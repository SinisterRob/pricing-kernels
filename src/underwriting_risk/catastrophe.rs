//! Catastrophe Event Loss Table (ELT) engine for Occurrence Exceedance Probability (OEP),
//! Aggregate Exceedance Probability (AEP), Probable Maximum Loss (PML), and cat XL layer pricing.
//!
//! References:
//! - G. Homer & D. Li (2017), *"Notes on Using Property Catastrophe Model Results"*, *CAS E-Forum*.
//! - D. R. Clark (2014), *Basics of Reinsurance Pricing*, CAS Study Note, Ch. 6.
//! - S. J. Mildenhall & J. A. Major, *Introduction to Capital Modeling and Portfolio Management*,
//!   CAS Monograph No. 15.

use super::aggregate::{AggregateLossDistribution, FrequencyDistribution, fft_aggregate_loss};
use crate::error::PricingError;
use crate::layers::CommercialLayer;
use crate::special::regularized_incomplete_beta;
use serde::{Deserialize, Serialize};

/// Validated catastrophe Event Loss Table (ELT) container anchoring OEP, PML,
/// occurrence layer pricing, and FFT aggregate exceedance probability methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLossTable {
    entries: Vec<EventLossEntry>,
}

impl EventLossTable {
    /// Construct and validate an Event Loss Table from event records.
    pub fn try_new(entries: Vec<EventLossEntry>) -> Result<Self, PricingError> {
        if entries.is_empty() {
            return Err(PricingError::EmptyInput(
                "ELT must contain at least one event entry",
            ));
        }
        for entry in &entries {
            entry.validate()?;
        }
        Ok(Self { entries })
    }

    /// Underlying slice of validated [`EventLossEntry`] records.
    #[inline]
    pub fn entries(&self) -> &[EventLossEntry] {
        &self.entries
    }

    /// Number of events in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table has no events.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Analytical ELT summary (total arrival rate, AAL, aggregate annual variance and SD).
    pub fn summary(&self) -> Result<EltSummary, PricingError> {
        elt_summary(&self.entries)
    }

    /// Occurrence Exceedance Probability `OEP(x) = P(max_k X_k > x)`.
    pub fn oep(&self, loss_threshold: f64) -> Result<f64, PricingError> {
        elt_oep(&self.entries, loss_threshold)
    }

    /// Occurrence Probable Maximum Loss at `return_period` years (e.g., `100.0` for 1-in-100).
    pub fn oep_pml(&self, return_period: f64) -> Result<f64, PricingError> {
        elt_oep_pml(&self.entries, return_period)
    }

    /// Expected annual occurrence ceded loss for a catastrophe XL `layer`.
    pub fn occurrence_layer_expected_loss(
        &self,
        layer: &CommercialLayer,
    ) -> Result<f64, PricingError> {
        elt_occurrence_layer_expected_loss(&self.entries, layer)
    }

    /// Compound Poisson annual aggregate loss distribution via tilted radix-2 FFT.
    pub fn aggregate_distribution(
        &self,
        step: f64,
        fft_size: usize,
        tilt_theta: Option<f64>,
    ) -> Result<AggregateLossDistribution, PricingError> {
        elt_aggregate_distribution(&self.entries, step, fft_size, tilt_theta)
    }
}

/// Single catastrophe event record in an Event Loss Table (ELT).
///
/// Follows the standard RMS / AIR ELT convention: annual Poisson event rate `lambda_i`,
/// conditional mean loss `mean_loss`, independent and correlated severity standard deviations
/// (`std_dev_indep`, `std_dev_corr`), and maximum possible event loss `max_loss`.
/// When `total_std_dev() > 0`, event severity `X / max_loss` follows a Beta distribution
/// fitted by method of moments; when `total_std_dev() == 0`, the event has deterministic
/// point-mass severity `mean_loss`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EventLossEntry {
    /// Event identifier.
    pub event_id: u64,
    /// Annual Poisson arrival rate `lambda_i >= 0`.
    pub annual_rate: f64,
    /// Conditional expected event loss `mu_i >= 0`.
    pub mean_loss: f64,
    /// Independent severity standard deviation `sigma_{indep, i} >= 0`.
    pub std_dev_indep: f64,
    /// Correlated severity standard deviation `sigma_{corr, i} >= 0`.
    pub std_dev_corr: f64,
    /// Maximum possible loss for this event `M_i >= mean_loss`.
    pub max_loss: f64,
}

impl EventLossEntry {
    /// Total conditional event severity standard deviation `sqrt(sd_indep^2 + sd_corr^2)`.
    #[inline]
    pub fn total_std_dev(&self) -> f64 {
        self.std_dev_indep.hypot(self.std_dev_corr)
    }

    /// Validate event loss table entry parameters and Beta moment feasibility.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.annual_rate.is_finite()
            || !self.mean_loss.is_finite()
            || !self.std_dev_indep.is_finite()
            || !self.std_dev_corr.is_finite()
            || !self.max_loss.is_finite()
        {
            return Err(PricingError::InvalidParameter(
                "ELT entry fields must be finite",
            ));
        }
        if !(self.annual_rate >= 0.0)
            || !(self.mean_loss >= 0.0)
            || !(self.std_dev_indep >= 0.0)
            || !(self.std_dev_corr >= 0.0)
            || !(self.max_loss >= self.mean_loss)
        {
            return Err(PricingError::InvalidParameter(
                "ELT requires non-negative rate/mean/sd and max_loss >= mean_loss",
            ));
        }
        let sd = self.total_std_dev();
        if sd > 0.0 {
            if !(self.max_loss > 0.0) || !(self.mean_loss > 0.0) || self.mean_loss >= self.max_loss
            {
                return Err(PricingError::InvalidParameter(
                    "positive ELT severity sd requires 0 < mean_loss < max_loss",
                ));
            }
            let mu0 = self.mean_loss / self.max_loss;
            let var0 = (sd / self.max_loss).powi(2);
            let common = mu0 * (1.0 - mu0) / var0 - 1.0;
            if var0 >= mu0 * (1.0 - mu0) || !(common > 0.0) || !common.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "ELT variance exceeds maximum feasible Beta variance mu0 * (1 - mu0)",
                ));
            }
        }
        Ok(())
    }

    /// Fitted `(alpha, beta)` parameters for `X / max_loss ~ Beta(alpha, beta)`, or `None` if point-mass.
    pub fn beta_params(&self) -> Result<Option<(f64, f64)>, PricingError> {
        self.validate()?;
        let sd = self.total_std_dev();
        if sd == 0.0 {
            return Ok(None);
        }
        let mu0 = self.mean_loss / self.max_loss;
        let var0 = (sd / self.max_loss).powi(2);
        let common = mu0 * (1.0 - mu0) / var0 - 1.0;
        Ok(Some((mu0 * common, (1.0 - mu0) * common)))
    }

    #[inline]
    fn survival_with_params(&self, x: f64, beta_params: Option<(f64, f64)>) -> f64 {
        if x < 0.0 {
            return 1.0;
        }
        match beta_params {
            None => {
                if x < self.mean_loss {
                    1.0
                } else {
                    0.0
                }
            }
            Some((alpha, beta)) => {
                if x >= self.max_loss {
                    return 0.0;
                }
                let u = x / self.max_loss;
                let cdf = regularized_incomplete_beta(u, alpha, beta);
                (1.0 - cdf).clamp(0.0, 1.0)
            }
        }
    }

    /// Conditional event severity survival function `S_i(x) = P(X_i > x)`.
    pub fn survival(&self, x: f64) -> Result<f64, PricingError> {
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter("threshold x must be finite"));
        }
        let params = self.beta_params()?;
        Ok(self.survival_with_params(x, params))
    }
}

/// Summary metrics for an Event Loss Table (exact compound Poisson analytical moments).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EltSummary {
    /// Total annual event arrival rate `Lambda = sum lambda_i`.
    pub total_annual_rate: f64,
    /// Analytical Average Annual Loss `AAL = sum lambda_i * mu_i`.
    pub average_annual_loss: f64,
    /// Analytical aggregate annual variance `Var(S) = sum lambda_i * (mu_i^2 + sd_i^2)`.
    pub aggregate_variance: f64,
    /// Analytical aggregate annual standard deviation `sqrt(Var(S))`.
    pub aggregate_std_dev: f64,
}

/// Compute exact analytical compound-Poisson ELT moments (`AAL`, aggregate variance, total rate).
pub fn elt_summary(entries: &[EventLossEntry]) -> Result<EltSummary, PricingError> {
    if entries.is_empty() {
        return Err(PricingError::EmptyInput("ELT entries must not be empty"));
    }
    let mut total_rate = 0.0;
    let mut aal = 0.0;
    let mut agg_var = 0.0;
    for entry in entries {
        entry.validate()?;
        let sd = entry.total_std_dev();
        total_rate += entry.annual_rate;
        aal += entry.annual_rate * entry.mean_loss;
        agg_var += entry.annual_rate * (entry.mean_loss * entry.mean_loss + sd * sd);
    }
    Ok(EltSummary {
        total_annual_rate: total_rate,
        average_annual_loss: aal,
        aggregate_variance: agg_var,
        aggregate_std_dev: agg_var.sqrt(),
    })
}

/// Exact Poisson Occurrence Exceedance Probability `OEP(x) = 1 - exp(-sum_i lambda_i * S_i(x))`.
pub fn elt_oep(entries: &[EventLossEntry], loss_threshold: f64) -> Result<f64, PricingError> {
    if entries.is_empty() {
        return Err(PricingError::EmptyInput("ELT entries must not be empty"));
    }
    if !loss_threshold.is_finite() || !(loss_threshold >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "loss_threshold must be finite and non-negative",
        ));
    }
    let mut rate_exceeding = 0.0;
    for entry in entries {
        rate_exceeding += entry.annual_rate * entry.survival(loss_threshold)?;
    }
    Ok((-(-rate_exceeding).exp_m1()).clamp(0.0, 1.0))
}

/// Invert the exact OEP curve via bisection to find the Occurrence Probable Maximum Loss (PML)
/// at a target return period `return_period > 1.0` (e.g. `100.0` for 1-in-100, `250.0` for 1-in-250).
pub fn elt_oep_pml(entries: &[EventLossEntry], return_period: f64) -> Result<f64, PricingError> {
    if entries.is_empty() {
        return Err(PricingError::EmptyInput("ELT entries must not be empty"));
    }
    if !return_period.is_finite() || !(return_period > 1.0) {
        return Err(PricingError::InvalidParameter(
            "return_period must be finite and > 1.0",
        ));
    }
    let target_prob = 1.0 / return_period;
    let mut prepared = Vec::with_capacity(entries.len());
    let mut max_possible = 0.0_f64;
    for entry in entries {
        let params = entry.beta_params()?;
        max_possible = max_possible.max(entry.max_loss);
        prepared.push((entry, params));
    }

    let eval_oep = |threshold: f64| -> f64 {
        let mut rate_exceeding = 0.0;
        for &(entry, params) in &prepared {
            rate_exceeding += entry.annual_rate * entry.survival_with_params(threshold, params);
        }
        (-(-rate_exceeding).exp_m1()).clamp(0.0, 1.0)
    };

    if eval_oep(0.0) <= target_prob {
        return Ok(0.0);
    }

    let mut lo = 0.0;
    let mut hi = max_possible;
    for _ in 0..80 {
        let mid = f64::midpoint(lo, hi);
        let p = eval_oep(mid);
        if p > target_prob {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= 1e-9 * hi.max(1.0) {
            break;
        }
    }
    Ok(f64::midpoint(lo, hi))
}

/// Expected annual ceded loss to a per-occurrence catastrophe excess-of-loss layer `layer`
/// from an Event Loss Table (exact for point-mass events; 128-point composite midpoint
/// integral of `S_i(x)` over `[attachment, attachment + limit]` for Beta-uncertain events).
pub fn elt_occurrence_layer_expected_loss(
    entries: &[EventLossEntry],
    layer: &CommercialLayer,
) -> Result<f64, PricingError> {
    if entries.is_empty() {
        return Err(PricingError::EmptyInput("ELT entries must not be empty"));
    }
    layer.validate()?;
    let attach = layer.attachment;
    let detach = attach + layer.limit;
    let mut total_ceded = 0.0;

    for entry in entries {
        let params = entry.beta_params()?;
        if entry.annual_rate == 0.0 {
            continue;
        }
        if params.is_none() {
            total_ceded += entry.annual_rate * layer.payout(entry.mean_loss);
        } else {
            if attach >= entry.max_loss {
                continue;
            }
            let hi = detach.min(entry.max_loss);
            let n = 128_i32;
            let h = (hi - attach) / f64::from(n);
            let mut integral = 0.0;
            for k in 0..n {
                let x = attach + (f64::from(k) + 0.5) * h;
                integral += entry.survival_with_params(x, params) * h;
            }
            total_ceded += entry.annual_rate * layer.participation * integral;
        }
    }
    Ok(total_ceded)
}

/// Build the exact compound-Poisson Aggregate Exceedance Probability (AEP) distribution
/// from an Event Loss Table via FFT convolution (`step` = grid spacing, `fft_size` = power of 2).
pub fn elt_aggregate_distribution(
    entries: &[EventLossEntry],
    step: f64,
    fft_size: usize,
    tilt: Option<f64>,
) -> Result<AggregateLossDistribution, PricingError> {
    let summary = elt_summary(entries)?;
    if !(summary.total_annual_rate > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "ELT total annual rate must be positive",
        ));
    }
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "step must be finite and positive",
        ));
    }
    if fft_size < 2 || !fft_size.is_power_of_two() {
        return Err(PricingError::InvalidParameter(
            "fft_size must be a power of two >= 2",
        ));
    }

    let mut mix_pmf = vec![0.0; fft_size];
    for entry in entries {
        if entry.annual_rate == 0.0 {
            continue;
        }
        let weight = entry.annual_rate / summary.total_annual_rate;
        let params = entry.beta_params()?;
        if params.is_none() {
            let exact_bin = entry.mean_loss / step;
            let lo = (exact_bin.floor() as usize).min(fft_size - 1);
            let hi = (lo + 1).min(fft_size - 1);
            if lo == hi {
                mix_pmf[lo] += weight;
            } else {
                let frac = (exact_bin - (lo as f64)).clamp(0.0, 1.0);
                mix_pmf[lo] += weight * (1.0 - frac);
                mix_pmf[hi] += weight * frac;
            }
        } else {
            let max_bin = ((entry.max_loss / step).ceil() as usize).min(fft_size - 1);
            let mut prev_cdf = 0.0;
            for k in 0..max_bin {
                let upper = ((k as f64) + 0.5) * step;
                let cdf = 1.0 - entry.survival_with_params(upper, params);
                mix_pmf[k] += weight * (cdf - prev_cdf).max(0.0);
                prev_cdf = cdf;
            }
            mix_pmf[max_bin] += weight * (1.0 - prev_cdf).max(0.0);
        }
    }

    fft_aggregate_loss(
        FrequencyDistribution::Poisson {
            lambda: summary.total_annual_rate,
        },
        &mix_pmf,
        step,
        fft_size,
        tilt,
    )
}

// ===========================================================================
// Catastrophe Year Loss Table (YLT) & Aggregate XL (AAD / AAL)
// ===========================================================================

/// Single simulated catastrophe event occurrence in a Year Loss Table (YLT).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct YltEvent {
    /// Zero-based simulation year index (`0..n_years`).
    pub year: u32,
    /// Catastrophe event identifier.
    pub event_id: u64,
    /// Simulated gross occurrence loss (`>= 0`).
    pub loss: f64,
}

/// Validated Catastrophe Year Loss Table (YLT) supporting empirical OEP/AEP/PML,
/// per-occurrence XL, and aggregate XL (Annual Aggregate Deductible / Annual
/// Aggregate Limit) catastrophe treaty pricing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YearLossTable {
    n_years: u32,
    events: Vec<YltEvent>,
}

impl YearLossTable {
    /// Construct and validate a [`YearLossTable`] spanning `n_years >= 1` simulation years.
    pub fn try_new(n_years: u32, events: Vec<YltEvent>) -> Result<Self, PricingError> {
        if n_years == 0 {
            return Err(PricingError::InvalidParameter(
                "YearLossTable requires n_years >= 1",
            ));
        }
        for ev in &events {
            if ev.year >= n_years {
                return Err(PricingError::OutOfBounds(
                    "YLT event year must be < n_years",
                ));
            }
            if !ev.loss.is_finite() || !(ev.loss >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "YLT event loss must be finite and non-negative",
                ));
            }
        }
        Ok(Self { n_years, events })
    }

    /// Number of simulated years in the YLT.
    #[inline]
    pub const fn n_years(&self) -> u32 {
        self.n_years
    }

    /// Underlying simulated event records.
    #[inline]
    pub fn events(&self) -> &[YltEvent] {
        &self.events
    }

    /// Annual maximum (occurrence) loss vector of length `n_years` (`0.0` in event-free years).
    pub fn annual_occurrence_losses(&self) -> Vec<f64> {
        let mut maxes = vec![0.0_f64; self.n_years as usize];
        for ev in &self.events {
            let slot = &mut maxes[ev.year as usize];
            if ev.loss > *slot {
                *slot = ev.loss;
            }
        }
        maxes
    }

    /// Annual aggregate (sum of events) loss vector of length `n_years` (`0.0` in event-free years).
    pub fn annual_aggregate_losses(&self) -> Vec<f64> {
        let mut sums = vec![0.0_f64; self.n_years as usize];
        for ev in &self.events {
            sums[ev.year as usize] += ev.loss;
        }
        sums
    }

    /// Empirical Average Annual Loss `AAL = (1 / n_years) * sum(loss)`.
    pub fn aal(&self) -> f64 {
        let total: f64 = self.events.iter().map(|e| e.loss).sum();
        total / f64::from(self.n_years)
    }

    fn exceedance_prob(&self, annual_losses: &[f64], threshold: f64) -> Result<f64, PricingError> {
        if !threshold.is_finite() || !(threshold >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "threshold must be finite and non-negative",
            ));
        }
        let exceed = annual_losses.iter().filter(|&&v| v > threshold).count();
        Ok((exceed as f64) / f64::from(self.n_years))
    }

    /// Empirical Occurrence Exceedance Probability `OEP(x) = P(max_k X_k > x)`.
    pub fn oep(&self, threshold: f64) -> Result<f64, PricingError> {
        self.exceedance_prob(&self.annual_occurrence_losses(), threshold)
    }

    /// Empirical Aggregate Exceedance Probability `AEP(x) = P(sum_k X_k > x)`.
    pub fn aep(&self, threshold: f64) -> Result<f64, PricingError> {
        self.exceedance_prob(&self.annual_aggregate_losses(), threshold)
    }

    /// Empirical Occurrence PML at `return_period > 1.0` years (e.g., `100.0` for 1-in-100).
    pub fn oep_pml(&self, return_period: f64) -> Result<f64, PricingError> {
        empirical_pml(self.annual_occurrence_losses(), return_period)
    }

    /// Empirical Aggregate PML at `return_period > 1.0` years (e.g., `100.0` for 1-in-100).
    pub fn aep_pml(&self, return_period: f64) -> Result<f64, PricingError> {
        empirical_pml(self.annual_aggregate_losses(), return_period)
    }

    /// Expected annual ceded loss for a combined per-occurrence + aggregate excess-of-loss
    /// catastrophe treaty (`per_occ_limit xs per_occ_attachment`, subject to Annual Aggregate
    /// Deductible `aggregate_deductible` and Annual Aggregate Limit `aggregate_limit`).
    ///
    /// Pass `per_occ_attachment = 0.0`, `per_occ_limit = f64::INFINITY` for a pure annual
    /// aggregate stop-loss treaty; pass `aggregate_deductible = 0.0`, `aggregate_limit = f64::INFINITY`
    /// for a pure per-occurrence XL treaty.
    pub fn aggregate_xl_expected_loss(
        &self,
        per_occ_attachment: f64,
        per_occ_limit: f64,
        aggregate_deductible: f64,
        aggregate_limit: f64,
    ) -> Result<f64, PricingError> {
        if !per_occ_attachment.is_finite()
            || !(per_occ_attachment >= 0.0)
            || per_occ_limit.is_nan()
            || !(per_occ_limit > 0.0)
            || !aggregate_deductible.is_finite()
            || !(aggregate_deductible >= 0.0)
            || aggregate_limit.is_nan()
            || !(aggregate_limit > 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "attachments/deductibles must be finite >= 0 and limits must be > 0",
            ));
        }

        let mut annual_subject = vec![0.0_f64; self.n_years as usize];
        for ev in &self.events {
            if ev.loss > per_occ_attachment {
                let occ_ceded = (ev.loss - per_occ_attachment).min(per_occ_limit);
                annual_subject[ev.year as usize] += occ_ceded;
            }
        }

        let total_ceded: f64 = annual_subject
            .iter()
            .map(|&s| (s - aggregate_deductible).max(0.0).min(aggregate_limit))
            .sum();
        Ok(total_ceded / f64::from(self.n_years))
    }
}

fn empirical_pml(mut annual_losses: Vec<f64>, return_period: f64) -> Result<f64, PricingError> {
    if !return_period.is_finite() || !(return_period > 1.0) {
        return Err(PricingError::InvalidParameter(
            "return_period must be finite and > 1.0",
        ));
    }
    let n = annual_losses.len();
    annual_losses.sort_by(f64::total_cmp);
    let alpha = 1.0 - 1.0 / return_period;
    let idx = ((alpha * (n as f64) - 1e-15).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    Ok(annual_losses[idx])
}

/// Deterministic SplitMix64-seeded Xoshiro256** PRNG for reproducible YLT simulation.
struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn from_seed(mut seed: u64) -> Self {
        let mut next_sm = || {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next_sm(), next_sm(), next_sm(), next_sm()],
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_std_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn sample_gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let u = self.next_f64().max(1e-300);
            return self.sample_gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_std_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.next_f64().max(1e-300);
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.sample_gamma(alpha);
        let y = self.sample_gamma(beta);
        if x + y > 0.0 { x / (x + y) } else { 0.5 }
    }

    fn sample_poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0u32;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.next_f64().max(1e-300);
                if p <= l {
                    return k - 1;
                }
            }
        }
        // Normal approximation for high annual rate (> 30 events/yr).
        let draw = (lambda + lambda.sqrt() * self.next_std_normal())
            .round()
            .max(0.0);
        draw as u32
    }
}

/// Deterministically simulate a [`YearLossTable`] of `n_years` simulation years from a
/// catastrophe Event Loss Table using a seeded `Xoshiro256**` PRNG (recommended default
/// seed: `42069`).
///
/// Each year draws a compound-Poisson event count (`Poisson(sum lambda_i)`), selects each
/// event proportional to `lambda_i`, and draws its severity from `max_loss * Beta(alpha, beta)`
/// (or point-mass `mean_loss` when `total_std_dev() == 0`).
pub fn simulate_ylt_from_elt(
    entries: &[EventLossEntry],
    n_years: u32,
    seed: u64,
) -> Result<YearLossTable, PricingError> {
    if n_years == 0 {
        return Err(PricingError::InvalidParameter("n_years must be >= 1"));
    }
    let summary = elt_summary(entries)?;
    if !(summary.total_annual_rate > 0.0) {
        return Ok(YearLossTable {
            n_years,
            events: Vec::new(),
        });
    }

    let mut prepared = Vec::with_capacity(entries.len());
    let mut cum_rates = Vec::with_capacity(entries.len());
    let mut running = 0.0;
    for entry in entries {
        if entry.annual_rate == 0.0 {
            continue;
        }
        let params = entry.beta_params()?;
        running += entry.annual_rate;
        prepared.push((entry, params));
        cum_rates.push(running / summary.total_annual_rate);
    }
    if let Some(last) = cum_rates.last_mut() {
        *last = 1.0;
    }

    let mut rng = Xoshiro256StarStar::from_seed(seed);
    let mut events = Vec::new();
    for year in 0..n_years {
        let count = rng.sample_poisson(summary.total_annual_rate);
        for _ in 0..count {
            let u = rng.next_f64();
            let idx = cum_rates
                .partition_point(|&c| c < u)
                .min(prepared.len() - 1);
            let (entry, params) = prepared[idx];
            let loss = match params {
                None => entry.mean_loss,
                Some((a, b)) => entry.max_loss * rng.sample_beta(a, b),
            };
            events.push(YltEvent {
                year,
                event_id: entry.event_id,
                loss,
            });
        }
    }

    Ok(YearLossTable { n_years, events })
}
