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

    /// Analytical ELT summary (total arrival rate, AAL, occurrence SD).
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
            if var0 >= mu0 * (1.0 - mu0) {
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

    /// Conditional event severity survival function `S_i(x) = P(X_i > x)`.
    pub fn survival(&self, x: f64) -> Result<f64, PricingError> {
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter("threshold x must be finite"));
        }
        if x < 0.0 {
            return Ok(1.0);
        }
        match self.beta_params()? {
            None => Ok(if x < self.mean_loss { 1.0 } else { 0.0 }),
            Some((alpha, beta)) => {
                if x >= self.max_loss {
                    return Ok(0.0);
                }
                let u = x / self.max_loss;
                let cdf = regularized_incomplete_beta(u, alpha, beta);
                Ok((1.0 - cdf).clamp(0.0, 1.0))
            }
        }
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
    let mut max_possible = 0.0_f64;
    for entry in entries {
        entry.validate()?;
        max_possible = max_possible.max(entry.max_loss);
    }

    if elt_oep(entries, 0.0)? <= target_prob {
        return Ok(0.0);
    }

    let mut lo = 0.0;
    let mut hi = max_possible;
    for _ in 0..80 {
        let mid = f64::midpoint(lo, hi);
        let p = elt_oep(entries, mid)?;
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
/// from an Event Loss Table (exact for point-mass events; 64-point Gauss-Legendre / midpoint
/// integral of `S_i(x)` over `[attachment, attachment + limit]` for Beta-uncertain events).
pub fn elt_occurrence_layer_expected_loss(
    entries: &[EventLossEntry],
    layer: &CommercialLayer,
) -> Result<f64, PricingError> {
    if entries.is_empty() {
        return Err(PricingError::EmptyInput("ELT entries must not be empty"));
    }
    let attach = layer.attachment;
    let detach = attach + layer.limit;
    let mut total_ceded = 0.0;

    for entry in entries {
        if entry.beta_params()?.is_none() {
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
                integral += entry.survival(x)? * h;
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
        if entry.beta_params()?.is_none() {
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
                let cdf = 1.0 - entry.survival(upper)?;
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

/// Log-gamma function via Lanczos approximation (`g = 7`, 9 coefficients, accuracy `< 1e-14`).
fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + (i as f64));
    }
    let t = z + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized incomplete beta function `I_x(a, b)` via modified Lentz continued fraction.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-15;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}
