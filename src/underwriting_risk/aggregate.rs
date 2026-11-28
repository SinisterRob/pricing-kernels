//! Prospective compound aggregate underwriting loss distributions via Panjer (1981)
//! `(a, b, 0)` recursion, Robertson (1992) exponentially tilted radix-2 FFT, and
//! Klugman–Panjer–Willmot mean-preserving severity discretization.
//!
//! References:
//! - H. H. Panjer (1981), *"Recursive Evaluation of a Family of Compound Distributions"*,
//!   *ASTIN Bulletin*, Vol. 12(1), pp. 22–26.
//! - J. P. Robertson (1992), *"The Computation of Aggregate Loss Distributions"*,
//!   *PCAS*, Vol. LXXIX, pp. 57–133.
//! - P. Embrechts & M. Frei (2009), *"Panjer Recursion versus FFT for Compound Distributions"*,
//!   *Mathematical Methods of Operations Research*, Vol. 69, pp. 497–508.
//! - S. A. Klugman, H. H. Panjer & G. E. Willmot, *Loss Models* (5th ed.), Ch. 9.

use crate::error::PricingError;
use crate::layers::SeverityDistribution;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Maximum probability mass (and maximum relative mean error) an aggregate grid may lose to
/// truncation (Panjer) or circular wrap-around aliasing (FFT) before the kernel returns
/// [`PricingError::OutOfBounds`] instead of a silently biased distribution.
///
/// `1e-6` keeps the lost mass two orders of magnitude below a `1 - 0.9999` (1-in-10,000)
/// tail probability, so every `VaR` / `TVaR` level used in practice is supported by
/// retained grid mass. Increase `n_steps` / `fft_size` (or `step`) if this guard fires.
///
/// The same tolerance bounds the accepted input severity-PMF mass deficit (see
/// [`panjer_recursion`]) and the total-mass error of each [`convolve_independent_aggregates`]
/// component, so an input deficit is never misreported as grid truncation.
pub const AGGREGATE_GRID_TOLERANCE: f64 = 1e-6;

/// Safety multiple applied to the estimated per-bin floating-point noise of the radix-2 FFT
/// inversion before untilting.
///
/// The per-bin noise in the tilted domain is estimated as
/// `max(max_k |Im g_tilde_k|, log2(N) * f64::EPSILON * max_k |g_tilde_k|)` (the imaginary part
/// of the inverse transform is exactly zero in exact arithmetic, so its magnitude measures the
/// realized round-off; the second term is the standard `O(eps log2 N)` FFT error bound).
/// Untilting multiplies the noise at bin `k` by `exp(theta * k)` (Embrechts & Frei 2009, §3),
/// so bins with `|g_k| <= FFT_ROUNDOFF_SAFETY_FACTOR * noise * exp(theta * k)` are set to `0`.
pub const FFT_ROUNDOFF_SAFETY_FACTOR: f64 = 8.0;

/// Maximum fraction of the `TVaR_alpha` tail mass `1 - alpha` that may be missing from the
/// aggregate grid (mass lost beyond `(pmf.len() - 1) * step`) before
/// [`AggregateLossDistribution::tvar`] returns [`PricingError::OutOfBounds`].
///
/// At `1e-3` at least 99.9% of the tail is represented on the grid; because missing mass lies
/// above every grid point, the returned value is then a lower bound on the true `TVaR`.
pub const TVAR_MAX_LOST_TAIL_FRACTION: f64 = 1e-3;

/// Claim-count frequency distribution in the Panjer `(a, b, 0)` class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrequencyDistribution {
    /// `Poisson(lambda)`: `E[N] = Var(N) = lambda`, `a = 0`, `b = lambda`.
    Poisson {
        /// Poisson mean intensity `lambda > 0`.
        lambda: f64,
    },
    /// `Binomial(m, q)`: `E[N] = m q`, `Var(N) = m q (1 - q)`,
    /// `a = -q / (1 - q)`, `b = (m + 1) q / (1 - q)`.
    Binomial {
        /// Number of trials `m >= 1`.
        m: u64,
        /// Per-trial claim probability `q in (0, 1)`.
        q: f64,
    },
    /// `NegativeBinomial(r, beta)` (Klugman–Panjer–Willmot parameterization):
    /// `E[N] = r * beta`, `Var(N) = r * beta * (1 + beta)`,
    /// `a = beta / (1 + beta)`, `b = (r - 1) * beta / (1 + beta)`.
    NegativeBinomial {
        /// Shape / dispersion parameter `r > 0`.
        r: f64,
        /// Scale parameter `beta > 0`.
        beta: f64,
    },
}

impl FrequencyDistribution {
    /// Validate distribution parameters.
    pub fn validate(self) -> Result<(), PricingError> {
        match self {
            Self::Poisson { lambda } => {
                if !lambda.is_finite() || !(lambda > 0.0) {
                    return Err(PricingError::NonPositiveValue(
                        "Poisson lambda must be finite and positive",
                    ));
                }
            }
            Self::Binomial { m, q } => {
                if m == 0 || !q.is_finite() || !(q > 0.0) || !(q < 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "Binomial requires m >= 1 and q in (0, 1)",
                    ));
                }
            }
            Self::NegativeBinomial { r, beta } => {
                if !r.is_finite() || !beta.is_finite() || !(r > 0.0) || !(beta > 0.0) {
                    return Err(PricingError::NonPositiveValue(
                        "NegativeBinomial requires finite r > 0 and beta > 0",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Expected claim count `E[N]`.
    pub fn mean(self) -> Result<f64, PricingError> {
        self.validate()?;
        Ok(match self {
            Self::Poisson { lambda } => lambda,
            Self::Binomial { m, q } => (m as f64) * q,
            Self::NegativeBinomial { r, beta } => r * beta,
        })
    }

    /// Claim-count variance `Var(N)`.
    pub fn variance(self) -> Result<f64, PricingError> {
        self.validate()?;
        Ok(match self {
            Self::Poisson { lambda } => lambda,
            Self::Binomial { m, q } => (m as f64) * q * (1.0 - q),
            Self::NegativeBinomial { r, beta } => r * beta * (1.0 + beta),
        })
    }

    fn panjer_ab(self) -> (f64, f64) {
        match self {
            Self::Poisson { lambda } => (0.0, lambda),
            Self::Binomial { m, q } => {
                let ratio = q / (1.0 - q);
                (-ratio, ((m as f64) + 1.0) * ratio)
            }
            Self::NegativeBinomial { r, beta } => {
                let a = beta / (1.0 + beta);
                (a, (r - 1.0) * a)
            }
        }
    }

    /// Probability generating function `P_N(z)` evaluated at a real scalar `z`.
    fn pgf_real(self, z: f64) -> f64 {
        match self {
            Self::Poisson { lambda } => (lambda * (z - 1.0)).exp(),
            Self::Binomial { m, q } => (1.0 - q + q * z).powf(m as f64),
            Self::NegativeBinomial { r, beta } => (1.0 - beta * (z - 1.0)).powf(-r),
        }
    }

    /// Probability generating function `P_N(z)` evaluated at complex `z = (re, im)`.
    fn pgf_complex(self, re: f64, im: f64) -> (f64, f64) {
        match self {
            Self::Poisson { lambda } => {
                let x = lambda * (re - 1.0);
                let y = lambda * im;
                let ex = x.exp();
                (ex * y.cos(), ex * y.sin())
            }
            Self::Binomial { m, q } => {
                let wr = 1.0 - q + q * re;
                let wi = q * im;
                let r = wr.hypot(wi);
                let theta = wi.atan2(wr);
                let rm = r.powf(m as f64);
                let ang = (m as f64) * theta;
                (rm * ang.cos(), rm * ang.sin())
            }
            Self::NegativeBinomial { r, beta } => {
                let wr = 1.0 - beta * (re - 1.0);
                let wi = -beta * im;
                let rad = wr.hypot(wi);
                let theta = wi.atan2(wr);
                let mag = rad.powf(-r);
                let ang = -r * theta;
                (mag * ang.cos(), mag * ang.sin())
            }
        }
    }
}

/// Discretized prospective aggregate underwriting loss distribution on `0, step, 2*step, ...`.
///
/// Fields are private so the derived moments can never go stale relative to the PMF;
/// construct with [`AggregateLossDistribution::from_pmf`] (deserialization re-validates
/// through the same constructor and recomputes the moments).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AggregateLossDistributionRepr")]
pub struct AggregateLossDistribution {
    step: f64,
    pmf: Vec<f64>,
    mean: f64,
    variance: f64,
    std_dev: f64,
}

/// Serialized form of [`AggregateLossDistribution`]; derived moments, if present, are
/// ignored and recomputed from `pmf`.
#[derive(Deserialize)]
struct AggregateLossDistributionRepr {
    step: f64,
    pmf: Vec<f64>,
}

impl TryFrom<AggregateLossDistributionRepr> for AggregateLossDistribution {
    type Error = PricingError;

    fn try_from(repr: AggregateLossDistributionRepr) -> Result<Self, Self::Error> {
        Self::from_pmf(repr.step, repr.pmf)
    }
}

impl AggregateLossDistribution {
    /// Monetary grid spacing `h > 0`.
    pub const fn step(&self) -> f64 {
        self.step
    }

    /// Discrete probability mass function `g_k = P(S = k * step)`.
    pub fn pmf(&self) -> &[f64] {
        &self.pmf
    }

    /// Mean `E[S]` computed from the discrete PMF.
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Variance `Var(S)` computed from the discrete PMF.
    pub const fn variance(&self) -> f64 {
        self.variance
    }

    /// Standard deviation `sqrt(Var(S))`.
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Construct from a validated grid spacing `step > 0` and probability mass vector.
    pub fn from_pmf(step: f64, pmf: Vec<f64>) -> Result<Self, PricingError> {
        if !step.is_finite() || !(step > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "grid step must be finite and positive",
            ));
        }
        if pmf.is_empty() {
            return Err(PricingError::EmptyInput("pmf must not be empty"));
        }
        let mut mean = 0.0;
        for (k, &p) in pmf.iter().enumerate() {
            if !p.is_finite() || !(p >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "pmf probabilities must be finite and non-negative",
                ));
            }
            let x = (k as f64) * step;
            mean += p * x;
        }
        let mut variance = 0.0;
        for (k, &p) in pmf.iter().enumerate() {
            let diff = (k as f64) * step - mean;
            variance += p * diff * diff;
        }
        let variance = variance.max(0.0);
        Ok(Self {
            step,
            pmf,
            mean,
            variance,
            std_dev: variance.sqrt(),
        })
    }

    /// Total probability mass `sum_k g_k` retained on the grid (`1.0` for a complete distribution).
    ///
    /// `1.0 - total_mass()` is the mass lost to grid truncation beyond `(pmf.len() - 1) * step`.
    pub fn total_mass(&self) -> f64 {
        self.pmf.iter().sum()
    }

    /// Recursive-summation round-off bound (`n * f64::EPSILON`) used when comparing the
    /// accumulated CDF against a confidence level.
    fn cdf_round_off(&self) -> f64 {
        (self.pmf.len() as f64) * f64::EPSILON
    }

    /// Exceedance probability `P(S > x)` evaluated on the discrete grid.
    pub fn exceedance_probability(&self, x: f64) -> Result<f64, PricingError> {
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter("threshold x must be finite"));
        }
        if x < 0.0 {
            return Ok(self.pmf.iter().sum::<f64>().clamp(0.0, 1.0));
        }
        let mut prob = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            if (k as f64) * self.step > x {
                prob += p;
            }
        }
        Ok(prob.clamp(0.0, 1.0))
    }

    /// Value-at-Risk `VaR_alpha(S) = inf { x : F(x) >= alpha }` for `alpha in (0, 1)`.
    ///
    /// Returns [`PricingError::OutOfBounds`] when `alpha` exceeds the probability mass retained
    /// on the grid (the quantile lies beyond `(pmf.len() - 1) * step` and is not representable).
    pub fn var(&self, alpha: f64) -> Result<f64, PricingError> {
        if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
            return Err(PricingError::InvalidParameter(
                "confidence level alpha must be finite and in (0, 1)",
            ));
        }
        let threshold = alpha - self.cdf_round_off();
        let mut cdf = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            cdf += p;
            if cdf >= threshold {
                return Ok((k as f64) * self.step);
            }
        }
        Err(PricingError::OutOfBounds(
            "VaR confidence level exceeds the probability mass retained on the aggregate grid",
        ))
    }

    /// Coherent Tail Value-at-Risk `TVaR_alpha(S) = (1 / (1 - alpha)) * integral_alpha^1 VaR_u(S) du`.
    ///
    /// Any mass missing from the grid lies above every grid point, i.e. inside every tail. The
    /// tail integral is normalized by the *retained* tail mass, so the result is a
    /// probability-weighted average of grid points at or above `VaR_alpha` (`TVaR_alpha >=
    /// VaR_alpha` by construction) and a lower bound on the true `TVaR` when mass is missing.
    ///
    /// Returns [`PricingError::OutOfBounds`] when `|1 - total_mass()| >
    /// AGGREGATE_GRID_TOLERANCE`, or when the missing mass `1 - total_mass()` exceeds
    /// [`TVAR_MAX_LOST_TAIL_FRACTION`]` * (1 - alpha)`, i.e. when less than 99.9% of the tail
    /// is represented on the grid (relevant for `alpha` close to `1`).
    pub fn tvar(&self, alpha: f64) -> Result<f64, PricingError> {
        if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
            return Err(PricingError::InvalidParameter(
                "confidence level alpha must be finite and in (0, 1)",
            ));
        }
        if (1.0 - self.total_mass()).abs() > AGGREGATE_GRID_TOLERANCE {
            return Err(PricingError::OutOfBounds(
                "TVaR undefined: aggregate grid total mass differs from 1 by more than AGGREGATE_GRID_TOLERANCE",
            ));
        }
        let lost_mass = (1.0 - self.total_mass()).max(0.0);
        if lost_mass > TVAR_MAX_LOST_TAIL_FRACTION * (1.0 - alpha) {
            return Err(PricingError::OutOfBounds(
                "TVaR undefined: mass lost beyond the grid exceeds TVAR_MAX_LOST_TAIL_FRACTION of the tail mass 1 - alpha; extend the grid",
            ));
        }
        let mut cdf = 0.0;
        let mut weighted_sum = 0.0;
        let mut tail_mass = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            let next_cdf = cdf + p;
            if next_cdf > alpha {
                let contrib = next_cdf.min(1.0) - cdf.max(alpha);
                if contrib > 0.0 {
                    weighted_sum += contrib * (k as f64) * self.step;
                    tail_mass += contrib;
                }
            }
            cdf = next_cdf;
        }
        if !(tail_mass > 0.0) {
            return Err(PricingError::OutOfBounds(
                "TVaR confidence level exceeds the probability mass retained on the aggregate grid",
            ));
        }
        Ok(weighted_sum / tail_mass)
    }

    /// Expected aggregate loss ceded to an aggregate stop-loss layer `limit xs attachment`.
    pub fn layer_expected_loss(&self, attachment: f64, limit: f64) -> Result<f64, PricingError> {
        if !attachment.is_finite() || !limit.is_finite() || !(attachment >= 0.0) || !(limit > 0.0) {
            return Err(PricingError::InvalidParameter(
                "attachment must be >= 0 and limit must be > 0",
            ));
        }
        let mut total = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            let x = (k as f64) * self.step;
            if x > attachment {
                total += p * (x - attachment).min(limit);
            }
        }
        Ok(total)
    }
}

/// Mean-preserving LEV discretization of a continuous `SeverityDistribution` onto grid
/// `0, step, 2*step, ..., (n_bins - 1)*step` (Klugman, Panjer & Willmot, Ch. 9).
///
/// The upper tail beyond `(n_bins - 1) * step` is folded into the last bin so probabilities
/// sum to `1.0` and the discrete mean matches `E[X ^ ((n_bins - 1) * step)]` cleanly.
pub fn discretize_severity_mean_preserving(
    dist: &SeverityDistribution,
    step: f64,
    n_bins: usize,
) -> Result<Vec<f64>, PricingError> {
    dist.validate()?;
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "discretization step must be finite and positive",
        ));
    }
    if n_bins < 2 {
        return Err(PricingError::InvalidParameter("n_bins must be at least 2"));
    }

    let mut lev = Vec::with_capacity(n_bins);
    lev.push(0.0);
    for j in 1..n_bins {
        lev.push(dist.limited_expected_value((j as f64) * step)?);
    }
    Ok(mean_preserving_pmf_from_lev(&lev, step))
}

/// Klugman–Panjer–Willmot mean-preserving (local moment matching, order 1) PMF from limited
/// expected values `lev[j] = E[X ^ (j * step)]` (`lev[0] = 0`).
///
/// `f_0 = 1 - lev[1] / h`, `f_j = (2 lev[j] - lev[j-1] - lev[j+1]) / h`, and the remaining tail
/// mass is folded into the last bin, so `sum_j j h f_j = lev[n - 1]` telescopes exactly.
pub(crate) fn mean_preserving_pmf_from_lev(lev: &[f64], step: f64) -> Vec<f64> {
    let n_bins = lev.len();
    let mut pmf = vec![0.0; n_bins];
    if n_bins < 2 {
        if let Some(first) = pmf.first_mut() {
            *first = 1.0;
        }
        return pmf;
    }
    pmf[0] = (1.0 - lev[1] / step).clamp(0.0, 1.0);
    for j in 1..(n_bins - 1) {
        pmf[j] = ((2.0 * lev[j] - lev[j - 1] - lev[j + 1]) / step).max(0.0);
    }
    let prefix_sum: f64 = pmf[..(n_bins - 1)].iter().sum();
    pmf[n_bins - 1] = (1.0 - prefix_sum).max(0.0);
    pmf
}

/// Exact mean `E[X] = sum_k k h f_k` of a discrete severity PMF on grid spacing `step`.
fn severity_grid_mean(severity_pmf: &[f64], step: f64) -> f64 {
    severity_pmf
        .iter()
        .enumerate()
        .map(|(k, &p)| (k as f64) * step * p)
        .sum()
}

/// Floating-point round-off allowance carried into [`check_grid_adequacy`].
#[derive(Debug, Clone, Copy, Default)]
struct RoundOffAllowance {
    /// Allowance on total probability mass.
    mass: f64,
    /// Allowance on the mean `E[S]` (monetary units).
    mean: f64,
}

/// Reject a discretized aggregate whose retained mass or mean shows truncation / aliasing.
///
/// For a finite-support severity PMF the compound mean `E[S] = E[N] E[X]` is exact, so a
/// relative mean error beyond [`AGGREGATE_GRID_TOLERANCE`] detects mass truncated off the grid
/// or wrapped around to low indices by circular convolution, even when total mass is `1`.
/// `round_off` widens both checks by the computed floating-point floor (zero for Panjer).
fn check_grid_adequacy(
    dist: &AggregateLossDistribution,
    expected_mean: f64,
    round_off: RoundOffAllowance,
    mass_msg: &'static str,
    mean_msg: &'static str,
) -> Result<(), PricingError> {
    if (1.0 - dist.total_mass()).abs() > AGGREGATE_GRID_TOLERANCE + round_off.mass {
        return Err(PricingError::OutOfBounds(mass_msg));
    }
    let scale = expected_mean.abs().max(dist.step);
    if (dist.mean - expected_mean).abs() > AGGREGATE_GRID_TOLERANCE * scale + round_off.mean {
        return Err(PricingError::OutOfBounds(mean_msg));
    }
    Ok(())
}

/// Validate a severity PMF and return it renormalized to unit mass.
///
/// Entries must be finite and non-negative and the total mass must lie within
/// [`AGGREGATE_GRID_TOLERANCE`] of `1` (otherwise an input-mass error is returned, distinct from
/// the grid-truncation errors of the aggregate kernels). Accepted rounding-level deficits or
/// excesses are removed by dividing by the total, so the aggregate guards measure grid
/// truncation / aliasing only.
fn normalized_severity_pmf(severity_pmf: &[f64]) -> Result<Vec<f64>, PricingError> {
    if severity_pmf.is_empty() {
        return Err(PricingError::EmptyInput("severity_pmf must not be empty"));
    }
    let mut sum = 0.0;
    for &p in severity_pmf {
        if !p.is_finite() || !(p >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "severity_pmf entries must be finite and non-negative",
            ));
        }
        sum += p;
    }
    if !((sum - 1.0).abs() <= AGGREGATE_GRID_TOLERANCE) {
        return Err(PricingError::InvalidParameter(
            "severity_pmf input mass differs from 1 by more than AGGREGATE_GRID_TOLERANCE; normalize the severity input (this is not a grid-truncation error)",
        ));
    }
    if sum == 1.0 {
        return Ok(severity_pmf.to_vec());
    }
    Ok(severity_pmf.iter().map(|&p| p / sum).collect())
}

/// Exact Panjer `(a, b, 0)` compound aggregate loss recursion (Panjer 1981).
///
/// `severity_pmf` must sum to `1` within [`AGGREGATE_GRID_TOLERANCE`] (else
/// [`PricingError::InvalidParameter`]) and is renormalized to unit mass.
///
/// The recursion is exact on grid points `0..n_steps`; mass beyond `(n_steps - 1) * step` is
/// truncated. Returns [`PricingError::OutOfBounds`] if the lost mass `1 - sum g_k` or the
/// relative mean deficit versus `E[N] E[X]` exceeds [`AGGREGATE_GRID_TOLERANCE`] (this also
/// fires when `g_0 = P_N(f_0)` underflows to zero for very large expected claim counts).
pub fn panjer_recursion(
    freq: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
) -> Result<AggregateLossDistribution, PricingError> {
    freq.validate()?;
    let severity = normalized_severity_pmf(severity_pmf)?;
    let severity_pmf = severity.as_slice();
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "step must be finite and positive",
        ));
    }
    if n_steps == 0 {
        return Err(PricingError::InvalidParameter("n_steps must be >= 1"));
    }

    let (a, b) = freq.panjer_ab();
    let f0 = severity_pmf[0];
    let denom = 1.0 - a * f0;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(PricingError::InvalidParameter(
            "Panjer recursion denominator 1 - a * f_0 must be positive",
        ));
    }

    let mut g = vec![0.0; n_steps];
    g[0] = freq.pgf_real(f0);
    let m = severity_pmf.len() - 1;

    for x in 1..n_steps {
        let max_y = x.min(m);
        let mut acc = 0.0;
        let xf = x as f64;
        for y in 1..=max_y {
            let fy = severity_pmf[y];
            if fy != 0.0 {
                acc += (a + b * (y as f64) / xf) * fy * g[x - y];
            }
        }
        g[x] = (acc / denom).max(0.0);
    }

    let dist = AggregateLossDistribution::from_pmf(step, g)?;
    check_grid_adequacy(
        &dist,
        freq.mean()? * severity_grid_mean(severity_pmf, step),
        RoundOffAllowance::default(),
        "Panjer grid truncation: lost probability mass exceeds AGGREGATE_GRID_TOLERANCE; increase n_steps",
        "Panjer grid truncation: aggregate mean deficit exceeds AGGREGATE_GRID_TOLERANCE; increase n_steps",
    )?;
    Ok(dist)
}

/// Pure-Rust in-place iterative radix-2 Cooley–Tukey FFT (`inverse = false` for DFT, `true` for IDFT).
pub(crate) fn radix2_fft(
    re: &mut [f64],
    im: &mut [f64],
    inverse: bool,
) -> Result<(), PricingError> {
    let n = re.len();
    if n == 0 || n != im.len() || !n.is_power_of_two() {
        return Err(PricingError::InvalidParameter(
            "FFT size must be a positive power of two",
        ));
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = sign * TAU / (len as f64);
        let wlen_re = angle.cos();
        let wlen_im = angle.sin();
        let mut start = 0usize;
        while start < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for k in 0..half {
                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + half] * w_re - im[start + k + half] * w_im;
                let v_im = re[start + k + half] * w_im + im[start + k + half] * w_re;
                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + half] = u_re - v_re;
                im[start + k + half] = u_im - v_im;

                let next_re = w_re * wlen_re - w_im * wlen_im;
                let next_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_re;
                w_im = next_im;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
    Ok(())
}

/// Compound aggregate loss distribution via exponentially tilted radix-2 FFT
/// (Robertson 1992 / Embrechts & Frei 2009).
///
/// - `fft_size`: Must be a power of two `>= severity_pmf.len()`.
/// - `tilt`: Optional exponential tilt `theta >= 0` (`f_tilde_k = exp(-theta * k) * f_k`)
///   to suppress circular wrap-around aliasing. Use `None` for no tilting (`theta = 0`).
///
/// Untilting multiplies the inverse-FFT round-off at bin `k` by `exp(theta * k)`. Bins whose
/// magnitude is at or below that amplified noise floor (see [`FFT_ROUNDOFF_SAFETY_FACTOR`])
/// are set to `0` (this also removes negative round-off), and the root-sum-square of the floor
/// over retained bins is carried into the guards below as a round-off allowance.
///
/// Returns [`PricingError::OutOfBounds`] when
/// - the round-off allowance itself exceeds [`AGGREGATE_GRID_TOLERANCE`] (tilt too large for
///   the grid: reduce `tilt`; `tilt * fft_size` of about `10..=20` is typical), or
/// - wrap-around aliasing / truncation is detected: total mass differs from `1` (tilted case)
///   or the PMF mean differs from the exact compound mean `E[N] E[X]` by more than
///   [`AGGREGATE_GRID_TOLERANCE`] relative plus the round-off allowance (untilted aliasing
///   conserves mass but moves it to low indices, so the mean check is what detects it).
///   Increase `fft_size` (or `step`) when this fires.
///
/// `severity_pmf` must sum to `1` within [`AGGREGATE_GRID_TOLERANCE`] (else
/// [`PricingError::InvalidParameter`]) and is renormalized to unit mass.
pub fn fft_aggregate_loss(
    freq: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    fft_size: usize,
    tilt: Option<f64>,
) -> Result<AggregateLossDistribution, PricingError> {
    freq.validate()?;
    let severity = normalized_severity_pmf(severity_pmf)?;
    let severity_pmf = severity.as_slice();
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "step must be finite and positive",
        ));
    }
    if fft_size < 2 || !fft_size.is_power_of_two() || fft_size < severity_pmf.len() {
        return Err(PricingError::InvalidParameter(
            "fft_size must be a power of two >= severity_pmf.len()",
        ));
    }
    let theta = tilt.unwrap_or(0.0);
    if !theta.is_finite() || !(theta >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "tilt parameter must be finite and non-negative",
        ));
    }
    if theta * (fft_size as f64) > 700.0 {
        return Err(PricingError::OutOfBounds(
            "tilt * fft_size exceeds safe exponential range (700.0)",
        ));
    }

    let mut re = vec![0.0; fft_size];
    let mut im = vec![0.0; fft_size];
    for (k, &p) in severity_pmf.iter().enumerate() {
        re[k] = if theta == 0.0 {
            p
        } else {
            p * (-theta * (k as f64)).exp()
        };
    }

    radix2_fft(&mut re, &mut im, false)?;

    for k in 0..fft_size {
        let (pr, pi) = freq.pgf_complex(re[k], im[k]);
        re[k] = pr;
        im[k] = pi;
    }

    radix2_fft(&mut re, &mut im, true)?;

    let max_abs = |v: &[f64]| v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let log2_n = f64::from(fft_size.trailing_zeros());
    let tilted_noise =
        FFT_ROUNDOFF_SAFETY_FACTOR * max_abs(&im).max(log2_n * f64::EPSILON * max_abs(&re));
    let mut round_off_mass_sq = 0.0;
    let mut round_off_mean_sq = 0.0;
    for (k, val) in re.iter_mut().enumerate() {
        let amplification = if theta == 0.0 {
            1.0
        } else {
            (theta * (k as f64)).exp()
        };
        let floor = tilted_noise * amplification;
        let untilted = *val * amplification;
        if untilted <= floor {
            *val = 0.0;
        } else {
            *val = untilted;
            let x = (k as f64) * step;
            round_off_mass_sq += floor * floor;
            round_off_mean_sq += (floor * x) * (floor * x);
        }
    }
    let round_off = RoundOffAllowance {
        mass: round_off_mass_sq.sqrt(),
        mean: round_off_mean_sq.sqrt(),
    };

    let expected_mean = freq.mean()? * severity_grid_mean(severity_pmf, step);
    if round_off.mass > AGGREGATE_GRID_TOLERANCE
        || round_off.mean > AGGREGATE_GRID_TOLERANCE * expected_mean.abs().max(step)
    {
        return Err(PricingError::OutOfBounds(
            "FFT round-off amplified by exp(tilt * k) exceeds AGGREGATE_GRID_TOLERANCE; reduce tilt (tilt * fft_size of 10-20 is typical) or use panjer_recursion",
        ));
    }

    let dist = AggregateLossDistribution::from_pmf(step, re)?;
    check_grid_adequacy(
        &dist,
        expected_mean,
        round_off,
        "FFT truncation/aliasing: aggregate grid mass differs from 1 by more than AGGREGATE_GRID_TOLERANCE; increase fft_size or step",
        "FFT truncation/aliasing: aggregate mean deviates from E[N]E[X] by more than AGGREGATE_GRID_TOLERANCE; increase fft_size or step",
    )?;
    Ok(dist)
}

/// PMF length after trimming trailing exact-zero masses (at least `1`).
fn effective_support_len(pmf: &[f64]) -> usize {
    pmf.iter().rposition(|&p| p > 0.0).map_or(1, |i| i + 1)
}

/// Convolve multiple independent aggregate loss distributions sharing the same grid `step`
/// (e.g. combining attritional + large + catastrophe prospective underwriting risk) via FFT.
///
/// For the circular convolution to equal the linear convolution (no wrap-around), `fft_size`
/// must be at least the combined support `sum_i (len_i - 1) + 1`, where `len_i` is the
/// component PMF length after trimming trailing zero masses; otherwise
/// [`PricingError::InvalidParameter`] is returned. With that sizing the convolution is exact
/// (no aliasing). Each component must also be a complete distribution:
/// `|1 - total_mass()| <= AGGREGATE_GRID_TOLERANCE`, otherwise
/// [`PricingError::InvalidParameter`] is returned (a truncated component would otherwise
/// silently produce a defective convolution, e.g. `0.8 * 0.8 = 0.64` total mass).
pub fn convolve_independent_aggregates(
    components: &[&AggregateLossDistribution],
    fft_size: usize,
) -> Result<AggregateLossDistribution, PricingError> {
    if components.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one aggregate component is required",
        ));
    }
    if fft_size < 2 || !fft_size.is_power_of_two() {
        return Err(PricingError::InvalidParameter(
            "fft_size must be a power of two >= 2",
        ));
    }
    let step = components[0].step;
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "aggregate component grid step must be finite and positive",
        ));
    }
    let mut combined_support = 1usize;
    for comp in components {
        if !comp.step.is_finite() || (comp.step - step).abs() > 1e-12 * step.max(1.0) {
            return Err(PricingError::DimensionMismatch(
                "all aggregate components must share the same grid step",
            ));
        }
        if comp.pmf.iter().any(|&p| !p.is_finite() || p < 0.0) {
            return Err(PricingError::InvalidParameter(
                "aggregate component pmf entries must be finite and non-negative",
            ));
        }
        if !((1.0 - comp.total_mass()).abs() <= AGGREGATE_GRID_TOLERANCE) {
            return Err(PricingError::InvalidParameter(
                "aggregate component total mass differs from 1 by more than AGGREGATE_GRID_TOLERANCE (truncated or defective component)",
            ));
        }
        combined_support = combined_support.saturating_add(effective_support_len(&comp.pmf) - 1);
    }
    if combined_support > fft_size {
        return Err(PricingError::InvalidParameter(
            "fft_size must be >= combined support sum_i (len_i - 1) + 1 to avoid circular wrap-around",
        ));
    }

    let mut prod_re = vec![1.0; fft_size];
    let mut prod_im = vec![0.0; fft_size];
    let mut buf_re = vec![0.0; fft_size];
    let mut buf_im = vec![0.0; fft_size];

    for comp in components {
        buf_re.fill(0.0);
        buf_im.fill(0.0);
        let n = effective_support_len(&comp.pmf);
        buf_re[..n].copy_from_slice(&comp.pmf[..n]);
        radix2_fft(&mut buf_re, &mut buf_im, false)?;

        for k in 0..fft_size {
            let pr = prod_re[k] * buf_re[k] - prod_im[k] * buf_im[k];
            let pi = prod_re[k] * buf_im[k] + prod_im[k] * buf_re[k];
            prod_re[k] = pr;
            prod_im[k] = pi;
        }
    }

    radix2_fft(&mut prod_re, &mut prod_im, true)?;
    for v in &mut prod_re {
        *v = v.max(0.0);
    }

    AggregateLossDistribution::from_pmf(step, prod_re)
}
