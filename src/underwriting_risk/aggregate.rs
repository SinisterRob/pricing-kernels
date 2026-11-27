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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateLossDistribution {
    /// Monetary grid spacing `h > 0`.
    pub step: f64,
    /// Discrete probability mass function `g_k = P(S = k * step)`.
    pub pmf: Vec<f64>,
    /// Mean `E[S]` computed from the discrete PMF.
    pub mean: f64,
    /// Variance `Var(S)` computed from the discrete PMF.
    pub variance: f64,
    /// Standard deviation `sqrt(Var(S))`.
    pub std_dev: f64,
}

impl AggregateLossDistribution {
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
        let mut m2 = 0.0;
        for (k, &p) in pmf.iter().enumerate() {
            if !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "pmf contains non-finite probability",
                ));
            }
            let x = (k as f64) * step;
            mean += p * x;
            m2 += p * x * x;
        }
        let variance = (m2 - mean * mean).max(0.0);
        Ok(Self {
            step,
            pmf,
            mean,
            variance,
            std_dev: variance.sqrt(),
        })
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
    pub fn var(&self, alpha: f64) -> Result<f64, PricingError> {
        if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
            return Err(PricingError::InvalidParameter(
                "confidence level alpha must be finite and in (0, 1)",
            ));
        }
        let mut cdf = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            cdf += p;
            if cdf >= alpha - 1e-15 {
                return Ok((k as f64) * self.step);
            }
        }
        Ok(((self.pmf.len() - 1) as f64) * self.step)
    }

    /// Coherent Tail Value-at-Risk `TVaR_alpha(S) = (1 / (1 - alpha)) * integral_alpha^1 VaR_u(S) du`.
    pub fn tvar(&self, alpha: f64) -> Result<f64, PricingError> {
        if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
            return Err(PricingError::InvalidParameter(
                "confidence level alpha must be finite and in (0, 1)",
            ));
        }
        let tail = 1.0 - alpha;
        let mut cdf = 0.0;
        let mut weighted_sum = 0.0;
        for (k, &p) in self.pmf.iter().enumerate() {
            let next_cdf = cdf + p;
            if next_cdf > alpha {
                let contrib = next_cdf.min(1.0) - cdf.max(alpha);
                if contrib > 0.0 {
                    weighted_sum += contrib * (k as f64) * self.step;
                }
            }
            cdf = next_cdf;
        }
        Ok(weighted_sum / tail)
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

    let mut pmf = vec![0.0; n_bins];
    pmf[0] = (1.0 - lev[1] / step).clamp(0.0, 1.0);
    for j in 1..(n_bins - 1) {
        pmf[j] = ((2.0 * lev[j] - lev[j - 1] - lev[j + 1]) / step).max(0.0);
    }
    let prefix_sum: f64 = pmf[..(n_bins - 1)].iter().sum();
    pmf[n_bins - 1] = (1.0 - prefix_sum).max(0.0);
    Ok(pmf)
}

fn validate_severity_pmf(severity_pmf: &[f64]) -> Result<(), PricingError> {
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
    if (sum - 1.0).abs() > 1e-4 {
        return Err(PricingError::InvalidParameter(
            "severity_pmf probabilities must sum to 1.0",
        ));
    }
    Ok(())
}

/// Exact Panjer `(a, b, 0)` compound aggregate loss recursion (Panjer 1981).
pub fn panjer_recursion(
    freq: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    n_steps: usize,
) -> Result<AggregateLossDistribution, PricingError> {
    freq.validate()?;
    validate_severity_pmf(severity_pmf)?;
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

    AggregateLossDistribution::from_pmf(step, g)
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
pub fn fft_aggregate_loss(
    freq: FrequencyDistribution,
    severity_pmf: &[f64],
    step: f64,
    fft_size: usize,
    tilt: Option<f64>,
) -> Result<AggregateLossDistribution, PricingError> {
    freq.validate()?;
    validate_severity_pmf(severity_pmf)?;
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

    for (k, val) in re.iter_mut().enumerate() {
        let untitled = if theta == 0.0 {
            *val
        } else {
            *val * (theta * (k as f64)).exp()
        };
        *val = untitled.max(0.0);
    }

    AggregateLossDistribution::from_pmf(step, re)
}

/// Convolve multiple independent aggregate loss distributions sharing the same grid `step`
/// (e.g. combining attritional + large + catastrophe prospective underwriting risk) via FFT.
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
    for comp in components {
        if (comp.step - step).abs() > 1e-12 * step.max(1.0) {
            return Err(PricingError::DimensionMismatch(
                "all aggregate components must share the same grid step",
            ));
        }
        if comp.pmf.len() > fft_size {
            return Err(PricingError::InvalidParameter(
                "fft_size must be >= every component pmf length",
            ));
        }
    }

    let mut prod_re = vec![1.0; fft_size];
    let mut prod_im = vec![0.0; fft_size];
    let mut buf_re = vec![0.0; fft_size];
    let mut buf_im = vec![0.0; fft_size];

    for comp in components {
        buf_re.fill(0.0);
        buf_im.fill(0.0);
        buf_re[..comp.pmf.len()].copy_from_slice(&comp.pmf);
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
