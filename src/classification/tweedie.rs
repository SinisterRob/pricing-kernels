//! Exact Tweedie compound Poisson–Gamma log-density (`1 < p < 2`) via the
//! Dunn & Smyth (2005) series, plus profile maximum-likelihood dispersion.
//!
//! Reference: P. K. Dunn & G. K. Smyth, *"Series Evaluation of Tweedie Exponential
//! Dispersion Model Densities"*, *Statistics and Computing* 15(4) (2005), pp. 267–280.
//!
//! For `y > 0` the density is
//! `f(y; μ, φ, p) = (1/y) · W(y, φ, p) · exp{(y θ − κ(θ)) / φ}` with
//! `θ = μ^{1−p}/(1−p)`, `κ(θ) = μ^{2−p}/(2−p)`, `α = (2−p)/(1−p) < 0`, and
//! `W = Σ_{j≥1} W_j`, `W_j = z^j / (j! Γ(−jα))`,
//! `z = y^{−α} (p−1)^{α} / (φ^{1−α} (2−p))`.
//! The series is summed in log space outward from its modal index
//! `j_max = y^{2−p} / (φ (2−p))` until terms fall below `1e-17` of the peak.
//! At `y = 0` the point mass is `P(Y = 0) = exp{−μ^{2−p} / (φ (2−p))}`.

use crate::error::PricingError;
use crate::special::ln_gamma;

/// Relative magnitude (vs. the modal term) below which series terms are dropped.
const SERIES_REL_TOL: f64 = 1e-17;
/// Hard cap on series terms evaluated on each side of the mode.
const SERIES_MAX_TERMS: usize = 1_000_000;
/// Half-width (in `ln φ` units) of the golden-section bracket around the moment start.
const LN_PHI_BRACKET_HALF_WIDTH: f64 = 12.0;
/// Golden-section iterations on `ln φ` (final bracket width `24 · 0.618^80 ≈ 5e-16`).
const LN_PHI_GOLDEN_ITERS: usize = 80;

/// Exact log-density `ln f(y; μ, φ, p)` of the Tweedie compound Poisson–Gamma
/// distribution for `1 < p < 2` (point mass at `y = 0`, continuous density for `y > 0`),
/// evaluated with the Dunn & Smyth (2005) series.
pub fn tweedie_log_density(y: f64, mu: f64, phi: f64, power: f64) -> Result<f64, PricingError> {
    TweedieSeries::new(power)?.log_density(y, mu, phi)
}

/// Dunn & Smyth series evaluator for a fixed power `p`, caching the
/// `y`/`μ`/`φ`-independent constants `ln j! + ln Γ(−jα)` across evaluations.
pub(crate) struct TweedieSeries {
    power: f64,
    alpha: f64,
    /// `ln_const[j] = ln Γ(j + 1) + ln Γ(−j α)` (index `0` unused).
    ln_const: Vec<f64>,
}

impl TweedieSeries {
    /// Validate `power ∈ (1, 2)` and create an empty constant cache.
    pub(crate) fn new(power: f64) -> Result<Self, PricingError> {
        if !(power.is_finite() && power > 1.0 && power < 2.0) {
            return Err(PricingError::InvalidParameter(
                "Tweedie power must lie strictly in (1, 2)",
            ));
        }
        Ok(Self {
            power,
            alpha: (2.0 - power) / (1.0 - power),
            ln_const: vec![0.0],
        })
    }

    #[inline]
    fn ln_const(&mut self, j: usize) -> f64 {
        while self.ln_const.len() <= j {
            let jf = self.ln_const.len() as f64;
            self.ln_const
                .push(ln_gamma(jf + 1.0) + ln_gamma(-jf * self.alpha));
        }
        self.ln_const[j]
    }

    /// `ln f(y; μ, φ, p)`.
    pub(crate) fn log_density(&mut self, y: f64, mu: f64, phi: f64) -> Result<f64, PricingError> {
        if !(y.is_finite() && y >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "Tweedie response must be finite and >= 0",
            ));
        }
        if !(mu.is_finite() && mu > 0.0 && phi.is_finite() && phi > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "Tweedie mean and dispersion must be finite and > 0",
            ));
        }
        let power = self.power;
        let alpha = self.alpha;
        let two_m_p = 2.0 - power;
        let one_m_p = 1.0 - power;
        let lambda = mu.powf(two_m_p) / (phi * two_m_p);
        if y == 0.0 {
            return Ok(-lambda);
        }
        let ln_z =
            -alpha * y.ln() + alpha * (power - 1.0).ln() - (1.0 - alpha) * phi.ln() - two_m_p.ln();

        let j_mode_f = (y.powf(two_m_p) / (phi * two_m_p)).round().max(1.0);
        if !(j_mode_f < SERIES_MAX_TERMS as f64) {
            return Err(PricingError::NonConvergence(
                "Tweedie series modal index exceeds SERIES_MAX_TERMS",
            ));
        }
        let j_mode = j_mode_f as usize;
        let log_peak = j_mode_f * ln_z - self.ln_const(j_mode);
        if !log_peak.is_finite() {
            return Err(PricingError::NonConvergence(
                "Tweedie series modal term is not finite",
            ));
        }
        let cutoff = log_peak + SERIES_REL_TOL.ln();
        let mut sum = 1.0; // modal term, scaled by exp(-log_peak)

        let mut converged_up = false;
        for j in (j_mode + 1)..(j_mode + 1 + SERIES_MAX_TERMS) {
            let lt = j as f64 * ln_z - self.ln_const(j);
            if lt < cutoff {
                converged_up = true;
                break;
            }
            sum += (lt - log_peak).exp();
        }
        for j in (1..j_mode).rev() {
            let lt = j as f64 * ln_z - self.ln_const(j);
            if lt < cutoff {
                break;
            }
            sum += (lt - log_peak).exp();
        }
        if !converged_up {
            return Err(PricingError::NonConvergence(
                "Tweedie series did not converge within SERIES_MAX_TERMS",
            ));
        }
        let ln_w = log_peak + sum.ln();
        let theta = mu.powf(one_m_p) / one_m_p;
        let kappa = mu.powf(two_m_p) / two_m_p;
        let ll = -y.ln() + ln_w + (y * theta - kappa) / phi;
        if ll.is_finite() {
            Ok(ll)
        } else {
            Err(PricingError::NonConvergence(
                "Tweedie series log-density is not finite",
            ))
        }
    }
}

/// Weighted Tweedie log-likelihood `Σ ln f(y_i; μ_i, φ / w_i, p)` over rows with `w_i > 0`
/// (prior weights scale the dispersion, as in the exponential-dispersion GLM convention).
pub(crate) fn tweedie_log_likelihood(
    series: &mut TweedieSeries,
    y: &[f64],
    mu: &[f64],
    weights: &[f64],
    phi: f64,
) -> Result<f64, PricingError> {
    let mut ll = 0.0;
    for i in 0..y.len() {
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        ll += series.log_density(y[i], mu[i], phi / wi)?;
    }
    Ok(ll)
}

/// Maximize a log-likelihood `f(φ)` over the dispersion `φ > 0` by golden-section search on
/// `ln φ` in `[ln φ_0 − H, ln φ_0 + H]` (`H =` [`LN_PHI_BRACKET_HALF_WIDTH`]).
///
/// Returns `(φ̂, f(φ̂))`. Errors if `phi_start` is not finite and positive, or if `f`
/// fails or is non-finite at the optimum.
pub(crate) fn maximize_over_dispersion<F>(
    phi_start: f64,
    mut f: F,
) -> Result<(f64, f64), PricingError>
where
    F: FnMut(f64) -> Result<f64, PricingError>,
{
    if !(phi_start.is_finite() && phi_start > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "dispersion start value must be finite and > 0",
        ));
    }
    let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
    let centre = phi_start.ln();
    let mut a = centre - LN_PHI_BRACKET_HALF_WIDTH;
    let mut b = centre + LN_PHI_BRACKET_HALF_WIDTH;
    // Treat evaluation failures (e.g., series overflow at extreme φ) as -inf so the search
    // moves away from them; the optimum itself must evaluate cleanly.
    let mut eval = |lp: f64| -> f64 { f(lp.exp()).unwrap_or(f64::NEG_INFINITY) };
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = eval(c);
    let mut fd = eval(d);
    for _ in 0..LN_PHI_GOLDEN_ITERS {
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = eval(d);
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = eval(c);
        }
    }
    let phi_hat = f64::midpoint(a, b).exp();
    let ll = f(phi_hat)?;
    if !ll.is_finite() {
        return Err(PricingError::NonConvergence(
            "profile log-likelihood is not finite at the dispersion optimum",
        ));
    }
    Ok((phi_hat, ll))
}
