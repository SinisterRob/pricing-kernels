//! Closed-form truncated power moments `E[X^p; X <= y]` and `E[X^p; X > y]` for the
//! [`SeverityDistribution`] families (Klugman, Panjer & Willmot, *Loss Models*, App. A).
//!
//! Every function returns `None` when no closed form applies (the caller then falls back
//! to quadrature); none of them is called for a moment that is infinite — use
//! [`power_moment_is_finite`] first.

use crate::layers::ilf::{SeverityDistribution, spliced_body_weight};
use crate::special::{
    gamma_p, gamma_q, ln_gamma, log_normal_survival, normal_cdf_exact, regularized_incomplete_beta,
};

/// `|p - alpha|` below which the single-parameter Pareto moment uses its logarithmic limit.
const SPP_LOG_LIMIT_TOL: f64 = 1e-12;
/// GPD shapes with `|xi|` at or below this are treated as exponential (matches `ilf.rs`).
const GPD_EXPONENTIAL_XI_TOL: f64 = 1e-10;
/// Log-magnitude beyond which a moment prefactor is combined in log-space.
const LOG_DOMAIN_THRESHOLD: f64 = 700.0;

/// `E[X^p] < ∞` for `p > 0` (Lomax/SPP: `p < alpha`; GPD: `xi p < 1`;
/// spliced Gamma–Pareto: `p < tail_alpha`; all other families: always).
pub(crate) fn power_moment_is_finite(sev: &SeverityDistribution, p: f64) -> bool {
    match *sev {
        SeverityDistribution::Lomax { alpha, .. }
        | SeverityDistribution::SingleParameterPareto { alpha, .. } => p < alpha,
        SeverityDistribution::GeneralizedPareto { xi, .. } => xi * p < 1.0,
        SeverityDistribution::SplicedGammaPareto { tail_alpha, .. } => p < tail_alpha,
        _ => true,
    }
}

/// `Lognormal`: `E[X^p; X <= y]` (`upper = false`) or `E[X^p; X > y]` (`upper = true`)
/// `= exp(p mu + p^2 sigma^2 / 2) Φ(±(ln y - mu - p sigma^2) / sigma)`.
fn lognormal_part(mu: f64, sigma: f64, p: f64, y: f64, upper: bool) -> f64 {
    let log_m = p.mul_add(mu, 0.5 * p * p * sigma * sigma);
    if !(y > 0.0) {
        return if upper { log_m.exp() } else { 0.0 };
    }
    let z = (y.ln() - mu - p * sigma * sigma) / sigma;
    let z = if upper { -z } else { z };
    if log_m.abs() > LOG_DOMAIN_THRESHOLD {
        (log_m + log_normal_survival(-z)).exp()
    } else {
        log_m.exp() * normal_cdf_exact(z)
    }
}

/// `Gamma(shape a, scale θ)`: `θ^p Γ(a + p) / Γ(a) · {P, Q}(a + p, y / θ)`; needs `a + p > 0`.
fn gamma_part(a: f64, theta: f64, p: f64, y: f64, upper: bool) -> Option<f64> {
    let ap = a + p;
    if !(ap > 0.0) {
        return None;
    }
    let log_g = p.mul_add(theta.ln(), ln_gamma(ap) - ln_gamma(a));
    let z = (y / theta).max(0.0);
    let frac = if upper {
        gamma_q(ap, z)
    } else {
        gamma_p(ap, z)
    };
    Some(log_g.exp() * frac)
}

/// Unscaled Lomax/beta moment pieces for `Y ~ Lomax(alpha, 1)`:
/// `E[Y^p; Y <= r]` or `E[Y^p; Y > r]` `= alpha B(p + 1, alpha - p) I_{·}(…)` with
/// `ln_r = ln r`. Needs `p > -1` and `alpha > p`.
pub(crate) fn lomax_unit_part(alpha: f64, p: f64, ln_r: f64, upper: bool) -> Option<f64> {
    let (a, b) = (p + 1.0, alpha - p);
    if !(a > 0.0 && b > 0.0) {
        return None;
    }
    // u = r / (1 + r), 1 - u = 1 / (1 + r) evaluated without overflow.
    let u = 1.0 / (1.0 + (-ln_r).exp());
    let v = 1.0 / (1.0 + ln_r.exp());
    let log_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let frac = if upper {
        regularized_incomplete_beta(v, b, a)
    } else {
        regularized_incomplete_beta(u, a, b)
    };
    if !frac.is_finite() {
        return None;
    }
    Some(alpha * log_beta.exp() * frac)
}

fn lomax_part(alpha: f64, theta: f64, p: f64, y: f64, upper: bool) -> Option<f64> {
    if !(y > 0.0) {
        return if upper {
            lomax_unit_part(alpha, p, f64::INFINITY, false).map(|m| theta.powf(p) * m)
        } else {
            Some(0.0)
        };
    }
    lomax_unit_part(alpha, p, (y / theta).ln(), upper).map(|m| theta.powf(p) * m)
}

/// `SingleParameterPareto(alpha, x_m)` truncated power moments (density `alpha x_m^alpha x^(-alpha-1)`).
fn spp_part(alpha: f64, x_m: f64, p: f64, y: f64, upper: bool) -> Option<f64> {
    let q = p - alpha;
    let scale = alpha * x_m.powf(p);
    if upper {
        if !(q < 0.0) {
            return None;
        }
        let lo = y.max(x_m);
        Some(scale * (q * (lo / x_m).ln()).exp() / -q)
    } else {
        if y <= x_m {
            return Some(0.0);
        }
        let ln_ratio = (y / x_m).ln();
        if q.abs() <= SPP_LOG_LIMIT_TOL {
            Some(scale * ln_ratio)
        } else {
            Some(scale * (q * ln_ratio).exp_m1() / q)
        }
    }
}

/// `E[X^p; X <= y]` (`upper = false`) or `E[X^p; X > y]` (`upper = true`) in closed form,
/// for `p > -1`, `y >= 0`. `None` when no closed form applies.
pub(crate) fn truncated_power_moment(
    sev: &SeverityDistribution,
    p: f64,
    y: f64,
    upper: bool,
) -> Option<f64> {
    if !(p > -1.0) || !p.is_finite() || !(y >= 0.0) || !y.is_finite() {
        return None;
    }
    let v = match sev {
        SeverityDistribution::Lognormal { mu, sigma } => {
            Some(lognormal_part(*mu, *sigma, p, y, upper))
        }
        SeverityDistribution::Exponential { lambda } => gamma_part(1.0, 1.0 / lambda, p, y, upper),
        SeverityDistribution::Gamma { shape, scale } => gamma_part(*shape, *scale, p, y, upper),
        SeverityDistribution::MixedExponential { weights, means } => {
            let mut acc = 0.0;
            for (&w, &m) in weights.iter().zip(means.iter()) {
                acc += w * gamma_part(1.0, m, p, y, upper)?;
            }
            Some(acc)
        }
        SeverityDistribution::Lomax { alpha, theta } => lomax_part(*alpha, *theta, p, y, upper),
        SeverityDistribution::GeneralizedPareto { xi, beta } => {
            if xi.abs() <= GPD_EXPONENTIAL_XI_TOL {
                gamma_part(1.0, *beta, p, y, upper)
            } else {
                lomax_part(1.0 / xi, beta / xi, p, y, upper)
            }
        }
        SeverityDistribution::SingleParameterPareto { alpha, x_m } => {
            spp_part(*alpha, *x_m, p, y, upper)
        }
        SeverityDistribution::SplicedGammaPareto {
            shape,
            scale,
            splice,
            tail_alpha,
        } => {
            let (w, f_u) = spliced_body_weight(*shape, *scale, *splice, *tail_alpha);
            if !(f_u > 0.0) {
                return None;
            }
            let ap = shape + p;
            if !(ap > 0.0) {
                return None;
            }
            let g = p.mul_add(scale.ln(), ln_gamma(ap) - ln_gamma(*shape)).exp();
            let p_u = gamma_p(ap, splice / scale);
            let p_y = gamma_p(ap, y.min(*splice) / scale);
            let body = (w / f_u) * g * if upper { p_u - p_y } else { p_y };
            let tail = (1.0 - w) * spp_part(*tail_alpha, *splice, p, y, upper)?;
            Some(body + tail)
        }
    }?;
    v.is_finite().then_some(v.max(0.0))
}
