//! ALAE-aware Excess & Increased Limits Factor (ILF) pricing kernels
//! (Miccolis 1977; ISO proportional-ALAE and power-law / bivariate-Lognormal
//! joint Loss–ALAE severity models).
//!
//! Supports all three standard policy ALAE treatments:
//!
//! * **ALAE-Included (Eroding Limits):** `E[(X + A) ∧ L]`.
//! * **ALAE-Pro-Rata (Defense Outside Limits, Proportional):** `E[(X ∧ L) + A * (X ∧ L) / X]`.
//! * **ALAE-In-Addition (Uncapped Defense Outside Limits):** `E[X ∧ L] + E[A]`.
//!
//! Uses exact closed-form formulas for proportional ALAE (`A = k X`), closed-form
//! truncated power moments `E[X^p; X <= y]`, `E[X^p; X > y]` for power-law ALAE
//! (globally adaptive, scale-aware 15-point Gauss–Kronrod quadrature only as a
//! fallback), and — for ALAE-included limits with a
//! random conditional ALAE (`PowerLawLognormal`, `BivariateLognormal`) — the
//! exact 1-D reduction
//! `E[(X + A) ∧ L] = E[X ∧ L] + ∫_0^L E[A ∧ (L - x) | X = x] f_X(x) dx`
//! with the inner conditional lognormal limited expected value in closed form.

use crate::error::PricingError;
use crate::layers::SeverityDistribution;
use crate::layers::moments::{power_moment_is_finite, truncated_power_moment};
use crate::layers::quadrature::{QUAD_REL_TOL, integrate_finite, integrate_semi_infinite};
use crate::special::{ln_gamma, normal_cdf_exact};
use crate::validate::{require_finite, require_non_negative, require_positive};
use serde::{Deserialize, Serialize};

// ===========================================================================
// Public Types
// ===========================================================================

/// Policy ALAE treatment relative to the policy limit `L`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlaePolicyTreatment {
    /// Eroding limits: policy limit `L` applies to `X + A`.
    Included,
    /// Defense outside limits, proportional to indemnity share: `(X ∧ L) + A * (X ∧ L) / X`.
    ProRata,
    /// Uncapped defense outside limits: `(X ∧ L) + A`.
    InAddition,
}

/// Dependence model linking claim ALAE `A` to ground-up indemnity `X`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AlaeDependenceModel {
    /// Miccolis (1977) / ISO proportional ALAE: `A = k * X` (`k >= 0`).
    Proportional {
        /// Constant ALAE-to-loss ratio `k >= 0`.
        k: f64,
    },
    /// Deterministic power-law ALAE: `A = c * X^gamma` exactly (`c >= 0, gamma ∈ (0, 1]`).
    ///
    /// `ProRata` and `InAddition` are linear in `A`, so their results hold for *any*
    /// conditional ALAE distribution with mean `E[A | X = x] = c x^gamma`. `Included`
    /// (`E[(X + A) ∧ L]`) is concave in `A`; treating `A` as deterministic overstates it
    /// by Jensen's inequality when `A | X` is dispersed — use
    /// [`AlaeDependenceModel::PowerLawLognormal`] in that case.
    PowerLaw {
        /// Scale coefficient `c >= 0`.
        c: f64,
        /// Power exponent `gamma ∈ (0, 1]`.
        gamma: f64,
    },
    /// Power-law conditional mean with Lognormal conditional dispersion:
    /// `A | X = x ~ Lognormal` with `E[A | X = x] = c * x^gamma` and coefficient of
    /// variation `cv >= 0` (`cv = 0` is [`AlaeDependenceModel::PowerLaw`]).
    PowerLawLognormal {
        /// Scale coefficient `c >= 0`.
        c: f64,
        /// Power exponent `gamma ∈ (0, 1]`.
        gamma: f64,
        /// Conditional coefficient of variation `CV[A | X = x] >= 0`.
        cv: f64,
    },
    /// Joint bivariate Lognormal `(ln X, ln A) ~ N((mu_x, mu_a), Σ)` with correlation `rho ∈ (-1, 1)`.
    ///
    /// The model owns the indemnity marginal: the `severity` argument of the ALAE kernels
    /// must be `SeverityDistribution::Lognormal { mu: mu_x, sigma: sigma_x }`
    /// (see [`AlaeDependenceModel::indemnity_marginal`]); anything else is rejected.
    BivariateLognormal {
        /// Log-indemnity mean `mu_x`.
        mu_x: f64,
        /// Log-indemnity SD `sigma_x > 0`.
        sigma_x: f64,
        /// Log-ALAE mean `mu_a`.
        mu_a: f64,
        /// Log-ALAE SD `sigma_a > 0`.
        sigma_a: f64,
        /// Correlation `rho ∈ (-1, 1)` between `ln X` and `ln A`.
        rho: f64,
    },
}

/// Output of [`alae_aware_ilf`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlaeIlfResult {
    /// Basic limit `B`.
    pub basic_limit: f64,
    /// Increased limit `L`.
    pub increased_limit: f64,
    /// ALAE policy treatment evaluated.
    pub treatment: AlaePolicyTreatment,
    /// Loss-and-ALAE severity at basic limit `LAS_τ(B)`.
    pub basic_las: f64,
    /// Loss-and-ALAE severity at increased limit `LAS_τ(L)`.
    pub increased_las: f64,
    /// ALAE-aware Increased Limits Factor `ILF_τ(L) = LAS_τ(L) / LAS_τ(B)`.
    pub ilf: f64,
}

// ===========================================================================
// 4.4 Globally Adaptive, Scale-Aware 15-Point Gauss–Kronrod Quadrature (G7-K15)
// ===========================================================================

/// Integrate `f(x)` on `[a, b]` (finite) with globally adaptive 15-point Gauss–Kronrod
/// quadrature started from breakpoints graded geometrically (14 decades) towards both
/// endpoints, so integrands whose mass sits on a scale much smaller than `|b - a|` are
/// resolved. Target relative accuracy `1e-12` of the global estimate.
///
/// # Errors
/// [`PricingError::InvalidParameter`] for non-finite bounds or a non-finite integrand value;
/// [`PricingError::NonConvergence`] if the global error estimate cannot be brought below
/// `1e-9` relative.
pub fn gauss_kronrod_integrate<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
) -> Result<f64, PricingError> {
    integrate_finite(f, a, b, &[], 0.0)
}

/// Integrate `f(x)` on `[lower, ∞)` via `x = lower + scale (1 - t) / t`, `t ∈ (0, 1]`,
/// with breakpoints at `x - lower = scale · 10^j`, `|j| <= 14`. `scale` should be the
/// integrand's natural scale (e.g. the severity median); any value within a few decades works.
///
/// # Errors
/// [`PricingError::InvalidParameter`] for a non-finite `lower`, a non-positive or non-finite
/// `scale`, or a non-finite integrand value; [`PricingError::NonConvergence`] as for
/// [`gauss_kronrod_integrate`].
pub fn gauss_kronrod_integrate_semi_infinite<F: Fn(f64) -> f64>(
    f: F,
    lower: f64,
    scale: f64,
) -> Result<f64, PricingError> {
    integrate_semi_infinite(f, lower, scale, &[], 0.0)
}

// ===========================================================================
// Conditional Lognormal Limited Expected Value & 1-D ALAE-Included Reduction
// ===========================================================================

/// Standard-normal half-width beyond which the integrand mass is below `1e-31`.
const STD_NORMAL_TAIL_Z: f64 = 12.0;
/// Relative tolerance for matching `severity` to the bivariate-Lognormal marginal.
const MARGINAL_MATCH_TOL: f64 = 1e-12;

/// Lognormal limited expected value `E[A ∧ m]` for `ln A ~ N(mu, sigma^2)`.
fn lognormal_lev(m: f64, mu: f64, sigma: f64) -> f64 {
    if !(m > 0.0) {
        return 0.0;
    }
    let z = (m.ln() - mu) / sigma;
    let mean = (mu + 0.5 * sigma * sigma).exp();
    mean * normal_cdf_exact(z - sigma) + m * normal_cdf_exact(-z)
}

/// Lognormal `(mu, sigma)` with mean `mean > 0` and coefficient of variation `cv > 0`.
fn lognormal_from_mean_cv(mean: f64, cv: f64) -> (f64, f64) {
    let s2 = (cv * cv).ln_1p();
    (mean.ln() - 0.5 * s2, s2.sqrt())
}

// ===========================================================================
// Core ALAE-Aware Loss-and-ALAE Severity & ILF Kernels
// ===========================================================================

fn validate_dependence(dep: AlaeDependenceModel) -> Result<(), PricingError> {
    match dep {
        AlaeDependenceModel::Proportional { k } => {
            require_non_negative(k, "proportional ALAE ratio k must be >= 0")
        }
        AlaeDependenceModel::PowerLaw { c, gamma } => validate_power_law(c, gamma),
        AlaeDependenceModel::PowerLawLognormal { c, gamma, cv } => {
            validate_power_law(c, gamma)?;
            require_non_negative(cv, "power-law conditional CV must be >= 0")
        }
        AlaeDependenceModel::BivariateLognormal {
            mu_x,
            sigma_x,
            mu_a,
            sigma_a,
            rho,
        } => {
            require_finite(mu_x, "mu_x must be finite")?;
            require_finite(mu_a, "mu_a must be finite")?;
            require_positive(sigma_x, "sigma_x must be > 0")?;
            require_positive(sigma_a, "sigma_a must be > 0")?;
            if !(rho.is_finite() && -1.0 < rho && rho < 1.0) {
                return Err(PricingError::InvalidParameter(
                    "bivariate lognormal correlation rho must lie in (-1, 1)",
                ));
            }
            Ok(())
        }
    }
}

fn validate_power_law(c: f64, gamma: f64) -> Result<(), PricingError> {
    require_non_negative(c, "power-law scale c must be >= 0")?;
    if !(gamma.is_finite() && gamma > 0.0 && gamma <= 1.0) {
        return Err(PricingError::InvalidParameter(
            "power-law exponent gamma must lie in (0, 1]",
        ));
    }
    Ok(())
}

impl AlaeDependenceModel {
    /// Indemnity marginal owned by the dependence model, if any
    /// (`Some(Lognormal { mu_x, sigma_x })` for `BivariateLognormal`, else `None`).
    pub const fn indemnity_marginal(&self) -> Option<SeverityDistribution> {
        match *self {
            Self::BivariateLognormal { mu_x, sigma_x, .. } => {
                Some(SeverityDistribution::Lognormal {
                    mu: mu_x,
                    sigma: sigma_x,
                })
            }
            _ => None,
        }
    }
}

fn require_matching_marginal(
    severity: &SeverityDistribution,
    mu_x: f64,
    sigma_x: f64,
) -> Result<(), PricingError> {
    let matches = match *severity {
        SeverityDistribution::Lognormal { mu, sigma } => {
            (mu - mu_x).abs() <= MARGINAL_MATCH_TOL * mu_x.abs().max(1.0)
                && (sigma - sigma_x).abs() <= MARGINAL_MATCH_TOL * sigma_x
        }
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(PricingError::InvalidParameter(
            "BivariateLognormal owns the indemnity marginal: severity must equal Lognormal { mu: mu_x, sigma: sigma_x }",
        ))
    }
}

/// GPD shapes with `|xi|` at or below this are treated as exponential (matches `ilf.rs`).
const GPD_EXPONENTIAL_XI_TOL: f64 = 1e-10;
/// Bisection iterations for the ALAE-included kink `x* + c (x*)^gamma = L`.
const KINK_BISECTION_MAX_ITER: usize = 200;
/// Relative bracket width at which the kink bisection stops.
const KINK_BISECTION_REL_TOL: f64 = 1e-15;

fn gamma_pdf(shape: f64, scale: f64, x: f64) -> f64 {
    ((shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - ln_gamma(shape)).exp()
}

/// Exact density `f(x)` of every [`SeverityDistribution`] family (`0` outside the support).
fn severity_pdf(severity: &SeverityDistribution, x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return 0.0;
    }
    let d = match severity {
        SeverityDistribution::Lognormal { mu, sigma } => {
            let z = (x.ln() - mu) / sigma;
            (-0.5 * z * z).exp() / (x * sigma * std::f64::consts::TAU.sqrt())
        }
        SeverityDistribution::Lomax { alpha, theta } => {
            (alpha / theta) * (-(alpha + 1.0) * (x / theta).ln_1p()).exp()
        }
        SeverityDistribution::SingleParameterPareto { alpha, x_m } => {
            if x <= *x_m {
                0.0
            } else {
                (alpha / x) * (x_m / x).powf(*alpha)
            }
        }
        SeverityDistribution::Exponential { lambda } => lambda * (-lambda * x).exp(),
        SeverityDistribution::MixedExponential { weights, means } => weights
            .iter()
            .zip(means.iter())
            .map(|(&w, &m)| (w / m) * (-x / m).exp())
            .sum(),
        SeverityDistribution::GeneralizedPareto { xi, beta } => {
            if xi.abs() <= GPD_EXPONENTIAL_XI_TOL {
                (-x / beta).exp() / beta
            } else {
                (-(1.0 / xi + 1.0) * (xi * x / beta).ln_1p()).exp() / beta
            }
        }
        SeverityDistribution::Gamma { shape, scale } => gamma_pdf(*shape, *scale, x),
        SeverityDistribution::SplicedGammaPareto {
            shape,
            scale,
            splice,
            tail_alpha,
        } => {
            let (w, f_u) =
                crate::layers::ilf::spliced_body_weight(*shape, *scale, *splice, *tail_alpha);
            if x <= *splice {
                if f_u > 0.0 {
                    (w / f_u) * gamma_pdf(*shape, *scale, x)
                } else {
                    0.0
                }
            } else {
                (1.0 - w) * (tail_alpha / x) * (splice / x).powf(*tail_alpha)
            }
        }
    };
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Natural scale points (median, mean, scale, threshold, support start, component means)
/// used to seed the quadrature subdivision.
fn severity_anchors(severity: &SeverityDistribution) -> Vec<f64> {
    match severity {
        SeverityDistribution::Lognormal { mu, sigma } => {
            vec![
                mu.exp(),
                (mu + 0.5 * sigma * sigma).exp(),
                (mu - sigma * sigma).exp(),
            ]
        }
        SeverityDistribution::Lomax { theta, .. } => vec![*theta],
        SeverityDistribution::SingleParameterPareto { x_m, .. } => vec![*x_m],
        SeverityDistribution::Exponential { lambda } => vec![1.0 / lambda],
        SeverityDistribution::MixedExponential { means, .. } => means.clone(),
        SeverityDistribution::GeneralizedPareto { xi, beta } => {
            if xi.abs() > GPD_EXPONENTIAL_XI_TOL {
                vec![*beta, beta / xi]
            } else {
                vec![*beta]
            }
        }
        SeverityDistribution::Gamma { shape, scale } => vec![*scale, shape * scale],
        SeverityDistribution::SplicedGammaPareto {
            shape,
            scale,
            splice,
            ..
        } => vec![*splice, *scale, shape * scale],
    }
}

/// `E[X^p; X <= y]` (`upper = false`) or `E[X^p; X > y]` (`upper = true`): closed form when
/// available, else scale-aware quadrature of `x^p f(x)` (caller guarantees finiteness).
fn truncated_moment(
    severity: &SeverityDistribution,
    p: f64,
    y: f64,
    upper: bool,
) -> Result<f64, PricingError> {
    if !upper && p == 1.0 {
        // E[X; X <= y] = E[X ∧ y] - y S(y), exact for every family.
        let v = severity.limited_expected_value(y)? - y * severity.survival(y)?;
        return Ok(v.max(0.0));
    }
    if let Some(v) = truncated_power_moment(severity, p, y, upper) {
        return Ok(v);
    }
    let anchors = severity_anchors(severity);
    let g = |x: f64| -> f64 {
        let d = severity_pdf(severity, x);
        if d > 0.0 { x.powf(p) * d } else { 0.0 }
    };
    if upper {
        let scale = anchors.iter().copied().fold(y, f64::max);
        integrate_semi_infinite(g, y, scale, &anchors, 0.0)
    } else {
        integrate_finite(g, 0.0, y, &anchors, 0.0)
    }
}

/// Evaluate Loss-and-ALAE Severity `LAS_τ(L)` at policy limit `L > 0` for a
/// ground-up indemnity distribution `severity`, ALAE dependence model `dependence`,
/// and policy treatment `treatment`.
///
/// For `AlaeDependenceModel::BivariateLognormal`, `severity` must equal the model's
/// marginal `Lognormal(mu_x, sigma_x)` ([`AlaeDependenceModel::indemnity_marginal`]);
/// a conflicting severity returns [`PricingError::InvalidParameter`].
pub fn alae_loss_severity(
    severity: &SeverityDistribution,
    limit: f64,
    dependence: AlaeDependenceModel,
    treatment: AlaePolicyTreatment,
) -> Result<f64, PricingError> {
    require_positive(limit, "policy limit must be > 0")?;
    validate_dependence(dependence)?;

    match dependence {
        AlaeDependenceModel::Proportional { k } => {
            let lev_l = severity.limited_expected_value(limit)?;
            match treatment {
                AlaePolicyTreatment::Included => {
                    let eff_lim = limit / (1.0 + k);
                    Ok((1.0 + k) * severity.limited_expected_value(eff_lim)?)
                }
                AlaePolicyTreatment::ProRata => Ok((1.0 + k) * lev_l),
                AlaePolicyTreatment::InAddition => {
                    let mean_x = severity.unbounded_mean()?.ok_or(
                        PricingError::InvalidParameter(
                            "ALAE-in-addition requires a severity distribution with finite mean",
                        ),
                    )?;
                    Ok(lev_l + k * mean_x)
                }
            }
        }
        AlaeDependenceModel::PowerLaw { c, gamma }
        | AlaeDependenceModel::PowerLawLognormal { c, gamma, .. } => {
            let cv = match dependence {
                AlaeDependenceModel::PowerLawLognormal { cv, .. } => cv,
                _ => 0.0,
            };
            let lev_l = severity.limited_expected_value(limit)?;
            let m_fn = |x: f64| c * x.powf(gamma);
            match treatment {
                AlaePolicyTreatment::ProRata => {
                    // E[c X^g; X <= L] + L E[c X^(g-1); X > L] (finite for g ∈ (0, 1]).
                    let below = c * truncated_moment(severity, gamma, limit, false)?;
                    let above = c * truncated_moment(severity, gamma - 1.0, limit, true)?;
                    Ok(lev_l + below + limit * above)
                }
                AlaePolicyTreatment::InAddition => {
                    if c == 0.0 {
                        return Ok(lev_l);
                    }
                    if !power_moment_is_finite(severity, gamma) {
                        return Err(PricingError::InvalidParameter(
                            "ALAE-in-addition with power-law ALAE requires a severity distribution with finite E[X^gamma]",
                        ));
                    }
                    Ok(lev_l + c * truncated_moment(severity, gamma, 0.0, true)?)
                }
                AlaePolicyTreatment::Included if cv > 0.0 && c > 0.0 => {
                    // E[(X + A) ∧ L] = E[X ∧ L] + ∫_0^L E[A ∧ (L - x) | x] f(x) dx with
                    // A | x Lognormal(mean c x^gamma, CV cv) in closed form.
                    let mut anchors = severity_anchors(severity);
                    anchors.push(limit);
                    let alae_part = integrate_finite(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            let mean_a = m_fn(x);
                            if !(x > 0.0) || !(pdf > 0.0) || !(mean_a > 0.0) {
                                return 0.0;
                            }
                            let (mu_c, sigma_c) = lognormal_from_mean_cv(mean_a, cv);
                            lognormal_lev(limit - x, mu_c, sigma_c) * pdf
                        },
                        0.0,
                        limit,
                        &anchors,
                        QUAD_REL_TOL * lev_l,
                    )?;
                    Ok(lev_l + alae_part)
                }
                AlaePolicyTreatment::Included => {
                    // Analytic kink-split: solve x* + c (x*)^gamma = L (strictly increasing),
                    // then E[(X + A) ∧ L] = E[X; X <= x*] + c E[X^gamma; X <= x*] + L S(x*).
                    let x_star = if c == 0.0 {
                        limit
                    } else {
                        let mut lo = 0.0_f64;
                        let mut hi = limit;
                        for _ in 0..KINK_BISECTION_MAX_ITER {
                            let mid = f64::midpoint(lo, hi);
                            if mid + m_fn(mid) < limit {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                            if (hi - lo) <= KINK_BISECTION_REL_TOL * hi {
                                break;
                            }
                        }
                        f64::midpoint(lo, hi)
                    };
                    let body_x = truncated_moment(severity, 1.0, x_star, false)?;
                    let body_a = if c == 0.0 {
                        0.0
                    } else {
                        c * truncated_moment(severity, gamma, x_star, false)?
                    };
                    let tail = limit * severity.survival(x_star)?;
                    Ok(body_x + body_a + tail)
                }
            }
        }
        AlaeDependenceModel::BivariateLognormal {
            mu_x,
            sigma_x,
            mu_a,
            sigma_a,
            rho,
        } => {
            require_matching_marginal(severity, mu_x, sigma_x)?;
            let lev_l = severity.limited_expected_value(limit)?;
            let e_a = (mu_a + 0.5 * sigma_a * sigma_a).exp();

            match treatment {
                AlaePolicyTreatment::InAddition => Ok(lev_l + e_a),
                AlaePolicyTreatment::ProRata => {
                    // Exact closed-form for bivariate Lognormal:
                    // E[A * I(X <= L)] = E[A] * Φ((ln L - mu_x - rho*sigma_x*sigma_a) / sigma_x)
                    // L * E[(A/X) * I(X > L)]: ln(A/X) has mean mu_a - mu_x,
                    // cov(ln(A/X), ln X) = rho*sigma_a*sigma_x - sigma_x^2.
                    let z_below = (limit.ln() - mu_x - rho * sigma_x * sigma_a) / sigma_x;
                    let below = e_a * normal_cdf_exact(z_below);

                    let mu_r = mu_a - mu_x;
                    let var_r =
                        sigma_a * sigma_a + sigma_x * sigma_x - 2.0 * rho * sigma_a * sigma_x;
                    let e_ratio = (mu_r + 0.5 * var_r).exp();
                    let cov_rx = rho * sigma_a * sigma_x - sigma_x * sigma_x;
                    let z_above = (limit.ln() - mu_x - cov_rx) / sigma_x;
                    let above = limit * e_ratio * normal_cdf_exact(-z_above);
                    Ok(lev_l + below + above)
                }
                AlaePolicyTreatment::Included => {
                    // Exact 1-D reduction in z = (ln x - mu_x) / sigma_x:
                    // E[(X + A) ∧ L] = E[X ∧ L] + ∫_{-∞}^{z_L} E[A ∧ (L - x(z)) | z] φ(z) dz,
                    // ln A | z ~ N(mu_a + rho sigma_a z, sigma_a^2 (1 - rho^2)).
                    // The integrand is smooth on (-∞, z_L) (the kink x = L is the endpoint).
                    let cond_sd = sigma_a * (1.0 - rho * rho).sqrt();
                    let z_l = (limit.ln() - mu_x) / sigma_x;
                    // E[A | z] φ(z) = E[A] φ(z - rho sigma_a): truncate below that centre.
                    let z_lo = z_l.min(rho * sigma_a) - STD_NORMAL_TAIL_Z;
                    let inv_sqrt_tau = 1.0 / std::f64::consts::TAU.sqrt();
                    let integrand = |z: f64| -> f64 {
                        let x = sigma_x.mul_add(z, mu_x).exp();
                        let mu_c = rho.mul_add(sigma_a * z, mu_a);
                        let phi = inv_sqrt_tau * (-0.5 * z * z).exp();
                        lognormal_lev(limit - x, mu_c, cond_sd) * phi
                    };
                    let alae_part =
                        integrate_finite(integrand, z_lo, z_l, &[], QUAD_REL_TOL * lev_l)?;
                    Ok(lev_l + alae_part)
                }
            }
        }
    }
}

/// Compute the ALAE-aware Increased Limits Factor `ILF_τ(L) = LAS_τ(L) / LAS_τ(B)`.
pub fn alae_aware_ilf(
    severity: &SeverityDistribution,
    basic_limit: f64,
    increased_limit: f64,
    dependence: AlaeDependenceModel,
    treatment: AlaePolicyTreatment,
) -> Result<AlaeIlfResult, PricingError> {
    require_positive(basic_limit, "basic_limit must be > 0")?;
    require_positive(increased_limit, "increased_limit must be > 0")?;
    let basic_las = alae_loss_severity(severity, basic_limit, dependence, treatment)?;
    let increased_las = alae_loss_severity(severity, increased_limit, dependence, treatment)?;
    require_positive(basic_las, "basic_las must be > 0")?;
    Ok(AlaeIlfResult {
        basic_limit,
        increased_limit,
        treatment,
        basic_las,
        increased_las,
        ilf: increased_las / basic_las,
    })
}

/// Compute ALAE-aware excess layer expected loss `LAS_τ(attachment + limit) - LAS_τ(attachment)`.
pub fn alae_excess_layer_expected_loss(
    severity: &SeverityDistribution,
    attachment: f64,
    limit: f64,
    dependence: AlaeDependenceModel,
    treatment: AlaePolicyTreatment,
) -> Result<f64, PricingError> {
    require_non_negative(attachment, "attachment must be >= 0")?;
    require_positive(limit, "layer limit must be > 0")?;
    let top = alae_loss_severity(severity, attachment + limit, dependence, treatment)?;
    let bot = if attachment > 0.0 {
        alae_loss_severity(severity, attachment, dependence, treatment)?
    } else {
        0.0
    };
    Ok((top - bot).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_pareto_ii_proportional_alae_ilf_and_ordering() {
        // Pareto II (alpha = 2, lambda = 100_000), proportional ALAE k = 0.20,
        // B = 100_000, L = 500_000.
        let sev = SeverityDistribution::Lomax {
            alpha: 2.0,
            theta: 100_000.0,
        };
        let dep = AlaeDependenceModel::Proportional { k: 0.20 };
        let b = 100_000.0;
        let l = 500_000.0;

        let pro = alae_aware_ilf(&sev, b, l, dep, AlaePolicyTreatment::ProRata).expect("pro");
        let add = alae_aware_ilf(&sev, b, l, dep, AlaePolicyTreatment::InAddition).expect("add");
        let inc = alae_aware_ilf(&sev, b, l, dep, AlaePolicyTreatment::Included).expect("inc");

        assert!((pro.basic_las - 60_000.0).abs() < 1e-6);
        assert!((pro.increased_las - 100_000.0).abs() < 1e-6);
        assert!((pro.ilf - (5.0 / 3.0)).abs() < 1e-8);

        assert!((add.basic_las - 70_000.0).abs() < 1e-6);
        assert!((add.increased_las - (83_333.333_333_333_33 + 20_000.0)).abs() < 1e-4);
        assert!((add.ilf - 1.476_190_48).abs() < 1e-6);

        assert!((inc.basic_las - 54_545.454_545).abs() < 1e-4);
        assert!((inc.increased_las - 96_774.193_548).abs() < 1e-4);
        assert!((inc.ilf - 1.774_193_55).abs() < 1e-6);

        // Exact LAS ordering: LAS_inc <= LAS_pro <= LAS_add at both B and L.
        assert!(inc.basic_las <= pro.basic_las && pro.basic_las <= add.basic_las);
        assert!(inc.increased_las <= pro.increased_las && pro.increased_las <= add.increased_las);
        // Proportional ALAE ILF ordering for Pareto II: ILF_add <= ILF_pro <= ILF_inc.
        assert!(add.ilf <= pro.ilf && pro.ilf <= inc.ilf);
    }

    #[test]
    fn power_law_and_bivariate_lognormal_satisfy_las_ordering_and_concavity() {
        let sev = SeverityDistribution::Lognormal {
            mu: 10.0,
            sigma: 1.2,
        };
        let dep_pw = AlaeDependenceModel::PowerLaw {
            c: 1.5,
            gamma: 0.75,
        };
        for lim in [50_000.0, 250_000.0, 1_000_000.0] {
            let inc =
                alae_loss_severity(&sev, lim, dep_pw, AlaePolicyTreatment::Included).expect("inc");
            let pro =
                alae_loss_severity(&sev, lim, dep_pw, AlaePolicyTreatment::ProRata).expect("pro");
            let add = alae_loss_severity(&sev, lim, dep_pw, AlaePolicyTreatment::InAddition)
                .expect("add");
            assert!(inc <= pro + 1e-6 && pro <= add + 1e-6);
        }

        let dep_biv = AlaeDependenceModel::BivariateLognormal {
            mu_x: 10.0,
            sigma_x: 1.0,
            mu_a: 8.5,
            sigma_a: 0.8,
            rho: 0.65,
        };
        let sev_biv = dep_biv.indemnity_marginal().expect("bivariate marginal");
        assert!(
            alae_loss_severity(&sev, 100_000.0, dep_biv, AlaePolicyTreatment::Included).is_err(),
            "conflicting severity must be rejected"
        );
        let l1 = 100_000.0;
        let l2 = 200_000.0;
        let l3 = 300_000.0;
        for tr in [
            AlaePolicyTreatment::Included,
            AlaePolicyTreatment::ProRata,
            AlaePolicyTreatment::InAddition,
        ] {
            let s1 = alae_loss_severity(&sev_biv, l1, dep_biv, tr).expect("s1");
            let s2 = alae_loss_severity(&sev_biv, l2, dep_biv, tr).expect("s2");
            let s3 = alae_loss_severity(&sev_biv, l3, dep_biv, tr).expect("s3");
            assert!(s1 < s2 && s2 < s3);
            // Concavity: diminishing marginal cost.
            assert!((s3 - s2) <= (s2 - s1) + 1e-6);
        }
    }
}
