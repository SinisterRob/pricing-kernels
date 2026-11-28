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
//! Uses exact closed-form formulas for proportional ALAE (`A = k X`), adaptive
//! 15-point Gauss–Kronrod quadrature (`G7-K15`) for general conditional ALAE
//! expectations `m(x) = E[A | X = x]`, and — for ALAE-included limits with a
//! random conditional ALAE (`PowerLawLognormal`, `BivariateLognormal`) — the
//! exact 1-D reduction
//! `E[(X + A) ∧ L] = E[X ∧ L] + ∫_0^L E[A ∧ (L - x) | X = x] f_X(x) dx`
//! with the inner conditional lognormal limited expected value in closed form.

use crate::error::PricingError;
use crate::layers::SeverityDistribution;
use crate::special::normal_cdf_exact;
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
// 4.4 Adaptive 15-Point Gauss–Kronrod Quadrature (G7-K15)
// ===========================================================================

const GK15_X: [f64; 8] = [
    0.0,
    0.207_784_955_007_898_5,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_5,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

const GK15_W: [f64; 8] = [
    0.209_482_141_084_727_82,
    0.204_432_940_075_298_89,
    0.190_350_578_064_785_42,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_18,
    0.063_092_092_629_978_55,
    0.022_935_322_010_529_224,
];

const G7_W: [f64; 4] = [
    0.417_959_183_673_469_4,
    0.381_830_050_505_118_9,
    0.279_705_391_489_276_64,
    0.129_484_966_168_869_7,
];

fn gk15_panel<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let mid = f64::midpoint(a, b);
    let half = 0.5 * (b - a);
    let f_mid = f(mid);
    let mut k15 = GK15_W[0] * f_mid;
    let mut g7 = G7_W[0] * f_mid;
    for i in 1..8 {
        let dx = half * GK15_X[i];
        let f_sum = f(mid - dx) + f(mid + dx);
        k15 += GK15_W[i] * f_sum;
        if i % 2 == 0 {
            g7 += G7_W[i / 2] * f_sum;
        }
    }
    (k15 * half, (k15 - g7).abs() * half)
}

fn gk15_adaptive<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64, depth: usize) -> f64 {
    let (val, err) = gk15_panel(f, a, b);
    if err <= tol.max(1e-10 * val.abs()) || depth == 0 || (b - a).abs() <= 1e-14 {
        return val;
    }
    let mid = f64::midpoint(a, b);
    gk15_adaptive(f, a, mid, 0.5 * tol, depth - 1) + gk15_adaptive(f, mid, b, 0.5 * tol, depth - 1)
}

/// Integrate `f(x)` on `[a, b]` using adaptive 15-point Gauss–Kronrod quadrature.
pub fn gauss_kronrod_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }
    if (b - a).abs() <= 1e-15 {
        return 0.0;
    }
    gk15_adaptive(&f, a, b, 1e-11, 20)
}

/// Integrate `f(x)` on `[lower, ∞)` via transformation `x = lower + (1 - t) / t`, `t ∈ (0, 1]`.
pub fn gauss_kronrod_integrate_semi_infinite<F: Fn(f64) -> f64>(f: F, lower: f64) -> f64 {
    if !lower.is_finite() {
        return f64::NAN;
    }
    let transformed = |t: f64| -> f64 {
        if !(t > 1e-14) {
            return 0.0;
        }
        let x = lower + (1.0 - t) / t;
        let val = f(x) / (t * t);
        if val.is_finite() { val } else { 0.0 }
    };
    gk15_adaptive(&transformed, 0.0, 1.0, 1e-11, 20)
}

// ===========================================================================
// Conditional Lognormal Limited Expected Value & 1-D ALAE-Included Reduction
// ===========================================================================

/// Relative tolerance target of the 1-D ALAE-included quadratures.
const INCLUDED_QUAD_REL_TOL: f64 = 1e-12;
/// Maximum bisection depth of the adaptive ALAE-included quadratures.
const INCLUDED_QUAD_MAX_DEPTH: usize = 30;
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

/// Adaptive Gauss–Kronrod with an absolute tolerance `tol_abs` (relative to the caller's
/// magnitude scale) and an explicit depth cap.
fn integrate_to_tol<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol_abs: f64) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    gk15_adaptive(&f, a, b, tol_abs, INCLUDED_QUAD_MAX_DEPTH)
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

fn severity_pdf(severity: &SeverityDistribution, x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return 0.0;
    }
    match severity {
        SeverityDistribution::Lognormal { mu, sigma } => {
            let z = (x.ln() - mu) / sigma;
            (-0.5 * z * z).exp() / (x * sigma * std::f64::consts::TAU.sqrt())
        }
        SeverityDistribution::Lomax { alpha, theta } => {
            (alpha / theta) * (1.0 + x / theta).powf(-(alpha + 1.0))
        }
        SeverityDistribution::Exponential { lambda } => lambda * (-lambda * x).exp(),
        _ => {
            let h = (1e-6 * x).max(1e-9);
            let s_hi = severity.survival(x + h).unwrap_or(0.0);
            let s_lo = severity.survival((x - h).max(0.0)).unwrap_or(1.0);
            let denom = (x + h) - (x - h).max(0.0);
            ((s_lo - s_hi) / denom).max(0.0)
        }
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
                    let below = gauss_kronrod_integrate(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            if x > 0.0 && pdf.is_finite() {
                                m_fn(x) * pdf
                            } else {
                                0.0
                            }
                        },
                        0.0,
                        limit,
                    );
                    let above = gauss_kronrod_integrate_semi_infinite(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            if x > 0.0 && pdf.is_finite() {
                                (m_fn(x) / x) * pdf
                            } else {
                                0.0
                            }
                        },
                        limit,
                    );
                    Ok(lev_l + below + limit * above)
                }
                AlaePolicyTreatment::InAddition => {
                    let e_a = gauss_kronrod_integrate_semi_infinite(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            if x > 0.0 && pdf.is_finite() {
                                m_fn(x) * pdf
                            } else {
                                0.0
                            }
                        },
                        0.0,
                    );
                    Ok(lev_l + e_a)
                }
                AlaePolicyTreatment::Included if cv > 0.0 && c > 0.0 => {
                    // E[(X + A) ∧ L] = E[X ∧ L] + ∫_0^L E[A ∧ (L - x) | x] f(x) dx with
                    // A | x Lognormal(mean c x^gamma, CV cv) in closed form.
                    let alae_part = gauss_kronrod_integrate(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            let mean_a = m_fn(x);
                            if !(x > 0.0) || !pdf.is_finite() || !(mean_a > 0.0) {
                                return 0.0;
                            }
                            let (mu_c, sigma_c) = lognormal_from_mean_cv(mean_a, cv);
                            lognormal_lev(limit - x, mu_c, sigma_c) * pdf
                        },
                        0.0,
                        limit,
                    );
                    Ok(lev_l + alae_part)
                }
                AlaePolicyTreatment::Included => {
                    // Analytic kink-split: solve x* + c * (x*)^gamma = limit (monotone
                    // strictly increasing on [0, limit]), integrate smoothly on [0, x*],
                    // and add limit * S(x*) analytically to avoid integrand kink error.
                    let x_star = if c == 0.0 {
                        limit
                    } else {
                        let mut lo = 0.0_f64;
                        let mut hi = limit;
                        for _ in 0..80 {
                            let mid = f64::midpoint(lo, hi);
                            if mid + m_fn(mid) < limit {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                            if (hi - lo) <= 1e-13 * hi.max(1.0) {
                                break;
                            }
                        }
                        f64::midpoint(lo, hi)
                    };
                    let body = gauss_kronrod_integrate(
                        |x| {
                            let pdf = severity_pdf(severity, x);
                            if x > 0.0 && pdf.is_finite() {
                                (x + m_fn(x)) * pdf
                            } else {
                                0.0
                            }
                        },
                        0.0,
                        x_star,
                    );
                    let tail = limit * severity.survival(x_star)?;
                    Ok(body + tail)
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
                    let scale = e_a.min(limit);
                    let alae_part =
                        integrate_to_tol(integrand, z_lo, z_l, INCLUDED_QUAD_REL_TOL * scale);
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
