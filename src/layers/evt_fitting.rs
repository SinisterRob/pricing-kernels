//! Censored & truncated Maximum Likelihood Estimation (MLE + EVT) and
//! spliced body–tail severity fitting with observed Fisher information SEs
//! and goodness-of-fit diagnostics (Klugman–Panjer–Willmot *Loss Models*, Ch. 13).
//!
//! Covers all 7 parametric families (`Lognormal`, `ParetoII`, `Gpd`, `BurrXii`,
//! `Weibull`, `Gamma`, `InverseGaussian`) plus `SplicedBodyGpd` ($C^0$ continuous
//! and $C^1$ differentiable Lognormal/Weibull body + GPD tail), with:
//!
//! * Exact left-truncated, right-censored log-likelihood in log-domain.
//! * Method-of-Moments (MoM) starting-value initializers.
//! * Deterministic L-BFGS (`m = 7`) + Nelder–Mead fallback optimizer.
//! * Observed Fisher information & delta-method standard errors.
//! * AIC, BIC (with `n = Σ w_i`), weighted Lynden-Bell / Kaplan–Meier product-limit
//!   Kolmogorov–Smirnov (`D_n`, delayed entry at each deductible), weighted PIT
//!   Anderson–Darling (`A^2`), and empirical mean-excess `e_n(u)`.
//!
//! # Parameter constraints
//!
//! * GPD shape (stand-alone and spliced tail) is constrained to `xi > -1`: for `xi < -1`
//!   the likelihood is unbounded (Smith 1985). Fits ending within `1e-3` of the bound are
//!   reported with `converged = false`. For `-1 < xi < -0.5` the MLE exists but is
//!   non-regular, so the reported standard errors are not asymptotically valid.
//! * Burr XII is optimized in `(ln alpha, ln gamma, ln theta)` with `lambda = theta^gamma`
//!   (scale-equivariant) and started from a log-logistic fit at the sample median.
//! * Any fit whose optimizer iterate lies within `1` unit of the internal `|phi| <= 30`
//!   parameter box, or whose standard errors are not finite, is reported with
//!   `converged = false`.

use crate::error::{PricingError, SolverOptions};
use crate::layers::alae_ilf::gauss_kronrod_integrate;
use crate::layers::fitting::SeverityObservation;
use crate::layers::ilf::{SeverityDistribution, ilf_from_lev};
use crate::linalg::cholesky_invert;
use crate::special::{gamma_p, ln_gamma, log_gamma_q, log_normal_survival, normal_cdf_exact};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

// ===========================================================================
// Public Distribution & Result Types
// ===========================================================================

/// Body distribution choice for spliced body + GPD tail models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpliceBodyFamily {
    /// Lognormal body on `(0, u]`.
    Lognormal,
    /// Weibull body on `(0, u]`.
    Weibull,
}

/// Continuity order enforced at the splice threshold `u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpliceContinuity {
    /// $C^0$ density continuity at `u` (eliminates weight `w`).
    #[default]
    C0Continuous,
    /// $C^1$ density & first-derivative continuity at `u` (eliminates `w` and `β`).
    C1Differentiable,
}

/// Distribution family selector for [`fit_evt_mle`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvtFamily {
    /// Lognormal `(mu, sigma)`.
    Lognormal,
    /// Pareto Type II / Lomax `(alpha, lambda)`.
    ParetoII,
    /// Generalized Pareto Distribution (POT) on excesses `y = x - threshold > 0`.
    Gpd {
        /// Peaks-Over-Threshold cutoff `u >= 0`.
        threshold: f64,
    },
    /// Burr Type XII `(alpha, gamma, lambda)`.
    BurrXii,
    /// Weibull `(k, lambda)`.
    Weibull,
    /// Gamma `(alpha, beta)` in shape–rate parameterization.
    Gamma,
    /// Inverse Gaussian `(mu, lambda)`.
    InverseGaussian,
    /// Spliced body (`Lognormal` or `Weibull`) on `(0, threshold]` + GPD tail on `(threshold, ∞)`.
    SplicedBodyGpd {
        /// Body family.
        body: SpliceBodyFamily,
        /// Splice threshold `u > 0`.
        threshold: f64,
        /// Continuity order (`C0Continuous` or `C1Differentiable`).
        continuity: SpliceContinuity,
    },
}

/// Fitted parametric parameters returned by [`fit_evt_mle`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FittedEvtDistribution {
    /// Lognormal `ln(X) ~ N(mu, sigma^2)`.
    Lognormal {
        /// Log-location `mu`.
        mu: f64,
        /// Log-scale `sigma > 0`.
        sigma: f64,
    },
    /// Pareto Type II (Lomax): `S(x) = (lambda / (x + lambda))^alpha`.
    ParetoII {
        /// Shape `alpha > 0`.
        alpha: f64,
        /// Scale `lambda > 0`.
        lambda: f64,
    },
    /// Generalized Pareto Distribution on excesses `y = x - threshold > 0`.
    Gpd {
        /// Threshold `u >= 0`.
        threshold: f64,
        /// Shape `xi`.
        xi: f64,
        /// Scale `beta > 0`.
        beta: f64,
    },
    /// Burr Type XII: `S(x) = (lambda / (lambda + x^gamma))^alpha`.
    BurrXii {
        /// Outer shape `alpha > 0`.
        alpha: f64,
        /// Power shape `gamma > 0`.
        gamma: f64,
        /// Scale `lambda > 0`.
        lambda: f64,
    },
    /// Weibull: `S(x) = exp(-(x / lambda)^k)`.
    Weibull {
        /// Shape `k > 0`.
        k: f64,
        /// Scale `lambda > 0`.
        lambda: f64,
    },
    /// Gamma in shape–rate parameterization: `f(x) = beta^alpha x^(alpha-1) e^(-beta x) / Γ(alpha)`.
    Gamma {
        /// Shape `alpha > 0`.
        alpha: f64,
        /// Rate `beta > 0`.
        beta: f64,
    },
    /// Inverse Gaussian `(mu, lambda)`.
    InverseGaussian {
        /// Mean `mu > 0`.
        mu: f64,
        /// Shape `lambda > 0`.
        lambda: f64,
    },
    /// Spliced body + GPD tail.
    SplicedBodyGpd {
        /// Body family.
        body: SpliceBodyFamily,
        /// Splice threshold `u > 0`.
        threshold: f64,
        /// Continuity order enforced.
        continuity: SpliceContinuity,
        /// First body parameter (`mu` for Lognormal, `k` for Weibull).
        body_param1: f64,
        /// Second body parameter (`sigma` for Lognormal, `lambda` for Weibull).
        body_param2: f64,
        /// Body probability weight `w = P(X <= u)`.
        body_weight: f64,
        /// GPD tail shape `xi`.
        xi: f64,
        /// GPD tail scale `beta > 0`.
        beta: f64,
    },
}

/// Goodness-of-fit diagnostics from [`fit_evt_mle`] or [`goodness_of_fit`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GoodnessOfFitDiagnostics {
    /// Akaike Information Criterion `2p - 2 * ll`.
    pub aic: f64,
    /// Bayesian Information Criterion `p * ln(n) - 2 * ll` with `n = Σ w_i`.
    pub bic: f64,
    /// Weighted product-limit (Lynden-Bell / Kaplan–Meier with delayed entry) Kolmogorov–Smirnov
    /// statistic `D_n` against the model conditional on `X > min_i d_i`.
    pub ks_statistic: f64,
    /// Weighted PIT Anderson–Darling statistic `A^2` with `n = Σ w_i`, using per-observation
    /// conditional PITs `1 - S(x_i)/S(d_i)` (`None` if censored observations are present).
    pub anderson_darling: Option<f64>,
}

/// Complete output of [`fit_evt_mle`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvtFitResult {
    /// Fitted distribution parameters.
    pub distribution: FittedEvtDistribution,
    /// Maximized left-truncated, right-censored log-likelihood.
    pub log_likelihood: f64,
    /// Delta-method standard errors of every estimated numeric field of the fitted
    /// [`FittedEvtDistribution`] variant, in declaration order, excluding fixed inputs
    /// (`threshold`, `body`, `continuity`). Derived fields (`body_weight`, and `beta` under
    /// $C^1$) are included via the full Jacobian. Names: [`FittedEvtDistribution::parameter_names`].
    pub standard_errors: Vec<f64>,
    /// Goodness-of-fit statistics (`AIC`, `BIC`, `KS`, `AD`).
    pub diagnostics: GoodnessOfFitDiagnostics,
    /// Number of observations used in the fit.
    pub n_observations: usize,
    /// Number of right-censored observations.
    pub n_censored: usize,
    /// Whether the optimizer met the convergence tolerance at an interior point with
    /// finite standard errors (see module docs for boundary rules).
    pub converged: bool,
    /// Total optimizer iterations (L-BFGS + Nelder–Mead + polish).
    #[serde(default)]
    pub iterations: usize,
}

// ===========================================================================
// Log-PDF & Log-Survival Evaluators (with Numerical Stability Guards)
// ===========================================================================

#[inline]
fn log_diff_exp(a: f64, b: f64) -> f64 {
    if b >= a {
        return f64::NEG_INFINITY;
    }
    a + (-(b - a).exp()).ln_1p()
}

/// Numerically stable `ln(1 + e^t)`.
#[inline]
fn softplus(t: f64) -> f64 {
    if t > 0.0 {
        t + (-t).exp().ln_1p()
    } else {
        t.exp().ln_1p()
    }
}

#[inline]
fn gpd_log_one_plus_xi_t_over_xi(xi: f64, t: f64) -> f64 {
    if xi.abs() < 1e-7 {
        t - 0.5 * xi * t * t + (xi * xi * t * t * t) / 3.0
    } else {
        let arg = 1.0 + xi * t;
        if !(arg > 0.0) {
            return f64::INFINITY;
        }
        arg.ln() / xi
    }
}

impl FittedEvtDistribution {
    /// Evaluate `ln f(x)` for `x > 0` (returns `-inf` outside support).
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || !(x > 0.0) {
            return f64::NEG_INFINITY;
        }
        match *self {
            Self::Lognormal { mu, sigma } => {
                let lx = x.ln();
                let z = (lx - mu) / sigma;
                -lx - sigma.ln() - 0.5 * TAU.ln() - 0.5 * z * z
            }
            Self::ParetoII { alpha, lambda } => {
                alpha.ln() + alpha * lambda.ln() - (alpha + 1.0) * (x + lambda).ln()
            }
            Self::Gpd {
                threshold,
                xi,
                beta,
            } => {
                let y = x - threshold;
                if !(y > 0.0) {
                    return f64::NEG_INFINITY;
                }
                let t = y / beta;
                let arg = 1.0 + xi * t;
                if !(arg > 0.0) {
                    return f64::NEG_INFINITY;
                }
                let term = gpd_log_one_plus_xi_t_over_xi(xi, t);
                -beta.ln() - term - arg.ln()
            }
            Self::BurrXii {
                alpha,
                gamma,
                lambda,
            } => {
                // ln(lambda + x^gamma) = ln lambda + softplus(gamma ln x - ln lambda).
                let lx = x.ln();
                let ln_lambda = lambda.ln();
                (alpha * gamma).ln() - ln_lambda + (gamma - 1.0) * lx
                    - (alpha + 1.0) * softplus(gamma.mul_add(lx, -ln_lambda))
            }
            Self::Weibull { k, lambda } => {
                let r = x / lambda;
                k.ln() - k * lambda.ln() + (k - 1.0) * x.ln() - r.powf(k)
            }
            Self::Gamma { alpha, beta } => {
                alpha * beta.ln() - ln_gamma(alpha) + (alpha - 1.0) * x.ln() - beta * x
            }
            Self::InverseGaussian { mu, lambda } => {
                0.5 * (lambda / (TAU * x * x * x)).ln()
                    - lambda * (x - mu).powi(2) / (2.0 * mu * mu * x)
            }
            Self::SplicedBodyGpd {
                body,
                threshold,
                body_param1,
                body_param2,
                body_weight,
                xi,
                beta,
                ..
            } => {
                if x <= threshold {
                    let body_dist = match body {
                        SpliceBodyFamily::Lognormal => Self::Lognormal {
                            mu: body_param1,
                            sigma: body_param2,
                        },
                        SpliceBodyFamily::Weibull => Self::Weibull {
                            k: body_param1,
                            lambda: body_param2,
                        },
                    };
                    let f1_u = body_dist.cdf(threshold).max(1e-300);
                    body_weight.max(1e-300).ln() + body_dist.log_pdf(x) - f1_u.ln()
                } else {
                    let gpd = Self::Gpd {
                        threshold,
                        xi,
                        beta,
                    };
                    (1.0 - body_weight).max(1e-300).ln() + gpd.log_pdf(x)
                }
            }
        }
    }

    /// Evaluate log survival `ln S(x)` for `x >= 0` (`ln S(0) = 0.0`).
    pub fn log_survival(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        match *self {
            Self::Lognormal { mu, sigma } => {
                let z = (x.ln() - mu) / sigma;
                log_normal_survival(z)
            }
            Self::ParetoII { alpha, lambda } => alpha * (lambda / (x + lambda)).ln(),
            Self::Gpd {
                threshold,
                xi,
                beta,
            } => {
                let y = x - threshold;
                if y <= 0.0 {
                    return 0.0;
                }
                let t = y / beta;
                if !(1.0 + xi * t > 0.0) {
                    return f64::NEG_INFINITY;
                }
                -gpd_log_one_plus_xi_t_over_xi(xi, t)
            }
            Self::BurrXii {
                alpha,
                gamma,
                lambda,
            } => -alpha * softplus(gamma.mul_add(x.ln(), -lambda.ln())),
            Self::Weibull { k, lambda } => -(x / lambda).powf(k),
            Self::Gamma { alpha, beta } => log_gamma_q(alpha, beta * x),
            Self::InverseGaussian { mu, lambda } => {
                let sq = (lambda / x).sqrt();
                let z1 = sq * (x / mu - 1.0);
                let z2 = sq * (x / mu + 1.0);
                let a = log_normal_survival(z1);
                let b = (2.0 * lambda / mu) + log_normal_survival(z2);
                log_diff_exp(a, b)
            }
            Self::SplicedBodyGpd {
                body,
                threshold,
                body_param1,
                body_param2,
                body_weight,
                xi,
                beta,
                ..
            } => {
                if x <= threshold {
                    let body_dist = match body {
                        SpliceBodyFamily::Lognormal => Self::Lognormal {
                            mu: body_param1,
                            sigma: body_param2,
                        },
                        SpliceBodyFamily::Weibull => Self::Weibull {
                            k: body_param1,
                            lambda: body_param2,
                        },
                    };
                    let f1_u = body_dist.cdf(threshold).max(1e-300);
                    let f_x = body_weight * (body_dist.cdf(x) / f1_u).clamp(0.0, 1.0);
                    (1.0 - f_x).max(1e-300).ln()
                } else {
                    let gpd = Self::Gpd {
                        threshold,
                        xi,
                        beta,
                    };
                    (1.0 - body_weight).max(1e-300).ln() + gpd.log_survival(x)
                }
            }
        }
    }

    /// Evaluate CDF `F(x) = 1 - S(x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        match *self {
            Self::Lognormal { mu, sigma } => normal_cdf_exact((x.ln() - mu) / sigma),
            Self::Gamma { alpha, beta } => gamma_p(alpha, beta * x),
            _ => (1.0 - self.log_survival(x).exp()).clamp(0.0, 1.0),
        }
    }

    /// Evaluate Limited Expected Value `E[X ∧ L] = ∫_0^L S(x) dx` for `L >= 0`.
    ///
    /// Uses exact closed-form formulas via [`SeverityDistribution`] when available
    /// (`Lognormal`, `ParetoII`, `Gamma`, `Gpd` with `threshold == 0, xi >= 0`) and
    /// adaptive 15-point Gauss–Kronrod quadrature (split cleanly at the splice/threshold
    /// point to avoid kinks) for `BurrXii`, `Weibull`, `InverseGaussian`, general `Gpd`,
    /// and `SplicedBodyGpd`.
    pub fn limited_expected_value(&self, limit: f64) -> Result<f64, PricingError> {
        if !limit.is_finite() || !(limit >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "limit must be finite and non-negative",
            ));
        }
        if limit == 0.0 {
            return Ok(0.0);
        }
        if let Ok(sev) = SeverityDistribution::try_from(*self) {
            return sev.limited_expected_value(limit);
        }
        let surv = |x: f64| -> f64 {
            let ls = self.log_survival(x);
            if ls.is_finite() { ls.exp() } else { 0.0 }
        };
        let lev = match *self {
            Self::Gpd { threshold, .. } | Self::SplicedBodyGpd { threshold, .. }
                if limit > threshold && threshold > 0.0 =>
            {
                let part1 = if matches!(*self, Self::Gpd { .. }) {
                    threshold
                } else {
                    gauss_kronrod_integrate(surv, 0.0, threshold)
                };
                part1 + gauss_kronrod_integrate(surv, threshold, limit)
            }
            Self::Gpd { threshold, .. } if limit <= threshold => limit,
            _ => gauss_kronrod_integrate(surv, 0.0, limit),
        };
        Ok(lev.clamp(0.0, limit))
    }

    /// Pure-loss Increased Limits Factor `ILF(L; B) = E[X ∧ L] / E[X ∧ B]`.
    pub fn increased_limits_factor(
        &self,
        limit: f64,
        basic_limit: f64,
    ) -> Result<f64, PricingError> {
        ilf_from_lev(limit, basic_limit, |x| self.limited_expected_value(x))
    }

    /// Expected loss in a 100%-participation excess layer `limit xs attachment`:
    /// `E[X ∧ (attachment + limit)] - E[X ∧ attachment]`.
    pub fn layer_expected_loss(&self, attachment: f64, limit: f64) -> Result<f64, PricingError> {
        if !attachment.is_finite() || !limit.is_finite() || !(attachment >= 0.0) || !(limit > 0.0) {
            return Err(PricingError::InvalidParameter(
                "attachment must be >= 0 and limit must be > 0",
            ));
        }
        let top = self.limited_expected_value(attachment + limit)?;
        let bot = self.limited_expected_value(attachment)?;
        Ok((top - bot).max(0.0))
    }
}

impl FittedEvtDistribution {
    /// Names of the estimated numeric fields, in the order used by
    /// [`EvtFitResult::standard_errors`].
    pub const fn parameter_names(&self) -> &'static [&'static str] {
        match self {
            Self::Lognormal { .. } => &["mu", "sigma"],
            Self::ParetoII { .. } => &["alpha", "lambda"],
            Self::Gpd { .. } => &["xi", "beta"],
            Self::BurrXii { .. } => &["alpha", "gamma", "lambda"],
            Self::Weibull { .. } => &["k", "lambda"],
            Self::Gamma { .. } => &["alpha", "beta"],
            Self::InverseGaussian { .. } => &["mu", "lambda"],
            Self::SplicedBodyGpd { .. } => {
                &["body_param1", "body_param2", "body_weight", "xi", "beta"]
            }
        }
    }

    /// Values of the estimated numeric fields (same order as [`Self::parameter_names`]).
    fn estimated_values(&self) -> Vec<f64> {
        match *self {
            Self::Lognormal { mu, sigma } => vec![mu, sigma],
            Self::ParetoII { alpha, lambda } => vec![alpha, lambda],
            Self::Gpd { xi, beta, .. } => vec![xi, beta],
            Self::BurrXii {
                alpha,
                gamma,
                lambda,
            } => vec![alpha, gamma, lambda],
            Self::Weibull { k, lambda } => vec![k, lambda],
            Self::Gamma { alpha, beta } => vec![alpha, beta],
            Self::InverseGaussian { mu, lambda } => vec![mu, lambda],
            Self::SplicedBodyGpd {
                body_param1,
                body_param2,
                body_weight,
                xi,
                beta,
                ..
            } => vec![body_param1, body_param2, body_weight, xi, beta],
        }
    }
}

impl TryFrom<FittedEvtDistribution> for SeverityDistribution {
    type Error = PricingError;

    /// Convert overlapping [`FittedEvtDistribution`] variants (`Lognormal`, `ParetoII`,
    /// `Gamma`, and `Gpd` with `threshold == 0.0 && xi >= 0.0`) directly into a
    /// [`SeverityDistribution`] for downstream `LayerTower`, deductible, and ALAE pricing.
    fn try_from(value: FittedEvtDistribution) -> Result<Self, Self::Error> {
        let dist = match value {
            FittedEvtDistribution::Lognormal { mu, sigma } => Self::Lognormal { mu, sigma },
            FittedEvtDistribution::ParetoII { alpha, lambda } => Self::Lomax {
                alpha,
                theta: lambda,
            },
            FittedEvtDistribution::Gamma { alpha, beta } => Self::Gamma {
                shape: alpha,
                scale: 1.0 / beta,
            },
            FittedEvtDistribution::Gpd {
                threshold,
                xi,
                beta,
            } if threshold == 0.0 && xi >= 0.0 => Self::GeneralizedPareto { xi, beta },
            _ => {
                return Err(PricingError::InvalidParameter(
                    "FittedEvtDistribution variant has no direct SeverityDistribution equivalent; use FittedEvtDistribution::limited_expected_value directly",
                ));
            }
        };
        dist.validate()?;
        Ok(dist)
    }
}

// ===========================================================================
// Unconstrained Parameterization φ -> FittedEvtDistribution & Jacobian
// ===========================================================================

/// Internal box on every unconstrained coordinate `|phi_j| <= PHI_BOUND`.
const PHI_BOUND: f64 = 30.0;
/// An optimum within this distance of [`PHI_BOUND`] is treated as a boundary (non-converged) fit.
const PHI_BOUND_MARGIN: f64 = 1.0;
/// GPD shape lower bound: the likelihood is unbounded for `xi < -1` (Smith 1985).
const GPD_XI_MIN: f64 = -1.0;
/// A GPD shape estimate within this distance of [`GPD_XI_MIN`] is a boundary (non-converged) fit.
const GPD_XI_BOUNDARY_TOL: f64 = 1e-3;
/// Relative central-difference step for the parameter-map Jacobian.
const JACOBIAN_REL_STEP: f64 = 1e-6;

fn unpack_params(family: EvtFamily, phi: &[f64]) -> Option<FittedEvtDistribution> {
    if phi.iter().any(|v| !v.is_finite() || v.abs() > PHI_BOUND) {
        return None;
    }
    match family {
        EvtFamily::Lognormal => Some(FittedEvtDistribution::Lognormal {
            mu: phi[0],
            sigma: phi[1].exp(),
        }),
        EvtFamily::ParetoII => Some(FittedEvtDistribution::ParetoII {
            alpha: phi[0].exp(),
            lambda: phi[1].exp(),
        }),
        EvtFamily::Gpd { threshold } => {
            let xi = phi[0];
            if !(xi > GPD_XI_MIN) {
                return None;
            }
            Some(FittedEvtDistribution::Gpd {
                threshold,
                xi,
                beta: phi[1].exp(),
            })
        }
        EvtFamily::BurrXii => {
            // phi = (ln alpha, ln gamma, ln theta), lambda = theta^gamma.
            let gamma = phi[1].exp();
            let lambda = (gamma * phi[2]).exp();
            if !(lambda > 0.0) || !lambda.is_finite() {
                return None;
            }
            Some(FittedEvtDistribution::BurrXii {
                alpha: phi[0].exp(),
                gamma,
                lambda,
            })
        }
        EvtFamily::Weibull => Some(FittedEvtDistribution::Weibull {
            k: phi[0].exp(),
            lambda: phi[1].exp(),
        }),
        EvtFamily::Gamma => Some(FittedEvtDistribution::Gamma {
            alpha: phi[0].exp(),
            beta: phi[1].exp(),
        }),
        EvtFamily::InverseGaussian => Some(FittedEvtDistribution::InverseGaussian {
            mu: phi[0].exp(),
            lambda: phi[1].exp(),
        }),
        EvtFamily::SplicedBodyGpd {
            body,
            threshold,
            continuity,
        } => {
            let (p1, p2, d_log_f1, f1_u, pdf1_u) = match body {
                SpliceBodyFamily::Lognormal => {
                    let mu = phi[0];
                    let sigma = phi[1].exp();
                    let d = FittedEvtDistribution::Lognormal { mu, sigma };
                    let f1 = d.cdf(threshold);
                    let pdf1 = d.log_pdf(threshold).exp();
                    let d_lf = -(1.0 + (threshold.ln() - mu) / (sigma * sigma)) / threshold;
                    (mu, sigma, d_lf, f1, pdf1)
                }
                SpliceBodyFamily::Weibull => {
                    let k = phi[0].exp();
                    let lambda = phi[1].exp();
                    let d = FittedEvtDistribution::Weibull { k, lambda };
                    let f1 = d.cdf(threshold);
                    let pdf1 = d.log_pdf(threshold).exp();
                    let d_lf =
                        (k - 1.0) / threshold - (k / lambda) * (threshold / lambda).powf(k - 1.0);
                    (k, lambda, d_lf, f1, pdf1)
                }
            };
            if !(f1_u > 1e-12) || !(pdf1_u > 0.0) || !f1_u.is_finite() || !pdf1_u.is_finite() {
                return None;
            }
            let xi = phi[2];
            if !(xi > GPD_XI_MIN) {
                return None;
            }
            let beta = match continuity {
                SpliceContinuity::C0Continuous => phi[3].exp(),
                SpliceContinuity::C1Differentiable => {
                    if !(d_log_f1 < -1e-12) || !(xi > GPD_XI_MIN + 1e-6) {
                        return None;
                    }
                    let beta = -(1.0 + xi) / d_log_f1;
                    if !(beta > 0.0) || !beta.is_finite() {
                        return None;
                    }
                    beta
                }
            };
            let w = f1_u / (f1_u + beta * pdf1_u);
            if !(w > 1e-12 && w < 1.0 - 1e-12) {
                return None;
            }
            Some(FittedEvtDistribution::SplicedBodyGpd {
                body,
                threshold,
                continuity,
                body_param1: p1,
                body_param2: p2,
                body_weight: w,
                xi,
                beta,
            })
        }
    }
}

/// Central-difference Jacobian `d(estimated_values) / d(phi)` (row-major `q x p`).
fn parameter_jacobian(family: EvtFamily, phi: &[f64]) -> Option<(Vec<f64>, usize)> {
    let p = phi.len();
    let q = unpack_params(family, phi)?.estimated_values().len();
    let mut jac = vec![0.0; q * p];
    let mut work = phi.to_vec();
    for j in 0..p {
        let h = JACOBIAN_REL_STEP * (1.0 + phi[j].abs());
        work[j] = phi[j] + h;
        let up = unpack_params(family, &work)?.estimated_values();
        work[j] = phi[j] - h;
        let dn = unpack_params(family, &work)?.estimated_values();
        work[j] = phi[j];
        for i in 0..q {
            jac[i * p + j] = (up[i] - dn[i]) / (2.0 * h);
        }
    }
    Some((jac, q))
}

fn eval_log_likelihood(dist: &FittedEvtDistribution, obs: &[SeverityObservation]) -> f64 {
    let mut ll = 0.0;
    for o in obs {
        let term = if o.is_censored {
            dist.log_survival(o.loss)
        } else {
            dist.log_pdf(o.loss)
        };
        let trunc = if o.deductible > 0.0 {
            dist.log_survival(o.deductible)
        } else {
            0.0
        };
        if !term.is_finite() || !trunc.is_finite() {
            return f64::NEG_INFINITY;
        }
        ll += o.weight * (term - trunc);
    }
    ll
}

// ===========================================================================
// 1.4 Method-of-Moments Starting-Value Initializers
// ===========================================================================

/// Clamp range for the log-logistic Burr XII starting `gamma`.
const BURR_GAMMA_START_MIN: f64 = 0.1;
/// Clamp range for the log-logistic Burr XII starting `gamma`.
const BURR_GAMMA_START_MAX: f64 = 50.0;

fn mom_initial_phi(family: EvtFamily, obs: &[SeverityObservation]) -> Vec<f64> {
    let unc: Vec<f64> = obs
        .iter()
        .filter(|o| !o.is_censored)
        .map(|o| o.loss)
        .collect();
    let sample = if unc.is_empty() {
        obs.iter().map(|o| o.loss).collect::<Vec<_>>()
    } else {
        unc
    };
    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;
    let var = if sample.len() > 1 {
        sample.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        mean * mean
    }
    .max(1e-12);
    let log_mean = sample.iter().map(|&x| x.ln()).sum::<f64>() / n;
    let log_var = if sample.len() > 1 {
        sample
            .iter()
            .map(|&x| (x.ln() - log_mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        1.0
    }
    .max(1e-6);

    match family {
        EvtFamily::Lognormal => vec![log_mean, log_var.sqrt().max(1e-3).ln()],
        EvtFamily::ParetoII => {
            let (alpha, lambda) = if var > mean * mean {
                let a = (2.0 * var / (var - mean * mean)).clamp(2.05, 50.0);
                (a, (mean * (a - 1.0)).max(1e-6))
            } else {
                (5.0, (4.0 * mean).max(1e-6))
            };
            vec![alpha.ln(), lambda.ln()]
        }
        EvtFamily::Gpd { threshold } => {
            let exc: Vec<f64> = sample
                .iter()
                .filter_map(|&x| (x > threshold).then_some(x - threshold))
                .collect();
            let (em, ev) = if exc.len() > 1 {
                let m = exc.iter().sum::<f64>() / exc.len() as f64;
                let v = exc.iter().map(|&y| (y - m).powi(2)).sum::<f64>() / (exc.len() - 1) as f64;
                (m.max(1e-6), v.max(1e-12))
            } else {
                (mean.max(1e-6), var)
            };
            let ratio = em * em / ev;
            let xi = (0.5 * (1.0 - ratio)).clamp(-0.45, 0.45);
            let beta = (0.5 * em * (1.0 + ratio)).max(1e-6);
            vec![xi, beta.ln()]
        }
        EvtFamily::BurrXii => {
            // Log-logistic start (alpha = 1): theta = sample median, and ln X logistic with
            // SD pi / (gamma sqrt(3)).
            let mut sorted = sample.clone();
            sorted.sort_by(f64::total_cmp);
            let mid = sorted.len() / 2;
            let median = if sorted.len() % 2 == 0 {
                f64::midpoint(sorted[mid - 1], sorted[mid])
            } else {
                sorted[mid]
            };
            let gamma = (std::f64::consts::PI / (3.0_f64.sqrt() * log_var.sqrt()))
                .clamp(BURR_GAMMA_START_MIN, BURR_GAMMA_START_MAX);
            vec![0.0, gamma.ln(), median.max(f64::MIN_POSITIVE).ln()]
        }
        EvtFamily::Weibull => {
            let k = (std::f64::consts::PI / (6.0_f64.sqrt() * log_var.sqrt())).clamp(0.1, 20.0);
            let lambda = (log_mean + std::f64::consts::EULER_GAMMA / k)
                .exp()
                .max(1e-6);
            vec![k.ln(), lambda.ln()]
        }
        EvtFamily::Gamma => {
            let alpha = (mean * mean / var).clamp(0.05, 500.0);
            let beta = (mean / var).max(1e-9);
            vec![alpha.ln(), beta.ln()]
        }
        EvtFamily::InverseGaussian => {
            let mu = mean.max(1e-6);
            let lambda = (mu.powi(3) / var).max(1e-9);
            vec![mu.ln(), lambda.ln()]
        }
        EvtFamily::SplicedBodyGpd {
            body,
            threshold,
            continuity,
        } => {
            let (p1, p2) = match body {
                SpliceBodyFamily::Lognormal => {
                    // Ensure mode of body < threshold so C1 derivative is negative at u.
                    let mu = log_mean.min(threshold.ln() - 0.1);
                    let sigma = log_var.sqrt().clamp(0.2, 3.0);
                    (mu, sigma.ln())
                }
                SpliceBodyFamily::Weibull => {
                    let k = 1.5_f64;
                    let lambda = (0.7 * threshold).max(1e-6);
                    (k.ln(), lambda.ln())
                }
            };
            let exc_mean = sample
                .iter()
                .filter_map(|&x| (x > threshold).then_some(x - threshold))
                .sum::<f64>()
                / sample.iter().filter(|&&x| x > threshold).count().max(1) as f64;
            let beta0 = exc_mean.max(0.2 * threshold).max(1e-6);
            match continuity {
                SpliceContinuity::C0Continuous => vec![p1, p2, 0.1, beta0.ln()],
                SpliceContinuity::C1Differentiable => vec![p1, p2, 0.1],
            }
        }
    }
}

// ===========================================================================
// 1.5 Deterministic L-BFGS (m = 7) + Nelder–Mead Fallback & Fisher Info
// ===========================================================================

fn central_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64]) -> Vec<f64> {
    let p = x.len();
    let eps_cbrt = f64::EPSILON.cbrt();
    let mut g = vec![0.0; p];
    let mut work = x.to_vec();
    for j in 0..p {
        // Absolute step: every coordinate of phi is O(1)-scaled (logs, xi, mu), and a
        // |phi|-proportional step inflates truncation error on log-scale coordinates.
        let h = eps_cbrt;
        work[j] = x[j] + h;
        let fp = f(&work);
        work[j] = x[j] - h;
        let fm = f(&work);
        work[j] = x[j];
        g[j] = if fp.is_finite() && fm.is_finite() {
            (fp - fm) / (2.0 * h)
        } else {
            0.0
        };
    }
    g
}

/// L-BFGS gradient tolerance: `||g||_inf < tol * LBFGS_GRAD_FACTOR` (floored at the
/// finite-difference noise level, see [`gradient_tolerance`]).
const LBFGS_GRAD_FACTOR: f64 = 1.0;
/// L-BFGS relative objective-change tolerance: `|Δf| <= tol * LBFGS_REL_F_FACTOR * (1 + |f|)`.
const LBFGS_REL_F_FACTOR: f64 = 1e-4;
/// Gradient bound accompanying the objective-change test: `||g||_inf < tol * LBFGS_LOOSE_GRAD_FACTOR`.
const LBFGS_LOOSE_GRAD_FACTOR: f64 = 1e2;
/// Nelder–Mead simplex-diameter tolerance factor (`tol * NM_X_FACTOR`).
const NM_X_FACTOR: f64 = 0.1;
/// Nelder–Mead relative objective-spread tolerance factor (`tol * NM_F_FACTOR`).
const NM_F_FACTOR: f64 = 1e-3;

/// Multiple of the central-difference noise floor `eps^(2/3) (1 + |f|)` below which a
/// gradient is indistinguishable from zero.
const GRAD_NOISE_FACTOR: f64 = 10.0;

/// Gradient stopping threshold: `requested`, floored at the finite-difference noise level
/// (large-sample log-likelihoods cannot resolve gradients below `~eps^(2/3) |f|`).
fn gradient_tolerance(requested: f64, fx: f64) -> f64 {
    let noise = GRAD_NOISE_FACTOR * f64::EPSILON.powf(2.0 / 3.0) * (1.0 + fx.abs());
    requested.max(noise)
}

struct OptimOutcome {
    x: Vec<f64>,
    fx: f64,
    converged: bool,
    iterations: usize,
}

fn lbfgs_minimize<F: Fn(&[f64]) -> f64>(
    f: &F,
    mut x: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> OptimOutcome {
    let p = x.len();
    let m = 7usize;
    let mut fx = f(&x);
    if !fx.is_finite() {
        return OptimOutcome {
            x,
            fx,
            converged: false,
            iterations: 0,
        };
    }
    let mut iterations = 0usize;
    let mut gx = central_gradient(f, &x);
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);
    let mut converged = false;

    for _ in 0..max_iter {
        let g_inf = gx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < gradient_tolerance(tol * LBFGS_GRAD_FACTOR, fx) {
            converged = true;
            break;
        }
        iterations += 1;

        let mut q = gx.clone();
        let h_len = s_hist.len();
        let mut alphas = vec![0.0; h_len];
        for i in (0..h_len).rev() {
            let dot: f64 = s_hist[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            alphas[i] = rho_hist[i] * dot;
            for k in 0..p {
                q[k] -= alphas[i] * y_hist[i][k];
            }
        }
        let gamma = if h_len > 0 {
            let last = h_len - 1;
            let sy: f64 = s_hist[last]
                .iter()
                .zip(y_hist[last].iter())
                .map(|(a, b)| a * b)
                .sum();
            let yy: f64 = y_hist[last].iter().map(|v| v * v).sum();
            if yy > 1e-24 && sy > 0.0 { sy / yy } else { 1.0 }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();
        for i in 0..h_len {
            let dot: f64 = y_hist[i].iter().zip(r.iter()).map(|(a, b)| a * b).sum();
            let beta = rho_hist[i] * dot;
            for k in 0..p {
                r[k] += s_hist[i][k] * (alphas[i] - beta);
            }
        }
        let mut d: Vec<f64> = r.iter().map(|&v| -v).collect();
        let mut dir_deriv: f64 = gx.iter().zip(d.iter()).map(|(g, di)| g * di).sum();
        if !dir_deriv.is_finite() || dir_deriv >= 0.0 {
            d = gx.iter().map(|&v| -v).collect();
            dir_deriv = -gx.iter().map(|v| v * v).sum::<f64>();
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        let mut step = 1.0;
        let mut x_next = vec![0.0; p];
        let mut f_next = f64::INFINITY;
        for _ in 0..35 {
            for k in 0..p {
                x_next[k] = x[k] + step * d[k];
            }
            f_next = f(&x_next);
            if f_next.is_finite() && f_next <= fx + 1e-4 * step * dir_deriv {
                break;
            }
            step *= 0.5;
        }
        if !f_next.is_finite() || f_next > fx {
            break;
        }

        let g_next = central_gradient(f, &x_next);
        let s_vec: Vec<f64> = x_next.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y_vec: Vec<f64> = g_next.iter().zip(gx.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if sy > 1e-14 {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        let df = (fx - f_next).abs();
        x = x_next;
        fx = f_next;
        gx = g_next;
        if df <= tol * LBFGS_REL_F_FACTOR * (1.0 + fx.abs())
            && gx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                < gradient_tolerance(tol * LBFGS_LOOSE_GRAD_FACTOR, fx)
        {
            converged = true;
            break;
        }
    }

    OptimOutcome {
        x,
        fx,
        converged,
        iterations,
    }
}

fn nelder_mead_minimize<F: Fn(&[f64]) -> f64>(
    f: &F,
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> OptimOutcome {
    let p = x0.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(p + 1);
    simplex.push((x0.to_vec(), f(x0)));
    for j in 0..p {
        let mut pt = x0.to_vec();
        pt[j] += 0.25 * (1.0 + pt[j].abs());
        let fv = f(&pt);
        simplex.push((pt, fv));
    }

    let mut converged = false;
    let mut iterations = 0usize;
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let spread_f = (simplex[p].1 - simplex[0].1).abs();
        let mut spread_x = 0.0_f64;
        for i in 1..=p {
            for j in 0..p {
                spread_x = spread_x.max((simplex[i].0[j] - simplex[0].0[j]).abs());
            }
        }
        if spread_x < tol * NM_X_FACTOR && spread_f < tol * NM_F_FACTOR * (1.0 + simplex[0].1.abs())
        {
            converged = true;
            break;
        }
        iterations += 1;

        let mut centroid = vec![0.0; p];
        for item in simplex.iter().take(p) {
            for j in 0..p {
                centroid[j] += item.0[j] / p as f64;
            }
        }

        let xr: Vec<f64> = (0..p)
            .map(|j| centroid[j] + (centroid[j] - simplex[p].0[j]))
            .collect();
        let fr = f(&xr);

        if fr < simplex[0].1 {
            let xe: Vec<f64> = (0..p)
                .map(|j| centroid[j] + 2.0 * (xr[j] - centroid[j]))
                .collect();
            let fe = f(&xe);
            simplex[p] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[p - 1].1 {
            simplex[p] = (xr, fr);
        } else {
            let xc: Vec<f64> = if fr < simplex[p].1 {
                (0..p)
                    .map(|j| centroid[j] + 0.5 * (xr[j] - centroid[j]))
                    .collect()
            } else {
                (0..p)
                    .map(|j| centroid[j] + 0.5 * (simplex[p].0[j] - centroid[j]))
                    .collect()
            };
            let fc = f(&xc);
            if fc < simplex[p].1.min(fr) {
                simplex[p] = (xc, fc);
            } else {
                let best = simplex[0].0.clone();
                for item in simplex.iter_mut().skip(1) {
                    for j in 0..p {
                        item.0[j] = f64::midpoint(best[j], item.0[j]);
                    }
                    item.1 = f(&item.0);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (x, fx) = simplex.swap_remove(0);
    OptimOutcome {
        x,
        fx,
        converged,
        iterations,
    }
}

/// Delta-method SEs `sqrt(diag(J H^{-1} J^T))` of the estimated fields (NaN where undefined).
fn delta_method_standard_errors<F: Fn(&[f64]) -> f64>(
    neg_ll: &F,
    family: EvtFamily,
    phi_hat: &[f64],
    n_out: usize,
) -> Vec<f64> {
    let p = phi_hat.len();
    let mut hess = vec![0.0; p * p];
    let mut work = phi_hat.to_vec();
    for j in 0..p {
        let h = 1e-4 * (1.0 + phi_hat[j].abs());
        work[j] = phi_hat[j] + h;
        let gp = central_gradient(neg_ll, &work);
        work[j] = phi_hat[j] - h;
        let gm = central_gradient(neg_ll, &work);
        work[j] = phi_hat[j];
        for i in 0..p {
            hess[i * p + j] = (gp[i] - gm[i]) / (2.0 * h);
        }
    }
    for i in 0..p {
        for j in (i + 1)..p {
            let sym = f64::midpoint(hess[i * p + j], hess[j * p + i]);
            hess[i * p + j] = sym;
            hess[j * p + i] = sym;
        }
    }

    let (Ok(cov_phi), Some((jac, q))) = (
        cholesky_invert(&hess, p),
        parameter_jacobian(family, phi_hat),
    ) else {
        return vec![f64::NAN; n_out];
    };
    (0..q)
        .map(|r| {
            let mut v = 0.0;
            for a in 0..p {
                for b in 0..p {
                    v += jac[r * p + a] * cov_phi[a * p + b] * jac[r * p + b];
                }
            }
            if v.is_finite() && v >= 0.0 {
                v.sqrt()
            } else {
                f64::NAN
            }
        })
        .collect()
}

// ===========================================================================
// 1.6 Goodness-of-Fit & Empirical Mean Excess
// ===========================================================================

/// Empirical mean excess function `e_n(u) = mean(x_i - u | x_i > u)` for
/// uncensored losses strictly exceeding `threshold`.
pub fn empirical_mean_excess(losses: &[f64], threshold: f64) -> Result<f64, PricingError> {
    if losses.is_empty() {
        return Err(PricingError::EmptyInput("losses must not be empty"));
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(PricingError::InvalidParameter(
            "threshold must be finite >= 0",
        ));
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for &x in losses {
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter("losses must be finite"));
        }
        if x > threshold {
            sum += x - threshold;
            count += 1;
        }
    }
    if count == 0 {
        return Err(PricingError::OutOfBounds(
            "no observations strictly exceed threshold",
        ));
    }
    Ok(sum / count as f64)
}

/// Lower/upper clamp applied to PIT values before the Anderson–Darling logarithms.
const PIT_CLAMP: f64 = 1e-15;

/// `∫_a^b (c - u)^2 / (u (1 - u)) du = -(b - a) + c^2 ln(b / a) + (1 - c)^2 ln((1 - a) / (1 - b))`
/// for a constant EDF level `c` on `[a, b] ⊂ [0, 1]` (terms with zero coefficient dropped).
fn ad_segment(a: f64, b: f64, c: f64) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    let mut v = -(b - a);
    if c > 0.0 {
        v += c * c * (b.ln() - a.ln());
    }
    if c < 1.0 {
        v += (1.0 - c) * (1.0 - c) * ((-a).ln_1p() - (-b).ln_1p());
    }
    v
}

/// Compute AIC, BIC (`n = Σ w_i`), weighted product-limit KS statistic `D_n`, and weighted
/// PIT Anderson–Darling statistic `A^2` (when uncensored).
///
/// * **KS:** Lynden-Bell / Kaplan–Meier product-limit estimator with delayed entry — the
///   risk set at `t` is `Σ w_i 1{d_i < t <= x_i}` — compared with the model conditional on
///   `X > min_i d_i`, i.e. `1 - S(t) / S(min_i d_i)`.
/// * **AD:** exact integral `W ∫ (F_W(u) - u)^2 / (u (1 - u)) du` of the weighted EDF `F_W`
///   of the conditional PITs `u_i = 1 - S(x_i) / S(d_i)`, `W = Σ w_i`; with unit weights this
///   is the classical `-n - (1/n) Σ (2i - 1) [ln u_(i) + ln(1 - u_(n+1-i))]`.
///
/// Integer weights are therefore equivalent to replicated rows for all statistics.
pub fn goodness_of_fit(
    dist: &FittedEvtDistribution,
    observations: &[SeverityObservation],
    log_likelihood: f64,
    n_params: usize,
) -> GoodnessOfFitDiagnostics {
    let n = observations.len();
    let aic = 2.0 * n_params as f64 - 2.0 * log_likelihood;
    let total_w: f64 = observations.iter().map(|o| o.weight).sum();
    if n == 0 || !(total_w > 0.0) {
        return GoodnessOfFitDiagnostics {
            aic,
            bic: f64::NAN,
            ks_statistic: 0.0,
            anderson_darling: None,
        };
    }
    let bic = (n_params as f64) * total_w.ln() - 2.0 * log_likelihood;

    let has_censored = observations.iter().any(|o| o.is_censored);
    let d_min = observations
        .iter()
        .map(|o| o.deductible)
        .fold(f64::INFINITY, f64::min);
    let ls_dmin = dist.log_survival(d_min);

    // Product-limit with delayed entry (Lynden-Bell 1971; Tsai, Jewell & Wang 1987).
    let mut by_loss: Vec<&SeverityObservation> = observations.iter().collect();
    by_loss.sort_by(|a, b| a.loss.total_cmp(&b.loss));
    let mut by_ded: Vec<(f64, f64)> = observations
        .iter()
        .map(|o| (o.deductible, o.weight))
        .collect();
    by_ded.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut km_s = 1.0_f64;
    let mut entered_w = 0.0_f64;
    let mut exited_w = 0.0_f64;
    let mut ded_idx = 0usize;
    let mut ks = 0.0_f64;
    let mut idx = 0usize;
    while idx < n {
        let x = by_loss[idx].loss;
        while ded_idx < n && by_ded[ded_idx].0 < x {
            entered_w += by_ded[ded_idx].1;
            ded_idx += 1;
        }
        let at_risk = entered_w - exited_w;
        let mut events = 0.0;
        let mut total_at_x = 0.0;
        while idx < n && by_loss[idx].loss == x {
            total_at_x += by_loss[idx].weight;
            if !by_loss[idx].is_censored {
                events += by_loss[idx].weight;
            }
            idx += 1;
        }
        let f_prev = 1.0 - km_s;
        if events > 0.0 && at_risk > 0.0 {
            km_s *= (1.0 - events / at_risk).max(0.0);
        }
        let f_curr = 1.0 - km_s;
        let f_model = (-(dist.log_survival(x) - ls_dmin).exp_m1()).clamp(0.0, 1.0);
        ks = ks
            .max((f_prev - f_model).abs())
            .max((f_curr - f_model).abs());
        exited_w += total_at_x;
    }

    let anderson_darling = if has_censored {
        None
    } else {
        let mut u: Vec<(f64, f64)> = observations
            .iter()
            .map(|o| {
                let log_ratio = dist.log_survival(o.loss) - dist.log_survival(o.deductible);
                let pit = (-log_ratio.exp_m1()).clamp(PIT_CLAMP, 1.0 - PIT_CLAMP);
                (pit, o.weight)
            })
            .collect();
        u.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut integral = 0.0;
        let mut level = 0.0_f64;
        let mut prev = 0.0_f64;
        let mut cum_w = 0.0_f64;
        for &(ui, wi) in &u {
            integral += ad_segment(prev, ui, level);
            cum_w += wi;
            level = (cum_w / total_w).min(1.0);
            prev = ui;
        }
        integral += ad_segment(prev, 1.0, 1.0);
        Some(total_w * integral)
    };

    GoodnessOfFitDiagnostics {
        aic,
        bic,
        ks_statistic: ks,
        anderson_darling,
    }
}

// ===========================================================================
// Main Entry Point: fit_evt_mle
// ===========================================================================

/// Fit any of the 7 parametric severity/EVT distributions or a $C^0$/$C^1$
/// spliced body + GPD tail model via left-truncated, right-censored MLE
/// using default solver tolerances.
pub fn fit_evt_mle(
    family: EvtFamily,
    observations: &[SeverityObservation],
) -> Result<EvtFitResult, PricingError> {
    fit_evt_mle_with_options(
        family,
        observations,
        &SolverOptions {
            max_iter: 500,
            tol: 1e-8,
        },
    )
}

/// Fit any of the 7 parametric severity/EVT distributions or a $C^0$/$C^1$
/// spliced body + GPD tail model via left-truncated, right-censored MLE
/// with explicit [`SolverOptions`].
///
/// `options.tol` drives every stopping rule: L-BFGS stops when `||∇(-ll)||_inf < tol` or when
/// `|Δll| <= 1e-4 tol (1 + |ll|)` with `||∇||_inf < 1e2 tol`; the Nelder–Mead fallback stops when
/// the simplex diameter is `< 0.1 tol` and its objective spread `< 1e-3 tol (1 + |ll|)`
/// (the [`fit_evt_mle`] default `tol = 1e-8` reproduces the historical constants).
pub fn fit_evt_mle_with_options(
    family: EvtFamily,
    observations: &[SeverityObservation],
    options: &SolverOptions,
) -> Result<EvtFitResult, PricingError> {
    if options.max_iter == 0 || !options.tol.is_finite() || !(options.tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "solver options require max_iter > 0 and finite tol > 0",
        ));
    }
    if observations.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one observation is required for MLE",
        ));
    }
    let mut n_uncensored = 0usize;
    for o in observations {
        if !o.loss.is_finite() || !o.deductible.is_finite() || !o.weight.is_finite() {
            return Err(PricingError::InvalidParameter(
                "observation fields must be finite",
            ));
        }
        if !(o.loss > 0.0) || !(o.deductible >= 0.0) || !(o.weight > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "loss and weight must be > 0 and deductible >= 0",
            ));
        }
        if o.loss <= o.deductible {
            return Err(PricingError::InvalidParameter(
                "observed loss must be strictly greater than deductible",
            ));
        }
        if !o.is_censored {
            n_uncensored += 1;
        }
    }
    if n_uncensored == 0 {
        return Err(PricingError::InvalidParameter(
            "at least one uncensored observation is required",
        ));
    }

    // For GPD, filter observations strictly exceeding the threshold (adjusting deductibles).
    let filtered: Vec<SeverityObservation> = match family {
        EvtFamily::Gpd { threshold } => {
            if !threshold.is_finite() || threshold < 0.0 {
                return Err(PricingError::InvalidParameter(
                    "GPD threshold must be finite >= 0",
                ));
            }
            let v: Vec<SeverityObservation> = observations
                .iter()
                .filter(|o| o.loss > threshold)
                .map(|o| SeverityObservation {
                    loss: o.loss,
                    deductible: o.deductible.max(threshold),
                    is_censored: o.is_censored,
                    weight: o.weight,
                })
                .collect();
            if v.is_empty() {
                return Err(PricingError::OutOfBounds(
                    "no observations exceed GPD threshold",
                ));
            }
            if !v.iter().any(|o| !o.is_censored) {
                return Err(PricingError::InvalidParameter(
                    "at least one uncensored observation must exceed GPD threshold",
                ));
            }
            v
        }
        EvtFamily::SplicedBodyGpd { threshold, .. } => {
            if !threshold.is_finite() || !(threshold > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "splice threshold must be finite > 0",
                ));
            }
            observations.to_vec()
        }
        _ => observations.to_vec(),
    };
    let n_censored = filtered.iter().filter(|o| o.is_censored).count();

    let phi0 = mom_initial_phi(family, &filtered);
    let neg_ll = |phi: &[f64]| -> f64 {
        match unpack_params(family, phi) {
            Some(dist) => {
                let ll = eval_log_likelihood(&dist, &filtered);
                if ll.is_finite() { -ll } else { f64::INFINITY }
            }
            None => f64::INFINITY,
        }
    };

    let tol = options.tol;
    let first = lbfgs_minimize(&neg_ll, phi0.clone(), options.max_iter, tol);
    let mut iterations = first.iterations;
    let (mut best_phi, mut best_nll, mut converged) = (first.x, first.fx, first.converged);
    if !converged || !best_nll.is_finite() {
        let start = if best_nll.is_finite() {
            best_phi.as_slice()
        } else {
            phi0.as_slice()
        };
        let nm_iters = options.max_iter.saturating_mul(3).max(500);
        let nm = nelder_mead_minimize(&neg_ll, start, nm_iters, tol);
        iterations += nm.iterations;
        if nm.fx <= best_nll || !best_nll.is_finite() {
            best_phi = nm.x;
            best_nll = nm.fx;
            converged = nm.converged;
        }
    }

    // One L-BFGS polish pass if Nelder–Mead improved the point.
    if best_nll.is_finite() {
        let polish_iters = (options.max_iter / 2).max(50);
        let pol = lbfgs_minimize(&neg_ll, best_phi.clone(), polish_iters, tol);
        iterations += pol.iterations;
        if pol.fx <= best_nll {
            best_phi = pol.x;
            best_nll = pol.fx;
            converged = converged || pol.converged;
        }
    }

    let Some(distribution) = unpack_params(family, &best_phi) else {
        return Err(PricingError::NonConvergence(
            "MLE failed to locate valid parameters",
        ));
    };
    if !best_nll.is_finite() {
        return Err(PricingError::NonConvergence(
            "MLE log-likelihood is non-finite",
        ));
    }

    let log_likelihood = -best_nll;
    let n_out = distribution.parameter_names().len();
    let standard_errors = delta_method_standard_errors(&neg_ll, family, &best_phi, n_out);
    let diagnostics = goodness_of_fit(&distribution, &filtered, log_likelihood, best_phi.len());

    // Boundary / degeneracy rules: never report convergence on the internal box, at the
    // GPD `xi -> -1` boundary (Smith 1985), or with undefined standard errors.
    let on_box = best_phi
        .iter()
        .any(|v| v.abs() > PHI_BOUND - PHI_BOUND_MARGIN);
    let on_xi_boundary = match distribution {
        FittedEvtDistribution::Gpd { xi, .. }
        | FittedEvtDistribution::SplicedBodyGpd { xi, .. } => xi < GPD_XI_MIN + GPD_XI_BOUNDARY_TOL,
        _ => false,
    };
    let se_ok = standard_errors.iter().all(|v| v.is_finite());
    let converged = converged && !on_box && !on_xi_boundary && se_ok;

    Ok(EvtFitResult {
        distribution,
        log_likelihood,
        standard_errors,
        diagnostics,
        n_observations: filtered.len(),
        n_censored,
        converged,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KPW_DATA_B: [f64; 20] = [
        27.0, 82.0, 115.0, 126.0, 155.0, 161.0, 243.0, 294.0, 340.0, 384.0, 457.0, 680.0, 855.0,
        877.0, 974.0, 1193.0, 1340.0, 1884.0, 2558.0, 15743.0,
    ];

    #[test]
    fn golden_kpw_data_set_b_lognormal_pareto_weibull() {
        let obs: Vec<SeverityObservation> = KPW_DATA_B
            .iter()
            .map(|&x| SeverityObservation::exact(x))
            .collect();

        // Lognormal exact MLE on KPW Data Set B: mu = 6.13788, sigma = 1.38941, ll = -157.711
        let ln_fit = fit_evt_mle(EvtFamily::Lognormal, &obs).expect("lognormal");
        let FittedEvtDistribution::Lognormal { mu, sigma } = ln_fit.distribution else {
            panic!("expected Lognormal");
        };
        assert!((mu - 6.137_88).abs() < 1e-4, "mu={mu}");
        assert!((sigma - 1.389_408).abs() < 1e-4, "sigma={sigma}");
        assert!(
            (ln_fit.log_likelihood - (-157.711)).abs() < 0.05,
            "ll={}",
            ln_fit.log_likelihood
        );
        assert!(ln_fit.diagnostics.anderson_darling.is_some());

        // Pareto II (Lomax): verify exact parity against existing `fit_severity_mle(Lomax)`.
        let p_fit = fit_evt_mle(EvtFamily::ParetoII, &obs).expect("pareto");
        let FittedEvtDistribution::ParetoII { alpha, lambda } = p_fit.distribution else {
            panic!("expected ParetoII");
        };
        let ref_lomax = crate::layers::fitting::fit_severity_mle(
            crate::layers::fitting::SeverityFamily::Lomax,
            &obs,
            &crate::error::SolverOptions::default(),
        )
        .expect("ref lomax");
        let crate::layers::SeverityDistribution::Lomax {
            alpha: ref_a,
            theta: ref_l,
        } = ref_lomax.distribution
        else {
            panic!("expected Lomax");
        };
        assert!((alpha - ref_a).abs() < 1e-4, "alpha={alpha} ref={ref_a}");
        assert!(
            (lambda - ref_l).abs() / ref_l < 1e-4,
            "lambda={lambda} ref={ref_l}"
        );
        assert!((p_fit.log_likelihood - ref_lomax.log_likelihood).abs() < 1e-5);

        // Weibull on KPW Data Set B: k ~ 0.659, ll ~ -159.45.
        let w_fit = fit_evt_mle(EvtFamily::Weibull, &obs).expect("weibull");
        let FittedEvtDistribution::Weibull { k, lambda } = w_fit.distribution else {
            panic!("expected Weibull");
        };
        assert!(k > 0.5 && k < 0.8, "k={k}");
        assert!(lambda > 500.0 && lambda < 1200.0, "lambda={lambda}");
        assert!(w_fit.log_likelihood.is_finite() && w_fit.converged);
    }

    #[test]
    fn golden_pareto_ii_truncation_shift_invariance() {
        let d = 100.0;
        let trunc_obs: Vec<SeverityObservation> = KPW_DATA_B
            .iter()
            .filter(|&&x| x > d)
            .map(|&x| SeverityObservation::truncated(x, d))
            .collect();
        let shifted_obs: Vec<SeverityObservation> = KPW_DATA_B
            .iter()
            .filter(|&&x| x > d)
            .map(|&x| SeverityObservation::exact(x - d))
            .collect();

        let f_trunc = fit_evt_mle(EvtFamily::ParetoII, &trunc_obs).expect("trunc");
        let f_shift = fit_evt_mle(EvtFamily::ParetoII, &shifted_obs).expect("shift");
        let (
            FittedEvtDistribution::ParetoII {
                alpha: a1,
                lambda: l1,
            },
            FittedEvtDistribution::ParetoII {
                alpha: a2,
                lambda: l2,
            },
        ) = (f_trunc.distribution, f_shift.distribution)
        else {
            panic!("expected ParetoII");
        };
        assert!((a1 - a2).abs() < 1e-6, "a1={a1} a2={a2}");
        assert!(((l1 + d) - l2).abs() < 1e-4, "l1+d={} l2={l2}", l1 + d);
    }

    #[test]
    fn gpd_burr_gamma_ig_and_spliced_fit_cleanly_with_mean_excess() {
        let obs: Vec<SeverityObservation> = KPW_DATA_B
            .iter()
            .map(|&x| SeverityObservation::exact(x))
            .collect();
        for fam in [
            EvtFamily::Gpd { threshold: 200.0 },
            EvtFamily::BurrXii,
            EvtFamily::Gamma,
            EvtFamily::InverseGaussian,
            EvtFamily::SplicedBodyGpd {
                body: SpliceBodyFamily::Lognormal,
                threshold: 1000.0,
                continuity: SpliceContinuity::C0Continuous,
            },
            EvtFamily::SplicedBodyGpd {
                body: SpliceBodyFamily::Lognormal,
                threshold: 1000.0,
                continuity: SpliceContinuity::C1Differentiable,
            },
        ] {
            let res = fit_evt_mle(fam, &obs).expect("evt family fit");
            assert!(res.log_likelihood.is_finite());
            assert!(res.diagnostics.ks_statistic >= 0.0 && res.diagnostics.ks_statistic <= 1.0);
        }

        let me1 = empirical_mean_excess(&KPW_DATA_B, 200.0).expect("me 200");
        let me2 = empirical_mean_excess(&KPW_DATA_B, 800.0).expect("me 800");
        assert!(me2 > me1, "heavy-tailed mean excess should increase");
    }
}
