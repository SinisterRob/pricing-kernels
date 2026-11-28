//! Maximum Likelihood Estimation (MLE) for parametric severity distributions under
//! left-truncation (per-occurrence deductibles) and right-censoring (policy limits).
//!
//! Reference: S. A. Klugman, H. H. Panjer & G. E. Willmot, *Loss Models: From Data to
//! Decisions* (5th ed.), Ch. 13.

use crate::error::{PricingError, SolverOptions};
use crate::layers::SeverityDistribution;
use crate::special::{gamma_p, ln_gamma, normal_cdf_exact};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Individual claim observation supporting left-truncation (deductible) and right-censoring
/// (policy limit) for actuarial severity fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeverityObservation {
    /// Observed claim amount `x > deductible` (or policy limit `u` when `is_censored == true`).
    pub loss: f64,
    /// Left-truncation threshold / deductible `d >= 0` (`0.0` for ground-up claims).
    pub deductible: f64,
    /// Whether the observation is right-censored at `loss` (e.g. policy-limit-capped claim).
    pub is_censored: bool,
    /// Observation multiplicity / weight (`> 0`, default `1.0`).
    pub weight: f64,
}

impl SeverityObservation {
    /// Exact ground-up uncensored claim observation (`deductible = 0`, `weight = 1`).
    #[inline]
    pub const fn exact(loss: f64) -> Self {
        Self {
            loss,
            deductible: 0.0,
            is_censored: false,
            weight: 1.0,
        }
    }

    /// Uncensored claim left-truncated at `deductible` (`loss > deductible`).
    #[inline]
    pub const fn truncated(loss: f64, deductible: f64) -> Self {
        Self {
            loss,
            deductible,
            is_censored: false,
            weight: 1.0,
        }
    }

    /// Right-censored claim capped at `policy_limit` with left-truncation `deductible`.
    #[inline]
    pub const fn censored(policy_limit: f64, deductible: f64) -> Self {
        Self {
            loss: policy_limit,
            deductible,
            is_censored: true,
            weight: 1.0,
        }
    }

    /// Override observation weight / multiplicity (`> 0`).
    #[inline]
    pub const fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Parametric severity family selector for maximum likelihood fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SeverityFamily {
    /// Exponential (`1` parameter: `lambda`).
    Exponential,
    /// Lognormal (`2` parameters: `mu`, `sigma`).
    Lognormal,
    /// Pareto Type II / Lomax (`2` parameters: `alpha`, `theta`).
    Lomax,
    /// Gamma (`2` parameters: `shape`, `scale`).
    Gamma,
    /// Single-Parameter Pareto with known threshold `x_m > 0` (`1` parameter: `alpha`).
    SingleParameterPareto {
        /// Known minimum threshold `x_m > 0`.
        threshold: f64,
    },
}

/// Fitted parametric severity distribution with information criteria and convergence diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeverityFitResult {
    /// Fitted [`SeverityDistribution`] ready for LEV / ILF / layer pricing.
    pub distribution: SeverityDistribution,
    /// Maximized truncated/censored log-likelihood.
    pub log_likelihood: f64,
    /// Akaike Information Criterion `AIC = 2k - 2 * log_likelihood`.
    pub aic: f64,
    /// Bayesian Information Criterion `BIC = k * ln(n) - 2 * log_likelihood`.
    pub bic: f64,
    /// Total observation weight (`sum w_i`).
    pub total_weight: f64,
    /// Number of right-censored observations in the sample.
    pub n_censored: usize,
    /// Number of optimizer iterations (`1` for closed-form solutions).
    pub iterations: usize,
}

fn validate_observations(
    observations: &[SeverityObservation],
) -> Result<(f64, f64, usize), PricingError> {
    if observations.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one severity observation is required",
        ));
    }
    let mut total_w = 0.0;
    let mut uncensored_w = 0.0;
    let mut n_censored = 0usize;
    for obs in observations {
        if !obs.loss.is_finite() || !obs.deductible.is_finite() || !obs.weight.is_finite() {
            return Err(PricingError::InvalidParameter(
                "severity observation fields must be finite",
            ));
        }
        if !(obs.loss > 0.0) || !(obs.deductible >= 0.0) || !(obs.weight > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "loss and weight must be > 0 and deductible >= 0",
            ));
        }
        if obs.loss <= obs.deductible {
            return Err(PricingError::InvalidParameter(
                "observed loss must be strictly greater than deductible",
            ));
        }
        total_w += obs.weight;
        if obs.is_censored {
            n_censored += 1;
        } else {
            uncensored_w += obs.weight;
        }
    }
    if !(uncensored_w > 0.0) {
        return Err(PricingError::InvalidParameter(
            "at least one uncensored observation is required for MLE",
        ));
    }
    Ok((total_w, uncensored_w, n_censored))
}

#[inline]
fn log_pdf(dist: &SeverityDistribution, x: f64) -> f64 {
    match dist {
        SeverityDistribution::Exponential { lambda } => lambda.ln() - lambda * x,
        SeverityDistribution::Lognormal { mu, sigma } => {
            let z = (x.ln() - mu) / sigma;
            -x.ln() - sigma.ln() - 0.5 * TAU.ln() - 0.5 * z * z
        }
        SeverityDistribution::Lomax { alpha, theta } => {
            alpha.ln() - theta.ln() - (alpha + 1.0) * (x / theta).ln_1p()
        }
        SeverityDistribution::Gamma { shape, scale } => {
            (shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - ln_gamma(*shape)
        }
        SeverityDistribution::SingleParameterPareto { alpha, x_m } => {
            if x < *x_m {
                f64::NEG_INFINITY
            } else {
                alpha.ln() + alpha * x_m.ln() - (alpha + 1.0) * x.ln()
            }
        }
        _ => f64::NEG_INFINITY,
    }
}

#[inline]
fn log_survival(dist: &SeverityDistribution, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    match dist {
        SeverityDistribution::Exponential { lambda } => -lambda * x,
        SeverityDistribution::Lognormal { mu, sigma } => {
            let z = (x.ln() - mu) / sigma;
            let s = normal_cdf_exact(-z);
            if s > 0.0 { s.ln() } else { -1e30 }
        }
        SeverityDistribution::Lomax { alpha, theta } => -alpha * (x / theta).ln_1p(),
        SeverityDistribution::SingleParameterPareto { alpha, x_m } => {
            if x <= *x_m {
                0.0
            } else {
                -alpha * (x / x_m).ln()
            }
        }
        SeverityDistribution::Gamma { shape, scale } => {
            let s = (1.0 - gamma_p(*shape, x / *scale)).clamp(0.0, 1.0);
            if s > 0.0 { s.ln() } else { -1e30 }
        }
        _ => f64::NEG_INFINITY,
    }
}

fn eval_log_likelihood(dist: &SeverityDistribution, observations: &[SeverityObservation]) -> f64 {
    let mut ll = 0.0;
    for obs in observations {
        let trunc = if obs.deductible > 0.0 {
            log_survival(dist, obs.deductible)
        } else {
            0.0
        };
        let contrib = if obs.is_censored {
            log_survival(dist, obs.loss) - trunc
        } else {
            log_pdf(dist, obs.loss) - trunc
        };
        if !contrib.is_finite() {
            return f64::NEG_INFINITY;
        }
        ll += obs.weight * contrib;
    }
    ll
}

/// Deterministic 2D Nelder–Mead minimizer for smooth unconstrained log-parameter objectives.
/// Returns `(best_point, best_value, iterations, converged)`.
fn nelder_mead_2d<F>(
    start: [f64; 2],
    step: [f64; 2],
    max_iter: usize,
    tol: f64,
    f: F,
) -> ([f64; 2], f64, usize, bool)
where
    F: Fn([f64; 2]) -> f64,
{
    let mut pts = [
        start,
        [start[0] + step[0], start[1]],
        [start[0], start[1] + step[1]],
    ];
    let mut vals = [f(pts[0]), f(pts[1]), f(pts[2])];

    let mut iters = 0usize;
    let mut converged = false;
    for iter in 1..=max_iter {
        iters = iter;
        // Sort 3 vertices ascending by objective value (no allocations).
        if vals[1] < vals[0] {
            pts.swap(0, 1);
            vals.swap(0, 1);
        }
        if vals[2] < vals[1] {
            pts.swap(1, 2);
            vals.swap(1, 2);
        }
        if vals[1] < vals[0] {
            pts.swap(0, 1);
            vals.swap(0, 1);
        }

        let spread_f = (vals[2] - vals[0]).abs();
        let spread_x = (pts[1][0] - pts[0][0])
            .abs()
            .max((pts[1][1] - pts[0][1]).abs())
            .max((pts[2][0] - pts[0][0]).abs())
            .max((pts[2][1] - pts[0][1]).abs());
        if spread_f <= tol && spread_x <= tol.sqrt().max(tol) {
            converged = true;
            break;
        }

        let centroid = [
            f64::midpoint(pts[0][0], pts[1][0]),
            f64::midpoint(pts[0][1], pts[1][1]),
        ];

        // Reflection
        let xr = [
            centroid[0] + (centroid[0] - pts[2][0]),
            centroid[1] + (centroid[1] - pts[2][1]),
        ];
        let fr = f(xr);

        if fr < vals[0] {
            // Expansion
            let xe = [
                centroid[0] + 2.0 * (xr[0] - centroid[0]),
                centroid[1] + 2.0 * (xr[1] - centroid[1]),
            ];
            let fe = f(xe);
            if fe < fr {
                pts[2] = xe;
                vals[2] = fe;
            } else {
                pts[2] = xr;
                vals[2] = fr;
            }
        } else if fr < vals[1] {
            pts[2] = xr;
            vals[2] = fr;
        } else {
            // Contraction
            let xc = if fr < vals[2] {
                [
                    centroid[0] + 0.5 * (xr[0] - centroid[0]),
                    centroid[1] + 0.5 * (xr[1] - centroid[1]),
                ]
            } else {
                [
                    centroid[0] + 0.5 * (pts[2][0] - centroid[0]),
                    centroid[1] + 0.5 * (pts[2][1] - centroid[1]),
                ]
            };
            let fc = f(xc);
            if fc < vals[2].min(fr) {
                pts[2] = xc;
                vals[2] = fc;
            } else {
                // Shrink toward best vertex
                for i in 1..3 {
                    pts[i] = [
                        f64::midpoint(pts[0][0], pts[i][0]),
                        f64::midpoint(pts[0][1], pts[i][1]),
                    ];
                    vals[i] = f(pts[i]);
                }
            }
        }
    }

    // Final sort
    if vals[1] < vals[0] {
        pts.swap(0, 1);
        vals.swap(0, 1);
    }
    if vals[2] < vals[0] {
        pts.swap(0, 2);
        vals.swap(0, 2);
    }
    (pts[0], vals[0], iters, converged)
}

fn make_fit_result(
    distribution: SeverityDistribution,
    n_params: usize,
    log_likelihood: f64,
    total_weight: f64,
    n_censored: usize,
    iterations: usize,
) -> Result<SeverityFitResult, PricingError> {
    distribution.validate()?;
    if !log_likelihood.is_finite() {
        return Err(PricingError::NonConvergence(
            "severity MLE failed to obtain a finite log-likelihood",
        ));
    }
    let k = n_params as f64;
    let aic = 2.0 * k - 2.0 * log_likelihood;
    let bic = k * total_weight.max(1.0).ln() - 2.0 * log_likelihood;
    Ok(SeverityFitResult {
        distribution,
        log_likelihood,
        aic,
        bic,
        total_weight,
        n_censored,
        iterations,
    })
}

/// Fit a parametric [`SeverityDistribution`] via Maximum Likelihood Estimation (MLE)
/// supporting per-claim left-truncation (deductibles) and right-censoring (policy limits).
pub fn fit_severity_mle(
    family: SeverityFamily,
    observations: &[SeverityObservation],
    options: &SolverOptions,
) -> Result<SeverityFitResult, PricingError> {
    if options.max_iter == 0 || !options.tol.is_finite() || !(options.tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "solver options require max_iter > 0 and finite tol > 0",
        ));
    }
    let (total_w, uncensored_w, n_censored) = validate_observations(observations)?;

    // Sample moments on uncensored excess/ground-up observations for robust starting values.
    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_lnx = 0.0;
    let mut sum_lnx2 = 0.0;
    let mut has_trunc = false;
    for obs in observations {
        if obs.deductible > 0.0 {
            has_trunc = true;
        }
        if !obs.is_censored {
            sum_x += obs.weight * obs.loss;
            sum_x2 += obs.weight * obs.loss * obs.loss;
            let lnx = obs.loss.ln();
            sum_lnx += obs.weight * lnx;
            sum_lnx2 += obs.weight * lnx * lnx;
        }
    }
    let mean_x = (sum_x / uncensored_w).max(1e-6);
    let var_x = (sum_x2 / uncensored_w - mean_x * mean_x).max(mean_x * mean_x * 0.1);
    let mu0 = sum_lnx / uncensored_w;
    let sigma0 = (sum_lnx2 / uncensored_w - mu0 * mu0).max(0.04).sqrt();

    let max_iter = options.max_iter;

    match family {
        SeverityFamily::Exponential => {
            // Exact closed-form MLE under arbitrary left-truncation and right-censoring:
            // lambda_hat = (sum_{uncensored} w_i) / (sum_{all} w_i * (x_i - d_i))
            let denom: f64 = observations
                .iter()
                .map(|o| o.weight * (o.loss - o.deductible))
                .sum();
            let lambda = uncensored_w / denom;
            let dist = SeverityDistribution::Exponential { lambda };
            let ll = eval_log_likelihood(&dist, observations);
            make_fit_result(dist, 1, ll, total_w, n_censored, 1)
        }
        SeverityFamily::SingleParameterPareto { threshold } => {
            if !threshold.is_finite() || !(threshold > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "SingleParameterPareto threshold must be finite and positive",
                ));
            }
            let mut denom = 0.0;
            for obs in observations {
                if obs.loss <= threshold {
                    return Err(PricingError::InvalidParameter(
                        "all observations for SingleParameterPareto must exceed threshold",
                    ));
                }
                let lower = obs.deductible.max(threshold);
                denom += obs.weight * (obs.loss / lower).ln();
            }
            if !(denom > 0.0) {
                return Err(PricingError::InvalidParameter(
                    "degenerate SingleParameterPareto sample",
                ));
            }
            let alpha = uncensored_w / denom;
            let dist = SeverityDistribution::SingleParameterPareto {
                alpha,
                x_m: threshold,
            };
            let ll = eval_log_likelihood(&dist, observations);
            make_fit_result(dist, 1, ll, total_w, n_censored, 1)
        }
        SeverityFamily::Lognormal => {
            if n_censored == 0 && !has_trunc && uncensored_w > 1.0 {
                let var_ln = (sum_lnx2 / uncensored_w - mu0 * mu0).max(1e-12);
                let dist = SeverityDistribution::Lognormal {
                    mu: mu0,
                    sigma: var_ln.sqrt(),
                };
                let ll = eval_log_likelihood(&dist, observations);
                return make_fit_result(dist, 2, ll, total_w, n_censored, 1);
            }
            let obj = |p: [f64; 2]| {
                let mu = p[0];
                let sigma = p[1].clamp(-12.0, 6.0).exp();
                let dist = SeverityDistribution::Lognormal { mu, sigma };
                let ll = eval_log_likelihood(&dist, observations);
                if ll.is_finite() { -ll } else { 1e30 }
            };
            let (best, neg_ll, iters, converged) =
                nelder_mead_2d([mu0, sigma0.ln()], [0.5, 0.3], max_iter, options.tol, obj);
            if !converged {
                return Err(PricingError::NonConvergence(
                    "Lognormal severity MLE did not converge within max_iter",
                ));
            }
            let dist = SeverityDistribution::Lognormal {
                mu: best[0],
                sigma: best[1].clamp(-12.0, 6.0).exp(),
            };
            make_fit_result(dist, 2, -neg_ll, total_w, n_censored, iters)
        }
        SeverityFamily::Lomax => {
            // Method-of-moments starting guess (if CV > 1) or sensible heavy-tail default.
            let cv2 = var_x / (mean_x * mean_x);
            let alpha0 = if cv2 > 1.05 {
                (2.0 * cv2 / (cv2 - 1.0)).clamp(1.2, 20.0)
            } else {
                2.5
            };
            let theta0 = (mean_x * (alpha0 - 1.0)).max(1e-3);
            let obj = |p: [f64; 2]| {
                let alpha = p[0].clamp(-8.0, 10.0).exp();
                let theta = p[1].clamp(-10.0, 25.0).exp();
                let dist = SeverityDistribution::Lomax { alpha, theta };
                let ll = eval_log_likelihood(&dist, observations);
                if ll.is_finite() { -ll } else { 1e30 }
            };
            let (best, neg_ll, iters, converged) = nelder_mead_2d(
                [alpha0.ln(), theta0.ln()],
                [0.4, 0.6],
                max_iter,
                options.tol,
                obj,
            );
            if !converged {
                return Err(PricingError::NonConvergence(
                    "Lomax severity MLE did not converge within max_iter",
                ));
            }
            let dist = SeverityDistribution::Lomax {
                alpha: best[0].clamp(-8.0, 10.0).exp(),
                theta: best[1].clamp(-10.0, 25.0).exp(),
            };
            make_fit_result(dist, 2, -neg_ll, total_w, n_censored, iters)
        }
        SeverityFamily::Gamma => {
            let shape0 = (mean_x * mean_x / var_x).clamp(0.1, 100.0);
            let scale0 = (var_x / mean_x).max(1e-6);
            let obj = |p: [f64; 2]| {
                let shape = p[0].clamp(-6.0, 8.0).exp();
                let scale = p[1].clamp(-10.0, 25.0).exp();
                let dist = SeverityDistribution::Gamma { shape, scale };
                let ll = eval_log_likelihood(&dist, observations);
                if ll.is_finite() { -ll } else { 1e30 }
            };
            let (best, neg_ll, iters, converged) = nelder_mead_2d(
                [shape0.ln(), scale0.ln()],
                [0.35, 0.5],
                max_iter,
                options.tol,
                obj,
            );
            if !converged {
                return Err(PricingError::NonConvergence(
                    "Gamma severity MLE did not converge within max_iter",
                ));
            }
            let dist = SeverityDistribution::Gamma {
                shape: best[0].clamp(-6.0, 8.0).exp(),
                scale: best[1].clamp(-10.0, 25.0).exp(),
            };
            make_fit_result(dist, 2, -neg_ll, total_w, n_censored, iters)
        }
    }
}

/// Fit multiple candidate [`SeverityFamily`] models and return them ranked from best to worst
/// by Akaike Information Criterion (`AIC`).
pub fn fit_best_severity(
    families: &[SeverityFamily],
    observations: &[SeverityObservation],
    options: &SolverOptions,
) -> Result<Vec<SeverityFitResult>, PricingError> {
    if families.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one candidate severity family is required",
        ));
    }
    let mut results = Vec::with_capacity(families.len());
    for &family in families {
        results.push(fit_severity_mle(family, observations, options)?);
    }
    results.sort_by(|a, b| a.aic.total_cmp(&b.aic));
    Ok(results)
}
