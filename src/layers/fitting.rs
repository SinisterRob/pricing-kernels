//! Maximum Likelihood Estimation (MLE) for parametric severity distributions under
//! left-truncation (per-occurrence deductibles) and right-censoring (policy limits).
//!
//! Reference: S. A. Klugman, H. H. Panjer & G. E. Willmot, *Loss Models: From Data to
//! Decisions* (5th ed.), Ch. 13.

use crate::error::{PricingError, SolverOptions};
use crate::layers::SeverityDistribution;
use crate::special::{ln_gamma, log_gamma_q, log_normal_survival};
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
    /// Whether the optimizer converged at an interior point (`true` for closed forms).
    /// `false` exactly when [`Self::at_bound`] is set; interior non-convergence is an error.
    pub converged: bool,
    /// Whether the optimum lies on the internal overflow guard `|p_j| <= LOG_PARAM_BOUND` of the
    /// scale-normalized log coordinates, i.e. the likelihood supremum is approached at a
    /// degenerate limit (e.g. Lomax `alpha -> inf` on a sample with CV < 1) and the reported
    /// parameters are not an MLE.
    pub at_bound: bool,
}

/// Overflow guard on every scale-normalized log coordinate of the iterative fits
/// (`e^30 ≈ 1.1e13` relative to the data scale); not a modelling constraint.
pub const LOG_PARAM_BOUND: f64 = 30.0;
/// An optimum within this distance of [`LOG_PARAM_BOUND`] is reported with `at_bound = true`.
pub const LOG_PARAM_BOUND_MARGIN: f64 = 1.0;
/// Lower clamp on the starting squared coefficient of variation.
const MIN_START_CV2: f64 = 1e-12;
/// Upper clamp on the starting squared coefficient of variation.
const MAX_START_CV2: f64 = 1e12;
/// Lower clamp on the starting log-variance for Lognormal.
const MIN_START_LOG_VARIANCE: f64 = 0.04;
/// Minimum CV² for the Lomax method-of-moments start `alpha = 2 CV² / (CV² - 1)`.
const LOMAX_MOM_MIN_CV2: f64 = 1.05;
/// Lomax starting `alpha` for light-tailed samples (CV² ≤ `LOMAX_MOM_MIN_CV2`).
const LOMAX_LIGHT_TAIL_START_ALPHA: f64 = 10.0;
/// Floor on `alpha - 1` in the Lomax starting scale.
const LOMAX_MIN_START_EXCESS: f64 = 0.05;

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
            log_normal_survival(z)
        }
        SeverityDistribution::Lomax { alpha, theta } => -alpha * (x / theta).ln_1p(),
        SeverityDistribution::SingleParameterPareto { alpha, x_m } => {
            if x <= *x_m {
                0.0
            } else {
                -alpha * (x / x_m).ln()
            }
        }
        SeverityDistribution::Gamma { shape, scale } => log_gamma_q(*shape, x / *scale),
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
/// Stops when the objective spread is `<= tol (1 + |f_best|)` and the simplex diameter is
/// `<= sqrt(tol)`.
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
        // Relative objective spread: the summed log-likelihood carries O(n·ulp·|ll|) round-off,
        // so an absolute `tol` is unreachable for large samples.
        if spread_f <= tol * (1.0 + vals[0].abs()) && spread_x <= tol.sqrt().max(tol) {
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
    at_bound: bool,
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
        converged: !at_bound,
        at_bound,
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
    // Data scale for scale-equivariant optimizer coordinates (weighted mean of uncensored losses).
    let scale_ref = sum_x / uncensored_w;
    if !scale_ref.is_finite() || !(scale_ref > 0.0) {
        return Err(PricingError::InvalidParameter(
            "severity sample mean must be finite and positive",
        ));
    }
    let ln_scale = scale_ref.ln();
    let mean_x = scale_ref;
    let var_x = (sum_x2 / uncensored_w - mean_x * mean_x)
        .max(MIN_START_CV2 * mean_x * mean_x)
        .min(MAX_START_CV2 * mean_x * mean_x);
    let mu0 = sum_lnx / uncensored_w;
    let sigma0 = (sum_lnx2 / uncensored_w - mu0 * mu0)
        .max(MIN_START_LOG_VARIANCE)
        .sqrt();
    let fit_ctx = FitContext {
        observations,
        options,
        total_w,
        n_censored,
    };

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
            make_fit_result(dist, 1, ll, total_w, n_censored, 1, false)
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
            make_fit_result(dist, 1, ll, total_w, n_censored, 1, false)
        }
        SeverityFamily::Lognormal => {
            if n_censored == 0 && !has_trunc && uncensored_w > 1.0 {
                let var_ln = (sum_lnx2 / uncensored_w - mu0 * mu0).max(1e-12);
                let dist = SeverityDistribution::Lognormal {
                    mu: mu0,
                    sigma: var_ln.sqrt(),
                };
                let ll = eval_log_likelihood(&dist, observations);
                return make_fit_result(dist, 2, ll, total_w, n_censored, 1, false);
            }
            // p = (mu - ln m, ln sigma)
            let unpack = |p: [f64; 2]| SeverityDistribution::Lognormal {
                mu: p[0] + ln_scale,
                sigma: p[1].exp(),
            };
            fit_ctx.optimize(
                [mu0 - ln_scale, sigma0.ln()],
                [0.5, 0.3],
                unpack,
                "Lognormal severity MLE did not converge within max_iter",
            )
        }
        SeverityFamily::Lomax => {
            // Method-of-moments starting guess (if CV > 1) or a near-exponential default.
            // p = (ln alpha, ln(theta / alpha) - ln m): the second coordinate is the scale of
            // the exponential limit (alpha -> inf, theta / alpha fixed), so the objective stays
            // well conditioned for light-tailed samples and is invariant to rescaling the data.
            let cv2 = var_x / (mean_x * mean_x);
            let alpha0 = if cv2 > LOMAX_MOM_MIN_CV2 {
                2.0 * cv2 / (cv2 - 1.0)
            } else {
                LOMAX_LIGHT_TAIL_START_ALPHA
            };
            let theta_over_alpha0 = mean_x * (alpha0 - 1.0).max(LOMAX_MIN_START_EXCESS) / alpha0;
            let unpack = |p: [f64; 2]| {
                let alpha = p[0].exp();
                SeverityDistribution::Lomax {
                    alpha,
                    theta: alpha * (p[1] + ln_scale).exp(),
                }
            };
            fit_ctx.optimize(
                [alpha0.ln(), theta_over_alpha0.ln() - ln_scale],
                [0.4, 0.6],
                unpack,
                "Lomax severity MLE did not converge within max_iter",
            )
        }
        SeverityFamily::Gamma => {
            // p = (ln shape, ln(shape * scale) - ln m): the mean coordinate is nearly
            // orthogonal to the shape (exactly so for uncensored data) and scale-equivariant.
            let shape0 = mean_x * mean_x / var_x;
            let unpack = |p: [f64; 2]| {
                let shape = p[0].exp();
                SeverityDistribution::Gamma {
                    shape,
                    scale: (p[1] + ln_scale).exp() / shape,
                }
            };
            fit_ctx.optimize(
                [shape0.ln(), 0.0],
                [0.35, 0.2],
                unpack,
                "Gamma severity MLE did not converge within max_iter",
            )
        }
    }
}

/// Shared inputs of the iterative two-parameter fits.
struct FitContext<'a> {
    observations: &'a [SeverityObservation],
    options: &'a SolverOptions,
    total_w: f64,
    n_censored: usize,
}

impl FitContext<'_> {
    /// Minimize `-ll` over scale-normalized log coordinates `p` with `|p_j| <= LOG_PARAM_BOUND`
    /// (an overflow guard far beyond realistic parameters, not a modelling constraint), then
    /// restart once from the optimum to guard against simplex collapse.
    ///
    /// An optimum within [`LOG_PARAM_BOUND_MARGIN`] of the guard is returned as `Ok` with
    /// `at_bound = true` and `converged = false` (the MLE does not exist in the interior, e.g.
    /// Lomax on a sample with CV < 1, whose supremum is the exponential limit). An interior
    /// point that fails to converge within `max_iter` is an error.
    fn optimize<U>(
        &self,
        start: [f64; 2],
        step: [f64; 2],
        unpack: U,
        non_convergence: &'static str,
    ) -> Result<SeverityFitResult, PricingError>
    where
        U: Fn([f64; 2]) -> SeverityDistribution,
    {
        let obj = |p: [f64; 2]| {
            if p.iter()
                .any(|v| !v.is_finite() || v.abs() > LOG_PARAM_BOUND)
            {
                return f64::INFINITY;
            }
            let ll = eval_log_likelihood(&unpack(p), self.observations);
            if ll.is_finite() { -ll } else { f64::INFINITY }
        };
        let start = start.map(|v| v.clamp(-LOG_PARAM_BOUND, LOG_PARAM_BOUND));
        let (tol, max_iter) = (self.options.tol, self.options.max_iter);
        let (first, f_first, it_first, conv_first) =
            nelder_mead_2d(start, step, max_iter, tol, obj);
        let (best, neg_ll, iters, converged) = if conv_first {
            let (second, f_second, it_second, conv_second) =
                nelder_mead_2d(first, step, max_iter, tol, obj);
            if f_second <= f_first {
                (second, f_second, it_first + it_second, conv_second)
            } else {
                (first, f_first, it_first + it_second, conv_first)
            }
        } else {
            (first, f_first, it_first, false)
        };
        let at_bound = best
            .iter()
            .any(|v| v.abs() > LOG_PARAM_BOUND - LOG_PARAM_BOUND_MARGIN);
        if !converged && !at_bound {
            return Err(PricingError::NonConvergence(non_convergence));
        }
        make_fit_result(
            unpack(best),
            2,
            -neg_ll,
            self.total_w,
            self.n_censored,
            iters,
            at_bound,
        )
    }
}

/// Fit multiple candidate [`SeverityFamily`] models and return the converged fits ranked from
/// best to worst by Akaike Information Criterion (`AIC`).
///
/// Fits with `converged == false` (optimum on the overflow guard, see
/// [`SeverityFitResult::at_bound`]) are excluded from the ranking: their parameters are not an
/// MLE and a degenerate limit (e.g. Lomax → Exponential) would be double-counted with a
/// spurious extra parameter. Errors with [`PricingError::NonConvergence`] if no candidate
/// converges; any other per-family error is propagated.
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
        let fit = fit_severity_mle(family, observations, options)?;
        if fit.converged {
            results.push(fit);
        }
    }
    if results.is_empty() {
        return Err(PricingError::NonConvergence(
            "no candidate severity family converged to an interior MLE",
        ));
    }
    results.sort_by(|a, b| a.aic.total_cmp(&b.aic));
    Ok(results)
}
