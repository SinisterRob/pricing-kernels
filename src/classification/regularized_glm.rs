//! Regularized GLMs (Elastic Net / Ridge / LASSO via Penalized IRLS + Cyclic
//! Coordinate Descent), B-spline & Natural Cubic Spline bases, Negative Binomial /
//! Hurdle / Zero-Inflated Poisson frequency models, and Bühlmann-in-GLM /
//! Laplace-approximated Random-Intercept GLMMs for high-cardinality territories
//! and class codes.

use crate::error::PricingError;
use crate::linalg::cholesky_solve;
use crate::special::{digamma, ln_gamma};
use crate::validate::{
    require_finite, require_non_negative, require_positive, require_unit_interval,
};
use serde::{Deserialize, Serialize};

/// Overflow-safe bound on the linear predictor `|η|` (`exp(709.78)` is the `f64` limit).
/// A fit whose linear predictor leaves `[-ETA_OVERFLOW_BOUND, ETA_OVERFLOW_BOUND]` is
/// reported as [`PricingError::NonConvergence`] (prediction: [`PricingError::OutOfBounds`]).
pub const ETA_OVERFLOW_BOUND: f64 = 700.0;
/// Columns with weighted standard deviation below this are treated as constant and dropped.
const ZERO_SD_TOL: f64 = 1e-12;
/// Inner coordinate-descent stopping threshold on the largest coordinate change.
const CD_INNER_TOL: f64 = 1e-10;
/// Initial Negative Binomial dispersion `θ` when estimated by profile MLE.
const NB_THETA_INIT: f64 = 5.0;
/// Floor on the initial zero-truncated Poisson rate `λ_0 = max(ȳ - 1, floor)`.
const ZTP_INIT_RATE_FLOOR: f64 = 0.1;
/// Below this rate `λ`, zero-truncated Poisson mean/variance use their Taylor series.
const ZTP_SERIES_THRESHOLD: f64 = 1e-4;
/// Floor on IRLS working weights `V(μ)` used as divisors.
const IRLS_WEIGHT_FLOOR: f64 = 1e-12;
/// Floor on the initial mean `ȳ` used to set the intercept start `ln(ȳ)`.
const MEAN_INIT_FLOOR: f64 = 1e-4;
/// Probability clamp for ZIP E-step structural-zero probabilities and M-step weights.
const ZIP_PROB_EPS: f64 = 1e-9;
/// EM convergence threshold on the largest change in ZIP posterior structural-zero probabilities.
const ZIP_EM_TOL: f64 = 1e-6;
/// GLMM joint-mode Newton stop: Newton decrement `½ Δθᵀ∇h` (nats).
const GLMM_NEWTON_DECREMENT_TOL: f64 = 1e-10;
/// Maximum step-halvings per GLMM joint-mode Newton iteration.
const GLMM_MAX_STEP_HALVINGS: usize = 30;
/// Relative round-off slack accepted when testing ascent of the GLMM penalized log-likelihood.
const GLMM_ASCENT_SLACK: f64 = 1e-12;
/// Grid step on `ln τ²` for the GLMM variance search.
const GLMM_LN_TAU2_GRID_STEP: f64 = 1.0;
/// Initial upper end of the `ln τ²` grid (expanded while the maximum sits at the top).
const GLMM_LN_TAU2_GRID_HI: f64 = 2.0;
/// Golden-section stopping width on `ln τ²`.
const GLMM_LN_TAU2_TOL: f64 = 1e-6;
/// Maximum golden-section iterations on `ln τ²`.
const GLMM_TAU_GOLDEN_MAX_ITERS: usize = 100;
/// Hard upper bound on the GLMM random-effect variance `τ²` (log-scale sd 100).
const GLMM_TAU2_MAX: f64 = 1e4;
/// Lower bound on the GLMM random-effect variance `τ²`.
const GLMM_TAU2_FLOOR: f64 = 1e-8;
/// Clamp `[eps, 1 - eps]` on the mean response used to initialize Binomial intercepts.
const BINOMIAL_INIT_EPS: f64 = 1e-4;
/// Floor on Binomial variance / derivative terms used as IRLS divisors.
const BINOMIAL_VAR_FLOOR: f64 = 1e-9;

/// Return `lin` if it is finite and within [`ETA_OVERFLOW_BOUND`], else `NonConvergence`.
#[inline]
fn checked_eta(lin: f64) -> Result<f64, PricingError> {
    if lin.is_finite() && lin.abs() <= ETA_OVERFLOW_BOUND {
        Ok(lin)
    } else {
        Err(PricingError::NonConvergence(
            "linear predictor left the overflow-safe range |eta| <= ETA_OVERFLOW_BOUND",
        ))
    }
}

/// Zero-truncated Poisson conditional mean `m = λ / (1 - e^{-λ})` and variance
/// `v = m (1 + λ - m)` (`= dm/dη` for `η = ln λ`), with Taylor series for small `λ`.
#[inline]
fn ztp_mean_var(lambda: f64) -> (f64, f64) {
    if lambda < ZTP_SERIES_THRESHOLD {
        let l2 = lambda * lambda;
        (1.0 + 0.5 * lambda + l2 / 12.0, 0.5 * lambda + l2 / 6.0)
    } else {
        let m = lambda / -(-lambda).exp_m1();
        (m, m * (1.0 + lambda - m))
    }
}

/// Complementary log-log inverse link `π = 1 - exp(-e^η)` (cancellation-free via `expm1`).
#[inline]
fn cloglog_mean(eta: f64) -> f64 {
    -(-eta.exp()).exp_m1()
}

/// Inverse link / conditional mean for a regularized family at linear predictor `e`.
#[inline]
fn family_mean(family: RegularizedGlmFamily, e: f64) -> f64 {
    match family {
        RegularizedGlmFamily::BinomialLogit => 1.0 / (1.0 + (-e).exp()),
        RegularizedGlmFamily::BinomialCloglog => cloglog_mean(e),
        RegularizedGlmFamily::ZeroTruncatedPoisson => ztp_mean_var(e.exp()).0,
        _ => e.exp(),
    }
}

// ===========================================================================
// 3.1 Penalized IRLS + Cyclic Coordinate Descent (Elastic Net GLM)
// ===========================================================================

/// Exponential-dispersion family for [`fit_regularized_glm`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RegularizedGlmFamily {
    /// Poisson log-link (`V(μ) = μ`).
    Poisson,
    /// Gamma log-link (`V(μ) = μ^2`).
    Gamma,
    /// Tweedie log-link (`V(μ) = μ^p`, `p ∈ (1, 2)`).
    Tweedie {
        /// Tweedie variance power `p ∈ (1, 2)`.
        power: f64,
    },
    /// Negative Binomial log-link (`V(μ) = μ + μ^2 / θ`) with optional fixed
    /// dispersion `theta` (`None` estimates `θ` via 1D Newton–Raphson profile MLE).
    NegativeBinomial {
        /// Optional fixed dispersion `θ > 0` (`None` = estimate via profile MLE).
        fixed_theta: Option<f64>,
    },
    /// Binary Logistic (`logit(π) = η`, used internally by ZIP and standalone).
    BinomialLogit,
    /// Binary complementary log-log (`ln(-ln(1 - π)) = η`, i.e. `π = 1 - exp(-e^η)`),
    /// fitted by Fisher scoring (McCullagh & Nelder 1989, §4.3.1). With offset `ln E_i`
    /// this is `P(N_i > 0)` for a Poisson process with rate `e^{x_i^T β}` over exposure
    /// `E_i`, so it is exposure-consistent; used by the Hurdle zero part.
    BinomialCloglog,
    /// Zero-truncated Poisson for positive counts `y >= 1` (`ln λ = η`,
    /// `E[Y | Y > 0] = λ / (1 - e^{-λ})`), fitted by exact Fisher scoring on
    /// `Σ w [y η - e^η - ln(1 - e^{-e^η})]`. Used by the Hurdle count part.
    /// `fitted_means` / `predict` return the conditional mean `E[Y | Y > 0]`.
    ZeroTruncatedPoisson,
}

/// Configuration for [`fit_regularized_glm`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegularizedGlmOptions {
    /// Distribution family.
    pub family: RegularizedGlmFamily,
    /// Overall regularization strength `λ >= 0` (`0.0` = unpenalized).
    pub lambda: f64,
    /// Elastic Net mixing parameter `α ∈ [0, 1]` (`0.0` = Ridge, `1.0` = LASSO).
    pub alpha: f64,
    /// Maximum outer IRLS iterations.
    pub max_outer_iter: usize,
    /// Maximum inner coordinate-descent sweeps per IRLS step.
    pub max_inner_iter: usize,
    /// Convergence tolerance for outer IRLS objective and parameter change.
    pub tolerance: f64,
}

impl Default for RegularizedGlmOptions {
    fn default() -> Self {
        Self {
            family: RegularizedGlmFamily::Poisson,
            lambda: 0.0,
            alpha: 0.0,
            max_outer_iter: 150,
            max_inner_iter: 200,
            tolerance: 1e-8,
        }
    }
}

/// Output of [`fit_regularized_glm`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegularizedGlmResult {
    /// Unpenalized intercept `β_0` on the original feature scale.
    pub intercept: f64,
    /// Penalized slope coefficients `β_1..β_p` on the original feature scale.
    pub coefficients: Vec<f64>,
    /// Fitted means `μ_i` on the training observations.
    pub fitted_means: Vec<f64>,
    /// Fitted Negative Binomial dispersion `θ` (`Some` only for `NegativeBinomial`).
    pub nb_theta: Option<f64>,
    /// Penalized objective value at convergence.
    pub objective: f64,
    /// Outer IRLS iterations executed.
    pub iterations: usize,
    /// Whether outer IRLS converged within tolerance.
    pub converged: bool,
}

impl RegularizedGlmResult {
    /// Predict expected responses `μ_i = g⁻¹(β_0 + x_i^T β + offset_i)` on new data.
    pub fn predict(
        &self,
        family: RegularizedGlmFamily,
        x: &[f64],
        n: usize,
        p: usize,
        offsets: Option<&[f64]>,
    ) -> Result<Vec<f64>, PricingError> {
        if n == 0 || p == 0 {
            return Err(PricingError::EmptyInput("n and p must be >= 1"));
        }
        if p != self.coefficients.len() || n.checked_mul(p) != Some(x.len()) {
            return Err(PricingError::DimensionMismatch(
                "x dimensions must match n * coefficients.len()",
            ));
        }
        if let Some(os) = offsets
            && os.len() != n
        {
            return Err(PricingError::DimensionMismatch(
                "offsets length must match n",
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PricingError::InvalidParameter(
                "design matrix entries must be finite",
            ));
        }
        if let Some(os) = offsets
            && os.iter().any(|v| !v.is_finite())
        {
            return Err(PricingError::InvalidParameter("offsets must be finite"));
        }

        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let mut eta = self.intercept + offsets.map_or(0.0, |os| os[i]);
            let row = &x[i * p..(i + 1) * p];
            for j in 0..p {
                eta += row[j] * self.coefficients[j];
            }
            if !(eta.is_finite() && eta.abs() <= ETA_OVERFLOW_BOUND) {
                return Err(PricingError::OutOfBounds(
                    "linear predictor outside |eta| <= ETA_OVERFLOW_BOUND",
                ));
            }
            preds.push(family_mean(family, eta));
        }
        Ok(preds)
    }

    /// Return `(name, exp(beta_j))` multiplicative relativities (intercept first, then slopes).
    pub fn relativities(
        &self,
        feature_names: Option<&[&str]>,
    ) -> Result<Vec<(String, f64)>, PricingError> {
        if let Some(names) = feature_names
            && names.len() != self.coefficients.len()
        {
            return Err(PricingError::DimensionMismatch(
                "feature_names length must match number of slope coefficients",
            ));
        }
        let mut out = Vec::with_capacity(self.coefficients.len() + 1);
        out.push(("intercept".to_owned(), self.intercept.exp()));
        for (j, &b) in self.coefficients.iter().enumerate() {
            let name = match feature_names {
                Some(names) => names[j].to_owned(),
                None => format!("x{}", j + 1),
            };
            out.push((name, b.exp()));
        }
        Ok(out)
    }
}

#[inline]
fn soft_threshold(u: f64, tau: f64) -> f64 {
    if u > tau {
        u - tau
    } else if u < -tau {
        u + tau
    } else {
        0.0
    }
}

fn eval_neg_ll(
    family: RegularizedGlmFamily,
    y: &[f64],
    eta: &[f64],
    weights: &[f64],
    nb_theta: f64,
) -> f64 {
    let mut nll = 0.0;
    for i in 0..y.len() {
        let wi = weights[i];
        let yi = y[i];
        let ei = eta[i];
        let term = match family {
            RegularizedGlmFamily::Poisson => ei.exp() - yi * ei,
            RegularizedGlmFamily::Gamma => yi * (-ei).exp() + ei,
            RegularizedGlmFamily::Tweedie { power } => {
                let mu = ei.exp();
                -(yi * mu.powf(1.0 - power) / (1.0 - power) - mu.powf(2.0 - power) / (2.0 - power))
            }
            RegularizedGlmFamily::NegativeBinomial { .. } => {
                let mu = ei.exp();
                let th = nb_theta;
                -(ln_gamma(yi + th) - ln_gamma(th) + th * th.ln() + yi * mu.max(1e-300).ln()
                    - (yi + th) * (mu + th).ln())
            }
            RegularizedGlmFamily::BinomialLogit => {
                // ln(1 + exp(ei)) - yi * ei
                let log1p_exp = if ei > 0.0 {
                    ei + (-ei).exp().ln_1p()
                } else {
                    ei.exp().ln_1p()
                };
                log1p_exp - yi * ei
            }
            RegularizedGlmFamily::BinomialCloglog => {
                // -[y ln π + (1 - y) ln(1 - π)] with ln(1 - π) = -e^η.
                let lam = ei.exp();
                -yi * (-(-lam).exp_m1()).ln() + (1.0 - yi) * lam
            }
            RegularizedGlmFamily::ZeroTruncatedPoisson => {
                // e^η - y η + ln(1 - e^{-e^η})
                let lam = ei.exp();
                lam - yi * ei + (-(-lam).exp_m1()).ln()
            }
        };
        nll += wi * term;
    }
    nll
}

/// 1D Newton–Raphson profile MLE for Negative Binomial dispersion `θ > 0`
/// using `digamma` and finite-difference trigamma.
fn update_nb_theta(y: &[f64], mu: &[f64], weights: &[f64], init_theta: f64) -> f64 {
    let score = |log_th: f64| -> f64 {
        let th = log_th.exp();
        let mut s = 0.0;
        for i in 0..y.len() {
            let yi = y[i];
            let mui = mu[i];
            s += weights[i]
                * (digamma(yi + th) - digamma(th) + th.ln() + 1.0
                    - (mui + th).ln()
                    - (yi + th) / (mui + th));
        }
        s * th // d/d(ln theta)
    };

    let mut log_th = init_theta.clamp(1e-3, 1e8).ln();
    let h = 1e-4;
    for _ in 0..25 {
        let g = score(log_th);
        if !g.is_finite() || g.abs() < 1e-8 {
            break;
        }
        let d = (score(log_th + h) - score(log_th - h)) / (2.0 * h);
        if !d.is_finite() || d >= -1e-12 {
            // Concave ascent step fallback.
            log_th = (log_th + 0.1 * g.signum()).clamp(-6.0, 18.0);
        } else {
            let step = (-g / d).clamp(-1.5, 1.5);
            log_th = (log_th + step).clamp(-6.0, 18.0);
        }
    }
    log_th.exp()
}

/// Fit a penalized Elastic Net GLM (`Poisson`, `Gamma`, `Tweedie`,
/// `NegativeBinomial`, `BinomialLogit`, `BinomialCloglog`, or `ZeroTruncatedPoisson`)
/// via Penalized IRLS with cyclic coordinate descent.
///
/// * `x` — row-major `n × p` feature matrix (without intercept column; `p >= 1`).
/// * `y` — response vector of length `n`.
/// * `weights` — optional positive observation weights (defaults to `1.0`).
/// * `offsets` — optional linear-predictor offsets `o_i` (e.g. `ln(exposure_i)`, defaults to `0.0`).
pub fn fit_regularized_glm(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    offsets: Option<&[f64]>,
    options: RegularizedGlmOptions,
) -> Result<RegularizedGlmResult, PricingError> {
    if n == 0 || p == 0 {
        return Err(PricingError::EmptyInput(
            "design matrix must have n > 0 and p > 0",
        ));
    }
    if n.checked_mul(p) != Some(x.len()) || y.len() != n {
        return Err(PricingError::DimensionMismatch(
            "x must have length n * p and y must have length n",
        ));
    }
    require_non_negative(options.lambda, "lambda must be >= 0")?;
    require_unit_interval(options.alpha, "alpha must be in [0, 1]")?;
    require_positive(options.tolerance, "tolerance must be > 0")?;
    if options.max_outer_iter == 0 || options.max_inner_iter == 0 {
        return Err(PricingError::InvalidParameter(
            "max_outer_iter and max_inner_iter must be >= 1",
        ));
    }
    if let RegularizedGlmFamily::Tweedie { power } = options.family
        && !(power.is_finite() && power > 1.0 && power < 2.0)
    {
        return Err(PricingError::InvalidParameter(
            "Tweedie power must lie in (1, 2)",
        ));
    }
    if let RegularizedGlmFamily::NegativeBinomial {
        fixed_theta: Some(th),
    } = options.family
    {
        require_positive(th, "fixed_theta must be > 0")?;
    }

    let w_vec: Vec<f64> = match weights {
        Some(ws) => {
            if ws.len() != n {
                return Err(PricingError::DimensionMismatch(
                    "weights length must equal n",
                ));
            }
            for &wi in ws {
                require_non_negative(wi, "weights must be >= 0")?;
            }
            ws.to_vec()
        }
        None => vec![1.0; n],
    };
    let o_vec: Vec<f64> = match offsets {
        Some(os) => {
            if os.len() != n {
                return Err(PricingError::DimensionMismatch(
                    "offsets length must equal n",
                ));
            }
            for &oi in os {
                require_finite(oi, "offsets must be finite")?;
            }
            os.to_vec()
        }
        None => vec![0.0; n],
    };

    for &xi in x {
        require_finite(xi, "design matrix entries must be finite")?;
    }
    for &yi in y {
        require_finite(yi, "response values must be finite")?;
        match options.family {
            RegularizedGlmFamily::Gamma => require_positive(yi, "Gamma response must be > 0")?,
            RegularizedGlmFamily::BinomialLogit | RegularizedGlmFamily::BinomialCloglog => {
                require_unit_interval(yi, "Binomial response must be in [0, 1]")?;
            }
            RegularizedGlmFamily::ZeroTruncatedPoisson => {
                if !(yi >= 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "ZeroTruncatedPoisson response must be >= 1",
                    ));
                }
            }
            _ => require_non_negative(yi, "count/Tweedie response must be >= 0")?,
        }
    }

    let total_w: f64 = w_vec.iter().sum();
    require_positive(total_w, "sum of weights must be > 0")?;

    // Weighted column standardization (dropping zero-variance columns cleanly).
    let mut col_means = vec![0.0; p];
    let mut col_sds = vec![0.0; p];
    let mut x_std = vec![0.0; n * p]; // column-major: col j at j*n..(j+1)*n
    for j in 0..p {
        let mut m = 0.0;
        for i in 0..n {
            m += w_vec[i] * x[i * p + j];
        }
        m /= total_w;
        col_means[j] = m;
        let mut var = 0.0;
        for i in 0..n {
            let d = x[i * p + j] - m;
            var += w_vec[i] * d * d;
        }
        let sd = (var / total_w).sqrt();
        col_sds[j] = sd;
        if sd >= ZERO_SD_TOL {
            for i in 0..n {
                x_std[j * n + i] = (x[i * p + j] - m) / sd;
            }
        }
    }

    // Initialize intercept from weighted mean response.
    let y_bar: f64 = y
        .iter()
        .zip(w_vec.iter())
        .map(|(yi, wi)| yi * wi)
        .sum::<f64>()
        / total_w;
    let off_bar: f64 = o_vec
        .iter()
        .zip(w_vec.iter())
        .map(|(oi, wi)| oi * wi)
        .sum::<f64>()
        / total_w;
    let mut beta0 = if options.family == RegularizedGlmFamily::BinomialLogit {
        let p0 = y_bar.clamp(BINOMIAL_INIT_EPS, 1.0 - BINOMIAL_INIT_EPS);
        (p0 / (1.0 - p0)).ln()
    } else if options.family == RegularizedGlmFamily::BinomialCloglog {
        let p0 = y_bar.clamp(BINOMIAL_INIT_EPS, 1.0 - BINOMIAL_INIT_EPS);
        (-(-p0).ln_1p()).ln() - off_bar
    } else {
        let start_mean = if options.family == RegularizedGlmFamily::ZeroTruncatedPoisson {
            (y_bar - 1.0).max(ZTP_INIT_RATE_FLOOR)
        } else {
            y_bar.max(MEAN_INIT_FLOOR)
        };
        start_mean.ln() - off_bar
    };
    let mut beta = vec![0.0; p];
    let mut nb_theta = match options.family {
        RegularizedGlmFamily::NegativeBinomial { fixed_theta } => {
            fixed_theta.unwrap_or(NB_THETA_INIT)
        }
        _ => 1.0,
    };

    let mut eta = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut irls_w = vec![0.0; n];
    let mut z_work = vec![0.0; n];
    let mut resid = vec![0.0; n];

    let mut prev_obj = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;

    let l1_pen = options.lambda * options.alpha;
    let l2_pen = options.lambda * (1.0 - options.alpha);

    for outer in 0..options.max_outer_iter {
        iterations = outer + 1;
        let prev_beta0 = beta0;
        let prev_beta = beta.clone();

        // Compute linear predictor, mean, IRLS working weights, and working response.
        for i in 0..n {
            let mut lin = beta0 + o_vec[i];
            for j in 0..p {
                if beta[j] != 0.0 {
                    lin += x_std[j * n + i] * beta[j];
                }
            }
            let e = checked_eta(lin)?;
            eta[i] = e;
            match options.family {
                RegularizedGlmFamily::Poisson => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m;
                    z_work[i] = e + (y[i] - m) / m.max(IRLS_WEIGHT_FLOOR);
                }
                RegularizedGlmFamily::Gamma => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i];
                    z_work[i] = e + (y[i] - m) / m.max(IRLS_WEIGHT_FLOOR);
                }
                RegularizedGlmFamily::Tweedie { power } => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m.powf(2.0 - power);
                    z_work[i] = e + (y[i] - m) / m.max(IRLS_WEIGHT_FLOOR);
                }
                RegularizedGlmFamily::NegativeBinomial { .. } => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m / (1.0 + m / nb_theta);
                    z_work[i] = e + (y[i] - m) / m.max(IRLS_WEIGHT_FLOOR);
                }
                RegularizedGlmFamily::ZeroTruncatedPoisson => {
                    // Canonical (natural-parameter) family in η: Newton = Fisher scoring,
                    // score y - m, information Var(Y | Y > 0) = m (1 + λ - m).
                    let (m, v) = ztp_mean_var(e.exp());
                    let v = v.max(IRLS_WEIGHT_FLOOR);
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * v;
                    z_work[i] = e + (y[i] - m) / v;
                }
                RegularizedGlmFamily::BinomialLogit => {
                    let prob = 1.0 / (1.0 + (-e).exp());
                    mu[i] = prob;
                    let pq = (prob * (1.0 - prob)).max(BINOMIAL_VAR_FLOOR);
                    irls_w[i] = w_vec[i] * pq;
                    z_work[i] = e + (y[i] - prob) / pq;
                }
                RegularizedGlmFamily::BinomialCloglog => {
                    // Fisher scoring: dπ/dη = λ (1 - π) with λ = e^η, working weight
                    // (dπ/dη)² / (π (1 - π)) = λ² (1 - π) / π, z = η + (y - π) / (dπ/dη).
                    let lam = e.exp();
                    let prob = cloglog_mean(e);
                    let surv = (-lam).exp();
                    mu[i] = prob;
                    let dmu = (lam * surv).max(BINOMIAL_VAR_FLOOR);
                    let var = (prob * surv).max(BINOMIAL_VAR_FLOOR);
                    irls_w[i] = w_vec[i] * dmu * dmu / var;
                    z_work[i] = e + (y[i] - prob) / dmu;
                }
            }
            resid[i] = z_work[i] - e;
        }

        let sum_irls_w: f64 = irls_w.iter().sum();
        if !(sum_irls_w > 0.0) || !sum_irls_w.is_finite() {
            return Err(PricingError::NonConvergence(
                "IRLS working weights collapsed",
            ));
        }

        // Cyclic coordinate descent on the weighted quadratic approximation.
        for _inner in 0..options.max_inner_iter {
            let mut max_delta = 0.0_f64;

            // 1. Unpenalized intercept update.
            let num0: f64 = irls_w.iter().zip(resid.iter()).map(|(w, r)| w * r).sum();
            let d0 = num0 / sum_irls_w;
            if d0.abs() > 0.0 {
                beta0 += d0;
                for r in &mut resid {
                    *r -= d0;
                }
                max_delta = max_delta.max(d0.abs());
            }

            // 2. Penalized coordinates j = 0..p.
            for j in 0..p {
                if col_sds[j] < ZERO_SD_TOL {
                    continue;
                }
                let col = &x_std[j * n..(j + 1) * n];
                let bj = beta[j];
                let mut u_j = 0.0;
                let mut v_j = 0.0;
                for i in 0..n {
                    let xij = col[i];
                    let wi = irls_w[i];
                    u_j += wi * xij * (resid[i] + xij * bj);
                    v_j += wi * xij * xij;
                }
                u_j /= total_w;
                v_j /= total_w;
                let b_new = soft_threshold(u_j, l1_pen) / (v_j + l2_pen).max(1e-15);
                let db = b_new - bj;
                if db != 0.0 {
                    beta[j] = b_new;
                    for i in 0..n {
                        resid[i] -= col[i] * db;
                    }
                    max_delta = max_delta.max(db.abs());
                }
            }

            if max_delta < CD_INNER_TOL {
                break;
            }
        }

        // Update Negative Binomial dispersion if estimated.
        if let RegularizedGlmFamily::NegativeBinomial { fixed_theta: None } = options.family {
            for i in 0..n {
                let mut lin = beta0 + o_vec[i];
                for j in 0..p {
                    lin += x_std[j * n + i] * beta[j];
                }
                mu[i] = checked_eta(lin)?.exp();
            }
            nb_theta = update_nb_theta(y, &mu, &w_vec, nb_theta);
        }

        // Evaluate penalized objective.
        for i in 0..n {
            let mut lin = beta0 + o_vec[i];
            for j in 0..p {
                lin += x_std[j * n + i] * beta[j];
            }
            eta[i] = checked_eta(lin)?;
        }
        let nll = eval_neg_ll(options.family, y, &eta, &w_vec, nb_theta) / total_w;
        let l1: f64 = beta.iter().map(|b| b.abs()).sum();
        let l2: f64 = beta.iter().map(|b| b * b).sum();
        let obj = nll + l1_pen * l1 + 0.5 * l2_pen * l2;

        let max_param_diff = (beta0 - prev_beta0).abs().max(
            beta.iter()
                .zip(prev_beta.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max),
        );
        if !obj.is_finite() {
            return Err(PricingError::NonConvergence(
                "penalized objective became non-finite",
            ));
        }
        if (obj - prev_obj).abs() <= options.tolerance * (1.0 + obj.abs())
            && max_param_diff <= options.tolerance
        {
            prev_obj = obj;
            converged = true;
            break;
        }
        prev_obj = obj;
    }

    // Back-transform coefficients to original feature scale.
    let mut orig_beta = vec![0.0; p];
    let mut orig_intercept = beta0;
    for j in 0..p {
        if col_sds[j] >= ZERO_SD_TOL {
            orig_beta[j] = beta[j] / col_sds[j];
            orig_intercept -= col_means[j] * orig_beta[j];
        }
    }

    let mut fitted_means = Vec::with_capacity(n);
    for i in 0..n {
        let mut lin = orig_intercept + o_vec[i];
        for j in 0..p {
            lin += x[i * p + j] * orig_beta[j];
        }
        fitted_means.push(family_mean(options.family, checked_eta(lin)?));
    }

    Ok(RegularizedGlmResult {
        intercept: orig_intercept,
        coefficients: orig_beta,
        fitted_means,
        nb_theta: matches!(
            options.family,
            RegularizedGlmFamily::NegativeBinomial { .. }
        )
        .then_some(nb_theta),
        objective: prev_obj,
        iterations,
        converged,
    })
}

// ===========================================================================
// Hurdle Poisson & Zero-Inflated Poisson (ZIP) Frequency Models
// ===========================================================================

/// Output of [`fit_hurdle_poisson`] and [`fit_zero_inflated_poisson`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZeroModifiedPoissonResult {
    /// Binary part: for Hurdle, `P(Y > 0)` from a complementary log-log GLM with the
    /// exposure offset ([`RegularizedGlmFamily::BinomialCloglog`]; predict with that family
    /// and `ln E` offsets); for ZIP, the logistic structural-zero probability `π`
    /// ([`RegularizedGlmFamily::BinomialLogit`], no offset — `π` is exposure-independent).
    pub zero_part: RegularizedGlmResult,
    /// Count part: zero-truncated Poisson fit on `y > 0` for Hurdle
    /// ([`RegularizedGlmFamily::ZeroTruncatedPoisson`]; `intercept`/`coefficients` are on
    /// `ln λ`, `fitted_means` are `E[Y | Y > 0]` on the positive rows); Poisson component for ZIP.
    pub count_part: RegularizedGlmResult,
    /// Predicted unconditional means `E[Y_i]` on training observations.
    pub fitted_means: Vec<f64>,
    /// EM iterations executed (`1` for Hurdle).
    pub iterations: usize,
    /// `true` iff every inner GLM fit converged and (ZIP) the EM loop met its tolerance.
    pub converged: bool,
}

/// Validate `x`/`y` dimensions plus optional `weights` (finite, `>= 0`) and `offsets` (finite).
fn validate_zero_modified_inputs(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    offsets: Option<&[f64]>,
) -> Result<(), PricingError> {
    if n == 0 || p == 0 {
        return Err(PricingError::EmptyInput("n and p must be >= 1"));
    }
    if n.checked_mul(p) != Some(x.len()) || y.len() != n {
        return Err(PricingError::DimensionMismatch(
            "x must have length n * p and y must have length n",
        ));
    }
    if let Some(ws) = weights {
        if ws.len() != n {
            return Err(PricingError::DimensionMismatch(
                "weights length must equal n",
            ));
        }
        for &wi in ws {
            require_non_negative(wi, "weights must be finite and >= 0")?;
        }
    }
    if let Some(os) = offsets {
        if os.len() != n {
            return Err(PricingError::DimensionMismatch(
                "offsets length must equal n",
            ));
        }
        for &oi in os {
            require_finite(oi, "offsets must be finite")?;
        }
    }
    for &yi in y {
        require_non_negative(yi, "count response must be finite and >= 0")?;
    }
    Ok(())
}

/// Fit a Hurdle Poisson model: (1) a complementary log-log binary GLM for `P(Y > 0)`
/// and (2) a true zero-truncated Poisson GLM ([`RegularizedGlmFamily::ZeroTruncatedPoisson`])
/// on the positive counts `y_i > 0`, so that `E[Y | Y > 0] = λ / (1 - e^{-λ})` is
/// calibrated to the observed positive counts (intercept score equation).
///
/// **Exposure handling.** Both parts use `offsets` (`ln E_i`). The zero part is
/// `P(Y_i > 0) = 1 - exp(-E_i e^{x_i^T γ})` ([`RegularizedGlmFamily::BinomialCloglog`]
/// with offset `ln E_i`): the hurdle probability of a Poisson process with rate
/// `e^{x_i^T γ}` observed over exposure `E_i`, so it scales correctly with exposure
/// (a logit zero part without offset would make `P(Y > 0)` exposure-independent).
/// The count part uses `ln λ_i = x_i^T β + ln E_i`.
///
/// `E[Y_i] = P(Y_i > 0) · λ_i / (1 - e^{-λ_i})`.
#[allow(clippy::too_many_arguments)]
pub fn fit_hurdle_poisson(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    offsets: Option<&[f64]>,
    lambda: f64,
    alpha: f64,
) -> Result<ZeroModifiedPoissonResult, PricingError> {
    validate_zero_modified_inputs(x, n, p, y, weights, offsets)?;
    let y_bin: Vec<f64> = y.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
    let zero_part = fit_regularized_glm(
        x,
        n,
        p,
        &y_bin,
        weights,
        offsets,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::BinomialCloglog,
            lambda,
            alpha,
            ..RegularizedGlmOptions::default()
        },
    )?;

    let mut pos_x = Vec::new();
    let mut pos_y = Vec::new();
    let mut pos_w = Vec::new();
    let mut pos_o = Vec::new();
    for i in 0..n {
        if y[i] > 0.0 {
            pos_x.extend_from_slice(&x[i * p..(i + 1) * p]);
            pos_y.push(y[i]);
            pos_w.push(weights.map_or(1.0, |w| w[i]));
            pos_o.push(offsets.map_or(0.0, |o| o[i]));
        }
    }
    if pos_y.is_empty() {
        return Err(PricingError::InvalidParameter(
            "Hurdle Poisson requires at least one positive count observation",
        ));
    }
    let count_family = RegularizedGlmFamily::ZeroTruncatedPoisson;
    let count_part = fit_regularized_glm(
        &pos_x,
        pos_y.len(),
        p,
        &pos_y,
        Some(&pos_w),
        Some(&pos_o),
        RegularizedGlmOptions {
            family: count_family,
            lambda,
            alpha,
            ..RegularizedGlmOptions::default()
        },
    )?;

    let ztp_means = count_part.predict(count_family, x, n, p, offsets)?;
    let fitted_means: Vec<f64> = zero_part
        .fitted_means
        .iter()
        .zip(ztp_means.iter())
        .map(|(&pi_pos, &m)| pi_pos * m)
        .collect();
    let converged = zero_part.converged && count_part.converged;

    Ok(ZeroModifiedPoissonResult {
        zero_part,
        count_part,
        fitted_means,
        iterations: 1,
        converged,
    })
}

/// Fit a Zero-Inflated Poisson (ZIP) model via Expectation-Maximization (EM).
///
/// The EM loop stops once the largest change in posterior structural-zero probabilities
/// falls below `1e-6`; `converged` reports whether that happened within `max_em_iter`
/// and all final inner fits converged.
#[allow(clippy::too_many_arguments)]
pub fn fit_zero_inflated_poisson(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    offsets: Option<&[f64]>,
    lambda: f64,
    alpha: f64,
    max_em_iter: usize,
) -> Result<ZeroModifiedPoissonResult, PricingError> {
    validate_zero_modified_inputs(x, n, p, y, weights, offsets)?;
    if max_em_iter == 0 {
        return Err(PricingError::InvalidParameter("max_em_iter must be >= 1"));
    }
    let base_w: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let zero_opts = RegularizedGlmOptions {
        family: RegularizedGlmFamily::BinomialLogit,
        lambda,
        alpha,
        ..RegularizedGlmOptions::default()
    };
    let count_opts = RegularizedGlmOptions {
        family: RegularizedGlmFamily::Poisson,
        lambda,
        alpha,
        ..RegularizedGlmOptions::default()
    };

    // Initialize posterior structural-zero probabilities gamma_i.
    let mut gamma: Vec<f64> = y
        .iter()
        .map(|&v| if v == 0.0 { 0.5 } else { 0.0 })
        .collect();
    let mut zero_part = fit_regularized_glm(x, n, p, &gamma, Some(&base_w), None, zero_opts)?;
    let mut count_part = fit_regularized_glm(x, n, p, y, Some(&base_w), offsets, count_opts)?;

    let mut iterations = 0usize;
    let mut em_converged = false;
    for iter in 0..max_em_iter {
        iterations = iter + 1;
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            let new_g = if y[i] == 0.0 {
                let pi_i = zero_part.fitted_means[i].clamp(ZIP_PROB_EPS, 1.0 - ZIP_PROB_EPS);
                let p0 = (-count_part.fitted_means[i]).exp();
                pi_i / (pi_i + (1.0 - pi_i) * p0).max(1e-300)
            } else {
                0.0
            };
            max_diff = max_diff.max((new_g - gamma[i]).abs());
            gamma[i] = new_g;
        }

        zero_part = fit_regularized_glm(x, n, p, &gamma, Some(&base_w), None, zero_opts)?;
        let count_w: Vec<f64> = base_w
            .iter()
            .zip(gamma.iter())
            .map(|(&w, &g)| w * (1.0 - g).max(ZIP_PROB_EPS))
            .collect();
        count_part = fit_regularized_glm(x, n, p, y, Some(&count_w), offsets, count_opts)?;

        if max_diff < ZIP_EM_TOL {
            em_converged = true;
            break;
        }
    }

    let fitted_means: Vec<f64> = (0..n)
        .map(|i| (1.0 - zero_part.fitted_means[i]) * count_part.fitted_means[i])
        .collect();
    let converged = em_converged && zero_part.converged && count_part.converged;

    Ok(ZeroModifiedPoissonResult {
        zero_part,
        count_part,
        fitted_means,
        iterations,
        converged,
    })
}

// ===========================================================================
// 3.2 B-Splines (Cox–de Boor) & Natural Cubic Splines
// ===========================================================================

/// Evaluate the B-spline basis of degree `degree` at points `xs` over
/// non-decreasing knot vector `knots` (`t_0 <= t_1 <= ... <= t_M`) via the
/// Cox–de Boor recursion (`0/0 ≡ 0`).
///
/// The basis is a partition of unity only on the valid domain
/// `[t_degree, t_{M-degree}]`; points outside it return [`PricingError::OutOfBounds`].
/// Each `x` is located in the knot span `[t_k, t_{k+1})` (half-open, `t_k < t_{k+1}`,
/// `degree <= k < M - degree`); at the right endpoint `x = t_{M-degree}` the last non-empty
/// span is used, i.e. the left limit (de Boor 1978). For clamped knots (end multiplicity
/// `degree + 1`) this gives `B_{last}(t_M) = 1`; for unclamped knots it is the continuous
/// left-limit value.
///
/// Returns `(n_basis, row_major_matrix)` where `n_basis = knots.len() - degree - 1`.
pub fn bspline_basis(
    xs: &[f64],
    knots: &[f64],
    degree: usize,
) -> Result<(usize, Vec<f64>), PricingError> {
    if xs.is_empty() {
        return Err(PricingError::EmptyInput("xs must not be empty"));
    }
    if knots.len() < 2 * degree + 2 {
        return Err(PricingError::InvalidParameter(
            "knots.len() must be at least 2 * degree + 2",
        ));
    }
    for w in knots.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || w[1] < w[0] {
            return Err(PricingError::InvalidParameter(
                "knots must be finite and non-decreasing",
            ));
        }
    }
    let m = knots.len() - 1;
    let n_basis = m - degree;
    let lo = knots[degree];
    let hi = knots[n_basis];
    if !(hi > lo) {
        return Err(PricingError::InvalidParameter(
            "B-spline valid domain [t_degree, t_{M-degree}] must have positive length",
        ));
    }
    let n = xs.len();
    let mut out = vec![0.0; n * n_basis];

    let mut b_prev = vec![0.0; m];
    let mut b_curr = vec![0.0; m];

    for (row, &x) in xs.iter().enumerate() {
        require_finite(x, "evaluation point x must be finite")?;
        if x < lo || x > hi {
            return Err(PricingError::OutOfBounds(
                "B-spline evaluation point outside valid domain [t_degree, t_{M-degree}]",
            ));
        }
        // Largest non-empty span k in [degree, n_basis) with t_k <= x (x < t_{k+1}
        // automatically, except at x = hi where this selects the last non-empty span).
        let mut span = degree;
        for k in degree..n_basis {
            if knots[k] < knots[k + 1] && knots[k] <= x {
                span = k;
            }
        }
        b_prev.fill(0.0);
        b_prev[span] = 1.0;
        for k in 1..=degree {
            let len = m - k;
            for j in 0..len {
                let d1 = knots[j + k] - knots[j];
                let d2 = knots[j + k + 1] - knots[j + 1];
                let t1 = if d1 > 0.0 {
                    (x - knots[j]) / d1 * b_prev[j]
                } else {
                    0.0
                };
                let t2 = if d2 > 0.0 {
                    (knots[j + k + 1] - x) / d2 * b_prev[j + 1]
                } else {
                    0.0
                };
                b_curr[j] = t1 + t2;
            }
            std::mem::swap(&mut b_prev, &mut b_curr);
        }
        out[row * n_basis..(row + 1) * n_basis].copy_from_slice(&b_prev[..n_basis]);
    }

    Ok((n_basis, out))
}

/// Evaluate the Natural Cubic Spline basis of dimension `K = knots.len()`
/// over strictly increasing knots `ξ_1 < ξ_2 < ... < ξ_K` (`K >= 3`).
///
/// Guarantees linear extrapolation outside `[ξ_1, ξ_K]` (`N_k'' = 0` at and
/// beyond both boundary knots). Returns `(K, row_major_matrix)`.
pub fn natural_cubic_spline_basis(
    xs: &[f64],
    knots: &[f64],
) -> Result<(usize, Vec<f64>), PricingError> {
    if xs.is_empty() {
        return Err(PricingError::EmptyInput("xs must not be empty"));
    }
    let k_len = knots.len();
    if k_len < 3 {
        return Err(PricingError::InvalidParameter(
            "natural cubic spline requires at least 3 strictly increasing knots",
        ));
    }
    for w in knots.windows(2) {
        if !w[0].is_finite() || !w[1].is_finite() || !(w[1] > w[0]) {
            return Err(PricingError::InvalidParameter(
                "natural cubic spline knots must be finite and strictly increasing",
            ));
        }
    }

    let xi_k = knots[k_len - 1];
    let xi_km1 = knots[k_len - 2];
    let denom_km1 = xi_k - xi_km1;
    let n = xs.len();
    let mut out = vec![0.0; n * k_len];

    for (row, &x) in xs.iter().enumerate() {
        require_finite(x, "evaluation point x must be finite")?;
        let base = row * k_len;
        out[base] = 1.0;
        out[base + 1] = x;
        let pos_k = (x - xi_k).max(0.0).powi(3);
        let d_km1 = ((x - xi_km1).max(0.0).powi(3) - pos_k) / denom_km1;
        for j in 0..(k_len - 2) {
            let d_j = ((x - knots[j]).max(0.0).powi(3) - pos_k) / (xi_k - knots[j]);
            out[base + 2 + j] = d_j - d_km1;
        }
    }

    Ok((k_len, out))
}

// ===========================================================================
// 3.3 Bühlmann-in-GLM / Laplace-Approximated Random-Intercept GLMM
// ===========================================================================

/// Output of [`fit_random_intercept_glmm`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomInterceptGlmmResult {
    /// Fixed-effect intercept `β_0` (unpenalized; estimated jointly with the random intercepts).
    pub intercept: f64,
    /// Fixed-effect slopes `β_1..β_p`.
    pub fixed_effects: Vec<f64>,
    /// Fitted random-effect variance `τ^2 ∈ [1e-8, 1e4]`.
    pub tau_squared: f64,
    /// Conditional-mode (BLUP) random intercepts `û_j` for each group `j = 0..n_groups`.
    /// At the joint mode the intercept score equation implies `Σ_j û_j = 0`.
    pub random_intercepts: Vec<f64>,
    /// Implied Bühlmann–Straub credibility weight `Z_j = Σ μ_{ij} / (Σ μ_{ij} + 1/τ^2)` per group.
    pub group_credibilities: Vec<f64>,
    /// Laplace-approximated marginal log-likelihood (including `-ln y!`) at the optimum.
    pub laplace_log_likelihood: f64,
    /// Number of profiled Laplace log-likelihood evaluations (candidate `τ²` values).
    pub iterations: usize,
    /// `τ²` optimum is at the lower bound `1e-8` (no detectable between-group
    /// heterogeneity; a legitimate boundary MLE, does not affect `converged`).
    #[serde(default)]
    pub tau_squared_at_lower_bound: bool,
    /// `τ²` optimum hit the hard upper bound `1e4`; the reported `τ²` is not an interior
    /// optimum and `converged` is `false`.
    #[serde(default)]
    pub tau_squared_at_upper_bound: bool,
    /// `true` iff every joint-mode Newton solve converged, the `ln τ²` search met its
    /// tolerance, and `τ²` is not at the upper bound.
    pub converged: bool,
}

/// Validated GLMM data (`x` row-major `n × p`, no intercept column).
struct GlmmData<'a> {
    x: &'a [f64],
    n: usize,
    p: usize,
    y: &'a [f64],
    w: Vec<f64>,
    o: Vec<f64>,
    group_ids: &'a [usize],
    n_groups: usize,
    /// `Σ_i w_i ln Γ(y_i + 1)`.
    log_factorial_sum: f64,
}

/// Joint conditional mode `(β̂(τ²), û(τ²))` and the Laplace marginal log-likelihood there.
#[derive(Clone)]
struct JointMode {
    /// `[β_0, β_1..β_p]`.
    beta: Vec<f64>,
    u: Vec<f64>,
    laplace_ll: f64,
    converged: bool,
}

impl GlmmData<'_> {
    /// Linear predictors `η_i = o_i + β_0 + x_i^T β + u_{g(i)}` (overflow-checked).
    fn eta(&self, beta: &[f64], u: &[f64], eta: &mut [f64]) -> Result<(), PricingError> {
        for i in 0..self.n {
            let mut lin = self.o[i] + beta[0] + u[self.group_ids[i]];
            for j in 0..self.p {
                lin += self.x[i * self.p + j] * beta[1 + j];
            }
            eta[i] = checked_eta(lin)?;
        }
        Ok(())
    }

    /// Penalized log-likelihood `h = Σ w (y η - μ) - Σ u² / (2τ²)` (without `ln y!`),
    /// or `None` if the linear predictor overflows.
    fn penalized_ll(&self, beta: &[f64], u: &[f64], inv_t2: f64, eta: &mut [f64]) -> Option<f64> {
        self.eta(beta, u, eta).ok()?;
        let mut h = 0.0;
        for i in 0..self.n {
            h += self.w[i] * (self.y[i] * eta[i] - eta[i].exp());
        }
        h -= 0.5 * inv_t2 * u.iter().map(|v| v * v).sum::<f64>();
        h.is_finite().then_some(h)
    }

    /// Penalized Newton–Raphson (= PIRLS for the canonical log link) on `θ = [β; u]` with
    /// ridge `1/τ²` on `u` only, solved by Schur complement on the diagonal `u` block.
    fn joint_mode(
        &self,
        tau2: f64,
        start: &JointMode,
        max_iter: usize,
        param_rtol: f64,
    ) -> Result<JointMode, PricingError> {
        let q = self.p + 1;
        let g_n = self.n_groups;
        let inv_t2 = 1.0 / tau2;
        let mut beta = start.beta.clone();
        let mut u = start.u.clone();
        let mut eta = vec![0.0; self.n];
        let mut h =
            self.penalized_ll(&beta, &u, inv_t2, &mut eta)
                .ok_or(PricingError::NonConvergence(
                    "GLMM linear predictor overflowed at the start point",
                ))?;
        let mut converged = false;

        let mut a_mat = vec![0.0; q * q];
        let mut b_mat = vec![0.0; g_n * q]; // row g: Σ_{i∈g} w μ x̃_i
        let mut d_diag = vec![0.0; g_n];
        let mut g_beta = vec![0.0; q];
        let mut g_u = vec![0.0; g_n];
        let mut xt = vec![0.0; q];
        let mut trial_beta = vec![0.0; q];
        let mut trial_u = vec![0.0; g_n];

        for _ in 0..max_iter {
            // Gradient and negative Hessian at the current iterate (`eta` is current).
            a_mat.fill(0.0);
            b_mat.fill(0.0);
            g_beta.fill(0.0);
            for (g, ug) in u.iter().enumerate() {
                d_diag[g] = inv_t2;
                g_u[g] = -ug * inv_t2;
            }
            for i in 0..self.n {
                let mu = eta[i].exp();
                let wm = self.w[i] * mu;
                let r = self.w[i] * (self.y[i] - mu);
                let g = self.group_ids[i];
                xt[0] = 1.0;
                xt[1..].copy_from_slice(&self.x[i * self.p..(i + 1) * self.p]);
                for a in 0..q {
                    g_beta[a] += r * xt[a];
                    b_mat[g * q + a] += wm * xt[a];
                    for b in 0..=a {
                        a_mat[a * q + b] += wm * xt[a] * xt[b];
                    }
                }
                g_u[g] += r;
                d_diag[g] += wm;
            }
            // Schur complement S = A - B^T D^{-1} B, rhs = g_β - B^T D^{-1} g_u.
            let mut rhs = g_beta.clone();
            for g in 0..g_n {
                let bg = &b_mat[g * q..(g + 1) * q];
                let inv_d = 1.0 / d_diag[g];
                for a in 0..q {
                    rhs[a] -= bg[a] * inv_d * g_u[g];
                    for b in 0..=a {
                        a_mat[a * q + b] -= bg[a] * bg[b] * inv_d;
                    }
                }
            }
            for a in 0..q {
                for b in 0..a {
                    a_mat[b * q + a] = a_mat[a * q + b];
                }
            }
            let d_beta = cholesky_solve(&a_mat, q, &rhs)?;
            let mut decrement = 0.0;
            let mut max_rel_step = 0.0_f64;
            for a in 0..q {
                decrement += d_beta[a] * g_beta[a];
                max_rel_step = max_rel_step.max(d_beta[a].abs() / (1.0 + beta[a].abs()));
            }
            let mut d_u = vec![0.0; g_n];
            for g in 0..g_n {
                let bg = &b_mat[g * q..(g + 1) * q];
                let dot: f64 = bg.iter().zip(&d_beta).map(|(b, d)| b * d).sum();
                d_u[g] = (g_u[g] - dot) / d_diag[g];
                decrement += d_u[g] * g_u[g];
                max_rel_step = max_rel_step.max(d_u[g].abs() / (1.0 + u[g].abs()));
            }
            if !(decrement.is_finite() && max_rel_step.is_finite()) {
                return Err(PricingError::NonConvergence(
                    "GLMM joint-mode Newton step is non-finite",
                ));
            }

            // Backtracking on the concave penalized log-likelihood.
            let mut t = 1.0;
            let mut accepted = false;
            for _ in 0..=GLMM_MAX_STEP_HALVINGS {
                for a in 0..q {
                    trial_beta[a] = beta[a] + t * d_beta[a];
                }
                for g in 0..g_n {
                    trial_u[g] = u[g] + t * d_u[g];
                }
                if let Some(h_new) = self.penalized_ll(&trial_beta, &trial_u, inv_t2, &mut eta)
                    && h_new >= h - GLMM_ASCENT_SLACK * (1.0 + h.abs())
                {
                    h = h_new;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                // Restore `eta` at the current iterate and stop (reported as non-converged).
                self.eta(&beta, &u, &mut eta)?;
                break;
            }
            beta.copy_from_slice(&trial_beta);
            u.copy_from_slice(&trial_u);
            if 0.5 * decrement <= GLMM_NEWTON_DECREMENT_TOL && max_rel_step <= param_rtol {
                converged = true;
                break;
            }
        }

        // Laplace marginal log-likelihood at the joint mode:
        // Σ w (y η - μ - ln y!) - Σ u²/(2τ²) - ½ Σ_g ln(1 + τ² Σ_{i∈g} w μ).
        let mut sum_wmu = vec![0.0; g_n];
        for i in 0..self.n {
            sum_wmu[self.group_ids[i]] += self.w[i] * eta[i].exp();
        }
        let log_det: f64 = sum_wmu.iter().map(|s| (tau2 * s).ln_1p()).sum();
        let laplace_ll = h - self.log_factorial_sum - 0.5 * log_det;
        if !laplace_ll.is_finite() {
            return Err(PricingError::NonConvergence(
                "GLMM Laplace log-likelihood is non-finite",
            ));
        }
        Ok(JointMode {
            beta,
            u,
            laplace_ll,
            converged,
        })
    }
}

/// Fit a Laplace-approximated Poisson random-intercept GLMM
/// `η_{ij} = β_0 + x_{ij}^T β + o_{ij} + u_j`, `u_j ~ N(0, τ^2)`
/// for high-cardinality territories or class codes (`group_ids[i] ∈ 0..n_groups`).
///
/// **Method** (Bates, Mächler, Bolker & Walker 2015, *JSS* 67(1), `glmer(nAGQ = 0)`;
/// Klinker 2011). For each `τ²` the joint conditional mode `(β̂, û)` maximizes the
/// penalized log-likelihood `Σ w (y η - μ) - Σ_j u_j² / (2τ²)` by penalized Newton/IRLS on
/// `[β; u]` (ridge `1/τ²` on `u` only, intercept unpenalized), solved by a Schur complement
/// on the diagonal `u` block (`O(n p² + G p²)` per step). Solving jointly removes the
/// intercept/random-effect confounding of alternating updates; at the mode `Σ_j û_j = 0`.
/// Each Newton solve stops when the Newton decrement `½ Δθᵀ∇h ≤ 1e-10` and
/// `max |Δθ_k| / (1 + |θ_k|) ≤ 1e-8` (scale-free, not relative to `|ℓ|`).
///
/// `τ²` maximizes the profiled Laplace marginal log-likelihood
/// `ℓ_L(τ²) = Σ w (y η̂ - μ̂ - ln y!) - Σ û²/(2τ²) - ½ Σ_j ln(1 + τ² Σ_{i∈j} w μ̂)`:
/// a unit-step grid on `ln τ²` from `ln 1e-8` to `2`, expanded upward while the maximum is at
/// the top (hard bound `τ² ≤ 1e4`), then golden-section refinement to `|Δ ln τ²| ≤ 1e-6`.
/// An optimum at the upper bound sets `tau_squared_at_upper_bound` and `converged = false`
/// instead of being silently clamped.
///
/// * `options.max_iter` — maximum penalized-Newton iterations per joint-mode solve (`>= 1`).
/// * `options.tol` — relative parameter-step tolerance of each joint-mode solve (`> 0`);
///   the solve also requires the Newton decrement to fall below `1e-10`.
///
/// The columns of `x` together with the intercept must have full column rank
/// (otherwise [`PricingError::SingularMatrix`]).
#[allow(clippy::too_many_arguments)]
pub fn fit_random_intercept_glmm(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    group_ids: &[usize],
    n_groups: usize,
    weights: Option<&[f64]>,
    offsets: Option<&[f64]>,
    options: &crate::error::SolverOptions,
) -> Result<RandomInterceptGlmmResult, PricingError> {
    if n == 0 || p == 0 || n_groups == 0 {
        return Err(PricingError::EmptyInput(
            "n, p, and n_groups must all be >= 1",
        ));
    }
    if options.max_iter == 0 {
        return Err(PricingError::InvalidParameter("max_iter must be >= 1"));
    }
    if !options.tol.is_finite() || !(options.tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "tolerance must be finite and positive",
        ));
    }
    if n.checked_mul(p) != Some(x.len()) || y.len() != n || group_ids.len() != n {
        return Err(PricingError::DimensionMismatch(
            "x, y, and group_ids dimensions must match",
        ));
    }
    validate_zero_modified_inputs(x, n, p, y, weights, offsets)?;
    for &xi in x {
        require_finite(xi, "design matrix entries must be finite")?;
    }
    for &g in group_ids {
        if g >= n_groups {
            return Err(PricingError::OutOfBounds("group_id >= n_groups"));
        }
    }
    let w_vec: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let o_vec: Vec<f64> = match offsets {
        Some(o) => o.to_vec(),
        None => vec![0.0; n],
    };
    let log_factorial_sum = y
        .iter()
        .zip(&w_vec)
        .map(|(&yi, &wi)| wi * ln_gamma(yi + 1.0))
        .sum();

    // Start: ordinary Poisson GLM (u = 0).
    let fixed_opts = RegularizedGlmOptions {
        family: RegularizedGlmFamily::Poisson,
        ..RegularizedGlmOptions::default()
    };
    let glm = fit_regularized_glm(x, n, p, y, Some(&w_vec), Some(&o_vec), fixed_opts)?;
    let data = GlmmData {
        x,
        n,
        p,
        y,
        w: w_vec,
        o: o_vec,
        group_ids,
        n_groups,
        log_factorial_sum,
    };
    let mut beta_start = Vec::with_capacity(p + 1);
    beta_start.push(glm.intercept);
    beta_start.extend_from_slice(&glm.coefficients);
    let start = JointMode {
        beta: beta_start,
        u: vec![0.0; n_groups],
        laplace_ll: f64::NEG_INFINITY,
        converged: true,
    };

    let mut evaluations = 0usize;
    let mut all_converged = true;
    let mut eval = |s: f64, from: &JointMode| -> Result<JointMode, PricingError> {
        evaluations += 1;
        let m = data.joint_mode(s.exp(), from, options.max_iter, options.tol)?;
        all_converged &= m.converged;
        Ok(m)
    };

    // 1. Unit-step grid on ln τ², warm-started upward; expand while the maximum is at the top.
    let ln_min = GLMM_TAU2_FLOOR.ln();
    let ln_max = GLMM_TAU2_MAX.ln();
    let mut grid: Vec<(f64, JointMode)> = Vec::new();
    let mut s = ln_min;
    let mut prev = start;
    loop {
        let m = eval(s, &prev)?;
        grid.push((s, m.clone()));
        prev = m;
        if s >= ln_max {
            break;
        }
        let best_is_last = grid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.1.laplace_ll.total_cmp(&b.1.1.laplace_ll))
            .is_some_and(|(k, _)| k + 1 == grid.len());
        if s >= GLMM_LN_TAU2_GRID_HI && !best_is_last {
            break;
        }
        s = (s + GLMM_LN_TAU2_GRID_STEP).min(ln_max);
    }
    let (k_best, _) = grid
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.laplace_ll.total_cmp(&b.1.1.laplace_ll))
        .ok_or(PricingError::NonConvergence("GLMM tau^2 grid is empty"))?;
    let at_lower = k_best == 0;
    let at_upper = k_best + 1 == grid.len() && grid[k_best].0 >= ln_max;

    // 2. Golden-section refinement on the bracketing grid cells.
    let mut search_converged = true;
    let (mut best_s, mut best) = (grid[k_best].0, grid[k_best].1.clone());
    if !at_lower && !at_upper {
        let warm = grid[k_best].1.clone();
        let mut a = grid[k_best - 1].0;
        let mut b = grid[k_best + 1].0;
        let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
        let mut c = b - gr * (b - a);
        let mut d = a + gr * (b - a);
        let mut mc = eval(c, &warm)?;
        let mut md = eval(d, &warm)?;
        search_converged = false;
        for _ in 0..GLMM_TAU_GOLDEN_MAX_ITERS {
            if b - a <= GLMM_LN_TAU2_TOL {
                search_converged = true;
                break;
            }
            if mc.laplace_ll < md.laplace_ll {
                a = c;
                c = d;
                mc = md;
                d = a + gr * (b - a);
                md = eval(d, &warm)?;
            } else {
                b = d;
                d = c;
                md = mc;
                c = b - gr * (b - a);
                mc = eval(c, &warm)?;
            }
        }
        let (s_c, m_c) = if mc.laplace_ll >= md.laplace_ll {
            (c, mc)
        } else {
            (d, md)
        };
        if m_c.laplace_ll >= best.laplace_ll {
            best_s = s_c;
            best = m_c;
        }
    }
    let tau_sq = best_s.exp();

    let inv_t2 = 1.0 / tau_sq;
    let mut eta = vec![0.0; n];
    data.eta(&best.beta, &best.u, &mut eta)?;
    let mut sum_mu = vec![0.0; n_groups];
    for i in 0..n {
        sum_mu[group_ids[i]] += data.w[i] * eta[i].exp();
    }
    let group_credibilities: Vec<f64> = sum_mu.iter().map(|s| s / (s + inv_t2)).collect();
    let converged = all_converged && search_converged && !at_upper;

    Ok(RandomInterceptGlmmResult {
        intercept: best.beta[0],
        fixed_effects: best.beta[1..].to_vec(),
        tau_squared: tau_sq,
        random_intercepts: best.u,
        group_credibilities,
        laplace_log_likelihood: best.laplace_ll,
        iterations: evaluations,
        tau_squared_at_lower_bound: at_lower,
        tau_squared_at_upper_bound: at_upper,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_elastic_net_kkt_optimality_and_nb_to_poisson_limit() {
        // Synthetic Poisson count dataset with 2 features (one active, one weak).
        let n = 20usize;
        let p = 2usize;
        let mut x = Vec::with_capacity(n * p);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let x1 = (i as f64 - 9.5) / 5.0;
            let x2 = ((i * 7) % 5) as f64 - 2.0;
            x.push(x1);
            x.push(x2);
            let mu = (0.5 + 0.4 * x1).exp();
            y.push(mu.round());
        }

        // Elastic Net KKT check on standardized features.
        let lambda = 0.15;
        let alpha = 0.5;
        let fit = fit_regularized_glm(
            &x,
            n,
            p,
            &y,
            None,
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                lambda,
                alpha,
                ..RegularizedGlmOptions::default()
            },
        )
        .expect("elastic net");
        assert!(fit.converged);

        // Unpenalized intercept KKT: (1/n) sum(y_i - mu_i) == 0.
        let res_sum: f64 = y
            .iter()
            .zip(fit.fitted_means.iter())
            .map(|(yi, mui)| yi - mui)
            .sum::<f64>()
            / n as f64;
        assert!(res_sum.abs() < 1e-6, "intercept KKT={res_sum}");

        // Standardized-feature KKT check.
        for j in 0..p {
            let m = (0..n).map(|i| x[i * p + j]).sum::<f64>() / n as f64;
            let sd = ((0..n).map(|i| (x[i * p + j] - m).powi(2)).sum::<f64>() / n as f64).sqrt();
            let b_std = fit.coefficients[j] * sd;
            let grad_j = (0..n)
                .map(|i| ((x[i * p + j] - m) / sd) * (y[i] - fit.fitted_means[i]))
                .sum::<f64>()
                / n as f64;
            if b_std.abs() > 1e-12 {
                let kkt = grad_j - lambda * (1.0 - alpha) * b_std - lambda * alpha * b_std.signum();
                assert!(kkt.abs() < 1e-5, "active KKT j={j}: {kkt}");
            } else {
                assert!(
                    grad_j.abs() <= lambda * alpha + 1e-5,
                    "inactive KKT j={j}: {grad_j}"
                );
            }
        }

        // Negative Binomial -> Poisson limit at theta = 1e8.
        let pois = fit_regularized_glm(
            &x,
            n,
            p,
            &y,
            None,
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                ..RegularizedGlmOptions::default()
            },
        )
        .expect("poisson");
        let nb = fit_regularized_glm(
            &x,
            n,
            p,
            &y,
            None,
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::NegativeBinomial {
                    fixed_theta: Some(1.0e8),
                },
                ..RegularizedGlmOptions::default()
            },
        )
        .expect("nb");
        assert!((pois.intercept - nb.intercept).abs() < 1e-6);
        for j in 0..p {
            assert!((pois.coefficients[j] - nb.coefficients[j]).abs() < 1e-6);
        }
    }

    #[test]
    fn golden_bspline_partition_of_unity_and_natural_cubic_zero_second_derivative() {
        // Clamped cubic B-spline knots on [0, 1].
        let knots = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let xs = [0.0, 0.1, 0.25, 0.42, 0.5, 0.75, 0.91, 1.0];
        let (nb, mat) = bspline_basis(&xs, &knots, 3).expect("bspline");
        for r in 0..xs.len() {
            let sum: f64 = mat[r * nb..(r + 1) * nb].iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "row={r} sum={sum}");
        }

        // Natural cubic spline zero second derivative outside [xi_1, xi_K].
        let ns_knots = [0.0, 1.0, 2.0, 3.0];
        let h = 1e-2;
        let eval_pts = [-2.0 * h, -h, 0.0, 3.0, 3.0 + h, 3.0 + 2.0 * h];
        let (nk, ns) = natural_cubic_spline_basis(&eval_pts, &ns_knots).expect("ns");
        for col in 0..nk {
            let d2_lo = (ns[2 * nk + col] - 2.0 * ns[nk + col] + ns[col]) / (h * h);
            let d2_hi = (ns[5 * nk + col] - 2.0 * ns[4 * nk + col] + ns[3 * nk + col]) / (h * h);
            assert!(d2_lo.abs() < 1e-10, "col={col} d2_lo={d2_lo}");
            assert!(d2_hi.abs() < 1e-10, "col={col} d2_hi={d2_hi}");
        }
    }

    #[test]
    fn hurdle_zip_and_random_intercept_glmm_fit_cleanly() {
        let x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y = [0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 2.0, 5.0, 1.0, 3.0];
        let hurdle = fit_hurdle_poisson(&x, 12, 1, &y, None, None, 0.01, 0.0).expect("hurdle");
        assert!(
            hurdle
                .fitted_means
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );

        let zip = fit_zero_inflated_poisson(&x, 12, 1, &y, None, None, 0.01, 0.0, 30).expect("zip");
        assert!(zip.fitted_means.iter().all(|v| v.is_finite() && *v > 0.0));

        let groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let glmm = fit_random_intercept_glmm(
            &x,
            12,
            1,
            &y,
            &groups,
            3,
            None,
            None,
            &crate::error::SolverOptions::default().with_max_iter(15),
        )
        .expect("glmm");
        assert!(glmm.tau_squared >= 1e-8);
        assert!(
            glmm.group_credibilities
                .iter()
                .all(|&z| (0.0..=1.0).contains(&z))
        );
    }
}
