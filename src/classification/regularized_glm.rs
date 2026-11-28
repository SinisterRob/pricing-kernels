//! Regularized GLMs (Elastic Net / Ridge / LASSO via Penalized IRLS + Cyclic
//! Coordinate Descent), B-spline & Natural Cubic Spline bases, Negative Binomial /
//! Hurdle / Zero-Inflated Poisson frequency models, and Bühlmann-in-GLM /
//! Laplace-approximated Random-Intercept GLMMs for high-cardinality territories
//! and class codes.

use crate::error::PricingError;
use crate::special::{digamma, ln_gamma};
use crate::validate::{
    require_finite, require_non_negative, require_positive, require_unit_interval,
};
use serde::{Deserialize, Serialize};

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
    /// Binary Logistic (`logit(π) = η`, used internally by Hurdle/ZIP and standalone).
    BinomialLogit,
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
            let e = eta.clamp(-20.0, 20.0);
            let mu = match family {
                RegularizedGlmFamily::BinomialLogit => 1.0 / (1.0 + (-e).exp()),
                _ => e.exp(),
            };
            preds.push(mu);
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
        let ei = eta[i].clamp(-20.0, 20.0);
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
/// `NegativeBinomial`, or `BinomialLogit`) via Penalized IRLS with cyclic
/// coordinate descent.
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
            RegularizedGlmFamily::BinomialLogit => {
                require_unit_interval(yi, "Binomial response must be in [0, 1]")?;
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
        if sd >= 1e-12 {
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
    let mut beta0 = if options.family == RegularizedGlmFamily::BinomialLogit {
        let p0 = y_bar.clamp(1e-4, 1.0 - 1e-4);
        (p0 / (1.0 - p0)).ln()
    } else {
        let off_bar: f64 = o_vec
            .iter()
            .zip(w_vec.iter())
            .map(|(oi, wi)| oi * wi)
            .sum::<f64>()
            / total_w;
        y_bar.max(1e-4).ln() - off_bar
    };
    let mut beta = vec![0.0; p];
    let mut nb_theta = match options.family {
        RegularizedGlmFamily::NegativeBinomial { fixed_theta } => fixed_theta.unwrap_or(5.0),
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
            let e = lin.clamp(-20.0, 20.0);
            eta[i] = e;
            match options.family {
                RegularizedGlmFamily::Poisson => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m;
                    z_work[i] = e + (y[i] - m) / m.max(1e-12);
                }
                RegularizedGlmFamily::Gamma => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i];
                    z_work[i] = e + (y[i] - m) / m.max(1e-12);
                }
                RegularizedGlmFamily::Tweedie { power } => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m.powf(2.0 - power);
                    z_work[i] = e + (y[i] - m) / m.max(1e-12);
                }
                RegularizedGlmFamily::NegativeBinomial { .. } => {
                    let m = e.exp();
                    mu[i] = m;
                    irls_w[i] = w_vec[i] * m / (1.0 + m / nb_theta);
                    z_work[i] = e + (y[i] - m) / m.max(1e-12);
                }
                RegularizedGlmFamily::BinomialLogit => {
                    let prob = 1.0 / (1.0 + (-e).exp());
                    mu[i] = prob;
                    let pq = (prob * (1.0 - prob)).max(1e-9);
                    irls_w[i] = w_vec[i] * pq;
                    z_work[i] = e + (y[i] - prob) / pq;
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
                if col_sds[j] < 1e-12 {
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

            if max_delta < 1e-10 {
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
                mu[i] = lin.clamp(-20.0, 20.0).exp();
            }
            nb_theta = update_nb_theta(y, &mu, &w_vec, nb_theta);
        }

        // Evaluate penalized objective.
        for i in 0..n {
            let mut lin = beta0 + o_vec[i];
            for j in 0..p {
                lin += x_std[j * n + i] * beta[j];
            }
            eta[i] = lin.clamp(-20.0, 20.0);
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
        if (obj - prev_obj).abs() <= options.tolerance * (1.0 + obj.abs()) && max_param_diff < 1e-7
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
        if col_sds[j] >= 1e-12 {
            orig_beta[j] = beta[j] / col_sds[j];
            orig_intercept -= col_means[j] * orig_beta[j];
        }
    }

    let fitted_means: Vec<f64> = (0..n)
        .map(|i| {
            let mut lin = orig_intercept + o_vec[i];
            for j in 0..p {
                lin += x[i * p + j] * orig_beta[j];
            }
            let e = lin.clamp(-20.0, 20.0);
            match options.family {
                RegularizedGlmFamily::BinomialLogit => 1.0 / (1.0 + (-e).exp()),
                _ => e.exp(),
            }
        })
        .collect();

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
    /// Binary logistic part (`P(Y > 0)` for Hurdle; structural-zero probability `π` for ZIP).
    pub zero_part: RegularizedGlmResult,
    /// Count Poisson part (truncated-count rate for Hurdle; Poisson component rate for ZIP).
    pub count_part: RegularizedGlmResult,
    /// Predicted unconditional means `E[Y_i]` on training observations.
    pub fitted_means: Vec<f64>,
    /// EM iterations executed (`1` for Hurdle).
    pub iterations: usize,
}

/// Fit a Hurdle Poisson model: (1) Binary Logistic GLM for `P(Y > 0)` and
/// (2) Zero-Truncated Poisson GLM on positive counts `y_i > 0` via IRLS.
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
    if n == 0 || p == 0 || x.len() != n * p || y.len() != n {
        return Err(PricingError::DimensionMismatch(
            "invalid dimensions for Hurdle Poisson",
        ));
    }
    let y_bin: Vec<f64> = y.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
    let zero_part = fit_regularized_glm(
        x,
        n,
        p,
        &y_bin,
        weights,
        None,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::BinomialLogit,
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
    let count_part = fit_regularized_glm(
        &pos_x,
        pos_y.len(),
        p,
        &pos_y,
        Some(&pos_w),
        Some(&pos_o),
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::Poisson,
            lambda,
            alpha,
            ..RegularizedGlmOptions::default()
        },
    )?;

    let fitted_means: Vec<f64> = (0..n)
        .map(|i| {
            let pi_pos = zero_part.fitted_means[i];
            let mut lin = count_part.intercept + offsets.map_or(0.0, |o| o[i]);
            for j in 0..p {
                lin += x[i * p + j] * count_part.coefficients[j];
            }
            let mu = lin.clamp(-20.0, 20.0).exp();
            let ztp_mean = mu / (1.0 - (-mu).exp()).max(1e-12);
            pi_pos * ztp_mean
        })
        .collect();

    Ok(ZeroModifiedPoissonResult {
        zero_part,
        count_part,
        fitted_means,
        iterations: 1,
    })
}

/// Fit a Zero-Inflated Poisson (ZIP) model via Expectation-Maximization (EM).
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
    if n == 0 || p == 0 || x.len() != n * p || y.len() != n {
        return Err(PricingError::DimensionMismatch(
            "invalid dimensions for ZIP",
        ));
    }
    if max_em_iter == 0 {
        return Err(PricingError::InvalidParameter("max_em_iter must be >= 1"));
    }
    let base_w: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };

    // Initialize posterior structural-zero probabilities gamma_i.
    let mut gamma: Vec<f64> = y
        .iter()
        .map(|&v| if v == 0.0 { 0.5 } else { 0.0 })
        .collect();
    let mut zero_part = fit_regularized_glm(
        x,
        n,
        p,
        &gamma,
        Some(&base_w),
        None,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::BinomialLogit,
            lambda,
            alpha,
            ..RegularizedGlmOptions::default()
        },
    )?;
    let mut count_part = fit_regularized_glm(
        x,
        n,
        p,
        y,
        Some(&base_w),
        offsets,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::Poisson,
            lambda,
            alpha,
            ..RegularizedGlmOptions::default()
        },
    )?;

    let mut iterations = 0usize;
    for iter in 0..max_em_iter {
        iterations = iter + 1;
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            let new_g = if y[i] == 0.0 {
                let pi_i = zero_part.fitted_means[i].clamp(1e-9, 1.0 - 1e-9);
                let p0 = (-count_part.fitted_means[i]).exp();
                pi_i / (pi_i + (1.0 - pi_i) * p0).max(1e-300)
            } else {
                0.0
            };
            max_diff = max_diff.max((new_g - gamma[i]).abs());
            gamma[i] = new_g;
        }

        zero_part = fit_regularized_glm(
            x,
            n,
            p,
            &gamma,
            Some(&base_w),
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::BinomialLogit,
                lambda,
                alpha,
                ..RegularizedGlmOptions::default()
            },
        )?;
        let count_w: Vec<f64> = base_w
            .iter()
            .zip(gamma.iter())
            .map(|(&w, &g)| w * (1.0 - g).max(1e-9))
            .collect();
        count_part = fit_regularized_glm(
            x,
            n,
            p,
            y,
            Some(&count_w),
            offsets,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                lambda,
                alpha,
                ..RegularizedGlmOptions::default()
            },
        )?;

        if max_diff < 1e-6 {
            break;
        }
    }

    let fitted_means: Vec<f64> = (0..n)
        .map(|i| (1.0 - zero_part.fitted_means[i]) * count_part.fitted_means[i])
        .collect();

    Ok(ZeroModifiedPoissonResult {
        zero_part,
        count_part,
        fitted_means,
        iterations,
    })
}

// ===========================================================================
// 3.2 B-Splines (Cox–de Boor) & Natural Cubic Splines
// ===========================================================================

/// Evaluate the B-spline basis of degree `degree` at points `xs` over
/// non-decreasing knot vector `knots` (`t_0 <= t_1 <= ... <= t_M`) via the
/// Cox–de Boor recursion (`0/0 ≡ 0`, right boundary `B_{M-d-1, d}(t_M) = 1`).
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
    if knots.len() < degree + 2 {
        return Err(PricingError::InvalidParameter(
            "knots.len() must be at least degree + 2",
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
    let n = xs.len();
    let mut out = vec![0.0; n * n_basis];
    let t_max = knots[m];

    let mut b_prev = vec![0.0; m];
    let mut b_curr = vec![0.0; m];

    for (row, &x) in xs.iter().enumerate() {
        require_finite(x, "evaluation point x must be finite")?;
        if (x - t_max).abs() <= 1e-15 {
            out[row * n_basis + (n_basis - 1)] = 1.0;
            continue;
        }
        for j in 0..m {
            b_prev[j] = if x >= knots[j] && x < knots[j + 1] {
                1.0
            } else {
                0.0
            };
        }
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

/// Output of [`fit_random_intercept_glmm`] (Poisson log-link random-intercept GLMM).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomInterceptGlmmResult {
    /// Fixed-effect intercept `β_0`.
    pub intercept: f64,
    /// Fixed-effect slopes `β_1..β_p`.
    pub fixed_effects: Vec<f64>,
    /// Fitted random-effect variance `τ^2 >= 1e-8`.
    pub tau_squared: f64,
    /// Posterior mode random intercepts `u_hat_j` for each group `j = 0..n_groups`.
    pub random_intercepts: Vec<f64>,
    /// Implied Bühlmann–Straub credibility weight `Z_j = Σ μ_{ij} / (Σ μ_{ij} + 1/τ^2)` per group.
    pub group_credibilities: Vec<f64>,
    /// Laplace-approximated marginal log-likelihood at the optimum.
    pub laplace_log_likelihood: f64,
    /// Outer alternating iterations executed.
    pub iterations: usize,
}

/// Fit a Laplace-approximated Poisson random-intercept GLMM
/// `η_{ij} = β_0 + x_{ij}^T β + o_{ij} + u_j`, `u_j ~ N(0, τ^2)`
/// for high-cardinality territories or class codes (`group_ids[i] ∈ 0..n_groups`).
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
    max_iter: usize,
) -> Result<RandomInterceptGlmmResult, PricingError> {
    if n == 0 || p == 0 || n_groups == 0 {
        return Err(PricingError::EmptyInput(
            "n, p, and n_groups must all be >= 1",
        ));
    }
    if max_iter == 0 {
        return Err(PricingError::InvalidParameter("max_iter must be >= 1"));
    }
    if n.checked_mul(p) != Some(x.len()) || y.len() != n || group_ids.len() != n {
        return Err(PricingError::DimensionMismatch(
            "x, y, and group_ids dimensions must match",
        ));
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

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        members[g].push(i);
    }

    // Initialize fixed effects with a standard Poisson GLM.
    let mut fixed = fit_regularized_glm(
        x,
        n,
        p,
        y,
        Some(&w_vec),
        Some(&o_vec),
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::Poisson,
            ..RegularizedGlmOptions::default()
        },
    )?;

    let mut u_hat = vec![0.0; n_groups];
    let mut tau_sq = 0.1_f64;
    let mut laplace_ll = f64::NEG_INFINITY;
    let mut iterations = 0usize;

    let solve_group_modes = |base_eta: &[f64], t2: f64| -> (Vec<f64>, Vec<f64>, f64) {
        let inv_t2 = 1.0 / t2;
        let mut u = vec![0.0; n_groups];
        let mut h = vec![0.0; n_groups];
        let mut ll = 0.0;
        for g in 0..n_groups {
            let mut ug = 0.0_f64;
            for _ in 0..25 {
                let mut g1 = -ug * inv_t2;
                let mut g2 = inv_t2;
                for &i in &members[g] {
                    let mu = (base_eta[i] + ug).clamp(-20.0, 20.0).exp();
                    g1 += w_vec[i] * (y[i] - mu);
                    g2 += w_vec[i] * mu;
                }
                let step = (g1 / g2.max(1e-12)).clamp(-2.0, 2.0);
                ug += step;
                if step.abs() < 1e-10 {
                    break;
                }
            }
            let mut g_val = -0.5 * ug * ug * inv_t2;
            let mut hg = inv_t2;
            for &i in &members[g] {
                let eta_i = (base_eta[i] + ug).clamp(-20.0, 20.0);
                let mu = eta_i.exp();
                g_val += w_vec[i] * (y[i] * eta_i - mu);
                hg += w_vec[i] * mu;
            }
            u[g] = ug;
            h[g] = hg;
            ll += g_val - 0.5 * (t2 * hg).max(1e-300).ln();
        }
        (u, h, ll)
    };

    for iter in 0..max_iter.max(1) {
        iterations = iter + 1;
        let prev_ll = laplace_ll;

        let base_eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut lin = fixed.intercept + o_vec[i];
                for j in 0..p {
                    lin += x[i * p + j] * fixed.coefficients[j];
                }
                lin.clamp(-20.0, 20.0)
            })
            .collect();

        // Optimize log(tau_sq) via golden-section search on Laplace marginal log-likelihood.
        let mut a = (-16.0_f64).max(1e-8_f64.ln());
        let mut b = 2.0_f64;
        let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
        let mut c = b - gr * (b - a);
        let mut d = a + gr * (b - a);
        let mut fc = solve_group_modes(&base_eta, c.exp()).2;
        let mut fd = solve_group_modes(&base_eta, d.exp()).2;
        for _ in 0..24 {
            if fc < fd {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = solve_group_modes(&base_eta, d.exp()).2;
            } else {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = solve_group_modes(&base_eta, c.exp()).2;
            }
        }
        tau_sq = f64::midpoint(a, b).exp().max(1e-8);
        let (u_new, _h_new, ll_new) = solve_group_modes(&base_eta, tau_sq);
        u_hat = u_new;
        laplace_ll = ll_new;

        // Refit fixed effects holding random intercepts as offsets.
        let combined_offsets: Vec<f64> = (0..n).map(|i| o_vec[i] + u_hat[group_ids[i]]).collect();
        fixed = fit_regularized_glm(
            x,
            n,
            p,
            y,
            Some(&w_vec),
            Some(&combined_offsets),
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                ..RegularizedGlmOptions::default()
            },
        )?;

        if (laplace_ll - prev_ll).abs() < 1e-6 * (1.0 + laplace_ll.abs()) {
            break;
        }
    }

    let inv_t2 = 1.0 / tau_sq;
    let mut group_credibilities = vec![0.0; n_groups];
    for g in 0..n_groups {
        let mut sum_mu = 0.0;
        for &i in &members[g] {
            let mut lin = fixed.intercept + o_vec[i] + u_hat[g];
            for j in 0..p {
                lin += x[i * p + j] * fixed.coefficients[j];
            }
            sum_mu += w_vec[i] * lin.clamp(-20.0, 20.0).exp();
        }
        group_credibilities[g] = sum_mu / (sum_mu + inv_t2);
    }

    Ok(RandomInterceptGlmmResult {
        intercept: fixed.intercept,
        fixed_effects: fixed.coefficients,
        tau_squared: tau_sq,
        random_intercepts: u_hat,
        group_credibilities,
        laplace_log_likelihood: laplace_ll,
        iterations,
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
        let glmm =
            fit_random_intercept_glmm(&x, 12, 1, &y, &groups, 3, None, None, 15).expect("glmm");
        assert!(glmm.tau_squared >= 1e-8);
        assert!(
            glmm.group_credibilities
                .iter()
                .all(|&z| (0.0..=1.0).contains(&z))
        );
    }
}
