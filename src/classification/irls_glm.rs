use serde::{Deserialize, Serialize};
use crate::error::PricingError;
use crate::linalg::{cholesky_solve, cholesky_invert};

/// GLM error family for pricing kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// Poisson family with log link: log(mu) = X beta + offset. Variance V(mu) = mu.
    Poisson,
    /// Gamma family with log link: log(mu) = X beta + offset. Variance V(mu) = mu^2.
    Gamma,
}

/// Fitted GLM result with coefficients, standard errors, dispersion, and deviance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmResult {
    /// Estimated regression coefficients (beta).
    pub coefficients: Vec<f64>,
    /// Asymptotic standard errors for coefficients: sqrt(diag(phi * (X^T W X)^-1)).
    pub standard_errors: Vec<f64>,
    /// Estimated dispersion parameter (phi). For Poisson, fixed or quasi-dispersion.
    pub dispersion: f64,
    /// Residual deviance: 2 * sum w_i * d(y_i, mu_i).
    pub deviance: f64,
    /// Degrees of freedom of the residuals (N - p).
    pub df_residual: usize,
    /// Number of IRLS iterations until convergence.
    pub iterations: usize,
}

/// Fit a Generalized Linear Model (GLM) via Iteratively Reweighted Least Squares (IRLS).
///
/// Designed specifically for actuarial commercial rating tables:
/// - `design_matrix`: Row-major slice of size `n_samples * n_features`.
/// - `y`: Observed responses (e.g. claim count for Poisson, average severity for Gamma).
/// - `weights`: Prior weights (e.g. 1.0 for counts, claim count for severity).
/// - `offsets`: Exposure offset $\ln(E)$ for Poisson log-link (use 0.0 if none).
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_irls(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<GlmResult, PricingError> {
    if n_samples == 0 || n_features == 0 {
        return Err(PricingError::EmptyInput("samples and features must be non-zero"));
    }
    if n_samples < n_features {
        return Err(PricingError::DimensionMismatch("number of samples must be >= features"));
    }
    if design_matrix.len() != n_samples * n_features {
        return Err(PricingError::DimensionMismatch("design matrix length != n_samples * n_features"));
    }
    if y.len() != n_samples || weights.len() != n_samples || offsets.len() != n_samples {
        return Err(PricingError::DimensionMismatch("vector dimensions do not match n_samples"));
    }

    // Initialize beta = 0.0 or intercept-based
    let mut beta = vec![0.0; n_features];

    // Compute sample mean with offsets for sensible starting value
    let mut sum_y = 0.0;
    let mut sum_w = 0.0;
    for i in 0..n_samples {
        if weights[i] > 0.0 {
            sum_y += weights[i] * y[i];
            sum_w += weights[i];
        }
    }
    let mean_y = if sum_w > 0.0 { (sum_y / sum_w).max(1e-4) } else { 1.0 };
    beta[0] = mean_y.ln();

    let mut eta = vec![0.0; n_samples];
    let mut mu = vec![0.0; n_samples];
    let mut w_irls = vec![0.0; n_samples];
    let mut z = vec![0.0; n_samples]; // working response

    let mut xtwx = vec![0.0; n_features * n_features];
    let mut xtwz = vec![0.0; n_features];

    let mut iter = 0;
    let mut converged = false;

    while iter < max_iter {
        iter += 1;

        // 1. Calculate linear predictor eta_i = sum_j X_ij * beta_j + offset_i
        for i in 0..n_samples {
            let mut line = offsets[i];
            let row = &design_matrix[i * n_features..(i + 1) * n_features];
            for j in 0..n_features {
                line += row[j] * beta[j];
            }
            eta[i] = line;
            
            // Link function: log link -> mu = exp(eta), clamped to avoid float overflow
            let mu_i = line.clamp(-700.0, 700.0).exp();
            mu[i] = mu_i.max(1e-12);

            // Working response and weights
            match family {
                GlmFamily::Poisson => {
                    // g'(mu) = 1/mu, V(mu) = mu
                    // W = w_prior / (V(mu) * [g']^2) = w_prior * mu
                    // z = eta + (y - mu) * g'(mu) = eta + (y - mu) / mu
                    let prior_w = weights[i];
                    w_irls[i] = (prior_w * mu[i]).max(1e-12);
                    z[i] = (eta[i] - offsets[i]) + (y[i] - mu[i]) / mu[i];
                }
                GlmFamily::Gamma => {
                    // g'(mu) = 1/mu, V(mu) = mu^2
                    // W = w_prior / (V(mu) * [g']^2) = w_prior
                    // z = eta + (y - mu) / mu
                    let prior_w = weights[i];
                    w_irls[i] = prior_w.max(1e-12);
                    z[i] = (eta[i] - offsets[i]) + (y[i] - mu[i]) / mu[i];
                }
            }
        }

        // 2. Form normal equations: (X^T W X) beta_new = X^T W z
        xtwx.fill(0.0);
        xtwz.fill(0.0);

        for i in 0..n_samples {
            let wi = w_irls[i];
            let zi = z[i];
            let row = &design_matrix[i * n_features..(i + 1) * n_features];

            for j in 0..n_features {
                let x_ij = row[j];
                xtwz[j] += wi * x_ij * zi;
                for k in 0..=j {
                    let val = wi * x_ij * row[k];
                    xtwx[j * n_features + k] += val;
                    if k != j {
                        xtwx[k * n_features + j] += val;
                    }
                }
            }
        }

        // Small Tikhonov regularization on diagonal for numerical stability (1e-9)
        for j in 0..n_features {
            xtwx[j * n_features + j] += 1e-9;
        }

        // 3. Solve for beta_new
        let beta_new = cholesky_solve(&xtwx, n_features, &xtwz)?;

        // Check convergence
        let mut max_change = 0.0;
        for j in 0..n_features {
            let change = (beta_new[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        beta = beta_new;
        if max_change < tol {
            converged = true;
            break;
        }
    }

    if !converged && iter >= max_iter {
        return Err(PricingError::NonConvergence("GLM IRLS did not converge within max iterations"));
    }

    // Compute deviance
    let mut deviance = 0.0;
    for i in 0..n_samples {
        let yi = y[i];
        let mui = mu[i];
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        let cell_dev = match family {
            GlmFamily::Poisson => {
                if yi > 0.0 {
                    2.0 * wi * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * wi * mui
                }
            }
            GlmFamily::Gamma => {
                if yi > 0.0 {
                    2.0 * wi * (-((yi / mui).ln()) + (yi - mui) / mui)
                } else {
                    0.0
                }
            }
        };
        deviance += cell_dev;
    }

    let df_residual = n_samples.saturating_sub(n_features);
    let dispersion = if df_residual > 0 {
        match family {
            GlmFamily::Poisson => (deviance / df_residual as f64).max(1.0), // Pearson/Deviance quasi-dispersion
            GlmFamily::Gamma => deviance / df_residual as f64,
        }
    } else {
        1.0
    };

    // Covariance matrix = phi * (X^T W X)^-1
    let inv_xtwx = cholesky_invert(&xtwx, n_features)?;
    let mut standard_errors = vec![0.0; n_features];
    for j in 0..n_features {
        let var = dispersion * inv_xtwx[j * n_features + j];
        standard_errors[j] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }

    Ok(GlmResult {
        coefficients: beta,
        standard_errors,
        dispersion,
        deviance,
        df_residual,
        iterations: iter,
    })
}
