use crate::error::PricingError;
use crate::linalg::{cholesky_invert, cholesky_solve};
use serde::{Deserialize, Serialize};

/// GLM error family for pricing kernels (all using canonical/standard log link).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// Poisson family with log link: `log(mu) = X beta + offset`, `V(mu) = mu`, `phi = 1.0`.
    Poisson,
    /// Overdispersed (quasi-)Poisson family with log link: `V(mu) = mu`, `phi` estimated via Pearson chi-squared / df.
    QuasiPoisson,
    /// Gamma family with log link: `log(mu) = X beta + offset`, `V(mu) = mu^2`, `phi` estimated via Pearson chi-squared / df.
    Gamma,
}

/// Fitted GLM result with coefficients, standard errors, dispersion, and deviance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmResult {
    /// Estimated regression coefficients (`beta`).
    pub coefficients: Vec<f64>,
    /// Asymptotic standard errors for coefficients: `sqrt(diag(phi * (X^T W X)^-1))`.
    pub standard_errors: Vec<f64>,
    /// Dispersion parameter (`phi`): `1.0` for `Poisson`, Pearson `chi^2 / df` for `QuasiPoisson` and `Gamma`.
    pub dispersion: f64,
    /// Residual deviance: `2 * sum w_i * d(y_i, mu_i)`.
    pub deviance: f64,
    /// Residual degrees of freedom (`n_active - p`).
    pub df_residual: usize,
    /// Number of IRLS iterations until convergence.
    pub iterations: usize,
}

/// Fit a Generalized Linear Model (GLM) via Iteratively Reweighted Least Squares (IRLS).
///
/// Designed specifically for actuarial commercial rating tables:
/// - `design_matrix`: Row-major slice of size `n_samples * n_features`.
/// - `y`: Observed responses (`>= 0` for Poisson/QuasiPoisson, `> 0` for Gamma on active rows).
/// - `weights`: Non-negative prior weights (e.g. `1.0` for counts, claim count for severity).
/// - `offsets`: Exposure offset `ln(E)` for log-link (use `0.0` if none).
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
        return Err(PricingError::EmptyInput(
            "samples and features must be non-zero",
        ));
    }
    if max_iter == 0 {
        return Err(PricingError::InvalidParameter("max_iter must be > 0"));
    }
    if !tol.is_finite() || !(tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "tolerance must be finite and positive",
        ));
    }
    if n_samples < n_features {
        return Err(PricingError::DimensionMismatch(
            "number of samples must be >= features",
        ));
    }
    if design_matrix.len() != n_samples * n_features {
        return Err(PricingError::DimensionMismatch(
            "design matrix length != n_samples * n_features",
        ));
    }
    if y.len() != n_samples || weights.len() != n_samples || offsets.len() != n_samples {
        return Err(PricingError::DimensionMismatch(
            "vector dimensions do not match n_samples",
        ));
    }
    if design_matrix.iter().any(|v| !v.is_finite()) {
        return Err(PricingError::InvalidParameter(
            "design matrix contains non-finite value",
        ));
    }

    let mut active_samples = 0usize;
    for i in 0..n_samples {
        let yi = y[i];
        let wi = weights[i];
        let oi = offsets[i];
        if !yi.is_finite() || !wi.is_finite() || !oi.is_finite() {
            return Err(PricingError::InvalidParameter(
                "y, weights, and offsets must be finite",
            ));
        }
        if wi < 0.0 {
            return Err(PricingError::InvalidParameter(
                "weights must be non-negative",
            ));
        }
        if wi > 0.0 {
            active_samples += 1;
            match family {
                GlmFamily::Poisson | GlmFamily::QuasiPoisson => {
                    if yi < 0.0 {
                        return Err(PricingError::InvalidParameter(
                            "Poisson response y must be non-negative",
                        ));
                    }
                }
                GlmFamily::Gamma => {
                    if !(yi > 0.0) {
                        return Err(PricingError::NonPositiveValue(
                            "Gamma response y must be strictly positive",
                        ));
                    }
                }
            }
        }
    }

    if active_samples < n_features {
        return Err(PricingError::DimensionMismatch(
            "number of positively weighted samples must be >= features",
        ));
    }

    // Standard R glm.fit initialization (mustart): invariant to offsets and whether
    // column 0 is an intercept.
    let mut mu = vec![0.0; n_samples];
    let mut eta = vec![0.0; n_samples];
    for i in 0..n_samples {
        let mu0 = match family {
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => y[i].max(0.1),
            GlmFamily::Gamma => y[i].max(1e-6),
        };
        mu[i] = mu0;
        eta[i] = mu0.ln();
    }

    let mut beta = vec![0.0; n_features];
    let mut w_irls = vec![0.0; n_samples];
    let mut z = vec![0.0; n_samples];
    let mut xtwx = vec![0.0; n_features * n_features];
    let mut xtwz = vec![0.0; n_features];

    let mut iter = 0;
    let mut converged = false;

    while iter < max_iter {
        iter += 1;

        for i in 0..n_samples {
            let wi = weights[i];
            if wi <= 0.0 {
                w_irls[i] = 0.0;
                z[i] = 0.0;
                continue;
            }
            let mui = mu[i];
            w_irls[i] = match family {
                GlmFamily::Poisson | GlmFamily::QuasiPoisson => wi * mui,
                GlmFamily::Gamma => wi,
            };
            z[i] = (eta[i] - offsets[i]) + (y[i] - mui) / mui;
        }

        // Form normal equations: (X^T W X) beta_new = X^T W z
        xtwx.fill(0.0);
        xtwz.fill(0.0);
        for i in 0..n_samples {
            let wi = w_irls[i];
            if wi <= 0.0 {
                continue;
            }
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

        let beta_new = cholesky_solve(&xtwx, n_features, &xtwz)?;

        let mut max_change = 0.0;
        for j in 0..n_features {
            let change = (beta_new[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        beta = beta_new;

        // Recompute eta and mu at the updated beta so diagnostics & next iteration are exact.
        for i in 0..n_samples {
            let mut line = offsets[i];
            let row = &design_matrix[i * n_features..(i + 1) * n_features];
            for j in 0..n_features {
                line += row[j] * beta[j];
            }
            eta[i] = line;
            mu[i] = line.clamp(-700.0, 700.0).exp().max(1e-15);
        }

        if iter > 1 && max_change <= tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(PricingError::NonConvergence(
            "GLM IRLS did not converge within max iterations",
        ));
    }

    // Recompute final Fisher information X^T W X, deviance, and Pearson chi-squared at converged beta.
    xtwx.fill(0.0);
    let mut deviance = 0.0;
    let mut pearson_chi2 = 0.0;

    for i in 0..n_samples {
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        let yi = y[i];
        let mui = mu[i];

        let w_final = match family {
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => wi * mui,
            GlmFamily::Gamma => wi,
        };
        let row = &design_matrix[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            let x_ij = row[j];
            for k in 0..=j {
                let val = w_final * x_ij * row[k];
                xtwx[j * n_features + k] += val;
                if k != j {
                    xtwx[k * n_features + j] += val;
                }
            }
        }

        match family {
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => {
                deviance += if yi > 0.0 {
                    2.0 * wi * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * wi * mui
                };
                let diff = yi - mui;
                pearson_chi2 += wi * (diff * diff) / mui;
            }
            GlmFamily::Gamma => {
                deviance += 2.0 * wi * (-((yi / mui).ln()) + (yi - mui) / mui);
                let rel = (yi - mui) / mui;
                pearson_chi2 += wi * rel * rel;
            }
        }
    }

    let df_residual = active_samples.saturating_sub(n_features);
    let dispersion = match family {
        GlmFamily::Poisson => 1.0,
        GlmFamily::QuasiPoisson | GlmFamily::Gamma => {
            if df_residual > 0 {
                pearson_chi2 / df_residual as f64
            } else {
                1.0
            }
        }
    };

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
