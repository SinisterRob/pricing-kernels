use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Bailey-Simon Multiplicative Minimum Bias Model result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimumBiasResult {
    /// Fitted relativities for dimension 1 (e.g., driver class or territory).
    pub relativities_dim1: Vec<f64>,
    /// Fitted relativities for dimension 2 (e.g., merit rating or vehicle group).
    pub relativities_dim2: Vec<f64>,
    /// Base loss cost (scale factor).
    pub base_rate: f64,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Final absolute maximum difference across relativities between the last two iterations.
    pub max_diff: f64,
    /// Weighted Chi-squared / balance diagnostic: sum_ij P_ij * (A_ij - r_i * c_j)^2 / (r_i * c_j).
    pub chi_squared: f64,
}

/// Fit a two-dimensional multiplicative rating model using Bailey & Simon (1960)
/// minimum bias procedure with the balance principle:
/// sum_j P_ij * (A_ij - r_i * c_j) = 0 for all i, and
/// sum_i P_ij * (A_ij - r_i * c_j) = 0 for all j.
///
/// Here:
/// - `exposures` is a row-major slice of size `n_rows * n_cols` (P_ij).
/// - `losses` (or average loss costs / claim amounts) of size `n_rows * n_cols` (A_ij).
/// - `max_iter`: Maximum iteration cycles.
/// - `tol`: Absolute convergence threshold for relativities.
pub fn bailey_simon_multiplicative(
    n_rows: usize,
    n_cols: usize,
    exposures: &[f64],
    losses: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<MinimumBiasResult, PricingError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(PricingError::EmptyInput("rows and columns must be non-zero"));
    }
    let total_cells = n_rows * n_cols;
    if exposures.len() != total_cells || losses.len() != total_cells {
        return Err(PricingError::DimensionMismatch("slice length does not match n_rows * n_cols"));
    }

    // Check for negative exposures or losses
    for (&p, &a) in exposures.iter().zip(losses.iter()) {
        if p < 0.0 || a < 0.0 {
            return Err(PricingError::InvalidParameter("exposures and losses must be non-negative"));
        }
    }

    // Initialize relativities: c_j = 1.0 for all j
    let mut r = vec![1.0; n_rows];
    let mut c = vec![1.0; n_cols];

    let mut iter = 0;
    let mut max_diff = 1.0;

    while iter < max_iter && max_diff > tol {
        iter += 1;
        max_diff = 0.0;

        // 1. Update row relativities r_i:
        // r_i = (sum_j P_ij * A_ij) / (sum_j P_ij * c_j)
        for i in 0..n_rows {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n_cols {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                let a = losses[idx];
                num += p * a;
                den += p * c[j];
            }
            let new_r = if den > 0.0 { num / den } else { r[i] };
            let diff = (new_r - r[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            r[i] = new_r;
        }

        // 2. Update column relativities c_j:
        // c_j = (sum_i P_ij * A_ij) / (sum_i P_ij * r_i)
        for j in 0..n_cols {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n_rows {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                let a = losses[idx];
                num += p * a;
                den += p * r[i];
            }
            let new_c = if den > 0.0 { num / den } else { c[j] };
            let diff = (new_c - c[j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            c[j] = new_c;
        }

        // Base rate normalization: normalize c relative to c[0] = 1.0
        if c[0] > 0.0 {
            let factor = c[0];
            for val in &mut c {
                *val /= factor;
            }
            for val in &mut r {
                *val *= factor;
            }
        }
    }

    if max_diff > tol && iter >= max_iter {
        return Err(PricingError::NonConvergence("Bailey-Simon failed to converge within max_iter"));
    }

    // Set base rate as r[0], and normalize r so r[0] = 1.0, base_rate = r[0]
    let base_rate = r[0];
    let mut normalized_r = r.clone();
    if base_rate > 0.0 {
        for val in &mut normalized_r {
            *val /= base_rate;
        }
    }

    // Compute Chi-squared goodness-of-fit diagnostic
    let mut chi_sq = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            let p = exposures[idx];
            let a = losses[idx];
            let fitted = base_rate * normalized_r[i] * c[j];
            if fitted > 0.0 && p > 0.0 {
                let diff = a - fitted;
                chi_sq += p * (diff * diff) / fitted;
            }
        }
    }

    Ok(MinimumBiasResult {
        relativities_dim1: normalized_r,
        relativities_dim2: c,
        base_rate,
        iterations: iter,
        max_diff,
        chi_squared: chi_sq,
    })
}
