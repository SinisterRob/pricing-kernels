//! Bailey (1963) balance-principle and Bailey–Simon (1960) chi-squared minimum bias
//! classification ratemaking solvers.

use crate::error::{PricingError, SolverOptions};
use serde::{Deserialize, Serialize};

/// Validated two-dimensional classification rating table (`n_rows × n_cols`) of
/// exposures and observed average loss costs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingTable {
    n_rows: usize,
    n_cols: usize,
    exposures: Vec<f64>,
    losses: Vec<f64>,
}

impl RatingTable {
    /// Construct and validate a row-major `n_rows × n_cols` rating table.
    pub fn try_new(
        n_rows: usize,
        n_cols: usize,
        exposures: Vec<f64>,
        losses: Vec<f64>,
    ) -> Result<Self, PricingError> {
        validate_grid(n_rows, n_cols, &exposures, &losses, 1, 1e-6)?;
        Ok(Self {
            n_rows,
            n_cols,
            exposures,
            losses,
        })
    }

    /// Number of dimension-1 levels (rows).
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of dimension-2 levels (columns).
    #[inline]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Flat row-major exposure slice.
    #[inline]
    pub fn exposures(&self) -> &[f64] {
        &self.exposures
    }

    /// Flat row-major average loss-cost slice.
    #[inline]
    pub fn losses(&self) -> &[f64] {
        &self.losses
    }

    /// Fit Bailey (1963) multiplicative balance-principle relativities.
    pub fn bailey_simon_multiplicative(
        &self,
        options: &SolverOptions,
    ) -> Result<MinimumBiasResult, PricingError> {
        bailey_simon_multiplicative(
            self.n_rows,
            self.n_cols,
            &self.exposures,
            &self.losses,
            options,
        )
    }

    /// Fit Bailey & Simon (1960) chi-squared minimum bias relativities.
    pub fn bailey_simon_chi_squared(
        &self,
        options: &SolverOptions,
    ) -> Result<MinimumBiasResult, PricingError> {
        bailey_simon_chi_squared(
            self.n_rows,
            self.n_cols,
            &self.exposures,
            &self.losses,
            options,
        )
    }
}

/// Multiplicative Minimum Bias Model result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimumBiasResult {
    /// Fitted relativities for dimension 1 (e.g., driver class or territory),
    /// normalized so the reference level (first positive row) equals `1.0`.
    pub relativities_dim1: Vec<f64>,
    /// Fitted relativities for dimension 2 (e.g., merit rating or vehicle group),
    /// normalized so the reference level (first positive column) equals `1.0`.
    pub relativities_dim2: Vec<f64>,
    /// Base loss cost (scale factor) at the reference cell `(ref_row, ref_col)`.
    pub base_rate: f64,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Final maximum change across normalized relativities and relative `base_rate` between the last two iterations.
    pub max_diff: f64,
    /// Weighted Chi-squared goodness-of-fit diagnostic: `sum_ij P_ij * (A_ij - fitted_ij)^2 / fitted_ij`.
    pub chi_squared: f64,
}

fn validate_grid(
    n_rows: usize,
    n_cols: usize,
    exposures: &[f64],
    losses: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(), PricingError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(PricingError::EmptyInput(
            "rows and columns must be non-zero",
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
    let Some(total_cells) = n_rows.checked_mul(n_cols) else {
        return Err(PricingError::DimensionMismatch(
            "n_rows * n_cols overflowed usize",
        ));
    };
    if exposures.len() != total_cells || losses.len() != total_cells {
        return Err(PricingError::DimensionMismatch(
            "slice length does not match n_rows * n_cols",
        ));
    }
    let mut total_weighted_loss = 0.0;
    for (&p, &a) in exposures.iter().zip(losses.iter()) {
        if !p.is_finite() || !a.is_finite() || p < 0.0 || a < 0.0 {
            return Err(PricingError::InvalidParameter(
                "exposures and losses must be finite and non-negative",
            ));
        }
        total_weighted_loss += p * a;
    }
    if !(total_weighted_loss > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "total weighted loss must be positive",
        ));
    }
    Ok(())
}

fn compute_chi_squared(
    n_rows: usize,
    n_cols: usize,
    exposures: &[f64],
    losses: &[f64],
    base_rate: f64,
    r: &[f64],
    c: &[f64],
) -> f64 {
    let mut chi_sq = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            let p = exposures[idx];
            let a = losses[idx];
            let fitted = base_rate * r[i] * c[j];
            if fitted > 0.0 && p > 0.0 {
                let diff = a - fitted;
                chi_sq += p * (diff * diff) / fitted;
            }
        }
    }
    chi_sq
}

/// Normalize `(r, c)` against the first positive row/column, write normalized row
/// relativities into `norm_r`, and return `(base_rate, max_diff)` vs previous iterate.
fn normalize_and_diff(
    r: &mut [f64],
    c: &mut [f64],
    norm_r: &mut [f64],
    prev_r: &mut [f64],
    prev_c: &mut [f64],
    prev_base: &mut f64,
) -> Result<(f64, f64), PricingError> {
    let ref_c = c
        .iter()
        .copied()
        .find(|&v| v > 0.0)
        .ok_or(PricingError::SingularMatrix(
            "all column relativities collapsed to zero",
        ))?;
    for val in c.iter_mut() {
        *val /= ref_c;
    }
    for val in r.iter_mut() {
        *val *= ref_c;
    }

    let base_rate = r
        .iter()
        .copied()
        .find(|&v| v > 0.0)
        .ok_or(PricingError::SingularMatrix(
            "all row relativities collapsed to zero",
        ))?;
    for (dst, &src) in norm_r.iter_mut().zip(r.iter()) {
        *dst = src / base_rate;
    }

    let mut max_diff = (base_rate - *prev_base).abs() / base_rate.max(1.0);
    for (&nr, &pr) in norm_r.iter().zip(prev_r.iter()) {
        max_diff = max_diff.max((nr - pr).abs());
    }
    for (&cv, &pc) in c.iter().zip(prev_c.iter()) {
        max_diff = max_diff.max((cv - pc).abs());
    }

    *prev_base = base_rate;
    prev_r.copy_from_slice(norm_r);
    prev_c.copy_from_slice(c);
    Ok((base_rate, max_diff))
}

/// Fit a two-dimensional multiplicative rating model using the **Bailey (1963)**
/// balance-principle minimum bias procedure (mathematically equivalent to a
/// Poisson log-link GLM; Brown 1988):
///
/// `sum_j P_ij * (A_ij - r_i * c_j) = 0` for all `i`, and
/// `sum_i P_ij * (A_ij - r_i * c_j) = 0` for all `j`.
///
/// - `exposures`: Row-major slice of size `n_rows * n_cols` (`P_ij`).
/// - `losses`: Row-major slice of average loss costs / claim frequencies (`A_ij`).
/// - `options`: Convergence controls (`max_iter`, `tol`).
pub fn bailey_simon_multiplicative(
    n_rows: usize,
    n_cols: usize,
    exposures: &[f64],
    losses: &[f64],
    options: &SolverOptions,
) -> Result<MinimumBiasResult, PricingError> {
    let SolverOptions { max_iter, tol } = *options;
    validate_grid(n_rows, n_cols, exposures, losses, max_iter, tol)?;

    let mut r = vec![1.0; n_rows];
    let mut c = vec![1.0; n_cols];
    let mut norm_r = vec![1.0; n_rows];
    let mut prev_r = vec![1.0; n_rows];
    let mut prev_c = vec![1.0; n_cols];
    let mut prev_base = 0.0;

    for iter in 1..=max_iter {
        // 1. Update row factors: r_i = (sum_j P_ij * A_ij) / (sum_j P_ij * c_j)
        for i in 0..n_rows {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n_cols {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                num += p * losses[idx];
                den += p * c[j];
            }
            r[i] = if den > 0.0 { num / den } else { 0.0 };
        }

        // 2. Update column factors: c_j = (sum_i P_ij * A_ij) / (sum_i P_ij * r_i)
        for j in 0..n_cols {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n_rows {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                num += p * losses[idx];
                den += p * r[i];
            }
            c[j] = if den > 0.0 { num / den } else { 0.0 };
        }

        let (base_rate, max_diff) = normalize_and_diff(
            &mut r,
            &mut c,
            &mut norm_r,
            &mut prev_r,
            &mut prev_c,
            &mut prev_base,
        )?;

        if max_diff <= tol {
            let chi_squared =
                compute_chi_squared(n_rows, n_cols, exposures, losses, base_rate, &norm_r, &c);
            return Ok(MinimumBiasResult {
                relativities_dim1: norm_r,
                relativities_dim2: c,
                base_rate,
                iterations: iter,
                max_diff,
                chi_squared,
            });
        }
    }

    Err(PricingError::NonConvergence(
        "Bailey multiplicative minimum bias failed to converge within max_iter",
    ))
}

/// Fit a two-dimensional multiplicative rating model using the original
/// **Bailey & Simon (1960)** weighted Chi-squared minimum bias procedure:
///
/// Minimizes `sum_ij P_ij * (A_ij - r_i * c_j)^2 / (r_i * c_j)` via the closed-form
/// coordinate updates:
/// `r_i = sqrt( (sum_j P_ij * A_ij^2 / c_j) / (sum_j P_ij * c_j) )`,
/// `c_j = sqrt( (sum_i P_ij * A_ij^2 / r_i) / (sum_i P_ij * r_i) )`.
pub fn bailey_simon_chi_squared(
    n_rows: usize,
    n_cols: usize,
    exposures: &[f64],
    losses: &[f64],
    options: &SolverOptions,
) -> Result<MinimumBiasResult, PricingError> {
    let SolverOptions { max_iter, tol } = *options;
    validate_grid(n_rows, n_cols, exposures, losses, max_iter, tol)?;

    let mut r = vec![1.0; n_rows];
    let mut c = vec![1.0; n_cols];
    let mut norm_r = vec![1.0; n_rows];
    let mut prev_r = vec![1.0; n_rows];
    let mut prev_c = vec![1.0; n_cols];
    let mut prev_base = 0.0;

    for iter in 1..=max_iter {
        for i in 0..n_rows {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n_cols {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                if p > 0.0 && c[j] > 0.0 {
                    let a = losses[idx];
                    num += p * (a * a) / c[j];
                    den += p * c[j];
                }
            }
            r[i] = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
        }

        for j in 0..n_cols {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n_rows {
                let idx = i * n_cols + j;
                let p = exposures[idx];
                if p > 0.0 && r[i] > 0.0 {
                    let a = losses[idx];
                    num += p * (a * a) / r[i];
                    den += p * r[i];
                }
            }
            c[j] = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
        }

        let (base_rate, max_diff) = normalize_and_diff(
            &mut r,
            &mut c,
            &mut norm_r,
            &mut prev_r,
            &mut prev_c,
            &mut prev_base,
        )?;

        if max_diff <= tol {
            let chi_squared =
                compute_chi_squared(n_rows, n_cols, exposures, losses, base_rate, &norm_r, &c);
            return Ok(MinimumBiasResult {
                relativities_dim1: norm_r,
                relativities_dim2: c,
                base_rate,
                iterations: iter,
                max_diff,
                chi_squared,
            });
        }
    }

    Err(PricingError::NonConvergence(
        "Bailey-Simon chi-squared minimum bias failed to converge within max_iter",
    ))
}
