//! Pure-Rust dense linear algebra helpers for Cholesky factorization and matrix equations.

use crate::error::PricingError;

/// Compute the lower-triangular Cholesky factor `L` of symmetric positive-definite
/// `dim × dim` matrix `a` in row-major layout (`A = L L^T`).
pub fn cholesky_lower(a: &[f64], dim: usize) -> Result<Vec<f64>, PricingError> {
    if dim == 0 || a.len() != dim * dim {
        return Err(PricingError::DimensionMismatch(
            "matrix slice size does not match dim * dim",
        ));
    }
    for i in 0..dim {
        let d = a[i * dim + i];
        if !d.is_finite() || d <= 0.0 {
            return Err(PricingError::SingularMatrix(
                "matrix is not positive definite: diagonal non-positive",
            ));
        }
    }
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        let row_tol = 1e-14 * a[i * dim + i];
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            if !sum.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "matrix contains non-finite entry",
                ));
            }
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= row_tol {
                    return Err(PricingError::SingularMatrix(
                        "matrix is not positive definite: breakdown in diagonal",
                    ));
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Ok(l)
}

/// Solve lower-triangular system `L y = b` by forward substitution into `y`.
pub fn solve_lower_into(
    l: &[f64],
    dim: usize,
    b: &[f64],
    y: &mut [f64],
) -> Result<(), PricingError> {
    if dim == 0 || l.len() != dim * dim || b.len() != dim || y.len() != dim {
        return Err(PricingError::DimensionMismatch(
            "dimensions do not match dim for lower-triangular solve",
        ));
    }
    for i in 0..dim {
        let diag = l[i * dim + i];
        if !diag.is_finite() || diag == 0.0 {
            return Err(PricingError::SingularMatrix(
                "zero or non-finite diagonal in lower-triangular factor",
            ));
        }
        let mut sum = b[i];
        let row = &l[i * dim..i * dim + i];
        for (lij, &yk) in row.iter().zip(y[..i].iter()) {
            sum -= lij * yk;
        }
        y[i] = sum / diag;
    }
    Ok(())
}

/// Solve upper-triangular system `L^T x = y` (where `L` is lower-triangular) by back substitution into `x`.
pub fn solve_upper_transpose_into(
    l: &[f64],
    dim: usize,
    y: &[f64],
    x: &mut [f64],
) -> Result<(), PricingError> {
    if dim == 0 || l.len() != dim * dim || y.len() != dim || x.len() != dim {
        return Err(PricingError::DimensionMismatch(
            "dimensions do not match dim for upper-transpose solve",
        ));
    }
    for i in (0..dim).rev() {
        let diag = l[i * dim + i];
        if !diag.is_finite() || diag == 0.0 {
            return Err(PricingError::SingularMatrix(
                "zero or non-finite diagonal in upper-transpose factor",
            ));
        }
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= l[k * dim + i] * x[k];
        }
        x[i] = sum / diag;
    }
    Ok(())
}

/// Solve symmetric positive-definite system `A x = b` using Cholesky factorization.
pub fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Result<Vec<f64>, PricingError> {
    if b.len() != dim {
        return Err(PricingError::DimensionMismatch(
            "vector b length does not match dim",
        ));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(PricingError::InvalidParameter(
            "vector b contains non-finite value",
        ));
    }
    let l = cholesky_lower(a, dim)?;
    let mut y = vec![0.0; dim];
    solve_lower_into(&l, dim, b, &mut y)?;
    let mut x = vec![0.0; dim];
    solve_upper_transpose_into(&l, dim, &y, &mut x)?;
    Ok(x)
}

/// Invert a symmetric positive-definite `dim × dim` matrix `A` using Cholesky factor `L`.
pub fn cholesky_invert(a: &[f64], dim: usize) -> Result<Vec<f64>, PricingError> {
    let l = cholesky_lower(a, dim)?;
    let mut inv = vec![0.0; dim * dim];
    let mut e = vec![0.0; dim];
    let mut y = vec![0.0; dim];
    let mut x = vec![0.0; dim];

    for col in 0..dim {
        e.fill(0.0);
        e[col] = 1.0;
        solve_lower_into(&l, dim, &e, &mut y)?;
        solve_upper_transpose_into(&l, dim, &y, &mut x)?;
        for row in 0..dim {
            inv[row * dim + col] = x[row];
        }
    }
    Ok(inv)
}

/// Multiply matrix `A` (`rows_a × cols_a`) by matrix `B` (`cols_a × cols_b`) into `C` (`rows_a × cols_b`).
#[allow(dead_code)]
pub fn mat_mul(
    a: &[f64],
    b: &[f64],
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
    c: &mut [f64],
) -> Result<(), PricingError> {
    if a.len() != rows_a * cols_a || b.len() != cols_a * cols_b || c.len() != rows_a * cols_b {
        return Err(PricingError::DimensionMismatch(
            "matrix dimensions do not match slice lengths for mat_mul",
        ));
    }
    c.fill(0.0);
    for i in 0..rows_a {
        for k in 0..cols_a {
            let aik = a[i * cols_a + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..cols_b {
                c[i * cols_b + j] += aik * b[k * cols_b + j];
            }
        }
    }
    Ok(())
}

/// Compute weighted dot product: `sum_i w_i * x_i * y_i`.
#[allow(dead_code)]
#[inline]
pub fn weighted_dot(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .zip(w.iter())
        .map(|((&xi, &yi), &wi)| wi * xi * yi)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linalg_rejects_mismatched_and_non_finite_inputs() {
        let mut y = [0.0; 1];
        assert!(solve_lower_into(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0, 2.0], &mut y).is_err());
        let mut c = [0.0; 1];
        assert!(mat_mul(&[1.0, 2.0], &[3.0, 4.0], 1, 2, 2, &mut c).is_err());
        assert!(cholesky_solve(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0, f64::NAN]).is_err());
    }
}
