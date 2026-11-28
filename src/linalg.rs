//! Pure-Rust dense linear algebra helpers for Cholesky factorization and matrix equations.

use crate::error::PricingError;

/// Compute the lower-triangular Cholesky factor `L` of symmetric positive-definite
/// `dim × dim` matrix `a` in row-major layout (`A = L L^T`).
pub fn cholesky_lower(a: &[f64], dim: usize) -> Result<Vec<f64>, PricingError> {
    if dim == 0 || dim.checked_mul(dim) != Some(a.len()) {
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
    if dim == 0 || dim.checked_mul(dim) != Some(l.len()) || b.len() != dim || y.len() != dim {
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
    if dim == 0 || dim.checked_mul(dim) != Some(l.len()) || y.len() != dim || x.len() != dim {
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

/// Solve `A x = b` for symmetric positive-definite row-major `dim × dim` matrix `A`
/// via its Cholesky factor (`L y = b`, then `L^T x = y`).
pub fn cholesky_solve(a: &[f64], dim: usize, b: &[f64]) -> Result<Vec<f64>, PricingError> {
    let l = cholesky_lower(a, dim)?;
    let mut y = vec![0.0; dim];
    let mut x = vec![0.0; dim];
    solve_lower_into(&l, dim, b, &mut y)?;
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

/// Reusable workspace for allocation-free column-major Householder QR WLS solves
/// (unpenalized or L2 Ridge-augmented) inside iterative loops such as GLM IRLS.
#[derive(Debug, Default)]
pub(crate) struct QrWorkspace {
    /// Column-major `m × p` weighted/augmented matrix (`col j` at `j * m .. (j + 1) * m`).
    a_cm: Vec<f64>,
    b: Vec<f64>,
    sqrt_w: Vec<f64>,
    v: Vec<f64>,
    /// Row-major `p × p` copy of upper-triangular `R` for contiguous back-sub / `R^{-1}`.
    r_rm: Vec<f64>,
    /// Column-major `p × p` inverse `R^{-1}`.
    r_inv_cm: Vec<f64>,
    /// Euclidean norms of the `p` columns of `a_cm` before factorization (rank test scale).
    col_norm: Vec<f64>,
}

impl QrWorkspace {
    pub(crate) fn with_capacity(m: usize, p: usize) -> Self {
        Self {
            a_cm: vec![0.0; m * p],
            b: vec![0.0; m],
            sqrt_w: vec![0.0; m],
            v: vec![0.0; m],
            r_rm: vec![0.0; p * p],
            r_inv_cm: vec![0.0; p * p],
            col_norm: vec![0.0; p],
        }
    }

    #[inline]
    fn ensure_dims(&mut self, m: usize, n: usize, p: usize) {
        if self.col_norm.len() != p {
            self.col_norm.resize(p, 0.0);
        }
        if self.a_cm.len() != m * p {
            self.a_cm.resize(m * p, 0.0);
        }
        if self.b.len() != m {
            self.b.resize(m, 0.0);
        }
        if self.sqrt_w.len() != n {
            self.sqrt_w.resize(n, 0.0);
        }
        if self.v.len() != m {
            self.v.resize(m, 0.0);
        }
    }
}

/// Relative rank tolerance of the Householder QR: column `k` is declared linearly dependent on
/// columns `0..k` when `|R_kk| <= QR_RANK_RTOL * ||a_k||_2` (the sine of the angle between
/// `a_k` and the span of the preceding columns). Comparing each `R_kk` with its own original
/// column norm (as LINPACK `dqrdc2` used by R `lm`/`glm` does) makes the test invariant to
/// column scaling, so rescaling a covariate by `1e12` or `1e-13` cannot trigger a false
/// [`PricingError::SingularMatrix`].
const QR_RANK_RTOL: f64 = 1e-12;

fn qr_factorize_a_cm_in_place(
    m: usize,
    p: usize,
    ws: &mut QrWorkspace,
    apply_rhs: bool,
) -> Result<(), PricingError> {
    for k in 0..p {
        let col = &ws.a_cm[k * m..(k + 1) * m];
        let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        ws.col_norm[k] = norm;
    }
    for k in 0..p {
        let len = m - k;
        let col_k_start = k * m + k;
        let col_k = &ws.a_cm[col_k_start..col_k_start + len];

        let mut norm_sq = 0.0;
        for &val in col_k {
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(PricingError::SingularMatrix(
                "design matrix is rank-deficient or collinear",
            ));
        }
        let x0 = col_k[0];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let v_slice = &mut ws.v[..len];
        v_slice[0] = x0 - alpha;
        let mut v_norm_sq = v_slice[0] * v_slice[0];
        for (dst, &src) in v_slice[1..].iter_mut().zip(col_k[1..].iter()) {
            *dst = src;
            v_norm_sq += src * src;
        }
        if v_norm_sq <= 1e-30 * norm_sq {
            continue;
        }
        let tau = 2.0 / v_norm_sq;

        ws.a_cm[col_k_start] = alpha;
        ws.a_cm[col_k_start + 1..col_k_start + len].fill(0.0);

        let v_ref = &ws.v[..len];
        for j in (k + 1)..p {
            let col_j_start = j * m + k;
            let col_j = &mut ws.a_cm[col_j_start..col_j_start + len];
            let mut dot = 0.0;
            for (&vi, &aij) in v_ref.iter().zip(col_j.iter()) {
                dot += vi * aij;
            }
            let scale = tau * dot;
            for (aij, &vi) in col_j.iter_mut().zip(v_ref.iter()) {
                *aij -= scale * vi;
            }
        }

        if apply_rhs {
            let b_slice = &mut ws.b[k..m];
            let mut dot_b = 0.0;
            for (&vi, &bi) in v_ref.iter().zip(b_slice.iter()) {
                dot_b += vi * bi;
            }
            let scale_b = tau * dot_b;
            for (bi, &vi) in b_slice.iter_mut().zip(v_ref.iter()) {
                *bi -= scale_b * vi;
            }
        }
    }

    for k in 0..p {
        let d = ws.a_cm[k * m + k].abs();
        if !d.is_finite() {
            return Err(PricingError::InvalidParameter(
                "QR diagonal overflowed to non-finite",
            ));
        }
        let col_norm = ws.col_norm[k];
        if !(col_norm > 0.0) || !(d > QR_RANK_RTOL * col_norm) {
            return Err(PricingError::SingularMatrix(
                "design matrix is rank-deficient or collinear",
            ));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qr_factorize_weighted_ridge_cm_into(
    x_cm: &[f64],
    y: Option<&[f64]>,
    weights: &[f64],
    n: usize,
    p: usize,
    ridge_lambda: f64,
    penalize_intercept: bool,
    ws: &mut QrWorkspace,
) -> Result<usize, PricingError> {
    let start = usize::from(!penalize_intercept);
    let n_pen = if ridge_lambda > 0.0 && start < p {
        p - start
    } else {
        0
    };
    let m = n + n_pen;
    if n == 0 || p == 0 || p > m || x_cm.len() != n * p || weights.len() != n {
        return Err(PricingError::DimensionMismatch(
            "invalid dimensions for QR weighted least squares",
        ));
    }
    if let Some(yv) = y
        && yv.len() != n
    {
        return Err(PricingError::DimensionMismatch(
            "response vector length does not match n",
        ));
    }

    ws.ensure_dims(m, n, p);

    if let Some(yv) = y {
        for i in 0..n {
            let w = weights[i];
            let yi = yv[i];
            if !w.is_finite() || w < 0.0 || !yi.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "WLS requires finite non-negative weights and finite responses",
                ));
            }
            let sw = w.sqrt();
            ws.sqrt_w[i] = sw;
            ws.b[i] = sw * yi;
        }
        if n_pen > 0 {
            ws.b[n..m].fill(0.0);
        }
    } else {
        for i in 0..n {
            let w = weights[i];
            if !w.is_finite() || w < 0.0 {
                return Err(PricingError::InvalidParameter(
                    "WLS requires finite non-negative weights",
                ));
            }
            ws.sqrt_w[i] = w.sqrt();
        }
    }

    let sqrt_lam = if n_pen > 0 { ridge_lambda.sqrt() } else { 0.0 };
    let sw = &ws.sqrt_w[..n];
    for j in 0..p {
        let src = &x_cm[j * n..(j + 1) * n];
        let dst = &mut ws.a_cm[j * m..(j + 1) * m];
        for ((d, &s), &w) in dst[..n].iter_mut().zip(src.iter()).zip(sw.iter()) {
            *d = w * s;
        }
        if n_pen > 0 {
            dst[n..m].fill(0.0);
            if j >= start {
                dst[n + (j - start)] = sqrt_lam;
            }
        }
    }

    qr_factorize_a_cm_in_place(m, p, ws, y.is_some())?;
    Ok(m)
}

/// Zero-allocation Householder QR weighted least squares (with optional L2 Ridge augmentation)
/// from pre-validated column-major `x_cm`, writing `beta` into `beta_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_least_squares_qr_ridge_cm_beta_into(
    x_cm: &[f64],
    y: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ridge_lambda: f64,
    penalize_intercept: bool,
    ws: &mut QrWorkspace,
    beta_out: &mut [f64],
) -> Result<(), PricingError> {
    if beta_out.len() != p {
        return Err(PricingError::DimensionMismatch(
            "beta_out length does not match p",
        ));
    }
    let m = qr_factorize_weighted_ridge_cm_into(
        x_cm,
        Some(y),
        weights,
        n,
        p,
        ridge_lambda,
        penalize_intercept,
        ws,
    )?;
    for i in (0..p).rev() {
        let mut sum = ws.b[i];
        for j in (i + 1)..p {
            sum -= ws.a_cm[j * m + i] * beta_out[j];
        }
        beta_out[i] = sum / ws.a_cm[i * m + i];
    }
    Ok(())
}

/// Factorize `[W^{1/2} X; sqrt(lambda) D]` and return `(X^T W X + Lambda)^{-1} = R^{-1} R^{-T}`
/// in row-major `p × p` layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_least_squares_qr_ridge_cm_cov(
    x_cm: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
    ridge_lambda: f64,
    penalize_intercept: bool,
    ws: &mut QrWorkspace,
) -> Result<Vec<f64>, PricingError> {
    let m = qr_factorize_weighted_ridge_cm_into(
        x_cm,
        None,
        weights,
        n,
        p,
        ridge_lambda,
        penalize_intercept,
        ws,
    )?;

    if ws.r_rm.len() != p * p {
        ws.r_rm.resize(p * p, 0.0);
    }
    if ws.r_inv_cm.len() != p * p {
        ws.r_inv_cm.resize(p * p, 0.0);
    }
    for row in 0..p {
        let r_dst = &mut ws.r_rm[row * p..row * p + p];
        for col in row..p {
            r_dst[col] = ws.a_cm[col * m + row];
        }
    }

    ws.r_inv_cm.fill(0.0);
    for col in 0..p {
        let inv_col_base = col * p;
        ws.r_inv_cm[inv_col_base + col] = 1.0 / ws.r_rm[col * p + col];
        for row in (0..col).rev() {
            let r_row = &ws.r_rm[(row * p + row + 1)..=(row * p + col)];
            let inv_col = &ws.r_inv_cm[(inv_col_base + row + 1)..=(inv_col_base + col)];
            let mut sum = 0.0;
            for (&rik, &yk) in r_row.iter().zip(inv_col.iter()) {
                sum -= rik * yk;
            }
            ws.r_inv_cm[inv_col_base + row] = sum / ws.r_rm[row * p + row];
        }
    }

    let mut xtwx_inv = vec![0.0; p * p];
    for k in 0..p {
        let c_k = &ws.r_inv_cm[(k * p)..=(k * p + k)];
        for i in 0..=k {
            let ci = c_k[i];
            let dst = &mut xtwx_inv[(i * p + i)..=(i * p + k)];
            for (d, &cj) in dst.iter_mut().zip(c_k[i..].iter()) {
                *d += ci * cj;
            }
        }
    }
    for i in 0..p {
        for j in (i + 1)..p {
            xtwx_inv[j * p + i] = xtwx_inv[i * p + j];
        }
    }
    Ok(xtwx_inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linalg_rejects_mismatched_and_non_finite_inputs() {
        let mut y = [0.0; 1];
        assert!(solve_lower_into(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0, 2.0], &mut y).is_err());
        assert!(cholesky_invert(&[1.0, f64::NAN, f64::NAN, 1.0], 2).is_err());
        assert!(cholesky_solve(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0]).is_err());
    }

    #[test]
    fn cholesky_solve_matches_hand_solution() {
        // [4 2; 2 3] x = [2; 1] -> x = [0.5, 0].
        let x = cholesky_solve(&[4.0, 2.0, 2.0, 3.0], 2, &[2.0, 1.0]).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-15 && x[1].abs() < 1e-15, "{x:?}");
    }
}
