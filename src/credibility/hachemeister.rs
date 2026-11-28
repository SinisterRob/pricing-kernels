//! Hachemeister (1975) Regression Credibility for credibility-weighted trend estimation
//! across commercial states, classes, or risk groups.
//!
//! Reference: C. A. Hachemeister (1975), *"Credibility for Regression Models with Application
//! to Trend"*, in *Credibility: Theory and Applications* (ed. P. M. Kahn), Academic Press,
//! pp. 129–163; CAS Exam 8 syllabus.
//!
//! # Time-origin invariance (centred parameterization)
//!
//! The PSD projection of the VHM matrix `A` (eigenvalue clamping) is not invariant under the
//! affine reparameterization `t -> t - c`, so estimating in the user's raw time origin makes the
//! credibility estimates depend on whether time is coded `1..5`, `2021..2025` or `-2..2`.
//! Following Bühlmann & Gisler (2005), *A Course in Credibility Theory*, §8.4 (choice of the
//! design origin at the barycentre), all structural parameters, the PSD projection and the
//! credibility matrices are computed in **centred coordinates** `t_c = t - t_bar`, where
//! `t_bar = sum_ij w_ij t_ij / sum_ij w_ij` is the exposure-weighted collective time centre
//! (reported as [`HachemeisterResult::time_centre`]). Results are then mapped back to the
//! user's original time coordinates with `T = [[1, -t_bar], [0, 1]]`
//! (`beta = T beta_c`, `A = T A_c T^T`, `Z_i = T Z_{c,i} T^-1`), so predictions
//! `beta_0 + beta_1 t` are invariant (up to floating-point rounding) to any shift of the
//! time origin.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Per-group linear regression credibility result `(intercept, slope)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HachemeisterGroupResult {
    /// Standalone weighted least squares `[intercept, slope]` for this group.
    pub wls_beta: [f64; 2],
    /// Credibility-blended `[intercept, slope]` shrinking `wls_beta` toward collective prior `b`.
    pub credibility_beta: [f64; 2],
    /// Row-major `2x2` credibility matrix `Z_i = A (A + s^2 (X_i^T W_i X_i)^-1)^-1`.
    pub credibility_matrix: [f64; 4],
}

impl HachemeisterGroupResult {
    /// Evaluate the credibility-blended linear trend prediction `beta_0 + beta_1 * t`.
    #[inline]
    pub fn predict(&self, t: f64) -> f64 {
        self.credibility_beta[0] + self.credibility_beta[1] * t
    }
}

/// Output of Hachemeister (1975) regression credibility across `K` risk groups.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HachemeisterResult {
    /// Expected Process Variance (`s^2`): pooled unbiased weighted residual variance.
    pub epv: f64,
    /// Exposure-weighted collective time centre `t_bar = sum_ij w_ij t_ij / sum_ij w_ij`
    /// (original time units) at which the estimation and PSD projection are performed.
    pub time_centre: f64,
    /// Collective prior regression vector `[intercept, slope]` (`b`), the GLS estimator
    /// `b = (sum_i (A + s^2 V_i)^-1)^-1 sum_i (A + s^2 V_i)^-1 beta_i` with
    /// `V_i = (X_i^T W_i X_i)^-1`. It equals Hachemeister's credibility-weighted
    /// `(sum_i Z_i)^-1 sum_i Z_i beta_i` whenever `A` is full rank, and stays well defined (and
    /// continuous in `A`) when `A` is rank-deficient, including `A = 0` (plain pooled GLS).
    pub collective_beta: [f64; 2],
    /// Row-major `2x2` Variance of Hypothetical Means covariance matrix `A` (PSD-projected),
    /// estimated by the exposure-weighted multivariate Bühlmann-Straub moment estimator,
    /// unbiased (before projection) for arbitrary per-group designs. The projection is done in
    /// centred coordinates (see module docs); the matrix is reported in the original time
    /// coordinates as `T A_c T^T` (still PSD).
    pub vhm_matrix: [f64; 4],
    /// Row-major `2x2` unbiased VHM moment estimate **before** PSD projection, in original
    /// time coordinates (may be indefinite in small samples; exposed for diagnostics and
    /// unbiasedness checks).
    pub vhm_matrix_raw: [f64; 4],
    /// Per-group standalone WLS and credibility-blended regression estimates.
    pub groups: Vec<HachemeisterGroupResult>,
}

#[inline]
fn invert_2x2(m: [f64; 4]) -> Result<[f64; 4], PricingError> {
    let det = m[0] * m[3] - m[1] * m[2];
    let scale = (m[0] * m[3]).abs().max((m[1] * m[2]).abs());
    if !det.is_finite() || det.abs() <= 1e-14 * scale.max(1e-30) {
        return Err(PricingError::SingularMatrix(
            "2x2 normal matrix is singular in Hachemeister regression credibility",
        ));
    }
    let inv_det = 1.0 / det;
    Ok([
        m[3] * inv_det,
        -m[1] * inv_det,
        -m[2] * inv_det,
        m[0] * inv_det,
    ])
}

#[inline]
fn mat_mul_2x2(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Change-of-origin matrix `T = [[1, -t_bar], [0, 1]]` mapping centred `[intercept at t_bar,
/// slope]` to original-coordinate `[intercept at t = 0, slope]`.
#[inline]
fn origin_transform(t_bar: f64) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let t = [1.0, -t_bar, 0.0, 1.0];
    let t_inv = [1.0, t_bar, 0.0, 1.0];
    let t_tr = [1.0, 0.0, -t_bar, 1.0];
    (t, t_inv, t_tr)
}

#[inline]
fn mat_vec_2x2(m: [f64; 4], v: [f64; 2]) -> [f64; 2] {
    [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]
}

/// Project a symmetric `2x2` matrix onto the positive semi-definite cone by clamping
/// negative eigenvalues to `0.0`.
fn project_psd_2x2(a: f64, b: f64, d: f64) -> [f64; 4] {
    let tr = a + d;
    let diff = a - d;
    let disc = diff.hypot(2.0 * b);
    let l1 = f64::midpoint(tr, disc);
    let l2 = 0.5 * (tr - disc);
    if l2 >= 0.0 {
        return [a, b, b, d];
    }
    if l1 <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    // Reconstruct rank-1 PSD matrix l1 * v1 v1^T
    let (vx, vy) = if b.abs() > 1e-15 {
        let x = l1 - d;
        let y = b;
        let norm = x.hypot(y);
        (x / norm, y / norm)
    } else if a >= d {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    [l1 * vx * vx, l1 * vx * vy, l1 * vx * vy, l1 * vy * vy]
}

/// Estimate Hachemeister (1975) linear regression credibility across `K >= 2` risk groups.
///
/// Each group `i` provides matching slices `(times[i], values[i], weights[i])` with at least
/// 3 observations (or at least 2 distinct time points and total `sum(n_i - 2) >= 1` for EPV).
/// Models `y_{ij} = beta_{i,0} + beta_{i,1} * t_{ij} + eps_{ij}` with `Var(eps_{ij}) = s^2 / w_{ij}`.
///
/// # Structural parameters
/// - `s^2`: pooled weighted residual variance `sum_ij w_ij r_ij^2 / sum_i (n_i - 2)` (unbiased).
/// - `A`: with group exposures `w_i = sum_j w_ij`, `w = sum_i w_i`, `a_i = w_i / w`,
///   `beta_bar = sum_i a_i beta_i`, `V_i = (X_i^T W_i X_i)^-1`:
///   `A_hat = [sum_i w_i (beta_i - beta_bar)(beta_i - beta_bar)^T - s^2 sum_i w_i (1 - a_i) V_i]
///   / (w (1 - sum_i a_i^2))`, unbiased for any designs (multivariate analogue of the
///   Bühlmann-Straub VHM estimator; cf. Bühlmann & Gisler (2005) §8.4, Dannenburg, Kaas &
///   Goovaerts (1996)), then projected onto the PSD cone.
/// - `Z_i = A (A + s^2 V_i)^-1` (Hachemeister 1975) and the collective
///   `b = (sum_i (A + s^2 V_i)^-1)^-1 sum_i (A + s^2 V_i)^-1 beta_i`, the GLS estimator of the
///   common mean of `beta_i ~ (b, A + s^2 V_i)`. Since `Z_i = A (A + s^2 V_i)^-1`, this equals
///   `(sum_i Z_i)^-1 sum_i Z_i beta_i` whenever `A` is invertible; unlike that form it remains
///   defined for a rank-deficient (PSD-projected) `A`, which is common in small panels.
///   Degenerate case `s^2 = 0` (all groups fit exactly): `Z_i = I` and `b` is the unweighted
///   mean of the `beta_i` (the `sum_i Z_i` form with `Z_i = I`); it does not affect the
///   credibility estimates, which then equal the standalone fits.
///
/// All of the above is evaluated in centred time coordinates `t - t_bar` (exposure-weighted
/// collective barycentre, Bühlmann & Gisler (2005) §8.4) so that the PSD projection — and hence
/// every prediction — is invariant to the time origin; reported coefficients and matrices are
/// mapped back to the caller's original time coordinates (see the module docs).
///
/// Trade-off: the scalar exposure weights `w_i` give a closed-form, non-iterative unbiased
/// estimator; the matrix-weighted (`X_i^T W_i X_i`) pseudo-estimator of Bühlmann-Gisler is more
/// efficient for very heterogeneous designs but requires a fixed-point iteration.
pub fn hachemeister_credibility(
    groups_times: &[&[f64]],
    groups_values: &[&[f64]],
    groups_weights: &[&[f64]],
) -> Result<HachemeisterResult, PricingError> {
    let k = groups_times.len();
    if k < 2 {
        return Err(PricingError::InvalidParameter(
            "Hachemeister credibility requires at least 2 groups",
        ));
    }
    if groups_values.len() != k || groups_weights.len() != k {
        return Err(PricingError::DimensionMismatch(
            "groups_times, groups_values, and groups_weights must have equal length",
        ));
    }

    // Pass 1: validate inputs and compute the exposure-weighted collective time centre.
    let mut sum_w = 0.0;
    let mut sum_wt = 0.0;
    for i in 0..k {
        let ts = groups_times[i];
        let ys = groups_values[i];
        let ws = groups_weights[i];
        let n = ts.len();
        if n < 2 || ys.len() != n || ws.len() != n {
            return Err(PricingError::DimensionMismatch(
                "each group must have matching slices with at least 2 observations",
            ));
        }
        for j in 0..n {
            let (t, y, w) = (ts[j], ys[j], ws[j]);
            if !t.is_finite() || !y.is_finite() || !w.is_finite() || !(w > 0.0) {
                return Err(PricingError::InvalidParameter(
                    "times, values must be finite and weights must be finite and positive",
                ));
            }
            sum_w += w;
            sum_wt += w * t;
        }
    }
    let time_centre = sum_wt / sum_w;
    if !time_centre.is_finite() {
        return Err(PricingError::InvalidParameter(
            "Hachemeister collective time centre is not finite",
        ));
    }

    // Pass 2: everything below is in centred coordinates t_c = t - time_centre.
    let mut xtwx_list = Vec::with_capacity(k);
    let mut xtwx_inv_list = Vec::with_capacity(k);
    let mut wls_betas = Vec::with_capacity(k);
    let mut total_sse = 0.0;
    let mut total_df = 0usize;

    for i in 0..k {
        let ts = groups_times[i];
        let ys = groups_values[i];
        let ws = groups_weights[i];
        let n = ts.len();

        let mut s00 = 0.0;
        let mut s01 = 0.0;
        let mut s11 = 0.0;
        let mut r0 = 0.0;
        let mut r1 = 0.0;

        for j in 0..n {
            let t = ts[j] - time_centre;
            let y = ys[j];
            let w = ws[j];
            s00 += w;
            s01 += w * t;
            s11 += w * t * t;
            r0 += w * y;
            r1 += w * t * y;
        }

        let xtwx = [s00, s01, s01, s11];
        let inv = invert_2x2(xtwx)?;
        let beta = mat_vec_2x2(inv, [r0, r1]);

        for j in 0..n {
            let resid = ys[j] - (beta[0] + beta[1] * (ts[j] - time_centre));
            total_sse += ws[j] * resid * resid;
        }
        total_df += n.saturating_sub(2);
        xtwx_list.push(xtwx);
        xtwx_inv_list.push(inv);
        wls_betas.push(beta);
    }

    if total_df == 0 {
        return Err(PricingError::InvalidParameter(
            "need at least one group with >= 3 observations to estimate EPV",
        ));
    }

    let epv = (total_sse / (total_df as f64)).max(0.0);

    // Unbiased VHM covariance matrix estimator for arbitrary per-group designs.
    //
    // Model: beta_i = b + u_i + e_i with Cov(u_i) = A, Cov(e_i) = s^2 V_i, V_i = (X_i^T W_i X_i)^-1.
    // With scalar group exposures w_i = sum_j w_ij, w = sum_i w_i, a_i = w_i / w and
    // beta_bar_w = sum_i a_i beta_i, the multivariate Bühlmann-Straub moment identity is
    //   E[ sum_i w_i (beta_i - beta_bar_w)(beta_i - beta_bar_w)^T ]
    //     = c * A + s^2 * sum_i w_i (1 - a_i) V_i,     c = w * (1 - sum_i a_i^2),
    // so A_hat = (S_w - s_hat^2 * sum_i w_i (1 - a_i) V_i) / c is unbiased for any designs
    // (it reduces exactly to the scalar Bühlmann-Straub VHM estimator for X_i = 1).
    let total_exposure: f64 = xtwx_list.iter().map(|m| m[0]).sum();
    let mut beta_bar = [0.0; 2];
    let mut sum_rel_sq = 0.0;
    for i in 0..k {
        let a_i = xtwx_list[i][0] / total_exposure;
        beta_bar[0] += a_i * wls_betas[i][0];
        beta_bar[1] += a_i * wls_betas[i][1];
        sum_rel_sq += a_i * a_i;
    }
    let c = total_exposure * (1.0 - sum_rel_sq);
    if !c.is_finite() || !(c > 0.0) {
        return Err(PricingError::InvalidParameter(
            "Hachemeister VHM normalizer is non-positive; group exposures are degenerate",
        ));
    }
    let mut scatter = [0.0; 3];
    let mut v_bar = [0.0; 3];
    for i in 0..k {
        let w_i = xtwx_list[i][0];
        let a_i = w_i / total_exposure;
        let d0 = wls_betas[i][0] - beta_bar[0];
        let d1 = wls_betas[i][1] - beta_bar[1];
        scatter[0] += w_i * d0 * d0;
        scatter[1] += w_i * d0 * d1;
        scatter[2] += w_i * d1 * d1;
        let inv = xtwx_inv_list[i];
        let f = w_i * (1.0 - a_i);
        v_bar[0] += f * inv[0];
        v_bar[1] += f * f64::midpoint(inv[1], inv[2]);
        v_bar[2] += f * inv[3];
    }
    let a00 = (scatter[0] - epv * v_bar[0]) / c;
    let a01 = (scatter[1] - epv * v_bar[1]) / c;
    let a11 = (scatter[2] - epv * v_bar[2]) / c;
    let vhm = project_psd_2x2(a00, a01, a11);

    let vhm_is_zero = vhm.iter().all(|&v| v == 0.0);
    let mut z_mats = Vec::with_capacity(k);
    for i in 0..k {
        let z_mat = if epv == 0.0 {
            [1.0, 0.0, 0.0, 1.0]
        } else if vhm_is_zero {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            let cov_total = [
                vhm[0] + epv * xtwx_inv_list[i][0],
                vhm[1] + epv * xtwx_inv_list[i][1],
                vhm[2] + epv * xtwx_inv_list[i][2],
                vhm[3] + epv * xtwx_inv_list[i][3],
            ];
            let inv_total = invert_2x2(cov_total)?;
            mat_mul_2x2(vhm, inv_total)
        };
        z_mats.push(z_mat);
    }

    // Collective prior: GLS estimator of the common mean of beta_i ~ (b, A + s^2 V_i),
    //   b = (sum_i Omega_i^-1)^-1 sum_i Omega_i^-1 beta_i,   Omega_i = A + s^2 V_i.
    // Because Z_i = A Omega_i^-1, this coincides with Hachemeister's (sum Z_i)^-1 sum Z_i beta_i
    // when A is invertible, and it is the continuous extension to rank-deficient A (A = 0 gives
    // the pooled GLS estimate). For s^2 > 0 each Omega_i is positive definite (V_i is).
    let collective_beta = if epv == 0.0 {
        // Z_i = I for every group: (sum Z_i)^-1 sum Z_i beta_i is the unweighted mean.
        let inv_k = 1.0 / (k as f64);
        let mut mean = [0.0; 2];
        for beta in &wls_betas {
            mean[0] += inv_k * beta[0];
            mean[1] += inv_k * beta[1];
        }
        mean
    } else {
        let mut sum_prec = [0.0; 4];
        let mut sum_prec_b = [0.0; 2];
        for i in 0..k {
            let omega = [
                vhm[0] + epv * xtwx_inv_list[i][0],
                vhm[1] + epv * xtwx_inv_list[i][1],
                vhm[2] + epv * xtwx_inv_list[i][2],
                vhm[3] + epv * xtwx_inv_list[i][3],
            ];
            let prec = invert_2x2(omega)?;
            for idx in 0..4 {
                sum_prec[idx] += prec[idx];
            }
            let pb = mat_vec_2x2(prec, wls_betas[i]);
            sum_prec_b[0] += pb[0];
            sum_prec_b[1] += pb[1];
        }
        mat_vec_2x2(invert_2x2(sum_prec)?, sum_prec_b)
    };

    // Map centred-coordinate results back to the user's original time coordinates.
    let (t_mat, t_inv, t_tr) = origin_transform(time_centre);
    let mut groups = Vec::with_capacity(k);
    for i in 0..k {
        let z_mat = z_mats[i];
        let diff = [
            wls_betas[i][0] - collective_beta[0],
            wls_betas[i][1] - collective_beta[1],
        ];
        let adj = mat_vec_2x2(z_mat, diff);
        let cred_beta = [collective_beta[0] + adj[0], collective_beta[1] + adj[1]];

        groups.push(HachemeisterGroupResult {
            wls_beta: mat_vec_2x2(t_mat, wls_betas[i]),
            credibility_beta: mat_vec_2x2(t_mat, cred_beta),
            credibility_matrix: mat_mul_2x2(mat_mul_2x2(t_mat, z_mat), t_inv),
        });
    }

    Ok(HachemeisterResult {
        epv,
        time_centre,
        collective_beta: mat_vec_2x2(t_mat, collective_beta),
        vhm_matrix: mat_mul_2x2(mat_mul_2x2(t_mat, vhm), t_tr),
        vhm_matrix_raw: mat_mul_2x2(mat_mul_2x2(t_mat, [a00, a01, a01, a11]), t_tr),
        groups,
    })
}
