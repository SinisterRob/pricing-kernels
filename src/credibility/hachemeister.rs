//! Hachemeister (1975) Regression Credibility for credibility-weighted trend estimation
//! across commercial states, classes, or risk groups.
//!
//! Reference: C. A. Hachemeister (1975), *"Credibility for Regression Models with Application
//! to Trend"*, in *Credibility: Theory and Applications* (ed. P. M. Kahn), Academic Press,
//! pp. 129–163; CAS Exam 8 syllabus.

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
    /// Collective prior regression vector `[intercept, slope]` (`b`), credibility-weighted
    /// `b = (sum_i Z_i)^-1 sum_i Z_i beta_i` (GLS-weighted fallback when `sum_i Z_i` is singular).
    pub collective_beta: [f64; 2],
    /// Row-major `2x2` Variance of Hypothetical Means covariance matrix `A` (PSD-projected),
    /// estimated by the exposure-weighted multivariate Bühlmann-Straub moment estimator,
    /// unbiased (before projection) for arbitrary per-group designs.
    pub vhm_matrix: [f64; 4],
    /// Row-major `2x2` unbiased VHM moment estimate **before** PSD projection (may be
    /// indefinite in small samples; exposed for diagnostics and unbiasedness checks).
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

/// Relative determinant threshold below which `sum_i Z_i` is treated as rank-deficient and the
/// collective prior falls back to the GLS (`A -> 0`) limit.
const SUM_Z_RANK_TOL: f64 = 1e-10;

/// Invert a `2x2` matrix only when it is well conditioned relative to [`SUM_Z_RANK_TOL`].
fn invert_well_conditioned_2x2(m: [f64; 4]) -> Option<[f64; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    let scale = (m[0] * m[3]).abs().max((m[1] * m[2]).abs());
    if !det.is_finite() || !(scale > 0.0) || det.abs() <= SUM_Z_RANK_TOL * scale {
        return None;
    }
    invert_2x2(m).ok()
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
/// - `b = (sum_i Z_i)^-1 sum_i Z_i beta_i` with `Z_i = A (A + s^2 V_i)^-1` (Hachemeister 1975).
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
        if n < 2 || ys.len() != n || ws.len() != n {
            return Err(PricingError::DimensionMismatch(
                "each group must have matching slices with at least 2 observations",
            ));
        }

        let mut s00 = 0.0;
        let mut s01 = 0.0;
        let mut s11 = 0.0;
        let mut r0 = 0.0;
        let mut r1 = 0.0;

        for j in 0..n {
            let t = ts[j];
            let y = ys[j];
            let w = ws[j];
            if !t.is_finite() || !y.is_finite() || !w.is_finite() || !(w > 0.0) {
                return Err(PricingError::InvalidParameter(
                    "times, values must be finite and weights must be finite and positive",
                ));
            }
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
            let resid = ys[j] - (beta[0] + beta[1] * ts[j]);
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

    // Hachemeister (1975) collective prior is credibility-weighted:
    // b = (sum_i Z_i)^-1 sum_i Z_i beta_i (cf. Bühlmann-Straub mu = sum Z_i X_i / sum Z_i).
    // When sum_i Z_i is (numerically) singular (A = 0 or rank-deficient A), fall back to the
    // GLS limit b = (sum_i X_i^T W_i X_i)^-1 sum_i X_i^T W_i X_i beta_i, which is the A -> 0 limit
    // of the credibility-weighted estimator.
    let mut sum_z = [0.0; 4];
    let mut sum_zb = [0.0; 2];
    for i in 0..k {
        for idx in 0..4 {
            sum_z[idx] += z_mats[i][idx];
        }
        let zb = mat_vec_2x2(z_mats[i], wls_betas[i]);
        sum_zb[0] += zb[0];
        sum_zb[1] += zb[1];
    }
    let collective_beta = if let Some(sum_z_inv) = invert_well_conditioned_2x2(sum_z) {
        mat_vec_2x2(sum_z_inv, sum_zb)
    } else {
        let mut sum_m = [0.0; 4];
        let mut sum_mb = [0.0; 2];
        for i in 0..k {
            for idx in 0..4 {
                sum_m[idx] += xtwx_list[i][idx];
            }
            let mb = mat_vec_2x2(xtwx_list[i], wls_betas[i]);
            sum_mb[0] += mb[0];
            sum_mb[1] += mb[1];
        }
        mat_vec_2x2(invert_2x2(sum_m)?, sum_mb)
    };

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
            wls_beta: wls_betas[i],
            credibility_beta: cred_beta,
            credibility_matrix: z_mat,
        });
    }

    Ok(HachemeisterResult {
        epv,
        collective_beta,
        vhm_matrix: vhm,
        vhm_matrix_raw: [a00, a01, a01, a11],
        groups,
    })
}
