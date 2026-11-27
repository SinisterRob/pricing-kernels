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
    /// Collective prior regression vector `[intercept, slope]` (`b`).
    pub collective_beta: [f64; 2],
    /// Row-major `2x2` Variance of Hypothetical Means covariance matrix `A` (PSD-projected).
    pub vhm_matrix: [f64; 4],
    /// Per-group standalone WLS and credibility-blended regression estimates.
    pub groups: Vec<HachemeisterGroupResult>,
}

#[inline]
fn invert_2x2(m: [f64; 4]) -> Result<[f64; 4], PricingError> {
    let det = m[0] * m[3] - m[1] * m[2];
    if !det.is_finite() || det.abs() <= 1e-18 {
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

#[inline]
fn mat_vec_2x2(m: [f64; 4], v: [f64; 2]) -> [f64; 2] {
    [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]
}

/// Project a symmetric `2x2` matrix onto the positive semi-definite cone by clamping
/// negative eigenvalues to `0.0`.
fn project_psd_2x2(a: f64, b: f64, d: f64) -> [f64; 4] {
    let tr = a + d;
    let diff = a - d;
    let disc = (diff * diff + 4.0 * b * b).sqrt();
    let l1 = 0.5 * (tr + disc);
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

    // Collective prior b = (sum_i X_i^T W_i X_i)^-1 * sum_i (X_i^T W_i X_i beta_i)
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
    let sum_m_inv = invert_2x2(sum_m)?;
    let collective_beta = mat_vec_2x2(sum_m_inv, sum_mb);

    // Unbiased Hachemeister / Bichsel-Straub VHM covariance matrix estimator:
    // A_hat = (1 / (K - 1)) * sum_i (beta_i - b)(beta_i - b)^T - (s^2 / K) * sum_i (X_i^T W_i X_i)^-1
    let mut emp_cov = [0.0; 4];
    let mut mean_inv = [0.0; 4];
    let kf = k as f64;
    for i in 0..k {
        let d0 = wls_betas[i][0] - collective_beta[0];
        let d1 = wls_betas[i][1] - collective_beta[1];
        emp_cov[0] += d0 * d0;
        emp_cov[1] += d0 * d1;
        emp_cov[2] += d1 * d0;
        emp_cov[3] += d1 * d1;
        for idx in 0..4 {
            mean_inv[idx] += xtwx_inv_list[i][idx] / kf;
        }
    }
    let denom = (k - 1) as f64;
    let a00 = emp_cov[0] / denom - epv * mean_inv[0];
    let a01 = 0.5 * (emp_cov[1] + emp_cov[2]) / denom - epv * 0.5 * (mean_inv[1] + mean_inv[2]);
    let a11 = emp_cov[3] / denom - epv * mean_inv[3];
    let vhm = project_psd_2x2(a00, a01, a11);

    let mut groups = Vec::with_capacity(k);
    for i in 0..k {
        let z_mat = if epv == 0.0 {
            [1.0, 0.0, 0.0, 1.0]
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
        groups,
    })
}
