use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Result of non-parametric Bühlmann-Straub credibility estimation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuhlmannStraubResult {
    /// Credibility parameter K = s^2 / a = Expected Process Variance / Variance of Hypothetical Means.
    pub k: f64,
    /// Expected value of process variance (s^2 or EPV).
    pub epv: f64,
    /// Variance of hypothetical means (a or VHM).
    pub vhm: f64,
    /// Collective overall grand mean (mu).
    pub grand_mean: f64,
    /// Credibility weights Z_i = P_i / (P_i + K) for each group/risk.
    pub credibilities: Vec<f64>,
    /// Blended credibility estimates for each group: Z_i * X_bar_i + (1 - Z_i) * mu.
    pub blended_estimates: Vec<f64>,
}

/// Computes Bühlmann-Straub credibility for heterogeneous risk groups with non-uniform exposures.
///
/// Ref: Bühlmann & Straub (1970) / CAS Exam 8 syllabus.
///
/// - `groups_losses`: Slice of loss rates / loss ratios $X_{ij}$ for each group $i$ and period $j$.
/// - `groups_weights`: Slice of exposures / weights $P_{ij}$ corresponding to each observation.
/// - `periods_per_group`: Number of observation periods for each group.
pub fn buhlmann_straub(
    groups_losses: &[&[f64]],
    groups_weights: &[&[f64]],
) -> Result<BuhlmannStraubResult, PricingError> {
    let n_groups = groups_losses.len();
    if n_groups < 2 {
        return Err(PricingError::InvalidParameter("at least 2 groups required for Bühlmann-Straub"));
    }
    if groups_weights.len() != n_groups {
        return Err(PricingError::DimensionMismatch("weights slice length != groups length"));
    }

    let mut total_periods = 0;
    let mut group_total_weights = vec![0.0; n_groups];
    let mut group_weighted_means = vec![0.0; n_groups];

    for i in 0..n_groups {
        let losses = groups_losses[i];
        let weights = groups_weights[i];
        if losses.len() != weights.len() || losses.is_empty() {
            return Err(PricingError::DimensionMismatch("each group must have non-empty matching losses and weights"));
        }
        total_periods += losses.len();
        let mut sum_p = 0.0;
        let mut sum_px = 0.0;
        for (&x, &p) in losses.iter().zip(weights.iter()) {
            if p <= 0.0 {
                return Err(PricingError::NonPositiveValue("weights must be strictly positive"));
            }
            sum_p += p;
            sum_px += p * x;
        }
        group_total_weights[i] = sum_p;
        group_weighted_means[i] = sum_px / sum_p;
    }

    if total_periods <= n_groups {
        return Err(PricingError::InvalidParameter("total observations must exceed number of groups"));
    }

    // 1. Calculate Expected Process Variance (s^2):
    // s^2 = (1 / sum_i (n_i - 1)) * sum_i sum_j P_ij * (X_ij - X_bar_i)^2
    let mut s2_num = 0.0;
    for i in 0..n_groups {
        let losses = groups_losses[i];
        let weights = groups_weights[i];
        let x_bar = group_weighted_means[i];
        for (&x, &p) in losses.iter().zip(weights.iter()) {
            let diff = x - x_bar;
            s2_num += p * diff * diff;
        }
    }
    let df_epv = total_periods - n_groups;
    let epv = (s2_num / df_epv as f64).max(1e-12);

    // 2. Calculate Grand Mean (mu) and Variance of Hypothetical Means (a / VHM):
    let total_weight: f64 = group_total_weights.iter().sum();
    let mu_unweighted: f64 = group_weighted_means.iter().zip(group_total_weights.iter())
        .map(|(&m, &w)| m * w)
        .sum::<f64>() / total_weight;

    // m* calculation:
    // c = total_weight - (sum_i P_i^2) / total_weight
    let sum_p_sq: f64 = group_total_weights.iter().map(|&w| w * w).sum();
    let c = total_weight - (sum_p_sq / total_weight);

    // Unbiased estimator of VHM:
    // a = (1 / c) * [ sum_i P_i * (X_bar_i - mu)^2 - (n_groups - 1) * s^2 ]
    let mut vhm_num = 0.0;
    for i in 0..n_groups {
        let p_i = group_total_weights[i];
        let diff = group_weighted_means[i] - mu_unweighted;
        vhm_num += p_i * diff * diff;
    }
    let a_raw = (vhm_num - ((n_groups - 1) as f64) * epv) / c;
    let vhm = a_raw.max(0.0);

    // 3. Credibility K and Z_i
    let k = if vhm > 0.0 { epv / vhm } else { f64::INFINITY };

    let mut credibilities = vec![0.0; n_groups];
    let mut blended_estimates = vec![0.0; n_groups];

    // Iteratively re-estimate mu with credibility weights if K is finite
    let mu = if vhm > 0.0 {
        let mut sum_z = 0.0;
        let mut sum_zx = 0.0;
        for i in 0..n_groups {
            let p_i = group_total_weights[i];
            let z_i = p_i / (p_i + k);
            credibilities[i] = z_i;
            sum_z += z_i;
            sum_zx += z_i * group_weighted_means[i];
        }
        if sum_z > 0.0 { sum_zx / sum_z } else { mu_unweighted }
    } else {
        mu_unweighted
    };

    for i in 0..n_groups {
        let z_i = credibilities[i];
        blended_estimates[i] = z_i * group_weighted_means[i] + (1.0 - z_i) * mu;
    }

    Ok(BuhlmannStraubResult {
        k,
        epv,
        vhm,
        grand_mean: mu,
        credibilities,
        blended_estimates,
    })
}
