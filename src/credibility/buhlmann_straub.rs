//! Non-parametric Bühlmann–Straub (1970) empirical Bayes credibility for heterogeneous risk groups.

use super::hachemeister::{HachemeisterResult, hachemeister_credibility};
use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Single risk group's longitudinal experience for credibility modeling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredibilityGroup {
    /// Optional human-readable label (e.g., state, class code, or account name).
    pub label: Option<String>,
    /// Per-period loss rates, loss ratios, or pure premiums `X_{ij}`.
    pub losses: Vec<f64>,
    /// Per-period strictly positive exposures or weights `P_{ij}`.
    pub weights: Vec<f64>,
    /// Optional time coordinates `t_{ij}` for Hachemeister regression credibility
    /// (defaults to `1.0, 2.0, ..., n_i` if omitted).
    pub times: Option<Vec<f64>>,
}

/// Typed multi-group experience panel for Bühlmann–Straub and Hachemeister credibility.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CredibilityPanel {
    groups: Vec<CredibilityGroup>,
}

impl CredibilityPanel {
    /// Create an empty [`CredibilityPanel`].
    pub const fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Append a validated risk group with matching `losses` and strictly positive `weights`.
    pub fn add_group(
        &mut self,
        label: impl Into<String>,
        losses: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<&mut Self, PricingError> {
        if losses.is_empty() || losses.len() != weights.len() {
            return Err(PricingError::DimensionMismatch(
                "group losses and weights must be non-empty and have matching lengths",
            ));
        }
        for (&x, &p) in losses.iter().zip(weights.iter()) {
            if !x.is_finite() || !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "group losses and weights must be finite",
                ));
            }
            if !(p > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "group weights must be strictly positive",
                ));
            }
        }
        self.groups.push(CredibilityGroup {
            label: Some(label.into()),
            losses,
            weights,
            times: None,
        });
        Ok(self)
    }

    /// Append a validated risk group with explicit time coordinates for Hachemeister trend credibility.
    pub fn add_trend_group(
        &mut self,
        label: impl Into<String>,
        times: Vec<f64>,
        losses: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<&mut Self, PricingError> {
        if losses.is_empty() || losses.len() != weights.len() || losses.len() != times.len() {
            return Err(PricingError::DimensionMismatch(
                "group times, losses, and weights must be non-empty and have matching lengths",
            ));
        }
        for ((&t, &x), &p) in times.iter().zip(losses.iter()).zip(weights.iter()) {
            if !t.is_finite() || !x.is_finite() || !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "group times, losses, and weights must be finite",
                ));
            }
            if !(p > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "group weights must be strictly positive",
                ));
            }
        }
        self.groups.push(CredibilityGroup {
            label: Some(label.into()),
            losses,
            weights,
            times: Some(times),
        });
        Ok(self)
    }

    /// Number of risk groups in the panel.
    #[inline]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Returns `true` if the panel has no groups.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Slice of underlying [`CredibilityGroup`] records.
    #[inline]
    pub fn groups(&self) -> &[CredibilityGroup] {
        &self.groups
    }

    /// Fit non-parametric Bühlmann–Straub credibility across all groups in this panel.
    pub fn fit_buhlmann_straub(&self) -> Result<BuhlmannStraubResult, PricingError> {
        let loss_slices: Vec<&[f64]> = self.groups.iter().map(|g| g.losses.as_slice()).collect();
        let weight_slices: Vec<&[f64]> = self.groups.iter().map(|g| g.weights.as_slice()).collect();
        buhlmann_straub(&loss_slices, &weight_slices)
    }

    /// Fit Hachemeister (1975) regression trend credibility across all groups in this panel.
    ///
    /// Groups without explicit `times` use sequential periods `1.0, 2.0, ..., n_i`.
    pub fn fit_hachemeister(&self) -> Result<HachemeisterResult, PricingError> {
        let owned_times: Vec<Vec<f64>> = self
            .groups
            .iter()
            .map(|g| {
                g.times
                    .clone()
                    .unwrap_or_else(|| (1..=g.losses.len()).map(|t| t as f64).collect())
            })
            .collect();
        let time_slices: Vec<&[f64]> = owned_times.iter().map(Vec::as_slice).collect();
        let loss_slices: Vec<&[f64]> = self.groups.iter().map(|g| g.losses.as_slice()).collect();
        let weight_slices: Vec<&[f64]> = self.groups.iter().map(|g| g.weights.as_slice()).collect();
        hachemeister_credibility(&time_slices, &loss_slices, &weight_slices)
    }
}

/// Result of non-parametric Bühlmann-Straub credibility estimation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuhlmannStraubResult {
    /// Credibility parameter `K = s^2 / a = EPV / VHM`.
    pub k: f64,
    /// Expected value of process variance (`s^2` or EPV).
    pub epv: f64,
    /// Variance of hypothetical means (`a` or VHM).
    pub vhm: f64,
    /// Collective overall grand mean (`mu`), credibility-weighted when `VHM > 0`.
    pub grand_mean: f64,
    /// Credibility weights `Z_i = P_i / (P_i + K)` for each group/risk.
    pub credibilities: Vec<f64>,
    /// Blended credibility estimates for each group: `Z_i * X_bar_i + (1 - Z_i) * mu`.
    pub blended_estimates: Vec<f64>,
}

/// Computes Bühlmann-Straub credibility for heterogeneous risk groups with non-uniform exposures.
///
/// Ref: Bühlmann & Straub (1970) / Klugman, Panjer & Willmot / CAS Exam 8 syllabus.
///
/// - `groups_losses`: Slice of per-period loss rates / loss ratios `X_{ij}` for each group `i`.
/// - `groups_weights`: Slice of strictly positive exposures / weights `P_{ij}` for each observation.
pub fn buhlmann_straub(
    groups_losses: &[&[f64]],
    groups_weights: &[&[f64]],
) -> Result<BuhlmannStraubResult, PricingError> {
    let n_groups = groups_losses.len();
    if n_groups < 2 {
        return Err(PricingError::InvalidParameter(
            "at least 2 groups required for Bühlmann-Straub",
        ));
    }
    if groups_weights.len() != n_groups {
        return Err(PricingError::DimensionMismatch(
            "weights slice length != groups length",
        ));
    }

    let mut total_periods = 0usize;
    let mut group_total_weights = vec![0.0; n_groups];
    let mut group_weighted_means = vec![0.0; n_groups];

    for i in 0..n_groups {
        let losses = groups_losses[i];
        let weights = groups_weights[i];
        if losses.len() != weights.len() || losses.is_empty() {
            return Err(PricingError::DimensionMismatch(
                "each group must have non-empty matching losses and weights",
            ));
        }
        total_periods += losses.len();
        let mut sum_p = 0.0;
        let mut sum_px = 0.0;
        for (&x, &p) in losses.iter().zip(weights.iter()) {
            if !x.is_finite() || !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "losses and weights must be finite",
                ));
            }
            if !(p > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "weights must be strictly positive",
                ));
            }
            sum_p += p;
            sum_px += p * x;
        }
        group_total_weights[i] = sum_p;
        group_weighted_means[i] = sum_px / sum_p;
    }

    if total_periods <= n_groups {
        return Err(PricingError::InvalidParameter(
            "total observations must exceed number of groups",
        ));
    }

    // 1. Expected Process Variance (s^2 = EPV):
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
    let epv = (s2_num / df_epv as f64).max(0.0);

    // 2. Collective Exposure-Weighted Mean and Variance of Hypothetical Means (a = VHM):
    let total_weight: f64 = group_total_weights.iter().sum();
    let mu_exposure_weighted: f64 = group_weighted_means
        .iter()
        .zip(group_total_weights.iter())
        .map(|(&m, &w)| m * w)
        .sum::<f64>()
        / total_weight;

    // c = P * (1 - sum_i (P_i / P)^2) avoids intermediate w^2 overflow for large exposures.
    let sum_rel_sq: f64 = group_total_weights
        .iter()
        .map(|&w| {
            let r = w / total_weight;
            r * r
        })
        .sum();
    let c = total_weight * (1.0 - sum_rel_sq);

    // Unbiased estimator of VHM:
    // a = (1 / c) * [ sum_i P_i * (X_bar_i - mu)^2 - (n_groups - 1) * s^2 ]
    let mut vhm_num = 0.0;
    for i in 0..n_groups {
        let p_i = group_total_weights[i];
        let diff = group_weighted_means[i] - mu_exposure_weighted;
        vhm_num += p_i * diff * diff;
    }
    let vhm = if c > 0.0 {
        ((vhm_num - ((n_groups - 1) as f64) * epv) / c).max(0.0)
    } else {
        0.0
    };

    // 3. Credibility K and Z_i
    let k = if vhm > 0.0 { epv / vhm } else { f64::INFINITY };

    let mut credibilities = vec![0.0; n_groups];
    let mut blended_estimates = vec![0.0; n_groups];

    let mu = if vhm > 0.0 {
        let mut sum_z = 0.0;
        let mut sum_zx = 0.0;
        for i in 0..n_groups {
            let p_i = group_total_weights[i];
            let z_i = if k == 0.0 { 1.0 } else { p_i / (p_i + k) };
            credibilities[i] = z_i;
            sum_z += z_i;
            sum_zx += z_i * group_weighted_means[i];
        }
        if sum_z > 0.0 {
            sum_zx / sum_z
        } else {
            mu_exposure_weighted
        }
    } else {
        mu_exposure_weighted
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
