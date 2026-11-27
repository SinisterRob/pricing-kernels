//! NCCI Workers' Compensation Experience Rating Plan (ERP) and off-balance kernels.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Individual claim record in the experience period.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExperienceClaim {
    /// Total reported/incurred loss amount.
    pub incurred: f64,
    /// Whether the claim is medical-only (subject to 70% ERA reduction in NCCI states).
    pub is_medical_only: bool,
}

/// NCCI Experience Rating Plan calculation inputs and parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NcciErpSpecification {
    /// State split point between primary and excess loss (e.g. $18,500).
    pub split_point: f64,
    /// Total expected losses (`E = sum_class Payroll * Expected Loss Rate`).
    pub expected_losses: f64,
    /// Expected primary ratio (`D`-ratio, strictly between `0.0` and `1.0`).
    pub d_ratio: f64,
    /// Weighting value `W` (credibility assigned to actual excess losses, in `[0.0, 1.0]`).
    pub weight_w: f64,
    /// Ballast value `B` (stabilizing constant added to numerator and denominator, `>= 0.0`).
    pub ballast_b: f64,
    /// Experience Rating Adjustment (ERA) applies 70% reduction to medical-only claims.
    pub apply_era: bool,
}

/// Concise alias for [`NcciErpSpecification`] matching [`super::IsoErpSpec`] and [`super::RetroPlanSpec`].
pub type NcciErpSpec = NcciErpSpecification;

/// Detailed result of NCCI Experience Modification calculation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NcciErpResult {
    /// Final Experience Modification Factor rounded to 3 decimal places (e.g., `0.925` or `1.150`).
    pub mod_factor: f64,
    /// Unrounded Experience Modification Factor (`numerator / denominator`).
    pub mod_factor_unrounded: f64,
    /// Actual primary losses (`A_p`).
    pub actual_primary: f64,
    /// Actual excess losses (`A_e`).
    pub actual_excess: f64,
    /// Expected primary losses (`E_p = E * D`).
    pub expected_primary: f64,
    /// Expected excess losses (`E_e = E * (1 - D)`).
    pub expected_excess: f64,
    /// Credibility-weighted actual excess: `W * A_e`.
    pub ratable_excess_actual: f64,
    /// Credibility-weighted expected excess: `(1 - W) * E_e`.
    pub ratable_excess_expected: f64,
}

/// Compute Workers' Compensation Experience Modification according to the NCCI formula:
///
/// `Mod = (A_p + W * A_e + (1 - W) * E_e + B) / (E + B)`
///
/// where each claim's loss is split at `split_point` (after 70% ERA reduction for
/// medical-only claims when `apply_era` is `true`).
pub fn calculate_ncci_experience_mod(
    spec: &NcciErpSpecification,
    claims: &[ExperienceClaim],
) -> Result<NcciErpResult, PricingError> {
    if !spec.expected_losses.is_finite() || !(spec.expected_losses > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "expected losses must be finite and positive",
        ));
    }
    if !spec.split_point.is_finite() || !(spec.split_point > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "split point must be finite and positive",
        ));
    }
    if !(spec.d_ratio.is_finite() && spec.d_ratio > 0.0 && spec.d_ratio < 1.0) {
        return Err(PricingError::InvalidParameter(
            "d_ratio must be finite and strictly between 0.0 and 1.0",
        ));
    }
    if !(spec.weight_w.is_finite() && spec.weight_w >= 0.0 && spec.weight_w <= 1.0) {
        return Err(PricingError::InvalidParameter(
            "weight_w must be finite and between 0.0 and 1.0",
        ));
    }
    if !spec.ballast_b.is_finite() || !(spec.ballast_b >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "ballast_b must be finite and non-negative",
        ));
    }

    let mut actual_primary = 0.0;
    let mut actual_excess = 0.0;

    for claim in claims {
        let mut loss = claim.incurred;
        if !loss.is_finite() || !(loss >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "claim incurred must be finite and non-negative",
            ));
        }
        if spec.apply_era && claim.is_medical_only {
            loss *= 0.30; // 70% reduction under ERA
        }
        let primary = loss.min(spec.split_point);
        let excess = (loss - spec.split_point).max(0.0);
        actual_primary += primary;
        actual_excess += excess;
    }

    let expected_primary = spec.expected_losses * spec.d_ratio;
    let expected_excess = spec.expected_losses * (1.0 - spec.d_ratio);

    let ratable_excess_actual = spec.weight_w * actual_excess;
    let ratable_excess_expected = (1.0 - spec.weight_w) * expected_excess;

    let numerator =
        actual_primary + ratable_excess_actual + ratable_excess_expected + spec.ballast_b;
    let denominator = spec.expected_losses + spec.ballast_b;

    let mod_factor_unrounded = numerator / denominator;
    let mod_factor = (mod_factor_unrounded * 1000.0).round() / 1000.0;

    Ok(NcciErpResult {
        mod_factor,
        mod_factor_unrounded,
        actual_primary,
        actual_excess,
        expected_primary,
        expected_excess,
        ratable_excess_actual,
        ratable_excess_expected,
    })
}

/// Compute statewide or line-wide experience rating off-balance factor:
/// `B = (sum_k ExpectedLoss_k) / (sum_k ExpectedLoss_k * Mod_k)`
///
/// Applied to manual rates to maintain overall premium adequacy.
pub fn calculate_off_balance(expected_losses: &[f64], mods: &[f64]) -> Result<f64, PricingError> {
    if expected_losses.is_empty() || expected_losses.len() != mods.len() {
        return Err(PricingError::DimensionMismatch(
            "expected_losses and mods must have identical non-zero lengths",
        ));
    }
    let mut sum_e = 0.0;
    let mut sum_e_mod = 0.0;
    for (&e, &m) in expected_losses.iter().zip(mods.iter()) {
        if !e.is_finite() || !m.is_finite() || !(e > 0.0) || !(m > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "expected losses and mods must be finite and positive",
            ));
        }
        sum_e += e;
        sum_e_mod += e * m;
    }
    if !sum_e.is_finite() || !sum_e_mod.is_finite() || !(sum_e_mod > 0.0) {
        return Err(PricingError::SingularMatrix(
            "sum of modified losses is non-positive or non-finite",
        ));
    }
    Ok(sum_e / sum_e_mod)
}
