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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcciErpSpecification {
    /// State split point between primary and excess loss (e.g. $18,500).
    pub split_point: f64,
    /// Total expected losses (E = sum_class Payroll * Expected Loss Rate).
    pub expected_losses: f64,
    /// Expected primary ratio (D-ratio, typically between 0.20 and 0.50).
    pub d_ratio: f64,
    /// Weighting value W (credibility assigned to actual excess losses).
    pub weight_w: f64,
    /// Ballast value B (stabilizing constant added to numerator and denominator).
    pub ballast_b: f64,
    /// Experience Rating Adjustment (ERA) applies 70% reduction to medical-only claims.
    pub apply_era: bool,
}

/// Detailed result of NCCI Experience Modification calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcciErpResult {
    /// Final Experience Modification Factor (e.g., 0.925 or 1.150).
    pub mod_factor: f64,
    /// Actual primary losses (A_p).
    pub actual_primary: f64,
    /// Actual excess losses (A_e).
    pub actual_excess: f64,
    /// Expected primary losses (E_p = E * D).
    pub expected_primary: f64,
    /// Expected excess losses (E_e = E * (1 - D)).
    pub expected_excess: f64,
    /// Credibility-weighted actual excess: W * A_e.
    pub ratable_excess_actual: f64,
    /// Credibility-weighted expected excess: (1 - W) * E_e.
    pub ratable_excess_expected: f64,
}

/// Compute Workers' Compensation Experience Modification according to the NCCI formula:
///
/// Mod = (A_p + W * A_e + (1 - W) * E_e + B) / (E + B)
///
/// where:
/// - Each claim's loss is split at `split_point`:
///   - Primary: min(Claim, split_point)
///   - Excess: max(0, Claim - split_point)
///   - If ERA is enabled and claim is medical-only: Claim is discounted by 70% (multiplied by 0.30).
pub fn calculate_ncci_experience_mod(
    spec: &NcciErpSpecification,
    claims: &[ExperienceClaim],
) -> Result<NcciErpResult, PricingError> {
    if spec.expected_losses <= 0.0 {
        return Err(PricingError::NonPositiveValue(
            "expected losses must be positive",
        ));
    }
    if spec.split_point <= 0.0 {
        return Err(PricingError::NonPositiveValue(
            "split point must be positive",
        ));
    }
    if spec.d_ratio <= 0.0 || spec.d_ratio >= 1.0 {
        return Err(PricingError::InvalidParameter(
            "d_ratio must be between 0.0 and 1.0",
        ));
    }
    if spec.weight_w < 0.0 || spec.weight_w > 1.0 {
        return Err(PricingError::InvalidParameter(
            "weight_w must be between 0.0 and 1.0",
        ));
    }
    if spec.ballast_b < 0.0 {
        return Err(PricingError::InvalidParameter(
            "ballast_b must be non-negative",
        ));
    }

    let mut actual_primary = 0.0;
    let mut actual_excess = 0.0;

    for claim in claims {
        let mut loss = claim.incurred;
        if loss < 0.0 {
            return Err(PricingError::InvalidParameter(
                "claim incurred cannot be negative",
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

    let mod_factor = (numerator / denominator * 1000.0).round() / 1000.0;

    Ok(NcciErpResult {
        mod_factor,
        actual_primary,
        actual_excess,
        expected_primary,
        expected_excess,
        ratable_excess_actual,
        ratable_excess_expected,
    })
}

/// Compute statewide or line-wide experience rating off-balance factor:
/// B = (sum_k ExpectedLoss_k) / (sum_k ExpectedLoss_k * Mod_k)
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
        if e <= 0.0 || m <= 0.0 {
            return Err(PricingError::NonPositiveValue(
                "expected losses and mods must be positive",
            ));
        }
        sum_e += e;
        sum_e_mod += e * m;
    }
    if sum_e_mod <= 0.0 {
        return Err(PricingError::SingularMatrix(
            "sum of modified losses is non-positive",
        ));
    }
    Ok(sum_e / sum_e_mod)
}
