//! ISO Commercial General Liability / Commercial Auto Experience and Schedule Rating Plan
//! (CAS Exam 8).

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Parameters for an ISO Commercial Lines Experience Rating Plan (CGL / Commercial Auto).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsoErpSpec {
    /// Expected basic-limits losses (`E`) over the experience period (after AER/DET adjustments).
    pub expected_basic_limits_loss: f64,
    /// Maximum Single Limit (`MSL`): per-occurrence cap applied to each experience claim.
    pub max_single_limit: f64,
    /// Credibility constant `K > 0` (`Z = E / (E + K)`).
    pub credibility_k: f64,
    /// Expected Unlimited / Excess-to-Basic adjustment or balance factor (default `1.0`).
    pub balance_factor: f64,
}

/// Detailed result of an ISO Commercial Experience Rating calculation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsoErpResult {
    /// Credibility weight `Z = E / (E + K)`.
    pub credibility_z: f64,
    /// Sum of experience claims after applying per-occurrence `MSL` cap (`A`).
    pub actual_ratable_loss: f64,
    /// Actual-to-Expected ratio `A / E`.
    pub actual_to_expected: f64,
    /// Unrounded experience modification factor: `(1 + Z * (A - E) / E) * balance_factor`.
    pub mod_factor_unrounded: f64,
    /// Standard 3-decimal rounded experience modification factor.
    pub mod_factor: f64,
}

/// Compute an ISO Commercial GL / Auto Experience Modification Factor:
///
/// Each claim `x_i` is capped at `max_single_limit`, summed to `A = sum min(x_i, MSL)`,
/// and blended with expected basic-limits losses `E` via credibility `Z = E / (E + K)`:
/// `Mod = (1 + Z * (A - E) / E) * balance_factor`.
pub fn calculate_iso_experience_mod(
    spec: &IsoErpSpec,
    claim_losses: &[f64],
) -> Result<IsoErpResult, PricingError> {
    if !spec.expected_basic_limits_loss.is_finite()
        || !spec.max_single_limit.is_finite()
        || !spec.credibility_k.is_finite()
        || !spec.balance_factor.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "ISO ERP parameters must be finite",
        ));
    }
    if !(spec.expected_basic_limits_loss > 0.0)
        || !(spec.max_single_limit > 0.0)
        || !(spec.credibility_k > 0.0)
        || !(spec.balance_factor > 0.0)
    {
        return Err(PricingError::NonPositiveValue(
            "expected_basic_limits_loss, max_single_limit, credibility_k, and balance_factor must be > 0",
        ));
    }

    let mut actual_ratable_loss = 0.0;
    for &loss in claim_losses {
        if !loss.is_finite() || !(loss >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "claim losses must be finite and non-negative",
            ));
        }
        actual_ratable_loss += loss.min(spec.max_single_limit);
    }

    let z =
        spec.expected_basic_limits_loss / (spec.expected_basic_limits_loss + spec.credibility_k);
    let aer = actual_ratable_loss / spec.expected_basic_limits_loss;
    let mod_unrounded = (1.0 + z * (aer - 1.0)) * spec.balance_factor;
    let mod_rounded = (mod_unrounded * 1000.0).round() / 1000.0;

    Ok(IsoErpResult {
        credibility_z: z,
        actual_ratable_loss,
        actual_to_expected: aer,
        mod_factor_unrounded: mod_unrounded,
        mod_factor: mod_rounded,
    })
}

/// Individual schedule rating / IRPM debit or credit category with regulatory bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRatingItem {
    /// Selected credit (negative) or debit (positive), e.g. `-0.05` for a 5% credit.
    pub adjustment: f64,
    /// Minimum allowed adjustment (maximum credit, e.g. `-0.10`).
    pub min_adjustment: f64,
    /// Maximum allowed adjustment (maximum debit, e.g. `0.10`).
    pub max_adjustment: f64,
}

/// Result of commercial schedule rating (IRPM) aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRatingResult {
    /// Unclamped sum of category adjustments.
    pub raw_total_adjustment: f64,
    /// Clamped total schedule adjustment within `[overall_min, overall_max]`.
    pub applied_adjustment: f64,
    /// Multiplicative schedule modification factor (`1.0 + applied_adjustment`).
    pub schedule_mod_factor: f64,
}

/// Validate and aggregate commercial schedule rating (IRPM) adjustments within
/// per-category and overall state-filed bounds (typically `[-0.25, +0.25]`).
pub fn calculate_schedule_mod(
    items: &[ScheduleRatingItem],
    overall_min: f64,
    overall_max: f64,
) -> Result<ScheduleRatingResult, PricingError> {
    if !overall_min.is_finite()
        || !overall_max.is_finite()
        || !(overall_min <= 0.0)
        || !(overall_max >= 0.0)
        || !(overall_min <= overall_max)
        || overall_min <= -1.0
    {
        return Err(PricingError::InvalidParameter(
            "overall schedule bounds must be finite with -1 < overall_min <= 0 <= overall_max",
        ));
    }

    let mut raw_total = 0.0;
    for item in items {
        if !item.adjustment.is_finite()
            || !item.min_adjustment.is_finite()
            || !item.max_adjustment.is_finite()
            || !(item.min_adjustment <= item.max_adjustment)
        {
            return Err(PricingError::InvalidParameter(
                "schedule rating item bounds and adjustment must be finite with min <= max",
            ));
        }
        if item.adjustment < item.min_adjustment || item.adjustment > item.max_adjustment {
            return Err(PricingError::OutOfBounds(
                "schedule rating item adjustment exceeds its category bounds",
            ));
        }
        raw_total += item.adjustment;
    }

    let applied = raw_total.clamp(overall_min, overall_max);
    Ok(ScheduleRatingResult {
        raw_total_adjustment: raw_total,
        applied_adjustment: applied,
        schedule_mod_factor: 1.0 + applied,
    })
}
