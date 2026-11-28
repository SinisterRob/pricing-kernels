//! NCCI Workers' Compensation Experience Rating Plan (ERP) and off-balance kernels.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Fraction of a medical-only claim's incurred loss that remains ratable under the NCCI
/// Experience Rating Adjustment (ERA): a 70% reduction leaves `0.30`.
pub const ERA_MEDICAL_ONLY_RATABLE_FACTOR: f64 = 0.30;

/// NCCI Experience Rating Plan Manual Rule 2-B: the multiple claim accident limitation equals
/// this multiple of the state per claim accident limitation.
pub const NCCI_MULTIPLE_CLAIM_LIMIT_MULTIPLIER: f64 = 2.0;

/// Decimal places of the published NCCI experience modification. NCCI rounds the mod to two
/// decimals, half-up (NCCI, *ABCs of Experience Rating*, worksheet example: `243,120 / 242,971
/// = 1.0006 -> 1.00`; Experience Rating Plan Manual Rule 3).
pub const NCCI_MOD_DECIMALS: i32 = 2;

/// Absolute guard (in units of the last published decimal) added before half-up rounding so
/// that a mod whose exact value lies on a `.xx5` boundary but is represented in binary just
/// below it still rounds up.
const MOD_HALF_UP_GUARD: f64 = 1e-9;

/// Round a non-negative modification half-up to [`NCCI_MOD_DECIMALS`] decimals.
fn round_mod_half_up(x: f64) -> f64 {
    let scale = 10f64.powi(NCCI_MOD_DECIMALS);
    (x * scale + 0.5 + MOD_HALF_UP_GUARD).floor() / scale
}

/// NCCI Experience Rating Plan Manual Rule 2-B accident limitations (state "per claim" and
/// "multiple claim" accident limitations, published per state in the ERM state values).
///
/// Applied per accident **before** the primary/excess split: each claim's ratable loss (after
/// ERA) is capped at `per_claim_limit`, then the sum over all claims of one accident is capped at
/// `multiple_claim_limit`. Disease limitations are not modeled.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NcciAccidentLimitation {
    /// State per claim accident limitation (ratable loss cap per claim).
    pub per_claim_limit: f64,
    /// Multiple claim accident limitation (ratable loss cap per accident).
    pub multiple_claim_limit: f64,
}

impl NcciAccidentLimitation {
    /// Per claim limitation with the NCCI default multiple claim limitation
    /// `NCCI_MULTIPLE_CLAIM_LIMIT_MULTIPLIER * per_claim_limit`.
    pub fn new(per_claim_limit: f64) -> Result<Self, PricingError> {
        Self::with_multiple_claim_limit(
            per_claim_limit,
            NCCI_MULTIPLE_CLAIM_LIMIT_MULTIPLIER * per_claim_limit,
        )
    }

    /// Explicit per claim and multiple claim limitations (`0 < per_claim <= multiple_claim`).
    pub fn with_multiple_claim_limit(
        per_claim_limit: f64,
        multiple_claim_limit: f64,
    ) -> Result<Self, PricingError> {
        if !per_claim_limit.is_finite() || !(per_claim_limit > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "per claim accident limitation must be finite and positive",
            ));
        }
        if !multiple_claim_limit.is_finite() || !(multiple_claim_limit >= per_claim_limit) {
            return Err(PricingError::InvalidParameter(
                "multiple claim accident limitation must be finite and >= per claim limitation",
            ));
        }
        Ok(Self {
            per_claim_limit,
            multiple_claim_limit,
        })
    }
}

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
pub struct NcciErpSpec {
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

impl NcciErpSpec {
    /// Compute the NCCI Experience Modification Factor for the given experience-period claims.
    pub fn calculate(&self, claims: &[ExperienceClaim]) -> Result<NcciErpResult, PricingError> {
        calculate_ncci_experience_mod(self, claims)
    }

    /// Compute the Experience Modification from claims grouped by accident with optional
    /// Rule 2-B accident limitations (see [`calculate_ncci_experience_mod_by_accident`]).
    pub fn calculate_by_accident(
        &self,
        accidents: &[&[ExperienceClaim]],
        limitation: Option<&NcciAccidentLimitation>,
    ) -> Result<NcciErpResult, PricingError> {
        calculate_ncci_experience_mod_by_accident(self, accidents, limitation)
    }
}

/// Detailed result of NCCI Experience Modification calculation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NcciErpResult {
    /// Final Experience Modification Factor rounded half-up to [`NCCI_MOD_DECIMALS`] (= 2)
    /// decimal places as published by NCCI (e.g., `0.93` or `1.15`).
    /// Use [`Self::mod_factor_unrounded`] for the exact ratio.
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

fn validate_spec(spec: &NcciErpSpec) -> Result<(), PricingError> {
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
    Ok(())
}

/// Ratable loss of one claim after ERA and the per claim accident limitation.
fn ratable_claim_loss(
    spec: &NcciErpSpec,
    claim: &ExperienceClaim,
    limitation: Option<&NcciAccidentLimitation>,
) -> Result<f64, PricingError> {
    let mut loss = claim.incurred;
    if !loss.is_finite() || !(loss >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "claim incurred must be finite and non-negative",
        ));
    }
    if spec.apply_era && claim.is_medical_only {
        loss *= ERA_MEDICAL_ONLY_RATABLE_FACTOR;
    }
    if let Some(lim) = limitation {
        loss = loss.min(lim.per_claim_limit);
    }
    Ok(loss)
}

fn finalize_mod(spec: &NcciErpSpec, actual_primary: f64, actual_excess: f64) -> NcciErpResult {
    let expected_primary = spec.expected_losses * spec.d_ratio;
    let expected_excess = spec.expected_losses * (1.0 - spec.d_ratio);

    let ratable_excess_actual = spec.weight_w * actual_excess;
    let ratable_excess_expected = (1.0 - spec.weight_w) * expected_excess;

    let numerator =
        actual_primary + ratable_excess_actual + ratable_excess_expected + spec.ballast_b;
    let denominator = spec.expected_losses + spec.ballast_b;

    let mod_factor_unrounded = numerator / denominator;
    let mod_factor = round_mod_half_up(mod_factor_unrounded);

    NcciErpResult {
        mod_factor,
        mod_factor_unrounded,
        actual_primary,
        actual_excess,
        expected_primary,
        expected_excess,
        ratable_excess_actual,
        ratable_excess_expected,
    }
}

/// Compute Workers' Compensation Experience Modification according to the NCCI formula:
///
/// `Mod = (A_p + W * A_e + (1 - W) * E_e + B) / (E + B)`
///
/// where each claim's loss is split at `split_point` (after 70% ERA reduction for
/// medical-only claims when `apply_era` is `true`). Each claim is treated as a separate
/// accident and **no accident limitation is applied**: pass already-limited ratable losses, or
/// use [`calculate_ncci_experience_mod_by_accident`] with an [`NcciAccidentLimitation`].
pub fn calculate_ncci_experience_mod(
    spec: &NcciErpSpec,
    claims: &[ExperienceClaim],
) -> Result<NcciErpResult, PricingError> {
    validate_spec(spec)?;

    let mut actual_primary = 0.0;
    let mut actual_excess = 0.0;
    for claim in claims {
        let loss = ratable_claim_loss(spec, claim, None)?;
        actual_primary += loss.min(spec.split_point);
        actual_excess += (loss - spec.split_point).max(0.0);
    }

    Ok(finalize_mod(spec, actual_primary, actual_excess))
}

/// NCCI Experience Modification from claims grouped by accident (`accidents[a]` = all claims
/// arising from accident `a`), with optional ERM Rule 2-B accident limitations.
///
/// Per claim: ratable loss = incurred (x `ERA_MEDICAL_ONLY_RATABLE_FACTOR` for medical-only
/// claims under ERA), capped at `per_claim_limit`; primary = `min(ratable, split_point)`.
/// Per accident: total ratable loss is capped at `multiple_claim_limit`; the reduction is taken
/// from excess loss first (accident primary = `min(sum primary, accident limited total)`).
/// With `limitation = None` this equals [`calculate_ncci_experience_mod`] on the flattened claims.
pub fn calculate_ncci_experience_mod_by_accident(
    spec: &NcciErpSpec,
    accidents: &[&[ExperienceClaim]],
    limitation: Option<&NcciAccidentLimitation>,
) -> Result<NcciErpResult, PricingError> {
    validate_spec(spec)?;
    if let Some(lim) = limitation {
        // Re-validate in case the struct was built literally.
        NcciAccidentLimitation::with_multiple_claim_limit(
            lim.per_claim_limit,
            lim.multiple_claim_limit,
        )?;
    }

    let mut actual_primary = 0.0;
    let mut actual_excess = 0.0;
    for claims in accidents {
        let mut acc_primary = 0.0;
        let mut acc_total = 0.0;
        for claim in *claims {
            let loss = ratable_claim_loss(spec, claim, limitation)?;
            acc_primary += loss.min(spec.split_point);
            acc_total += loss;
        }
        if let Some(lim) = limitation {
            acc_total = acc_total.min(lim.multiple_claim_limit);
        }
        let acc_primary = acc_primary.min(acc_total);
        actual_primary += acc_primary;
        actual_excess += (acc_total - acc_primary).max(0.0);
    }

    Ok(finalize_mod(spec, actual_primary, actual_excess))
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
