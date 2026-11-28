//! Commercial large deductible pricing with ALAE treatment and optional aggregate
//! deductible cap charge (CAS Exam 8 / Gillam & Snader).

use super::table_m::TableMEngine;
use crate::error::PricingError;
use crate::layers::SeverityDistribution;
use serde::{Deserialize, Serialize};

/// Contractual treatment of Allocated Loss Adjustment Expense (ALAE) under a large deductible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlaeTreatment {
    /// ALAE is included with indemnity inside the deductible; deductible credit applies to `Loss + ALAE`
    /// using the combined-severity LER.
    InsideLimit,
    /// ALAE is shared pro-rata between insured and insurer in proportion to indemnity below/above
    /// the deductible; indemnity LER applies equally to ALAE.
    ProRata,
    /// Insurer pays all ALAE from the first dollar (outside the deductible); deductible credit
    /// applies to indemnity only.
    OutsideLimit,
}

/// Commercial large deductible pricing specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LargeDeductibleSpec {
    /// Expected full-coverage ground-up indemnity loss up to `policy_limit`.
    pub expected_indemnity: f64,
    /// Expected full-coverage ALAE (`>= 0`).
    pub expected_alae: f64,
    /// Per-occurrence deductible `d >= 0`.
    pub deductible: f64,
    /// Per-occurrence policy limit `L > deductible`.
    pub policy_limit: f64,
    /// Contractual ALAE treatment relative to the deductible.
    pub alae_treatment: AlaeTreatment,
    /// Fixed expense dollars (e.g. underwriting, loss control, claims administration).
    pub fixed_expenses: f64,
    /// Variable expense & profit ratio `v in [0, 1)` applied to net premium (commission, taxes, profit).
    pub variable_expense_ratio: f64,
    /// Optional aggregate deductible cap expressed as an entry ratio `r_agg` relative to
    /// expected retained loss (requires a [`TableMEngine`] for aggregate cap pricing).
    pub aggregate_cap_entry_ratio: Option<f64>,
}

impl LargeDeductibleSpec {
    /// Price this commercial large deductible program against a parametric severity distribution
    /// and optional [`TableMEngine`] (required when `aggregate_cap_entry_ratio` is `Some`).
    pub fn price(
        &self,
        severity: &SeverityDistribution,
        retained_table_m: Option<&TableMEngine>,
    ) -> Result<LargeDeductibleResult, PricingError> {
        price_large_deductible(self, severity, retained_table_m)
    }
}

/// Financial breakdown of a priced commercial large deductible program.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LargeDeductibleResult {
    /// Per-occurrence Loss Elimination Ratio `LER(d; L) = E[X ∧ d] / E[X ∧ L]`.
    pub loss_elimination_ratio: f64,
    /// Per-occurrence Excess Loss Factor `ELF = 1 - LER`.
    pub excess_loss_factor: f64,
    /// Expected indemnity retained by the insured below the deductible.
    pub retained_indemnity: f64,
    /// Expected ALAE retained by the insured.
    pub retained_alae: f64,
    /// Expected excess indemnity paid by the insurer above the deductible.
    pub excess_indemnity: f64,
    /// Expected ALAE paid by the insurer.
    pub insurer_alae: f64,
    /// Expected cost to the insurer of an aggregate deductible cap (`0.0` if uncapped).
    pub aggregate_cap_charge: f64,
    /// Total expected deductible credit dollars (`retained_indemnity + retained_alae - aggregate_cap_charge`).
    pub net_deductible_credit_dollars: f64,
    /// Net deductible credit as a fraction of full-coverage `expected_indemnity + expected_alae`.
    pub net_deductible_credit_ratio: f64,
    /// Indicated large-deductible net premium:
    /// `(excess_indemnity + insurer_alae + aggregate_cap_charge + fixed_expenses) / (1 - variable_expense_ratio)`.
    pub indicated_premium: f64,
}

/// Price a commercial large deductible program using a parametric [`SeverityDistribution`]
/// (and optional [`TableMEngine`] when `aggregate_cap_entry_ratio` is specified).
///
/// Under [`AlaeTreatment::InsideLimit`], pass a combined Loss+ALAE severity distribution;
/// under [`AlaeTreatment::ProRata`] or [`AlaeTreatment::OutsideLimit`], pass the indemnity
/// severity distribution.
pub fn price_large_deductible(
    spec: &LargeDeductibleSpec,
    severity: &SeverityDistribution,
    retained_table_m: Option<&TableMEngine>,
) -> Result<LargeDeductibleResult, PricingError> {
    if !spec.expected_indemnity.is_finite()
        || !spec.expected_alae.is_finite()
        || !spec.fixed_expenses.is_finite()
        || !spec.variable_expense_ratio.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "expected losses and expenses must be finite",
        ));
    }
    if !(spec.expected_indemnity > 0.0) || !(spec.expected_alae >= 0.0) {
        return Err(PricingError::NonPositiveValue(
            "expected_indemnity must be > 0 and expected_alae >= 0",
        ));
    }
    if !(spec.fixed_expenses >= 0.0)
        || !(spec.variable_expense_ratio >= 0.0)
        || !(spec.variable_expense_ratio < 1.0)
    {
        return Err(PricingError::InvalidParameter(
            "fixed_expenses must be >= 0 and variable_expense_ratio in [0, 1)",
        ));
    }

    let ler = severity.loss_elimination_ratio(spec.deductible, spec.policy_limit)?;
    let elf = (1.0 - ler).max(0.0);

    let retained_indemnity = spec.expected_indemnity * ler;
    let excess_indemnity = spec.expected_indemnity * elf;

    let (retained_alae, insurer_alae) = match spec.alae_treatment {
        AlaeTreatment::InsideLimit | AlaeTreatment::ProRata => {
            (spec.expected_alae * ler, spec.expected_alae * elf)
        }
        AlaeTreatment::OutsideLimit => (0.0, spec.expected_alae),
    };

    let expected_retained_total = retained_indemnity + retained_alae;

    let aggregate_cap_charge = match spec.aggregate_cap_entry_ratio {
        None => 0.0,
        Some(r_agg) => {
            if !r_agg.is_finite() || !(r_agg >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "aggregate_cap_entry_ratio must be finite and non-negative",
                ));
            }
            let Some(tm) = retained_table_m else {
                return Err(PricingError::InvalidParameter(
                    "retained_table_m is required when aggregate_cap_entry_ratio is set",
                ));
            };
            expected_retained_total * tm.charge_at(r_agg)
        }
    };

    let full_loss_and_alae = spec.expected_indemnity + spec.expected_alae;
    let net_deductible_credit_dollars = expected_retained_total - aggregate_cap_charge;
    let net_deductible_credit_ratio = net_deductible_credit_dollars / full_loss_and_alae;

    let insurer_loss_and_alae = excess_indemnity + insurer_alae + aggregate_cap_charge;
    let indicated_premium =
        (insurer_loss_and_alae + spec.fixed_expenses) / (1.0 - spec.variable_expense_ratio);

    Ok(LargeDeductibleResult {
        loss_elimination_ratio: ler,
        excess_loss_factor: elf,
        retained_indemnity,
        retained_alae,
        excess_indemnity,
        insurer_alae,
        aggregate_cap_charge,
        net_deductible_credit_dollars,
        net_deductible_credit_ratio,
        indicated_premium,
    })
}
