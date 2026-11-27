use super::table_m::TableMEngine;
use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Parameters defining a Commercial Incurred Loss Retrospective Rating Plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetroPlanSpec {
    /// Standard Premium (P).
    pub standard_premium: f64,
    /// Expected loss ratio (ELR = Expected Losses / Standard Premium).
    pub expected_loss_ratio: f64,
    /// Loss Conversion Factor (LCF, covers unallocated loss adjustment expenses, e.g. 1.10).
    pub lcf: f64,
    /// Tax Multiplier (TM, e.g. 1.035).
    pub tax_multiplier: f64,
    /// Maximum Retrospective Premium Ratio (G = MaxPremium / StandardPremium).
    pub max_retro_ratio: f64,
    /// Minimum Retrospective Premium Ratio (H = MinPremium / StandardPremium).
    pub min_retro_ratio: f64,
    /// Expense loading ratio (excluding taxes and ULAE).
    pub expense_ratio: f64,
}

/// Solved retrospective rating plan parameters and financial breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetroPlanResult {
    /// Basic Premium Factor (BPF).
    pub basic_premium_factor: f64,
    /// Dollar Basic Premium: BPF * Standard Premium.
    pub basic_premium_dollars: f64,
    /// Maximum loss entry ratio r_G.
    pub entry_ratio_max: f64,
    /// Minimum loss entry ratio r_H.
    pub entry_ratio_min: f64,
    /// Insurance charge at maximum cap: phi(r_G).
    pub charge_at_max: f64,
    /// Insurance savings at minimum floor: psi(r_H).
    pub savings_at_min: f64,
    /// Net insurance charge: I = phi(r_G) - psi(r_H).
    pub net_insurance_charge: f64,
}

/// Solves for the balanced Basic Premium Factor (BPF) of an incurred retrospective plan:
///
/// Under actuarial balance (expected retro premium == standard expected cost):
/// BPF = ExpenseRatio - (LCF - 1.0) * ELR + [ phi(r_G) - psi(r_H) ] * LCF * ELR
///
/// where entry ratios r_G and r_H depend on BPF:
/// r_G = ( (G / TM) - BPF ) / (LCF * ELR)
/// r_H = ( (H / TM) - BPF ) / (LCF * ELR)
pub fn solve_balanced_retro_plan(
    spec: &RetroPlanSpec,
    table_m: &TableMEngine,
    max_iter: usize,
    tol: f64,
) -> Result<RetroPlanResult, PricingError> {
    if spec.standard_premium <= 0.0 || spec.expected_loss_ratio <= 0.0 {
        return Err(PricingError::NonPositiveValue(
            "standard premium and ELR must be positive",
        ));
    }
    if spec.max_retro_ratio <= spec.min_retro_ratio {
        return Err(PricingError::InvalidParameter(
            "max retro ratio must be > min retro ratio",
        ));
    }
    if spec.tax_multiplier <= 0.0 || spec.lcf <= 0.0 {
        return Err(PricingError::NonPositiveValue(
            "tax multiplier and LCF must be positive",
        ));
    }

    let e_lcf = spec.lcf * spec.expected_loss_ratio;
    let non_loss_cost = spec.expense_ratio - (spec.lcf - 1.0) * spec.expected_loss_ratio;

    // Iterative solution for BPF
    let mut bpf = spec.expense_ratio; // initial seed
    let mut iter = 0;
    let mut diff = 1.0;

    let mut r_g = 0.0;
    let mut r_h = 0.0;
    let mut phi_g = 0.0;
    let mut psi_h = 0.0;
    let mut net_charge = 0.0;

    while iter < max_iter && diff > tol {
        iter += 1;

        // Compute loss entry ratios
        r_g = (((spec.max_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);
        r_h = (((spec.min_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);

        phi_g = table_m.charge_at(r_g);
        psi_h = table_m.savings_at(r_h);
        net_charge = phi_g - psi_h;

        let new_bpf = non_loss_cost + net_charge * e_lcf;
        diff = (new_bpf - bpf).abs();
        bpf = f64::midpoint(bpf, new_bpf); // damped relaxation
    }

    if diff > tol && iter >= max_iter {
        return Err(PricingError::NonConvergence(
            "retro plan basic premium solver failed to converge",
        ));
    }

    Ok(RetroPlanResult {
        basic_premium_factor: bpf,
        basic_premium_dollars: bpf * spec.standard_premium,
        entry_ratio_max: r_g,
        entry_ratio_min: r_h,
        charge_at_max: phi_g,
        savings_at_min: psi_h,
        net_insurance_charge: net_charge,
    })
}
