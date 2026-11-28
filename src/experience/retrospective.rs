//! Actuarially balanced incurred-loss retrospective rating plan solver (CAS Exam 8 / Brosius 1989).

use super::table_l::{TableLEngine, solve_balanced_retro_plan_table_l};
use super::table_m::TableMEngine;
use crate::error::{PricingError, SolverOptions};
use serde::{Deserialize, Serialize};

/// Parameters defining a Commercial Incurred Loss Retrospective Rating Plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetroPlanSpec {
    /// Standard Premium (`P`).
    pub standard_premium: f64,
    /// Expected loss ratio (`ELR = Expected Losses / Standard Premium`).
    pub expected_loss_ratio: f64,
    /// Loss Conversion Factor (`LCF`, covers unallocated loss adjustment expenses, e.g. `1.10`).
    pub lcf: f64,
    /// Tax Multiplier (`TM`, e.g. `1.035`).
    pub tax_multiplier: f64,
    /// Maximum Retrospective Premium Ratio (`G = MaxPremium / StandardPremium`).
    pub max_retro_ratio: f64,
    /// Minimum Retrospective Premium Ratio (`H = MinPremium / StandardPremium`).
    pub min_retro_ratio: f64,
    /// Expense loading ratio (excluding taxes and ULAE).
    pub expense_ratio: f64,
}

impl RetroPlanSpec {
    /// Solve for the balanced retrospective plan using a [`TableMEngine`] and [`SolverOptions`].
    pub fn solve_table_m(
        &self,
        table_m: &TableMEngine,
        options: &SolverOptions,
    ) -> Result<RetroPlanResult, PricingError> {
        solve_balanced_retro_plan(self, table_m, options.max_iter, options.tol)
    }

    /// Solve for the balanced retrospective plan with per-occurrence loss limit using a
    /// [`TableLEngine`] and [`SolverOptions`].
    pub fn solve_table_l(
        &self,
        table_l: &TableLEngine,
        options: &SolverOptions,
    ) -> Result<RetroPlanResult, PricingError> {
        solve_balanced_retro_plan_table_l(self, table_l, options.max_iter, options.tol)
    }
}

/// Solved retrospective rating plan parameters and financial breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetroPlanResult {
    /// Basic Premium Factor (`BPF`).
    pub basic_premium_factor: f64,
    /// Dollar Basic Premium: `BPF * Standard Premium`.
    pub basic_premium_dollars: f64,
    /// Maximum loss entry ratio `r_G`.
    pub entry_ratio_max: f64,
    /// Minimum loss entry ratio `r_H`.
    pub entry_ratio_min: f64,
    /// Insurance charge at maximum cap: `phi(r_G)`.
    pub charge_at_max: f64,
    /// Insurance savings at minimum floor: `psi(r_H)`.
    pub savings_at_min: f64,
    /// Net insurance charge: `I = phi(r_G) - psi(r_H)`.
    pub net_insurance_charge: f64,
}

/// Shared damped fixed-point solver for balanced Table M and Table L retrospective plans.
pub(crate) fn solve_balanced_retro_plan_with<F, G>(
    spec: &RetroPlanSpec,
    initial_bpf_offset: f64,
    charge_at: F,
    savings_at: G,
    options: SolverOptions,
    non_convergence_msg: &'static str,
    infeasible_msg: &'static str,
) -> Result<RetroPlanResult, PricingError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let SolverOptions { max_iter, tol } = options;
    if max_iter == 0 {
        return Err(PricingError::InvalidParameter("max_iter must be > 0"));
    }
    if !tol.is_finite() || !(tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "tolerance must be finite and positive",
        ));
    }
    if !spec.standard_premium.is_finite()
        || !spec.expected_loss_ratio.is_finite()
        || !(spec.standard_premium > 0.0)
        || !(spec.expected_loss_ratio > 0.0)
    {
        return Err(PricingError::NonPositiveValue(
            "standard premium and ELR must be finite and positive",
        ));
    }
    if !spec.max_retro_ratio.is_finite()
        || !spec.min_retro_ratio.is_finite()
        || !(spec.max_retro_ratio > spec.min_retro_ratio)
        || !(spec.min_retro_ratio >= 0.0)
    {
        return Err(PricingError::InvalidParameter(
            "retro ratios must be finite with max_retro_ratio > min_retro_ratio >= 0",
        ));
    }
    if !spec.tax_multiplier.is_finite()
        || !spec.lcf.is_finite()
        || !spec.expense_ratio.is_finite()
        || !(spec.tax_multiplier > 0.0)
        || !(spec.lcf > 0.0)
    {
        return Err(PricingError::NonPositiveValue(
            "tax multiplier and LCF must be finite and positive, and expense_ratio finite",
        ));
    }

    let e_lcf = spec.lcf * spec.expected_loss_ratio;
    let non_loss_cost = spec.expense_ratio - (spec.lcf - 1.0) * spec.expected_loss_ratio;

    let mut bpf = spec.expense_ratio + initial_bpf_offset * e_lcf;
    let mut converged = false;

    for _ in 0..max_iter {
        let r_g = (((spec.max_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);
        let r_h = (((spec.min_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);

        let net_charge = charge_at(r_g) - savings_at(r_h);
        let new_bpf = non_loss_cost + net_charge * e_lcf;
        let diff = (new_bpf - bpf).abs();
        bpf = f64::midpoint(bpf, new_bpf);

        if diff <= tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(PricingError::NonConvergence(non_convergence_msg));
    }

    // Re-evaluate all plan components at the final converged BPF so outputs are self-consistent.
    let r_g_raw = ((spec.max_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf;
    if r_g_raw < 0.0 {
        return Err(PricingError::OutOfBounds(infeasible_msg));
    }
    let r_h = (((spec.min_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);
    let phi_g = charge_at(r_g_raw);
    let psi_h = savings_at(r_h);
    let net_charge = phi_g - psi_h;

    Ok(RetroPlanResult {
        basic_premium_factor: bpf,
        basic_premium_dollars: bpf * spec.standard_premium,
        entry_ratio_max: r_g_raw,
        entry_ratio_min: r_h,
        charge_at_max: phi_g,
        savings_at_min: psi_h,
        net_insurance_charge: net_charge,
    })
}

/// Solves for the balanced Basic Premium Factor (`BPF`) of an incurred retrospective plan:
///
/// Under actuarial balance (expected retro premium == standard expected cost):
/// `BPF = ExpenseRatio - (LCF - 1.0) * ELR + [ phi(r_G) - psi(r_H) ] * LCF * ELR`
///
/// where entry ratios `r_G` and `r_H` depend on `BPF`:
/// `r_G = ( (G / TM) - BPF ) / (LCF * ELR)`
/// `r_H = ( (H / TM) - BPF ) / (LCF * ELR)`
pub fn solve_balanced_retro_plan(
    spec: &RetroPlanSpec,
    table_m: &TableMEngine,
    max_iter: usize,
    tol: f64,
) -> Result<RetroPlanResult, PricingError> {
    solve_balanced_retro_plan_with(
        spec,
        0.0,
        |r| table_m.charge_at(r),
        |r| table_m.savings_at(r),
        SolverOptions { max_iter, tol },
        "retro plan basic premium solver failed to converge",
        "infeasible retro plan: max_retro_ratio / tax_multiplier is below balanced basic premium",
    )
}
