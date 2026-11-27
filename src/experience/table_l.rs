//! Skurnick (1974) Table L aggregate loss charge engine with per-occurrence loss limitations
//! and balanced loss-limited retrospective rating plan solver.
//!
//! Reference: D. Skurnick (1974), *"Measuring the Effect of Loss Limitations on Retrospective
//! Rating"*, *Proceedings of the Casualty Actuarial Society (PCAS)*, Vol. LXI, pp. 1–28.

use super::retrospective::{RetroPlanResult, RetroPlanSpec};
use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Discrete Table L aggregate distribution row for per-occurrence limited losses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TableLRow {
    /// Entry ratio `r = LimitedLoss / UnlimitedExpectedLoss`.
    pub entry_ratio: f64,
    /// Table L total insurance charge `phi*(r) = E[max(0, Y* - r)] + k`, where `k` is the LER.
    pub charge: f64,
    /// Table L aggregate excess charge excluding per-occurrence `k`: `E[max(0, Y* - r)]`.
    pub aggregate_excess_charge: f64,
    /// Table L insurance savings `psi*(r) = E[max(0, r - Y*)] = phi*(r) + r - 1`.
    pub savings: f64,
}

/// Skurnick (1974) Table L engine for retrospective rating with per-occurrence loss limits.
///
/// Let `k in [0, 1)` be the per-occurrence Loss Elimination Ratio (`LER(L)`) so that
/// limited aggregate loss ratio `Y* = LimitedAggregateLoss / UnlimitedExpectedLoss` satisfies
/// `E[Y*] = 1 - k`. Table L satisfies the exact Skurnick balance identity:
/// `phi*(r) - psi*(r) = 1 - r` for all `r >= 0`, with `phi*(0) = 1` and `lim_{r -> inf} phi*(r) = k`.
/// When `k = 0`, Table L reduces identically to Table M.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableLEngine {
    /// Per-occurrence Loss Elimination Ratio `k = 1 - E[Y*] in [0, 1)`.
    pub loss_elimination_ratio: f64,
    /// Sorted grid of Table L rows.
    pub rows: Vec<TableLRow>,
}

impl TableLEngine {
    /// Build Table L from a discrete probability distribution of **limited** aggregate
    /// entry ratios `(y*_i, p_i)` normalized by **unlimited** expected loss.
    ///
    /// Automatically sorts entry ratios, merges duplicates, and infers `k = 1 - E[Y*]`.
    /// Requires `sum p_i ≈ 1.0` and `0 < E[Y*] <= 1.0 + 1e-4`.
    pub fn from_limited_discrete_pdf(
        limited_entry_ratios: &[f64],
        probabilities: &[f64],
    ) -> Result<Self, PricingError> {
        if limited_entry_ratios.len() != probabilities.len() || limited_entry_ratios.is_empty() {
            return Err(PricingError::DimensionMismatch(
                "limited_entry_ratios and probabilities must have matching non-empty length",
            ));
        }

        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(limited_entry_ratios.len());
        let mut sum_p = 0.0;
        let mut mean_y_star = 0.0;

        for (&y, &p) in limited_entry_ratios.iter().zip(probabilities.iter()) {
            if !y.is_finite() || !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "entry ratios and probabilities must be finite",
                ));
            }
            if !(y >= 0.0) || !(p >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "entry ratios and probabilities must be non-negative",
                ));
            }
            sum_p += p;
            mean_y_star += y * p;
            pairs.push((y, p));
        }

        if !sum_p.is_finite() || (sum_p - 1.0).abs() > 1e-4 {
            return Err(PricingError::InvalidParameter(
                "probabilities must sum to 1.0",
            ));
        }
        if !mean_y_star.is_finite() || !(mean_y_star > 0.0) || mean_y_star > 1.0 + 1e-4 {
            return Err(PricingError::InvalidParameter(
                "mean limited entry ratio E[Y*] must lie in (0, 1]",
            ));
        }

        let k = (1.0 - mean_y_star).max(0.0);

        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
        for (y, p) in pairs {
            if let Some(last) = merged.last_mut()
                && (last.0 - y).abs() <= 1e-15
            {
                last.1 += p;
                continue;
            }
            merged.push((y, p));
        }

        // O(N) cancellation-free right-to-left recurrence for E[max(0, Y* - r_i)].
        let n = merged.len();
        let mut rows = vec![
            TableLRow {
                entry_ratio: 0.0,
                charge: 0.0,
                aggregate_excess_charge: 0.0,
                savings: 0.0,
            };
            n
        ];
        let mut agg_excess = 0.0;
        let mut tail_prob = 0.0;
        for i in (0..n).rev() {
            let (r, _) = merged[i];
            if i + 1 < n {
                let (next_r, next_p) = merged[i + 1];
                tail_prob += next_p;
                agg_excess += tail_prob * (next_r - r);
            }
            let charge = agg_excess + k;
            let savings = (charge + r - 1.0).max(0.0);
            rows[i] = TableLRow {
                entry_ratio: r,
                charge,
                aggregate_excess_charge: agg_excess,
                savings,
            };
        }

        Ok(Self {
            loss_elimination_ratio: k,
            rows,
        })
    }

    /// Evaluates the Table L total insurance charge `phi*(r) = E[max(0, Y* - r)] + k`
    /// at any entry ratio `r >= 0`.
    pub fn charge_at(&self, r: f64) -> f64 {
        if !r.is_finite() {
            return f64::NAN;
        }
        if self.rows.is_empty() {
            return self.loss_elimination_ratio;
        }
        if r <= 0.0 {
            return self.rows[0].charge + self.rows[0].entry_ratio;
        }
        if r <= self.rows[0].entry_ratio {
            return self.rows[0].charge + (self.rows[0].entry_ratio - r);
        }
        let last = &self.rows[self.rows.len() - 1];
        if r >= last.entry_ratio {
            return self.loss_elimination_ratio;
        }

        let pos = match self
            .rows
            .binary_search_by(|row| row.entry_ratio.total_cmp(&r))
        {
            Ok(idx) => return self.rows[idx].charge,
            Err(idx) => idx,
        };

        let r0 = &self.rows[pos - 1];
        let r1 = &self.rows[pos];
        let t = (r - r0.entry_ratio) / (r1.entry_ratio - r0.entry_ratio);
        (1.0 - t) * r0.charge + t * r1.charge
    }

    /// Evaluates the aggregate-only excess charge `E[max(0, Y* - r)] = phi*(r) - k`.
    pub fn aggregate_excess_charge_at(&self, r: f64) -> f64 {
        let c = self.charge_at(r);
        if c.is_nan() {
            return f64::NAN;
        }
        (c - self.loss_elimination_ratio).max(0.0)
    }

    /// Evaluates Table L insurance savings `psi*(r) = E[max(0, r - Y*)] = phi*(r) + r - 1`.
    pub fn savings_at(&self, r: f64) -> f64 {
        if !r.is_finite() {
            return f64::NAN;
        }
        (self.charge_at(r) + r - 1.0).max(0.0)
    }
}

/// Solves for the balanced Basic Premium Factor (`BPF`) of a retrospective rating plan
/// with a **per-occurrence loss limit** using Skurnick's (1974) Table L.
///
/// Because `phi*(r_G)` embeds both the aggregate excess charge and the per-occurrence
/// Loss Elimination Ratio `k`, the exact actuarial balance equation is:
/// `BPF = ExpenseRatio - (LCF - 1.0) * ELR + [ phi*(r_G) - psi*(r_H) ] * LCF * ELR`
/// (with `r_G` and `r_H` measured against unlimited expected losses `ELR`).
pub fn solve_balanced_retro_plan_table_l(
    spec: &RetroPlanSpec,
    table_l: &TableLEngine,
    max_iter: usize,
    tol: f64,
) -> Result<RetroPlanResult, PricingError> {
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

    let mut bpf = spec.expense_ratio + table_l.loss_elimination_ratio * e_lcf;
    let mut converged = false;

    for _ in 0..max_iter {
        let r_g = (((spec.max_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);
        let r_h = (((spec.min_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);

        let phi_g = table_l.charge_at(r_g);
        let psi_h = table_l.savings_at(r_h);
        let net_charge = phi_g - psi_h;

        let new_bpf = non_loss_cost + net_charge * e_lcf;
        let diff = (new_bpf - bpf).abs();
        bpf = f64::midpoint(bpf, new_bpf);

        if diff <= tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(PricingError::NonConvergence(
            "Table L retro plan basic premium solver failed to converge",
        ));
    }

    let r_g_raw = ((spec.max_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf;
    if r_g_raw < 0.0 {
        return Err(PricingError::OutOfBounds(
            "infeasible Table L retro plan: max_retro_ratio / tax_multiplier is below balanced basic premium",
        ));
    }
    let r_h = (((spec.min_retro_ratio / spec.tax_multiplier) - bpf) / e_lcf).max(0.0);
    let phi_g = table_l.charge_at(r_g_raw);
    let psi_h = table_l.savings_at(r_h);
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
