//! Skurnick (1974) Table L aggregate loss charge engine with per-occurrence loss limitations
//! and balanced loss-limited retrospective rating plan solver.
//!
//! Reference: D. Skurnick (1974), *"Measuring the Effect of Loss Limitations on Retrospective
//! Rating"*, *Proceedings of the Casualty Actuarial Society (PCAS)*, Vol. LXI, pp. 1–28.

use super::retrospective::{RetroPlanResult, RetroPlanSpec, solve_balanced_retro_plan_with};
use super::table_m::{
    PROBABILITY_SUM_TOL, TABLE_IDENTITY_TOL, excess_and_shortfall, sort_and_merge,
};
use crate::error::{PricingError, SolverOptions};
use crate::underwriting_risk::AggregateLossDistribution;
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
///
/// `rows` is private so the sorted, non-empty grid invariant required by the binary-search
/// evaluator cannot be broken; use [`TableLEngine::rows`]. Deserialization is validated through
/// [`TableLEngine::from_rows`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TableLEngineRaw")]
pub struct TableLEngine {
    /// Per-occurrence Loss Elimination Ratio `k = 1 - E[Y*] in [0, 1)`.
    pub loss_elimination_ratio: f64,
    /// Sorted grid of Table L rows.
    rows: Vec<TableLRow>,
}

#[derive(Deserialize)]
struct TableLEngineRaw {
    loss_elimination_ratio: f64,
    rows: Vec<TableLRow>,
}

impl TryFrom<TableLEngineRaw> for TableLEngine {
    type Error = PricingError;
    fn try_from(raw: TableLEngineRaw) -> Result<Self, Self::Error> {
        Self::from_rows(raw.loss_elimination_ratio, raw.rows)
    }
}

impl TableLEngine {
    /// Sorted (strictly increasing entry ratio), non-empty grid of Table L rows.
    #[inline]
    pub fn rows(&self) -> &[TableLRow] {
        &self.rows
    }

    /// Build Table L from pre-computed rows and LER `k in [0, 1)`, validating a non-empty,
    /// finite, strictly increasing grid with non-increasing charges and the Skurnick identities
    /// `phi*(0) = 1`, `phi*(r) = E[(Y* - r)+] + k`, `psi*(r) = phi*(r) + r - 1` within
    /// [`TABLE_IDENTITY_TOL`].
    pub fn from_rows(
        loss_elimination_ratio: f64,
        rows: Vec<TableLRow>,
    ) -> Result<Self, PricingError> {
        let k = loss_elimination_ratio;
        if !k.is_finite() || !(k >= 0.0) || !(k < 1.0) {
            return Err(PricingError::InvalidParameter(
                "Table L loss elimination ratio must be finite and in [0, 1)",
            ));
        }
        let Some(first) = rows.first() else {
            return Err(PricingError::EmptyInput("Table L rows must be non-empty"));
        };
        if (first.charge + first.entry_ratio - 1.0).abs() > TABLE_IDENTITY_TOL {
            return Err(PricingError::InvalidParameter(
                "Table L rows violate phi*(0) = 1",
            ));
        }
        let mut prev: Option<&TableLRow> = None;
        for row in &rows {
            if !row.entry_ratio.is_finite()
                || !row.charge.is_finite()
                || !row.aggregate_excess_charge.is_finite()
                || !row.savings.is_finite()
                || !(row.entry_ratio >= 0.0)
                || !(row.aggregate_excess_charge >= 0.0)
                || !(row.savings >= 0.0)
            {
                return Err(PricingError::InvalidParameter(
                    "Table L rows must be finite and non-negative",
                ));
            }
            if (row.charge - (row.aggregate_excess_charge + k)).abs() > TABLE_IDENTITY_TOL
                || (row.savings - (row.charge + row.entry_ratio - 1.0)).abs() > TABLE_IDENTITY_TOL
            {
                return Err(PricingError::InvalidParameter(
                    "Table L rows violate phi* = excess + k or psi* = phi* + r - 1",
                ));
            }
            if let Some(p) = prev
                && (!(row.entry_ratio > p.entry_ratio) || row.charge > p.charge)
            {
                return Err(PricingError::InvalidParameter(
                    "Table L rows must be strictly increasing in entry ratio with non-increasing charge",
                ));
            }
            prev = Some(row);
        }
        Ok(Self {
            loss_elimination_ratio: k,
            rows,
        })
    }

    /// Build Table L from a discrete probability distribution of **limited** aggregate
    /// entry ratios `(y*_i, p_i)` normalized by **unlimited** expected loss.
    ///
    /// Automatically sorts entry ratios, merges duplicates, and infers `k = 1 - E[Y*]`.
    /// Requires `sum p_i ≈ 1.0` and `0 < E[Y*] <= 1.0 + PROBABILITY_SUM_TOL`. Probabilities are
    /// renormalized to sum to `1`; when `1 < E[Y*] <= 1 + PROBABILITY_SUM_TOL` (rounding noise in
    /// the limited mean), entry ratios are rescaled to `y*_i / E[Y*]` exactly as in
    /// [`super::TableMEngine::from_discrete_pdf`], giving `k = 0` and `phi*(0) = 1`.
    ///
    /// The result is re-validated through [`Self::from_rows`], so any successfully constructed
    /// engine also round-trips through serde.
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

        if !sum_p.is_finite() || (sum_p - 1.0).abs() > PROBABILITY_SUM_TOL {
            return Err(PricingError::InvalidParameter(
                "probabilities must sum to 1.0",
            ));
        }
        // Renormalize so that sum p = 1 exactly; otherwise phi*(0) = E[Y*] + k != 1.
        let mean_y_star = mean_y_star / sum_p;
        if !mean_y_star.is_finite()
            || !(mean_y_star > 0.0)
            || mean_y_star > 1.0 + PROBABILITY_SUM_TOL
        {
            return Err(PricingError::InvalidParameter(
                "mean limited entry ratio E[Y*] must lie in (0, 1]",
            ));
        }
        // E[Y*] slightly above 1 is rounding noise: rescale so E[Y*] = 1 (k = 0), matching
        // Table M; otherwise phi*(0) = E[Y*] + k would exceed 1 and from_rows would reject.
        let y_scale = if mean_y_star > 1.0 { mean_y_star } else { 1.0 };
        for pair in &mut pairs {
            pair.0 /= y_scale;
            pair.1 /= sum_p;
        }

        let k = (1.0 - mean_y_star).max(0.0);

        let merged = sort_and_merge(pairs);
        // E[(Y* - r)+] and E[(r - Y*)+] via cancellation-free recurrences; psi* = E[(r - Y*)+].
        let rows: Vec<TableLRow> = merged
            .iter()
            .zip(excess_and_shortfall(&merged))
            .map(|(&(r, _), (agg_excess, savings))| TableLRow {
                entry_ratio: r,
                charge: agg_excess + k,
                aggregate_excess_charge: agg_excess,
                savings,
            })
            .collect();

        Self::from_rows(k, rows)
    }

    /// Build Skurnick (1974) Table L directly from a per-occurrence-limited
    /// [`AggregateLossDistribution`] and the corresponding `unlimited_expected_loss > 0`.
    ///
    /// Normalizes each grid point `x*_k = k * limited_agg.step()` into limited entry ratio
    /// `y*_k = x*_k / unlimited_expected_loss`, automatically inferring the per-occurrence
    /// Loss Elimination Ratio `k = 1 - E[Y*]`.
    pub fn from_limited_aggregate(
        limited_agg: &AggregateLossDistribution,
        unlimited_expected_loss: f64,
    ) -> Result<Self, PricingError> {
        if !unlimited_expected_loss.is_finite() || !(unlimited_expected_loss > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "unlimited_expected_loss must be finite and positive",
            ));
        }
        let mut ratios = Vec::with_capacity(limited_agg.pmf().len());
        let mut probs = Vec::with_capacity(limited_agg.pmf().len());
        let mut total_p = 0.0;
        for (k, &p) in limited_agg.pmf().iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(PricingError::InvalidParameter(
                    "aggregate pmf probabilities must be finite and non-negative",
                ));
            }
            if p > 0.0 {
                ratios.push((k as f64) * limited_agg.step() / unlimited_expected_loss);
                probs.push(p);
                total_p += p;
            }
        }
        if !(total_p > 0.0) {
            return Err(PricingError::EmptyInput(
                "limited aggregate distribution has no positive probability mass",
            ));
        }
        if (total_p - 1.0).abs() > 1e-3 {
            return Err(PricingError::InvalidParameter(
                "limited aggregate distribution tail truncation exceeds 1e-3; increase grid size",
            ));
        }
        for p in &mut probs {
            *p /= total_p;
        }
        Self::from_limited_discrete_pdf(&ratios, &probs)
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
    options: &SolverOptions,
) -> Result<RetroPlanResult, PricingError> {
    solve_balanced_retro_plan_with(
        spec,
        table_l.loss_elimination_ratio,
        |r| table_l.charge_at(r),
        |r| table_l.savings_at(r),
        *options,
        "Table L retro plan basic premium solver failed to converge",
        "infeasible Table L retro plan: max_retro_ratio / tax_multiplier is below balanced basic premium",
    )
}
