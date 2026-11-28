//! Lee (1988) Table M aggregate insurance charge and savings engine.

use crate::error::PricingError;
use crate::underwriting_risk::AggregateLossDistribution;
use serde::{Deserialize, Serialize};

/// Maximum relative deviation of the input mean entry ratio from `1.0` accepted by
/// [`TableMEngine::from_discrete_pdf`]; inside this band entry ratios are rescaled so that
/// `E[Y] = 1` holds to machine precision, outside it the input is rejected.
pub const MEAN_ENTRY_RATIO_BAND: f64 = 0.05;

/// Maximum absolute deviation of `sum p_k` from `1.0` accepted before renormalization.
pub const PROBABILITY_SUM_TOL: f64 = 1e-4;

/// Tolerance used when validating Lee's identities on externally supplied rows
/// (`phi(0) = 1`, `psi(r) = phi(r) + r - 1`) in [`TableMEngine::from_rows`] and
/// [`super::TableLEngine::from_rows`].
pub const TABLE_IDENTITY_TOL: f64 = 1e-9;

/// Entry ratios closer than this are merged into a single support point.
pub(crate) const ENTRY_RATIO_MERGE_TOL: f64 = 1e-15;

/// Sort ascending by entry ratio and merge (numerically) identical support points.
pub(crate) fn sort_and_merge(mut pairs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
    for (r, p) in pairs {
        if let Some(last) = merged.last_mut()
            && (last.0 - r).abs() <= ENTRY_RATIO_MERGE_TOL
        {
            last.1 += p;
            continue;
        }
        merged.push((r, p));
    }
    merged
}

/// Cancellation-free `O(N)` recurrences over a sorted, merged discrete pmf `(y_i, p_i)`:
/// returns `(E[(Y - y_i)+], E[(y_i - Y)+])` for each support point.
///
/// `phi(y_i) = phi(y_{i+1}) + P(Y > y_i) (y_{i+1} - y_i)` (right to left) and
/// `psi(y_i) = psi(y_{i-1}) + P(Y <= y_{i-1}) (y_i - y_{i-1})` (left to right).
pub(crate) fn excess_and_shortfall(merged: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = merged.len();
    let mut out = vec![(0.0, 0.0); n];
    let mut excess = 0.0;
    let mut tail_prob = 0.0;
    for i in (0..n).rev() {
        if i + 1 < n {
            tail_prob += merged[i + 1].1;
            excess += tail_prob * (merged[i + 1].0 - merged[i].0);
        }
        out[i].0 = excess;
    }
    let mut shortfall = 0.0;
    let mut cdf = 0.0;
    for i in 0..n {
        if i > 0 {
            cdf += merged[i - 1].1;
            shortfall += cdf * (merged[i].0 - merged[i - 1].0);
        }
        out[i].1 = shortfall;
    }
    out
}

/// Discrete Table M aggregate distribution row.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TableMRow {
    /// Entry ratio `r = Loss / ExpectedLoss`.
    pub entry_ratio: f64,
    /// Insurance charge `phi(r) = E[max(0, Y - r)]`.
    pub charge: f64,
    /// Insurance savings `psi(r) = phi(r) + r - 1 = E[max(0, r - Y)]`.
    pub savings: f64,
}

/// Table M aggregate loss charge engine based on Lee (1988) PCAS.
///
/// Rows are private so the sorted, non-empty grid invariant required by the binary-search
/// evaluator cannot be broken; use [`TableMEngine::rows`] for read access. Deserialization is
/// validated through [`TableMEngine::from_rows`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TableMEngineRaw")]
pub struct TableMEngine {
    rows: Vec<TableMRow>,
}

#[derive(Deserialize)]
struct TableMEngineRaw {
    rows: Vec<TableMRow>,
}

impl TryFrom<TableMEngineRaw> for TableMEngine {
    type Error = PricingError;
    fn try_from(raw: TableMEngineRaw) -> Result<Self, Self::Error> {
        Self::from_rows(raw.rows)
    }
}

impl TableMEngine {
    /// Sorted (strictly increasing entry ratio), non-empty grid of Table M rows.
    #[inline]
    pub fn rows(&self) -> &[TableMRow] {
        &self.rows
    }

    /// Build Table M from pre-computed rows, validating that the grid is non-empty, finite,
    /// strictly increasing in entry ratio, has non-negative non-increasing charges, and satisfies
    /// `phi(0) = 1` and `psi(r) = phi(r) + r - 1` within [`TABLE_IDENTITY_TOL`].
    pub fn from_rows(rows: Vec<TableMRow>) -> Result<Self, PricingError> {
        let Some(first) = rows.first() else {
            return Err(PricingError::EmptyInput("Table M rows must be non-empty"));
        };
        // phi(0) = phi(r_0) + r_0 because F(r) = 0 below the first support point.
        if (first.charge + first.entry_ratio - 1.0).abs() > TABLE_IDENTITY_TOL {
            return Err(PricingError::InvalidParameter(
                "Table M rows violate phi(0) = 1 (mean entry ratio must be 1)",
            ));
        }
        let mut prev: Option<&TableMRow> = None;
        for row in &rows {
            if !row.entry_ratio.is_finite()
                || !row.charge.is_finite()
                || !row.savings.is_finite()
                || !(row.entry_ratio >= 0.0)
                || !(row.charge >= 0.0)
                || !(row.savings >= 0.0)
            {
                return Err(PricingError::InvalidParameter(
                    "Table M rows must be finite and non-negative",
                ));
            }
            if (row.savings - (row.charge + row.entry_ratio - 1.0)).abs() > TABLE_IDENTITY_TOL {
                return Err(PricingError::InvalidParameter(
                    "Table M rows violate psi(r) = phi(r) + r - 1",
                ));
            }
            if let Some(p) = prev
                && (!(row.entry_ratio > p.entry_ratio) || row.charge > p.charge)
            {
                return Err(PricingError::InvalidParameter(
                    "Table M rows must be strictly increasing in entry ratio with non-increasing charge",
                ));
            }
            prev = Some(row);
        }
        Ok(Self { rows })
    }

    /// Build Table M directly from a discrete probability distribution of entry ratios `(y_k, p_k)`.
    ///
    /// Automatically sorts entry ratios in ascending order and merges duplicates.
    ///
    /// Normalization: probabilities must sum to `1` within [`PROBABILITY_SUM_TOL`] and are
    /// renormalized to sum exactly to `1`; the mean entry ratio must lie within
    /// [`MEAN_ENTRY_RATIO_BAND`] of `1`, and entry ratios are then **rescaled** to `y_k / E[Y]`
    /// so that `E[Y] = 1` exactly (to rounding). Inputs outside either band are rejected.
    ///
    /// Enforces Lee's fundamental relations: `phi(0) = 1`,
    /// `d(phi)/dr = -(1 - F(r)) <= 0` (monotonically non-increasing and convex) and
    /// `psi(r) = phi(r) + r - 1` (both `phi` and `psi` are computed directly by
    /// cancellation-free recurrences, so the identity holds to rounding).
    pub fn from_discrete_pdf(
        entry_ratios: &[f64],
        probabilities: &[f64],
    ) -> Result<Self, PricingError> {
        if entry_ratios.len() != probabilities.len() || entry_ratios.is_empty() {
            return Err(PricingError::DimensionMismatch(
                "entry ratios and probabilities must have matching non-empty length",
            ));
        }

        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(entry_ratios.len());
        let mut sum_p = 0.0;
        let mut sum_rp = 0.0;

        for (&r, &p) in entry_ratios.iter().zip(probabilities.iter()) {
            if !r.is_finite() || !p.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "entry ratios and probabilities must be finite",
                ));
            }
            if !(r >= 0.0) || !(p >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "entry ratios and probabilities must be non-negative",
                ));
            }
            sum_p += p;
            sum_rp += r * p;
            pairs.push((r, p));
        }

        if !sum_p.is_finite() || (sum_p - 1.0).abs() > PROBABILITY_SUM_TOL {
            return Err(PricingError::InvalidParameter(
                "probabilities must sum to 1.0",
            ));
        }
        let mean_r = sum_rp / sum_p;
        if !mean_r.is_finite() || (mean_r - 1.0).abs() > MEAN_ENTRY_RATIO_BAND {
            return Err(PricingError::InvalidParameter(
                "mean entry ratio must be within MEAN_ENTRY_RATIO_BAND of 1.0",
            ));
        }
        for pair in &mut pairs {
            pair.0 /= mean_r;
            pair.1 /= sum_p;
        }

        let merged = sort_and_merge(pairs);
        let rows = merged
            .iter()
            .zip(excess_and_shortfall(&merged))
            .map(|(&(r, _), (charge, savings))| TableMRow {
                entry_ratio: r,
                charge,
                savings,
            })
            .collect();

        Ok(Self { rows })
    }

    /// Build Table M directly from a discrete [`AggregateLossDistribution`] produced by
    /// Panjer recursion, FFT convolution, or Cat ELT aggregation.
    ///
    /// Retains strictly positive probability masses, renormalizes them to sum to `1` (tail
    /// truncation up to `1e-3` is tolerated), and normalizes each grid point `x_k = k * agg.step`
    /// into entry ratio `r_k = x_k / m`, where `m = sum_k x_k p_k / sum_k p_k` is the mean of the
    /// **renormalized retained pmf** (not `agg.mean`, which is only checked to be finite and
    /// positive), so that `E[r] = 1` and `phi(0) = 1` hold exactly.
    pub fn from_aggregate(agg: &AggregateLossDistribution) -> Result<Self, PricingError> {
        if !agg.mean.is_finite() || !(agg.mean > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "aggregate distribution mean must be finite and positive to normalize entry ratios",
            ));
        }
        let mut xs = Vec::with_capacity(agg.pmf.len());
        let mut probs = Vec::with_capacity(agg.pmf.len());
        let mut total_p = 0.0;
        for (k, &p) in agg.pmf.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(PricingError::InvalidParameter(
                    "aggregate pmf probabilities must be finite and non-negative",
                ));
            }
            if p > 0.0 {
                xs.push((k as f64) * agg.step);
                probs.push(p);
                total_p += p;
            }
        }
        if !(total_p > 0.0) {
            return Err(PricingError::EmptyInput(
                "aggregate distribution has no positive probability mass",
            ));
        }
        if (total_p - 1.0).abs() > 1e-3 {
            return Err(PricingError::InvalidParameter(
                "aggregate distribution tail truncation exceeds 1e-3; increase grid size",
            ));
        }
        // Renormalize small numerical tail mass first, then compute exact normalized mean so E[r] == 1.0.
        let mut norm_mean = 0.0;
        for (x, p) in xs.iter().zip(probs.iter_mut()) {
            *p /= total_p;
            norm_mean += *x * *p;
        }
        if !norm_mean.is_finite() || !(norm_mean > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "normalized aggregate distribution mean must be positive",
            ));
        }
        let ratios: Vec<f64> = xs.into_iter().map(|x| x / norm_mean).collect();
        Self::from_discrete_pdf(&ratios, &probs)
    }

    /// Evaluates insurance charge `phi(r)` at any entry ratio `r >= 0` using exact piecewise-linear
    /// evaluation for discrete distributions (values `r <= 0` evaluate at `r = 0`).
    ///
    /// For `0 <= r < rows[0].entry_ratio`, `P(Y >= rows[0].entry_ratio) = 1`, so
    /// `phi'(r) = -1` and `phi(r) = rows[0].charge + (rows[0].entry_ratio - r)`.
    pub fn charge_at(&self, r: f64) -> f64 {
        if !r.is_finite() {
            return f64::NAN;
        }
        let (Some(first), Some(last)) = (self.rows.first(), self.rows.last()) else {
            return 0.0;
        };
        if r <= 0.0 {
            return first.charge + first.entry_ratio;
        }
        if r <= first.entry_ratio {
            return first.charge + (first.entry_ratio - r);
        }
        if r >= last.entry_ratio {
            return 0.0;
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

    /// Evaluates insurance savings `psi(r)` at any entry ratio `r`: `psi(r) = phi(r) + r - 1`.
    pub fn savings_at(&self, r: f64) -> f64 {
        if !r.is_finite() {
            return f64::NAN;
        }
        (self.charge_at(r) + r - 1.0).max(0.0)
    }
}
