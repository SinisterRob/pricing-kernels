//! Lee (1988) Table M aggregate insurance charge and savings engine.

use crate::error::PricingError;
use crate::underwriting_risk::AggregateLossDistribution;
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableMEngine {
    /// Sorted grid of entry ratios and corresponding charges.
    pub rows: Vec<TableMRow>,
}

impl TableMEngine {
    /// Build Table M directly from a discrete probability distribution of entry ratios `(y_k, p_k)`.
    ///
    /// Automatically sorts entry ratios in ascending order and merges duplicates.
    /// Enforces Lee's fundamental relations:
    /// `d(phi)/dr = -(1 - F(r)) <= 0` (monotonically non-increasing and convex) and
    /// `psi(r) = phi(r) + r - 1`.
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
        let mut mean_r = 0.0;

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
            mean_r += r * p;
            pairs.push((r, p));
        }

        if !sum_p.is_finite() || (sum_p - 1.0).abs() > 1e-4 {
            return Err(PricingError::InvalidParameter(
                "probabilities must sum to 1.0",
            ));
        }
        if !mean_r.is_finite() || (mean_r - 1.0).abs() > 0.05 {
            return Err(PricingError::InvalidParameter(
                "mean entry ratio must be approximately 1.0",
            ));
        }

        // Sort ascending by entry ratio and merge identical points.
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
        for (r, p) in pairs {
            if let Some(last) = merged.last_mut()
                && (last.0 - r).abs() <= 1e-15
            {
                last.1 += p;
                continue;
            }
            merged.push((r, p));
        }

        // O(N) cancellation-free right-to-left recurrence:
        // phi(r_i) = phi(r_{i+1}) + P(Y > r_i) * (r_{i+1} - r_i).
        let n = merged.len();
        let mut rows = vec![
            TableMRow {
                entry_ratio: 0.0,
                charge: 0.0,
                savings: 0.0,
            };
            n
        ];
        let mut charge = 0.0;
        let mut tail_prob = 0.0;
        for i in (0..n).rev() {
            let (r, _) = merged[i];
            if i + 1 < n {
                let (next_r, next_p) = merged[i + 1];
                tail_prob += next_p;
                charge += tail_prob * (next_r - r);
            }
            let savings = (charge + r - 1.0).max(0.0);
            rows[i] = TableMRow {
                entry_ratio: r,
                charge,
                savings,
            };
        }

        Ok(Self { rows })
    }

    /// Build Table M directly from a discrete [`AggregateLossDistribution`] produced by
    /// Panjer recursion, FFT convolution, or Cat ELT aggregation.
    ///
    /// Normalizes each grid point `x_k = k * agg.step` into entry ratio `r_k = x_k / agg.mean`
    /// and retains strictly positive probability masses.
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
        if self.rows.is_empty() {
            return 0.0;
        }
        if r <= 0.0 {
            return self.rows[0].charge + self.rows[0].entry_ratio;
        }
        if r <= self.rows[0].entry_ratio {
            return self.rows[0].charge + (self.rows[0].entry_ratio - r);
        }
        let last = &self.rows[self.rows.len() - 1];
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
