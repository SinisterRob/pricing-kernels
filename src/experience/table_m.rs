use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Discrete Table M aggregate distribution row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableMRow {
    /// Entry ratio r = Loss / ExpectedLoss.
    pub entry_ratio: f64,
    /// Insurance charge phi(r) = E[max(0, Y - r)].
    pub charge: f64,
    /// Insurance savings psi(r) = phi(r) + r - 1 = E[max(0, r - Y)].
    pub savings: f64,
}

/// Table M aggregate loss charge engine based on Lee (1988) PCAS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableMEngine {
    /// Sorted grid of entry ratios and corresponding charges.
    pub rows: Vec<TableMRow>,
}

impl TableMEngine {
    /// Build Table M directly from a discrete probability distribution of entry ratios (y_k, p_k).
    /// Enforces Lee's fundamental relations:
    /// d(phi)/dr = -(1 - F(r)) <= 0 (monotonically non-increasing and convex).
    pub fn from_discrete_pdf(entry_ratios: &[f64], probabilities: &[f64]) -> Result<Self, PricingError> {
        if entry_ratios.len() != probabilities.len() || entry_ratios.is_empty() {
            return Err(PricingError::DimensionMismatch("entry ratios and probabilities must have matching length"));
        }

        // Verify probabilities sum to ~1.0 and expected entry ratio is ~1.0
        let sum_p: f64 = probabilities.iter().sum();
        if (sum_p - 1.0).abs() > 1e-4 {
            return Err(PricingError::InvalidParameter("probabilities must sum to 1.0"));
        }

        let mean_r: f64 = entry_ratios.iter().zip(probabilities.iter())
            .map(|(&r, &p)| r * p)
            .sum();
        if (mean_r - 1.0).abs() > 0.05 {
            return Err(PricingError::InvalidParameter("mean entry ratio must be approximately 1.0"));
        }

        let mut rows = Vec::with_capacity(entry_ratios.len());
        for &r in entry_ratios {
            // phi(r) = sum_k p_k * max(0, y_k - r)
            let mut charge = 0.0;
            for (&y, &p) in entry_ratios.iter().zip(probabilities.iter()) {
                if y > r {
                    charge += p * (y - r);
                }
            }
            // Lee identity: psi(r) = phi(r) + r - 1.0
            let savings = charge + r - 1.0;
            rows.push(TableMRow {
                entry_ratio: r,
                charge,
                savings: savings.max(0.0),
            });
        }

        Ok(Self { rows })
    }

    /// Evaluates insurance charge phi(r) at any entry ratio r using linear interpolation.
    pub fn charge_at(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 1.0;
        }
        if self.rows.is_empty() {
            return 0.0;
        }
        if r <= self.rows[0].entry_ratio {
            return self.rows[0].charge;
        }
        let last = &self.rows[self.rows.len() - 1];
        if r >= last.entry_ratio {
            return 0.0;
        }

        // Binary search for bracket
        let pos = match self.rows.binary_search_by(|row| row.entry_ratio.partial_cmp(&r).unwrap()) {
            Ok(idx) => return self.rows[idx].charge,
            Err(idx) => idx,
        };

        let r0 = &self.rows[pos - 1];
        let r1 = &self.rows[pos];
        let t = (r - r0.entry_ratio) / (r1.entry_ratio - r0.entry_ratio);
        (1.0 - t) * r0.charge + t * r1.charge
    }

    /// Evaluates insurance savings psi(r) at any entry ratio r: psi(r) = phi(r) + r - 1.
    pub fn savings_at(&self, r: f64) -> f64 {
        (self.charge_at(r) + r - 1.0).max(0.0)
    }
}
