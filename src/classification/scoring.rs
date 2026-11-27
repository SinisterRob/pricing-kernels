//! Zero-allocation multiplicative rating plan scorer for real-time and batch pricing.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// High-performance multi-factor scoring table for commercial rating plans.
///
/// Models the standard actuarial rating structure:
/// `Premium = BaseRate * Exposure * Product_k(Factor_k)`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiplicativeRater {
    /// Base rate per unit of exposure (e.g., rate per power unit or per $100 payroll).
    pub base_rate: f64,
    /// Minimum premium floor.
    pub minimum_premium: Option<f64>,
    /// Maximum premium cap.
    pub maximum_premium: Option<f64>,
}

impl MultiplicativeRater {
    /// Construct a rater with given base rate.
    pub const fn new(base_rate: f64) -> Self {
        Self {
            base_rate,
            minimum_premium: None,
            maximum_premium: None,
        }
    }

    /// Set a minimum premium floor.
    pub const fn with_minimum_premium(mut self, min_prem: f64) -> Self {
        self.minimum_premium = Some(min_prem);
        self
    }

    /// Set a maximum premium cap.
    pub const fn with_maximum_premium(mut self, max_prem: f64) -> Self {
        self.maximum_premium = Some(max_prem);
        self
    }

    /// Evaluates policy premium for a given exposure and slice of multiplicative factors.
    /// Executes in single-digit nanoseconds with zero dynamic memory allocation.
    #[inline]
    pub fn score(&self, exposure: f64, factors: &[f64]) -> f64 {
        let mut prem = self.base_rate * exposure;
        for &f in factors {
            prem *= f;
        }
        if let Some(min_p) = self.minimum_premium
            && prem < min_p
        {
            prem = min_p;
        }
        if let Some(max_p) = self.maximum_premium
            && prem > max_p
        {
            prem = max_p;
        }
        prem
    }

    /// Batch score a slice of exposure-factor tuples into an output slice `results`.
    pub fn batch_score_into(
        &self,
        exposures: &[f64],
        factor_matrix: &[&[f64]],
        results: &mut [f64],
    ) -> Result<(), PricingError> {
        if exposures.len() != factor_matrix.len() || results.len() != exposures.len() {
            return Err(PricingError::DimensionMismatch(
                "exposures, factor_matrix, and results must have identical lengths",
            ));
        }
        for ((&exp, &factors), out) in exposures
            .iter()
            .zip(factor_matrix.iter())
            .zip(results.iter_mut())
        {
            *out = self.score(exp, factors);
        }
        Ok(())
    }
}
