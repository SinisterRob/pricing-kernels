use serde::{Deserialize, Serialize};

/// High-performance multi-factor scoring table for commercial rating plans.
///
/// Models the standard actuarial rating structure:
/// Premium = BaseRate * Exposure * Product_k(Factor_k)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiplicativeRater {
    /// Base rate per unit of exposure (e.g., rate per power unit or per $100 payroll).
    pub base_rate: f64,
    /// Minimum premium floor.
    pub minimum_premium: Option<f64>,
    /// Maximum premium cap.
    pub maximum_premium: Option<f64>,
}

impl MultiplicativeRater {
    pub fn new(base_rate: f64) -> Self {
        Self {
            base_rate,
            minimum_premium: None,
            maximum_premium: None,
        }
    }

    pub fn with_minimum_premium(mut self, min_prem: f64) -> Self {
        self.minimum_premium = Some(min_prem);
        self
    }

    pub fn with_maximum_premium(mut self, max_prem: f64) -> Self {
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
        if let Some(min_p) = self.minimum_premium {
            #[allow(clippy::collapsible_if)]
            if prem < min_p {
                prem = min_p;
            }
        }
        if let Some(max_p) = self.maximum_premium {
            #[allow(clippy::collapsible_if)]
            if prem > max_p {
                prem = max_p;
            }
        }
        prem
    }

    /// Batch score a slice of exposure-factor tuples into an output slice `results`.
    pub fn batch_score_into(
        &self,
        exposures: &[f64],
        factor_matrix: &[&[f64]],
        results: &mut [f64],
    ) {
        for (i, (&exp, &factors)) in exposures.iter().zip(factor_matrix.iter()).enumerate() {
            results[i] = self.score(exp, factors);
        }
    }
}
