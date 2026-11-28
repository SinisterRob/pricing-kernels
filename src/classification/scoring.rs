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
    /// Construct a rater with given base rate (unvalidated; see [`Self::try_new`]).
    pub const fn new(base_rate: f64) -> Self {
        Self {
            base_rate,
            minimum_premium: None,
            maximum_premium: None,
        }
    }

    /// Construct a rater with a validated finite, strictly positive base rate.
    pub fn try_new(base_rate: f64) -> Result<Self, PricingError> {
        let rater = Self::new(base_rate);
        rater.validate()?;
        Ok(rater)
    }

    /// Validate the rating plan: `base_rate` finite and `> 0`; `minimum_premium` /
    /// `maximum_premium` finite and `>= 0`; and `minimum_premium <= maximum_premium`.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.base_rate.is_finite() || !(self.base_rate > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "base_rate must be finite and > 0",
            ));
        }
        for bound in [self.minimum_premium, self.maximum_premium]
            .into_iter()
            .flatten()
        {
            if !bound.is_finite() || !(bound >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "minimum/maximum premium must be finite and >= 0",
                ));
            }
        }
        if let (Some(min_p), Some(max_p)) = (self.minimum_premium, self.maximum_premium)
            && min_p > max_p
        {
            return Err(PricingError::InvalidParameter(
                "minimum_premium must not exceed maximum_premium",
            ));
        }
        Ok(())
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
    ///
    /// Unchecked hot path: inputs are not validated (NaN propagates; with
    /// `minimum_premium > maximum_premium` the cap wins). Use [`Self::try_score`] for
    /// validated scoring.
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

    /// Validated premium: checks the plan ([`Self::validate`]), that `exposure` is finite and
    /// `>= 0`, and that every factor is finite and `>= 0`, then returns [`Self::score`].
    pub fn try_score(&self, exposure: f64, factors: &[f64]) -> Result<f64, PricingError> {
        self.validate()?;
        Self::validate_policy(exposure, factors)?;
        Ok(self.score(exposure, factors))
    }

    #[inline]
    fn validate_policy(exposure: f64, factors: &[f64]) -> Result<(), PricingError> {
        if !exposure.is_finite() || !(exposure >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "exposure must be finite and >= 0",
            ));
        }
        if factors.iter().any(|f| !f.is_finite() || !(*f >= 0.0)) {
            return Err(PricingError::InvalidParameter(
                "rating factors must be finite and >= 0",
            ));
        }
        Ok(())
    }

    /// Batch score a slice of exposure-factor tuples into an output slice `results`
    /// with the same validation as [`Self::try_score`] (plan validated once).
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
        self.validate()?;
        for ((&exp, &factors), out) in exposures
            .iter()
            .zip(factor_matrix.iter())
            .zip(results.iter_mut())
        {
            Self::validate_policy(exp, factors)?;
            *out = self.score(exp, factors);
        }
        Ok(())
    }
}
