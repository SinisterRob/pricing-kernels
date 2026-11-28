//! Classical (limited-fluctuation) credibility standards (Longley-Cook 1962 / CAS Exam 5).

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

fn validate_z_and_tolerance(z_score: f64, tolerance: f64) -> Result<f64, PricingError> {
    if !z_score.is_finite() || !(z_score > 0.0) {
        return Err(PricingError::InvalidParameter(
            "z_score must be finite and strictly positive",
        ));
    }
    if !tolerance.is_finite() || !(tolerance > 0.0) {
        return Err(PricingError::InvalidParameter(
            "tolerance must be finite and strictly positive",
        ));
    }
    let n0 = (z_score / tolerance).powi(2);
    if !n0.is_finite() {
        return Err(PricingError::OutOfBounds(
            "full credibility standard overflowed floating-point range",
        ));
    }
    Ok(n0)
}

fn validate_cv(cv_severity: f64) -> Result<f64, PricingError> {
    if !cv_severity.is_finite() || !(cv_severity >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "severity coefficient of variation must be finite and non-negative",
        ));
    }
    Ok(cv_severity)
}

/// Classical / Limited Fluctuation Credibility standard (CAS Exam 5 / Longley-Cook).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassicalCredibility {
    /// Number of claims required for full credibility (e.g., `1,082.41` for +/- 5% with 90% confidence,
    /// or `682.95` for +/- 7.5% with 95% confidence; `384.16` for +/- 10% with 95% confidence).
    pub full_credibility_standard: f64,
}

impl ClassicalCredibility {
    /// Claim-count (frequency) standard for Poisson frequency given two-sided normal quantile
    /// `z_score` (e.g. `1.645` for `p = 90%`) and tolerance `k` (e.g. `0.05`):
    /// `n_0 = (z_score / tolerance)^2` claims.
    ///
    /// Returns [`PricingError::InvalidParameter`] unless `z_score > 0` and `tolerance > 0`
    /// (a zero tolerance would otherwise give an infinite standard and `NaN` credibility).
    pub fn from_confidence_and_tolerance(
        z_score: f64,
        tolerance: f64,
    ) -> Result<Self, PricingError> {
        Ok(Self {
            full_credibility_standard: validate_z_and_tolerance(z_score, tolerance)?,
        })
    }

    /// Pure-premium (aggregate loss) standard in **expected claims** under Poisson frequency
    /// (Werner & Modlin *Basic Ratemaking*, Ch. 12; Mahler & Dean (2001)):
    /// `n_PP = n_0 * (1 + CV_S^2)`, where `CV_S` is the severity coefficient of variation.
    pub fn pure_premium_standard(
        z_score: f64,
        tolerance: f64,
        cv_severity: f64,
    ) -> Result<Self, PricingError> {
        let n0 = validate_z_and_tolerance(z_score, tolerance)?;
        let cv = validate_cv(cv_severity)?;
        Ok(Self {
            full_credibility_standard: n0 * (1.0 + cv * cv),
        })
    }

    /// Severity standard in **claims** (Mahler & Dean (2001)): `n_S = n_0 * CV_S^2`.
    ///
    /// Returns [`PricingError::InvalidParameter`] when `CV_S = 0` (degenerate standard).
    pub fn severity_standard(
        z_score: f64,
        tolerance: f64,
        cv_severity: f64,
    ) -> Result<Self, PricingError> {
        let n0 = validate_z_and_tolerance(z_score, tolerance)?;
        let cv = validate_cv(cv_severity)?;
        if !(cv > 0.0) {
            return Err(PricingError::InvalidParameter(
                "severity standard requires a strictly positive severity CV",
            ));
        }
        Ok(Self {
            full_credibility_standard: n0 * cv * cv,
        })
    }

    /// Default 1,082.41 claim standard (90% probability within +/- 5% error under Poisson).
    pub const fn standard_90_5() -> Self {
        Self {
            full_credibility_standard: 1082.41,
        }
    }

    /// Compute partial credibility `Z = sqrt(min(1.0, observed_volume / full_credibility_standard))`.
    ///
    /// `observed_volume = +inf` returns the well-defined limit `Z = 1` (full credibility).
    ///
    /// # Errors
    /// [`PricingError::InvalidParameter`] when `observed_volume` is `NaN` or negative, or the
    /// full-credibility standard is not finite and strictly positive.
    #[inline]
    pub fn credibility(&self, observed_volume: f64) -> Result<f64, PricingError> {
        if !self.full_credibility_standard.is_finite() || !(self.full_credibility_standard > 0.0) {
            return Err(PricingError::InvalidParameter(
                "full credibility standard must be finite and strictly positive",
            ));
        }
        if !(observed_volume >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "observed credibility volume must be non-negative and not NaN",
            ));
        }
        Ok(if observed_volume >= self.full_credibility_standard {
            1.0
        } else {
            (observed_volume / self.full_credibility_standard).sqrt()
        })
    }

    /// Blend observed indication with prior complement:
    /// `Blended = Z * observed + (1 - Z) * prior`.
    ///
    /// # Errors
    /// Propagates [`Self::credibility`] errors; [`PricingError::InvalidParameter`] when
    /// `observed` or `prior` is not finite.
    #[inline]
    pub fn blend(
        &self,
        observed_volume: f64,
        observed: f64,
        prior: f64,
    ) -> Result<f64, PricingError> {
        if !observed.is_finite() || !prior.is_finite() {
            return Err(PricingError::InvalidParameter(
                "observed indication and prior complement must be finite",
            ));
        }
        let z = self.credibility(observed_volume)?;
        Ok(z * observed + (1.0 - z) * prior)
    }
}
