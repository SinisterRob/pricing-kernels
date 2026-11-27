use serde::{Deserialize, Serialize};

/// Classical / Limited Fluctuation Credibility standard (CAS Exam 5 / Longley-Cook).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassicalCredibility {
    /// Number of claims required for full credibility (e.g., `1,082.41` for +/- 5% with 90% confidence,
    /// or `683` for +/- 10% with 95% confidence).
    pub full_credibility_standard: f64,
}

impl ClassicalCredibility {
    /// Create standard given confidence level `z_score` and tolerance `p`:
    /// `N_full = (z_score / tolerance)^2`.
    pub fn from_confidence_and_tolerance(z_score: f64, tolerance: f64) -> Self {
        let n_full = (z_score / tolerance).powi(2);
        Self {
            full_credibility_standard: n_full,
        }
    }

    /// Default 1,082.41 claim standard (90% probability within +/- 5% error under Poisson).
    pub fn standard_90_5() -> Self {
        Self {
            full_credibility_standard: 1082.41,
        }
    }

    /// Compute partial credibility `Z = sqrt(min(1.0, observed_volume / full_credibility_standard))`.
    #[inline]
    pub fn credibility(&self, observed_volume: f64) -> f64 {
        if !observed_volume.is_finite()
            || !self.full_credibility_standard.is_finite()
            || !(self.full_credibility_standard > 0.0)
        {
            return f64::NAN;
        }
        if observed_volume <= 0.0 {
            0.0
        } else if observed_volume >= self.full_credibility_standard {
            1.0
        } else {
            (observed_volume / self.full_credibility_standard).sqrt()
        }
    }

    /// Blend observed indication with prior complement:
    /// `Blended = Z * observed + (1 - Z) * prior`.
    #[inline]
    pub fn blend(&self, observed_volume: f64, observed: f64, prior: f64) -> f64 {
        let z = self.credibility(observed_volume);
        z * observed + (1.0 - z) * prior
    }
}
