use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Continuous parametric severity distribution for commercial liability and property pricing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeverityDistribution {
    /// Pareto Type II (Lomax): F(x) = 1 - (1 + x / theta)^(-alpha).
    /// Heavy-tailed standard in commercial liability (Auto Liability / General Liability).
    Lomax { alpha: f64, theta: f64 },
    /// Single-Parameter Pareto (strictly positive above threshold x_m):
    /// F(x) = 1 - (x_m / x)^alpha for x >= x_m.
    SingleParameterPareto { alpha: f64, x_m: f64 },
    /// Exponential distribution: F(x) = 1 - exp(-lambda * x).
    Exponential { lambda: f64 },
}

impl SeverityDistribution {
    /// Limited Expected Value E[X ^ L] = integral_0^L (1 - F(x)) dx.
    pub fn limited_expected_value(&self, limit: f64) -> Result<f64, PricingError> {
        if limit < 0.0 {
            return Err(PricingError::NonPositiveValue("limit cannot be negative"));
        }
        if limit == 0.0 {
            return Ok(0.0);
        }

        match *self {
            Self::Lomax { alpha, theta } => {
                if alpha <= 0.0 || theta <= 0.0 {
                    return Err(PricingError::InvalidParameter("alpha and theta must be positive"));
                }
                if (alpha - 1.0).abs() < 1e-9 {
                    // alpha == 1 case: E[X ^ L] = theta * ln(1 + L / theta)
                    Ok(theta * (limit / theta).ln_1p())
                } else {
                    // E[X ^ L] = (theta / (alpha - 1)) * [ 1 - (1 + L / theta)^(1 - alpha) ]
                    let term = (1.0 + limit / theta).powf(1.0 - alpha);
                    Ok((theta / (alpha - 1.0)) * (1.0 - term))
                }
            }
            Self::SingleParameterPareto { alpha, x_m } => {
                if alpha <= 0.0 || x_m <= 0.0 {
                    return Err(PricingError::InvalidParameter("alpha and x_m must be positive"));
                }
                if limit <= x_m {
                    Ok(limit)
                } else if (alpha - 1.0).abs() < 1e-9 {
                    Ok(x_m * (1.0 + (limit / x_m).ln()))
                } else {
                    let term = (x_m / limit).powf(alpha - 1.0);
                    Ok((alpha * x_m - limit * term) / (alpha - 1.0))
                }
            }
            Self::Exponential { lambda } => {
                if lambda <= 0.0 {
                    return Err(PricingError::InvalidParameter("lambda must be positive"));
                }
                // E[X ^ L] = (1 - exp(-lambda * L)) / lambda
                Ok((1.0 - (-lambda * limit).exp()) / lambda)
            }
        }
    }

    /// Calculate Increased Limits Factor (ILF) relative to a base limit:
    /// ILF(L) = E[X ^ L] / E[X ^ B]
    pub fn increased_limits_factor(&self, limit: f64, base_limit: f64) -> Result<f64, PricingError> {
        if base_limit <= 0.0 {
            return Err(PricingError::NonPositiveValue("base limit must be positive"));
        }
        let lev_limit = self.limited_expected_value(limit)?;
        let lev_base = self.limited_expected_value(base_limit)?;
        if lev_base <= 0.0 {
            return Err(PricingError::SingularMatrix("base limit LEV is zero"));
        }
        Ok(lev_limit / lev_base)
    }

    /// Calculate Loss Elimination Ratio (LER) for a deductible d relative to full expected loss (or policy limit L):
    /// LER(d) = E[X ^ d] / E[X ^ L]
    pub fn loss_elimination_ratio(&self, deductible: f64, policy_limit: f64) -> Result<f64, PricingError> {
        if deductible < 0.0 || policy_limit <= 0.0 {
            return Err(PricingError::InvalidParameter("deductible must be non-negative and policy limit positive"));
        }
        let lev_d = self.limited_expected_value(deductible)?;
        let lev_l = self.limited_expected_value(policy_limit)?;
        if lev_l <= 0.0 {
            return Err(PricingError::SingularMatrix("policy limit LEV is zero"));
        }
        Ok((lev_d / lev_l).min(1.0))
    }
}

/// Specifications for a commercial excess insurance layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommercialLayer {
    /// Attachment point (retention) A.
    pub attachment: f64,
    /// Layer limit L (covers loss from A to A + L).
    pub limit: f64,
    /// Insurer participation percentage (1.0 = 100% share).
    pub participation: f64,
}

impl CommercialLayer {
    pub fn new(attachment: f64, limit: f64, participation: f64) -> Result<Self, PricingError> {
        if attachment < 0.0 || limit <= 0.0 {
            return Err(PricingError::InvalidParameter("attachment must be >= 0 and limit > 0"));
        }
        if participation <= 0.0 || participation > 1.0 {
            return Err(PricingError::InvalidParameter("participation must be between 0.0 and 1.0"));
        }
        Ok(Self { attachment, limit, participation })
    }

    /// Evaluates expected loss cost in this specific layer:
    /// LayerLoss = Participation * ( E[X ^ (A + L)] - E[X ^ A] )
    pub fn layer_expected_loss(&self, severity: &SeverityDistribution) -> Result<f64, PricingError> {
        let lev_top = severity.limited_expected_value(self.attachment + self.limit)?;
        let lev_bot = severity.limited_expected_value(self.attachment)?;
        Ok(self.participation * (lev_top - lev_bot).max(0.0))
    }

    /// Evaluates actual loss payout of this layer given a gross occurrence loss X:
    /// Payout = Participation * min(L, max(0, X - A))
    #[inline]
    pub fn payout(&self, gross_loss: f64) -> f64 {
        if gross_loss <= self.attachment {
            0.0
        } else {
            let inside = (gross_loss - self.attachment).min(self.limit);
            self.participation * inside
        }
    }
}
