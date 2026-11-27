//! Parametric severity distributions, Limited Expected Values (LEVs),
//! Increased Limits Factors (ILFs), Loss Elimination Ratios (LERs),
//! and commercial excess layer towers.

use crate::error::PricingError;
use crate::special::{gamma_p, ln_gamma, normal_cdf_exact};
use serde::{Deserialize, Serialize};

/// Continuous density `f_G(x)` for `Gamma(shape, scale)` at `x > 0`.
#[inline]
fn gamma_pdf_raw(shape: f64, scale: f64, x: f64) -> f64 {
    if !(x > 0.0) {
        return 0.0;
    }
    ((shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - ln_gamma(shape)).exp()
}

/// Continuity-enforced body weight `w = P(X <= splice)` for `SplicedGammaPareto`.
#[inline]
fn spliced_body_weight(shape: f64, scale: f64, splice: f64, tail_alpha: f64) -> (f64, f64) {
    let f_u = gamma_p(shape, splice / scale);
    let pdf_u = gamma_pdf_raw(shape, scale, splice);
    let denom = tail_alpha * f_u + splice * pdf_u;
    let w = if denom > 0.0 && denom.is_finite() {
        (tail_alpha * f_u / denom).clamp(0.0, 1.0)
    } else {
        1.0
    };
    (w, f_u)
}

/// Continuous parametric severity distributions for commercial liability, auto, and property pricing.
///
/// Closed-form formulas follow Klugman, Panjer & Willmot (*Loss Models*, 5th ed.),
/// Miccolis (1977), Lee (1988), and Keatinge (1999).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeverityDistribution {
    /// Lognormal `ln(X) ~ N(mu, sigma^2)` with `sigma > 0`.
    Lognormal {
        /// Log-scale location parameter `mu`.
        mu: f64,
        /// Log-scale standard deviation `sigma > 0`.
        sigma: f64,
    },
    /// Pareto Type II (Lomax): `S(x) = (1 + x / theta)^(-alpha)` for `x >= 0`.
    Lomax {
        /// Shape parameter `alpha > 0`.
        alpha: f64,
        /// Scale parameter `theta > 0`.
        theta: f64,
    },
    /// Single-Parameter Pareto: `S(x) = (x_m / x)^alpha` for `x >= x_m`.
    SingleParameterPareto {
        /// Shape parameter `alpha > 0`.
        alpha: f64,
        /// Threshold parameter `x_m > 0`.
        x_m: f64,
    },
    /// Exponential distribution: `S(x) = exp(-lambda * x)`.
    Exponential {
        /// Rate parameter `lambda > 0`.
        lambda: f64,
    },
    /// Mixed Exponential (Keatinge 1999; ISO commercial liability ILF standard):
    /// `S(x) = sum_k w_k * exp(-x / mean_k)` with `w_k > 0`, `sum w_k = 1`, `mean_k > 0`.
    MixedExponential {
        /// Positive mixture weights summing to `1.0`.
        weights: Vec<f64>,
        /// Positive component means matching `weights` in length.
        means: Vec<f64>,
    },
    /// Generalized Pareto Distribution (GPD): `S(x) = (1 + xi * x / beta)^(-1 / xi)`
    /// for `xi >= 0`, `beta > 0`.
    GeneralizedPareto {
        /// Tail shape parameter `xi >= 0`.
        xi: f64,
        /// Scale parameter `beta > 0`.
        beta: f64,
    },
    /// Gamma distribution: `f(x) = x^(shape - 1) exp(-x / scale) / (Γ(shape) scale^shape)`
    /// for `shape > 0`, `scale > 0`.
    Gamma {
        /// Shape parameter `shape > 0`.
        shape: f64,
        /// Scale parameter `scale > 0`.
        scale: f64,
    },
    /// Composite spliced Gamma body (`x <= splice`) and Single-Parameter Pareto tail
    /// (`x > splice`) with continuity-enforced body weight `w = α F_G(u) / (α F_G(u) + u f_G(u))`
    /// (Klugman, Panjer & Willmot *Loss Models* §4.4; Scollnik & Sun 2012).
    SplicedGammaPareto {
        /// Gamma body shape parameter `shape > 0`.
        shape: f64,
        /// Gamma body scale parameter `scale > 0`.
        scale: f64,
        /// Splice threshold `splice > 0` where the Gamma body transitions to the Pareto tail.
        splice: f64,
        /// Pareto tail index `tail_alpha > 0` (`> 1` for finite mean).
        tail_alpha: f64,
    },
}

impl SeverityDistribution {
    /// Construct a `Lognormal` distribution from ground-up `mean > 0` and coefficient of variation `cv > 0`.
    pub fn lognormal_from_mean_cv(mean: f64, cv: f64) -> Result<Self, PricingError> {
        if !mean.is_finite() || !cv.is_finite() || !(mean > 0.0) || !(cv > 0.0) {
            return Err(PricingError::InvalidParameter(
                "mean and cv must be finite and strictly positive",
            ));
        }
        let sigma2 = (cv * cv).ln_1p();
        let sigma = sigma2.sqrt();
        let mu = mean.ln() - 0.5 * sigma2;
        Ok(Self::Lognormal { mu, sigma })
    }

    /// Construct a `Gamma` distribution from ground-up `mean > 0` and coefficient of variation `cv > 0`.
    pub fn gamma_from_mean_cv(mean: f64, cv: f64) -> Result<Self, PricingError> {
        if !mean.is_finite() || !cv.is_finite() || !(mean > 0.0) || !(cv > 0.0) {
            return Err(PricingError::InvalidParameter(
                "mean and cv must be finite and strictly positive",
            ));
        }
        let shape = 1.0 / (cv * cv);
        let scale = mean / shape;
        Ok(Self::Gamma { shape, scale })
    }

    /// Construct a `SplicedGammaPareto` distribution normalized to exact ground-up `mean > 0`
    /// for a given Gamma body `shape > 0`, dimensionless splice ratio `splice_ratio = splice / scale > 0`,
    /// and Pareto tail index `tail_alpha > 1.0`.
    pub fn spliced_gamma_pareto_from_mean(
        mean: f64,
        shape: f64,
        splice_ratio: f64,
        tail_alpha: f64,
    ) -> Result<Self, PricingError> {
        if !mean.is_finite()
            || !shape.is_finite()
            || !splice_ratio.is_finite()
            || !tail_alpha.is_finite()
            || !(mean > 0.0)
            || !(shape > 0.0)
            || !(splice_ratio > 0.0)
            || !(tail_alpha > 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "SplicedGammaPareto requires finite mean > 0, shape > 0, splice_ratio > 0, tail_alpha > 1",
            ));
        }
        let (w, f_u) = spliced_body_weight(shape, 1.0, splice_ratio, tail_alpha);
        let body_unit = if f_u > 0.0 {
            (w / f_u) * shape * gamma_p(shape + 1.0, splice_ratio)
        } else {
            0.0
        };
        let tail_unit = (1.0 - w) * tail_alpha * splice_ratio / (tail_alpha - 1.0);
        let unit_mean = body_unit + tail_unit;
        if !(unit_mean > 0.0) || !unit_mean.is_finite() {
            return Err(PricingError::InvalidParameter(
                "SplicedGammaPareto unit mean must be finite and positive",
            ));
        }
        let scale = mean / unit_mean;
        let splice = splice_ratio * scale;
        Ok(Self::SplicedGammaPareto {
            shape,
            scale,
            splice,
            tail_alpha,
        })
    }

    /// Validate distribution parameters for finiteness and mathematical domain constraints.
    pub fn validate(&self) -> Result<(), PricingError> {
        match self {
            Self::Lognormal { mu, sigma } => {
                if !mu.is_finite() || !sigma.is_finite() || !(*sigma > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "Lognormal requires finite mu and sigma > 0",
                    ));
                }
            }
            Self::Lomax { alpha, theta } => {
                if !alpha.is_finite() || !theta.is_finite() || !(*alpha > 0.0) || !(*theta > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "Lomax requires finite alpha > 0 and theta > 0",
                    ));
                }
            }
            Self::SingleParameterPareto { alpha, x_m } => {
                if !alpha.is_finite() || !x_m.is_finite() || !(*alpha > 0.0) || !(*x_m > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "SingleParameterPareto requires finite alpha > 0 and x_m > 0",
                    ));
                }
            }
            Self::Exponential { lambda } => {
                if !lambda.is_finite() || !(*lambda > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "Exponential requires finite lambda > 0",
                    ));
                }
            }
            Self::MixedExponential { weights, means } => {
                if weights.is_empty() || weights.len() != means.len() {
                    return Err(PricingError::DimensionMismatch(
                        "MixedExponential weights and means must be non-empty with equal length",
                    ));
                }
                let mut sum_w = 0.0;
                for (&w, &m) in weights.iter().zip(means.iter()) {
                    if !w.is_finite() || !m.is_finite() || !(w > 0.0) || !(m > 0.0) {
                        return Err(PricingError::InvalidParameter(
                            "MixedExponential weights and means must be finite and positive",
                        ));
                    }
                    sum_w += w;
                }
                if (sum_w - 1.0).abs() > 1e-6 {
                    return Err(PricingError::InvalidParameter(
                        "MixedExponential weights must sum to 1.0",
                    ));
                }
            }
            Self::GeneralizedPareto { xi, beta } => {
                if !xi.is_finite() || !beta.is_finite() || !(*xi >= 0.0) || !(*beta > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "GeneralizedPareto requires finite xi >= 0 and beta > 0",
                    ));
                }
            }
            Self::Gamma { shape, scale } => {
                if !shape.is_finite() || !scale.is_finite() || !(*shape > 0.0) || !(*scale > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "Gamma requires finite shape > 0 and scale > 0",
                    ));
                }
            }
            Self::SplicedGammaPareto {
                shape,
                scale,
                splice,
                tail_alpha,
            } => {
                if !shape.is_finite()
                    || !scale.is_finite()
                    || !splice.is_finite()
                    || !tail_alpha.is_finite()
                    || !(*shape > 0.0)
                    || !(*scale > 0.0)
                    || !(*splice > 0.0)
                    || !(*tail_alpha > 0.0)
                {
                    return Err(PricingError::InvalidParameter(
                        "SplicedGammaPareto requires finite shape, scale, splice, and tail_alpha > 0",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Survival function `S(x) = P(X > x)`.
    pub fn survival(&self, x: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter("x must be finite"));
        }
        if x <= 0.0 {
            return Ok(1.0);
        }
        let s = match self {
            Self::Lognormal { mu, sigma } => {
                let z = (x.ln() - mu) / sigma;
                normal_cdf_exact(-z)
            }
            Self::Lomax { alpha, theta } => (-alpha * (x / theta).ln_1p()).exp(),
            Self::SingleParameterPareto { alpha, x_m } => {
                if x <= *x_m {
                    1.0
                } else {
                    (x_m / x).powf(*alpha)
                }
            }
            Self::Exponential { lambda } => (-lambda * x).exp(),
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * (-x / m).exp())
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= 1e-10 {
                    (-x / beta).exp()
                } else {
                    (-(1.0 / xi) * (xi * x / beta).ln_1p()).exp()
                }
            }
            Self::Gamma { shape, scale } => 1.0 - gamma_p(*shape, x / scale),
            Self::SplicedGammaPareto {
                shape,
                scale,
                splice,
                tail_alpha,
            } => {
                let (w, f_u) = spliced_body_weight(*shape, *scale, *splice, *tail_alpha);
                if x <= *splice {
                    if f_u > 0.0 {
                        1.0 - w * gamma_p(*shape, x / scale) / f_u
                    } else {
                        1.0
                    }
                } else {
                    (1.0 - w) * (splice / x).powf(*tail_alpha)
                }
            }
        };
        Ok(s.clamp(0.0, 1.0))
    }

    /// Unbounded ground-up expected severity `E[X]`, or `None` if the mean is infinite.
    pub fn unbounded_mean(&self) -> Result<Option<f64>, PricingError> {
        self.validate()?;
        let mean = match self {
            Self::Lognormal { mu, sigma } => Some((mu + 0.5 * sigma * sigma).exp()),
            Self::Lomax { alpha, theta } => {
                if *alpha > 1.0 {
                    Some(theta / (alpha - 1.0))
                } else {
                    None
                }
            }
            Self::SingleParameterPareto { alpha, x_m } => {
                if *alpha > 1.0 {
                    Some(alpha * x_m / (alpha - 1.0))
                } else {
                    None
                }
            }
            Self::Exponential { lambda } => Some(1.0 / lambda),
            Self::MixedExponential { weights, means } => {
                Some(weights.iter().zip(means.iter()).map(|(&w, &m)| w * m).sum())
            }
            Self::GeneralizedPareto { xi, beta } => {
                if *xi < 1.0 {
                    Some(beta / (1.0 - xi))
                } else {
                    None
                }
            }
            Self::Gamma { shape, scale } => Some(shape * scale),
            Self::SplicedGammaPareto {
                shape,
                scale,
                splice,
                tail_alpha,
            } => {
                if *tail_alpha > 1.0 {
                    let (w, f_u) = spliced_body_weight(*shape, *scale, *splice, *tail_alpha);
                    let body = if f_u > 0.0 {
                        (w / f_u) * shape * scale * gamma_p(shape + 1.0, splice / scale)
                    } else {
                        0.0
                    };
                    let tail = (1.0 - w) * tail_alpha * splice / (tail_alpha - 1.0);
                    Some(body + tail)
                } else {
                    None
                }
            }
        };
        Ok(mean)
    }

    /// Limited Expected Value `E[X ∧ L] = integral_0^L S(x) dx`.
    pub fn limited_expected_value(&self, limit: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !limit.is_finite() {
            return Err(PricingError::InvalidParameter("limit must be finite"));
        }
        if !(limit >= 0.0) {
            return Err(PricingError::NonPositiveValue("limit cannot be negative"));
        }
        if limit == 0.0 {
            return Ok(0.0);
        }

        let lev = match self {
            Self::Lognormal { mu, sigma } => {
                let z = (limit.ln() - mu) / sigma;
                let mean = (mu + 0.5 * sigma * sigma).exp();
                let cdf_shifted = normal_cdf_exact(z - sigma);
                let body = if cdf_shifted == 0.0 {
                    0.0
                } else if !mean.is_finite() {
                    return Err(PricingError::OutOfBounds(
                        "lognormal mean overflowed floating-point range",
                    ));
                } else {
                    mean * cdf_shifted
                };
                body + limit * normal_cdf_exact(-z)
            }
            Self::Lomax { alpha, theta } => {
                if (alpha - 1.0).abs() < 1e-10 {
                    theta * (limit / theta).ln_1p()
                } else {
                    let ln_u = (limit / theta).ln_1p();
                    let term = ((1.0 - alpha) * ln_u).exp_m1();
                    -(theta / (alpha - 1.0)) * term
                }
            }
            Self::SingleParameterPareto { alpha, x_m } => {
                if limit <= *x_m {
                    limit
                } else if (alpha - 1.0).abs() < 1e-10 {
                    x_m * (1.0 + (limit / x_m).ln())
                } else {
                    let excess_term = -(((alpha - 1.0) * (x_m / limit).ln()).exp_m1());
                    x_m + x_m * excess_term / (alpha - 1.0)
                }
            }
            Self::Exponential { lambda } => -((-lambda * limit).exp_m1()) / lambda,
            Self::MixedExponential { weights, means } => weights
                .iter()
                .zip(means.iter())
                .map(|(&w, &m)| w * m * -((-limit / m).exp_m1()))
                .sum(),
            Self::GeneralizedPareto { xi, beta } => {
                if xi.abs() <= 1e-10 {
                    beta * -((-limit / beta).exp_m1())
                } else {
                    let ln_u = (xi * limit / beta).ln_1p();
                    if (xi - 1.0).abs() <= 1e-10 {
                        beta * ln_u
                    } else {
                        let omx = 1.0 - xi;
                        beta * -(((-omx / xi) * ln_u).exp_m1()) / omx
                    }
                }
            }
            Self::Gamma { shape, scale } => {
                let z = limit / scale;
                shape * scale * gamma_p(shape + 1.0, z) + limit * (1.0 - gamma_p(*shape, z))
            }
            Self::SplicedGammaPareto {
                shape,
                scale,
                splice,
                tail_alpha,
            } => {
                let (w, f_u) = spliced_body_weight(*shape, *scale, *splice, *tail_alpha);
                if limit <= *splice {
                    let z = limit / scale;
                    let body = if f_u > 0.0 {
                        (w / f_u) * shape * scale * gamma_p(shape + 1.0, z)
                    } else {
                        0.0
                    };
                    let surv_part = if f_u > 0.0 {
                        limit * (1.0 - w * gamma_p(*shape, z) / f_u)
                    } else {
                        limit
                    };
                    body + surv_part
                } else {
                    let zu = splice / scale;
                    let lev_u = if f_u > 0.0 {
                        (w / f_u) * shape * scale * gamma_p(shape + 1.0, zu) + (1.0 - w) * splice
                    } else {
                        *splice
                    };
                    let tail_excess = if (tail_alpha - 1.0).abs() <= 1e-10 {
                        (1.0 - w) * splice * (limit / splice).ln()
                    } else {
                        let excess_term = -(((tail_alpha - 1.0) * (splice / limit).ln()).exp_m1());
                        (1.0 - w) * (splice / (tail_alpha - 1.0)) * excess_term
                    };
                    lev_u + tail_excess
                }
            }
        };
        Ok(lev)
    }

    /// Calculate pure-loss Increased Limits Factor (ILF) relative to a base limit
    /// (Miccolis 1977; Lee 1988): `ILF(L; B) = E[X ∧ L] / E[X ∧ B]`.
    pub fn increased_limits_factor(
        &self,
        limit: f64,
        base_limit: f64,
    ) -> Result<f64, PricingError> {
        if !base_limit.is_finite() || !(base_limit > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "base limit must be finite and positive",
            ));
        }
        let lev_limit = self.limited_expected_value(limit)?;
        let lev_base = self.limited_expected_value(base_limit)?;
        if !(lev_base > 0.0) {
            return Err(PricingError::SingularMatrix("base limit LEV is zero"));
        }
        Ok(lev_limit / lev_base)
    }

    /// Calculate Loss Elimination Ratio (LER) for a deductible `d` relative to policy limit `L`:
    /// `LER(d; L) = E[X ∧ d] / E[X ∧ L]`.
    pub fn loss_elimination_ratio(
        &self,
        deductible: f64,
        policy_limit: f64,
    ) -> Result<f64, PricingError> {
        if !deductible.is_finite()
            || !policy_limit.is_finite()
            || !(deductible >= 0.0)
            || !(policy_limit > 0.0)
            || deductible > policy_limit
        {
            return Err(PricingError::InvalidParameter(
                "must satisfy 0 <= deductible <= policy_limit with finite values and policy_limit > 0",
            ));
        }
        let lev_d = self.limited_expected_value(deductible)?;
        let lev_l = self.limited_expected_value(policy_limit)?;
        if !(lev_l > 0.0) {
            return Err(PricingError::SingularMatrix("policy limit LEV is zero"));
        }
        Ok((lev_d / lev_l).clamp(0.0, 1.0))
    }

    /// Calculate excess ratio `R(retention) = E[(X - retention)_+] / E[X] = 1 - E[X ∧ retention] / E[X]`.
    pub fn excess_ratio(&self, retention: f64) -> Result<f64, PricingError> {
        let mean = self
            .unbounded_mean()?
            .ok_or(PricingError::InvalidParameter(
                "excess ratio requires a finite unbounded mean",
            ))?;
        let lev = self.limited_expected_value(retention)?;
        Ok(((mean - lev) / mean).clamp(0.0, 1.0))
    }
}

/// Specifications for a commercial excess insurance layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommercialLayer {
    /// Attachment point (retention) `A`.
    pub attachment: f64,
    /// Layer limit `L` (covers loss from `A` to `A + L`).
    pub limit: f64,
    /// Insurer participation percentage (`1.0 = 100%` share).
    pub participation: f64,
}

impl CommercialLayer {
    /// Construct a commercial excess layer with given attachment, limit, and participation share.
    pub fn new(attachment: f64, limit: f64, participation: f64) -> Result<Self, PricingError> {
        if !attachment.is_finite()
            || !limit.is_finite()
            || !participation.is_finite()
            || !(attachment >= 0.0)
            || !(limit > 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "attachment must be finite >= 0 and limit finite > 0",
            ));
        }
        if !(participation > 0.0 && participation <= 1.0) {
            return Err(PricingError::InvalidParameter(
                "participation must be finite and in (0.0, 1.0]",
            ));
        }
        Ok(Self {
            attachment,
            limit,
            participation,
        })
    }

    /// Validate layer parameters (`attachment >= 0`, `limit > 0`, `participation in (0, 1]`).
    pub fn validate(&self) -> Result<(), PricingError> {
        Self::new(self.attachment, self.limit, self.participation).map(|_| ())
    }

    /// Evaluates expected loss cost in this specific layer:
    /// `LayerLoss = Participation * ( E[X ∧ (A + L)] - E[X ∧ A] )`
    pub fn layer_expected_loss(
        &self,
        severity: &SeverityDistribution,
    ) -> Result<f64, PricingError> {
        self.validate()?;
        let lev_top = severity.limited_expected_value(self.attachment + self.limit)?;
        let lev_bot = severity.limited_expected_value(self.attachment)?;
        Ok(self.participation * (lev_top - lev_bot).max(0.0))
    }

    /// Evaluates actual loss payout of this layer given a gross occurrence loss `X`:
    /// `Payout = Participation * min(L, max(0, X - A))`
    #[inline]
    pub fn payout(&self, gross_loss: f64) -> f64 {
        if !gross_loss.is_finite() {
            return f64::NAN;
        }
        if gross_loss <= self.attachment {
            0.0
        } else {
            let inside = (gross_loss - self.attachment).min(self.limit);
            self.participation * inside
        }
    }
}

/// Ordered tower of commercial excess insurance layers (`Vec<CommercialLayer>`).
///
/// Useful for pricing multi-layer excess towers and treaty programs against a common
/// [`SeverityDistribution`] or occurrence loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerTower {
    layers: Vec<CommercialLayer>,
}

impl LayerTower {
    /// Construct a tower from validated [`CommercialLayer`] entries.
    pub fn try_new(layers: Vec<CommercialLayer>) -> Result<Self, PricingError> {
        if layers.is_empty() {
            return Err(PricingError::EmptyInput(
                "layer tower must contain at least one layer",
            ));
        }
        for layer in &layers {
            layer.validate()?;
        }
        Ok(Self { layers })
    }

    /// Underlying slice of [`CommercialLayer`] entries.
    #[inline]
    pub fn layers(&self) -> &[CommercialLayer] {
        &self.layers
    }

    /// Number of layers in the tower.
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if the tower has no layers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Evaluate expected loss cost for each layer in the tower.
    pub fn expected_losses(
        &self,
        severity: &SeverityDistribution,
    ) -> Result<Vec<f64>, PricingError> {
        let mut out = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            out.push(layer.layer_expected_loss(severity)?);
        }
        Ok(out)
    }

    /// Evaluate total expected loss across all layers in the tower.
    pub fn total_expected_loss(
        &self,
        severity: &SeverityDistribution,
    ) -> Result<f64, PricingError> {
        let mut total = 0.0;
        for layer in &self.layers {
            total += layer.layer_expected_loss(severity)?;
        }
        Ok(total)
    }

    /// Evaluate total tower payout for a single gross occurrence loss `X`.
    #[inline]
    pub fn total_payout(&self, gross_loss: f64) -> f64 {
        self.layers.iter().map(|l| l.payout(gross_loss)).sum()
    }
}
