use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// High-accuracy standard normal CDF `Φ(z)` via Cody (1969) rational `erfc`
/// approximation (max absolute error `< 1e-15`).
pub(crate) fn normal_cdf_exact(z: f64) -> f64 {
    0.5 * erfc_approx(-z / std::f64::consts::SQRT_2)
}

/// Complementary error function `erfc(x)` accurate to ~1e-15 (Cody 1969).
fn erfc_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    if x < 0.46875 {
        const A: [f64; 5] = [
            3.161_123_743_870_565_6e0,
            1.138_641_541_510_501_6e2,
            3.774_852_376_853_02e2,
            3.209_377_589_138_469_5e3,
            1.857_777_061_846_031_5e-1,
        ];
        const B: [f64; 4] = [
            2.360_129_095_234_412_1e1,
            2.440_246_379_344_441_7e2,
            1.282_616_526_077_372_3e3,
            2.844_236_833_439_171e3,
        ];
        let y = x * x;
        let num = (((A[4] * y + A[0]) * y + A[1]) * y + A[2]) * y + A[3];
        let den = (((y + B[0]) * y + B[1]) * y + B[2]) * y + B[3];
        1.0 - x * num / den
    } else if x <= 4.0 {
        const C: [f64; 9] = [
            5.641_884_969_886_701e-1,
            8.883_149_794_388_375,
            6.611_919_063_714_163e1,
            2.986_351_381_974_001e2,
            8.819_522_212_417_69e2,
            1.712_047_612_634_070_6e3,
            2.051_078_377_826_071_5e3,
            1.230_339_354_797_997_2e3,
            2.153_115_354_744_038_5e-8,
        ];
        const D: [f64; 8] = [
            1.574_492_611_070_983_5e1,
            1.176_939_508_913_125e2,
            5.371_811_018_620_099e2,
            1.621_389_574_566_690_2e3,
            3.290_799_235_733_459_6e3,
            4.362_619_090_143_247e3,
            3.439_367_674_143_721_6e3,
            1.230_339_354_803_749_4e3,
        ];
        let num = (((((((C[8] * x + C[0]) * x + C[1]) * x + C[2]) * x + C[3]) * x + C[4]) * x
            + C[5])
            * x
            + C[6])
            * x
            + C[7];
        let den = (((((((x + D[0]) * x + D[1]) * x + D[2]) * x + D[3]) * x + D[4]) * x + D[5]) * x
            + D[6])
            * x
            + D[7];
        (-x * x).exp() * num / den
    } else if x < 26.5 {
        const P: [f64; 6] = [
            3.053_266_349_612_323_4e-1,
            3.603_448_999_498_044_4e-1,
            1.257_817_261_112_292_5e-1,
            1.608_378_514_874_228e-2,
            6.587_491_615_298_378e-4,
            1.631_538_713_730_209_8e-2,
        ];
        const Q: [f64; 5] = [
            2.568_520_192_289_822,
            1.872_952_849_923_460_5e0,
            5.279_051_029_514_285e-1,
            6.051_834_131_244_132e-2,
            2.335_204_976_268_691_9e-3,
        ];
        let y = 1.0 / (x * x);
        let num = ((((P[5] * y + P[0]) * y + P[1]) * y + P[2]) * y + P[3]) * y + P[4];
        let den = ((((y + Q[0]) * y + Q[1]) * y + Q[2]) * y + Q[3]) * y + Q[4];
        let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
        (-x * x).exp() / x * (inv_sqrt_pi - y * num / den)
    } else {
        0.0
    }
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
                mean * normal_cdf_exact(z - sigma) + limit * normal_cdf_exact(-z)
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
                    let term = (x_m / limit).powf(alpha - 1.0);
                    x_m * (alpha - term) / (alpha - 1.0)
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
        {
            return Err(PricingError::InvalidParameter(
                "deductible must be finite and non-negative, policy limit finite and positive",
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

    /// Evaluates expected loss cost in this specific layer:
    /// `LayerLoss = Participation * ( E[X ∧ (A + L)] - E[X ∧ A] )`
    pub fn layer_expected_loss(
        &self,
        severity: &SeverityDistribution,
    ) -> Result<f64, PricingError> {
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
