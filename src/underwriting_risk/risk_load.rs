//! Pricing risk loads and distortion risk measures: Wang (1996, 2000) transforms,
//! Proportional Hazards, Dual Power, Kreps (1990) marginal surplus risk load, and
//! Ruhm–Mango–Kreps (2005) risk-adjusted scenario pricing.
//!
//! References:
//! - S. S. Wang (1996), *"Premium Calculation by Transforming the Layer Premium Density"*,
//!   *ASTIN Bulletin*, Vol. 26(1), pp. 71–92.
//! - S. S. Wang (2000), *"A Class of Distortion Operators for Pricing Financial and Insurance
//!   Risks"*, *Journal of Risk and Insurance*, Vol. 67(1), pp. 15–36.
//! - R. E. Kreps (1990), *"Reinsurer Risk Loads from Marginal Surplus Requirements"*,
//!   *PCAS*, Vol. LXXVII, pp. 196–203.
//! - D. Ruhm, D. Mango & R. Kreps (2005), *"Risk-Adjusted Pricing and Capital Allocation"*.

use super::aggregate::AggregateLossDistribution;
use crate::error::PricingError;
use crate::special::{normal_cdf_exact, normal_inv_cdf};
use serde::{Deserialize, Serialize};

/// Concave distortion operator `g: [0, 1] -> [0, 1]` with `g(0) = 0, g(1) = 1`
/// defining a coherent spectral / distortion risk measure `rho_g(X) = integral_0^inf g(S_X(x)) dx`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistortionOperator {
    /// Proportional Hazards (PH) transform (Wang 1996): `g(s) = s^(1 / rho)` with `rho >= 1`.
    ProportionalHazards {
        /// Risk-aversion parameter `rho >= 1` (`rho = 1` is risk-neutral expected value).
        rho: f64,
    },
    /// Wang transform (Wang 2000): `g(s) = Phi(Phi^-1(s) + lambda)` with `lambda >= 0`.
    Wang {
        /// Market price of risk `lambda >= 0` (`lambda = 0` is risk-neutral expected value).
        lambda: f64,
    },
    /// Dual Power distortion: `g(s) = 1 - (1 - s)^alpha` with `alpha >= 1`.
    DualPower {
        /// Power parameter `alpha >= 1` (`alpha = 1` is risk-neutral expected value).
        alpha: f64,
    },
    /// Tail Value-at-Risk (TVaR / Expected Shortfall) distortion at confidence `alpha in (0, 1)`:
    /// `g(s) = min(s / (1 - alpha), 1)`.
    Tvar {
        /// Confidence level `alpha in (0, 1)`.
        alpha: f64,
    },
}

impl DistortionOperator {
    /// Validate distortion parameters.
    pub fn validate(self) -> Result<(), PricingError> {
        match self {
            Self::ProportionalHazards { rho } => {
                if !rho.is_finite() || !(rho >= 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "PH distortion requires finite rho >= 1.0",
                    ));
                }
            }
            Self::Wang { lambda } => {
                if !lambda.is_finite() || !(lambda >= 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "Wang distortion requires finite lambda >= 0.0",
                    ));
                }
            }
            Self::DualPower { alpha } => {
                if !alpha.is_finite() || !(alpha >= 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "DualPower distortion requires finite alpha >= 1.0",
                    ));
                }
            }
            Self::Tvar { alpha } => {
                if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "TVaR distortion requires finite alpha in (0, 1)",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Evaluate `g(s)` for survival probability `s in [0, 1]`.
    pub fn evaluate(self, s: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !s.is_finite() {
            return Err(PricingError::InvalidParameter(
                "survival probability s must be finite",
            ));
        }
        if s <= 0.0 {
            return Ok(0.0);
        }
        if s >= 1.0 {
            return Ok(1.0);
        }
        let val = match self {
            Self::ProportionalHazards { rho } => s.powf(1.0 / rho),
            Self::Wang { lambda } => normal_cdf_exact(normal_inv_cdf(s) + lambda),
            Self::DualPower { alpha } => 1.0 - (1.0 - s).powf(alpha),
            Self::Tvar { alpha } => (s / (1.0 - alpha)).min(1.0),
        };
        Ok(val.clamp(0.0, 1.0))
    }
}

/// Output of distortion risk-adjusted pricing on a loss distribution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistortionPricingResult {
    /// Risk-neutral expected loss `E[X]`.
    pub expected_loss: f64,
    /// Distortion risk-adjusted price `rho_g(X) = integral_0^inf g(S(x)) dx`.
    pub risk_adjusted_price: f64,
    /// Risk load above expected loss: `risk_adjusted_price - expected_loss`.
    pub risk_load: f64,
    /// Risk load as a fraction of expected loss (`risk_load / expected_loss`, or `0.0` if `E[X] == 0`).
    pub risk_load_ratio: f64,
}

/// Price a discrete loss distribution `(losses, probs)` using a concave distortion operator
/// via the exact Choquet integral `sum_k (g(S_{k-1}) - g(S_k)) * x_k`.
pub fn distortion_price_discrete(
    losses: &[f64],
    probabilities: &[f64],
    distortion: DistortionOperator,
) -> Result<DistortionPricingResult, PricingError> {
    distortion.validate()?;
    if losses.is_empty() || losses.len() != probabilities.len() {
        return Err(PricingError::DimensionMismatch(
            "losses and probabilities must have matching non-empty length",
        ));
    }
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(losses.len());
    let mut sum_p = 0.0;
    for (&x, &p) in losses.iter().zip(probabilities.iter()) {
        if !x.is_finite() || !p.is_finite() || !(p >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "losses must be finite and probabilities must be finite and non-negative",
            ));
        }
        sum_p += p;
        pairs.push((x, p));
    }
    if (sum_p - 1.0).abs() > 1e-4 {
        return Err(PricingError::InvalidParameter(
            "probabilities must sum to 1.0",
        ));
    }

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Accumulate survival probabilities from the upper tail downward so far-tail
    // probabilities retain full 53-bit precision under Wang's inverse-normal transform.
    let mut expected_loss = 0.0;
    let mut price = 0.0;
    let mut surv_after = 0.0;
    let mut g_after = 0.0;

    for &(x, p) in pairs.iter().rev() {
        expected_loss += p * x;
        let surv_before = (surv_after + p).min(1.0);
        let g_before = distortion.evaluate(surv_before)?;
        let weight = (g_before - g_after).max(0.0);
        price += weight * x;
        surv_after = surv_before;
        g_after = g_before;
    }

    let risk_load = price - expected_loss;
    let risk_load_ratio = if expected_loss.abs() > 1e-15 {
        risk_load / expected_loss
    } else {
        0.0
    };

    Ok(DistortionPricingResult {
        expected_loss,
        risk_adjusted_price: price,
        risk_load,
        risk_load_ratio,
    })
}

/// Price an [`AggregateLossDistribution`] using a concave distortion operator.
pub fn distortion_price_aggregate(
    dist: &AggregateLossDistribution,
    distortion: DistortionOperator,
) -> Result<DistortionPricingResult, PricingError> {
    let losses: Vec<f64> = (0..dist.pmf.len())
        .map(|k| (k as f64) * dist.step)
        .collect();
    distortion_price_discrete(&losses, &dist.pmf, distortion)
}

/// Result of Kreps (1990) marginal surplus risk load calculation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KrepsRiskLoadResult {
    /// Portfolio standard deviation before adding the contract `sigma_old`.
    pub portfolio_std_dev_before: f64,
    /// Portfolio standard deviation after adding the contract `sigma_new`.
    pub portfolio_std_dev_after: f64,
    /// Marginal surplus required: `z * (sigma_new - sigma_old)`.
    pub marginal_surplus: f64,
    /// Exact Kreps (1990) risk load: `(y / (1 + y)) * marginal_surplus` (with `y = cost_of_capital`).
    pub risk_load: f64,
    /// Linearized covariance approximation: `(y / (1 + y)) * z * (Var(X) + 2 Cov(X, S)) / (2 sigma_old)`.
    pub linearized_risk_load: f64,
    /// Total risk-adjusted premium (`contract_mean + risk_load`).
    pub risk_adjusted_price: f64,
}

/// Compute Kreps (1990) marginal surplus risk load for adding a new contract/layer `X`
/// to an existing portfolio `S` with standard deviation `portfolio_std_dev`.
///
/// - `portfolio_std_dev`: Existing portfolio standard deviation `sigma_S >= 0`.
/// - `contract_mean`: Expected loss `E[X] >= 0` of the added contract/layer.
/// - `contract_variance`: Variance `Var(X) >= 0` of the added contract/layer.
/// - `covariance_with_portfolio`: `Cov(X, S)` (must satisfy `|Cov(X, S)| <= sigma_S * sqrt(Var(X))`).
/// - `capital_multiplier_z`: Capital-to-standard-deviation multiple `z > 0` (e.g. `3.0`).
/// - `cost_of_capital_y`: Target return on marginal surplus `y >= 0` (e.g. `0.15`).
pub fn kreps_marginal_risk_load(
    portfolio_std_dev: f64,
    contract_mean: f64,
    contract_variance: f64,
    covariance_with_portfolio: f64,
    capital_multiplier_z: f64,
    cost_of_capital_y: f64,
) -> Result<KrepsRiskLoadResult, PricingError> {
    if !portfolio_std_dev.is_finite()
        || !contract_mean.is_finite()
        || !contract_variance.is_finite()
        || !covariance_with_portfolio.is_finite()
        || !capital_multiplier_z.is_finite()
        || !cost_of_capital_y.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "Kreps risk load inputs must be finite",
        ));
    }
    if !(portfolio_std_dev >= 0.0)
        || !(contract_mean >= 0.0)
        || !(contract_variance >= 0.0)
        || !(capital_multiplier_z > 0.0)
        || !(cost_of_capital_y >= 0.0)
    {
        return Err(PricingError::InvalidParameter(
            "std_dev, mean, variance, cost_of_capital must be >= 0 and capital_multiplier_z > 0",
        ));
    }

    let max_cov = portfolio_std_dev * contract_variance.sqrt();
    if covariance_with_portfolio.abs() > max_cov + 1e-9 * max_cov.max(1.0) {
        return Err(PricingError::InvalidParameter(
            "covariance violates Cauchy-Schwarz bound |Cov(X, S)| <= sigma_S * sigma_X",
        ));
    }

    let var_after = (portfolio_std_dev * portfolio_std_dev
        + contract_variance
        + 2.0 * covariance_with_portfolio)
        .max(0.0);
    let sd_after = var_after.sqrt();
    let marginal_surplus = capital_multiplier_z * (sd_after - portfolio_std_dev);
    let load_factor = cost_of_capital_y / (1.0 + cost_of_capital_y);
    let risk_load = load_factor * marginal_surplus;

    let linearized = if portfolio_std_dev > 0.0 {
        load_factor * capital_multiplier_z * (contract_variance + 2.0 * covariance_with_portfolio)
            / (2.0 * portfolio_std_dev)
    } else {
        risk_load
    };

    Ok(KrepsRiskLoadResult {
        portfolio_std_dev_before: portfolio_std_dev,
        portfolio_std_dev_after: sd_after,
        marginal_surplus,
        risk_load,
        linearized_risk_load: linearized,
        risk_adjusted_price: contract_mean + risk_load,
    })
}

/// Output of Ruhm–Mango–Kreps (RMK) risk-adjusted pricing across aligned portfolio lines/layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RmkPricingResult {
    /// Total portfolio expected loss `E[S]`.
    pub portfolio_expected_loss: f64,
    /// Total portfolio risk-adjusted price `rho_g(S)`.
    pub portfolio_price: f64,
    /// Total portfolio risk load `rho_g(S) - E[S]`.
    pub portfolio_risk_load: f64,
    /// Per-line expected loss `E[X_i]`.
    pub line_expected_losses: Vec<f64>,
    /// Per-line risk-adjusted price `P_i = sum_k w_k X_{i, k}` (`sum_i P_i == portfolio_price`).
    pub line_prices: Vec<f64>,
    /// Per-line risk load `P_i - E[X_i]` (`sum_i R_i == portfolio_risk_load`).
    pub line_risk_loads: Vec<f64>,
    /// Per-line risk-load share `R_i / portfolio_risk_load`.
    pub risk_load_shares: Vec<f64>,
}

/// Ruhm–Mango–Kreps (2005) risk-adjusted pricing across `d` aligned lines/layers/contracts
/// using a concave distortion operator applied to the portfolio aggregate `S_k = sum_i X_{i, k}`.
///
/// Ties in aggregate loss `S_k` receive symmetric averaged distortion weights so the result
/// is strictly permutation-invariant and satisfies exact portfolio additivity
/// `sum_i line_prices[i] == portfolio_price`.
pub fn rmk_risk_adjusted_pricing(
    lines: &[&[f64]],
    distortion: DistortionOperator,
) -> Result<RmkPricingResult, PricingError> {
    distortion.validate()?;
    if lines.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one line of business is required",
        ));
    }
    let n = lines[0].len();
    if n == 0 {
        return Err(PricingError::EmptyInput(
            "scenario vectors must be non-empty",
        ));
    }
    for line in lines {
        if line.len() != n {
            return Err(PricingError::DimensionMismatch(
                "all lines must have the same number of scenarios",
            ));
        }
        if line.iter().any(|v| !v.is_finite()) {
            return Err(PricingError::InvalidParameter(
                "scenario values must be finite",
            ));
        }
    }

    let d = lines.len();
    let mut agg = vec![0.0; n];
    for line in lines {
        for (k, &v) in line.iter().enumerate() {
            agg[k] += v;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| agg[a].total_cmp(&agg[b]));

    // Assign exact tie-symmetric distortion weights w_k summing to 1.0.
    let mut weights = vec![0.0; n];
    let nf = n as f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && agg[order[j]] == agg[order[i]] {
            j += 1;
        }
        let surv_hi = ((n - i) as f64) / nf;
        let surv_lo = ((n - j) as f64) / nf;
        let total_w = (distortion.evaluate(surv_hi)? - distortion.evaluate(surv_lo)?).max(0.0);
        let each_w = total_w / ((j - i) as f64);
        for idx in i..j {
            weights[order[idx]] = each_w;
        }
        i = j;
    }

    let mut line_expected_losses = Vec::with_capacity(d);
    let mut line_prices = Vec::with_capacity(d);
    let mut line_risk_loads = Vec::with_capacity(d);

    let inv_n = 1.0 / nf;
    let mut portfolio_expected_loss = 0.0;
    let mut portfolio_price = 0.0;

    for line in lines {
        let mean: f64 = line.iter().sum::<f64>() * inv_n;
        let price: f64 = line.iter().zip(weights.iter()).map(|(&x, &w)| x * w).sum();
        line_expected_losses.push(mean);
        line_prices.push(price);
        line_risk_loads.push(price - mean);
        portfolio_expected_loss += mean;
        portfolio_price += price;
    }

    let portfolio_risk_load = portfolio_price - portfolio_expected_loss;
    let risk_load_shares = line_risk_loads
        .iter()
        .map(|&r| {
            if portfolio_risk_load.abs() > 1e-15 {
                r / portfolio_risk_load
            } else {
                0.0
            }
        })
        .collect();

    Ok(RmkPricingResult {
        portfolio_expected_loss,
        portfolio_price,
        portfolio_risk_load,
        line_expected_losses,
        line_prices,
        line_risk_loads,
        risk_load_shares,
    })
}
