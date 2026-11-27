//! Coherent underwriting risk measures (`VaR`, `TVaR`, `RVaR`) and capital allocation
//! via Tasche's (2008) Euler principle and Myers & Read's (2001) marginal default-put method.
//!
//! References:
//! - D. Tasche (2008), *"Capital Allocation to Business Units and Sub-Portfolios: the Euler
//!   Principle"*, in *Pillar II in the New Basel Accord*, pp. 423–453.
//! - S. C. Myers & J. A. Read (2001), *"Capital Allocation for Insurance Companies"*,
//!   *Journal of Risk and Insurance*, Vol. 68(4), pp. 545–580.
//! - S. J. Mildenhall & J. A. Major, *Introduction to Capital Modeling and Portfolio Management*,
//!   CAS Monograph No. 15, §3 & §5.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

fn validate_samples(samples: &[f64]) -> Result<(), PricingError> {
    if samples.is_empty() {
        return Err(PricingError::EmptyInput("samples must not be empty"));
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(PricingError::InvalidParameter(
            "samples must contain only finite values",
        ));
    }
    Ok(())
}

fn validate_lines(lines: &[&[f64]]) -> Result<usize, PricingError> {
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
    Ok(n)
}

/// Empirical Value-at-Risk `VaR_alpha(X)` (least-quantile order statistic) for `alpha in (0, 1)`.
pub fn sample_var(samples: &[f64], alpha: f64) -> Result<f64, PricingError> {
    validate_samples(samples)?;
    if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
        return Err(PricingError::InvalidParameter(
            "confidence level alpha must be finite and in (0, 1)",
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((alpha * (sorted.len() as f64) - 1e-15).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[idx])
}

/// Exact coherent sample Range Value-at-Risk `RVaR_{p_low, p_high}(X)` (average of quantile
/// function over `[p_low, p_high]`, CAS Monograph No. 15 Algorithm 3.4.3).
pub fn sample_rvar(samples: &[f64], p_low: f64, p_high: f64) -> Result<f64, PricingError> {
    validate_samples(samples)?;
    if !p_low.is_finite()
        || !p_high.is_finite()
        || !(p_low >= 0.0)
        || !(p_high <= 1.0)
        || !(p_low < p_high)
    {
        return Err(PricingError::InvalidParameter(
            "must satisfy 0.0 <= p_low < p_high <= 1.0",
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let nf = n as f64;
    let width = p_high - p_low;

    let mut acc = 0.0;
    for (k, &v) in sorted.iter().enumerate() {
        let u0 = (k as f64) / nf;
        let u1 = ((k + 1) as f64) / nf;
        let overlap = (u1.min(p_high) - u0.max(p_low)).max(0.0);
        if overlap > 0.0 {
            acc += (overlap / width) * v;
        }
    }
    Ok(acc)
}

/// Exact coherent sample Tail Value-at-Risk `TVaR_alpha(X) = RVaR_{alpha, 1}(X)` for `alpha in (0, 1)`.
pub fn sample_tvar(samples: &[f64], alpha: f64) -> Result<f64, PricingError> {
    if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
        return Err(PricingError::InvalidParameter(
            "confidence level alpha must be finite and in (0, 1)",
        ));
    }
    sample_rvar(samples, alpha, 1.0)
}

/// Output of an Euler coherent risk-measure underwriting capital allocation across `d` lines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EulerAllocationResult {
    /// Portfolio aggregate risk measure `rho(S)`.
    pub portfolio_risk_measure: f64,
    /// Portfolio expected loss `E[S]`.
    pub portfolio_mean: f64,
    /// Portfolio risk capital above mean `rho(S) - E[S]`.
    pub portfolio_risk_capital: f64,
    /// Standalone expected loss `E[X_i]` per line.
    pub standalone_means: Vec<f64>,
    /// Standalone risk measure `rho(X_i)` per line.
    pub standalone_risk_measures: Vec<f64>,
    /// Euler allocated total risk measure `AC_i` per line (`sum_i AC_i == rho(S)`).
    pub allocated_total: Vec<f64>,
    /// Euler allocated risk capital above mean `AC_i - E[X_i]` per line.
    pub allocated_capital: Vec<f64>,
    /// Allocation shares `AC_i / rho(S)` per line.
    pub allocation_shares: Vec<f64>,
    /// Diversification credit per line: `standalone_risk_measures[i] - allocated_total[i]`.
    pub diversification_credit: Vec<f64>,
    /// Portfolio diversification ratio `rho(S) / sum_i rho(X_i)`.
    pub diversification_ratio: f64,
}

/// Compute exact tie-symmetric Euler allocation for `RVaR_{p_low, p_high}(S)` across `lines`
/// (Tasche 2008; CAS Monograph No. 15, §5).
///
/// Guarantees `sum_i allocated_total[i] == sample_rvar(S, p_low, p_high)` identically.
pub fn euler_rvar_allocation(
    lines: &[&[f64]],
    p_low: f64,
    p_high: f64,
) -> Result<EulerAllocationResult, PricingError> {
    let n = validate_lines(lines)?;
    if !p_low.is_finite()
        || !p_high.is_finite()
        || !(p_low >= 0.0)
        || !(p_high <= 1.0)
        || !(p_low < p_high)
    {
        return Err(PricingError::InvalidParameter(
            "must satisfy 0.0 <= p_low < p_high <= 1.0",
        ));
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

    let nf = n as f64;
    let width = p_high - p_low;
    let mut weights = vec![0.0; n];

    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && agg[order[j]] == agg[order[i]] {
            j += 1;
        }
        let u0 = (i as f64) / nf;
        let u1 = (j as f64) / nf;
        let overlap = (u1.min(p_high) - u0.max(p_low)).max(0.0);
        let each_w = (overlap / width) / ((j - i) as f64);
        for idx in i..j {
            weights[order[idx]] = each_w;
        }
        i = j;
    }

    let inv_n = 1.0 / nf;
    let mut standalone_means = Vec::with_capacity(d);
    let mut standalone_risk_measures = Vec::with_capacity(d);
    let mut allocated_total = Vec::with_capacity(d);
    let mut allocated_capital = Vec::with_capacity(d);
    let mut diversification_credit = Vec::with_capacity(d);

    let mut portfolio_mean = 0.0;
    let mut portfolio_risk_measure = 0.0;
    let mut sum_standalone = 0.0;

    for line in lines {
        let mean = line.iter().sum::<f64>() * inv_n;
        let standalone = sample_rvar(line, p_low, p_high)?;
        let alloc: f64 = line.iter().zip(weights.iter()).map(|(&x, &w)| x * w).sum();
        standalone_means.push(mean);
        standalone_risk_measures.push(standalone);
        allocated_total.push(alloc);
        allocated_capital.push(alloc - mean);
        diversification_credit.push(standalone - alloc);
        portfolio_mean += mean;
        portfolio_risk_measure += alloc;
        sum_standalone += standalone;
    }

    let allocation_shares = allocated_total
        .iter()
        .map(|&a| {
            if portfolio_risk_measure.abs() > 1e-15 {
                a / portfolio_risk_measure
            } else {
                0.0
            }
        })
        .collect();

    let diversification_ratio = if sum_standalone.abs() > 1e-15 {
        portfolio_risk_measure / sum_standalone
    } else {
        1.0
    };

    Ok(EulerAllocationResult {
        portfolio_risk_measure,
        portfolio_mean,
        portfolio_risk_capital: portfolio_risk_measure - portfolio_mean,
        standalone_means,
        standalone_risk_measures,
        allocated_total,
        allocated_capital,
        allocation_shares,
        diversification_credit,
        diversification_ratio,
    })
}

/// Compute exact tie-symmetric Euler allocation for `TVaR_alpha(S)` across `lines`
/// (Tasche 2008; CAS Monograph No. 15, §5).
pub fn euler_tvar_allocation(
    lines: &[&[f64]],
    alpha: f64,
) -> Result<EulerAllocationResult, PricingError> {
    if !alpha.is_finite() || !(alpha > 0.0) || !(alpha < 1.0) {
        return Err(PricingError::InvalidParameter(
            "confidence level alpha must be finite and in (0, 1)",
        ));
    }
    euler_rvar_allocation(lines, alpha, 1.0)
}

/// Compute Euler Standard-Deviation principle allocation `AC_i = E[X_i] + kappa * Cov(X_i, S) / sigma(S)`.
///
/// Conserves `sum_i allocated_total[i] == E[S] + kappa * sigma(S)` identically.
pub fn euler_sd_allocation(
    lines: &[&[f64]],
    kappa: f64,
) -> Result<EulerAllocationResult, PricingError> {
    let n = validate_lines(lines)?;
    if !kappa.is_finite() || !(kappa >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "standard deviation multiplier kappa must be finite and non-negative",
        ));
    }

    let d = lines.len();
    let nf = n as f64;
    let inv_n = 1.0 / nf;

    let mut agg = vec![0.0; n];
    for line in lines {
        for (k, &v) in line.iter().enumerate() {
            agg[k] += v;
        }
    }
    let portfolio_mean = agg.iter().sum::<f64>() * inv_n;
    let portfolio_var = agg
        .iter()
        .map(|&s| (s - portfolio_mean).powi(2))
        .sum::<f64>()
        * inv_n;
    let portfolio_sd = portfolio_var.sqrt();
    let portfolio_risk_measure = portfolio_mean + kappa * portfolio_sd;

    let mut standalone_means = Vec::with_capacity(d);
    let mut standalone_risk_measures = Vec::with_capacity(d);
    let mut allocated_total = Vec::with_capacity(d);
    let mut allocated_capital = Vec::with_capacity(d);
    let mut diversification_credit = Vec::with_capacity(d);
    let mut sum_standalone = 0.0;

    for line in lines {
        let mean = line.iter().sum::<f64>() * inv_n;
        let var = line.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() * inv_n;
        let standalone = mean + kappa * var.sqrt();
        let cap = if portfolio_sd > 0.0 {
            let cov = line
                .iter()
                .zip(agg.iter())
                .map(|(&x, &s)| (x - mean) * (s - portfolio_mean))
                .sum::<f64>()
                * inv_n;
            kappa * cov / portfolio_sd
        } else {
            0.0
        };
        let alloc = mean + cap;
        standalone_means.push(mean);
        standalone_risk_measures.push(standalone);
        allocated_total.push(alloc);
        allocated_capital.push(cap);
        diversification_credit.push(standalone - alloc);
        sum_standalone += standalone;
    }

    let allocation_shares = allocated_total
        .iter()
        .map(|&a| {
            if portfolio_risk_measure.abs() > 1e-15 {
                a / portfolio_risk_measure
            } else {
                0.0
            }
        })
        .collect();
    let diversification_ratio = if sum_standalone.abs() > 1e-15 {
        portfolio_risk_measure / sum_standalone
    } else {
        1.0
    };

    Ok(EulerAllocationResult {
        portfolio_risk_measure,
        portfolio_mean,
        portfolio_risk_capital: portfolio_risk_measure - portfolio_mean,
        standalone_means,
        standalone_risk_measures,
        allocated_total,
        allocated_capital,
        allocation_shares,
        diversification_credit,
        diversification_ratio,
    })
}

/// Output of Myers & Read (2001) marginal default-put surplus allocation across `d` lines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MyersReadAllocationResult {
    /// Total portfolio expected liabilities `L = E[S]`.
    pub portfolio_liability: f64,
    /// Total portfolio assets `A = L + Q`.
    pub portfolio_assets: f64,
    /// Total portfolio surplus `Q = A - L`.
    pub portfolio_surplus: f64,
    /// Portfolio insolvency probability `P(S > A)`.
    pub default_probability: f64,
    /// Insolvency default put value `D(A) = E[(S - A)_+]`.
    pub default_put_value: f64,
    /// Portfolio default-value-to-liability ratio `D(A) / L`.
    pub default_value_ratio: f64,
    /// Expected liability `L_i = E[X_i]` per line.
    pub line_liabilities: Vec<f64>,
    /// Conditional tail expectation `E[X_i | S > A]` per line.
    pub conditional_tail_losses: Vec<f64>,
    /// Allocated assets `A_i = L_i + Q_i` per line (`sum_i A_i == A`).
    pub allocated_assets: Vec<f64>,
    /// Allocated surplus `Q_i` per line (`sum_i Q_i == Q`).
    pub allocated_surplus: Vec<f64>,
    /// Allocated surplus-to-liability ratio `Q_i / L_i` per line.
    pub surplus_ratios: Vec<f64>,
    /// Portfolio aggregate surplus-to-liability ratio `Q / L`.
    pub portfolio_surplus_ratio: f64,
}

/// Compute Myers & Read (2001) marginal default-put surplus allocation at asset level `portfolio_assets`:
///
/// `Q_i = L_i * (Q / L) + (E[X_i | S > A] - (L_i / L) * E[S | S > A])`,
/// conserving `sum_i Q_i == Q` and `sum_i (L_i + Q_i) == A` identically.
/// Requires at least one default scenario (`S > portfolio_assets`) and `portfolio_liability > 0`.
pub fn myers_read_allocation(
    lines: &[&[f64]],
    portfolio_assets: f64,
) -> Result<MyersReadAllocationResult, PricingError> {
    let n = validate_lines(lines)?;
    if !portfolio_assets.is_finite() || !(portfolio_assets > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "portfolio_assets must be finite and positive",
        ));
    }

    let d = lines.len();
    let inv_n = 1.0 / (n as f64);
    let mut agg = vec![0.0; n];
    for line in lines {
        for (k, &v) in line.iter().enumerate() {
            agg[k] += v;
        }
    }

    let portfolio_liability = agg.iter().sum::<f64>() * inv_n;
    if !(portfolio_liability > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "portfolio expected liability must be positive",
        ));
    }

    let mut default_count = 0usize;
    let mut tail_agg_sum = 0.0;
    let mut default_put_sum = 0.0;
    for &s in &agg {
        if s > portfolio_assets {
            default_count += 1;
            tail_agg_sum += s;
            default_put_sum += s - portfolio_assets;
        }
    }
    if default_count == 0 {
        return Err(PricingError::OutOfBounds(
            "no scenarios exceed portfolio_assets; cannot estimate conditional default tail",
        ));
    }

    let portfolio_surplus = portfolio_assets - portfolio_liability;
    let portfolio_surplus_ratio = portfolio_surplus / portfolio_liability;
    let default_probability = (default_count as f64) * inv_n;
    let default_put_value = default_put_sum * inv_n;
    let default_value_ratio = default_put_value / portfolio_liability;
    let cond_tail_agg = tail_agg_sum / (default_count as f64);

    let mut line_liabilities = Vec::with_capacity(d);
    let mut conditional_tail_losses = Vec::with_capacity(d);
    let mut allocated_surplus = Vec::with_capacity(d);
    let mut allocated_assets = Vec::with_capacity(d);
    let mut surplus_ratios = Vec::with_capacity(d);

    for line in lines {
        let li = line.iter().sum::<f64>() * inv_n;
        let tail_i = line
            .iter()
            .zip(agg.iter())
            .filter(|&(_, &s)| s > portfolio_assets)
            .map(|(&x, _)| x)
            .sum::<f64>()
            / (default_count as f64);
        let share = li / portfolio_liability;
        let qi = li * portfolio_surplus_ratio + (tail_i - share * cond_tail_agg);
        let ai = li + qi;
        line_liabilities.push(li);
        conditional_tail_losses.push(tail_i);
        allocated_surplus.push(qi);
        allocated_assets.push(ai);
        surplus_ratios.push(if li.abs() > 1e-15 { qi / li } else { 0.0 });
    }

    Ok(MyersReadAllocationResult {
        portfolio_liability,
        portfolio_assets,
        portfolio_surplus,
        default_probability,
        default_put_value,
        default_value_ratio,
        line_liabilities,
        conditional_tail_losses,
        allocated_assets,
        allocated_surplus,
        surplus_ratios,
        portfolio_surplus_ratio,
    })
}
