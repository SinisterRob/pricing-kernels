//! Overall rate-level indication, underwriting profit & contingency loads
//! (Myers–Cohn 1987 and ROE-implied), extension-of-exposures premium trending,
//! and constrained rate-capping off-balance iteration (Werner & Modlin *Basic Ratemaking*).

use crate::error::PricingError;
use crate::validate::{
    require_finite, require_non_negative, require_positive, require_unit_interval,
};
use serde::{Deserialize, Serialize};

// ===========================================================================
// 2.1 Pure Premium & Loss Ratio Rate-Level Indications
// ===========================================================================

/// Output of [`pure_premium_indication`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PurePremiumIndicationResult {
    /// Permissible Loss Ratio `PLR = 1 - V - Q_T`.
    pub permissible_loss_ratio: f64,
    /// Indicated average rate per exposure `R_I = (L_bar + E_F) / (1 - V - Q_T)`.
    pub indicated_rate: f64,
    /// Indicated overall rate change `Δ_PP = R_I / P_C - 1`.
    pub indicated_change: f64,
}

/// Pure Premium Method overall rate-level indication (Werner & Modlin, Ch. 8).
///
/// * `projected_loss_per_exposure` — `L_bar >= 0`.
/// * `fixed_expense_per_exposure` — `E_F >= 0`.
/// * `variable_expense_ratio` — `V in [0, 1)`.
/// * `profit_and_contingencies` — `Q_T` with `1 - V - Q_T > 0`.
/// * `current_average_rate` — `P_C > 0`.
pub fn pure_premium_indication(
    projected_loss_per_exposure: f64,
    fixed_expense_per_exposure: f64,
    variable_expense_ratio: f64,
    profit_and_contingencies: f64,
    current_average_rate: f64,
) -> Result<PurePremiumIndicationResult, PricingError> {
    require_non_negative(
        projected_loss_per_exposure,
        "projected_loss_per_exposure must be finite >= 0",
    )?;
    require_non_negative(
        fixed_expense_per_exposure,
        "fixed_expense_per_exposure must be finite >= 0",
    )?;
    require_unit_interval(
        variable_expense_ratio,
        "variable_expense_ratio must be in [0, 1)",
    )?;
    if !(variable_expense_ratio < 1.0) {
        return Err(PricingError::InvalidParameter(
            "variable_expense_ratio must be < 1.0",
        ));
    }
    require_finite(
        profit_and_contingencies,
        "profit_and_contingencies must be finite",
    )?;
    require_positive(current_average_rate, "current_average_rate must be > 0")?;

    let plr = 1.0 - variable_expense_ratio - profit_and_contingencies;
    require_positive(plr, "permissible loss ratio (1 - V - Q_T) must be > 0")?;

    let indicated_rate = (projected_loss_per_exposure + fixed_expense_per_exposure) / plr;
    let indicated_change = indicated_rate / current_average_rate - 1.0;

    Ok(PurePremiumIndicationResult {
        permissible_loss_ratio: plr,
        indicated_rate,
        indicated_change,
    })
}

/// Output of [`loss_ratio_indication`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossRatioIndicationResult {
    /// Permissible Loss Ratio `PLR = 1 - V - Q_T`.
    pub permissible_loss_ratio: f64,
    /// Indicated overall rate change `Δ_LR = (L_R + F) / (1 - V - Q_T) - 1`.
    pub indicated_change: f64,
}

/// Loss Ratio Method overall rate-level indication (Werner & Modlin, Ch. 8).
pub fn loss_ratio_indication(
    projected_loss_ratio: f64,
    fixed_expense_ratio: f64,
    variable_expense_ratio: f64,
    profit_and_contingencies: f64,
) -> Result<LossRatioIndicationResult, PricingError> {
    require_non_negative(
        projected_loss_ratio,
        "projected_loss_ratio must be finite >= 0",
    )?;
    require_non_negative(
        fixed_expense_ratio,
        "fixed_expense_ratio must be finite >= 0",
    )?;
    require_unit_interval(
        variable_expense_ratio,
        "variable_expense_ratio must be in [0, 1)",
    )?;
    if !(variable_expense_ratio < 1.0) {
        return Err(PricingError::InvalidParameter(
            "variable_expense_ratio must be < 1.0",
        ));
    }
    require_finite(
        profit_and_contingencies,
        "profit_and_contingencies must be finite",
    )?;

    let plr = 1.0 - variable_expense_ratio - profit_and_contingencies;
    require_positive(plr, "permissible loss ratio (1 - V - Q_T) must be > 0")?;

    let indicated_change = (projected_loss_ratio + fixed_expense_ratio) / plr - 1.0;
    Ok(LossRatioIndicationResult {
        permissible_loss_ratio: plr,
        indicated_change,
    })
}

// ===========================================================================
// 2.2 Underwriting Profit & Contingency Provision (Myers–Cohn & ROE-Implied)
// ===========================================================================

/// Specification for the Myers–Cohn (1987) discounted cash-flow underwriting
/// profit model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MyersCohnSpec {
    /// Total expected nominal losses `L > 0`.
    pub expected_losses: f64,
    /// Loss payout fractions `w_t >= 0` for periods `t = 1..=T` (must sum to `1.0`).
    pub loss_payout_pattern: Vec<f64>,
    /// Nominal underwriting expenses `E >= 0` paid at `t = 0` (or discounted via `pv_expenses`).
    pub expected_expenses: f64,
    /// Optional present value of expenses `PV(E)` (defaults to `expected_expenses` paid at `t = 0`).
    pub pv_expenses: Option<f64>,
    /// Surplus balance `S_{t-1} >= 0` held at the start of each period `t = 1..=T`.
    pub surplus_schedule: Vec<f64>,
    /// Risk-free interest rate `r_f > -1.0`.
    pub risk_free_rate: f64,
    /// Risk-adjusted loss discount rate `r_L = r_f + β_L (r_m - r_f) > -1.0`.
    pub loss_discount_rate: f64,
    /// Corporate federal income tax rate `τ ∈ [0, 1)`.
    pub tax_rate: f64,
    /// Timing `t_tax >= 0` (in periods) when underwriting profit tax `τ(P - L - E)` is paid
    /// (default `1.0`). The premium and expense components of this tax are discounted at
    /// `r_f`; the loss component (tax shield `τ L`) at `r_L` (Myers & Cohn 1987).
    pub uw_tax_time: f64,
}

/// Output of [`myers_cohn_profit_load`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MyersCohnResult {
    /// Present value of losses `PV(L)` discounted at `r_L`.
    pub pv_losses: f64,
    /// Present value of underwriting expenses `PV(E)`.
    pub pv_expenses: f64,
    /// Present value of investment-income tax on surplus `PV(IBT)`.
    pub pv_investment_balance_tax: f64,
    /// Present value of underwriting profit tax
    /// `PV(UWPT) = τ [P / (1 + r_f)^{t_tax} - L / (1 + r_L)^{t_tax} - E / (1 + r_f)^{t_tax}]`.
    pub pv_underwriting_profit_tax: f64,
    /// Fair Myers–Cohn premium `P`.
    pub fair_premium: f64,
    /// Implied underwriting profit provision `Q_T = 1 - (L + E) / P`.
    pub profit_provision: f64,
}

/// Solve the Myers–Cohn (1987) discounted cash-flow model for fair premium `P`
/// and implied underwriting profit provision `Q_T = 1 - (L + E) / P`.
///
/// Uses the closed-form linear balance equation
/// `P = PV(L) + PV(E) + PV(IBT) + PV(UWPT)` with
/// `PV(UWPT) = τ [P / (1 + r_f)^{t_tax} - L / (1 + r_L)^{t_tax} - E / (1 + r_f)^{t_tax}]`,
/// i.e. `P = [PV(L) + PV(E) + PV(IBT) - τ L d_L - τ E d_f] / (1 - τ d_f)` with
/// `d_f = (1 + r_f)^{-t_tax}`, `d_L = (1 + r_L)^{-t_tax}`.
///
/// Convention (Myers & Cohn 1987; Robbin, *The Underwriting Profit Provision*, CAS Exam 9
/// study note, Myers–Cohn section): each cash flow is discounted at a rate reflecting its risk.
/// Premium and expense flows (and their tax effects) are treated as riskless and discounted at
/// `r_f`; losses **and the loss-related tax shield** `τ L` carry the systematic risk of losses
/// and are discounted at the risk-adjusted rate `r_L`. Investment-income tax on surplus is
/// discounted at `r_f`. With `r_L = r_f` this reduces to discounting the whole underwriting
/// tax `τ (P - L - E)` at `r_f`.
pub fn myers_cohn_profit_load(spec: &MyersCohnSpec) -> Result<MyersCohnResult, PricingError> {
    let t_len = spec.loss_payout_pattern.len();
    if t_len == 0 {
        return Err(PricingError::EmptyInput(
            "loss_payout_pattern must not be empty",
        ));
    }
    if spec.surplus_schedule.len() != t_len {
        return Err(PricingError::DimensionMismatch(
            "surplus_schedule length must match loss_payout_pattern",
        ));
    }
    require_positive(spec.expected_losses, "expected_losses must be > 0")?;
    require_non_negative(spec.expected_expenses, "expected_expenses must be >= 0")?;
    require_non_negative(spec.uw_tax_time, "uw_tax_time must be >= 0")?;
    require_unit_interval(spec.tax_rate, "tax_rate must be in [0, 1)")?;
    if !(spec.tax_rate < 1.0) {
        return Err(PricingError::InvalidParameter("tax_rate must be < 1.0"));
    }
    if !spec.risk_free_rate.is_finite() || !(spec.risk_free_rate > -1.0) {
        return Err(PricingError::InvalidParameter(
            "risk_free_rate must be finite and > -1.0",
        ));
    }
    if !spec.loss_discount_rate.is_finite() || !(spec.loss_discount_rate > -1.0) {
        return Err(PricingError::InvalidParameter(
            "loss_discount_rate must be finite and > -1.0",
        ));
    }

    let mut w_sum = 0.0;
    let mut pv_losses = 0.0;
    let mut pv_ibt = 0.0;
    for (idx, (&w, &s)) in spec
        .loss_payout_pattern
        .iter()
        .zip(spec.surplus_schedule.iter())
        .enumerate()
    {
        require_non_negative(w, "payout weight must be >= 0")?;
        require_non_negative(s, "surplus balance must be >= 0")?;
        let t = (idx + 1) as i32;
        w_sum += w;
        pv_losses += spec.expected_losses * w / (1.0 + spec.loss_discount_rate).powi(t);
        pv_ibt += spec.tax_rate * spec.risk_free_rate * s / (1.0 + spec.risk_free_rate).powi(t);
    }
    if (w_sum - 1.0).abs() > 1e-6 {
        return Err(PricingError::InvalidParameter(
            "loss_payout_pattern weights must sum to 1.0",
        ));
    }

    let pv_expenses = match spec.pv_expenses {
        Some(v) => {
            require_non_negative(v, "pv_expenses must be >= 0")?;
            v
        }
        None => spec.expected_expenses,
    };

    // Premium/expense tax flows at r_f; loss tax shield at the risk-adjusted r_L.
    let tax_disc = spec.tax_rate / (1.0 + spec.risk_free_rate).powf(spec.uw_tax_time);
    let loss_tax_disc = spec.tax_rate / (1.0 + spec.loss_discount_rate).powf(spec.uw_tax_time);
    let denom = 1.0 - tax_disc;
    require_positive(denom, "1 - tax_rate / (1 + r_f)^t_tax must be > 0")?;

    let tax_shield = loss_tax_disc * spec.expected_losses + tax_disc * spec.expected_expenses;
    let nom_outflow = spec.expected_losses + spec.expected_expenses;
    let fair_premium = (pv_losses + pv_expenses + pv_ibt - tax_shield) / denom;
    require_positive(fair_premium, "fair_premium must be > 0")?;

    let pv_uwpt = tax_disc * fair_premium - tax_shield;
    let profit_provision = 1.0 - nom_outflow / fair_premium;

    Ok(MyersCohnResult {
        pv_losses,
        pv_expenses,
        pv_investment_balance_tax: pv_ibt,
        pv_underwriting_profit_tax: pv_uwpt,
        fair_premium,
        profit_provision,
    })
}

/// Output of [`roe_implied_profit_load`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoeProfitLoadResult {
    /// After-tax investment yield `y_AT = y_pre * (1 - τ)`.
    pub after_tax_yield: f64,
    /// Implied underwriting profit & contingency provision `Q_T`.
    pub profit_provision: f64,
}

/// ROE / IRR-implied underwriting profit & contingency provision:
/// `Q_T = (k_S * (ROE* - y_AT) - k_R * y_AT) / (1 - τ)`.
pub fn roe_implied_profit_load(
    target_roe: f64,
    pretax_investment_yield: f64,
    tax_rate: f64,
    surplus_to_premium: f64,
    reserves_to_premium: f64,
) -> Result<RoeProfitLoadResult, PricingError> {
    require_finite(target_roe, "target_roe must be finite")?;
    require_finite(
        pretax_investment_yield,
        "pretax_investment_yield must be finite",
    )?;
    require_unit_interval(tax_rate, "tax_rate must be in [0, 1)")?;
    if !(tax_rate < 1.0) {
        return Err(PricingError::InvalidParameter("tax_rate must be < 1.0"));
    }
    require_positive(surplus_to_premium, "surplus_to_premium must be > 0")?;
    require_non_negative(reserves_to_premium, "reserves_to_premium must be >= 0")?;

    let y_at = pretax_investment_yield * (1.0 - tax_rate);
    let q_t =
        (surplus_to_premium * (target_roe - y_at) - reserves_to_premium * y_at) / (1.0 - tax_rate);

    Ok(RoeProfitLoadResult {
        after_tax_yield: y_at,
        profit_provision: q_t,
    })
}

// ===========================================================================
// 2.3 Extension of Exposures & Two-Step Premium Trend (AWP Drift)
// ===========================================================================

/// Exact on-level earned premium via Extension of Exposures:
/// `OLEP = Σ_i exposures[i] * current_manual_rates[i]`.
pub fn extension_of_exposures_olep(
    exposures: &[f64],
    current_manual_rates: &[f64],
) -> Result<f64, PricingError> {
    if exposures.is_empty() {
        return Err(PricingError::EmptyInput("exposures must not be empty"));
    }
    if exposures.len() != current_manual_rates.len() {
        return Err(PricingError::DimensionMismatch(
            "exposures and current_manual_rates must have equal length",
        ));
    }
    let mut olep = 0.0;
    for (&e, &r) in exposures.iter().zip(current_manual_rates.iter()) {
        require_non_negative(e, "exposure must be >= 0")?;
        require_non_negative(r, "current_manual_rate must be >= 0")?;
        olep += e * r;
    }
    Ok(olep)
}

/// Output of [`two_step_premium_trend`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PremiumTrendResult {
    /// Fitted annual exponential drift rate `exp(b) - 1` from log-linear AWP regression.
    pub fitted_annual_drift: f64,
    /// Historical trend factor `(1 + g_past)^Δt_1`.
    pub past_trend_factor: f64,
    /// Prospective trend factor `(1 + g_fut)^Δt_2`.
    pub future_trend_factor: f64,
    /// Combined two-step premium trend factor.
    pub combined_trend_factor: f64,
}

/// Fit log-linear average-written-premium (AWP) drift `ln(AWP_t) = a + b * t`
/// and compute the two-step premium trend factor
/// `(1 + g_past)^dt_past * (1 + g_fut)^dt_future`.
///
/// If `prospective_drift_override` is `None`, the fitted annual drift is used
/// for both steps.
pub fn two_step_premium_trend(
    times: &[f64],
    on_level_awp: &[f64],
    dt_past: f64,
    dt_future: f64,
    prospective_drift_override: Option<f64>,
) -> Result<PremiumTrendResult, PricingError> {
    let n = times.len();
    if n < 2 {
        return Err(PricingError::InvalidParameter(
            "at least 2 AWP observations are required to fit premium drift",
        ));
    }
    if on_level_awp.len() != n {
        return Err(PricingError::DimensionMismatch(
            "times and on_level_awp must have equal length",
        ));
    }
    require_finite(dt_past, "dt_past must be finite")?;
    require_finite(dt_future, "dt_future must be finite")?;

    let mut mean_t = 0.0;
    let mut mean_y = 0.0;
    let mut ys = Vec::with_capacity(n);
    for (&t, &awp) in times.iter().zip(on_level_awp.iter()) {
        require_finite(t, "time value must be finite")?;
        require_positive(awp, "on_level_awp must be > 0")?;
        let y = awp.ln();
        ys.push(y);
        mean_t += t;
        mean_y += y;
    }
    mean_t /= n as f64;
    mean_y /= n as f64;

    let mut s_tt = 0.0;
    let mut s_ty = 0.0;
    for (&t, &y) in times.iter().zip(ys.iter()) {
        let dt = t - mean_t;
        s_tt += dt * dt;
        s_ty += dt * (y - mean_y);
    }
    require_positive(s_tt, "times must have non-zero variance")?;

    let slope = s_ty / s_tt;
    let fitted_annual_drift = slope.exp_m1();
    let g_fut = match prospective_drift_override {
        Some(g) => {
            if !g.is_finite() || !(g > -1.0) {
                return Err(PricingError::InvalidParameter(
                    "prospective_drift_override must be finite and > -1.0",
                ));
            }
            g
        }
        None => fitted_annual_drift,
    };

    let past_trend_factor = (1.0 + fitted_annual_drift).powf(dt_past);
    let future_trend_factor = (1.0 + g_fut).powf(dt_future);
    let combined_trend_factor = past_trend_factor * future_trend_factor;

    Ok(PremiumTrendResult {
        fitted_annual_drift,
        past_trend_factor,
        future_trend_factor,
        combined_trend_factor,
    })
}

// ===========================================================================
// 2.4 Constrained Rate-Capping & Disruption Off-Balance Iteration
// ===========================================================================

/// Output of [`solve_capped_rate_off_balance`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateCappingResult {
    /// Off-balance multiplier `γ > 0` satisfying `Σ clamp(γ P_i^ind, lo_i, hi_i) = P_target`.
    pub off_balance_factor: f64,
    /// Final capped premium per policy.
    pub capped_premiums: Vec<f64>,
    /// Final capped rate change `(capped_i / curr_i) - 1` per policy.
    pub capped_changes: Vec<f64>,
    /// Total target premium `(1 + Δ_target) * Σ P_i^curr`.
    pub target_total_premium: f64,
    /// Achieved total capped premium.
    pub achieved_total_premium: f64,
    /// Number of policies hitting the lower cap `min_change`.
    pub n_at_lower_cap: usize,
    /// Number of policies hitting the upper cap `max_change`.
    pub n_at_upper_cap: usize,
    /// Root-finder iterations executed.
    pub iterations: usize,
}

/// Solve for the off-balance multiplier `γ > 0` so that per-policy capped
/// premiums `clamp(γ * P_i^ind, (1 + min_change) * P_i^curr, (1 + max_change) * P_i^curr)`
/// achieve the exact target total premium `(1 + target_change) * Σ P_i^curr`.
///
/// Uses feasibility validation followed by bracketing and Brent's method.
pub fn solve_capped_rate_off_balance(
    current_premiums: &[f64],
    indicated_premiums: &[f64],
    target_change: f64,
    min_change: f64,
    max_change: f64,
) -> Result<RateCappingResult, PricingError> {
    let n = current_premiums.len();
    if n == 0 {
        return Err(PricingError::EmptyInput(
            "current_premiums must not be empty",
        ));
    }
    if indicated_premiums.len() != n {
        return Err(PricingError::DimensionMismatch(
            "current_premiums and indicated_premiums must have equal length",
        ));
    }
    require_finite(target_change, "target_change must be finite")?;
    require_finite(min_change, "min_change must be finite")?;
    require_finite(max_change, "max_change must be finite")?;
    if !(min_change > -1.0) || !(max_change >= min_change) {
        return Err(PricingError::InvalidParameter(
            "capping bounds must satisfy -1.0 < min_change <= max_change",
        ));
    }

    let mut total_curr = 0.0;
    let mut lo_bounds = Vec::with_capacity(n);
    let mut hi_bounds = Vec::with_capacity(n);
    for (&pc, &pi) in current_premiums.iter().zip(indicated_premiums.iter()) {
        require_positive(pc, "current_premium must be > 0")?;
        require_positive(pi, "indicated_premium must be > 0")?;
        total_curr += pc;
        lo_bounds.push((1.0 + min_change) * pc);
        hi_bounds.push((1.0 + max_change) * pc);
    }

    let target_total = (1.0 + target_change) * total_curr;
    let sum_lo: f64 = lo_bounds.iter().sum();
    let sum_hi: f64 = hi_bounds.iter().sum();

    if target_total < sum_lo - 1e-12 * total_curr || target_total > sum_hi + 1e-12 * total_curr {
        return Err(PricingError::OutOfBounds(
            "target_change is infeasible given [min_change, max_change] capping bounds",
        ));
    }

    let eval_f = |gamma: f64| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            s += (gamma * indicated_premiums[i]).clamp(lo_bounds[i], hi_bounds[i]);
        }
        s - target_total
    };

    let tol = 1e-10 * target_total.max(1.0);

    // Bracket [a, b] such that F(a) <= 0 <= F(b).
    let mut a = 0.5_f64;
    let mut fa = eval_f(a);
    let mut expand = 0;
    while fa > 0.0 && expand < 80 {
        a *= 0.5;
        fa = eval_f(a);
        expand += 1;
    }
    if fa > 0.0 {
        a = 0.0;
        fa = eval_f(a);
    }

    let mut b = 2.0_f64;
    let mut fb = eval_f(b);
    expand = 0;
    while fb < 0.0 && expand < 80 {
        b *= 2.0;
        fb = eval_f(b);
        expand += 1;
    }

    // Brent's method on monotone piecewise-linear F(gamma) = 0.
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut iterations = 0usize;

    for iter in 0..200 {
        iterations = iter + 1;
        if fb.abs() <= tol || (b - a).abs() <= 1e-14 * b.abs().max(1.0) {
            break;
        }
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let m = 0.5 * (c - b);
        if e.abs() >= 1e-14 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if (a - c).abs() <= 1e-15 {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            let min1 = 3.0 * m * q - (1e-14 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > 1e-14 {
            d
        } else if m > 0.0 {
            1e-14
        } else {
            -1e-14
        };
        fb = eval_f(b);
    }

    let gamma = b;
    let mut capped_premiums = Vec::with_capacity(n);
    let mut capped_changes = Vec::with_capacity(n);
    let mut achieved_total = 0.0;
    let mut n_at_lower = 0usize;
    let mut n_at_upper = 0usize;

    for i in 0..n {
        let raw = gamma * indicated_premiums[i];
        let capped = raw.clamp(lo_bounds[i], hi_bounds[i]);
        if raw <= lo_bounds[i] + 1e-12 {
            n_at_lower += 1;
        } else if raw >= hi_bounds[i] - 1e-12 {
            n_at_upper += 1;
        }
        capped_premiums.push(capped);
        capped_changes.push(capped / current_premiums[i] - 1.0);
        achieved_total += capped;
    }

    Ok(RateCappingResult {
        off_balance_factor: gamma,
        capped_premiums,
        capped_changes,
        target_total_premium: target_total,
        achieved_total_premium: achieved_total,
        n_at_lower_cap: n_at_lower,
        n_at_upper_cap: n_at_upper,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_werner_modlin_ch8_pp_and_lr_equivalence() {
        let pp = pure_premium_indication(450.0, 45.0, 0.15, 0.05, 580.0).expect("pp");
        assert!((pp.permissible_loss_ratio - 0.80).abs() < 1e-12);
        assert!((pp.indicated_rate - 618.75).abs() < 1e-12);
        assert!((pp.indicated_change - 0.066_810_344_827_586_2).abs() < 1e-12);

        let lr = loss_ratio_indication(450.0 / 580.0, 45.0 / 580.0, 0.15, 0.05).expect("lr");
        assert!((lr.indicated_change - pp.indicated_change).abs() < 1e-12);
    }

    #[test]
    fn golden_roe_implied_profit_load() {
        let res = roe_implied_profit_load(0.12, 0.05, 0.21, 0.50, 1.20).expect("roe");
        assert!((res.after_tax_yield - 0.0395).abs() < 1e-12);
        assert!((res.profit_provision - (-0.009_050_632_911_392_406)).abs() < 1e-10);
    }

    #[test]
    fn golden_myers_cohn_1_period_exact() {
        let spec = MyersCohnSpec {
            expected_losses: 100.0,
            loss_payout_pattern: vec![1.0],
            expected_expenses: 0.0,
            pv_expenses: None,
            surplus_schedule: vec![50.0],
            risk_free_rate: 0.06,
            loss_discount_rate: 0.04,
            tax_rate: 0.34,
            uw_tax_time: 1.0,
        };
        let res = myers_cohn_profit_load(&spec).expect("myers-cohn");
        assert!((res.pv_losses - 100.0 / 1.04).abs() < 1e-12);
        assert!((res.pv_investment_balance_tax - 0.34 * 0.06 * 50.0 / 1.06).abs() < 1e-12);
        // Hand derivation (Myers & Cohn 1987: loss tax shield discounted at r_L):
        //   PV(L)    = 100 / 1.04
        //   PV(IBT)  = 0.34 * 0.06 * 50 / 1.06 = 1.02 / 1.06
        //   PV(UWPT) = 0.34 P / 1.06 - 0.34 * 100 / 1.04
        //   P (1 - 0.34 / 1.06) = (100 - 34) / 1.04 + 1.02 / 1.06
        //   P = [66 * 1.06 / 1.04 + 1.02] / 0.72 = (69.96 / 1.04 + 1.02) / 0.72
        //     = 71.0208 / 0.7488 = 1233 / 13 = 94.846153846...
        // (pre-fix r_f-discounted loss shield gave 95.7542735.)
        assert!(
            (res.fair_premium - 1233.0 / 13.0).abs() < 1e-10,
            "P={}",
            res.fair_premium
        );
        assert!(
            (res.pv_underwriting_profit_tax - (0.34 * res.fair_premium / 1.06 - 34.0 / 1.04)).abs()
                < 1e-12
        );
        // Balance identity P = PV(L) + PV(E) + PV(IBT) + PV(UWPT).
        let rhs = res.pv_losses
            + res.pv_expenses
            + res.pv_investment_balance_tax
            + res.pv_underwriting_profit_tax;
        assert!((res.fair_premium - rhs).abs() < 1e-10);
    }

    #[test]
    fn myers_cohn_reduces_to_rf_tax_discount_when_rl_equals_rf() {
        let spec = MyersCohnSpec {
            expected_losses: 100.0,
            loss_payout_pattern: vec![0.5, 0.5],
            expected_expenses: 20.0,
            pv_expenses: None,
            surplus_schedule: vec![50.0, 25.0],
            risk_free_rate: 0.05,
            loss_discount_rate: 0.05,
            tax_rate: 0.21,
            uw_tax_time: 1.0,
        };
        let res = myers_cohn_profit_load(&spec).expect("myers-cohn");
        let d_f = 1.0 / 1.05;
        let expected = (res.pv_losses + res.pv_expenses + res.pv_investment_balance_tax
            - 0.21 * d_f * 120.0)
            / (1.0 - 0.21 * d_f);
        assert!((res.fair_premium - expected).abs() < 1e-10);
    }

    #[test]
    fn rate_capping_off_balance_hits_exact_target_and_detects_infeasible_caps() {
        let curr = [100.0, 200.0, 300.0, 400.0];
        let ind = [80.0, 250.0, 310.0, 520.0];
        let res = solve_capped_rate_off_balance(&curr, &ind, 0.05, -0.10, 0.15)
            .expect("capped off-balance");
        assert!(
            (res.achieved_total_premium - res.target_total_premium).abs()
                <= 1e-9 * res.target_total_premium
        );
        for &chg in &res.capped_changes {
            assert!((-0.10 - 1e-12..=0.15 + 1e-12).contains(&chg));
        }

        // Infeasible target (+20% when max cap is +15%).
        assert!(solve_capped_rate_off_balance(&curr, &ind, 0.20, -0.10, 0.15).is_err());
    }
}
