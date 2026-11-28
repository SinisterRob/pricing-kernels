//! Overall rate level indication kernels: Loss Ratio Method and Pure Premium Method
//! (Werner & Modlin *Basic Ratemaking*, Ch. 5–8).

use crate::error::PricingError;
use crate::validate::{
    require_finite, require_non_negative, require_positive, require_unit_interval,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Single historical experience period (accident year or policy year) for rate level indication.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RateIndicationPeriod {
    /// Experience year identifier (e.g. `2022`).
    pub year: i32,
    /// Earned exposures in the experience period (must be `> 0`).
    pub earned_exposures: f64,
    /// Historical earned premium (must be `> 0`).
    pub earned_premium: f64,
    /// On-level factor bringing historical earned premium to current rate level (`> 0`).
    pub on_level_factor: f64,
    /// Premium trend factor from experience period to prospective policy period (`> 0`).
    pub premium_trend_factor: f64,
    /// Historical reported (or paid) losses (`>= 0`).
    pub reported_loss: f64,
    /// Loss development factor to ultimate (`> 0`, `1.0` if already at ultimate).
    pub loss_development_factor: f64,
    /// Loss cost / severity-frequency trend factor to prospective period (`> 0`).
    pub loss_trend_factor: f64,
    /// Allocated Loss Adjustment Expense (ALAE) multiplicative load (`>= 1.0`, `1.0` if included in loss).
    pub alae_load_factor: f64,
    /// Optional experience-year weight (`None` uses period's adjusted earned premium or exposures).
    pub weight: Option<f64>,
}

/// Underwriting expense, ULAE, and target profit provisions (Werner & Modlin Ch. 7).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpenseProvisions {
    /// Fixed underwriting expense ratio `F` for Loss Ratio Method (`>= 0`, e.g. `0.08`).
    pub fixed_expense_ratio: f64,
    /// Fixed underwriting expense per exposure `E_F` for Pure Premium Method (`>= 0`).
    pub fixed_expense_per_exposure: f64,
    /// Variable expense ratio `V` (commissions, premium taxes, other acquisition; in `[0, 1)`).
    pub variable_expense_ratio: f64,
    /// Underwriting profit and contingencies provision `Q_T` (in `(-1, 1)` with `V + Q_T < 1`).
    pub profit_and_contingencies_ratio: f64,
    /// Unallocated Loss Adjustment Expense (ULAE) multiplicative factor (`>= 1.0`, e.g. `1.06`).
    pub ulae_factor: f64,
}

impl ExpenseProvisions {
    /// Permissible Loss Ratio `PLR = 1.0 - V - Q_T` (excluding fixed expenses `F`).
    #[inline]
    pub fn permissible_loss_ratio(&self) -> f64 {
        1.0 - self.variable_expense_ratio - self.profit_and_contingencies_ratio
    }

    /// Validate all expense and profit provisions.
    pub fn validate(&self) -> Result<(), PricingError> {
        require_non_negative(
            self.fixed_expense_ratio,
            "fixed_expense_ratio must be finite >= 0",
        )?;
        require_non_negative(
            self.fixed_expense_per_exposure,
            "fixed_expense_per_exposure must be finite >= 0",
        )?;
        require_unit_interval(
            self.variable_expense_ratio,
            "variable_expense_ratio must be finite in [0, 1)",
        )?;
        if !(self.variable_expense_ratio < 1.0) {
            return Err(PricingError::InvalidParameter(
                "variable_expense_ratio must be strictly < 1.0",
            ));
        }
        require_finite(
            self.profit_and_contingencies_ratio,
            "profit_and_contingencies_ratio must be finite",
        )?;
        if !self.ulae_factor.is_finite() || !(self.ulae_factor >= 1.0) {
            return Err(PricingError::InvalidParameter(
                "ulae_factor must be finite and >= 1.0",
            ));
        }
        require_positive(
            self.permissible_loss_ratio(),
            "permissible loss ratio (1 - V - Q_T) must be strictly positive",
        )?;
        Ok(())
    }
}

/// Per-period adjusted premium, ultimate trended loss & LAE, and loss ratio breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RateIndicationPeriodResult {
    /// Experience year identifier.
    pub year: i32,
    /// On-level, trended earned premium at current rate level.
    pub projected_earned_premium: f64,
    /// Developed, trended ultimate loss and LAE (including ALAE and ULAE).
    pub projected_ultimate_loss_and_lae: f64,
    /// Period projected loss & LAE ratio (`projected_ultimate_loss_and_lae / projected_earned_premium`).
    pub projected_loss_ratio: f64,
    /// Period projected pure premium (`projected_ultimate_loss_and_lae / earned_exposures`).
    pub projected_pure_premium: f64,
    /// Normalized weight assigned to this experience period.
    pub normalized_weight: f64,
}

/// Complete overall rate level indication output covering both Loss Ratio and Pure Premium methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateIndicationResult {
    /// Per-period exhibit breakdown.
    pub by_period: Vec<RateIndicationPeriodResult>,
    /// Total on-level, trended earned premium across all periods.
    pub total_projected_earned_premium: f64,
    /// Total developed, trended ultimate loss & LAE across all periods.
    pub total_projected_loss_and_lae: f64,
    /// Total earned exposures across all periods.
    pub total_earned_exposures: f64,
    /// Weighted projected loss & LAE ratio across the experience period.
    pub weighted_loss_ratio: f64,
    /// Projected pure premium (loss & LAE per exposure).
    pub projected_pure_premium: f64,
    /// Permissible loss ratio `1 - V - Q_T`.
    pub permissible_loss_ratio: f64,
    /// Indicated overall rate change via Loss Ratio Method:
    /// `(weighted_loss_ratio + F) / (1 - V - Q_T) - 1`.
    pub indicated_rate_change: f64,
    /// Indicated rate per exposure via Pure Premium Method:
    /// `(projected_pure_premium + E_F) / (1 - V - Q_T)`.
    pub indicated_rate_per_exposure: f64,
    /// Current average rate per exposure (`total_projected_earned_premium / total_earned_exposures`).
    pub current_average_rate: f64,
    /// Indicated overall rate change via Pure Premium Method:
    /// `indicated_rate_per_exposure / current_average_rate - 1`.
    pub pure_premium_indicated_change: f64,
    /// Optional credibility-weighted indicated rate change `Z * indicated + (1 - Z) * complement`.
    pub credibility_weighted_change: Option<f64>,
}

impl fmt::Display for RateIndicationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rate Level Indication Summary")?;
        writeln!(
            f,
            "  Projected Loss & LAE Ratio : {:>7.2}%",
            self.weighted_loss_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Permissible Loss Ratio     : {:>7.2}%",
            self.permissible_loss_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Indicated Change (LR)      : {:>+7.2}%",
            self.indicated_rate_change * 100.0
        )?;
        writeln!(
            f,
            "  Indicated Rate (PP)        : {:>10.2} (current {:.2}, {:+.2}%)",
            self.indicated_rate_per_exposure,
            self.current_average_rate,
            self.pure_premium_indicated_change * 100.0
        )?;
        if let Some(cw) = self.credibility_weighted_change {
            writeln!(f, "  Credibility-Weighted Change: {:>+7.2}%", cw * 100.0)?;
        }
        Ok(())
    }
}

/// Complete specification for an overall rate level indication (Werner & Modlin Ch. 8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateIndicationSpec {
    /// Historical experience periods.
    pub periods: Vec<RateIndicationPeriod>,
    /// Underwriting expense, ULAE, and profit provisions.
    pub expenses: ExpenseProvisions,
    /// Optional classical credibility `Z in [0, 1]` for blending with a complement of credibility.
    pub credibility_z: Option<f64>,
    /// Complement of credibility rate change (e.g. trended present rates / competitor change; default `0.0`).
    pub complement_of_credibility: f64,
}

impl RateIndicationSpec {
    /// Compute the overall rate level indication (both Loss Ratio and Pure Premium methods).
    pub fn calculate(&self) -> Result<RateIndicationResult, PricingError> {
        calculate_rate_indication(self)
    }
}

/// Compute an overall rate level indication using both the Loss Ratio Method and Pure Premium
/// Method (Werner & Modlin *Basic Ratemaking*, Ch. 8).
pub fn calculate_rate_indication(
    spec: &RateIndicationSpec,
) -> Result<RateIndicationResult, PricingError> {
    if spec.periods.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one experience period is required for rate indication",
        ));
    }
    spec.expenses.validate()?;
    if !spec.complement_of_credibility.is_finite() {
        return Err(PricingError::InvalidParameter(
            "complement_of_credibility must be finite",
        ));
    }
    if let Some(z) = spec.credibility_z
        && (!z.is_finite() || !(z >= 0.0) || !(z <= 1.0))
    {
        return Err(PricingError::InvalidParameter(
            "credibility_z must be finite and in [0, 1]",
        ));
    }

    let has_custom_weights = spec.periods.iter().any(|p| p.weight.is_some());
    if has_custom_weights && spec.periods.iter().any(|p| p.weight.is_none()) {
        return Err(PricingError::InvalidParameter(
            "period weights must be either all Some or all None",
        ));
    }
    let mut raw_weights = Vec::with_capacity(spec.periods.len());
    let mut proj_prems = Vec::with_capacity(spec.periods.len());
    let mut proj_losses = Vec::with_capacity(spec.periods.len());
    let mut total_prem = 0.0;
    let mut total_loss = 0.0;
    let mut total_exp = 0.0;
    let mut weight_sum = 0.0;

    for p in &spec.periods {
        if !p.earned_exposures.is_finite()
            || !p.earned_premium.is_finite()
            || !p.on_level_factor.is_finite()
            || !p.premium_trend_factor.is_finite()
            || !p.reported_loss.is_finite()
            || !p.loss_development_factor.is_finite()
            || !p.loss_trend_factor.is_finite()
            || !p.alae_load_factor.is_finite()
        {
            return Err(PricingError::InvalidParameter(
                "rate indication period fields must be finite",
            ));
        }
        if !(p.earned_exposures > 0.0)
            || !(p.earned_premium > 0.0)
            || !(p.on_level_factor > 0.0)
            || !(p.premium_trend_factor > 0.0)
            || !(p.reported_loss >= 0.0)
            || !(p.loss_development_factor > 0.0)
            || !(p.loss_trend_factor > 0.0)
            || !(p.alae_load_factor >= 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "exposures, premiums, factors must be > 0, losses >= 0, and alae_load_factor >= 1",
            ));
        }

        let proj_prem = p.earned_premium * p.on_level_factor * p.premium_trend_factor;
        let proj_loss = p.reported_loss
            * p.loss_development_factor
            * p.loss_trend_factor
            * p.alae_load_factor
            * spec.expenses.ulae_factor;

        let w = match p.weight {
            Some(w) => {
                if !w.is_finite() || !(w >= 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "period weight must be finite and non-negative",
                    ));
                }
                w
            }
            None => proj_prem,
        };

        proj_prems.push(proj_prem);
        proj_losses.push(proj_loss);
        raw_weights.push(w);
        total_prem += proj_prem;
        total_loss += proj_loss;
        total_exp += p.earned_exposures;
        weight_sum += w;
    }

    if !(weight_sum > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "sum of period weights must be strictly positive",
        ));
    }

    let mut by_period = Vec::with_capacity(spec.periods.len());
    let mut weighted_lr = 0.0;
    let mut weighted_pp = 0.0;
    let mut weighted_rate = 0.0;
    for (i, p) in spec.periods.iter().enumerate() {
        let proj_prem = proj_prems[i];
        let proj_loss = proj_losses[i];
        let lr = proj_loss / proj_prem;
        let pp = proj_loss / p.earned_exposures;
        let rate = proj_prem / p.earned_exposures;
        let norm_w = raw_weights[i] / weight_sum;
        weighted_lr += norm_w * lr;
        weighted_pp += norm_w * pp;
        weighted_rate += norm_w * rate;
        by_period.push(RateIndicationPeriodResult {
            year: p.year,
            projected_earned_premium: proj_prem,
            projected_ultimate_loss_and_lae: proj_loss,
            projected_loss_ratio: lr,
            projected_pure_premium: pp,
            normalized_weight: norm_w,
        });
    }

    let plr = spec.expenses.permissible_loss_ratio();
    let indicated_rate_change = (weighted_lr + spec.expenses.fixed_expense_ratio) / plr - 1.0;

    let (projected_pure_premium, current_average_rate) = if has_custom_weights {
        (weighted_pp, weighted_rate)
    } else {
        (total_loss / total_exp, total_prem / total_exp)
    };
    let indicated_rate_per_exposure =
        (projected_pure_premium + spec.expenses.fixed_expense_per_exposure) / plr;
    let pure_premium_indicated_change = indicated_rate_per_exposure / current_average_rate - 1.0;

    let credibility_weighted_change = spec
        .credibility_z
        .map(|z| z * indicated_rate_change + (1.0 - z) * spec.complement_of_credibility);

    Ok(RateIndicationResult {
        by_period,
        total_projected_earned_premium: total_prem,
        total_projected_loss_and_lae: total_loss,
        total_earned_exposures: total_exp,
        weighted_loss_ratio: weighted_lr,
        projected_pure_premium,
        permissible_loss_ratio: plr,
        indicated_rate_change,
        indicated_rate_per_exposure,
        current_average_rate,
        pure_premium_indicated_change,
        credibility_weighted_change,
    })
}
