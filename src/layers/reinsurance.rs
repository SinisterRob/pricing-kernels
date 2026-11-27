//! Commercial treaty reinsurance pricing kernels: experience-rated burning cost,
//! reinstatement pricing (Sundt 1991), sliding-scale commission, loss corridors,
//! and profit commission (Clark 2014 *Basics of Reinsurance Pricing*).

use crate::error::PricingError;
use crate::experience::TableMEngine;
use crate::layers::CommercialLayer;
use serde::{Deserialize, Serialize};

/// Historical treaty-year subject premium and adjustment factors for burning-cost rating.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoricalTreatyYear {
    /// Underwriting or accident year identifier.
    pub year: i32,
    /// Historical subject earned premium.
    pub subject_premium: f64,
    /// On-level factor bringing historical rate level to prospective treaty period.
    pub on_level_factor: f64,
    /// Exposure trend factor bringing historical exposure base to prospective period.
    pub exposure_trend_factor: f64,
}

/// Historical individual large claim record for excess-of-loss experience rating.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoricalTreatyClaim {
    /// Underwriting or accident year matching a `HistoricalTreatyYear`.
    pub year: i32,
    /// Historical incurred (or paid) ground-up claim amount.
    pub incurred: f64,
    /// Severity trend factor bringing claim cost to prospective treaty period.
    pub severity_trend_factor: f64,
    /// Loss development factor to ultimate (use `1.0` if already at ultimate).
    pub development_factor: f64,
}

/// Per-year burning cost breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BurningCostYearResult {
    /// Underwriting or accident year.
    pub year: i32,
    /// Adjusted subject premium (`subject_premium * on_level * exposure_trend`).
    pub adjusted_subject_premium: f64,
    /// Total trended/developed ceded layer losses in this year.
    pub layer_ceded_loss: f64,
    /// Annual burning cost rate (`layer_ceded_loss / adjusted_subject_premium`).
    pub burning_cost_rate: f64,
    /// Count of claims penetrating the layer in this year.
    pub layer_claim_count: usize,
}

/// Aggregate experience-rated burning cost result for an excess-of-loss treaty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurningCostResult {
    /// Per-year experience breakdown in input order.
    pub by_year: Vec<BurningCostYearResult>,
    /// Total adjusted subject premium across all experience years.
    pub total_adjusted_subject_premium: f64,
    /// Total trended/developed ceded layer losses across all experience years.
    pub total_layer_ceded_loss: f64,
    /// Overall experience-rated burning cost rate (`total_layer_ceded_loss / total_adjusted_subject_premium`).
    pub overall_burning_cost_rate: f64,
    /// Total count of claims penetrating the layer.
    pub total_layer_claims: usize,
}

/// Compute experience-rated burning cost for an excess-of-loss treaty (Clark 2014).
///
/// Each claim is trended and developed to prospective ultimate ground-up dollars
/// (`incurred * severity_trend_factor * development_factor`) before applying `layer.payout`.
/// Each historical year's subject premium is adjusted by `on_level_factor * exposure_trend_factor`.
pub fn burning_cost_xl(
    years: &[HistoricalTreatyYear],
    claims: &[HistoricalTreatyClaim],
    layer: &CommercialLayer,
) -> Result<BurningCostResult, PricingError> {
    if years.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one historical treaty year is required",
        ));
    }
    layer.validate()?;

    let mut by_year: Vec<BurningCostYearResult> = Vec::with_capacity(years.len());
    let mut total_adj_prem = 0.0;

    for y in years {
        if by_year.iter().any(|entry| entry.year == y.year) {
            return Err(PricingError::InvalidParameter(
                "duplicate year encountered in historical treaty years",
            ));
        }
        if !y.subject_premium.is_finite()
            || !y.on_level_factor.is_finite()
            || !y.exposure_trend_factor.is_finite()
            || !(y.subject_premium > 0.0)
            || !(y.on_level_factor > 0.0)
            || !(y.exposure_trend_factor > 0.0)
        {
            return Err(PricingError::NonPositiveValue(
                "subject_premium, on_level_factor, and exposure_trend_factor must be finite and positive",
            ));
        }
        let adj = y.subject_premium * y.on_level_factor * y.exposure_trend_factor;
        total_adj_prem += adj;
        by_year.push(BurningCostYearResult {
            year: y.year,
            adjusted_subject_premium: adj,
            layer_ceded_loss: 0.0,
            burning_cost_rate: 0.0,
            layer_claim_count: 0,
        });
    }

    let mut total_ceded = 0.0;
    let mut total_layer_claims = 0usize;

    for c in claims {
        if !c.incurred.is_finite()
            || !c.severity_trend_factor.is_finite()
            || !c.development_factor.is_finite()
            || !(c.incurred >= 0.0)
            || !(c.severity_trend_factor > 0.0)
            || !(c.development_factor > 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "claim incurred must be >= 0 and trend/development factors must be finite and positive",
            ));
        }
        let Some(yr_entry) = by_year.iter_mut().find(|entry| entry.year == c.year) else {
            return Err(PricingError::InvalidParameter(
                "claim year does not match any provided HistoricalTreatyYear",
            ));
        };
        let adj_loss = c.incurred * c.severity_trend_factor * c.development_factor;
        let payout = layer.payout(adj_loss);
        if payout > 0.0 {
            yr_entry.layer_ceded_loss += payout;
            yr_entry.layer_claim_count += 1;
            total_ceded += payout;
            total_layer_claims += 1;
        }
    }

    for entry in &mut by_year {
        entry.burning_cost_rate = entry.layer_ceded_loss / entry.adjusted_subject_premium;
    }

    Ok(BurningCostResult {
        by_year,
        total_adjusted_subject_premium: total_adj_prem,
        total_layer_ceded_loss: total_ceded,
        overall_burning_cost_rate: total_ceded / total_adj_prem,
        total_layer_claims,
    })
}

/// Reinstatement premium basis for excess-of-loss treaties (Sundt 1991 / Clark 2014).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReinstatementBasis {
    /// Pro-rata as to amount only (standard property catastrophe / per-occurrence XL).
    ProRataAmount,
    /// Pro-rata as to amount and time (uniform occurrence assumption scales reinstatement by `0.5`).
    ProRataAmountAndTime,
}

/// Solved reinstatement pricing breakdown for an excess-of-loss treaty.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReinstatementPricingResult {
    /// Total expected ceded loss across initial limit and all reinstatements.
    pub total_expected_ceded_loss: f64,
    /// Initial (upfront) treaty premium `P_0`.
    pub initial_premium: f64,
    /// Expected reinstatement premium collected `E[P_reinst]`.
    pub expected_reinstatement_premium: f64,
    /// Total expected premium (`initial_premium + expected_reinstatement_premium`).
    pub total_expected_premium: f64,
    /// Initial Rate-on-Line (`initial_premium / layer_limit`).
    pub initial_rate_on_line: f64,
    /// Effective Rate-on-Line including expected reinstatements (`total_expected_premium / layer_limit`).
    pub effective_rate_on_line: f64,
}

/// Price an excess-of-loss treaty with `K` reinstatements using Sundt (1991) exact balance.
///
/// - `layer_limit`: Occurrence layer limit `L > 0`.
/// - `expected_layer_losses`: Expected ceded loss in each aggregate layer slice `[E[S_0], E[S_1], ..., E[S_K]]`
///   where `S_0` is the initial layer (`0..L`) and `S_k` is the `k`-th reinstated layer (`k*L..(k+1)*L`).
///   Length is `K + 1` (at least 1).
/// - `reinstatement_rates`: Reinstatement premium percentage `c_k >= 0` for each reinstatement `k = 1..=K`
///   (e.g. `1.0` for 100% pro-rata, `0.0` for free reinstatement). Must have length `expected_layer_losses.len() - 1`.
/// - `expense_ratio`: Reinsurer expense + brokerage ratio `e in [0, 1)`.
/// - `basis`: Pro-rata by amount vs. pro-rata by amount and time.
///
/// Under Sundt (1991) actuarial balance:
/// `P_0 = (sum_{k=0}^K E[S_k]) / ((1 - e) * (1 + tau * sum_{k=1}^K c_k * E[S_{k-1}] / L))`
/// where `tau = 1.0` for `ProRataAmount` and `tau = 0.5` for `ProRataAmountAndTime`.
pub fn price_reinstatements(
    layer_limit: f64,
    expected_layer_losses: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
    basis: ReinstatementBasis,
) -> Result<ReinstatementPricingResult, PricingError> {
    if expected_layer_losses.is_empty() {
        return Err(PricingError::EmptyInput(
            "expected_layer_losses must contain at least the initial layer E[S_0]",
        ));
    }
    if reinstatement_rates.len() + 1 != expected_layer_losses.len() {
        return Err(PricingError::DimensionMismatch(
            "reinstatement_rates length must equal expected_layer_losses.len() - 1",
        ));
    }
    if !layer_limit.is_finite() || !(layer_limit > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "layer_limit must be finite and positive",
        ));
    }
    if !expense_ratio.is_finite() || !(expense_ratio >= 0.0) || !(expense_ratio < 1.0) {
        return Err(PricingError::InvalidParameter(
            "expense_ratio must be finite and in [0, 1)",
        ));
    }

    let mut total_expected_loss = 0.0;
    for &s in expected_layer_losses {
        if !s.is_finite() || !(s >= 0.0) || s > layer_limit + 1e-9 {
            return Err(PricingError::InvalidParameter(
                "each expected layer loss must be finite and in [0, layer_limit]",
            ));
        }
        total_expected_loss += s;
    }

    let time_factor = match basis {
        ReinstatementBasis::ProRataAmount => 1.0,
        ReinstatementBasis::ProRataAmountAndTime => 0.5,
    };

    let mut reinst_multiplier = 0.0;
    for (k, &c_k) in reinstatement_rates.iter().enumerate() {
        if !c_k.is_finite() || !(c_k >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "reinstatement rates must be finite and non-negative",
            ));
        }
        // Reinstatement k+1 is consumed by losses in aggregate layer k.
        reinst_multiplier += time_factor * c_k * (expected_layer_losses[k] / layer_limit);
    }

    let initial_premium = total_expected_loss / ((1.0 - expense_ratio) * (1.0 + reinst_multiplier));
    let expected_reinstatement_premium = initial_premium * reinst_multiplier;
    let total_expected_premium = initial_premium + expected_reinstatement_premium;

    Ok(ReinstatementPricingResult {
        total_expected_ceded_loss: total_expected_loss,
        initial_premium,
        expected_reinstatement_premium,
        total_expected_premium,
        initial_rate_on_line: initial_premium / layer_limit,
        effective_rate_on_line: total_expected_premium / layer_limit,
    })
}

/// Specification for a proportional treaty sliding-scale ceding commission (Clark 2014).
///
/// Commission slides linearly with slope `slide_ratio` (e.g. `1.0` for 1-for-1 or `0.5` for
/// 0.5% commission per 1% loss ratio) from `max_commission` at `min_lr` down to
/// `min_commission` at `max_lr = min_lr + (max_commission - min_commission) / slide_ratio`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlidingScaleSpec {
    /// Minimum ceding commission ratio (floor, e.g. `0.20`).
    pub min_commission: f64,
    /// Maximum ceding commission ratio (cap, e.g. `0.35`).
    pub max_commission: f64,
    /// Loss ratio at or below which `max_commission` is paid (e.g. `0.50`).
    pub min_lr: f64,
    /// Slide slope: reduction in commission per unit increase in loss ratio (e.g. `1.0` or `0.5`).
    pub slide_ratio: f64,
}

impl SlidingScaleSpec {
    /// Validate specification and return the upper loss ratio `max_lr` where `min_commission` is reached.
    pub fn max_lr(&self) -> Result<f64, PricingError> {
        if !self.min_commission.is_finite()
            || !self.max_commission.is_finite()
            || !self.min_lr.is_finite()
            || !self.slide_ratio.is_finite()
        {
            return Err(PricingError::InvalidParameter(
                "sliding scale parameters must be finite",
            ));
        }
        if !(self.min_commission >= 0.0)
            || !(self.max_commission >= self.min_commission)
            || !(self.max_commission <= 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "must satisfy 0 <= min_commission <= max_commission <= 1",
            ));
        }
        if !(self.min_lr >= 0.0) || !(self.slide_ratio > 0.0) {
            return Err(PricingError::InvalidParameter(
                "min_lr must be >= 0 and slide_ratio must be > 0",
            ));
        }
        let max_lr = self.min_lr + (self.max_commission - self.min_commission) / self.slide_ratio;
        if !max_lr.is_finite() {
            return Err(PricingError::OutOfBounds(
                "sliding scale max_lr overflowed floating-point range",
            ));
        }
        Ok(max_lr)
    }

    /// Deterministic ceding commission rate at realized loss ratio `lr`.
    pub fn commission_at(&self, lr: f64) -> Result<f64, PricingError> {
        let max_lr = self.max_lr()?;
        if !lr.is_finite() || !(lr >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "loss ratio must be finite and non-negative",
            ));
        }
        if lr <= self.min_lr {
            Ok(self.max_commission)
        } else if lr >= max_lr {
            Ok(self.min_commission)
        } else {
            Ok(self.max_commission - self.slide_ratio * (lr - self.min_lr))
        }
    }

    /// Expected sliding-scale ceding commission `E[C(LR)]` given an aggregate loss ratio
    /// distribution (`table_m`) and expected loss ratio `expected_lr` (CAS Exam 8 / Clark 2014):
    ///
    /// `E[C(LR)] = max_commission - slide_ratio * expected_lr * (phi(r_min) - phi(r_max))`
    pub fn expected_commission(
        &self,
        expected_lr: f64,
        table_m: &TableMEngine,
    ) -> Result<f64, PricingError> {
        let max_lr = self.max_lr()?;
        if !expected_lr.is_finite() || !(expected_lr > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "expected_lr must be finite and positive",
            ));
        }
        let r_min = self.min_lr / expected_lr;
        let r_max = max_lr / expected_lr;
        if !r_min.is_finite() || !r_max.is_finite() {
            return Err(PricingError::OutOfBounds(
                "sliding scale entry ratios overflowed floating-point range",
            ));
        }
        let phi_min = table_m.charge_at(r_min);
        let phi_max = table_m.charge_at(r_max);
        Ok(self.max_commission - self.slide_ratio * expected_lr * (phi_min - phi_max))
    }
}

/// Expected cedent retention under a proportional treaty loss corridor (Clark 2014).
///
/// The cedent retains `cedent_share` (in `[0, 1]`) of aggregate losses whose loss ratio falls
/// between `attach_lr` and `detach_lr`:
/// `E[Corridor] = cedent_share * expected_lr * (phi(attach_lr / expected_lr) - phi(detach_lr / expected_lr))`.
pub fn expected_loss_corridor_retention(
    attach_lr: f64,
    detach_lr: f64,
    cedent_share: f64,
    expected_lr: f64,
    table_m: &TableMEngine,
) -> Result<f64, PricingError> {
    if !attach_lr.is_finite()
        || !detach_lr.is_finite()
        || !cedent_share.is_finite()
        || !expected_lr.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "loss corridor parameters must be finite",
        ));
    }
    if !(attach_lr >= 0.0) || !(detach_lr > attach_lr) {
        return Err(PricingError::InvalidParameter(
            "must satisfy 0 <= attach_lr < detach_lr",
        ));
    }
    if !(cedent_share >= 0.0) || !(cedent_share <= 1.0) {
        return Err(PricingError::InvalidParameter(
            "cedent_share must be in [0, 1]",
        ));
    }
    if !(expected_lr > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "expected_lr must be positive",
        ));
    }
    let r_a = attach_lr / expected_lr;
    let r_d = detach_lr / expected_lr;
    if !r_a.is_finite() || !r_d.is_finite() {
        return Err(PricingError::OutOfBounds(
            "corridor entry ratios overflowed floating-point range",
        ));
    }
    let phi_a = table_m.charge_at(r_a);
    let phi_d = table_m.charge_at(r_d);
    Ok(cedent_share * expected_lr * (phi_a - phi_d))
}

/// Expected proportional treaty profit commission ratio (Clark 2014).
///
/// Reinsurer pays `profit_commission_rate * max(0, 1 - ceding_commission - reinsurer_expense - LR)`.
/// Using Table M savings `psi(r) = E[max(0, r - Y)]`:
/// `E[PC] = profit_commission_rate * expected_lr * psi(break_even_lr / expected_lr)`.
pub fn expected_profit_commission(
    ceding_commission: f64,
    reinsurer_expense: f64,
    profit_commission_rate: f64,
    expected_lr: f64,
    table_m: &TableMEngine,
) -> Result<f64, PricingError> {
    if !ceding_commission.is_finite()
        || !reinsurer_expense.is_finite()
        || !profit_commission_rate.is_finite()
        || !expected_lr.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "profit commission parameters must be finite",
        ));
    }
    if !(ceding_commission >= 0.0)
        || !(reinsurer_expense >= 0.0)
        || !(profit_commission_rate >= 0.0)
        || !(profit_commission_rate <= 1.0)
        || !(expected_lr > 0.0)
    {
        return Err(PricingError::InvalidParameter(
            "commission/expense must be >= 0, profit_commission_rate in [0, 1], expected_lr > 0",
        ));
    }
    let break_even_lr = 1.0 - ceding_commission - reinsurer_expense;
    if break_even_lr <= 0.0 {
        return Ok(0.0);
    }
    let r_be = break_even_lr / expected_lr;
    if !r_be.is_finite() {
        return Err(PricingError::OutOfBounds(
            "profit commission break-even entry ratio overflowed floating-point range",
        ));
    }
    let psi = table_m.savings_at(r_be);
    Ok(profit_commission_rate * expected_lr * psi)
}
