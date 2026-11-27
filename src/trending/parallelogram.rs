use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Historic rate change event.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RateChange {
    /// Date of rate change expressed as fractional year (e.g. `2021.5` for July 1, 2021).
    pub effective_date: f64,
    /// Multiplicative rate level change factor (e.g. `+10%` rate increase is `1.10`).
    pub rate_factor: f64,
}

fn validate_and_sort_changes(
    experience_year: f64,
    rate_changes: &[RateChange],
) -> Result<Vec<RateChange>, PricingError> {
    if !experience_year.is_finite() {
        return Err(PricingError::InvalidParameter(
            "experience_year must be finite",
        ));
    }
    let mut sorted = Vec::with_capacity(rate_changes.len());
    for &chg in rate_changes {
        if !chg.effective_date.is_finite() || !chg.rate_factor.is_finite() {
            return Err(PricingError::InvalidParameter(
                "effective_date and rate_factor must be finite",
            ));
        }
        if !(chg.rate_factor > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "rate factor must be strictly positive",
            ));
        }
        sorted.push(chg);
    }
    sorted.sort_by(|a, b| a.effective_date.total_cmp(&b.effective_date));
    Ok(sorted)
}

/// Fraction of Calendar-Year `Y` earned exposures (under uniform 1-year annual policies)
/// written on or after `effective_date = d`.
///
/// Exact geometric parallelogram area:
/// - `1.0` if `d <= Y - 1`
/// - `1.0 - 0.5 * (d - (Y - 1))^2` if `Y - 1 < d <= Y`
/// - `0.5 * (Y + 1 - d)^2` if `Y < d < Y + 1`
/// - `0.0` if `d >= Y + 1`
#[inline]
fn cy_earned_area_after(experience_year: f64, d: f64) -> f64 {
    let prev_year = experience_year - 1.0;
    let next_year = experience_year + 1.0;
    if d <= prev_year {
        1.0
    } else if d <= experience_year {
        let dt = d - prev_year;
        1.0 - 0.5 * dt * dt
    } else if d < next_year {
        let dt = next_year - d;
        0.5 * dt * dt
    } else {
        0.0
    }
}

/// Fraction of Policy-Year `Y` written exposures (uniform over `[Y, Y + 1]`)
/// written on or after `effective_date = d`.
#[inline]
fn py_written_area_after(policy_year: f64, d: f64) -> f64 {
    (policy_year + 1.0 - d).clamp(0.0, 1.0)
}

fn compute_on_level_from_area_fn<F>(
    experience_year: f64,
    rate_changes: &[RateChange],
    area_after: F,
) -> Result<f64, PricingError>
where
    F: Fn(f64, f64) -> f64,
{
    if !experience_year.is_finite() {
        return Err(PricingError::InvalidParameter(
            "experience_year must be finite",
        ));
    }
    if rate_changes.is_empty() {
        return Ok(1.0);
    }

    let sorted = validate_and_sort_changes(experience_year, rate_changes)?;

    // Telescoping area decomposition across arbitrarily many rate changes:
    // Weight on level L_0 = 1 - A(d_1)
    // Weight on level L_k = A(d_k) - A(d_{k+1})
    // Weight on level L_m = A(d_m)
    let mut level = 1.0;
    let mut prev_area_before = 0.0;
    let mut average_historical_level = 0.0;

    for chg in &sorted {
        let area_after_d = area_after(experience_year, chg.effective_date);
        let area_before_d = 1.0 - area_after_d;
        let slice_weight = (area_before_d - prev_area_before).max(0.0);
        average_historical_level += slice_weight * level;
        prev_area_before = area_before_d;
        level *= chg.rate_factor;
    }

    // Remaining area written at the final cumulative rate level
    let final_weight = (1.0 - prev_area_before).max(0.0);
    average_historical_level += final_weight * level;

    if !(average_historical_level > 0.0) {
        return Err(PricingError::SingularMatrix(
            "historical average rate level is non-positive",
        ));
    }

    Ok(level / average_historical_level)
}

/// Exact Geometric Parallelogram Method for Calendar-Year **earned** on-level factors
/// (CAS Exam 5 / Kallop 1975 / Werner & Modlin).
///
/// Assumes uniform policy writings and 1-year annual policy terms. Supports arbitrarily
/// many rate changes (sorted or unsorted), including rate changes in the prior year
/// `[Y - 1, Y]` that earn partially into calendar year `Y`.
pub fn parallelogram_on_level_factor(
    experience_year: f64,
    rate_changes: &[RateChange],
) -> Result<f64, PricingError> {
    compute_on_level_from_area_fn(experience_year, rate_changes, cy_earned_area_after)
}

/// Exact Policy-Year **written** on-level factor for uniform writings over `[Y, Y + 1]`.
///
/// Supports arbitrarily many rate changes (sorted or unsorted).
pub fn policy_year_on_level_factor(
    policy_year: f64,
    rate_changes: &[RateChange],
) -> Result<f64, PricingError> {
    compute_on_level_from_area_fn(policy_year, rate_changes, py_written_area_after)
}
