use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Historic rate change event.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RateChange {
    /// Date of rate change expressed as fractional year (e.g. 2021.5 for July 1, 2021).
    pub effective_date: f64,
    /// Multiplicative rate level change factor (e.g. +10% rate increase is 1.10).
    pub rate_factor: f64,
}

/// Geometric Parallelogram Method for calculating historical on-level factors (CAS Exam 5 / Kallop).
/// Assumes policies are written uniformly throughout the experience year and have 1-year policy terms.
pub fn parallelogram_on_level_factor(
    experience_year: f64,
    rate_changes: &[RateChange],
) -> Result<f64, PricingError> {
    if rate_changes.is_empty() {
        return Ok(1.0);
    }

    // Historical policy writing block for year Y: written from Y to Y+1, earned from Y to Y+2.
    // The policy parallelogram has vertices: (Y, Y), (Y+1, Y+1), (Y+1, Y+2), (Y, Y+1).
    let year_start = experience_year;
    let year_end = experience_year + 1.0;

    // Current cumulative rate index across all historic changes
    let mut current_rate_level = 1.0;
    for chg in rate_changes {
        if chg.rate_factor <= 0.0 {
            return Err(PricingError::NonPositiveValue("rate factor must be positive"));
        }
        current_rate_level *= chg.rate_factor;
    }

    // If all rate changes happened after the entire earned period (year_end + 1.0), historical level = 1.0
    // If all rate changes happened before year_start, historical level = current_rate_level
    // We compute the average earned rate level across the parallelogram.
    // Area of writing parallelogram = 1.0.
    // For a rate change occurring at time t during writing year [Y, Y+1]:
    // The portion of exposures written under the new rate is (Y+1 - t).
    // The portion of earned exposures during the experience year is calculated based on coverage geometry.
    // For earned premium in calendar year Y:
    // If rate change happens at date d in [year_start, year_end]:
    // Portion of earned premium in year Y affected by new rate is: 0.5 * (year_end - d)^2.
    
    // For policy year Y (written in year Y):
    // If rate change occurs at date d in [year_start, year_end]:
    // Area written at new rate = (year_end - d).
    let mut historical_rate_level = 1.0;
    let mut prev_level = 1.0;

    for chg in rate_changes {
        let d = chg.effective_date;
        let next_level = prev_level * chg.rate_factor;

        if d <= year_start {
            // Already fully effective before this policy year began
            historical_rate_level = next_level;
        } else if d >= year_end {
            // Not effective at all during this policy year's writings
            // Remains at prev_level
        } else {
            // Effective midway through writing year:
            // Area at new rate = (year_end - d)
            // Area at old rate = (d - year_start)
            let p_new = (year_end - d) / (year_end - year_start);
            let p_old = 1.0 - p_new;
            historical_rate_level = p_old * prev_level + p_new * next_level;
        }
        prev_level = next_level;
    }

    if historical_rate_level <= 0.0 {
        return Err(PricingError::SingularMatrix("historical rate level is non-positive"));
    }

    // On-level factor = Current Rate Level / Historical Average Rate Level
    Ok(current_rate_level / historical_rate_level)
}
