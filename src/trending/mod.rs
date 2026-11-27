//! Historical rate-level on-leveling and loss trending kernels.

/// Geometric parallelogram on-level factor calculation.
pub mod parallelogram;
/// Exponential loss cost and exposure trend models.
pub mod trend;

pub use parallelogram::{RateChange, parallelogram_on_level_factor, policy_year_on_level_factor};
pub use trend::TrendModel;
