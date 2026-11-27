//! Historical rate-level on-leveling and loss trending kernels.

/// Geometric parallelogram on-level factor calculation (annual and arbitrary policy terms).
pub mod parallelogram;
/// Single-step and two-step exponential loss cost and exposure trend models.
pub mod trend;

pub use parallelogram::{
    RateChange, parallelogram_on_level_factor, parallelogram_on_level_factor_with_term,
    policy_year_on_level_factor,
};
pub use trend::{TrendModel, TwoStepTrendModel};
