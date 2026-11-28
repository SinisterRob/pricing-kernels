//! Historical rate-level on-leveling and loss trending kernels.

/// Overall rate level indication (Loss Ratio and Pure Premium methods, Werner & Modlin Ch. 8).
pub mod indication;
/// Geometric parallelogram on-level factor calculation (annual and arbitrary policy terms).
pub mod parallelogram;
/// Werner & Modlin ratemaking mechanics, Myers–Cohn / ROE profit loads, AWP drift, and rate capping.
pub mod ratemaking;
/// Single-step and two-step exponential loss cost and exposure trend models.
pub mod trend;

pub use indication::{
    ExpenseProvisions, RateIndicationPeriod, RateIndicationPeriodResult, RateIndicationResult,
    RateIndicationSpec, calculate_rate_indication,
};
pub use parallelogram::{
    RateChange, parallelogram_on_level_factor, parallelogram_on_level_factor_with_term,
    policy_year_on_level_factor,
};
pub use ratemaking::{
    LossRatioIndicationResult, MyersCohnResult, MyersCohnSpec, PremiumTrendResult,
    PurePremiumIndicationResult, RateCappingResult, RoeProfitLoadResult,
    extension_of_exposures_olep, loss_ratio_indication, myers_cohn_profit_load,
    pure_premium_indication, roe_implied_profit_load, solve_capped_rate_off_balance,
    two_step_premium_trend,
};
pub use trend::{TrendModel, TwoStepTrendModel};
