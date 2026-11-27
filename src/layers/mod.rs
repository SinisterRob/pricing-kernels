//! Severity layering, Increased Limit Factors (ILFs), deductible pricing,
//! MBBEFD exposure curves, and treaty reinsurance pricing.

/// Continuous severity distributions, ILFs, and excess layers.
pub mod ilf;
/// Bernegger (1997) MBBEFD and Swiss Re exposure rating curves.
pub mod mbbefd;
/// Treaty reinsurance pricing (burning cost, reinstatements, sliding scale, corridors, profit commission).
pub mod reinsurance;

pub use ilf::{CommercialLayer, SeverityDistribution};
pub use mbbefd::{MbbefdCurve, SwissReCurve};
pub use reinsurance::{
    BurningCostResult, BurningCostYearResult, HistoricalTreatyClaim, HistoricalTreatyYear,
    ReinstatementBasis, ReinstatementPricingResult, SlidingScaleSpec, burning_cost_xl,
    expected_loss_corridor_retention, expected_profit_commission, price_reinstatements,
};
