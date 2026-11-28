//! Severity layering, Increased Limit Factors (ILFs), deductible pricing,
//! MBBEFD exposure curves, and treaty reinsurance pricing.

/// Censored and truncated maximum likelihood fitting for parametric severity distributions.
pub mod fitting;
/// Continuous severity distributions, ILFs, and excess layers.
pub mod ilf;
/// Bernegger (1997) MBBEFD and Swiss Re exposure rating curves.
pub mod mbbefd;
/// Treaty reinsurance pricing (burning cost, reinstatements, sliding scale, corridors, profit commission).
pub mod reinsurance;

pub use fitting::{
    SeverityFamily, SeverityFitResult, SeverityObservation, fit_best_severity, fit_severity_mle,
};
pub use ilf::{CommercialLayer, LayerTower, SeverityDistribution};
pub use mbbefd::{MbbefdCurve, SwissReCurve};
pub use reinsurance::{
    BlendCredibilityRule, BurningCostResult, BurningCostYearResult, ExposureExperienceBlendResult,
    HistoricalTreatyClaim, HistoricalTreatyYear, ReinstatementBasis, ReinstatementPricingResult,
    SlidingScaleSpec, TreatyExperience, blend_exposure_and_experience, burning_cost_xl,
    expected_loss_corridor_retention, expected_profit_commission, price_reinstatements,
};
