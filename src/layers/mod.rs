//! Severity layering, Increased Limit Factors (ILFs), deductible pricing,
//! MBBEFD exposure curves, and treaty reinsurance pricing.
//!
//! # Example
//! ```rust
//! use pricing_kernels::layers::{SeverityDistribution, MbbefdCurve, SwissReCurve};
//!
//! let sev = SeverityDistribution::Lomax { alpha: 2.0, theta: 100_000.0 };
//! let ilf = sev.increased_limits_factor(500_000.0, 100_000.0)?;
//! assert!((ilf - 5.0 / 3.0).abs() < 1e-12);
//!
//! let factor = MbbefdCurve::from_preset(SwissReCurve::Y3)
//!     .layer_exposure_factor(1_000_000.0, 4_000_000.0, 10_000_000.0)?;
//! assert!(factor > 0.0 && factor < 1.0);
//! # Ok::<(), pricing_kernels::error::PricingError>(())
//! ```

/// ALAE-aware excess layer and Increased Limits Factor (ILF) pricing kernels.
pub mod alae_ilf;
/// Censored and truncated MLE + EVT and spliced body–tail severity fitting.
pub mod evt_fitting;
/// Censored and truncated maximum likelihood fitting for parametric severity distributions.
pub mod fitting;
/// Continuous severity distributions, ILFs, and excess layers.
pub mod ilf;
/// Bernegger (1997) MBBEFD and Swiss Re exposure rating curves.
pub mod mbbefd;
/// Treaty reinsurance pricing (burning cost, reinstatements, sliding scale, corridors, profit commission).
pub mod reinsurance;

pub use alae_ilf::{
    AlaeDependenceModel, AlaeIlfResult, AlaePolicyTreatment, alae_aware_ilf,
    alae_excess_layer_expected_loss, alae_loss_severity, gauss_kronrod_integrate,
    gauss_kronrod_integrate_semi_infinite,
};
pub use evt_fitting::{
    EvtFamily, EvtFitResult, FittedEvtDistribution, GoodnessOfFitDiagnostics, SpliceBodyFamily,
    SpliceContinuity, empirical_mean_excess, fit_evt_mle, fit_evt_mle_with_options,
    goodness_of_fit,
};

pub use fitting::{
    SeverityFamily, SeverityFitResult, SeverityObservation, fit_best_severity, fit_severity_mle,
};
pub use ilf::{CommercialLayer, LayerTower, SeverityDistribution};
pub use mbbefd::{MbbefdCurve, SwissReCurve};
pub use reinsurance::{
    BlendCredibilityRule, BurningCostResult, BurningCostYearResult, ExposureExperienceBlendResult,
    HistoricalTreatyClaim, HistoricalTreatyYear, ProportionalTreatyResult, QuotaShareSpec,
    ReinstatementBasis, ReinstatementPricingResult, SlidingScaleSpec, SurplusShareSpec,
    TreatyExperience, blend_exposure_and_experience, burning_cost_xl,
    expected_loss_corridor_retention, expected_profit_commission, price_reinstatements,
};
