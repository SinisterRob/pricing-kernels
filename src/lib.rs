#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "wasm"), forbid(unsafe_code))]
#![warn(missing_docs)]

pub mod classification;
pub mod credibility;
pub mod datasets;
pub mod error;
pub mod experience;
pub mod layers;
pub(crate) mod linalg;
pub(crate) mod special;
pub mod trending;
pub mod underwriting_risk;

#[cfg(feature = "wasm")]
#[doc(hidden)]
pub mod wasm_exports;

/// Declarative constructor for a validated [`crate::layers::LayerTower`].
///
/// Syntax: `tower![ limit xs attachment @ participation, ... ]` (omitting `@ participation`
/// defaults to `1.0` = 100% share). Returns `Result<LayerTower, PricingError>`.
///
/// # Example
/// ```rust
/// use pricing_kernels::{prelude::*, tower};
///
/// let t = tower![
///     500_000.0 xs 500_000.0,
///     4_000_000.0 xs 1_000_000.0 @ 0.5,
/// ]?;
/// assert_eq!(t.len(), 2);
/// # Ok::<(), PricingError>(())
/// ```
#[macro_export]
macro_rules! tower {
    ( $( $limit:tt xs $attachment:tt $( @ $share:tt )? ),+ $(,)? ) => {{
        (|| -> ::core::result::Result<$crate::layers::LayerTower, $crate::error::PricingError> {
            let mut layers = ::std::vec::Vec::new();
            $(
                let share: f64 = $crate::__tower_share!($( $share )?);
                layers.push($crate::layers::CommercialLayer::new(
                    ($attachment) as f64,
                    ($limit) as f64,
                    share,
                )?);
            )+
            $crate::layers::LayerTower::try_new(layers)
        })()
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __tower_share {
    () => {
        1.0_f64
    };
    ($share:expr) => {
        ($share) as f64
    };
}

/// Declarative constructor for a `Vec<RateChange>` on-level rate schedule.
///
/// Syntax: `rate_changes![ effective_date => rate_factor, ... ]`.
///
/// # Example
/// ```rust
/// use pricing_kernels::{prelude::*, rate_changes};
///
/// let changes = rate_changes![2021.5 => 1.10, 2022.25 => 1.05];
/// let olf = parallelogram_on_level_factor(2022.0, &changes)?;
/// assert!(olf > 1.0);
/// # Ok::<(), PricingError>(())
/// ```
#[macro_export]
macro_rules! rate_changes {
    ( $( $date:expr => $factor:expr ),* $(,)? ) => {{
        vec![
            $(
                $crate::trending::RateChange {
                    effective_date: ($date) as f64,
                    rate_factor: ($factor) as f64,
                }
            ),*
        ]
    }};
}

/// Standard prelude re-exporting common pricing kernels and structs.
pub mod prelude {
    pub use crate::classification::{
        DesignMatrix, DoubleLiftBucket, GiniResult, GlmFamily, GlmFit, GlmOptions, GlmResult,
        InformationCriteria, LorenzCurvePoint, MinimumBiasResult, MultiplicativeRater, RatingTable,
        TweediePowerSelection, bailey_simon_chi_squared, bailey_simon_multiplicative,
        double_lift_chart, fit_glm_irls, fit_glm_irls_with_options, glm_information_criteria,
        predict_glm, select_tweedie_power, variance_inflation_factors, weighted_gini_index,
    };
    pub use crate::credibility::{
        BuhlmannStraubResult, ClassicalCredibility, HachemeisterGroupResult, HachemeisterResult,
        buhlmann_straub, hachemeister_credibility,
    };
    pub use crate::error::{PricingError, SolverOptions};
    pub use crate::experience::{
        AlaeTreatment, ExperienceClaim, IsoErpResult, IsoErpSpec, LargeDeductibleResult,
        LargeDeductibleSpec, NcciErpResult, NcciErpSpec, NcciErpSpecification, RetroPlanResult,
        RetroPlanSpec, ScheduleRatingItem, ScheduleRatingResult, TableLEngine, TableLRow,
        TableMEngine, TableMRow, calculate_iso_experience_mod, calculate_ncci_experience_mod,
        calculate_off_balance, calculate_schedule_mod, price_large_deductible,
        solve_balanced_retro_plan, solve_balanced_retro_plan_table_l,
    };
    pub use crate::layers::{
        BurningCostResult, BurningCostYearResult, CommercialLayer, HistoricalTreatyClaim,
        HistoricalTreatyYear, LayerTower, MbbefdCurve, ReinstatementBasis,
        ReinstatementPricingResult, SeverityDistribution, SlidingScaleSpec, SwissReCurve,
        burning_cost_xl, expected_loss_corridor_retention, expected_profit_commission,
        price_reinstatements,
    };
    pub use crate::trending::{
        RateChange, TrendModel, TwoStepTrendModel, parallelogram_on_level_factor,
        parallelogram_on_level_factor_with_term, policy_year_on_level_factor,
    };
    pub use crate::underwriting_risk::{
        AggregateLossDistribution, DistortionOperator, DistortionPricingResult, EltSummary,
        EulerAllocationResult, EventLossEntry, EventLossTable, FrequencyDistribution,
        KrepsRiskLoadResult, MyersReadAllocationResult, RmkPricingResult,
        convolve_independent_aggregates, discretize_severity_mean_preserving,
        distortion_price_aggregate, distortion_price_discrete, elt_aggregate_distribution,
        elt_occurrence_layer_expected_loss, elt_oep, elt_oep_pml, elt_summary,
        euler_rvar_allocation, euler_sd_allocation, euler_tvar_allocation, fft_aggregate_loss,
        kreps_marginal_risk_load, myers_read_allocation, panjer_recursion,
        rmk_risk_adjusted_pricing, sample_rvar, sample_tvar, sample_var,
    };
}
