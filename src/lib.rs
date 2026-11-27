#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "wasm"), forbid(unsafe_code))]
#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod classification;
pub mod credibility;
pub mod datasets;
pub mod error;
pub mod experience;
pub mod layers;
pub(crate) mod linalg;
pub mod trending;
pub mod underwriting_risk;

#[cfg(feature = "wasm")]
#[doc(hidden)]
pub mod wasm_exports;

/// Standard prelude re-exporting common pricing kernels and structs.
pub mod prelude {
    pub use crate::classification::{
        GlmFamily, GlmResult, MinimumBiasResult, MultiplicativeRater, bailey_simon_chi_squared,
        bailey_simon_multiplicative, fit_glm_irls,
    };
    pub use crate::credibility::{BuhlmannStraubResult, ClassicalCredibility, buhlmann_straub};
    pub use crate::error::PricingError;
    pub use crate::experience::{
        AlaeTreatment, ExperienceClaim, IsoErpResult, IsoErpSpec, LargeDeductibleResult,
        LargeDeductibleSpec, NcciErpResult, NcciErpSpecification, RetroPlanResult, RetroPlanSpec,
        ScheduleRatingItem, ScheduleRatingResult, TableLEngine, TableLRow, TableMEngine, TableMRow,
        calculate_iso_experience_mod, calculate_ncci_experience_mod, calculate_off_balance,
        calculate_schedule_mod, price_large_deductible, solve_balanced_retro_plan,
        solve_balanced_retro_plan_table_l,
    };
    pub use crate::layers::{
        BurningCostResult, BurningCostYearResult, CommercialLayer, HistoricalTreatyClaim,
        HistoricalTreatyYear, MbbefdCurve, ReinstatementBasis, ReinstatementPricingResult,
        SeverityDistribution, SlidingScaleSpec, SwissReCurve, burning_cost_xl,
        expected_loss_corridor_retention, expected_profit_commission, price_reinstatements,
    };
    pub use crate::trending::{
        RateChange, TrendModel, parallelogram_on_level_factor, policy_year_on_level_factor,
    };
    pub use crate::underwriting_risk::{
        AggregateLossDistribution, DistortionOperator, DistortionPricingResult, EltSummary,
        EulerAllocationResult, EventLossEntry, FrequencyDistribution, KrepsRiskLoadResult,
        MyersReadAllocationResult, RmkPricingResult, convolve_independent_aggregates,
        discretize_severity_mean_preserving, distortion_price_aggregate, distortion_price_discrete,
        elt_aggregate_distribution, elt_occurrence_layer_expected_loss, elt_oep, elt_oep_pml,
        elt_summary, euler_rvar_allocation, euler_sd_allocation, euler_tvar_allocation,
        fft_aggregate_loss, kreps_marginal_risk_load, myers_read_allocation, panjer_recursion,
        rmk_risk_adjusted_pricing, sample_rvar, sample_tvar, sample_var,
    };
}
