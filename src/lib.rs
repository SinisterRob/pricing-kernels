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
}
