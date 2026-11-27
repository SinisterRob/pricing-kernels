#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "wasm"), forbid(unsafe_code))]
#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod classification;
pub mod credibility;
pub mod error;
pub mod experience;
pub mod layers;
pub mod linalg;
pub mod trending;

/// Standard prelude re-exporting common pricing kernels and structs.
pub mod prelude {
    pub use crate::classification::{
        GlmFamily, GlmResult, MinimumBiasResult, MultiplicativeRater, bailey_simon_chi_squared,
        bailey_simon_multiplicative, fit_glm_irls,
    };
    pub use crate::credibility::{BuhlmannStraubResult, ClassicalCredibility, buhlmann_straub};
    pub use crate::error::PricingError;
    pub use crate::experience::{
        ExperienceClaim, NcciErpResult, NcciErpSpecification, RetroPlanResult, RetroPlanSpec,
        TableMEngine, TableMRow, calculate_ncci_experience_mod, calculate_off_balance,
        solve_balanced_retro_plan,
    };
    pub use crate::layers::{CommercialLayer, SeverityDistribution};
    pub use crate::trending::{
        RateChange, TrendModel, parallelogram_on_level_factor, policy_year_on_level_factor,
    };
}
