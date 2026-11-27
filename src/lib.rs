//! # pricing-kernels
//!
//! Pure Rust numeric kernels for P&C commercial actuarial pricing, classification,
//! experience rating, and retrospective plans.
//!
//! Zero heavy dependencies, WASM-ready, deterministic, and verified against published
//! actuarial literature (Bailey & Simon 1960, Lee 1988, NCCI ERP, Bühlmann-Straub, CAS Exam 5 & 8).

pub mod error;
pub mod linalg;
pub mod classification;
pub mod credibility;
pub mod experience;
pub mod layers;
pub mod trending;

/// Standard prelude re-exporting common pricing kernels and structs.
pub mod prelude {
    pub use crate::error::PricingError;
    pub use crate::classification::{
        bailey_simon_multiplicative, fit_glm_irls, GlmFamily, GlmResult,
        MinimumBiasResult, MultiplicativeRater,
    };
    pub use crate::credibility::{
        buhlmann_straub, BuhlmannStraubResult, ClassicalCredibility,
    };
    pub use crate::experience::{
        calculate_ncci_experience_mod, calculate_off_balance, solve_balanced_retro_plan,
        ExperienceClaim, NcciErpResult, NcciErpSpecification, RetroPlanResult,
        RetroPlanSpec, TableMEngine, TableMRow,
    };
    pub use crate::layers::{CommercialLayer, SeverityDistribution};
    pub use crate::trending::{
        parallelogram_on_level_factor, RateChange, TrendModel,
    };
}
