//! # pricing-kernels
//!
//! Pure Rust numeric kernels for P&C commercial actuarial pricing, classification,
//! experience rating, and retrospective plans.
//!
//! Zero heavy dependencies, WASM-ready, deterministic, and verified against published
//! actuarial literature (Bailey & Simon 1960, Lee 1988, NCCI ERP, Bühlmann-Straub, CAS Exam 5 & 8).

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
        GlmFamily, GlmResult, MinimumBiasResult, MultiplicativeRater, bailey_simon_multiplicative,
        fit_glm_irls,
    };
    pub use crate::credibility::{BuhlmannStraubResult, ClassicalCredibility, buhlmann_straub};
    pub use crate::error::PricingError;
    pub use crate::experience::{
        ExperienceClaim, NcciErpResult, NcciErpSpecification, RetroPlanResult, RetroPlanSpec,
        TableMEngine, TableMRow, calculate_ncci_experience_mod, calculate_off_balance,
        solve_balanced_retro_plan,
    };
    pub use crate::layers::{CommercialLayer, SeverityDistribution};
    pub use crate::trending::{RateChange, TrendModel, parallelogram_on_level_factor};
}
