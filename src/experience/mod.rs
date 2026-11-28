//! Commercial experience rating plans, Table M / Table L aggregate loss charges,
//! retrospective rating plans, large deductibles, and ISO commercial ERP/IRPM.
//!
//! # Example
//! ```rust
//! use pricing_kernels::datasets::ncci_erp_worksheet;
//!
//! let data = ncci_erp_worksheet();
//! let res = data.spec.calculate(&data.claims)?;
//! assert!((res.mod_factor - data.expected_mod_rounded).abs() < 1e-12);
//! # Ok::<(), pricing_kernels::error::PricingError>(())
//! ```

/// Commercial large deductible pricing with ALAE treatment and aggregate caps.
pub mod deductible;
/// ISO Commercial General Liability / Auto experience and schedule rating (IRPM).
pub mod iso_erp;
/// NCCI Workers' Compensation Experience Rating Plan formula.
pub mod ncci_erp;
/// Balanced retrospective rating plan solver.
pub mod retrospective;
/// Skurnick (1974) Table L aggregate loss charge engine with per-occurrence loss limits.
pub mod table_l;
/// Lee (1988) Table M aggregate insurance charge engine.
pub mod table_m;

pub use deductible::{
    AlaeTreatment, LargeDeductibleResult, LargeDeductibleSpec, price_large_deductible,
};
pub use iso_erp::{
    IsoErpResult, IsoErpSpec, ScheduleRatingItem, ScheduleRatingResult,
    calculate_iso_experience_mod, calculate_schedule_mod,
};
pub use ncci_erp::{
    ERA_MEDICAL_ONLY_RATABLE_FACTOR, ExperienceClaim, NCCI_MULTIPLE_CLAIM_LIMIT_MULTIPLIER,
    NcciAccidentLimitation, NcciErpResult, NcciErpSpec, calculate_ncci_experience_mod,
    calculate_ncci_experience_mod_by_accident, calculate_off_balance,
};
pub use retrospective::{RetroPlanResult, RetroPlanSpec, solve_balanced_retro_plan};
pub use table_l::{TableLEngine, TableLRow, solve_balanced_retro_plan_table_l};
pub use table_m::{TableMEngine, TableMRow};
