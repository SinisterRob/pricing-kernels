//! Commercial experience rating plans, Table M aggregate loss charges, and retrospective plans.

/// NCCI Workers' Compensation Experience Rating Plan formula.
pub mod ncci_erp;
/// Balanced retrospective rating plan solver.
pub mod retrospective;
/// Lee (1988) Table M aggregate insurance charge engine.
pub mod table_m;

pub use ncci_erp::{
    ExperienceClaim, NcciErpResult, NcciErpSpecification, calculate_ncci_experience_mod,
    calculate_off_balance,
};
pub use retrospective::{RetroPlanResult, RetroPlanSpec, solve_balanced_retro_plan};
pub use table_m::{TableMEngine, TableMRow};
