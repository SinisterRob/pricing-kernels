//! Credibility models for commercial risk accounts, classification plans, and trend.

/// Non-parametric Bühlmann-Straub credibility for heterogeneous risk groups.
pub mod buhlmann_straub;
/// Classical / limited fluctuation credibility standard.
pub mod classical;
/// Hachemeister (1975) regression credibility for credibility-weighted trend.
pub mod hachemeister;

pub use buhlmann_straub::{BuhlmannStraubResult, buhlmann_straub};
pub use classical::ClassicalCredibility;
pub use hachemeister::{HachemeisterGroupResult, HachemeisterResult, hachemeister_credibility};
