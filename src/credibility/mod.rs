//! Credibility models for commercial risk accounts and classification plans.

/// Non-parametric Bühlmann-Straub credibility for heterogeneous risk groups.
pub mod buhlmann_straub;
/// Classical / limited fluctuation credibility standard.
pub mod classical;

pub use buhlmann_straub::{BuhlmannStraubResult, buhlmann_straub};
pub use classical::ClassicalCredibility;
