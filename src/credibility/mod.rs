//! Credibility models for commercial risk accounts, classification plans, and trend.
//!
//! # Example
//! ```rust
//! use pricing_kernels::credibility::{ClassicalCredibility, CredibilityPanel};
//!
//! let z = ClassicalCredibility::standard_90_5().credibility(400.0);
//! assert!(z > 0.6 && z < 0.62);
//!
//! let mut panel = CredibilityPanel::new();
//! panel.add_group("Class A", vec![100.0, 110.0], vec![10.0, 12.0])?;
//! panel.add_group("Class B", vec![150.0, 160.0], vec![20.0, 22.0])?;
//! let res = panel.fit_buhlmann_straub()?;
//! assert!(res.credibilities[1] > res.credibilities[0]);
//! # Ok::<(), pricing_kernels::error::PricingError>(())
//! ```

/// Non-parametric Bühlmann-Straub credibility for heterogeneous risk groups.
pub mod buhlmann_straub;
/// Classical / limited fluctuation credibility standard.
pub mod classical;
/// Hachemeister (1975) regression credibility for credibility-weighted trend.
pub mod hachemeister;
/// Jewell (1975) two-level hierarchical credibility (Sector → Risk → Period).
pub mod hierarchical;

pub use buhlmann_straub::{
    BuhlmannStraubResult, CredibilityGroup, CredibilityPanel, buhlmann_straub,
};
pub use classical::ClassicalCredibility;
pub use hachemeister::{HachemeisterGroupResult, HachemeisterResult, hachemeister_credibility};
pub use hierarchical::{
    HierarchicalCredibilityResult, HierarchicalRisk, HierarchicalRiskResult, HierarchicalSector,
    HierarchicalSectorResult, jewell_hierarchical_credibility,
};
