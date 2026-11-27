//! Severity layering, Increased Limit Factors (ILFs), and deductible pricing.

/// Continuous severity distributions, ILFs, and excess layers.
pub mod ilf;

pub use ilf::{CommercialLayer, SeverityDistribution};
