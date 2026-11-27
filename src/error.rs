//! Error types and error definitions for numerical pricing kernels.

use std::fmt;

/// Standard error type for pricing numerical kernels and validations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingError {
    /// Provided slice or vector input was empty.
    EmptyInput(&'static str),
    /// Dimension or slice length mismatch between inputs.
    DimensionMismatch(&'static str),
    /// Non-positive or negative value encountered where strictly positive is required.
    NonPositiveValue(&'static str),
    /// Parameter value outside valid mathematical domain.
    InvalidParameter(&'static str),
    /// Iterative numerical solver failed to reach specified tolerance.
    NonConvergence(&'static str),
    /// Non-invertible or singular matrix encountered.
    SingularMatrix(&'static str),
    /// Value evaluated outside supported boundaries.
    OutOfBounds(&'static str),
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput(msg) => write!(f, "empty input: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::NonPositiveValue(msg) => write!(f, "non-positive value encountered: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::NonConvergence(msg) => write!(f, "algorithm failed to converge: {msg}"),
            Self::SingularMatrix(msg) => write!(f, "singular or non-invertible matrix: {msg}"),
            Self::OutOfBounds(msg) => write!(f, "value out of bounds: {msg}"),
        }
    }
}

impl std::error::Error for PricingError {}
