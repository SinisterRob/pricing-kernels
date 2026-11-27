use std::fmt;

/// Standard error type for pricing numerical kernels and validations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingError {
    EmptyInput(&'static str),
    DimensionMismatch(&'static str),
    NonPositiveValue(&'static str),
    InvalidParameter(&'static str),
    NonConvergence(&'static str),
    SingularMatrix(&'static str),
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
