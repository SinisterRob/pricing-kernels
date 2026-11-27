//! Error types and error definitions for numerical pricing kernels.

use std::fmt;

/// Standard error type for pricing numerical kernels and validations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
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

/// Common iterative solver convergence options (`max_iter`, `tol`) shared across
/// minimum-bias and retrospective rating plan solvers.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SolverOptions {
    /// Maximum number of solver iterations (default `200`).
    pub max_iter: usize,
    /// Absolute convergence tolerance (default `1e-9`).
    pub tol: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-9,
        }
    }
}

impl SolverOptions {
    /// Set maximum solver iterations.
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set absolute convergence tolerance.
    pub const fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}
