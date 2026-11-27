//! Classification ratemaking and relativity estimation kernels.

/// Iteratively Reweighted Least Squares (IRLS) GLM solver for actuarial count and severity.
pub mod irls_glm;
/// Bailey-Simon iterative minimum bias solver with balance property.
pub mod minimum_bias;
/// Zero-allocation multiplicative rating plan scorer.
pub mod scoring;

pub use irls_glm::{GlmFamily, GlmResult, fit_glm_irls};
pub use minimum_bias::{MinimumBiasResult, bailey_simon_multiplicative};
pub use scoring::MultiplicativeRater;
