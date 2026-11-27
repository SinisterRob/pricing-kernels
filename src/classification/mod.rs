//! Classification ratemaking and relativity estimation kernels.

/// Iteratively Reweighted Least Squares (IRLS) GLM solver for actuarial count and severity.
pub mod irls_glm;
/// Bailey (1963) balance-principle and Bailey-Simon (1960) chi-squared minimum bias solvers.
pub mod minimum_bias;
/// Zero-allocation multiplicative rating plan scorer.
pub mod scoring;

pub use irls_glm::{GlmFamily, GlmResult, fit_glm_irls};
pub use minimum_bias::{MinimumBiasResult, bailey_simon_chi_squared, bailey_simon_multiplicative};
pub use scoring::MultiplicativeRater;
