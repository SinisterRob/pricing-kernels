//! Classification ratemaking, IRLS GLMs, and actuarial lift diagnostics.
//!
//! # Example
//! ```rust
//! use pricing_kernels::classification::{DesignMatrix, GlmFamily, GlmFit};
//!
//! let dm = DesignMatrix::try_from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])?;
//! let fit = GlmFit::new(GlmFamily::Poisson, &dm, &[2.0, 4.0, 8.0]).fit()?;
//! assert!(fit.coefficients[1] > 0.6);
//! # Ok::<(), pricing_kernels::error::PricingError>(())
//! ```

/// Actuarial lift and model diagnostics (Gini, Lorenz, double-lift, AIC/BIC, VIF).
pub mod diagnostics;
/// Iteratively Reweighted Least Squares (IRLS) GLM solver for actuarial count, severity, and pure premium.
pub mod irls_glm;
/// Bailey (1963) balance-principle and Bailey-Simon (1960) chi-squared minimum bias solvers.
pub mod minimum_bias;
/// Regularized Elastic Net GLMs, B-splines / Natural Cubic Splines, Hurdle/ZIP, and random-intercept GLMMs.
pub mod regularized_glm;
/// Zero-allocation multiplicative rating plan scorer.
pub mod scoring;
/// Exact Tweedie compound Poisson–Gamma density via the Dunn & Smyth (2005) series.
pub mod tweedie;

pub use diagnostics::{
    DoubleLiftBucket, GiniResult, InformationCriteria, LorenzCurvePoint, double_lift_chart,
    glm_information_criteria, variance_inflation_factors, weighted_gini_index,
};
pub use irls_glm::{
    DesignMatrix, GlmFamily, GlmFit, GlmOptions, GlmResult, RelativityRow, TweediePowerSelection,
    fit_glm_irls, fit_glm_irls_with_options, predict_glm, select_tweedie_power,
};
pub use minimum_bias::{
    MinimumBiasResult, RatingTable, bailey_simon_chi_squared, bailey_simon_multiplicative,
};
pub use regularized_glm::{
    RandomInterceptGlmmResult, RegularizedGlmFamily, RegularizedGlmOptions, RegularizedGlmResult,
    ZeroModifiedPoissonResult, bspline_basis, fit_hurdle_poisson, fit_random_intercept_glmm,
    fit_regularized_glm, fit_zero_inflated_poisson, natural_cubic_spline_basis,
};
pub use scoring::MultiplicativeRater;
pub use tweedie::tweedie_log_density;
