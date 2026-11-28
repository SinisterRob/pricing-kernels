//! Classification ratemaking, IRLS GLMs, and actuarial lift diagnostics.

/// Actuarial lift and model diagnostics (Gini, Lorenz, double-lift, AIC/BIC, VIF).
pub mod diagnostics;
/// Iteratively Reweighted Least Squares (IRLS) GLM solver for actuarial count, severity, and pure premium.
pub mod irls_glm;
/// Bailey (1963) balance-principle and Bailey-Simon (1960) chi-squared minimum bias solvers.
pub mod minimum_bias;
/// Zero-allocation multiplicative rating plan scorer.
pub mod scoring;

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
pub use scoring::MultiplicativeRater;
