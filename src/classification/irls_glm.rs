//! Iteratively Reweighted Least Squares (IRLS) GLM solver with Householder QR WLS,
//! optional L2 (Ridge) penalization, deviance-monitored step-halving, and Tweedie power
//! selection by exact Dunn & Smyth (2005) series profile likelihood.
//!
//! Step-halving follows the `glm.fit` / Marschner (2011, *The R Journal* 3(2), "glm2: Fitting
//! Generalized Linear Models with Convergence Problems") convention: whenever a full IRLS step
//! yields a non-finite or increased (penalized) deviance, the step is halved toward the previous
//! iterate up to [`MAX_STEP_HALVINGS`] times before returning [`PricingError::NonConvergence`].

use super::tweedie::{TweedieSeries, maximize_over_dispersion, tweedie_log_likelihood};
use crate::error::PricingError;
use crate::linalg::{
    QrWorkspace, weighted_least_squares_qr_ridge_cm_beta_into,
    weighted_least_squares_qr_ridge_cm_cov,
};
use serde::{Deserialize, Serialize};

/// Maximum number of IRLS step-halvings per iteration before declaring divergence.
pub const MAX_STEP_HALVINGS: usize = 30;
/// Relative tolerance on a (penalized) deviance increase that triggers step-halving.
const DEVIANCE_INCREASE_RTOL: f64 = 1e-10;
/// Additive constant in the `glm.fit` relative-deviance stop `|D - D_old| / (|D| + 0.1) < tol`.
const DEVIANCE_RTOL_OFFSET: f64 = 0.1;
/// If IRLS stops on the deviance criterion while some fitted linear predictor `eta_i = x_i^T beta`
/// still moves by more than this per iteration, the estimates are drifting to infinity
/// (zero-claim level / separation). Measured on `eta` (dimensionless on the link scale), not on
/// raw coefficients, so the flag is invariant to rescaling any covariate.
const SEPARATION_STEP_THRESHOLD: f64 = 1e-2;
/// Binomial fitted means within this distance of 0 or 1 are numerically on the boundary
/// (`10 ×` the inverse-link clamp, as in R `glm.fit`'s "fitted probabilities 0 or 1").
const BINOMIAL_BOUNDARY_EPS: f64 = 1e-11;
/// Log-link fitted means below this are numerically zero (`10 ×` the inverse-link floor).
const LOG_LINK_BOUNDARY_MU: f64 = 1e-14;

/// GLM error family for commercial actuarial pricing kernels.
///
/// All count/severity/pure-premium families use the standard log link (`log(mu) = X beta + offset`),
/// while `Binomial` uses the canonical logit link (`logit(mu) = X beta + offset`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// Poisson family with log link: `V(mu) = mu`, `phi = 1.0`.
    Poisson,
    /// Overdispersed (quasi-)Poisson family with log link: `V(mu) = mu`, `phi = Pearson chi^2 / df`.
    QuasiPoisson,
    /// Gamma family with log link: `V(mu) = mu^2`, `phi = Pearson chi^2 / df`.
    Gamma,
    /// Compound Poisson–Gamma Tweedie family with log link and variance power `power in (1, 2)`:
    /// `V(mu) = mu^power`, `phi = Pearson chi^2 / df`.
    Tweedie {
        /// Tweedie variance power `1.0 < power < 2.0` (typical P&C pure premium: `1.5`–`1.7`).
        power: f64,
    },
    /// Negative Binomial family with log link and overdispersion `alpha > 0`:
    /// `V(mu) = mu + alpha * mu^2`, `phi = 1.0`.
    NegativeBinomial {
        /// Overdispersion parameter `alpha > 0` (`alpha -> 0` recovers Poisson).
        alpha: f64,
    },
    /// Binomial / Logistic family with canonical logit link for binary/proportion outcomes `y in [0, 1]`:
    /// `V(mu) = mu * (1 - mu)`, `phi = 1.0`.
    Binomial,
    /// Inverse Gaussian family with log link for heavy-tailed severity (`y > 0`):
    /// `V(mu) = mu^3`, `phi = Pearson chi^2 / df`.
    InverseGaussian,
}

impl GlmFamily {
    /// Validate family-specific parameters.
    pub fn validate(self) -> Result<(), PricingError> {
        match self {
            Self::Tweedie { power } => {
                if !power.is_finite() || !(power > 1.0) || !(power < 2.0) {
                    return Err(PricingError::InvalidParameter(
                        "Tweedie power must be finite and strictly in (1, 2)",
                    ));
                }
            }
            Self::NegativeBinomial { alpha } => {
                if !alpha.is_finite() || !(alpha > 0.0) {
                    return Err(PricingError::NonPositiveValue(
                        "NegativeBinomial alpha must be finite and positive",
                    ));
                }
            }
            Self::Poisson
            | Self::QuasiPoisson
            | Self::Gamma
            | Self::Binomial
            | Self::InverseGaussian => {}
        }
        Ok(())
    }

    #[inline]
    fn inverse_link(self, eta: f64) -> f64 {
        match self {
            Self::Binomial => {
                let e = eta.clamp(-35.0, 35.0);
                let p = if e >= 0.0 {
                    1.0 / (1.0 + (-e).exp())
                } else {
                    let ex = e.exp();
                    ex / (1.0 + ex)
                };
                p.clamp(1e-12, 1.0 - 1e-12)
            }
            _ => eta.clamp(-700.0, 700.0).exp().max(1e-15),
        }
    }

    #[inline]
    fn irls_weight(self, wi: f64, mui: f64) -> f64 {
        match self {
            Self::Poisson | Self::QuasiPoisson => wi * mui,
            Self::Gamma => wi,
            Self::Tweedie { power } => wi * mui.powf(2.0 - power),
            Self::NegativeBinomial { alpha } => wi * mui / (1.0 + alpha * mui),
            Self::Binomial => wi * (mui * (1.0 - mui)).max(1e-12),
            Self::InverseGaussian => wi / mui,
        }
    }

    #[inline]
    fn working_response(self, eta: f64, offset: f64, yi: f64, mui: f64) -> f64 {
        match self {
            Self::Binomial => (eta - offset) + (yi - mui) / (mui * (1.0 - mui)).max(1e-12),
            _ => (eta - offset) + (yi - mui) / mui,
        }
    }

    #[inline]
    pub(crate) fn unit_deviance(self, yi: f64, mui: f64) -> f64 {
        match self {
            Self::Poisson | Self::QuasiPoisson => {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * mui
                }
            }
            Self::Gamma => 2.0 * (-((yi / mui).ln()) + (yi - mui) / mui),
            Self::Tweedie { power } => {
                let one_minus_p = 1.0 - power;
                let two_minus_p = 2.0 - power;
                if yi > 0.0 {
                    2.0 * (yi.powf(two_minus_p) / (one_minus_p * two_minus_p)
                        - yi * mui.powf(one_minus_p) / one_minus_p
                        + mui.powf(two_minus_p) / two_minus_p)
                } else {
                    2.0 * mui.powf(two_minus_p) / two_minus_p
                }
            }
            Self::NegativeBinomial { alpha } => {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln()
                        - (yi + 1.0 / alpha) * ((1.0 + alpha * yi) / (1.0 + alpha * mui)).ln())
                } else {
                    (2.0 / alpha) * (alpha * mui).ln_1p()
                }
            }
            Self::Binomial => {
                let t1 = if yi > 0.0 { yi * (yi / mui).ln() } else { 0.0 };
                let t2 = if yi < 1.0 {
                    (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln()
                } else {
                    0.0
                };
                2.0 * (t1 + t2)
            }
            Self::InverseGaussian => {
                let ratio = (yi - mui) / mui;
                (ratio * ratio) / yi
            }
        }
    }

    /// Returns `true` for families whose dispersion parameter `phi` is estimated
    /// from the data rather than fixed at `1.0`.
    #[inline]
    pub(crate) const fn has_free_dispersion(self) -> bool {
        matches!(
            self,
            Self::QuasiPoisson | Self::Gamma | Self::Tweedie { .. } | Self::InverseGaussian
        )
    }

    #[inline]
    pub(crate) fn variance_fn(self, mui: f64) -> f64 {
        match self {
            Self::Poisson | Self::QuasiPoisson => mui,
            Self::Gamma => mui * mui,
            Self::Tweedie { power } => mui.powf(power),
            Self::NegativeBinomial { alpha } => mui + alpha * mui * mui,
            Self::Binomial => (mui * (1.0 - mui)).max(1e-12),
            Self::InverseGaussian => mui * mui * mui,
        }
    }
}

/// Configuration options for penalized or unpenalized IRLS GLM fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlmOptions {
    /// Maximum IRLS iterations (default `100`).
    pub max_iter: usize,
    /// Convergence tolerance (default `1e-9`). IRLS stops when the relative change in
    /// (penalized) deviance `|D - D_old| / (|D| + 0.1) < tol` (R `glm.fit`) or, secondarily,
    /// when `max_j |beta_new[j] - beta[j]| <= tol`.
    pub tol: f64,
    /// Optional L2 (Ridge) penalty `lambda >= 0` added to `(X^T W X)` diagonal for
    /// credibility-shrunk commercial relativities (default `0.0`).
    pub ridge_lambda: f64,
    /// Whether to apply `ridge_lambda` to column `0` (intercept). Defaults to `false`.
    pub penalize_intercept: bool,
}

impl Default for GlmOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            ridge_lambda: 0.0,
            penalize_intercept: false,
        }
    }
}

impl GlmOptions {
    /// Set maximum IRLS iterations.
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance (relative deviance change / absolute coefficient change).
    pub const fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set L2 (Ridge) penalty `lambda >= 0`.
    pub const fn with_ridge_lambda(mut self, ridge_lambda: f64) -> Self {
        self.ridge_lambda = ridge_lambda;
        self
    }

    /// Set whether to penalize column `0` (intercept).
    pub const fn with_penalize_intercept(mut self, penalize_intercept: bool) -> Self {
        self.penalize_intercept = penalize_intercept;
        self
    }
}

/// Rate-filing relativity row derived from a log-link GLM coefficient (`exp(beta_j)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativityRow {
    /// Feature / rating-factor name (`"intercept"`, `"x1"`, etc. when unnamed).
    pub name: String,
    /// Log-scale GLM coefficient `beta_j`.
    pub beta: f64,
    /// Asymptotic standard error `SE(beta_j)`.
    pub std_error: f64,
    /// Multiplicative relativity `exp(beta_j)`.
    pub relativity: f64,
    /// Lower 95% Wald confidence bound `exp(beta_j - 1.959964 * SE)`.
    pub ci_low_95: f64,
    /// Upper 95% Wald confidence bound `exp(beta_j + 1.959964 * SE)`.
    pub ci_high_95: f64,
    /// Wald z-statistic `beta_j / SE(beta_j)` (`0.0` if `SE == 0.0`).
    pub z_score: f64,
}

/// Fitted GLM result with family, coefficients, standard errors, dispersion, and deviance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmResult {
    /// Error family and link function used to fit this GLM.
    pub family: GlmFamily,
    /// Estimated regression coefficients (`beta`).
    pub coefficients: Vec<f64>,
    /// Asymptotic standard errors for coefficients: `sqrt(diag(phi * (X^T W X + Lambda)^-1))`.
    pub standard_errors: Vec<f64>,
    /// Dispersion parameter (`phi`): `1.0` for `Poisson`, `NegativeBinomial`, `Binomial`;
    /// Pearson `chi^2 / df` for `QuasiPoisson`, `Gamma`, `Tweedie`, and `InverseGaussian`.
    pub dispersion: f64,
    /// Residual deviance: `sum w_i * d(y_i, mu_i)`.
    pub deviance: f64,
    /// Residual degrees of freedom (`n_active - p`).
    pub df_residual: usize,
    /// Number of IRLS iterations until convergence.
    pub iterations: usize,
    /// Potentially infinite estimates: the fit stopped on the deviance criterion while some
    /// linear predictor `eta_i` was still drifting (`> 1e-2` per iteration on the link scale,
    /// hence invariant to covariate units), or some fitted mean is
    /// numerically on the boundary (`μ ≈ 0`, or `μ ≈ 1` for Binomial). Typical causes are a
    /// rating level with zero claims (log link) or complete / quasi-complete separation
    /// (Binomial); the affected relativities are effectively `0` (or `∞`) and their standard
    /// errors are meaningless (Hauck–Donner). Mirrors R `glm.fit`'s boundary warnings.
    #[serde(default)]
    pub separation_suspected: bool,
}

impl GlmResult {
    /// Convert log-link GLM coefficients into multiplicative rating relativities (or odds
    /// ratios for [`GlmFamily::Binomial`]) with 95% Wald confidence intervals and z-scores.
    pub fn relativities(
        &self,
        feature_names: Option<&[&str]>,
    ) -> Result<Vec<RelativityRow>, PricingError> {
        const Z_95: f64 = 1.959_963_984_540_054;
        if let Some(names) = feature_names
            && names.len() != self.coefficients.len()
        {
            return Err(PricingError::DimensionMismatch(
                "feature_names length must match number of GLM coefficients",
            ));
        }
        let mut rows = Vec::with_capacity(self.coefficients.len());
        for (j, (&beta, &se)) in self
            .coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .enumerate()
        {
            let name = match feature_names {
                Some(names) => names[j].to_owned(),
                None if j == 0 => "intercept".to_owned(),
                None => format!("x{j}"),
            };
            let z_score = if se > 0.0 { beta / se } else { 0.0 };
            rows.push(RelativityRow {
                name,
                beta,
                std_error: se,
                relativity: beta.exp(),
                ci_low_95: (beta - Z_95 * se).exp(),
                ci_high_95: (beta + Z_95 * se).exp(),
                z_score,
            });
        }
        Ok(rows)
    }

    /// Construct a [`crate::classification::MultiplicativeRater`] using `exp(coefficients[0])`
    /// as the base rate for log-link GLMs (with intercept in column `0`).
    ///
    /// Rejects [`GlmFamily::Binomial`] (logit link), where `exp(beta[0])` is baseline odds
    /// rather than a multiplicative rate.
    pub fn to_rater(&self) -> Result<crate::classification::MultiplicativeRater, PricingError> {
        if matches!(self.family, GlmFamily::Binomial) {
            return Err(PricingError::InvalidParameter(
                "to_rater requires a log-link GLM; Binomial uses a logit link",
            ));
        }
        let Some(&b0) = self.coefficients.first() else {
            return Err(PricingError::EmptyInput(
                "GLM result has no intercept coefficient",
            ));
        };
        let base_rate = b0.exp();
        if !base_rate.is_finite() || !(base_rate > 0.0) {
            return Err(PricingError::InvalidParameter(
                "exp(intercept) must be finite and positive",
            ));
        }
        Ok(crate::classification::MultiplicativeRater::new(base_rate))
    }
}

/// Validated row-major `n_samples × n_features` design matrix for GLM fitting and prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignMatrix {
    n_samples: usize,
    n_features: usize,
    data: Vec<f64>,
}

impl DesignMatrix {
    /// Construct and validate a row-major `n_samples × n_features` design matrix.
    pub fn try_new(
        n_samples: usize,
        n_features: usize,
        data: Vec<f64>,
    ) -> Result<Self, PricingError> {
        if n_samples == 0 || n_features == 0 {
            return Err(PricingError::EmptyInput(
                "samples and features must be non-zero",
            ));
        }
        let expected = n_samples
            .checked_mul(n_features)
            .ok_or(PricingError::DimensionMismatch(
                "n_samples * n_features overflowed",
            ))?;
        if data.len() != expected {
            return Err(PricingError::DimensionMismatch(
                "design matrix length != n_samples * n_features",
            ));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(PricingError::InvalidParameter(
                "design matrix contains non-finite value",
            ));
        }
        Ok(Self {
            n_samples,
            n_features,
            data,
        })
    }

    /// Construct and validate a design matrix from rectangular row slices.
    pub fn try_from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, PricingError> {
        let n_samples = rows.len();
        if n_samples == 0 {
            return Err(PricingError::EmptyInput("rows must be non-empty"));
        }
        let n_features = rows[0].as_ref().len();
        if n_features == 0 {
            return Err(PricingError::EmptyInput("columns must be non-zero"));
        }
        let mut data = Vec::with_capacity(n_samples * n_features);
        for row in rows {
            let slice = row.as_ref();
            if slice.len() != n_features {
                return Err(PricingError::DimensionMismatch(
                    "all design matrix rows must have identical length",
                ));
            }
            if slice.iter().any(|v| !v.is_finite()) {
                return Err(PricingError::InvalidParameter(
                    "design matrix contains non-finite value",
                ));
            }
            data.extend_from_slice(slice);
        }
        Ok(Self {
            n_samples,
            n_features,
            data,
        })
    }

    /// Number of observations (rows).
    #[inline]
    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of model features (columns).
    #[inline]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Flat row-major backing slice of length `n_samples * n_features`.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Row slice for observation `i`, or `None` if out of bounds.
    #[inline]
    pub fn row(&self, i: usize) -> Option<&[f64]> {
        if i >= self.n_samples {
            return None;
        }
        let start = i * self.n_features;
        Some(&self.data[start..start + self.n_features])
    }

    /// Predict expected responses `mu_i = g^-1(X_i beta + offset_i)`.
    pub fn predict(
        &self,
        family: GlmFamily,
        coefficients: &[f64],
        offsets: Option<&[f64]>,
    ) -> Result<Vec<f64>, PricingError> {
        if let Some(off) = offsets {
            predict_glm(
                family,
                &self.data,
                self.n_samples,
                self.n_features,
                coefficients,
                off,
            )
        } else {
            let zeros = vec![0.0; self.n_samples];
            predict_glm(
                family,
                &self.data,
                self.n_samples,
                self.n_features,
                coefficients,
                &zeros,
            )
        }
    }

    /// Fit an Elastic Net / Ridge / LASSO penalized GLM directly on this [`DesignMatrix`]
    /// (treat columns as predictor features; intercept is added automatically and any
    /// constant column — such as the explicit intercept column used by [`GlmFit`] — is
    /// dropped, with coefficient `0.0`).
    pub fn fit_regularized(
        &self,
        y: &[f64],
        weights: Option<&[f64]>,
        offsets: Option<&[f64]>,
        options: super::regularized_glm::RegularizedGlmOptions,
    ) -> Result<super::regularized_glm::RegularizedGlmResult, PricingError> {
        super::regularized_glm::fit_regularized_glm(
            &self.data,
            self.n_samples,
            self.n_features,
            y,
            weights,
            offsets,
            options,
        )
    }

    /// Fit a Laplace-approximated Poisson random-intercept GLMM directly on this [`DesignMatrix`].
    ///
    /// The model intercept is added automatically; a constant column (such as the explicit
    /// intercept column used by [`GlmFit`]) is dropped exactly as in
    /// [`DesignMatrix::fit_regularized`] and reported with fixed effect `0.0`.
    pub fn fit_random_intercept_glmm(
        &self,
        y: &[f64],
        group_ids: &[usize],
        n_groups: usize,
        weights: Option<&[f64]>,
        offsets: Option<&[f64]>,
        options: &crate::error::SolverOptions,
    ) -> Result<super::regularized_glm::RandomInterceptGlmmResult, PricingError> {
        super::regularized_glm::fit_random_intercept_glmm(
            &self.data,
            self.n_samples,
            self.n_features,
            y,
            group_ids,
            n_groups,
            weights,
            offsets,
            options,
        )
    }
}

/// Fluent builder for fitting an unpenalized or L2-penalized GLM via Householder QR IRLS.
///
/// Defaults to unit prior weights (`1.0`), zero offsets (`0.0`), and [`GlmOptions::default()`].
#[derive(Debug, Clone)]
pub struct GlmFit<'a> {
    family: GlmFamily,
    design: &'a DesignMatrix,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    offsets: Option<&'a [f64]>,
    options: GlmOptions,
}

impl<'a> GlmFit<'a> {
    /// Start a GLM fit for `family` on validated `design` and observed responses `y`.
    pub fn new(family: GlmFamily, design: &'a DesignMatrix, y: &'a [f64]) -> Self {
        Self {
            family,
            design,
            y,
            weights: None,
            offsets: None,
            options: GlmOptions::default(),
        }
    }

    /// Set non-negative prior observation weights (default: all `1.0`).
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set linear-predictor exposure offsets `ln(E)` (default: all `0.0`).
    pub const fn offsets(mut self, offsets: &'a [f64]) -> Self {
        self.offsets = Some(offsets);
        self
    }

    /// Override the full [`GlmOptions`] configuration.
    pub const fn options(mut self, options: GlmOptions) -> Self {
        self.options = options;
        self
    }

    /// Set L2 (Ridge) penalty `lambda >= 0`.
    pub const fn ridge_lambda(mut self, lambda: f64) -> Self {
        self.options.ridge_lambda = lambda;
        self
    }

    /// Set whether to penalize column `0` (intercept).
    pub const fn penalize_intercept(mut self, penalize: bool) -> Self {
        self.options.penalize_intercept = penalize;
        self
    }

    /// Set maximum IRLS iterations.
    pub const fn max_iter(mut self, max_iter: usize) -> Self {
        self.options.max_iter = max_iter;
        self
    }

    /// Set absolute coefficient convergence tolerance.
    pub const fn tol(mut self, tol: f64) -> Self {
        self.options.tol = tol;
        self
    }

    /// Execute IRLS fitting and return [`GlmResult`].
    pub fn fit(&self) -> Result<GlmResult, PricingError> {
        let n = self.design.n_samples();
        let default_w;
        let w = if let Some(slice) = self.weights {
            slice
        } else {
            default_w = vec![1.0; n];
            &default_w
        };
        let default_off;
        let off = if let Some(slice) = self.offsets {
            slice
        } else {
            default_off = vec![0.0; n];
            &default_off
        };
        fit_glm_irls(
            self.family,
            self.design.data(),
            n,
            self.design.n_features(),
            self.y,
            w,
            off,
            &self.options,
        )
    }
}

/// Fit an unpenalized or L2 (Ridge) penalized Generalized Linear Model (GLM) via
/// Iteratively Reweighted Least Squares (IRLS) with configurable [`GlmOptions`].
///
/// Designed specifically for actuarial commercial rating tables:
/// - `design_matrix`: Row-major slice of size `n_samples * n_features`.
/// - `y`: Observed responses (`>= 0` for Poisson/QuasiPoisson/Tweedie/NegativeBinomial,
///   `> 0` for Gamma/InverseGaussian, `[0, 1]` for Binomial on active rows).
/// - `weights`: Non-negative prior weights (e.g. `1.0` for counts, claim count for severity).
/// - `offsets`: Exposure offset `ln(E)` for log-link (use `0.0` if none).
/// - `options`: Convergence and optional Ridge regularization controls ([`GlmOptions`]).
///
/// Each IRLS step is deviance-monitored: a step producing a non-finite or increased
/// penalized deviance `D(β) + λ Σ β_j²` is halved toward the previous iterate (up to
/// [`MAX_STEP_HALVINGS`] times), otherwise [`PricingError::NonConvergence`] is returned.
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_irls(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    options: &GlmOptions,
) -> Result<GlmResult, PricingError> {
    fit_glm_irls_impl(
        family,
        design_matrix,
        n_samples,
        n_features,
        y,
        weights,
        offsets,
        options,
        None,
    )
}

/// Penalized deviance `Σ w_i d(y_i, μ_i) + λ Σ_j β_j²` (intercept excluded unless penalized).
#[allow(clippy::too_many_arguments)]
fn penalized_deviance(
    family: GlmFamily,
    y: &[f64],
    weights: &[f64],
    mu: &[f64],
    beta: &[f64],
    ridge_lambda: f64,
    penalize_intercept: bool,
) -> f64 {
    let mut dev = 0.0;
    for i in 0..y.len() {
        if weights[i] > 0.0 {
            dev += weights[i] * family.unit_deviance(y[i], mu[i]);
        }
    }
    if ridge_lambda > 0.0 {
        let start = usize::from(!penalize_intercept);
        dev += ridge_lambda * beta.iter().skip(start).map(|b| b * b).sum::<f64>();
    }
    dev
}

/// IRLS core; `deviance_trace` (when `Some`) records the accepted penalized deviance per iteration.
#[allow(clippy::too_many_arguments)]
fn fit_glm_irls_impl(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    options: &GlmOptions,
    mut deviance_trace: Option<&mut Vec<f64>>,
) -> Result<GlmResult, PricingError> {
    family.validate()?;
    if n_samples == 0 || n_features == 0 {
        return Err(PricingError::EmptyInput(
            "samples and features must be non-zero",
        ));
    }
    if options.max_iter == 0 {
        return Err(PricingError::InvalidParameter("max_iter must be > 0"));
    }
    if !options.tol.is_finite() || !(options.tol > 0.0) {
        return Err(PricingError::InvalidParameter(
            "tolerance must be finite and positive",
        ));
    }
    if !options.ridge_lambda.is_finite() || !(options.ridge_lambda >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "ridge_lambda must be finite and non-negative",
        ));
    }
    if options.ridge_lambda == 0.0 && n_samples < n_features {
        return Err(PricingError::DimensionMismatch(
            "number of samples must be >= features for unpenalized GLM",
        ));
    }
    if n_samples.checked_mul(n_features) != Some(design_matrix.len()) {
        return Err(PricingError::DimensionMismatch(
            "design matrix length != n_samples * n_features",
        ));
    }
    if y.len() != n_samples || weights.len() != n_samples || offsets.len() != n_samples {
        return Err(PricingError::DimensionMismatch(
            "vector dimensions do not match n_samples",
        ));
    }
    if design_matrix.iter().any(|v| !v.is_finite()) {
        return Err(PricingError::InvalidParameter(
            "design matrix contains non-finite value",
        ));
    }

    let mut active_samples = 0usize;
    for i in 0..n_samples {
        let yi = y[i];
        let wi = weights[i];
        let oi = offsets[i];
        if !yi.is_finite() || !wi.is_finite() || !oi.is_finite() {
            return Err(PricingError::InvalidParameter(
                "y, weights, and offsets must be finite",
            ));
        }
        if wi < 0.0 {
            return Err(PricingError::InvalidParameter(
                "weights must be non-negative",
            ));
        }
        if wi > 0.0 {
            active_samples += 1;
            match family {
                GlmFamily::Poisson
                | GlmFamily::QuasiPoisson
                | GlmFamily::Tweedie { .. }
                | GlmFamily::NegativeBinomial { .. } => {
                    if yi < 0.0 {
                        return Err(PricingError::InvalidParameter(
                            "response y must be non-negative for count/Tweedie GLM",
                        ));
                    }
                }
                GlmFamily::Gamma | GlmFamily::InverseGaussian => {
                    if !(yi > 0.0) {
                        return Err(PricingError::NonPositiveValue(
                            "response y must be strictly positive for Gamma/InverseGaussian GLM",
                        ));
                    }
                }
                GlmFamily::Binomial => {
                    if !(0.0..=1.0).contains(&yi) {
                        return Err(PricingError::OutOfBounds(
                            "Binomial response y must lie in [0, 1]",
                        ));
                    }
                }
            }
        }
    }

    if options.ridge_lambda == 0.0 && active_samples < n_features {
        return Err(PricingError::DimensionMismatch(
            "number of positively weighted samples must be >= features",
        ));
    }
    if active_samples == 0 {
        return Err(PricingError::EmptyInput(
            "at least one positively weighted sample is required",
        ));
    }

    // Standard R glm.fit initialization (mustart).
    let mut mu = vec![0.0; n_samples];
    let mut eta = vec![0.0; n_samples];
    for i in 0..n_samples {
        if weights[i] <= 0.0 {
            continue;
        }
        let mu0 = match family {
            GlmFamily::Binomial => f64::midpoint(y[i], 0.5),
            GlmFamily::Gamma | GlmFamily::InverseGaussian => y[i].max(1e-6),
            _ => y[i].max(0.1),
        };
        mu[i] = mu0;
        eta[i] = match family {
            GlmFamily::Binomial => (mu0 / (1.0 - mu0)).ln(),
            _ => mu0.ln(),
        };
    }

    // Pre-transpose design matrix into column-major layout once so every Householder QR
    // step operates on contiguous column slices without squaring the condition number.
    let mut x_cm = vec![0.0; n_samples * n_features];
    for i in 0..n_samples {
        let row = &design_matrix[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            x_cm[j * n_samples + i] = row[j];
        }
    }

    let n_aug = n_samples
        + if options.ridge_lambda > 0.0 {
            n_features
        } else {
            0
        };
    let mut beta = vec![0.0; n_features];
    let mut beta_new = vec![0.0; n_features];
    let mut w_irls = vec![0.0; n_samples];
    let mut z = vec![0.0; n_samples];
    let mut qr_ws = QrWorkspace::with_capacity(n_aug, n_features);

    let mut iter = 0;
    let mut converged = false;
    let mut dev_old = f64::INFINITY;
    let mut last_eta_step = 0.0_f64;

    let update_linear_predictor = |beta: &[f64], eta: &mut [f64], mu: &mut [f64]| {
        for i in 0..n_samples {
            let mut line = offsets[i];
            let row = &design_matrix[i * n_features..(i + 1) * n_features];
            for j in 0..n_features {
                line += row[j] * beta[j];
            }
            eta[i] = line;
            mu[i] = family.inverse_link(line);
        }
    };

    while iter < options.max_iter {
        iter += 1;

        for i in 0..n_samples {
            let wi = weights[i];
            if wi <= 0.0 {
                w_irls[i] = 0.0;
                z[i] = 0.0;
                continue;
            }
            let mui = mu[i];
            w_irls[i] = family.irls_weight(wi, mui);
            z[i] = family.working_response(eta[i], offsets[i], y[i], mui);
        }

        weighted_least_squares_qr_ridge_cm_beta_into(
            &x_cm,
            &z,
            &w_irls,
            n_samples,
            n_features,
            options.ridge_lambda,
            options.penalize_intercept,
            &mut qr_ws,
            &mut beta_new,
        )?;

        if beta_new.iter().any(|b| !b.is_finite()) {
            return Err(PricingError::NonConvergence(
                "GLM IRLS produced non-finite coefficient",
            ));
        }

        // Deviance-monitored step-halving (glm.fit / glm2 convention). On the first
        // iteration only non-finite deviance triggers halving (the start is `mustart`).
        update_linear_predictor(&beta_new, &mut eta, &mut mu);
        let mut dev_new = penalized_deviance(
            family,
            y,
            weights,
            &mu,
            &beta_new,
            options.ridge_lambda,
            options.penalize_intercept,
        );
        let mut halvings = 0usize;
        while !dev_new.is_finite()
            || (iter > 1 && dev_new > dev_old + DEVIANCE_INCREASE_RTOL * (1.0 + dev_old.abs()))
        {
            if halvings == MAX_STEP_HALVINGS {
                return Err(PricingError::NonConvergence(
                    "GLM IRLS step-halving failed to reduce deviance",
                ));
            }
            halvings += 1;
            for j in 0..n_features {
                beta_new[j] = f64::midpoint(beta_new[j], beta[j]);
            }
            update_linear_predictor(&beta_new, &mut eta, &mut mu);
            dev_new = penalized_deviance(
                family,
                y,
                weights,
                &mu,
                &beta_new,
                options.ridge_lambda,
                options.penalize_intercept,
            );
        }
        // R glm.fit: relative deviance change (primary); coefficient change (secondary).
        let dev_rel_change = (dev_new - dev_old).abs() / (dev_new.abs() + DEVIANCE_RTOL_OFFSET);
        dev_old = dev_new;
        if let Some(trace) = deviance_trace.as_deref_mut() {
            trace.push(dev_new);
        }

        let mut max_change = 0.0;
        for j in 0..n_features {
            let change = (beta_new[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if iter > 1 && (dev_rel_change < options.tol || max_change <= options.tol) {
            converged = true;
            // Scale-free drift measure: largest per-iteration change of an active eta_i.
            last_eta_step = (0..n_samples)
                .filter(|&i| weights[i] > 0.0)
                .map(|i| {
                    let row = &design_matrix[i * n_features..(i + 1) * n_features];
                    row.iter()
                        .zip(beta_new.iter().zip(beta.iter()))
                        .map(|(&x, (&bn, &bo))| x * (bn - bo))
                        .sum::<f64>()
                        .abs()
                })
                .fold(0.0_f64, f64::max);
            beta.copy_from_slice(&beta_new);
            break;
        }
        beta.copy_from_slice(&beta_new);
    }

    if !converged {
        return Err(PricingError::NonConvergence(
            "GLM IRLS did not converge within max iterations",
        ));
    }

    // Recompute final IRLS weights, deviance, and Pearson chi-squared at converged beta.
    let mut deviance = 0.0;
    let mut pearson_chi2 = 0.0;

    for i in 0..n_samples {
        let wi = weights[i];
        if wi <= 0.0 {
            w_irls[i] = 0.0;
            continue;
        }
        let yi = y[i];
        let mui = mu[i];
        w_irls[i] = family.irls_weight(wi, mui);
        deviance += wi * family.unit_deviance(yi, mui);
        let diff = yi - mui;
        pearson_chi2 += wi * (diff * diff) / family.variance_fn(mui);
    }

    let mu_on_boundary = (0..n_samples).any(|i| {
        weights[i] > 0.0
            && match family {
                GlmFamily::Binomial => {
                    mu[i] <= BINOMIAL_BOUNDARY_EPS || mu[i] >= 1.0 - BINOMIAL_BOUNDARY_EPS
                }
                _ => mu[i] <= LOG_LINK_BOUNDARY_MU,
            }
    });
    let separation_suspected = mu_on_boundary || last_eta_step > SEPARATION_STEP_THRESHOLD;

    let df_residual = active_samples.saturating_sub(n_features);
    let dispersion = if family.has_free_dispersion() && df_residual > 0 {
        pearson_chi2 / df_residual as f64
    } else {
        1.0
    };

    let inv_xtwx = weighted_least_squares_qr_ridge_cm_cov(
        &x_cm,
        &w_irls,
        n_samples,
        n_features,
        options.ridge_lambda,
        options.penalize_intercept,
        &mut qr_ws,
    )?;
    let mut standard_errors = vec![0.0; n_features];
    for j in 0..n_features {
        let var = dispersion * inv_xtwx[j * n_features + j];
        standard_errors[j] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }

    Ok(GlmResult {
        family,
        coefficients: beta,
        standard_errors,
        dispersion,
        deviance,
        df_residual,
        iterations: iter,
        separation_suspected,
    })
}

/// Predict expected responses `mu_i = g^-1(X_i beta + offset_i)` from a fitted GLM.
pub fn predict_glm(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    coefficients: &[f64],
    offsets: &[f64],
) -> Result<Vec<f64>, PricingError> {
    family.validate()?;
    if n_samples == 0 || n_features == 0 {
        return Err(PricingError::EmptyInput(
            "samples and features must be non-zero",
        ));
    }
    if n_samples.checked_mul(n_features) != Some(design_matrix.len())
        || coefficients.len() != n_features
        || offsets.len() != n_samples
    {
        return Err(PricingError::DimensionMismatch(
            "design_matrix, coefficients, or offsets length mismatch",
        ));
    }
    if design_matrix.iter().any(|v| !v.is_finite())
        || coefficients.iter().any(|v| !v.is_finite())
        || offsets.iter().any(|v| !v.is_finite())
    {
        return Err(PricingError::InvalidParameter(
            "predict_glm inputs must be finite",
        ));
    }

    let mut preds = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut eta = offsets[i];
        let row = &design_matrix[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            eta += row[j] * coefficients[j];
        }
        preds.push(family.inverse_link(eta));
    }
    Ok(preds)
}

/// Result of Tweedie power grid search by exact profile log-likelihood
/// (Dunn & Smyth 2005 series density with profile-ML dispersion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TweediePowerSelection {
    /// Selected variance power `p in (1, 2)` maximizing the profile log-likelihood.
    pub best_power: f64,
    /// Profile log-likelihood `max_φ ℓ(μ̂_p, φ, p)` at `best_power`.
    pub best_log_likelihood: f64,
    /// Profile maximum-likelihood dispersion `φ̂` at `best_power`.
    pub best_dispersion: f64,
    /// Fitted GLM result at `best_power`.
    pub best_fit: GlmResult,
    /// Evaluated `(power, profile_log_likelihood)` pairs for every candidate that fitted.
    pub profile: Vec<(f64, f64)>,
    /// Candidates that were skipped, with the error message (invalid power, IRLS or
    /// dispersion-profile failure).
    #[serde(default)]
    pub failed_powers: Vec<(f64, String)>,
}

/// Select the Tweedie variance power `p in (1, 2)` over `candidate_powers` by maximizing the
/// exact profile log-likelihood.
///
/// For each candidate `p`: fit the IRLS Tweedie GLM for `μ̂_p`, then maximize the exact
/// Dunn & Smyth (2005) series log-likelihood `Σ ln f(y_i; μ̂_i, φ / w_i, p)` over `φ`
/// (golden-section on `ln φ`, started at the Pearson estimate). This follows the
/// `tweedie.profile` procedure of Dunn & Smyth; the saddlepoint approximation is not used
/// because it is biased for small `y` and can favour boundary powers.
///
/// A candidate whose fit or profile fails (including powers outside `(1, 2)`) is skipped and
/// recorded in [`TweediePowerSelection::failed_powers`]; the function errors only when no
/// candidate succeeds.
#[allow(clippy::too_many_arguments)]
pub fn select_tweedie_power(
    candidate_powers: &[f64],
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    options: &GlmOptions,
) -> Result<TweediePowerSelection, PricingError> {
    if candidate_powers.is_empty() {
        return Err(PricingError::EmptyInput(
            "candidate_powers must not be empty",
        ));
    }

    let mut profile = Vec::with_capacity(candidate_powers.len());
    let mut failed_powers = Vec::new();
    let mut best: Option<(f64, f64, f64, GlmResult)> = None;

    for &p in candidate_powers {
        let family = GlmFamily::Tweedie { power: p };
        let candidate = (|| -> Result<(f64, f64, GlmResult), PricingError> {
            let fit = fit_glm_irls(
                family,
                design_matrix,
                n_samples,
                n_features,
                y,
                weights,
                offsets,
                options,
            )?;
            let mu = predict_glm(
                family,
                design_matrix,
                n_samples,
                n_features,
                &fit.coefficients,
                offsets,
            )?;
            let phi_start = if fit.dispersion.is_finite() && fit.dispersion > 0.0 {
                fit.dispersion
            } else {
                1.0
            };
            let mut series = TweedieSeries::new(p)?;
            let (phi_hat, ll) = maximize_over_dispersion(phi_start, |phi| {
                tweedie_log_likelihood(&mut series, y, &mu, weights, phi)
            })?;
            Ok((phi_hat, ll, fit))
        })();
        match candidate {
            Ok((phi_hat, ll, fit)) => {
                profile.push((p, ll));
                if best.as_ref().is_none_or(|b| ll > b.1) {
                    best = Some((p, ll, phi_hat, fit));
                }
            }
            Err(e) => failed_powers.push((p, e.to_string())),
        }
    }

    let Some((best_power, best_log_likelihood, best_dispersion, best_fit)) = best else {
        return Err(PricingError::NonConvergence(
            "no valid Tweedie power converged",
        ));
    };

    Ok(TweediePowerSelection {
        best_power,
        best_log_likelihood,
        best_dispersion,
        best_fit,
        profile,
        failed_powers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gamma log-link severity fit with a poor (`mustart = y`) start and widely spread
    /// responses: the accepted deviance sequence must be non-increasing.
    #[test]
    fn irls_step_halving_keeps_deviance_monotone() {
        let n = 12usize;
        let mut x = Vec::with_capacity(2 * n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f64 / 2.0;
            x.push(1.0);
            x.push(xi);
            // Heavily dispersed severities spanning ~8 orders of magnitude.
            y.push(if i % 3 == 0 { 1e-3 } else { 10f64.powf(xi) });
        }
        let w = vec![1.0; n];
        let off = vec![0.0; n];
        for family in [
            GlmFamily::Gamma,
            GlmFamily::Poisson,
            GlmFamily::Tweedie { power: 1.5 },
        ] {
            let mut trace = Vec::new();
            let res = fit_glm_irls_impl(
                family,
                &x,
                n,
                2,
                &y,
                &w,
                &off,
                &GlmOptions::default().with_max_iter(500),
                Some(&mut trace),
            );
            assert!(trace.len() >= 2, "{family:?}: {res:?}");
            for pair in trace[1..].windows(2) {
                assert!(
                    pair[1] <= pair[0] * (1.0 + 1e-9) + 1e-12,
                    "{family:?} deviance increased: {trace:?}"
                );
            }
            assert!(res.is_ok(), "{family:?}: {res:?}");
        }
    }
}
