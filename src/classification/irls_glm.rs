//! Iteratively Reweighted Least Squares (IRLS) GLM solver with Householder QR WLS,
//! optional L2 (Ridge) penalization, and Dunn & Smyth (2005) Tweedie power selection.

use crate::error::PricingError;
use crate::linalg::{
    QrWorkspace, weighted_least_squares_qr_ridge_cm_beta_into,
    weighted_least_squares_qr_ridge_cm_cov,
};
use serde::{Deserialize, Serialize};

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
    /// Absolute convergence threshold on `max_j |beta_new[j] - beta[j]|` (default `1e-9`).
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

    /// Set absolute convergence tolerance on coefficients.
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

/// Fitted GLM result with coefficients, standard errors, dispersion, and deviance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmResult {
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
        fit_glm_irls_with_options(
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

/// Fit a Generalized Linear Model (GLM) via Iteratively Reweighted Least Squares (IRLS).
///
/// Designed specifically for actuarial commercial rating tables:
/// - `design_matrix`: Row-major slice of size `n_samples * n_features`.
/// - `y`: Observed responses (`>= 0` for Poisson/QuasiPoisson/Tweedie/NegativeBinomial,
///   `> 0` for Gamma/InverseGaussian, `[0, 1]` for Binomial on active rows).
/// - `weights`: Non-negative prior weights (e.g. `1.0` for counts, claim count for severity).
/// - `offsets`: Exposure offset `ln(E)` for log-link (use `0.0` if none).
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_irls(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<GlmResult, PricingError> {
    fit_glm_irls_with_options(
        family,
        design_matrix,
        n_samples,
        n_features,
        y,
        weights,
        offsets,
        &GlmOptions {
            max_iter,
            tol,
            ridge_lambda: 0.0,
            penalize_intercept: false,
        },
    )
}

/// Fit an unpenalized or L2 (Ridge) penalized GLM via IRLS with configurable [`GlmOptions`].
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_irls_with_options(
    family: GlmFamily,
    design_matrix: &[f64],
    n_samples: usize,
    n_features: usize,
    y: &[f64],
    weights: &[f64],
    offsets: &[f64],
    options: &GlmOptions,
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
    if design_matrix.len() != n_samples * n_features {
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

        let mut max_change = 0.0;
        for j in 0..n_features {
            if !beta_new[j].is_finite() {
                return Err(PricingError::NonConvergence(
                    "GLM IRLS produced non-finite coefficient",
                ));
            }
            let change = (beta_new[j] - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
            beta[j] = beta_new[j];
        }

        for i in 0..n_samples {
            let mut line = offsets[i];
            let row = &design_matrix[i * n_features..(i + 1) * n_features];
            for j in 0..n_features {
                line += row[j] * beta[j];
            }
            eta[i] = line;
            mu[i] = family.inverse_link(line);
        }

        if iter > 1 && max_change <= options.tol {
            converged = true;
            break;
        }
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
        coefficients: beta,
        standard_errors,
        dispersion,
        deviance,
        df_residual,
        iterations: iter,
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
    if design_matrix.len() != n_samples * n_features
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

/// Result of Tweedie power parameter grid search via saddlepoint profile log-likelihood
/// (Dunn & Smyth 2005 / CAS Monograph No. 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TweediePowerSelection {
    /// Selected variance power `p in (1, 2)` maximizing saddlepoint profile log-likelihood.
    pub best_power: f64,
    /// Saddlepoint profile log-likelihood at `best_power`.
    pub best_log_likelihood: f64,
    /// Fitted GLM result at `best_power`.
    pub best_fit: GlmResult,
    /// Evaluated `(power, saddlepoint_log_likelihood)` pairs across all candidate powers.
    pub profile: Vec<(f64, f64)>,
}

/// Select the Tweedie variance power `p in (1, 2)` over `candidate_powers` by maximizing
/// the Dunn & Smyth (2005) saddlepoint profile log-likelihood.
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
    let mut best_power = f64::NAN;
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_fit: Option<GlmResult> = None;

    for &p in candidate_powers {
        let family = GlmFamily::Tweedie { power: p };
        let fit = fit_glm_irls_with_options(
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
        let phi = fit.dispersion.max(1e-12);

        // Dunn & Smyth (2005) saddlepoint log-likelihood approximation
        let mut ll = 0.0;
        for i in 0..n_samples {
            let wi = weights[i];
            if wi <= 0.0 {
                continue;
            }
            let yi = y[i];
            let d_i = family.unit_deviance(yi, mu[i]);
            if yi > 0.0 {
                let log_norm = (std::f64::consts::TAU * (phi / wi)).ln() + p * yi.ln();
                ll -= f64::midpoint(log_norm, wi * d_i / phi);
            } else {
                ll -= 0.5 * wi * d_i / phi;
            }
        }

        profile.push((p, ll));
        if ll > best_ll {
            best_ll = ll;
            best_power = p;
            best_fit = Some(fit);
        }
    }

    let Some(fit) = best_fit else {
        return Err(PricingError::NonConvergence(
            "no valid Tweedie power converged",
        ));
    };

    Ok(TweediePowerSelection {
        best_power,
        best_log_likelihood: best_ll,
        best_fit: fit,
        profile,
    })
}
