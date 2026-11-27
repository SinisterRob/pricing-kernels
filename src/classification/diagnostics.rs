//! Actuarial classification model diagnostics: exposure-weighted Gini index and Lorenz curve
//! (Frees, Meyers & Cummings 2011), double-lift charts, AIC/BIC, and Variance Inflation Factors.
//!
//! References:
//! - E. W. Frees, G. Meyers & A. D. Cummings (2011), *"Summarizing Insurance Scores Using a
//!   Gini Index"*, *Journal of the American Statistical Association*, Vol. 106(495), pp. 1085–1098.
//! - M. Goldburd, A. Khare & D. Tevet, *Generalized Linear Models for Insurance Rating*,
//!   CAS Monograph No. 5, Ch. 6–7.

use super::irls_glm::GlmFamily;
use crate::error::PricingError;
use crate::linalg::cholesky_invert;
use serde::{Deserialize, Serialize};

/// Point on an exposure-weighted Lorenz curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LorenzCurvePoint {
    /// Cumulative share of earned exposure in `[0, 1]`.
    pub cumulative_exposure: f64,
    /// Cumulative share of observed loss in `[0, 1]`.
    pub cumulative_loss: f64,
}

/// Exposure-weighted Gini index and Lorenz curve output (Frees, Meyers & Cummings 2011).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GiniResult {
    /// Normalized Gini index `G = 1 - 2 * Area(Lorenz)` in `[-1, 1]`.
    /// Higher values indicate stronger risk-ordering lift (`0.0` = flat/random pricing).
    pub gini_index: f64,
    /// Lorenz curve points starting at `(0, 0)` and ending at `(1, 1)`.
    pub lorenz_curve: Vec<LorenzCurvePoint>,
}

/// Compute the exposure-weighted Lorenz curve and normalized Gini index for a rating plan.
///
/// Policies are ordered ascending by `predicted_pure_premium`, with tied predictions grouped
/// symmetrically so the Gini index is strictly permutation-invariant.
pub fn weighted_gini_index(
    exposures: &[f64],
    actual_losses: &[f64],
    predicted_pure_premium: &[f64],
) -> Result<GiniResult, PricingError> {
    let n = exposures.len();
    if n == 0 {
        return Err(PricingError::EmptyInput(
            "exposures must not be empty for Gini calculation",
        ));
    }
    if actual_losses.len() != n || predicted_pure_premium.len() != n {
        return Err(PricingError::DimensionMismatch(
            "exposures, actual_losses, and predicted_pure_premium must have equal length",
        ));
    }

    let mut total_exp = 0.0;
    let mut total_loss = 0.0;
    for i in 0..n {
        if !exposures[i].is_finite()
            || !actual_losses[i].is_finite()
            || !predicted_pure_premium[i].is_finite()
        {
            return Err(PricingError::InvalidParameter("Gini inputs must be finite"));
        }
        if !(exposures[i] > 0.0) || !(actual_losses[i] >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "exposures must be > 0 and actual_losses must be >= 0",
            ));
        }
        total_exp += exposures[i];
        total_loss += actual_losses[i];
    }
    if !(total_loss > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "total actual_losses must be positive for Lorenz/Gini calculation",
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| predicted_pure_premium[a].total_cmp(&predicted_pure_premium[b]));

    let mut curve = Vec::with_capacity(n + 1);
    curve.push(LorenzCurvePoint {
        cumulative_exposure: 0.0,
        cumulative_loss: 0.0,
    });

    let mut cum_exp = 0.0;
    let mut cum_loss = 0.0;
    let mut area = 0.0;
    let mut prev_x = 0.0;
    let mut prev_y = 0.0;

    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n
            && (predicted_pure_premium[order[j]] - predicted_pure_premium[order[i]]).abs() <= 1e-15
        {
            j += 1;
        }
        for idx in i..j {
            cum_exp += exposures[order[idx]];
            cum_loss += actual_losses[order[idx]];
        }
        let x = cum_exp / total_exp;
        let y = cum_loss / total_loss;
        area += 0.5 * (x - prev_x) * (y + prev_y);
        curve.push(LorenzCurvePoint {
            cumulative_exposure: x,
            cumulative_loss: y,
        });
        prev_x = x;
        prev_y = y;
        i = j;
    }

    Ok(GiniResult {
        gini_index: 1.0 - 2.0 * area,
        lorenz_curve: curve,
    })
}

/// Single quantile bucket of a double-lift chart comparing candidate vs. incumbent rating plans.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DoubleLiftBucket {
    /// Zero-based quantile bucket index (`0..n_buckets`).
    pub bucket: usize,
    /// Total earned exposure in this bucket.
    pub exposure: f64,
    /// Observed actual pure premium (`sum actual_loss / sum exposure`).
    pub actual_pure_premium: f64,
    /// Incumbent model (Model A) predicted pure premium.
    pub model_a_pure_premium: f64,
    /// Candidate model (Model B) predicted pure premium.
    pub model_b_pure_premium: f64,
    /// Exposure-weighted average sort ratio (`Model B / Model A`).
    pub mean_ratio_b_to_a: f64,
}

/// Construct an exposure-weighted double-lift chart comparing incumbent (`pred_a`) against
/// candidate (`pred_b`) pure-premium predictions across `n_buckets` quantiles sorted by `pred_b / pred_a`
/// (CAS Monograph No. 5, Ch. 7).
pub fn double_lift_chart(
    exposures: &[f64],
    actual_losses: &[f64],
    pred_a: &[f64],
    pred_b: &[f64],
    n_buckets: usize,
) -> Result<Vec<DoubleLiftBucket>, PricingError> {
    let n = exposures.len();
    if n == 0 || n_buckets == 0 {
        return Err(PricingError::EmptyInput(
            "exposures and n_buckets must be non-zero",
        ));
    }
    if actual_losses.len() != n || pred_a.len() != n || pred_b.len() != n {
        return Err(PricingError::DimensionMismatch(
            "exposures, actual_losses, pred_a, and pred_b must have equal length",
        ));
    }

    let mut total_exp = 0.0;
    let mut ratios = Vec::with_capacity(n);
    for i in 0..n {
        if !exposures[i].is_finite()
            || !actual_losses[i].is_finite()
            || !pred_a[i].is_finite()
            || !pred_b[i].is_finite()
            || !(exposures[i] > 0.0)
            || !(pred_a[i] > 0.0)
            || !(pred_b[i] > 0.0)
            || !(actual_losses[i] >= 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "exposures and predictions must be finite and positive, actual_losses >= 0",
            ));
        }
        total_exp += exposures[i];
        ratios.push(pred_b[i] / pred_a[i]);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ratios[a].total_cmp(&ratios[b]));

    let mut buckets = Vec::with_capacity(n_buckets);
    let bucket_target = total_exp / (n_buckets as f64);
    let mut idx = 0usize;

    for b in 0..n_buckets {
        let mut b_exp = 0.0;
        let mut b_actual = 0.0;
        let mut b_a = 0.0;
        let mut b_b = 0.0;
        let mut b_ratio_sum = 0.0;

        let target_cum = if b + 1 == n_buckets {
            f64::INFINITY
        } else {
            bucket_target
        };

        while idx < n {
            let k = order[idx];
            if b_exp > 0.0 && b_exp + 0.5 * exposures[k] > target_cum {
                break;
            }
            b_exp += exposures[k];
            b_actual += actual_losses[k];
            b_a += exposures[k] * pred_a[k];
            b_b += exposures[k] * pred_b[k];
            b_ratio_sum += exposures[k] * ratios[k];
            idx += 1;
        }

        if b_exp > 0.0 {
            buckets.push(DoubleLiftBucket {
                bucket: b,
                exposure: b_exp,
                actual_pure_premium: b_actual / b_exp,
                model_a_pure_premium: b_a / b_exp,
                model_b_pure_premium: b_b / b_exp,
                mean_ratio_b_to_a: b_ratio_sum / b_exp,
            });
        }
    }

    Ok(buckets)
}

/// Goodness-of-fit information criteria for a fitted GLM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InformationCriteria {
    /// Scaled deviance `D / phi`.
    pub scaled_deviance: f64,
    /// Pearson chi-squared statistic `sum w_i (y_i - mu_i)^2 / V(mu_i)`.
    pub pearson_chi2: f64,
    /// Akaike Information Criterion (`scaled_deviance + 2 * k`, where `k` includes dispersion if estimated).
    pub aic: f64,
    /// Bayesian Information Criterion (`scaled_deviance + ln(n_active) * k`).
    pub bic: f64,
}

/// Compute deviance-based `AIC`, `BIC`, and Pearson `chi^2` for a fitted GLM.
pub fn glm_information_criteria(
    family: GlmFamily,
    y: &[f64],
    mu: &[f64],
    weights: &[f64],
    dispersion: f64,
    n_features: usize,
) -> Result<InformationCriteria, PricingError> {
    family.validate()?;
    let n = y.len();
    if n == 0 || mu.len() != n || weights.len() != n {
        return Err(PricingError::DimensionMismatch(
            "y, mu, and weights must have matching non-empty length",
        ));
    }
    if !dispersion.is_finite() || !(dispersion > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "dispersion must be finite and positive",
        ));
    }

    let mut deviance = 0.0;
    let mut pearson_chi2 = 0.0;
    let mut n_active = 0usize;

    for i in 0..n {
        let yi = y[i];
        let mui = mu[i];
        let wi = weights[i];
        if !yi.is_finite() || !mui.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(PricingError::InvalidParameter(
                "y, mu, and weights must be finite with weights >= 0",
            ));
        }
        if wi <= 0.0 {
            continue;
        }
        match family {
            GlmFamily::Binomial => {
                if !((0.0..=1.0).contains(&yi) && mui > 0.0 && mui < 1.0) {
                    return Err(PricingError::InvalidParameter(
                        "Binomial information criteria require y in [0, 1] and mu in (0, 1)",
                    ));
                }
            }
            GlmFamily::Gamma | GlmFamily::InverseGaussian => {
                if !(yi > 0.0) || !(mui > 0.0) {
                    return Err(PricingError::NonPositiveValue(
                        "Gamma/InverseGaussian information criteria require y > 0 and mu > 0",
                    ));
                }
            }
            _ => {
                if !(yi >= 0.0) || !(mui > 0.0) {
                    return Err(PricingError::InvalidParameter(
                        "count/Tweedie information criteria require y >= 0 and mu > 0",
                    ));
                }
            }
        }
        n_active += 1;
        deviance += wi * family.unit_deviance(yi, mui);
        let diff = yi - mui;
        pearson_chi2 += wi * (diff * diff) / family.variance_fn(mui);
    }
    if n_active == 0 {
        return Err(PricingError::EmptyInput(
            "at least one positively weighted observation is required",
        ));
    }

    let scaled_deviance = deviance / dispersion;
    let extra_disp_param = if family.has_free_dispersion() {
        1.0
    } else {
        0.0
    };
    let k = (n_features as f64) + extra_disp_param;
    let aic = scaled_deviance + 2.0 * k;
    let bic = scaled_deviance + (n_active as f64).ln() * k;

    Ok(InformationCriteria {
        scaled_deviance,
        pearson_chi2,
        aic,
        bic,
    })
}

/// Compute Variance Inflation Factors (VIF) for non-constant predictor columns of a
/// row-major design matrix (`n_samples × n_features`, excluding any intercept column).
///
/// Pass only the predictor columns (without the constant intercept column); returns
/// `VIF_j = (R^-1)_{jj} >= 1.0` for each column `j = 0..n_features`.
pub fn variance_inflation_factors(
    predictors: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, PricingError> {
    if n_samples < 2 || n_features == 0 {
        return Err(PricingError::EmptyInput(
            "at least 2 samples and 1 predictor column are required for VIF",
        ));
    }
    if n_samples.checked_mul(n_features) != Some(predictors.len()) {
        return Err(PricingError::DimensionMismatch(
            "predictors length != n_samples * n_features",
        ));
    }
    if predictors.iter().any(|v| !v.is_finite()) {
        return Err(PricingError::InvalidParameter(
            "predictors contain non-finite value",
        ));
    }

    let nf = n_samples as f64;
    let mut means = vec![0.0; n_features];
    for i in 0..n_samples {
        let row = &predictors[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            means[j] += row[j];
        }
    }
    for m in &mut means {
        *m /= nf;
    }

    let mut stds = vec![0.0; n_features];
    for i in 0..n_samples {
        let row = &predictors[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            let d = row[j] - means[j];
            stds[j] += d * d;
        }
    }
    for (s, &mean) in stds.iter_mut().zip(means.iter()) {
        *s = (*s).sqrt();
        if *s <= 1e-14 * mean.abs().max(1.0) {
            return Err(PricingError::SingularMatrix(
                "constant predictor column encountered in VIF (exclude intercept column)",
            ));
        }
    }
    if n_features == 1 {
        return Ok(vec![1.0]);
    }

    let mut corr = vec![0.0; n_features * n_features];
    for i in 0..n_samples {
        let row = &predictors[i * n_features..(i + 1) * n_features];
        for j in 0..n_features {
            let zj = (row[j] - means[j]) / stds[j];
            for k in 0..=j {
                let zk = (row[k] - means[k]) / stds[k];
                corr[j * n_features + k] += zj * zk;
            }
        }
    }
    for j in 0..n_features {
        corr[j * n_features + j] = 1.0;
        for k in 0..j {
            corr[k * n_features + j] = corr[j * n_features + k];
        }
    }

    let inv_corr = cholesky_invert(&corr, n_features)?;
    Ok((0..n_features)
        .map(|j| inv_corr[j * n_features + j].max(1.0))
        .collect())
}
