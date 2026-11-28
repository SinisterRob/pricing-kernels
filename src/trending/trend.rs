//! Single-step and two-step exponential loss-cost and exposure trending kernels
//! (Werner & Modlin *Basic Ratemaking*, Ch. 6).

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Exponential loss & exposure trending kernel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrendModel {
    /// Annual trend rate (e.g., `+0.045` for `+4.5%` annual inflation).
    pub annual_rate: f64,
}

impl TrendModel {
    /// Construct a trend model with specified annual decimal rate (e.g., `0.045` for `+4.5%`).
    pub const fn new(annual_rate: f64) -> Self {
        Self { annual_rate }
    }

    /// Multiplicative exponential trend factor for period `delta_years`:
    /// `Factor = (1 + r)^t`
    #[inline]
    pub fn compound_factor(&self, delta_years: f64) -> f64 {
        (1.0 + self.annual_rate).powf(delta_years)
    }

    /// Continuous exponential trend factor:
    /// `Factor = exp(r * t)`
    #[inline]
    pub fn continuous_factor(&self, delta_years: f64) -> f64 {
        (self.annual_rate * delta_years).exp()
    }

    /// Trend a historical loss cost to future projected level:
    /// `Trended = Historic * (1 + r)^t`
    #[inline]
    pub fn apply_compound(&self, historic_val: f64, delta_years: f64) -> f64 {
        historic_val * self.compound_factor(delta_years)
    }

    /// Fit an annual exponential trend rate from a series of historical points `(t_i, y_i)`
    /// via log-linear least squares: `ln(y) = a + b * t -> annual_rate = exp(b) - 1`.
    pub fn fit_exponential(time_points: &[f64], values: &[f64]) -> Result<Self, PricingError> {
        let n = time_points.len();
        if n < 2 || n != values.len() {
            return Err(PricingError::InvalidParameter(
                "at least 2 matching points required",
            ));
        }
        let mut sum_t = 0.0;
        let mut sum_lny = 0.0;
        for (&t, &y) in time_points.iter().zip(values.iter()) {
            if !t.is_finite() || !y.is_finite() {
                return Err(PricingError::InvalidParameter(
                    "time points and values must be finite",
                ));
            }
            if !(y > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "values must be strictly positive for log trend",
                ));
            }
            sum_t += t;
            sum_lny += y.ln();
        }

        let n_f = n as f64;
        let mean_t = sum_t / n_f;
        let mean_lny = sum_lny / n_f;

        // Center coordinates around mean_t to avoid catastrophic cancellation on calendar years.
        let mut s_tt = 0.0;
        let mut s_ty = 0.0;
        for (&t, &y) in time_points.iter().zip(values.iter()) {
            let dt = t - mean_t;
            s_tt += dt * dt;
            s_ty += dt * (y.ln() - mean_lny);
        }

        if !s_tt.is_finite() || s_tt <= 1e-12 * mean_t.abs().max(1.0) {
            return Err(PricingError::SingularMatrix(
                "time points are collinear or identical",
            ));
        }

        let slope = s_ty / s_tt;
        let annual_rate = slope.exp_m1();
        if !annual_rate.is_finite() {
            return Err(PricingError::OutOfBounds(
                "fitted annual trend rate overflowed floating-point range",
            ));
        }

        Ok(Self { annual_rate })
    }
}

/// Two-step exponential trending model (Werner & Modlin *Basic Ratemaking*, Ch. 6).
///
/// Separates historical experience-to-latest trend (`historical_rate`, applied over
/// `historical_years`) from prospective latest-to-future projection trend
/// (`prospective_rate`, applied over `prospective_years`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TwoStepTrendModel {
    /// Annual trend rate from historical experience period to latest available data point.
    pub historical_rate: f64,
    /// Annual trend rate from latest available data point to prospective policy period.
    pub prospective_rate: f64,
}

impl TwoStepTrendModel {
    /// Construct a two-step trend model with validated rates (`> -1.0`).
    pub fn try_new(historical_rate: f64, prospective_rate: f64) -> Result<Self, PricingError> {
        if !historical_rate.is_finite()
            || !prospective_rate.is_finite()
            || !(historical_rate > -1.0)
            || !(prospective_rate > -1.0)
        {
            return Err(PricingError::InvalidParameter(
                "trend rates must be finite and strictly greater than -1.0",
            ));
        }
        Ok(Self {
            historical_rate,
            prospective_rate,
        })
    }

    /// Combined multiplicative two-step trend factor:
    /// `(1 + historical_rate)^historical_years * (1 + prospective_rate)^prospective_years`.
    pub fn compound_factor(
        &self,
        historical_years: f64,
        prospective_years: f64,
    ) -> Result<f64, PricingError> {
        if !historical_years.is_finite() || !prospective_years.is_finite() {
            return Err(PricingError::InvalidParameter(
                "trend periods must be finite",
            ));
        }
        Ok((1.0 + self.historical_rate).powf(historical_years)
            * (1.0 + self.prospective_rate).powf(prospective_years))
    }

    /// Apply the two-step trend factor to a historical loss cost or average premium.
    pub fn apply_compound(
        &self,
        historic_val: f64,
        historical_years: f64,
        prospective_years: f64,
    ) -> Result<f64, PricingError> {
        if !historic_val.is_finite() {
            return Err(PricingError::InvalidParameter(
                "historic_val must be finite",
            ));
        }
        Ok(historic_val * self.compound_factor(historical_years, prospective_years)?)
    }
}
