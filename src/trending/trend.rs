use serde::{Deserialize, Serialize};
use crate::error::PricingError;

/// Exponential or linear loss & exposure trending kernels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrendModel {
    /// Annual trend rate (e.g., +0.045 for +4.5% annual inflation).
    pub annual_rate: f64,
}

impl TrendModel {
    pub fn new(annual_rate: f64) -> Self {
        Self { annual_rate }
    }

    /// Multiplicative exponential trend factor for period delta_years:
    /// Factor = (1 + r)^t
    #[inline]
    pub fn compound_factor(&self, delta_years: f64) -> f64 {
        (1.0 + self.annual_rate).powf(delta_years)
    }

    /// Continuous exponential trend factor:
    /// Factor = exp(r * t)
    #[inline]
    pub fn continuous_factor(&self, delta_years: f64) -> f64 {
        (self.annual_rate * delta_years).exp()
    }

    /// Trend a historical loss cost to future projected level:
    /// Trended = Historic * (1 + r)^t
    #[inline]
    pub fn apply_compound(&self, historic_val: f64, delta_years: f64) -> f64 {
        historic_val * self.compound_factor(delta_years)
    }

    /// Fit an annual exponential trend rate from a series of historical points (t_i, y_i)
    /// via log-linear least squares: ln(y) = a + b * t -> annual_rate = exp(b) - 1.
    pub fn fit_exponential(time_points: &[f64], values: &[f64]) -> Result<Self, PricingError> {
        let n = time_points.len();
        if n < 2 || n != values.len() {
            return Err(PricingError::InvalidParameter("at least 2 matching points required"));
        }
        let mut sum_t = 0.0;
        let mut sum_lny = 0.0;
        let mut sum_tt = 0.0;
        let mut sum_t_lny = 0.0;

        for (&t, &y) in time_points.iter().zip(values.iter()) {
            if y <= 0.0 {
                return Err(PricingError::NonPositiveValue("values must be positive for log trend"));
            }
            let lny = y.ln();
            sum_t += t;
            sum_lny += lny;
            sum_tt += t * t;
            sum_t_lny += t * lny;
        }

        let n_f = n as f64;
        let denom = n_f * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-12 {
            return Err(PricingError::SingularMatrix("time points are collinear or identical"));
        }

        let slope = (n_f * sum_t_lny - sum_t * sum_lny) / denom;
        let annual_rate = slope.exp_m1();

        Ok(Self { annual_rate })
    }
}
