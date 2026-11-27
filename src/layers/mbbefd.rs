//! Bernegger (1997) MBBEFD and Swiss Re exposure rating curves for property and
//! commercial excess-of-loss reinsurance pricing.
//!
//! Reference: S. Bernegger (1997), *"The Swiss Re Exposure Curves and the MBBEFD
//! Distribution Class"*, *ASTIN Bulletin*, Vol. 27(1), pp. 99–111.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Canonical Swiss Re property exposure curve presets (Bernegger 1997).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwissReCurve {
    /// Swiss Re `Y1` (`c = 1.5`): personal lines / light commercial property.
    Y1,
    /// Swiss Re `Y2` (`c = 2.0`): standard commercial property.
    Y2,
    /// Swiss Re `Y3` (`c = 3.0`): industrial property.
    Y3,
    /// Swiss Re `Y4` (`c = 4.0`): heavy industrial property.
    Y4,
    /// Lloyd's / very heavy industrial curve (`c = 5.0`).
    Lloyds,
}

impl SwissReCurve {
    /// Returns the Bernegger (1997) continuous parameter `c` for the preset.
    #[inline]
    pub const fn c_parameter(self) -> f64 {
        match self {
            Self::Y1 => 1.5,
            Self::Y2 => 2.0,
            Self::Y3 => 3.0,
            Self::Y4 => 4.0,
            Self::Lloyds => 5.0,
        }
    }
}

/// MBBEFD (Maxwell–Boltzmann, Bose–Einstein, Fermi–Dirac) exposure curve
/// parameterized by `(b, g)` with `b >= 0` and `g >= 1` (Bernegger 1997).
///
/// The exposure curve `G(d)` maps the normalized retention/limit `d in [0, 1]`
/// (as a fraction of Maximum Possible Loss or Total Insured Value) to the
/// fraction of expected ground-up loss capped at `d`:
/// `G(d) = E[min(X, d)] / E[X]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MbbefdCurve {
    /// Shape parameter `b >= 0` (`b = 0` or `g = 1` corresponds to total-loss-only linear curve).
    pub b: f64,
    /// Reciprocal of total-loss probability `g = 1 / p >= 1`.
    pub g: f64,
}

impl MbbefdCurve {
    /// Construct an MBBEFD exposure curve directly from `(b, g)` parameters.
    ///
    /// Requires finite `b >= 0.0` and `g >= 1.0`.
    pub fn new(b: f64, g: f64) -> Result<Self, PricingError> {
        if !b.is_finite() || !g.is_finite() {
            return Err(PricingError::InvalidParameter(
                "MBBEFD parameters b and g must be finite",
            ));
        }
        if !(b >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "MBBEFD parameter b must be >= 0",
            ));
        }
        if !(g >= 1.0) {
            return Err(PricingError::InvalidParameter(
                "MBBEFD parameter g must be >= 1",
            ));
        }
        Ok(Self { b, g })
    }

    #[inline]
    fn swiss_re_bg(c: f64) -> (f64, f64) {
        let b = (3.1 - 0.15 * (1.0 + c) * c).exp();
        let g = ((0.78 + 0.12 * c) * c).exp();
        (b, g)
    }

    /// Construct an MBBEFD exposure curve from the Swiss Re continuous parameter `c >= 0`
    /// using Bernegger (1997) equation (4.1):
    /// - `b(c) = exp(3.1 - 0.15 * (1 + c) * c)`
    /// - `g(c) = exp((0.78 + 0.12 * c) * c)`
    pub fn from_swiss_re_c(c: f64) -> Result<Self, PricingError> {
        if !c.is_finite() || !(c >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "Swiss Re parameter c must be finite and non-negative",
            ));
        }
        let (b, g) = Self::swiss_re_bg(c);
        Self::new(b, g)
    }

    /// Construct one of the canonical Swiss Re preset curves (`Y1`–`Y4`, `Lloyds`).
    pub fn from_preset(preset: SwissReCurve) -> Self {
        let (b, g) = Self::swiss_re_bg(preset.c_parameter());
        Self { b, g }
    }

    /// Validate curve parameters.
    pub fn validate(&self) -> Result<(), PricingError> {
        Self::new(self.b, self.g).map(|_| ())
    }

    /// Evaluate the exposure curve `G(d) = E[min(X, d)] / E[X]` for normalized
    /// deductible/limit `d in [0, 1]`.
    ///
    /// Values `d <= 0` return `0.0` and `d >= 1` return `1.0` (after checking finiteness).
    pub fn exposure_curve(&self, d: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !d.is_finite() {
            return Err(PricingError::InvalidParameter(
                "normalized limit d must be finite",
            ));
        }
        if d <= 0.0 {
            return Ok(0.0);
        }
        if d >= 1.0 {
            return Ok(1.0);
        }

        let b = self.b;
        let g = self.g;

        // Limiting case 1: total-loss-only (g == 1 or b == 0)
        if (g - 1.0).abs() <= 1e-14 || b == 0.0 {
            return Ok(d);
        }

        // Limiting case 2: b == 1 (g > 1)
        if (b - 1.0).abs() <= 1e-12 {
            return Ok(((g - 1.0) * d).ln_1p() / g.ln());
        }

        // Limiting case 3: b * g == 1 (g > 1)
        let bg = b * g;
        if (bg - 1.0).abs() <= 1e-12 {
            return Ok((1.0 - b.powf(d)) / (1.0 - b));
        }

        // General MBBEFD formula (Bernegger 1997, eq. 3.3)
        let num = (g - 1.0) * b + (1.0 - bg) * b.powf(d);
        let ratio = num / (1.0 - b);
        if !(ratio > 0.0) || !ratio.is_finite() {
            return Err(PricingError::OutOfBounds(
                "MBBEFD exposure curve ratio underflowed or became non-positive",
            ));
        }
        Ok((ratio.ln() / bg.ln()).clamp(0.0, 1.0))
    }

    /// Evaluate the implied severity survival function `S(x) = P(X > x)` on `[0, 1)`.
    ///
    /// Note that `S(0) = 1.0`, `lim_{x -> 1-} S(x) = 1 / g` (the probability mass of a
    /// total loss at `x = 1`), and `S(x) = 0.0` for `x >= 1.0`.
    pub fn survival(&self, x: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !x.is_finite() {
            return Err(PricingError::InvalidParameter(
                "loss ratio x must be finite",
            ));
        }
        if x <= 0.0 {
            return Ok(1.0);
        }
        if x >= 1.0 {
            return Ok(0.0);
        }

        let b = self.b;
        let g = self.g;

        if (g - 1.0).abs() <= 1e-14 || b == 0.0 {
            return Ok(1.0);
        }
        if (b - 1.0).abs() <= 1e-12 {
            return Ok(1.0 / (1.0 + (g - 1.0) * x));
        }
        let bg = b * g;
        if (bg - 1.0).abs() <= 1e-12 {
            return Ok(b.powf(x));
        }

        let bx = b.powf(x);
        let denom = (g - 1.0) * b + (1.0 - bg) * bx;
        let s = (1.0 - b) * bx / denom;
        if !s.is_finite() {
            return Err(PricingError::OutOfBounds(
                "MBBEFD survival calculation overflowed or became undefined",
            ));
        }
        Ok(s.clamp(0.0, 1.0))
    }

    /// Expected normalized loss `E[X] = 1 / G'(0)` on `[0, 1]`.
    pub fn mean_loss_ratio(&self) -> Result<f64, PricingError> {
        self.validate()?;
        let b = self.b;
        let g = self.g;

        if (g - 1.0).abs() <= 1e-14 || b == 0.0 {
            return Ok(1.0);
        }
        if (b - 1.0).abs() <= 1e-12 {
            return Ok(g.ln() / (g - 1.0));
        }
        let bg = b * g;
        if (bg - 1.0).abs() <= 1e-12 {
            return Ok((b - 1.0) / b.ln());
        }
        let mean = ((1.0 - b) * bg.ln()) / ((1.0 - bg) * b.ln());
        if !mean.is_finite() {
            return Err(PricingError::OutOfBounds(
                "MBBEFD mean loss ratio overflowed or became undefined",
            ));
        }
        Ok(mean)
    }

    /// Probability of a total loss (`X = 1`), equal to `1 / g`.
    #[inline]
    pub fn total_loss_probability(&self) -> Result<f64, PricingError> {
        self.validate()?;
        Ok(1.0 / self.g)
    }

    /// Exposure-rated excess-of-loss layer factor `G(min((attachment + limit) / mpl, 1)) - G(min(attachment / mpl, 1))`.
    ///
    /// Represents the fraction of expected ground-up loss falling into the layer
    /// `limit xs attachment` for a risk with Maximum Possible Loss / TIV `mpl`.
    pub fn layer_exposure_factor(
        &self,
        attachment: f64,
        limit: f64,
        mpl: f64,
    ) -> Result<f64, PricingError> {
        self.validate()?;
        if !attachment.is_finite()
            || !limit.is_finite()
            || !mpl.is_finite()
            || !(attachment >= 0.0)
            || !(limit > 0.0)
            || !(mpl > 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "attachment must be >= 0, and limit and mpl must be finite and > 0",
            ));
        }
        let d_lo = (attachment / mpl).min(1.0);
        let d_hi = ((attachment + limit) / mpl).min(1.0);
        let g_lo = self.exposure_curve(d_lo)?;
        let g_hi = self.exposure_curve(d_hi)?;
        Ok((g_hi - g_lo).max(0.0))
    }

    /// Expected ceded loss to an excess-of-loss layer `limit xs attachment` for a risk
    /// with expected ground-up loss `expected_ground_up_loss` and insured value `mpl`.
    pub fn layer_expected_loss(
        &self,
        expected_ground_up_loss: f64,
        attachment: f64,
        limit: f64,
        mpl: f64,
    ) -> Result<f64, PricingError> {
        if !expected_ground_up_loss.is_finite() || !(expected_ground_up_loss >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "expected_ground_up_loss must be finite and non-negative",
            ));
        }
        let factor = self.layer_exposure_factor(attachment, limit, mpl)?;
        Ok(expected_ground_up_loss * factor)
    }
}
