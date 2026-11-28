//! Internal NaN-rejecting numeric validation helpers mirroring `reserving-kernels::validate`.

use crate::error::PricingError;

/// Verify that `value` is finite.
#[inline]
pub(crate) fn require_finite(value: f64, msg: &'static str) -> Result<(), PricingError> {
    if !value.is_finite() {
        return Err(PricingError::InvalidParameter(msg));
    }
    Ok(())
}

/// Verify that `value` is finite and strictly positive (`> 0.0`).
#[inline]
pub(crate) fn require_positive(value: f64, msg: &'static str) -> Result<(), PricingError> {
    if !value.is_finite() || !(value > 0.0) {
        return Err(PricingError::NonPositiveValue(msg));
    }
    Ok(())
}

/// Verify that `value` is finite and non-negative (`>= 0.0`).
#[inline]
pub(crate) fn require_non_negative(value: f64, msg: &'static str) -> Result<(), PricingError> {
    if !value.is_finite() || !(value >= 0.0) {
        return Err(PricingError::InvalidParameter(msg));
    }
    Ok(())
}

/// Verify that `value` is finite and lies in `[0.0, 1.0]`.
#[inline]
pub(crate) fn require_unit_interval(value: f64, msg: &'static str) -> Result<(), PricingError> {
    if !value.is_finite() || !(value >= 0.0) || !(value <= 1.0) {
        return Err(PricingError::InvalidParameter(msg));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_reject_nan_inf_and_out_of_domain() {
        assert!(require_finite(f64::NAN, "x").is_err());
        assert!(require_finite(f64::INFINITY, "x").is_err());
        assert!(require_finite(0.0, "x").is_ok());

        assert!(require_positive(0.0, "x").is_err());
        assert!(require_positive(-1.0, "x").is_err());
        assert!(require_positive(f64::NAN, "x").is_err());
        assert!(require_positive(1e-12, "x").is_ok());

        assert!(require_non_negative(-1e-12, "x").is_err());
        assert!(require_non_negative(f64::NAN, "x").is_err());
        assert!(require_non_negative(0.0, "x").is_ok());

        assert!(require_unit_interval(-0.01, "x").is_err());
        assert!(require_unit_interval(1.01, "x").is_err());
        assert!(require_unit_interval(f64::NAN, "x").is_err());
        assert!(require_unit_interval(0.0, "x").is_ok());
        assert!(require_unit_interval(1.0, "x").is_ok());
    }
}
