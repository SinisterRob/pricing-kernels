//! C-ABI exports for browser WebAssembly builds (`--features wasm`).
//!
//! All exports accept and return `f64` / `u32` so JavaScript hosts can call them
//! directly through `WebAssembly.Instance.exports` without bindgen glue. Invalid
//! inputs return `f64::NAN` rather than panicking across the FFI boundary.
//!
//! # Safety
//!
//! Batch exports exchange claim vectors through a process-global `static mut`
//! scratch buffer and are **strictly for single-threaded `wasm32` hosts**. Do not
//! invoke them concurrently from multiple threads on native targets.

#![allow(unsafe_code)]

use crate::credibility::ClassicalCredibility;
use crate::experience::{ExperienceClaim, NcciErpSpecification, calculate_ncci_experience_mod};
use crate::layers::{MbbefdCurve, SeverityDistribution};
use crate::trending::{RateChange, parallelogram_on_level_factor};

const MAX_CLAIMS: usize = 256;

static mut CLAIM_AMOUNTS: [f64; MAX_CLAIMS] = [0.0; MAX_CLAIMS];
static mut CLAIM_MED_ONLY: [u8; MAX_CLAIMS] = [0; MAX_CLAIMS];

/// Maximum number of claims supported by the static WASM scratch buffers.
#[unsafe(no_mangle)]
pub extern "C" fn pk_max_claims() -> u32 {
    MAX_CLAIMS as u32
}

/// Pointer to the `f64` claim incurred-amount scratch buffer (`MAX_CLAIMS` elements).
#[unsafe(no_mangle)]
pub extern "C" fn pk_claim_amounts_ptr() -> *mut f64 {
    core::ptr::addr_of_mut!(CLAIM_AMOUNTS).cast()
}

/// Pointer to the `u8` medical-only flag scratch buffer (`0` = indemnity, non-zero = med-only).
#[unsafe(no_mangle)]
pub extern "C" fn pk_claim_med_only_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(CLAIM_MED_ONLY).cast()
}

macro_rules! pk_ilf_export {
    ($(#[$meta:meta])* $fn_name:ident ( $($arg:ident : f64),+ ) => $variant:ident { $($field:ident),+ }) => {
        $(#[$meta])*
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn_name($($arg: f64),+, limit: f64, basic_limit: f64) -> f64 {
            SeverityDistribution::$variant { $($field),+ }
                .increased_limits_factor(limit, basic_limit)
                .unwrap_or(f64::NAN)
        }
    };
}

pk_ilf_export!(
    /// Compute Increased Limit Factor for a Lomax (Pareto Type II) severity distribution.
    pk_ilf_lomax(alpha: f64, theta: f64) => Lomax { alpha, theta }
);
pk_ilf_export!(
    /// Compute Increased Limit Factor for a Lognormal severity distribution.
    pk_ilf_lognormal(mu: f64, sigma: f64) => Lognormal { mu, sigma }
);
pk_ilf_export!(
    /// Compute Increased Limit Factor for a Gamma severity distribution.
    pk_ilf_gamma(shape: f64, scale: f64) => Gamma { shape, scale }
);
pk_ilf_export!(
    /// Compute Increased Limit Factor for a composite Spliced Gamma–Pareto severity distribution.
    pk_ilf_spliced_gamma_pareto(shape: f64, scale: f64, splice: f64, tail_alpha: f64)
        => SplicedGammaPareto { shape, scale, splice, tail_alpha }
);

/// Evaluate Swiss Re / Bernegger (1997) MBBEFD exposure curve `G(d)` for continuous `c >= 0`.
#[unsafe(no_mangle)]
pub extern "C" fn pk_mbbefd_swiss_re_g(c: f64, d: f64) -> f64 {
    MbbefdCurve::from_swiss_re_c(c)
        .and_then(|curve| curve.exposure_curve(d))
        .unwrap_or(f64::NAN)
}

/// Evaluate Swiss Re / Bernegger (1997) excess-of-loss layer exposure factor `G((A+L)/M) - G(A/M)`.
#[unsafe(no_mangle)]
pub extern "C" fn pk_mbbefd_layer_factor(c: f64, attachment: f64, limit: f64, mpl: f64) -> f64 {
    MbbefdCurve::from_swiss_re_c(c)
        .and_then(|curve| curve.layer_exposure_factor(attachment, limit, mpl))
        .unwrap_or(f64::NAN)
}

/// Evaluate classical limited-fluctuation credibility weight `Z` for given full-credibility standard.
#[unsafe(no_mangle)]
pub extern "C" fn pk_classical_credibility(full_standard: f64, observed_volume: f64) -> f64 {
    ClassicalCredibility {
        full_credibility_standard: full_standard,
    }
    .credibility(observed_volume)
}

/// Evaluate single-change Calendar-Year earned parallelogram on-level factor.
#[unsafe(no_mangle)]
pub extern "C" fn pk_parallelogram_olf(
    calendar_year: f64,
    effective_date: f64,
    rate_factor: f64,
) -> f64 {
    let changes = [RateChange {
        effective_date,
        rate_factor,
    }];
    parallelogram_on_level_factor(calendar_year, &changes).unwrap_or(f64::NAN)
}

/// Compute NCCI Experience Modification Factor from the first `n_claims` entries
/// populated in `pk_claim_amounts_ptr` and `pk_claim_med_only_ptr`.
///
/// Returns `f64::NAN` if `n_claims > pk_max_claims()` or if any parameter is invalid.
#[unsafe(no_mangle)]
pub extern "C" fn pk_ncci_experience_mod(
    split_point: f64,
    expected_losses: f64,
    d_ratio: f64,
    weight_w: f64,
    ballast_b: f64,
    apply_era: u32,
    n_claims: u32,
) -> f64 {
    let count = n_claims as usize;
    if count > MAX_CLAIMS {
        return f64::NAN;
    }
    let mut claims = Vec::with_capacity(count);
    let amounts_ptr = core::ptr::addr_of!(CLAIM_AMOUNTS).cast::<f64>();
    let med_ptr = core::ptr::addr_of!(CLAIM_MED_ONLY).cast::<u8>();
    for i in 0..count {
        // SAFETY: Single-threaded WASM host; `i < count <= MAX_CLAIMS` stays within
        // the static scratch buffers initialized to valid `f64` / `u8` values.
        let (incurred, med_only) =
            unsafe { (amounts_ptr.add(i).read(), med_ptr.add(i).read() != 0) };
        claims.push(ExperienceClaim {
            incurred,
            is_medical_only: med_only,
        });
    }
    let spec = NcciErpSpecification {
        split_point,
        expected_losses,
        d_ratio,
        weight_w,
        ballast_b,
        apply_era: apply_era != 0,
    };
    calculate_ncci_experience_mod(&spec, &claims).map_or(f64::NAN, |res| res.mod_factor)
}
