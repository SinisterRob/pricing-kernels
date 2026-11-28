//! Prospective underwriting-risk capital modeling, catastrophe exceedance probability (ELT/YLT),
//! pricing risk loads (Wang, PH, Kreps, RMK), and coherent capital allocation (Euler, Myers–Read).
//!
//! # Example
//! ```rust
//! use pricing_kernels::underwriting_risk::{FrequencyDistribution, panjer_recursion};
//!
//! let agg = panjer_recursion(
//!     FrequencyDistribution::Poisson { lambda: 2.0 },
//!     &[0.0, 0.5, 0.5],
//!     100.0,
//!     64,
//! )?;
//! assert!((agg.mean - 300.0).abs() < 1e-6);
//! # Ok::<(), pricing_kernels::error::PricingError>(())
//! ```

/// Compound aggregate underwriting loss distributions (Panjer recursion & tilted radix-2 FFT).
pub mod aggregate;
/// Coherent risk measures (`VaR`, `TVaR`, `RVaR`) and Euler / Myers–Read capital allocation.
pub mod capital;
/// Catastrophe Event Loss Table (ELT) engine for OEP, AEP, PML, and cat XL layer pricing.
pub mod catastrophe;
/// Distortion pricing transforms (Wang, PH, Dual Power, TVaR), Kreps risk loads, and RMK pricing.
pub mod risk_load;

pub use aggregate::{
    AggregateLossDistribution, FrequencyDistribution, convolve_independent_aggregates,
    discretize_severity_mean_preserving, fft_aggregate_loss, panjer_recursion,
};
pub use capital::{
    EulerAllocationResult, MyersReadAllocationResult, ScenarioMatrix, euler_rvar_allocation,
    euler_sd_allocation, euler_tvar_allocation, myers_read_allocation, sample_rvar, sample_tvar,
    sample_var,
};
pub use catastrophe::{
    EltSummary, EventLossEntry, EventLossTable, YearLossTable, YltEvent,
    elt_aggregate_distribution, elt_occurrence_layer_expected_loss, elt_oep, elt_oep_pml,
    elt_summary, simulate_ylt_from_elt,
};
pub use risk_load::{
    DistortionOperator, DistortionPricingResult, KrepsRiskLoadResult, RmkPricingResult,
    distortion_price_aggregate, distortion_price_discrete, kreps_marginal_risk_load,
    rmk_risk_adjusted_pricing,
};
