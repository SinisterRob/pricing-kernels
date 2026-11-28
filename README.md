<p align="center">
  <img src="https://raw.githubusercontent.com/SinisterRob/pricing-kernels/main/assets/banner.png" alt="pricing-kernels — Actuarial ratemaking in Rust" width="100%" />
</p>

<p align="center">
  <strong>Pure Rust numeric kernels for P&amp;C actuarial pricing and underwriting-risk capital modeling.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/pricing-kernels"><img src="https://img.shields.io/crates/v/pricing-kernels?style=flat" alt="Crates.io" /></a>
  <a href="https://docs.rs/pricing-kernels"><img src="https://img.shields.io/docsrs/pricing-kernels?style=flat" alt="docs.rs" /></a>
  <a href="https://github.com/SinisterRob/pricing-kernels/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SinisterRob/pricing-kernels/ci.yml?branch=main&label=CI&style=flat" alt="CI" /></a>
  <a href="https://github.com/SinisterRob/pricing-kernels/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/Rust-1.97%2B-f97316?style=flat" alt="Rust 1.97+" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-30363d?style=flat" alt="License: MIT or Apache-2.0" /></a>
  <a href="https://github.com/SinisterRob/pricing-kernels/blob/main/src/lib.rs"><img src="https://img.shields.io/badge/unsafe-forbidden-30363d?style=flat" alt="No unsafe code" /></a>
  <a href="https://github.com/SinisterRob/pricing-kernels/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/deps-serde%20only-30363d?style=flat" alt="Dependencies: serde only" /></a>
  <a href="#building-and-testing"><img src="https://img.shields.io/badge/target-wasm32-30363d?style=flat" alt="WebAssembly target" /></a>
</p>

<p align="center">
  <a href="#performance">Performance</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#building-and-testing">Build &amp; Test</a> &nbsp;·&nbsp;
  <a href="https://github.com/SinisterRob/pricing-kernels/blob/main/docs/SOURCES.md">Literature</a> &nbsp;·&nbsp;
  <a href="https://github.com/SinisterRob/reserving-kernels">reserving-kernels</a>
</p>

---

`pricing-kernels` covers prospective commercial-lines ratemaking (classification, credibility, experience rating, ILFs, retrospective plans, large deductibles), treaty reinsurance pricing (MBBEFD exposure curves, burning cost, reinstatements, sliding-scale & profit commissions), and the **underwriting risk** module of capital modeling. Every method is implemented from peer-reviewed actuarial literature ([`docs/SOURCES.md`](https://github.com/SinisterRob/pricing-kernels/blob/main/docs/SOURCES.md)) and verified against published benchmark problems ([`tests/source_benchmarks.rs`](https://github.com/SinisterRob/pricing-kernels/blob/main/tests/source_benchmarks.rs)). Standalone sister crate to [`reserving-kernels`](https://github.com/SinisterRob/reserving-kernels) (retrospective loss development and reserve risk).

## Implemented Modules

| Area | Modules | Capabilities |
| --- | --- | --- |
| **Classification & Scoring** | `classification::{minimum_bias, irls_glm, regularized_glm, diagnostics, scoring}` | `RatingTable` Bailey (1963) & Bailey–Simon (1960) minimum bias, `DesignMatrix` / `GlmFit` Householder-QR IRLS GLMs, Penalized Elastic Net (Ridge/LASSO) IRLS + coordinate descent, Negative Binomial / Hurdle / ZIP frequency, B-splines & natural cubic splines, Bühlmann-in-GLM / Laplace random-intercept GLMMs, Gini/Lorenz/double-lift/AIC/BIC/VIF diagnostics, zero-alloc `MultiplicativeRater` |
| **Credibility** | `credibility::{classical, buhlmann_straub, hachemeister}` | Longley-Cook limited-fluctuation credibility, non-parametric Bühlmann–Straub credibility, Hachemeister (1975) regression trend credibility |
| **Experience, Retro & Deductibles** | `experience::{ncci_erp, iso_erp, table_m, table_l, retrospective, deductible}` | NCCI WC ERP & off-balance, ISO Commercial ERP & bounded IRPM schedule rating, Lee (1988) Table M, Skurnick (1974) Table L, balanced retro solvers, large deductibles (ALAE inside/pro-rata/outside + aggregate caps) |
| **Severity, ILFs & Reinsurance** | `layers::{ilf, fitting, evt_fitting, alae_ilf, mbbefd, reinsurance}` | 8 parametric severity families + EVT/spliced $C^0$/$C^1$ body+GPD censored/truncated MLE (`fit_evt_mle` with Fisher SEs, KS, AD, mean excess), ALAE-aware ILF & excess pricing (Included/Pro-Rata/In-Addition via Miccolis, power-law, bivariate Lognormal + Gauss–Kronrod G7-K15), `CommercialLayer` & `LayerTower` (`tower!`), MBBEFD / Swiss Re exposure curves, burning cost, reinstatements, sliding-scale / corridor / profit commissions |
| **Trending & Rate Indication** | `trending::{parallelogram, trend, indication, ratemaking}` | Exact Calendar-Year & Policy-Year on-leveling (`rate_changes!`), extension of exposures, single/two-step exponential & AWP drift trending, Pure Premium & Loss Ratio indications, Myers–Cohn (1987) & ROE-implied profit loads, constrained rate-capping off-balance solver |
| **Underwriting-Risk Capital** | `underwriting_risk::{aggregate, catastrophe, risk_load, capital}` | Panjer `(a, b, 0)` & tilted radix-2 FFT aggregate risk, `EventLossTable` Cat ELT OEP/AEP/PML, `ScenarioMatrix` Wang / PH / Kreps / RMK risk loads, Euler & Myers–Read capital allocation |
| **Fixtures & WASM** | `datasets`, `wasm_exports` (`--features wasm`) | Zero-I/O CAS Exam 5/8 benchmark fixtures, single-threaded `wasm32` C-ABI exports |

## Why `pricing-kernels`?

- **Zero heavy dependencies:** Pure Rust with no BLAS, LAPACK, or large external frameworks. Linear algebra, IRLS, and root-finding routines are built in and compile cleanly to `wasm32` (`--features wasm`).
- **Deterministic by design:** Numerical algorithms converge deterministically across native and browser targets.
- **Microsecond execution:** High-performance classification and commercial rating routines designed for real-time quotation engines, large-account underwriting workflows, and batch raters.
- **Actuarial verification:** Standard models verified against CAS Exam 5 and Exam 8 literature — Bailey & Simon (1960), Brown (1988), Bühlmann & Straub (1970), NCCI Experience Rating Plan, and Lee (1988) Table M.

## Performance

Single-threaded `--release` Criterion medians on Intel Core i9-14900K (`rustc 1.97`, pure Rust, zero BLAS/LAPACK, `#![forbid(unsafe_code)]`). Reproduce locally with `cargo bench --bench pricing_bench`.

<p align="center">
  <img src="https://raw.githubusercontent.com/SinisterRob/pricing-kernels/main/assets/benchmarks.svg" alt="pricing-kernels steady-state benchmark latency" width="100%" />
</p>

<details>
<summary><strong>Full benchmark table (17 kernels across all 6 pricing domains)</strong></summary>

| Kernel | Median (`i9-14900K`) | Domain |
| --- | :---: | :---: |
| Multiplicative Rater (single policy) | `1.55 ns` | Classification |
| Parallelogram On-Level (3 rate changes) | `11.1 ns` | Trending |
| NCCI Experience Mod (ERP) | `11.1 ns` | Experience |
| ILF Mixed Exponential | `37.9 ns` | Layers |
| MBBEFD Layer Exposure Factor | `43.2 ns` | Layers |
| ILF Lognormal | `43.8 ns` | Layers |
| Bühlmann–Straub Credibility (`3×3`) | `45.6 ns` | Credibility |
| Balanced Retro Plan (Table L) | `0.49 µs` | Experience |
| Bailey–Simon Minimum Bias (`10×10`) | `0.60 µs` | Classification |
| Balanced Retro Plan (Table M) | `0.64 µs` | Experience |
| Poisson IRLS GLM (`3×2`) | `0.73 µs` | Classification |
| Weighted Gini Index (200 obs) | `0.86 µs` | Classification |
| Panjer Recursion (Poisson, 256 pts) | `1.77 µs` | Capital |
| Tweedie Ridge IRLS GLM (`3×2`) | `3.59 µs` | Classification |
| Cat ELT OEP / PML (100 events) | `11.9 µs` | Capital |
| Euler TVaR Allocation (500 scenarios) | `13.0 µs` | Capital |
| Tilted FFT Aggregate (1,024 pts) | `30.7 µs` | Capital |

</details>

## Quick Start

Add `pricing-kernels` to your `Cargo.toml`:

```toml
[dependencies]
pricing-kernels = "0.4"
```

### 1. Classification Minimum Bias (Bailey & Simon 1960)

```rust
use pricing_kernels::prelude::*;

fn main() -> Result<(), PricingError> {
    // 2x2 multi-dimensional classification table (exposures and average losses)
    let exposures = vec![1000.0, 800.0, 500.0, 400.0];
    let losses = vec![100.0, 120.0, 150.0, 180.0];

    let result = bailey_simon_multiplicative(2, 2, &exposures, &losses, 100, 1e-6)?;
    println!("Base Rate: {:.2}", result.base_rate);
    println!("Row Relativities: {:?}", result.relativities_dim1);
    println!("Column Relativities: {:?}", result.relativities_dim2);
    Ok(())
}
```

### 2. Workers' Compensation Experience Rating (NCCI ERP)

```rust
use pricing_kernels::prelude::*;

fn main() -> Result<(), PricingError> {
    let spec = NcciErpSpecification {
        split_point: 18_500.0,
        expected_losses: 120_000.0,
        d_ratio: 0.38,
        weight_w: 0.18,
        ballast_b: 25_000.0,
        apply_era: true,
    };

    let claims = vec![
        ExperienceClaim { incurred: 45_000.0, is_medical_only: false },
        ExperienceClaim { incurred: 8_200.0, is_medical_only: false },
        ExperienceClaim { incurred: 12_000.0, is_medical_only: true },
    ];

    let erp = calculate_ncci_experience_mod(&spec, &claims)?;
    println!("Experience Modification Factor: {:.3}", erp.mod_factor);
    Ok(())
}
```

### 3. Severity Layering & Increased Limits Factors (ILFs)

```rust
use pricing_kernels::prelude::*;

fn main() -> Result<(), PricingError> {
    let lomax = SeverityDistribution::Lomax { alpha: 2.2, theta: 150_000.0 };

    // Calculate $1M ILF relative to $500k basic limit
    let ilf_1m = lomax.increased_limits_factor(1_000_000.0, 500_000.0)?;
    println!("$1M ILF: {:.4}", ilf_1m);

    // Price a commercial excess layer: $4M excess of $1M
    let layer = CommercialLayer::new(1_000_000.0, 4_000_000.0, 1.0)?;
    let layer_cost = layer.layer_expected_loss(&lomax)?;
    println!("Layer Expected Loss: ${:.2}", layer_cost);
    Ok(())
}
```

### 4. Commercial Retrospective Rating Plan Solver

```rust
use pricing_kernels::prelude::*;

fn main() -> Result<(), PricingError> {
    let entry_ratios = vec![0.2, 0.6, 1.0, 1.4, 1.8];
    let probs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
    let table_m = TableMEngine::from_discrete_pdf(&entry_ratios, &probs)?;

    let spec = RetroPlanSpec {
        standard_premium: 750_000.0,
        expected_loss_ratio: 0.65,
        lcf: 1.12,
        tax_multiplier: 1.035,
        max_retro_ratio: 1.35,
        min_retro_ratio: 0.55,
        expense_ratio: 0.18,
    };

    let plan = solve_balanced_retro_plan(&spec, &table_m, 100, 1e-5)?;
    println!("Basic Premium Factor: {:.4}", plan.basic_premium_factor);
    println!("Basic Premium Dollars: ${:.2}", plan.basic_premium_dollars);
    Ok(())
}
```

## Building and Testing

```bash
# Run unit, integration, doctest, and benchmark-parity tests
cargo test --all-targets --all-features
cargo test --doc --all-features

# Build for WebAssembly
cargo build --release --target wasm32-unknown-unknown --features wasm
```

## License

Licensed under either of [MIT](https://github.com/SinisterRob/pricing-kernels/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/SinisterRob/pricing-kernels/blob/main/LICENSE-APACHE) at your option.
