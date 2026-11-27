# pricing-kernels

Pure Rust numeric kernels for P&C commercial actuarial pricing, classification, experience rating, and retrospective plans.

[![CI](https://github.com/SinisterRob/pricing-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/SinisterRob/pricing-kernels/actions/workflows/ci.yml)
· Rust 1.97+ · MIT OR Apache-2.0 · no unsafe · serde only · wasm32 ready

Every method is implemented from peer-reviewed actuarial literature and verified against published benchmark problems in [`tests/source_benchmarks.rs`](tests/source_benchmarks.rs). See [`docs/SOURCES.md`](docs/SOURCES.md) for full citations and equation mappings.

## Why `pricing-kernels`?

* **Zero heavy dependencies**: Pure Rust with no BLAS, LAPACK, or large external frameworks. Linear algebra, IRLS, and root-finding routines are built in and compile cleanly to `wasm32` (`--features wasm`).
* **Deterministic by design**: Numerical algorithms converge deterministically across native and browser targets.
* **Microsecond execution**: High-performance classification and commercial rating routines designed for real-time quotation engines, large account underwriting workflows, and batch raters.
* **Actuarial verification**: Standard models verified against CAS Exam 5 and Exam 8 literature: Bailey & Simon (1960), Brown (1988), Bühlmann & Straub (1970), NCCI Experience Rating Plan, and Lee (1988) Table M.

## Quick Start

Add `pricing-kernels` to your `Cargo.toml`:

```toml
[dependencies]
pricing-kernels = { git = "https://github.com/SinisterRob/pricing-kernels" }
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

## Building & Testing

```bash
# Run all unit and benchmark-parity tests
cargo test --all-targets --all-features

# Build for WebAssembly
cargo build --target wasm32-unknown-unknown --features wasm
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
