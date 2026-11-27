# Changelog

All notable changes to `pricing-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-26

### Added
- **MBBEFD / Swiss Re exposure rating** (`layers::mbbefd`): Bernegger (1997) exact $(b, g)$ and Swiss Re $c$-curve exposure curves $G(d)$, excess-of-loss layer factors, implied severity survival, and Table 2 benchmark parity.
- **Treaty reinsurance pricing** (`layers::reinsurance`): Clark (2014) experience-rated burning cost for excess-of-loss treaties, Sundt (1991) reinstatement pricing (pro-rata by amount and by time), sliding-scale commission, profit commission, and loss corridor evaluation.
- **Table L & per-occurrence loss-limited retro plans** (`experience::table_l`): Skurnick (1974) $\phi^*(r, k)$ and $\psi^*(r, k)$ insurance charge/savings engine with exact balance identity $\phi^*(r, k) - \psi^*(r, k) = 1 - k - r$, integrated with `solve_balanced_retro_plan_table_l`.
- **Commercial large deductible pricing** (`experience::deductible`): Deductible credit with ALAE treatment (inside limit, pro-rata, outside limit), Excess Loss Factor (ELF), and aggregate deductible cap charge.
- **ISO Commercial Experience Rating & Schedule Rating** (`experience::iso_erp`): Basic-limits expected losses, Maximum Single Limit (MSL) capping, credibility $Z = E / (E + K)$, and bounded IRPM schedule rating.
- **Zero-I/O CAS benchmark fixtures** (`datasets`): Published Exam 5/8 benchmark tables and worksheets for reproducible tests and doctests.
- **WebAssembly C-ABI exports** (`wasm_exports`, behind `--features wasm`): Single-threaded `wasm32` exports for ILFs, MBBEFD exposure curves, NCCI ERP, and Table M/L queries.

### Changed
- Moved `serde_json` from `[dependencies]` to `[dev-dependencies]` so `serde` is the sole runtime dependency.
- Marked `PricingError` as `#[non_exhaustive]` and derived `Copy` and `serde::Serialize`.
- Scoped `linalg` to `pub(crate)` to keep internal dense Cholesky routines out of the public SemVer surface.
- Added `[package.metadata.docs.rs]`, `homepage`, `documentation`, and `exclude` metadata in `Cargo.toml`.
- Expanded Criterion benchmarks to cover GLM IRLS, Bühlmann–Straub, ILFs, MBBEFD exposure curves, and Table L.

## [0.1.0] - 2026-11-26

### Added
- Initial implementation of pure-Rust P&C commercial pricing kernels: classification minimum bias (Bailey 1963, Bailey & Simon 1960), IRLS GLMs (Poisson, Quasi-Poisson, Gamma), multiplicative scorer, classical & Bühlmann–Straub credibility, NCCI ERP, Lee (1988) Table M, balanced retrospective plan solver, 6 parametric severity families with LEV/ILF/LER/excess layers, and parallelogram/exponential trending.
