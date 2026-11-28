# Changelog

All notable changes to `pricing-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (full actuarial review, 2026-11-27)
- **Aggregate / cat (`underwriting_risk`)**: Panjer and FFT now detect grid truncation and circular wrap-around, and multi-component convolution rejects an `fft_size` below the combined support (exact, no aliasing; component mass deficits are checked since round 2) (`AGGREGATE_GRID_TOLERANCE` = 1e-6 lost mass / mean error) instead of returning silently biased distributions; `var`/`tvar` error beyond retained mass and TVaR ≥ VaR is guaranteed; ELT AEP rejects losses beyond the grid and uses mean-preserving Beta discretization (AEP mean = AAL to 1e-9); YLT Poisson sampler is exact for λ > 30 (normal approximation removed); floating-point quantile-rank errors fixed in `sample_var` and empirical PML.
- **Credibility**: Hachemeister VHM is now an unbiased exposure-weighted moment estimator for unequal designs (was ~2× biased); collective trend is credibility-weighted $b = (\sum Z_i)^{-1}\sum Z_i\beta_i$. Classical standards validate inputs (tolerance 0 no longer yields NaN).
- **Experience rating**: Table M rescales mean entry ratio to exactly 1 and renormalizes probabilities (φ(0)=1, ψ=φ+r−1 to 1e-12); Table L renormalizes probabilities; retro `expense_ratio` correctly documented as excluding premium taxes but including LCF-provided LAE.
- **GLMs (`classification`)**: Hurdle count part is a true zero-truncated Poisson MLE (previously double-corrected, +58% bias); Tweedie power selection uses the exact Dunn & Smyth (2005) series density with profiled φ (saddlepoint removed); AIC/BIC use the full log-likelihood at ML dispersion; regularized GLMs no longer clamp η to ±20 (overflow bound 700, surfaced as `NonConvergence`/`OutOfBounds`); Hurdle/ZIP/GLMM validate lengths (no panic) and report `converged`; outer loop honours `options.tolerance`; B-spline endpoint/domain handling fixed; deviance-monitored IRLS step-halving (Marschner 2011).
- **Severity / EVT (`layers`)**: Burr XII reparametrized in (ln α, ln γ, ln θ), scale-equivariant; GPD ξ > −1 (Smith 1985); boundary or NaN-SE fits report `converged = false`; `SolverOptions::tol` honoured everywhere; SEs cover every estimated field in declared order; weighted BIC/KS/AD with delayed-entry product-limit KS under left truncation.
- **ALAE ILFs**: ALAE-Included under bivariate Lognormal uses an exact 1-D conditional-LEV reduction with adaptive Gauss–Kronrod (fixed 32×32 Gauss–Legendre had errors up to 3.8%). Power-Law documented as deterministic ALAE.
- **Reinsurance**: pro-rata-as-to-amount-and-time reinstatements are exact under compound Poisson with uniform arrivals (Sundt 1991; Mata 2000); the flat 0.5 time factor (first reinstatement −16% at λ=1) is removed.
- **Tilted FFT round-off (`fft_aggregate_loss`)**: untilting no longer amplifies inverse-FFT round-off by $e^{\theta k}$ into false aliasing errors at standard tilts ($\theta N \approx 20$); bins below `FFT_ROUNDOFF_SAFETY_FACTOR`·noise·$e^{\theta k}$ are zeroed (also clips negatives) and the guard allows the root-sum-square round-off floor. Excess tilt now errors with "reduce tilt" (the old message advised increasing it). Tilted FFT matches Panjer to 1e-10 per bin for θN ∈ {5, 10, 20, 30}.
- **Severity input mass**: Panjer/FFT accept a severity PMF only within `AGGREGATE_GRID_TOLERANCE` of unit mass (was 1e-4, then misreported as grid truncation) and renormalize it; the error names the input deficit.
- **Convolution**: `convolve_independent_aggregates` rejects components whose total mass differs from 1 by more than `AGGREGATE_GRID_TOLERANCE` (0.8 × 0.8 → 0.64 was silent).
- **TVaR tail coverage**: `AggregateLossDistribution::tvar` errors when lost mass exceeds `TVAR_MAX_LOST_TAIL_FRACTION` (1e-3) of the tail mass 1 − α (at α = 0.99999 a 9e-7 loss left only 91% of the tail); doc now states the result is a lower bound over retained tail mass.
- **Cat XL layer loss**: Beta events use the exact LEV difference instead of a 128-point midpoint integral (wrong for concentrated Betas).
- **YLT Beta sampler**: Gamma draws in log space (Marsaglia–Tsang boost `ln G = ln G(α+1) + ln U / α`); shapes < 1e-3 no longer underflow to the 0.5·M fallback (74% of mass), which is removed.
- **`special::regularized_incomplete_beta`**: prefactor via Stirling-remainder differences and `rlog1` (DiDonato & Morris 1992 `brcomp`) instead of cancelling `ln Γ` terms; binomial-tail parity ≤ 1e-12 relative at a + b up to 1e6 (old prefactor error ~1e-9).
- **MBBEFD**: continuous across b=1 and bg=1 via `expm1`/`ln_1p` (removed 1e-12 tolerance branches with 5e-5 cancellation error).
- **Hachemeister time-origin invariance**: estimation, PSD projection of `A` and credibility matrices now run in centred time `t − t̄` (exposure-weighted collective barycentre, Bühlmann & Gisler 2005 §8.4) and are mapped back; predictions are invariant to time-origin shifts (previously `t = 2021..2025` collapsed `A` to 0 and changed predictions by up to 0.44).
- **NCCI ERP**: mods are rounded half-up to 2 decimals (`NCCI_MOD_DECIMALS`), as published by NCCI (*ABCs of Experience Rating* worksheet golden: 243,120.72 / 242,971 → 1.00); previously 3 decimals.
- **Myers–Cohn**: the loss-related underwriting tax shield `τL` is discounted at the risk-adjusted `r_L` (Myers & Cohn 1987; Robbin), not `r_f`; 1-period golden `P` 95.7543 → 1233/13 = 94.8462.
- **Table L**: `from_limited_discrete_pdf` with `1 < E[Y*] ≤ 1 + 1e-4` rescales entry ratios (k = 0, φ*(0) = 1) and re-validates via `from_rows`, so every constructed engine serde round-trips.
- **Classical credibility**: `credibility(+∞)` = 1 (was NaN); NaN/negative volume is an error.
- Docs: `TableMEngine::from_aggregate` normalizes by the renormalized retained-pmf mean, not `agg.mean`.
- **Quadrature (`layers`)**: Gauss–Kronrod is globally adaptive with scale-aware initial subdivision (geometric grading to both endpoints and around natural scale points); a first panel with both rules ≈ 0 is no longer accepted. **Breaking:** `gauss_kronrod_integrate*` return `Result` (no silent NaN/0) and `gauss_kronrod_integrate_semi_infinite` takes a `scale` argument.
- **EVT LEVs**: `FittedEvtDistribution::limited_expected_value` uses closed forms for Weibull (incomplete gamma), Burr XII with αγ > 1 (incomplete beta), Inverse Gaussian (normal CDF), shifted GPD and spliced body + GPD (integrating only to the support end u + β/|ξ| when ξ < 0). Previously Weibull/Burr/IG LEV(1e7–1e9) returned ≈ 0 and GPD ξ < 0 was 25% low.
- **Power-law ALAE**: Pro-Rata, In-Addition and Included (cv = 0) use closed-form truncated power moments per family (was a unit-scale semi-infinite map: In-Addition −20.8% at Lognormal(15, 1), E[A] = 0 for SPP(1.5, 1e5)); In-Addition errors when E[X^γ] is infinite (Lomax/SPP γ ≥ α, GPD ξγ ≥ 1); exact densities replace finite differences in the fallback quadrature.
- **Hurdle Poisson exposure**: zero part is now a complementary log-log GLM with the `ln E` offset, $P(Y>0)=1-e^{-E e^{x^T\gamma}}$ (previously logit without offset: $P(Y>0)$ exposure-independent, fitted means 4.1× off at low exposure).
- **GLMM**: `fit_random_intercept_glmm` solves the joint conditional mode $(\hat\beta, \hat u)$ by penalized Newton (Schur complement; `glmer(nAGQ = 0)`) instead of alternating fixed/random updates that stopped ~0.15 nats from the optimum (τ²=9: b̂₀ = 2.51 vs ≈ −1.5); Σû = 0 holds exactly; scale-free Newton-decrement / relative-step stops; `ln τ²` search expands up to τ² = 1e4 instead of silently capping at e² ≈ 7.39.
- **IRLS convergence** (`fit_glm_irls*`): R `glm.fit` relative-deviance stop $|D-D_{old}|/(|D|+0.1)<$ `tol` (|Δβ| ≤ `tol` kept as secondary); zero-claim levels and separated Binomial fits now return `Ok` (were `NonConvergence` even at 2000 iterations).
- **`select_tweedie_power`**: a failing candidate no longer aborts the grid; failures are skipped and recorded, error only if none succeed.
- **`double_lift_chart`**: never drops empty buckets silently; each bucket receives ≥ 1 observation.

### Changed (breaking)
- YLT `oep_pml`/`aep_pml` default to the rank convention $T = N/r$ and error when $T > N$; `EmpiricalPmlConvention::LowerQuantile` restores the old behaviour.
- `var`/`tvar`, Panjer, FFT, convolution and ELT-AEP return errors where they previously returned silently biased values.
- Panjer/FFT reject severity PMFs with |Σf − 1| > 1e-6 (`InvalidParameter`; previously 1e-4) and renormalize accepted ones; `tvar` errors on under-represented tails; convolution rejects non-unit-mass components; FFT out-of-bounds messages changed (tilt advice corrected). YLT Beta draws change at the ulp level for all shapes (log-space sampler) and materially for shapes < 1e-3; cat XL Beta layer losses change (exact LEV).
- `price_reinstatements` rejects `ReinstatementBasis::ProRataAmountAndTime`; use `price_reinstatements_compound_poisson` (exact) or `price_reinstatements_time_weighted` (explicit time factor). The 0.4.0 note "pro-rata by amount and by time" referred to the flat-0.5 approximation.
- `AlaeDependenceModel::BivariateLognormal` rejects a severity that differs from its owned marginal (`indemnity_marginal()`).
- Spliced EVT SE vectors have length 5 (body_weight included); new field `EvtFitResult.iterations`.
- `TrendModel::new` and `ClassicalCredibility::from_confidence_and_tolerance` return `Result`; `TableMEngine.rows` / `TableLEngine.rows` are private (`rows()`, validated `from_rows`, validated serde).
- `HachemeisterResult` adds `vhm_matrix_raw`; Hachemeister numbers change.
- `HachemeisterResult` adds `time_centre`; results change whenever the raw-origin PSD projection was active (coefficients/matrices still reported in the caller's time coordinates).
- `NcciErpResult.mod_factor` is 2-decimal (e.g. 0.858 → 0.86); `datasets::ncci_erp_worksheet().expected_mod_rounded` = 0.86. New pub const `NCCI_MOD_DECIMALS`.
- `myers_cohn_profit_load` fair premium / `pv_underwriting_profit_tax` change whenever `loss_discount_rate ≠ risk_free_rate`.
- `ClassicalCredibility::credibility` and `blend` return `Result<f64, PricingError>` (FFI `pk_classical_credibility` still returns NaN on error).
- New fields: `TweediePowerSelection.best_dispersion`, `InformationCriteria.{log_likelihood, ml_dispersion}`, `converged` on `ZeroModifiedPoissonResult` and `RandomInterceptGlmmResult`; new variant `RegularizedGlmFamily::ZeroTruncatedPoisson`.
- `glm_information_criteria` rejects `QuasiPoisson` (no likelihood); AIC/BIC values change.
- New fields `GlmResult.separation_suspected`, `TweediePowerSelection.failed_powers`, `RandomInterceptGlmmResult.{tau_squared_at_lower_bound, tau_squared_at_upper_bound}`; new variant `RegularizedGlmFamily::BinomialCloglog`. `ZeroModifiedPoissonResult.zero_part` from `fit_hurdle_poisson` is a cloglog fit (predict with `BinomialCloglog` and `ln E` offsets); Hurdle fitted values change whenever offsets are supplied.
- `fit_random_intercept_glmm`: `max_iter` now bounds Newton iterations per joint-mode solve; `iterations` counts profiled-likelihood (τ²) evaluations; `laplace_log_likelihood` includes $-\ln y!$; GLMM estimates change; rank-deficient `[1, x]` returns `SingularMatrix`.
- `GlmOptions::tol` also drives the relative-deviance stop, so IRLS may stop earlier (results agree to within the tolerance).
- `double_lift_chart` errors when `n_buckets > n` (previously returned fewer buckets).
- `bspline_basis` errors outside $[t_d, t_{M-d}]$ and requires `knots.len() >= 2·degree + 2`; `batch_score_into` rejects NaN/negative inputs and min > max.

### Added
- `EmpiricalPmlConvention`, `oep_pml_with` / `aep_pml_with`, `AggregateLossDistribution::total_mass`, `AGGREGATE_GRID_TOLERANCE`, `FFT_ROUNDOFF_SAFETY_FACTOR`, `TVAR_MAX_LOST_TAIL_FRACTION`.
- NCCI Rule 2-B per-claim and multiple-claim (default 2×) accident limitations: `NcciAccidentLimitation`, `calculate_ncci_experience_mod_by_accident`.
- Classical credibility pure-premium $n_0(1+CV^2)$ and severity $n_0 CV^2$ standards.
- `tweedie_log_density` (Dunn & Smyth series), `MultiplicativeRater::{try_new, validate, try_score}`.
- `AlaeDependenceModel::PowerLawLognormal { c, gamma, cv }`, `price_reinstatements_compound_poisson`, `price_reinstatements_time_weighted`, `compound_poisson_reinstatement_slices`, `ReinstatementSliceExpectations`.
- Seeded (42069) regression suites: `tests/{underwriting_risk,classification,credibility_experience,layers}_regressions.rs` — brute-force convolution parity, Monte Carlo unbiasedness (Bühlmann–Straub, Hachemeister, Jewell), parameter recovery (Burr, GPD, Tweedie p, ZTP), hand-computed OEP/AEP/AggXL/Kreps/NCCI, and special-function reference values.

### Added
- **Phase 1 & Phase 2 API ergonomics**:
  - Validated domain containers and fluent builders: `DesignMatrix`, `GlmFit`, `RatingTable`, `EventLossTable`, `LayerTower`, `CredibilityPanel` / `CredibilityGroup`, `ScenarioMatrix`, `TreatyExperience`, `SolverOptions`, `NcciErpSpec`, and declarative `tower!` / `rate_changes!` macros.
  - Householder QR IRLS solver in `linalg` for improved numerical conditioning.
  - Spec entry points (`NcciErpSpec::calculate`, `IsoErpSpec::calculate`, `LargeDeductibleSpec::price`, `SeverityDistribution::layer_expected_loss`), cross-module bridges (`TableMEngine::from_aggregate`, `TableLEngine::from_limited_aggregate`), and GLM rate-filing helpers (`GlmResult::relativities`, `GlmResult::to_rater`).
- **End-to-end Rate Indication & Commercial Pricing loop**:
  - `trending::indication`: Overall rate level indication (`RateIndicationSpec`, `ExpenseProvisions`, `calculate_rate_indication`) supporting both Loss Ratio and Pure Premium methods (Werner & Modlin Ch. 8) with credibility weighting and formatted `Display` output.
  - `layers::fitting`: Left-truncated (deductible) and right-censored (policy-limit) Maximum Likelihood Estimation (`SeverityObservation`, `SeverityFamily`, `fit_severity_mle`, `fit_best_severity`) for Exponential, Single-Parameter Pareto, Lognormal, Lomax, and Gamma with AIC/BIC ranking (Klugman–Panjer–Willmot Ch. 13).
  - `layers::reinsurance`: Credibility-weighted XL exposure + experience blending (`blend_exposure_and_experience`, `BlendCredibilityRule`, `ExposureExperienceBlendResult`, Clark 2014).
- **Examples & Benchmark CI parity**:
  - Runnable end-to-end examples: `examples/rate_indication.rs` and `examples/commercial_xl_pricing.rs` (smoke-tested in CI).
  - Self-hosted `theliomira` benchmark regression workflow (`.github/workflows/benchmarks.yml`), `scripts/bench-report.py`, and checked-in `benches/baseline-i9-14900k.json`.

## [0.4.0] - 2026-11-26

### Added
- **Expanded GLM families & penalized IRLS** (`classification::irls_glm`): Added `GlmFamily::{Tweedie, NegativeBinomial, Binomial, InverseGaussian}`, L2 (Ridge) penalized IRLS via `GlmOptions` / `fit_glm_irls_with_options`, `predict_glm`, and Dunn & Smyth (2005) saddlepoint profile log-likelihood `select_tweedie_power`.
- **Actuarial lift & model diagnostics** (`classification::diagnostics`): Exposure-weighted Lorenz curve and tie-symmetric Gini index (Frees, Meyers & Cummings 2011), double-lift charts (`double_lift_chart`), deviance-based `glm_information_criteria` (AIC/BIC/Pearson $\chi^2$), and `variance_inflation_factors`.
- **Hachemeister (1975) regression credibility** (`credibility::hachemeister`): Credibility-weighted linear trend estimation across heterogeneous risk groups with PSD-projected $2\times 2$ VHM covariance matrix.
- **Arbitrary-term parallelogram & two-step trending** (`trending`): `parallelogram_on_level_factor_with_term` (exact trapezoid geometry for 6-month, annual, or multi-year policy terms) and `TwoStepTrendModel` (Werner & Modlin Ch. 6).

## [0.3.0] - 2026-11-26

### Added
- **Prospective compound aggregate risk** (`underwriting_risk::aggregate`): Panjer (1981) `(a, b, 0)` recursion, Robertson (1992) exponentially tilted radix-2 FFT, Klugman–Panjer–Willmot mean-preserving LEV discretization, and independent multi-component convolution (`convolve_independent_aggregates`) for attritional + large + cat underwriting risk.
- **Catastrophe Event Loss Table (ELT) engine** (`underwriting_risk::catastrophe`): Exact Poisson OEP, bisection PML inversion, Beta event severity via regularized incomplete beta, occurrence cat XL layer pricing, and compound-Poisson AEP via FFT.
- **Pricing risk loads & distortion transforms** (`underwriting_risk::risk_load`): Wang (1996, 2000) Proportional Hazards and Normal transforms, Dual Power, TVaR distortion, Kreps (1990) marginal surplus risk load, and Ruhm–Mango–Kreps (2005) tie-symmetric scenario pricing.
- **Underwriting capital allocation** (`underwriting_risk::capital`): Coherent sample `VaR`, `TVaR`, `RVaR`, Tasche (2008) Euler TVaR/RVaR/SD allocations, and Myers & Read (2001) marginal default-put surplus allocation.

## [0.2.0] - 2026-11-26

### Added
- **MBBEFD / Swiss Re exposure rating** (`layers::mbbefd`): Bernegger (1997) exact $(b, g)$ and Swiss Re $c$-curve exposure curves $G(d)$, excess-of-loss layer factors, implied severity survival, and Table 2 benchmark parity.
- **Treaty reinsurance pricing** (`layers::reinsurance`): Clark (2014) experience-rated burning cost for excess-of-loss treaties, Sundt (1991) reinstatement pricing (pro-rata by amount and by time), sliding-scale commission, profit commission, and loss corridor evaluation.
- **Table L & per-occurrence loss-limited retro plans** (`experience::table_l`): Skurnick (1974) $\phi^*(r, k)$ and $\psi^*(r, k)$ insurance charge/savings engine with exact balance identity $\phi^*(r, k) - \psi^*(r, k) = 1 - k - r$, integrated with `solve_balanced_retro_plan_table_l`.
- **Commercial large deductible pricing** (`experience::deductible`): Deductible credit with ALAE treatment (inside limit, pro-rata, outside limit), Excess Loss Factor (ELF), and aggregate deductible cap charge.
- **ISO Commercial Experience Rating & Schedule Rating** (`experience::iso_erp`): Basic-limits expected losses, Maximum Single Limit (MSL) capping, credibility $Z = E / (E + K)$, and bounded IRPM schedule rating.
- **Zero-I/O CAS benchmark fixtures** (`datasets`): Published Exam 5/8 benchmark tables and worksheets for reproducible tests and doctests.
- **WebAssembly C-ABI exports** (`wasm_exports`, behind `--features wasm`): Single-threaded `wasm32` exports for ILFs, MBBEFD exposure curves, classical credibility, parallelogram on-leveling, and NCCI ERP.

### Changed
- Moved `serde_json` from `[dependencies]` to `[dev-dependencies]` so `serde` is the sole runtime dependency.
- Marked `PricingError` as `#[non_exhaustive]` and derived `Copy` and `serde::Serialize`.
- Scoped `linalg` to `pub(crate)` to keep internal dense Cholesky routines out of the public SemVer surface.
- Added `[package.metadata.docs.rs]`, `homepage`, `documentation`, and `exclude` metadata in `Cargo.toml`.
- Expanded Criterion benchmarks to cover GLM IRLS, Bühlmann–Straub, ILFs, MBBEFD exposure curves, and Table L.

## [0.1.0] - 2026-11-26

### Added
- Initial implementation of pure-Rust P&C commercial pricing kernels: classification minimum bias (Bailey 1963, Bailey & Simon 1960), IRLS GLMs (Poisson, Quasi-Poisson, Gamma), multiplicative scorer, classical & Bühlmann–Straub credibility, NCCI ERP, Lee (1988) Table M, balanced retrospective plan solver, 6 parametric severity families with LEV/ILF/LER/excess layers, and parallelogram/exponential trending.
