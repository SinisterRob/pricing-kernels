//! Regression tests for classification-module review findings (ZTP hurdle, Tweedie series
//! density / power selection, likelihood-based AIC/BIC, eta overflow bounds, convergence
//! propagation, B-spline domain handling, scorer validation), plus round-2 findings
//! (exposure-aware cloglog hurdle, joint-mode GLMM, deviance-based IRLS convergence with
//! separation flag, Tweedie grid skipping, double-lift bucket count) and closed-form
//! external references (saturated one-way Poisson / Gamma / cloglog MLEs).
//!
//! All simulations use a local SplitMix64 PRNG seeded with `SEED = 42069`.

use pricing_kernels::classification::{
    DesignMatrix, GlmFamily, GlmOptions, MultiplicativeRater, RegularizedGlmFamily,
    RegularizedGlmOptions, bspline_basis, double_lift_chart, fit_glm_irls, fit_hurdle_poisson,
    fit_random_intercept_glmm, fit_regularized_glm, fit_zero_inflated_poisson,
    glm_information_criteria, predict_glm, select_tweedie_power, tweedie_log_density,
};
use pricing_kernels::error::{PricingError, SolverOptions};

const SEED: u64 = 42069;

/// SplitMix64 (Steele, Lea & Flood 2014) — deterministic, dependency-free.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn exponential(&mut self) -> f64 {
        -self.uniform().ln()
    }

    /// Exact Poisson: Knuth multiplication on chunks of rate `<= POISSON_CHUNK`
    /// (a sum of independent Poissons is Poisson; avoids `exp(-λ)` underflow).
    fn poisson(&mut self, lambda: f64) -> u64 {
        const POISSON_CHUNK: f64 = 30.0;
        let mut rest = lambda;
        let mut k = 0u64;
        while rest > POISSON_CHUNK {
            k += self.poisson(POISSON_CHUNK);
            rest -= POISSON_CHUNK;
        }
        let limit = (-rest).exp();
        let mut prod = self.uniform();
        while prod > limit {
            k += 1;
            prod *= self.uniform();
        }
        k
    }

    /// Standard normal by Box–Muller (cosine branch).
    fn normal(&mut self) -> f64 {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        r * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

fn ln_factorial(k: u64) -> f64 {
    (1..=k).map(|i| (i as f64).ln()).sum()
}

// ---------------------------------------------------------------------------
// Finding 1: true zero-truncated Poisson count part
// ---------------------------------------------------------------------------

#[test]
fn ztp_score_equations_and_simulated_recovery() {
    let mut rng = SplitMix64(SEED);
    let n = 4_000usize;
    let (b0, b1) = (0.3, 0.5);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    while y.len() < n {
        let xi = 2.0 * rng.uniform() - 1.0;
        let lam = (b0 + b1 * xi).exp();
        let k = rng.poisson(lam);
        if k > 0 {
            x.push(xi);
            y.push(k as f64);
        }
    }
    let fit = fit_regularized_glm(
        &x,
        n,
        1,
        &y,
        None,
        None,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::ZeroTruncatedPoisson,
            ..RegularizedGlmOptions::default()
        },
    )
    .unwrap();
    assert!(fit.converged);
    // Intercept score equation: Σ E[Y|Y>0] = Σ y.
    let sum_y: f64 = y.iter().sum();
    let sum_m: f64 = fit.fitted_means.iter().sum();
    assert!(
        ((sum_m - sum_y) / sum_y).abs() < 1e-8,
        "sum m={sum_m} sum y={sum_y}"
    );
    // Slope score equation: Σ x (y - m) = 0.
    let slope_score: f64 = (0..n).map(|i| x[i] * (y[i] - fit.fitted_means[i])).sum();
    assert!(
        slope_score.abs() < 1e-6 * sum_y,
        "slope score={slope_score}"
    );
    // fitted_means really are λ/(1-e^{-λ}).
    let lam0 = (fit.intercept + fit.coefficients[0] * x[0]).exp();
    assert!((fit.fitted_means[0] - lam0 / (1.0 - (-lam0).exp())).abs() < 1e-12);
    // Recovery (SE ≈ 0.03 at n = 4000).
    assert!((fit.intercept - b0).abs() < 0.1, "b0={}", fit.intercept);
    assert!(
        (fit.coefficients[0] - b1).abs() < 0.1,
        "b1={}",
        fit.coefficients[0]
    );
}

#[test]
fn hurdle_count_part_calibrates_to_positive_counts() {
    // 20 rows: 5 zeros and 15 positives (mostly ones); the old Poisson-then-ZTP-factor
    // double-corrected and overstated the positive total.
    let x: Vec<f64> = (0..20).map(|i| f64::from(i % 2)).collect();
    let mut y = vec![1.0; 20];
    for i in [0usize, 3, 6, 9, 12] {
        y[i] = 0.0;
    }
    y[1] = 2.0;
    y[4] = 3.0;
    y[7] = 2.0;
    let h = fit_hurdle_poisson(&x, 20, 1, &y, None, None, 0.0, 0.0).unwrap();
    assert!(h.converged);
    let pos_total: f64 = y.iter().filter(|&&v| v > 0.0).sum();
    let ztp_total: f64 = h.count_part.fitted_means.iter().sum();
    assert!(
        (ztp_total - pos_total).abs() < 1e-8 * pos_total,
        "ztp total={ztp_total} observed={pos_total}"
    );
    // Unpenalized logistic + ZTP with saturated group design: E[Y] reproduces group totals.
    let fitted_total: f64 = h.fitted_means.iter().sum();
    let total: f64 = y.iter().sum();
    assert!(
        (fitted_total - total).abs() < 1e-6 * total,
        "{fitted_total} vs {total}"
    );
}

// ---------------------------------------------------------------------------
// Finding 6: hurdle / ZIP / GLMM input lengths validated before indexing
// ---------------------------------------------------------------------------

#[test]
fn zero_modified_models_reject_bad_lengths_without_panicking() {
    let x: Vec<f64> = (0..20).map(|i| f64::from(i % 2)).collect();
    let y: Vec<f64> = (0..20).map(|i| f64::from(i % 3)).collect();
    let short = [0.0; 3];
    assert!(matches!(
        fit_hurdle_poisson(&x, 20, 1, &y, None, Some(&short), 0.0, 0.0),
        Err(PricingError::DimensionMismatch(_))
    ));
    assert!(matches!(
        fit_hurdle_poisson(&x, 20, 1, &y, Some(&short), None, 0.0, 0.0),
        Err(PricingError::DimensionMismatch(_))
    ));
    assert!(matches!(
        fit_zero_inflated_poisson(&x, 20, 1, &y, None, Some(&short), 0.0, 0.0, 10),
        Err(PricingError::DimensionMismatch(_))
    ));
    let groups: Vec<usize> = (0..20).map(|i| i % 4).collect();
    assert!(matches!(
        fit_random_intercept_glmm(
            &x,
            20,
            1,
            &y,
            &groups,
            4,
            Some(&short),
            None,
            &SolverOptions::default().with_max_iter(5)
        ),
        Err(PricingError::DimensionMismatch(_))
    ));
}

// ---------------------------------------------------------------------------
// Finding 5: convergence propagated for ZIP / GLMM
// ---------------------------------------------------------------------------

#[test]
fn zip_and_glmm_report_convergence() {
    let mut rng = SplitMix64(SEED);
    let n = 400usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for i in 0..n {
        let xi = f64::from(u8::from(rng.uniform() < 0.5));
        x.push(xi);
        let structural_zero = rng.uniform() < 0.3;
        let k = if structural_zero {
            0
        } else {
            rng.poisson((0.5 + 0.4 * xi).exp())
        };
        y.push(k as f64);
        groups.push(i % 8);
    }
    let zip = fit_zero_inflated_poisson(&x, n, 1, &y, None, None, 0.0, 0.0, 500).unwrap();
    assert!(zip.converged, "ZIP iterations={}", zip.iterations);
    let zip_one = fit_zero_inflated_poisson(&x, n, 1, &y, None, None, 0.0, 0.0, 1).unwrap();
    assert!(!zip_one.converged);

    let glmm = fit_random_intercept_glmm(
        &x,
        n,
        1,
        &y,
        &groups,
        8,
        None,
        None,
        &SolverOptions::default().with_max_iter(50),
    )
    .unwrap();
    assert!(glmm.converged, "GLMM iterations={}", glmm.iterations);
}

// ---------------------------------------------------------------------------
// Findings 4 & 8: no ±20 eta clamp; options.tolerance drives convergence
// ---------------------------------------------------------------------------

#[test]
fn gamma_large_scale_response_is_recovered() {
    let x: Vec<f64> = (0..10).map(|i| f64::from(i % 2)).collect();
    let y: Vec<f64> = (0..10)
        .map(|i| 2.0e9 * (1.0 + 0.01 * f64::from(i)) * if i % 2 == 1 { 1.5 } else { 1.0 })
        .collect();
    let fit = fit_regularized_glm(
        &x,
        10,
        1,
        &y,
        None,
        None,
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::Gamma,
            ..RegularizedGlmOptions::default()
        },
    )
    .unwrap();
    assert!(fit.converged);
    for g in 0..2 {
        let idx: Vec<usize> = (0..10).filter(|i| i % 2 == g).collect();
        let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
        for &i in &idx {
            let rel = (fit.fitted_means[i] - mean).abs() / mean;
            assert!(rel < 1e-6, "group {g} rel err {rel}");
        }
    }
    // Prediction beyond the overflow-safe bound is a typed error, not a silent clamp.
    assert!(matches!(
        fit.predict(RegularizedGlmFamily::Gamma, &[1.0e6], 1, 1, None),
        Err(PricingError::OutOfBounds(_))
    ));
}

#[test]
fn regularized_outer_convergence_honours_tolerance() {
    let x: Vec<f64> = (0..30).map(|i| f64::from(i) / 10.0 - 1.5).collect();
    let y: Vec<f64> = x.iter().map(|v| (0.2 + 0.7 * v).exp().round()).collect();
    let run = |tolerance: f64| {
        fit_regularized_glm(
            &x,
            30,
            1,
            &y,
            None,
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                tolerance,
                ..RegularizedGlmOptions::default()
            },
        )
        .unwrap()
    };
    let loose = run(1e-2);
    let tight = run(1e-12);
    assert!(loose.converged && tight.converged);
    assert!(
        loose.iterations < tight.iterations,
        "loose={} tight={}",
        loose.iterations,
        tight.iterations
    );
}

// ---------------------------------------------------------------------------
// Finding 2: Dunn & Smyth series density and profile-likelihood power selection
// ---------------------------------------------------------------------------

/// Independent compound Poisson–Gamma convolution density (Poisson-weighted Gamma pdfs).
fn cpg_density(y: f64, mu: f64, phi: f64, p: f64) -> f64 {
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let shape = (2.0 - p) / (p - 1.0);
    let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let mut f = 0.0;
    for j in 1..400u64 {
        let jf = j as f64;
        let a = jf * shape;
        let ln_pois = -lambda + jf * lambda.ln() - ln_factorial(j);
        let ln_gamma_pdf = (a - 1.0) * y.ln() - y / scale - a * scale.ln() - ln_gamma_ref(a);
        f += (ln_pois + ln_gamma_pdf).exp();
    }
    f
}

/// Reference `ln Γ(a)` (shift to a >= 10, then Stirling series).
fn ln_gamma_ref(mut a: f64) -> f64 {
    let mut shift = 0.0;
    while a < 10.0 {
        shift -= a.ln();
        a += 1.0;
    }
    let inv = 1.0 / a;
    let inv2 = inv * inv;
    shift
        + (a - 0.5) * a.ln()
        + 0.5 * (2.0 * std::f64::consts::PI).ln()
        + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
        - a
}

#[test]
fn tweedie_series_density_matches_convolution_and_integrates_to_one() {
    for &(mu, phi, p) in &[(2.0, 1.0, 1.5), (5.0, 2.0, 1.3), (1.0, 0.5, 1.7)] {
        for &y in &[0.05, 0.7, 2.0, 6.5, 15.0] {
            let series = tweedie_log_density(y, mu, phi, p).unwrap().exp();
            let exact = cpg_density(y, mu, phi, p);
            assert!(
                ((series - exact) / exact).abs() < 1e-9,
                "mu={mu} phi={phi} p={p} y={y}: series={series} exact={exact}"
            );
        }
    }
    // Total mass: P(Y=0) + ∫ f(y) dy (composite Simpson; p = 1.5 has finite f(0+)).
    let (mu, phi, p) = (2.0, 1.0, 1.5);
    let p0 = tweedie_log_density(0.0, mu, phi, p).unwrap().exp();
    let upper = 80.0;
    let steps = 20_000usize;
    let h = upper / steps as f64;
    let f = |y: f64| {
        if y == 0.0 {
            // Right limit at 0 for shape 1: λ e^{-λ} / scale.
            let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
            let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
            lambda * (-lambda).exp() / scale
        } else {
            tweedie_log_density(y, mu, phi, p).unwrap().exp()
        }
    };
    let mut s = f(0.0) + f(upper);
    for k in 1..steps {
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        s += w * f(k as f64 * h);
    }
    let mass = p0 + s * h / 3.0;
    assert!((mass - 1.0).abs() < 1e-8, "total mass={mass}");
}

#[test]
fn tweedie_power_selection_recovers_simulated_p() {
    let mut rng = SplitMix64(SEED);
    let n = 1_500usize;
    let (p_true, phi_true) = (1.5_f64, 2.0_f64);
    let mut design = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.uniform();
        design.push(1.0);
        design.push(xi);
        let mu = (1.0 + 0.5 * xi).exp();
        let lambda = mu.powf(2.0 - p_true) / (phi_true * (2.0 - p_true));
        // p = 1.5 -> Gamma shape (2-p)/(p-1) = 1 (exponential severities).
        let scale = phi_true * (p_true - 1.0) * mu.powf(p_true - 1.0);
        let k = rng.poisson(lambda);
        let total: f64 = (0..k).map(|_| rng.exponential() * scale).sum();
        y.push(total);
    }
    let w = vec![1.0; n];
    let off = vec![0.0; n];
    let grid: Vec<f64> = (0..17).map(|i| 1.1 + 0.05 * f64::from(i)).collect();
    let sel =
        select_tweedie_power(&grid, &design, n, 2, &y, &w, &off, &GlmOptions::default()).unwrap();
    assert!(
        (sel.best_power - p_true).abs() <= 0.1 + 1e-12,
        "best p={} profile={:?}",
        sel.best_power,
        sel.profile
    );
    assert!(
        (sel.best_dispersion / phi_true - 1.0).abs() < 0.25,
        "phi={}",
        sel.best_dispersion
    );
    // Interior optimum (not the boundary candidate the saddlepoint picked).
    assert!(sel.best_power > grid[0] && sel.best_power < grid[grid.len() - 1]);
}

// ---------------------------------------------------------------------------
// Finding 3: AIC/BIC from the full log-likelihood
// ---------------------------------------------------------------------------

#[test]
fn aic_matches_hand_computed_poisson_log_likelihood() {
    let y = [0.0, 1.0, 3.0];
    let mu = [0.5, 1.2, 2.5];
    let w = [1.0; 3];
    let ll: f64 = (0..3)
        .map(|i| y[i] * f64::ln(mu[i]) - mu[i] - ln_factorial(y[i] as u64))
        .sum();
    let ic = glm_information_criteria(GlmFamily::Poisson, &y, &mu, &w, 1.0, 1).unwrap();
    assert!((ic.log_likelihood - ll).abs() < 1e-12);
    assert!((ic.aic - (-2.0 * ll + 2.0)).abs() < 1e-12);
    assert!((ic.bic - (-2.0 * ll + 3.0_f64.ln())).abs() < 1e-12);
    assert!(matches!(
        glm_information_criteria(GlmFamily::QuasiPoisson, &y, &mu, &w, 1.0, 1),
        Err(PricingError::InvalidParameter(_))
    ));
}

#[test]
fn aic_prefers_true_gamma_model_over_intercept_only() {
    let mut rng = SplitMix64(SEED);
    let n = 200usize;
    let shape = 2.0; // phi = 0.5
    let mut x_full = Vec::with_capacity(2 * n);
    let mut x_null = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 2.0 * rng.uniform();
        x_full.push(1.0);
        x_full.push(xi);
        x_null.push(1.0);
        let mu = (1.0 + 0.8 * xi).exp();
        // Gamma(shape = 2, scale = mu / 2) as a sum of two exponentials.
        y.push((rng.exponential() + rng.exponential()) * mu / shape);
    }
    let w = vec![1.0; n];
    let off = vec![0.0; n];
    let opts = GlmOptions::default();
    let fit_full = fit_glm_irls(GlmFamily::Gamma, &x_full, n, 2, &y, &w, &off, &opts).unwrap();
    let fit_null = fit_glm_irls(GlmFamily::Gamma, &x_null, n, 1, &y, &w, &off, &opts).unwrap();
    let mu_full = predict_glm(
        GlmFamily::Gamma,
        &x_full,
        n,
        2,
        &fit_full.coefficients,
        &off,
    )
    .unwrap();
    let mu_null = predict_glm(
        GlmFamily::Gamma,
        &x_null,
        n,
        1,
        &fit_null.coefficients,
        &off,
    )
    .unwrap();
    let ic_full =
        glm_information_criteria(GlmFamily::Gamma, &y, &mu_full, &w, fit_full.dispersion, 2)
            .unwrap();
    let ic_null =
        glm_information_criteria(GlmFamily::Gamma, &y, &mu_null, &w, fit_null.dispersion, 1)
            .unwrap();
    assert!(
        ic_full.aic < ic_null.aic,
        "full={} null={}",
        ic_full.aic,
        ic_null.aic
    );
    assert!(ic_full.bic < ic_null.bic);
    // ML dispersion is a stationary point of the Gamma log-likelihood in phi.
    let ll_at = |phi: f64| -> f64 {
        let a = 1.0 / phi;
        (0..n)
            .map(|i| {
                let r = y[i] / mu_full[i];
                a * (a * r).ln() - a * r - y[i].ln() - ln_gamma_ref(a)
            })
            .sum()
    };
    let phi = ic_full.ml_dispersion;
    assert!((ll_at(phi) - ic_full.log_likelihood).abs() < 1e-8);
    let h = 1e-5 * phi;
    let deriv = (ll_at(phi + h) - ll_at(phi - h)) / (2.0 * h);
    assert!(deriv.abs() < 1e-3, "dl/dphi={deriv}");
    assert!((phi / 0.5 - 1.0).abs() < 0.3, "phi_ml={phi}");

    // Tweedie AIC uses the series density with profiled phi.
    let ic_tw =
        glm_information_criteria(GlmFamily::Tweedie { power: 1.5 }, &y, &mu_full, &w, 1.0, 2)
            .unwrap();
    assert!(ic_tw.aic.is_finite() && ic_tw.ml_dispersion > 0.0);
}

// ---------------------------------------------------------------------------
// Finding 7: B-spline right endpoint and valid domain
// ---------------------------------------------------------------------------

#[test]
fn bspline_partition_of_unity_and_domain_for_clamped_and_unclamped_knots() {
    let degree = 3;
    let unclamped: Vec<f64> = (0..=10).map(f64::from).collect(); // domain [3, 7]
    let pts = [3.0, 3.5, 4.0, 5.25, 6.999_999, 7.0];
    let (nb, mat) = bspline_basis(&pts, &unclamped, degree).unwrap();
    for r in 0..pts.len() {
        let sum: f64 = mat[r * nb..(r + 1) * nb].iter().sum();
        assert!((sum - 1.0).abs() < 1e-13, "x={} sum={sum}", pts[r]);
    }
    // Right-endpoint value equals the left limit (continuity, no forced 1.0).
    let (_, left) = bspline_basis(&[7.0 - 1e-10], &unclamped, degree).unwrap();
    let (_, at) = bspline_basis(&[7.0], &unclamped, degree).unwrap();
    for j in 0..nb {
        assert!(
            (left[j] - at[j]).abs() < 1e-8,
            "basis {j}: {} vs {}",
            left[j],
            at[j]
        );
    }
    // Uniform cubic: last basis (support [6, 10]) equals 1/6 at x = 7, not a forced 1.0.
    assert!(
        (at[nb - 1] - 1.0 / 6.0).abs() < 1e-12,
        "last basis={}",
        at[nb - 1]
    );
    for bad in [2.9, 7.1, 0.0, 10.0] {
        assert!(matches!(
            bspline_basis(&[bad], &unclamped, degree),
            Err(PricingError::OutOfBounds(_))
        ));
    }

    // Clamped: last basis is exactly 1 at the right endpoint.
    let clamped = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let (nc, cm) = bspline_basis(&[0.0, 0.3, 1.0], &clamped, degree).unwrap();
    assert!((cm[2 * nc + nc - 1] - 1.0).abs() < 1e-15);
    assert!((cm[0] - 1.0).abs() < 1e-15);
    for r in 0..3 {
        let sum: f64 = cm[r * nc..(r + 1) * nc].iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }
}

// ---------------------------------------------------------------------------
// Finding 9: validated scoring
// ---------------------------------------------------------------------------

#[test]
fn multiplicative_rater_try_score_validates_inputs() {
    let rater = MultiplicativeRater::try_new(100.0).unwrap();
    assert_eq!(
        rater.try_score(2.0, &[1.2, 0.9]).unwrap(),
        rater.score(2.0, &[1.2, 0.9])
    );
    assert!(rater.try_score(f64::NAN, &[1.0]).is_err());
    assert!(rater.try_score(1.0, &[f64::NAN]).is_err());
    assert!(rater.try_score(-1.0, &[1.0]).is_err());
    assert!(MultiplicativeRater::try_new(f64::NAN).is_err());
    let inverted = MultiplicativeRater::new(100.0)
        .with_minimum_premium(500.0)
        .with_maximum_premium(200.0);
    assert!(matches!(
        inverted.try_score(1.0, &[1.0]),
        Err(PricingError::InvalidParameter(_))
    ));
    let mut out = [0.0; 1];
    assert!(
        inverted
            .batch_score_into(&[1.0], &[&[1.0]], &mut out)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Round 2, finding 1: hurdle zero part is exposure-aware (cloglog + ln E offset)
// ---------------------------------------------------------------------------

#[test]
fn hurdle_zero_part_scales_with_exposure() {
    let mut rng = SplitMix64(SEED);
    let n = 4_000usize;
    let rate = 0.5;
    let (e_lo, e_hi) = (0.25, 4.0);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n);
    for i in 0..n {
        let e = if i % 2 == 0 { e_lo } else { e_hi };
        y.push(rng.poisson(rate * e) as f64);
        x.push(f64::from(u8::from((i / 2) % 2 == 1)));
        off.push(f64::ln(e));
    }
    let h = fit_hurdle_poisson(&x, n, 1, &y, None, Some(&off), 0.0, 0.0).unwrap();
    assert!(h.converged);
    for (parity, e) in [(0usize, e_lo), (1, e_hi)] {
        let fitted: f64 = (parity..n).step_by(2).map(|i| h.fitted_means[i]).sum();
        let actual: f64 = (parity..n).step_by(2).map(|i| y[i]).sum();
        let expected = rate * e * (n / 2) as f64;
        // Poisson total: sd = sqrt(expected); allow 3 sd vs truth and vs observed.
        let tol = 3.0 * expected.sqrt();
        assert!(
            (fitted - expected).abs() < tol && (fitted - actual).abs() < tol,
            "E={e}: fitted={fitted} actual={actual} expected={expected}"
        );
    }
    // P(Y > 0) is monotone in exposure and matches 1 - exp(-rate E) (binomial SE <= 0.011).
    for i in 0..4 {
        let (lo, hi) = (
            h.zero_part.fitted_means[2 * i],
            h.zero_part.fitted_means[2 * i + 1],
        );
        assert!(hi > lo, "P(Y>0) not monotone in exposure: {lo} vs {hi}");
        assert!((lo - (1.0 - (-rate * e_lo).exp())).abs() < 0.035, "lo={lo}");
        assert!((hi - (1.0 - (-rate * e_hi).exp())).abs() < 0.035, "hi={hi}");
    }
}

#[test]
fn cloglog_one_way_matches_closed_form_mle() {
    // Single cell, common exposure E: MLE solves 1 - exp(-E e^b) = p_hat.
    let e: f64 = 2.0;
    let y = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let n = y.len();
    let p_hat = y.iter().sum::<f64>() / n as f64;
    let x = vec![0.0; n];
    let off = vec![e.ln(); n];
    let fit = fit_regularized_glm(
        &x,
        n,
        1,
        &y,
        None,
        Some(&off),
        RegularizedGlmOptions {
            family: RegularizedGlmFamily::BinomialCloglog,
            ..RegularizedGlmOptions::default()
        },
    )
    .unwrap();
    assert!(fit.converged);
    let b_ref = (-(1.0 - p_hat).ln()).ln() - e.ln();
    assert!(
        (fit.intercept - b_ref).abs() < 1e-8,
        "{} vs {b_ref}",
        fit.intercept
    );
}

// ---------------------------------------------------------------------------
// Round 2, finding 2: GLMM joint mode, BLUP constraint, τ² recovery
// ---------------------------------------------------------------------------

#[test]
fn glmm_joint_mode_recovers_variance_and_intercept() {
    let (n_groups, per, b0, b1) = (60usize, 40usize, -1.0, 0.5);
    for tau2 in [1.0_f64, 4.0, 9.0] {
        let mut rng = SplitMix64(SEED);
        let n = n_groups * per;
        let u: Vec<f64> = (0..n_groups).map(|_| tau2.sqrt() * rng.normal()).collect();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);
        for i in 0..n {
            let g = i / per;
            let xi = rng.uniform();
            x.push(xi);
            groups.push(g);
            y.push(rng.poisson((b0 + b1 * xi + u[g]).exp()) as f64);
        }
        let f = fit_random_intercept_glmm(
            &x,
            n,
            1,
            &y,
            &groups,
            n_groups,
            None,
            None,
            &SolverOptions::default().with_max_iter(100),
        )
        .unwrap();
        assert!(f.converged, "tau2={tau2}: {f:?}");
        assert!(!f.tau_squared_at_upper_bound && !f.tau_squared_at_lower_bound);
        // BLUP constraint at the joint mode (intercept score equation).
        let sum_u: f64 = f.random_intercepts.iter().sum();
        assert!(sum_u.abs() < 1e-6, "tau2={tau2}: sum u_hat={sum_u}");
        // τ̂² vs the realized variance of the true effects (removes most MC noise) and vs τ².
        let u_bar = u.iter().sum::<f64>() / n_groups as f64;
        let s2 = u.iter().map(|v| (v - u_bar).powi(2)).sum::<f64>() / (n_groups - 1) as f64;
        assert!(
            (f.tau_squared / s2 - 1.0).abs() < 0.35,
            "tau2={tau2}: tau2_hat={} realized={s2}",
            f.tau_squared
        );
        // sd(τ̂²) ≈ τ² sqrt(2 / G) ≈ 0.18 τ²: allow ~2.5 sd.
        assert!(
            (f.tau_squared / tau2 - 1.0).abs() < 0.45,
            "tau2={tau2}: tau2_hat={}",
            f.tau_squared
        );
        // Intercept identifies b0 + mean(u) (Σû = 0); groups with zero claims shrink, so
        // allow a fraction of τ. Was 2.51 vs ≈ -1.5 under alternating updates at τ² = 9.
        let tau = tau2.sqrt();
        assert!(
            (f.intercept - (b0 + u_bar)).abs() < 0.15 * tau,
            "tau2={tau2}: b0_hat={} b0+u_bar={}",
            f.intercept,
            b0 + u_bar
        );
        assert!(
            (f.fixed_effects[0] - b1).abs() < 0.12,
            "slope={}",
            f.fixed_effects[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Round 2, finding 3: deviance-based IRLS convergence + separation flag
// ---------------------------------------------------------------------------

/// 3-level one-way design (intercept + 2 dummies); level 2 has no claims.
fn zero_claim_design() -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..30u32 {
        let l = i % 3;
        x.extend_from_slice(&[
            1.0,
            f64::from(u8::from(l == 1)),
            f64::from(u8::from(l == 2)),
        ]);
        y.push(if l == 2 { 0.0 } else { f64::from(1 + i % 4) });
    }
    (x, y)
}

#[test]
fn irls_zero_claim_level_converges_with_separation_flag() {
    let (x, y) = zero_claim_design();
    let w = vec![1.0; 30];
    let off = vec![0.0; 30];
    let fit = fit_glm_irls(
        GlmFamily::Poisson,
        &x,
        30,
        3,
        &y,
        &w,
        &off,
        &GlmOptions::default(),
    )
    .unwrap();
    assert!(fit.separation_suspected);
    assert!(fit.coefficients[2] < -10.0, "beta2={}", fit.coefficients[2]);
    // Closed form for the finite levels: saturated one-way relativity = observed ratio.
    let mean = |lvl: usize| {
        let v: Vec<f64> = (0..30).filter(|i| i % 3 == lvl).map(|i| y[i]).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    assert!((fit.coefficients[0] - mean(0).ln()).abs() < 1e-8);
    assert!((fit.coefficients[1] - (mean(1) / mean(0)).ln()).abs() < 1e-8);

    // Separated Binomial (level 2 all zeros) likewise returns Ok with the flag.
    let yb: Vec<f64> = y.iter().map(|&v| f64::from(u8::from(v > 2.0))).collect();
    let fb = fit_glm_irls(
        GlmFamily::Binomial,
        &x,
        30,
        3,
        &yb,
        &w,
        &off,
        &GlmOptions::default().with_max_iter(2000),
    )
    .unwrap();
    assert!(fb.separation_suspected);
    assert!(fb.coefficients[2] < -10.0, "beta2={}", fb.coefficients[2]);

    // Complete separation on a continuous covariate.
    let xs: Vec<f64> = (0..20).flat_map(|i| [1.0, f64::from(i) / 19.0]).collect();
    let ys: Vec<f64> = (0..20).map(|i| f64::from(u8::from(i >= 10))).collect();
    let fs = fit_glm_irls(
        GlmFamily::Binomial,
        &xs,
        20,
        2,
        &ys,
        &[1.0; 20],
        &[0.0; 20],
        &GlmOptions::default(),
    )
    .unwrap();
    assert!(
        fs.separation_suspected && fs.coefficients[1] > 20.0,
        "{fs:?}"
    );
}

#[test]
fn saturated_one_way_poisson_and_gamma_match_closed_form_mle() {
    // Poisson with exposure offsets: exp(beta_k) = (C_k / E_k) / (C_0 / E_0).
    let claims = [12.0, 30.0, 7.0, 3.0, 25.0, 9.0];
    let exposure = [10.0, 20.0, 4.0, 5.0, 15.0, 6.0];
    let level = [0usize, 1, 2, 0, 1, 2];
    let mut x = Vec::new();
    for &l in &level {
        x.extend_from_slice(&[
            1.0,
            f64::from(u8::from(l == 1)),
            f64::from(u8::from(l == 2)),
        ]);
    }
    let off: Vec<f64> = exposure.iter().map(|e: &f64| e.ln()).collect();
    let w = [1.0; 6];
    let fit = fit_glm_irls(
        GlmFamily::Poisson,
        &x,
        6,
        3,
        &claims,
        &w,
        &off,
        &GlmOptions::default(),
    )
    .unwrap();
    assert!(!fit.separation_suspected);
    let freq = |l: usize| {
        let (c, e) = (0..6)
            .filter(|&i| level[i] == l)
            .fold((0.0, 0.0), |(c, e), i| (c + claims[i], e + exposure[i]));
        c / e
    };
    assert!((fit.coefficients[0] - freq(0).ln()).abs() < 1e-10);
    for l in 1..3 {
        let r = fit.coefficients[l].exp();
        assert!((r - freq(l) / freq(0)).abs() < 1e-10, "level {l}: {r}");
    }

    // Gamma (log link) with claim-count weights: fitted level mean = weighted mean severity.
    let sev = [1200.0, 900.0, 3000.0, 1500.0, 1100.0, 2600.0];
    let cnt = [3.0, 5.0, 2.0, 4.0, 1.0, 6.0];
    let g = fit_glm_irls(
        GlmFamily::Gamma,
        &x,
        6,
        3,
        &sev,
        &cnt,
        &[0.0; 6],
        &GlmOptions::default(),
    )
    .unwrap();
    let wmean = |l: usize| {
        let (s, c) = (0..6)
            .filter(|&i| level[i] == l)
            .fold((0.0, 0.0), |(s, c), i| (s + cnt[i] * sev[i], c + cnt[i]));
        s / c
    };
    assert!((g.coefficients[0] - wmean(0).ln()).abs() < 1e-9);
    for l in 1..3 {
        assert!(
            (g.coefficients[l] - (wmean(l) / wmean(0)).ln()).abs() < 1e-9,
            "level {l}"
        );
    }
}

// ---------------------------------------------------------------------------
// Round 2, findings 4 & 5: Tweedie grid skips failures; double-lift bucket count
// ---------------------------------------------------------------------------

#[test]
fn tweedie_power_grid_skips_failed_candidates() {
    let n = 200usize;
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.extend_from_slice(&[1.0, (i % 5) as f64]);
        y.push(if i % 3 == 0 {
            0.0
        } else {
            (1 + i % 7) as f64 * 10.0
        });
    }
    let (w, off) = (vec![1.0; n], vec![0.0; n]);
    let sel =
        select_tweedie_power(&[1.5, 2.0], &x, n, 2, &y, &w, &off, &GlmOptions::default()).unwrap();
    assert!((sel.best_power - 1.5).abs() < 1e-15);
    assert_eq!(sel.profile.len(), 1);
    assert_eq!(sel.failed_powers.len(), 1);
    assert!((sel.failed_powers[0].0 - 2.0).abs() < 1e-15);
    assert!(matches!(
        select_tweedie_power(&[2.0, 0.5], &x, n, 2, &y, &w, &off, &GlmOptions::default()),
        Err(PricingError::NonConvergence(_))
    ));
}

#[test]
fn double_lift_rejects_too_many_buckets_and_never_returns_empty_ones() {
    let r = double_lift_chart(&[1.0; 3], &[1.0, 2.0, 3.0], &[1.0; 3], &[0.5, 1.0, 2.0], 5);
    assert!(matches!(r, Err(PricingError::InvalidParameter(_))));
    // Exposure-skewed data: the greedy fill used to leave the last bucket empty.
    let exposures = [1.0, 1.0, 1.0, 1.0, 100.0];
    let pred_b = [0.5, 0.6, 0.7, 0.8, 0.9];
    let d = double_lift_chart(&exposures, &[1.0; 5], &[1.0; 5], &pred_b, 3).unwrap();
    assert_eq!(d.len(), 3);
    assert_eq!(
        d.iter().map(|b| b.bucket).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let total: f64 = d.iter().map(|b| b.exposure).sum();
    assert!((total - 104.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Round 3: scale-free separation flag, scale-invariant QR rank test, GLMM intercept column
// ---------------------------------------------------------------------------

/// Covariate rescaling factors spanning the unit-dependence that broke the absolute rules.
const COVARIATE_SCALES: [f64; 3] = [1e-7, 1.0, 1e7];

/// Well-posed Poisson design `[1, x1, x2]` with a smooth continuous `x1`; `x2` is a dummy.
fn well_posed_poisson(scale_x1: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64(SEED);
    let n = 60usize;
    let mut x = Vec::with_capacity(3 * n);
    let mut y = Vec::with_capacity(n);
    let mut off = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f64) / (n as f64);
        let x2 = f64::from(u8::from(i % 3 == 0));
        x.extend_from_slice(&[1.0, x1 * scale_x1, x2]);
        let exposure = 1.0 + rng.uniform();
        off.push(exposure.ln());
        y.push(rng.poisson(exposure * (0.4 + 0.8 * x1 - 0.3 * x2).exp()) as f64);
    }
    (x, y, off)
}

#[test]
fn separation_flag_is_invariant_to_covariate_scale() {
    let opts = GlmOptions::default();
    for &c in &COVARIATE_SCALES {
        // Well-posed fit: never flagged, at any covariate scale.
        let (x, y, off) = well_posed_poisson(c);
        let n = y.len();
        let fit =
            fit_glm_irls(GlmFamily::Poisson, &x, n, 3, &y, &vec![1.0; n], &off, &opts).unwrap();
        assert!(!fit.separation_suspected, "scale {c}: {fit:?}");

        // Zero-claim level (dummy column rescaled by c): flagged at every scale.
        let (mut xz, yz) = zero_claim_design();
        for row in xz.chunks_mut(3) {
            row[2] *= c;
        }
        let w = vec![1.0; 30];
        let off0 = vec![0.0; 30];
        let fz = fit_glm_irls(GlmFamily::Poisson, &xz, 30, 3, &yz, &w, &off0, &opts).unwrap();
        assert!(fz.separation_suspected, "zero-claim, scale {c}");

        // Complete separation of a Binomial on a rescaled continuous covariate.
        let xs: Vec<f64> = (0..20)
            .flat_map(|i| [1.0, c * f64::from(i) / 19.0])
            .collect();
        let ys: Vec<f64> = (0..20).map(|i| f64::from(u8::from(i >= 10))).collect();
        let fs = fit_glm_irls(
            GlmFamily::Binomial,
            &xs,
            20,
            2,
            &ys,
            &[1.0; 20],
            &[0.0; 20],
            &opts,
        )
        .unwrap();
        assert!(fs.separation_suspected, "separated binomial, scale {c}");
    }
}

#[test]
fn qr_rank_test_is_invariant_to_column_scale() {
    let opts = GlmOptions::default().with_tol(1e-14).with_max_iter(100);
    let (x, y, off) = well_posed_poisson(1.0);
    let n = y.len();
    let w = vec![1.0; n];
    let base = fit_glm_irls(GlmFamily::Poisson, &x, n, 3, &y, &w, &off, &opts).unwrap();
    let mu_base = predict_glm(GlmFamily::Poisson, &x, n, 3, &base.coefficients, &off).unwrap();
    for c in [1e12, 1e-13] {
        let mut xs = x.clone();
        for row in xs.chunks_mut(3) {
            row[1] *= c;
        }
        let fit = fit_glm_irls(GlmFamily::Poisson, &xs, n, 3, &y, &w, &off, &opts)
            .unwrap_or_else(|e| panic!("scale {c}: {e:?}"));
        // Coefficient of the rescaled column scales exactly inversely; the rest are unchanged.
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        assert!(
            rel(fit.coefficients[1] * c, base.coefficients[1]) < 1e-9,
            "scale {c}"
        );
        assert!(rel(fit.coefficients[0], base.coefficients[0]) < 1e-9);
        assert!(rel(fit.coefficients[2], base.coefficients[2]) < 1e-9);
        assert!(rel(fit.standard_errors[1] * c, base.standard_errors[1]) < 1e-8);
        let mu = predict_glm(GlmFamily::Poisson, &xs, n, 3, &fit.coefficients, &off).unwrap();
        for (a, b) in mu.iter().zip(&mu_base) {
            assert!(rel(*a, *b) < 1e-10, "scale {c}: {a} vs {b}");
        }

        // A genuinely collinear design (x3 = scaled x1 + x2) is still rejected.
        let xc: Vec<f64> = xs
            .chunks(3)
            .flat_map(|r| [r[0], r[1], r[2], r[1] + c * r[2]])
            .collect();
        assert!(matches!(
            fit_glm_irls(GlmFamily::Poisson, &xc, n, 4, &y, &w, &off, &opts),
            Err(PricingError::SingularMatrix(_))
        ));
    }
}

#[test]
fn design_matrix_glmm_drops_explicit_intercept_column_like_fit_regularized() {
    let mut rng = SplitMix64(SEED);
    let n = 240usize;
    let n_groups = 6usize;
    let mut x = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    let effects = [-0.4, -0.1, 0.0, 0.1, 0.25, 0.4];
    for i in 0..n {
        let xi = f64::from(u8::from(rng.uniform() < 0.5));
        let g = i % n_groups;
        x.push(xi);
        rows.push([1.0, xi]);
        y.push(rng.poisson((0.3 + 0.5 * xi + effects[g]).exp()) as f64);
        groups.push(g);
    }
    let opts = SolverOptions::default().with_max_iter(50);
    let direct =
        fit_random_intercept_glmm(&x, n, 1, &y, &groups, n_groups, None, None, &opts).unwrap();
    // DesignMatrix carries the explicit intercept column (GlmFit convention).
    let dm = DesignMatrix::try_from_rows(&rows).unwrap();
    let via_dm = dm
        .fit_random_intercept_glmm(&y, &groups, n_groups, None, None, &opts)
        .unwrap();
    assert_eq!(via_dm.fixed_effects.len(), 2);
    assert_eq!(via_dm.fixed_effects[0], 0.0);
    assert!((via_dm.fixed_effects[1] - direct.fixed_effects[0]).abs() < 1e-10);
    assert!((via_dm.intercept - direct.intercept).abs() < 1e-10);
    assert!((via_dm.tau_squared - direct.tau_squared).abs() < 1e-10 * direct.tau_squared);
    assert!(via_dm.converged && direct.converged);

    // Same convention as DesignMatrix::fit_regularized (constant column -> coefficient 0).
    let reg = dm
        .fit_regularized(
            &y,
            None,
            None,
            RegularizedGlmOptions {
                family: RegularizedGlmFamily::Poisson,
                ..RegularizedGlmOptions::default()
            },
        )
        .unwrap();
    assert_eq!(reg.coefficients[0], 0.0);
}
