//! Regression tests for classification-module review findings (ZTP hurdle, Tweedie series
//! density / power selection, likelihood-based AIC/BIC, eta overflow bounds, convergence
//! propagation, B-spline domain handling, scorer validation).
//!
//! All simulations use a local SplitMix64 PRNG seeded with `SEED = 42069`.

use pricing_kernels::classification::{
    GlmFamily, GlmOptions, MultiplicativeRater, RegularizedGlmFamily, RegularizedGlmOptions,
    bspline_basis, fit_glm_irls_with_options, fit_hurdle_poisson, fit_random_intercept_glmm,
    fit_regularized_glm, fit_zero_inflated_poisson, glm_information_criteria, predict_glm,
    select_tweedie_power, tweedie_log_density,
};
use pricing_kernels::error::PricingError;

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

    /// Poisson by Knuth multiplication (fine for the small rates used here).
    fn poisson(&mut self, lambda: f64) -> u64 {
        let limit = (-lambda).exp();
        let mut k = 0u64;
        let mut prod = self.uniform();
        while prod > limit {
            k += 1;
            prod *= self.uniform();
        }
        k
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
        fit_random_intercept_glmm(&x, 20, 1, &y, &groups, 4, Some(&short), None, 5),
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

    let glmm = fit_random_intercept_glmm(&x, n, 1, &y, &groups, 8, None, None, 50).unwrap();
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
    let fit_full =
        fit_glm_irls_with_options(GlmFamily::Gamma, &x_full, n, 2, &y, &w, &off, &opts).unwrap();
    let fit_null =
        fit_glm_irls_with_options(GlmFamily::Gamma, &x_null, n, 1, &y, &w, &off, &opts).unwrap();
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
