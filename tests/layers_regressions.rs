//! Regression tests for verified review findings in `pricing_kernels::layers`
//! (EVT fitting, ALAE-aware ILFs, reinstatement pricing, MBBEFD limits).
//!
//! All stochastic inputs come from a seeded SplitMix64 generator (default seed `42069`).

use pricing_kernels::error::{PricingError, SolverOptions};
use pricing_kernels::layers::{
    AlaeDependenceModel, AlaePolicyTreatment, EvtFamily, FittedEvtDistribution, MbbefdCurve,
    ReinstatementBasis, SeverityDistribution, SeverityObservation, SpliceBodyFamily,
    SpliceContinuity, alae_loss_severity, compound_poisson_reinstatement_slices, fit_evt_mle,
    fit_evt_mle_with_options, gauss_kronrod_integrate, gauss_kronrod_integrate_semi_infinite,
    goodness_of_fit, price_reinstatements, price_reinstatements_compound_poisson,
};

const SEED: u64 = 42069;

/// SplitMix64 (Steele, Lea & Flood 2014): small deterministic PRNG for tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller (one draw per call).
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Poisson via inversion (small means only).
    fn poisson(&mut self, lambda: f64) -> usize {
        let u = self.uniform();
        let mut p = (-lambda).exp();
        let mut cdf = p;
        let mut n = 0usize;
        while u > cdf {
            n += 1;
            p *= lambda / n as f64;
            cdf += p;
        }
        n
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

fn total_ll(dist: &FittedEvtDistribution, obs: &[SeverityObservation]) -> f64 {
    obs.iter()
        .map(|o| {
            let t = if o.is_censored {
                dist.log_survival(o.loss)
            } else {
                dist.log_pdf(o.loss)
            };
            o.weight * (t - dist.log_survival(o.deductible))
        })
        .sum()
}

// ===========================================================================
// Finding 1: Burr XII dollar-scale fit (reparametrized in theta) + scale equivariance
// ===========================================================================

fn burr_sample(n: usize, alpha: f64, gamma: f64, theta: f64, rng: &mut SplitMix64) -> Vec<f64> {
    (0..n)
        .map(|_| theta * (rng.uniform().powf(-1.0 / alpha) - 1.0).powf(1.0 / gamma))
        .collect()
}

fn burr_params(d: FittedEvtDistribution) -> (f64, f64, f64) {
    let FittedEvtDistribution::BurrXii {
        alpha,
        gamma,
        lambda,
    } = d
    else {
        panic!("expected BurrXii, got {d:?}");
    };
    (alpha, gamma, lambda.powf(1.0 / gamma))
}

#[test]
fn burr_xii_recovers_dollar_scale_parameters_and_is_scale_equivariant() {
    let (alpha, gamma, theta) = (1.5, 3.0, 1e5);
    let mut rng = SplitMix64(SEED);
    let xs = burr_sample(2000, alpha, gamma, theta, &mut rng);
    let obs: Vec<SeverityObservation> = xs.iter().map(|&x| SeverityObservation::exact(x)).collect();

    let fit = fit_evt_mle(EvtFamily::BurrXii, &obs).expect("burr fit");
    assert!(fit.converged, "burr fit must converge: {fit:?}");
    let (a_hat, g_hat, t_hat) = burr_params(fit.distribution);
    // ~3-4 standard errors at n = 2000.
    assert!(rel(a_hat, alpha) < 0.20, "alpha={a_hat}");
    assert!(rel(g_hat, gamma) < 0.08, "gamma={g_hat}");
    assert!(rel(t_hat, theta) < 0.12, "theta={t_hat}");
    // MLE dominates the truth.
    let truth = FittedEvtDistribution::BurrXii {
        alpha,
        gamma,
        lambda: theta.powf(gamma),
    };
    assert!(fit.log_likelihood >= total_ll(&truth, &obs) - 1e-6);
    assert_eq!(fit.standard_errors.len(), 3);
    assert!(
        fit.standard_errors
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );

    // Scale equivariance: X -> 1000 X gives theta x 1000, same (alpha, gamma).
    let obs_k: Vec<SeverityObservation> = xs
        .iter()
        .map(|&x| SeverityObservation::exact(1000.0 * x))
        .collect();
    let fit_k = fit_evt_mle(EvtFamily::BurrXii, &obs_k).expect("burr fit scaled");
    assert!(fit_k.converged);
    let (a_k, g_k, t_k) = burr_params(fit_k.distribution);
    assert!(rel(a_k, a_hat) < 1e-4, "alpha {a_k} vs {a_hat}");
    assert!(rel(g_k, g_hat) < 1e-4, "gamma {g_k} vs {g_hat}");
    assert!(
        rel(t_k / t_hat, 1000.0) < 1e-4,
        "theta ratio {}",
        t_k / t_hat
    );
    // ll shifts by -n ln(1000) exactly.
    let shift = -(xs.len() as f64) * 1000.0_f64.ln();
    assert!((fit_k.log_likelihood - (fit.log_likelihood + shift)).abs() < 1e-5);
}

// ===========================================================================
// GPD and spliced fits recover parameters on seeded data
// ===========================================================================

fn gpd_excess(xi: f64, beta: f64, rng: &mut SplitMix64) -> f64 {
    beta * (rng.uniform().powf(-xi) - 1.0) / xi
}

#[test]
fn gpd_fit_recovers_seeded_parameters() {
    let (u, xi, beta) = (50_000.0, 0.3, 10_000.0);
    let mut rng = SplitMix64(SEED);
    let obs: Vec<SeverityObservation> = (0..3000)
        .map(|_| SeverityObservation::exact(u + gpd_excess(xi, beta, &mut rng)))
        .collect();
    let fit = fit_evt_mle(EvtFamily::Gpd { threshold: u }, &obs).expect("gpd");
    assert!(fit.converged);
    let FittedEvtDistribution::Gpd {
        xi: xi_hat,
        beta: b_hat,
        ..
    } = fit.distribution
    else {
        panic!("expected Gpd");
    };
    assert!((xi_hat - xi).abs() < 0.08, "xi={xi_hat}");
    assert!(rel(b_hat, beta) < 0.10, "beta={b_hat}");
    assert_eq!(fit.distribution.parameter_names(), &["xi", "beta"]);
    // SE(xi) ~ (1 + xi) / sqrt(n) for the GPD.
    let se_xi_asym = (1.0 + xi) / 3000.0_f64.sqrt();
    assert!(rel(fit.standard_errors[0], se_xi_asym) < 0.25);
}

#[test]
fn spliced_lognormal_gpd_recovers_parameters_and_reports_named_ses() {
    let (mu, sigma, u, w, xi): (f64, f64, f64, f64, f64) = (9.0, 1.0, 30_000.0, 0.85, 0.35);
    // Choose beta so the simulated model satisfies the C0 density-continuity constraint
    // w f_body(u) / F_body(u) = (1 - w) / beta (the fitted family is then well specified).
    let z_u = (u.ln() - mu) / sigma;
    let f_body_u = std_normal_pdf(z_u) / (u * sigma);
    let cdf_body_u = simpson(std_normal_pdf, -12.0, z_u, 4000);
    let beta = (1.0 - w) * cdf_body_u / (w * f_body_u);
    let mut rng = SplitMix64(SEED);
    let mut xs = Vec::with_capacity(4000);
    while xs.len() < 4000 {
        if rng.uniform() < w {
            // Body: Lognormal(mu, sigma) truncated to (0, u] by rejection.
            loop {
                let x = (mu + sigma * rng.normal()).exp();
                if x <= u {
                    xs.push(x);
                    break;
                }
            }
        } else {
            xs.push(u + gpd_excess(xi, beta, &mut rng));
        }
    }
    let obs: Vec<SeverityObservation> = xs.iter().map(|&x| SeverityObservation::exact(x)).collect();
    let fam = EvtFamily::SplicedBodyGpd {
        body: SpliceBodyFamily::Lognormal,
        threshold: u,
        continuity: SpliceContinuity::C0Continuous,
    };
    let fit = fit_evt_mle(fam, &obs).expect("spliced");
    let FittedEvtDistribution::SplicedBodyGpd {
        body_param1,
        body_param2,
        body_weight,
        xi: xi_hat,
        beta: b_hat,
        ..
    } = fit.distribution
    else {
        panic!("expected spliced");
    };
    assert!((body_param1 - mu).abs() < 0.15, "mu={body_param1}");
    assert!((body_param2 - sigma).abs() < 0.1, "sigma={body_param2}");
    assert!((body_weight - w).abs() < 0.03, "w={body_weight}");
    assert!((xi_hat - xi).abs() < 0.12, "xi={xi_hat}");
    assert!(rel(b_hat, beta) < 0.2, "beta={b_hat}");

    // Finding 9: SE vector follows variant field order incl. derived body_weight.
    assert_eq!(
        fit.distribution.parameter_names(),
        &["body_param1", "body_param2", "body_weight", "xi", "beta"]
    );
    assert_eq!(fit.standard_errors.len(), 5);
    assert!(
        fit.standard_errors
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    );
    // body_weight SE is a probability SE: sqrt(w (1 - w) / n) order of magnitude.
    let se_w = fit.standard_errors[2];
    assert!(se_w > 1e-4 && se_w < 0.05, "se_w={se_w}");

    let fam_c1 = EvtFamily::SplicedBodyGpd {
        body: SpliceBodyFamily::Lognormal,
        threshold: u,
        continuity: SpliceContinuity::C1Differentiable,
    };
    let fit_c1 = fit_evt_mle(fam_c1, &obs).expect("spliced c1");
    assert_eq!(fit_c1.standard_errors.len(), 5);
}

// ===========================================================================
// Finding 4: GPD xi > -1 constraint (Smith 1985)
// ===========================================================================

#[test]
fn gpd_uniform_excesses_never_report_converged_with_nan_ses() {
    let mut rng = SplitMix64(SEED);
    let obs: Vec<SeverityObservation> = (0..30)
        .map(|_| SeverityObservation::exact(1.0 + 100.0 * rng.uniform()))
        .collect();
    match fit_evt_mle(EvtFamily::Gpd { threshold: 0.0 }, &obs) {
        Ok(fit) => {
            let FittedEvtDistribution::Gpd { xi, .. } = fit.distribution else {
                panic!("expected Gpd");
            };
            assert!(xi > -1.0, "xi={xi} violates the xi > -1 constraint");
            // Uniform excesses put the constrained MLE on the xi -> -1 boundary (pre-fix:
            // xi = -1.25, NaN SEs, converged = true).
            assert!(xi < -0.99, "xi={xi}");
            assert!(!fit.converged, "boundary fit must not report convergence");
            if fit.converged {
                assert!(fit.standard_errors.iter().all(|v| v.is_finite()));
            }
        }
        Err(e) => assert!(matches!(e, PricingError::NonConvergence(_))),
    }
}

// ===========================================================================
// Finding 3: product-limit KS with delayed entry (left truncation)
// ===========================================================================

#[test]
fn ks_uses_delayed_entry_risk_sets_under_left_truncation() {
    let (alpha, lambda, d) = (2.0, 1000.0, 500.0);
    let n: usize = 4000;
    let mut rng = SplitMix64(SEED);
    let obs: Vec<SeverityObservation> = (0..n)
        .map(|i| {
            let u = rng.uniform();
            if i % 2 == 0 {
                SeverityObservation::exact(lambda * (u.powf(-1.0 / alpha) - 1.0))
            } else {
                // X | X > d for Pareto II: S(x)/S(d) = ((lambda + d)/(lambda + x))^alpha.
                let x = (lambda + d) * u.powf(-1.0 / alpha) - lambda;
                SeverityObservation::truncated(x.max(d * (1.0 + 1e-12)), d)
            }
        })
        .collect();
    let truth = FittedEvtDistribution::ParetoII { alpha, lambda };
    let ll = total_ll(&truth, &obs);
    let gof = goodness_of_fit(&truth, &obs, ll, 2);
    // Below the truncation point only the untruncated half is at risk, so the product-limit
    // estimator there has the precision of an n/2 sample: the asymptotic 5% Kolmogorov
    // critical value is 1.36 / sqrt(n/2) (seeded 40-replicate check: q95 of sqrt(n) D_n
    // = 1.94 ≈ 1.36 sqrt(2)). The pre-fix estimator (risk set n from the start) gave
    // D_n ≈ F(500)/2 ≈ 0.28 here.
    let crit_5pct = 1.36 / (n as f64 / 2.0).sqrt();
    assert!(
        gof.ks_statistic < crit_5pct,
        "D_n={} >= {crit_5pct}",
        gof.ks_statistic
    );
    let ad = gof.anderson_darling.expect("uncensored");
    assert!(ad < 2.492, "A^2={ad} above 5% critical value");
}

// ===========================================================================
// Finding 10: weights are multiplicities (BIC, KS, AD, fit)
// ===========================================================================

#[test]
fn integer_weights_equal_replicated_rows() {
    let mut rng = SplitMix64(SEED);
    let base: Vec<(f64, f64, bool)> = (0..60)
        .map(|i| {
            let x = (8.0 + 1.2 * rng.normal()).exp() + 600.0;
            let d = if i % 3 == 0 { 500.0 } else { 0.0 };
            (x, d, i % 7 == 0)
        })
        .collect();
    let mult = |i: usize| (i % 3 + 1) as f64;
    let mk = |&(x, d, c): &(f64, f64, bool)| -> SeverityObservation {
        let o = if c {
            SeverityObservation::censored(x, d)
        } else {
            SeverityObservation::truncated(x, d)
        };
        if d == 0.0 && !c {
            SeverityObservation::exact(x)
        } else {
            o
        }
    };
    let weighted: Vec<SeverityObservation> = base
        .iter()
        .enumerate()
        .map(|(i, r)| mk(r).with_weight(mult(i)))
        .collect();
    let replicated: Vec<SeverityObservation> = base
        .iter()
        .enumerate()
        .flat_map(|(i, r)| std::iter::repeat_n(mk(r), mult(i) as usize))
        .collect();

    let dist = FittedEvtDistribution::Lognormal {
        mu: 8.1,
        sigma: 1.1,
    };
    let ll_w = total_ll(&dist, &weighted);
    let ll_r = total_ll(&dist, &replicated);
    assert!((ll_w - ll_r).abs() < 1e-8 * ll_r.abs());
    let g_w = goodness_of_fit(&dist, &weighted, ll_w, 2);
    let g_r = goodness_of_fit(&dist, &replicated, ll_r, 2);
    assert!((g_w.bic - g_r.bic).abs() < 1e-8 * g_r.bic.abs());
    assert!((g_w.ks_statistic - g_r.ks_statistic).abs() < 1e-12);
    assert!(g_w.anderson_darling.is_none() && g_r.anderson_darling.is_none());

    // Uncensored subset: AD must also agree.
    let w_unc: Vec<SeverityObservation> = weighted
        .iter()
        .filter(|o| !o.is_censored)
        .copied()
        .collect();
    let r_unc: Vec<SeverityObservation> = replicated
        .iter()
        .filter(|o| !o.is_censored)
        .copied()
        .collect();
    let a_w = goodness_of_fit(&dist, &w_unc, total_ll(&dist, &w_unc), 2)
        .anderson_darling
        .expect("ad");
    let a_r = goodness_of_fit(&dist, &r_unc, total_ll(&dist, &r_unc), 2)
        .anderson_darling
        .expect("ad");
    assert!(
        (a_w - a_r).abs() < 1e-9 * a_r.abs().max(1.0),
        "{a_w} vs {a_r}"
    );

    let f_w = fit_evt_mle(EvtFamily::Lognormal, &weighted).expect("fit w");
    let f_r = fit_evt_mle(EvtFamily::Lognormal, &replicated).expect("fit r");
    let (
        FittedEvtDistribution::Lognormal { mu: m1, sigma: s1 },
        FittedEvtDistribution::Lognormal { mu: m2, sigma: s2 },
    ) = (f_w.distribution, f_r.distribution)
    else {
        panic!("expected Lognormal");
    };
    assert!((m1 - m2).abs() < 1e-6 && (s1 - s2).abs() < 1e-6);
    assert!((f_w.diagnostics.bic - f_r.diagnostics.bic).abs() < 1e-5);
    assert!((f_w.diagnostics.ks_statistic - f_r.diagnostics.ks_statistic).abs() < 1e-6);
}

// ===========================================================================
// Finding 7: SolverOptions::tol is honoured
// ===========================================================================

#[test]
fn solver_tolerance_changes_iterations_and_results() {
    let mut rng = SplitMix64(SEED);
    let xs = burr_sample(500, 2.0, 1.5, 20_000.0, &mut rng);
    let obs: Vec<SeverityObservation> = xs.iter().map(|&x| SeverityObservation::exact(x)).collect();
    let loose = fit_evt_mle_with_options(
        EvtFamily::BurrXii,
        &obs,
        &SolverOptions::default().with_max_iter(500).with_tol(1e-1),
    )
    .expect("loose");
    let tight = fit_evt_mle_with_options(
        EvtFamily::BurrXii,
        &obs,
        &SolverOptions::default().with_max_iter(500).with_tol(1e-8),
    )
    .expect("tight");
    assert!(
        loose.iterations < tight.iterations,
        "loose {} vs tight {}",
        loose.iterations,
        tight.iterations
    );
    assert_ne!(loose.distribution, tight.distribution);
    assert!(tight.log_likelihood >= loose.log_likelihood - 1e-9);
}

// ===========================================================================
// Finding 11: MBBEFD continuity across b = 1 and b g = 1
// ===========================================================================

#[test]
fn mbbefd_is_continuous_across_b_equals_one_and_bg_equals_one() {
    let g = 10.0;
    let at = |b: f64, g: f64| MbbefdCurve::new(b, g).expect("curve");
    for &delta in &[2e-12, -2e-12, 1e-10, -1e-10, 1e-8, -1e-8] {
        // b -> 1 limit.
        let c0 = at(1.0, g);
        let c1 = at(1.0 + delta, g);
        // bg -> 1 limit (b = 1 / g).
        let e0 = at(1.0 / g, g);
        let e1 = at((1.0 + delta) / g, g);
        for &d in &[0.01, 0.1, 0.37, 0.8, 0.99] {
            let bound = 10.0 * delta.abs() + 1e-15;
            let gc = (c1.exposure_curve(d).unwrap() - c0.exposure_curve(d).unwrap()).abs();
            assert!(gc < bound, "G b=1+{delta} d={d}: {gc}");
            let sc = (c1.survival(d).unwrap() - c0.survival(d).unwrap()).abs();
            assert!(sc < bound, "S b=1+{delta} d={d}: {sc}");
            let ge = (e1.exposure_curve(d).unwrap() - e0.exposure_curve(d).unwrap()).abs();
            assert!(ge < bound, "G bg=1+{delta} d={d}: {ge}");
            let se = (e1.survival(d).unwrap() - e0.survival(d).unwrap()).abs();
            assert!(se < bound, "S bg=1+{delta} d={d}: {se}");
        }
        let mc = (c1.mean_loss_ratio().unwrap() - c0.mean_loss_ratio().unwrap()).abs();
        let me = (e1.mean_loss_ratio().unwrap() - e0.mean_loss_ratio().unwrap()).abs();
        assert!(mc < 10.0 * delta.abs() + 1e-15 && me < 10.0 * delta.abs() + 1e-15);
    }
    // Closed-form limits: b = 1 -> ln(1 + (g - 1) d) / ln g, mean ln g / (g - 1).
    let c = at(1.0, g);
    let d = 0.3_f64;
    assert!((c.exposure_curve(d).unwrap() - (9.0 * d).ln_1p() / g.ln()).abs() < 1e-15);
    assert!((c.mean_loss_ratio().unwrap() - g.ln() / (g - 1.0)).abs() < 1e-15);
    // bg = 1 -> (1 - b^d) / (1 - b), S = b^d.
    let b = 0.25;
    let e = at(b, 4.0);
    assert!((e.exposure_curve(d).unwrap() - (1.0 - b.powf(d)) / (1.0 - b)).abs() < 1e-15);
    assert!((e.survival(d).unwrap() - b.powf(d)).abs() < 1e-15);
    // General point agrees with Bernegger (1997) eq. 3.3 directly.
    let (b, g) = (0.05, 30.0);
    let general = at(b, g);
    let direct = (((g - 1.0) * b + (1.0 - b * g) * b.powf(d)) / (1.0 - b)).ln() / (b * g).ln();
    assert!((general.exposure_curve(d).unwrap() - direct).abs() < 1e-13);
}

// ===========================================================================
// Findings 2 & 8: bivariate Lognormal ALAE-included accuracy + marginal ownership
// ===========================================================================

fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    let n = n + n % 2;
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * f(a + h * i as f64);
    }
    s * h / 3.0
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / std::f64::consts::TAU.sqrt()
}

/// Independent reference: nested composite Simpson in standard-normal coordinates, split
/// exactly at both kinks (`x = L` and `x + a = L`).
fn bln_included_reference(mu_x: f64, s_x: f64, mu_a: f64, s_a: f64, rho: f64, lim: f64) -> f64 {
    const Z: f64 = 10.0;
    const N: usize = 1200;
    let cond = s_a * (1.0 - rho * rho).sqrt();
    let z_l = ((lim.ln() - mu_x) / s_x).clamp(-Z, Z);
    let inner = |z1: f64| -> f64 {
        let x = (mu_x + s_x * z1).exp();
        let mu_c = mu_a + rho * s_a * z1;
        let k = (((lim - x).max(f64::MIN_POSITIVE).ln() - mu_c) / cond).clamp(-Z, Z);
        let below = simpson(
            |z2| std_normal_pdf(z2) * (x + (mu_c + cond * z2).exp()),
            -Z,
            k,
            N,
        );
        let above = simpson(|z2| std_normal_pdf(z2) * lim, k, Z, N);
        std_normal_pdf(z1) * (below + above)
    };
    // The outer integrand changes on the scale E[A] / (L sigma_x) just below z_L, so grade
    // the mesh geometrically towards z_L: z1 = z_L - e^v.
    let v_hi = (z_l + Z).ln();
    let graded = simpson(
        |v| {
            let e = v.exp();
            inner(z_l - e) * e
        },
        V_MIN,
        v_hi,
        N,
    );
    graded + simpson(|z1| std_normal_pdf(z1) * lim, z_l, Z, N)
}

/// Lower log-offset of the graded outer mesh (`e^-30 ≈ 1e-13`; remainder negligible).
const V_MIN: f64 = -30.0;

#[test]
fn bivariate_lognormal_included_matches_high_resolution_reference() {
    // (mu_x, sigma_x, mu_a, sigma_a, rho, L)
    let cases = [
        (10.0, 1.0, 8.5, 0.8, 0.65, 100_000.0),
        (10.0, 2.5, 8.5, 1.5, 0.5, 1_000_000.0),
        (10.0, 2.5, 8.5, 1.5, 0.5, 50_000.0),
    ];
    for &(mu_x, sigma_x, mu_a, sigma_a, rho, lim) in &cases {
        let dep = AlaeDependenceModel::BivariateLognormal {
            mu_x,
            sigma_x,
            mu_a,
            sigma_a,
            rho,
        };
        let sev = dep.indemnity_marginal().expect("marginal");
        let got = alae_loss_severity(&sev, lim, dep, AlaePolicyTreatment::Included).expect("inc");
        let reference = bln_included_reference(mu_x, sigma_x, mu_a, sigma_a, rho, lim);
        assert!(
            rel(got, reference) < 1e-8,
            "case L={lim} sx={sigma_x}: got {got} ref {reference} rel {}",
            rel(got, reference)
        );
    }
}

#[test]
fn bivariate_lognormal_rejects_conflicting_severity() {
    let dep = AlaeDependenceModel::BivariateLognormal {
        mu_x: 10.0,
        sigma_x: 1.0,
        mu_a: 8.5,
        sigma_a: 0.8,
        rho: 0.65,
    };
    let other = SeverityDistribution::Lognormal {
        mu: 10.0,
        sigma: 1.2,
    };
    for tr in [
        AlaePolicyTreatment::Included,
        AlaePolicyTreatment::ProRata,
        AlaePolicyTreatment::InAddition,
    ] {
        assert!(matches!(
            alae_loss_severity(&other, 1e5, dep, tr),
            Err(PricingError::InvalidParameter(_))
        ));
        let own = dep.indemnity_marginal().expect("marginal");
        assert!(alae_loss_severity(&own, 1e5, dep, tr).is_ok());
    }
}

// ===========================================================================
// Finding 6: PowerLaw is deterministic ALAE; PowerLawLognormal adds dispersion
// ===========================================================================

#[test]
fn power_law_lognormal_matches_equivalent_bivariate_lognormal_and_jensen() {
    // A | X = x Lognormal with mean c x^g, CV cv and X ~ LN(mu_x, s_x) is bivariate Lognormal:
    // ln A = ln c - s^2/2 + g ln X + s Z,  s^2 = ln(1 + cv^2).
    let (mu_x, s_x, c, g, cv): (f64, f64, f64, f64, f64) = (10.0, 1.2, 1.5, 0.75, 0.9);
    let s2 = (cv * cv).ln_1p();
    let mu_a = c.ln() - 0.5 * s2 + g * mu_x;
    let sigma_a = (g * g * s_x * s_x + s2).sqrt();
    let rho = g * s_x / sigma_a;
    let biv = AlaeDependenceModel::BivariateLognormal {
        mu_x,
        sigma_x: s_x,
        mu_a,
        sigma_a,
        rho,
    };
    let sev = biv.indemnity_marginal().expect("marginal");
    let plln = AlaeDependenceModel::PowerLawLognormal { c, gamma: g, cv };
    let pl = AlaeDependenceModel::PowerLaw { c, gamma: g };
    for lim in [50_000.0, 250_000.0, 1_000_000.0] {
        for tr in [
            AlaePolicyTreatment::Included,
            AlaePolicyTreatment::ProRata,
            AlaePolicyTreatment::InAddition,
        ] {
            let a = alae_loss_severity(&sev, lim, plln, tr).expect("plln");
            let b = alae_loss_severity(&sev, lim, biv, tr).expect("biv");
            assert!(rel(a, b) < 1e-7, "L={lim} {tr:?}: {a} vs {b}");
        }
        let det = alae_loss_severity(&sev, lim, pl, AlaePolicyTreatment::Included).expect("pl");
        let rnd = alae_loss_severity(&sev, lim, plln, AlaePolicyTreatment::Included).expect("plln");
        assert!(rnd < det, "Jensen: dispersed ALAE must reduce E[(X+A) ∧ L]");
        let zero_cv = AlaeDependenceModel::PowerLawLognormal {
            c,
            gamma: g,
            cv: 0.0,
        };
        let z = alae_loss_severity(&sev, lim, zero_cv, AlaePolicyTreatment::Included).expect("z");
        assert_eq!(z, det);
    }
}

// ===========================================================================
// Finding 5: exact pro-rata-as-to-time reinstatement premium
// ===========================================================================

#[test]
fn time_basis_total_loss_poisson_one_gives_exp_minus_one() {
    let lim = 1_000_000.0;
    let slices = compound_poisson_reinstatement_slices(lim, 1.0, lim, &[0.0, 1.0], 2).expect("s");
    let e_inv = (-1.0_f64).exp();
    assert!((slices.expected_layer_losses[0] / lim - (1.0 - e_inv)).abs() < 1e-12);
    assert!((slices.expected_layer_losses[1] / lim - (1.0 - 2.0 * e_inv)).abs() < 1e-12);
    assert!((slices.time_weighted_layer_losses[0] / lim - e_inv).abs() < 1e-12);

    let priced = price_reinstatements_compound_poisson(
        lim,
        1.0,
        lim,
        &[0.0, 1.0],
        &[1.0],
        0.0,
        ReinstatementBasis::ProRataAmountAndTime,
    )
    .expect("time basis");
    let factor = priced.expected_reinstatement_premium / priced.initial_premium;
    assert!((factor - e_inv).abs() < 1e-9, "factor={factor}");
    let total_loss = lim * (2.0 - 3.0 * e_inv);
    assert!(rel(priced.initial_premium, total_loss / (1.0 + e_inv)) < 1e-12);

    // Amount basis equals the slice-expectation API.
    let amount = price_reinstatements_compound_poisson(
        lim,
        1.0,
        lim,
        &[0.0, 1.0],
        &[1.0],
        0.0,
        ReinstatementBasis::ProRataAmount,
    )
    .expect("amount");
    let direct = price_reinstatements(
        lim,
        &slices.expected_layer_losses,
        &[1.0],
        0.0,
        ReinstatementBasis::ProRataAmount,
    )
    .expect("direct");
    assert!(rel(amount.initial_premium, direct.initial_premium) < 1e-14);

    // Slice expectations alone cannot price the time basis.
    assert!(matches!(
        price_reinstatements(
            lim,
            &slices.expected_layer_losses,
            &[1.0],
            0.0,
            ReinstatementBasis::ProRataAmountAndTime,
        ),
        Err(PricingError::InvalidParameter(_))
    ));
}

#[test]
fn time_basis_slices_match_seeded_monte_carlo() {
    let lim = 4.0;
    let step = 1.0;
    let pmf = [0.30, 0.25, 0.20, 0.15, 0.10];
    let lambda = 2.5;
    let n_slices = 3;
    let exact =
        compound_poisson_reinstatement_slices(lim, lambda, step, &pmf, n_slices).expect("s");

    let mut rng = SplitMix64(SEED);
    let n_sims: usize = 200_000;
    let mut sum_e = vec![0.0; n_slices];
    let mut sum_t = vec![0.0; n_slices];
    let mut sum_t2 = vec![0.0; n_slices];
    let mut times = Vec::new();
    for _ in 0..n_sims {
        let n = rng.poisson(lambda);
        times.clear();
        times.extend((0..n).map(|_| rng.uniform()));
        times.sort_by(f64::total_cmp);
        let mut agg = 0.0_f64;
        let mut tw = vec![0.0; n_slices];
        for &t in &times {
            let u = rng.uniform();
            let mut cdf = 0.0;
            let mut y = 0.0;
            for (j, &p) in pmf.iter().enumerate() {
                cdf += p;
                if u <= cdf {
                    y = j as f64 * step;
                    break;
                }
            }
            let before = agg;
            agg += y;
            for k in 0..n_slices {
                let g = |s: f64| (s - k as f64 * lim).clamp(0.0, lim);
                tw[k] += (1.0 - t) * (g(agg) - g(before));
            }
        }
        for k in 0..n_slices {
            sum_e[k] += (agg - k as f64 * lim).clamp(0.0, lim);
            sum_t[k] += tw[k];
            sum_t2[k] += tw[k] * tw[k];
        }
    }
    let nf = n_sims as f64;
    for k in 0..n_slices {
        let mean_t = sum_t[k] / nf;
        let sd_t = (sum_t2[k] / nf - mean_t * mean_t).max(0.0).sqrt();
        let se = sd_t / nf.sqrt();
        let z = (mean_t - exact.time_weighted_layer_losses[k]) / se;
        assert!(
            z.abs() < 4.0,
            "slice {k}: MC {mean_t} exact {} z={z}",
            exact.time_weighted_layer_losses[k]
        );
        let mean_e = sum_e[k] / nf;
        assert!(
            (mean_e - exact.expected_layer_losses[k]).abs() < 0.01 * lim,
            "slice {k}: E MC {mean_e} exact {}",
            exact.expected_layer_losses[k]
        );
    }
    // The flat 0.5 factor is wrong in both directions across slices.
    assert!(exact.time_weighted_layer_losses[0] > 0.5 * exact.expected_layer_losses[0]);
    assert!(exact.time_weighted_layer_losses[2] < 0.5 * exact.expected_layer_losses[2]);
}

// ===========================================================================
// Round 2: scale-aware quadrature, closed-form LEVs, power-law ALAE moments
// ===========================================================================

const ALL_TREATMENTS: [AlaePolicyTreatment; 3] = [
    AlaePolicyTreatment::Included,
    AlaePolicyTreatment::ProRata,
    AlaePolicyTreatment::InAddition,
];

#[test]
fn power_law_gamma_one_equals_proportional_for_all_treatments_and_scales() {
    let sevs = [
        SeverityDistribution::Lognormal {
            mu: 11.0,
            sigma: 2.5,
        },
        SeverityDistribution::Lognormal {
            mu: 15.0,
            sigma: 1.0,
        },
        SeverityDistribution::Lognormal {
            mu: 20.0,
            sigma: 1.0,
        },
        SeverityDistribution::SingleParameterPareto {
            alpha: 1.5,
            x_m: 1e5,
        },
        SeverityDistribution::Lomax {
            alpha: 1.8,
            theta: 2e5,
        },
        SeverityDistribution::Gamma {
            shape: 2.0,
            scale: 5e4,
        },
    ];
    let c = 0.35;
    for sev in &sevs {
        for lim in [1e5, 1e6, 1e7] {
            for tr in ALL_TREATMENTS {
                let prop =
                    alae_loss_severity(sev, lim, AlaeDependenceModel::Proportional { k: c }, tr)
                        .expect("proportional");
                let pw = alae_loss_severity(
                    sev,
                    lim,
                    AlaeDependenceModel::PowerLaw { c, gamma: 1.0 },
                    tr,
                )
                .expect("power law");
                assert!(
                    rel(pw, prop) < 1e-8,
                    "{sev:?} L={lim} {tr:?}: power-law {pw} vs proportional {prop}"
                );
            }
        }
    }
}

#[test]
fn power_law_lognormal_matches_bivariate_lognormal_at_realistic_scale() {
    let (mu_x, s_x, c, g, cv): (f64, f64, f64, f64, f64) = (15.0, 1.0, 0.5, 0.8, 1.0);
    let s2 = (cv * cv).ln_1p();
    let sigma_a = (g * g * s_x * s_x + s2).sqrt();
    let biv = AlaeDependenceModel::BivariateLognormal {
        mu_x,
        sigma_x: s_x,
        mu_a: c.ln() + g * mu_x - 0.5 * s2,
        sigma_a,
        rho: g * s_x / sigma_a,
    };
    let sev = biv.indemnity_marginal().expect("marginal");
    let plln = AlaeDependenceModel::PowerLawLognormal { c, gamma: g, cv };
    for lim in [1e5, 1e6, 1e7, 1e8, 1e9] {
        for tr in ALL_TREATMENTS {
            let a = alae_loss_severity(&sev, lim, plln, tr).expect("plln");
            let b = alae_loss_severity(&sev, lim, biv, tr).expect("biv");
            assert!(rel(a, b) < 1e-7, "L={lim:e} {tr:?}: {a} vs {b}");
        }
    }
}

#[test]
fn power_law_in_addition_and_pro_rata_are_exact_at_dollar_scale() {
    let (c, g) = (0.5, 0.8);
    let dep = AlaeDependenceModel::PowerLaw { c, gamma: g };
    // Lognormal(15, 1): E[X^g] = exp(g mu + g^2 sigma^2 / 2).
    let ln = SeverityDistribution::Lognormal {
        mu: 15.0,
        sigma: 1.0,
    };
    // SPP(1.5, 1e5): E[X^g] = alpha x_m^g / (alpha - g).
    let spp = SeverityDistribution::SingleParameterPareto {
        alpha: 1.5,
        x_m: 1e5,
    };
    let cases = [
        (ln.clone(), (g * 15.0 + 0.5 * g * g).exp()),
        (spp.clone(), 1.5 * 1e5_f64.powf(g) / (1.5 - g)),
    ];
    for (sev, moment) in &cases {
        for lim in [1e5, 1e6, 1e7] {
            let lev = sev.limited_expected_value(lim).expect("lev");
            let add =
                alae_loss_severity(sev, lim, dep, AlaePolicyTreatment::InAddition).expect("add");
            let expected = lev + c * moment;
            assert!(
                rel(add, expected) < 1e-10,
                "{sev:?} L={lim}: {add} vs {expected}"
            );
        }
    }
    // SPP pro-rata: E[c X^g; X <= L] + L E[c X^(g-1); X > L] in closed form.
    let (a, xm, lim) = (1.5_f64, 1e5_f64, 1e7_f64);
    let below = a * xm.powf(g) * ((lim / xm).powf(g - a) - 1.0) / (g - a);
    let above = a * xm.powf(g - 1.0) * (lim / xm).powf(g - 1.0 - a) / (a + 1.0 - g);
    let expected = spp.limited_expected_value(lim).expect("lev") + c * (below + lim * above);
    let got = alae_loss_severity(&spp, lim, dep, AlaePolicyTreatment::ProRata).expect("pro");
    assert!(rel(got, expected) < 1e-10, "{got} vs {expected}");
}

#[test]
fn power_law_in_addition_rejects_infinite_alae_moment() {
    let dep = AlaeDependenceModel::PowerLaw {
        c: 2.0,
        gamma: 0.75,
    };
    let heavy = [
        SeverityDistribution::Lomax {
            alpha: 0.5,
            theta: 1e5,
        },
        SeverityDistribution::SingleParameterPareto {
            alpha: 0.6,
            x_m: 1e5,
        },
        SeverityDistribution::GeneralizedPareto { xi: 2.0, beta: 1e5 },
    ];
    for sev in &heavy {
        assert!(
            matches!(
                alae_loss_severity(sev, 1e6, dep, AlaePolicyTreatment::InAddition),
                Err(PricingError::InvalidParameter(_))
            ),
            "{sev:?}: infinite E[X^gamma] must be rejected"
        );
        let plln = AlaeDependenceModel::PowerLawLognormal {
            c: 2.0,
            gamma: 0.75,
            cv: 0.5,
        };
        assert!(alae_loss_severity(sev, 1e6, plln, AlaePolicyTreatment::InAddition).is_err());
        // Capped treatments stay finite.
        for tr in [AlaePolicyTreatment::Included, AlaePolicyTreatment::ProRata] {
            let v = alae_loss_severity(sev, 1e6, dep, tr).expect("capped treatment");
            assert!(v.is_finite() && v > 0.0, "{sev:?} {tr:?}: {v}");
        }
    }
}

/// Independent reference `∫_a^b S(x) dx` by composite Simpson in `t = ln x`
/// (starting 60 e-folds below `b` when `a == 0`).
fn lev_reference(d: &FittedEvtDistribution, a: f64, b: f64) -> f64 {
    const N: usize = 400_000;
    const LOG_SPAN: f64 = 60.0;
    if !(b > a) {
        return 0.0;
    }
    let t_lo = if a > 0.0 { a.ln() } else { b.ln() - LOG_SPAN };
    simpson(
        |t| {
            let x = t.exp();
            x * d.log_survival(x).exp()
        },
        t_lo,
        b.ln(),
        N,
    )
}

fn assert_lev(d: &FittedEvtDistribution, lim: f64, expected: f64) {
    let got = d.limited_expected_value(lim).expect("lev");
    assert!(
        rel(got, expected) < 1e-8,
        "{d:?} L={lim:e}: got {got}, expected {expected}, rel {}",
        rel(got, expected)
    );
}

#[test]
fn evt_lev_weibull_burr_inverse_gaussian_match_closed_forms_up_to_1e9() {
    let limits = [1e2, 1e3, 1e5, 1e7, 1e9];
    let lam = 1000.0;
    let w1 = FittedEvtDistribution::Weibull {
        k: 1.0,
        lambda: lam,
    };
    let w_half = FittedEvtDistribution::Weibull {
        k: 0.5,
        lambda: lam,
    };
    for lim in limits {
        assert_lev(&w1, lim, -lam * (-lim / lam).exp_m1());
        let s = (lim / lam).sqrt();
        assert_lev(&w_half, lim, 2.0 * lam * (1.0 - (1.0 + s) * (-s).exp()));
    }
    // Weibull k = 2: λ √π / 2 erf(L / λ); erf(2) = 0.995_322_265_018_952_7.
    let w2 = FittedEvtDistribution::Weibull {
        k: 2.0,
        lambda: lam,
    };
    let half_sqrt_pi = 0.886_226_925_452_758;
    assert_lev(&w2, 2.0 * lam, lam * half_sqrt_pi * 0.995_322_265_018_952_7);
    assert_lev(&w2, 1e9, lam * half_sqrt_pi);

    // Burr XII S(x) = (λ / (λ + x^γ))^α.
    let lb: f64 = 1e6;
    let rl = lb.sqrt();
    let burr_atan = FittedEvtDistribution::BurrXii {
        alpha: 1.0,
        gamma: 2.0,
        lambda: lb,
    };
    let burr_asinh = FittedEvtDistribution::BurrXii {
        alpha: 0.5,
        gamma: 2.0,
        lambda: lb,
    };
    let burr_lomax = FittedEvtDistribution::BurrXii {
        alpha: 2.0,
        gamma: 1.0,
        lambda: 5e4,
    };
    for lim in limits {
        assert_lev(&burr_atan, lim, rl * (lim / rl).atan());
        assert_lev(&burr_asinh, lim, rl * (lim / rl).asinh());
        assert_lev(&burr_lomax, lim, 5e4 * (1.0 - 5e4 / (5e4 + lim)));
    }

    // Inverse Gaussian: LEV -> mu as L -> ∞; intermediate limits vs Simpson reference.
    for (mu, shape) in [(1000.0, 500.0), (1000.0, 5000.0), (2e5, 1e5)] {
        let ig = FittedEvtDistribution::InverseGaussian { mu, lambda: shape };
        assert_lev(&ig, 1e9, mu);
        for lim in [0.3 * mu, mu, 4.0 * mu, 1e7] {
            assert_lev(&ig, lim, lev_reference(&ig, 0.0, lim));
        }
    }
}

#[test]
fn evt_lev_gpd_and_spliced_respect_finite_support_and_thresholds() {
    // Shifted GPD with xi < 0: support ends at u + β / |ξ| = 2000.
    let (u, xi, beta) = (1000.0, -0.5, 500.0);
    let gpd_neg = FittedEvtDistribution::Gpd {
        threshold: u,
        xi,
        beta,
    };
    let full = u + beta / (1.0 - xi);
    for lim in [2000.0, 1e4, 1e6, 1e9] {
        assert_lev(&gpd_neg, lim, full);
    }
    assert_lev(
        &gpd_neg,
        1200.0,
        u + beta / (1.0 - xi) * (1.0 - 0.8_f64.powi(3)),
    );
    assert_lev(&gpd_neg, 500.0, 500.0);

    // Shifted GPD with xi > 0.
    let (xi_p, beta_p) = (0.3, 2000.0);
    let gpd_pos = FittedEvtDistribution::Gpd {
        threshold: u,
        xi: xi_p,
        beta: beta_p,
    };
    for lim in [1e4, 1e6, 1e9] {
        let y: f64 = lim - u;
        let tail = (1.0 + xi_p * y / beta_p).powf(1.0 - 1.0 / xi_p);
        assert_lev(&gpd_pos, lim, u + beta_p / (1.0 - xi_p) * (1.0 - tail));
    }

    // Spliced Lognormal body + GPD tail with xi < 0 (finite support end).
    let (u_s, w_s, xi_s, beta_s) = (2000.0, 0.8, -0.25, 800.0);
    let spl_ln = FittedEvtDistribution::SplicedBodyGpd {
        body: SpliceBodyFamily::Lognormal,
        threshold: u_s,
        continuity: SpliceContinuity::C0Continuous,
        body_param1: 7.0,
        body_param2: 1.0,
        body_weight: w_s,
        xi: xi_s,
        beta: beta_s,
    };
    let body_ref = lev_reference(&spl_ln, 0.0, u_s);
    for lim in [1e4, 1e6, 1e9] {
        assert_lev(&spl_ln, lim, body_ref + (1.0 - w_s) * beta_s / (1.0 - xi_s));
    }
    assert_lev(&spl_ln, 900.0, lev_reference(&spl_ln, 0.0, 900.0));

    // Spliced Weibull body + GPD tail with xi > 0.
    let (u_w, w_w, xi_w, beta_w) = (3000.0, 0.7, 0.4, 2000.0);
    let spl_w = FittedEvtDistribution::SplicedBodyGpd {
        body: SpliceBodyFamily::Weibull,
        threshold: u_w,
        continuity: SpliceContinuity::C0Continuous,
        body_param1: 0.8,
        body_param2: 1500.0,
        body_weight: w_w,
        xi: xi_w,
        beta: beta_w,
    };
    let body_w = lev_reference(&spl_w, 0.0, u_w);
    for lim in [1e5, 1e7, 1e9] {
        let y: f64 = lim - u_w;
        let tail = (1.0 + xi_w * y / beta_w).powf(1.0 - 1.0 / xi_w);
        let expected = body_w + (1.0 - w_w) * beta_w / (1.0 - xi_w) * (1.0 - tail);
        assert_lev(&spl_w, lim, expected);
    }
    assert_lev(&spl_w, 1000.0, lev_reference(&spl_w, 0.0, 1000.0));
}

#[test]
fn gauss_kronrod_public_api_is_scale_aware_and_reports_errors() {
    // Mass on scale 1e3 inside [0, 1e9] (a single panel sees ~0).
    let v = gauss_kronrod_integrate(|x| (-x / 1e3).exp(), 0.0, 1e9).expect("finite");
    assert!(rel(v, 1e3) < 1e-10, "v={v}");
    let r = gauss_kronrod_integrate(|x| (-x / 1e3).exp(), 1e9, 0.0).expect("reversed");
    assert!(rel(r, -1e3) < 1e-10, "r={r}");
    assert_eq!(gauss_kronrod_integrate(|x| x, 2.0, 2.0), Ok(0.0));
    // Semi-infinite with a scale 6 decades off the integrand's.
    let s = gauss_kronrod_integrate_semi_infinite(|x| (-x / 1e6).exp(), 0.0, 1.0).expect("semi");
    assert!(rel(s, 1e6) < 1e-10, "s={s}");
    let s2 = gauss_kronrod_integrate_semi_infinite(|x| (-(x - 5e5) / 1e3).exp(), 5e5, 1e3)
        .expect("shifted");
    assert!(rel(s2, 1e3) < 1e-10, "s2={s2}");

    assert!(gauss_kronrod_integrate(|x| x, f64::NAN, 1.0).is_err());
    assert!(gauss_kronrod_integrate(|x| x, 0.0, f64::INFINITY).is_err());
    assert!(gauss_kronrod_integrate(|_| f64::NAN, 0.0, 1.0).is_err());
    assert!(gauss_kronrod_integrate_semi_infinite(|x| x, f64::NAN, 1.0).is_err());
    assert!(gauss_kronrod_integrate_semi_infinite(|x| x, 0.0, 0.0).is_err());
    assert!(gauss_kronrod_integrate_semi_infinite(|x| x, 0.0, -1.0).is_err());
    assert!(gauss_kronrod_integrate_semi_infinite(|_| f64::NAN, 0.0, 1.0).is_err());
    // A divergent integral is reported, not silently truncated.
    assert!(gauss_kronrod_integrate_semi_infinite(|x| 1.0 / (1.0 + x), 0.0, 1.0).is_err());
}
