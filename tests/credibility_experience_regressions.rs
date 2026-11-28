//! Regression tests for credibility, experience-rating, and trending review findings:
//! seeded Monte Carlo unbiasedness checks (Bühlmann–Straub, Hachemeister, Jewell),
//! retro plan balance, Table M/L normalization and row invariants, NCCI accident
//! limitations, classical credibility standards, trend validation, and parallelogram
//! properties. Default seed `42069`; no external test dependencies.

use pricing_kernels::credibility::{
    ClassicalCredibility, HierarchicalRisk, HierarchicalSector, buhlmann_straub,
    hachemeister_credibility, jewell_hierarchical_credibility,
};
use pricing_kernels::error::{PricingError, SolverOptions};
use pricing_kernels::experience::{
    ExperienceClaim, NcciAccidentLimitation, NcciErpSpec, RetroPlanSpec, TableLEngine,
    TableMEngine, TableMRow, calculate_ncci_experience_mod,
    calculate_ncci_experience_mod_by_accident,
};
use pricing_kernels::trending::{
    RateChange, TrendModel, parallelogram_on_level_factor, policy_year_on_level_factor,
};

/// Default deterministic seed (house rule).
const SEED: u64 = 42069;

/// Monte Carlo replications for fast (default) unbiasedness tests.
const FAST_SIMS: usize = 4000;

/// Monte Carlo replications for the heavier `#[ignore]` check.
const SLOW_SIMS: usize = 40_000;

/// Acceptance band in standard errors of the Monte Carlo mean. With ~10 independent
/// assertions, a 5-sigma band gives a family-wise false-failure rate < 1e-5 while still
/// rejecting the pre-fix Hachemeister bias (~+4 on a true value of 4, i.e. > 25 SE).
const MC_SIGMA_BAND: f64 = 5.0;

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Running mean / standard error accumulator.
#[derive(Default)]
struct MeanSe {
    n: f64,
    sum: f64,
    sum_sq: f64,
}

impl MeanSe {
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        self.sum += x;
        self.sum_sq += x * x;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n
    }
    fn se(&self) -> f64 {
        let m = self.mean();
        let var = (self.sum_sq / self.n - m * m) * self.n / (self.n - 1.0);
        (var / self.n).sqrt()
    }
    fn assert_unbiased(&self, truth: f64, what: &str) {
        let (m, se) = (self.mean(), self.se());
        assert!(
            (m - truth).abs() <= MC_SIGMA_BAND * se + 1e-12,
            "{what}: MC mean {m} vs truth {truth} (se {se})"
        );
    }
}

// ---------------------------------------------------------------------------
// Hachemeister (findings 1, 2)
// ---------------------------------------------------------------------------

/// Simulate one Hachemeister panel: `beta_i ~ N(b, A)` with `A = L L^T`,
/// `y_ij = beta_i0 + beta_i1 t_j + eps`, `Var(eps) = s2 / w_ij`.
fn simulate_hachemeister(
    rng: &mut SplitMix64,
    times: &[f64],
    group_weights: &[f64],
    b: [f64; 2],
    chol_a: [f64; 3],
    s2: f64,
) -> Vec<Vec<f64>> {
    group_weights
        .iter()
        .map(|&w| {
            let z0 = rng.normal();
            let z1 = rng.normal();
            let beta0 = b[0] + chol_a[0] * z0;
            let beta1 = b[1] + chol_a[1] * z0 + chol_a[2] * z1;
            times
                .iter()
                .map(|&t| beta0 + beta1 * t + (s2 / w).sqrt() * rng.normal())
                .collect()
        })
        .collect()
}

fn run_hachemeister_mc(sims: usize, group_weights: &[f64]) {
    let times = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 1.0];
    // A = diag(4, 1): Cholesky [l00, l10, l11] = [2, 0, 1].
    let chol_a = [2.0, 0.0, 1.0];
    let true_a = [4.0, 0.0, 1.0];
    let s2 = 4.0;

    let mut rng = SplitMix64::new(SEED);
    let mut acc: [MeanSe; 4] = Default::default();
    for _ in 0..sims {
        let ys = simulate_hachemeister(&mut rng, &times, group_weights, b, chol_a, s2);
        let ws: Vec<Vec<f64>> = group_weights
            .iter()
            .map(|&w| vec![w; times.len()])
            .collect();
        let t_sl: Vec<&[f64]> = group_weights.iter().map(|_| times.as_slice()).collect();
        let y_sl: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
        let w_sl: Vec<&[f64]> = ws.iter().map(Vec::as_slice).collect();
        let res = hachemeister_credibility(&t_sl, &y_sl, &w_sl).unwrap();
        acc[0].push(res.vhm_matrix_raw[0]);
        acc[1].push(res.vhm_matrix_raw[1]);
        acc[2].push(res.vhm_matrix_raw[3]);
        acc[3].push(res.epv);
    }
    acc[0].assert_unbiased(true_a[0], "Hachemeister A00");
    acc[1].assert_unbiased(true_a[1], "Hachemeister A01");
    acc[2].assert_unbiased(true_a[2], "Hachemeister A11");
    acc[3].assert_unbiased(s2, "Hachemeister s^2");
}

/// Reviewer probe design: K = 3, per-observation weights {1, 1, 500}, true A = diag(4, 1).
/// The pre-fix estimator averaged ~[8.04, 2.01] on the diagonal.
#[test]
fn hachemeister_vhm_unbiased_for_unequal_designs() {
    run_hachemeister_mc(FAST_SIMS, &[1.0, 1.0, 500.0]);
}

#[test]
fn hachemeister_vhm_unbiased_for_moderate_heterogeneity() {
    run_hachemeister_mc(FAST_SIMS, &[0.5, 1.0, 2.0, 5.0, 10.0, 25.0]);
}

/// Heavier Monte Carlo confirmation (10x replications): `cargo test -- --ignored`.
#[test]
#[ignore = "slow: 40k-replication Monte Carlo confirmation of Hachemeister unbiasedness"]
fn hachemeister_vhm_unbiased_heavy() {
    run_hachemeister_mc(SLOW_SIMS, &[1.0, 1.0, 500.0]);
    run_hachemeister_mc(SLOW_SIMS, &[0.5, 1.0, 2.0, 5.0, 10.0, 25.0]);
}

fn inv2(m: [f64; 4]) -> [f64; 4] {
    let det = m[0] * m[3] - m[1] * m[2];
    [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det]
}

#[test]
fn hachemeister_collective_is_credibility_weighted() {
    let mut rng = SplitMix64::new(SEED);
    let times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = [1.0, 3.0, 8.0, 20.0, 60.0];
    let ys = simulate_hachemeister(
        &mut rng,
        &times,
        &weights,
        [100.0, 3.0],
        [8.0, 0.5, 1.5],
        25.0,
    );
    let ws: Vec<Vec<f64>> = weights.iter().map(|&w| vec![w; times.len()]).collect();
    let t_sl: Vec<&[f64]> = weights.iter().map(|_| times.as_slice()).collect();
    let y_sl: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
    let w_sl: Vec<&[f64]> = ws.iter().map(Vec::as_slice).collect();
    let res = hachemeister_credibility(&t_sl, &y_sl, &w_sl).unwrap();

    let mut sum_z = [0.0; 4];
    let mut sum_zb = [0.0; 2];
    for g in &res.groups {
        let z = g.credibility_matrix;
        for i in 0..4 {
            sum_z[i] += z[i];
        }
        sum_zb[0] += z[0] * g.wls_beta[0] + z[1] * g.wls_beta[1];
        sum_zb[1] += z[2] * g.wls_beta[0] + z[3] * g.wls_beta[1];
    }
    let inv = inv2(sum_z);
    let b0 = inv[0] * sum_zb[0] + inv[1] * sum_zb[1];
    let b1 = inv[2] * sum_zb[0] + inv[3] * sum_zb[1];
    assert!((res.collective_beta[0] - b0).abs() < 1e-9 * b0.abs().max(1.0));
    assert!((res.collective_beta[1] - b1).abs() < 1e-9 * b1.abs().max(1.0));
    // VHM must be full rank here so the credibility-weighted branch (not GLS fallback) ran.
    let a = res.vhm_matrix;
    assert!(a[0] * a[3] - a[1] * a[2] > 0.0);
}

/// Reviewer probe panel: K = 4 groups, 5 unit-weight years. Pre-fix, the PSD projection in
/// the raw (t = 0 intercept) parameterization made year-6 predictions depend on the time origin
/// (16.417 / 15.973 / 16.119 for group 2 under t = 1..5 / 2021..2025 / -2..2).
#[test]
fn hachemeister_predictions_invariant_to_time_origin() {
    let ys: Vec<Vec<f64>> = vec![
        vec![10.0, 11.5, 11.0, 13.2, 14.0],
        vec![12.0, 12.1, 13.9, 14.2, 15.5],
        vec![9.0, 10.5, 10.0, 12.5, 12.0],
        vec![11.0, 11.0, 12.5, 12.0, 13.9],
    ];
    let ws = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 3.0, 1.0],
        [2.0; 5],
        [0.5; 5],
    ];
    let y_sl: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
    let w_sl: Vec<&[f64]> = ws.iter().map(<[f64; 5]>::as_slice).collect();
    let fit = |offset: f64| {
        let ts: Vec<f64> = (1..=5).map(|t| f64::from(t) + offset).collect();
        let t_sl: Vec<&[f64]> = (0..ys.len()).map(|_| ts.as_slice()).collect();
        hachemeister_credibility(&t_sl, &y_sl, &w_sl).unwrap()
    };
    let base = fit(0.0);
    // Projection must be active here (raw A indefinite) for the test to be meaningful.
    let raw = base.vhm_matrix_raw;
    assert!(raw[0] * raw[3] - raw[1] * raw[2] < 0.0);
    for offset in [2020.0, -3.0] {
        let shifted = fit(offset);
        assert!((shifted.time_centre - (base.time_centre + offset)).abs() < 1e-9);
        assert!((shifted.epv - base.epv).abs() < 1e-9 * base.epv);
        for (g0, g1) in base.groups.iter().zip(&shifted.groups) {
            // Slopes are origin-free; predictions at matching calendar points must agree.
            assert!((g0.credibility_beta[1] - g1.credibility_beta[1]).abs() < 1e-9);
            for t in [0.0, 3.0, 6.0, 10.0] {
                let (p0, p1) = (g0.predict(t), g1.predict(t + offset));
                assert!(
                    (p0 - p1).abs() < 1e-9,
                    "offset {offset}, t {t}: {p0} vs {p1}"
                );
            }
        }
        // Reported matrices are in original coordinates: A_shift = S A S^T with S = [[1, o], [0, 1]]
        // mapping intercept-at-0 under the base origin to intercept-at-0 under the shifted one.
        let a = base.vhm_matrix;
        let o = -offset;
        let a00 = a[0] + 2.0 * o * a[1] + o * o * a[3];
        let a01 = a[1] + o * a[3];
        let b = shifted.vhm_matrix;
        let scale = a00.abs().max(1.0);
        assert!((b[0] - a00).abs() < 1e-9 * scale, "{} vs {a00}", b[0]);
        assert!((b[1] - a01).abs() < 1e-9 * scale);
        assert!((b[3] - a[3]).abs() < 1e-9);
        assert!(b.iter().any(|&v| v != 0.0));
    }
}

/// Heterogeneous designs: every group has its own time grid and per-observation exposures.
/// Checks the raw VHM / EPV moment estimators and the credibility predictions are unbiased
/// (the latter exactly, by the reflection symmetry `y -> 2 X b - y` of the estimator).
#[test]
fn hachemeister_unbiased_for_heterogeneous_designs() {
    let times: [&[f64]; 5] = [
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[0.0, 2.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[1.0, 3.0, 5.0, 7.0, 9.0],
    ];
    let weights: [&[f64]; 5] = [
        &[1.0, 2.0, 1.5, 3.0, 2.0],
        &[10.0, 12.0, 8.0, 9.0, 11.0, 10.0],
        &[0.5, 0.8, 0.6],
        &[30.0, 25.0, 40.0, 35.0],
        &[4.0, 5.0, 6.0, 5.0, 4.0],
    ];
    let b = [10.0, 1.0];
    // A = [[4, 0.6], [0.6, 1]] = L L^T with L = [[2, 0], [0.3, sqrt(0.91)]].
    let chol_a = [2.0, 0.3, 0.91_f64.sqrt()];
    let true_a = [4.0, 0.6, 1.0];
    let s2 = 4.0;
    let t_pred = 10.0;

    let mut rng = SplitMix64::new(SEED);
    let mut acc: [MeanSe; 5] = Default::default();
    for _ in 0..FAST_SIMS {
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(times.len());
        let mut truth = Vec::with_capacity(times.len());
        for (ts, ws) in times.iter().zip(&weights) {
            let z0 = rng.normal();
            let z1 = rng.normal();
            let beta0 = b[0] + chol_a[0] * z0;
            let beta1 = b[1] + chol_a[1] * z0 + chol_a[2] * z1;
            truth.push(beta0 + beta1 * t_pred);
            ys.push(
                ts.iter()
                    .zip(ws.iter())
                    .map(|(&t, &w)| beta0 + beta1 * t + (s2 / w).sqrt() * rng.normal())
                    .collect(),
            );
        }
        let y_sl: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
        let res = hachemeister_credibility(&times, &y_sl, &weights).unwrap();
        acc[0].push(res.vhm_matrix_raw[0]);
        acc[1].push(res.vhm_matrix_raw[1]);
        acc[2].push(res.vhm_matrix_raw[3]);
        acc[3].push(res.epv);
        let err: f64 = res
            .groups
            .iter()
            .zip(&truth)
            .map(|(g, &m)| g.predict(t_pred) - m)
            .sum::<f64>()
            / truth.len() as f64;
        acc[4].push(err);
    }
    acc[0].assert_unbiased(true_a[0], "heterogeneous Hachemeister A00");
    acc[1].assert_unbiased(true_a[1], "heterogeneous Hachemeister A01");
    acc[2].assert_unbiased(true_a[2], "heterogeneous Hachemeister A11");
    acc[3].assert_unbiased(s2, "heterogeneous Hachemeister s^2");
    acc[4].assert_unbiased(0.0, "heterogeneous Hachemeister prediction error");
}

// ---------------------------------------------------------------------------
// Bühlmann–Straub & Jewell Monte Carlo unbiasedness (finding 9)
// ---------------------------------------------------------------------------

#[test]
fn buhlmann_straub_epv_vhm_unbiased() {
    let mu = 50.0;
    let a: f64 = 16.0;
    let s2 = 9.0;
    // Heterogeneous exposures per group and period.
    let weights: [[f64; 4]; 6] = [
        [1.0, 2.0, 1.5, 3.0],
        [5.0, 4.0, 6.0, 5.0],
        [10.0, 12.0, 8.0, 9.0],
        [20.0, 25.0, 30.0, 15.0],
        [2.0, 2.0, 2.0, 2.0],
        [40.0, 35.0, 45.0, 50.0],
    ];
    let mut rng = SplitMix64::new(SEED);
    let mut epv = MeanSe::default();
    let mut vhm = MeanSe::default();
    let mut clamped = 0usize;
    for _ in 0..FAST_SIMS {
        let losses: Vec<Vec<f64>> = weights
            .iter()
            .map(|ws| {
                let m = mu + a.sqrt() * rng.normal();
                ws.iter()
                    .map(|&w| m + (s2 / w).sqrt() * rng.normal())
                    .collect()
            })
            .collect();
        let l_sl: Vec<&[f64]> = losses.iter().map(Vec::as_slice).collect();
        let w_sl: Vec<&[f64]> = weights.iter().map(<[f64; 4]>::as_slice).collect();
        let res = buhlmann_straub(&l_sl, &w_sl).unwrap();
        epv.push(res.epv);
        vhm.push(res.vhm);
        if res.vhm == 0.0 {
            clamped += 1;
        }
    }
    epv.assert_unbiased(s2, "Bühlmann–Straub EPV");
    // VHM is clamped at 0; this design keeps clamping negligible so the mean tests unbiasedness.
    assert!(
        clamped * 200 < FAST_SIMS,
        "VHM clamped too often: {clamped}"
    );
    vhm.assert_unbiased(a, "Bühlmann–Straub VHM");
}

#[test]
fn jewell_epv_and_within_sector_vhm_unbiased() {
    let mu = 100.0;
    let a: f64 = 25.0; // between-sector
    let v: f64 = 9.0; // within-sector
    let s2 = 16.0; // process
    let n_sectors: usize = 6;
    let n_risks = 4;
    let n_periods = 4;
    let mut rng = SplitMix64::new(SEED);
    let mut epv = MeanSe::default();
    let mut within = MeanSe::default();
    let mut between = MeanSe::default();
    for _ in 0..FAST_SIMS {
        let sectors: Vec<HierarchicalSector> = (0..n_sectors)
            .map(|p| {
                let sector_mean = mu + a.sqrt() * rng.normal();
                let risks = (0..n_risks)
                    .map(|j| {
                        let risk_mean = sector_mean + v.sqrt() * rng.normal();
                        let weights: Vec<f64> = (0..n_periods)
                            .map(|t| 1.0 + ((p + 2 * j + t) % 5) as f64)
                            .collect();
                        let losses = weights
                            .iter()
                            .map(|&w| risk_mean + (s2 / w).sqrt() * rng.normal())
                            .collect();
                        HierarchicalRisk {
                            label: None,
                            losses,
                            weights,
                        }
                    })
                    .collect();
                HierarchicalSector { label: None, risks }
            })
            .collect();
        let res = jewell_hierarchical_credibility(&sectors).unwrap();
        epv.push(res.epv);
        within.push(res.within_sector_vhm);
        between.push(res.between_sector_vhm);
    }
    epv.assert_unbiased(s2, "Jewell EPV");
    within.assert_unbiased(v, "Jewell within-sector VHM");
    // The between-sector estimator uses estimated (z-weighted) structural weights and is a
    // pseudo-estimator; only check it is in the right neighborhood.
    assert!(
        (between.mean() - a).abs() < 0.25 * a,
        "Jewell between-sector VHM mean {} vs {a}",
        between.mean()
    );
}

// ---------------------------------------------------------------------------
// Credibility properties & classical standards (findings 6, 9)
// ---------------------------------------------------------------------------

#[test]
fn classical_standards_werner_modlin() {
    // p = 90%, k = 5% -> n0 = (1.645 / 0.05)^2 = 1082.41 (Werner & Modlin Ch. 12).
    let freq = ClassicalCredibility::from_confidence_and_tolerance(1.645, 0.05).unwrap();
    assert!((freq.full_credibility_standard - 1082.41).abs() < 1e-9);
    // Pure premium standard n0 (1 + CV^2); CV_S = 2 -> 5 * 1082.41.
    let pp = ClassicalCredibility::pure_premium_standard(1.645, 0.05, 2.0).unwrap();
    assert!((pp.full_credibility_standard - 5412.05).abs() < 1e-9);
    // Severity standard n0 CV^2.
    let sev = ClassicalCredibility::severity_standard(1.645, 0.05, 2.0).unwrap();
    assert!((sev.full_credibility_standard - 4329.64).abs() < 1e-9);
    // CV = 0 collapses the pure premium standard to the frequency standard.
    let pp0 = ClassicalCredibility::pure_premium_standard(1.645, 0.05, 0.0).unwrap();
    assert_eq!(pp0, freq);

    for bad_tol in [0.0, -0.05, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ClassicalCredibility::from_confidence_and_tolerance(1.645, bad_tol),
            Err(PricingError::InvalidParameter(_))
        ));
    }
    assert!(ClassicalCredibility::from_confidence_and_tolerance(0.0, 0.05).is_err());
    assert!(ClassicalCredibility::pure_premium_standard(1.645, 0.05, -1.0).is_err());
    assert!(ClassicalCredibility::severity_standard(1.645, 0.05, 0.0).is_err());
}

#[test]
fn credibility_bounded_and_monotone_in_exposure() {
    let mut rng = SplitMix64::new(SEED);
    let cred = ClassicalCredibility::standard_90_5();
    let mut prev = 0.0;
    for i in 0..500 {
        let n = f64::from(i) * 5.0;
        let z = cred.credibility(n).unwrap();
        assert!((0.0..=1.0).contains(&z));
        assert!(z >= prev);
        prev = z;
    }

    for _ in 0..200 {
        let k = 2 + (rng.next_u64() % 6) as usize;
        let losses: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..3).map(|_| 50.0 + 20.0 * rng.normal()).collect())
            .collect();
        let weights: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..3).map(|_| 0.5 + 50.0 * rng.uniform()).collect())
            .collect();
        let l_sl: Vec<&[f64]> = losses.iter().map(Vec::as_slice).collect();
        let w_sl: Vec<&[f64]> = weights.iter().map(Vec::as_slice).collect();
        let res = buhlmann_straub(&l_sl, &w_sl).unwrap();
        let totals: Vec<f64> = weights.iter().map(|w| w.iter().sum()).collect();
        for i in 0..k {
            let zi = res.credibilities[i];
            assert!((0.0..=1.0).contains(&zi));
            for j in 0..k {
                if totals[i] < totals[j] {
                    assert!(zi <= res.credibilities[j] + 1e-15);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Retro plan balance (finding 3)
// ---------------------------------------------------------------------------

#[test]
fn retro_plan_balances_to_expense_plus_loss_times_tax() {
    let ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5];
    let probs = [0.05, 0.10, 0.15, 0.18, 0.17, 0.15, 0.10, 0.07, 0.03];
    let mean: f64 = ratios.iter().zip(&probs).map(|(r, p)| r * p).sum();
    let ratios: Vec<f64> = ratios.iter().map(|r| r / mean).collect();
    let tm = TableMEngine::from_discrete_pdf(&ratios, &probs).unwrap();

    let spec = RetroPlanSpec {
        standard_premium: 1_000_000.0,
        expected_loss_ratio: 0.65,
        lcf: 1.12,
        tax_multiplier: 1.03,
        max_retro_ratio: 1.40,
        min_retro_ratio: 0.55,
        // e: total expenses ex-taxes, including the LAE (LCF - 1) * ELR = 0.078.
        expense_ratio: 0.25,
    };
    let res = spec
        .solve_table_m(
            &tm,
            &SolverOptions::default()
                .with_tol(1e-13)
                .with_max_iter(10_000),
        )
        .unwrap();

    let p = spec.standard_premium;
    let expected_losses = spec.expected_loss_ratio * p;
    let mut e_retro = 0.0;
    for (r, pr) in ratios.iter().zip(&probs) {
        let loss = expected_losses * r;
        let raw = (res.basic_premium_dollars + spec.lcf * loss) * spec.tax_multiplier;
        let capped = raw.clamp(spec.min_retro_ratio * p, spec.max_retro_ratio * p);
        e_retro += pr * capped;
    }
    let target = (spec.expense_ratio + spec.expected_loss_ratio) * spec.tax_multiplier * p;
    assert!(
        (e_retro - target).abs() < 1e-6 * target,
        "E[R] = {e_retro}, (e + E) T P = {target}"
    );
    // Both caps must bind with positive probability for the test to exercise I = phi - psi.
    assert!(res.charge_at_max > 0.0 && res.savings_at_min > 0.0);
}

// ---------------------------------------------------------------------------
// Table M / Table L normalization and row invariants (findings 4, 7)
// ---------------------------------------------------------------------------

#[test]
fn table_m_rescales_mean_to_one_and_identity_holds() {
    let ratios = [0.2, 0.6, 1.0, 1.4, 1.8];
    let probs = [0.1, 0.2, 0.4, 0.2, 0.1];
    // Input mean 1.04 (inside the 5% band): entries are rescaled by 1 / 1.04.
    let shifted: Vec<f64> = ratios.iter().map(|r| r * 1.04).collect();
    let tm = TableMEngine::from_discrete_pdf(&shifted, &probs).unwrap();
    assert!((tm.charge_at(0.0) - 1.0).abs() < 1e-12, "phi(0) = 1");
    for row in tm.rows() {
        assert!((row.savings - (row.charge + row.entry_ratio - 1.0)).abs() < 1e-12);
    }
    let reference = TableMEngine::from_discrete_pdf(&ratios, &probs).unwrap();
    for i in 0..=40 {
        let r = f64::from(i) * 0.05;
        assert!((tm.charge_at(r) - reference.charge_at(r)).abs() < 1e-12);
        assert!(
            (tm.savings_at(r) - (tm.charge_at(r) + r - 1.0).max(0.0)).abs() < 1e-12,
            "psi = phi + r - 1"
        );
        let direct: f64 = ratios
            .iter()
            .zip(&probs)
            .map(|(y, p)| p * (r - y).max(0.0))
            .sum();
        assert!((tm.savings_at(r) - direct).abs() < 1e-12);
    }
    // Probabilities summing to 1 + 5e-5 are renormalized so phi(0) stays exactly 1.
    let probs_off: Vec<f64> = probs.iter().map(|p| p * (1.0 + 5e-5)).collect();
    let tm2 = TableMEngine::from_discrete_pdf(&ratios, &probs_off).unwrap();
    assert!((tm2.charge_at(0.0) - 1.0).abs() < 1e-12);
    // Outside the band: rejected.
    let far: Vec<f64> = ratios.iter().map(|r| r * 1.10).collect();
    assert!(TableMEngine::from_discrete_pdf(&far, &probs).is_err());
}

/// serde_json float parsing is not guaranteed bit-exact without `float_roundtrip`,
/// so compare deserialized tables on a charge grid with a ULP-level tolerance.
fn assert_close_charges(a: impl Fn(f64) -> f64, b: impl Fn(f64) -> f64) {
    for i in 0..=40 {
        let r = f64::from(i) * 0.05;
        assert!((a(r) - b(r)).abs() < 1e-12, "charge mismatch at r = {r}");
    }
}

#[test]
fn table_rows_are_validated() {
    let tm = TableMEngine::from_discrete_pdf(&[0.5, 1.5], &[0.5, 0.5]).unwrap();
    let rows = tm.rows().to_vec();
    assert_eq!(TableMEngine::from_rows(rows.clone()).unwrap(), tm);
    assert!(matches!(
        TableMEngine::from_rows(Vec::new()),
        Err(PricingError::EmptyInput(_))
    ));
    let mut unsorted = rows.clone();
    unsorted.reverse();
    assert!(TableMEngine::from_rows(unsorted).is_err());
    let bad_identity = vec![TableMRow {
        entry_ratio: 1.0,
        charge: 0.1,
        savings: 0.1,
    }];
    assert!(TableMEngine::from_rows(bad_identity).is_err());

    // Serde round trip is validated: valid JSON ok, unsorted JSON rejected.
    let json = serde_json::to_string(&tm).unwrap();
    let back: TableMEngine = serde_json::from_str(&json).unwrap();
    assert_close_charges(|r| back.charge_at(r), |r| tm.charge_at(r));
    let unsorted_json = r#"{"rows":[{"entry_ratio":1.5,"charge":0.0,"savings":0.5},{"entry_ratio":0.5,"charge":0.5,"savings":0.0}]}"#;
    assert!(serde_json::from_str::<TableMEngine>(unsorted_json).is_err());

    let tl = TableLEngine::from_limited_discrete_pdf(&[0.4, 1.4], &[0.5, 0.5]).unwrap();
    assert!((tl.loss_elimination_ratio - 0.1).abs() < 1e-12);
    assert!((tl.charge_at(0.0) - 1.0).abs() < 1e-12);
    for row in tl.rows() {
        assert!((row.savings - (row.charge + row.entry_ratio - 1.0)).abs() < 1e-12);
    }
    let rebuilt = TableLEngine::from_rows(tl.loss_elimination_ratio, tl.rows().to_vec()).unwrap();
    assert_eq!(rebuilt, tl);
    let mut tl_unsorted = tl.rows().to_vec();
    tl_unsorted.reverse();
    assert!(TableLEngine::from_rows(tl.loss_elimination_ratio, tl_unsorted).is_err());
    assert!(TableLEngine::from_rows(0.1, Vec::new()).is_err());
    let json = serde_json::to_string(&tl).unwrap();
    let back: TableLEngine = serde_json::from_str(&json).unwrap();
    assert_close_charges(|r| back.charge_at(r), |r| tl.charge_at(r));
}

// ---------------------------------------------------------------------------
// NCCI accident limitations (finding 5)
// ---------------------------------------------------------------------------

fn ncci_spec() -> NcciErpSpec {
    NcciErpSpec {
        split_point: 20_000.0,
        expected_losses: 125_000.0,
        d_ratio: 0.30,
        weight_w: 0.20,
        ballast_b: 10_000.0,
        apply_era: true,
    }
}

#[test]
fn ncci_per_claim_accident_limitation_hand_computation() {
    let spec = ncci_spec();
    let claim = [ExperienceClaim {
        incurred: 1_000_000.0,
        is_medical_only: false,
    }];
    let lim = NcciAccidentLimitation::new(250_000.0).unwrap();
    assert!((lim.multiple_claim_limit - 500_000.0).abs() < 1e-9);
    let res = calculate_ncci_experience_mod_by_accident(&spec, &[&claim], Some(&lim)).unwrap();
    // Ratable = 250,000 -> A_p = 20,000, A_e = 230,000; E_e = 87,500.
    // Mod = (20,000 + 0.2 * 230,000 + 0.8 * 87,500 + 10,000) / (125,000 + 10,000)
    //     = 146,000 / 135,000 = 1.08148... -> 1.08 (NCCI 2-dp).
    assert!((res.actual_primary - 20_000.0).abs() < 1e-9);
    assert!((res.actual_excess - 230_000.0).abs() < 1e-9);
    assert!((res.mod_factor_unrounded - 146_000.0 / 135_000.0).abs() < 1e-12);
    assert!((res.mod_factor - 1.08).abs() < 1e-12);

    // Unlimited: (20,000 + 196,000 + 70,000 + 10,000) / 135,000 = 2.1926 -> 2.19.
    let unlimited = calculate_ncci_experience_mod_by_accident(&spec, &[&claim], None).unwrap();
    assert!((unlimited.mod_factor - 2.19).abs() < 1e-12);
    let legacy = calculate_ncci_experience_mod(&spec, &claim).unwrap();
    assert_eq!(unlimited, legacy);
}

#[test]
fn ncci_multiple_claim_accident_limitation() {
    let spec = ncci_spec();
    let lim = NcciAccidentLimitation::new(250_000.0).unwrap();
    let accident = [
        ExperienceClaim {
            incurred: 300_000.0,
            is_medical_only: false,
        },
        ExperienceClaim {
            incurred: 400_000.0,
            is_medical_only: false,
        },
        ExperienceClaim {
            incurred: 260_000.0,
            is_medical_only: false,
        },
    ];
    let small = [ExperienceClaim {
        incurred: 10_000.0,
        is_medical_only: true,
    }];
    let res =
        calculate_ncci_experience_mod_by_accident(&spec, &[&accident, &small], Some(&lim)).unwrap();
    // Accident 1: 3 x 250,000 = 750,000 capped at 500,000; primary 3 x 20,000 = 60,000.
    // Accident 2: medical-only ERA -> 3,000 primary.
    assert!((res.actual_primary - 63_000.0).abs() < 1e-9);
    assert!((res.actual_excess - 440_000.0).abs() < 1e-9);

    assert!(NcciAccidentLimitation::new(0.0).is_err());
    assert!(NcciAccidentLimitation::with_multiple_claim_limit(250_000.0, 100_000.0).is_err());
    let literal = NcciAccidentLimitation {
        per_claim_limit: -1.0,
        multiple_claim_limit: 1.0,
    };
    assert!(calculate_ncci_experience_mod_by_accident(&spec, &[&small], Some(&literal)).is_err());
}

// ---------------------------------------------------------------------------
// Trending (findings 8, 9)
// ---------------------------------------------------------------------------

#[test]
fn trend_model_rate_validated() {
    assert!(TrendModel::try_new(0.05).is_ok());
    assert!(TrendModel::try_new(-0.5).is_ok());
    for bad in [-1.0, -1.5, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            TrendModel::try_new(bad),
            Err(PricingError::InvalidParameter(_))
        ));
    }
}

#[test]
fn parallelogram_identity_and_multiplicativity() {
    let mut rng = SplitMix64::new(SEED);
    for _ in 0..300 {
        let year = 2015.0 + (rng.next_u64() % 10) as f64;
        assert_eq!(parallelogram_on_level_factor(year, &[]).unwrap(), 1.0);
        assert_eq!(policy_year_on_level_factor(year, &[]).unwrap(), 1.0);

        // Fully earned (pre-period) changes leave the OLF at 1.
        let early = [RateChange {
            effective_date: year - 1.0 - 2.0 * rng.uniform(),
            rate_factor: 0.8 + 0.4 * rng.uniform(),
        }];
        assert!((parallelogram_on_level_factor(year, &early).unwrap() - 1.0).abs() < 1e-12);

        // Random in-period changes.
        let n = 1 + (rng.next_u64() % 4) as usize;
        let mut changes: Vec<RateChange> = (0..n)
            .map(|_| RateChange {
                effective_date: year - 1.0 + 2.0 * rng.uniform(),
                rate_factor: 0.85 + 0.3 * rng.uniform(),
            })
            .collect();
        let base = parallelogram_on_level_factor(year, &changes).unwrap();
        let base_py = policy_year_on_level_factor(year, &changes).unwrap();

        // Unit rate factors are no-ops.
        let mut with_unit = changes.clone();
        with_unit.push(RateChange {
            effective_date: year - 0.5 + rng.uniform(),
            rate_factor: 1.0,
        });
        assert!((parallelogram_on_level_factor(year, &with_unit).unwrap() - base).abs() < 1e-12);

        // A change effective after the period is unearned: OLF scales by exactly its factor.
        let f = 0.8 + 0.4 * rng.uniform();
        changes.push(RateChange {
            effective_date: year + 1.0 + rng.uniform(),
            rate_factor: f,
        });
        let scaled = parallelogram_on_level_factor(year, &changes).unwrap();
        assert!((scaled - f * base).abs() < 1e-12 * base.max(1.0));
        let scaled_py = policy_year_on_level_factor(year, &changes).unwrap();
        assert!((scaled_py - f * base_py).abs() < 1e-12 * base_py.max(1.0));

        // A fully earned prior change cancels (current and historical levels scale equally).
        changes.push(early[0]);
        let with_early = parallelogram_on_level_factor(year, &changes).unwrap();
        assert!((with_early - scaled).abs() < 1e-12 * scaled.max(1.0));
    }
}

// ---------------------------------------------------------------------------
// Round-2 review: Table L near-unit mean, classical credibility limits, NCCI rounding
// ---------------------------------------------------------------------------

/// `E[Y*]` within `PROBABILITY_SUM_TOL` above 1 is accepted; construction must then imply a
/// valid `from_rows` / serde round trip (pre-fix: `k` clamped to 0 left `phi*(0) = E[Y*] > 1`).
#[test]
fn table_l_mean_slightly_above_one_round_trips() {
    let tl = TableLEngine::from_limited_discrete_pdf(&[0.5, 1.5001], &[0.5, 0.5]).unwrap();
    assert_eq!(tl.loss_elimination_ratio, 0.0);
    assert!((tl.charge_at(0.0) - 1.0).abs() < 1e-12);
    let json = serde_json::to_string(&tl).unwrap();
    let back: TableLEngine = serde_json::from_str(&json).unwrap();
    // serde_json's default float parser may differ in the last ulp; compare to rounding.
    for (r0, r1) in tl.rows().iter().zip(back.rows()) {
        assert!((r0.entry_ratio - r1.entry_ratio).abs() < 1e-14);
        assert!((r0.charge - r1.charge).abs() < 1e-14);
    }
    assert!(TableLEngine::from_rows(tl.loss_elimination_ratio, tl.rows().to_vec()).is_ok());
    // Beyond the tolerance band it is still rejected.
    assert!(TableLEngine::from_limited_discrete_pdf(&[0.5, 1.6], &[0.5, 0.5]).is_err());
}

#[test]
fn classical_credibility_infinite_volume_is_full_and_invalid_is_err() {
    let cred = ClassicalCredibility::standard_90_5();
    assert_eq!(cred.credibility(f64::INFINITY).unwrap(), 1.0);
    assert_eq!(cred.blend(f64::INFINITY, 120.0, 100.0).unwrap(), 120.0);
    assert!(matches!(
        cred.credibility(f64::NAN),
        Err(PricingError::InvalidParameter(_))
    ));
    assert!(cred.credibility(-1.0).is_err());
    assert!(cred.credibility(f64::NEG_INFINITY).is_err());
    assert!(cred.blend(100.0, f64::NAN, 1.0).is_err());
    assert!(cred.blend(100.0, 1.0, f64::INFINITY).is_err());
    let bad = ClassicalCredibility {
        full_credibility_standard: 0.0,
    };
    assert!(bad.credibility(10.0).is_err());
}

/// NCCI publishes mods to two decimals, rounded half-up.
#[test]
fn ncci_mod_rounds_half_up_to_two_decimals() {
    // E = 100,000, D = 0.5, W = 0, B = 0 -> Mod = (A_p + 50,000) / 100,000; A_p = 35,500
    // gives 0.855 -> 0.86 (half-up, not banker's rounding or truncation).
    let spec = NcciErpSpec {
        split_point: 1_000_000.0,
        expected_losses: 100_000.0,
        d_ratio: 0.5,
        weight_w: 0.0,
        ballast_b: 0.0,
        apply_era: false,
    };
    let claim = [ExperienceClaim {
        incurred: 35_500.0,
        is_medical_only: false,
    }];
    let res = calculate_ncci_experience_mod(&spec, &claim).unwrap();
    assert!((res.mod_factor_unrounded - 0.855).abs() < 1e-12);
    assert_eq!(res.mod_factor, 0.86);
    let claim = [ExperienceClaim {
        incurred: 35_499.0,
        is_medical_only: false,
    }];
    let res = calculate_ncci_experience_mod(&spec, &claim).unwrap();
    assert_eq!(res.mod_factor, 0.85);
}
