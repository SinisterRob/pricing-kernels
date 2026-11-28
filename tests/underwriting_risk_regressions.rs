//! Regression and golden tests for `underwriting_risk`: grid truncation / aliasing guards,
//! circular-convolution support, ELT discretization, empirical quantile/PML conventions,
//! and hand-derived reference values (default seed: `42069`).

use pricing_kernels::error::PricingError;
use pricing_kernels::underwriting_risk::{
    AggregateLossDistribution, EmpiricalPmlConvention, EventLossEntry, FrequencyDistribution,
    YearLossTable, YltEvent, convolve_independent_aggregates, elt_aggregate_distribution,
    elt_occurrence_layer_expected_loss, elt_summary, fft_aggregate_loss, kreps_marginal_risk_load,
    panjer_recursion, sample_var, simulate_ylt_from_elt,
};

const SEED: u64 = 42069;

fn close_rel(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() <= tol * expected.abs().max(1.0)
}

/// Claim-count pmf `P(N = n)` for `n = 0..len`, from closed forms (not the Panjer recursion).
fn count_pmf(freq: FrequencyDistribution, len: usize) -> Vec<f64> {
    let mut p = vec![0.0; len];
    match freq {
        FrequencyDistribution::Poisson { lambda } => {
            p[0] = (-lambda).exp();
            for n in 1..len {
                p[n] = p[n - 1] * lambda / (n as f64);
            }
        }
        FrequencyDistribution::Binomial { m, q } => {
            let m = m as usize;
            for (n, slot) in p.iter_mut().enumerate().take(m + 1) {
                let mut binom = 1.0;
                for i in 0..n {
                    binom *= ((m - i) as f64) / ((i + 1) as f64);
                }
                *slot = binom * q.powi(n as i32) * (1.0 - q).powi((m - n) as i32);
            }
        }
        FrequencyDistribution::NegativeBinomial { r, beta } => {
            // P(N = n) = Gamma(r + n) / (Gamma(r) n!) * (1 / (1 + beta))^r * (beta / (1 + beta))^n
            let ratio = beta / (1.0 + beta);
            p[0] = (1.0 + beta).powf(-r);
            for n in 1..len {
                p[n] = p[n - 1] * (r + (n as f64) - 1.0) / (n as f64) * ratio;
            }
        }
    }
    p
}

/// Brute-force compound pmf `sum_n P(N = n) f^{*n}` truncated to `grid` points.
fn brute_force_compound(freq: FrequencyDistribution, sev: &[f64], grid: usize) -> Vec<f64> {
    let max_n = 400;
    let pn = count_pmf(freq, max_n);
    let mut conv = vec![0.0; grid];
    conv[0] = 1.0; // f^{*0} = delta_0
    let mut out = vec![0.0; grid];
    for &p in &pn {
        for (o, &c) in out.iter_mut().zip(conv.iter()) {
            *o += p * c;
        }
        let mut next = vec![0.0; grid];
        for (i, &c) in conv.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            for (j, &f) in sev.iter().enumerate() {
                if i + j < grid {
                    next[i + j] += c * f;
                }
            }
        }
        conv = next;
    }
    out
}

// ---------------------------------------------------------------------------
// Finding 7: Panjer / FFT golden values vs independent brute-force convolution
// ---------------------------------------------------------------------------

#[test]
fn panjer_poisson_hand_values() {
    // Poisson(2), f = (0, 0.5, 0.3, 0.2):
    // g0 = e^-2; g1 = 2 * 0.5 e^-2 = e^-2; g2 = (2 * 0.3 + 2 * 0.25) e^-2 = 1.1 e^-2;
    // g3 = (2 * 0.2 + 2 * 2 * 0.5 * 0.3 + (8/6) * 0.125) e^-2 = (7/6) e^-2.
    let agg = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 2.0 },
        &[0.0, 0.5, 0.3, 0.2],
        1.0,
        64,
    )
    .expect("panjer");
    let e2 = (-2.0_f64).exp();
    assert!((agg.pmf[0] - e2).abs() < 1e-15);
    assert!((agg.pmf[1] - e2).abs() < 1e-15);
    assert!((agg.pmf[2] - 1.1 * e2).abs() < 1e-15);
    assert!((agg.pmf[3] - 7.0 / 6.0 * e2).abs() < 1e-15);
    assert!(close_rel(agg.mean, 2.0 * 1.7, 1e-12));
}

#[test]
fn panjer_and_fft_match_brute_force_convolution_for_ab0_class() {
    let sev = [0.1, 0.45, 0.25, 0.2];
    let grid = 128;
    for freq in [
        FrequencyDistribution::Poisson { lambda: 2.0 },
        FrequencyDistribution::Binomial { m: 8, q: 0.35 },
        FrequencyDistribution::NegativeBinomial { r: 2.5, beta: 1.2 },
    ] {
        let reference = brute_force_compound(freq, &sev, grid);
        let panjer = panjer_recursion(freq, &sev, 1.0, grid).expect("panjer");
        let fft = fft_aggregate_loss(freq, &sev, 1.0, grid, Some(0.05)).expect("fft");
        for k in 0..grid {
            assert!(
                (panjer.pmf[k] - reference[k]).abs() < 1e-14,
                "{freq:?} panjer k={k}: {} vs {}",
                panjer.pmf[k],
                reference[k]
            );
            assert!(
                (fft.pmf[k] - reference[k]).abs() < 1e-12,
                "{freq:?} fft k={k}: {} vs {}",
                fft.pmf[k],
                reference[k]
            );
        }
        let ex: f64 = sev.iter().enumerate().map(|(k, &p)| k as f64 * p).sum();
        let exact_mean = freq.mean().expect("mean") * ex;
        assert!(close_rel(panjer.mean, exact_mean, 1e-12));
        assert!(close_rel(fft.mean, exact_mean, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// Finding 3: grid truncation / aliasing guards and VaR/TVaR consistency
// ---------------------------------------------------------------------------

fn point_mass_severity(bin: usize) -> Vec<f64> {
    let mut sev = vec![0.0; bin + 1];
    sev[bin] = 1.0;
    sev
}

#[test]
fn panjer_rejects_truncated_grid() {
    // Poisson(10), severity 10: retained mass on 64 bins is ~0.13.
    let freq = FrequencyDistribution::Poisson { lambda: 10.0 };
    let sev = point_mass_severity(10);
    let err = panjer_recursion(freq, &sev, 1.0, 64);
    assert!(matches!(err, Err(PricingError::OutOfBounds(_))), "{err:?}");

    let ok = panjer_recursion(freq, &sev, 1.0, 512).expect("adequate grid");
    assert!((1.0 - ok.total_mass()).abs() < 1e-12);
    assert!(close_rel(ok.mean, 100.0, 1e-10));
    let var = ok.var(0.99).expect("var");
    let tvar = ok.tvar(0.99).expect("tvar");
    assert!(tvar >= var, "tvar={tvar} < var={var}");
}

#[test]
fn fft_rejects_aliasing_with_and_without_tilt() {
    let freq = FrequencyDistribution::Poisson { lambda: 10.0 };
    let sev = point_mass_severity(10);
    let untilted = fft_aggregate_loss(freq, &sev, 1.0, 64, None);
    assert!(
        matches!(untilted, Err(PricingError::OutOfBounds(_))),
        "{untilted:?}"
    );
    let tilted = fft_aggregate_loss(freq, &sev, 1.0, 64, Some(0.05));
    assert!(
        matches!(tilted, Err(PricingError::OutOfBounds(_))),
        "{tilted:?}"
    );
    let ok = fft_aggregate_loss(freq, &sev, 1.0, 512, Some(0.01)).expect("adequate fft grid");
    assert!(close_rel(ok.mean, 100.0, 1e-9));
}

#[test]
fn var_tvar_error_beyond_retained_mass_and_tvar_dominates_var() {
    let truncated = AggregateLossDistribution::from_pmf(100.0, vec![0.3, 0.3, 0.2]).expect("pmf");
    assert!((truncated.total_mass() - 0.8).abs() < 1e-15);
    assert_eq!(truncated.var(0.5).expect("var 0.5"), 100.0);
    assert!(matches!(
        truncated.var(0.9),
        Err(PricingError::OutOfBounds(_))
    ));
    assert!(matches!(
        truncated.tvar(0.5),
        Err(PricingError::OutOfBounds(_))
    ));

    let agg = panjer_recursion(
        FrequencyDistribution::NegativeBinomial { r: 2.5, beta: 1.2 },
        &[0.0, 0.5, 0.3, 0.2],
        100.0,
        256,
    )
    .expect("panjer");
    for k in 1..1000 {
        let alpha = f64::from(k) / 1000.0;
        let var = agg.var(alpha).expect("var");
        let tvar = agg.tvar(alpha).expect("tvar");
        assert!(tvar >= var, "alpha={alpha}: tvar={tvar} < var={var}");
    }
}

// ---------------------------------------------------------------------------
// Finding 1: convolution support / wrap-around
// ---------------------------------------------------------------------------

#[test]
fn convolution_rejects_wraparound_and_trims_trailing_zeros() {
    let mass_at_100 =
        AggregateLossDistribution::from_pmf(1.0, point_mass_severity(100)).expect("pmf");
    let err = convolve_independent_aggregates(&[&mass_at_100, &mass_at_100], 128);
    assert!(
        matches!(err, Err(PricingError::InvalidParameter(_))),
        "{err:?}"
    );
    let ok = convolve_independent_aggregates(&[&mass_at_100, &mass_at_100], 256).expect("conv");
    assert!(close_rel(ok.mean, 200.0, 1e-12));
    assert!((ok.pmf[200] - 1.0).abs() < 1e-12);

    // Trailing zeros do not count toward the required support.
    let mut padded = vec![0.0; 300];
    padded[0] = 0.5;
    padded[3] = 0.5;
    let padded = AggregateLossDistribution::from_pmf(1.0, padded).expect("pmf");
    let small = convolve_independent_aggregates(&[&padded, &padded], 8).expect("conv");
    assert!(close_rel(small.mean, 3.0, 1e-12));
    assert!((small.pmf[6] - 0.25).abs() < 1e-14);
}

// ---------------------------------------------------------------------------
// Finding 4: ELT severity grid bounds and mean-preserving Beta discretization
// ---------------------------------------------------------------------------

fn sample_elt() -> [EventLossEntry; 2] {
    [
        EventLossEntry {
            event_id: 1,
            annual_rate: 0.05,
            mean_loss: 2_000_000.0,
            std_dev_indep: 400_000.0,
            std_dev_corr: 300_000.0,
            max_loss: 10_000_000.0,
        },
        EventLossEntry {
            event_id: 2,
            annual_rate: 0.02,
            mean_loss: 6_000_000.0,
            std_dev_indep: 0.0,
            std_dev_corr: 0.0,
            max_loss: 6_000_000.0,
        },
    ]
}

#[test]
fn elt_aggregate_rejects_losses_beyond_grid() {
    // 16M grid (62,500 * 256): point mass at 50M must not be clamped into the last bin.
    let point = [EventLossEntry {
        event_id: 9,
        annual_rate: 0.5,
        mean_loss: 50_000_000.0,
        std_dev_indep: 0.0,
        std_dev_corr: 0.0,
        max_loss: 50_000_000.0,
    }];
    let err = elt_aggregate_distribution(&point, 62_500.0, 256, Some(0.01));
    assert!(matches!(err, Err(PricingError::OutOfBounds(_))), "{err:?}");

    let beta = [EventLossEntry {
        event_id: 10,
        annual_rate: 0.5,
        mean_loss: 5_000_000.0,
        std_dev_indep: 2_000_000.0,
        std_dev_corr: 0.0,
        max_loss: 50_000_000.0,
    }];
    let err = elt_aggregate_distribution(&beta, 62_500.0, 256, Some(0.01));
    assert!(matches!(err, Err(PricingError::OutOfBounds(_))), "{err:?}");
}

#[test]
fn elt_aggregate_mean_matches_aal_exactly() {
    let entries = sample_elt();
    let aal = elt_summary(&entries).expect("summary").average_annual_loss;
    let aep = elt_aggregate_distribution(&entries, 250_000.0, 256, Some(0.01)).expect("aep");
    assert!(
        close_rel(aep.mean, aal, 1e-9),
        "aep mean={}, aal={aal}",
        aep.mean
    );
    assert!((1.0 - aep.total_mass()).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Finding 5: integer-safe empirical quantile rank
// ---------------------------------------------------------------------------

#[test]
fn sample_var_is_integer_safe() {
    let samples: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(sample_var(&samples, 0.28).expect("var"), 28.0);
    for k in 1..100 {
        let alpha = f64::from(k) / 100.0;
        assert_eq!(sample_var(&samples, alpha).expect("var"), f64::from(k));
    }
    let big: Vec<f64> = (1..=1000).map(f64::from).collect();
    for k in 1..1000 {
        let alpha = f64::from(k) * 0.001;
        assert_eq!(sample_var(&big, alpha).expect("var"), f64::from(k));
    }
    // Genuinely fractional ranks still round up: ceil(0.285 * 100) = 29.
    assert_eq!(sample_var(&samples, 0.285).expect("var"), 29.0);
}

// ---------------------------------------------------------------------------
// Findings 6 & 7: exact OEP / AEP / PML / AggXL on hand-constructed YLTs
// ---------------------------------------------------------------------------

fn hand_ylt() -> YearLossTable {
    let raw: [(u32, f64); 12] = [
        (0, 5.0),
        (0, 3.0),
        (1, 12.0),
        (3, 2.0),
        (3, 2.0),
        (3, 2.0),
        (4, 8.0),
        (5, 1.0),
        (6, 20.0),
        (7, 4.0),
        (7, 9.0),
        (9, 6.0),
    ];
    let events = raw
        .iter()
        .enumerate()
        .map(|(i, &(year, loss))| YltEvent {
            year,
            event_id: i as u64,
            loss,
        })
        .collect();
    YearLossTable::try_new(10, events).expect("ylt")
}

#[test]
fn hand_ylt_exceedance_pml_and_aggregate_xl() {
    let ylt = hand_ylt();
    // Annual occurrence maxima: [5, 12, 0, 2, 8, 1, 20, 9, 0, 6]
    // Annual aggregates:        [8, 12, 0, 6, 8, 1, 20, 13, 0, 6]
    assert!(close_rel(ylt.aal(), 7.4, 1e-15));
    assert_eq!(ylt.oep(5.0).expect("oep"), 0.5);
    assert_eq!(ylt.aep(5.0).expect("aep"), 0.7);
    assert_eq!(ylt.oep(8.0).expect("oep"), 0.3);
    assert_eq!(ylt.aep(8.0).expect("aep"), 0.3);

    // Rank convention: rank r <-> return period 10 / r.
    assert_eq!(ylt.oep_pml(10.0).expect("pml"), 20.0);
    assert_eq!(ylt.oep_pml(5.0).expect("pml"), 12.0);
    assert_eq!(ylt.oep_pml(3.0).expect("pml"), 9.0); // r = floor(10/3) = 3
    assert_eq!(ylt.oep_pml(2.0).expect("pml"), 6.0);
    assert_eq!(ylt.aep_pml(5.0).expect("pml"), 13.0);
    assert_eq!(ylt.aep_pml(2.0).expect("pml"), 8.0);
    assert!(matches!(
        ylt.oep_pml(20.0),
        Err(PricingError::OutOfBounds(_))
    ));
    // Lower-quantile convention sits one rank lower at integer N / T.
    assert_eq!(
        ylt.oep_pml_with(10.0, EmpiricalPmlConvention::LowerQuantile)
            .expect("pml"),
        12.0
    );

    // 5 xs 3 per occurrence, AAD 2, AAL 6:
    // annual subject [2, 5, 0, 0, 5, 0, 5, 6, 0, 3] -> ceded [0, 3, 0, 0, 3, 0, 3, 4, 0, 1] = 14 / 10.
    let ceded = ylt
        .aggregate_xl_expected_loss(3.0, 5.0, 2.0, 6.0)
        .expect("agg xl");
    assert!(close_rel(ceded, 1.4, 1e-15));
}

#[test]
fn rank_based_pml_on_hundred_year_ladder() {
    let events: Vec<YltEvent> = (0..100)
        .map(|y| YltEvent {
            year: y,
            event_id: u64::from(y),
            loss: f64::from(y + 1),
        })
        .collect();
    let ylt = YearLossTable::try_new(100, events).expect("ylt");
    assert_eq!(ylt.oep_pml(100.0).expect("pml"), 100.0);
    assert_eq!(ylt.oep_pml(50.0).expect("pml"), 99.0);
    assert_eq!(ylt.oep_pml(10.0).expect("pml"), 91.0);
    assert_eq!(ylt.aep_pml(100.0).expect("pml"), 100.0);
    assert_eq!(
        ylt.oep_pml_with(100.0, EmpiricalPmlConvention::LowerQuantile)
            .expect("pml"),
        99.0
    );
}

// ---------------------------------------------------------------------------
// Finding 7: Kreps (1990) risk load against hand-computed numbers
// ---------------------------------------------------------------------------

#[test]
fn kreps_risk_load_hand_computed() {
    // sigma_S = 300, Var(X) = 5625, Cov = 5000: sigma_new = sqrt(90000 + 5625 + 10000) = 325.
    // z = 2 -> marginal surplus 50; y = 0.25 -> y / (1 + y) = 0.2 -> risk load 10.
    // Linearized: 0.2 * 2 * 15625 / 600 = 125 / 12.
    let res = kreps_marginal_risk_load(300.0, 50.0, 5_625.0, 5_000.0, 2.0, 0.25).expect("kreps");
    assert!((res.portfolio_std_dev_after - 325.0).abs() < 1e-12);
    assert!((res.marginal_surplus - 50.0).abs() < 1e-12);
    assert!((res.risk_load - 10.0).abs() < 1e-12);
    assert!((res.linearized_risk_load - 125.0 / 12.0).abs() < 1e-12);
    assert!((res.risk_adjusted_price - 60.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Finding 2: exact Poisson sampling through the public YLT simulator
// ---------------------------------------------------------------------------

#[test]
fn simulated_annual_counts_have_poisson_skewness_at_high_rate() {
    // Single deterministic unit-loss event at rate 40: annual aggregate == annual count.
    let elt = [EventLossEntry {
        event_id: 1,
        annual_rate: 40.0,
        mean_loss: 1.0,
        std_dev_indep: 0.0,
        std_dev_corr: 0.0,
        max_loss: 1.0,
    }];
    let n_years = 200_000_u32;
    let ylt = simulate_ylt_from_elt(&elt, n_years, SEED).expect("ylt");
    let counts = ylt.annual_aggregate_losses();
    let n = f64::from(n_years);
    let mean = counts.iter().sum::<f64>() / n;
    let m2 = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
    let m3 = counts.iter().map(|c| (c - mean).powi(3)).sum::<f64>() / n;
    let skew = m3 / m2.powf(1.5);
    // n = 200k: SE(mean) = sqrt(40 / n) ~ 0.014, SE(var) ~ sqrt(2 * 40^2 / n) ~ 0.13,
    // SE(skew) ~ sqrt(6 / n) ~ 0.0055 (exact skew 1/sqrt(40) = 0.158); tolerances ~4 SE.
    assert!((mean - 40.0).abs() < 0.06, "mean={mean}");
    assert!((m2 - 40.0).abs() < 0.6, "var={m2}");
    assert!((skew - 1.0 / 40.0_f64.sqrt()).abs() < 0.022, "skew={skew}");
}

// ---------------------------------------------------------------------------
// Round 2: tilted-FFT round-off floor, input-mass vs truncation, TVaR tail coverage,
// convolution component mass
// ---------------------------------------------------------------------------

#[test]
fn tilted_fft_matches_panjer_across_standard_tilts() {
    // Standard Robertson / Embrechts–Frei tilts theta * N in {5, 10, 20, 30}: untilting
    // amplifies round-off by exp(theta * k), which must be floored, not reported as aliasing.
    let n = 4096_usize;
    let severities: [&[f64]; 2] = [
        &[0.0, 0.5, 0.3, 0.2],
        &[0.05, 0.2, 0.2, 0.15, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02],
    ];
    for sev in severities {
        for freq in [
            FrequencyDistribution::Poisson { lambda: 5.0 },
            FrequencyDistribution::Binomial { m: 20, q: 0.3 },
            FrequencyDistribution::NegativeBinomial { r: 3.0, beta: 2.0 },
        ] {
            let panjer = panjer_recursion(freq, sev, 1.0, n).expect("panjer");
            for theta_n in [5.0, 10.0, 20.0, 30.0] {
                let theta = theta_n / (n as f64);
                let fft = fft_aggregate_loss(freq, sev, 1.0, n, Some(theta))
                    .unwrap_or_else(|e| panic!("{freq:?} theta*N={theta_n}: {e}"));
                let max_err = panjer
                    .pmf
                    .iter()
                    .zip(&fft.pmf)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_err <= 1e-10,
                    "{freq:?} theta*N={theta_n}: max |fft - panjer| = {max_err:e}"
                );
                assert!(fft.pmf.iter().all(|&p| p >= 0.0));
            }
        }
    }
}

#[test]
fn excessive_tilt_reports_round_off_not_aliasing() {
    // theta * N = 600 amplifies round-off by up to e^600: the error must advise reducing tilt.
    let n = 4096_usize;
    let err = fft_aggregate_loss(
        FrequencyDistribution::Poisson { lambda: 200.0 },
        &[0.0, 0.5, 0.3, 0.2],
        1.0,
        n,
        Some(600.0 / (n as f64)),
    );
    match err {
        Err(PricingError::OutOfBounds(msg)) => {
            assert!(msg.contains("reduce tilt"), "{msg}");
        }
        other => panic!("expected round-off OutOfBounds, got {other:?}"),
    }
}

#[test]
fn severity_input_mass_deficit_is_rejected_as_input_error() {
    // Sum 0.99995: previously accepted, then misreported as grid truncation forever.
    let sev = [0.0, 0.5 - 5e-5, 0.3, 0.2];
    let freq = FrequencyDistribution::Poisson { lambda: 2.0 };
    for res in [
        panjer_recursion(freq, &sev, 1.0, 1 << 12),
        fft_aggregate_loss(freq, &sev, 1.0, 1 << 12, None),
    ] {
        match res {
            Err(PricingError::InvalidParameter(msg)) => {
                assert!(msg.contains("input mass"), "{msg}");
            }
            other => panic!("expected input-mass InvalidParameter, got {other:?}"),
        }
    }
    // A rounding-level deficit (1e-9) is renormalized: unit mass and exact compound mean.
    let sev_rounded = [0.0, 0.5 - 1e-9, 0.3, 0.2];
    let agg = panjer_recursion(freq, &sev_rounded, 1.0, 256).expect("panjer");
    let total = 1.0 - 1e-9;
    let ex = (0.5 - 1e-9 + 0.6 + 0.6) / total;
    assert!((agg.total_mass() - 1.0).abs() < 1e-13);
    assert!(close_rel(agg.mean, 2.0 * ex, 1e-12));
}

#[test]
fn tvar_rejects_poorly_represented_tail() {
    // Lost mass 9e-7 (< AGGREGATE_GRID_TOLERANCE) is 9% of the 1e-5 tail at alpha = 0.99999.
    let mut pmf = vec![0.0; 101];
    pmf[0] = 1.0 - 1e-5;
    pmf[100] = 1e-5 - 9e-7;
    let d = AggregateLossDistribution::from_pmf(1.0, pmf).expect("pmf");
    assert!(matches!(d.tvar(0.99999), Err(PricingError::OutOfBounds(_))));
    // At alpha = 0.99 the same lost mass is 9e-5 of the tail (< 1e-3): accepted, lower bound.
    let tvar = d.tvar(0.99).expect("tvar 0.99");
    let exact_lower = 100.0 * (1e-5 - 9e-7) / (1e-2 - 9e-7);
    assert!(close_rel(tvar, exact_lower, 1e-12), "tvar={tvar}");
}

#[test]
fn convolution_rejects_defective_components() {
    let defective = AggregateLossDistribution::from_pmf(1.0, vec![0.5, 0.3]).expect("pmf");
    let res = convolve_independent_aggregates(&[&defective, &defective], 8);
    assert!(
        matches!(res, Err(PricingError::InvalidParameter(_))),
        "{res:?}"
    );
}

// ---------------------------------------------------------------------------
// Round 2: exact cat-layer Beta LEV and log-space Beta sampler
// ---------------------------------------------------------------------------

#[test]
fn cat_layer_expected_loss_uses_exact_beta_lev() {
    // Concentrated Beta (a, b ~ 1.25e7): the survival is a near-step at 5.03M, which a fixed
    // 128-point midpoint rule cannot resolve.
    let entry = EventLossEntry {
        event_id: 1,
        annual_rate: 1.0,
        mean_loss: 5_030_000.0,
        std_dev_indep: 1_000.0,
        std_dev_corr: 0.0,
        max_loss: 10_000_000.0,
    };
    // 10M xs 0 covers the full support: E[X ^ M] - E[X ^ 0] = E[X] exactly.
    let full =
        pricing_kernels::layers::CommercialLayer::new(0.0, 10_000_000.0, 1.0).expect("layer");
    let el = elt_occurrence_layer_expected_loss(&[entry], &full).expect("el");
    assert!((el - 5_030_000.0).abs() <= 1e-9 * 5_030_000.0, "el={el}");
    // 2M xs 4M: E[X ^ 6M] - E[X ^ 4M] = 1.03M up to Gaussian tails beyond ~1000 SD (< 1e-300).
    let mid = pricing_kernels::layers::CommercialLayer::new(4_000_000.0, 2_000_000.0, 1.0)
        .expect("layer");
    let el = elt_occurrence_layer_expected_loss(&[entry], &mid).expect("el");
    assert!((el - 1_030_000.0).abs() <= 1e-9 * 1_030_000.0, "el={el}");
}

#[test]
fn ylt_beta_sampler_is_exact_for_tiny_shapes() {
    // a = b = 2e-4: X / M is ~Bernoulli(1/2) on {0, 1} with P(0.01 < U < 0.99) = 9.19e-4.
    // The old sampler underflowed both gamma draws and returned 0.5 * M for ~74% of events.
    let sd = 0.5 / (1.0 + 4e-4_f64).sqrt();
    let entry = EventLossEntry {
        event_id: 1,
        annual_rate: 10.0,
        mean_loss: 0.5,
        std_dev_indep: sd,
        std_dev_corr: 0.0,
        max_loss: 1.0,
    };
    let (a, b) = entry.beta_params().expect("params").expect("beta");
    assert!((a - 2e-4).abs() < 1e-12 && (b - 2e-4).abs() < 1e-12);
    let exact_mid = entry.survival(0.01).expect("s") - entry.survival(0.99).expect("s");
    // Leading-order check of the reference: a * ln(99) (exact to O(a^2)).
    assert!(
        close_rel(exact_mid, 2e-4 * 99.0_f64.ln(), 2e-3),
        "{exact_mid}"
    );

    let ylt = simulate_ylt_from_elt(&[entry], 100_000, SEED).expect("ylt");
    let events = ylt.events();
    let n = events.len() as f64;
    assert!(n > 990_000.0, "n={n}");
    let mid = events
        .iter()
        .filter(|e| e.loss > 0.01 && e.loss < 0.99)
        .count() as f64
        / n;
    // Binomial SE sqrt(p / n) ~ 3.0e-5: 4 SE.
    let se = (exact_mid / n).sqrt();
    assert!(
        (mid - exact_mid).abs() < 4.0 * se,
        "mid={mid} exact={exact_mid}"
    );
    // Mean 0.5, SD ~0.5: SE ~5e-4.
    let mean = events.iter().map(|e| e.loss).sum::<f64>() / n;
    assert!((mean - 0.5).abs() < 2e-3, "mean={mean}");
    assert!(
        events
            .iter()
            .all(|e| e.loss.is_finite() && (0.0..=1.0).contains(&e.loss))
    );
}
