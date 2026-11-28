//! Deterministic seeded property-invariant, extreme-regime, and bridge tests
//! for `pricing-kernels` (default seed: `42069`, zero external test-framework deps).

use pricing_kernels::prelude::*;

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

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[test]
fn severity_and_evt_lev_derivative_equals_survival_and_concavity() {
    let sevs = vec![
        SeverityDistribution::Lognormal {
            mu: 9.5,
            sigma: 1.4,
        },
        SeverityDistribution::Lomax {
            alpha: 1.0 + 1e-9,
            theta: 50_000.0,
        },
        SeverityDistribution::Lomax {
            alpha: 2.3,
            theta: 75_000.0,
        },
        SeverityDistribution::SingleParameterPareto {
            alpha: 1.8,
            x_m: 10_000.0,
        },
        SeverityDistribution::Exponential { lambda: 1e-4 },
        SeverityDistribution::MixedExponential {
            weights: vec![0.6, 0.3, 0.1],
            means: vec![20_000.0, 120_000.0, 800_000.0],
        },
        SeverityDistribution::GeneralizedPareto {
            xi: 0.35,
            beta: 40_000.0,
        },
        SeverityDistribution::Gamma {
            shape: 2.5,
            scale: 25_000.0,
        },
        SeverityDistribution::SplicedGammaPareto {
            shape: 2.0,
            scale: 15_000.0,
            splice: 50_000.0,
            tail_alpha: 2.2,
        },
    ];

    for dist in &sevs {
        let l1 = 30_000.0;
        let l2 = 90_000.0;
        let l3 = 150_000.0;
        let e1 = dist.limited_expected_value(l1).expect("e1");
        let e2 = dist.limited_expected_value(l2).expect("e2");
        let e3 = dist.limited_expected_value(l3).expect("e3");
        assert!(0.0 <= e1 && e1 <= e2 && e2 <= e3 && e3 <= l3 + 1e-9);
        // Concavity: diminishing marginal expected loss per equal limit step.
        assert!((e3 - e2) <= (e2 - e1) + 1e-6);

        // Fundamental theorem: d/dL E[X ^ L] == S(L).
        let h = 1e-2;
        let deriv = (dist.limited_expected_value(l2 + h).expect("hi")
            - dist.limited_expected_value(l2 - h).expect("lo"))
            / (2.0 * h);
        let surv = dist.survival(l2).expect("surv");
        assert!(
            (deriv - surv).abs() < 1e-5,
            "dLEV/dL={deriv}, S(L)={surv} for {dist:?}"
        );
    }

    // Overflow-free Lognormal LEV even when sigma = 40 (E[X] = exp(805) overflows f64::MAX).
    let extreme_ln = SeverityDistribution::Lognormal {
        mu: 5.0,
        sigma: 40.0,
    };
    let lev_ext = extreme_ln
        .limited_expected_value(100_000.0)
        .expect("extreme lognormal LEV should not overflow");
    assert!(lev_ext.is_finite() && lev_ext > 0.0 && lev_ext <= 100_000.0);

    // FittedEvtDistribution LEV, ILF, layer_expected_loss, and TryFrom bridge.
    let evt_dists = vec![
        FittedEvtDistribution::Lognormal {
            mu: 9.0,
            sigma: 1.1,
        },
        FittedEvtDistribution::ParetoII {
            alpha: 2.1,
            lambda: 50_000.0,
        },
        FittedEvtDistribution::Gpd {
            threshold: 0.0,
            xi: 0.25,
            beta: 30_000.0,
        },
        FittedEvtDistribution::BurrXii {
            alpha: 1.5,
            gamma: 1.8,
            lambda: 1e8,
        },
        FittedEvtDistribution::Weibull {
            k: 1.4,
            lambda: 40_000.0,
        },
        FittedEvtDistribution::Gamma {
            alpha: 2.0,
            beta: 1.0 / 25_000.0,
        },
        FittedEvtDistribution::InverseGaussian {
            mu: 50_000.0,
            lambda: 80_000.0,
        },
        FittedEvtDistribution::SplicedBodyGpd {
            body: SpliceBodyFamily::Lognormal,
            threshold: 50_000.0,
            continuity: SpliceContinuity::C0Continuous,
            body_param1: 9.8,
            body_param2: 0.9,
            body_weight: 0.85,
            xi: 0.2,
            beta: 35_000.0,
        },
    ];
    for ed in &evt_dists {
        let e1 = ed.limited_expected_value(40_000.0).expect("evt e1");
        let e2 = ed.limited_expected_value(120_000.0).expect("evt e2");
        assert!(e1 > 0.0 && e2 >= e1 && e2 <= 120_000.0 + 1e-6);
        let ilf = ed
            .increased_limits_factor(120_000.0, 40_000.0)
            .expect("ilf");
        assert!((ilf - e2 / e1).abs() < 1e-10);
        let layer = ed.layer_expected_loss(40_000.0, 80_000.0).expect("layer");
        assert!((layer - (e2 - e1)).abs() < 1e-8);
    }

    // Bridge TryFrom<FittedEvtDistribution> for SeverityDistribution.
    let bridged = SeverityDistribution::try_from(FittedEvtDistribution::ParetoII {
        alpha: 2.5,
        lambda: 60_000.0,
    })
    .expect("bridge ParetoII");
    assert_eq!(
        bridged,
        SeverityDistribution::Lomax {
            alpha: 2.5,
            theta: 60_000.0
        }
    );
    assert!(
        SeverityDistribution::try_from(FittedEvtDistribution::Weibull {
            k: 1.5,
            lambda: 100.0
        })
        .is_err()
    );
}

#[test]
fn table_m_and_table_l_identities_hold_across_seeded_random_pmfs() {
    let mut rng = SplitMix64::new(42069);
    for _ in 0..25 {
        let k = 8usize;
        let mut raw_x = Vec::with_capacity(k);
        let mut raw_p = Vec::with_capacity(k);
        for _ in 0..k {
            raw_x.push(0.1 + 3.0 * rng.next_f64());
            raw_p.push(0.05 + rng.next_f64());
        }
        let sum_p: f64 = raw_p.iter().sum();
        for p in &mut raw_p {
            *p /= sum_p;
        }
        let mean_x: f64 = raw_x.iter().zip(raw_p.iter()).map(|(x, p)| x * p).sum();
        let entry_ratios: Vec<f64> = raw_x.iter().map(|&x| x / mean_x).collect();

        let tm = TableMEngine::from_discrete_pdf(&entry_ratios, &raw_p).expect("table m");
        let limited: Vec<f64> = entry_ratios.iter().map(|&r| (0.85 * r).min(1.5)).collect();
        let tl = TableLEngine::from_limited_discrete_pdf(&limited, &raw_p).expect("table l");

        let grid = [0.0, 0.25, 0.6, 1.0, 1.5, 2.5, 4.0];
        for &r in &grid {
            // Lee identity: phi(r) - psi(r) == 1 - r
            assert!(((tm.charge_at(r) - tm.savings_at(r)) - (1.0 - r)).abs() < 1e-11);
            // Skurnick identity: phi*(r) - psi*(r) == 1 - r
            assert!(((tl.charge_at(r) - tl.savings_at(r)) - (1.0 - r)).abs() < 1e-11);
        }
        // Convexity of phi(r): midpoint <= average of endpoints.
        let mid = tm.charge_at(1.0);
        assert!(mid <= f64::midpoint(tm.charge_at(0.5), tm.charge_at(1.5)) + 1e-12);
    }
}

#[test]
fn capital_allocation_permutation_invariance_and_additivity() {
    let mut rng = SplitMix64::new(42069);
    let n = 400usize;
    let mut l1 = Vec::with_capacity(n);
    let mut l2 = Vec::with_capacity(n);
    let mut l3 = Vec::with_capacity(n);
    for _ in 0..n {
        let common = rng.next_f64();
        l1.push(50.0 + 100.0 * common + 20.0 * rng.next_f64());
        l2.push(30.0 + 80.0 * common + 40.0 * rng.next_f64());
        l3.push(20.0 + 40.0 * rng.next_f64());
    }

    // Create a deterministic Fisher-Yates shuffled copy of scenarios.
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        perm.swap(i, j);
    }
    let s1: Vec<f64> = perm.iter().map(|&i| l1[i]).collect();
    let s2: Vec<f64> = perm.iter().map(|&i| l2[i]).collect();
    let s3: Vec<f64> = perm.iter().map(|&i| l3[i]).collect();

    let m_orig = ScenarioMatrix::try_new(
        vec!["L1".into(), "L2".into(), "L3".into()],
        vec![l1.clone(), l2.clone(), l3.clone()],
    )
    .expect("orig");
    let m_shuf = ScenarioMatrix::try_new(
        vec!["L1".into(), "L2".into(), "L3".into()],
        vec![s1.clone(), s2.clone(), s3.clone()],
    )
    .expect("shuf");

    let e1 = m_orig.euler_tvar(0.90).expect("euler orig");
    let e2 = m_shuf.euler_tvar(0.90).expect("euler shuf");
    assert!((e1.portfolio_risk_measure - e2.portfolio_risk_measure).abs() < 1e-10);
    for i in 0..3 {
        assert!((e1.allocated_total[i] - e2.allocated_total[i]).abs() < 1e-10);
    }
    let sum_alloc: f64 = e1.allocated_total.iter().sum();
    assert!((sum_alloc - e1.portfolio_risk_measure).abs() < 1e-10);

    let mr1 = m_orig.myers_read(300.0).expect("mr orig");
    let mr2 = m_shuf.myers_read(300.0).expect("mr shuf");
    for i in 0..3 {
        assert!((mr1.allocated_surplus[i] - mr2.allocated_surplus[i]).abs() < 1e-10);
    }
    let sum_q: f64 = mr1.allocated_surplus.iter().sum();
    assert!((sum_q - mr1.portfolio_surplus).abs() < 1e-10);

    let r1 = rmk_risk_adjusted_pricing(
        &[l1.as_slice(), l2.as_slice(), l3.as_slice()],
        DistortionOperator::Wang { lambda: 0.35 },
    )
    .expect("rmk orig");
    let r2 = rmk_risk_adjusted_pricing(
        &[s1.as_slice(), s2.as_slice(), s3.as_slice()],
        DistortionOperator::Wang { lambda: 0.35 },
    )
    .expect("rmk shuf");
    for i in 0..3 {
        assert!((r1.line_prices[i] - r2.line_prices[i]).abs() < 1e-10);
    }
}

#[test]
fn ylt_simulation_aggregate_xl_and_proportional_treaties() {
    let elt = [
        EventLossEntry {
            event_id: 101,
            annual_rate: 0.50,
            mean_loss: 2_000_000.0,
            std_dev_indep: 400_000.0,
            std_dev_corr: 200_000.0,
            max_loss: 8_000_000.0,
        },
        EventLossEntry {
            event_id: 102,
            annual_rate: 0.10,
            mean_loss: 10_000_000.0,
            std_dev_indep: 2_000_000.0,
            std_dev_corr: 1_000_000.0,
            max_loss: 30_000_000.0,
        },
    ];
    let exact = elt_summary(&elt).expect("elt summary");
    let ylt = simulate_ylt_from_elt(&elt, 10_000, 42069).expect("ylt sim");
    // Deterministic reproducibility with same seed.
    let ylt2 = simulate_ylt_from_elt(&elt, 10_000, 42069).expect("ylt sim 2");
    assert_eq!(ylt, ylt2);

    // Monte Carlo AAL should match exact ELT AAL within 5% over 10k years.
    assert!(
        (ylt.aal() - exact.average_annual_loss).abs() / exact.average_annual_loss < 0.05,
        "sim_aal={}, exact_aal={}",
        ylt.aal(),
        exact.average_annual_loss
    );
    // Exact ordering: OEP(x) <= AEP(x) and OEP PML <= AEP PML.
    assert!(ylt.oep(5_000_000.0).expect("oep") <= ylt.aep(5_000_000.0).expect("aep") + 1e-12);
    assert!(ylt.oep_pml(100.0).expect("oep pml") <= ylt.aep_pml(100.0).expect("aep pml") + 1e-12);

    // Aggregate XL (AAD/AAL): higher AAD strictly lowers expected ceded loss.
    let agg_low_aad = ylt
        .aggregate_xl_expected_loss(1_000_000.0, 10_000_000.0, 2_000_000.0, 15_000_000.0)
        .expect("agg low");
    let agg_high_aad = ylt
        .aggregate_xl_expected_loss(1_000_000.0, 10_000_000.0, 8_000_000.0, 15_000_000.0)
        .expect("agg high");
    assert!(agg_low_aad >= agg_high_aad && agg_high_aad >= 0.0);

    // Quota Share & Surplus Share accounting conservation.
    let qs = QuotaShareSpec {
        cession_share: 0.40,
        ceding_commission: 0.25,
        ceded_loss_cap: Some(500_000.0),
    };
    let qs_res = qs.apply(1_000_000.0, 600_000.0).expect("qs");
    assert!((qs_res.ceded_premium + qs_res.retained_premium - 1_000_000.0).abs() < 1e-9);
    assert!((qs_res.ceded_loss + qs_res.retained_loss - 600_000.0).abs() < 1e-9);
    assert!((qs_res.ceded_loss - 240_000.0).abs() < 1e-9);

    let ss = SurplusShareSpec {
        retention_line: 1_000_000.0,
        number_of_lines: 4.0,
        ceding_commission: 0.28,
    };
    // Risk 1: TIV 500k (below retention -> 0% ceded)
    // Risk 2: TIV 3M (1M retained, 2M ceded -> 2/3 ceded)
    // Risk 3: TIV 10M (4M max ceded -> 40% ceded)
    assert!((ss.risk_cession_share(500_000.0).expect("r1") - 0.0).abs() < 1e-12);
    assert!((ss.risk_cession_share(3_000_000.0).expect("r2") - 2.0 / 3.0).abs() < 1e-12);
    assert!((ss.risk_cession_share(10_000_000.0).expect("r3") - 0.4).abs() < 1e-12);

    let ss_res = ss
        .apply_to_risks(
            &[500_000.0, 3_000_000.0, 10_000_000.0],
            &[5_000.0, 30_000.0, 100_000.0],
            &[2_000.0, 15_000.0, 50_000.0],
        )
        .expect("ss");
    assert!((ss_res.ceded_premium + ss_res.retained_premium - ss_res.gross_premium).abs() < 1e-9);
    assert!((ss_res.ceded_loss + ss_res.retained_loss - ss_res.gross_loss).abs() < 1e-9);
}

#[test]
fn design_matrix_regularized_glm_ergonomics_and_adversarial_guards() {
    let dm = DesignMatrix::try_from_rows(&[
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [4.0, 1.0],
        [5.0, 0.0],
    ])
    .expect("dm");
    let y = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0];
    let opts = RegularizedGlmOptions {
        family: RegularizedGlmFamily::Poisson,
        lambda: 0.05,
        alpha: 0.5,
        ..RegularizedGlmOptions::default()
    };
    let fit = dm.fit_regularized(&y, None, None, opts).expect("reg fit");
    let preds = fit
        .predict(RegularizedGlmFamily::Poisson, dm.data(), 6, 2, None)
        .expect("predict");
    assert_eq!(preds.len(), 6);
    for (p, m) in preds.iter().zip(fit.fitted_means.iter()) {
        assert!((p - m).abs() < 1e-9);
    }
    let rels = fit
        .relativities(Some(&["age_band", "urban"]))
        .expect("rels");
    assert_eq!(rels.len(), 3);
    assert_eq!(rels[0].0, "intercept");

    // Adversarial guards:
    // 1. usize overflow on matrix dimensions.
    assert!(
        fit_glm_irls(
            GlmFamily::Poisson,
            &[],
            usize::MAX,
            2,
            &[],
            &[],
            &[],
            10,
            1e-6
        )
        .is_err()
    );
    assert!(bailey_simon_multiplicative(usize::MAX, 2, &[], &[], 10, 1e-6).is_err());
    // 2. Non-finite quadrature bounds must return NaN immediately (no deep recursion).
    assert!(gauss_kronrod_integrate(|x| x, f64::NAN, 1.0).is_nan());
    assert!(gauss_kronrod_integrate_semi_infinite(|x| x, f64::INFINITY).is_nan());
    // 3. GLMM max_iter == 0 rejected.
    assert!(
        dm.fit_random_intercept_glmm(&y, &[0, 0, 1, 1, 2, 2], 3, None, None, 0)
            .is_err()
    );
    // 4. Empty goodness_of_fit returns finite/NaN cleanly without panicking.
    let gof = goodness_of_fit(
        &FittedEvtDistribution::Lognormal {
            mu: 1.0,
            sigma: 1.0,
        },
        &[],
        0.0,
        2,
    );
    assert!(gof.anderson_darling.is_none());
}
