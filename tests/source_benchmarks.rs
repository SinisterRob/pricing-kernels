//! Source-document benchmark, exact actuarial identity, and adversarial validation suite.

use pricing_kernels::datasets;
use pricing_kernels::prelude::*;

fn close_rel(actual: f64, expected: f64, tol: f64) -> bool {
    let denom = expected.abs().max(1.0);
    (actual - expected).abs() / denom <= tol
}

#[test]
fn test_bailey_1963_balance_and_brown_1988_poisson_glm_equivalence() {
    // 3x2 commercial risk grid with heterogeneous exposure mix
    let n_rows = 3;
    let n_cols = 2;
    let exposures = [1235.0, 582.0, 2845.0, 1204.0, 890.0, 410.0];
    let loss_costs = [105.4, 132.8, 92.1, 116.5, 81.0, 102.3];

    let mb = bailey_simon_multiplicative(n_rows, n_cols, &exposures, &loss_costs, 500, 1e-12)
        .expect("Bailey 1963 balance");

    // 1. Verify exact row and column balance identities:
    // sum_j P_ij * fitted_ij == sum_j P_ij * A_ij
    for i in 0..n_rows {
        let mut obs = 0.0;
        let mut fit = 0.0;
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            obs += exposures[idx] * loss_costs[idx];
            fit +=
                exposures[idx] * mb.base_rate * mb.relativities_dim1[i] * mb.relativities_dim2[j];
        }
        assert!(close_rel(fit, obs, 1e-9), "row {i} balance: {fit} vs {obs}");
    }
    for j in 0..n_cols {
        let mut obs = 0.0;
        let mut fit = 0.0;
        for i in 0..n_rows {
            let idx = i * n_cols + j;
            obs += exposures[idx] * loss_costs[idx];
            fit +=
                exposures[idx] * mb.base_rate * mb.relativities_dim1[i] * mb.relativities_dim2[j];
        }
        assert!(close_rel(fit, obs, 1e-9), "col {j} balance: {fit} vs {obs}");
    }

    // 2. Brown (1988) theorem: Bailey (1963) multiplicative balance relativities are
    // mathematically identical to Poisson log-link GLM with exposure offsets ln(P_ij).
    // Design: [intercept, row1, row2, col1]
    let design = [
        1.0, 0.0, 0.0, 0.0, // (0,0)
        1.0, 0.0, 0.0, 1.0, // (0,1)
        1.0, 1.0, 0.0, 0.0, // (1,0)
        1.0, 1.0, 0.0, 1.0, // (1,1)
        1.0, 0.0, 1.0, 0.0, // (2,0)
        1.0, 0.0, 1.0, 1.0, // (2,1)
    ];
    let total_losses: Vec<f64> = exposures
        .iter()
        .zip(loss_costs.iter())
        .map(|(&p, &a)| p * a)
        .collect();
    let weights = [1.0; 6];
    let offsets: Vec<f64> = exposures.iter().map(|&p| p.ln()).collect();

    let glm = fit_glm_irls(
        GlmFamily::Poisson,
        &design,
        6,
        4,
        &total_losses,
        &weights,
        &offsets,
        100,
        1e-12,
    )
    .expect("Poisson GLM");

    assert!(close_rel(glm.coefficients[0].exp(), mb.base_rate, 1e-8));
    assert!(close_rel(
        glm.coefficients[1].exp(),
        mb.relativities_dim1[1],
        1e-8
    ));
    assert!(close_rel(
        glm.coefficients[2].exp(),
        mb.relativities_dim1[2],
        1e-8
    ));
    assert!(close_rel(
        glm.coefficients[3].exp(),
        mb.relativities_dim2[1],
        1e-8
    ));
    assert_eq!(glm.dispersion, 1.0);
}

#[test]
fn test_bailey_simon_1960_chi_squared_minimum_bias() {
    // Noisy 2x2 table: Bailey-Simon (1960) chi-squared minimum bias achieves strictly
    // lower chi-squared than Bailey (1963) balance principle.
    let exposures = [1000.0, 800.0, 500.0, 400.0];
    let losses = [100.0, 128.0, 155.0, 172.0];

    let bal = bailey_simon_multiplicative(2, 2, &exposures, &losses, 200, 1e-12).unwrap();
    let chi = bailey_simon_chi_squared(2, 2, &exposures, &losses, 200, 1e-12).unwrap();

    assert!(chi.chi_squared <= bal.chi_squared + 1e-12);
    assert!(chi.chi_squared < bal.chi_squared);

    // Exact multiplicative table recovers exact factors with zero chi-squared.
    let exact_losses = [100.0, 120.0, 150.0, 180.0];
    let exact = bailey_simon_chi_squared(2, 2, &exposures, &exact_losses, 200, 1e-12).unwrap();
    assert!(close_rel(exact.base_rate, 100.0, 1e-9));
    assert!(close_rel(exact.relativities_dim1[1], 1.5, 1e-9));
    assert!(close_rel(exact.relativities_dim2[1], 1.2, 1e-9));
    assert!(exact.chi_squared < 1e-12);
}

#[test]
fn test_glm_irls_gamma_and_collinearity_rejection() {
    // Weighted Gamma severity model: group 0 weighted mean = (10*505 + 12*495)/22 = 499.545454...
    // group 1 weighted mean = (8*1010 + 9*990)/17 = 999.411764...
    let design = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y = [505.0, 495.0, 1010.0, 990.0];
    let weights = [10.0, 12.0, 8.0, 9.0];
    let offsets = [0.0; 4];

    let res = fit_glm_irls(
        GlmFamily::Gamma,
        &design,
        4,
        2,
        &y,
        &weights,
        &offsets,
        100,
        1e-12,
    )
    .unwrap();

    let expected_m0 = (10.0 * 505.0 + 12.0 * 495.0) / 22.0;
    let expected_m1 = (8.0 * 1010.0 + 9.0 * 990.0) / 17.0;
    assert!(close_rel(res.coefficients[0].exp(), expected_m0, 1e-9));
    assert!(close_rel(
        (res.coefficients[0] + res.coefficients[1]).exp(),
        expected_m1,
        1e-9
    ));
    assert!(res.dispersion > 0.0);
    assert!(res.standard_errors[0] > 0.0 && res.standard_errors[1] > 0.0);

    // Collinear design matrix (dummy variable trap: col0 = col1 + col2) must return SingularMatrix
    let singular_design = [
        1.0, 1.0, 0.0, //
        1.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, //
        1.0, 0.0, 1.0, //
    ];
    let err = fit_glm_irls(
        GlmFamily::Poisson,
        &singular_design,
        4,
        3,
        &[10.0, 12.0, 20.0, 22.0],
        &[1.0; 4],
        &[0.0; 4],
        50,
        1e-8,
    )
    .unwrap_err();
    assert!(matches!(err, PricingError::SingularMatrix(_)));
}

#[test]
fn test_multiplicative_rater_and_batch() {
    let rater = MultiplicativeRater::new(100.0)
        .with_minimum_premium(250.0)
        .with_maximum_premium(5000.0);

    assert_eq!(rater.score(2.0, &[1.2, 0.9]), 250.0);
    assert_eq!(rater.score(10.0, &[1.5, 1.2]), 1800.0);
    assert_eq!(rater.score(50.0, &[2.0]), 5000.0);

    let mut out = [0.0; 3];
    rater
        .batch_score_into(
            &[2.0, 10.0, 50.0],
            &[&[1.2, 0.9], &[1.5, 1.2], &[2.0]],
            &mut out,
        )
        .unwrap();
    assert_eq!(out, [250.0, 1800.0, 5000.0]);

    // Mismatched output slice returns DimensionMismatch instead of panicking
    let mut bad_out = [0.0; 2];
    assert!(matches!(
        rater.batch_score_into(&[2.0, 10.0, 50.0], &[&[1.0], &[1.0], &[1.0]], &mut bad_out),
        Err(PricingError::DimensionMismatch(_))
    ));
}

#[test]
fn test_classical_credibility() {
    let cred = ClassicalCredibility::standard_90_5();
    assert_eq!(cred.credibility(1082.41), 1.0);
    assert_eq!(cred.credibility(2000.0), 1.0);
    assert_eq!(cred.credibility(0.0), 0.0);
    assert!(cred.credibility(f64::NAN).is_nan());

    let z = cred.credibility(1082.41 * 0.5);
    assert!(close_rel(z, 0.5_f64.sqrt(), 1e-12));

    let blended = cred.blend(1082.41 * 0.5, 120.0, 100.0);
    assert!(close_rel(blended, z * 120.0 + (1.0 - z) * 100.0, 1e-12));
}

#[test]
fn test_buhlmann_straub_exact_worked_example() {
    // Exact hand-verified Bühlmann-Straub problem:
    // Group 1: X = [100, 110, 105], P = [10, 10, 10] -> P_1 = 30, X_bar_1 = 105
    // Group 2: X = [150, 160, 155], P = [20, 20, 20] -> P_2 = 60, X_bar_2 = 155
    // Group 3: X = [200, 210, 195], P = [15, 15, 15] -> P_3 = 45, X_bar_3 = 605/3 = 201.666666...
    //
    // s^2_num = 10*(25+25+0) + 20*(25+25+0) + 15*((5/3)^2 + (25/3)^2 + (20/3)^2)
    //         = 500 + 1000 + 15*(25 + 625 + 400)/9 = 1500 + 15*1050/9 = 3250
    // df_epv = 9 - 3 = 6 -> EPV = 3250 / 6 = 541.6666666666666
    let g1_losses = [100.0, 110.0, 105.0];
    let g1_weights = [10.0, 10.0, 10.0];
    let g2_losses = [150.0, 160.0, 155.0];
    let g2_weights = [20.0, 20.0, 20.0];
    let g3_losses = [200.0, 210.0, 195.0];
    let g3_weights = [15.0, 15.0, 15.0];

    let losses: Vec<&[f64]> = vec![&g1_losses, &g2_losses, &g3_losses];
    let weights: Vec<&[f64]> = vec![&g1_weights, &g2_weights, &g3_weights];

    let res = buhlmann_straub(&losses, &weights).unwrap();

    let expected_epv = 3250.0 / 6.0;
    assert!(close_rel(res.epv, expected_epv, 1e-12));

    let total_p = 135.0;
    let x1 = 105.0;
    let x2 = 155.0;
    let x3 = 605.0 / 3.0;
    let mu_raw = (30.0 * x1 + 60.0 * x2 + 45.0 * x3) / total_p;
    let c = total_p - (30.0_f64.powi(2) + 60.0_f64.powi(2) + 45.0_f64.powi(2)) / total_p;
    let vhm_num =
        30.0 * (x1 - mu_raw).powi(2) + 60.0 * (x2 - mu_raw).powi(2) + 45.0 * (x3 - mu_raw).powi(2);
    let expected_vhm = (vhm_num - 2.0 * expected_epv) / c;
    let expected_k = expected_epv / expected_vhm;

    assert!(close_rel(res.vhm, expected_vhm, 1e-12));
    assert!(close_rel(res.k, expected_k, 1e-12));
    assert!(close_rel(
        res.credibilities[0],
        30.0 / (30.0 + expected_k),
        1e-12
    ));
    assert!(close_rel(
        res.credibilities[1],
        60.0 / (60.0 + expected_k),
        1e-12
    ));
    assert!(close_rel(
        res.credibilities[2],
        45.0 / (45.0 + expected_k),
        1e-12
    ));
}

#[test]
fn test_ncci_experience_mod_exact_calculation() {
    let spec = NcciErpSpecification {
        split_point: 18_500.0,
        expected_losses: 100_000.0,
        d_ratio: 0.40,
        weight_w: 0.15,
        ballast_b: 25_000.0,
        apply_era: true,
    };

    let claims = vec![
        ExperienceClaim {
            incurred: 50_000.0,
            is_medical_only: false,
        }, // 18.5k primary, 31.5k excess
        ExperienceClaim {
            incurred: 5_000.0,
            is_medical_only: false,
        }, // 5k primary, 0 excess
        ExperienceClaim {
            incurred: 10_000.0,
            is_medical_only: true,
        }, // ERA -> 3k primary, 0 excess
    ];

    let res = calculate_ncci_experience_mod(&spec, &claims).unwrap();
    assert_eq!(res.actual_primary, 26_500.0);
    assert_eq!(res.actual_excess, 31_500.0);
    assert_eq!(res.expected_primary, 40_000.0);
    assert_eq!(res.expected_excess, 60_000.0);
    // Exact numerator = 26,500 + 0.15*31,500 + 0.85*60,000 + 25,000 = 107,225
    // Exact denominator = 100,000 + 25,000 = 125,000
    // Exact mod = 107,225 / 125,000 = 0.8578 -> rounds to 0.858
    assert!(close_rel(
        res.mod_factor_unrounded,
        107_225.0 / 125_000.0,
        1e-12
    ));
    assert_eq!(res.mod_factor, 0.858);
}

#[test]
fn test_off_balance_calculation() {
    let e = [100_000.0, 200_000.0, 100_000.0];
    let m = [0.90, 1.10, 1.00];
    let b = calculate_off_balance(&e, &m).unwrap();
    assert!(close_rel(b, 400.0 / 410.0, 1e-12));
}

#[test]
fn test_table_m_lee_identities_and_retro_plan_balance() {
    // Pass unsorted entry ratios to verify internal sorting
    let entry_ratios = [1.4, 0.2, 1.8, 0.6, 1.0];
    let probs = [0.2, 0.1, 0.1, 0.2, 0.4];

    let table_m = TableMEngine::from_discrete_pdf(&entry_ratios, &probs).unwrap();

    // Exact Lee (1988) identity below minimum support point (r in [0, 0.2]):
    // phi(r) = E[Y] - r = 1.0 - r
    assert!(close_rel(table_m.charge_at(0.0), 1.0, 1e-12));
    assert!(close_rel(table_m.charge_at(0.1), 0.9, 1e-12));
    assert!(close_rel(table_m.charge_at(0.2), 0.8, 1e-12));
    // At r = 1.0: phi(1.0) = 0.2*(0.4) + 0.1*(0.8) = 0.16; psi(1.0) = 0.16
    assert!(close_rel(table_m.charge_at(1.0), 0.16, 1e-12));
    assert!(close_rel(table_m.savings_at(1.0), 0.16, 1e-12));
    assert_eq!(table_m.charge_at(2.5), 0.0);

    let spec = RetroPlanSpec {
        standard_premium: 500_000.0,
        expected_loss_ratio: 0.60,
        lcf: 1.12,
        tax_multiplier: 1.04,
        max_retro_ratio: 1.40,
        min_retro_ratio: 0.50,
        expense_ratio: 0.20,
    };

    let retro = solve_balanced_retro_plan(&spec, &table_m, 200, 1e-10).unwrap();

    // Verify exact actuarial balance equation at the converged solution:
    let e_lcf = spec.lcf * spec.expected_loss_ratio;
    let rhs = spec.expense_ratio - (spec.lcf - 1.0) * spec.expected_loss_ratio
        + retro.net_insurance_charge * e_lcf;
    assert!(
        close_rel(retro.basic_premium_factor, rhs, 1e-9),
        "balance violated: bpf={} vs rhs={}",
        retro.basic_premium_factor,
        rhs
    );
    // Verify auxiliary fields match the final BPF:
    let expected_r_g =
        ((spec.max_retro_ratio / spec.tax_multiplier) - retro.basic_premium_factor) / e_lcf;
    assert!(close_rel(retro.entry_ratio_max, expected_r_g, 1e-12));
}

#[test]
fn test_all_six_severity_families_lev_ilf_ler_and_lee_survival_integral() {
    let dists = [
        SeverityDistribution::lognormal_from_mean_cv(100_000.0, 1.5).unwrap(),
        SeverityDistribution::Lomax {
            alpha: 2.5,
            theta: 150_000.0,
        },
        SeverityDistribution::Lomax {
            alpha: 1.0,
            theta: 100_000.0,
        },
        SeverityDistribution::SingleParameterPareto {
            alpha: 2.0,
            x_m: 50_000.0,
        },
        SeverityDistribution::Exponential { lambda: 1e-5 },
        SeverityDistribution::MixedExponential {
            weights: vec![0.5, 0.35, 0.15],
            means: vec![25_000.0, 150_000.0, 800_000.0],
        },
        SeverityDistribution::GeneralizedPareto {
            xi: 0.35,
            beta: 80_000.0,
        },
        SeverityDistribution::gamma_from_mean_cv(100_000.0, 0.7).unwrap(),
        SeverityDistribution::spliced_gamma_pareto_from_mean(100_000.0, 2.5, 3.5, 2.5).unwrap(),
    ];

    // Lee (1988) identity: E[X ∧ L] = integral_0^L S(x) dx
    let limit = 250_000.0;
    let steps = 5_000usize;
    let dx = limit / steps as f64;

    for dist in &dists {
        let exact_lev = dist.limited_expected_value(limit).unwrap();
        let mut trap = 0.0;
        for i in 0..steps {
            let x_mid = (i as f64 + 0.5) * dx;
            trap += dist.survival(x_mid).unwrap() * dx;
        }
        assert!(
            close_rel(exact_lev, trap, 5e-4),
            "dist={dist:?} exact={exact_lev} trap={trap}"
        );

        let ilf = dist.increased_limits_factor(250_000.0, 100_000.0).unwrap();
        assert!(ilf >= 1.0);

        let ler = dist.loss_elimination_ratio(10_000.0, 250_000.0).unwrap();
        assert!(ler > 0.0 && ler < 1.0);

        // Layer sum conservation: [0, 100k] + [100k, 250k] == [0, 250k]
        let l1 = CommercialLayer::new(0.0, 100_000.0, 1.0)
            .unwrap()
            .layer_expected_loss(dist)
            .unwrap();
        let l2 = CommercialLayer::new(100_000.0, 150_000.0, 1.0)
            .unwrap()
            .layer_expected_loss(dist)
            .unwrap();
        assert!(close_rel(l1 + l2, exact_lev, 1e-12));

        // Excess ratio conservation when mean is finite
        if let Some(mean) = dist.unbounded_mean().unwrap() {
            let xs = dist.excess_ratio(limit).unwrap();
            assert!(close_rel(exact_lev + xs * mean, mean, 1e-12));
        }
    }
}

#[test]
fn test_parallelogram_multi_change_calendar_year_and_policy_year() {
    // Two +10% rate changes in 2022 (passed unsorted!) at 2022.25 and 2022.75
    let changes = [
        RateChange {
            effective_date: 2022.75,
            rate_factor: 1.10,
        },
        RateChange {
            effective_date: 2022.25,
            rate_factor: 1.10,
        },
    ];

    // 1. Policy-Year 2022 (written uniformly in [2022.0, 2023.0]):
    // 0.25 at 1.00, 0.50 at 1.10, 0.25 at 1.21 -> avg = 1.1025
    // Current level = 1.21 -> OLF = 1.21 / 1.1025
    let py_olf = policy_year_on_level_factor(2022.0, &changes).unwrap();
    let expected_py_avg = 0.25 * 1.00 + 0.50 * 1.10 + 0.25 * 1.21;
    assert!(close_rel(py_olf, 1.21 / expected_py_avg, 1e-12));

    // 2. Calendar-Year 2022 earned parallelogram (1-year policies):
    // Area after 2022.25 = 0.5 * 0.75^2 = 0.28125
    // Area after 2022.75 = 0.5 * 0.25^2 = 0.03125
    // Weights: 1.00 -> 1 - 0.28125 = 0.71875; 1.10 -> 0.25; 1.21 -> 0.03125
    let cy_olf = parallelogram_on_level_factor(2022.0, &changes).unwrap();
    let expected_cy_avg = 0.71875 * 1.00 + 0.25 * 1.10 + 0.03125 * 1.21;
    assert!(close_rel(cy_olf, 1.21 / expected_cy_avg, 1e-12));

    // 3. Prior-year rate change (+10% on July 1, 2021) earning into Calendar Year 2022:
    // Area written on/after 2021.5 in CY 2022 = 1 - 0.5 * 0.5^2 = 0.875
    // Average CY 2022 level = 0.125 * 1.0 + 0.875 * 1.10 = 1.0875 -> OLF = 1.10 / 1.0875
    let prior_only = [RateChange {
        effective_date: 2021.5,
        rate_factor: 1.10,
    }];
    let cy_prior_olf = parallelogram_on_level_factor(2022.0, &prior_only).unwrap();
    assert!(close_rel(cy_prior_olf, 1.10 / 1.0875, 1e-12));

    // Trending model fit
    let fitted = TrendModel::fit_exponential(&[0.0, 1.0, 2.0], &[100.0, 105.0, 110.25]).unwrap();
    assert!(close_rel(fitted.annual_rate, 0.05, 1e-12));
}

#[test]
fn test_adversarial_nan_inf_and_dimension_guards() {
    // NaN rejections across every kernel
    assert!(
        bailey_simon_multiplicative(2, 2, &[1.0, f64::NAN, 1.0, 1.0], &[1.0; 4], 50, 1e-6).is_err()
    );
    assert!(
        fit_glm_irls(
            GlmFamily::Poisson,
            &[1.0, 1.0],
            2,
            1,
            &[f64::NAN, 1.0],
            &[1.0; 2],
            &[0.0; 2],
            50,
            1e-6
        )
        .is_err()
    );
    assert!(
        fit_glm_irls(
            GlmFamily::Gamma,
            &[1.0, 1.0],
            2,
            1,
            &[0.0, 1.0],
            &[1.0; 2],
            &[0.0; 2],
            50,
            1e-6
        )
        .is_err()
    );
    assert!(
        buhlmann_straub(
            &[&[1.0, f64::NAN], &[2.0, 3.0]],
            &[&[1.0, 1.0], &[1.0, 1.0]]
        )
        .is_err()
    );
    assert!(TableMEngine::from_discrete_pdf(&[0.5, 1.5], &[f64::NAN, 0.5]).is_err());
    assert!(TableMEngine::from_discrete_pdf(&[0.5, 1.5], &[-0.2, 1.2]).is_err());
    assert!(parallelogram_on_level_factor(f64::NAN, &[]).is_err());
    assert!(TrendModel::fit_exponential(&[0.0, f64::INFINITY], &[1.0, 2.0]).is_err());

    let lomax = SeverityDistribution::Lomax {
        alpha: 2.0,
        theta: 100_000.0,
    };
    assert!(lomax.limited_expected_value(f64::NAN).is_err());
    assert!(CommercialLayer::new(0.0, f64::INFINITY, 1.0).is_err());

    // Phase 2 adversarial guards
    assert!(MbbefdCurve::new(-0.1, 2.0).is_err());
    assert!(MbbefdCurve::new(1.0, 0.5).is_err());
    assert!(MbbefdCurve::from_swiss_re_c(-1.0).is_err());
    assert!(
        MbbefdCurve::from_preset(SwissReCurve::Y2)
            .exposure_curve(f64::NAN)
            .is_err()
    );
    assert!(burning_cost_xl(&[], &[], &CommercialLayer::new(0.0, 100.0, 1.0).unwrap()).is_err());
    assert!(
        price_reinstatements(
            100.0,
            &[10.0, 5.0],
            &[],
            0.1,
            ReinstatementBasis::ProRataAmount
        )
        .is_err()
    );
    assert!(TableLEngine::from_limited_discrete_pdf(&[0.5, 1.5], &[0.5, 0.5]).is_ok());
    assert!(TableLEngine::from_limited_discrete_pdf(&[1.2, 1.4], &[0.5, 0.5]).is_err());
    assert!(calculate_schedule_mod(&[], -1.5, 0.25).is_err());
}

#[test]
fn test_bernegger_1997_mbbefd_swiss_re_curves_and_survival_integral() {
    // Verify G(d) = (integral_0^d S(x) dx) / E[X] across all 5 Swiss Re presets + all 3 limiting cases.
    let curves = [
        MbbefdCurve::from_preset(SwissReCurve::Y1),
        MbbefdCurve::from_preset(SwissReCurve::Y2),
        MbbefdCurve::from_preset(SwissReCurve::Y3),
        MbbefdCurve::from_preset(SwissReCurve::Y4),
        MbbefdCurve::from_preset(SwissReCurve::Lloyds),
        MbbefdCurve::new(0.5, 1.0).unwrap(), // g = 1 limiting case (linear)
        MbbefdCurve::new(1.0, 5.0).unwrap(), // b = 1 limiting case
        MbbefdCurve::new(0.25, 4.0).unwrap(), // b * g = 1 limiting case
    ];

    for curve in &curves {
        assert_eq!(curve.exposure_curve(0.0).unwrap(), 0.0);
        assert_eq!(curve.exposure_curve(1.0).unwrap(), 1.0);
        let mean = curve.mean_loss_ratio().unwrap();
        assert!(mean > 0.0 && mean <= 1.0 + 1e-12);

        // Numerical midpoint integration of S(x) on [0, d] for d = 0.25 and d = 0.70
        for &d in &[0.25, 0.70] {
            let n = 40_000_i32;
            let h = d / f64::from(n);
            let mut integral = 0.0;
            for i in 0..n {
                let x = (f64::from(i) + 0.5) * h;
                integral += curve.survival(x).unwrap() * h;
            }
            let g_numeric = integral / mean;
            let g_exact = curve.exposure_curve(d).unwrap();
            assert!(
                close_rel(g_exact, g_numeric, 1e-5),
                "MBBEFD ({}, {}) at d={d}: exact={g_exact} vs numeric={g_numeric}",
                curve.b,
                curve.g
            );
        }

        // Layer factor additivity: [0, 0.2M] + [0.2M, 0.5M] + [0.5M, M] == 1.0
        let f1 = curve
            .layer_exposure_factor(0.0, 200_000.0, 1_000_000.0)
            .unwrap();
        let f2 = curve
            .layer_exposure_factor(200_000.0, 300_000.0, 1_000_000.0)
            .unwrap();
        let f3 = curve
            .layer_exposure_factor(500_000.0, 500_000.0, 1_000_000.0)
            .unwrap();
        assert!(close_rel(f1 + f2 + f3, 1.0, 1e-12));
    }

    // Ordering: higher Swiss Re c-curve is more concave (c = 0 is linear G(d) = d;
    // higher c concentrates more expected loss in lower layers -> higher G(d) for 0 < d < 1).
    let g_y1 = MbbefdCurve::from_preset(SwissReCurve::Y1)
        .exposure_curve(0.2)
        .unwrap();
    let g_y2 = MbbefdCurve::from_preset(SwissReCurve::Y2)
        .exposure_curve(0.2)
        .unwrap();
    let g_y3 = MbbefdCurve::from_preset(SwissReCurve::Y3)
        .exposure_curve(0.2)
        .unwrap();
    let g_y4 = MbbefdCurve::from_preset(SwissReCurve::Y4)
        .exposure_curve(0.2)
        .unwrap();
    let g_ll = MbbefdCurve::from_preset(SwissReCurve::Lloyds)
        .exposure_curve(0.2)
        .unwrap();
    assert!(g_y1 < g_y2 && g_y2 < g_y3 && g_y3 < g_y4 && g_y4 < g_ll);
}

#[test]
fn test_clark_2014_reinsurance_burning_cost_reinstatements_and_proportional_terms() {
    // 1. Experience-rated burning cost for $500k xs $500k treaty
    let layer = CommercialLayer::new(500_000.0, 500_000.0, 1.0).unwrap();
    let years = [
        HistoricalTreatyYear {
            year: 2023,
            subject_premium: 10_000_000.0,
            on_level_factor: 1.10,
            exposure_trend_factor: 1.05,
        },
        HistoricalTreatyYear {
            year: 2024,
            subject_premium: 12_000_000.0,
            on_level_factor: 1.05,
            exposure_trend_factor: 1.02,
        },
    ];
    let claims = [
        HistoricalTreatyClaim {
            year: 2023,
            incurred: 600_000.0,
            severity_trend_factor: 1.10,
            development_factor: 1.0,
        }, // adj = 660k -> payout = 160k
        HistoricalTreatyClaim {
            year: 2024,
            incurred: 1_200_000.0,
            severity_trend_factor: 1.05,
            development_factor: 1.0,
        }, // adj = 1.26M -> capped payout = 500k
        HistoricalTreatyClaim {
            year: 2024,
            incurred: 200_000.0,
            severity_trend_factor: 1.05,
            development_factor: 1.0,
        }, // below attachment -> 0
    ];
    let bc = burning_cost_xl(&years, &claims, &layer).unwrap();
    assert_eq!(bc.total_layer_claims, 2);
    assert!(close_rel(bc.total_layer_ceded_loss, 660_000.0, 1e-12));
    let expected_prem = 10_000_000.0 * 1.10 * 1.05 + 12_000_000.0 * 1.05 * 1.02;
    assert!(close_rel(
        bc.total_adjusted_subject_premium,
        expected_prem,
        1e-12
    ));
    assert!(close_rel(
        bc.overall_burning_cost_rate,
        660_000.0 / expected_prem,
        1e-12
    ));

    // 2. Sundt (1991) reinstatement pricing: 1 reinstatement at 100% pro-rata by amount
    let reinst = price_reinstatements(
        1_000_000.0,
        &[200_000.0, 50_000.0],
        &[1.0],
        0.10,
        ReinstatementBasis::ProRataAmount,
    )
    .unwrap();
    // Exact Sundt formula: P0 = 250_000 / (0.9 * (1 + 0.2)) = 250_000 / 1.08
    let expected_p0 = 250_000.0 / 1.08;
    assert!(close_rel(reinst.initial_premium, expected_p0, 1e-12));
    assert!(close_rel(
        reinst.total_expected_premium * 0.90,
        250_000.0,
        1e-12
    ));

    // Pro-rata by time reduces reinstatement premium collection -> higher initial premium P0
    let reinst_time = price_reinstatements(
        1_000_000.0,
        &[200_000.0, 50_000.0],
        &[1.0],
        0.10,
        ReinstatementBasis::ProRataAmountAndTime,
    )
    .unwrap();
    assert!(reinst_time.initial_premium > reinst.initial_premium);
    assert!(close_rel(
        reinst_time.total_expected_premium * 0.90,
        250_000.0,
        1e-12
    ));

    // 3. Sliding-scale commission, loss corridor, and profit commission:
    // Verify Table M closed-form formulas match exact discrete expectation over (r_k, p_k).
    let retro = datasets::retro_plan_benchmark();
    let tm = TableMEngine::from_discrete_pdf(&retro.entry_ratios, &retro.probabilities).unwrap();
    let elr = 0.65;

    let slide = SlidingScaleSpec {
        min_commission: 0.20,
        max_commission: 0.35,
        min_lr: 0.50,
        slide_ratio: 0.50,
    };
    let expected_comm_formula = slide.expected_commission(elr, &tm).unwrap();
    let expected_comm_direct: f64 = retro
        .entry_ratios
        .iter()
        .zip(retro.probabilities.iter())
        .map(|(&r, &p)| p * slide.commission_at(r * elr).unwrap())
        .sum();
    assert!(close_rel(
        expected_comm_formula,
        expected_comm_direct,
        1e-12
    ));

    let corridor_formula = expected_loss_corridor_retention(0.60, 0.85, 0.50, elr, &tm).unwrap();
    let corridor_direct: f64 = retro
        .entry_ratios
        .iter()
        .zip(retro.probabilities.iter())
        .map(|(&r, &p)| {
            let lr = r * elr;
            let ret = 0.50 * (lr.min(0.85) - 0.60).max(0.0);
            p * ret
        })
        .sum();
    assert!(close_rel(corridor_formula, corridor_direct, 1e-12));

    let pc_formula = expected_profit_commission(0.25, 0.05, 0.20, elr, &tm).unwrap();
    let pc_direct: f64 = retro
        .entry_ratios
        .iter()
        .zip(retro.probabilities.iter())
        .map(|(&r, &p)| {
            let profit = (1.0 - 0.25 - 0.05 - r * elr).max(0.0);
            p * 0.20 * profit
        })
        .sum();
    assert!(close_rel(pc_formula, pc_direct, 1e-12));
}

#[test]
fn test_skurnick_1974_table_l_balance_and_retro_solver() {
    // 1. Reduction to Table M when k = 0 (E[Y*] = 1.0)
    let retro = datasets::retro_plan_benchmark();
    let tm = TableMEngine::from_discrete_pdf(&retro.entry_ratios, &retro.probabilities).unwrap();
    let tl_zero_k =
        TableLEngine::from_limited_discrete_pdf(&retro.entry_ratios, &retro.probabilities).unwrap();
    assert!(tl_zero_k.loss_elimination_ratio < 1e-12);
    for &r in &[0.0, 0.35, 0.80, 1.15, 2.0] {
        assert!(close_rel(tl_zero_k.charge_at(r), tm.charge_at(r), 1e-12));
        assert!(close_rel(tl_zero_k.savings_at(r), tm.savings_at(r), 1e-12));
    }

    // 2. Non-zero per-occurrence LER k = 0.10 (E[Y*] = 0.90)
    let limited_ratios = [0.18, 0.54, 0.90, 1.26, 1.62];
    let tl =
        TableLEngine::from_limited_discrete_pdf(&limited_ratios, &retro.probabilities).unwrap();
    assert!(close_rel(tl.loss_elimination_ratio, 0.10, 1e-12));

    // Skurnick fundamental identity: phi*(r) - psi*(r) == 1 - r for all r >= 0
    for &r in &[0.0, 0.10, 0.45, 0.90, 1.40, 2.50] {
        let lhs = tl.charge_at(r) - tl.savings_at(r);
        assert!(
            close_rel(lhs, 1.0 - r, 1e-12),
            "Skurnick identity at r={r}: {lhs} vs {}",
            1.0 - r
        );
    }
    assert!(close_rel(tl.charge_at(0.0), 1.0, 1e-12));
    assert!(close_rel(tl.charge_at(5.0), 0.10, 1e-12));

    // 3. Balanced Table L retro plan satisfies exact actuarial balance equation
    let plan = solve_balanced_retro_plan_table_l(&retro.spec, &tl, 200, 1e-12).unwrap();
    let e_lcf = retro.spec.lcf * retro.spec.expected_loss_ratio;
    let expected_bpf = retro.spec.expense_ratio
        - (retro.spec.lcf - 1.0) * retro.spec.expected_loss_ratio
        + plan.net_insurance_charge * e_lcf;
    assert!(close_rel(plan.basic_premium_factor, expected_bpf, 1e-9));

    // Adding a per-occurrence limit (k > 0) raises the basic premium factor relative to unlimited Table M
    let plan_unlimited = solve_balanced_retro_plan(&retro.spec, &tm, 200, 1e-12).unwrap();
    assert!(plan.basic_premium_factor > plan_unlimited.basic_premium_factor);
}

#[test]
fn test_large_deductible_alae_treatments_and_iso_erp() {
    let sev = SeverityDistribution::Lomax {
        alpha: 2.5,
        theta: 100_000.0,
    };
    let retro = datasets::retro_plan_benchmark();
    let tm = TableMEngine::from_discrete_pdf(&retro.entry_ratios, &retro.probabilities).unwrap();

    let base_spec = LargeDeductibleSpec {
        expected_indemnity: 1_000_000.0,
        expected_alae: 150_000.0,
        deductible: 100_000.0,
        policy_limit: 1_000_000.0,
        alae_treatment: AlaeTreatment::ProRata,
        fixed_expenses: 50_000.0,
        variable_expense_ratio: 0.20,
        aggregate_cap_entry_ratio: None,
    };

    let res_prorata = price_large_deductible(&base_spec, &sev, None).unwrap();
    assert!(close_rel(
        res_prorata.loss_elimination_ratio + res_prorata.excess_loss_factor,
        1.0,
        1e-12
    ));
    assert!(close_rel(
        res_prorata.retained_indemnity + res_prorata.excess_indemnity,
        1_000_000.0,
        1e-12
    ));

    // OutsideLimit means insurer pays all ALAE -> higher premium and lower credit than ProRata
    let outside_spec = LargeDeductibleSpec {
        alae_treatment: AlaeTreatment::OutsideLimit,
        ..base_spec
    };
    let res_outside = price_large_deductible(&outside_spec, &sev, None).unwrap();
    assert_eq!(res_outside.retained_alae, 0.0);
    assert!(close_rel(res_outside.insurer_alae, 150_000.0, 1e-12));
    assert!(res_outside.indicated_premium > res_prorata.indicated_premium);

    // Adding an aggregate deductible cap adds a positive charge and raises indicated premium
    let capped_spec = LargeDeductibleSpec {
        aggregate_cap_entry_ratio: Some(1.20),
        ..base_spec
    };
    let res_capped = price_large_deductible(&capped_spec, &sev, Some(&tm)).unwrap();
    assert!(res_capped.aggregate_cap_charge > 0.0);
    assert!(res_capped.indicated_premium > res_prorata.indicated_premium);

    // ISO Commercial ERP with MSL capping
    let iso_spec = IsoErpSpec {
        expected_basic_limits_loss: 200_000.0,
        max_single_limit: 50_000.0,
        credibility_k: 50_000.0, // Z = 200k / 250k = 0.80
        balance_factor: 1.0,
    };
    // Claims: 120k (capped at 50k) + 30k + 20k = 100k ratable -> A/E = 0.50 -> Mod = 1 + 0.8*(0.5 - 1) = 0.60
    let iso_res =
        calculate_iso_experience_mod(&iso_spec, &[120_000.0, 30_000.0, 20_000.0]).unwrap();
    assert!(close_rel(iso_res.credibility_z, 0.80, 1e-12));
    assert!(close_rel(iso_res.actual_ratable_loss, 100_000.0, 1e-12));
    assert!(close_rel(iso_res.mod_factor_unrounded, 0.60, 1e-12));

    // Schedule rating / IRPM clamping to [-0.25, +0.25]
    let items = [
        ScheduleRatingItem {
            adjustment: -0.15,
            min_adjustment: -0.15,
            max_adjustment: 0.15,
        },
        ScheduleRatingItem {
            adjustment: -0.15,
            min_adjustment: -0.15,
            max_adjustment: 0.15,
        },
    ];
    let sched = calculate_schedule_mod(&items, -0.25, 0.25).unwrap();
    assert!(close_rel(sched.raw_total_adjustment, -0.30, 1e-12));
    assert!(close_rel(sched.applied_adjustment, -0.25, 1e-12));
    assert!(close_rel(sched.schedule_mod_factor, 0.75, 1e-12));

    // Zero-I/O datasets fixture parity check
    let ncci = datasets::ncci_erp_worksheet();
    let ncci_res = calculate_ncci_experience_mod(&ncci.spec, &ncci.claims).unwrap();
    assert!(close_rel(
        ncci_res.mod_factor_unrounded,
        ncci.expected_mod_unrounded,
        1e-12
    ));
    assert_eq!(ncci_res.mod_factor, ncci.expected_mod_rounded);
}

#[test]
fn test_panjer_vs_tilted_fft_and_compound_moments() {
    let sev_pmf = [0.0, 0.50, 0.30, 0.20];
    let step = 100.0;
    let ex = 0.50 * 100.0 + 0.30 * 200.0 + 0.20 * 300.0; // 170.0
    let ex2 = 0.50 * 10_000.0 + 0.30 * 40_000.0 + 0.20 * 90_000.0; // 35_000.0
    let var_x = ex2 - ex * ex;

    let freqs = [
        FrequencyDistribution::Poisson { lambda: 3.0 },
        FrequencyDistribution::Binomial { m: 8, q: 0.35 },
        FrequencyDistribution::NegativeBinomial { r: 2.5, beta: 1.2 },
    ];

    for freq in freqs {
        let panjer = panjer_recursion(freq, &sev_pmf, step, 256).unwrap();
        let fft = fft_aggregate_loss(freq, &sev_pmf, step, 256, Some(0.02)).unwrap();

        let en = freq.mean().unwrap();
        let var_n = freq.variance().unwrap();
        let exact_mean = en * ex;
        let exact_var = en * var_x + var_n * ex * ex;

        assert!(close_rel(panjer.mean, exact_mean, 1e-8));
        assert!(close_rel(panjer.variance, exact_var, 1e-7));
        assert!(close_rel(fft.mean, panjer.mean, 1e-8));
        assert!(close_rel(fft.variance, panjer.variance, 1e-7));
        assert!(close_rel(
            fft.tvar(0.95).unwrap(),
            panjer.tvar(0.95).unwrap(),
            1e-7
        ));
    }

    // Independent convolution of two components preserves sum of means and sum of variances
    let comp1 = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 2.0 },
        &sev_pmf,
        step,
        128,
    )
    .unwrap();
    let comp2 = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 1.5 },
        &sev_pmf,
        step,
        128,
    )
    .unwrap();
    let combined = convolve_independent_aggregates(&[&comp1, &comp2], 256).unwrap();
    assert!(close_rel(combined.mean, comp1.mean + comp2.mean, 1e-8));
    assert!(close_rel(
        combined.variance,
        comp1.variance + comp2.variance,
        1e-7
    ));

    // Mean-preserving LEV discretization of Lomax matches E[X ^ max_grid]
    let lomax = SeverityDistribution::Lomax {
        alpha: 3.0,
        theta: 10_000.0,
    };
    let disc = discretize_severity_mean_preserving(&lomax, 1_000.0, 64).unwrap();
    let sum_p: f64 = disc.iter().sum();
    assert!(close_rel(sum_p, 1.0, 1e-12));
    let disc_mean: f64 = disc
        .iter()
        .enumerate()
        .map(|(k, &p)| (k as f64) * 1_000.0 * p)
        .sum();
    let exact_lev = lomax.limited_expected_value(63_000.0).unwrap();
    assert!(close_rel(disc_mean, exact_lev, 1e-6));
}

#[test]
fn test_catastrophe_elt_oep_aep_pml_and_layer_pricing() {
    let entries = [
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
            max_loss: 6_000_000.0, // point-mass event
        },
    ];

    let summary = elt_summary(&entries).unwrap();
    let expected_aal = 0.05 * 2_000_000.0 + 0.02 * 6_000_000.0;
    let expected_var =
        0.05 * (2_000_000.0_f64.powi(2) + 500_000.0_f64.powi(2)) + 0.02 * 6_000_000.0_f64.powi(2);
    assert!(close_rel(summary.average_annual_loss, expected_aal, 1e-12));
    assert!(close_rel(summary.aggregate_variance, expected_var, 1e-12));

    // Mixed Beta + point-mass ELT: PML satisfies exact quantile bracket across jump discontinuities
    for &rp in &[50.0, 100.0] {
        let pml = elt_oep_pml(&entries, rp).unwrap();
        let target = 1.0 / rp;
        assert!(elt_oep(&entries, (pml - 1.0).max(0.0)).unwrap() >= target - 1e-6);
        assert!(elt_oep(&entries, pml + 1.0).unwrap() <= target + 1e-6);
    }
    // Continuous Beta-only ELT: exact continuous inversion OEP(PML(T)) == 1 / T
    let pml_beta = elt_oep_pml(&entries[..1], 50.0).unwrap();
    assert!(close_rel(
        elt_oep(&entries[..1], pml_beta).unwrap(),
        1.0 / 50.0,
        1e-7
    ));

    // Compound-Poisson AEP via FFT: mean matches AAL and AEP(x) >= OEP(x)
    let aep = elt_aggregate_distribution(&entries, 250_000.0, 256, Some(0.01)).unwrap();
    assert!(close_rel(aep.mean, summary.average_annual_loss, 1e-2));
    let oep_3m = elt_oep(&entries, 3_000_000.0).unwrap();
    let aep_3m = aep.exceedance_probability(3_000_000.0).unwrap();
    assert!(aep_3m >= oep_3m - 1e-3);

    // Occurrence layer expected loss for $4M xs $2M
    let layer = CommercialLayer::new(2_000_000.0, 4_000_000.0, 1.0).unwrap();
    let ceded = elt_occurrence_layer_expected_loss(&entries, &layer).unwrap();
    // Event 2 alone contributes 0.02 * 4M = 80k; Event 1 adds positive Beta tail above 2M
    assert!(ceded > 80_000.0);
    assert!(ceded < summary.average_annual_loss);
}

#[test]
fn test_wang_ph_kreps_and_rmk_risk_loads() {
    let losses = [0.0, 100.0, 250.0, 600.0];
    let probs = [0.40, 0.30, 0.20, 0.10];
    let mean = 0.30 * 100.0 + 0.20 * 250.0 + 0.10 * 600.0; // 140.0

    // Risk-neutral limits recover exact expected loss
    for op in [
        DistortionOperator::ProportionalHazards { rho: 1.0 },
        DistortionOperator::Wang { lambda: 0.0 },
        DistortionOperator::DualPower { alpha: 1.0 },
    ] {
        let res = distortion_price_discrete(&losses, &probs, op).unwrap();
        assert!(close_rel(res.expected_loss, mean, 1e-12));
        assert!(close_rel(res.risk_adjusted_price, mean, 1e-12));
        assert!(res.risk_load.abs() < 1e-12);
    }

    // Wang (2000) exact Lognormal theorem: E_lambda[X] = exp(mu + 0.5*sigma^2 + lambda*sigma)
    let mu = 4.0;
    let sigma = 0.5;
    let lambda = 0.35;
    let ln_dist = SeverityDistribution::Lognormal { mu, sigma };
    let pmf = discretize_severity_mean_preserving(&ln_dist, 1.0, 1024).unwrap();
    let agg = AggregateLossDistribution::from_pmf(1.0, pmf).unwrap();
    let wang_res = distortion_price_aggregate(&agg, DistortionOperator::Wang { lambda }).unwrap();
    let exact_wang = (mu + 0.5 * sigma * sigma + lambda * sigma).exp();
    assert!(close_rel(wang_res.risk_adjusted_price, exact_wang, 1e-3));

    // Kreps (1990) marginal surplus risk load
    let kreps = kreps_marginal_risk_load(1_000.0, 200.0, 10_000.0, 20_000.0, 3.0, 0.15).unwrap();
    let expected_sd_after = (1_000_000.0 + 10_000.0 + 40_000.0_f64).sqrt();
    let expected_rl = (0.15 / 1.15) * 3.0 * (expected_sd_after - 1_000.0);
    assert!(close_rel(kreps.risk_load, expected_rl, 1e-12));
    assert!(kreps_marginal_risk_load(100.0, 50.0, 100.0, 5_000.0, 3.0, 0.15).is_err());

    // Ruhm-Mango-Kreps exact portfolio additivity and permutation invariance under ties
    let l1 = [10.0, 20.0, 30.0, 30.0, 80.0];
    let l2 = [5.0, 15.0, 20.0, 20.0, 40.0];
    let rmk =
        rmk_risk_adjusted_pricing(&[&l1, &l2], DistortionOperator::Wang { lambda: 0.4 }).unwrap();
    let sum_prices: f64 = rmk.line_prices.iter().sum();
    assert!(close_rel(sum_prices, rmk.portfolio_price, 1e-12));
    assert!(rmk.portfolio_risk_load > 0.0);

    // Permute scenarios (reversing order) -> identical RMK prices
    let l1_rev = [80.0, 30.0, 30.0, 20.0, 10.0];
    let l2_rev = [40.0, 20.0, 20.0, 15.0, 5.0];
    let rmk_rev = rmk_risk_adjusted_pricing(
        &[&l1_rev, &l2_rev],
        DistortionOperator::Wang { lambda: 0.4 },
    )
    .unwrap();
    assert!(close_rel(rmk.line_prices[0], rmk_rev.line_prices[0], 1e-12));
    assert!(close_rel(rmk.line_prices[1], rmk_rev.line_prices[1], 1e-12));
}

#[test]
fn test_euler_tvar_rvar_sd_and_myers_read_capital_allocation() {
    // 10 scenarios with an intentional tie in aggregate loss (scenarios 5 & 6 both sum to 100)
    let line_a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 40.0, 80.0, 90.0, 150.0];
    let line_b = [30.0, 25.0, 20.0, 35.0, 40.0, 40.0, 60.0, 50.0, 70.0, 100.0];
    let agg: Vec<f64> = line_a
        .iter()
        .zip(line_b.iter())
        .map(|(a, b)| a + b)
        .collect();
    let lines = [line_a.as_slice(), line_b.as_slice()];

    // 1. Euler TVaR conservation: sum AC_i == sample_tvar(S, 0.80)
    let tvar_alloc = euler_tvar_allocation(&lines, 0.80).unwrap();
    let port_tvar = sample_tvar(&agg, 0.80).unwrap();
    assert!(close_rel(
        tvar_alloc.portfolio_risk_measure,
        port_tvar,
        1e-12
    ));
    let sum_ac: f64 = tvar_alloc.allocated_total.iter().sum();
    assert!(close_rel(sum_ac, port_tvar, 1e-12));
    assert!(tvar_alloc.diversification_ratio <= 1.0 + 1e-12);

    // 2. Euler RVaR conservation: sum AC_i == sample_rvar(S, 0.50, 0.90)
    let rvar_alloc = euler_rvar_allocation(&lines, 0.50, 0.90).unwrap();
    let port_rvar = sample_rvar(&agg, 0.50, 0.90).unwrap();
    assert!(close_rel(
        rvar_alloc.portfolio_risk_measure,
        port_rvar,
        1e-12
    ));
    let sum_rvar: f64 = rvar_alloc.allocated_total.iter().sum();
    assert!(close_rel(sum_rvar, port_rvar, 1e-12));

    // 3. Euler SD conservation: sum AC_i == E[S] + 2.5 * SD(S)
    let sd_alloc = euler_sd_allocation(&lines, 2.5).unwrap();
    let sum_sd: f64 = sd_alloc.allocated_total.iter().sum();
    assert!(close_rel(sum_sd, sd_alloc.portfolio_risk_measure, 1e-12));
    assert!(sd_alloc.diversification_ratio <= 1.0 + 1e-12);

    // 4. Myers & Read (2001) exact conservation: sum Q_i == Q and sum A_i == A
    let mr = myers_read_allocation(&lines, 150.0).unwrap();
    let sum_q: f64 = mr.allocated_surplus.iter().sum();
    let sum_a: f64 = mr.allocated_assets.iter().sum();
    assert!(close_rel(sum_q, mr.portfolio_surplus, 1e-12));
    assert!(close_rel(sum_a, 150.0, 1e-12));
    assert!(mr.default_probability > 0.0 && mr.default_put_value > 0.0);
    assert!(sample_var(&agg, 0.90).unwrap() >= sample_var(&agg, 0.50).unwrap());
}

#[test]
fn test_tweedie_negbin_binomial_invgauss_and_ridge_glm() {
    // 1. Tweedie (p = 1.5) with zero-inflated pure-premium outcomes + power selection
    let design = [
        1.0, 0.0, //
        1.0, 0.0, //
        1.0, 0.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
    ];
    let y_tw = [0.0, 80.0, 100.0, 0.0, 220.0, 260.0];
    let w = [1.0; 6];
    let offsets = [0.0; 6];

    let tw = fit_glm_irls_with_options(
        GlmFamily::Tweedie { power: 1.5 },
        &design,
        6,
        2,
        &y_tw,
        &w,
        &offsets,
        &GlmOptions::default(),
    )
    .unwrap();
    // Group 0 mean = 60, Group 1 mean = 160 -> exact log-link recovery
    assert!(close_rel(tw.coefficients[0].exp(), 60.0, 1e-6));
    assert!(close_rel(tw.coefficients[1].exp(), 160.0 / 60.0, 1e-6));

    let sel = select_tweedie_power(
        &[1.2, 1.5, 1.8],
        &design,
        6,
        2,
        &y_tw,
        &w,
        &offsets,
        &GlmOptions::default(),
    )
    .unwrap();
    assert_eq!(sel.profile.len(), 3);
    assert!(sel.best_power > 1.0 && sel.best_power < 2.0);

    // 2. Ridge penalty shrinks non-intercept slope strictly toward 0
    let tw_ridge = fit_glm_irls_with_options(
        GlmFamily::Tweedie { power: 1.5 },
        &design,
        6,
        2,
        &y_tw,
        &w,
        &offsets,
        &GlmOptions {
            ridge_lambda: 10.0,
            ..GlmOptions::default()
        },
    )
    .unwrap();
    assert!(tw_ridge.coefficients[1].abs() < tw.coefficients[1].abs());

    // 3. Negative Binomial approaches Poisson as alpha -> 0
    let y_cnt = [2.0, 4.0, 3.0, 8.0, 10.0, 9.0];
    let pois = fit_glm_irls(
        GlmFamily::Poisson,
        &design,
        6,
        2,
        &y_cnt,
        &w,
        &offsets,
        100,
        1e-10,
    )
    .unwrap();
    let nb_small = fit_glm_irls(
        GlmFamily::NegativeBinomial { alpha: 1e-6 },
        &design,
        6,
        2,
        &y_cnt,
        &w,
        &offsets,
        100,
        1e-10,
    )
    .unwrap();
    assert!(close_rel(
        pois.coefficients[0],
        nb_small.coefficients[0],
        1e-5
    ));
    assert!(close_rel(
        pois.coefficients[1],
        nb_small.coefficients[1],
        1e-5
    ));

    // 4. Binomial / Logistic GLM + AIC/BIC
    let y_bin = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let logit = fit_glm_irls(
        GlmFamily::Binomial,
        &design,
        6,
        2,
        &y_bin,
        &w,
        &offsets,
        100,
        1e-10,
    )
    .unwrap();
    let probs = predict_glm(
        GlmFamily::Binomial,
        &design,
        6,
        2,
        &logit.coefficients,
        &offsets,
    )
    .unwrap();
    // Group 0 empirical p = 1/3, Group 1 empirical p = 2/3
    assert!(close_rel(probs[0], 1.0 / 3.0, 1e-7));
    assert!(close_rel(probs[3], 2.0 / 3.0, 1e-7));
    let ic = glm_information_criteria(GlmFamily::Binomial, &y_bin, &probs, &w, logit.dispersion, 2)
        .unwrap();
    assert!(close_rel(ic.scaled_deviance, logit.deviance, 1e-10));
    assert!(ic.aic > ic.scaled_deviance);

    // 5. Inverse Gaussian severity GLM
    let y_ig = [500.0, 620.0, 580.0, 1_400.0, 1_650.0, 1_550.0];
    let ig = fit_glm_irls(
        GlmFamily::InverseGaussian,
        &design,
        6,
        2,
        &y_ig,
        &w,
        &offsets,
        100,
        1e-10,
    )
    .unwrap();
    assert!(ig.coefficients[1] > 0.5);
    assert!(ig.deviance > 0.0);
}

#[test]
fn test_gini_lorenz_double_lift_and_vif_diagnostics() {
    let exposures = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let actuals = [100.0, 200.0, 300.0, 300.0, 600.0, 900.0];
    let pred_good = [10.0, 20.0, 30.0, 30.0, 60.0, 90.0];
    let pred_flat = [40.0; 6];
    let pred_rev = [90.0, 60.0, 30.0, 30.0, 20.0, 10.0];

    let g_flat = weighted_gini_index(&exposures, &actuals, &pred_flat).unwrap();
    assert!(g_flat.gini_index.abs() < 1e-12);

    let g_good = weighted_gini_index(&exposures, &actuals, &pred_good).unwrap();
    let g_rev = weighted_gini_index(&exposures, &actuals, &pred_rev).unwrap();
    assert!(g_good.gini_index > 0.30);
    assert!(close_rel(g_good.gini_index, -g_rev.gini_index, 1e-12));

    // Double-lift chart conserves total exposure and actual loss across buckets
    let dl = double_lift_chart(&exposures, &actuals, &pred_flat, &pred_good, 3).unwrap();
    assert_eq!(dl.len(), 3);
    let tot_exp: f64 = dl.iter().map(|b| b.exposure).sum();
    let tot_act: f64 = dl.iter().map(|b| b.exposure * b.actual_pure_premium).sum();
    assert!(close_rel(tot_exp, 60.0, 1e-12));
    assert!(close_rel(tot_act, 2400.0, 1e-12));
    assert!(dl[0].mean_ratio_b_to_a < dl[1].mean_ratio_b_to_a);
    assert!(dl[1].mean_ratio_b_to_a < dl[2].mean_ratio_b_to_a);

    // VIF: orthogonal predictors -> VIF == 1.0; correlated predictors -> VIF > 1.0
    let ortho = [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let vif_ortho = variance_inflation_factors(&ortho, 4, 2).unwrap();
    assert!(close_rel(vif_ortho[0], 1.0, 1e-12));
    assert!(close_rel(vif_ortho[1], 1.0, 1e-12));

    let corr = [1.0, 1.1, 2.0, 2.0, 3.0, 2.9, 4.0, 4.1];
    let vif_corr = variance_inflation_factors(&corr, 4, 2).unwrap();
    assert!(vif_corr[0] > 10.0 && vif_corr[1] > 10.0);
}

#[test]
fn test_subannual_parallelogram_two_step_trend_and_hachemeister() {
    // 1. Annual policy_term_years = 1.0 matches parallelogram_on_level_factor identically
    let changes = [
        RateChange {
            effective_date: 2021.5,
            rate_factor: 1.08,
        },
        RateChange {
            effective_date: 2022.25,
            rate_factor: 1.05,
        },
    ];
    let ann = parallelogram_on_level_factor(2022.0, &changes).unwrap();
    let ann_term = parallelogram_on_level_factor_with_term(2022.0, 1.0, &changes).unwrap();
    assert!(close_rel(ann, ann_term, 1e-12));

    // 2. 6-month policy term (T = 0.5) with single mid-year rate change (+10% on July 1):
    // Area after d = 2022.5 is the upper-right triangle of width 0.5 and height 1.0 -> 0.25.
    // Avg level = 0.75 * 1.0 + 0.25 * 1.10 = 1.025 -> OLF = 1.10 / 1.025
    let mid_year = [RateChange {
        effective_date: 2022.5,
        rate_factor: 1.10,
    }];
    let semi_olf = parallelogram_on_level_factor_with_term(2022.0, 0.5, &mid_year).unwrap();
    assert!(close_rel(semi_olf, 1.10 / 1.025, 1e-12));

    // 3. Two-step exponential trend model
    let two_step = TwoStepTrendModel::new(0.04, 0.06).unwrap();
    let f = two_step.compound_factor(2.0, 1.5).unwrap();
    assert!(close_rel(f, 1.04_f64.powf(2.0) * 1.06_f64.powf(1.5), 1e-12));

    // 4. Hachemeister (1975) regression credibility:
    // Large-exposure group retains near-full credibility; small noisy group shrinks toward collective.
    let t = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y_large = [100.0, 108.0, 117.0, 124.0, 133.0];
    let w_large = [1000.0; 5];
    let y_med = [90.0, 95.0, 101.0, 106.0, 112.0];
    let w_med = [500.0; 5];
    let y_small_noisy = [120.0, 95.0, 150.0, 110.0, 185.0];
    let w_small = [5.0; 5];

    let hach = hachemeister_credibility(
        &[&t, &t, &t],
        &[&y_large, &y_med, &y_small_noisy],
        &[&w_large, &w_med, &w_small],
    )
    .unwrap();

    assert!(hach.epv > 0.0);
    // Small noisy group's slope is shrunk closer to collective slope than its raw WLS slope
    let raw_dist = (hach.groups[2].wls_beta[1] - hach.collective_beta[1]).abs();
    let cred_dist = (hach.groups[2].credibility_beta[1] - hach.collective_beta[1]).abs();
    assert!(cred_dist < raw_dist);
    assert!(hach.groups[0].predict(6.0) > hach.groups[0].predict(5.0));
}

#[test]
fn test_phase1_anchor_types_builders_and_declarative_macros() {
    use pricing_kernels::{rate_changes, tower};

    // 1. DesignMatrix + GlmFit fluent builder (unpenalized & ridge QR parity)
    let design = DesignMatrix::try_from_rows(&[
        &[1.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[1.0, 1.0, 0.0, 0.0],
        &[1.0, 1.0, 0.0, 1.0],
        &[1.0, 0.0, 1.0, 0.0],
        &[1.0, 0.0, 1.0, 1.0],
    ])
    .expect("design matrix");
    let exposures = [1235.0, 582.0, 2845.0, 1204.0, 890.0, 410.0];
    let loss_costs = [105.4, 132.8, 92.1, 116.5, 81.0, 102.3];
    let total_losses: Vec<f64> = exposures
        .iter()
        .zip(loss_costs.iter())
        .map(|(&p, &a)| p * a)
        .collect();
    let offsets: Vec<f64> = exposures.iter().map(|&p| p.ln()).collect();

    let fit = GlmFit::new(GlmFamily::Poisson, &design, &total_losses)
        .offsets(&offsets)
        .tol(1e-12)
        .fit()
        .expect("GlmFit Poisson");
    let preds = design
        .predict(GlmFamily::Poisson, &fit.coefficients, Some(&offsets))
        .expect("predict");
    let obs_sum: f64 = total_losses.iter().sum();
    let pred_sum: f64 = preds.iter().sum();
    assert!(close_rel(pred_sum, obs_sum, 1e-9));

    // Ridge-penalized QR GLM shrinks non-intercept coefficients toward 0
    let fit_ridge = GlmFit::new(GlmFamily::Poisson, &design, &total_losses)
        .offsets(&offsets)
        .ridge_lambda(50.0)
        .fit()
        .expect("GlmFit Ridge");
    let unpen_norm: f64 = fit.coefficients[1..].iter().map(|b| b * b).sum();
    let ridge_norm: f64 = fit_ridge.coefficients[1..].iter().map(|b| b * b).sum();
    assert!(ridge_norm < unpen_norm);

    // 2. RatingTable + SolverOptions
    let table =
        RatingTable::try_new(3, 2, exposures.to_vec(), loss_costs.to_vec()).expect("rating table");
    let mb = table
        .bailey_simon_multiplicative(&SolverOptions::default().with_tol(1e-12))
        .expect("mb");
    assert!(close_rel(fit.coefficients[0].exp(), mb.base_rate, 1e-8));

    // 3. LayerTower + tower! macro
    let sev = SeverityDistribution::lognormal_from_mean_cv(100_000.0, 1.5).unwrap();
    let t = tower![
        100_000.0 xs 0.0,
        150_000.0 xs 100_000.0 @ 1.0,
    ]
    .expect("tower");
    assert_eq!(t.len(), 2);
    let total_el = t.total_expected_loss(&sev).unwrap();
    let lev_250k = sev.limited_expected_value(250_000.0).unwrap();
    assert!(close_rel(total_el, lev_250k, 1e-12));
    assert!(close_rel(t.total_payout(180_000.0), 180_000.0, 1e-12));

    // 4. rate_changes! macro
    let changes = rate_changes![2021.5 => 1.10];
    let olf = parallelogram_on_level_factor(2022.0, &changes).unwrap();
    assert!(close_rel(olf, 1.10 / 1.0875, 1e-12));

    // 5. EventLossTable anchor
    let elt = EventLossTable::try_new(vec![
        EventLossEntry {
            event_id: 1,
            annual_rate: 0.05,
            mean_loss: 1_000_000.0,
            std_dev_indep: 200_000.0,
            std_dev_corr: 100_000.0,
            max_loss: 5_000_000.0,
        },
        EventLossEntry {
            event_id: 2,
            annual_rate: 0.02,
            mean_loss: 3_000_000.0,
            std_dev_indep: 500_000.0,
            std_dev_corr: 300_000.0,
            max_loss: 12_000_000.0,
        },
    ])
    .expect("elt");
    assert_eq!(elt.len(), 2);
    let summary = elt.summary().unwrap();
    assert!(close_rel(
        summary.average_annual_loss,
        0.05 * 1e6 + 0.02 * 3e6,
        1e-9
    ));
    assert!(elt.oep_pml(100.0).unwrap() > 0.0);

    // 6. RetroPlanSpec::solve_table_m + NcciErpSpec alias
    let table_m =
        TableMEngine::from_discrete_pdf(&[0.2, 0.6, 1.0, 1.4, 1.8], &[0.1, 0.2, 0.4, 0.2, 0.1])
            .unwrap();
    let retro_spec = RetroPlanSpec {
        standard_premium: 500_000.0,
        expected_loss_ratio: 0.60,
        lcf: 1.12,
        tax_multiplier: 1.04,
        max_retro_ratio: 1.40,
        min_retro_ratio: 0.50,
        expense_ratio: 0.20,
    };
    let retro = retro_spec
        .solve_table_m(&table_m, &SolverOptions::default())
        .unwrap();
    assert!(retro.basic_premium_factor > 0.0);

    let ncci_alias: NcciErpSpec = NcciErpSpec {
        split_point: 18_500.0,
        expected_losses: 100_000.0,
        d_ratio: 0.40,
        weight_w: 0.15,
        ballast_b: 25_000.0,
        apply_era: true,
    };
    let ncci_res = calculate_ncci_experience_mod(&ncci_alias, &[]).unwrap();
    assert!(ncci_res.mod_factor < 1.0);
}

#[test]
fn test_numerical_stability_and_validation_refinements() {
    // 1. Centered TrendModel::fit_exponential on calendar years (2020..2025)
    let years = [2020.0_f64, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0];
    let true_rate = 0.065_f64;
    let vals: Vec<f64> = years
        .iter()
        .map(|&y| 100.0 * (1.0 + true_rate).powf(y - 2020.0))
        .collect();
    let fitted = TrendModel::fit_exponential(&years, &vals).unwrap();
    assert!(close_rel(fitted.annual_rate, true_rate, 1e-12));

    // 2. LER rejects deductible > policy_limit
    let lomax = SeverityDistribution::Lomax {
        alpha: 2.2,
        theta: 100_000.0,
    };
    assert!(lomax.loss_elimination_ratio(600_000.0, 500_000.0).is_err());

    // 3. burning_cost_xl rejects duplicate treaty years and invalid CommercialLayer
    let dup_years = [
        HistoricalTreatyYear {
            year: 2023,
            subject_premium: 1_000_000.0,
            on_level_factor: 1.0,
            exposure_trend_factor: 1.0,
        },
        HistoricalTreatyYear {
            year: 2023,
            subject_premium: 1_200_000.0,
            on_level_factor: 1.0,
            exposure_trend_factor: 1.0,
        },
    ];
    let valid_layer = CommercialLayer::new(100_000.0, 400_000.0, 1.0).unwrap();
    assert!(burning_cost_xl(&dup_years, &[], &valid_layer).is_err());
    let invalid_layer = CommercialLayer {
        attachment: -1.0,
        limit: 100_000.0,
        participation: 1.0,
    };
    assert!(burning_cost_xl(&dup_years[..1], &[], &invalid_layer).is_err());
    assert!(LayerTower::try_new(vec![invalid_layer]).is_err());

    // 4. glm_information_criteria rejects out-of-domain mu/y
    assert!(
        glm_information_criteria(
            GlmFamily::Binomial,
            &[0.5, 0.5],
            &[1.2, 0.5],
            &[1.0, 1.0],
            1.0,
            1
        )
        .is_err()
    );
    assert!(
        glm_information_criteria(
            GlmFamily::InverseGaussian,
            &[0.0, 1.0],
            &[0.5, 1.0],
            &[1.0, 1.0],
            1.0,
            1
        )
        .is_err()
    );

    // 5. VIF rejects constant single predictor column
    assert!(variance_inflation_factors(&[5.0, 5.0, 5.0, 5.0], 4, 1).is_err());

    // 6. AggregateLossDistribution::from_pmf rejects negative mass; FFT guards tilt overflow
    assert!(AggregateLossDistribution::from_pmf(1.0, vec![0.5, -0.1, 0.6]).is_err());
    assert!(
        fft_aggregate_loss(
            FrequencyDistribution::Poisson { lambda: 1.0 },
            &[0.5, 0.5],
            1.0,
            1024,
            Some(1.0),
        )
        .is_err()
    );

    // 7. Hachemeister with identical groups (zero VHM) and tiny EPV does not falsely error
    let t = [1.0, 2.0, 3.0, 4.0];
    let w = [10.0, 10.0, 10.0, 10.0];
    let y1 = [10.0 + 1e-6, 12.0, 14.0 - 1e-6, 16.0];
    let y2 = [10.0 - 1e-6, 12.0, 14.0 + 1e-6, 16.0];
    let hach = hachemeister_credibility(&[&t, &t], &[&y1, &y2], &[&w, &w]).unwrap();
    assert!(hach.epv > 0.0);
    for g in &hach.groups {
        assert!(g.credibility_beta[0].is_finite() && g.credibility_beta[1].is_finite());
    }
}

#[cfg(feature = "wasm")]
#[test]
fn test_wasm_exports_guard_overflow_and_basic_calls() {
    use pricing_kernels::wasm_exports::{pk_ilf_gamma, pk_max_claims, pk_ncci_experience_mod};

    let ilf = pk_ilf_gamma(2.5, 40_000.0, 1_000_000.0, 250_000.0);
    assert!(ilf.is_finite() && ilf > 1.0);

    // Exceeding MAX_CLAIMS must return NaN rather than silently truncating.
    let overflow_mod = pk_ncci_experience_mod(
        18_500.0,
        100_000.0,
        0.4,
        0.15,
        25_000.0,
        1,
        pk_max_claims() + 1,
    );
    assert!(overflow_mod.is_nan());
}
