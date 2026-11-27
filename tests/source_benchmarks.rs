//! Source-document benchmark, exact actuarial identity, and adversarial validation suite.

use pricing_kernels::linalg::{cholesky_solve, mat_mul, solve_lower_into};
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

    // Public linalg functions reject mismatched slices instead of panicking
    let mut y = [0.0; 1];
    assert!(solve_lower_into(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0, 2.0], &mut y).is_err());
    let mut c = [0.0; 1];
    assert!(mat_mul(&[1.0, 2.0], &[3.0, 4.0], 1, 2, 2, &mut c).is_err());
    assert!(cholesky_solve(&[1.0, 0.0, 0.0, 1.0], 2, &[1.0, f64::NAN]).is_err());
}
