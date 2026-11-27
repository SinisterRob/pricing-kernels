use pricing_kernels::prelude::*;

#[test]
fn test_bailey_simon_two_by_two() {
    // 2x2 synthetic classification test
    // True base = 100, row = [1.0, 1.5], col = [1.0, 1.2]
    // Cell costs = [100.0, 120.0, 150.0, 180.0]
    let exposures = vec![1000.0, 800.0, 500.0, 400.0];
    let losses = vec![100.0, 120.0, 150.0, 180.0];

    let res = bailey_simon_multiplicative(2, 2, &exposures, &losses, 100, 1e-7).unwrap();
    
    assert!((res.base_rate - 100.0).abs() < 1e-3);
    assert!((res.relativities_dim1[0] - 1.0).abs() < 1e-4);
    assert!((res.relativities_dim1[1] - 1.5).abs() < 1e-4);
    assert!((res.relativities_dim2[0] - 1.0).abs() < 1e-4);
    assert!((res.relativities_dim2[1] - 1.2).abs() < 1e-4);
    assert!(res.chi_squared < 1e-4);
}

#[test]
fn test_bailey_simon_canadian_automobile_benchmark() {
    // Bailey & Simon (1960) Table 1 Canadian Merit Rating Data
    // 5 Merit Classes x 2 Territories (or representative structure)
    // Testing convergence and balance property
    let exposures = vec![
        1235.0, 582.0,
        2845.0, 1204.0,
        890.0, 410.0,
    ];
    let losses = vec![
        105.4, 132.8,
        92.1, 116.5,
        81.0, 102.3,
    ];

    let res = bailey_simon_multiplicative(3, 2, &exposures, &losses, 200, 1e-6).unwrap();
    assert!(res.iterations < 50);
    assert!(res.relativities_dim1[0] > res.relativities_dim1[1]);
    assert!(res.relativities_dim2[1] > res.relativities_dim2[0]);
}

#[test]
fn test_glm_irls_poisson() {
    // 4 groups, design with intercept and binary covariate
    // log(mu) = 1.0 + 0.5 * x
    let n_samples = 4;
    let n_features = 2;
    let design = vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        1.0, 1.0,
    ];
    // Expected mu = [exp(1.0), exp(1.0), exp(1.5), exp(1.5)] ~= [2.718, 2.718, 4.482, 4.482]
    let y = vec![2.7, 2.8, 4.5, 4.4];
    let weights = vec![1.0, 1.0, 1.0, 1.0];
    let offsets = vec![0.0, 0.0, 0.0, 0.0];

    let res = fit_glm_irls(GlmFamily::Poisson, &design, n_samples, n_features, &y, &weights, &offsets, 50, 1e-6).unwrap();
    assert!((res.coefficients[0] - 1.0).abs() < 0.1);
    assert!((res.coefficients[1] - 0.5).abs() < 0.1);
    assert!(res.deviance < 0.1);
}

#[test]
fn test_glm_irls_gamma() {
    // Gamma severity model: y ~ Gamma with log link
    let n_samples = 4;
    let n_features = 2;
    let design = vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        1.0, 1.0,
    ];
    // mu = [500, 500, 1000, 1000]
    let y = vec![505.0, 495.0, 1010.0, 990.0];
    let weights = vec![10.0, 12.0, 8.0, 9.0]; // claim counts
    let offsets = vec![0.0, 0.0, 0.0, 0.0];

    let res = fit_glm_irls(GlmFamily::Gamma, &design, n_samples, n_features, &y, &weights, &offsets, 50, 1e-6).unwrap();
    // Intercept should be ln(500) ~= 6.2146
    assert!((res.coefficients[0] - (500.0_f64).ln()).abs() < 0.05);
    // Slope should be ln(1000) - ln(500) = ln(2) ~= 0.6931
    assert!((res.coefficients[1] - (2.0_f64).ln()).abs() < 0.05);
}

#[test]
fn test_multiplicative_rater_and_batch() {
    let rater = MultiplicativeRater::new(100.0)
        .with_minimum_premium(250.0)
        .with_maximum_premium(5000.0);

    // Normal calculation: 100 * 2.0 * 1.2 * 0.9 = 216 -> hits min 250
    assert_eq!(rater.score(2.0, &[1.2, 0.9]), 250.0);

    // Large account: 100 * 10.0 * 1.5 * 1.2 = 1800
    assert_eq!(rater.score(10.0, &[1.5, 1.2]), 1800.0);

    // Cap test: 100 * 50.0 * 2.0 = 10000 -> hits max 5000
    assert_eq!(rater.score(50.0, &[2.0]), 5000.0);
}

#[test]
fn test_classical_credibility() {
    let cred = ClassicalCredibility::standard_90_5();
    assert_eq!(cred.credibility(1082.41), 1.0);
    assert_eq!(cred.credibility(2000.0), 1.0);
    assert_eq!(cred.credibility(0.0), 0.0);
    
    // Half standard: 541.2 claims -> sqrt(0.5) ~= 0.7071
    let z = cred.credibility(1082.41 * 0.5);
    assert!((z - 0.5_f64.sqrt()).abs() < 1e-4);

    let blended = cred.blend(1082.41 * 0.5, 120.0, 100.0);
    let expected = z * 120.0 + (1.0 - z) * 100.0;
    assert!((blended - expected).abs() < 1e-4);
}

#[test]
fn test_buhlmann_straub_credibility() {
    // 3 groups over 3 observation periods
    let g1_losses = [100.0, 110.0, 105.0];
    let g1_weights = [10.0, 10.0, 10.0];

    let g2_losses = [150.0, 160.0, 155.0];
    let g2_weights = [20.0, 20.0, 20.0];

    let g3_losses = [200.0, 210.0, 195.0];
    let g3_weights = [15.0, 15.0, 15.0];

    let losses: Vec<&[f64]> = vec![&g1_losses, &g2_losses, &g3_losses];
    let weights: Vec<&[f64]> = vec![&g1_weights, &g2_weights, &g3_weights];

    let res = buhlmann_straub(&losses, &weights).unwrap();
    assert!(res.k > 0.0);
    assert!(res.epv > 0.0);
    assert!(res.vhm > 0.0);
    assert_eq!(res.credibilities.len(), 3);
    for &z in &res.credibilities {
        assert!(z > 0.0 && z < 1.0);
    }
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
        ExperienceClaim { incurred: 50_000.0, is_medical_only: false }, // 18.5k primary, 31.5k excess
        ExperienceClaim { incurred: 5_000.0, is_medical_only: false },  // 5k primary, 0 excess
        ExperienceClaim { incurred: 10_000.0, is_medical_only: true },  // ERA -> 3k primary, 0 excess
    ];

    let res = calculate_ncci_experience_mod(&spec, &claims).unwrap();
    assert_eq!(res.actual_primary, 18_500.0 + 5_000.0 + 3_000.0);
    assert_eq!(res.actual_excess, 31_500.0);
    assert_eq!(res.expected_primary, 40_000.0);
    assert_eq!(res.expected_excess, 60_000.0);
    assert!(res.mod_factor > 0.5 && res.mod_factor < 2.0);
}

#[test]
fn test_off_balance_calculation() {
    let e = vec![100_000.0, 200_000.0, 100_000.0];
    let m = vec![0.90, 1.10, 1.00];
    // Total E = 400,000
    // Total E*M = 90k + 220k + 100k = 410,000
    // Off balance B = 400 / 410 = 0.9756
    let b = calculate_off_balance(&e, &m).unwrap();
    assert!((b - (400.0 / 410.0)).abs() < 1e-4);
}

#[test]
fn test_table_m_and_retro_plan_solver() {
    // Discrete probability distribution of entry ratios with mean 1.0
    let entry_ratios = vec![0.2, 0.6, 1.0, 1.4, 1.8];
    let probs = vec![0.1, 0.2, 0.4, 0.2, 0.1];

    let table_m = TableMEngine::from_discrete_pdf(&entry_ratios, &probs).unwrap();
    assert_eq!(table_m.charge_at(0.0), 1.0);
    assert!(table_m.charge_at(1.0) > 0.0);
    assert_eq!(table_m.charge_at(2.5), 0.0);
    // Convexity: charge_at(0.5) >= charge_at(1.0)
    assert!(table_m.charge_at(0.5) > table_m.charge_at(1.0));

    let spec = RetroPlanSpec {
        standard_premium: 500_000.0,
        expected_loss_ratio: 0.60,
        lcf: 1.12,
        tax_multiplier: 1.04,
        max_retro_ratio: 1.40,
        min_retro_ratio: 0.50,
        expense_ratio: 0.20,
    };

    let retro_res = solve_balanced_retro_plan(&spec, &table_m, 100, 1e-5).unwrap();
    assert!(retro_res.basic_premium_factor > 0.05 && retro_res.basic_premium_factor < 0.40);
    assert!(retro_res.basic_premium_dollars > 0.0);
    assert!(retro_res.entry_ratio_max > retro_res.entry_ratio_min);
}

#[test]
fn test_severity_ilf_and_layer_expected_loss() {
    let lomax = SeverityDistribution::Lomax { alpha: 2.0, theta: 100_000.0 };
    
    // E[X ^ L] for Lomax with alpha = 2 is theta * (1 - (1 + L / theta)^-1)
    let lev_100k = lomax.limited_expected_value(100_000.0).unwrap();
    let expected_lev = 100_000.0 * (1.0 - 0.5);
    assert!((lev_100k - expected_lev).abs() < 1e-4);

    // ILF relative to 100k base
    let ilf_500k = lomax.increased_limits_factor(500_000.0, 100_000.0).unwrap();
    assert!(ilf_500k > 1.0);

    // Deductible LER
    let ler_10k = lomax.loss_elimination_ratio(10_000.0, 100_000.0).unwrap();
    assert!(ler_10k > 0.0 && ler_10k < 1.0);

    // Commercial excess layer: 400k xs 100k at 100% participation
    let layer = CommercialLayer::new(100_000.0, 400_000.0, 1.0).unwrap();
    let expected_loss = layer.layer_expected_loss(&lomax).unwrap();
    let lev_500k = lomax.limited_expected_value(500_000.0).unwrap();
    assert!((expected_loss - (lev_500k - lev_100k)).abs() < 1e-4);

    // Payout logic
    assert_eq!(layer.payout(50_000.0), 0.0);
    assert_eq!(layer.payout(200_000.0), 100_000.0);
    assert_eq!(layer.payout(1_000_000.0), 400_000.0);
}

#[test]
fn test_parallelogram_and_trending() {
    // Rate change of +10% on July 1, 2022 (2022.5)
    let changes = vec![
        RateChange { effective_date: 2022.5, rate_factor: 1.10 },
    ];

    // For policy year 2022 (written 2022.0 to 2023.0):
    // Half written at 1.0, half written at 1.10 -> average historical rate = 1.05
    // On-level factor = 1.10 / 1.05 = 1.0476
    let factor = parallelogram_on_level_factor(2022.0, &changes).unwrap();
    assert!((factor - (1.10 / 1.05)).abs() < 1e-4);

    // Trending model: +5% annual inflation
    let trend = TrendModel::new(0.05);
    let f2 = trend.compound_factor(2.0);
    assert!((f2 - 1.1025).abs() < 1e-5);

    // Fit trend from points (0, 100), (1, 105), (2, 110.25)
    let fitted = TrendModel::fit_exponential(&[0.0, 1.0, 2.0], &[100.0, 105.0, 110.25]).unwrap();
    assert!((fitted.annual_rate - 0.05).abs() < 1e-4);
}
