use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pricing_kernels::datasets;
use pricing_kernels::prelude::*;

fn bench_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification");

    // 10x10 synthetic grid
    let n_rows = 10;
    let n_cols = 10;
    let mut exposures = Vec::with_capacity(100);
    let mut losses = Vec::with_capacity(100);
    for i in 0..n_rows {
        for j in 0..n_cols {
            exposures.push(1000.0 + (i as f64 * 50.0) + (j as f64 * 30.0));
            losses.push(100.0 * (1.0 + i as f64 * 0.1) * (1.0 + j as f64 * 0.05));
        }
    }

    group.bench_function("bailey_simon_10x10", |b| {
        b.iter(|| {
            bailey_simon_multiplicative(
                black_box(n_rows),
                black_box(n_cols),
                black_box(&exposures),
                black_box(&losses),
                black_box(50),
                black_box(1e-5),
            )
        });
    });

    let grid = datasets::commercial_classification_grid();
    let design = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
    ];
    let total_losses: Vec<f64> = grid
        .exposures
        .iter()
        .zip(grid.loss_costs.iter())
        .map(|(&p, &a)| p * a)
        .collect();
    let weights = [1.0; 6];
    let offsets: Vec<f64> = grid.exposures.iter().map(|&p| p.ln()).collect();

    group.bench_function("glm_irls_poisson_3x2", |b| {
        b.iter(|| {
            fit_glm_irls(
                GlmFamily::Poisson,
                black_box(&design),
                6,
                4,
                black_box(&total_losses),
                black_box(&weights),
                black_box(&offsets),
                50,
                1e-9,
            )
        });
    });

    let rater = MultiplicativeRater::new(850.0)
        .with_minimum_premium(250.0)
        .with_maximum_premium(10_000.0);
    let factors = [1.25, 0.90, 1.15, 1.05, 0.95];

    group.bench_function("scoring_multiplicative_single", |b| {
        b.iter(|| rater.score(black_box(12.5), black_box(&factors)));
    });

    let tweedie_y = [0.0, 45.0, 0.0, 120.0, 35.0, 210.0];
    let tweedie_opts = GlmOptions {
        ridge_lambda: 0.5,
        ..GlmOptions::default()
    };
    group.bench_function("glm_irls_tweedie_ridge_3x2", |b| {
        b.iter(|| {
            fit_glm_irls_with_options(
                GlmFamily::Tweedie { power: 1.5 },
                black_box(&design),
                6,
                4,
                black_box(&tweedie_y),
                black_box(&weights),
                black_box(&offsets),
                black_box(&tweedie_opts),
            )
        });
    });

    let gini_exp: Vec<f64> = (1..=200_i32).map(f64::from).collect();
    let gini_pred: Vec<f64> = (1..=200_i32).map(|i| 50.0 + f64::from(i) * 1.5).collect();
    let gini_act: Vec<f64> = gini_exp
        .iter()
        .zip(gini_pred.iter())
        .map(|(&e, &p)| e * p)
        .collect();
    group.bench_function("weighted_gini_200", |b| {
        b.iter(|| {
            weighted_gini_index(
                black_box(&gini_exp),
                black_box(&gini_act),
                black_box(&gini_pred),
            )
        });
    });

    group.finish();
}

fn bench_credibility_and_trending(c: &mut Criterion) {
    let mut group = c.benchmark_group("credibility_and_trending");

    let bs = datasets::buhlmann_straub_example();
    let loss_slices: Vec<&[f64]> = bs.group_losses.iter().map(Vec::as_slice).collect();
    let weight_slices: Vec<&[f64]> = bs.group_weights.iter().map(Vec::as_slice).collect();

    group.bench_function("buhlmann_straub_3x3", |b| {
        b.iter(|| buhlmann_straub(black_box(&loss_slices), black_box(&weight_slices)));
    });

    let changes = [
        RateChange {
            effective_date: 2021.5,
            rate_factor: 1.05,
        },
        RateChange {
            effective_date: 2022.25,
            rate_factor: 1.08,
        },
        RateChange {
            effective_date: 2022.75,
            rate_factor: 1.04,
        },
    ];
    group.bench_function("parallelogram_on_level_3_changes", |b| {
        b.iter(|| parallelogram_on_level_factor(black_box(2022.0), black_box(&changes)));
    });

    group.finish();
}

fn bench_experience_and_retro(c: &mut Criterion) {
    let mut group = c.benchmark_group("experience_and_retro");

    let ncci = datasets::ncci_erp_worksheet();
    group.bench_function("ncci_experience_mod", |b| {
        b.iter(|| calculate_ncci_experience_mod(black_box(&ncci.spec), black_box(&ncci.claims)));
    });

    let retro = datasets::retro_plan_benchmark();
    let table_m =
        TableMEngine::from_discrete_pdf(&retro.entry_ratios, &retro.probabilities).unwrap();

    group.bench_function("solve_balanced_retro_plan", |b| {
        b.iter(|| solve_balanced_retro_plan(black_box(&retro.spec), black_box(&table_m), 50, 1e-5));
    });

    let limited_ratios = [0.18, 0.54, 0.90, 1.26, 1.62];
    let table_l =
        TableLEngine::from_limited_discrete_pdf(&limited_ratios, &retro.probabilities).unwrap();
    group.bench_function("solve_balanced_retro_plan_table_l", |b| {
        b.iter(|| {
            solve_balanced_retro_plan_table_l(black_box(&retro.spec), black_box(&table_l), 50, 1e-5)
        });
    });

    group.finish();
}

fn bench_layers_and_reinsurance(c: &mut Criterion) {
    let mut group = c.benchmark_group("layers_and_reinsurance");

    let lognormal = SeverityDistribution::Lognormal {
        mu: 10.5,
        sigma: 1.35,
    };
    group.bench_function("ilf_lognormal", |b| {
        b.iter(|| lognormal.increased_limits_factor(black_box(1_000_000.0), black_box(100_000.0)));
    });

    let mixed_exp = SeverityDistribution::MixedExponential {
        weights: vec![0.6, 0.3, 0.1],
        means: vec![25_000.0, 150_000.0, 1_200_000.0],
    };
    group.bench_function("ilf_mixed_exponential", |b| {
        b.iter(|| mixed_exp.increased_limits_factor(black_box(2_000_000.0), black_box(250_000.0)));
    });

    let y3 = MbbefdCurve::from_preset(SwissReCurve::Y3);
    group.bench_function("mbbefd_layer_exposure_factor", |b| {
        b.iter(|| {
            y3.layer_exposure_factor(
                black_box(1_000_000.0),
                black_box(4_000_000.0),
                black_box(10_000_000.0),
            )
        });
    });

    group.finish();
}

fn bench_underwriting_risk(c: &mut Criterion) {
    let mut group = c.benchmark_group("underwriting_risk");

    let freq = FrequencyDistribution::Poisson { lambda: 4.5 };
    let sev_pmf = [0.0, 0.45, 0.30, 0.15, 0.10];

    group.bench_function("panjer_poisson_256", |b| {
        b.iter(|| panjer_recursion(black_box(freq), black_box(&sev_pmf), 100.0, 256));
    });

    group.bench_function("fft_aggregate_1024", |b| {
        b.iter(|| {
            fft_aggregate_loss(
                black_box(freq),
                black_box(&sev_pmf),
                100.0,
                1024,
                Some(0.01),
            )
        });
    });

    let elt = [
        EventLossEntry {
            event_id: 1,
            annual_rate: 0.04,
            mean_loss: 2_500_000.0,
            std_dev_indep: 600_000.0,
            std_dev_corr: 300_000.0,
            max_loss: 10_000_000.0,
        },
        EventLossEntry {
            event_id: 2,
            annual_rate: 0.01,
            mean_loss: 8_000_000.0,
            std_dev_indep: 1_500_000.0,
            std_dev_corr: 1_000_000.0,
            max_loss: 25_000_000.0,
        },
    ];
    group.bench_function("elt_oep_pml_100", |b| {
        b.iter(|| elt_oep_pml(black_box(&elt), black_box(100.0)));
    });

    let line1: Vec<f64> = (0..500_i32).map(|i| 100.0 + f64::from(i) * 0.8).collect();
    let line2: Vec<f64> = (0..500_i32)
        .map(|i| 80.0 + f64::from((i * 7) % 500) * 0.6)
        .collect();
    let lines = [line1.as_slice(), line2.as_slice()];

    group.bench_function("euler_tvar_500_scenarios", |b| {
        b.iter(|| euler_tvar_allocation(black_box(&lines), black_box(0.95)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_classification,
    bench_credibility_and_trending,
    bench_experience_and_retro,
    bench_layers_and_reinsurance,
    bench_underwriting_risk
);
criterion_main!(benches);
