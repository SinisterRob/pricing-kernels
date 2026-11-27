use criterion::{Criterion, black_box, criterion_group, criterion_main};
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

criterion_group!(
    benches,
    bench_classification,
    bench_credibility_and_trending,
    bench_experience_and_retro,
    bench_layers_and_reinsurance
);
criterion_main!(benches);
