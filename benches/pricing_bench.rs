use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
        })
    });

    let rater = MultiplicativeRater::new(850.0)
        .with_minimum_premium(250.0)
        .with_maximum_premium(10_000.0);
    let factors = [1.25, 0.90, 1.15, 1.05, 0.95];

    group.bench_function("scoring_multiplicative_single", |b| {
        b.iter(|| rater.score(black_box(12.5), black_box(&factors)))
    });

    group.finish();
}

fn bench_experience_and_retro(c: &mut Criterion) {
    let mut group = c.benchmark_group("experience_and_retro");

    let spec = NcciErpSpecification {
        split_point: 18_500.0,
        expected_losses: 150_000.0,
        d_ratio: 0.35,
        weight_w: 0.22,
        ballast_b: 22_500.0,
        apply_era: true,
    };

    let claims = vec![
        ExperienceClaim { incurred: 45_000.0, is_medical_only: false },
        ExperienceClaim { incurred: 8_200.0, is_medical_only: false },
        ExperienceClaim { incurred: 12_500.0, is_medical_only: true },
        ExperienceClaim { incurred: 3_100.0, is_medical_only: true },
        ExperienceClaim { incurred: 65_000.0, is_medical_only: false },
    ];

    group.bench_function("ncci_experience_mod", |b| {
        b.iter(|| calculate_ncci_experience_mod(black_box(&spec), black_box(&claims)))
    });

    let entry_ratios = vec![0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
    let probs = vec![0.05, 0.10, 0.15, 0.20, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02];
    let table_m = TableMEngine::from_discrete_pdf(&entry_ratios, &probs).unwrap();

    let retro_spec = RetroPlanSpec {
        standard_premium: 800_000.0,
        expected_loss_ratio: 0.62,
        lcf: 1.12,
        tax_multiplier: 1.035,
        max_retro_ratio: 1.40,
        min_retro_ratio: 0.50,
        expense_ratio: 0.18,
    };

    group.bench_function("solve_balanced_retro_plan", |b| {
        b.iter(|| solve_balanced_retro_plan(black_box(&retro_spec), black_box(&table_m), 50, 1e-5))
    });

    group.finish();
}

criterion_group!(benches, bench_classification, bench_experience_and_retro);
criterion_main!(benches);
