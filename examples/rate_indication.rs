//! End-to-end commercial lines overall rate indication and GLM relativity exhibit
//! (Werner & Modlin *Basic Ratemaking*, Ch. 5–9).
//!
//! Run with: `cargo run --example rate_indication`

use pricing_kernels::{datasets, prelude::*, rate_changes};

fn main() -> Result<(), PricingError> {
    // 1. Historical rate change schedule -> Calendar-Year parallelogram on-level factors.
    let changes = rate_changes![
        2021.5 => 1.06,
        2022.5 => 1.08,
        2023.75 => 1.05,
    ];
    let olf_2022 = parallelogram_on_level_factor(2022.0, &changes)?;
    let olf_2023 = parallelogram_on_level_factor(2023.0, &changes)?;
    let olf_2024 = parallelogram_on_level_factor(2024.0, &changes)?;

    // 2. Two-step exponential loss trend (historical +4.0%, prospective +5.5%).
    let loss_trend = TwoStepTrendModel::new(0.04, 0.055)?;
    let prem_trend = TrendModel::new(0.02);

    let spec = RateIndicationSpec {
        periods: vec![
            RateIndicationPeriod {
                year: 2022,
                earned_exposures: 10_000.0,
                earned_premium: 8_200_000.0,
                on_level_factor: olf_2022,
                premium_trend_factor: prem_trend.compound_factor(3.5),
                reported_loss: 5_100_000.0,
                loss_development_factor: 1.04,
                loss_trend_factor: loss_trend.compound_factor(2.0, 1.5)?,
                alae_load_factor: 1.08,
                weight: None,
            },
            RateIndicationPeriod {
                year: 2023,
                earned_exposures: 10_400.0,
                earned_premium: 9_100_000.0,
                on_level_factor: olf_2023,
                premium_trend_factor: prem_trend.compound_factor(2.5),
                reported_loss: 5_400_000.0,
                loss_development_factor: 1.12,
                loss_trend_factor: loss_trend.compound_factor(1.0, 1.5)?,
                alae_load_factor: 1.08,
                weight: None,
            },
            RateIndicationPeriod {
                year: 2024,
                earned_exposures: 10_800.0,
                earned_premium: 10_200_000.0,
                on_level_factor: olf_2024,
                premium_trend_factor: prem_trend.compound_factor(1.5),
                reported_loss: 5_200_000.0,
                loss_development_factor: 1.32,
                loss_trend_factor: loss_trend.compound_factor(0.0, 1.5)?,
                alae_load_factor: 1.08,
                weight: None,
            },
        ],
        expenses: ExpenseProvisions {
            fixed_expense_ratio: 0.08,
            fixed_expense_per_exposure: 85.0,
            variable_expense_ratio: 0.17,
            profit_and_contingencies_ratio: 0.05,
            ulae_factor: 1.05,
        },
        credibility_z: Some(0.85),
        complement_of_credibility: 0.04,
    };

    let indication = spec.calculate()?;
    println!("{indication}");

    // 3. Fit a Poisson classification GLM and export rate-filing relativities + scorer.
    let grid = datasets::commercial_classification_grid();
    let dm = DesignMatrix::try_from_rows(&[
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
    ])?;
    let total_losses: Vec<f64> = grid
        .exposures
        .iter()
        .zip(grid.loss_costs.iter())
        .map(|(&p, &a)| p * a)
        .collect();
    let offsets: Vec<f64> = grid.exposures.iter().map(|&p| p.ln()).collect();

    let fit = GlmFit::new(GlmFamily::Poisson, &dm, &total_losses)
        .offsets(&offsets)
        .fit()?;
    let rels = fit.relativities(Some(&["Base", "Class B", "Class C", "Territory 2"]))?;
    println!("GLM Rating Relativities (95% Wald CI):");
    for row in &rels {
        println!(
            "  {:<12} rel={:.4}  95% CI=[{:.4}, {:.4}]  z={:+.2}",
            row.name, row.relativity, row.ci_low_95, row.ci_high_95, row.z_score
        );
    }

    let rater = fit.to_rater()?;
    let quote = rater.score(10.0, &[rels[1].relativity, rels[3].relativity]);
    println!("\nSample 10-exposure quote (Class B, Territory 2): ${quote:.2}");

    Ok(())
}
