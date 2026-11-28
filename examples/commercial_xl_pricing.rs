//! End-to-end commercial excess severity fitting, treaty XL exposure + experience blending,
//! and aggregate-to-Table-M retrospective rating workflow.
//!
//! Run with: `cargo run --example commercial_xl_pricing`

use pricing_kernels::{prelude::*, tower};

fn main() -> Result<(), PricingError> {
    // 1. Fit parametric severity to left-truncated ($10k deductible) and right-censored ($1M limit) claims.
    let observations = vec![
        SeverityObservation::truncated(22_000.0, 10_000.0),
        SeverityObservation::truncated(35_000.0, 10_000.0),
        SeverityObservation::truncated(48_000.0, 10_000.0),
        SeverityObservation::truncated(75_000.0, 10_000.0),
        SeverityObservation::truncated(110_000.0, 10_000.0),
        SeverityObservation::truncated(185_000.0, 10_000.0),
        SeverityObservation::truncated(320_000.0, 10_000.0),
        SeverityObservation::truncated(540_000.0, 10_000.0),
        SeverityObservation::censored(1_000_000.0, 10_000.0),
        SeverityObservation::censored(1_000_000.0, 10_000.0),
    ];

    let candidates = fit_best_severity(
        &[
            SeverityFamily::Lomax,
            SeverityFamily::Lognormal,
            SeverityFamily::Gamma,
            SeverityFamily::Exponential,
        ],
        &observations,
        &SolverOptions::default(),
    )?;
    let best = &candidates[0];
    println!(
        "Best AIC severity fit: {:?} (AIC = {:.2}, LL = {:.2})",
        best.distribution, best.aic, best.log_likelihood
    );

    // 2. Price a 2-layer commercial excess tower via declarative `tower!`.
    let xl_tower = tower![
        500_000.0 xs 500_000.0,
        4_000_000.0 xs 1_000_000.0 @ 0.5,
    ]?;
    let layer_costs = xl_tower.expected_losses(&best.distribution)?;
    println!("Tower expected claim severity by layer: {layer_costs:.2?}");

    // 3. Treaty experience-rated burning cost blended with Swiss Re MBBEFD exposure curve.
    let experience = TreatyExperience::try_new(
        vec![
            HistoricalTreatyYear {
                year: 2022,
                subject_premium: 10_000_000.0,
                on_level_factor: 1.12,
                exposure_trend_factor: 1.06,
            },
            HistoricalTreatyYear {
                year: 2023,
                subject_premium: 11_000_000.0,
                on_level_factor: 1.05,
                exposure_trend_factor: 1.03,
            },
        ],
        vec![
            HistoricalTreatyClaim {
                year: 2022,
                incurred: 850_000.0,
                severity_trend_factor: 1.15,
                development_factor: 1.05,
            },
            HistoricalTreatyClaim {
                year: 2023,
                incurred: 620_000.0,
                severity_trend_factor: 1.08,
                development_factor: 1.15,
            },
        ],
    )?;

    let working_layer = xl_tower.layers()[0];
    let bc = experience.burning_cost(&working_layer)?;
    let y3 = MbbefdCurve::from_preset(SwissReCurve::Y3);
    let exposure_factor = y3.layer_exposure_factor(500_000.0, 500_000.0, 5_000_000.0)?;
    let exposure_rate = 0.60 * exposure_factor;

    let blend = blend_exposure_and_experience(
        exposure_rate,
        bc.overall_burning_cost_rate,
        BlendCredibilityRule::SquareRoot {
            claim_count: bc.total_layer_claims as f64,
            full_credibility_claims: 25.0,
        },
        Some(12_000_000.0),
        0.15,
    )?;
    println!(
        "XL Blend: exposure={:.4}, experience={:.4}, Z={:.3}, blended={:.4}, tech_prem=${:.2}",
        blend.exposure_rate,
        blend.experience_rate,
        blend.credibility_z,
        blend.blended_rate,
        blend.technical_premium.unwrap_or(0.0)
    );

    // 4. Bridge Panjer compound aggregate distribution directly into Table M -> Retro Plan.
    let disc = discretize_severity_mean_preserving(&best.distribution, 50_000.0, 21)?;
    let agg = panjer_recursion(
        FrequencyDistribution::Poisson { lambda: 4.0 },
        &disc,
        50_000.0,
        512,
    )?;
    let table_m = TableMEngine::from_aggregate(&agg)?;
    let retro = RetroPlanSpec {
        standard_premium: 1_500_000.0,
        expected_loss_ratio: 0.62,
        lcf: 1.10,
        tax_multiplier: 1.035,
        max_retro_ratio: 1.35,
        min_retro_ratio: 0.55,
        expense_ratio: 0.18,
    }
    .solve_table_m(&table_m, &SolverOptions::default())?;
    println!(
        "Balanced Retro Plan from Panjer Table M: BPF={:.4}, Basic=${:.2}",
        retro.basic_premium_factor, retro.basic_premium_dollars
    );

    Ok(())
}
