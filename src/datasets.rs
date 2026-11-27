//! Zero-I/O published CAS Exam 5 and Exam 8 benchmark datasets.
//!
//! Provides canonical worked examples and tables for reproducible tests,
//! doctests, and benchmarks without external file I/O.

use crate::experience::{ExperienceClaim, NcciErpSpecification, RetroPlanSpec};
use serde::{Deserialize, Serialize};

/// Bailey (1963) / Brown (1988) 3x2 commercial classification grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationGridDataset {
    /// Number of row classes.
    pub n_rows: usize,
    /// Number of column classes.
    pub n_cols: usize,
    /// Row-major earned exposures (`P_ij`).
    pub exposures: Vec<f64>,
    /// Row-major observed average loss costs (`A_ij`).
    pub loss_costs: Vec<f64>,
}

/// Returns a 3x2 commercial risk grid with heterogeneous exposure mix for verifying
/// Bailey (1963) balance-principle minimum bias and Brown (1988) Poisson GLM equivalence.
pub fn commercial_classification_grid() -> ClassificationGridDataset {
    ClassificationGridDataset {
        n_rows: 3,
        n_cols: 2,
        exposures: vec![1235.0, 582.0, 2845.0, 1204.0, 890.0, 410.0],
        loss_costs: vec![105.4, 132.8, 92.1, 116.5, 81.0, 102.3],
    }
}

/// NCCI Workers' Compensation Experience Rating worksheet benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcciWorksheetDataset {
    /// Plan parameters (split point, expected losses, D-ratio, W, B, ERA).
    pub spec: NcciErpSpecification,
    /// Experience period claims.
    pub claims: Vec<ExperienceClaim>,
    /// Exact unrounded modification factor (`107_225 / 125_000 = 0.8578`).
    pub expected_mod_unrounded: f64,
    /// Published 3-decimal rounded modification factor (`0.858`).
    pub expected_mod_rounded: f64,
}

/// Returns the standard NCCI Experience Rating Plan worksheet benchmark with ERA enabled.
pub fn ncci_erp_worksheet() -> NcciWorksheetDataset {
    NcciWorksheetDataset {
        spec: NcciErpSpecification {
            split_point: 18_500.0,
            expected_losses: 100_000.0,
            d_ratio: 0.40,
            weight_w: 0.15,
            ballast_b: 25_000.0,
            apply_era: true,
        },
        claims: vec![
            ExperienceClaim {
                incurred: 50_000.0,
                is_medical_only: false,
            },
            ExperienceClaim {
                incurred: 5_000.0,
                is_medical_only: false,
            },
            ExperienceClaim {
                incurred: 10_000.0,
                is_medical_only: true,
            },
        ],
        expected_mod_unrounded: 107_225.0 / 125_000.0,
        expected_mod_rounded: 0.858,
    }
}

/// Discrete entry-ratio distribution and balanced retrospective rating plan fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetroBenchmarkDataset {
    /// Discrete aggregate entry ratios `r_k` with mean `1.0`.
    pub entry_ratios: Vec<f64>,
    /// Probability masses `p_k` summing to `1.0`.
    pub probabilities: Vec<f64>,
    /// Standard commercial retrospective plan specification.
    pub spec: RetroPlanSpec,
}

/// Returns a symmetric 5-point entry-ratio distribution and balanced commercial
/// retrospective rating plan specification (CAS Exam 8).
pub fn retro_plan_benchmark() -> RetroBenchmarkDataset {
    RetroBenchmarkDataset {
        entry_ratios: vec![0.2, 0.6, 1.0, 1.4, 1.8],
        probabilities: vec![0.10, 0.20, 0.40, 0.20, 0.10],
        spec: RetroPlanSpec {
            standard_premium: 750_000.0,
            expected_loss_ratio: 0.65,
            lcf: 1.12,
            tax_multiplier: 1.035,
            max_retro_ratio: 1.35,
            min_retro_ratio: 0.55,
            expense_ratio: 0.18,
        },
    }
}

/// Non-parametric Bühlmann–Straub credibility worked example (3 risk groups, 3 periods each).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuhlmannStraubDataset {
    /// Per-group loss rates `X_{ij}`.
    pub group_losses: Vec<Vec<f64>>,
    /// Per-group exposures `P_{ij}`.
    pub group_weights: Vec<Vec<f64>>,
}

/// Returns a 3-group heterogeneous-exposure Bühlmann–Straub credibility dataset.
pub fn buhlmann_straub_example() -> BuhlmannStraubDataset {
    BuhlmannStraubDataset {
        group_losses: vec![
            vec![100.0, 120.0, 110.0],
            vec![150.0, 140.0, 160.0],
            vec![80.0, 90.0, 85.0],
        ],
        group_weights: vec![
            vec![10.0, 12.0, 15.0],
            vec![20.0, 25.0, 22.0],
            vec![5.0, 6.0, 7.0],
        ],
    }
}
