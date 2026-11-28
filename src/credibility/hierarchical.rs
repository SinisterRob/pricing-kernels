//! Jewell (1975) two-level hierarchical credibility model (Sector → Risk → Period)
//! following Bühlmann & Gisler (*A Course in Credibility Theory and its Applications*, Ch. 5).
//!
//! Reference: W. S. Jewell (1975), *"The Use of Collateral Data in Credibility Theory:
//! A Hierarchical Model"*, *Giornale dell'Istituto Italiano degli Attuari*, Vol. 38, pp. 1–16.

use crate::error::PricingError;
use serde::{Deserialize, Serialize};

/// Single risk group inside a hierarchical sector (e.g., class code inside an industry group,
/// or territory inside a state).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalRisk {
    /// Optional human-readable risk label.
    pub label: Option<String>,
    /// Per-period observed loss rates, pure premiums, or loss ratios `X_{pjt}`.
    pub losses: Vec<f64>,
    /// Per-period strictly positive exposures or weights `w_{pjt} > 0`.
    pub weights: Vec<f64>,
}

/// Top-level sector containing one or more [`HierarchicalRisk`] members.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalSector {
    /// Optional human-readable sector label.
    pub label: Option<String>,
    /// Member risks belonging to this sector.
    pub risks: Vec<HierarchicalRisk>,
}

/// Credibility estimate for a single risk within a sector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalRiskResult {
    /// Total exposure `w_{pj•} = ∑_t w_{pjt}`.
    pub total_weight: f64,
    /// Exposure-weighted observed risk mean `X̄_{pj}`.
    pub observed_mean: f64,
    /// Risk-level credibility weight `z_{pj} = w_{pj•} / (w_{pj•} + s^2 / v)`.
    pub credibility_z: f64,
    /// Credibility-blended risk estimate `z_{pj} X̄_{pj} + (1 - z_{pj}) B̂_p`.
    pub credibility_estimate: f64,
}

/// Credibility estimate for a sector and its member risks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalSectorResult {
    /// Total sector exposure `w_{p••} = ∑_j w_{pj•}`.
    pub total_weight: f64,
    /// Sum of member risk credibilities `z_{p•} = ∑_j z_{pj}`.
    pub total_risk_credibility: f64,
    /// Credibility-weighted sector mean `X̄_p^{(z)}`.
    pub sector_mean: f64,
    /// Sector-level credibility weight `α_p = z_{p•} / (z_{p•} + v / a)`.
    pub sector_credibility: f64,
    /// Credibility-blended sector estimate `B̂_p = α_p X̄_p^{(z)} + (1 - α_p) μ̂`.
    pub sector_estimate: f64,
    /// Per-risk results in input order.
    pub risks: Vec<HierarchicalRiskResult>,
}

/// Complete output of Jewell (1975) two-level hierarchical credibility estimation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalCredibilityResult {
    /// Within-risk Expected Value of Process Variance `s^2` (level-3 EPV).
    pub epv: f64,
    /// Within-sector Variance of Hypothetical Risk Means `v` (level-2 VHM).
    pub within_sector_vhm: f64,
    /// Between-sector Variance of Hypothetical Sector Means `a` (level-1 VHM).
    pub between_sector_vhm: f64,
    /// Collective grand mean `μ̂`.
    pub grand_mean: f64,
    /// Per-sector credibility results (each containing its member risk results).
    pub sectors: Vec<HierarchicalSectorResult>,
}

/// Fit Jewell (1975) two-level hierarchical credibility across `sectors`
/// (`Sector → Risk → Period`), using the unbiased Bühlmann–Gisler (2005)
/// variance-component estimators for `(s^2, v, a)`.
///
/// Requirements:
/// - At least 2 sectors (`sectors.len() >= 2`).
/// - Every sector must contain at least 1 risk, and at least one sector must have `>= 2` risks
///   so `within_sector_vhm` (`v`) is identifiable.
/// - Total period observations across all risks must strictly exceed total risk count
///   so `epv` (`s^2`) is identifiable.
pub fn jewell_hierarchical_credibility(
    sectors: &[HierarchicalSector],
) -> Result<HierarchicalCredibilityResult, PricingError> {
    let n_sectors = sectors.len();
    if n_sectors < 2 {
        return Err(PricingError::InvalidParameter(
            "Jewell hierarchical credibility requires at least 2 sectors",
        ));
    }

    let mut total_risks = 0usize;
    let mut total_periods = 0usize;
    let mut has_multi_risk_sector = false;

    // Precompute per-risk total weights and weighted means.
    let mut risk_w: Vec<Vec<f64>> = Vec::with_capacity(n_sectors);
    let mut risk_mean: Vec<Vec<f64>> = Vec::with_capacity(n_sectors);

    for sector in sectors {
        if sector.risks.is_empty() {
            return Err(PricingError::EmptyInput(
                "each hierarchical sector must contain at least one risk",
            ));
        }
        if sector.risks.len() >= 2 {
            has_multi_risk_sector = true;
        }
        let mut sw = Vec::with_capacity(sector.risks.len());
        let mut sm = Vec::with_capacity(sector.risks.len());
        for risk in &sector.risks {
            if risk.losses.is_empty() || risk.losses.len() != risk.weights.len() {
                return Err(PricingError::DimensionMismatch(
                    "each risk must have non-empty matching losses and weights",
                ));
            }
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            for (&x, &w) in risk.losses.iter().zip(risk.weights.iter()) {
                if !x.is_finite() || !w.is_finite() {
                    return Err(PricingError::InvalidParameter(
                        "risk losses and weights must be finite",
                    ));
                }
                if !(w > 0.0) {
                    return Err(PricingError::NonPositiveValue(
                        "risk weights must be strictly positive",
                    ));
                }
                sum_w += w;
                sum_wx += w * x;
            }
            sw.push(sum_w);
            sm.push(sum_wx / sum_w);
            total_risks += 1;
            total_periods += risk.losses.len();
        }
        risk_w.push(sw);
        risk_mean.push(sm);
    }

    if total_periods <= total_risks {
        return Err(PricingError::InvalidParameter(
            "total period observations must exceed total number of risks to estimate EPV",
        ));
    }
    if !has_multi_risk_sector {
        return Err(PricingError::InvalidParameter(
            "at least one sector must contain >= 2 risks to estimate within-sector VHM",
        ));
    }

    // 1. Level-3 EPV (s^2): pooled within-risk weighted variance.
    let mut s2_num = 0.0;
    for (p, sector) in sectors.iter().enumerate() {
        for (j, risk) in sector.risks.iter().enumerate() {
            let x_bar = risk_mean[p][j];
            for (&x, &w) in risk.losses.iter().zip(risk.weights.iter()) {
                let d = x - x_bar;
                s2_num += w * d * d;
            }
        }
    }
    let epv = (s2_num / ((total_periods - total_risks) as f64)).max(0.0);

    // 2. Level-2 VHM (v): pooled within-sector between-risk variance (Bühlmann–Gisler Eq. 5.14).
    let mut v_num = 0.0;
    let mut v_den = 0.0;
    let mut sector_total_w = Vec::with_capacity(n_sectors);
    let mut sector_exp_mean = Vec::with_capacity(n_sectors);

    for p in 0..n_sectors {
        let wp_tot: f64 = risk_w[p].iter().sum();
        let xp_w: f64 = risk_w[p]
            .iter()
            .zip(risk_mean[p].iter())
            .map(|(&w, &m)| w * m)
            .sum::<f64>()
            / wp_tot;
        sector_total_w.push(wp_tot);
        sector_exp_mean.push(xp_w);

        let j_p = risk_w[p].len();
        if j_p >= 2 {
            let mut sum_rel_sq = 0.0;
            let mut ss_between = 0.0;
            for j in 0..j_p {
                let rel = risk_w[p][j] / wp_tot;
                sum_rel_sq += rel * rel;
                let d = risk_mean[p][j] - xp_w;
                ss_between += risk_w[p][j] * d * d;
            }
            let c_p = wp_tot * (1.0 - sum_rel_sq);
            v_num += ss_between - ((j_p - 1) as f64) * epv;
            v_den += c_p;
        }
    }
    let v = if v_den > 0.0 {
        (v_num / v_den).max(0.0)
    } else {
        0.0
    };

    // 3. Risk-level credibilities z_{pj} and credibility-weighted sector means X̄_p^{(z)}.
    let mut risk_z: Vec<Vec<f64>> = Vec::with_capacity(n_sectors);
    let mut sector_z_sum = Vec::with_capacity(n_sectors);
    let mut sector_z_mean = Vec::with_capacity(n_sectors);

    for p in 0..n_sectors {
        let j_p = risk_w[p].len();
        let mut zs = Vec::with_capacity(j_p);
        let mut z_tot = 0.0;
        let mut zx_sum = 0.0;
        for j in 0..j_p {
            let z = if v > 0.0 {
                if epv == 0.0 {
                    1.0
                } else {
                    let k2 = epv / v;
                    risk_w[p][j] / (risk_w[p][j] + k2)
                }
            } else {
                0.0
            };
            zs.push(z);
            z_tot += z;
            zx_sum += z * risk_mean[p][j];
        }
        risk_z.push(zs);
        sector_z_sum.push(z_tot);
        sector_z_mean.push(if z_tot > 0.0 {
            zx_sum / z_tot
        } else {
            sector_exp_mean[p]
        });
    }

    // 4. Level-1 VHM (a): between-sector variance (Bühlmann–Gisler Eq. 5.16).
    let z_grand: f64 = sector_z_sum.iter().sum();
    let (a, mu_prior) = if v > 0.0 && z_grand > 0.0 {
        let mu_z: f64 = sector_z_sum
            .iter()
            .zip(sector_z_mean.iter())
            .map(|(&zp, &mp)| zp * mp)
            .sum::<f64>()
            / z_grand;
        let mut sum_rel_sq = 0.0;
        let mut ss_sect = 0.0;
        for p in 0..n_sectors {
            let rel = sector_z_sum[p] / z_grand;
            sum_rel_sq += rel * rel;
            let d = sector_z_mean[p] - mu_z;
            ss_sect += sector_z_sum[p] * d * d;
        }
        let c_sect = z_grand * (1.0 - sum_rel_sq);
        let a_est = if c_sect > 0.0 {
            ((ss_sect - ((n_sectors - 1) as f64) * v) / c_sect).max(0.0)
        } else {
            0.0
        };
        (a_est, mu_z)
    } else {
        // Fallback when v == 0 (all within-sector variance is process noise):
        // estimate between-sector variance directly from exposure-weighted sector means.
        let w_grand: f64 = sector_total_w.iter().sum();
        let mu_w: f64 = sector_total_w
            .iter()
            .zip(sector_exp_mean.iter())
            .map(|(&wp, &mp)| wp * mp)
            .sum::<f64>()
            / w_grand;
        let mut sum_rel_sq = 0.0;
        let mut ss_sect = 0.0;
        for p in 0..n_sectors {
            let rel = sector_total_w[p] / w_grand;
            sum_rel_sq += rel * rel;
            let d = sector_exp_mean[p] - mu_w;
            ss_sect += sector_total_w[p] * d * d;
        }
        let c_sect = w_grand * (1.0 - sum_rel_sq);
        let a_est = if c_sect > 0.0 {
            ((ss_sect - ((n_sectors - 1) as f64) * epv) / c_sect).max(0.0)
        } else {
            0.0
        };
        (a_est, mu_w)
    };

    // 5. Sector-level credibilities alpha_p and collective grand mean mu_hat.
    let mut alpha_p = Vec::with_capacity(n_sectors);
    let mut sum_alpha = 0.0;
    let mut sum_alpha_x = 0.0;

    for p in 0..n_sectors {
        let alpha = if a > 0.0 {
            if v > 0.0 {
                let k1 = v / a;
                sector_z_sum[p] / (sector_z_sum[p] + k1)
            } else if epv > 0.0 {
                let k1 = epv / a;
                sector_total_w[p] / (sector_total_w[p] + k1)
            } else {
                1.0
            }
        } else {
            0.0
        };
        alpha_p.push(alpha);
        sum_alpha += alpha;
        sum_alpha_x += alpha * sector_z_mean[p];
    }

    let grand_mean = if sum_alpha > 0.0 {
        sum_alpha_x / sum_alpha
    } else {
        mu_prior
    };

    // 6. Top-down credibility shrinkage: grand_mean -> sector_estimate -> risk_estimate.
    let mut sector_results = Vec::with_capacity(n_sectors);
    for p in 0..n_sectors {
        let sector_est = alpha_p[p] * sector_z_mean[p] + (1.0 - alpha_p[p]) * grand_mean;
        let j_p = risk_w[p].len();
        let mut risk_results = Vec::with_capacity(j_p);
        for j in 0..j_p {
            let z = risk_z[p][j];
            let est = z * risk_mean[p][j] + (1.0 - z) * sector_est;
            risk_results.push(HierarchicalRiskResult {
                total_weight: risk_w[p][j],
                observed_mean: risk_mean[p][j],
                credibility_z: z,
                credibility_estimate: est,
            });
        }
        sector_results.push(HierarchicalSectorResult {
            total_weight: sector_total_w[p],
            total_risk_credibility: sector_z_sum[p],
            sector_mean: sector_z_mean[p],
            sector_credibility: alpha_p[p],
            sector_estimate: sector_est,
            risks: risk_results,
        });
    }

    Ok(HierarchicalCredibilityResult {
        epv,
        within_sector_vhm: v,
        between_sector_vhm: a,
        grand_mean,
        sectors: sector_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jewell_hierarchical_shrinks_risks_toward_sector_and_sectors_toward_grand_mean() {
        let sectors = vec![
            HierarchicalSector {
                label: Some("Manufacturing".into()),
                risks: vec![
                    HierarchicalRisk {
                        label: Some("Light".into()),
                        losses: vec![80.0, 84.0, 76.0],
                        weights: vec![10.0, 12.0, 10.0],
                    },
                    HierarchicalRisk {
                        label: Some("Heavy".into()),
                        losses: vec![110.0, 118.0, 112.0],
                        weights: vec![15.0, 15.0, 14.0],
                    },
                ],
            },
            HierarchicalSector {
                label: Some("Construction".into()),
                risks: vec![
                    HierarchicalRisk {
                        label: Some("Residential".into()),
                        losses: vec![160.0, 170.0, 165.0],
                        weights: vec![20.0, 22.0, 18.0],
                    },
                    HierarchicalRisk {
                        label: Some("Commercial".into()),
                        losses: vec![210.0, 200.0, 215.0],
                        weights: vec![5.0, 6.0, 5.0],
                    },
                ],
            },
        ];

        let res = jewell_hierarchical_credibility(&sectors).expect("jewell");
        assert!(res.epv > 0.0);
        assert!(res.within_sector_vhm > 0.0);
        assert!(res.between_sector_vhm > 0.0);

        for s in &res.sectors {
            assert!(s.sector_credibility > 0.0 && s.sector_credibility < 1.0);
            // Sector estimate lies strictly between grand mean and observed sector mean.
            let lo = res.grand_mean.min(s.sector_mean);
            let hi = res.grand_mean.max(s.sector_mean);
            assert!(s.sector_estimate >= lo - 1e-12 && s.sector_estimate <= hi + 1e-12);

            for r in &s.risks {
                assert!(r.credibility_z > 0.0 && r.credibility_z < 1.0);
                // Risk estimate lies strictly between sector estimate and observed risk mean.
                let r_lo = s.sector_estimate.min(r.observed_mean);
                let r_hi = s.sector_estimate.max(r.observed_mean);
                assert!(
                    r.credibility_estimate >= r_lo - 1e-12
                        && r.credibility_estimate <= r_hi + 1e-12
                );
            }
        }
        // Larger exposure risk gets higher credibility weight than smaller risk.
        assert!(res.sectors[1].risks[0].credibility_z > res.sectors[1].risks[1].credibility_z);
    }
}
