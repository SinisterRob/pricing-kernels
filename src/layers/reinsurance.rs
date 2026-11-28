//! Commercial treaty reinsurance pricing kernels: experience-rated burning cost,
//! reinstatement pricing (Sundt 1991), sliding-scale commission, loss corridors,
//! and profit commission (Clark 2014 *Basics of Reinsurance Pricing*).

use crate::error::PricingError;
use crate::experience::TableMEngine;
use crate::layers::{CommercialLayer, LayerTower};
use serde::{Deserialize, Serialize};

/// Validated historical treaty experience container pairing subject-premium years and large claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreatyExperience {
    years: Vec<HistoricalTreatyYear>,
    claims: Vec<HistoricalTreatyClaim>,
}

impl TreatyExperience {
    /// Construct a validated [`TreatyExperience`] dataset, eagerly verifying all treaty years
    /// and individual claim records.
    pub fn try_new(
        years: Vec<HistoricalTreatyYear>,
        claims: Vec<HistoricalTreatyClaim>,
    ) -> Result<Self, PricingError> {
        if years.is_empty() {
            return Err(PricingError::EmptyInput(
                "at least one historical treaty year is required",
            ));
        }
        for y in &years {
            if !y.subject_premium.is_finite()
                || !y.on_level_factor.is_finite()
                || !y.exposure_trend_factor.is_finite()
                || !(y.subject_premium > 0.0)
                || !(y.on_level_factor > 0.0)
                || !(y.exposure_trend_factor > 0.0)
            {
                return Err(PricingError::NonPositiveValue(
                    "subject_premium, on_level_factor, and exposure_trend_factor must be finite and positive",
                ));
            }
        }
        for c in &claims {
            if !c.incurred.is_finite()
                || !c.severity_trend_factor.is_finite()
                || !c.development_factor.is_finite()
                || !(c.incurred >= 0.0)
                || !(c.severity_trend_factor > 0.0)
                || !(c.development_factor > 0.0)
            {
                return Err(PricingError::InvalidParameter(
                    "claim incurred must be >= 0 and trend/development factors must be finite and positive",
                ));
            }
            if !years.iter().any(|y| y.year == c.year) {
                return Err(PricingError::OutOfBounds(
                    "claim year does not match any historical treaty year",
                ));
            }
        }
        Ok(Self { years, claims })
    }

    /// Historical treaty years.
    #[inline]
    pub fn years(&self) -> &[HistoricalTreatyYear] {
        &self.years
    }

    /// Historical individual claims.
    #[inline]
    pub fn claims(&self) -> &[HistoricalTreatyClaim] {
        &self.claims
    }

    /// Compute experience-rated burning cost for a single [`CommercialLayer`].
    pub fn burning_cost(&self, layer: &CommercialLayer) -> Result<BurningCostResult, PricingError> {
        burning_cost_xl(&self.years, &self.claims, layer)
    }

    /// Compute experience-rated burning cost for every layer in a [`LayerTower`].
    pub fn burning_cost_tower(
        &self,
        tower: &LayerTower,
    ) -> Result<Vec<BurningCostResult>, PricingError> {
        let mut results = Vec::with_capacity(tower.len());
        for layer in tower.layers() {
            results.push(burning_cost_xl(&self.years, &self.claims, layer)?);
        }
        Ok(results)
    }
}

/// Credibility weighting rule for blending XL treaty exposure and experience rates (Clark 2014).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BlendCredibilityRule {
    /// Explicit credibility weight `Z in [0, 1]` assigned to the experience rate.
    Fixed(f64),
    /// Square-root classical credibility `Z = min(1, sqrt(n / full_credibility_claims))`.
    SquareRoot {
        /// Number of historical or expected claims penetrating the layer (`>= 0`).
        claim_count: f64,
        /// Full-credibility claim count standard `n_0 > 0`.
        full_credibility_claims: f64,
    },
    /// Bühlmann credibility `Z = volume / (volume + k)` where `k > 0`.
    Buhlmann {
        /// Experience volume (e.g. expected layer claims or adjusted subject premium, `>= 0`).
        volume: f64,
        /// Bühlmann credibility constant `K > 0`.
        credibility_k: f64,
    },
}

/// Result of credibility-blending an exposure-rated and experience-rated XL treaty loss cost.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExposureExperienceBlendResult {
    /// Exposure-rated layer loss cost rate (complement of credibility).
    pub exposure_rate: f64,
    /// Experience-rated (burning-cost) layer loss cost rate.
    pub experience_rate: f64,
    /// Credibility weight `Z in [0, 1]` assigned to the experience rate.
    pub credibility_z: f64,
    /// Blended treaty loss cost rate `Z * experience_rate + (1 - Z) * exposure_rate`.
    pub blended_rate: f64,
    /// Indicated treaty technical premium (`blended_rate * subject_premium / (1 - load_ratio)`),
    /// if `subject_premium` was provided.
    pub technical_premium: Option<f64>,
}

/// Credibility-blend an exposure-rated XL layer loss cost (e.g. from MBBEFD or ILF curves)
/// with an experience-rated burning cost (Clark 2014 *Basics of Reinsurance Pricing*).
pub fn blend_exposure_and_experience(
    exposure_rate: f64,
    experience_rate: f64,
    rule: BlendCredibilityRule,
    subject_premium: Option<f64>,
    expense_and_margin_load: f64,
) -> Result<ExposureExperienceBlendResult, PricingError> {
    if !exposure_rate.is_finite()
        || !experience_rate.is_finite()
        || !expense_and_margin_load.is_finite()
        || !(exposure_rate >= 0.0)
        || !(experience_rate >= 0.0)
        || !(expense_and_margin_load >= 0.0)
        || !(expense_and_margin_load < 1.0)
    {
        return Err(PricingError::InvalidParameter(
            "rates must be finite >= 0 and expense_and_margin_load in [0, 1)",
        ));
    }

    let z = match rule {
        BlendCredibilityRule::Fixed(z) => {
            if !z.is_finite() || !(z >= 0.0) || !(z <= 1.0) {
                return Err(PricingError::InvalidParameter(
                    "fixed credibility_z must be finite and in [0, 1]",
                ));
            }
            z
        }
        BlendCredibilityRule::SquareRoot {
            claim_count,
            full_credibility_claims,
        } => {
            if !claim_count.is_finite()
                || !full_credibility_claims.is_finite()
                || !(claim_count >= 0.0)
                || !(full_credibility_claims > 0.0)
            {
                return Err(PricingError::InvalidParameter(
                    "claim_count must be >= 0 and full_credibility_claims > 0",
                ));
            }
            (claim_count / full_credibility_claims).sqrt().min(1.0)
        }
        BlendCredibilityRule::Buhlmann {
            volume,
            credibility_k,
        } => {
            if !volume.is_finite()
                || !credibility_k.is_finite()
                || !(volume >= 0.0)
                || !(credibility_k > 0.0)
            {
                return Err(PricingError::InvalidParameter(
                    "volume must be >= 0 and credibility_k > 0",
                ));
            }
            volume / (volume + credibility_k)
        }
    };

    let blended_rate = z * experience_rate + (1.0 - z) * exposure_rate;
    let technical_premium = match subject_premium {
        None => None,
        Some(prem) => {
            if !prem.is_finite() || !(prem > 0.0) {
                return Err(PricingError::NonPositiveValue(
                    "subject_premium must be finite and positive",
                ));
            }
            Some(blended_rate * prem / (1.0 - expense_and_margin_load))
        }
    };

    Ok(ExposureExperienceBlendResult {
        exposure_rate,
        experience_rate,
        credibility_z: z,
        blended_rate,
        technical_premium,
    })
}

/// Historical treaty-year subject premium and adjustment factors for burning-cost rating.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoricalTreatyYear {
    /// Underwriting or accident year identifier.
    pub year: i32,
    /// Historical subject earned premium.
    pub subject_premium: f64,
    /// On-level factor bringing historical rate level to prospective treaty period.
    pub on_level_factor: f64,
    /// Exposure trend factor bringing historical exposure base to prospective period.
    pub exposure_trend_factor: f64,
}

/// Historical individual large claim record for excess-of-loss experience rating.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoricalTreatyClaim {
    /// Underwriting or accident year matching a `HistoricalTreatyYear`.
    pub year: i32,
    /// Historical incurred (or paid) ground-up claim amount.
    pub incurred: f64,
    /// Severity trend factor bringing claim cost to prospective treaty period.
    pub severity_trend_factor: f64,
    /// Loss development factor to ultimate (use `1.0` if already at ultimate).
    pub development_factor: f64,
}

/// Per-year burning cost breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BurningCostYearResult {
    /// Underwriting or accident year.
    pub year: i32,
    /// Adjusted subject premium (`subject_premium * on_level * exposure_trend`).
    pub adjusted_subject_premium: f64,
    /// Total trended/developed ceded layer losses in this year.
    pub layer_ceded_loss: f64,
    /// Annual burning cost rate (`layer_ceded_loss / adjusted_subject_premium`).
    pub burning_cost_rate: f64,
    /// Count of claims penetrating the layer in this year.
    pub layer_claim_count: usize,
}

/// Aggregate experience-rated burning cost result for an excess-of-loss treaty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurningCostResult {
    /// Per-year experience breakdown in input order.
    pub by_year: Vec<BurningCostYearResult>,
    /// Total adjusted subject premium across all experience years.
    pub total_adjusted_subject_premium: f64,
    /// Total trended/developed ceded layer losses across all experience years.
    pub total_layer_ceded_loss: f64,
    /// Overall experience-rated burning cost rate (`total_layer_ceded_loss / total_adjusted_subject_premium`).
    pub overall_burning_cost_rate: f64,
    /// Total count of claims penetrating the layer.
    pub total_layer_claims: usize,
}

/// Compute experience-rated burning cost for an excess-of-loss treaty (Clark 2014).
///
/// Each claim is trended and developed to prospective ultimate ground-up dollars
/// (`incurred * severity_trend_factor * development_factor`) before applying `layer.payout`.
/// Each historical year's subject premium is adjusted by `on_level_factor * exposure_trend_factor`.
pub fn burning_cost_xl(
    years: &[HistoricalTreatyYear],
    claims: &[HistoricalTreatyClaim],
    layer: &CommercialLayer,
) -> Result<BurningCostResult, PricingError> {
    if years.is_empty() {
        return Err(PricingError::EmptyInput(
            "at least one historical treaty year is required",
        ));
    }
    layer.validate()?;

    let mut by_year: Vec<BurningCostYearResult> = Vec::with_capacity(years.len());
    let mut total_adj_prem = 0.0;

    for y in years {
        if by_year.iter().any(|entry| entry.year == y.year) {
            return Err(PricingError::InvalidParameter(
                "duplicate year encountered in historical treaty years",
            ));
        }
        if !y.subject_premium.is_finite()
            || !y.on_level_factor.is_finite()
            || !y.exposure_trend_factor.is_finite()
            || !(y.subject_premium > 0.0)
            || !(y.on_level_factor > 0.0)
            || !(y.exposure_trend_factor > 0.0)
        {
            return Err(PricingError::NonPositiveValue(
                "subject_premium, on_level_factor, and exposure_trend_factor must be finite and positive",
            ));
        }
        let adj = y.subject_premium * y.on_level_factor * y.exposure_trend_factor;
        total_adj_prem += adj;
        by_year.push(BurningCostYearResult {
            year: y.year,
            adjusted_subject_premium: adj,
            layer_ceded_loss: 0.0,
            burning_cost_rate: 0.0,
            layer_claim_count: 0,
        });
    }

    let mut total_ceded = 0.0;
    let mut total_layer_claims = 0usize;

    for c in claims {
        if !c.incurred.is_finite()
            || !c.severity_trend_factor.is_finite()
            || !c.development_factor.is_finite()
            || !(c.incurred >= 0.0)
            || !(c.severity_trend_factor > 0.0)
            || !(c.development_factor > 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "claim incurred must be >= 0 and trend/development factors must be finite and positive",
            ));
        }
        let Some(yr_entry) = by_year.iter_mut().find(|entry| entry.year == c.year) else {
            return Err(PricingError::InvalidParameter(
                "claim year does not match any provided HistoricalTreatyYear",
            ));
        };
        let adj_loss = c.incurred * c.severity_trend_factor * c.development_factor;
        let payout = layer.payout(adj_loss);
        if payout > 0.0 {
            yr_entry.layer_ceded_loss += payout;
            yr_entry.layer_claim_count += 1;
            total_ceded += payout;
            total_layer_claims += 1;
        }
    }

    for entry in &mut by_year {
        entry.burning_cost_rate = entry.layer_ceded_loss / entry.adjusted_subject_premium;
    }

    Ok(BurningCostResult {
        by_year,
        total_adjusted_subject_premium: total_adj_prem,
        total_layer_ceded_loss: total_ceded,
        overall_burning_cost_rate: total_ceded / total_adj_prem,
        total_layer_claims,
    })
}

/// Reinstatement premium basis for excess-of-loss treaties (Sundt 1991 / Clark 2014).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReinstatementBasis {
    /// Pro-rata as to amount only (standard property catastrophe / per-occurrence XL).
    ProRataAmount,
    /// Pro-rata as to amount and time: each reinstated amount is charged for the unexpired
    /// fraction `1 - t` of the term at its loss date `t`.
    ///
    /// The expected premium depends on *when* each aggregate slice is eroded, so it needs a
    /// timing model: use [`price_reinstatements_compound_poisson`] (exact under Poisson arrivals
    /// uniform over the term) or [`price_reinstatements_time_weighted`] (caller-supplied
    /// time-weighted slice losses). [`price_reinstatements`] rejects this basis.
    ProRataAmountAndTime,
}

/// Solved reinstatement pricing breakdown for an excess-of-loss treaty.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReinstatementPricingResult {
    /// Total expected ceded loss across initial limit and all reinstatements.
    pub total_expected_ceded_loss: f64,
    /// Initial (upfront) treaty premium `P_0`.
    pub initial_premium: f64,
    /// Expected reinstatement premium collected `E[P_reinst]`.
    pub expected_reinstatement_premium: f64,
    /// Total expected premium (`initial_premium + expected_reinstatement_premium`).
    pub total_expected_premium: f64,
    /// Initial Rate-on-Line (`initial_premium / layer_limit`).
    pub initial_rate_on_line: f64,
    /// Effective Rate-on-Line including expected reinstatements (`total_expected_premium / layer_limit`).
    pub effective_rate_on_line: f64,
}

/// Absolute tolerance (in currency units) allowed when checking slice expectations
/// against `layer_limit` and against each other.
const SLICE_BOUND_TOL: f64 = 1e-9;

fn validate_reinstatement_inputs(
    layer_limit: f64,
    expected_layer_losses: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
) -> Result<f64, PricingError> {
    if expected_layer_losses.is_empty() {
        return Err(PricingError::EmptyInput(
            "expected_layer_losses must contain at least the initial layer E[S_0]",
        ));
    }
    if reinstatement_rates.len() + 1 != expected_layer_losses.len() {
        return Err(PricingError::DimensionMismatch(
            "reinstatement_rates length must equal expected_layer_losses.len() - 1",
        ));
    }
    if !layer_limit.is_finite() || !(layer_limit > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "layer_limit must be finite and positive",
        ));
    }
    if !expense_ratio.is_finite() || !(expense_ratio >= 0.0) || !(expense_ratio < 1.0) {
        return Err(PricingError::InvalidParameter(
            "expense_ratio must be finite and in [0, 1)",
        ));
    }
    let mut total_expected_loss = 0.0;
    for &s in expected_layer_losses {
        if !s.is_finite() || !(s >= 0.0) || s > layer_limit + SLICE_BOUND_TOL {
            return Err(PricingError::InvalidParameter(
                "each expected layer loss must be finite and in [0, layer_limit]",
            ));
        }
        total_expected_loss += s;
    }
    for &c_k in reinstatement_rates {
        if !c_k.is_finite() || !(c_k >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "reinstatement rates must be finite and non-negative",
            ));
        }
    }
    Ok(total_expected_loss)
}

/// Sundt (1991) balance `P_0 (1 - e) (1 + sum_k c_k R_{k-1} / L) = sum_k E[S_k]`, where
/// `R_{k-1}` is the premium-relevant measure of slice `k - 1` (`E[S_{k-1}]` for pro-rata
/// as to amount, `E[∫(1 - t) dS_{k-1}(t)]` for pro-rata as to amount and time).
fn sundt_balance(
    layer_limit: f64,
    total_expected_loss: f64,
    premium_slice_measure: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
) -> ReinstatementPricingResult {
    // Reinstatement k+1 is consumed by losses in aggregate layer k.
    let reinst_multiplier: f64 = reinstatement_rates
        .iter()
        .zip(premium_slice_measure)
        .map(|(&c_k, &r_k)| c_k * r_k / layer_limit)
        .sum();
    let initial_premium = total_expected_loss / ((1.0 - expense_ratio) * (1.0 + reinst_multiplier));
    let expected_reinstatement_premium = initial_premium * reinst_multiplier;
    let total_expected_premium = initial_premium + expected_reinstatement_premium;
    ReinstatementPricingResult {
        total_expected_ceded_loss: total_expected_loss,
        initial_premium,
        expected_reinstatement_premium,
        total_expected_premium,
        initial_rate_on_line: initial_premium / layer_limit,
        effective_rate_on_line: total_expected_premium / layer_limit,
    }
}

/// Price an excess-of-loss treaty with `K` reinstatements pro-rata as to amount using
/// Sundt (1991) exact balance.
///
/// - `layer_limit`: Occurrence layer limit `L > 0`.
/// - `expected_layer_losses`: Expected ceded loss in each aggregate layer slice `[E[S_0], E[S_1], ..., E[S_K]]`
///   where `S_0` is the initial layer (`0..L`) and `S_k` is the `k`-th reinstated layer (`k*L..(k+1)*L`).
///   Length is `K + 1` (at least 1).
/// - `reinstatement_rates`: Reinstatement premium percentage `c_k >= 0` for each reinstatement `k = 1..=K`
///   (e.g. `1.0` for 100% pro-rata, `0.0` for free reinstatement). Must have length `expected_layer_losses.len() - 1`.
/// - `expense_ratio`: Reinsurer expense + brokerage ratio `e in [0, 1)`.
/// - `basis`: must be [`ReinstatementBasis::ProRataAmount`]; slice expectations alone carry no
///   timing information, so [`ReinstatementBasis::ProRataAmountAndTime`] returns
///   [`PricingError::InvalidParameter`] (use [`price_reinstatements_compound_poisson`] or
///   [`price_reinstatements_time_weighted`]).
///
/// Under Sundt (1991) actuarial balance:
/// `P_0 = (sum_{k=0}^K E[S_k]) / ((1 - e) * (1 + sum_{k=1}^K c_k * E[S_{k-1}] / L))`.
pub fn price_reinstatements(
    layer_limit: f64,
    expected_layer_losses: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
    basis: ReinstatementBasis,
) -> Result<ReinstatementPricingResult, PricingError> {
    let total_expected_loss = validate_reinstatement_inputs(
        layer_limit,
        expected_layer_losses,
        reinstatement_rates,
        expense_ratio,
    )?;
    if basis == ReinstatementBasis::ProRataAmountAndTime {
        return Err(PricingError::InvalidParameter(
            "pro-rata as to time needs a loss-timing model; use price_reinstatements_compound_poisson or price_reinstatements_time_weighted",
        ));
    }
    Ok(sundt_balance(
        layer_limit,
        total_expected_loss,
        expected_layer_losses,
        reinstatement_rates,
        expense_ratio,
    ))
}

/// Price an excess-of-loss treaty with `K` reinstatements pro-rata as to amount **and time**
/// from caller-supplied time-weighted slice losses (Sundt 1991 balance).
///
/// - `time_weighted_layer_losses[k] = E[∫_0^1 (1 - t) dS_k(t)] = ∫_0^1 E[S_k(t)] dt`, where
///   `S_k(t)` is the loss eroded from slice `k` by time `t` (term normalized to `[0, 1]`).
///   Must satisfy `0 <= time_weighted_layer_losses[k] <= expected_layer_losses[k]`.
///   Obtain it from [`compound_poisson_reinstatement_slices`] or a simulation.
///
/// `P_0 = (sum_k E[S_k]) / ((1 - e) * (1 + sum_{k=1}^K c_k * T_{k-1} / L))`.
pub fn price_reinstatements_time_weighted(
    layer_limit: f64,
    expected_layer_losses: &[f64],
    time_weighted_layer_losses: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
) -> Result<ReinstatementPricingResult, PricingError> {
    let total_expected_loss = validate_reinstatement_inputs(
        layer_limit,
        expected_layer_losses,
        reinstatement_rates,
        expense_ratio,
    )?;
    if time_weighted_layer_losses.len() != expected_layer_losses.len() {
        return Err(PricingError::DimensionMismatch(
            "time_weighted_layer_losses must have the same length as expected_layer_losses",
        ));
    }
    for (&t_k, &s_k) in time_weighted_layer_losses.iter().zip(expected_layer_losses) {
        if !t_k.is_finite() || !(t_k >= 0.0) || t_k > s_k + SLICE_BOUND_TOL {
            return Err(PricingError::InvalidParameter(
                "time-weighted slice loss must be finite and in [0, expected slice loss]",
            ));
        }
    }
    Ok(sundt_balance(
        layer_limit,
        total_expected_loss,
        time_weighted_layer_losses,
        reinstatement_rates,
        expense_ratio,
    ))
}

/// Expected and time-weighted aggregate slice losses from
/// [`compound_poisson_reinstatement_slices`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReinstatementSliceExpectations {
    /// `E[S_k]` for slices `k = 0..n_slices` (slice `k` covers aggregate `[kL, (k+1)L]`).
    pub expected_layer_losses: Vec<f64>,
    /// `T_k = E[∫_0^1 (1 - t) dS_k(t)] = ∫_0^1 E[S_k(t)] dt` (pro-rata-as-to-time exposure).
    pub time_weighted_layer_losses: Vec<f64>,
}

/// Tolerance on `sum(pmf) == 1` for the per-occurrence layer-loss pmf.
const PMF_SUM_TOL: f64 = 1e-9;
/// Relative tolerance for `layer_limit / step` to be an integer.
const GRID_ALIGNMENT_TOL: f64 = 1e-9;
/// Poisson count truncation: `n_max = λ + POISSON_TAIL_SD * sqrt(λ) + POISSON_TAIL_PAD`
/// (neglected tail mass far below `f64::EPSILON`).
const POISSON_TAIL_SD: f64 = 40.0;
/// Additive pad on the Poisson truncation point (see [`POISSON_TAIL_SD`]).
const POISSON_TAIL_PAD: f64 = 40.0;
/// Upper bound on `grid_points * pmf_support * count_terms` convolution work.
const MAX_CONVOLUTION_WORK: f64 = 2e10;

/// Exact expected and time-weighted aggregate slice losses for an occurrence XL layer under
/// a compound Poisson model with arrivals uniform over the (unit) term
/// (Sundt 1991; Mata 2000).
///
/// - `layer_limit`: occurrence layer limit `L > 0`; must be an integer multiple of `step`.
/// - `poisson_frequency`: expected number of occurrences per term `λ >= 0` (ground-up or
///   layer-hitting; occurrences with zero layer loss are thinned out exactly).
/// - `step`: grid span `h > 0` of the per-occurrence layer loss pmf.
/// - `occurrence_layer_pmf[j] = P(Y = j h)`, `Y = min((X - R)+, L)`; sums to 1, support `<= L`.
/// - `n_slices`: number of aggregate slices `K + 1 >= 1`.
///
/// With `S(t)` compound Poisson with intensity `λ t` and `g_k(s) = min((s - kL)+, L)`:
/// `E[S_k] = Σ_n p_n(λ) E[g_k(Y^{*n})]` and, since `∫_0^1 e^{-λt}(λt)^n/n! dt = P(N_λ > n)/λ`,
/// `T_k = Σ_n [P(N_λ > n)/λ] E[g_k(Y^{*n})]`. Both are evaluated by truncated convolution powers;
/// mass beyond `n_slices * L` is tracked exactly (it erodes every slice fully).
pub fn compound_poisson_reinstatement_slices(
    layer_limit: f64,
    poisson_frequency: f64,
    step: f64,
    occurrence_layer_pmf: &[f64],
    n_slices: usize,
) -> Result<ReinstatementSliceExpectations, PricingError> {
    if !layer_limit.is_finite() || !(layer_limit > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "layer_limit must be finite and positive",
        ));
    }
    if !poisson_frequency.is_finite() || !(poisson_frequency >= 0.0) {
        return Err(PricingError::InvalidParameter(
            "poisson_frequency must be finite and non-negative",
        ));
    }
    if !step.is_finite() || !(step > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "step must be finite and positive",
        ));
    }
    if occurrence_layer_pmf.is_empty() {
        return Err(PricingError::EmptyInput(
            "occurrence_layer_pmf must not be empty",
        ));
    }
    if n_slices == 0 {
        return Err(PricingError::InvalidParameter("n_slices must be >= 1"));
    }
    let cells_f = layer_limit / step;
    let cells_rounded = cells_f.round();
    if !(cells_rounded >= 1.0)
        || (cells_f - cells_rounded).abs() > GRID_ALIGNMENT_TOL * cells_rounded
    {
        return Err(PricingError::InvalidParameter(
            "layer_limit must be a positive integer multiple of step",
        ));
    }
    let cells = cells_rounded as usize;
    if occurrence_layer_pmf.len() > cells + 1 {
        return Err(PricingError::OutOfBounds(
            "occurrence_layer_pmf support exceeds layer_limit",
        ));
    }
    let mut total = 0.0;
    for &p in occurrence_layer_pmf {
        if !p.is_finite() || !(p >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "occurrence_layer_pmf entries must be finite and non-negative",
            ));
        }
        total += p;
    }
    if (total - 1.0).abs() > PMF_SUM_TOL {
        return Err(PricingError::InvalidParameter(
            "occurrence_layer_pmf must sum to 1",
        ));
    }

    let mut expected = vec![0.0; n_slices];
    let mut time_weighted = vec![0.0; n_slices];
    // Thin to layer-hitting occurrences: λ' = λ P(Y > 0), Y | Y > 0 has f_0 = 0.
    let hit_mass: f64 = occurrence_layer_pmf[1..].iter().sum();
    let lambda = poisson_frequency * hit_mass / total;
    if !(lambda > 0.0) || !(hit_mass > 0.0) {
        return Ok(ReinstatementSliceExpectations {
            expected_layer_losses: expected,
            time_weighted_layer_losses: time_weighted,
        });
    }
    let sev: Vec<f64> = occurrence_layer_pmf.iter().map(|&p| p / hit_mass).collect();
    let sev_support = sev.len() - 1;

    // Poisson pmf p_n (log-recursive, underflow-safe) and survival P(N > n) by backward sums.
    let n_max = (lambda + POISSON_TAIL_SD * lambda.sqrt() + POISSON_TAIL_PAD).ceil() as usize;
    let grid_max = n_slices * cells;
    let count_terms = n_max.min(grid_max) + 1;
    let work = (grid_max + 1) as f64 * sev_support as f64 * count_terms as f64;
    if work > MAX_CONVOLUTION_WORK {
        return Err(PricingError::OutOfBounds(
            "convolution grid too large; coarsen step or reduce n_slices",
        ));
    }
    let ln_lambda = lambda.ln();
    let mut log_p = -lambda;
    let mut pois = Vec::with_capacity(n_max + 1);
    for n in 0..=n_max {
        if n > 0 {
            log_p += ln_lambda - (n as f64).ln();
        }
        pois.push(log_p.exp());
    }
    // surv[n] = P(N > n); time weights w_n = surv[n] / λ.
    let mut surv = vec![0.0; n_max + 1];
    for n in (0..n_max).rev() {
        surv[n] = surv[n + 1] + pois[n + 1];
    }
    // Tail sums beyond index n for both weight sets.
    let mut pois_tail = vec![0.0; n_max + 1];
    let mut time_tail = vec![0.0; n_max + 1];
    for n in (0..n_max).rev() {
        pois_tail[n] = pois_tail[n + 1] + pois[n + 1];
        time_tail[n] = time_tail[n + 1] + surv[n + 1] / lambda;
    }

    // conv[s] = P(Y_1 + ... + Y_n = s h) for s <= grid_max; overflow = P(sum > grid_max h).
    let mut conv = vec![0.0; grid_max + 1];
    conv[0] = 1.0;
    let mut overflow = 0.0;
    let mut next = vec![0.0; grid_max + 1];
    for n in 1..=n_max {
        next.fill(0.0);
        let mut pushed = 0.0;
        for (s, &c) in conv.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            for (j, &f) in sev.iter().enumerate().skip(1) {
                let idx = s + j;
                if idx <= grid_max {
                    next[idx] += c * f;
                } else {
                    pushed += c * f;
                }
            }
        }
        std::mem::swap(&mut conv, &mut next);
        overflow += pushed;

        let w_pois = pois[n];
        let w_time = surv[n] / lambda;
        let mut in_range = 0.0;
        for k in 0..n_slices {
            let lo = k * cells;
            let mut e_gk = overflow * layer_limit;
            for (s, &c) in conv.iter().enumerate().skip(lo + 1) {
                e_gk += c * step * ((s - lo).min(cells) as f64);
            }
            expected[k] += w_pois * e_gk;
            time_weighted[k] += w_time * e_gk;
        }
        for &c in &conv {
            in_range += c;
        }
        if in_range == 0.0 {
            // Every later convolution power lies wholly beyond the top slice.
            for k in 0..n_slices {
                expected[k] += pois_tail[n] * layer_limit;
                time_weighted[k] += time_tail[n] * layer_limit;
            }
            break;
        }
    }
    for k in 0..n_slices {
        expected[k] = expected[k].clamp(0.0, layer_limit);
        time_weighted[k] = time_weighted[k].clamp(0.0, expected[k]);
    }
    Ok(ReinstatementSliceExpectations {
        expected_layer_losses: expected,
        time_weighted_layer_losses: time_weighted,
    })
}

/// Price an excess-of-loss treaty with `K = reinstatement_rates.len()` reinstatements under
/// a compound Poisson occurrence model with arrivals uniform over the term (Sundt 1991
/// balance; exact pro-rata-as-to-time expectation per Mata 2000).
///
/// See [`compound_poisson_reinstatement_slices`] for the frequency/severity inputs. For
/// `basis = ProRataAmount` this equals [`price_reinstatements`] on the exact `E[S_k]`; for
/// `ProRataAmountAndTime` the reinstatement premium uses `T_k = ∫_0^1 E[S_k(t)] dt`
/// (e.g. total losses with `λ = 1` give `T_0 / L = e^{-1}`).
pub fn price_reinstatements_compound_poisson(
    layer_limit: f64,
    poisson_frequency: f64,
    step: f64,
    occurrence_layer_pmf: &[f64],
    reinstatement_rates: &[f64],
    expense_ratio: f64,
    basis: ReinstatementBasis,
) -> Result<ReinstatementPricingResult, PricingError> {
    let slices = compound_poisson_reinstatement_slices(
        layer_limit,
        poisson_frequency,
        step,
        occurrence_layer_pmf,
        reinstatement_rates.len() + 1,
    )?;
    match basis {
        ReinstatementBasis::ProRataAmount => price_reinstatements(
            layer_limit,
            &slices.expected_layer_losses,
            reinstatement_rates,
            expense_ratio,
            basis,
        ),
        ReinstatementBasis::ProRataAmountAndTime => price_reinstatements_time_weighted(
            layer_limit,
            &slices.expected_layer_losses,
            &slices.time_weighted_layer_losses,
            reinstatement_rates,
            expense_ratio,
        ),
    }
}

/// Specification for a proportional treaty sliding-scale ceding commission (Clark 2014).
///
/// Commission slides linearly with slope `slide_ratio` (e.g. `1.0` for 1-for-1 or `0.5` for
/// 0.5% commission per 1% loss ratio) from `max_commission` at `min_lr` down to
/// `min_commission` at `max_lr = min_lr + (max_commission - min_commission) / slide_ratio`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlidingScaleSpec {
    /// Minimum ceding commission ratio (floor, e.g. `0.20`).
    pub min_commission: f64,
    /// Maximum ceding commission ratio (cap, e.g. `0.35`).
    pub max_commission: f64,
    /// Loss ratio at or below which `max_commission` is paid (e.g. `0.50`).
    pub min_lr: f64,
    /// Slide slope: reduction in commission per unit increase in loss ratio (e.g. `1.0` or `0.5`).
    pub slide_ratio: f64,
}

impl SlidingScaleSpec {
    /// Validate specification and return the upper loss ratio `max_lr` where `min_commission` is reached.
    pub fn max_lr(&self) -> Result<f64, PricingError> {
        if !self.min_commission.is_finite()
            || !self.max_commission.is_finite()
            || !self.min_lr.is_finite()
            || !self.slide_ratio.is_finite()
        {
            return Err(PricingError::InvalidParameter(
                "sliding scale parameters must be finite",
            ));
        }
        if !(self.min_commission >= 0.0)
            || !(self.max_commission >= self.min_commission)
            || !(self.max_commission <= 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "must satisfy 0 <= min_commission <= max_commission <= 1",
            ));
        }
        if !(self.min_lr >= 0.0) || !(self.slide_ratio > 0.0) {
            return Err(PricingError::InvalidParameter(
                "min_lr must be >= 0 and slide_ratio must be > 0",
            ));
        }
        let max_lr = self.min_lr + (self.max_commission - self.min_commission) / self.slide_ratio;
        if !max_lr.is_finite() {
            return Err(PricingError::OutOfBounds(
                "sliding scale max_lr overflowed floating-point range",
            ));
        }
        Ok(max_lr)
    }

    /// Deterministic ceding commission rate at realized loss ratio `lr`.
    pub fn commission_at(&self, lr: f64) -> Result<f64, PricingError> {
        let max_lr = self.max_lr()?;
        if !lr.is_finite() || !(lr >= 0.0) {
            return Err(PricingError::InvalidParameter(
                "loss ratio must be finite and non-negative",
            ));
        }
        if lr <= self.min_lr {
            Ok(self.max_commission)
        } else if lr >= max_lr {
            Ok(self.min_commission)
        } else {
            Ok(self.max_commission - self.slide_ratio * (lr - self.min_lr))
        }
    }

    /// Expected sliding-scale ceding commission `E[C(LR)]` given an aggregate loss ratio
    /// distribution (`table_m`) and expected loss ratio `expected_lr` (CAS Exam 8 / Clark 2014):
    ///
    /// `E[C(LR)] = max_commission - slide_ratio * expected_lr * (phi(r_min) - phi(r_max))`
    pub fn expected_commission(
        &self,
        expected_lr: f64,
        table_m: &TableMEngine,
    ) -> Result<f64, PricingError> {
        let max_lr = self.max_lr()?;
        if !expected_lr.is_finite() || !(expected_lr > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "expected_lr must be finite and positive",
            ));
        }
        let r_min = self.min_lr / expected_lr;
        let r_max = max_lr / expected_lr;
        if !r_min.is_finite() || !r_max.is_finite() {
            return Err(PricingError::OutOfBounds(
                "sliding scale entry ratios overflowed floating-point range",
            ));
        }
        let phi_min = table_m.charge_at(r_min);
        let phi_max = table_m.charge_at(r_max);
        Ok(self.max_commission - self.slide_ratio * expected_lr * (phi_min - phi_max))
    }
}

/// Expected cedent retention under a proportional treaty loss corridor (Clark 2014).
///
/// The cedent retains `cedent_share` (in `[0, 1]`) of aggregate losses whose loss ratio falls
/// between `attach_lr` and `detach_lr`:
/// `E[Corridor] = cedent_share * expected_lr * (phi(attach_lr / expected_lr) - phi(detach_lr / expected_lr))`.
pub fn expected_loss_corridor_retention(
    attach_lr: f64,
    detach_lr: f64,
    cedent_share: f64,
    expected_lr: f64,
    table_m: &TableMEngine,
) -> Result<f64, PricingError> {
    if !attach_lr.is_finite()
        || !detach_lr.is_finite()
        || !cedent_share.is_finite()
        || !expected_lr.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "loss corridor parameters must be finite",
        ));
    }
    if !(attach_lr >= 0.0) || !(detach_lr > attach_lr) {
        return Err(PricingError::InvalidParameter(
            "must satisfy 0 <= attach_lr < detach_lr",
        ));
    }
    if !(cedent_share >= 0.0) || !(cedent_share <= 1.0) {
        return Err(PricingError::InvalidParameter(
            "cedent_share must be in [0, 1]",
        ));
    }
    if !(expected_lr > 0.0) {
        return Err(PricingError::NonPositiveValue(
            "expected_lr must be positive",
        ));
    }
    let r_a = attach_lr / expected_lr;
    let r_d = detach_lr / expected_lr;
    if !r_a.is_finite() || !r_d.is_finite() {
        return Err(PricingError::OutOfBounds(
            "corridor entry ratios overflowed floating-point range",
        ));
    }
    let phi_a = table_m.charge_at(r_a);
    let phi_d = table_m.charge_at(r_d);
    Ok(cedent_share * expected_lr * (phi_a - phi_d))
}

/// Expected proportional treaty profit commission ratio (Clark 2014).
///
/// Reinsurer pays `profit_commission_rate * max(0, 1 - ceding_commission - reinsurer_expense - LR)`.
/// Using Table M savings `psi(r) = E[max(0, r - Y)]`:
/// `E[PC] = profit_commission_rate * expected_lr * psi(break_even_lr / expected_lr)`.
pub fn expected_profit_commission(
    ceding_commission: f64,
    reinsurer_expense: f64,
    profit_commission_rate: f64,
    expected_lr: f64,
    table_m: &TableMEngine,
) -> Result<f64, PricingError> {
    if !ceding_commission.is_finite()
        || !reinsurer_expense.is_finite()
        || !profit_commission_rate.is_finite()
        || !expected_lr.is_finite()
    {
        return Err(PricingError::InvalidParameter(
            "profit commission parameters must be finite",
        ));
    }
    if !(ceding_commission >= 0.0)
        || !(reinsurer_expense >= 0.0)
        || !(profit_commission_rate >= 0.0)
        || !(profit_commission_rate <= 1.0)
        || !(expected_lr > 0.0)
    {
        return Err(PricingError::InvalidParameter(
            "commission/expense must be >= 0, profit_commission_rate in [0, 1], expected_lr > 0",
        ));
    }
    let break_even_lr = 1.0 - ceding_commission - reinsurer_expense;
    if break_even_lr <= 0.0 {
        return Ok(0.0);
    }
    let r_be = break_even_lr / expected_lr;
    if !r_be.is_finite() {
        return Err(PricingError::OutOfBounds(
            "profit commission break-even entry ratio overflowed floating-point range",
        ));
    }
    let psi = table_m.savings_at(r_be);
    Ok(profit_commission_rate * expected_lr * psi)
}

// ===========================================================================
// Proportional Treaty Economics: Quota Share & Surplus Share (Clark 2014)
// ===========================================================================

/// Financial summary of a proportional reinsurance treaty (Quota Share or Surplus Share).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProportionalTreatyResult {
    /// Total gross subject premium.
    pub gross_premium: f64,
    /// Total gross subject loss.
    pub gross_loss: f64,
    /// Total ceded premium to the reinsurer.
    pub ceded_premium: f64,
    /// Total ceded loss to the reinsurer (after any event limit).
    pub ceded_loss: f64,
    /// Ceding commission paid by reinsurer back to cedent.
    pub ceding_commission_dollars: f64,
    /// Net premium retained by cedent (`gross_premium - ceded_premium`).
    pub retained_premium: f64,
    /// Net loss retained by cedent (`gross_loss - ceded_loss`).
    pub retained_loss: f64,
    /// Reinsurer net underwriting margin (`ceded_premium - ceded_loss - ceding_commission_dollars`).
    pub reinsurer_margin: f64,
    /// Average cession share across the portfolio (`ceded_premium / gross_premium`).
    pub effective_cession_share: f64,
}

/// Quota Share proportional treaty specification (Clark 2014).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuotaShareSpec {
    /// Fixed cession share `c in (0, 1]` ceded to the reinsurer.
    pub cession_share: f64,
    /// Flat ceding commission ratio `comm in [0, 1)` paid on ceded premium.
    pub ceding_commission: f64,
    /// Optional per-event or aggregate ceded loss cap (`None` = uncapped).
    pub ceded_loss_cap: Option<f64>,
}

impl QuotaShareSpec {
    /// Validate Quota Share treaty parameters.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.cession_share.is_finite()
            || !self.ceding_commission.is_finite()
            || !(self.cession_share > 0.0)
            || !(self.cession_share <= 1.0)
            || !(self.ceding_commission >= 0.0)
            || !(self.ceding_commission < 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "QuotaShare requires cession_share in (0, 1] and ceding_commission in [0, 1)",
            ));
        }
        if let Some(cap) = self.ceded_loss_cap
            && (!cap.is_finite() || !(cap > 0.0))
        {
            return Err(PricingError::NonPositiveValue(
                "ceded_loss_cap must be finite and positive",
            ));
        }
        Ok(())
    }

    /// Apply this Quota Share treaty to a portfolio `(gross_premium, gross_loss)`.
    pub fn apply(
        &self,
        gross_premium: f64,
        gross_loss: f64,
    ) -> Result<ProportionalTreatyResult, PricingError> {
        self.validate()?;
        if !gross_premium.is_finite()
            || !gross_loss.is_finite()
            || !(gross_premium > 0.0)
            || !(gross_loss >= 0.0)
        {
            return Err(PricingError::InvalidParameter(
                "gross_premium must be finite > 0 and gross_loss finite >= 0",
            ));
        }
        let ceded_premium = gross_premium * self.cession_share;
        let raw_ceded_loss = gross_loss * self.cession_share;
        let ceded_loss = match self.ceded_loss_cap {
            Some(cap) => raw_ceded_loss.min(cap),
            None => raw_ceded_loss,
        };
        let ceding_commission_dollars = ceded_premium * self.ceding_commission;
        Ok(ProportionalTreatyResult {
            gross_premium,
            gross_loss,
            ceded_premium,
            ceded_loss,
            ceding_commission_dollars,
            retained_premium: gross_premium - ceded_premium,
            retained_loss: gross_loss - ceded_loss,
            reinsurer_margin: ceded_premium - ceded_loss - ceding_commission_dollars,
            effective_cession_share: ceded_premium / gross_premium,
        })
    }
}

/// Surplus Share proportional treaty specification (Clark 2014).
///
/// For each risk with Total Insured Value `tiv_i`, the cedent retains `retention_line`
/// and cedes up to `number_of_lines * retention_line`:
/// `cession_share_i = clamp((tiv_i - retention_line) / tiv_i, 0, max_treaty_limit / tiv_i)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurplusShareSpec {
    /// Cedent's maximum per-risk retention ("one line") `R > 0`.
    pub retention_line: f64,
    /// Treaty capacity expressed as a multiple of `retention_line` (`> 0`, e.g., `9.0` for a 9-line surplus).
    pub number_of_lines: f64,
    /// Flat ceding commission ratio `comm in [0, 1)` paid on ceded premium.
    pub ceding_commission: f64,
}

impl SurplusShareSpec {
    /// Validate Surplus Share treaty parameters.
    pub fn validate(&self) -> Result<(), PricingError> {
        if !self.retention_line.is_finite()
            || !self.number_of_lines.is_finite()
            || !self.ceding_commission.is_finite()
            || !(self.retention_line > 0.0)
            || !(self.number_of_lines > 0.0)
            || !(self.ceding_commission >= 0.0)
            || !(self.ceding_commission < 1.0)
        {
            return Err(PricingError::InvalidParameter(
                "SurplusShare requires retention_line > 0, number_of_lines > 0, and ceding_commission in [0, 1)",
            ));
        }
        Ok(())
    }

    /// Cession share for a single risk with insured value `tiv > 0`.
    pub fn risk_cession_share(&self, tiv: f64) -> Result<f64, PricingError> {
        self.validate()?;
        if !tiv.is_finite() || !(tiv > 0.0) {
            return Err(PricingError::NonPositiveValue(
                "risk TIV must be finite and positive",
            ));
        }
        if tiv <= self.retention_line {
            return Ok(0.0);
        }
        let ceded_amount =
            (tiv - self.retention_line).min(self.number_of_lines * self.retention_line);
        Ok((ceded_amount / tiv).clamp(0.0, 1.0))
    }

    /// Apply this Surplus Share treaty across a portfolio of risks `(tivs, premiums, losses)`.
    pub fn apply_to_risks(
        &self,
        tivs: &[f64],
        premiums: &[f64],
        losses: &[f64],
    ) -> Result<ProportionalTreatyResult, PricingError> {
        self.validate()?;
        let n = tivs.len();
        if n == 0 {
            return Err(PricingError::EmptyInput(
                "at least one risk is required for SurplusShare evaluation",
            ));
        }
        if premiums.len() != n || losses.len() != n {
            return Err(PricingError::DimensionMismatch(
                "tivs, premiums, and losses must have equal length",
            ));
        }

        let mut gross_premium = 0.0;
        let mut gross_loss = 0.0;
        let mut ceded_premium = 0.0;
        let mut ceded_loss = 0.0;

        for i in 0..n {
            let p = premiums[i];
            let l = losses[i];
            if !p.is_finite() || !l.is_finite() || !(p > 0.0) || !(l >= 0.0) {
                return Err(PricingError::InvalidParameter(
                    "risk premiums must be finite > 0 and losses finite >= 0",
                ));
            }
            let c = self.risk_cession_share(tivs[i])?;
            gross_premium += p;
            gross_loss += l;
            ceded_premium += c * p;
            ceded_loss += c * l;
        }

        let ceding_commission_dollars = ceded_premium * self.ceding_commission;
        Ok(ProportionalTreatyResult {
            gross_premium,
            gross_loss,
            ceded_premium,
            ceded_loss,
            ceding_commission_dollars,
            retained_premium: gross_premium - ceded_premium,
            retained_loss: gross_loss - ceded_loss,
            reinsurer_margin: ceded_premium - ceded_loss - ceding_commission_dollars,
            effective_cession_share: ceded_premium / gross_premium,
        })
    }
}
