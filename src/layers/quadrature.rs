//! Globally adaptive 15-point Gauss–Kronrod (G7–K15) quadrature with a scale-aware
//! initial subdivision (Piessens et al. 1983, QUADPACK `QAG` strategy).
//!
//! A single G7–K15 panel over `[0, L]` places its smallest node at `≈ 0.004 L`; when the
//! integrand's mass sits on a much smaller scale both rules return `≈ 0` and a local
//! error test accepts a wrong answer. This module therefore
//!
//! 1. starts from breakpoints graded geometrically towards both endpoints and around
//!    caller-supplied natural scale points (median, mean, threshold, support end, …), and
//! 2. bisects the panel with the largest error estimate until the *global* error estimate
//!    meets `max(abs_tol, rel_tol * |I|)` with `I` the running global estimate.

use crate::error::PricingError;

const GK15_X: [f64; 8] = [
    0.0,
    0.207_784_955_007_898_5,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_5,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

const GK15_W: [f64; 8] = [
    0.209_482_141_084_727_82,
    0.204_432_940_075_298_89,
    0.190_350_578_064_785_42,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_18,
    0.063_092_092_629_978_55,
    0.022_935_322_010_529_224,
];

const G7_W: [f64; 4] = [
    0.417_959_183_673_469_4,
    0.381_830_050_505_118_9,
    0.279_705_391_489_276_64,
    0.129_484_966_168_869_7,
];

/// Target relative tolerance of the global error estimate.
pub(crate) const QUAD_REL_TOL: f64 = 1e-12;
/// Relative accuracy still accepted when roundoff or the panel budget stops refinement.
const QUAD_ACCEPT_REL_TOL: f64 = 1e-9;
/// Maximum number of panels in the global subdivision.
const QUAD_MAX_PANELS: usize = 4_000;
/// Decades of geometric grading towards each interval endpoint.
const QUAD_EDGE_DECADES: i32 = 14;
/// Decades of geometric grading on either side of each natural scale point.
const QUAD_ANCHOR_DECADES: i32 = 8;
/// Breakpoints closer than this relative gap are merged.
const QUAD_MERGE_REL_GAP: f64 = 1e-12;
/// A panel narrower than this multiple of `ε |mid|` is not bisected further.
const QUAD_MIN_WIDTH_ULPS: f64 = 64.0;
/// Decades of geometric grading of the semi-infinite map `u = (x - lower) / scale`.
const SEMI_INFINITE_DECADES: i32 = 14;

#[derive(Clone, Copy)]
struct Panel {
    a: f64,
    b: f64,
    val: f64,
    err: f64,
}

const NON_FINITE_INTEGRAND: PricingError =
    PricingError::InvalidParameter("integrand returned a non-finite value");

fn gk15_panel<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> Result<Panel, PricingError> {
    let mid = f64::midpoint(a, b);
    let half = 0.5 * (b - a);
    let f_mid = f(mid);
    if !f_mid.is_finite() {
        return Err(NON_FINITE_INTEGRAND);
    }
    let mut k15 = GK15_W[0] * f_mid;
    let mut g7 = G7_W[0] * f_mid;
    for i in 1..8 {
        let dx = half * GK15_X[i];
        let f_sum = f(mid - dx) + f(mid + dx);
        if !f_sum.is_finite() {
            return Err(NON_FINITE_INTEGRAND);
        }
        k15 += GK15_W[i] * f_sum;
        if i % 2 == 0 {
            g7 += G7_W[i / 2] * f_sum;
        }
    }
    Ok(Panel {
        a,
        b,
        val: k15 * half,
        err: ((k15 - g7) * half).abs(),
    })
}

/// Sorted, de-duplicated breakpoints of `[a, b]` (`a < b`, both finite): both endpoints,
/// geometric grading `a + (b - a) 10^-j` and `b - (b - a) 10^-j`, and every
/// `s * 10^j` (`|j| <= QUAD_ANCHOR_DECADES`) of each positive finite anchor `s` in `(a, b)`.
pub(crate) fn graded_breakpoints(a: f64, b: f64, anchors: &[f64]) -> Vec<f64> {
    let width = b - a;
    let mut pts = vec![a, b];
    for j in 1..=QUAD_EDGE_DECADES {
        let d = width * 10f64.powi(-j);
        pts.push(a + d);
        pts.push(b - d);
    }
    for &s in anchors {
        if !(s.is_finite() && s > 0.0) {
            continue;
        }
        for j in -QUAD_ANCHOR_DECADES..=QUAD_ANCHOR_DECADES {
            pts.push(s * 10f64.powi(j));
        }
    }
    pts.retain(|p| p.is_finite() && *p >= a && *p <= b);
    pts.sort_by(f64::total_cmp);
    let mut out: Vec<f64> = Vec::with_capacity(pts.len());
    for p in pts {
        match out.last() {
            Some(&q) if p - q <= QUAD_MERGE_REL_GAP * p.abs().max(q.abs()) => {}
            _ => out.push(p),
        }
    }
    if let Some(last) = out.last_mut() {
        *last = b;
    }
    out
}

/// Globally adaptive G7–K15 over the sorted breakpoints `pts` (length `>= 2`).
pub(crate) fn integrate_panels<F: Fn(f64) -> f64>(
    f: &F,
    pts: &[f64],
    abs_tol: f64,
) -> Result<f64, PricingError> {
    let mut panels = Vec::with_capacity(pts.len().max(2) * 4);
    for w in pts.windows(2) {
        if w[1] > w[0] {
            panels.push(gk15_panel(f, w[0], w[1])?);
        }
    }
    let tol = |total: f64, rel: f64| abs_tol.max(rel * total.abs());
    loop {
        let total: f64 = panels.iter().map(|p| p.val).sum();
        let err: f64 = panels.iter().map(|p| p.err).sum();
        if err <= tol(total, QUAD_REL_TOL) {
            return Ok(total);
        }
        let worst = panels
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.err.total_cmp(&y.1.err))
            .map(|(i, p)| (i, *p));
        let Some((idx, worst)) = worst else {
            return Ok(total);
        };
        let mid = f64::midpoint(worst.a, worst.b);
        let too_narrow =
            (worst.b - worst.a) <= QUAD_MIN_WIDTH_ULPS * f64::EPSILON * mid.abs().max(1.0);
        if too_narrow || panels.len() >= QUAD_MAX_PANELS {
            return if err <= tol(total, QUAD_ACCEPT_REL_TOL) {
                Ok(total)
            } else {
                Err(PricingError::NonConvergence(
                    "adaptive Gauss-Kronrod quadrature did not reach its error tolerance",
                ))
            };
        }
        panels[idx] = gk15_panel(f, worst.a, mid)?;
        panels.push(gk15_panel(f, mid, worst.b)?);
    }
}

/// `∫_a^b f` on a finite interval with graded breakpoints and natural scale `anchors`.
pub(crate) fn integrate_finite<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    anchors: &[f64],
    abs_tol: f64,
) -> Result<f64, PricingError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(PricingError::InvalidParameter(
            "quadrature bounds must be finite",
        ));
    }
    if a == b {
        return Ok(0.0);
    }
    if a > b {
        return integrate_finite(f, b, a, anchors, abs_tol).map(|v| -v);
    }
    integrate_panels(&f, &graded_breakpoints(a, b, anchors), abs_tol)
}

/// `∫_lower^∞ f` via `x = lower + scale (1 - t) / t`, `t ∈ (0, 1]`, with breakpoints at
/// `u = (x - lower) / scale = 10^j` (`|j| <= SEMI_INFINITE_DECADES`) and at every anchor.
pub(crate) fn integrate_semi_infinite<F: Fn(f64) -> f64>(
    f: F,
    lower: f64,
    scale: f64,
    anchors: &[f64],
    abs_tol: f64,
) -> Result<f64, PricingError> {
    if !lower.is_finite() {
        return Err(PricingError::InvalidParameter(
            "semi-infinite quadrature lower bound must be finite",
        ));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(PricingError::InvalidParameter(
            "semi-infinite quadrature scale must be finite and > 0",
        ));
    }
    let to_t = |u: f64| 1.0 / (1.0 + u);
    let mut pts = vec![0.0, 1.0];
    for j in -SEMI_INFINITE_DECADES..=SEMI_INFINITE_DECADES {
        pts.push(to_t(10f64.powi(j)));
    }
    for &x in anchors {
        if x.is_finite() && x > lower {
            pts.push(to_t((x - lower) / scale));
        }
    }
    pts.retain(|t| t.is_finite() && (0.0..=1.0).contains(t));
    pts.sort_by(f64::total_cmp);
    pts.dedup_by(|p, q| (*p - *q).abs() <= QUAD_MERGE_REL_GAP * p.abs().max(q.abs()));
    let transformed = |t: f64| -> f64 {
        if !(t > 0.0) {
            return 0.0;
        }
        let x = lower + scale * ((1.0 - t) / t);
        if !x.is_finite() {
            return 0.0;
        }
        let fx = f(x);
        if fx == 0.0 {
            0.0
        } else {
            fx * (scale / (t * t))
        }
    };
    integrate_panels(&transformed, &pts, abs_tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_scale_mass_on_huge_interval_is_found() {
        // Mass on scale 1e3 inside [0, 1e9]: a single panel sees ~0.
        let v = integrate_finite(|x| (-x / 1e3).exp(), 0.0, 1e9, &[], 0.0).expect("quad");
        assert!((v - 1e3).abs() < 1e-9 * 1e3, "v={v}");
        let s =
            integrate_semi_infinite(|x| (-x / 1e3).exp() / 1e3, 0.0, 1.0, &[], 0.0).expect("semi");
        assert!((s - 1.0).abs() < 1e-10, "s={s}");
    }

    #[test]
    fn breakpoints_are_sorted_unique_and_span_interval() {
        let p = graded_breakpoints(0.0, 1e7, &[1e3, 1e3, 5e6]);
        assert_eq!(p.first().copied(), Some(0.0));
        assert_eq!(p.last().copied(), Some(1e7));
        assert!(p.windows(2).all(|w| w[1] > w[0]));
    }
}
