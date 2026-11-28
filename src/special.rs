//! Pure-Rust special mathematical functions shared across pricing and capital kernels.
//!
//! Includes Cody (1969) `erfc` / standard normal CDF, Acklam inverse normal CDF,
//! Lanczos `ln Γ(z)`, regularized incomplete gamma `P(a, x)`, and regularized
//! incomplete beta `I_x(a, b)`.

/// High-accuracy standard normal CDF `Φ(z)` via Cody (1969) rational `erfc`
/// approximation (max absolute error `< 1e-15`).
pub(crate) fn normal_cdf_exact(z: f64) -> f64 {
    0.5 * erfc_approx(-z / std::f64::consts::SQRT_2)
}

/// Log standard normal survival `ln Φ(-z)` without underflow for large `z > 0`.
///
/// For `z <= 25.0`, evaluates `ln(Φ(-z))` directly via Cody's `erfc` (`< 1e-15` accuracy,
/// well above `f64::MIN_POSITIVE`). For `z > 25.0`, evaluates the asymptotic Mills-ratio
/// expansion directly in log-space (`< 2e-13` relative error, avoiding underflow to `0.0`).
pub(crate) fn log_normal_survival(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z > 25.0 {
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0
            - inv_z2 * (1.0 - inv_z2 * (3.0 - inv_z2 * (15.0 - inv_z2 * (105.0 - 945.0 * inv_z2))));
        -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - z.ln() + series.max(1e-300).ln()
    } else {
        normal_cdf_exact(-z).max(1e-300).ln()
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)` for `a > 0, x >= 0`.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if a < 1.0 && x <= SMALL_SHAPE_MAX_X {
        gamma_q_small_shape(a, x).clamp(0.0, 1.0)
    } else if x >= a + 1.0 {
        gamma_q_cf_log(a, x).exp().clamp(0.0, 1.0)
    } else {
        (1.0 - gamma_p(a, x)).clamp(0.0, 1.0)
    }
}

/// Log regularized upper incomplete gamma `ln Q(a, x)` for `a > 0, x >= 0`
/// without right-tail underflow for large `x`.
pub(crate) fn log_gamma_q(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if a < 1.0 && x <= SMALL_SHAPE_MAX_X {
        gamma_q_small_shape(a, x).clamp(f64::MIN_POSITIVE, 1.0).ln()
    } else if x >= a + 1.0 {
        gamma_q_cf_log(a, x).min(0.0)
    } else {
        (1.0 - gamma_p(a, x)).clamp(1e-300, 1.0).ln()
    }
}

/// Upper end of the `x` range handled by [`gamma_q_small_shape`] for `a < 1`
/// (DiDonato & Morris 1986 use the analogous series for `x <= 1.5`).
const SMALL_SHAPE_MAX_X: f64 = 1.5;
/// Largest `a` for which `ln Γ(1 + a)` uses the zeta Taylor series (`0.2^k` truncation
/// `< 1e-17` after [`LN_GAMMA_1P_TERMS`] terms).
const LN_GAMMA_1P_SERIES_MAX: f64 = 0.2;
/// Number of Taylor terms in [`ln_gamma_1p`].
const LN_GAMMA_1P_TERMS: usize = 26;
/// Relative truncation of the alternating series in [`gamma_q_small_shape`].
const SMALL_SHAPE_SERIES_EPS: f64 = 1e-17;
/// Hard cap on series terms (`x <= 1.5` converges in < 30 terms).
const SMALL_SHAPE_SERIES_MAX_TERMS: usize = 200;

/// Riemann zeta `ζ(k)` for integer `k >= 2` (closed forms / tabulated odd values for `k < 10`,
/// direct summation, which converges geometrically fast, for `k >= 10`).
fn zeta_int(k: usize) -> f64 {
    use std::f64::consts::PI;
    match k {
        2 => PI * PI / 6.0,
        3 => 1.202_056_903_159_594_3,
        4 => PI.powi(4) / 90.0,
        5 => 1.036_927_755_143_37,
        6 => PI.powi(6) / 945.0,
        7 => 1.008_349_277_381_922_8,
        8 => PI.powi(8) / 9450.0,
        9 => 1.002_008_392_826_082_2,
        _ => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let e = k as i32;
            let mut sum = 0.0;
            // n^-k < 1e-18 for n >= 64 when k >= 10; sum small terms first.
            for n in (2..=64_u32).rev() {
                sum += f64::from(n).powi(-e);
            }
            1.0 + sum
        }
    }
}

/// `ln Γ(1 + a)` with full *relative* accuracy as `a -> 0`, for `0 <= a < 1`.
///
/// For `a <= LN_GAMMA_1P_SERIES_MAX`: `ln Γ(1 + a) = -γ a + sum_{k>=2} (-a)^k ζ(k) / k`
/// (Abramowitz & Stegun 6.1.33). Otherwise the Lanczos value, whose magnitude there is
/// `>= 0.08`, has no damaging cancellation.
fn ln_gamma_1p(a: f64) -> f64 {
    if a > LN_GAMMA_1P_SERIES_MAX {
        return ln_gamma(1.0 + a);
    }
    let mut acc = 0.0;
    for k in (2..=LN_GAMMA_1P_TERMS).rev() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let term = sign * a.powi(k as i32) * zeta_int(k) / k as f64;
        acc += term;
    }
    -std::f64::consts::EULER_GAMMA * a + acc
}

/// `Q(a, x)` for `0 < a < 1`, `0 < x <= SMALL_SHAPE_MAX_X` without the `1 - P` cancellation
/// (`Q ≈ a E1(x)` as `a -> 0`; DiDonato & Morris 1986, *ACM TOMS* 12(4), small-`a` branch).
///
/// From `P = x^a / Γ(1 + a) · (1 - J)` with `J = -a sum_{n>=1} (-x)^n / (n! (a + n))`:
/// `Q = -expm1(L) + e^L J`, `L = a ln x - ln Γ(1 + a)`, with `ln Γ(1 + a)` from
/// [`ln_gamma_1p`] so `L = O(a)` keeps full relative accuracy.
fn gamma_q_small_shape(a: f64, x: f64) -> f64 {
    let l = a * x.ln() - ln_gamma_1p(a);
    let mut sum = 0.0;
    let mut pow = 1.0; // (-x)^n / n!
    for n in 1..=SMALL_SHAPE_SERIES_MAX_TERMS {
        let nf = n as f64;
        pow *= -x / nf;
        let term = pow / (a + nf);
        sum += term;
        if term.abs() <= SMALL_SHAPE_SERIES_EPS * sum.abs() {
            break;
        }
    }
    let j = -a * sum;
    -l.exp_m1() + l.exp() * j
}

/// Digamma function `ψ(x) = d/dx ln Γ(x)` for `x > 0` via recurrence shift and
/// asymptotic Bernoulli series (accurate to `< 1e-14`).
pub(crate) fn digamma(mut x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    acc + x.ln()
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0 - inv_x2 / 240.0)))
}

/// Complementary error function `erfc(x)` accurate to ~1e-15 (Cody 1969).
fn erfc_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    if x < 0.46875 {
        const A: [f64; 5] = [
            3.161_123_743_870_565_6e0,
            1.138_641_541_510_501_6e2,
            3.774_852_376_853_02e2,
            3.209_377_589_138_469_5e3,
            1.857_777_061_846_031_5e-1,
        ];
        const B: [f64; 4] = [
            2.360_129_095_234_412_1e1,
            2.440_246_379_344_441_7e2,
            1.282_616_526_077_372_3e3,
            2.844_236_833_439_171e3,
        ];
        let y = x * x;
        let num = (((A[4] * y + A[0]) * y + A[1]) * y + A[2]) * y + A[3];
        let den = (((y + B[0]) * y + B[1]) * y + B[2]) * y + B[3];
        1.0 - x * num / den
    } else if x <= 4.0 {
        const C: [f64; 9] = [
            5.641_884_969_886_701e-1,
            8.883_149_794_388_375,
            6.611_919_063_714_163e1,
            2.986_351_381_974_001e2,
            8.819_522_212_417_69e2,
            1.712_047_612_634_070_6e3,
            2.051_078_377_826_071_5e3,
            1.230_339_354_797_997_2e3,
            2.153_115_354_744_038_5e-8,
        ];
        const D: [f64; 8] = [
            1.574_492_611_070_983_5e1,
            1.176_939_508_913_125e2,
            5.371_811_018_620_099e2,
            1.621_389_574_566_690_2e3,
            3.290_799_235_733_459_6e3,
            4.362_619_090_143_247e3,
            3.439_367_674_143_721_6e3,
            1.230_339_354_803_749_4e3,
        ];
        let num = (((((((C[8] * x + C[0]) * x + C[1]) * x + C[2]) * x + C[3]) * x + C[4]) * x
            + C[5])
            * x
            + C[6])
            * x
            + C[7];
        let den = (((((((x + D[0]) * x + D[1]) * x + D[2]) * x + D[3]) * x + D[4]) * x + D[5]) * x
            + D[6])
            * x
            + D[7];
        (-x * x).exp() * num / den
    } else if x < 26.5 {
        const P: [f64; 6] = [
            3.053_266_349_612_323_4e-1,
            3.603_448_999_498_044_4e-1,
            1.257_817_261_112_292_5e-1,
            1.608_378_514_874_228e-2,
            6.587_491_615_298_378e-4,
            1.631_538_713_730_209_8e-2,
        ];
        const Q: [f64; 5] = [
            2.568_520_192_289_822,
            1.872_952_849_923_460_5e0,
            5.279_051_029_514_285e-1,
            6.051_834_131_244_132e-2,
            2.335_204_976_268_691_9e-3,
        ];
        let y = 1.0 / (x * x);
        let num = ((((P[5] * y + P[0]) * y + P[1]) * y + P[2]) * y + P[3]) * y + P[4];
        let den = ((((y + Q[0]) * y + Q[1]) * y + Q[2]) * y + Q[3]) * y + Q[4];
        let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
        (-x * x).exp() / x * (inv_sqrt_pi - y * num / den)
    } else {
        0.0
    }
}

/// High-accuracy inverse standard normal CDF `Φ⁻¹(p)` for `p in (0, 1)`
/// via Acklam rational approximation refined with one Newton-Raphson step (`< 1e-15`).
pub(crate) fn normal_inv_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    if p.is_nan() {
        return f64::NAN;
    }
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let mut x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Newton-Raphson refinement step using exact Cody normal CDF.
    let err = normal_cdf_exact(x) - p;
    let pdf = (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    if pdf > 1e-300 {
        x -= err / pdf;
    }
    x
}

/// High-accuracy `ln Γ(z)` for `z > 0` via the 9-term Lanczos approximation (`g = 7`) for
/// `z >= 1`, and `ln Γ(1 + z) - ln z` (series near 0) for `z < 1`.
pub(crate) fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if !z.is_finite() || z <= 0.0 {
        return f64::NAN;
    }
    if z < 1.0 {
        // Lanczos loses relative accuracy as z -> 0 (5e-9 in P(a, x) at a = 1e-8);
        // ln Γ(z) = ln Γ(1 + z) - ln z with the small-argument series.
        return ln_gamma_1p(z) - z.ln();
    }
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)` for `a > 0, x >= 0`.
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    if x < a + 1.0 {
        // Series representation: P(a, x) = e^{-x} x^a sum_{n=0}^∞ x^n / Γ(a + n + 1).
        let max_iter = 2_000_usize
            .max((12.0 * a.sqrt()).ceil() as usize)
            .min(50_000);
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..max_iter {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * 1e-15 {
                break;
            }
        }
        (sum * ln_gamma_prefactor(a, x).exp()).clamp(0.0, 1.0)
    } else {
        (1.0 - gamma_q_cf(a, x)).clamp(0.0, 1.0)
    }
}

/// Log of regularized upper incomplete gamma `ln Q(a, x)` via modified Lentz
/// continued fraction for `x >= a + 1` (avoids underflow for large `x`).
fn gamma_q_cf_log(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let max_iter = 2_000_usize
        .max((12.0 * a.sqrt()).ceil() as usize)
        .min(50_000);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let fi = i as f64;
        let an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= 1e-15 {
            break;
        }
    }
    ln_gamma_prefactor(a, x) + h.max(TINY).ln()
}

/// `ln(x^a e^{-x} / Γ(a))` for `a > 0, x > 0`, the common prefactor of the incomplete-gamma
/// series and continued fraction.
///
/// For `a >= STIRLING_MIN_SHAPE` the direct form `-x + a ln x - ln Γ(a)` loses ~`|ln Γ(a)|·ε`
/// absolute (relative error 1e-11 at `a = 1e4`, 2e-9 at `a = 1e6`). Writing
/// `ln Γ(a) = (a - 1/2) ln a - a + ln(2π)/2 + δ(a)` and `u = (x - a) / a` gives the
/// cancellation-free form (Temme 1979; DiDonato & Morris 1986, *ACM TOMS* 12(4), `rcomp`):
/// `-a·rlog1(u) + ln(a / 2π)/2 - δ(a)` with `rlog1(u) = u - ln(1 + u)`.
fn ln_gamma_prefactor(a: f64, x: f64) -> f64 {
    if a >= STIRLING_MIN_SHAPE {
        let u = (x - a) / a;
        -a * rlog1(u) + 0.5 * (a / std::f64::consts::TAU).ln() - stirling_correction(a)
    } else {
        -x + a * x.ln() - ln_gamma(a)
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)` via modified Lentz
/// continued fraction for `x >= a + 1`.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    gamma_q_cf_log(a, x).exp().clamp(0.0, 1.0)
}

/// Regularized incomplete beta function `I_x(a, b)` via modified Lentz continued fraction.
pub(crate) fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = ln_beta_kernel(x, a, b).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * betacf(x, a, b) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * betacf(1.0 - x, b, a) / b).clamp(0.0, 1.0)
    }
}

/// Shape at and above which the Stirling-series remainder `δ(z)` is used (7 Bernoulli terms
/// give `|error| < 1e-17` for `z >= 10`).
const STIRLING_MIN_SHAPE: f64 = 10.0;

/// Stirling-series remainder `δ(z) = ln Γ(z) - [(z - 1/2) ln z - z + ln(2π) / 2]` for
/// `z >= STIRLING_MIN_SHAPE` (R `lgammacor`; DiDonato & Morris 1992, *ACM TOMS* 18(3)).
fn stirling_correction(z: f64) -> f64 {
    // B_{2k} / (2k (2k - 1)) for k = 1..=7
    const C: [f64; 7] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
    ];
    let inv_z2 = 1.0 / (z * z);
    let mut acc = 0.0;
    for &c in C.iter().rev() {
        acc = acc * inv_z2 + c;
    }
    acc / z
}

/// `u - ln(1 + u)` for `u > -1` without cancellation near `u = 0` (DiDonato & Morris `rlog1`).
///
/// For `|u| <= 1/2`, with `r = u / (2 + u)`: `ln(1 + u) = 2 atanh(r)` and `u - 2r = r u`, so
/// `u - ln(1 + u) = r u - 2 sum_{k>=1} r^(2k+1) / (2k + 1)` with `|r| <= 1/3`.
fn rlog1(u: f64) -> f64 {
    const SERIES_MAX_ABS_U: f64 = 0.5;
    const MAX_TERMS: usize = 40;
    if u.abs() > SERIES_MAX_ABS_U {
        return u - u.ln_1p();
    }
    let r = u / (2.0 + u);
    let r2 = r * r;
    let mut power = r * r2;
    let mut series = 0.0;
    for k in 1..=MAX_TERMS {
        let term = power / ((2 * k + 1) as f64);
        series += term;
        if term.abs() <= f64::EPSILON * series.abs() {
            break;
        }
        power *= r2;
    }
    r * u - 2.0 * series
}

/// Log of the incomplete-beta prefactor `ln[x^a (1 - x)^b / B(a, b)]` for `0 < x < 1`.
///
/// Computing `ln Γ(a + b) - ln Γ(a) - ln Γ(b)` directly cancels terms of size `~(a + b) ln(a + b)`
/// (relative error `~2.6e-7` in `I_x` at `a + b = 1e6`). Following DiDonato & Morris (1992,
/// `brcomp`) / R `lbeta`, the large-shape cases use Stirling remainders `δ(·)`:
/// - `a, b >= 10`: with `s = a + b`, `t = s x - a`, `u = t / a`, `v = -t / b`,
///   `ln = -a rlog1(u) - b rlog1(v) + ln(a b / (2π s)) / 2 - [δ(a) + δ(b) - δ(s)]`
///   (the first-order terms `a u + b v = 0` cancel analytically);
/// - `p = min(a, b) < 10 <= q = max(a, b)`:
///   `ln B = ln Γ(p) + δ(q) - δ(s) + p - p ln s + (q - 1/2) ln(1 - p / s)`;
/// - both `< 10`: Lanczos `ln Γ` directly (no large-magnitude cancellation).
fn ln_beta_kernel(x: f64, a: f64, b: f64) -> f64 {
    let s = a + b;
    let lo = a.min(b);
    let hi = a.max(b);
    if lo >= STIRLING_MIN_SHAPE {
        let t = x.mul_add(s, -a);
        let u = t / a;
        let v = -t / b;
        let delta = stirling_correction(a) + stirling_correction(b) - stirling_correction(s);
        -a * rlog1(u) - b * rlog1(v) + 0.5 * ((a / s).ln() + b.ln() - std::f64::consts::TAU.ln())
            - delta
    } else if hi >= STIRLING_MIN_SHAPE {
        let ln_beta = ln_gamma(lo) + stirling_correction(hi) - stirling_correction(s) + lo
            - lo * s.ln()
            + (hi - 0.5) * (-lo / s).ln_1p();
        a * x.ln() + b * (-x).ln_1p() - ln_beta
    } else {
        ln_gamma(s) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p()
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const EPS: f64 = 3.0e-15;
    const FPMIN: f64 = 1.0e-300;
    let max_iter = 2_000_usize
        .max((12.0 * (a + b).sqrt()).ceil() as usize)
        .min(50_000);

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_and_inv_roundtrip() {
        for &p in &[1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-6] {
            let z = normal_inv_cdf(p);
            let p2 = normal_cdf_exact(z);
            assert!((p2 - p).abs() < 1e-13, "p={p}, z={z}, p2={p2}");
        }
        assert!(normal_cdf_exact(f64::NAN).is_nan());
        assert!(normal_inv_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn log_normal_survival_continuous_and_accurate_across_crossover() {
        // Verify continuity across the z = 25.0 Cody-erfc -> Mills-ratio crossover.
        let left = log_normal_survival(25.0);
        let right = log_normal_survival(25.000_001);
        assert!((left - right).abs() < 1e-4, "left={left}, right={right}");
        // At z = 10.0, Cody erfc gives exact ln Phi(-10) ≈ -53.222536619...
        let z10 = log_normal_survival(10.0);
        let exact_10 = normal_cdf_exact(-10.0).ln();
        assert!((z10 - exact_10).abs() < 1e-14);
        // Far tail z = 40.0 must remain finite without underflow.
        let z40 = log_normal_survival(40.0);
        assert!(z40.is_finite() && z40 < -800.0, "z40={z40}");
    }

    #[test]
    fn gamma_p_converges_for_large_shape() {
        // For Gamma(a = 10_000, scale = 1), P(a, a) ≈ 0.5 + 1 / (3 * sqrt(2 * pi * a)) ≈ 0.50133
        let p = gamma_p(10_000.0, 10_000.0);
        assert!((p - 0.501_330).abs() < 1e-4, "p={p}");
        assert!(gamma_p(1.0, f64::NAN).is_nan());
        assert!(ln_gamma(0.0).is_nan());
    }

    /// Independent `ln n! - (n ln n - n)`: exact product for `n <= 170`, otherwise the Stirling
    /// series `ln(2πn)/2 + 1/(12n) - 1/(360n³) + 1/(1260n⁵)` written out here (truncation
    /// `< 1/(1680 n⁷) < 1e-24` for `n >= 1000`), so no `O(n ln n)` cancellation occurs.
    fn ref_ln_factorial_minus_leading(n: u64) -> f64 {
        let m = n as f64;
        if n <= 170 {
            return (1..=n).map(|k| k as f64).product::<f64>().ln() - (m * m.ln() - m);
        }
        let inv = 1.0 / m;
        0.5 * (std::f64::consts::TAU * m).ln()
            + inv * (1.0 / 12.0 - inv * inv * (1.0 / 360.0 - inv * inv / 1260.0))
    }

    /// `t - ln(1 + t)` by its Taylor series for `|t| < 0.1` (independent of `rlog1`).
    fn ref_t_minus_ln1p(t: f64) -> f64 {
        if t.abs() >= 0.1 {
            return t - t.ln_1p();
        }
        let mut acc = 0.0;
        let mut pow = t * t;
        for k in 2..60_u32 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            acc += sign * pow / f64::from(k);
            pow *= t;
        }
        acc
    }

    /// `ln[e^{-x} x^m / m!]` (Poisson pmf at `m`) with `m ln(x/m) - (x - m)` evaluated as
    /// `-m (t - ln(1 + t))`, `t = (x - m)/m`.
    fn ref_ln_poisson_pmf(m: u64, x: f64) -> f64 {
        let mf = m as f64;
        let t = (x - mf) / mf;
        -mf * ref_t_minus_ln1p(t) - ref_ln_factorial_minus_leading(m)
    }

    /// Smaller Poisson tail by ratio recurrence from the pmf at the anchor (Neumaier sum):
    /// for `x >= n`: `Q(n, x) = P(N <= n - 1)`; for `x < n`: `P(n, x) = P(N >= n)`.
    fn ref_gamma_small_tail(n: u64, x: f64) -> (bool, f64) {
        // Neumaier-compensated sum of positive decreasing terms.
        let (mut sum, mut comp) = (0.0_f64, 0.0_f64);
        let mut add = |sum: &mut f64, v: f64| {
            let t = *sum + v;
            comp += if sum.abs() >= v.abs() {
                (*sum - t) + v
            } else {
                (v - t) + *sum
            };
            *sum = t;
        };
        let upper = x >= n as f64;
        if upper {
            let mut k = n - 1;
            let mut term = ref_ln_poisson_pmf(k, x).exp();
            loop {
                add(&mut sum, term);
                if k == 0 || term < 1e-20 * sum {
                    break;
                }
                term *= k as f64 / x;
                k -= 1;
            }
        } else {
            let mut k = n;
            let mut term = ref_ln_poisson_pmf(k, x).exp();
            loop {
                add(&mut sum, term);
                k += 1;
                term *= x / k as f64;
                if term < 1e-20 * sum {
                    break;
                }
            }
        }
        (upper, sum + comp)
    }

    /// Independent `Γ(a, x) = ∫_x^∞ t^(a-1) e^(-t) dt` by Romberg quadrature in `t = x e^s`
    /// (`dt = t ds`, smooth integrand `exp(a ln t - t)`), Neumaier-summed over unit panels.
    fn ref_upper_gamma_integral(a: f64, x: f64) -> f64 {
        const PANEL: f64 = 0.25;
        const DEPTH: usize = 9;
        let f = |s: f64| {
            let t = x * s.exp();
            (a * t.ln() - t).exp() * 1.0
        };
        let s_max = ((x + 60.0) / x).ln();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_panels = (s_max / PANEL).ceil() as usize;
        let (mut sum, mut comp) = (0.0_f64, 0.0_f64);
        for p in 0..n_panels {
            let lo = p as f64 * PANEL;
            let hi = (lo + PANEL).min(s_max);
            let mut r = [[0.0_f64; DEPTH]; DEPTH];
            let mut h = hi - lo;
            r[0][0] = 0.5 * h * (f(lo) + f(hi));
            for i in 1..DEPTH {
                h *= 0.5;
                let n_new = 1_usize << (i - 1);
                let mid: f64 = (0..n_new).map(|k| f(lo + (2 * k + 1) as f64 * h)).sum();
                r[i][0] = 0.5 * r[i - 1][0] + h * mid;
                let mut pow4 = 1.0;
                for j in 1..=i {
                    pow4 *= 4.0;
                    r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / (pow4 - 1.0);
                }
            }
            let v = r[DEPTH - 1][DEPTH - 1];
            let t = sum + v;
            comp += if sum.abs() >= v.abs() {
                (sum - t) + v
            } else {
                (v - t) + sum
            };
            sum = t;
        }
        sum + comp
    }

    #[test]
    fn small_shape_upper_gamma_matches_independent_quadrature() {
        for &a in &[1e-8_f64, 1e-4, 0.01, 0.1, 0.5] {
            // Q = a Γ(a, x) / Γ(1 + a); Γ(1 + a) ∈ [0.88, 1] carries no cancellation.
            let gamma_1p = ln_gamma(1.0 + a).exp();
            for &x in &[1e-6_f64, 0.01, 0.5, 1.0, 2.0] {
                let exact = a * ref_upper_gamma_integral(a, x) / gamma_1p;
                let got = gamma_q(a, x);
                let rel = (got - exact).abs() / exact;
                assert!(
                    rel <= 1e-12,
                    "a={a} x={x}: Q={got:e} ref={exact:e} rel={rel:e}"
                );
                let lq = log_gamma_q(a, x);
                let lrel = (lq - exact.ln()).abs() / exact.ln().abs().max(1.0);
                assert!(lrel <= 1e-12, "a={a} x={x}: lnQ={lq} ref={} ", exact.ln());
                // Complement stays consistent.
                let pq = gamma_p(a, x) + got - 1.0;
                assert!(pq.abs() <= 1e-13, "a={a} x={x}: P+Q-1={pq:e}");
            }
        }
        // a = 1/2: Q(1/2, x) = erfc(sqrt x) (tabulated erfc(1) = 0.157_299_207_050_285_13).
        assert!((gamma_q(0.5, 1.0) - 0.157_299_207_050_285_13).abs() <= 1e-15);
    }

    #[test]
    fn incomplete_gamma_matches_poisson_tails_at_large_shapes() {
        for &n in &[10_u64, 1_000, 10_000, 1_000_000] {
            let a = n as f64;
            for &k in &[-5.0_f64, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0] {
                let x = (a + k * a.sqrt()).round().max(1.0) + 0.25;
                let (is_q, exact) = ref_gamma_small_tail(n, x);
                let got = if is_q { gamma_q(a, x) } else { gamma_p(a, x) };
                let rel = (got - exact).abs() / exact;
                assert!(
                    rel <= 1e-12,
                    "a={a} x={x} {}: got={got:e} exact={exact:e} rel={rel:e}",
                    if is_q { "Q" } else { "P" }
                );
                if is_q {
                    let lq = log_gamma_q(a, x);
                    assert!((lq - exact.ln()).abs() <= 1e-12 * exact.ln().abs().max(1.0));
                }
            }
        }
    }

    #[test]
    fn incomplete_beta_converges_for_large_params() {
        // Symmetric Beta(5_000, 5_000) has exact median 0.5 (stable Stirling-remainder prefactor).
        let cdf = regularized_incomplete_beta(0.5, 5_000.0, 5_000.0);
        assert!((cdf - 0.5).abs() < 1e-13, "cdf={cdf}");
        assert!(regularized_incomplete_beta(f64::NAN, 2.0, 2.0).is_nan());
    }
}

#[cfg(test)]
mod large_shape_beta_tests {
    use super::*;

    /// Independent reference `P(Bin(n, x) >= k)` from pmf ratios
    /// `p_{j+1} / p_j = (n - j) x / ((j + 1)(1 - x))`, accumulated as compensated sums of
    /// `ln1p((p_{j+1} - p_j) / p_j)` outward from the mode and normalized by the total, so no
    /// large `ln Γ` / log-factorial differences enter.
    fn binomial_upper_tail(n: u64, k: u64, x: f64) -> f64 {
        const LN_CUTOFF: f64 = -80.0;
        let nf = n as f64;
        let q = 1.0 - x;
        let ln_ratio = |j: f64| -> f64 {
            // ln[(n - j) x / ((j + 1) q)] for the step j -> j + 1
            let num = (nf - j) * x;
            let den = (j + 1.0) * q;
            ((num - den) / den).ln_1p()
        };
        let mode = ((nf + 1.0) * x).floor().min(nf);
        let mut total = 1.0_f64;
        let mut tail = if mode >= k as f64 { 1.0 } else { 0.0 };
        // Upward from the mode.
        let (mut acc, mut comp) = (0.0_f64, 0.0_f64);
        let mut j = mode;
        while j < nf {
            let y = ln_ratio(j) - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
            j += 1.0;
            if acc < LN_CUTOFF {
                break;
            }
            let w = acc.exp();
            total += w;
            if j >= k as f64 {
                tail += w;
            }
        }
        // Downward from the mode: ln p_{j-1} - ln p_j = -ln_ratio(j - 1).
        let (mut acc, mut comp) = (0.0_f64, 0.0_f64);
        let mut j = mode;
        while j > 0.0 {
            let y = -ln_ratio(j - 1.0) - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
            j -= 1.0;
            if acc < LN_CUTOFF {
                break;
            }
            let w = acc.exp();
            total += w;
            if j >= k as f64 {
                tail += w;
            }
        }
        tail / total
    }

    #[test]
    fn incomplete_beta_matches_exact_binomial_tails_at_large_shapes() {
        // I_x(k, n - k + 1) = P(Bin(n, x) >= k), a + b = n + 1 in {1e4, 2e5, 1e6}.
        let cases: [(u64, u64, f64); 6] = [
            (10_000, 5_050, 0.5),
            (10_000, 1_030, 0.1),
            (200_000, 100_300, 0.5),
            (200_000, 99_500, 0.5),
            (1_000_000, 500_500, 0.5),
            (1_000_000, 100_500, 0.1),
        ];
        for (n, k, x) in cases {
            let got = regularized_incomplete_beta(x, k as f64, (n - k + 1) as f64);
            let exact = binomial_upper_tail(n, k, x);
            let rel = (got - exact).abs() / exact;
            assert!(
                rel <= 1e-12,
                "n={n} k={k} x={x}: got={got:e} exact={exact:e} rel={rel:e}"
            );
        }
    }

    #[test]
    fn stable_kernel_matches_direct_form_at_small_shapes() {
        // Below the Stirling threshold the kernel must agree with the direct Lanczos form.
        for &(a, b) in &[(0.5, 3.0), (2.0, 9.5), (1.0, 25.0), (12.0, 40.0)] {
            for &x in &[0.05, 0.3, 0.7] {
                let direct = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
                    + a * f64::ln(x)
                    + b * (1.0 - x).ln();
                let kernel = ln_beta_kernel(x, a, b);
                assert!(
                    (direct - kernel).abs() < 1e-12,
                    "a={a} b={b} x={x}: {direct} vs {kernel}"
                );
            }
        }
    }
}

#[cfg(test)]
mod reference_value_tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= TOL,
            "{label}: actual={actual:e}, expected={expected:e}, diff={:e}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn gamma_p_matches_closed_forms() {
        for &x in &[0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0] {
            // P(1, x) = 1 - e^{-x}
            assert_close(gamma_p(1.0, x), -(-x).exp_m1(), "P(1,x)");
            // P(2, x) = 1 - e^{-x} (1 + x)
            assert_close(gamma_p(2.0, x), 1.0 - (-x).exp() * (1.0 + x), "P(2,x)");
            // P(3, x) = 1 - e^{-x} (1 + x + x^2 / 2)
            assert_close(
                gamma_p(3.0, x),
                1.0 - (-x).exp() * (1.0 + x + 0.5 * x * x),
                "P(3,x)",
            );
            // Complement identity P + Q = 1
            assert_close(gamma_p(2.5, x) + gamma_q(2.5, x), 1.0, "P+Q");
        }
    }

    #[test]
    fn gamma_p_half_matches_tabulated_erf() {
        // P(1/2, x) = erf(sqrt(x)); erf reference values to 16 significant digits.
        let table = [
            (0.25_f64, 0.520_499_877_813_046_5_f64), // erf(0.5)
            (1.0, 0.842_700_792_949_714_9),          // erf(1)
            (2.25, 0.966_105_146_475_310_7),         // erf(1.5)
            (4.0, 0.995_322_265_018_952_7),          // erf(2)
            (9.0, 0.999_977_909_503_001_4),          // erf(3)
        ];
        for (x, erf_sqrt_x) in table {
            assert_close(gamma_p(0.5, x), erf_sqrt_x, "P(1/2,x)");
        }
    }

    #[test]
    fn incomplete_beta_matches_closed_forms() {
        for &x in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            for &a in &[0.5, 1.0, 2.5, 7.0] {
                // I_x(a, 1) = x^a
                assert_close(regularized_incomplete_beta(x, a, 1.0), x.powf(a), "I(a,1)");
                // I_x(1, b) = 1 - (1 - x)^b
                assert_close(
                    regularized_incomplete_beta(x, 1.0, a),
                    1.0 - (1.0 - x).powf(a),
                    "I(1,b)",
                );
                // Reflection I_x(a, b) = 1 - I_{1-x}(b, a)
                assert_close(
                    regularized_incomplete_beta(x, a, 3.0),
                    1.0 - regularized_incomplete_beta(1.0 - x, 3.0, a),
                    "reflection",
                );
            }
            // I_x(2, 2) = 3x^2 - 2x^3
            assert_close(
                regularized_incomplete_beta(x, 2.0, 2.0),
                3.0 * x * x - 2.0 * x * x * x,
                "I(2,2)",
            );
            // I_x(1/2, 1/2) = (2 / pi) asin(sqrt x)
            assert_close(
                regularized_incomplete_beta(x, 0.5, 0.5),
                std::f64::consts::FRAC_2_PI * x.sqrt().asin(),
                "I(1/2,1/2)",
            );
        }
        for &a in &[0.5, 3.0, 40.0] {
            assert_close(regularized_incomplete_beta(0.5, a, a), 0.5, "I_0.5(a,a)");
        }
    }
}
