#!/usr/bin/env node
/**
 * Runtime smoke test for the compiled wasm32-unknown-unknown C-ABI binary.
 *
 * Usage:
 *   node scripts/wasm-smoke.mjs [path/to/pricing_kernels.wasm]
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";

const wasmPath =
  process.argv[2] ?? "target/wasm32-unknown-unknown/release/pricing_kernels.wasm";

const bytes = fs.readFileSync(wasmPath);
const gzipBytes = zlib.gzipSync(bytes, { level: 9 }).length;

const { instance } = await WebAssembly.instantiate(bytes);
const e = instance.exports;

const requiredExports = [
  "memory",
  "pk_max_claims",
  "pk_claim_amounts_ptr",
  "pk_claim_med_only_ptr",
  "pk_ilf_lomax",
  "pk_ilf_lognormal",
  "pk_mbbefd_swiss_re_g",
  "pk_mbbefd_layer_factor",
  "pk_classical_credibility",
  "pk_parallelogram_olf",
  "pk_ncci_experience_mod",
];

for (const name of requiredExports) {
  assert.ok(name in e, `Missing WASM export: ${name}`);
}

const assertClose = (actual, expected, tol, label) => {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tol,
    `${label}: expected ${expected} ± ${tol}, got ${actual}`,
  );
};

// 1. Scratch buffer capacity
const maxClaims = e.pk_max_claims();
assert.ok(maxClaims >= 64, `Unexpected pk_max_claims=${maxClaims}`);

// 2. Increased Limit Factors (Lomax & Lognormal)
const ilfLomax = e.pk_ilf_lomax(2.5, 50_000.0, 500_000.0, 100_000.0);
assert.ok(Number.isFinite(ilfLomax) && ilfLomax > 1.0, `Invalid Lomax ILF: ${ilfLomax}`);
assertClose(e.pk_ilf_lomax(2.5, 50_000.0, 100_000.0, 100_000.0), 1.0, 1e-12, "Lomax ILF at basic limit");

const ilfLognorm = e.pk_ilf_lognormal(11.0, 1.0, 500_000.0, 100_000.0);
assert.ok(Number.isFinite(ilfLognorm) && ilfLognorm > 1.0, `Invalid Lognormal ILF: ${ilfLognorm}`);

// 3. Swiss Re / Bernegger MBBEFD exposure curve & layer factor
const gHalf = e.pk_mbbefd_swiss_re_g(3.0, 0.5);
assert.ok(Number.isFinite(gHalf) && gHalf > 0.5 && gHalf < 1.0, `Invalid MBBEFD G(0.5): ${gHalf}`);
const layerFactor = e.pk_mbbefd_layer_factor(3.0, 250_000.0, 500_000.0, 1_000_000.0);
assert.ok(Number.isFinite(layerFactor) && layerFactor > 0.0 && layerFactor < 1.0, `Invalid MBBEFD layer factor: ${layerFactor}`);

// 4. Classical credibility
assertClose(e.pk_classical_credibility(1082.0, 1082.0), 1.0, 1e-12, "Full credibility");
assertClose(e.pk_classical_credibility(1082.0, 270.5), 0.5, 1e-12, "Quarter volume -> Z=0.5");

// 5. Parallelogram on-level factor
const olf = e.pk_parallelogram_olf(2025.0, 2025.5, 1.10);
assert.ok(Number.isFinite(olf) && olf > 1.0 && olf < 1.10, `Invalid parallelogram OLF: ${olf}`);

// 6. NCCI Experience Mod via static WASM scratch buffers
const amounts = new Float64Array(e.memory.buffer, e.pk_claim_amounts_ptr(), maxClaims);
const medOnly = new Uint8Array(e.memory.buffer, e.pk_claim_med_only_ptr(), maxClaims);
amounts.set([5_000.0, 32_000.0, 12_500.0]);
medOnly.set([1, 0, 0]);
const mod = e.pk_ncci_experience_mod(18_500.0, 80_000.0, 0.30, 0.20, 25_000.0, 1, 3);
assert.ok(Number.isFinite(mod) && mod > 0.5 && mod < 1.5, `Invalid NCCI experience mod: ${mod}`);

// 7. Invalid inputs return NaN rather than trapping
assert.ok(Number.isNaN(e.pk_ilf_lomax(-1.0, 50_000.0, 500_000.0, 100_000.0)), "Negative alpha should return NaN");
assert.ok(Number.isNaN(e.pk_classical_credibility(0.0, 100.0)), "Zero full_standard should return NaN");
assert.ok(Number.isNaN(e.pk_mbbefd_swiss_re_g(-1.0, 0.5)), "Negative c should return NaN");

console.log(
  `WASM smoke test passed (${wasmPath}: ${bytes.length.toLocaleString()} bytes raw, ${gzipBytes.toLocaleString()} bytes gzip)`,
);
console.log(
  `  Lomax ILF = ${ilfLomax.toFixed(4)}, MBBEFD G(0.5) = ${gHalf.toFixed(4)}, NCCI mod = ${mod.toFixed(4)}`,
);
