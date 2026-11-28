#!/usr/bin/env python3
"""Parse Criterion benchmark results, check regressions, and refresh README / SVG assets.

Uses only the Python 3 standard library so it runs cleanly on any mesh node or CI runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def load_criterion(criterion_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    pattern = str(criterion_dir / "**" / "new" / "estimates.json")
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        path = Path(path_str)
        rel = path.relative_to(criterion_dir).as_posix()
        bench_id = rel[: -len("/new/estimates.json")]
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        results[bench_id] = {
            "median_ns": round(float(data["median"]["point_estimate"]), 2),
            "mean_ns": round(float(data["mean"]["point_estimate"]), 2),
        }
    return results


def fmt_latency(ns: float) -> str:
    if ns < 10.0:
        return f"{ns:.2f} ns"
    if ns < 100.0:
        return f"{ns:.1f} ns"
    if ns < 1_000.0:
        return f"{ns / 1_000.0:.2f} µs"
    us = ns / 1_000.0
    if us < 10.0:
        return f"{us:.2f} µs"
    if us < 100.0:
        return f"{us:.1f} µs"
    if us < 1_000.0:
        return f"{us:.0f} µs"
    ms = us / 1_000.0
    if ms < 10.0:
        return f"{ms:.2f} ms"
    return f"{ms:.1f} ms"


def detect_rustc_version() -> str:
    try:
        out = subprocess.check_output(["rustc", "--version"], text=True).strip()
        parts = out.split()
        return parts[1] if len(parts) >= 2 else out
    except Exception:
        return "unknown"


def detect_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def render_svg(baseline: dict, out_path: Path) -> None:
    meta = baseline["metadata"]
    benches = baseline["benchmarks"]
    entries = baseline["entries"]

    ordered = sorted(entries, key=lambda e: float(benches[e["id"]]["median_ns"]))
    width = 920
    row_step = 34
    first_y = 92
    height = first_y + len(ordered) * row_step + 40

    bar_x = 300.0
    bar_max_w = 430.0
    domain_x = 892.0
    min_ns = max(1.0, min(float(benches[e["id"]]["median_ns"]) for e in ordered))
    max_ns = max(10.0, max(float(benches[e["id"]]["median_ns"]) for e in ordered))
    log_min = math.log10(min_ns)
    log_span = max(1.0, math.log10(max_ns) - log_min)

    hw_short = meta["hardware"].replace("Intel Core ", "")
    rustc_short = ".".join(str(meta["rustc"]).split(".")[:2])

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="pricing-kernels steady-state benchmark latency">',
        "  <defs>",
        "    <style>",
        "      .bg { fill: #0d1117; stroke: #262c36; stroke-width: 1; }",
        '      .title { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f0f6fc; letter-spacing: -0.2px; }',
        '      .sub { font: 400 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
        '      .label { font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }',
        "      .val { font: 600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #fdba74; }",
        '      .dom { font: 500 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #8b949e; }',
        '      .hdr { font: 600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #6e7681; letter-spacing: 0.6px; }',
        '      .foot { font: 400 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #6e7681; }',
        "    </style>",
        "  </defs>",
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" class="bg"/>',
        '  <text x="26" y="36" class="title">Steady-State Kernel Latency (pricing-kernels)</text>',
        '  <text x="26" y="55" class="sub">Log10 scale · single-threaded --release Criterion medians · lower is faster</text>',
        f'  <line x1="26" y1="68" x2="{width - 26}" y2="68" stroke="#21262d" stroke-width="1"/>',
        '  <text x="26" y="84" class="hdr">KERNEL</text>',
        f'  <text x="{bar_x:.1f}" y="84" class="hdr">MEDIAN LATENCY (LOG SCALE)</text>',
        f'  <text x="{domain_x:.1f}" y="84" text-anchor="end" class="hdr">DOMAIN</text>',
    ]

    for idx, entry in enumerate(ordered):
        y0 = first_y + idx * row_step
        yc = y0 + row_step / 2.0
        ns = float(benches[entry["id"]]["median_ns"])
        frac = (math.log10(max(1.0, ns)) - log_min) / log_span
        bar_w = max(8.0, 24.0 + frac * (bar_max_w - 24.0))

        if idx % 2 == 1:
            lines.append(
                f'  <rect x="18" y="{y0 + 1:.1f}" width="{width - 36}" height="{row_step - 2:.1f}" rx="5" fill="#161b22" opacity="0.55"/>'
            )

        lines.append(
            f'  <text x="26" y="{yc + 4:.1f}" class="label">{entry["label"]}</text>'
        )
        lines.append(
            f'  <rect x="{bar_x:.1f}" y="{yc - 6:.1f}" width="{bar_w:.1f}" height="12" rx="2.5" fill="#f97316"/>'
        )
        lines.append(
            f'  <text x="{bar_x + bar_w + 8:.1f}" y="{yc + 4:.1f}" class="val">{fmt_latency(ns)}</text>'
        )
        lines.append(
            f'  <text x="{domain_x:.1f}" y="{yc + 4:.1f}" text-anchor="end" class="dom">{entry["domain"]}</text>'
        )

    lines.append(
        f'  <text x="26" y="{height - 14}" class="foot">{hw_short} · rustc {rustc_short} · pure Rust (serde only) · #![forbid(unsafe_code)]</text>'
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme_table(baseline: dict, readme_path: Path) -> None:
    benches = baseline["benchmarks"]
    entries = sorted(
        baseline["entries"], key=lambda e: float(benches[e["id"]]["median_ns"])
    )
    rows = [
        "| Kernel | Median (`i9-14900K`) | Domain |",
        "| --- | :---: | :---: |",
    ]
    for entry in entries:
        ns = float(benches[entry["id"]]["median_ns"])
        rows.append(
            f"| {entry['label']} | `{fmt_latency(ns)}` | {entry['domain']} |"
        )
    new_table = "\n".join(rows)

    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\| Kernel \| Median \(`i9-14900K`\) \| Domain \|\n"
        r"\| --- \| :---: \| :---: \|\n"
        r"(?:\|.*\n)+",
        re.MULTILINE,
    )
    updated, count = pattern.subn(new_table + "\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate performance table in {readme_path}")
    readme_path.write_text(updated, encoding="utf-8")


def compare_and_report(
    current: dict[str, dict[str, float]],
    baseline: dict,
    threshold_pct: float,
) -> tuple[str, list[str], dict]:
    base_benches: dict[str, dict[str, float]] = baseline.get("benchmarks", {})
    meta = baseline.get("metadata", {})

    lines = [
        f"## Criterion Benchmark Report (`{meta.get('hardware', 'unknown')}`)",
        "",
        f"- **Baseline:** `{meta.get('commit', 'unknown')}` ({meta.get('updated_at', '')}, rustc `{meta.get('rustc', '')}`)",
        f"- **Current:** `{detect_git_sha()}` (rustc `{detect_rustc_version()}`)",
        f"- **Regression threshold:** `+{threshold_pct:.1f}%` (median)",
        "",
        "| Benchmark | Baseline | Current | Delta | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    regressions: list[str] = []
    summary_rows: list[dict] = []

    for bench_id in sorted(current.keys()):
        cur_ns = current[bench_id]["median_ns"]
        base_entry = base_benches.get(bench_id)
        if base_entry is None:
            lines.append(
                f"| `{bench_id}` | — | {fmt_latency(cur_ns)} | new | new |"
            )
            summary_rows.append(
                {
                    "id": bench_id,
                    "baseline_ns": None,
                    "current_ns": cur_ns,
                    "delta_pct": None,
                }
            )
            continue

        base_ns = float(base_entry["median_ns"])
        delta_pct = ((cur_ns - base_ns) / base_ns) * 100.0 if base_ns > 0 else 0.0
        if delta_pct > threshold_pct:
            status = "regression"
            regressions.append(
                f"{bench_id}: +{delta_pct:.1f}% ({fmt_latency(base_ns)} -> {fmt_latency(cur_ns)})"
            )
        elif delta_pct > 5.0:
            status = "slower"
        elif delta_pct < -5.0:
            status = "faster"
        else:
            status = "steady"

        lines.append(
            f"| `{bench_id}` | {fmt_latency(base_ns)} | {fmt_latency(cur_ns)} | "
            f"{delta_pct:+.1f}% | {status} |"
        )
        summary_rows.append(
            {
                "id": bench_id,
                "baseline_ns": base_ns,
                "current_ns": cur_ns,
                "delta_pct": round(delta_pct, 2),
                "status": status,
            }
        )

    if regressions:
        lines.extend(
            [
                "",
                f"### Regressions exceeding +{threshold_pct:.1f}%",
                *[f"- `{r}`" for r in regressions],
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"No regressions exceeding `+{threshold_pct:.1f}%` detected across {len(current)} benchmarks.",
            ]
        )

    report_md = "\n".join(lines) + "\n"
    summary_json = {
        "commit": detect_git_sha(),
        "rustc": detect_rustc_version(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "threshold_pct": threshold_pct,
        "regressions": regressions,
        "benchmarks": summary_rows,
    }
    return report_md, regressions, summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "criterion",
        help="Path to Criterion output directory",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benches/baseline-i9-14900k.json"),
        help="Path to checked-in baseline JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Regression percentage threshold on median latency (default: 15.0)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any benchmark regresses beyond --threshold",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Append Markdown report to $GITHUB_STEP_SUMMARY if set",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write structured JSON benchmark report",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Merge current Criterion measurements into --baseline JSON",
    )
    parser.add_argument(
        "--render-svg",
        type=Path,
        default=None,
        help="Regenerate benchmark comparison SVG at the given path",
    )
    parser.add_argument(
        "--update-readme",
        type=Path,
        default=None,
        help="Update the Performance table in the given README.md path",
    )
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = load_criterion(args.criterion_dir) if args.criterion_dir.exists() else {}

    if not current and not (args.render_svg or args.update_readme):
        print(
            f"error: no Criterion estimates found in {args.criterion_dir}",
            file=sys.stderr,
        )
        return 1

    if current:
        report_md, regressions, summary_json = compare_and_report(
            current, baseline, args.threshold
        )
        print(report_md, end="")

        if args.github_summary:
            summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, "a", encoding="utf-8") as fh:
                    fh.write(report_md)

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
            )

        if args.update_baseline:
            baseline["benchmarks"].update(current)
            baseline["metadata"]["rustc"] = detect_rustc_version()
            baseline["metadata"]["commit"] = detect_git_sha()
            baseline["metadata"]["updated_at"] = dt.date.today().isoformat()
            args.baseline.write_text(
                json.dumps(baseline, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            print(f"Updated baseline: {args.baseline}")
    else:
        regressions = []

    if args.render_svg is not None:
        render_svg(baseline, args.render_svg)
        print(f"Rendered SVG: {args.render_svg}")

    if args.update_readme is not None:
        update_readme_table(baseline, args.update_readme)
        print(f"Updated README table: {args.update_readme}")

    if args.fail_on_regression and regressions:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
