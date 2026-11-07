#!/usr/bin/env python3
"""Extended validation for the RA-CLHPS manuscript.

This script adds independent-training replications, adaptation-gain sensitivity,
out-of-distribution demand/capability stress tests, objective-weight sensitivity,
and larger scale tests.  It imports the disclosed simulator so that the model,
policies, and exact action-trace checks are shared with the main experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import adaptive_lazy_seru_experiment as core


QUALITY = ("makespan", "mean_flow", "worker_cv2", "seru_cv2", "skill_gain", "coverage", "composite")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs)


def ci_half(xs: Sequence[float]) -> float:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def inverse_softplus(x: float) -> float:
    if x <= 1e-8:
        return -10.0
    return math.log(math.expm1(x))


def scale_adaptation(params: np.ndarray, factor: float) -> np.ndarray:
    out = params.copy()
    gains = np.log1p(np.exp(np.clip(out[10:14], -10.0, 6.0))) * factor
    out[10:14] = np.array([inverse_softplus(float(x)) for x in gains])
    return out


def transform_instance(
    instance: core.Instance,
    intensity: float = 1.0,
    skill_band: Tuple[float, float] = (0.30, 0.80),
) -> core.Instance:
    first = instance.orders[0].arrival
    orders = tuple(
        core.OrderSpec(first + (o.arrival - first) / intensity, o.tasks)
        for o in instance.orders
    )
    lo, hi = skill_band
    normalized = np.clip((instance.proficiency - 0.30) / 0.50, 0.0, 1.0)
    proficiency = lo + normalized * (hi - lo)
    for m in range(core.SERUS):
        ws = slice(m * core.WORKERS_PER_SERU, (m + 1) * core.WORKERS_PER_SERU)
        for k in range(core.TASK_TYPES):
            if proficiency[ws, k].max() < 0.34:
                proficiency[m * core.WORKERS_PER_SERU, k] = 0.34
    return core.Instance(instance.seed, instance.regime, orders, proficiency)


def run_with_trace(instance: core.Instance, policy: core.Policy) -> Tuple[Dict[str, float], list]:
    sim = core.Simulator(instance, policy)
    metrics = sim.run()
    return metrics, sim.action_trace


def assert_equal(reference: Dict[str, float], candidate: Dict[str, float], ref_trace: list, cand_trace: list) -> None:
    if ref_trace != cand_trace:
        raise AssertionError("certified lazy evaluation changed an action trace")
    for metric in QUALITY:
        if not math.isclose(reference[metric], candidate[metric], rel_tol=0.0, abs_tol=1e-10):
            raise AssertionError(f"certified lazy evaluation changed {metric}")


def adaptation_sensitivity(out: Path, adaptive: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    factors = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0)
    regimes = ("steady", "surge", "recovery")
    for factor in factors:
        params = scale_adaptation(adaptive, factor)
        metrics = []
        for i in range(60):
            instance = core.make_instance(50000 + i, regimes[i % 3])
            metrics.append(core.Simulator(instance, core.ScoringPolicy(params, True, f"gain-{factor}" )).run())
        row: Dict[str, object] = {"gain_factor": factor, "n": len(metrics)}
        for key in QUALITY:
            values = [float(x[key]) for x in metrics]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_ci_half"] = ci_half(values)
        rows.append(row)
    write_csv(out / "gain_sensitivity.csv", rows, list(rows[0].keys()))
    return rows


def scenario_stress(out: Path, fixed: np.ndarray, adaptive: np.ndarray) -> List[Dict[str, object]]:
    raw: List[Dict[str, object]] = []
    intensities = (0.75, 1.00, 1.25, 1.50)
    bands = {"low": (0.30, 0.55), "baseline": (0.30, 0.80), "high": (0.55, 0.80)}
    regimes = ("steady", "surge", "recovery")
    for intensity in intensities:
        for band_name, band in bands.items():
            for j in range(15):
                base_instance = core.make_instance(80000 + j, regimes[j % 3])
                instance = transform_instance(base_instance, intensity, band)
                fixed_m, _ = run_with_trace(instance, core.ScoringPolicy(fixed, False, "Fixed-HPS"))
                full_m, full_t = run_with_trace(instance, core.ScoringPolicy(adaptive, True, "Adaptive-HPS"))
                lazy_m, lazy_t = run_with_trace(
                    instance, core.CertifiedLazyScoringPolicy(adaptive, True, "RA-CLHPS", bound_order=False)
                )
                assert_equal(full_m, lazy_m, full_t, lazy_t)
                for name, metrics in (("Fixed-HPS", fixed_m), ("Adaptive-HPS", full_m), ("RA-CLHPS", lazy_m)):
                    raw.append({
                        "intensity": intensity,
                        "skill_band": band_name,
                        "seed": instance.seed,
                        "method": name,
                        **metrics,
                    })
    raw_fields = ["intensity", "skill_band", "seed", "method", *QUALITY,
                  "component_evals", "full_component_evals", "pruning_rate", "decision_count",
                  "wall_time_ms", "projection_calls", "projection_task_scans"]
    write_csv(out / "stress_raw.csv", raw, raw_fields)

    summary: List[Dict[str, object]] = []
    for intensity in intensities:
        for band_name in bands:
            for method in ("Fixed-HPS", "Adaptive-HPS", "RA-CLHPS"):
                subset = [r for r in raw if r["intensity"] == intensity and r["skill_band"] == band_name and r["method"] == method]
                row: Dict[str, object] = {"intensity": intensity, "skill_band": band_name, "method": method, "n": len(subset)}
                for key in (*QUALITY, "projection_calls", "projection_task_scans", "wall_time_ms"):
                    vals = [float(r[key]) for r in subset]
                    row[f"{key}_mean"] = mean(vals)
                summary.append(row)
    write_csv(out / "stress_summary.csv", summary, list(summary[0].keys()))
    return summary


def objective_weight_sensitivity(out: Path, main_raw: Path) -> List[Dict[str, object]]:
    rows = list(csv.DictReader(main_raw.open(encoding="utf-8")))
    fixed = {(r["seed"], r["regime"]): r for r in rows if r["method"] == "Fixed-HPS"}
    adaptive = {(r["seed"], r["regime"]): r for r in rows if r["method"] == "Adaptive-HPS"}
    result: List[Dict[str, object]] = []
    for w_gain in (0.0, 1.0, 2.0, 3.0, 4.0):
        for w_coverage in (0.0, 0.15, 0.30, 0.45, 0.60):
            diffs = []
            for key, f in fixed.items():
                a = adaptive[key]
                def score(r: Dict[str, str]) -> float:
                    return (float(r["makespan"]) / 600.0 + float(r["mean_flow"]) / 100.0
                            + 0.35 * float(r["worker_cv2"]) + 0.25 * float(r["seru_cv2"])
                            - w_gain * float(r["skill_gain"]) - w_coverage * float(r["coverage"]))
                diffs.append(score(a) - score(f))
            result.append({
                "skill_gain_weight": w_gain,
                "coverage_weight": w_coverage,
                "adaptive_minus_fixed_mean": mean(diffs),
                "adaptive_better_fraction": mean([d < 0 for d in diffs]),
                "two_sided_sign_test_p": core.sign_test_p(diffs),
                "n": len(diffs),
            })
    write_csv(out / "objective_weight_sensitivity.csv", result, list(result[0].keys()))
    return result


def extended_scale(out: Path, adaptive: np.ndarray) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    regimes = ("steady", "surge", "recovery")
    for order_count in (40, 60, 100, 200, 400):
        replications = 15 if order_count <= 100 else 10
        for j in range(replications):
            instance = core.make_instance(100000 + order_count * 100 + j, regimes[j % 3], orders=order_count)
            full_m, full_t = run_with_trace(instance, core.ScoringPolicy(adaptive, True, "Adaptive-HPS"))
            lazy_m, lazy_t = run_with_trace(
                instance, core.CertifiedLazyScoringPolicy(adaptive, True, "RA-CLHPS", bound_order=False)
            )
            assert_equal(full_m, lazy_m, full_t, lazy_t)
            for name, metrics in (("Adaptive-HPS", full_m), ("RA-CLHPS", lazy_m)):
                rows.append({"orders": order_count, "seed": instance.seed, "method": name, **metrics})
    fields = ["orders", "seed", "method", *QUALITY, "component_evals", "full_component_evals",
              "pruning_rate", "decision_count", "wall_time_ms", "projection_calls", "projection_task_scans"]
    write_csv(out / "extended_scale_raw.csv", rows, fields)
    summary: List[Dict[str, object]] = []
    for order_count in (40, 60, 100, 200, 400):
        for method in ("Adaptive-HPS", "RA-CLHPS"):
            subset = [r for r in rows if r["orders"] == order_count and r["method"] == method]
            summary.append({
                "orders": order_count,
                "method": method,
                "n": len(subset),
                "component_evals_mean": mean([float(r["component_evals"]) for r in subset]),
                "projection_calls_mean": mean([float(r["projection_calls"]) for r in subset]),
                "projection_task_scans_mean": mean([float(r["projection_task_scans"]) for r in subset]),
                "wall_time_ms_mean": mean([float(r["wall_time_ms"]) for r in subset]),
                "composite_mean": mean([float(r["composite"]) for r in subset]),
            })
    write_csv(out / "extended_scale_summary.csv", summary, list(summary[0].keys()))
    return summary


def independent_training(out: Path, replications: int = 5) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    parameters: List[Dict[str, object]] = []
    regimes = ("steady", "surge", "recovery")
    validation_instances = [core.make_instance(110000 + i, regimes[i % 3]) for i in range(15)]
    test_instances = [core.make_instance(120000 + i, regimes[i % 3]) for i in range(30)]
    for rep in range(replications):
        fixed, fixed_hist = core.train_cem(False, 31000 + 2 * rep, 8, 20)
        adaptive, adaptive_hist = core.train_cem(True, 31001 + 2 * rep, 8, 20, fixed_base=fixed)
        validation_fixed = mean([
            core.Simulator(x, core.ScoringPolicy(fixed, False, "Fixed-HPS")).run()["composite"]
            for x in validation_instances
        ])
        validation_adaptive = mean([
            core.Simulator(x, core.ScoringPolicy(adaptive, True, "Adaptive-HPS")).run()["composite"]
            for x in validation_instances
        ])
        selected_policy = "Adaptive-HPS" if validation_adaptive <= validation_fixed else "Fixed-HPS"
        fixed_values = []
        adaptive_values = []
        for instance in test_instances:
            fixed_values.append(core.Simulator(instance, core.ScoringPolicy(fixed, False, "Fixed-HPS")).run())
            adaptive_values.append(core.Simulator(instance, core.ScoringPolicy(adaptive, True, "Adaptive-HPS")).run())
        row: Dict[str, object] = {
            "training_replication": rep + 1,
            "fixed_training_seed": 31000 + 2 * rep,
            "adaptive_training_seed": 31001 + 2 * rep,
            "n_test": len(test_instances),
            "fixed_training_best": fixed_hist[-1],
            "adaptive_training_best": adaptive_hist[-1],
            "validation_fixed_composite": validation_fixed,
            "validation_adaptive_composite": validation_adaptive,
            "validation_adaptive_minus_fixed": validation_adaptive - validation_fixed,
            "validation_selected_policy": selected_policy,
        }
        for key in QUALITY:
            f = [float(x[key]) for x in fixed_values]
            a = [float(x[key]) for x in adaptive_values]
            row[f"fixed_{key}_mean"] = mean(f)
            row[f"adaptive_{key}_mean"] = mean(a)
            row[f"adaptive_minus_fixed_{key}"] = mean([x - y for x, y in zip(a, f)])
            row[f"selected_minus_fixed_{key}"] = (
                row[f"adaptive_minus_fixed_{key}"] if selected_policy == "Adaptive-HPS" else 0.0
            )
        rows.append(row)
        parameters.append({
            "training_replication": rep + 1,
            "fixed_parameters": fixed.tolist(),
            "adaptive_parameters": adaptive.tolist(),
        })
    write_csv(out / "independent_training.csv", rows, list(rows[0].keys()))
    (out / "independent_training_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    return rows


def color(value: float, low: float, high: float) -> str:
    if high <= low:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - low) / (high - low)))
    if t < 0.5:
        u = t / 0.5
        rgb = (int(49 + u * (255 - 49)), int(130 + u * (255 - 130)), int(189 + u * (255 - 189)))
    else:
        u = (t - 0.5) / 0.5
        rgb = (int(255 + u * (215 - 255)), int(255 + u * (48 - 255)), int(255 + u * (39 - 255)))
    return "#%02x%02x%02x" % rgb


def heatmap_svg(path: Path, rows: Sequence[Dict[str, object]], x_key: str, y_key: str, value_key: str,
                title: str, x_label: str, y_label: str) -> None:
    xs = sorted({r[x_key] for r in rows}, key=float)
    ys = sorted({r[y_key] for r in rows}, key=str)
    values = [float(r[value_key]) for r in rows]
    bound = max(abs(min(values)), abs(max(values)), 1e-9)
    cw, ch = 115, 62
    left, top = 150, 65
    width, height = left + cw * len(xs) + 40, top + ch * len(ys) + 90
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222}.t{font-size:19px;font-weight:bold}.a{font-size:14px}.v{font-size:13px;font-weight:bold}</style>',
             f'<text x="{width/2}" y="28" text-anchor="middle" class="t">{title}</text>']
    lookup = {(r[x_key], r[y_key]): float(r[value_key]) for r in rows}
    for yi, y in enumerate(ys):
        yy = top + yi * ch
        parts.append(f'<text x="{left-10}" y="{yy+ch/2+5}" text-anchor="end" class="a">{y}</text>')
        for xi, x in enumerate(xs):
            xx = left + xi * cw
            v = lookup[(x, y)]
            fill = color(v, -bound, bound)
            parts.append(f'<rect x="{xx}" y="{yy}" width="{cw}" height="{ch}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{xx+cw/2}" y="{yy+ch/2+5}" text-anchor="middle" class="v">{v:+.4f}</text>')
    for xi, x in enumerate(xs):
        parts.append(f'<text x="{left+xi*cw+cw/2}" y="{top+len(ys)*ch+23}" text-anchor="middle" class="a">{x}</text>')
    parts.append(f'<text x="{left+cw*len(xs)/2}" y="{height-15}" text-anchor="middle" class="a">{x_label}</text>')
    parts.append(f'<text x="18" y="{top+ch*len(ys)/2}" transform="rotate(-90 18 {top+ch*len(ys)/2})" text-anchor="middle" class="a">{y_label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def line_svg(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    xs = [float(r["gain_factor"]) for r in rows]
    series = (("composite_mean", "Composite J", "#0072B2"),
              ("mean_flow_mean", "Mean flow / 40", "#D55E00"),
              ("skill_gain_mean", "Skill gain x 12", "#009E73"))
    transformed = {
        "composite_mean": [float(r["composite_mean"]) for r in rows],
        "mean_flow_mean": [float(r["mean_flow_mean"]) / 40.0 for r in rows],
        "skill_gain_mean": [float(r["skill_gain_mean"]) * 12.0 for r in rows],
    }
    width, height, left, top, pw, ph = 760, 430, 80, 55, 620, 290
    allv = [v for vals in transformed.values() for v in vals]
    lo, hi = min(allv) * 0.96, max(allv) * 1.04
    def px(x: float) -> float: return left + (x - min(xs)) / (max(xs) - min(xs)) * pw
    def py(y: float) -> float: return top + (hi - y) / (hi - lo) * ph
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222}.t{font-size:20px;font-weight:bold}.a{font-size:14px}</style>',
             f'<text x="{width/2}" y="27" text-anchor="middle" class="t">Sensitivity to adaptation-gain scaling</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
             f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>']
    for key, label, stroke in series:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, transformed[key]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="3"/>')
        for x, y in zip(xs, transformed[key]):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{stroke}"/>')
    for x in xs:
        parts.append(f'<text x="{px(x):.1f}" y="{top+ph+23}" text-anchor="middle" class="a">{x:g}</text>')
    for i, (_, label, stroke) in enumerate(series):
        parts.append(f'<line x1="{left+100+i*180}" y1="{height-42}" x2="{left+125+i*180}" y2="{height-42}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(f'<text x="{left+132+i*180}" y="{height-37}" class="a">{label}</text>')
    parts.append(f'<text x="{left+pw/2}" y="{height-62}" text-anchor="middle" class="a">Adaptation-gain multiplier</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--main-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("extended_results"))
    parser.add_argument("--training-replications", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    saved = json.loads(args.parameter_file.read_text(encoding="utf-8"))
    fixed = np.asarray(saved["fixed_parameters"], dtype=float)
    adaptive = np.asarray(saved["adaptive_parameters"], dtype=float)

    gains = adaptation_sensitivity(args.output, adaptive)
    stress = scenario_stress(args.output, fixed, adaptive)
    preferences = objective_weight_sensitivity(args.output, args.main_raw)
    scales = extended_scale(args.output, adaptive)
    training = independent_training(args.output, args.training_replications)

    stress_delta = []
    for intensity in sorted({r["intensity"] for r in stress}, key=float):
        for band in ("low", "baseline", "high"):
            subset = [r for r in stress if r["intensity"] == intensity and r["skill_band"] == band]
            f = next(r for r in subset if r["method"] == "Fixed-HPS")
            a = next(r for r in subset if r["method"] == "Adaptive-HPS")
            stress_delta.append({"intensity": intensity, "skill_band": band,
                                 "delta": float(a["composite_mean"]) - float(f["composite_mean"])})
    heatmap_svg(args.output / "stress_heatmap.svg", stress_delta, "intensity", "skill_band", "delta",
                "Adaptive minus fixed composite score", "Arrival-intensity multiplier", "Initial capability band")
    heatmap_svg(args.output / "objective_weight_heatmap.svg", preferences,
                "skill_gain_weight", "coverage_weight", "adaptive_minus_fixed_mean",
                "Preference sensitivity: adaptive minus fixed", "Skill-gain weight", "Coverage weight")
    line_svg(args.output / "gain_sensitivity.svg", gains)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.time() - started,
        "training_replications": args.training_replications,
        "stress_scenarios": 12,
        "stress_replications_per_scenario": 15,
        "gain_factors": [float(r["gain_factor"]) for r in gains],
        "extended_order_sizes": sorted({int(r["orders"]) for r in scales}),
        "exact_trace_checks": True,
        "notes": [
            "All stress and scale comparisons use paired instances.",
            "No policy is retrained for the stress or scale transfer tests.",
            "Every full-versus-lazy comparison asserts identical action traces and terminal quality metrics.",
            "Wall-clock values are machine-specific; component and task-scan counts are deterministic.",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
