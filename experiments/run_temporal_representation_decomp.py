"""Representation decomposition behind the frozen p99.9 D9 source.

Compares, on identical candidates and horizons:
  dyadic signed-5, dyadic float (same IDs), raw-prefix signed-5, raw-prefix
  float, prefix Fisher-M signed-5, plus full-window teacher / 64x5.

    python -m Spatial.experiments.run_temporal_representation_decomp --pilot
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.causal_temporal_baselines import CausalTemporalBaselines
from Spatial.algorithms.causal_temporal_sketch import CausalTemporalSketch
from Spatial.algorithms.temporal_representation_decomp import (
    assign_among_candidates,
    fisher_sample_ids,
    float_dyadic_features,
    float_templates,
    integer_templates,
)
from Spatial.experiments.causal_temporal_selection import (
    PILOT_DATASET_IDS,
)
from Spatial.experiments.run_causal_temporal_sketch import (
    _prediction_metrics,
    _with_spatial_fallback,
    discover_pilot_datasets,
    frozen_uniform_candidate_source,
    prepare_causal_events,
    stable_three_way_split,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "temporal_representation_decomp"
DELAYS = (8, 16, 48)
FEATURE_COUNTS = (8, 12, 16)
WAVEFORM_BITS = 64 * 5


def _payload_bits(name: str, delay: int, m: int, prefix_len: int) -> int:
    if name in ("dyadic_s5", "fisher_prefix_s5"):
        return 5 * int(m)
    if name == "prefix_s5":
        return 5 * int(prefix_len)
    if name == "signed_full64x5":
        return WAVEFORM_BITS
    if name in ("dyadic_float", "prefix_float"):
        return 64 * int(m if name == "dyadic_float" else prefix_len)  # diagnostic only
    if name == "full_teacher":
        return 64 * 64
    return 0


def _row(
    *,
    dataset: str,
    family: str,
    name: str,
    delay: int,
    m: int,
    prefix_len: int,
    pred: np.ndarray,
    labels: np.ndarray,
    candidates: list[np.ndarray],
    reads: np.ndarray,
    n_units: int,
) -> dict[str, Any]:
    metric = _prediction_metrics(pred, labels, candidates)
    counts = np.asarray([len(c) for c in candidates], dtype=np.int64)
    n = int(labels.size)
    actual_reads = int(reads.sum())
    bits_per = _payload_bits(name, delay, m, prefix_len)
    actual_bits = actual_reads * bits_per
    global_bits = n * n_units * WAVEFORM_BITS
    return {
        "dataset": dataset,
        "family": family,
        "representation": name,
        "delay_samples": int(delay),
        "feature_count": int(m),
        "prefix_length": int(prefix_len),
        "n_test": n,
        "n_units": int(n_units),
        "overall_accuracy": float(metric["overall_accuracy"]),
        "ambiguous_accuracy": float(metric["ambiguous_accuracy"]),
        "worst_unit_accuracy": float(metric["worst_unit_accuracy"]),
        "template_reads": actual_reads,
        "template_bits_per_unit": int(bits_per),
        "actual_template_bits": int(actual_bits),
        "global_64x5_bits": int(global_bits),
        "traffic_x_vs_full64x5": float(global_bits / max(actual_bits, 1)),
        "frac_Cgt1": float(np.mean(counts > 1)),
    }


def run_recording(dataset, family: str) -> list[dict[str, Any]]:
    prepared = prepare_causal_events(dataset)
    fit, cal, test = stable_three_way_split(prepared["times"])
    y = prepared["labels"]
    wave = prepared["raw_waveforms"]
    source = frozen_uniform_candidate_source(
        prepared["descriptor"][fit], y[fit],
        prepared["descriptor"][cal], y[cal],
        prepared["descriptor"][test],
    )
    candidates = [np.asarray(row).copy() for row in source["candidates"]]
    y_test = y[test]
    wave_test = wave[test]
    wave_fit = wave[fit]
    y_fit = y[fit]
    units_fit = np.unique(y_fit)
    n_units = int(len(source["model"].units_))
    rows: list[dict[str, Any]] = []

    baselines = CausalTemporalBaselines().fit(wave_fit, y_fit)
    for name in ("fit_scale_float64", "signed_full64x5"):
        initial, diag = baselines.assign(wave_test, candidates, name)
        pred = _with_spatial_fallback(initial, candidates, source)
        reads = np.asarray(diag["logical_candidate_template_reads"], dtype=np.int64)
        rows.append(
            _row(
                dataset=dataset.name, family=family,
                name="full_teacher" if name == "fit_scale_float64" else "signed_full64x5",
                delay=48, m=64, prefix_len=64,
                pred=pred, labels=y_test, candidates=candidates,
                reads=reads, n_units=n_units,
            )
        )

    eligible = units_fit[np.array([np.count_nonzero(y_fit == u) >= 10 for u in units_fit])]
    for delay in DELAYS:
        for m in FEATURE_COUNTS:
            sketch = CausalTemporalSketch(delay_samples=delay, n_features=m).fit(wave_fit, y_fit)
            prefix_len = sketch.prefix_length
            scale = sketch.scale_
            # 1. signed-5 dyadic
            init, diag = sketch.assign_candidates(wave_test, candidates)
            pred = _with_spatial_fallback(init, candidates, source)
            reads = np.asarray(diag["logical_candidate_template_reads"], dtype=np.int64)
            rows.append(_row(
                dataset=dataset.name, family=family, name="dyadic_s5",
                delay=delay, m=m, prefix_len=prefix_len,
                pred=pred, labels=y_test, candidates=candidates,
                reads=reads, n_units=n_units,
            ))
            # 2. float dyadic, same IDs
            scaled_fit = np.clip(wave_fit[:, :prefix_len] / scale, -1.0, 1.0)
            scaled_test = np.clip(wave_test[:, :prefix_len] / scale, -1.0, 1.0)
            ids = np.asarray(sketch.selected_filter_ids_, dtype=np.int64)
            f_fit = float_dyadic_features(scaled_fit, ids)
            f_test = float_dyadic_features(scaled_test, ids)
            tmpl = float_templates(f_fit, y_fit, sketch.units_)
            pred, reads = assign_among_candidates(f_test, tmpl, sketch.units_, candidates)
            pred = _with_spatial_fallback(pred, candidates, source)
            rows.append(_row(
                dataset=dataset.name, family=family, name="dyadic_float",
                delay=delay, m=m, prefix_len=prefix_len,
                pred=pred, labels=y_test, candidates=candidates,
                reads=reads, n_units=n_units,
            ))
            # 3. raw-prefix signed-5
            codes_fit = sketch.quantize_prefix(wave_fit)
            codes_test = sketch.quantize_prefix(wave_test)
            tmpl_i = integer_templates(codes_fit, y_fit, sketch.units_)
            pred, reads = assign_among_candidates(codes_test, tmpl_i, sketch.units_, candidates)
            pred = _with_spatial_fallback(pred, candidates, source)
            rows.append(_row(
                dataset=dataset.name, family=family, name="prefix_s5",
                delay=delay, m=prefix_len, prefix_len=prefix_len,
                pred=pred, labels=y_test, candidates=candidates,
                reads=reads, n_units=n_units,
            ))
            # 4. raw-prefix float
            init = sketch.same_horizon_float_assign(wave_test, candidates)
            pred = _with_spatial_fallback(init, candidates, source)
            reads = np.array([len(c) if len(c) > 1 else 0 for c in candidates], dtype=np.int64)
            rows.append(_row(
                dataset=dataset.name, family=family, name="prefix_float",
                delay=delay, m=prefix_len, prefix_len=prefix_len,
                pred=pred, labels=y_test, candidates=candidates,
                reads=reads, n_units=n_units,
            ))
            # 5. Fisher-M taps on the prefix, signed-5
            if eligible.size >= 2:
                taps = fisher_sample_ids(codes_fit, y_fit, m, eligible)
                tmpl_f = integer_templates(codes_fit[:, taps], y_fit, sketch.units_)
                pred, reads = assign_among_candidates(
                    codes_test[:, taps], tmpl_f, sketch.units_, candidates
                )
                pred = _with_spatial_fallback(pred, candidates, source)
                rows.append(_row(
                    dataset=dataset.name, family=family, name="fisher_prefix_s5",
                    delay=delay, m=m, prefix_len=prefix_len,
                    pred=pred, labels=y_test, candidates=candidates,
                    reads=reads, n_units=n_units,
                ))
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {(r["representation"], r["delay_samples"], r["feature_count"]) for r in rows}
    out = []
    for name, delay, m in sorted(keys):
        for family in ("hj", "mearec", "all"):
            sub = [
                r for r in rows
                if r["representation"] == name
                and r["delay_samples"] == delay
                and r["feature_count"] == m
                and (family == "all" or r["family"] == family)
            ]
            if not sub:
                continue
            n = sum(int(r["n_test"]) for r in sub)
            acc = sum(float(r["overall_accuracy"]) * int(r["n_test"]) for r in sub) / max(n, 1)
            amb = sum(float(r["ambiguous_accuracy"]) * int(r["n_test"]) for r in sub) / max(n, 1)
            out.append({
                "representation": name,
                "delay_samples": delay,
                "feature_count": m,
                "family": family,
                "overall_accuracy": acc,
                "ambiguous_accuracy": amb,
                "n_recordings": len(sub),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot (locked four-recording development set)")
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s")
    discovered = discover_pilot_datasets(duration_s=60.0, mearec_npz_dir=Path(args.mearec_dir))
    rows: list[dict[str, Any]] = []
    for family in ("hj", "mearec"):
        wanted = set(PILOT_DATASET_IDS[family])
        for record in discovered[family]:
            if record.dataset_id not in wanted:
                continue
            dataset = record.load_dataset()
            rec_rows = run_recording(dataset, family)
            del dataset
            rows.extend(rec_rows)
            if not args.quiet:
                for r in rec_rows:
                    if r["representation"] in (
                        "full_teacher", "signed_full64x5", "dyadic_s5",
                        "dyadic_float", "prefix_float", "prefix_s5",
                        "fisher_prefix_s5",
                    ) and r["delay_samples"] in (8, 16, 48) and (
                        r["feature_count"] in (16, 64, r["prefix_length"])
                    ):
                        print(
                            f"{r['dataset']:<42} {r['representation']:<18} "
                            f"d={r['delay_samples']:>2} M={r['feature_count']:>2}  "
                            f"acc={r['overall_accuracy']:.4f} amb={r['ambiguous_accuracy']:.4f}"
                        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "temporal_representation_decomp_pilot.csv"
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    agg = _aggregate(rows)
    write_json(out_dir / "temporal_representation_decomp_summary.json", {"aggregate": agg, "n_rows": len(rows)})
    print(f"Saved {csv_path}")
    print("event-weighted family means (selected):")
    for a in agg:
        if a["family"] != "all":
            continue
        if a["representation"] in ("full_teacher", "signed_full64x5") or (
            a["feature_count"] in (16, a.get("feature_count"))
            and a["delay_samples"] in (8, 16, 48)
        ):
            print(
                f"  {a['representation']:<18} d={a['delay_samples']:>2} M={a['feature_count']:>2}  "
                f"acc={a['overall_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
