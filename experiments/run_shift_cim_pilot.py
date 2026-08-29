"""P1 pilot: relative footprint + shift-min L1/L2 + CiM multi-copy accounting.

Grid: S ∈ {0,1,2} × metric ∈ {l1,l2} × codes ∈ {float,4bit}
Data: all HJ + MEArec SNR5, duration 60s, chronological 50/50.

  python -m Spatial.experiments.run_shift_cim_pilot --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.shift_match import (
    cim_row_accounting,
    search_ops_per_event,
    shift_min_distance,
)
from Spatial.algorithms.spatial_footprint import quantize_p2p
from Spatial.algorithms.two_stage_assign import (
    assign_nearest,
    calibrate_thresholds,
    fit_unit_centroids,
    stage1_candidates,
)
from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import (
    _chronological_split,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "shift_cim_pilot"

SHIFT_RADII = (0, 1, 2)
METRICS = ("l1", "l2")
CODE_MODES = ("float", "4bit")


def _prepare_p2p(p2p: np.ndarray, code_mode: str) -> tuple[np.ndarray, int]:
    if code_mode == "float":
        return np.asarray(p2p, dtype=np.float64), 32
    if code_mode == "4bit":
        # quantize then re-normalize to [0,1] for comparable float distances
        codes = quantize_p2p(p2p, n_bits=4, mode="per_spike_max")
        scale = float((1 << 4) - 1)
        return codes / scale, 4
    raise ValueError(f"Unknown code_mode: {code_mode}")


def _median_true_distance(
    p2p: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    *,
    shift_radius: int,
    metric: str,
) -> float:
    units = np.array(sorted(centroids.keys()), dtype=np.int64)
    if units.size == 0:
        return float("nan")
    mat = np.stack([centroids[int(u)] for u in units], axis=0)
    unit_to_j = {int(u): j for j, u in enumerate(units)}
    dists = shift_min_distance(
        p2p, mat, shift_radius=int(shift_radius), metric=metric  # type: ignore[arg-type]
    )
    vals = []
    for i, y in enumerate(labels):
        j = unit_to_j.get(int(y))
        if j is not None:
            vals.append(float(dists[i, j]))
    return float(np.median(vals)) if vals else float("nan")


def _gate_stats(
    com_tr: np.ndarray,
    p2p_tr: np.ndarray,
    lab_tr: np.ndarray,
    com_te: np.ndarray,
    p2p_te: np.ndarray,
    lab_te: np.ndarray,
    *,
    shift_radius: int,
    p2p_metric: str,
    percentile: float = 95.0,
) -> dict[str, float]:
    com_cents = fit_unit_centroids(com_tr, lab_tr)
    p2p_cents = fit_unit_centroids(p2p_tr, lab_tr)
    tau_com, tau_p2p = calibrate_thresholds(
        com_tr,
        p2p_tr,
        lab_tr,
        percentile=percentile,
        com_metric="l2",
        p2p_metric=p2p_metric,  # type: ignore[arg-type]
        shift_radius=int(shift_radius),
    )
    hits = []
    sizes = []
    for i, y in enumerate(lab_te):
        yi = int(y)
        if yi not in com_cents or yi not in p2p_cents:
            continue
        cands = stage1_candidates(
            com_te[i],
            p2p_te[i],
            com_cents,
            p2p_cents,
            tau_com,
            tau_p2p,
            com_metric="l2",
            p2p_metric=p2p_metric,  # type: ignore[arg-type]
            shift_radius=int(shift_radius),
        )
        sizes.append(len(cands))
        hits.append(yi in cands)
    return {
        "gate_recall": float(np.mean(hits)) if hits else float("nan"),
        "mean_candidates": float(np.mean(sizes)) if sizes else float("nan"),
        "tau_com": float(tau_com),
        "tau_p2p": float(tau_p2p),
        "n_gate_eval": float(len(hits)),
    }


def evaluate_prepared(
    prep: dict[str, np.ndarray],
    *,
    dataset_name: str,
    shift_radius: int,
    metric: str,
    code_mode: str,
) -> dict[str, Any]:
    tr, te = _chronological_split(prep["times"])
    com = prep["com"]
    p2p_all, n_bits = _prepare_p2p(prep["p2p"], code_mode)
    lab = prep["labels"]
    k = int(p2p_all.shape[1])

    p2p_cents = fit_unit_centroids(p2p_all[tr], lab[tr])
    pred = assign_nearest(
        p2p_all[te],
        p2p_cents,
        metric=metric,  # type: ignore[arg-type]
        shift_radius=int(shift_radius),
    )
    keep = np.array([int(y) in p2p_cents for y in lab[te]])
    argmin = (
        float(np.mean(pred[keep] == lab[te][keep].astype(pred.dtype)))
        if keep.any()
        else float("nan")
    )
    true_d = _median_true_distance(
        p2p_all[te],
        lab[te],
        p2p_cents,
        shift_radius=int(shift_radius),
        metric=metric,
    )
    gate = _gate_stats(
        com[tr],
        p2p_all[tr],
        lab[tr],
        com[te],
        p2p_all[te],
        lab[te],
        shift_radius=int(shift_radius),
        p2p_metric=metric,
    )
    n_units = len(p2p_cents)
    acc = cim_row_accounting(
        n_units=n_units, k=k, shift_radius=int(shift_radius), n_bits=n_bits
    )
    ops = search_ops_per_event(
        gate["mean_candidates"] if np.isfinite(gate["mean_candidates"]) else float(n_units),
        k,
        int(shift_radius),
    )
    return {
        "dataset": dataset_name,
        "shift_radius": int(shift_radius),
        "metric": metric,
        "code_mode": code_mode,
        "n_bits": int(n_bits),
        "k": k,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "n_units_train": int(n_units),
        "p2p_argmin_accuracy": argmin,
        "median_true_p2p_distance": true_d,
        "gate_recall": gate["gate_recall"],
        "mean_candidates": gate["mean_candidates"],
        "tau_com": gate["tau_com"],
        "tau_p2p": gate["tau_p2p"],
        "rows_per_unit": int(acc["rows_per_unit"]),
        "total_rows": int(acc["total_rows"]),
        "bits_per_row": int(acc["bits_per_row"]),
        "total_bits": int(acc["total_bits"]),
        "area_multiplier_vs_s0": float(acc["area_multiplier_vs_s0"]),
        "search_ops_per_event": float(ops),
    }


def evaluate_config(
    dataset,
    *,
    shift_radius: int,
    metric: str,
    code_mode: str,
    k_neighbors: int = 7,
) -> dict[str, Any]:
    prep = prepare_dataset_events(
        dataset, k_neighbors=k_neighbors, footprint_layout="relative"
    )
    return evaluate_prepared(
        prep,
        dataset_name=dataset.name,
        shift_radius=shift_radius,
        metric=metric,
        code_mode=code_mode,
    )


def _iter_datasets(family: str, duration: float, mearec_dir: Path):
    if family in ("hj", "all"):
        for entry in list_hybrid_janelia_scenes(ready_only=True):
            key = entry.get("scene_key") or entry.get("short_name")
            ds = load_hybrid_janelia(scene=key, duration_s=duration)
            yield "hj", ds
    if family in ("mearec", "mearec_snr5", "all"):
        for path in discover_npz(mearec_dir):
            if family in ("mearec_snr5", "all") and "snr5" not in path.name:
                continue
            ds = load_mearec_npz(path, duration_s=duration)
            yield "mearec", ds


def _condition(name: str) -> str:
    if "drift" in name:
        return "drift"
    if "static" in name:
        return "static"
    return "other"


def _mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    if not rows:
        return {}
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["n"] = float(len(rows))
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "p2p_argmin_accuracy",
        "median_true_p2p_distance",
        "gate_recall",
        "mean_candidates",
        "total_bits",
        "search_ops_per_event",
        "area_multiplier_vs_s0",
    )
    summary: dict[str, Any] = {"n_rows": len(rows), "grids": {}, "gates": {}}

    for family in sorted({r["family"] for r in rows}):
        summary["grids"][family] = {}
        for S in SHIFT_RADII:
            for metric in METRICS:
                for code in CODE_MODES:
                    subset = [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["shift_radius"] == S
                        and r["metric"] == metric
                        and r["code_mode"] == code
                    ]
                    key = f"S{S}_{metric}_{code}"
                    summary["grids"][family][key] = _mean_rows(subset, metric_keys)

    # Gate 1: HJ drift true-distance drop for L1 float S=1 or S=2 vs S=0
    drift = [r for r in rows if r["family"] == "hj" and r["condition"] == "drift"]
    d0 = [r for r in drift if r["shift_radius"] == 0 and r["metric"] == "l1" and r["code_mode"] == "float"]
    best_drop = 0.0
    best_s = 0
    if d0:
        base = float(np.nanmean([r["median_true_p2p_distance"] for r in d0]))
        for S in (1, 2):
            ds = [
                r
                for r in drift
                if r["shift_radius"] == S and r["metric"] == "l1" and r["code_mode"] == "float"
            ]
            if not ds:
                continue
            cur = float(np.nanmean([r["median_true_p2p_distance"] for r in ds]))
            drop = base - cur
            if drop > best_drop:
                best_drop = drop
                best_s = S
    gate1 = bool(best_drop > 0.0)
    summary["gates"]["hj_drift_l1_true_distance_drop"] = {
        "pass": gate1,
        "best_S": best_s,
        "drop": best_drop,
    }

    # Gate 2: family-mean best-S L1 4bit within 1pp of float L1
    gate2: dict[str, Any] = {}
    all_pass = True
    for family in sorted({r["family"] for r in rows}):
        float_best = -1.0
        bit_at_same = -1.0
        best_s_f = 0
        for S in SHIFT_RADII:
            fl = [
                r
                for r in rows
                if r["family"] == family
                and r["shift_radius"] == S
                and r["metric"] == "l1"
                and r["code_mode"] == "float"
            ]
            if not fl:
                continue
            acc = float(np.nanmean([r["p2p_argmin_accuracy"] for r in fl]))
            if acc > float_best:
                float_best = acc
                best_s_f = S
                bit_rows = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["shift_radius"] == S
                    and r["metric"] == "l1"
                    and r["code_mode"] == "4bit"
                ]
                bit_at_same = (
                    float(np.nanmean([r["p2p_argmin_accuracy"] for r in bit_rows]))
                    if bit_rows
                    else float("nan")
                )
        delta_pp = 100.0 * (float_best - bit_at_same) if np.isfinite(bit_at_same) else float("nan")
        ok = bool(np.isfinite(delta_pp) and delta_pp <= 1.0 + 1e-9)
        gate2[family] = {
            "pass": ok,
            "best_S_float": best_s_f,
            "float_argmin": float_best,
            "4bit_argmin_at_best_S": bit_at_same,
            "delta_pp": delta_pp,
        }
        all_pass = all_pass and ok
    summary["gates"]["l1_4bit_within_1pp_of_float"] = {"pass": all_pass, "per_family": gate2}

    # Accounting table S=0/1/2 at representative U,K (from first HJ float row means)
    acct = {}
    for S in SHIFT_RADII:
        sample = next(
            (
                r
                for r in rows
                if r["shift_radius"] == S and r["metric"] == "l1" and r["code_mode"] == "4bit"
            ),
            None,
        )
        if sample is None:
            continue
        acct[f"S{S}"] = {
            "rows_per_unit": sample["rows_per_unit"],
            "area_multiplier_vs_s0": sample["area_multiplier_vs_s0"],
            "bits_per_row_4bit": sample["bits_per_row"],
        }
    summary["gates"]["cim_accounting_published"] = {"pass": len(acct) == 3, "table": acct}

    summary["p1_pass"] = bool(
        gate1
        and all_pass
        and summary["gates"]["cim_accounting_published"]["pass"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift-min + CiM multi-copy pilot")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--family",
        choices=("hj", "mearec", "mearec_snr5", "all"),
        default="all",
    )
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for family, ds in _iter_datasets(args.family, args.duration, Path(args.mearec_npz_dir)):
        if not args.quiet:
            print(f"prepare {ds.name}", flush=True)
        prep = prepare_dataset_events(
            ds, k_neighbors=args.k, footprint_layout="relative"
        )
        for S in SHIFT_RADII:
            for metric in METRICS:
                for code in CODE_MODES:
                    if not args.quiet:
                        print(f"  S={S} {metric} {code}", flush=True)
                    row = evaluate_prepared(
                        prep,
                        dataset_name=ds.name,
                        shift_radius=S,
                        metric=metric,
                        code_mode=code,
                    )
                    row["family"] = family
                    row["condition"] = _condition(ds.name)
                    rows.append(row)

    csv_path = out_dir / "shift_cim_pilot.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = _summarize(rows)
    summary_path = out_dir / "shift_cim_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        print(json.dumps(summary["gates"], indent=2, sort_keys=True))
        print(f"Wrote {csv_path}")
        print(f"Wrote {summary_path}")
        print(f"P1 pass: {summary['p1_pass']}")


if __name__ == "__main__":
    main()
