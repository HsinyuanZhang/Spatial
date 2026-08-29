"""Oracle A/B: KNN vs relative-offset P2P footprint (P0).

Compares distance-ranked KNN slots against a fixed primary-axis relative patch
on chronological 50/50 splits. Does not flip defaults in other experiments.

  python -m Spatial.experiments.run_relative_footprint_pilot --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

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
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "relative_footprint_pilot"


def _median_true_p2p_l1(
    p2p: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> float:
    dists = []
    for x, y in zip(p2p, labels):
        c = centroids.get(int(y))
        if c is None:
            continue
        dists.append(float(np.sum(np.abs(x - c))))
    return float(np.median(dists)) if dists else float("nan")


def _gate_stats(
    com_tr: np.ndarray,
    p2p_tr: np.ndarray,
    lab_tr: np.ndarray,
    com_te: np.ndarray,
    p2p_te: np.ndarray,
    lab_te: np.ndarray,
    percentile: float = 95.0,
) -> dict[str, float]:
    com_cents = fit_unit_centroids(com_tr, lab_tr)
    p2p_cents = fit_unit_centroids(p2p_tr, lab_tr)
    tau_com, tau_p2p = calibrate_thresholds(
        com_tr, p2p_tr, lab_tr, percentile=percentile, com_metric="l2", p2p_metric="l1"
    )
    hits = []
    sizes = []
    known = 0
    for i, y in enumerate(lab_te):
        yi = int(y)
        if yi not in com_cents or yi not in p2p_cents:
            continue
        known += 1
        cands = stage1_candidates(
            com_te[i],
            p2p_te[i],
            com_cents,
            p2p_cents,
            tau_com,
            tau_p2p,
            com_metric="l2",
            p2p_metric="l1",
        )
        sizes.append(len(cands))
        hits.append(yi in cands)
    return {
        "gate_recall": float(np.mean(hits)) if hits else float("nan"),
        "mean_candidates": float(np.mean(sizes)) if sizes else float("nan"),
        "tau_com": float(tau_com),
        "tau_p2p": float(tau_p2p),
        "n_gate_eval": float(known),
    }


def evaluate_recording(dataset, *, layout: str, k_neighbors: int = 7) -> dict[str, Any]:
    prep = prepare_dataset_events(
        dataset, k_neighbors=k_neighbors, footprint_layout=layout
    )
    tr, te = _chronological_split(prep["times"])
    com, p2p, lab = prep["com"], prep["p2p"], prep["labels"]

    p2p_cents = fit_unit_centroids(p2p[tr], lab[tr])
    pred = assign_nearest(p2p[te], p2p_cents, metric="l1")
    keep = np.array([int(y) in p2p_cents for y in lab[te]])
    if keep.any():
        p2p_argmin = float(np.mean(pred[keep] == lab[te][keep].astype(pred.dtype)))
    else:
        p2p_argmin = float("nan")

    gate = _gate_stats(com[tr], p2p[tr], lab[tr], com[te], p2p[te], lab[te])
    true_l1 = _median_true_p2p_l1(p2p[te], lab[te], p2p_cents)

    return {
        "dataset": dataset.name,
        "footprint_layout": layout,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "n_units_train": int(len(p2p_cents)),
        "p2p_argmin_accuracy": p2p_argmin,
        "median_true_p2p_l1": true_l1,
        **gate,
    }


def _iter_datasets(family: str, duration: float, mearec_dir: Path):
    if family in ("hj", "all"):
        for entry in list_hybrid_janelia_scenes(ready_only=True):
            key = entry.get("scene_key") or entry.get("short_name")
            ds = load_hybrid_janelia(scene=key, duration_s=duration)
            yield "hj", ds
    if family in ("mearec", "all", "mearec_snr5"):
        for path in discover_npz(mearec_dir):
            if family == "mearec_snr5" and "snr5" not in path.name:
                continue
            if family == "all" and "snr5" not in path.name:
                # Default "all" for this pilot = HJ + MEArec SNR5 only.
                continue
            ds = load_mearec_npz(path, duration_s=duration)
            yield "mearec", ds


def _family_means(rows: list[dict[str, Any]], family: str, layout: str) -> dict[str, float]:
    subset = [r for r in rows if r["family"] == family and r["footprint_layout"] == layout]
    if not subset:
        return {}
    keys = (
        "p2p_argmin_accuracy",
        "median_true_p2p_l1",
        "gate_recall",
        "mean_candidates",
    )
    return {k: float(np.nanmean([r[k] for r in subset])) for k in keys} | {"n": float(len(subset))}


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN vs relative-offset footprint pilot")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--family",
        choices=("hj", "mearec", "mearec_snr5", "all"),
        default="all",
        help="all = HJ + MEArec SNR5 (pilot default)",
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
        for layout in ("knn", "relative"):
            if not args.quiet:
                print(f"{ds.name} layout={layout}", flush=True)
            row = evaluate_recording(ds, layout=layout, k_neighbors=args.k)
            row["family"] = family
            rows.append(row)

    csv_path = out_dir / "relative_footprint_pilot.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary: dict[str, Any] = {"n_rows": len(rows), "by_family_layout": {}}
    for family in sorted({r["family"] for r in rows}):
        summary["by_family_layout"][family] = {
            layout: _family_means(rows, family, layout) for layout in ("knn", "relative")
        }

    # P0 success checks
    gates: dict[str, Any] = {}
    for family in ("hj", "mearec"):
        knn = summary["by_family_layout"].get(family, {}).get("knn", {})
        rel = summary["by_family_layout"].get(family, {}).get("relative", {})
        if not knn or not rel:
            continue
        argmin_ok = rel["p2p_argmin_accuracy"] + 1e-12 >= knn["p2p_argmin_accuracy"]
        l1_ok = rel["median_true_p2p_l1"] <= knn["median_true_p2p_l1"] + 1e-12
        gates[family] = {
            "argmin_relative_ge_knn": bool(argmin_ok),
            "true_l1_relative_le_knn": bool(l1_ok),
            "knn": knn,
            "relative": rel,
        }
    hj_pass = bool(gates.get("hj", {}).get("argmin_relative_ge_knn")) and bool(
        gates.get("hj", {}).get("true_l1_relative_le_knn")
    )
    summary["gates"] = gates
    summary["p0_hj_pass"] = hj_pass

    summary_path = out_dir / "relative_footprint_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {csv_path}")
        print(f"Wrote {summary_path}")
        print(f"P0 HJ pass: {hj_pass}")


if __name__ == "__main__":
    main()
