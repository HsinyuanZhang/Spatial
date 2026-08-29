"""Shift-CONV / cosine vs L1 classification with 4/5/6-bit sweep.

- conv: max_s query·shift(template,s)  (raw MAC)
- cosine: same after L2-normalize (normalized correlation / MAC-friendly)
- l1: min_s ||query-shift(template,s)||_1

Grid: method ∈ {conv, cosine, l1} × bits ∈ {4,5,6,float} × S ∈ {0,1,2}
Data: all HJ + MEArec SNR5, 60s, chronological 50/50, relative K=7.

  python -m Spatial.experiments.run_shift_conv_bit_sweep --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.shift_match import (
    assign_shift_max_dot,
    cim_row_accounting,
    search_ops_per_event,
    shift_max_dot,
    shift_min_distance,
)
from Spatial.algorithms.spatial_footprint import quantize_p2p
from Spatial.algorithms.two_stage_assign import (
    assign_nearest,
    calibrate_thresholds,
    fit_unit_centroids,
    stage1_candidates,
    vector_distance,
)
from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import (
    _chronological_split,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "shift_conv_bit_sweep"

SHIFT_RADII = (0, 1, 2)
METHODS = ("conv", "cosine", "l1")
BIT_MODES = ("float", "4", "5", "6")


def _prepare_features(p2p: np.ndarray, bit_mode: str) -> tuple[np.ndarray, int]:
    """Return features in [0,1] and nominal bit width for accounting."""
    p2p = np.asarray(p2p, dtype=np.float64)
    if bit_mode == "float":
        return p2p, 32
    n_bits = int(bit_mode)
    codes = quantize_p2p(p2p, n_bits=n_bits, mode="per_spike_max")
    scale = float((1 << n_bits) - 1)
    return codes / scale, n_bits


def _uses_cosine(method: str) -> bool:
    return method == "cosine"


def _uses_conv_family(method: str) -> bool:
    return method in ("conv", "cosine")


def _condition(name: str) -> str:
    if "drift" in name:
        return "drift"
    if "static" in name:
        return "static"
    return "other"


def _iter_datasets(family: str, duration: float, mearec_dir: Path):
    if family in ("hj", "all"):
        for entry in list_hybrid_janelia_scenes(ready_only=True):
            key = entry.get("scene_key") or entry.get("short_name")
            yield "hj", load_hybrid_janelia(scene=key, duration_s=duration)
    if family in ("mearec", "mearec_snr5", "all"):
        for path in discover_npz(mearec_dir):
            if family in ("mearec_snr5", "all") and "snr5" not in path.name:
                continue
            yield "mearec", load_mearec_npz(path, duration_s=duration)


def _true_unit_scores_or_dists(
    feats: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    *,
    method: str,
    shift_radius: int,
) -> np.ndarray:
    units = np.array(sorted(centroids.keys()), dtype=np.int64)
    mat = np.stack([centroids[int(u)] for u in units], axis=0)
    unit_to_j = {int(u): j for j, u in enumerate(units)}
    if _uses_conv_family(method):
        scores = shift_max_dot(
            feats,
            mat,
            shift_radius=int(shift_radius),
            normalize=_uses_cosine(method),
        )
        out = np.full(len(labels), np.nan)
        for i, y in enumerate(labels):
            j = unit_to_j.get(int(y))
            if j is not None:
                out[i] = scores[i, j]
        return out
    dists = shift_min_distance(
        feats, mat, shift_radius=int(shift_radius), metric="l1"
    )
    out = np.full(len(labels), np.nan)
    for i, y in enumerate(labels):
        j = unit_to_j.get(int(y))
        if j is not None:
            out[i] = dists[i, j]
    return out


def _gate_conv(
    com_tr: np.ndarray,
    feat_tr: np.ndarray,
    lab_tr: np.ndarray,
    com_te: np.ndarray,
    feat_te: np.ndarray,
    lab_te: np.ndarray,
    *,
    shift_radius: int,
    normalize: bool,
    percentile: float = 95.0,
) -> dict[str, float]:
    """COM L2 ∧ CONV/cosine score gate: keep units with score >= τ."""
    com_cents = fit_unit_centroids(com_tr, lab_tr)
    p2p_cents = fit_unit_centroids(feat_tr, lab_tr)
    d_com = []
    for x, y in zip(com_tr, lab_tr):
        c = com_cents.get(int(y))
        if c is not None:
            d_com.append(vector_distance(x, c, "l2"))
    tau_com = max(float(np.percentile(d_com, percentile)), 1e-8) if d_com else 1.0

    true_scores = _true_unit_scores_or_dists(
        feat_tr,
        lab_tr,
        p2p_cents,
        method="cosine" if normalize else "conv",
        shift_radius=shift_radius,
    )
    true_scores = true_scores[np.isfinite(true_scores)]
    low_pct = max(100.0 - float(percentile), 0.0)
    tau_score = float(np.percentile(true_scores, low_pct)) if true_scores.size else 0.0

    units = [u for u in sorted(com_cents.keys()) if u in p2p_cents]
    mat = np.stack([p2p_cents[u] for u in units], axis=0) if units else np.zeros((0, feat_te.shape[1]))
    hits, sizes = [], []
    for i, y in enumerate(lab_te):
        yi = int(y)
        if yi not in com_cents or yi not in p2p_cents:
            continue
        scores = shift_max_dot(
            feat_te[i : i + 1],
            mat,
            shift_radius=int(shift_radius),
            normalize=normalize,
        )[0]
        cands = []
        for j, u in enumerate(units):
            d_com_u = vector_distance(com_te[i], com_cents[u], "l2")
            if d_com_u < tau_com and float(scores[j]) >= tau_score:
                cands.append(int(u))
        sizes.append(len(cands))
        hits.append(yi in cands)
    return {
        "gate_recall": float(np.mean(hits)) if hits else float("nan"),
        "mean_candidates": float(np.mean(sizes)) if sizes else float("nan"),
        "tau_com": float(tau_com),
        "tau_score_or_p2p": float(tau_score),
        "n_gate_eval": float(len(hits)),
    }


def _gate_l1(
    com_tr: np.ndarray,
    feat_tr: np.ndarray,
    lab_tr: np.ndarray,
    com_te: np.ndarray,
    feat_te: np.ndarray,
    lab_te: np.ndarray,
    *,
    shift_radius: int,
    percentile: float = 95.0,
) -> dict[str, float]:
    com_cents = fit_unit_centroids(com_tr, lab_tr)
    p2p_cents = fit_unit_centroids(feat_tr, lab_tr)
    tau_com, tau_p2p = calibrate_thresholds(
        com_tr,
        feat_tr,
        lab_tr,
        percentile=percentile,
        com_metric="l2",
        p2p_metric="l1",
        shift_radius=int(shift_radius),
    )
    hits, sizes = [], []
    for i, y in enumerate(lab_te):
        yi = int(y)
        if yi not in com_cents or yi not in p2p_cents:
            continue
        cands = stage1_candidates(
            com_te[i],
            feat_te[i],
            com_cents,
            p2p_cents,
            tau_com,
            tau_p2p,
            com_metric="l2",
            p2p_metric="l1",
            shift_radius=int(shift_radius),
        )
        sizes.append(len(cands))
        hits.append(yi in cands)
    return {
        "gate_recall": float(np.mean(hits)) if hits else float("nan"),
        "mean_candidates": float(np.mean(sizes)) if sizes else float("nan"),
        "tau_com": float(tau_com),
        "tau_score_or_p2p": float(tau_p2p),
        "n_gate_eval": float(len(hits)),
    }


def evaluate_prepared(
    prep: dict[str, np.ndarray],
    *,
    dataset_name: str,
    method: str,
    bit_mode: str,
    shift_radius: int,
) -> dict[str, Any]:
    tr, te = _chronological_split(prep["times"])
    com = prep["com"]
    feats, n_bits = _prepare_features(prep["p2p"], bit_mode)
    lab = prep["labels"]
    k = int(feats.shape[1])
    S = int(shift_radius)

    cents = fit_unit_centroids(feats[tr], lab[tr])
    if method == "conv":
        pred = assign_shift_max_dot(
            feats[te], cents, shift_radius=S, normalize=False
        )
    elif method == "cosine":
        pred = assign_shift_max_dot(
            feats[te], cents, shift_radius=S, normalize=True
        )
    elif method == "l1":
        pred = assign_nearest(feats[te], cents, metric="l1", shift_radius=S)
    else:
        raise ValueError(f"Unknown method: {method}")

    keep = np.array([int(y) in cents for y in lab[te]])
    argmin = (
        float(np.mean(pred[keep] == lab[te][keep].astype(pred.dtype)))
        if keep.any()
        else float("nan")
    )
    true_vals = _true_unit_scores_or_dists(
        feats[te], lab[te], cents, method=method, shift_radius=S
    )
    true_vals = true_vals[np.isfinite(true_vals)]
    median_true = float(np.median(true_vals)) if true_vals.size else float("nan")

    if _uses_conv_family(method):
        gate = _gate_conv(
            com[tr],
            feats[tr],
            lab[tr],
            com[te],
            feats[te],
            lab[te],
            shift_radius=S,
            normalize=_uses_cosine(method),
        )
    else:
        gate = _gate_l1(
            com[tr], feats[tr], lab[tr], com[te], feats[te], lab[te], shift_radius=S
        )

    n_units = len(cents)
    acc = cim_row_accounting(n_units=n_units, k=k, shift_radius=S, n_bits=n_bits)
    ops = search_ops_per_event(
        gate["mean_candidates"] if np.isfinite(gate["mean_candidates"]) else float(n_units),
        k,
        S,
    )
    return {
        "dataset": dataset_name,
        "method": method,
        "bit_mode": bit_mode,
        "n_bits": int(n_bits),
        "shift_radius": S,
        "k": k,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "n_units_train": int(n_units),
        "p2p_assign_accuracy": argmin,
        "median_true_score_or_l1": median_true,
        "gate_recall": gate["gate_recall"],
        "mean_candidates": gate["mean_candidates"],
        "tau_com": gate["tau_com"],
        "tau_score_or_p2p": gate["tau_score_or_p2p"],
        "rows_per_unit": int(acc["rows_per_unit"]),
        "bits_per_row": int(acc["bits_per_row"]),
        "total_bits": int(acc["total_bits"]),
        "area_multiplier_vs_s0": float(acc["area_multiplier_vs_s0"]),
        "search_ops_per_event": float(ops),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.nanmean([r[key] for r in rows])) if rows else float("nan")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "p2p_assign_accuracy",
        "median_true_score_or_l1",
        "gate_recall",
        "mean_candidates",
        "total_bits",
        "search_ops_per_event",
    )
    grids: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        grids[family] = {}
        for method in METHODS:
            for bit in BIT_MODES:
                for S in SHIFT_RADII:
                    sub = [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["method"] == method
                        and r["bit_mode"] == bit
                        and r["shift_radius"] == S
                    ]
                    grids[family][f"{method}_b{bit}_S{S}"] = {
                        **{k: _mean(sub, k) for k in keys},
                        "n": float(len(sub)),
                    }

    # Bit sweep at best S for each method (by family-mean assign accuracy)
    bit_curves: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        bit_curves[family] = {}
        for method in METHODS:
            curve = {}
            for bit in BIT_MODES:
                best_acc, best_s = -1.0, 0
                for S in SHIFT_RADII:
                    sub = [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["method"] == method
                        and r["bit_mode"] == bit
                        and r["shift_radius"] == S
                    ]
                    acc = _mean(sub, "p2p_assign_accuracy")
                    if acc > best_acc:
                        best_acc, best_s = acc, S
                curve[bit] = {"best_S": best_s, "assign_accuracy": best_acc}
            bit_curves[family][method] = curve

    # Head-to-head: cosine/conv vs L1 at matched bit/S
    head_to_head = {}
    for family in sorted({r["family"] for r in rows}):
        head_to_head[family] = {}
        for method in ("conv", "cosine"):
            for bit in ("4", "5", "6", "float"):
                for S in SHIFT_RADII:
                    c = [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["method"] == method
                        and r["bit_mode"] == bit
                        and r["shift_radius"] == S
                    ]
                    l = [
                        r
                        for r in rows
                        if r["family"] == family
                        and r["method"] == "l1"
                        and r["bit_mode"] == bit
                        and r["shift_radius"] == S
                    ]
                    head_to_head[family][f"{method}_b{bit}_S{S}"] = {
                        f"{method}_acc": _mean(c, "p2p_assign_accuracy"),
                        "l1_acc": _mean(l, "p2p_assign_accuracy"),
                        f"delta_{method}_minus_l1_pp": 100.0
                        * (_mean(c, "p2p_assign_accuracy") - _mean(l, "p2p_assign_accuracy")),
                    }

    # Recommended operating points
    picks = {}
    for family in sorted({r["family"] for r in rows}):
        best = None
        for rkey, vals in grids[family].items():
            score = vals.get("p2p_assign_accuracy", float("nan"))
            if not np.isfinite(score):
                continue
            # prefer higher acc, then lower total_bits, then lower S
            cand = (score, -vals.get("total_bits", 0), rkey)
            if best is None or cand > best:
                best = cand
        picks[family] = best[2] if best else None

    return {
        "n_rows": len(rows),
        "grids": grids,
        "bit_curves_best_S": bit_curves,
        "head_to_head_conv_vs_l1": head_to_head,
        "best_config_by_family": picks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift-CONV vs L1 bit sweep")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--family", choices=("hj", "mearec", "mearec_snr5", "all"), default="all"
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
        for method in METHODS:
            for bit in BIT_MODES:
                for S in SHIFT_RADII:
                    if not args.quiet:
                        print(f"  {method} bits={bit} S={S}", flush=True)
                    row = evaluate_prepared(
                        prep,
                        dataset_name=ds.name,
                        method=method,
                        bit_mode=bit,
                        shift_radius=S,
                    )
                    row["family"] = family
                    row["condition"] = _condition(ds.name)
                    rows.append(row)

    csv_path = out_dir / "shift_conv_bit_sweep.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows)
    summary_path = out_dir / "shift_conv_bit_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        print(json.dumps(summary["bit_curves_best_S"], indent=2, sort_keys=True))
        print(json.dumps(summary["best_config_by_family"], indent=2))
        print(f"Wrote {csv_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
