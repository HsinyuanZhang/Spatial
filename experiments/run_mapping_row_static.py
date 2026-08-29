"""Frozen mapping-row static contrast J0–J5 on locked development recordings.

    python -m Spatial.experiments.run_mapping_row_static --pilot --duration 60 --quiet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.mapping_init import (
    calibrate_tau,
    column_abs_scales,
    com_codes_unit_interval,
    fit_gain_ranges,
    fit_interval_rows,
    fit_mean_l1_p999,
    gain_full_scale,
    j0_search_from_descriptor,
    query_mean_l1,
    shape_codes_from_extrema,
)
from Spatial.algorithms.mapping_row import (
    analog_gain,
    candidate_lists,
    quality_flags,
    quantize_gain,
    scale_alert,
    score_rows,
)
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_extrema,
)
from Spatial.data.loader import Dataset, resolve_hybrid_janelia_path
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.run_adaptive_weighted_5bit import (
    stable_chronological_three_way_split,
)
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _load_locked_recording,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "mapping_row_v0"
WINDOW = 15
K_NEIGHBORS = 7
WAVEFORM_TEMPLATE_BITS = 64 * 5
ARMS = ("J0", "J1", "J2", "J3", "J4", "J5")
SOURCE_PATHS = (
    "Spatial/algorithms/mapping_row.py",
    "Spatial/algorithms/mapping_init.py",
    "Spatial/experiments/run_mapping_row_static.py",
)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def prepare_mapping_events(dataset: Dataset) -> dict[str, np.ndarray]:
    prepared = prepare_dataset_events(dataset, k_neighbors=K_NEIGHBORS, footprint_layout="knn")
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    table = build_knn_table_with_self(dataset.geom, min(K_NEIGHBORS, dataset.n_channels))
    vmax, vmin, imax, imin, times, nbr = extract_local_extrema(
        filtered,
        prepared["times"],
        prepared["central"],
        table,
        window=WINDOW,
    )
    if not np.array_equal(times, prepared["times"]):
        raise AssertionError("extrema rows drifted from prepared GT events")
    g = analog_gain(vmax, vmin, nbr)
    sat = float(np.max(np.abs(filtered))) if filtered.size else 1.0
    flags = quality_flags(
        vmax, vmin, imax, imin, nbr, g,
        window=WINDOW, home_slot=0, sat_abs=sat,
    )
    prepared.update(
        {
            "vmax": vmax,
            "vmin": vmin,
            "imax": imax,
            "imin": imin,
            "ext_neighbors": nbr,
            "analog_g": g,
            "quality": flags,
        }
    )
    return prepared


def _split(prepared: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return stable_chronological_three_way_split(prepared["times"])


def _traffic(counts: np.ndarray, n_units: int) -> dict[str, float | int]:
    n = int(counts.size)
    reads = np.where(counts > 1, counts, 0).astype(np.int64)
    actual = int(reads.sum()) * WAVEFORM_TEMPLATE_BITS
    global_bits = n * int(n_units) * WAVEFORM_TEMPLATE_BITS
    return {
        "actual_waveform_template_bits_read": actual,
        "global_all_unit_waveform_template_bits": global_bits,
        "waveform_bit_traffic_reduction_factor": (
            float(global_bits / actual) if actual else float("inf")
        ),
        "mean_C": float(counts.mean()) if n else 0.0,
        "p95_C": float(np.percentile(counts, 95)) if n else 0.0,
        "frac_C0": float(np.mean(counts == 0)) if n else 0.0,
        "frac_C1": float(np.mean(counts == 1)) if n else 0.0,
        "frac_Cgt1": float(np.mean(counts > 1)) if n else 0.0,
    }


def _gaps(
    scores: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
) -> dict[str, float]:
    index = {_scalar(u): i for i, u in enumerate(units)}
    gaps = []
    s_true = []
    s_nn = []
    for i, y in enumerate(labels):
        key = _scalar(y)
        if key not in index:
            continue
        row = np.asarray(scores[i], dtype=np.float64).copy()
        ti = index[key]
        true = float(row[ti])
        row[ti] = np.inf
        nn = float(np.min(row)) if row.size > 1 else float("inf")
        gaps.append(nn - true)
        s_true.append(true)
        s_nn.append(nn)
    if not gaps:
        return {
            "gap_mean": float("nan"),
            "gap_p10": float("nan"),
            "gap_p50": float("nan"),
            "s_true_mean": float("nan"),
            "s_nn_mean": float("nan"),
        }
    g = np.asarray(gaps, dtype=np.float64)
    return {
        "gap_mean": float(np.mean(g)),
        "gap_p10": float(np.percentile(g, 10)),
        "gap_p50": float(np.percentile(g, 50)),
        "s_true_mean": float(np.mean(s_true)),
        "s_nn_mean": float(np.mean(s_nn)),
    }


def _per_unit_recall(candidates: list[np.ndarray], labels: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels)
    recalled = np.array(
        [any(_scalar(c) == _scalar(lab) for c in rows) for rows, lab in zip(candidates, y)],
        dtype=bool,
    )
    worst = 1.0
    n_units = 0
    for unit in np.unique(y):
        mask = y == unit
        if not np.any(mask):
            continue
        n_units += 1
        worst = min(worst, float(recalled[mask].mean()))
    return {
        "true_unit_candidate_recall": float(recalled.mean()) if y.size else 0.0,
        "worst_unit_recall": float(worst) if n_units else 0.0,
        "n_test_units": int(n_units),
    }


def _row_bits_interval(n_dim: int) -> int:
    # COM 2×5×2 bounds = 20; POSNEG 14×4×2 = 112; g range 8; tau 6.
    posneg_dims = max(n_dim - 2, 0)
    return 2 * 5 * 2 + posneg_dims * 4 * 2 + 8 + 6


def _evaluate_lists(
    candidates: list[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    units: np.ndarray,
    n_dim: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    counts = np.asarray([len(c) for c in candidates], dtype=np.int64)
    search = evaluate_candidate_search(candidates, labels)
    rec = _per_unit_recall(candidates, labels)
    traffic = _traffic(counts, int(units.size))
    gaps = _gaps(scores, labels, units)
    return {
        **search,
        **rec,
        **traffic,
        **gaps,
        "row_payload_bits": int(extra.get("row_payload_bits", _row_bits_interval(n_dim))),
        **{k: v for k, v in extra.items() if k != "row_payload_bits"},
    }


def run_j0(prepared: dict[str, np.ndarray], fit: np.ndarray, cal: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    desc = prepared["descriptor"]
    y = prepared["labels"]
    model = j0_search_from_descriptor(desc[fit], y[fit], desc[cal], y[cal])
    codes = model.transform(desc[test])
    candidates, diag = model.query_codes(codes, mode="l1")
    dist = np.abs(codes[:, None, :] - model.centroids_[None, :, :]).sum(axis=2)
    extra = {
        "row_payload_bits": int(model.l1_row_bits_full_scale),
        "n_active_row_comparisons": int(np.sum(diag["row_comparisons"])),
        "scale_alert_frac": 0.0,
    }
    return _evaluate_lists(candidates, y[test], dist, model.units_, desc.shape[1], extra)


def _shape_bundle(
    prepared: dict[str, np.ndarray],
    fit: np.ndarray,
    *,
    normalize_by_g: bool,
) -> dict[str, Any]:
    pos_scale = neg_scale = None
    if not normalize_by_g:
        pos_scale, neg_scale = column_abs_scales(
            prepared["vmax"][fit], prepared["vmin"][fit], prepared["ext_neighbors"][fit]
        )
    g_fs = gain_full_scale(prepared["analog_g"][fit])
    codes = shape_codes_from_extrema(
        prepared["com"],
        prepared["vmax"],
        prepared["vmin"],
        prepared["ext_neighbors"],
        prepared["analog_g"],
        normalize_by_g=normalize_by_g,
        pos_scale=pos_scale,
        neg_scale=neg_scale,
    )
    g_codes = quantize_gain(prepared["analog_g"], g_fs)
    return {"codes": codes, "g_codes": g_codes, "g_fs": g_fs}


def run_mean_l1_shape(
    prepared: dict[str, np.ndarray],
    fit: np.ndarray,
    cal: np.ndarray,
    test: np.ndarray,
    *,
    normalize_by_g: bool,
) -> dict[str, Any]:
    bundle = _shape_bundle(prepared, fit, normalize_by_g=normalize_by_g)
    y = prepared["labels"]
    model = fit_mean_l1_p999(bundle["codes"][fit], y[fit], bundle["codes"][cal], y[cal])
    cand, dist, meta = query_mean_l1(bundle["codes"][test], model)
    extra = {
        "row_payload_bits": int(model["centroids"].size * 5 + 9),
        "n_active_row_comparisons": int(meta["n_active_row_comparisons"]),
        "scale_alert_frac": 0.0,
    }
    return _evaluate_lists(cand, y[test], dist, model["units"], bundle["codes"].shape[1], extra)


def run_interval_arm(
    prepared: dict[str, np.ndarray],
    fit: np.ndarray,
    cal: np.ndarray,
    test: np.ndarray,
    *,
    normalize_by_g: bool,
    use_precision: bool,
    censoring: bool,
) -> dict[str, Any]:
    bundle = _shape_bundle(prepared, fit, normalize_by_g=normalize_by_g)
    y = prepared["labels"]
    rows = fit_interval_rows(bundle["codes"][fit], y[fit])
    if not use_precision:
        rows = {**rows, "precision": np.zeros_like(rows["precision"])}
    tau, tau_src = calibrate_tau(
        bundle["codes"][cal],
        y[cal],
        rows,
        censoring=censoring,
        codes_fit=bundle["codes"][fit],
        labels_fit=y[fit],
    )
    g_lo, g_hi = fit_gain_ranges(bundle["g_codes"][fit], y[fit], rows["units"])
    scored = score_rows(
        bundle["codes"][test],
        rows["lower"],
        rows["upper"],
        rows["precision"],
        rows["mu"],
        censoring=censoring,
    )
    cand, diag = candidate_lists(scored["S"], tau, rows["units"])
    alert = scale_alert(bundle["g_codes"][test], g_lo, g_hi)
    dim = bundle["codes"].shape[1]
    n_test, n_u = scored["S"].shape
    # Per-dimension contribution: mean penalty on true vs nearest interferer.
    contrib = _dim_contrib(scored["penalty"], y[test], rows["units"])
    extra = {
        "row_payload_bits": _row_bits_interval(dim),
        "n_active_row_comparisons": int(n_test * n_u),
        "n_lower_upper_compares": int(2 * n_test * n_u * dim),
        "n_shifts": int(n_test * n_u * dim),
        "n_saturations": int(n_test * n_u * dim),
        "n_adds": int(n_test * n_u * max(dim - 1, 0)),
        "scale_alert_frac": float(np.mean(alert)),
        "quality_sat_frac": float(np.mean(prepared["quality"][test, 0])),
        "quality_cut_frac": float(np.mean(prepared["quality"][test, 1])),
        "quality_overlap_frac": float(np.mean(prepared["quality"][test, 2])),
        "interval_width_mean": float(np.mean(rows["width"])),
        "tau_fallback_frac": float(np.mean(np.asarray(tau_src) != "calibration")),
        **contrib,
    }
    extra.update(_fit_half_stability(bundle["codes"][fit], y[fit], prepared["times"][fit]))
    return _evaluate_lists(cand, y[test], scored["S"], rows["units"], dim, extra)


def _dim_contrib(penalty: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[str, float]:
    index = {_scalar(u): i for i, u in enumerate(units)}
    true_pen = []
    nn_pen = []
    for i, y in enumerate(labels):
        key = _scalar(y)
        if key not in index:
            continue
        ti = index[key]
        s = penalty[i].sum(axis=1)
        true_pen.append(penalty[i, ti])
        s2 = s.copy()
        s2[ti] = np.iinfo(np.int64).max
        nn_pen.append(penalty[i, int(np.argmin(s2))])
    if not true_pen:
        return {}
    tp = np.mean(np.stack(true_pen, axis=0), axis=0)
    npn = np.mean(np.stack(nn_pen, axis=0), axis=0)
    return {
        "dim_gap_mean_com": float(np.mean((npn - tp)[:2])),
        "dim_gap_mean_posneg": float(np.mean((npn - tp)[2:])),
    }


def _fit_half_stability(codes: np.ndarray, labels: np.ndarray, times: np.ndarray) -> dict[str, float]:
    order = np.arange(codes.shape[0])
    if order.size < 4:
        return {"fit_half_width_ratio": float("nan")}
    mid = order.size // 2
    a = fit_interval_rows(codes[order[:mid]], labels[order[:mid]])
    b = fit_interval_rows(codes[order[mid:]], labels[order[mid:]])
    common = [u for u in a["units"] if u in set(b["units"].tolist())]
    if not common:
        return {"fit_half_width_ratio": float("nan")}
    ia = { _scalar(u): i for i, u in enumerate(a["units"]) }
    ib = { _scalar(u): i for i, u in enumerate(b["units"]) }
    ratios = []
    for u in common:
        wa = np.maximum(a["width"][ia[_scalar(u)]].astype(np.float64), 1.0)
        wb = np.maximum(b["width"][ib[_scalar(u)]].astype(np.float64), 1.0)
        ratios.append(float(np.mean(wb / wa)))
    return {"fit_half_width_ratio": float(np.median(ratios))}


def run_recording(dataset: Dataset, family: str, arms: Sequence[str]) -> list[dict[str, Any]]:
    prepared = prepare_mapping_events(dataset)
    fit, cal, test = _split(prepared)
    dispatch = {
        "J0": lambda: run_j0(prepared, fit, cal, test),
        "J1": lambda: run_mean_l1_shape(prepared, fit, cal, test, normalize_by_g=True),
        "J2": lambda: run_interval_arm(
            prepared, fit, cal, test, normalize_by_g=False, use_precision=True, censoring=False
        ),
        "J3": lambda: run_interval_arm(
            prepared, fit, cal, test, normalize_by_g=True, use_precision=True, censoring=False
        ),
        "J4": lambda: run_interval_arm(
            prepared, fit, cal, test, normalize_by_g=True, use_precision=False, censoring=False
        ),
        "J5": lambda: run_interval_arm(
            prepared, fit, cal, test, normalize_by_g=True, use_precision=True, censoring=True
        ),
    }
    rows = []
    for arm in arms:
        metrics = dispatch[arm]()
        rows.append(
            {
                "dataset": dataset.name,
                "family": family,
                "arm": arm,
                "n_fit": int(fit.size),
                "n_cal": int(cal.size),
                "n_test": int(test.size),
                **metrics,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        block = [r for r in rows if r["arm"] == arm]
        for family in ("hj", "mearec", "all"):
            sub = block if family == "all" else [r for r in block if r["family"] == family]
            if not sub:
                continue
            rec_unw = float(np.mean([r["true_unit_candidate_recall"] for r in sub]))
            n_test = sum(int(r["n_test"]) for r in sub)
            hits = sum(
                float(r["true_unit_candidate_recall"]) * int(r["n_test"]) for r in sub
            )
            actual = sum(int(r["actual_waveform_template_bits_read"]) for r in sub)
            glob = sum(int(r["global_all_unit_waveform_template_bits"]) for r in sub)
            out.append(
                {
                    "arm": arm,
                    "family": family,
                    "n_recordings": len(sub),
                    "recall_recording_unweighted": rec_unw,
                    "recall_event_weighted": float(hits / max(n_test, 1)),
                    "mean_C": float(np.mean([r["mean_C"] for r in sub])),
                    "traffic_x": float(glob / actual) if actual else float("inf"),
                    "gap_mean": float(np.mean([r["gap_mean"] for r in sub])),
                    "gap_p50": float(np.mean([r["gap_p50"] for r in sub])),
                    "worst_unit_recall_min": float(min(r["worst_unit_recall"] for r in sub)),
                }
            )
    return out


def _sha256_sources() -> dict[str, str]:
    out = {}
    for rel in SOURCE_PATHS:
        out[rel] = sha256_file(REPO_ROOT / rel)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--arms", type=str, default="J0,J1,J2,J3,J4,J5")
    parser.add_argument("--mearec-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.confirm:
        raise SystemExit(
            "confirmation recordings are not enabled until the development gate passes"
        )
    if not args.pilot:
        parser.error("v0 runs the locked development set; pass --pilot")
    if float(args.duration) != 60.0:
        parser.error("--pilot duration is locked to 60 seconds")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm}")
    recordings = list(LOCKED_DEVELOPMENT_RECORDINGS)
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for family, name in recordings:
        ds = _load_locked_recording(family, name, float(args.duration), Path(args.mearec_dir))
        if family == "hj":
            src = resolve_hybrid_janelia_path(scene=name)
        else:
            src = Path(args.mearec_dir) / f"{name}.npz"
        if Path(src).exists():
            input_hashes[str(src)] = sha256_file(Path(src))
        rec_rows = run_recording(ds, family, arms)
        rows.extend(rec_rows)
        if not args.quiet:
            for r in rec_rows:
                print(
                    f"{r['dataset']:<42} {r['arm']}  "
                    f"R={r['true_unit_candidate_recall']:.4f}  "
                    f"C={r['mean_C']:.2f}  x={r['waveform_bit_traffic_reduction_factor']:.2f}  "
                    f"gap={r['gap_mean']:.2f}"
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mapping_row_v0_pilot.csv"
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
    summary = {
        "arms": list(arms),
        "duration_s": float(args.duration),
        "recordings": [f"{f}:{n}" for f, n in recordings],
        "aggregate": agg,
        "input_sha256": input_hashes,
        "source_sha256": _sha256_sources(),
        "csv": str(csv_path),
    }
    write_json(out_dir / "mapping_row_v0_pilot_summary.json", summary)
    print(f"Saved {csv_path}")
    print("family aggregates:")
    for a in agg:
        print(
            f"  {a['arm']} {a['family']:<6}  "
            f"R_ev={a['recall_event_weighted']:.4f}  "
            f"R_rec={a['recall_recording_unweighted']:.4f}  "
            f"x={a['traffic_x']:.2f}  gap={a['gap_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
