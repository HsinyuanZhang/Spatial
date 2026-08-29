"""COM radius prefilter after 6-bit all-channel p99.9 ADC.

COM nearest-centroid unique-ID is weak on this MEA. This script treats COM as
a spatial candidate gate: keep train units whose COM centroid is within R of
the event COM, drop the rest.

ADC lock: signed 6-bit, one full-scale for all channels, Vref = p99.9 of
|sample| in ±15-sample event windows. Isolation remains the 3-pitch
same-spike disk. Radius is Euclidean micrometres in probe coordinates.

    python -m Spatial.experiments.analyze_com_radius_prefilter --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    com_features,
    extract_local_p2p,
    quantize_signed_adc,
)
from Spatial.algorithms.two_stage_assign import fit_unit_centroids
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR
from Spatial.experiments.analyze_adc_absolute_com import _com_xy_um, _event_abs_scales
from Spatial.experiments.run_com_mainchannel_lowbit import (
    WINDOW,
    _chronological_split,
    align_ks4_spike_positions,
    fill_missing_seed_xy,
    prepare_com_mainchannel_events,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "com_radius_prefilter"
ADC_BITS = 6
DEFAULT_RADII_UM: tuple[float, ...] = (
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    90.0,
    120.0,
    150.0,
    180.0,
)
DEFAULT_PERCENTILES: tuple[float, ...] = (90.0, 95.0, 99.0, 99.9)


def _ordered_split(
    com_um: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float,
) -> dict[str, np.ndarray]:
    labs = np.asarray(labels).ravel()
    t = np.asarray(times, dtype=np.int64).ravel()
    xy = np.asarray(com_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"com_um must be (n, 2), got {xy.shape}")
    if labs.shape[0] != xy.shape[0] or t.shape[0] != xy.shape[0]:
        raise ValueError("com_um, labels, and times length mismatch")
    order = np.argsort(t, kind="stable")
    xy, labs, t = xy[order], labs[order], t[order]
    train_idx, test_idx = _chronological_split(xy.shape[0], train_frac)
    return {
        "com_train": xy[train_idx],
        "com_test": xy[test_idx],
        "y_train": labs[train_idx],
        "y_test": labs[test_idx],
        "times_test": t[test_idx],
    }


def evaluate_com_radius_prefilter(
    com_um: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    radius_um: float | None = None,
    per_unit_percentile: float | None = None,
    train_frac: float = 0.5,
) -> dict[str, Any]:
    """True-unit recall of a Euclidean COM ball, train centroids only.

    Exactly one of ``radius_um`` (global) or ``per_unit_percentile`` (train
    intra-unit distance percentile, one radius per unit) must be set.
    """
    if (radius_um is None) == (per_unit_percentile is None):
        raise ValueError("set exactly one of radius_um or per_unit_percentile")
    split = _ordered_split(com_um, labels, times, train_frac)
    cents = fit_unit_centroids(split["com_train"], split["y_train"])
    if not cents:
        raise ValueError("no train centroids")
    units = np.array(sorted(cents.keys()), dtype=np.int64)
    mat = np.stack([cents[int(u)] for u in units], axis=0)
    y = split["y_test"]
    x = split["com_test"]
    col = {int(u): j for j, u in enumerate(units)}
    seen = np.array([int(u) in col for u in y], dtype=bool)
    dists = np.linalg.norm(x[:, None, :] - mat[None, :, :], axis=2)
    if radius_um is not None:
        radii = np.full(units.shape[0], float(radius_um), dtype=np.float64)
        policy = "global_um"
        policy_value = float(radius_um)
    else:
        pct = float(per_unit_percentile)
        radii = np.empty(units.shape[0], dtype=np.float64)
        y_tr = split["y_train"]
        x_tr = split["com_train"]
        for j, u in enumerate(units):
            intra = np.linalg.norm(x_tr[y_tr == u] - mat[j], axis=1)
            radii[j] = max(float(np.percentile(intra, pct)), 1e-8)
        policy = "per_unit_percentile"
        policy_value = pct
    keep = dists <= radii[None, :]
    n_cand = keep.sum(axis=1).astype(np.int64)
    n_units = int(units.size)
    n_eval = int(seen.sum())
    if n_eval:
        true_col = np.array([col[int(u)] for u in y[seen]], dtype=np.int64)
        hits = keep[seen, true_col]
        n_hit = int(hits.sum())
        recall = float(n_hit / n_eval)
    else:
        n_hit = 0
        recall = float("nan")
    d_masked = np.where(keep, dists, np.inf)
    pred_idx = np.argmin(d_masked, axis=1)
    pred = units[pred_idx]
    pred[~np.any(keep, axis=1)] = -1
    n_correct = int(np.sum((pred[seen] == y[seen]))) if n_eval else 0
    mean_c = float(np.mean(n_cand)) if n_cand.size else 0.0
    return {
        "policy": policy,
        "policy_value": policy_value,
        "n_train": int(split["com_train"].shape[0]),
        "n_test": int(x.shape[0]),
        "n_eval": n_eval,
        "n_units_train": n_units,
        "n_hit": n_hit,
        "candidate_recall": recall,
        "mean_candidates": mean_c,
        "median_candidates": float(np.median(n_cand)) if n_cand.size else 0.0,
        "p90_candidates": float(np.percentile(n_cand, 90)) if n_cand.size else 0.0,
        "max_candidates": int(n_cand.max()) if n_cand.size else 0,
        "frac_empty": float(np.mean(n_cand == 0)) if n_cand.size else float("nan"),
        "mean_units_dropped": float(n_units - mean_c),
        "frac_units_dropped": float(1.0 - mean_c / max(n_units, 1)),
        "traffic_reduction_x": float(n_units / max(mean_c, 1e-12)),
        "argmin_in_candidates": float(n_correct / n_eval) if n_eval else float("nan"),
        "mean_radius_um": float(np.mean(radii)),
        "median_radius_um": float(np.median(radii)),
        "p90_radius_um": float(np.percentile(radii, 90)),
        "max_radius_um": float(np.max(radii)),
    }


def _com_diagnostics(com_um: np.ndarray, labels: np.ndarray, times: np.ndarray) -> dict:
    split = _ordered_split(com_um, labels, times, 0.5)
    cents = fit_unit_centroids(split["com_train"], split["y_train"])
    units = np.array(sorted(cents.keys()), dtype=np.int64)
    mat = np.stack([cents[int(u)] for u in units], axis=0)
    intra = []
    for u, c in cents.items():
        pts = split["com_train"][split["y_train"] == u]
        intra.append(float(np.sqrt(np.mean(np.sum((pts - c) ** 2, axis=1)))))
    nn = []
    for j in range(mat.shape[0]):
        d = np.linalg.norm(mat - mat[j], axis=1)
        d[j] = np.inf
        nn.append(float(np.min(d)))
    return {
        "n_units_train": int(units.size),
        "intra_com_rms_um_median": float(np.median(intra)) if intra else float("nan"),
        "intra_com_rms_um_p90": float(np.percentile(intra, 90)) if intra else float("nan"),
        "centroid_nn_um_median": float(np.median(nn)) if nn else float("nan"),
        "centroid_nn_um_p10": float(np.percentile(nn, 10)) if nn else float("nan"),
    }


def _pick_point(rows: list[dict[str, Any]], min_recall: float) -> dict[str, Any] | None:
    feasible = [r for r in rows if r["candidate_recall"] >= min_recall]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda r: (r["mean_candidates"], -r["candidate_recall"]),
    )


def _adc6_all_channel_p999_com(
    dataset,
    seed_xy: np.ndarray,
    window: int,
    pitches: float,
) -> dict[str, np.ndarray]:
    prep = prepare_com_mainchannel_events(
        dataset,
        window=window,
        same_spike_max_pitches=pitches,
        seed_xy=seed_xy,
        return_filtered=True,
    )
    filtered = prep["filtered"]
    times = prep["times"]
    labels = prep["labels"]
    homes = prep["central"]
    analog_com = prep["com"]
    meta = prep["meta"]
    rng = np.random.default_rng(0)
    scales = _event_abs_scales(filtered, times, window, rng)
    recon = quantize_signed_adc(filtered, ADC_BITS, scales["global_p999"])
    table = build_radius_neighbor_table(dataset.geom, float(meta["radius_um"]))
    p2p, p2p_times, nbr = extract_local_p2p(recon, times, homes, table, window=window)
    if not np.array_equal(p2p_times, times):
        raise AssertionError("ADC P2P row alignment changed")
    adc_com = com_features(p2p, nbr, dataset.geom)
    analog_um = _com_xy_um(analog_com, dataset.geom)
    adc_um = _com_xy_um(adc_com, dataset.geom)
    return {
        "times": times,
        "labels": labels,
        "analog_com_um": analog_um,
        "adc_com_um": adc_um,
        "full_scale": np.array(scales["global_p999"], dtype=np.float64),
        "radius_um_isolation": np.array(meta["radius_um"], dtype=np.float64),
        "pitch_um": np.array(meta["pitch_um"], dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--same-spike-max-pitches", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ds = load_yger_ks4(path=YGER_KS4_NPZ_PATH, duration_s=args.duration)
    seed_xy = fill_missing_seed_xy(
        align_ks4_spike_positions(ds.spike_times, ds.spike_units, DEFAULT_KS4_TH13_DIR),
        ds.spike_units,
    )
    packed = _adc6_all_channel_p999_com(
        ds,
        seed_xy,
        int(args.window),
        float(args.same_spike_max_pitches),
    )
    times = packed["times"]
    labels = packed["labels"]
    sources = {
        "adc6_allch_p999": packed["adc_com_um"],
        "analog_p2p": packed["analog_com_um"],
    }

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for source, com_um in sources.items():
        diagnostics[source] = _com_diagnostics(com_um, labels, times)
        for radius in DEFAULT_RADII_UM:
            rec = evaluate_com_radius_prefilter(
                com_um, labels, times, radius_um=float(radius)
            )
            rec["source"] = source
            rec["dataset"] = ds.name
            rec["duration_s"] = float(args.duration)
            rec["adc_bits"] = ADC_BITS if source.startswith("adc") else -1
            rec["n_events"] = int(times.size)
            rows.append(rec)
        for pct in DEFAULT_PERCENTILES:
            rec = evaluate_com_radius_prefilter(
                com_um, labels, times, per_unit_percentile=float(pct)
            )
            rec["source"] = source
            rec["dataset"] = ds.name
            rec["duration_s"] = float(args.duration)
            rec["adc_bits"] = ADC_BITS if source.startswith("adc") else -1
            rec["n_events"] = int(times.size)
            rows.append(rec)

    adc_global = [
        r
        for r in rows
        if r["source"] == "adc6_allch_p999" and r["policy"] == "global_um"
    ]
    adc_pct = [
        r
        for r in rows
        if r["source"] == "adc6_allch_p999" and r["policy"] == "per_unit_percentile"
    ]
    pick99 = _pick_point(adc_global, 0.99)
    pick95 = _pick_point(adc_global, 0.95)
    pick99_pct = _pick_point(adc_pct, 0.99)
    pick95_pct = _pick_point(adc_pct, 0.95)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "com_radius_prefilter.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "policy": row["policy"],
            "policy_value": row["policy_value"],
            "candidate_recall": row["candidate_recall"],
            "mean_candidates": row["mean_candidates"],
            "frac_units_dropped": row["frac_units_dropped"],
            "traffic_reduction_x": row["traffic_reduction_x"],
            "argmin_in_candidates": row["argmin_in_candidates"],
            "mean_radius_um": row["mean_radius_um"],
        }

    summary = {
        "dataset": ds.name,
        "duration_s": float(args.duration),
        "adc": "signed 6-bit, all-channel Vref = event-window |x| p99.9",
        "isolation": "3-pitch same-spike disk; COM from ADC-reconstructed P2P",
        "full_scale_abs": float(packed["full_scale"]),
        "diagnostics": diagnostics,
        "selected": {
            "global_recall_ge_0.99": _brief(pick99),
            "global_recall_ge_0.95": _brief(pick95),
            "per_unit_percentile_recall_ge_0.99": _brief(pick99_pct),
            "per_unit_percentile_recall_ge_0.95": _brief(pick95_pct),
        },
        "n_rows": len(rows),
        "csv": str(csv_path),
    }
    json_path = out_dir / "com_radius_prefilter_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"ADC 6-bit all-channel p99.9 | FS={float(packed['full_scale']):.1f} | "
        f"events={times.size} | train units="
        f"{diagnostics['adc6_allch_p999']['n_units_train']}"
    )
    diag = diagnostics["adc6_allch_p999"]
    print(
        f"ADC COM intra RMS median {diag['intra_com_rms_um_median']:.2f} um | "
        f"centroid NN median {diag['centroid_nn_um_median']:.2f} um"
    )
    print(
        f"{'source':<18} {'policy':<22} {'R/p':>7} {'recall':>8} "
        f"{'mean C':>8} {'drop U':>8} {'x':>6} {'argmin':>8}"
    )
    for r in rows:
        if r["source"] != "adc6_allch_p999":
            continue
        print(
            f"{r['source']:<18} {r['policy']:<22} {r['policy_value']:7.1f} "
            f"{r['candidate_recall']:8.4f} {r['mean_candidates']:8.2f} "
            f"{r['frac_units_dropped']:8.3f} {r['traffic_reduction_x']:6.2f} "
            f"{r['argmin_in_candidates']:8.4f}"
        )

    def _show(title: str, row: dict[str, Any] | None) -> None:
        if row is None:
            print(f"Selected {title}: none (recall target not reached)")
            return
        print(
            f"Selected {title}: {row['policy']}={row['policy_value']:g} | "
            f"recall={row['candidate_recall']:.4f} | mean C={row['mean_candidates']:.2f} | "
            f"drop {row['frac_units_dropped']:.1%} of {adc_global[0]['n_units_train']} units | "
            f"{row['traffic_reduction_x']:.2f}x"
        )

    _show("global R @ recall>=0.99", pick99)
    _show("global R @ recall>=0.95", pick95)
    _show("per-unit percentile @ recall>=0.99", pick99_pct)
    _show("per-unit percentile @ recall>=0.95", pick95_pct)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
