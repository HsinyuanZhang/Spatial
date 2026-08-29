"""Direct nearest-centroid assignment: main-channel vs COM vs low-bit COM.

Yger 20160415_patch2 KS4 Th=13 pseudo-GT (362 units, 314 ``good``). Labels are
KS4 clusters, not juxta. This is oracle argmin accuracy on GT-isolated events,
not two-stage waveform refinement, not P2P-only, and not CiM candidate
recall/traffic. Do not treat the number as a STAR-Mem system-gate result.

Full-recording zero-phase bandpass (300–6000 Hz) is the same non-causal
isolation used by other Spatial DAC pilots.

``--same-spike-max-pitches 3`` restricts home and COM to a disk of 3 channel
pitches around the KS4 spike position. Extrema farther away are treated as a
different spike (overlap), not as wander of the same unit.

Usage (from parent workspace root)::

    python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 --quiet
    python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 \\
        --same-spike-max-pitches 3 \\
        --output-dir Spatial/output/com_mainchannel_lowbit_local3pitch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    build_radius_neighbor_table,
    com_features,
    extract_local_p2p,
    local_home_channels,
    main_channel_features,
    main_channel_xy_features,
    median_nearest_neighbor_spacing,
    normalize_geom_xy,
    quantize_com,
)
from Spatial.algorithms.two_stage_assign import assign_nearest, fit_unit_centroids
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR
from Spatial.evaluation.metrics import match_spikes_to_gt
from Spatial.experiments.run_adaptive_5bit_range_search import labels_for_peak_output


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "com_mainchannel_lowbit"
DEFAULT_COM_BITS: tuple[int, ...] = (4, 5, 6, 8)
JUXTA_TOL_MS = 0.4
WINDOW = 15
K_NEIGHBORS = 7
HOME_HALF_WINDOW = 2
FILTER_LOW_HZ = 300.0
FILTER_HIGH_HZ = 6000.0


def _chronological_split(n: int, train_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Equal chronological 50/50 on already time-sorted rows."""
    n_train = max(1, int(n * train_frac))
    if n_train >= n:
        n_train = max(1, n - 1) if n > 1 else 1
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n) if n_train < n else np.arange(n)
    return train_idx, test_idx


def oracle_argmin_assign(
    features: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
    metric: str = "l2",
) -> dict[str, Any]:
    """Nearest-centroid labels: centroids from the train split only.

    Primary accuracy is ``pred == true`` on test rows whose GT unit appeared
    in train (no Hungarian). Unseen-unit rows cannot match a train centroid.
    """
    feats = np.asarray(features, dtype=np.float64)
    labs = np.asarray(labels).ravel()
    t = np.asarray(times, dtype=np.int64).ravel()
    n = int(feats.shape[0])
    if n == 0 or feats.ndim != 2:
        raise ValueError("features must be a nonempty 2-D array")
    if labs.shape[0] != n or t.shape[0] != n:
        raise ValueError("features, labels, and times must have equal length")

    order = np.argsort(t, kind="stable")
    feats, labs, t = feats[order], labs[order], t[order]
    train_idx, test_idx = _chronological_split(n, train_frac)
    cents = fit_unit_centroids(feats[train_idx], labs[train_idx])
    pred = assign_nearest(feats[test_idx], cents, metric=metric)
    y = labs[test_idx]
    seen = np.array([int(u) in cents for u in y], dtype=bool)
    n_eval = int(seen.sum())
    n_correct = int(np.sum(pred[seen] == y[seen])) if n_eval else 0
    n_correct_all = int(np.sum(pred == y))
    return {
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_eval": n_eval,
        "n_correct": n_correct,
        "n_correct_all_test": n_correct_all,
        "argmin_accuracy": float(n_correct / n_eval) if n_eval else float("nan"),
        "argmin_accuracy_all_test": (
            float(n_correct_all / test_idx.size) if test_idx.size else float("nan")
        ),
        "n_units_train": int(len(cents)),
        "n_units_test": int(len(np.unique(y))),
        "n_units_unseen_test": int(len(np.unique(y[~seen]))) if np.any(~seen) else 0,
        "pred": pred,
        "y_true": y,
        "times_test": t[test_idx],
        "seen": seen,
    }


def juxta_match_rate(
    pred: np.ndarray,
    times_test: np.ndarray,
    juxta_times: np.ndarray,
    juxta_cluster: int,
    tolerance_samples: int,
) -> dict[str, float | int]:
    """Fraction of juxta-coincident test events assigned to the KS4 juxta cluster."""
    pred = np.asarray(pred)
    t = np.asarray(times_test, dtype=np.int64).ravel()
    jx = np.asarray(juxta_times, dtype=np.int64).ravel()
    if t.size == 0 or jx.size == 0:
        return {
            "juxta_n_coincident": 0,
            "juxta_n_pred_matched_cluster": 0,
            "juxta_match_rate": float("nan"),
        }
    matched_pred, _ = match_spikes_to_gt(t, jx, tolerance_samples=int(tolerance_samples))
    n_coin = int(matched_pred.size)
    if n_coin == 0:
        return {
            "juxta_n_coincident": 0,
            "juxta_n_pred_matched_cluster": 0,
            "juxta_match_rate": float("nan"),
        }
    n_hit = int(np.sum(pred[matched_pred] == int(juxta_cluster)))
    return {
        "juxta_n_coincident": n_coin,
        "juxta_n_pred_matched_cluster": n_hit,
        "juxta_match_rate": float(n_hit / n_coin),
    }


def load_yger_juxta(
    npz_path: Path,
    duration_s: float | None,
    fs: int,
) -> tuple[np.ndarray, int]:
    payload = np.load(str(npz_path), allow_pickle=True)
    times = np.asarray(payload["juxta_spike_times"], dtype=np.int64).ravel()
    cluster = int(payload["juxta_matched_cluster_id"])
    if duration_s is not None:
        times = times[times < int(duration_s * fs)]
    return times, cluster


def align_ks4_spike_positions(
    event_times: np.ndarray,
    event_labels: np.ndarray,
    ks4_dir: Path,
) -> np.ndarray:
    """Map each (time, cluster) event onto Kilosort4 ``spike_positions``.

    Duplicate (time, cluster) pairs consume KS4 rows in file order. Events
    with no matching KS4 row get NaN and should be filled by the caller.
    """
    ks_dir = Path(ks4_dir)
    ks_times = np.load(ks_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    ks_clu = np.load(ks_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    ks_pos = np.asarray(np.load(ks_dir / "spike_positions.npy"), dtype=np.float64)
    if ks_pos.ndim != 2 or ks_pos.shape[1] < 2 or ks_pos.shape[0] != ks_times.size:
        raise ValueError(
            f"KS4 spike_positions shape {ks_pos.shape} does not match "
            f"{ks_times.size} spike_times"
        )
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(ks_times.size):
        key = (int(ks_times[i]), int(ks_clu[i]))
        buckets.setdefault(key, []).append(i)
    cursors = {key: 0 for key in buckets}

    times = np.asarray(event_times, dtype=np.int64).ravel()
    labels = np.asarray(event_labels).ravel()
    if times.shape != labels.shape:
        raise ValueError("event_times and event_labels length mismatch")
    xy = np.full((times.size, 2), np.nan, dtype=np.float64)
    for j in range(times.size):
        key = (int(times[j]), int(labels[j]))
        rows = buckets.get(key)
        if not rows:
            continue
        c = cursors[key]
        if c >= len(rows):
            continue
        xy[j] = ks_pos[rows[c], :2]
        cursors[key] = c + 1
    return xy


def fill_missing_seed_xy(
    seed_xy: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Replace NaN seeds with that unit's median of finite positions."""
    xy = np.asarray(seed_xy, dtype=np.float64).copy()
    labs = np.asarray(labels).ravel()
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != labs.size:
        raise ValueError("seed_xy must be (n, 2) aligned with labels")
    missing = ~np.isfinite(xy).all(axis=1)
    if not np.any(missing):
        return xy
    for u in np.unique(labs[missing]):
        unit = labs == u
        finite = unit & ~missing
        if not np.any(finite):
            continue
        xy[unit & missing] = np.median(xy[finite], axis=0)
    return xy


def _global_localmin_homes(
    filtered: np.ndarray,
    times: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """Array-wide most-negative channel in ±half_window (overlap-theft diagnostic)."""
    n_samples = filtered.shape[1]
    homes = np.empty(len(times), dtype=np.int64)
    half = int(half_window)
    for i, t in enumerate(times):
        t0 = max(int(t) - half, 0)
        t1 = min(int(t) + half + 1, n_samples)
        homes[i] = int(np.argmin(filtered[:, t0:t1].min(axis=1)))
    return homes


def prepare_com_mainchannel_events(
    dataset,
    *,
    k_neighbors: int = K_NEIGHBORS,
    window: int = WINDOW,
    same_spike_max_pitches: float = 0.0,
    home_half_window: int = HOME_HALF_WINDOW,
    seed_xy: np.ndarray | None = None,
    return_filtered: bool = False,
) -> dict[str, np.ndarray]:
    """GT-event isolation: bandpass, then local P2P → COM.

    ``same_spike_max_pitches <= 0`` is the original path: whole-array
    ``argmax |amp|`` in ``±window`` and KNN K=``k_neighbors``.

    ``same_spike_max_pitches > 0`` treats extrema farther than that many
    channel pitches from ``seed_xy`` as a different spike. Home is the
    most-negative channel inside the disk (``±home_half_window`` samples);
    COM uses P2P on every electrode within the same radius of that home.
    """
    filtered = bandpass_filter(
        dataset.raw_data, dataset.fs, low=FILTER_LOW_HZ, high=FILTER_HIGH_HZ
    )
    pitches = float(same_spike_max_pitches)
    pitch_um = float(median_nearest_neighbor_spacing(dataset.geom))
    if pitches > 0.0:
        if seed_xy is None:
            raise ValueError(
                "same-spike radius isolation requires seed_xy (KS4 spike_positions)"
            )
        radius_um = pitches * pitch_um
        valid = (dataset.spike_times >= int(window)) & (
            dataset.spike_times < dataset.n_samples - int(window)
        )
        peak_times = np.asarray(dataset.spike_times[valid], dtype=np.int64)
        labels = np.asarray(dataset.spike_units[valid]).ravel()
        seeds = fill_missing_seed_xy(np.asarray(seed_xy, dtype=np.float64)[valid], labels)
        still_missing = ~np.isfinite(seeds).all(axis=1)
        if np.any(still_missing):
            raise ValueError(
                f"{int(still_missing.sum())} events have no KS4 position to seed "
                "the same-spike disk"
            )
        central, n_in_disk = local_home_channels(
            filtered,
            peak_times,
            seeds,
            dataset.geom,
            radius_um,
            half_window=int(home_half_window),
        )
        neighbor_table = build_radius_neighbor_table(dataset.geom, radius_um)
        global_homes = _global_localmin_homes(
            filtered, peak_times, half_window=int(home_half_window)
        )
        dist_global = np.linalg.norm(
            dataset.geom[global_homes] - seeds, axis=1
        )
        frac_stolen = float(np.mean(dist_global > radius_um + 1e-9))
        home_rule = (
            f"most-negative in ±{int(home_half_window)} samples among electrodes "
            f"within {pitches:g} pitches ({radius_um:.3f} um) of KS4 spike_positions"
        )
    else:
        radius_um = float("nan")
        _, peak_times, central = get_peak_amplitudes(
            filtered, dataset.spike_times, window=window
        )
        labels = labels_for_peak_output(
            dataset.spike_times,
            dataset.spike_units,
            peak_times,
            dataset.n_samples,
            window=window,
        )
        k = min(int(k_neighbors), dataset.n_channels)
        neighbor_table = build_knn_table_with_self(dataset.geom, k)
        seeds = np.full((peak_times.size, 2), np.nan, dtype=np.float64)
        n_in_disk = np.full(peak_times.size, neighbor_table.shape[1], dtype=np.int64)
        frac_stolen = float("nan")
        home_rule = f"global argmax |amp| in ±{int(window)} samples"

    p2p, p2p_times, neighbor_ids = extract_local_p2p(
        filtered, peak_times, central, neighbor_table, window=window
    )
    if not np.array_equal(p2p_times, peak_times):
        raise AssertionError("local P2P must preserve aligned event rows")
    com = com_features(p2p, neighbor_ids, dataset.geom)
    main_ch = main_channel_features(central, dataset.n_channels)
    main_xy = main_channel_xy_features(central, dataset.geom)
    order = np.argsort(peak_times, kind="stable")
    out: dict[str, np.ndarray] = {
        "times": peak_times[order],
        "labels": labels[order],
        "central": central[order],
        "p2p": p2p[order],
        "neighbor_ids": neighbor_ids[order],
        "com": com[order],
        "main_channel": main_ch[order],
        "main_channel_xy": main_xy[order],
        "n_in_disk": n_in_disk[order],
        "seed_xy": seeds[order],
    }
    if np.isfinite(seeds).all():
        out["ks4_xy_norm"] = normalize_geom_xy(seeds, dataset.geom)[order]
    if return_filtered:
        out["filtered"] = filtered
    out["meta"] = {
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "same_spike_max_pitches": pitches,
        "home_half_window": int(home_half_window),
        "home_rule": home_rule,
        "k_table": int(neighbor_table.shape[1]),
        "frac_global_home_outside_radius": frac_stolen,
        "n_neighbors_median": float(np.median(n_in_disk)) if n_in_disk.size else 0.0,
    }
    return out


def _bit_label(n_bits: int | None) -> str:
    return "float" if n_bits is None else f"{int(n_bits)}bit"


def _row_from_assign(
    *,
    dataset_name: str,
    method: str,
    n_bits: int | None,
    feature_dim: int,
    n_spikes: int,
    n_units_all: int,
    assign: dict[str, Any],
    juxta: dict[str, float | int],
    k_neighbors: int,
    window: int,
    metric: str,
    duration_s: float,
    good_only: bool,
    n_unique_x: int | None = None,
    n_unique_y: int | None = None,
    same_spike_max_pitches: float = 0.0,
    pitch_um: float | None = None,
    radius_um: float | None = None,
    home_rule: str = "",
    frac_global_home_outside_radius: float | None = None,
    n_neighbors_median: float | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "good_only": int(bool(good_only)),
        "duration_s": float(duration_s),
        "method": method,
        "n_bits": -1 if n_bits is None else int(n_bits),
        "bit_label": _bit_label(n_bits),
        "feature_dim": int(feature_dim),
        "n_spikes": int(n_spikes),
        "n_train": int(assign["n_train"]),
        "n_test": int(assign["n_test"]),
        "n_eval": int(assign["n_eval"]),
        "n_correct": int(assign["n_correct"]),
        "argmin_accuracy": float(assign["argmin_accuracy"]),
        "argmin_accuracy_all_test": float(assign["argmin_accuracy_all_test"]),
        "n_units_all": int(n_units_all),
        "n_units_train": int(assign["n_units_train"]),
        "n_units_test": int(assign["n_units_test"]),
        "n_units_unseen_test": int(assign["n_units_unseen_test"]),
        "n_unique_com_x": "" if n_unique_x is None else int(n_unique_x),
        "n_unique_com_y": "" if n_unique_y is None else int(n_unique_y),
        "juxta_n_coincident": int(juxta["juxta_n_coincident"]),
        "juxta_n_pred_matched_cluster": int(juxta["juxta_n_pred_matched_cluster"]),
        "juxta_match_rate": juxta["juxta_match_rate"],
        "k_neighbors": int(k_neighbors),
        "window": int(window),
        "metric": str(metric),
        "same_spike_max_pitches": float(same_spike_max_pitches),
        "pitch_um": "" if pitch_um is None else float(pitch_um),
        "radius_um": "" if radius_um is None else float(radius_um),
        "home_rule": str(home_rule),
        "frac_global_home_outside_radius": (
            ""
            if frac_global_home_outside_radius is None
            else float(frac_global_home_outside_radius)
        ),
        "n_neighbors_median": (
            "" if n_neighbors_median is None else float(n_neighbors_median)
        ),
    }


def run_study(
    dataset,
    *,
    com_bits: tuple[int, ...] = DEFAULT_COM_BITS,
    k_neighbors: int = K_NEIGHBORS,
    window: int = WINDOW,
    train_frac: float = 0.5,
    metric: str = "l2",
    duration_s: float = 60.0,
    good_only: bool = False,
    juxta_times: np.ndarray | None = None,
    juxta_cluster: int = 310,
    juxta_tol_samples: int = 8,
    same_spike_max_pitches: float = 0.0,
    home_half_window: int = HOME_HALF_WINDOW,
    seed_xy: np.ndarray | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    t0 = time.time()
    pitches = float(same_spike_max_pitches)
    if verbose:
        print(
            f"Dataset: {dataset.name} | {dataset.n_channels}ch | "
            f"{dataset.duration_s:.1f}s | units={len(dataset.unit_ids)} "
            f"| events={len(dataset.spike_times)} | good_only={good_only}"
        )
        if pitches > 0.0:
            print(
                "Isolation: GT event times, full-slice zero-phase bandpass "
                f"{FILTER_LOW_HZ:.0f}–{FILTER_HIGH_HZ:.0f} Hz, same-spike radius "
                f"{pitches:g} channel pitches (extrema farther away are excluded)."
            )
        else:
            print(
                "Isolation: GT event times, full-slice zero-phase bandpass "
                f"{FILTER_LOW_HZ:.0f}–{FILTER_HIGH_HZ:.0f} Hz, KNN K={k_neighbors}."
            )

    prep = prepare_com_mainchannel_events(
        dataset,
        k_neighbors=k_neighbors,
        window=window,
        same_spike_max_pitches=pitches,
        home_half_window=home_half_window,
        seed_xy=seed_xy,
    )
    meta = prep["meta"]
    times = prep["times"]
    labels = prep["labels"]
    com = prep["com"]
    main_ch = prep["main_channel"]
    main_xy = prep["main_channel_xy"]
    n_spikes = int(times.size)
    n_units_all = int(len(np.unique(labels)))
    k_used = int(meta["k_table"])

    if juxta_times is None:
        juxta_times = np.empty((0,), dtype=np.int64)

    rows: list[dict[str, Any]] = []

    def _eval_method(
        method: str,
        feats: np.ndarray,
        n_bits: int | None,
        unique_xy: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        assign = oracle_argmin_assign(
            feats, labels, times, train_frac=train_frac, metric=metric
        )
        juxta = juxta_match_rate(
            assign["pred"],
            assign["times_test"],
            juxta_times,
            juxta_cluster,
            juxta_tol_samples,
        )
        row = _row_from_assign(
            dataset_name=dataset.name,
            method=method,
            n_bits=n_bits,
            feature_dim=int(feats.shape[1]),
            n_spikes=n_spikes,
            n_units_all=n_units_all,
            assign=assign,
            juxta=juxta,
            k_neighbors=k_used,
            window=window,
            metric=metric,
            duration_s=duration_s,
            good_only=good_only,
            n_unique_x=None if unique_xy is None else unique_xy[0],
            n_unique_y=None if unique_xy is None else unique_xy[1],
            same_spike_max_pitches=float(meta["same_spike_max_pitches"]),
            pitch_um=float(meta["pitch_um"]),
            radius_um=None
            if not np.isfinite(meta["radius_um"])
            else float(meta["radius_um"]),
            home_rule=str(meta["home_rule"]),
            frac_global_home_outside_radius=(
                None
                if not np.isfinite(meta["frac_global_home_outside_radius"])
                else float(meta["frac_global_home_outside_radius"])
            ),
            n_neighbors_median=float(meta["n_neighbors_median"]),
        )
        if verbose:
            jx = juxta["juxta_match_rate"]
            jx_s = "nan" if jx != jx else f"{float(jx):.3f}"
            print(
                f"  {method:<18} dim={feats.shape[1]}  "
                f"argmin={row['argmin_accuracy']:.4f}  "
                f"n_eval={row['n_eval']}  "
                f"units_train={row['n_units_train']}  "
                f"juxta={jx_s}"
            )
        return row

    if verbose:
        print(f"\nValid events: {n_spikes} | units in slice: {n_units_all}")
        print(f"Home rule: {meta['home_rule']}")
        if pitches > 0.0:
            print(
                f"Pitch {meta['pitch_um']:.3f} um | radius {meta['radius_um']:.3f} um "
                f"| median electrodes in disk {meta['n_neighbors_median']:.1f} "
                f"| frac array-wide trough >radius "
                f"{meta['frac_global_home_outside_radius']:.3f}"
            )
        print("--- main_channel (1-D home-channel index) ---")
    rows.append(_eval_method("main_channel", main_ch, None))

    if verbose:
        print("--- main_channel_xy (2-D home-electrode coordinates) ---")
    rows.append(
        _eval_method(
            "main_channel_xy",
            main_xy,
            None,
            unique_xy=(
                int(np.unique(main_xy[:, 0]).size),
                int(np.unique(main_xy[:, 1]).size),
            ),
        )
    )

    if verbose:
        print("--- com float (2-D geometry-weighted centroid) ---")
    rows.append(
        _eval_method(
            "com_float",
            com,
            None,
            unique_xy=(int(np.unique(com[:, 0]).size), int(np.unique(com[:, 1]).size)),
        )
    )

    for bits in com_bits:
        codes = quantize_com(com, int(bits))
        uniq = (int(np.unique(codes[:, 0]).size), int(np.unique(codes[:, 1]).size))
        if verbose:
            print(f"--- com {bits}-bit (axis-wise, range [0,1] → 0..{ (1 << bits) - 1 }) ---")
        rows.append(_eval_method(f"com_{bits}bit", codes, int(bits), unique_xy=uniq))

    if "ks4_xy_norm" in prep:
        ks4_xy = prep["ks4_xy_norm"]
        if verbose:
            print("--- ks4_spike_xy (oracle location prior; not a COM feature) ---")
        rows.append(
            _eval_method(
                "ks4_spike_xy",
                ks4_xy,
                None,
                unique_xy=(
                    int(np.unique(ks4_xy[:, 0]).size),
                    int(np.unique(ks4_xy[:, 1]).size),
                ),
            )
        )

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_table(rows)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"\n{'method':<18} {'bits':>5} {'dim':>4} {'n_eval':>8} "
        f"{'argmin':>8} {'all_test':>8} {'juxta':>8}"
    )
    print("-" * 66)
    for r in rows:
        bits = "float" if int(r["n_bits"]) < 0 else str(int(r["n_bits"]))
        jx = r["juxta_match_rate"]
        jx_s = "nan" if jx != jx else f"{float(jx):.4f}"
        print(
            f"{r['method']:<18} {bits:>5} {r['feature_dim']:>4} {r['n_eval']:>8} "
            f"{r['argmin_accuracy']:>8.4f} {r['argmin_accuracy_all_test']:>8.4f} "
            f"{jx_s:>8}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct COM / main-channel / low-bit COM nearest-centroid eval"
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--npz",
        type=Path,
        default=YGER_KS4_NPZ_PATH,
        help="Yger KS4 Th=13 Spatial NPZ (HDD default).",
    )
    parser.add_argument("--good-only", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--metric", choices=("l2", "l1"), default="l2")
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=list(DEFAULT_COM_BITS),
        help="COM bit widths (float COM is always included).",
    )
    parser.add_argument("--juxta-tol-ms", type=float, default=JUXTA_TOL_MS)
    parser.add_argument(
        "--same-spike-max-pitches",
        type=float,
        default=0.0,
        help=(
            "Exclude extrema farther than this many channel pitches from the "
            "KS4 spike position (0 = legacy whole-array argmax + KNN). "
            "3 pitches on the 30 um Yger MEA is 90 um."
        ),
    )
    parser.add_argument(
        "--home-half-window",
        type=int,
        default=HOME_HALF_WINDOW,
        help="Samples on each side of the event for the local trough (radius path).",
    )
    parser.add_argument(
        "--ks4-dir",
        type=Path,
        default=DEFAULT_KS4_TH13_DIR,
        help="Kilosort4 output with spike_positions.npy (HDD; not copied to SSD).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise SystemExit(
            f"Yger KS4 NPZ not found: {npz_path}. "
            "Keep it on /mnt/data; do not copy the recording to the SSD."
        )

    ds = load_yger_ks4(
        path=npz_path, duration_s=args.duration, good_only=bool(args.good_only)
    )
    juxta_times, juxta_cluster = load_yger_juxta(npz_path, args.duration, ds.fs)
    juxta_tol_samples = max(1, int(round(float(args.juxta_tol_ms) * ds.fs / 1000.0)))

    seed_xy = None
    pitches = float(args.same_spike_max_pitches)
    if pitches > 0.0:
        ks4_dir = Path(args.ks4_dir)
        if not (ks4_dir / "spike_positions.npy").exists():
            raise SystemExit(
                f"KS4 spike_positions.npy not found in {ks4_dir}. "
                "Keep the sorting on /mnt/data; do not copy it to the SSD."
            )
        seed_xy = fill_missing_seed_xy(
            align_ks4_spike_positions(ds.spike_times, ds.spike_units, ks4_dir),
            ds.spike_units,
        )

    rows = run_study(
        ds,
        com_bits=tuple(int(b) for b in args.bits),
        k_neighbors=int(args.k_neighbors),
        window=int(args.window),
        train_frac=float(args.train_frac),
        metric=str(args.metric),
        duration_s=float(args.duration),
        good_only=bool(args.good_only),
        juxta_times=juxta_times,
        juxta_cluster=juxta_cluster,
        juxta_tol_samples=juxta_tol_samples,
        same_spike_max_pitches=pitches,
        home_half_window=int(args.home_half_window),
        seed_xy=seed_xy,
        verbose=not bool(args.quiet),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "com_mainchannel_lowbit.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": ds.name,
        "npz": str(npz_path),
        "good_only": bool(args.good_only),
        "duration_s": float(args.duration),
        "n_channels": int(ds.n_channels),
        "fs": int(ds.fs),
        "k_neighbors": int(args.k_neighbors),
        "window": int(args.window),
        "same_spike_max_pitches": float(args.same_spike_max_pitches),
        "home_half_window": int(args.home_half_window),
        "ks4_dir": str(args.ks4_dir),
        "train_frac": float(args.train_frac),
        "metric": str(args.metric),
        "filter_hz": [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        "filter_note": (
            "full-slice zero-phase Butterworth; non-causal GT-event isolation "
            "matching other Spatial DAC pilots"
        ),
        "labels": "KS4 Th=13 clusters (pseudo-GT); not juxta",
        "n_ks4_units_in_npz": 362,
        "n_ks4_good_in_npz": 314,
        "juxta_matched_cluster_id": int(juxta_cluster),
        "juxta_tol_ms": float(args.juxta_tol_ms),
        "juxta_tol_samples": int(juxta_tol_samples),
        "evidence_boundary": (
            "Oracle argmin accuracy of nearest-centroid assignment to KS4 "
            "cluster identities. Not candidate recall, not two-stage waveform "
            "refinement, not a STAR-Mem system-gate result."
        ),
        "rows": _json_safe(rows),
    }
    json_path = out_dir / "com_mainchannel_lowbit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
