"""COM / main-channel nearest-centroid on KS4-paper ``sim_no_drift``.

True simulator labels (1200 units), 384-ch Neuropixels, no probe drift.
Same isolation family as the Yger 3-pitch COM eval: GT times, zero-phase
bandpass, most-negative home in ±2 samples, COM on a 3-pitch disk around
that home. Not a STAR-Mem system-gate result.

    python -m Spatial.experiments.run_com_ks4sim_nodrift --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    com_features,
    extract_local_p2p,
    local_home_channels,
    main_channel_features,
    main_channel_xy_features,
    median_nearest_neighbor_spacing,
    quantize_com,
)
from Spatial.data.ks4_sim import load_ks4_sim_no_drift
from Spatial.experiments.run_com_mainchannel_lowbit import (
    HOME_HALF_WINDOW,
    WINDOW,
    oracle_argmin_assign,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "ks4sim_no_drift_com"
FILTER_LOW_HZ = 300.0
FILTER_HIGH_HZ = 6000.0
DEFAULT_PITCHES = 3.0
DEFAULT_BITS = (4, 5, 6, 8)


def _unit_seed_xy(
    filtered: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    geom: np.ndarray,
    half: int = HOME_HALF_WINDOW,
    max_spikes: int = 80,
) -> np.ndarray:
    """Per-unit electrode: mean trough over a spike subsample, then geom of that site.

    This is the analogue of Yger KS4 spike_positions: a stable location prior
    for the same-spike disk, not a classification feature.
    """
    n_ch = filtered.shape[0]
    n_samp = filtered.shape[1]
    seeds = np.zeros((int(np.max(labels)) + 1, 2), dtype=np.float64)
    rng = np.random.default_rng(0)
    for u in np.unique(labels):
        idx = np.flatnonzero(labels == u)
        if idx.size > max_spikes:
            idx = rng.choice(idx, size=max_spikes, replace=False)
        acc = np.zeros(n_ch, dtype=np.float64)
        for t in times[idx]:
            t0 = max(int(t) - half, 0)
            t1 = min(int(t) + half + 1, n_samp)
            acc += filtered[:, t0:t1].min(axis=1)
        seeds[int(u)] = geom[int(np.argmin(acc))]
    return seeds


def _trough_homes(filtered: np.ndarray, times: np.ndarray, half: int) -> np.ndarray:
    """Most-negative channel in ±half samples, vectorized over the slice."""
    n_ch, n_samp = filtered.shape
    trough = filtered.copy()
    for shift in range(-int(half), int(half) + 1):
        if shift == 0:
            continue
        rolled = np.empty_like(filtered)
        if shift > 0:
            rolled[:, shift:] = filtered[:, : n_samp - shift]
            rolled[:, :shift] = filtered[:, :shift]
        else:
            s = -shift
            rolled[:, : n_samp - s] = filtered[:, s:]
            rolled[:, n_samp - s :] = filtered[:, n_samp - s :]
        trough = np.minimum(trough, rolled)
    t = np.clip(np.asarray(times, dtype=np.int64), 0, n_samp - 1)
    return np.argmin(trough[:, t], axis=0).astype(np.int64)


def _unit_home_stats(homes: np.ndarray, labels: np.ndarray, geom: np.ndarray) -> dict:
    rows = []
    for u in np.unique(labels):
        ch = homes[labels == u].astype(np.int64)
        if ch.size == 0:
            continue
        counts = np.bincount(ch, minlength=geom.shape[0])
        maj = int(np.argmax(counts))
        dist = np.linalg.norm(geom[ch] - geom[maj], axis=1)
        rows.append(
            {
                "n": int(ch.size),
                "majority_frac": float(counts[maj] / ch.size),
                "n_unique": int(np.unique(ch).size),
                "rms_um": float(np.sqrt(np.mean(dist**2))),
            }
        )
    kept = [r for r in rows if r["n"] >= 20]
    xy = []
    for u in np.unique(labels):
        ch = homes[labels == u]
        if ch.size < 20:
            continue
        maj = int(np.bincount(ch, minlength=geom.shape[0]).argmax())
        xy.append(geom[maj])
    xy = np.asarray(xy, dtype=np.float64)
    nn = np.full(len(xy), np.nan)
    for i in range(len(xy)):
        d = np.linalg.norm(xy - xy[i], axis=1)
        d[i] = np.inf
        nn[i] = float(np.min(d))
    return {
        "n_units_ge20": len(kept),
        "majority_frac_median": float(np.median([r["majority_frac"] for r in kept])),
        "n_unique_homes_median": float(np.median([r["n_unique"] for r in kept])),
        "home_rms_um_median": float(np.median([r["rms_um"] for r in kept])),
        "home_nn_um_median": float(np.nanmedian(nn)) if nn.size else float("nan"),
        "home_nn_um_p10": float(np.nanpercentile(nn, 10)) if nn.size else float("nan"),
        "frac_home_nn_lt_pitch30": float(np.mean(nn < 30.0)) if nn.size else float("nan"),
    }


def prepare_ks4sim_com_events(
    duration_s: float,
    pitches: float = DEFAULT_PITCHES,
    cache_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Bandpass + 3-pitch COM/homes. Optional NPZ cache to skip the cbin."""
    if cache_path is not None and Path(cache_path).exists():
        payload = np.load(cache_path, allow_pickle=False)
        if verbose:
            print(f"Loaded COM cache: {cache_path}")
        return {k: payload[k] for k in payload.files}

    ds = load_ks4_sim_no_drift(duration_s=duration_s)
    if verbose:
        print(
            f"{ds.name}: {ds.n_channels}ch {ds.duration_s:.1f}s "
            f"units={len(ds.unit_ids)} events={len(ds.spike_times)}"
        )
    filtered = bandpass_filter(ds.raw_data, ds.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ)
    window = WINDOW
    valid = (ds.spike_times >= window) & (ds.spike_times < ds.n_samples - window)
    times = ds.spike_times[valid]
    labels = ds.spike_units[valid]
    unit_xy = _unit_seed_xy(filtered, times, labels, ds.geom)
    seed_xy = unit_xy[labels]
    pitch = float(median_nearest_neighbor_spacing(ds.geom))
    radius = float(pitches) * pitch
    homes, _ = local_home_channels(
        filtered, times, seed_xy, ds.geom, radius, half_window=HOME_HALF_WINDOW
    )
    table = build_radius_neighbor_table(ds.geom, radius)
    p2p, p2p_times, nbr = extract_local_p2p(filtered, times, homes, table, window=window)
    if not np.array_equal(p2p_times, times):
        raise AssertionError("P2P row alignment changed")
    com = com_features(p2p, nbr, ds.geom)
    geom_min = np.min(ds.geom, axis=0)
    geom_span = np.maximum(np.max(ds.geom, axis=0) - geom_min, 1e-8)
    com_um = com * geom_span + geom_min
    out = {
        "times": times,
        "labels": labels,
        "homes": homes,
        "com": com,
        "com_um": com_um,
        "geom": ds.geom,
        "pitch_um": np.array(pitch),
        "radius_um": np.array(radius),
        "k_table": np.array(table.shape[1]),
        "n_channels": np.array(ds.n_channels),
        "duration_s": np.array(duration_s),
    }
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
        if verbose:
            print(f"Saved COM cache: {cache_path}")
    return out


def run(duration_s: float, pitches: float, out_dir: Path, verbose: bool = True) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"events_{int(duration_s)}s.npz"
    prep = prepare_ks4sim_com_events(
        duration_s, pitches=pitches, cache_path=cache_path, verbose=verbose
    )
    times = prep["times"]
    labels = prep["labels"]
    homes = prep["homes"]
    com = prep["com"]
    geom = prep["geom"]
    pitch = float(np.asarray(prep["pitch_um"]))
    radius = float(np.asarray(prep["radius_um"]))
    n_channels = int(np.asarray(prep["n_channels"]))
    table_k = int(np.asarray(prep["k_table"]))
    main_ch = main_channel_features(homes, n_channels)
    main_xy = main_channel_xy_features(homes, geom)
    home_stats = _unit_home_stats(homes, labels, geom)

    rows = []

    def _eval(method: str, feats: np.ndarray, n_bits: int) -> dict:
        assign = oracle_argmin_assign(feats, labels, times)
        row = {
            "dataset": "ks4sim_no_drift",
            "duration_s": float(duration_s),
            "method": method,
            "n_bits": int(n_bits),
            "feature_dim": int(feats.shape[1]),
            "n_spikes": int(times.size),
            "n_eval": int(assign["n_eval"]),
            "n_correct": int(assign["n_correct"]),
            "argmin_accuracy": float(assign["argmin_accuracy"]),
            "n_units_all": int(len(np.unique(labels))),
            "n_units_train": int(assign["n_units_train"]),
            "n_units_unseen_test": int(assign["n_units_unseen_test"]),
            "pitch_um": pitch,
            "radius_um": radius,
            "k_table": table_k,
        }
        if verbose:
            print(
                f"  {method:<18} dim={feats.shape[1]}  "
                f"argmin={row['argmin_accuracy']:.4f}  "
                f"n_eval={row['n_eval']}  units_train={row['n_units_train']}"
            )
        return row

    if verbose:
        print(
            f"Pitch {pitch:.2f} um | 3-pitch radius {radius:.2f} um | "
            f"table width {table_k}"
        )
        print(
            f"Home stability (units>=20): majority {home_stats['majority_frac_median']:.3f} "
            f"| unique homes med {home_stats['n_unique_homes_median']:.1f} "
            f"| RMS {home_stats['home_rms_um_median']:.2f} um "
            f"| centroid NN {home_stats['home_nn_um_median']:.2f} um"
        )
    rows.append(_eval("main_channel", main_ch, -1))
    rows.append(_eval("main_channel_xy", main_xy, -1))
    rows.append(_eval("com_float", com, -1))
    for bits in DEFAULT_BITS:
        rows.append(_eval(f"com_{bits}bit", quantize_com(com, bits), bits))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "com_ks4sim_no_drift.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "ks4sim_no_drift",
        "duration_s": float(duration_s),
        "n_channels": n_channels,
        "fs": 30000,
        "labels": "simulator GT cluster ids (1200 units), not Kilosort",
        "home_rule": (
            "unit location = geom of the channel with the most-negative mean "
            "trough over a 80-spike subsample; event home = most-negative "
            "site in a 3-pitch disk around that location"
        ),
        "filter_hz": [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        "home_stats": home_stats,
        "rows": rows,
        "evidence_boundary": (
            "Oracle argmin on true simulator identities. Not candidate recall, "
            "not a STAR-Mem system-gate result."
        ),
    }
    json_path = out_dir / "com_ks4sim_no_drift_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if verbose:
        print(f"Saved: {csv_path}")
        print(f"Saved: {json_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--same-spike-max-pitches", type=float, default=DEFAULT_PITCHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(
        float(args.duration),
        float(args.same_spike_max_pitches),
        Path(args.output_dir),
        verbose=True,
    )


if __name__ == "__main__":
    main()
