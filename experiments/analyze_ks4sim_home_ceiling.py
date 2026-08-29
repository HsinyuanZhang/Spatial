"""Why is main_channel accuracy low on sim_no_drift: crowding vs far-spike theft.

    python -m Spatial.experiments.analyze_ks4sim_home_ceiling --duration 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    local_home_channels,
    median_nearest_neighbor_spacing,
)
from Spatial.data.ks4_sim import load_ks4_sim_no_drift
from Spatial.experiments.run_com_ks4sim_nodrift import (
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    _trough_homes,
    _unit_seed_xy,
)
from Spatial.experiments.run_com_mainchannel_lowbit import (
    HOME_HALF_WINDOW,
    WINDOW,
    oracle_argmin_assign,
)
from Spatial.algorithms.spatial_footprint import main_channel_features


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "ks4sim_no_drift_com"


def _majority_home(homes: np.ndarray, labels: np.ndarray, n_ch: int) -> np.ndarray:
    out = np.full(int(np.max(labels)) + 1, -1, dtype=np.int64)
    for u in np.unique(labels):
        ch = homes[labels == u].astype(np.int64)
        out[int(u)] = int(np.bincount(ch, minlength=n_ch).argmax())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--same-spike-max-pitches", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ds = load_ks4_sim_no_drift(duration_s=args.duration)
    filtered = bandpass_filter(ds.raw_data, ds.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ)
    window = WINDOW
    valid = (ds.spike_times >= window) & (ds.spike_times < ds.n_samples - window)
    times = ds.spike_times[valid]
    labels = ds.spike_units[valid]
    n_ch = ds.n_channels
    pitch = float(median_nearest_neighbor_spacing(ds.geom))
    radius = float(args.same_spike_max_pitches) * pitch

    homes_global = _trough_homes(filtered, times, HOME_HALF_WINDOW)
    unit_xy = _unit_seed_xy(filtered, times, labels, ds.geom)
    seed_xy = unit_xy[labels]
    homes_local, _ = local_home_channels(
        filtered, times, seed_xy, ds.geom, radius, half_window=HOME_HALF_WINDOW
    )

    maj = _majority_home(homes_local, labels, n_ch)
    true_home = maj[labels]
    dist_local = np.linalg.norm(ds.geom[homes_local] - ds.geom[true_home], axis=1)
    dist_global = np.linalg.norm(ds.geom[homes_global] - ds.geom[true_home], axis=1)

    # Crowding on the unit's own majority electrode.
    occ_units = np.bincount(maj[maj >= 0], minlength=n_ch)
    used = occ_units > 0
    # Spike-weighted majority share if every event used the unit's true home.
    spikes_on_true = np.bincount(true_home, minlength=n_ch)
    correct_if_true_home = np.zeros(n_ch, dtype=np.int64)
    for ch in np.flatnonzero(used):
        on_ch = true_home == ch
        counts = np.bincount(labels[on_ch])
        correct_if_true_home[ch] = int(counts.max())
    crowding_ceiling = float(correct_if_true_home.sum() / labels.size)

    # Among events whose *detected* 3-pitch home equals the unit majority home.
    on_true = homes_local == true_home
    if on_true.any():
        ch_hit = homes_local[on_true]
        y_hit = labels[on_true]
        ok = 0
        for ch in np.unique(ch_hit):
            sel = ch_hit == ch
            ok += int(np.bincount(y_hit[sel]).max())
        crowding_given_correct_home = float(ok / on_true.sum())
    else:
        crowding_given_correct_home = float("nan")

    assign_local = oracle_argmin_assign(
        main_channel_features(homes_local, n_ch), labels, times
    )
    assign_true = oracle_argmin_assign(
        main_channel_features(true_home, n_ch), labels, times
    )
    assign_global = oracle_argmin_assign(
        main_channel_features(homes_global, n_ch), labels, times
    )

    summary = {
        "n_events": int(times.size),
        "n_units": int(len(np.unique(labels))),
        "n_channels": int(n_ch),
        "mean_units_per_occupied_electrode": float(occ_units[used].mean()),
        "median_units_per_occupied_electrode": float(np.median(occ_units[used])),
        "max_units_on_one_electrode": int(occ_units.max()),
        "n_occupied_electrodes": int(used.sum()),
        "frac_units_sharing_an_electrode": float(np.mean(occ_units[maj[np.unique(labels)]] > 1)),
        "crowding_ceiling_if_every_spike_uses_unit_majority_home": crowding_ceiling,
        "frac_events_3pitch_home_equals_unit_majority": float(np.mean(on_true)),
        "frac_events_3pitch_home_gt_60um_from_majority": float(np.mean(dist_local > 60.0)),
        "frac_events_global_home_gt_60um_from_majority": float(np.mean(dist_global > 60.0)),
        "median_3pitch_home_error_um": float(np.median(dist_local)),
        "median_global_home_error_um": float(np.median(dist_global)),
        "crowding_ceiling_among_events_with_correct_3pitch_home": crowding_given_correct_home,
        "argmin_main_channel_3pitch_detected_home": float(assign_local["argmin_accuracy"]),
        "argmin_main_channel_oracle_unit_majority_home": float(assign_true["argmin_accuracy"]),
        "argmin_main_channel_global_trough": float(assign_global["argmin_accuracy"]),
        "chance": 1.0 / float(len(np.unique(labels))),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "home_crowding_vs_interference.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"Occupied electrodes: {summary['n_occupied_electrodes']}/{n_ch} | "
        f"units/site mean {summary['mean_units_per_occupied_electrode']:.2f} "
        f"median {summary['median_units_per_occupied_electrode']:.0f} "
        f"max {summary['max_units_on_one_electrode']}"
    )
    print(
        f"Share an electrode with another unit: "
        f"{summary['frac_units_sharing_an_electrode']:.1%}"
    )
    print(
        f"Crowding ceiling (perfect home = unit majority channel): "
        f"{crowding_ceiling:.4f}"
    )
    print(
        f"3-pitch home == majority: {summary['frac_events_3pitch_home_equals_unit_majority']:.3f} | "
        f">60 um: {summary['frac_events_3pitch_home_gt_60um_from_majority']:.3f} | "
        f"median error {summary['median_3pitch_home_error_um']:.1f} um"
    )
    print(
        f"Global trough >60 um from majority (far theft): "
        f"{summary['frac_events_global_home_gt_60um_from_majority']:.3f} | "
        f"median error {summary['median_global_home_error_um']:.1f} um"
    )
    print(
        f"Among events already on the majority channel, majority-unit share: "
        f"{crowding_given_correct_home:.4f}"
    )
    print(
        f"argmin main_channel: "
        f"oracle-home {assign_true['argmin_accuracy']:.4f} | "
        f"3-pitch detected {assign_local['argmin_accuracy']:.4f} | "
        f"global trough {assign_global['argmin_accuracy']:.4f}"
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
