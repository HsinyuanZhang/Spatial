"""Absolute-gain signed ADC (5/6-bit) then P2P → COM on the Yger 3-pitch disk.

This is the hardware-like path: one fixed scale for all channels, or one
scale per channel, then a uniform signed ADC on the filtered waveform.
It is not per-event AGC (``per_spike_max``).

    python -m Spatial.experiments.analyze_adc_absolute_com --duration 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    com_features,
    extract_local_p2p,
    quantize_com,
    quantize_signed_adc,
)
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR
from Spatial.experiments.run_com_mainchannel_lowbit import (
    WINDOW,
    align_ks4_spike_positions,
    fill_missing_seed_xy,
    oracle_argmin_assign,
    prepare_com_mainchannel_events,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "adc_absolute_com"


def _com_xy_um(com_norm: np.ndarray, geom: np.ndarray) -> np.ndarray:
    g = np.asarray(geom, dtype=np.float64)
    span = np.max(g, axis=0) - np.min(g, axis=0)
    return np.asarray(com_norm, dtype=np.float64) * span + np.min(g, axis=0)


def _event_abs_scales(
    filtered: np.ndarray,
    times: np.ndarray,
    window: int,
    rng: np.random.Generator,
    n_percentile_events: int = 8000,
) -> dict[str, np.ndarray | float]:
    n_ch, _ = filtered.shape
    w = int(window)
    times = np.asarray(times, dtype=np.int64).ravel()
    ch_max = np.zeros(n_ch, dtype=np.float64)
    global_max = 0.0
    for t in times:
        sl = np.abs(filtered[:, int(t) - w : int(t) + w])
        ch_max = np.maximum(ch_max, sl.max(axis=1))
        peak = float(sl.max())
        if peak > global_max:
            global_max = peak
    n_take = min(int(n_percentile_events), times.size)
    pick = rng.choice(times.size, size=n_take, replace=False)
    snips = np.empty((n_take, n_ch, 2 * w), dtype=np.float64)
    for j, i in enumerate(pick):
        t = int(times[i])
        snips[j] = filtered[:, t - w : t + w]
    abs_snips = np.abs(snips)
    ch_p999 = np.percentile(abs_snips, 99.9, axis=(0, 2))
    global_p999 = float(np.percentile(abs_snips, 99.9))
    ch_max = np.maximum(ch_max, 1e-8)
    ch_p999 = np.maximum(ch_p999, 1e-8)
    return {
        "global_max": float(max(global_max, 1e-8)),
        "global_p999": float(max(global_p999, 1e-8)),
        "channel_max": ch_max,
        "channel_p999": ch_p999,
        "n_percentile_events": int(n_take),
    }


def _clip_frac(
    original: np.ndarray,
    full_scale: float | np.ndarray,
    times: np.ndarray,
    window: int,
    rng: np.random.Generator,
    n_take: int = 4000,
) -> float:
    w = int(window)
    times = np.asarray(times, dtype=np.int64).ravel()
    take = min(int(n_take), times.size)
    pick = rng.choice(times.size, size=take, replace=False)
    fs = np.asarray(full_scale, dtype=np.float64)
    n_clip = 0
    n_tot = 0
    for i in pick:
        t = int(times[i])
        sl = original[:, t - w : t + w]
        if fs.ndim == 0:
            n_clip += int(np.sum(np.abs(sl) > float(fs) * (1.0 + 1e-12)))
        else:
            n_clip += int(np.sum(np.abs(sl) > fs[:, None] * (1.0 + 1e-12)))
        n_tot += sl.size
    return float(n_clip / max(n_tot, 1))


def _row(
    com_hat: np.ndarray,
    com_float: np.ndarray,
    geom: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    com_bits: int,
) -> dict:
    span = np.max(geom, axis=0) - np.min(geom, axis=0)
    err = _com_xy_um(com_hat, geom) - _com_xy_um(com_float, geom)
    rms = np.sqrt(np.mean(err**2, axis=0))
    euclid = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    bin_um = span / float((1 << int(com_bits)) - 1)
    q_hat = quantize_com(com_hat, com_bits)
    q_ref = quantize_com(com_float, com_bits)
    assign = oracle_argmin_assign(com_hat, labels, times)
    assign_q = oracle_argmin_assign(q_hat, labels, times)
    sigma = np.maximum(rms, 1e-12)
    bits_eq = np.log2(span / (sigma * np.sqrt(12.0)))
    return {
        "rms_um": [float(rms[0]), float(rms[1])],
        "rms_euclid_um": euclid,
        "median_abs_um": [
            float(np.median(np.abs(err[:, 0]))),
            float(np.median(np.abs(err[:, 1]))),
        ],
        "equivalent_bits_vs_probe_bbox": [float(bits_eq[0]), float(bits_eq[1])],
        "frac_same_com_code": float(np.mean(np.all(q_hat == q_ref, axis=1))),
        "frac_error_within_half_com_bin": float(
            np.mean(np.all(np.abs(err) <= 0.5 * bin_um, axis=1))
        ),
        "argmin_accuracy": float(assign["argmin_accuracy"]),
        "argmin_accuracy_com_%dbit" % com_bits: float(assign_q["argmin_accuracy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--same-spike-max-pitches", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--com-bits", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ds = load_yger_ks4(path=YGER_KS4_NPZ_PATH, duration_s=args.duration)
    seed_xy = fill_missing_seed_xy(
        align_ks4_spike_positions(ds.spike_times, ds.spike_units, DEFAULT_KS4_TH13_DIR),
        ds.spike_units,
    )
    prep = prepare_com_mainchannel_events(
        ds,
        window=int(args.window),
        same_spike_max_pitches=float(args.same_spike_max_pitches),
        seed_xy=seed_xy,
        return_filtered=True,
    )
    filtered = prep["filtered"]
    times = prep["times"]
    labels = prep["labels"]
    homes = prep["central"]
    com_float = prep["com"]
    meta = prep["meta"]
    geom = ds.geom
    table = build_radius_neighbor_table(geom, float(meta["radius_um"]))
    rng = np.random.default_rng(0)
    scales = _event_abs_scales(filtered, times, int(args.window), rng)

    configs = [
        ("all_channel_max", 5, scales["global_max"]),
        ("all_channel_max", 6, scales["global_max"]),
        ("all_channel_p999", 5, scales["global_p999"]),
        ("all_channel_p999", 6, scales["global_p999"]),
        ("per_channel_max", 5, scales["channel_max"]),
        ("per_channel_max", 6, scales["channel_max"]),
        ("per_channel_p999", 5, scales["channel_p999"]),
        ("per_channel_p999", 6, scales["channel_p999"]),
    ]

    rows: dict[str, dict] = {}
    for name, n_bits, fs in configs:
        recon = quantize_signed_adc(filtered, int(n_bits), fs)
        p2p, p2p_times, nbr = extract_local_p2p(
            recon, times, homes, table, window=int(args.window)
        )
        if not np.array_equal(p2p_times, times):
            raise AssertionError("ADC P2P row alignment changed")
        com_hat = com_features(p2p, nbr, geom)
        key = f"{name}_{n_bits}bit_adc"
        rec = _row(com_hat, com_float, geom, labels, times, int(args.com_bits))
        rec["adc_bits"] = int(n_bits)
        rec["scale"] = name
        rec["full_scale"] = (
            float(fs) if np.ndim(fs) == 0 else "per_channel_vector"
        )
        rec["frac_event_window_samples_clipped"] = _clip_frac(
            filtered, fs, times, int(args.window), rng
        )
        rec["p2p_event_max_median"] = float(np.median(np.max(p2p, axis=1)))
        rec["p2p_event_max_p99"] = float(np.percentile(np.max(p2p, axis=1), 99))
        rows[key] = rec
        del recon

    ch_max = np.asarray(scales["channel_max"])
    summary = {
        "dataset": ds.name,
        "duration_s": float(args.duration),
        "n_events": int(times.size),
        "same_spike_max_pitches": float(args.same_spike_max_pitches),
        "adc": "signed two's-complement on bandpass waveform; P2P from reconstructed samples",
        "homes": "unchanged from analog 3-pitch local trough (isolation not re-detected)",
        "full_scale_from": "max |sample| (or p99.9 of a 8000-event subsample) in ±15-sample event windows",
        "analog_event_window": {
            "global_max_abs": float(scales["global_max"]),
            "global_p999_abs": float(scales["global_p999"]),
            "channel_max_median": float(np.median(ch_max)),
            "channel_max_hottest": float(np.max(ch_max)),
            "hottest_over_median_channel": float(np.max(ch_max) / np.median(ch_max)),
        },
        "argmin_accuracy_analog_p2p_com_float": float(
            oracle_argmin_assign(com_float, labels, times)["argmin_accuracy"]
        ),
        "rows": rows,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adc_absolute_com.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"Analog |x| in event windows: max={scales['global_max']:.1f}  "
        f"p99.9={scales['global_p999']:.1f}  "
        f"hottest/median ch={summary['analog_event_window']['hottest_over_median_channel']:.2f}x"
    )
    print(
        f"{'config':<28} {'ADC':>3} {'clip':>7} {'RMS um':>8} {'eq.bit':>7} "
        f"{'same6b':>7} {'argmin':>8} {'6b argmin':>9}"
    )
    analog = summary["argmin_accuracy_analog_p2p_com_float"]
    print(f"{'analog P2P (no ADC)':<28} {'—':>3} {'0':>7} {0.0:8.3f} {'∞':>7} {1.0:7.3f} {analog:8.4f}")
    for key, rec in rows.items():
        eq = 0.5 * (
            rec["equivalent_bits_vs_probe_bbox"][0]
            + rec["equivalent_bits_vs_probe_bbox"][1]
        )
        print(
            f"{rec['scale']:<28} {rec['adc_bits']:>3} "
            f"{rec['frac_event_window_samples_clipped']:7.4f} "
            f"{rec['rms_euclid_um']:8.3f} {eq:7.2f} "
            f"{rec['frac_same_com_code']:7.3f} "
            f"{rec['argmin_accuracy']:8.4f} "
            f"{rec['argmin_accuracy_com_%dbit' % args.com_bits]:9.4f}"
        )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
