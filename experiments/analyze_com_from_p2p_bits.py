"""How many COM bits survive when P2P is quantized (Yger 3-pitch disk).

COM is a weighted average of known electrode coordinates. P2P bit-width
quantizes the weights; COM bit-width quantizes the two bbox-normalized
centroid axes. This script measures the COM error caused by 5-bit P2P
and whether that error is smaller than a 6-bit COM bin.

    python -m Spatial.experiments.analyze_com_from_p2p_bits --duration 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Spatial.algorithms.spatial_footprint import com_features, quantize_com, quantize_p2p
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR
from Spatial.experiments.run_com_mainchannel_lowbit import (
    align_ks4_spike_positions,
    fill_missing_seed_xy,
    oracle_argmin_assign,
    prepare_com_mainchannel_events,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "com_from_5bit_p2p"


def _com_xy_um(com_norm: np.ndarray, geom: np.ndarray) -> np.ndarray:
    g = np.asarray(geom, dtype=np.float64)
    span = np.max(g, axis=0) - np.min(g, axis=0)
    return np.asarray(com_norm, dtype=np.float64) * span + np.min(g, axis=0)


def _pct(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("min", "p10", "p50", "p90", "p99", "p99_9", "max")}
    qs = np.percentile(x, [0, 10, 50, 90, 99, 99.9, 100])
    return {
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p99": float(qs[4]),
        "p99_9": float(qs[5]),
        "max": float(qs[6]),
    }


def _valid_amp(p2p: np.ndarray, nbr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(nbr, dtype=np.int64) >= 0
    return np.asarray(p2p, dtype=np.float64)[valid], np.asarray(nbr, dtype=np.int64)[valid]


def _channel_scales(
    p2p: np.ndarray,
    nbr: np.ndarray,
    n_channels: int,
    percentile: float | None = None,
) -> np.ndarray:
    amp, ch = _valid_amp(p2p, nbr)
    scales = np.zeros(int(n_channels), dtype=np.float64)
    if percentile is None:
        np.maximum.at(scales, ch, amp)
        return scales
    for i in range(int(n_channels)):
        vals = amp[ch == i]
        if vals.size:
            scales[i] = float(np.percentile(vals, percentile))
    return scales


def _quantize_per_channel(
    p2p: np.ndarray,
    nbr: np.ndarray,
    scales: np.ndarray,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (integer codes, dequantized amplitudes) with a per-channel scale."""
    amp = np.maximum(np.asarray(p2p, dtype=np.float64), 0.0)
    ids = np.asarray(nbr, dtype=np.int64)
    valid = ids >= 0
    max_level = float((1 << int(n_bits)) - 1)
    safe_ids = np.where(valid, ids, 0)
    scale = np.maximum(scales[safe_ids], 1e-8)
    codes = np.clip(np.rint(amp / scale * max_level), 0.0, max_level)
    codes = np.where(valid, codes, 0.0)
    dequant = np.where(valid, codes / max_level * scale, 0.0)
    return codes, dequant


def _com_error_row(
    com_hat: np.ndarray,
    com_float: np.ndarray,
    geom: np.ndarray,
    span: np.ndarray,
    bin_um: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    com_bits: int,
) -> dict:
    err_um = _com_xy_um(com_hat, geom) - _com_xy_um(com_float, geom)
    q_hat = quantize_com(com_hat, com_bits)
    q_ref = quantize_com(com_float, com_bits)
    assign = oracle_argmin_assign(com_hat, labels, times)
    assign_q = oracle_argmin_assign(q_hat, labels, times)
    stats = _axis_stats(err_um, span)
    stats["frac_same_com_code"] = float(np.mean(np.all(q_hat == q_ref, axis=1)))
    stats["frac_error_within_half_bin"] = float(
        np.mean(np.all(np.abs(err_um) <= 0.5 * bin_um, axis=1))
    )
    stats["argmin_accuracy"] = float(assign["argmin_accuracy"])
    stats["argmin_accuracy_com_quantized"] = float(assign_q["argmin_accuracy"])
    return stats


def _axis_stats(err_um: np.ndarray, span_um: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(err_um**2, axis=0))
    med = np.median(np.abs(err_um), axis=0)
    p90 = np.percentile(np.abs(err_um), 90, axis=0)
    # Uniform-quantizer equivalent bits: LSB = σ√12, n_levels = span / LSB.
    sigma = np.maximum(rms, 1e-12)
    bits_eq = np.log2(span_um / (sigma * np.sqrt(12.0)))
    return {
        "rms_um": [float(rms[0]), float(rms[1])],
        "median_abs_um": [float(med[0]), float(med[1])],
        "p90_abs_um": [float(p90[0]), float(p90[1])],
        "rms_euclid_um": float(np.sqrt(np.mean(np.sum(err_um**2, axis=1)))),
        "equivalent_bits_vs_probe_bbox": [float(bits_eq[0]), float(bits_eq[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--p2p-bits", type=int, default=5)
    parser.add_argument("--com-bits", type=int, default=6)
    parser.add_argument("--same-spike-max-pitches", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ds = load_yger_ks4(path=YGER_KS4_NPZ_PATH, duration_s=args.duration)
    seed_xy = fill_missing_seed_xy(
        align_ks4_spike_positions(ds.spike_times, ds.spike_units, DEFAULT_KS4_TH13_DIR),
        ds.spike_units,
    )
    prep = prepare_com_mainchannel_events(
        ds,
        same_spike_max_pitches=float(args.same_spike_max_pitches),
        seed_xy=seed_xy,
    )
    p2p = prep["p2p"]
    nbr = prep["neighbor_ids"]
    com_float = prep["com"]
    labels = prep["labels"]
    times = prep["times"]
    geom = ds.geom
    span = np.max(geom, axis=0) - np.min(geom, axis=0)
    p2p_bits = int(args.p2p_bits)
    com_bits = int(args.com_bits)
    max_p2p = (1 << p2p_bits) - 1
    max_com = (1 << com_bits) - 1
    bin_um = span / float(max_com)

    valid_amp, valid_ch = _valid_amp(p2p, nbr)
    event_max = np.max(np.maximum(p2p, 0.0), axis=1)
    home_p2p = np.maximum(p2p[:, 0], 0.0)
    global_max = float(np.max(valid_amp)) if valid_amp.size else 0.0
    global_p99 = float(np.percentile(valid_amp, 99)) if valid_amp.size else 0.0
    peak_code_global = np.clip(np.rint(event_max / max(global_max, 1e-8) * max_p2p), 0, max_p2p)
    ch_max = _channel_scales(p2p, nbr, ds.n_channels, percentile=None)
    ch_p99 = _channel_scales(p2p, nbr, ds.n_channels, percentile=99.0)
    used_ch = ch_max > 0.0

    modes: dict[str, np.ndarray] = {}
    codes_spike = quantize_p2p(p2p, p2p_bits, mode="per_spike_max")
    modes["per_spike_max_codes_as_weights"] = com_features(codes_spike, nbr, geom)
    codes_global = quantize_p2p(p2p, p2p_bits, mode="global_max")
    modes["global_max_codes_as_weights"] = com_features(codes_global, nbr, geom)
    codes_gp99 = quantize_p2p(p2p, p2p_bits, mode="lsb", lsb=global_p99 / max_p2p)
    modes["global_p99_lsb_codes_as_weights"] = com_features(codes_gp99, nbr, geom)
    ch_codes, ch_deq = _quantize_per_channel(p2p, nbr, ch_max, p2p_bits)
    modes["per_channel_max_codes_as_weights"] = com_features(ch_codes, nbr, geom)
    modes["per_channel_max_dequantized"] = com_features(ch_deq, nbr, geom)
    _, ch_p99_deq = _quantize_per_channel(p2p, nbr, ch_p99, p2p_bits)
    modes["per_channel_p99_dequantized"] = com_features(ch_p99_deq, nbr, geom)

    scale_rows = {
        name: _com_error_row(
            com_hat, com_float, geom, span, bin_um, labels, times, com_bits
        )
        for name, com_hat in modes.items()
    }
    assign_float = oracle_argmin_assign(com_float, labels, times)
    assign_float_q = oracle_argmin_assign(quantize_com(com_float, com_bits), labels, times)

    summary = {
        "dataset": ds.name,
        "duration_s": float(args.duration),
        "n_events": int(times.size),
        "same_spike_max_pitches": float(args.same_spike_max_pitches),
        "p2p_bits": p2p_bits,
        "com_bits": com_bits,
        "probe_bbox_um": [float(span[0]), float(span[1])],
        "com_bin_um": [float(bin_um[0]), float(bin_um[1])],
        "previous_claim_used": "per_spike_max (each event / its own neighborhood max); not per-channel, not recording-wide",
        "p2p_amplitude_outliers": {
            "units": "filtered P2P on 3-pitch disk, ADC counts after int16→float",
            "all_valid_samples": _pct(valid_amp),
            "event_max": _pct(event_max),
            "home_channel": _pct(home_p2p),
            "global_max_over_median_event_max": float(global_max / max(np.median(event_max), 1e-8)),
            "global_max_over_p99_event_max": float(global_max / max(np.percentile(event_max, 99), 1e-8)),
            "frac_events_peak_uses_le_3_of_31_global_levels": float(np.mean(peak_code_global <= 3)),
            "frac_events_peak_uses_le_7_of_31_global_levels": float(np.mean(peak_code_global <= 7)),
            "frac_events_peak_saturates_global_5bit": float(np.mean(peak_code_global >= max_p2p)),
            "median_peak_code_under_global_max": float(np.median(peak_code_global)),
            "frac_home_is_event_max": float(np.mean(home_p2p >= event_max - 1e-12)),
        },
        "per_channel_scale_outliers": {
            "n_channels_with_any_p2p": int(used_ch.sum()),
            "channel_max": _pct(ch_max[used_ch]),
            "channel_p99": _pct(ch_p99[used_ch]),
            "hottest_over_median_channel_max": float(
                np.max(ch_max[used_ch]) / max(np.median(ch_max[used_ch]), 1e-8)
            ),
            "hottest_channel": int(np.argmax(ch_max)),
            "hottest_channel_max": float(np.max(ch_max)),
        },
        "scale_modes": scale_rows,
        "argmin_accuracy_float_p2p": {
            "com_float": float(assign_float["argmin_accuracy"]),
            "com_%dbit" % com_bits: float(assign_float_q["argmin_accuracy"]),
        },
        "note": (
            "per_spike_max: one scale per event (neighborhood max). "
            "global_max: one scale for the whole recording. "
            "per_channel: one scale per electrode. COM needs commensurate "
            "physical amplitudes, so per-channel codes must be dequantized "
            "before the weighted average."
        ),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "com_from_p2p_bits.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    amp = summary["p2p_amplitude_outliers"]
    print(
        f"Probe bbox {span[0]:.1f} x {span[1]:.1f} um | "
        f"{com_bits}-bit COM bin {bin_um[0]:.2f} x {bin_um[1]:.2f} um"
    )
    em = amp["event_max"]
    print(
        f"Event-max P2P: p50={em['p50']:.1f}  p90={em['p90']:.1f}  "
        f"p99={em['p99']:.1f}  p99.9={em['p99_9']:.1f}  max={em['max']:.1f}  "
        f"max/median={amp['global_max_over_median_event_max']:.1f}x"
    )
    print(
        f"Global 5-bit: median peak code {amp['median_peak_code_under_global_max']:.0f}/31 | "
        f"peak<=3 levels {amp['frac_events_peak_uses_le_3_of_31_global_levels']:.3f} | "
        f"peak<=7 {amp['frac_events_peak_uses_le_7_of_31_global_levels']:.3f} | "
        f"saturate {amp['frac_events_peak_saturates_global_5bit']:.4f}"
    )
    chs = summary["per_channel_scale_outliers"]
    print(
        f"Per-channel max: hottest ch {chs['hottest_channel']} = {chs['hottest_channel_max']:.1f} | "
        f"hottest/median channel {chs['hottest_over_median_channel_max']:.1f}x"
    )
    print(
        f"{'mode':<38} {'RMS um':>8} {'eq.bits':>8} {'same6b':>7} {'argmin':>8}"
    )
    for name, row in scale_rows.items():
        eq = 0.5 * (
            row["equivalent_bits_vs_probe_bbox"][0]
            + row["equivalent_bits_vs_probe_bbox"][1]
        )
        print(
            f"{name:<38} {row['rms_euclid_um']:8.3f} {eq:8.2f} "
            f"{row['frac_same_com_code']:7.3f} {row['argmin_accuracy']:8.4f}"
        )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
