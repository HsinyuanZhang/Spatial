"""Low-bit COM on the Neuropixels ``sim_no_drift`` 1-D probe.

Full-probe bbox quantization wastes bits: 6-bit over ~3.8 mm is a 61 µm
bin, coarser than the 20 µm pitch. This sweep keeps COM along the shank
(y) and spends 4/5/6/8 bits on either the full probe or a local window
around the home electrode. Also reports a COM_y radius candidate gate.

    python -m Spatial.experiments.analyze_np_com_lowbit --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    main_channel_features,
    quantize_com,
)
from Spatial.experiments.analyze_com_radius_prefilter import (
    evaluate_com_radius_prefilter,
)
from Spatial.experiments.run_com_ks4sim_nodrift import (
    DEFAULT_OUT,
    prepare_ks4sim_com_events,
)
from Spatial.experiments.run_com_mainchannel_lowbit import oracle_argmin_assign


BITS = (4, 5, 6, 8)
LOCAL_HALF_UM = (30.0, 60.0, 90.0)
RADIUS_UM = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0)


def quantize_span(
    values: np.ndarray,
    n_bits: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Unsigned n-bit codes of a 1-D axis over [lo, hi]."""
    x = np.asarray(values, dtype=np.float64).ravel()
    max_level = float((1 << int(n_bits)) - 1)
    span = max(float(hi) - float(lo), 1e-8)
    codes = np.clip(np.rint((x - float(lo)) / span * max_level), 0.0, max_level)
    return codes[:, None]


def _eval(name: str, feats: np.ndarray, labels, times, n_bits: int, note: str) -> dict:
    assign = oracle_argmin_assign(feats, labels, times)
    return {
        "method": name,
        "n_bits": int(n_bits),
        "feature_dim": int(feats.shape[1]),
        "n_eval": int(assign["n_eval"]),
        "n_correct": int(assign["n_correct"]),
        "argmin_accuracy": float(assign["argmin_accuracy"]),
        "n_units_train": int(assign["n_units_train"]),
        "note": note,
    }


def _com_y_diagnostics(com_y: np.ndarray, labels: np.ndarray) -> dict:
    cents = []
    intra = []
    for u in np.unique(labels):
        y = com_y[labels == u]
        if y.size < 20:
            continue
        c = float(np.mean(y))
        cents.append(c)
        intra.append(float(np.sqrt(np.mean((y - c) ** 2))))
    cents = np.asarray(cents)
    nn = np.full(cents.size, np.nan)
    for i, c in enumerate(cents):
        d = np.abs(cents - c)
        d[i] = np.inf
        nn[i] = float(np.min(d))
    return {
        "n_units": int(cents.size),
        "intra_rms_um_median": float(np.median(intra)) if intra else float("nan"),
        "intra_rms_um_p90": float(np.percentile(intra, 90)) if intra else float("nan"),
        "centroid_nn_um_median": float(np.nanmedian(nn)) if nn.size else float("nan"),
        "centroid_nn_um_p10": float(np.nanpercentile(nn, 10)) if nn.size else float("nan"),
        "frac_centroid_nn_lt_20um": float(np.mean(nn < 20.0)) if nn.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    cache = out_dir / f"events_{int(args.duration)}s.npz"
    prep = prepare_ks4sim_com_events(
        float(args.duration), cache_path=cache, verbose=True
    )
    times = prep["times"]
    labels = prep["labels"]
    homes = prep["homes"]
    com = prep["com"]
    com_um = prep["com_um"]
    geom = prep["geom"]
    n_ch = int(np.asarray(prep["n_channels"]))
    y_lo = float(np.min(geom[:, 1]))
    y_hi = float(np.max(geom[:, 1]))
    y_span = y_hi - y_lo
    home_y = geom[homes.astype(np.int64), 1]
    com_y = com_um[:, 1]
    rel_y = com_y - home_y

    rows: list[dict] = []
    main_ch = main_channel_features(homes, n_ch)
    rows.append(_eval("main_channel", main_ch, labels, times, -1, "channel index"))
    rows.append(_eval("com_xy_bbox_float", com, labels, times, -1, "full-probe bbox COM"))
    rows.append(
        _eval("com_y_float", com_y[:, None], labels, times, -1, "shank-axis COM in um")
    )
    rows.append(
        _eval(
            "com_y_rel_float",
            rel_y[:, None],
            labels,
            times,
            -1,
            "COM_y minus home electrode y",
        )
    )
    rows.append(
        _eval(
            "home_plus_rel_y_float",
            np.column_stack([homes.astype(np.float64), rel_y]),
            labels,
            times,
            -1,
            "home index + residual COM_y (um)",
        )
    )

    for bits in BITS:
        rows.append(
            _eval(
                f"com_xy_bbox_{bits}bit",
                quantize_com(com, bits),
                labels,
                times,
                bits,
                f"2-D bbox, {y_span / ((1 << bits) - 1):.1f} um y-bin",
            )
        )
        y_full = quantize_span(com_y, bits, y_lo, y_hi)
        rows.append(
            _eval(
                f"com_y_full_{bits}bit",
                y_full,
                labels,
                times,
                bits,
                f"1-D y over {y_span:.0f} um, bin {y_span / ((1 << bits) - 1):.1f} um",
            )
        )
        for half in LOCAL_HALF_UM:
            local = quantize_span(rel_y, bits, -half, half)
            rows.append(
                _eval(
                    f"com_y_rel_pm{half:.0f}_{bits}bit",
                    local,
                    labels,
                    times,
                    bits,
                    f"residual y in ±{half:.0f} um, bin {2 * half / ((1 << bits) - 1):.2f} um",
                )
            )
            hybrid = np.column_stack([homes.astype(np.float64), local.ravel()])
            rows.append(
                _eval(
                    f"home_plus_rel_pm{half:.0f}_{bits}bit",
                    hybrid,
                    labels,
                    times,
                    bits,
                    "home index + local residual COM_y",
                )
            )

    diag = _com_y_diagnostics(com_y, labels)
    radius_rows = []
    for r in RADIUS_UM:
        rec = evaluate_com_radius_prefilter(com_um, labels, times, radius_um=float(r))
        rec["source"] = "com_xy_um"
        radius_rows.append(
            {
                "radius_um": float(r),
                "candidate_recall": rec["candidate_recall"],
                "mean_candidates": rec["mean_candidates"],
                "frac_units_dropped": rec["frac_units_dropped"],
                "traffic_reduction_x": rec["traffic_reduction_x"],
                "argmin_in_candidates": rec["argmin_in_candidates"],
            }
        )
    # 1-D y-only ball: embed as (0, y)
    com_y2 = np.column_stack([np.zeros(com_y.size), com_y])
    radius_y_rows = []
    for r in RADIUS_UM:
        rec = evaluate_com_radius_prefilter(com_y2, labels, times, radius_um=float(r))
        radius_y_rows.append(
            {
                "radius_um": float(r),
                "candidate_recall": rec["candidate_recall"],
                "mean_candidates": rec["mean_candidates"],
                "frac_units_dropped": rec["frac_units_dropped"],
                "traffic_reduction_x": rec["traffic_reduction_x"],
                "argmin_in_candidates": rec["argmin_in_candidates"],
            }
        )

    def _pick(rows_r: list[dict], min_recall: float) -> dict | None:
        ok = [r for r in rows_r if r["candidate_recall"] >= min_recall]
        if not ok:
            return None
        return min(ok, key=lambda r: r["mean_candidates"])

    summary = {
        "dataset": "ks4sim_no_drift",
        "duration_s": float(args.duration),
        "y_span_um": y_span,
        "com_y_diagnostics": diag,
        "unique_id": rows,
        "radius_com_xy": radius_rows,
        "radius_com_y": radius_y_rows,
        "selected_xy_r99": _pick(radius_rows, 0.99),
        "selected_xy_r95": _pick(radius_rows, 0.95),
        "selected_y_r99": _pick(radius_y_rows, 0.99),
        "selected_y_r95": _pick(radius_y_rows, 0.95),
        "note": (
            "Full-probe N-bit COM_y bins are span/(2^N-1). Local residual "
            "quantizes COM_y-home_y over ±H um. Hybrid is (home index, local code)."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "np_com_lowbit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    rad_path = out_dir / "np_com_radius.csv"
    rad_fields = list(radius_rows[0].keys()) + ["axis"]
    with rad_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rad_fields)
        writer.writeheader()
        for r in radius_rows:
            writer.writerow({**r, "axis": "xy"})
        for r in radius_y_rows:
            writer.writerow({**r, "axis": "y"})
    json_path = out_dir / "np_com_lowbit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"COM_y intra RMS med {diag['intra_rms_um_median']:.2f} um | "
        f"centroid NN med {diag['centroid_nn_um_median']:.2f} um | "
        f"NN<20um {diag['frac_centroid_nn_lt_20um']:.1%}"
    )
    print(f"{'method':<36} {'bits':>4} {'dim':>3} {'argmin':>8}")
    for r in rows:
        print(
            f"{r['method']:<36} {r['n_bits']:>4} {r['feature_dim']:>3} "
            f"{r['argmin_accuracy']:8.4f}  {r['note']}"
        )
    print("--- COM_xy radius ---")
    for r in radius_rows:
        print(
            f"  R={r['radius_um']:6.0f}  recall={r['candidate_recall']:.4f}  "
            f"meanC={r['mean_candidates']:7.2f}  x={r['traffic_reduction_x']:6.2f}  "
            f"argmin={r['argmin_in_candidates']:.4f}"
        )
    print("--- COM_y radius ---")
    for r in radius_y_rows:
        print(
            f"  R={r['radius_um']:6.0f}  recall={r['candidate_recall']:.4f}  "
            f"meanC={r['mean_candidates']:7.2f}  x={r['traffic_reduction_x']:6.2f}  "
            f"argmin={r['argmin_in_candidates']:.4f}"
        )
    for title, rec in (
        ("xy R@0.99", summary["selected_xy_r99"]),
        ("xy R@0.95", summary["selected_xy_r95"]),
        ("y  R@0.99", summary["selected_y_r99"]),
        ("y  R@0.95", summary["selected_y_r95"]),
    ):
        if rec is None:
            print(f"Selected {title}: none")
        else:
            print(
                f"Selected {title}: R={rec['radius_um']:.0f} um  "
                f"recall={rec['candidate_recall']:.4f}  "
                f"meanC={rec['mean_candidates']:.2f}  "
                f"{rec['traffic_reduction_x']:.2f}x"
            )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
