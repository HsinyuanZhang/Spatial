"""Two-stage COM∧P2P → waveform upper-bound study (GT spikes).

Usage:
    python -m Spatial.experiments.run_two_stage_upper_bound --dataset hj --duration 60
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    com_features,
    footprint_p2p_features,
)
from Spatial.algorithms.two_stage_assign import (
    fit_unit_centroids,
    assign_nearest,
    calibrate_thresholds,
    two_stage_assign,
    assign_stage1_normalized,
    concat_zscore_blocks,
)
from GeoOsort.feature_extractor import FeatureExtractor


def _chronological_split(n: int, train_frac: float = 0.5):
    n_train = max(1, int(n * train_frac))
    if n_train >= n and n > 1:
        n_train = n - 1
    return np.arange(0, n_train), np.arange(n_train, n)


def extract_main_channel_waveforms(
    signal: np.ndarray,
    times: np.ndarray,
    central_channels: np.ndarray,
    pre: int = 15,
    post: int = 49,
) -> np.ndarray:
    n = len(times)
    win = pre + post
    out = np.zeros((n, win), dtype=np.float64)
    n_ch, n_samp = signal.shape
    for i, (t, ch) in enumerate(zip(times, central_channels)):
        ch = int(np.clip(ch, 0, n_ch - 1))
        s = int(t - pre)
        e = int(t + post)
        if s < 0 or e > n_samp:
            continue
        out[i] = signal[ch, s:e]
    return out


def normalize_waveforms(wfs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = np.asarray(wfs, dtype=np.float64)
    scale = np.max(np.abs(w), axis=1, keepdims=True)
    return w / np.maximum(scale, eps)


def peak_fsde_batch(waveforms: np.ndarray, normalize: bool = True) -> np.ndarray:
    ext = FeatureExtractor(normalize=normalize)
    return ext.process_batch([w for w in waveforms])


def oracle_acc(pred: np.ndarray, y_true: np.ndarray) -> dict:
    n = len(y_true)
    n_correct = int(np.sum(pred == y_true))
    return {
        "accuracy": float(n_correct / max(n, 1)),
        "n_correct": n_correct,
        "n_test": int(n),
    }


def run_study(
    dataset,
    k_neighbors: int = 7,
    window: int = 15,
    train_frac: float = 0.5,
    percentile: float = 95.0,
    p2p_metric: str = "l2",
    verbose: bool = True,
) -> list[dict]:
    t0 = time.time()
    if verbose:
        print(f"Dataset: {dataset.name} | {dataset.n_channels}ch | {dataset.duration_s:.1f}s")
        print(f"GT spikes: {len(dataset.spike_times)} | units: {len(dataset.unit_ids)}")
        print(f"Stage-1 gate percentile: {percentile} | P2P metric: {p2p_metric}")

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=window)

    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    labels = np.array([time_to_unit.get(int(t), -1) for t in times], dtype=np.int64)
    keep = labels >= 0
    times, central, labels = times[keep], central[keep], labels[keep]

    k = min(k_neighbors, dataset.geom.shape[0])
    nbr = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, times2, neighbor_ids = extract_local_p2p(
        filtered, times, central, nbr, window=window
    )
    assert len(times2) == len(times)

    com = com_features(p2p_raw, neighbor_ids, dataset.geom)
    p2p = footprint_p2p_features(p2p_raw, normalize=True)
    wfs = extract_main_channel_waveforms(filtered, times, central)
    wf_norm = normalize_waveforms(wfs)
    fsde = peak_fsde_batch(wfs, normalize=True)

    order = np.argsort(times)
    com = com[order]
    p2p = p2p[order]
    fsde = fsde[order]
    wf_norm = wf_norm[order]
    labels = labels[order]
    times = times[order]

    train_idx, test_idx = _chronological_split(len(labels), train_frac)
    y_train, y_test = labels[train_idx], labels[test_idx]

    tau_com, tau_p2p = calibrate_thresholds(
        com[train_idx], p2p[train_idx], y_train, percentile=percentile,
        p2p_metric=p2p_metric,
    )
    if verbose:
        print(f"τ_com={tau_com:.4g}  τ_p2p={tau_p2p:.4g}  "
              f"n_train={len(train_idx)} n_test={len(test_idx)}")

    rows: list[dict] = []

    def add_row(method: str, pred: np.ndarray, extra: dict | None = None):
        m = oracle_acc(pred, y_test)
        extra = extra or {}
        row = {
            "dataset": dataset.name,
            "p2p_metric": p2p_metric,
            "method": method,
            "oracle_accuracy": m["accuracy"],
            "n_correct": m["n_correct"],
            "n_test": m["n_test"],
            "tau_com": float(tau_com),
            "tau_p2p": float(tau_p2p),
            "percentile": float(percentile),
            "fallback_rate": extra.get("fallback_rate", ""),
            "mean_n_candidates": extra.get("mean_n_candidates", ""),
        }
        rows.append(row)
        if verbose:
            fb = ""
            if "fallback_rate" in extra:
                fb = (f"  fallback={extra['fallback_rate']:.3f}  "
                      f"mean_cands={extra['mean_n_candidates']:.2f}")
            print(f"  {method:<28} oracle={m['accuracy']:.3f}{fb}")

    # --- Baselines ---
    if verbose:
        print("\n--- baselines ---")
    add_row("com_only", assign_nearest(com[test_idx], fit_unit_centroids(com[train_idx], y_train)))
    add_row(
        "p2p_only",
        assign_nearest(
            p2p[test_idx], fit_unit_centroids(p2p[train_idx], y_train), metric=p2p_metric
        ),
    )
    add_row(
        "com_and_p2p",
        assign_stage1_normalized(
            com[test_idx], p2p[test_idx], y_train, com[train_idx], p2p[train_idx],
            percentile=percentile, p2p_metric=p2p_metric,
        ),
    )
    add_row("fsde_only", assign_nearest(fsde[test_idx], fit_unit_centroids(fsde[train_idx], y_train)))
    add_row(
        "waveform_only",
        assign_nearest(wf_norm[test_idx], fit_unit_centroids(wf_norm[train_idx], y_train)),
    )

    tr_c_f, te_c_f = concat_zscore_blocks(
        [com[train_idx], p2p[train_idx], fsde[train_idx]],
        [com[test_idx], p2p[test_idx], fsde[test_idx]],
    )
    add_row(
        "concat_com_p2p_fsde",
        assign_nearest(te_c_f, fit_unit_centroids(tr_c_f, y_train)),
    )
    tr_c_w, te_c_w = concat_zscore_blocks(
        [com[train_idx], p2p[train_idx], wf_norm[train_idx]],
        [com[test_idx], p2p[test_idx], wf_norm[test_idx]],
    )
    add_row(
        "concat_com_p2p_waveform",
        assign_nearest(te_c_w, fit_unit_centroids(tr_c_w, y_train)),
    )

    # --- Two-stage targets ---
    if verbose:
        print("\n--- two-stage ---")
    pred_fsde, meta_fsde = two_stage_assign(
        com[train_idx], p2p[train_idx], fsde[train_idx], y_train,
        com[test_idx], p2p[test_idx], fsde[test_idx],
        percentile=percentile, p2p_metric=p2p_metric,
    )
    add_row("two_stage_fsde", pred_fsde, meta_fsde)

    pred_wf, meta_wf = two_stage_assign(
        com[train_idx], p2p[train_idx], wf_norm[train_idx], y_train,
        com[test_idx], p2p[test_idx], wf_norm[test_idx],
        percentile=percentile, p2p_metric=p2p_metric,
    )
    add_row("two_stage_waveform", pred_wf, meta_wf)

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        print(f"\n{'method':<28} {'oracle':>8}")
        print("-" * 40)
        for r in rows:
            print(f"{r['method']:<28} {r['oracle_accuracy']:>8.3f}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Two-stage COM∧P2P → waveform upper bound")
    parser.add_argument("--dataset", choices=["hj"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--p2p-metric",
        choices=["l2", "l1"],
        default="l2",
        help="Distance for stage-1 P2P gate / com_and_p2p (COM stays L2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(
            f"Spatial/output/two_stage_upper_bound_hj60s_p2p{args.p2p_metric}.csv"
        )

    ds = load_hybrid_janelia(duration_s=args.duration)
    rows = run_study(
        ds,
        k_neighbors=args.k_neighbors,
        train_frac=args.train_frac,
        percentile=args.percentile,
        p2p_metric=args.p2p_metric,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
