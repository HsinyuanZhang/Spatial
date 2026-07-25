"""Sweep low-bit P2P quantization for spatial-only features on 1D probes.

Quantizes local neighborhood P2P amplitudes, then rebuilds soft_loc /
footprint features and reports GT-centroid oracle + SpatialSOM accuracy.

Usage:
    python -m Spatial.experiments.run_lowbit_p2p --dataset hj --duration 60
    python -m Spatial.experiments.run_lowbit_p2p --dataset hj --mode lsb --bits 8 5 4 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_cortexlab_npz
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    main_channel_features,
    soft_localization_features,
    footprint_p2p_features,
    quantize_p2p,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy
from Spatial.experiments.run_spatial_upper_bound import (
    SOM_PARAMS,
    gt_centroid_oracle_accuracy,
    separability_ratio,
)

FEATURE_METHODS = ("main_channel", "soft_loc", "footprint_p2p")
DEFAULT_BITS = (None, 8, 6, 5, 4, 3, 2)


def _bit_label(n_bits: int | None) -> str:
    return "float" if n_bits is None else f"{int(n_bits)}bit"


def run_lowbit_study(
    dataset,
    bits: tuple[int | None, ...] = DEFAULT_BITS,
    mode: str = "per_spike_max",
    lsb: float | None = None,
    feature_methods: tuple[str, ...] = FEATURE_METHODS,
    k_neighbors: int = 7,
    window: int = 15,
    som_grid: int = 32,
    som_beta: int = 8,
    som_alpha: int = 16,
    train_frac: float = 0.5,
    tolerance_samples: int = 30,
    verbose: bool = True,
) -> list[dict]:
    t0 = time.time()
    if verbose:
        print(f"Dataset: {dataset.name} | {dataset.n_channels}ch | {dataset.duration_s:.1f}s")
        print(f"GT spikes: {len(dataset.spike_times)} | units: {len(dataset.unit_ids)}")
        print(f"Quant mode: {mode}" + (f"  lsb={lsb}" if mode == "lsb" else ""))

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    amplitudes, valid_times, central_channels = get_peak_amplitudes(
        filtered, dataset.spike_times, window=window
    )
    k = min(k_neighbors, dataset.geom.shape[0])
    nbr_self = build_knn_table_with_self(dataset.geom, k)
    p2p_f, p2p_times, neighbor_ids = extract_local_p2p(
        filtered, valid_times, central_channels, nbr_self, window=window
    )
    if len(p2p_times) != len(valid_times):
        time_to_idx = {int(t): i for i, t in enumerate(valid_times)}
        keep = [time_to_idx[int(t)] for t in p2p_times]
        central_channels = central_channels[keep]
        valid_times = p2p_times

    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in valid_times], dtype=np.int64)
    keep = gt_labels >= 0
    valid_times = valid_times[keep]
    gt_labels = gt_labels[keep]
    central_channels = central_channels[keep]
    p2p_f = p2p_f[keep]
    neighbor_ids = neighbor_ids[keep]

    rows: list[dict] = []
    main_ch_feats = main_channel_features(central_channels, dataset.geom.shape[0])

    for n_bits in bits:
        label = _bit_label(n_bits)
        lsb_used = None
        if mode == "lsb" and n_bits is not None:
            if lsb is not None and lsb > 0:
                use_lsb = float(lsb)
            else:
                # Full-range ADC: map recording max P2P onto (2^n-1).
                max_level = float((1 << int(n_bits)) - 1)
                use_lsb = float(np.max(p2p_f)) / max_level if p2p_f.size else 1.0
            p2p_q = quantize_p2p(p2p_f, n_bits, mode="lsb", lsb=use_lsb)
            lsb_used = use_lsb
        else:
            p2p_q = quantize_p2p(p2p_f, n_bits, mode=mode, lsb=lsb)

        if verbose:
            uniq = "float" if n_bits is None else int(np.unique(p2p_q).size)
            extra = f"  lsb={lsb_used:.4g}" if lsb_used is not None else ""
            print(f"\n=== P2P={label}  unique_codes≈{uniq}{extra} ===")

        feature_sets: dict[str, np.ndarray] = {}
        for method in feature_methods:
            if method == "main_channel":
                # Independent of P2P quantization; still reported once per bit for table alignment
                feature_sets[method] = main_ch_feats
            elif method == "soft_loc":
                feature_sets[method] = soft_localization_features(
                    p2p_q, neighbor_ids, dataset.geom
                )
            elif method == "footprint_p2p":
                feature_sets[method] = footprint_p2p_features(p2p_q, normalize=True)
            else:
                raise ValueError(f"Unknown feature method: {method}")

        for method in feature_methods:
            feats = feature_sets[method]
            sep = separability_ratio(feats, gt_labels)
            oracle = gt_centroid_oracle_accuracy(
                feats, gt_labels, valid_times, train_frac=train_frac,
                tolerance_samples=tolerance_samples,
            )
            som = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta)
            som.train(feats, verbose=False)
            pred = som.assign(feats)
            som_metrics = clustering_accuracy(
                pred, gt_labels, valid_times, valid_times,
                tolerance_samples=tolerance_samples,
            )
            row = {
                "dataset": dataset.name,
                "quant_mode": mode,
                "p2p_bits": label,
                "n_bits": -1 if n_bits is None else int(n_bits),
                "lsb": "" if lsb_used is None else float(lsb_used),
                "method": method,
                "feature_dim": int(feats.shape[1]),
                "n_spikes": int(len(feats)),
                "sep_ratio": float(sep),
                "oracle_accuracy": float(oracle["accuracy"]),
                "som_accuracy": float(som_metrics["accuracy"]),
                "som_clusters": int(som.n_clusters),
                "gt_units": int(len(np.unique(gt_labels))),
            }
            rows.append(row)
            if verbose:
                print(
                    f"  {method:<14} sep={sep:.3f}  "
                    f"oracle={oracle['accuracy']:.3f}  "
                    f"SOM={som_metrics['accuracy']:.3f}  "
                    f"clust={som.n_clusters}/{len(np.unique(gt_labels))}"
                )

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_pivot(rows)

    return rows


def _print_pivot(rows: list[dict]):
    """Print soft_loc / footprint oracle & SOM vs bit width."""
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    bits = []
    for r in rows:
        if r["p2p_bits"] not in bits:
            bits.append(r["p2p_bits"])

    print("\nOracle accuracy")
    header = f"{'method':<14}" + "".join(f"{b:>9}" for b in bits)
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = []
        for b in bits:
            hit = next(r for r in rows if r["method"] == m and r["p2p_bits"] == b)
            cells.append(f"{hit['oracle_accuracy']:>9.3f}")
        print(f"{m:<14}" + "".join(cells))

    print("\nSOM accuracy")
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = []
        for b in bits:
            hit = next(r for r in rows if r["method"] == m and r["p2p_bits"] == b)
            cells.append(f"{hit['som_accuracy']:>9.3f}")
        print(f"{m:<14}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Low-bit P2P spatial feature sweep")
    parser.add_argument("--dataset", choices=["hj", "cortexlab"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--mode",
        choices=["per_spike_max", "global_max", "lsb"],
        default="per_spike_max",
        help="P2P quantization scheme (default: per-spike relative).",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        default=None,
        help="Bit widths to sweep; use 'float' for unquantized. Default: float 8 6 5 4 3 2",
    )
    parser.add_argument(
        "--lsb",
        type=float,
        default=None,
        help="Fixed LSB for mode=lsb. If omitted, LSB = max(P2P)/(2^n-1) per bit width.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(FEATURE_METHODS),
        choices=list(FEATURE_METHODS),
    )
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Spatial/output/lowbit_p2p_hj60s.csv"),
    )
    args = parser.parse_args()

    if args.bits is None:
        bits: tuple[int | None, ...] = DEFAULT_BITS
    else:
        parsed = []
        for b in args.bits:
            if str(b).lower() in ("float", "none", "full", "-1"):
                parsed.append(None)
            else:
                parsed.append(int(b))
        bits = tuple(parsed)

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
    else:
        ds = load_cortexlab_npz(duration_s=args.duration)

    som_defaults = SOM_PARAMS[args.dataset]
    grid = args.grid_size if args.grid_size is not None else som_defaults["grid_size"]
    beta = args.beta if args.beta is not None else som_defaults["beta"]
    alpha = args.alpha if args.alpha is not None else som_defaults["alpha"]

    rows = run_lowbit_study(
        ds,
        bits=bits,
        mode=args.mode,
        lsb=args.lsb,
        feature_methods=tuple(args.methods),
        k_neighbors=args.k_neighbors,
        som_grid=grid,
        som_beta=beta,
        som_alpha=alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
