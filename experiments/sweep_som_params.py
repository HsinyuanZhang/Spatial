"""Parameter sweep for SOM clustering on spatial features.

Usage:
    python -m Spatial.experiments.sweep_som_params --dataset hj --duration 60
"""

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_mearec_h5
from Spatial.algorithms.detection import bandpass_filter, detect_spikes_neo, get_peak_amplitudes
from Spatial.algorithms.spatial_features import extract_spatial_features_fast, build_knn_table
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy


def run_sweep(
    dataset,
    alphas=(8, 16, 32),
    betas=(2, 4, 8),
    grid_sizes=(32, 64, 128),
    c_neo: float = 40.0,
    n_neighbors: int = 6,
    use_gt_spikes: bool = True,
    verbose: bool = True,
) -> list:
    """Grid search over SOM hyperparameters."""

    # Pre-compute features (shared across all SOM configs)
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)

    if use_gt_spikes:
        amplitudes, valid_times, central_channels = get_peak_amplitudes(filtered, dataset.spike_times)
        time_to_unit = dict(zip(dataset.spike_times, dataset.spike_units))
        gt_labels = np.array([time_to_unit.get(t, -1) for t in valid_times], dtype=np.int64)
    else:
        spike_times, _ = detect_spikes_neo(filtered, dataset.fs, c_neo=c_neo)
        amplitudes, valid_times, central_channels = get_peak_amplitudes(filtered, spike_times)
        gt_labels = dataset.spike_units

    knn_table = build_knn_table(dataset.geom, n_neighbors)
    features = extract_spatial_features_fast(amplitudes, dataset.geom, central_channels, knn_table)

    if verbose:
        print(f"Features: {features.shape[0]} spikes x {features.shape[1]} dims")
        print(f"GT units: {len(np.unique(gt_labels))}")
        print()

    results = []
    configs = list(itertools.product(alphas, betas, grid_sizes))
    total = len(configs)

    for idx, (alpha, beta, grid_size) in enumerate(configs):
        t0 = time.time()
        som = SpatialSOM(grid_size=grid_size, alpha=alpha, beta=beta)
        som.train(features, verbose=False)
        pred_labels = som.assign(features)

        metrics = clustering_accuracy(pred_labels, gt_labels, valid_times, valid_times, tolerance_samples=30)
        elapsed = time.time() - t0

        row = {
            "alpha": alpha,
            "beta": beta,
            "grid_size": grid_size,
            "accuracy": metrics["accuracy"],
            "n_clusters": som.n_clusters,
            "n_correct": metrics["n_correct"],
            "n_gt": metrics["n_gt"],
            "elapsed_s": round(elapsed, 2),
        }
        results.append(row)

        if verbose:
            print(f"  [{idx+1}/{total}] alpha={alpha:2d} beta={beta} grid={grid_size:3d} "
                  f"-> acc={metrics['accuracy']:.3f} clusters={som.n_clusters:3d} ({elapsed:.1f}s)")

    # Sort by accuracy
    results.sort(key=lambda r: -r["accuracy"])
    return results


def main():
    parser = argparse.ArgumentParser(description="SOM parameter sweep")
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--mearec-path", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--c-neo", type=float, default=40.0)
    parser.add_argument("--use-gt", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
    else:
        if args.mearec_path is None:
            print("ERROR: --mearec-path required for mearec dataset")
            sys.exit(1)
        ds = load_mearec_h5(args.mearec_path, duration_s=args.duration)

    print(f"Dataset: {ds.name} ({ds.n_channels}ch, {ds.duration_s:.0f}s)")
    print(f"GT: {len(ds.spike_times)} spikes, {len(ds.unit_ids)} units")
    print()

    results = run_sweep(ds, c_neo=args.c_neo, use_gt_spikes=args.use_gt)

    # Print top results
    print(f"\n{'='*60}")
    print("Top 5 configurations:")
    print(f"{'='*60}")
    for r in results[:5]:
        print(f"  alpha={r['alpha']:2d} beta={r['beta']} grid={r['grid_size']:3d} "
              f"-> acc={r['accuracy']:.3f} clusters={r['n_clusters']}")

    # Save
    output_path = args.output or (Path(__file__).resolve().parents[1] / "output" / "sweep_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
