"""Main experiment: spatial feature extraction + SOM clustering on multi-channel data.

Usage:
    python -m Spatial.experiments.run_spatial_sorting --dataset hj
    python -m Spatial.experiments.run_spatial_sorting --dataset hj --duration 60
    python -m Spatial.experiments.run_spatial_sorting --dataset mearec --mearec-path <path.h5>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_mearec_h5, load_mearec_npz
from Spatial.algorithms.detection import bandpass_filter, detect_spikes_neo, get_peak_amplitudes
from Spatial.algorithms.spatial_features import extract_spatial_features_fast, build_knn_table
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy, detection_metrics


def run_pipeline(
    dataset,
    c_neo: float = 40.0,
    n_neighbors: int = 6,
    grid_size: int = 64,
    alpha: int = 16,
    beta: int = 4,
    use_gt_spikes: bool = False,
    verbose: bool = True,
) -> dict:
    """Run full spatial sorting pipeline.

    Args:
        dataset: Dataset object from loader.
        c_neo: NEO detection threshold multiplier.
        n_neighbors: Neighbors for spatial features.
        grid_size: SOM initial grid size.
        alpha: SOM BMU update factor.
        beta: SOM frequency pruning divisor.
        use_gt_spikes: If True, skip detection and use GT spike times.
        verbose: Print progress.

    Returns:
        dict with all metrics and intermediate results.
    """
    t0 = time.time()

    # Step 1: Filter
    if verbose:
        print(f"[1/5] Filtering ({dataset.n_channels}ch, {dataset.duration_s:.0f}s)...")
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)

    # Step 2: Detect spikes
    if use_gt_spikes:
        spike_times = dataset.spike_times
        central_channels = np.zeros(len(spike_times), dtype=np.int64)
        # Determine central channel from GT: max amplitude channel per spike
        if verbose:
            print(f"[2/5] Using GT spikes: {len(spike_times)} spikes")
        amplitudes, valid_times, central_channels = get_peak_amplitudes(filtered, spike_times)
        gt_times_valid = valid_times
        # Map valid times back to GT units
        time_to_unit = dict(zip(dataset.spike_times, dataset.spike_units))
        gt_labels = np.array([time_to_unit.get(t, -1) for t in valid_times], dtype=np.int64)
    else:
        if verbose:
            print(f"[2/5] Detecting spikes (c_neo={c_neo})...")
        spike_times, central_channels = detect_spikes_neo(filtered, dataset.fs, c_neo=c_neo)
        if verbose:
            print(f"       Detected {len(spike_times)} spikes")
        amplitudes, valid_times, central_channels = get_peak_amplitudes(filtered, spike_times)
        gt_times_valid = dataset.spike_times
        gt_labels = dataset.spike_units

    # Step 3: Extract spatial features
    if verbose:
        print(f"[3/5] Extracting spatial features (n_neighbors={n_neighbors})...")
    knn_table = build_knn_table(dataset.geom, n_neighbors)
    features = extract_spatial_features_fast(amplitudes, dataset.geom, central_channels, knn_table)
    if verbose:
        print(f"       Feature shape: {features.shape}")

    # Step 4: SOM clustering
    if verbose:
        print(f"[4/5] SOM clustering (grid={grid_size}, alpha={alpha}, beta={beta})...")
    som = SpatialSOM(grid_size=grid_size, alpha=alpha, beta=beta)
    som.train(features, verbose=verbose)
    pred_labels = som.assign(features)
    if verbose:
        print(f"       Converged to {som.n_clusters} clusters")

    # Step 5: Evaluate
    if verbose:
        print(f"[5/5] Evaluating...")

    det_metrics = detection_metrics(valid_times, gt_times_valid, tolerance_samples=30)
    clust_metrics = clustering_accuracy(
        pred_labels, gt_labels, valid_times, gt_times_valid, tolerance_samples=30
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.name}")
        print(f"Duration: {dataset.duration_s:.0f}s | Channels: {dataset.n_channels}")
        print(f"GT spikes: {len(gt_times_valid)} | Detected: {len(valid_times)}")
        print(f"{'='*50}")
        print(f"Detection:  P={det_metrics['precision']:.3f}  R={det_metrics['recall']:.3f}  F1={det_metrics['f1']:.3f}")
        print(f"Clustering: Accuracy={clust_metrics['accuracy']:.3f}  "
              f"({clust_metrics['n_correct']}/{clust_metrics['n_gt']})")
        print(f"SOM clusters: {som.n_clusters} | GT units: {len(np.unique(gt_labels))}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*50}")

    return {
        "dataset": dataset.name,
        "detection": det_metrics,
        "clustering": clust_metrics,
        "som_clusters": som.n_clusters,
        "gt_units": len(np.unique(gt_labels)),
        "n_features": features.shape[1],
        "params": {
            "c_neo": c_neo,
            "n_neighbors": n_neighbors,
            "grid_size": grid_size,
            "alpha": alpha,
            "beta": beta,
            "use_gt_spikes": use_gt_spikes,
        },
        "elapsed_s": elapsed,
        "convergence_history": som.n_active_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Spatial sorting experiment (Akhoundi 2025)")
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--mearec-path", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None, help="Truncate to N seconds")
    parser.add_argument("--c-neo", type=float, default=40.0)
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--beta", type=int, default=4)
    parser.add_argument("--use-gt", action="store_true", help="Use GT spike times (skip detection)")
    parser.add_argument("--output", type=Path, default=None, help="Save results CSV")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
    elif args.dataset == "mearec":
        if args.mearec_path is None:
            # Try default generated path
            default_path = Path(__file__).resolve().parents[1] / "output" / "mearec_generated"
            h5_files = list(default_path.glob("*/recording.h5"))
            if not h5_files:
                print("ERROR: No MEArec recordings found. Run generate_mearec.py first.")
                sys.exit(1)
            args.mearec_path = h5_files[0]
        if str(args.mearec_path).endswith(".npz"):
            ds = load_mearec_npz(args.mearec_path, duration_s=args.duration)
        else:
            ds = load_mearec_h5(args.mearec_path, duration_s=args.duration)

    results = run_pipeline(
        ds,
        c_neo=args.c_neo,
        n_neighbors=args.n_neighbors,
        grid_size=args.grid_size,
        alpha=args.alpha,
        beta=args.beta,
        use_gt_spikes=args.use_gt,
    )

    # Save results
    if args.output:
        import csv
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flat = {
            "dataset": results["dataset"],
            "det_precision": results["detection"]["precision"],
            "det_recall": results["detection"]["recall"],
            "det_f1": results["detection"]["f1"],
            "clust_accuracy": results["clustering"]["accuracy"],
            "som_clusters": results["som_clusters"],
            "gt_units": results["gt_units"],
            **{f"param_{k}": v for k, v in results["params"].items()},
            "elapsed_s": results["elapsed_s"],
        }
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat.keys())
            writer.writeheader()
            writer.writerow(flat)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
