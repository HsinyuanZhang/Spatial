"""Spatial-only upper-bound study on 1D probes (no FSDE / temporal waveform PCA).

Compares spatial feature representations under:
  1) GT-centroid oracle (supervised feature ceiling)
  2) Unsupervised SpatialSOM

Usage:
    python -m Spatial.experiments.run_spatial_upper_bound --dataset hj --duration 60
    python -m Spatial.experiments.run_spatial_upper_bound --dataset cortexlab --duration 60
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
from Spatial.algorithms.spatial_features import (
    extract_spatial_features_fast,
    build_knn_table,
)
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    main_channel_features,
    soft_localization_features,
    footprint_p2p_features,
    footprint_pca_features,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy

ALL_METHODS = ("main_channel", "akhoundi5", "soft_loc", "footprint_p2p", "footprint_pca")

# Dataset-optimal SOM params from configs/experiment.yaml
SOM_PARAMS = {
    "hj": {"grid_size": 32, "beta": 8, "alpha": 16},
    "cortexlab": {"grid_size": 32, "beta": 4, "alpha": 16},
}


def _chronological_split(n: int, train_frac: float = 0.5):
    n_train = max(1, int(n * train_frac))
    if n_train >= n:
        n_train = max(1, n - 1) if n > 1 else 1
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n) if n_train < n else np.arange(n)
    return train_idx, test_idx


def gt_centroid_oracle_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
    tolerance_samples: int = 30,
) -> dict:
    """Assign test spikes to nearest GT-unit centroid fitted on train split."""
    n = len(features)
    if n == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_gt": 0, "n_test": 0, "n_clusters": 0}

    # Chronological split on sorted spike times
    order = np.argsort(times)
    feat = features[order]
    lab = labels[order]
    t = times[order]
    train_idx, test_idx = _chronological_split(n, train_frac)

    train_feat, train_lab = feat[train_idx], lab[train_idx]
    test_feat, test_lab, test_times = feat[test_idx], lab[test_idx], t[test_idx]

    units = np.unique(train_lab)
    centroids = []
    centroid_units = []
    for u in units:
        mask = train_lab == u
        if not np.any(mask):
            continue
        centroids.append(np.mean(train_feat[mask], axis=0))
        centroid_units.append(u)

    if not centroids or len(test_feat) == 0:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_gt": int(len(test_lab)),
            "n_test": int(len(test_lab)),
            "n_clusters": len(centroids),
        }

    centroids = np.stack(centroids, axis=0)
    centroid_units = np.asarray(centroid_units, dtype=np.int64)

    dists = np.linalg.norm(test_feat[:, None, :] - centroids[None, :, :], axis=2)
    pred = centroid_units[np.argmin(dists, axis=1)]

    # With GT times, pred_times == test_times; Hungarian mapping still applied
    metrics = clustering_accuracy(
        pred, test_lab, test_times, test_times, tolerance_samples=tolerance_samples
    )
    metrics["n_test"] = int(len(test_lab))
    metrics["n_clusters"] = int(len(centroid_units))
    return metrics


def separability_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-unit scatter / mean nearest-other-unit centroid distance."""
    units = np.unique(labels)
    if len(units) < 2 or len(features) == 0:
        return float("nan")

    centroids = {}
    within = []
    for u in units:
        mask = labels == u
        c = np.mean(features[mask], axis=0)
        centroids[u] = c
        if np.sum(mask) > 1:
            within.append(np.mean(np.linalg.norm(features[mask] - c, axis=1)))
        else:
            within.append(0.0)

    nearest = []
    unit_list = list(units)
    for i, u in enumerate(unit_list):
        others = [centroids[v] for j, v in enumerate(unit_list) if j != i]
        d = np.linalg.norm(np.stack(others) - centroids[u], axis=1)
        nearest.append(float(np.min(d)))

    mean_within = float(np.mean(within))
    mean_nearest = float(np.mean(nearest))
    if mean_nearest < 1e-12:
        return float("inf")
    return mean_within / mean_nearest


def build_feature_sets(
    filtered: np.ndarray,
    geom: np.ndarray,
    spike_times: np.ndarray,
    methods: tuple[str, ...],
    k_neighbors: int,
    pca_dim: int,
    window: int = 15,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build requested spatial feature matrices aligned to valid spike times."""
    # Central channel from peak amplitude (spatial-only; no waveform shape)
    amplitudes, valid_times, central_channels = get_peak_amplitudes(
        filtered, spike_times, window=window
    )

    k = min(k_neighbors, geom.shape[0])
    nbr_self = build_knn_table_with_self(geom, k)
    p2p, p2p_times, neighbor_ids = extract_local_p2p(
        filtered, valid_times, central_channels, nbr_self, window=window
    )
    # Align amplitudes/central to p2p_times (same validity rule)
    if len(p2p_times) != len(valid_times):
        # Should be identical validity; be defensive
        time_to_idx = {int(t): i for i, t in enumerate(valid_times)}
        keep = [time_to_idx[int(t)] for t in p2p_times]
        amplitudes = amplitudes[keep]
        central_channels = central_channels[keep]
        valid_times = p2p_times

    features = {}
    for method in methods:
        if method == "main_channel":
            features[method] = main_channel_features(central_channels, geom.shape[0])
        elif method == "akhoundi5":
            knn = build_knn_table(geom, n_neighbors=min(6, geom.shape[0] - 1))
            features[method] = extract_spatial_features_fast(
                amplitudes, geom, central_channels, knn
            )
        elif method == "soft_loc":
            features[method] = soft_localization_features(p2p, neighbor_ids, geom)
        elif method == "footprint_p2p":
            features[method] = footprint_p2p_features(p2p, normalize=True)
        elif method == "footprint_pca":
            fp = footprint_p2p_features(p2p, normalize=True)
            features[method] = footprint_pca_features(fp, n_components=pca_dim)
        else:
            raise ValueError(f"Unknown method: {method}")

    return features, valid_times, central_channels


def run_study(
    dataset,
    methods: tuple[str, ...] = ALL_METHODS,
    k_neighbors: int = 7,
    pca_dim: int = 4,
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

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    feature_sets, valid_times, _ = build_feature_sets(
        filtered,
        dataset.geom,
        dataset.spike_times,
        methods=methods,
        k_neighbors=k_neighbors,
        pca_dim=pca_dim,
    )

    # Map GT labels onto valid times
    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in valid_times], dtype=np.int64)
    keep = gt_labels >= 0
    valid_times = valid_times[keep]
    gt_labels = gt_labels[keep]
    for m in list(feature_sets.keys()):
        feature_sets[m] = feature_sets[m][keep]

    rows = []
    for method in methods:
        feats = feature_sets[method]
        if verbose:
            print(f"\n--- {method}  dim={feats.shape[1]}  n={len(feats)} ---")

        sep = separability_ratio(feats, gt_labels)
        oracle = gt_centroid_oracle_accuracy(
            feats, gt_labels, valid_times, train_frac=train_frac, tolerance_samples=tolerance_samples
        )

        som = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta)
        som.train(feats, verbose=False)
        pred = som.assign(feats)
        som_metrics = clustering_accuracy(
            pred, gt_labels, valid_times, valid_times, tolerance_samples=tolerance_samples
        )

        row = {
            "dataset": dataset.name,
            "method": method,
            "feature_dim": int(feats.shape[1]),
            "n_spikes": int(len(feats)),
            "sep_ratio": float(sep),
            "oracle_accuracy": float(oracle["accuracy"]),
            "oracle_n_correct": int(oracle.get("n_correct", 0)),
            "oracle_n_gt": int(oracle.get("n_gt", 0)),
            "som_accuracy": float(som_metrics["accuracy"]),
            "som_n_correct": int(som_metrics.get("n_correct", 0)),
            "som_n_gt": int(som_metrics.get("n_gt", 0)),
            "som_clusters": int(som.n_clusters),
            "gt_units": int(len(np.unique(gt_labels))),
        }
        rows.append(row)

        if verbose:
            print(
                f"  sep_ratio={sep:.3f}  "
                f"oracle={oracle['accuracy']:.3f}  "
                f"SOM={som_metrics['accuracy']:.3f}  "
                f"clusters={som.n_clusters}/{len(np.unique(gt_labels))}"
            )

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_summary_table(rows)

    return rows


def _print_summary_table(rows: list[dict]):
    print(f"\n{'method':<16} {'dim':>4} {'sep':>8} {'oracle':>8} {'SOM':>8} {'clust':>8}")
    print("-" * 56)
    for r in rows:
        print(
            f"{r['method']:<16} {r['feature_dim']:>4} "
            f"{r['sep_ratio']:>8.3f} {r['oracle_accuracy']:>8.3f} "
            f"{r['som_accuracy']:>8.3f} {r['som_clusters']:>3}/{r['gt_units']:<3}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Spatial-only upper-bound study (no FSDE / waveform PCA)"
    )
    parser.add_argument("--dataset", choices=["hj", "cortexlab"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GT spike times (default True). Detection path not used in this study.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
    )
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=4)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.use_gt:
        print("WARNING: this study is designed for --use-gt; proceeding with GT times anyway.")

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
    else:
        ds = load_cortexlab_npz(duration_s=args.duration)

    som_defaults = SOM_PARAMS[args.dataset]
    grid = args.grid_size if args.grid_size is not None else som_defaults["grid_size"]
    beta = args.beta if args.beta is not None else som_defaults["beta"]
    alpha = args.alpha if args.alpha is not None else som_defaults["alpha"]

    rows = run_study(
        ds,
        methods=tuple(args.methods),
        k_neighbors=args.k_neighbors,
        pca_dim=args.pca_dim,
        som_grid=grid,
        som_beta=beta,
        som_alpha=alpha,
        train_frac=args.train_frac,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
