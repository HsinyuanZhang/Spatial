"""Multi-scale FPN footprint fusion experiment.

Compares single-K soft_loc baselines against multi-K fusion strategies
(concat, mean, PCA, weighted) for spatial spike sorting.

Usage:
    python -m Spatial.experiments.run_multiscale_fpn --duration 60
    python -m Spatial.experiments.run_multiscale_fpn --duration 60 --dataset mearec
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_mearec_npz
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    soft_localization_features,
)
from Spatial.algorithms.multiscale_footprint import (
    extract_multiscale_p2p,
    multiscale_soft_loc,
    fuse_features,
    multiscale_com_spread_sharpness,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy

# ---------------------------------------------------------------------------
# Helpers (reused pattern from run_spatial_upper_bound.py)
# ---------------------------------------------------------------------------

SOM_PARAMS = {
    "hj": {"grid_size": 32, "beta": 8, "alpha": 16},
    "mearec": {"grid_size": 32, "beta": 8, "alpha": 16},
}

K_LIST_HJ = [3, 5, 7, 9, 13, 16]
K_LIST_MEAREC = [3, 5, 7, 9, 13, 21]


def _chronological_split(n: int, train_frac: float = 0.5):
    n_train = max(1, int(n * train_frac))
    if n_train >= n:
        n_train = max(1, n - 1) if n > 1 else 1
    return np.arange(0, n_train), np.arange(n_train, n) if n_train < n else np.arange(n)


def gt_centroid_oracle_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
    tolerance_samples: int = 30,
) -> dict:
    n = len(features)
    if n == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_gt": 0, "n_test": 0, "n_clusters": 0}

    order = np.argsort(times)
    feat, lab, t = features[order], labels[order], times[order]
    train_idx, test_idx = _chronological_split(n, train_frac)
    train_feat, train_lab = feat[train_idx], lab[train_idx]
    test_feat, test_lab, test_times = feat[test_idx], lab[test_idx], t[test_idx]

    units = np.unique(train_lab)
    centroids, centroid_units = [], []
    for u in units:
        mask = train_lab == u
        if np.any(mask):
            centroids.append(np.mean(train_feat[mask], axis=0))
            centroid_units.append(u)

    if not centroids or len(test_feat) == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_gt": int(len(test_lab)),
                "n_test": int(len(test_lab)), "n_clusters": len(centroids)}

    centroids = np.stack(centroids)
    centroid_units = np.asarray(centroid_units, dtype=np.int64)
    dists = np.linalg.norm(test_feat[:, None, :] - centroids[None, :, :], axis=2)
    pred = centroid_units[np.argmin(dists, axis=1)]

    m = clustering_accuracy(pred, test_lab, test_times, test_times,
                            tolerance_samples=tolerance_samples)
    m["n_test"] = int(len(test_lab))
    m["n_clusters"] = int(len(centroid_units))
    return m


def separability_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    units = np.unique(labels)
    if len(units) < 2 or len(features) == 0:
        return float("nan")
    centroids, within = {}, []
    for u in units:
        mask = labels == u
        c = np.mean(features[mask], axis=0)
        centroids[u] = c
        within.append(np.mean(np.linalg.norm(features[mask] - c, axis=1)) if np.sum(mask) > 1 else 0.0)
    nearest = []
    unit_list = list(units)
    for i, u in enumerate(unit_list):
        others = [centroids[v] for j, v in enumerate(unit_list) if j != i]
        d = np.linalg.norm(np.stack(others) - centroids[u], axis=1)
        nearest.append(float(np.min(d)))
    mean_within = float(np.mean(within))
    mean_nearest = float(np.mean(nearest))
    return mean_within / mean_nearest if mean_nearest >= 1e-12 else float("inf")


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_study(
    dataset,
    k_list: list[int],
    fusion_methods: tuple[str, ...] = ("concat", "mean", "pca4", "pca8"),
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
        print(f"K list: {k_list}")

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, valid_times, central_channels = get_peak_amplitudes(
        filtered, dataset.spike_times, window=15
    )

    # Multi-scale P2P extraction
    p2p_dict = extract_multiscale_p2p(
        filtered, valid_times, central_channels, dataset.geom, k_list, window=15
    )

    # Align GT labels
    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    ref_times = p2p_dict[k_list[0]][1]
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in ref_times], dtype=np.int64)
    keep = gt_labels >= 0
    gt_labels = gt_labels[keep]
    ref_times = ref_times[keep]

    # Re-index p2p_dict to kept spikes
    p2p_kept: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k, (p2p, times, nbr_ids) in p2p_dict.items():
        p2p_kept[k] = (p2p[keep], times[keep], nbr_ids[keep])

    # Soft_loc at each K
    soft_dict = multiscale_soft_loc(p2p_kept, dataset.geom)
    for k in list(soft_dict.keys()):
        soft_dict[k] = soft_dict[k][keep]

    rows: list[dict] = []

    def eval_and_add(name: str, feats: np.ndarray, extra: dict | None = None):
        sep = separability_ratio(feats, gt_labels)
        oracle = gt_centroid_oracle_accuracy(
            feats, gt_labels, ref_times, train_frac=train_frac,
            tolerance_samples=tolerance_samples,
        )
        som = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta, seed=42)
        som.train(feats, verbose=False)
        pred = som.assign(feats)
        som_m = clustering_accuracy(pred, gt_labels, ref_times, ref_times,
                                    tolerance_samples=tolerance_samples)
        row = {
            "dataset": dataset.name,
            "method": name,
            "feature_dim": int(feats.shape[1]),
            "n_spikes": int(len(feats)),
            "sep_ratio": float(sep),
            "oracle_accuracy": float(oracle["accuracy"]),
            "oracle_n_correct": int(oracle.get("n_correct", 0)),
            "som_accuracy": float(som_m["accuracy"]),
            "som_n_correct": int(som_m.get("n_correct", 0)),
            "som_clusters": int(som.n_clusters),
            "gt_units": int(len(np.unique(gt_labels))),
            **(extra or {}),
        }
        rows.append(row)
        if verbose:
            print(
                f"  {name:<32} dim={feats.shape[1]:>3}  sep={sep:.3f}  "
                f"oracle={oracle['accuracy']:.3f}  SOM={som_m['accuracy']:.3f}  "
                f"K={som.n_clusters}/{len(np.unique(gt_labels))}"
            )

    # --- Single-K baselines ---
    if verbose:
        print("\n--- Single-K baselines ---")
    for k in k_list:
        eval_and_add(f"soft_loc_K{k}", soft_dict[k], {"K": k, "fusion": "single"})

    # --- Multi-scale fusions ---
    if verbose:
        print("\n--- Multi-scale fusions ---")

    # All-K fusion
    for method in fusion_methods:
        if method == "pca4":
            fused = fuse_features(soft_dict, method="pca", pca_components=4)
            name = "ms_pca4"
        elif method == "pca8":
            fused = fuse_features(soft_dict, method="pca", pca_components=8)
            name = "ms_pca8"
        elif method == "concat":
            fused = fuse_features(soft_dict, method="concat")
            name = "ms_concat"
        elif method == "mean":
            fused = fuse_features(soft_dict, method="mean")
            name = "ms_mean"
        else:
            continue
        eval_and_add(name, fused, {"K": "all", "fusion": method})

    # --- Sub-band fusions (small+large K pairs) ---
    if verbose:
        print("\n--- Sub-band fusions ---")
    ref_k = 7 if 7 in k_list else k_list[len(k_list) // 2]
    small_ks = [k for k in k_list if k < ref_k]
    large_ks = [k for k in k_list if k > ref_k]

    if small_ks and large_ks:
        # Small + ref + large
        for sk in small_ks:
            for lk in large_ks:
                sub_dict = {sk: soft_dict[sk], ref_k: soft_dict[ref_k], lk: soft_dict[lk]}
                fused = fuse_features(sub_dict, method="concat")
                eval_and_add(
                    f"ms_K{sk}+K{ref_k}+K{lk}_concat", fused,
                    {"K": f"{sk},{ref_k},{lk}", "fusion": "concat_sub"},
                )

    # --- COM + multi-scale spread/sharpness ---
    if verbose:
        print("\n--- COM + multi-scale spread/sharpness ---")
    ms_ss = multiscale_com_spread_sharpness(p2p_kept, dataset.geom, ref_k=ref_k)
    ms_ss = ms_ss[keep]
    eval_and_add("com+ms_spread_sharp", ms_ss, {"K": "all", "fusion": "com_ms_ss"})

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_summary(rows)

    return rows


def run_som_sweep(
    dataset,
    k_list: list[int],
    fusion_method: str = "concat",
    grid_sizes: tuple[int, ...] = (16, 32, 48, 64),
    betas: tuple[int, ...] = (4, 8, 16),
    train_frac: float = 0.5,
    tolerance_samples: int = 30,
    verbose: bool = True,
) -> list[dict]:
    """SOM hyperparameter sweep on a chosen multi-scale fusion."""
    t0 = time.time()
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, valid_times, central_channels = get_peak_amplitudes(
        filtered, dataset.spike_times, window=15
    )
    p2p_dict = extract_multiscale_p2p(
        filtered, valid_times, central_channels, dataset.geom, k_list, window=15
    )
    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    ref_times = p2p_dict[k_list[0]][1]
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in ref_times], dtype=np.int64)
    keep = gt_labels >= 0
    gt_labels = gt_labels[keep]
    ref_times = ref_times[keep]

    soft_dict = multiscale_soft_loc(p2p_dict, dataset.geom)
    for k in list(soft_dict.keys()):
        soft_dict[k] = soft_dict[k][keep]

    if fusion_method == "concat":
        feats = fuse_features(soft_dict, method="concat")
    elif fusion_method == "mean":
        feats = fuse_features(soft_dict, method="mean")
    elif fusion_method.startswith("pca"):
        n_comp = int(fusion_method[3:])
        feats = fuse_features(soft_dict, method="pca", pca_components=n_comp)
    else:
        feats = fuse_features(soft_dict, method="concat")

    rows: list[dict] = []
    for gs, beta in itertools.product(grid_sizes, betas):
        som = SpatialSOM(grid_size=gs, alpha=16, beta=beta, seed=42)
        som.train(feats, verbose=False)
        pred = som.assign(feats)
        m = clustering_accuracy(pred, gt_labels, ref_times, ref_times,
                                tolerance_samples=tolerance_samples)
        row = {
            "dataset": dataset.name,
            "fusion": fusion_method,
            "grid_size": gs,
            "beta": beta,
            "accuracy": float(m["accuracy"]),
            "n_clusters": int(som.n_clusters),
            "n_correct": int(m.get("n_correct", 0)),
            "n_gt": int(m["n_gt"]),
            "gt_units": int(len(np.unique(gt_labels))),
        }
        rows.append(row)
        if verbose:
            print(f"  grid={gs:>3} beta={beta:>2}  acc={m['accuracy']:.3f}  K={som.n_clusters}")

    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    if verbose:
        print(f"\nBest: grid={rows[0]['grid_size']} beta={rows[0]['beta']} "
              f"acc={rows[0]['accuracy']:.3f}")
        print(f"Elapsed: {time.time() - t0:.1f}s")
    return rows


def _print_summary(rows: list[dict]):
    print(f"\n{'method':<32} {'dim':>4} {'sep':>8} {'oracle':>8} {'SOM':>8} {'clust':>8}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['method']:<32} {r['feature_dim']:>4} "
            f"{r['sep_ratio']:>8.3f} {r['oracle_accuracy']:>8.3f} "
            f"{r['som_accuracy']:>8.3f} {r['som_clusters']:>3}/{r['gt_units']:<3}"
        )


def main():
    parser = argparse.ArgumentParser(description="Multi-scale FPN footprint fusion")
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-path", type=Path, default=None,
                        help="Path to MEArec NPZ (for --dataset mearec)")
    parser.add_argument("--k-list", type=int, nargs="+", default=None,
                        help="K values (default: auto per dataset)")
    parser.add_argument("--fusion-methods", nargs="+",
                        default=["concat", "mean", "pca4", "pca8"])
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Run SOM hyperparameter sweep on ms_concat")
    parser.add_argument("--sweep-grids", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument("--sweep-betas", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Spatial/output/multiscale_fpn"))
    args = parser.parse_args()

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
        k_list = args.k_list or K_LIST_HJ
    else:
        if args.mearec_path:
            ds = load_mearec_npz(args.mearec_path, duration_s=args.duration)
        else:
            # Auto-discover first MEArec NPZ
            import glob
            npz_dir = Path(__file__).resolve().parents[2] / "synthetic_spike_dataset/generated/geoosort_npz_v1"
            npz_files = sorted(glob.glob(str(npz_dir / "rec_v1_*.npz")))
            if not npz_files:
                print("No MEArec NPZ found. Provide --mearec-path.")
                return
            ds = load_mearec_npz(Path(npz_files[0]), duration_s=args.duration)
        k_list = args.k_list or K_LIST_MEAREC

    # Clamp K to n_channels
    k_list = [min(k, ds.n_channels) for k in k_list]
    k_list = sorted(set(k_list))

    som_defaults = SOM_PARAMS.get(args.dataset, SOM_PARAMS["hj"])
    grid = args.grid_size or som_defaults["grid_size"]
    beta = args.beta or som_defaults["beta"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Main study
    rows = run_study(
        ds, k_list=k_list,
        fusion_methods=tuple(args.fusion_methods),
        som_grid=grid, som_beta=beta,
    )
    out_path = args.output_dir / f"multiscale_fpn_{args.dataset}_{int(args.duration)}s.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Optional SOM sweep
    if args.sweep:
        print("\n=== SOM hyperparameter sweep on ms_concat ===")
        sweep_rows = run_som_sweep(
            ds, k_list=k_list, fusion_method="concat",
            grid_sizes=tuple(args.sweep_grids), betas=tuple(args.sweep_betas),
        )
        sweep_path = args.output_dir / f"som_sweep_{args.dataset}_{int(args.duration)}s.csv"
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Saved: {sweep_path}")


if __name__ == "__main__":
    main()
