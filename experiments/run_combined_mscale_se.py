"""Combined multi-scale FPN + SE-Net attention experiment.

Tests combinations of the best multi-scale fusion and attention variants.
Key new idea: SOM-discovered per-cluster profile attention (unsupervised
approximation of the oracle per-cluster attention that scored 0.966).

Usage:
    python -m Spatial.experiments.run_combined_mscale_se --duration 60
"""

from __future__ import annotations

import argparse
import csv
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
    footprint_p2p_features,
)
from Spatial.algorithms.multiscale_footprint import (
    extract_multiscale_p2p,
    multiscale_soft_loc,
    fuse_features,
)
from Spatial.algorithms.se_attention import (
    fisher_channel_weights,
    variance_channel_weights,
    cluster_profile_weights,
    apply_channel_weights,
    apply_per_cluster_weights,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.evaluation.metrics import clustering_accuracy

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
    features: np.ndarray, labels: np.ndarray, times: np.ndarray,
    train_frac: float = 0.5, tolerance_samples: int = 30,
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
    mw, mn = float(np.mean(within)), float(np.mean(nearest))
    return mw / mn if mn >= 1e-12 else float("inf")


def som_discovered_percluster_attention(
    p2p_raw: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    som_grid: int = 32,
    som_beta: int = 8,
    som_alpha: int = 16,
    n_iter: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Iterative SOM → per-cluster profile attention → re-extract → re-SOM.

    This is the unsupervised approximation of oracle per-cluster attention.

    Returns:
        (enhanced_soft_loc, som_labels, iteration_diagnostics)
    """
    soft = soft_localization_features(p2p_raw, neighbor_ids, geom)
    diagnostics = []

    for it in range(n_iter):
        som = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta, seed=seed)
        som.train(soft, verbose=False)
        labels = som.assign(soft)
        n_clusters = som.n_clusters

        # Per-cluster profile attention from SOM labels
        cw = cluster_profile_weights(p2p_raw, labels)
        p2p_att = apply_per_cluster_weights(p2p_raw, cw, labels)
        soft_new = soft_localization_features(p2p_att, neighbor_ids, geom)

        # Check convergence
        delta = np.mean(np.linalg.norm(soft_new - soft, axis=1))
        diagnostics.append({
            "iter": it, "n_clusters": n_clusters,
            "mean_feature_delta": float(delta),
        })

        soft = soft_new
        if delta < 1e-6:
            break

    # Final SOM on enhanced features
    som_final = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta, seed=seed)
    som_final.train(soft, verbose=False)
    labels_final = som_final.assign(soft)

    return soft, labels_final, diagnostics


def run_study(
    dataset,
    k_list: list[int],
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

    # Multi-scale P2P
    p2p_dict = extract_multiscale_p2p(
        filtered, valid_times, central_channels, dataset.geom, k_list, window=15
    )
    ref_times = p2p_dict[k_list[0]][1]
    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in ref_times], dtype=np.int64)
    keep = gt_labels >= 0
    gt_labels = gt_labels[keep]
    ref_times = ref_times[keep]

    p2p_kept = {}
    for k, (p2p, times, nbr_ids) in p2p_dict.items():
        p2p_kept[k] = (p2p[keep], times[keep], nbr_ids[keep])

    soft_dict = multiscale_soft_loc(p2p_kept, dataset.geom)
    for k in list(soft_dict.keys()):
        soft_dict[k] = soft_dict[k][keep]

    # Reference K=7 P2P for attention experiments
    ref_k = 7 if 7 in k_list else k_list[len(k_list) // 2]
    p2p_ref = p2p_kept[ref_k][0]
    nbr_ref = p2p_kept[ref_k][2]

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
            "som_accuracy": float(som_m["accuracy"]),
            "som_clusters": int(som.n_clusters),
            "gt_units": int(len(np.unique(gt_labels))),
            **(extra or {}),
        }
        rows.append(row)
        if verbose:
            print(
                f"  {name:<44} dim={feats.shape[1]:>3}  sep={sep:.3f}  "
                f"oracle={oracle['accuracy']:.3f}  SOM={som_m['accuracy']:.3f}  "
                f"K={som.n_clusters}/{len(np.unique(gt_labels))}"
            )

    # === Recap baselines ===
    if verbose:
        print("\n=== Baselines (recap) ===")
    eval_and_add("baseline_soft_loc_K7", soft_dict[ref_k], {"group": "baseline"})
    eval_and_add("baseline_ms_concat", fuse_features(soft_dict, "concat"),
                 {"group": "baseline"})

    # Best sub-band from FPN experiment
    sub3 = {k: soft_dict[k] for k in [3, ref_k, 16] if k in soft_dict}
    if len(sub3) == 3:
        eval_and_add("baseline_K3+K7+K16_concat", fuse_features(sub3, "concat"),
                     {"group": "baseline"})

    # === Oracle per-cluster attention (recap) ===
    if verbose:
        print("\n=== Oracle per-cluster attention (recap) ===")
    cw_oracle = cluster_profile_weights(p2p_ref, gt_labels)
    p2p_pc = apply_per_cluster_weights(p2p_ref, cw_oracle, gt_labels)
    soft_pc = soft_localization_features(p2p_pc, nbr_ref, dataset.geom)
    eval_and_add("oracle_percluster_K7", soft_pc, {"group": "oracle_attn"})

    # === SOM-discovered per-cluster attention (unsupervised, iterative) ===
    if verbose:
        print("\n=== SOM-discovered per-cluster attention (iterative) ===")
    for n_iter in [1, 2, 3]:
        soft_enh, labels_enh, diag = som_discovered_percluster_attention(
            p2p_ref, nbr_ref, dataset.geom,
            som_grid=som_grid, som_beta=som_beta, som_alpha=som_alpha,
            n_iter=n_iter,
        )
        eval_and_add(
            f"som_percluster_iter{n_iter}_K7", soft_enh,
            {"group": "som_attn", "n_iter": n_iter,
             "final_clusters": diag[-1]["n_clusters"] if diag else 0},
        )

    # === Multi-scale + per-scale attention ===
    if verbose:
        print("\n=== Multi-scale + per-scale Fisher attention ===")
    # Fisher attention at each K, then fuse
    fisher_per_k: dict[int, np.ndarray] = {}
    for k in k_list:
        p2p_k = p2p_kept[k][0]
        nbr_k = p2p_kept[k][2]
        fw = fisher_channel_weights(p2p_k, gt_labels)
        p2p_fw = apply_channel_weights(p2p_k, fw)
        fisher_per_k[k] = soft_localization_features(p2p_fw, nbr_k, dataset.geom)
    eval_and_add("ms_fisher_perK_concat", fuse_features(fisher_per_k, "concat"),
                 {"group": "ms_attn"})

    # === Multi-scale + variance attention (unsupervised) ===
    if verbose:
        print("\n=== Multi-scale + variance attention (unsupervised) ===")
    var_per_k: dict[int, np.ndarray] = {}
    for k in k_list:
        p2p_k = p2p_kept[k][0]
        nbr_k = p2p_kept[k][2]
        vw = variance_channel_weights(p2p_k)
        p2p_vw = apply_channel_weights(p2p_k, vw)
        var_per_k[k] = soft_localization_features(p2p_vw, nbr_k, dataset.geom)
    eval_and_add("ms_variance_perK_concat", fuse_features(var_per_k, "concat"),
                 {"group": "ms_attn"})

    # === Multi-scale + per-cluster attention (oracle) ===
    if verbose:
        print("\n=== Multi-scale + per-cluster attention (oracle) ===")
    pc_per_k: dict[int, np.ndarray] = {}
    for k in k_list:
        p2p_k = p2p_kept[k][0]
        nbr_k = p2p_kept[k][2]
        cw_k = cluster_profile_weights(p2p_k, gt_labels)
        p2p_cw = apply_per_cluster_weights(p2p_k, cw_k, gt_labels)
        pc_per_k[k] = soft_localization_features(p2p_cw, nbr_k, dataset.geom)
    eval_and_add("ms_percluster_perK_concat", fuse_features(pc_per_k, "concat"),
                 {"group": "ms_attn"})

    # === SOM per-cluster attention + multi-scale fusion ===
    if verbose:
        print("\n=== SOM per-cluster attention + multi-scale ===")
    # Run SOM-discovered attention at ref_k, then concat with other K's baseline
    soft_enh_ref, _, _ = som_discovered_percluster_attention(
        p2p_ref, nbr_ref, dataset.geom,
        som_grid=som_grid, som_beta=som_beta, som_alpha=som_alpha,
        n_iter=2,
    )
    combo_dict = dict(soft_dict)  # baseline at all K
    combo_dict[ref_k] = soft_enh_ref  # replace ref_k with enhanced
    eval_and_add("ms_som_percluster_K7+baseline", fuse_features(combo_dict, "concat"),
                 {"group": "combo"})

    # === Best sub-band + SOM per-cluster attention ===
    if verbose:
        print("\n=== Best sub-band + SOM per-cluster attention ===")
    # K3 + enhanced_K7 + K16
    sub_enh = {}
    for k in [3, ref_k, 16]:
        if k == ref_k:
            sub_enh[k] = soft_enh_ref
        elif k in soft_dict:
            sub_enh[k] = soft_dict[k]
    if len(sub_enh) >= 2:
        eval_and_add("K3+som_pc_K7+K16_concat", fuse_features(sub_enh, "concat"),
                     {"group": "combo"})

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_summary(rows)

    return rows


def _print_summary(rows: list[dict]):
    print(f"\n{'method':<44} {'dim':>4} {'sep':>8} {'oracle':>8} {'SOM':>8} {'clust':>8}")
    print("-" * 84)
    for r in rows:
        print(
            f"{r['method']:<44} {r['feature_dim']:>4} "
            f"{r['sep_ratio']:>8.3f} {r['oracle_accuracy']:>8.3f} "
            f"{r['som_accuracy']:>8.3f} {r['som_clusters']:>3}/{r['gt_units']:<3}"
        )


def main():
    parser = argparse.ArgumentParser(description="Combined multi-scale + SE-Net experiment")
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-path", type=Path, default=None)
    parser.add_argument("--k-list", type=int, nargs="+", default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Spatial/output/combined_mscale_se"))
    args = parser.parse_args()

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
        k_list = args.k_list or K_LIST_HJ
    else:
        if args.mearec_path:
            ds = load_mearec_npz(args.mearec_path, duration_s=args.duration)
        else:
            import glob
            npz_dir = Path(__file__).resolve().parents[2] / "synthetic_spike_dataset/generated/geoosort_npz_v1"
            npz_files = sorted(glob.glob(str(npz_dir / "rec_v1_*.npz")))
            if not npz_files:
                print("No MEArec NPZ found. Provide --mearec-path.")
                return
            ds = load_mearec_npz(Path(npz_files[0]), duration_s=args.duration)
        k_list = args.k_list or K_LIST_MEAREC

    k_list = sorted(set(min(k, ds.n_channels) for k in k_list))
    som_defaults = SOM_PARAMS.get(args.dataset, SOM_PARAMS["hj"])
    grid = args.grid_size or som_defaults["grid_size"]
    beta = args.beta or som_defaults["beta"]

    rows = run_study(ds, k_list=k_list, som_grid=grid, som_beta=beta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"combined_{args.dataset}_{int(args.duration)}s.csv"
    all_fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
