"""SE-Net channel attention experiment for spatial spike sorting.

Compares attention variants:
  - Baseline: no attention (standard soft_loc / footprint_p2p)
  - Fisher global: per-channel Fisher discriminant ratio (oracle, GT labels)
  - Variance: per-channel variance (unsupervised)
  - Inverse-CV: 1/(1+CV) weighting (unsupervised)
  - Per-cluster profile: per-unit mean footprint as weight (oracle)
  - SOM-Fisher: Fisher weights from SOM-discovered clusters (unsupervised)
  - Temperature sweep on best attention

Usage:
    python -m Spatial.experiments.run_se_attention --duration 60
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
from Spatial.algorithms.se_attention import (
    fisher_channel_weights,
    variance_channel_weights,
    cluster_profile_weights,
    inverse_cv_weights,
    apply_channel_weights,
    apply_per_cluster_weights,
    softmax_temperature,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.algorithms.two_stage_assign import (
    two_stage_assign,
    fit_unit_centroids,
    calibrate_thresholds,
)
from Spatial.evaluation.metrics import clustering_accuracy

SOM_PARAMS = {
    "hj": {"grid_size": 32, "beta": 8, "alpha": 16},
    "mearec": {"grid_size": 32, "beta": 8, "alpha": 16},
}


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


def run_study(
    dataset,
    k_neighbors: int = 7,
    som_grid: int = 32,
    som_beta: int = 8,
    som_alpha: int = 16,
    train_frac: float = 0.5,
    tolerance_samples: int = 30,
    temperatures: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0),
    verbose: bool = True,
) -> list[dict]:
    t0 = time.time()
    if verbose:
        print(f"Dataset: {dataset.name} | {dataset.n_channels}ch | {dataset.duration_s:.1f}s")
        print(f"GT spikes: {len(dataset.spike_times)} | units: {len(dataset.unit_ids)}")
        print(f"K={k_neighbors}")

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, valid_times, central_channels = get_peak_amplitudes(
        filtered, dataset.spike_times, window=15
    )

    k = min(k_neighbors, dataset.geom.shape[0])
    nbr = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, times, neighbor_ids = extract_local_p2p(
        filtered, valid_times, central_channels, nbr, window=15
    )

    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    gt_labels = np.array([time_to_unit.get(int(t), -1) for t in times], dtype=np.int64)
    keep = gt_labels >= 0
    p2p_raw, times, gt_labels = p2p_raw[keep], times[keep], gt_labels[keep]
    neighbor_ids = neighbor_ids[keep]

    # Baseline features
    soft_loc_base = soft_localization_features(p2p_raw, neighbor_ids, dataset.geom)
    fp2p_base = footprint_p2p_features(p2p_raw, normalize=True)

    # Chronological split for oracle
    order = np.argsort(times)
    p2p_ordered = p2p_raw[order]
    labels_ordered = gt_labels[order]
    times_ordered = times[order]
    train_idx, test_idx = _chronological_split(len(labels_ordered), train_frac)
    y_train = labels_ordered[train_idx]

    rows: list[dict] = []

    def eval_and_add(name: str, feats: np.ndarray, extra: dict | None = None):
        sep = separability_ratio(feats, gt_labels)
        oracle = gt_centroid_oracle_accuracy(
            feats, gt_labels, times, train_frac=train_frac,
            tolerance_samples=tolerance_samples,
        )
        som = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta, seed=42)
        som.train(feats, verbose=False)
        pred = som.assign(feats)
        som_m = clustering_accuracy(pred, gt_labels, times, times,
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
                f"  {name:<36} dim={feats.shape[1]:>3}  sep={sep:.3f}  "
                f"oracle={oracle['accuracy']:.3f}  SOM={som_m['accuracy']:.3f}  "
                f"K={som.n_clusters}/{len(np.unique(gt_labels))}"
            )

    # --- Baselines ---
    if verbose:
        print("\n--- Baselines (no attention) ---")
    eval_and_add("baseline_soft_loc", soft_loc_base, {"attention": "none"})
    eval_and_add("baseline_fp2p", fp2p_base, {"attention": "none"})

    # --- Fisher global attention (oracle) ---
    if verbose:
        print("\n--- Fisher global attention (oracle, GT labels) ---")
    fisher_w = fisher_channel_weights(p2p_raw, gt_labels)
    p2p_fisher = apply_channel_weights(p2p_raw, fisher_w)
    soft_fisher = soft_localization_features(p2p_fisher, neighbor_ids, dataset.geom)
    fp2p_fisher = footprint_p2p_features(p2p_fisher, normalize=True)
    eval_and_add("fisher_soft_loc", soft_fisher, {"attention": "fisher"})
    eval_and_add("fisher_fp2p", fp2p_fisher, {"attention": "fisher"})

    # --- Variance attention (unsupervised) ---
    if verbose:
        print("\n--- Variance attention (unsupervised) ---")
    var_w = variance_channel_weights(p2p_raw)
    p2p_var = apply_channel_weights(p2p_raw, var_w)
    soft_var = soft_localization_features(p2p_var, neighbor_ids, dataset.geom)
    fp2p_var = footprint_p2p_features(p2p_var, normalize=True)
    eval_and_add("variance_soft_loc", soft_var, {"attention": "variance"})
    eval_and_add("variance_fp2p", fp2p_var, {"attention": "variance"})

    # --- Inverse-CV attention (unsupervised) ---
    if verbose:
        print("\n--- Inverse-CV attention (unsupervised) ---")
    icv_w = inverse_cv_weights(p2p_raw, gt_labels)
    p2p_icv = apply_channel_weights(p2p_raw, icv_w)
    soft_icv = soft_localization_features(p2p_icv, neighbor_ids, dataset.geom)
    eval_and_add("inv_cv_soft_loc", soft_icv, {"attention": "inv_cv"})

    # --- Per-cluster profile attention (oracle) ---
    if verbose:
        print("\n--- Per-cluster profile attention (oracle) ---")
    cluster_w = cluster_profile_weights(p2p_raw, gt_labels)
    # For oracle eval: weight each spike by its GT unit's profile
    p2p_percluster = apply_per_cluster_weights(p2p_raw, cluster_w, gt_labels)
    soft_percluster = soft_localization_features(p2p_percluster, neighbor_ids, dataset.geom)
    eval_and_add("percluster_soft_loc", soft_percluster, {"attention": "percluster"})

    # --- SOM-discovered Fisher (unsupervised) ---
    if verbose:
        print("\n--- SOM-discovered Fisher (unsupervised) ---")
    som_disc = SpatialSOM(grid_size=som_grid, alpha=som_alpha, beta=som_beta, seed=42)
    som_disc.train(soft_loc_base, verbose=False)
    som_labels = som_disc.assign(soft_loc_base)
    som_fisher_w = fisher_channel_weights(p2p_raw, som_labels)
    p2p_somfisher = apply_channel_weights(p2p_raw, som_fisher_w)
    soft_somfisher = soft_localization_features(p2p_somfisher, neighbor_ids, dataset.geom)
    eval_and_add("som_fisher_soft_loc", soft_somfisher, {"attention": "som_fisher"})

    # --- Temperature sweep on Fisher attention ---
    if verbose:
        print("\n--- Temperature sweep on Fisher attention ---")
    for T in temperatures:
        w_t = softmax_temperature(fisher_w, temperature=T)
        p2p_t = apply_channel_weights(p2p_raw, w_t)
        soft_t = soft_localization_features(p2p_t, neighbor_ids, dataset.geom)
        eval_and_add(f"fisher_T{T}", soft_t, {"attention": "fisher", "temperature": T})

    # --- Two-stage integration: attention-weighted P2P in gate ---
    if verbose:
        print("\n--- Two-stage gate with attention-weighted P2P ---")
    _run_two_stage_attention(
        rows, p2p_raw, p2p_ordered, labels_ordered, times_ordered,
        train_idx, test_idx, neighbor_ids, dataset.geom,
        fisher_w, var_w, som_fisher_w,
        tolerance_samples, verbose,
    )

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        _print_summary(rows)

    return rows


def _run_two_stage_attention(
    rows: list[dict],
    p2p_raw: np.ndarray,
    p2p_ordered: np.ndarray,
    labels_ordered: np.ndarray,
    times_ordered: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    fisher_w: np.ndarray,
    var_w: np.ndarray,
    som_fisher_w: np.ndarray,
    tolerance_samples: int,
    verbose: bool,
):
    """Test attention-weighted P2P in the COM∧P2P two-stage gate (oracle)."""
    from Spatial.algorithms.spatial_footprint import com_features
    from Spatial.experiments.run_two_stage_upper_bound import (
        extract_main_channel_waveforms,
        normalize_waveforms,
    )

    # We need waveforms for stage 2 — extract from ordered data
    # Reconstruct central channels from neighbor_ids slot 0
    central_ordered = neighbor_ids[np.argsort(times_ordered)][test_idx, 0]
    # Actually, let's just use the ordered p2p for gate and skip waveform stage 2
    # Instead, test the gate quality: how many candidates pass with attention-weighted P2P

    com_all = com_features(p2p_raw, neighbor_ids, geom)
    com_ordered = com_all[np.argsort(times_ordered)]

    y_train = labels_ordered[train_idx]
    y_test = labels_ordered[test_idx]
    com_train, com_test = com_ordered[train_idx], com_ordered[test_idx]

    for attn_name, w in [("none", None), ("fisher", fisher_w),
                          ("variance", var_w), ("som_fisher", som_fisher_w)]:
        if w is not None:
            p2p_w = apply_channel_weights(p2p_raw, w)
        else:
            p2p_w = p2p_raw
        fp2p_w = footprint_p2p_features(p2p_w, normalize=True)
        fp2p_ordered = fp2p_w[np.argsort(times_ordered)]
        p2p_train, p2p_test = fp2p_ordered[train_idx], fp2p_ordered[test_idx]

        # Oracle two-stage (no waveform, just gate + stage-1 fallback)
        tau_com, tau_p2p = calibrate_thresholds(
            com_train, p2p_train, y_train, percentile=95.0,
        )
        com_cents = fit_unit_centroids(com_train, y_train)
        p2p_cents = fit_unit_centroids(p2p_train, y_train)

        from Spatial.algorithms.two_stage_assign import stage1_candidates, _fallback_stage1
        n_test = len(com_test)
        pred = np.full(n_test, -1, dtype=np.int64)
        n_cands_list = []
        n_fallback = 0
        for i in range(n_test):
            cands = stage1_candidates(
                com_test[i], p2p_test[i], com_cents, p2p_cents,
                tau_com, tau_p2p,
            )
            n_cands_list.append(len(cands))
            if not cands:
                n_fallback += 1
            pred[i] = _fallback_stage1(
                com_test[i], p2p_test[i], com_cents, p2p_cents,
                tau_com, tau_p2p,
            )

        n_correct = int(np.sum(pred == y_test))
        acc = n_correct / max(n_test, 1)
        row = {
            "dataset": "",
            "method": f"gate_{attn_name}",
            "feature_dim": p2p_train.shape[1],
            "n_spikes": n_test,
            "sep_ratio": float("nan"),
            "oracle_accuracy": float(acc),
            "som_accuracy": float("nan"),
            "som_clusters": int(len(np.unique(pred))),
            "gt_units": int(len(np.unique(y_test))),
            "attention": attn_name,
            "gate_tau_com": float(tau_com),
            "gate_tau_p2p": float(tau_p2p),
            "gate_fallback_rate": float(n_fallback / max(n_test, 1)),
            "gate_mean_candidates": float(np.mean(n_cands_list)),
        }
        rows.append(row)
        if verbose:
            print(
                f"  {'gate_' + attn_name:<36} acc={acc:.3f}  "
                f"fb={row['gate_fallback_rate']:.3f}  "
                f"cands={row['gate_mean_candidates']:.2f}  "
                f"tau_p2p={tau_p2p:.3f}"
            )


def _print_summary(rows: list[dict]):
    print(f"\n{'method':<36} {'dim':>4} {'sep':>8} {'oracle':>8} {'SOM':>8} {'clust':>8}")
    print("-" * 76)
    for r in rows:
        som_str = f"{r['som_accuracy']:.3f}" if r['som_accuracy'] == r['som_accuracy'] else "  N/A"
        print(
            f"{r['method']:<36} {r['feature_dim']:>4} "
            f"{r['sep_ratio']:>8.3f} {r['oracle_accuracy']:>8.3f} "
            f"{som_str:>8} {r['som_clusters']:>3}/{r['gt_units']:<3}"
        )


def main():
    parser = argparse.ArgumentParser(description="SE-Net channel attention experiment")
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-path", type=Path, default=None)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Spatial/output/se_attention"))
    args = parser.parse_args()

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
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

    som_defaults = SOM_PARAMS.get(args.dataset, SOM_PARAMS["hj"])
    grid = args.grid_size or som_defaults["grid_size"]
    beta = args.beta or som_defaults["beta"]

    rows = run_study(
        ds, k_neighbors=args.k_neighbors,
        som_grid=grid, som_beta=beta,
        temperatures=tuple(args.temperatures),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"se_attention_{args.dataset}_{int(args.duration)}s.csv"
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
