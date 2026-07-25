"""Unsupervised two-stage COM∧P2P → waveform (no GT centroids / τ).

Usage:
    python -m Spatial.experiments.run_two_stage_unsupervised --duration 60
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
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    com_features,
    footprint_p2p_features,
    soft_localization_features,
)
from Spatial.algorithms.two_stage_assign import (
    two_stage_assign,
    calibrate_thresholds_codes,
    _intra_unit_distances,
)
from Spatial.algorithms.unsupervised_two_stage import (
    unsupervised_two_stage_assign,
    _gate_p2p_from_raw,
)
from Spatial.evaluation.metrics import clustering_accuracy
from Spatial.experiments.run_two_stage_upper_bound import (
    extract_main_channel_waveforms,
    normalize_waveforms,
    peak_fsde_batch,
    _chronological_split,
)


def _hung_acc(
    pred: np.ndarray,
    gt: np.ndarray,
    times: np.ndarray,
    tolerance_samples: int = 30,
) -> dict:
    """Hungarian clustering accuracy.

    When ``pred``/``gt`` share the same spike times (GT spikes), ``tolerance_samples``
    does not change matching. Parameterize it for future detector-time evaluation.
    """
    m = clustering_accuracy(
        pred, gt, times, times, tolerance_samples=tolerance_samples
    )
    n_matched = max(int(m["n_matched"]), 1)
    return {
        "accuracy": float(m["accuracy"]),
        "accuracy_matched": float(m.get("n_correct", 0) / n_matched),
        "n_correct": int(m.get("n_correct", 0)),
        "n_matched": int(m["n_matched"]),
        "n_gt": int(m["n_gt"]),
        "n_pred_clusters": int(len(np.unique(pred))),
    }


def run_study(
    dataset,
    k_neighbors: int = 7,
    window: int = 15,
    train_frac: float = 0.5,
    percentile: float = 95.0,
    p2p_metrics: tuple[str, ...] = ("l2", "l1"),
    spatial_grid: int = 32,
    wave_grid: int = 16,
    min_cluster_size: int = 30,
    som_beta: int = 8,
    include_oracle: bool = True,
    tolerance_samples: int = 30,
    p2p_repr: str = "maxnorm",
    cim_bits: int | None = None,
    ratio_clip: float = 2.0,
    verbose: bool = True,
) -> list[dict]:
    t0 = time.time()
    if verbose:
        print(f"Dataset: {dataset.name} | {dataset.n_channels}ch | {dataset.duration_s:.1f}s")
        print(f"GT spikes: {len(dataset.spike_times)} | units: {len(dataset.unit_ids)}")
        print(
            f"Mode: UNSUPERVISED two-stage | p2p_repr={p2p_repr} "
            f"cim_bits={cim_bits} clip={ratio_clip}"
        )

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
    soft = soft_localization_features(p2p_raw, neighbor_ids, dataset.geom)
    wfs = extract_main_channel_waveforms(filtered, times, central)
    wf_norm = normalize_waveforms(wfs)
    fsde = peak_fsde_batch(wfs, normalize=True)

    order = np.argsort(times)
    com, p2p, soft = com[order], p2p[order], soft[order]
    p2p_raw = p2p_raw[order]
    fsde, wf_norm = fsde[order], wf_norm[order]
    labels, times = labels[order], times[order]

    train_idx, test_idx = _chronological_split(len(labels), train_frac)
    y_test = labels[test_idx]
    t_test = times[test_idx]

    rows: list[dict] = []

    def add_row(method: str, pred: np.ndarray, extra: dict | None = None):
        m = _hung_acc(pred, y_test, t_test, tolerance_samples=tolerance_samples)
        extra = extra or {}
        row = {
            "dataset": dataset.name,
            "mode": extra.get("mode", "unsupervised"),
            "p2p_metric": extra.get("p2p_metric", ""),
            "method": method,
            "accuracy": m["accuracy"],
            "accuracy_matched": m["accuracy_matched"],
            "n_correct": m["n_correct"],
            "n_matched": m["n_matched"],
            "n_gt": m["n_gt"],
            "n_pred_clusters": m["n_pred_clusters"],
            "n_discovered_train": extra.get("n_discovered_clusters", ""),
            "tau_com": extra.get("tau_com", ""),
            "tau_p2p": extra.get("tau_p2p", ""),
            "fallback_rate": extra.get("fallback_rate", ""),
            "mean_n_candidates": extra.get("mean_n_candidates", ""),
            "gt_units": int(len(np.unique(labels))),
        }
        rows.append(row)
        if verbose:
            fb = ""
            if extra.get("fallback_rate", "") != "":
                fb = (f"  fb={extra['fallback_rate']:.3f}"
                      f"  cands={extra.get('mean_n_candidates', 0):.2f}"
                      f"  K_train={extra.get('n_discovered_clusters', '?')}")
            print(
                f"  {method:<32} acc={m['accuracy']:.3f}  "
                f"matched={m['accuracy_matched']:.3f}  "
                f"K={m['n_pred_clusters']}/{row['gt_units']}{fb}"
            )

    # --- Unsupervised spatial SOM baselines (train→assign test via train centroids) ---
    if verbose:
        print("\n--- unsupervised spatial SOM baselines ---")
    for name, feats in (("som_com", com), ("som_soft_loc", soft)):
        som = SpatialSOM(grid_size=spatial_grid, alpha=16, beta=som_beta, seed=42)
        som.train(feats[train_idx], verbose=False)
        # Assign test by nearest active centroid from train SOM
        pred = som.assign(feats[test_idx])
        add_row(name, pred, {"mode": "unsupervised", "n_discovered_clusters": som.n_clusters})

    # Full-data SOM soft_loc (Akhoundi-style protocol used elsewhere)
    som_full = SpatialSOM(grid_size=spatial_grid, alpha=16, beta=som_beta, seed=42)
    som_full.train(soft, verbose=False)
    pred_full = som_full.assign(soft)
    m_full = clustering_accuracy(
        pred_full, labels, times, times, tolerance_samples=tolerance_samples
    )
    if verbose:
        print(
            f"  {'som_soft_loc_full':<32} acc={m_full['accuracy']:.3f}  "
            f"K={som_full.n_clusters}/{len(np.unique(labels))}  (train=all)"
        )
    rows.append({
        "dataset": dataset.name,
        "mode": "unsupervised_full",
        "p2p_metric": "",
        "method": "som_soft_loc_full",
        "accuracy": float(m_full["accuracy"]),
        "accuracy_matched": float(m_full.get("n_correct", 0) / max(m_full["n_matched"], 1)),
        "n_correct": int(m_full.get("n_correct", 0)),
        "n_matched": int(m_full["n_matched"]),
        "n_gt": int(m_full["n_gt"]),
        "n_pred_clusters": int(som_full.n_clusters),
        "n_discovered_train": int(som_full.n_clusters),
        "tau_com": "",
        "tau_p2p": "",
        "fallback_rate": "",
        "mean_n_candidates": "",
        "gt_units": int(len(np.unique(labels))),
    })

    # Code-space path forces L1; float path keeps requested metrics
    metrics_run: tuple[str, ...] = ("l1",) if cim_bits is not None else p2p_metrics

    # --- Unsupervised two-stage L1/L2 × FSDE/waveform ---
    for p2p_metric in metrics_run:
        if verbose:
            print(f"\n--- unsupervised two-stage | P2P={p2p_metric} ---")
        for wave_name, wave in (("fsde", fsde), ("waveform", wf_norm)):
            # FSDE over-splits easily → smaller wave grid / larger min size
            w_grid = 8 if wave_name == "fsde" else wave_grid
            min_sz = max(min_cluster_size, 50) if wave_name == "fsde" else min_cluster_size
            pred, meta = unsupervised_two_stage_assign(
                com[train_idx],
                p2p[train_idx],
                wave[train_idx],
                com[test_idx],
                p2p[test_idx],
                wave[test_idx],
                percentile=percentile,
                p2p_metric=p2p_metric,  # type: ignore[arg-type]
                spatial_train=soft[train_idx],
                p2p_raw_train=p2p_raw[train_idx],
                p2p_raw_test=p2p_raw[test_idx],
                p2p_repr=p2p_repr,  # type: ignore[arg-type]
                cim_bits=cim_bits,
                ratio_clip=ratio_clip,
                spatial_grid=spatial_grid,
                wave_grid=w_grid,
                min_cluster_size=min_sz,
                spatial_beta=som_beta,
                wave_beta=8 if wave_name == "fsde" else 4,
                seed=42,
            )
            add_row(
                f"unsup_two_stage_{wave_name}",
                pred,
                {**meta, "mode": "unsupervised", "p2p_metric": p2p_metric},
            )

        if include_oracle:
            if verbose:
                print(f"\n--- oracle two-stage (upper bound) | P2P={p2p_metric} ---")
            y_train = labels[train_idx]
            gate_p2p = _gate_p2p_from_raw(
                p2p_raw, p2p_repr, cim_bits, ratio_clip  # type: ignore[arg-type]
            )
            tau_com_o = None
            tau_p2p_o = None
            if cim_bits is not None:
                d_com = _intra_unit_distances(com[train_idx], y_train, metric="l2")
                d_com = d_com[np.isfinite(d_com)]
                tau_com_o = float(np.percentile(d_com, percentile)) if d_com.size else 1.0
                tau_com_o = max(tau_com_o, 1e-8)
                tau_p2p_o = float(
                    calibrate_thresholds_codes(
                        gate_p2p[train_idx], y_train, percentile=percentile
                    )
                )
            for wave_name, wave in (("fsde", fsde), ("waveform", wf_norm)):
                pred_o, meta_o = two_stage_assign(
                    com[train_idx],
                    gate_p2p[train_idx],
                    wave[train_idx],
                    y_train,
                    com[test_idx],
                    gate_p2p[test_idx],
                    wave[test_idx],
                    percentile=percentile,
                    tau_com=tau_com_o,
                    tau_p2p=tau_p2p_o,
                    p2p_metric=p2p_metric,  # type: ignore[arg-type]
                )
                # Oracle labels are GT ids — report exact match as accuracy_matched
                n_correct = int(np.sum(pred_o == y_test))
                n_test = len(y_test)
                row = {
                    "dataset": dataset.name,
                    "mode": "oracle",
                    "p2p_metric": p2p_metric,
                    "method": f"oracle_two_stage_{wave_name}",
                    "accuracy": float(n_correct / max(n_test, 1)),
                    "accuracy_matched": float(n_correct / max(n_test, 1)),
                    "n_correct": n_correct,
                    "n_matched": n_test,
                    "n_gt": n_test,
                    "n_pred_clusters": int(len(np.unique(pred_o))),
                    "n_discovered_train": int(len(np.unique(y_train))),
                    "tau_com": meta_o.get("tau_com", ""),
                    "tau_p2p": meta_o.get("tau_p2p", ""),
                    "fallback_rate": meta_o.get("fallback_rate", ""),
                    "mean_n_candidates": meta_o.get("mean_n_candidates", ""),
                    "gt_units": int(len(np.unique(labels))),
                }
                rows.append(row)
                if verbose:
                    print(
                        f"  {row['method']:<32} acc={row['accuracy']:.3f}  "
                        f"fb={meta_o['fallback_rate']:.3f}  "
                        f"cands={meta_o['mean_n_candidates']:.2f}"
                    )

    if verbose:
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        print(f"\n{'mode':<18} {'p2p':<4} {'method':<32} {'acc':>7}")
        print("-" * 70)
        for r in rows:
            print(
                f"{r['mode']:<18} {str(r['p2p_metric']):<4} "
                f"{r['method']:<32} {r['accuracy']:>7.3f}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Unsupervised two-stage L1/L2 study")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--spatial-grid", type=int, default=32)
    parser.add_argument("--wave-grid", type=int, default=16)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--som-beta", type=int, default=8)
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=30,
        help="Time match tolerance for Hungarian eval (relevant if pred/gt times differ).",
    )
    parser.add_argument(
        "--p2p-repr",
        choices=["maxnorm", "central_ratio"],
        default="maxnorm",
        help="Stage-1 P2P representation (built from p2p_raw).",
    )
    parser.add_argument(
        "--cim-bits",
        type=int,
        default=None,
        help="If set, quantize P2P to n-bit codes and gate with L1 + τ≥1.",
    )
    parser.add_argument(
        "--ratio-clip",
        type=float,
        default=2.0,
        help="Clip for central_ratio float/codes (ignored by maxnorm float).",
    )
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Spatial/output/two_stage_unsupervised_hj60s.csv"),
    )
    args = parser.parse_args()

    ds = load_hybrid_janelia(duration_s=args.duration)
    rows = run_study(
        ds,
        k_neighbors=args.k_neighbors,
        train_frac=args.train_frac,
        percentile=args.percentile,
        spatial_grid=args.spatial_grid,
        wave_grid=args.wave_grid,
        min_cluster_size=args.min_cluster_size,
        som_beta=args.som_beta,
        include_oracle=not args.no_oracle,
        tolerance_samples=args.tolerance_samples,
        p2p_repr=args.p2p_repr,
        cim_bits=args.cim_bits,
        ratio_clip=args.ratio_clip,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
