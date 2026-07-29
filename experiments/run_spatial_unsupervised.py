"""Stage C: unsupervised end-to-end on quantized spatial descriptors.

Discovers clusters with SpatialSOM on the quantized (or float) descriptor,
assigns test spikes by nearest discovered centroid (integer L1 when quantized),
and scores with Hungarian matching against GT — matching the protocol in
``docs/two_stage_unsupervised.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from Spatial.algorithms.adaptive_range_search import UniformUnsignedQuantizer
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz
from Spatial.experiments.spatial_cim_common import (
    prepare_spatial_events,
    sha256_json,
    write_csv,
    write_json,
)


def hungarian_accuracy(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    pred_u = np.unique(pred)
    truth_u = np.unique(truth)
    cost = np.zeros((len(pred_u), len(truth_u)), dtype=np.int64)
    for i, p in enumerate(pred_u):
        for j, t in enumerate(truth_u):
            cost[i, j] = int(np.sum((pred == p) & (truth == t)))
    if cost.size == 0:
        return {"accuracy": 0.0, "n_correct": 0, "n_events": int(truth.size)}
    r, c = linear_sum_assignment(-cost)
    n_correct = int(cost[r, c].sum())
    return {
        "accuracy": n_correct / max(int(truth.size), 1),
        "n_correct": n_correct,
        "n_events": int(truth.size),
        "n_pred_clusters": int(len(pred_u)),
        "n_gt_units": int(len(truth_u)),
    }


def run_one(
    dataset_id: str,
    family: str,
    dataset,
    *,
    feature_set: str = "com_and_p2p",
    n_bits: int | None = 5,
    grid_size: int = 32,
) -> dict[str, Any]:
    prepared = prepare_spatial_events(dataset, k_neighbors=7)
    # Chronological 50/50 for unsupervised (train discover, test assign).
    times = prepared["times"]
    n = times.size
    # Use first half / second half at a timestamp boundary.
    mid = n // 2
    # Snap to timestamp change.
    boundaries = np.flatnonzero(times[1:] != times[:-1]) + 1
    if boundaries.size:
        mid = int(boundaries[np.argmin(np.abs(boundaries - mid))])
    train_idx = np.arange(mid)
    test_idx = np.arange(mid, n)
    values = prepared["features"][feature_set]
    labels = prepared["labels"]

    if n_bits is not None:
        quantizer = UniformUnsignedQuantizer(n_bits, fixed_unit_interval=True)
        train_codes = quantizer.fit_transform(values[train_idx])
        test_codes = quantizer.transform(values[test_idx])
        train_feat = train_codes.astype(np.float64)
        test_feat = test_codes.astype(np.float64)
        metric = "l1"
    else:
        train_feat = values[train_idx]
        test_feat = values[test_idx]
        train_codes = None
        test_codes = None
        metric = "l2"

    som = SpatialSOM(grid_size=grid_size, alpha=16, beta=8, seed=42)
    som.train(train_feat, verbose=False)
    train_clusters = som.assign(train_feat)
    # Build cluster centroids in the same space used for assignment.
    cluster_ids = np.unique(train_clusters)
    if metric == "l1":
        assert train_codes is not None and test_codes is not None
        centroids = np.stack(
            [
                np.rint(train_codes[train_clusters == c].mean(axis=0)).astype(np.int64)
                for c in cluster_ids
            ],
            axis=0,
        )
        distances = np.abs(test_codes[:, np.newaxis, :] - centroids[np.newaxis, :, :]).sum(axis=2)
    else:
        centroids = np.stack(
            [train_feat[train_clusters == c].mean(axis=0) for c in cluster_ids],
            axis=0,
        )
        distances = np.linalg.norm(
            test_feat[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
    pred = cluster_ids[np.argmin(distances, axis=1)]
    metrics = hungarian_accuracy(pred, labels[test_idx])
    return {
        "dataset": dataset_id,
        "family": family,
        "feature_set": feature_set,
        "n_bits": -1 if n_bits is None else int(n_bits),
        "metric": metric,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_discovered_clusters": int(len(cluster_ids)),
        "som_n_epochs": int(som.n_epochs),
        "som_converged": bool(som.converged),
        **metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial unsupervised Stage C")
    parser.add_argument("--feature-set", default="com_and_p2p")
    parser.add_argument("--n-bits", type=int, default=5)
    parser.add_argument("--float", action="store_true", help="Use float features instead of low-bit")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "spatial_cim_unsupervised",
    )
    parser.add_argument("--pilot-only", action="store_true", help="Only the four locked pilots")
    args = parser.parse_args(argv)
    n_bits: int | None = None if args.float else args.n_bits

    rows: list[dict[str, Any]] = []

    if args.pilot_only:
        from Spatial.experiments.run_causal_temporal_sketch import discover_pilot_datasets

        found = discover_pilot_datasets(duration_s=args.duration)
        for family in ("hj", "mearec"):
            for spec in sorted(found[family], key=lambda s: s.dataset_id):
                print(f"unsup {spec.dataset_id} ...", flush=True)
                rows.append(
                    run_one(
                        spec.dataset_id, family, spec.load_dataset(),
                        feature_set=args.feature_set,
                        n_bits=n_bits,
                        grid_size=args.grid_size,
                    )
                )
    else:
        for scene in list_hybrid_janelia_scenes(ready_only=True):
            key = scene.get("short_name") or scene.get("scene_key")
            path = Path(scene.get("npz_path", ""))
            if not path.exists():
                continue
            dataset_id = f"hybrid_janelia_{str(key).replace('_filtered_gt', '')}"
            print(f"unsup HJ {dataset_id} ...", flush=True)
            dataset = load_hybrid_janelia(path=path, duration_s=args.duration)
            rows.append(
                run_one(
                    dataset_id, "hj", dataset,
                    feature_set=args.feature_set,
                    n_bits=n_bits,
                    grid_size=args.grid_size,
                )
            )
        for path in discover_npz(Path(DEFAULT_NPZ_DIR)):
            dataset_id = f"mearec_{path.stem}"
            print(f"unsup MEArec {dataset_id} ...", flush=True)
            dataset = load_mearec_npz(path, duration_s=args.duration)
            rows.append(
                run_one(
                    dataset_id, "mearec", dataset,
                    feature_set=args.feature_set,
                    n_bits=n_bits,
                    grid_size=args.grid_size,
                )
            )

    summaries: dict[str, Any] = {}
    for family in ("hj", "mearec"):
        subset = [r for r in rows if r["family"] == family]
        if not subset:
            continue
        n = sum(r["n_events"] for r in subset)
        summaries[family] = {
            "n_recordings": len(subset),
            "n_events": n,
            "accuracy": sum(r["accuracy"] * r["n_events"] for r in subset) / max(n, 1),
            "reference_float_unsupervised": 0.752 if family == "hj" else 0.754,
        }
        summaries[family]["holds_reference_line"] = (
            summaries[family]["accuracy"] >= summaries[family]["reference_float_unsupervised"] - 0.02
        )

    write_csv(args.output_dir / "spatial_unsupervised.csv", rows)
    write_json(
        args.output_dir / "spatial_unsupervised_summary.json",
        {
            "config": {
                "feature_set": args.feature_set,
                "n_bits": -1 if n_bits is None else n_bits,
                "grid_size": args.grid_size,
            },
            "summaries": summaries,
            "digest": sha256_json(
                [{"dataset": r["dataset"], "n_correct": r["n_correct"], "n_events": r["n_events"]}
                 for r in rows]
            ),
        },
    )
    print(summaries)


if __name__ == "__main__":
    main()
