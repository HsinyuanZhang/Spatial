"""Quick real-data sanity evaluation of StreamingClusterer on HJ scenes.

Compares three configurations on identical event rows:
  - offline-som: full-data SpatialSOM (upper bound, sees all samples)
  - streaming:   StreamingClusterer with 10% warm-up + online update + pool growth
  - frozen:      warm-up templates, no online update, no new units (ablation)

Uses row-aligned GT labels from prepare_spatial_events and Hungarian matching
to map discovered cluster ids to GT unit ids. Reports accuracy = correctly
clustered / total GT events under the best 1:1 label assignment.

This is a quick sanity probe, not a pre-registered experiment.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    com_features,
    extract_local_p2p,
    footprint_p2p_features,
    quantize_p2p,
)
from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.algorithms.streaming_clusterer import StreamingClusterer
from Spatial.data.loader import load_hybrid_janelia

# Filter / window constants (mirror run_causal_temporal_sketch defaults).
FILTER_LOW_HZ = 300.0
FILTER_HIGH_HZ = 6000.0
FILTER_ORDER = 3
PRE_ALIGNMENT_SAMPLES = 15
POST_ALIGNMENT_SAMPLES = 48


def _prepare_minimal(dataset, k_neighbors: int = 7):
    """Self-contained event preparation that avoids the GeoOsort import chain."""
    filtered = bandpass_filter(
        dataset.raw_data, dataset.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ, FILTER_ORDER
    )
    _, times, central = get_peak_amplitudes(
        filtered, dataset.spike_times, window=PRE_ALIGNMENT_SAMPLES
    )

    # Align GT labels to the boundary-filtered times.
    spike_times = np.asarray(dataset.spike_times, dtype=np.int64).ravel()
    spike_units = np.asarray(dataset.spike_units).ravel()
    mask = (spike_times >= PRE_ALIGNMENT_SAMPLES) & (
        spike_times < dataset.n_samples - PRE_ALIGNMENT_SAMPLES
    )
    expected_times = spike_times[mask]
    assert np.array_equal(times, expected_times), "time order mismatch"
    labels = spike_units[mask].copy()

    # Drop trailing events that don't fit the full waveform window (keep
    # consistency with the existing runners' boundary rule).
    joint = times + POST_ALIGNMENT_SAMPLES < dataset.n_samples
    times, central, labels = times[joint], central[joint], labels[joint]
    if times.size == 0:
        raise ValueError("no events remain after boundary filtering")

    table = build_knn_table_with_self(
        np.asarray(dataset.geom, dtype=np.float64), k_neighbors
    )
    p2p, _, neighbors = extract_local_p2p(
        filtered, times, central, table, window=PRE_ALIGNMENT_SAMPLES
    )
    com = com_features(p2p, neighbors, np.asarray(dataset.geom, dtype=np.float64))
    p2p_norm = footprint_p2p_features(p2p, normalize=True)

    order = np.argsort(times, kind="stable")
    return {
        "com": com[order],
        "p2p_norm": p2p_norm[order],
        "labels": labels[order],
        "times": times[order],
    }


def hungarian_row_accuracy(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Best 1:1 label-map accuracy over aligned rows."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    valid = (pred >= 0) & (truth >= 0)
    pred_v, truth_v = pred[valid], truth[valid]
    if pred_v.size == 0:
        return {
            "accuracy": 0.0,
            "n_discovered_units": 0,
            "n_unassigned_pred": int((pred < 0).sum()),
        }
    pred_u = np.unique(pred_v)
    gt_u = np.unique(truth_v)
    cost = np.zeros((len(pred_u), len(gt_u)), dtype=np.int64)
    pu = {u: i for i, u in enumerate(pred_u)}
    gu = {u: j for j, u in enumerate(gt_u)}
    for p, g in zip(pred_v, truth_v):
        cost[pu[p], gu[g]] += 1
    ri, ci = linear_sum_assignment(-cost)
    n_correct = int(cost[ri, ci].sum())
    return {
        "accuracy": float(n_correct / max(len(truth), 1)),
        "n_discovered_units": int(len(pred_u)),
        "n_unassigned_pred": int((pred < 0).sum()),
    }


def run_scene(scene: str, duration: float = 60.0, n_bits: int = 8, seed: int = 0) -> dict:
    ds = load_hybrid_janelia(scene=scene, duration_s=duration)
    prep = _prepare_minimal(ds, k_neighbors=7)
    com = prep["com"]
    p2p_norm = prep["p2p_norm"]
    labels = prep["labels"]
    n = len(labels)
    p2p_codes = quantize_p2p(p2p_norm, n_bits=n_bits, mode="per_spike_max").astype(np.int64)
    n_units_gt = len(np.unique(labels[labels >= 0]))

    # 1. Offline SOM (upper bound: sees all data).
    som = SpatialSOM(
        grid_size=32, alpha=16, beta=8, batch_samples=2000, max_epochs=40, seed=seed
    )
    som.train(com, verbose=False)
    som_labels = som.assign(com)
    offline = hungarian_row_accuracy(som_labels, labels)

    # 2. Streaming: warm-up on first 10%, stream the rest in time order.
    clf = StreamingClusterer(
        n_bits=n_bits,
        warmup_fraction=0.1,
        pool_size=50,
        min_new_cluster_size=5,
        com_grid=32,
        p2p_grid=8,
        margin_floor=1,
        lr_shift=4,
        seed=seed,
    )
    n_warmup = max(int(n * 0.1), 100)
    clf.warmup(com[:n_warmup], p2p_codes[:n_warmup])
    stream_pred = np.full(n - n_warmup, -1, dtype=np.int64)
    for i in range(n_warmup, n):
        res = clf.process(com[i], p2p_codes[i])
        stream_pred[i - n_warmup] = res["assigned_unit"]
    stream_eval = hungarian_row_accuracy(stream_pred, labels[n_warmup:])
    acct = clf.accounting()

    # 3. Frozen ablation: warm-up templates, assign-only (no update, no growth).
    frozen_pred = clf.assign_all(com[n_warmup:], p2p_codes[n_warmup:])
    frozen_eval = hungarian_row_accuracy(frozen_pred, labels[n_warmup:])

    return {
        "scene": scene,
        "n_events": n,
        "n_warmup": n_warmup,
        "n_stream": n - n_warmup,
        "n_gt_units": n_units_gt,
        "offline_som_accuracy": offline["accuracy"],
        "offline_som_n_clusters": int(len(np.unique(som_labels))),
        "streaming_accuracy": stream_eval["accuracy"],
        "streaming_n_discovered": stream_eval["n_discovered_units"],
        "streaming_n_unassigned": stream_eval["n_unassigned_pred"],
        "frozen_accuracy": frozen_eval["accuracy"],
        "frozen_n_discovered": frozen_eval["n_discovered_units"],
        "frozen_n_unassigned": frozen_eval["n_unassigned_pred"],
        "acct": {
            k: acct[k]
            for k in (
                "n_accepted",
                "n_rejected_distance",
                "n_rejected_margin",
                "n_rejected_no_com_candidate",
                "n_pooled",
                "n_recycled",
                "n_units_promoted",
                "n_units_discarded_duplicate",
                "n_pending_pool",
                "n_template_coordinate_writes",
            )
        },
    }


def main() -> None:
    scenes = [
        "drift16c_600s_11",
        "static16c_600s_11",
        "drift16c_600s_21",
        "static16c_600s_21",
    ]
    header = (
        f"{'scene':<22} {'gtu':>4} {'off':>6} {'strm':>6} {'frzn':>6} "
        f"{'disc':>5} {'prom':>5} {'pool':>5} {'rec':>5} {'rjD':>6} {'rjM':>6} {'rjC':>6}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for s in scenes:
        try:
            r = run_scene(s, duration=60.0, n_bits=8, seed=0)
        except Exception as exc:  # noqa: BLE001
            print(f"{s:<22} ERROR: {type(exc).__name__}: {exc}")
            continue
        a = r["acct"]
        print(
            f"{s:<22} {r['n_gt_units']:>4} "
            f"{r['offline_som_accuracy']:>6.3f} "
            f"{r['streaming_accuracy']:>6.3f} "
            f"{r['frozen_accuracy']:>6.3f} "
            f"{r['streaming_n_discovered']:>5} "
            f"{a['n_units_promoted']:>5} "
            f"{a['n_pending_pool']:>5} "
            f"{a['n_recycled']:>5} "
            f"{a['n_rejected_distance']:>6} "
            f"{a['n_rejected_margin']:>6} "
            f"{a['n_rejected_no_com_candidate']:>6}"
        )
        results.append(r)

    if results:
        off = float(np.mean([r["offline_som_accuracy"] for r in results]))
        stm = float(np.mean([r["streaming_accuracy"] for r in results]))
        frz = float(np.mean([r["frozen_accuracy"] for r in results]))
        print("-" * len(header))
        print(f"{'MEAN':<22} {'':>4} {off:>6.3f} {stm:>6.3f} {frz:>6.3f}")
        print()
        print("Legend:")
        print("  off   = full-data SOM (upper bound, sees all samples)")
        print("  strm  = StreamingClusterer (10% warm-up + online update + pool growth)")
        print("  frzn  = warm-up templates, assign-only (no update, no growth)")
        print("  gtu   = GT units; disc = discovered units; prom = units promoted from pool")
        print("  pool  = events still in pending pool; rec = recycled to existing units")
        print("  rjD/rjM/rjC = rejected by distance / margin / no-COM-candidate")


if __name__ == "__main__":
    main()
