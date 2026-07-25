"""Unsupervised two-stage sorting: spatial SOM → within-cluster wave SOM → COM∧P2P gate.

No GT labels are used for centroids, thresholds, or assignment. P2P gate metric
may be L1 or L2 (COM stays L2). Evaluation against GT uses Hungarian matching.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from Spatial.algorithms.som_clustering import SpatialSOM
from Spatial.algorithms.spatial_footprint import (
    central_ratio_features,
    footprint_p2p_features,
    quantize_p2p,
)
from Spatial.algorithms.two_stage_assign import (
    two_stage_assign,
    calibrate_thresholds_codes,
    _intra_unit_distances,
)

DistanceMetric = Literal["l2", "l1"]
P2PRepr = Literal["maxnorm", "central_ratio"]


def _gate_p2p_from_raw(
    p2p_raw: np.ndarray,
    p2p_repr: P2PRepr,
    cim_bits: int | None,
    ratio_clip: float,
) -> np.ndarray:
    """Build stage-1 P2P features/codes from raw neighborhood amplitudes."""
    raw = np.asarray(p2p_raw, dtype=np.float64)
    if cim_bits is not None:
        mode = "per_spike_max" if p2p_repr == "maxnorm" else "central_ratio"
        return quantize_p2p(raw, n_bits=int(cim_bits), mode=mode, clip=ratio_clip)
    if p2p_repr == "maxnorm":
        return footprint_p2p_features(raw, normalize=True)
    if p2p_repr == "central_ratio":
        return central_ratio_features(raw, clip=ratio_clip)
    raise ValueError(f"Unknown p2p_repr: {p2p_repr}")


def discover_hierarchical_labels(
    com: np.ndarray,
    wave: np.ndarray,
    spatial_grid: int = 32,
    wave_grid: int = 16,
    min_cluster_size: int = 30,
    spatial_alpha: int = 16,
    spatial_beta: int = 8,
    wave_alpha: int = 16,
    wave_beta: int = 4,
    batch_samples: int = 2000,
    max_epochs: int = 40,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Discover cluster labels without GT: COM SOM, then waveform SOM per spatial group.

    Returns:
        labels: (n,) integer cluster ids (contiguous from 0).
        meta: spatial / wave SOM epoch and convergence diagnostics.
    """
    com = np.asarray(com, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    n = len(com)
    if n == 0:
        return np.empty((0,), dtype=np.int64), {
            "spatial_n_epochs": 0,
            "spatial_converged": False,
            "wave_n_epochs": [],
            "wave_converged": [],
            "n_wave_soms": 0,
        }

    som_s = SpatialSOM(
        grid_size=spatial_grid,
        alpha=spatial_alpha,
        beta=spatial_beta,
        batch_samples=min(batch_samples, max(n, 1)),
        max_epochs=max_epochs,
        seed=seed,
    )
    som_s.train(com, verbose=False)
    spatial_lab = som_s.assign(com)

    raw = np.full(n, -1, dtype=np.int64)
    next_id = 0
    wave_n_epochs: list[int] = []
    wave_converged: list[bool] = []
    min_size = max(int(min_cluster_size), 2)

    for s in np.unique(spatial_lab):
        idx = np.where(spatial_lab == s)[0]
        if len(idx) < min_size:
            raw[idx] = next_id
            next_id += 1
            continue

        # Small grids for within-cluster waveform split
        g = min(wave_grid, max(2, int(np.sqrt(len(idx)))))
        som_w = SpatialSOM(
            grid_size=g,
            alpha=wave_alpha,
            beta=wave_beta,
            batch_samples=min(batch_samples, len(idx)),
            max_epochs=max_epochs,
            seed=seed + int(s) + 1,
        )
        som_w.train(wave[idx], verbose=False)
        wave_n_epochs.append(int(som_w.n_epochs))
        wave_converged.append(bool(som_w.converged))
        w_lab = som_w.assign(wave[idx])

        # Remap local wave labels to global ids
        remap: dict[int, int] = {}
        for wl in np.unique(w_lab):
            remap[int(wl)] = next_id
            next_id += 1
        for j, wl in zip(idx, w_lab):
            raw[j] = remap[int(wl)]

    # Compact to 0..K-1
    uniq = np.unique(raw)
    remap_all = {int(u): i for i, u in enumerate(uniq)}
    labels = np.array([remap_all[int(u)] for u in raw], dtype=np.int64)
    meta: dict[str, Any] = {
        "spatial_n_epochs": int(som_s.n_epochs),
        "spatial_converged": bool(som_s.converged),
        "wave_n_epochs": wave_n_epochs,
        "wave_converged": wave_converged,
        "n_wave_soms": int(len(wave_n_epochs)),
        "wave_all_converged": bool(all(wave_converged)) if wave_converged else True,
    }
    return labels, meta


def unsupervised_two_stage_assign(
    com_train: np.ndarray,
    p2p_train: np.ndarray,
    wave_train: np.ndarray,
    com_test: np.ndarray,
    p2p_test: np.ndarray,
    wave_test: np.ndarray,
    percentile: float = 95.0,
    p2p_metric: DistanceMetric = "l2",
    com_metric: DistanceMetric = "l2",
    spatial_train: np.ndarray | None = None,
    p2p_raw_train: np.ndarray | None = None,
    p2p_raw_test: np.ndarray | None = None,
    p2p_repr: P2PRepr = "maxnorm",
    cim_bits: int | None = None,
    ratio_clip: float = 2.0,
    spatial_grid: int = 32,
    wave_grid: int = 16,
    min_cluster_size: int = 30,
    spatial_alpha: int = 16,
    spatial_beta: int = 8,
    wave_alpha: int = 16,
    wave_beta: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit hierarchical clusters on train, assign test with COM∧P2P → wave gate.

    Args:
        p2p_metric: Distance for P2P gate / fallback (`l1` or `l2`).
        com_metric: Distance for COM gate (default `l2`).
        spatial_train: Features for spatial SOM discovery (default: ``com_train``).
            Gate still uses COM∧P2P; discovery can use e.g. soft_loc.
        p2p_raw_train/test: Optional raw neighborhood P2P. When set, rebuilds
            gate features via ``p2p_repr`` / ``cim_bits`` (overrides ``p2p_*``).

    Returns:
        pred_test: (n_test,) discovered cluster ids.
        meta: diagnostics including train labels, gate stats, and SOM convergence.
    """
    if p2p_raw_train is not None:
        if p2p_raw_test is None:
            raise ValueError("p2p_raw_test required when p2p_raw_train is set")
        p2p_train = _gate_p2p_from_raw(
            p2p_raw_train, p2p_repr, cim_bits, ratio_clip
        )
        p2p_test = _gate_p2p_from_raw(
            p2p_raw_test, p2p_repr, cim_bits, ratio_clip
        )

    spat = com_train if spatial_train is None else spatial_train
    labels_train, disc_meta = discover_hierarchical_labels(
        spat,
        wave_train,
        spatial_grid=spatial_grid,
        wave_grid=wave_grid,
        min_cluster_size=min_cluster_size,
        spatial_alpha=spatial_alpha,
        spatial_beta=spatial_beta,
        wave_alpha=wave_alpha,
        wave_beta=wave_beta,
        seed=seed,
    )

    tau_com = None
    tau_p2p = None
    metric: DistanceMetric = p2p_metric
    if cim_bits is not None:
        # Code-space L1 with precomputed τ≥1 (same path as oracle verify script)
        d_com = _intra_unit_distances(com_train, labels_train, metric=com_metric)
        d_com = d_com[np.isfinite(d_com)]
        tau_com = float(np.percentile(d_com, percentile)) if d_com.size else 1.0
        tau_com = max(tau_com, 1e-8)
        tau_p2p = float(
            calibrate_thresholds_codes(p2p_train, labels_train, percentile=percentile)
        )
        metric = "l1"

    pred, gate_meta = two_stage_assign(
        com_train,
        p2p_train,
        wave_train,
        labels_train,
        com_test,
        p2p_test,
        wave_test,
        percentile=percentile,
        tau_com=tau_com,
        tau_p2p=tau_p2p,
        com_metric=com_metric,
        p2p_metric=metric,
    )

    meta: dict[str, Any] = {
        **gate_meta,
        **disc_meta,
        "labels_train": labels_train,
        "n_discovered_clusters": int(len(np.unique(labels_train))),
        "spatial_grid": int(spatial_grid),
        "wave_grid": int(wave_grid),
        "min_cluster_size": int(min_cluster_size),
        "p2p_repr": p2p_repr,
        "cim_bits": cim_bits,
        "ratio_clip": float(ratio_clip),
        "supervised": False,
    }
    return pred, meta
