"""Multi-scale (FPN-style) footprint feature extraction and fusion.

Extracts soft_loc features at multiple neighborhood sizes K and fuses them
into a single representation. Inspired by Feature Pyramid Networks: small K
captures sharp/local footprints (near neurons), large K captures diffuse
footprints (far neurons).
"""

from __future__ import annotations

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    soft_localization_features,
    footprint_p2p_features,
    footprint_pca_features,
)


def extract_multiscale_p2p(
    filtered: np.ndarray,
    spike_times: np.ndarray,
    central_channels: np.ndarray,
    geom: np.ndarray,
    k_list: list[int],
    window: int = 15,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract local P2P at each K in k_list.

    Returns:
        {K: (p2p, valid_times, neighbor_ids)} for each K.
        All K share the same valid_times (intersection of validity masks).
    """
    n_channels = geom.shape[0]
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # First pass: extract per-K, collect valid time sets
    per_k_raw: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in k_list:
        k_eff = min(k, n_channels)
        nbr = build_knn_table_with_self(geom, k_eff)
        p2p, times, nbr_ids = extract_local_p2p(
            filtered, spike_times, central_channels, nbr, window=window
        )
        per_k_raw[k] = (p2p, times, nbr_ids)

    # Use the first K's valid times as reference (all should be identical
    # since validity depends only on spike_times and window, not K)
    ref_k = k_list[0]
    ref_times = per_k_raw[ref_k][1]

    for k in k_list:
        p2p, times, nbr_ids = per_k_raw[k]
        # Defensive: align to reference times if needed
        if len(times) != len(ref_times) or not np.array_equal(times, ref_times):
            time_to_idx = {int(t): i for i, t in enumerate(times)}
            keep = [time_to_idx[int(t)] for t in ref_times if int(t) in time_to_idx]
            p2p = p2p[keep]
            nbr_ids = nbr_ids[keep]
            times = ref_times[keep]
        results[k] = (p2p, times, nbr_ids)

    return results


def multiscale_soft_loc(
    p2p_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    geom: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute soft_loc features at each K.

    Args:
        p2p_dict: output of extract_multiscale_p2p
        geom: (n_channels, 2) electrode geometry

    Returns:
        {K: (n_spikes, 4) soft_loc features}
    """
    out: dict[int, np.ndarray] = {}
    for k, (p2p, _times, nbr_ids) in p2p_dict.items():
        out[k] = soft_localization_features(p2p, nbr_ids, geom)
    return out


def multiscale_footprint_p2p(
    p2p_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    normalize: bool = True,
) -> dict[int, np.ndarray]:
    """Compute normalized P2P footprint at each K.

    Returns:
        {K: (n_spikes, K) footprint features}
    """
    out: dict[int, np.ndarray] = {}
    for k, (p2p, _times, _nbr_ids) in p2p_dict.items():
        out[k] = footprint_p2p_features(p2p, normalize=normalize)
    return out


def fuse_features(
    feature_dict: dict[int, np.ndarray],
    method: str = "concat",
    pca_components: int = 4,
    weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Fuse multi-scale features into a single matrix.

    Args:
        feature_dict: {K: (n_spikes, D_K) features}. All must have same n_spikes.
        method: fusion strategy
            - "concat": concatenate along feature axis → (n, sum(D_K))
            - "mean": element-wise mean (requires all D_K equal) → (n, D)
            - "pca": concat then PCA → (n, pca_components)
            - "weighted": weighted sum (requires all D_K equal) → (n, D)
        pca_components: number of PCA components (for method="pca")
        weights: per-K weights for method="weighted". If None, uniform.

    Returns:
        (n_spikes, D_fused) fused feature matrix
    """
    if not feature_dict:
        raise ValueError("Empty feature_dict")

    k_list = sorted(feature_dict.keys())
    arrays = [feature_dict[k] for k in k_list]
    n_spikes = arrays[0].shape[0]

    for a in arrays:
        if a.shape[0] != n_spikes:
            raise ValueError(
                f"All feature arrays must have same n_spikes, got {a.shape[0]} vs {n_spikes}"
            )

    if method == "concat":
        return np.concatenate(arrays, axis=1)

    if method == "mean":
        dims = {a.shape[1] for a in arrays}
        if len(dims) != 1:
            raise ValueError(
                f"mean fusion requires equal feature dims, got {dims}"
            )
        return np.mean(np.stack(arrays, axis=0), axis=0)

    if method == "pca":
        concat = np.concatenate(arrays, axis=1)
        return footprint_pca_features(concat, n_components=pca_components)

    if method == "weighted":
        dims = {a.shape[1] for a in arrays}
        if len(dims) != 1:
            raise ValueError(
                f"weighted fusion requires equal feature dims, got {dims}"
            )
        if weights is None:
            w = {k: 1.0 / len(k_list) for k in k_list}
        else:
            total = sum(weights.get(k, 0.0) for k in k_list)
            w = {k: weights.get(k, 0.0) / max(total, 1e-8) for k in k_list}
        stacked = np.stack(arrays, axis=0)  # (n_scales, n_spikes, D)
        w_arr = np.array([w[k] for k in k_list])[:, None, None]
        return np.sum(stacked * w_arr, axis=0)

    raise ValueError(f"Unknown fusion method: {method}")


def multiscale_com_spread_sharpness(
    p2p_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    geom: np.ndarray,
    ref_k: int | None = None,
) -> np.ndarray:
    """COM from reference K + spread/sharpness at all scales.

    Returns (n_spikes, 2 + 2*len(k_list)):
        [cx, cy, spread_k1, sharp_k1, spread_k2, sharp_k2, ...]
    """
    k_list = sorted(p2p_dict.keys())
    if ref_k is None:
        ref_k = k_list[len(k_list) // 2]  # middle K as default

    soft_dict = multiscale_soft_loc(p2p_dict, geom)

    # COM from reference K
    com = soft_dict[ref_k][:, :2]

    # Spread + sharpness from all K
    ss_parts = []
    for k in k_list:
        ss_parts.append(soft_dict[k][:, 2:4])  # [spread, sharpness]

    return np.concatenate([com] + ss_parts, axis=1)
