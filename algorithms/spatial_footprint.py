"""Spatial-only footprint / soft-localization features (no temporal waveform PCA/FSDE).

All features are derived from per-channel peak-to-peak (P2P) amplitudes in a
local neighborhood around each spike's central channel, plus electrode geometry.
"""

from __future__ import annotations

import numpy as np


def build_knn_table_with_self(geom: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Per-channel KNN table including self at slot 0 (sorted by distance).

    Returns:
        (n_channels, K) int64 neighbor indices, K = min(k_neighbors, n_channels).
    """
    g = np.asarray(geom, dtype=np.float64)
    n_channels = g.shape[0]
    k = int(max(1, min(k_neighbors, n_channels)))
    table = np.zeros((n_channels, k), dtype=np.int64)
    for ch in range(n_channels):
        dists = np.linalg.norm(g - g[ch], axis=1)
        table[ch] = np.argsort(dists)[:k]
    return table


def extract_local_p2p(
    signal: np.ndarray,
    spike_times: np.ndarray,
    central_channels: np.ndarray,
    neighbor_table: np.ndarray,
    window: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract local-neighborhood P2P amplitudes around each spike.

    Args:
        signal: (n_channels, n_samples) filtered recording.
        spike_times: (n_spikes,) sample indices.
        central_channels: (n_spikes,) home channel per spike.
        neighbor_table: (n_channels, K) including self.
        window: half-window samples on each side of the spike time.

    Returns:
        p2p: (n_valid, K) peak-to-peak amplitudes.
        valid_times: (n_valid,) spike times kept.
        neighbor_ids: (n_valid, K) channel ids used for each spike.
    """
    sig = np.asarray(signal, dtype=np.float64)
    times = np.asarray(spike_times, dtype=np.int64).ravel()
    homes = np.asarray(central_channels, dtype=np.int64).ravel()
    nbr_table = np.asarray(neighbor_table, dtype=np.int64)

    if times.size != homes.size:
        raise ValueError("spike_times and central_channels length mismatch")

    n_channels, n_samples = sig.shape
    k = nbr_table.shape[1]
    valid_mask = (times >= window) & (times < n_samples - window)
    times = times[valid_mask]
    homes = homes[valid_mask]
    n = times.size

    p2p = np.zeros((n, k), dtype=np.float64)
    neighbor_ids = np.zeros((n, k), dtype=np.int64)
    for i, (t, ch) in enumerate(zip(times, homes)):
        ch = int(np.clip(ch, 0, n_channels - 1))
        nbr = nbr_table[ch]
        neighbor_ids[i] = nbr
        seg = sig[nbr, t - window : t + window]
        p2p[i] = np.max(seg, axis=1) - np.min(seg, axis=1)

    return p2p, times, neighbor_ids


def central_ratio_features(
    p2p: np.ndarray,
    clip: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """P2P relative to central channel (slot 0). Shape (n_spikes, K).

    Values in ``[0, clip]``. Central channel is always 1.0 when ``ref > 0``.
    Requires neighbor layout from ``build_knn_table_with_self`` (self at slot 0).
    """
    amp = np.maximum(np.asarray(p2p, dtype=np.float64), 0.0)
    if amp.ndim != 2:
        raise ValueError(f"Expected 2D p2p, got {amp.shape}")
    if amp.shape[0] == 0:
        return amp.copy()
    ref = amp[:, 0:1]
    # central P2P≈0 → all-zero footprint (avoid 1/eps blow-up)
    ratios = np.where(ref > eps, amp / np.maximum(ref, eps), 0.0)
    return np.clip(ratios, 0.0, float(clip))


def quantize_p2p(
    p2p: np.ndarray,
    n_bits: int | None,
    mode: str = "per_spike_max",
    lsb: float | None = None,
    eps: float = 1e-8,
    clip: float = 2.0,
) -> np.ndarray:
    """Quantize local P2P amplitudes to unsigned n-bit integer levels.

    Returns float arrays of integer codes in ``[0, 2**n_bits - 1]`` so that
    downstream ``soft_loc`` / ``footprint_*`` can consume them unchanged
    (they re-normalize). ``n_bits=None`` is a no-op passthrough.

    Modes:
      - ``per_spike_max``: each spike scaled by its own max P2P (relative footprint).
      - ``global_max``: all spikes scaled by the recording-wide max P2P.
      - ``lsb``: ``round(p2p / lsb)`` clipped to the n-bit range (requires ``lsb``).
      - ``central_ratio``: ÷ central (slot 0), clip to ``[0, clip]``, then n-bit.
    """
    amp = np.maximum(np.asarray(p2p, dtype=np.float64), 0.0)
    if n_bits is None:
        return amp.copy()
    n_bits = int(n_bits)
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1 or None, got {n_bits}")
    if amp.ndim != 2:
        raise ValueError(f"Expected 2D p2p, got {amp.shape}")

    max_level = float((1 << n_bits) - 1)
    if mode == "per_spike_max":
        scale = np.max(amp, axis=1, keepdims=True)
        scaled = amp / np.maximum(scale, eps) * max_level
    elif mode == "global_max":
        gmax = float(np.max(amp)) if amp.size else 0.0
        scaled = amp / max(gmax, eps) * max_level
    elif mode == "lsb":
        if lsb is None or float(lsb) <= 0:
            raise ValueError("mode='lsb' requires a positive lsb")
        scaled = amp / float(lsb)
    elif mode == "central_ratio":
        ref = amp[:, 0:1]
        ratios = np.where(ref > eps, amp / np.maximum(ref, eps), 0.0)
        ratios = np.clip(ratios, 0.0, float(clip))
        scaled = ratios / max(float(clip), eps) * max_level
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    codes = np.rint(scaled)
    return np.clip(codes, 0.0, max_level)


def main_channel_features(
    central_channels: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Normalized central-channel index. Shape (n_spikes, 1)."""
    ch = np.asarray(central_channels, dtype=np.float64).ravel()
    denom = float(max(int(n_channels) - 1, 1))
    return (ch / denom)[:, None]


def soft_localization_features(
    p2p: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Geometry-weighted COM (x,y), spread, sharpness. Shape (n_spikes, 4).

    Columns: [centroid_x_norm, centroid_y_norm, spread, sharpness], all in [0, 1].
    """
    amp = np.asarray(p2p, dtype=np.float64)
    nbr = np.asarray(neighbor_ids, dtype=np.int64)
    g = np.asarray(geom, dtype=np.float64)

    if amp.shape != nbr.shape:
        raise ValueError("p2p and neighbor_ids shape mismatch")
    if amp.ndim != 2:
        raise ValueError(f"Expected 2D p2p, got {amp.shape}")
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    if amp.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)

    amp = np.maximum(amp, 0.0)
    mass = np.sum(amp, axis=1, keepdims=True)
    weights = amp / np.maximum(mass, eps)
    xy = g[nbr]
    centroid = np.sum(weights[:, :, None] * xy, axis=1)

    geom_min = np.min(g, axis=0)
    geom_span = np.max(g, axis=0) - geom_min
    centroid_norm = np.zeros_like(centroid)
    active = geom_span > eps
    centroid_norm[:, active] = (centroid[:, active] - geom_min[active]) / geom_span[active]

    squared_radius = np.sum((xy - centroid[:, None, :]) ** 2, axis=2)
    probe_diag = max(float(np.linalg.norm(geom_span)), eps)
    spread = np.sqrt(np.sum(weights * squared_radius, axis=1)) / probe_diag
    sharpness = np.max(weights, axis=1)

    feats = np.column_stack([centroid_norm, spread, sharpness])
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(feats, 0.0, 1.0)


def com_features(
    p2p: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Geometry-weighted COM only. Shape (n_spikes, 2): [cx_norm, cy_norm]."""
    return soft_localization_features(p2p, neighbor_ids, geom, eps=eps)[:, :2]


def footprint_p2p_features(
    p2p: np.ndarray,
    eps: float = 1e-8,
    normalize: bool = True,
) -> np.ndarray:
    """Local K-neighbor P2P amplitude vector. Shape (n_spikes, K).

    If normalize=True, divide by per-spike max P2P (scale-invariant footprint).
    """
    amp = np.maximum(np.asarray(p2p, dtype=np.float64), 0.0)
    if amp.ndim != 2:
        raise ValueError(f"Expected 2D p2p, got {amp.shape}")
    if amp.shape[0] == 0:
        return amp.copy()
    if not normalize:
        return amp
    scale = np.max(amp, axis=1, keepdims=True)
    return amp / np.maximum(scale, eps)


def footprint_pca_features(
    footprint: np.ndarray,
    n_components: int = 4,
) -> np.ndarray:
    """PCA of footprint vectors via SVD. Shape (n_spikes, n_components)."""
    x = np.asarray(footprint, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D footprint, got {x.shape}")
    n, d = x.shape
    n_comp = int(max(1, min(n_components, d, max(n, 1))))
    if n == 0:
        return np.empty((0, n_comp), dtype=np.float64)
    if n == 1 or d == 1:
        out = np.zeros((n, n_comp), dtype=np.float64)
        out[:, 0] = x[:, 0] if d >= 1 else 0.0
        return out

    mu = np.mean(x, axis=0)
    xc = x - mu
    # Economy SVD; columns of Vt are principal axes
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:n_comp]
    return xc @ components.T
