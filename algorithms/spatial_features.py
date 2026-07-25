"""Spatial feature extraction following Akhoundi et al. 2025 (Eq. 3-5).

For each detected spike:
  - Central channel: channel with max absolute amplitude
  - Direction angle: angle of amplitude-weighted neighbor position vector
  - Distance ratio: max neighbor amplitude / central amplitude
  - Position estimate: central channel position + weighted displacement
"""

import numpy as np


def find_neighbors(geom: np.ndarray, central_idx: int, n_neighbors: int = 6) -> np.ndarray:
    """Find n nearest neighbor channel indices by Euclidean distance."""
    dists = np.linalg.norm(geom - geom[central_idx], axis=1)
    dists[central_idx] = np.inf
    return np.argsort(dists)[:n_neighbors]


def extract_spatial_features(
    waveforms: np.ndarray,
    geom: np.ndarray,
    central_channels: np.ndarray = None,
    n_neighbors: int = 6,
) -> np.ndarray:
    """Extract 5D spatial features for a batch of spikes.

    Args:
        waveforms: (n_spikes, n_channels) peak amplitudes or (n_spikes, n_channels, n_samples) snippets.
                   If 3D, peak amplitude per channel is computed automatically.
        geom: (n_channels, 2) channel positions in um.
        central_channels: (n_spikes,) pre-computed central channel indices.
                         If None, determined from max |amplitude|.
        n_neighbors: number of nearest neighbors to use.

    Returns:
        features: (n_spikes, 5) — [angle, distance_ratio, x_est, y_est, peak_amp]
    """
    if waveforms.ndim == 3:
        # (n_spikes, n_channels, n_samples) -> peak amplitude per channel
        amplitudes = np.max(np.abs(waveforms), axis=2)  # (n_spikes, n_channels)
        # Use signed peak (preserve polarity)
        peak_idx = np.argmax(np.abs(waveforms), axis=2)
        for i in range(waveforms.shape[0]):
            for ch in range(waveforms.shape[1]):
                amplitudes[i, ch] = waveforms[i, ch, peak_idx[i, ch]]
    else:
        amplitudes = waveforms  # (n_spikes, n_channels)

    n_spikes, n_channels = amplitudes.shape
    features = np.zeros((n_spikes, 5), dtype=np.float64)

    if central_channels is None:
        central_channels = np.argmax(np.abs(amplitudes), axis=1)

    # Pre-compute neighbor indices for each unique central channel
    neighbor_cache = {}
    for ch in np.unique(central_channels):
        neighbor_cache[ch] = find_neighbors(geom, ch, n_neighbors)

    for i in range(n_spikes):
        c = central_channels[i]
        a_c = amplitudes[i, c]
        neighbors = neighbor_cache[c]

        # Neighbor amplitudes and relative positions
        a_neighbors = amplitudes[i, neighbors]  # (n_neighbors,)
        rel_pos = geom[neighbors] - geom[c]  # (n_neighbors, 2)

        # Eq. 3: Direction angle — amplitude-weighted position vector
        # Δ_c,s = Σ A_i,s * (x_i, y_i)_s
        weighted_sum = np.sum(a_neighbors[:, None] * rel_pos, axis=0)  # (2,)

        # Normalize by central amplitude to get displacement
        if abs(a_c) > 1e-10:
            delta = weighted_sum / abs(a_c)
        else:
            delta = np.zeros(2)

        angle = np.arctan2(delta[1], delta[0])

        # Eq. 4: Distance ratio — max neighbor / central
        max_neighbor = np.max(np.abs(a_neighbors)) if len(a_neighbors) > 0 else 0.0
        distance_ratio = max_neighbor / abs(a_c) if abs(a_c) > 1e-10 else 0.0

        # Eq. 5: Position estimate
        x_est = geom[c, 0] + delta[0]
        y_est = geom[c, 1] + delta[1]

        features[i] = [angle, distance_ratio, x_est, y_est, a_c]

    return features


def extract_spatial_features_fast(
    amplitudes: np.ndarray,
    geom: np.ndarray,
    central_channels: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Vectorized spatial feature extraction for pre-computed neighbors.

    Args:
        amplitudes: (n_spikes, n_channels) signed peak amplitudes.
        geom: (n_channels, 2) positions.
        central_channels: (n_spikes,) central channel per spike.
        neighbor_indices: (n_channels, n_neighbors) pre-computed KNN table.

    Returns:
        features: (n_spikes, 5) — [angle, distance_ratio, x_est, y_est, peak_amp]
    """
    n_spikes = amplitudes.shape[0]
    n_neighbors = neighbor_indices.shape[1]

    # Gather neighbor amplitudes: (n_spikes, n_neighbors)
    a_neighbors = amplitudes[np.arange(n_spikes)[:, None], neighbor_indices[central_channels]]

    # Relative positions: (n_spikes, n_neighbors, 2)
    rel_pos = geom[neighbor_indices[central_channels]] - geom[central_channels][:, None, :]

    # Weighted displacement (Eq. 3)
    a_c = amplitudes[np.arange(n_spikes), central_channels]
    weighted_sum = np.sum(a_neighbors[:, :, None] * rel_pos, axis=1)  # (n_spikes, 2)

    safe_ac = np.where(np.abs(a_c) > 1e-10, np.abs(a_c), 1.0)
    delta = weighted_sum / safe_ac[:, None]

    angle = np.arctan2(delta[:, 1], delta[:, 0])

    # Distance ratio (Eq. 4)
    max_neighbor = np.max(np.abs(a_neighbors), axis=1)
    distance_ratio = max_neighbor / safe_ac

    # Position estimate (Eq. 5)
    x_est = geom[central_channels, 0] + delta[:, 0]
    y_est = geom[central_channels, 1] + delta[:, 1]

    features = np.column_stack([angle, distance_ratio, x_est, y_est, a_c])
    return features


def build_knn_table(geom: np.ndarray, n_neighbors: int = 6) -> np.ndarray:
    """Pre-compute KNN table for all channels.

    Returns:
        (n_channels, n_neighbors) array of neighbor indices.
    """
    n_channels = geom.shape[0]
    knn = np.zeros((n_channels, n_neighbors), dtype=np.int64)
    for ch in range(n_channels):
        knn[ch] = find_neighbors(geom, ch, n_neighbors)
    return knn
