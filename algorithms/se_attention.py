"""SE-Net style channel attention for spatial footprint features.

Adapts Squeeze-and-Excitation Networks (Hu et al. 2018) to spike sorting:
instead of learning channel weights via backprop, we derive them from
statistical properties of the footprint data (Fisher discriminant, variance,
cluster profiles).

All attention weights are non-negative and sum to 1 (soft attention).
"""

from __future__ import annotations

import numpy as np


def fisher_channel_weights(
    p2p: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel Fisher discriminant ratio as attention weight.

    For each channel k:
        weight_k = inter-cluster variance / intra-cluster variance

    Channels that discriminate well between units get high weight.

    Args:
        p2p: (n_spikes, K) footprint amplitudes
        labels: (n_spikes,) unit labels

    Returns:
        (K,) weights normalized to sum=1
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n, k_dim = p2p.shape
    units = np.unique(labels)

    if len(units) < 2 or n < 2:
        return np.ones(k_dim, dtype=np.float64) / k_dim

    global_mean = np.mean(p2p, axis=0)  # (K,)

    inter_var = np.zeros(k_dim, dtype=np.float64)
    intra_var = np.zeros(k_dim, dtype=np.float64)

    for u in units:
        mask = labels == u
        n_u = np.sum(mask)
        if n_u == 0:
            continue
        unit_mean = np.mean(p2p[mask], axis=0)
        inter_var += n_u * (unit_mean - global_mean) ** 2
        if n_u > 1:
            intra_var += np.sum((p2p[mask] - unit_mean) ** 2, axis=0)

    inter_var /= max(n, 1)
    intra_var /= max(n - len(units), 1)

    ratio = inter_var / np.maximum(intra_var, eps)
    return _normalize_weights(ratio)


def variance_channel_weights(
    p2p: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Unsupervised: per-channel variance across all spikes as weight.

    High-variance channels carry more information.

    Args:
        p2p: (n_spikes, K) footprint amplitudes

    Returns:
        (K,) weights normalized to sum=1
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    if p2p.shape[0] < 2:
        return np.ones(p2p.shape[1], dtype=np.float64) / p2p.shape[1]
    var = np.var(p2p, axis=0)
    return _normalize_weights(var)


def cluster_profile_weights(
    p2p: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-8,
) -> dict[int, np.ndarray]:
    """Per-unit mean footprint profile as attention weights.

    For each unit, channels where it has high relative amplitude get
    higher weight — emphasizing the channels most characteristic of
    that unit's spatial footprint.

    Args:
        p2p: (n_spikes, K) footprint amplitudes
        labels: (n_spikes,) unit labels

    Returns:
        {unit_id: (K,) weights normalized to sum=1}
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    result: dict[int, np.ndarray] = {}

    for u in np.unique(labels):
        mask = labels == u
        if not np.any(mask):
            continue
        profile = np.mean(p2p[mask], axis=0)
        result[int(u)] = _normalize_weights(profile)

    return result


def inverse_cv_weights(
    p2p: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Inverse coefficient-of-variation weighting (unsupervised variant).

    Channels with low CV (consistent across spikes) are reliable;
    channels with high CV are noisy. Weight = 1 / (1 + CV).

    Returns:
        (K,) weights normalized to sum=1
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    mu = np.mean(p2p, axis=0)
    sd = np.std(p2p, axis=0)
    cv = sd / np.maximum(np.abs(mu), eps)
    w = 1.0 / (1.0 + cv)
    return _normalize_weights(w)


def apply_channel_weights(
    p2p: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply global channel weights to footprint amplitudes.

    Args:
        p2p: (n_spikes, K)
        weights: (K,)

    Returns:
        (n_spikes, K) reweighted p2p
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    return p2p * w[None, :]


def apply_per_cluster_weights(
    p2p: np.ndarray,
    cluster_weights: dict[int, np.ndarray],
    assignments: np.ndarray,
) -> np.ndarray:
    """Apply per-cluster weights: each spike weighted by its assigned cluster's profile.

    Args:
        p2p: (n_spikes, K)
        cluster_weights: {unit_id: (K,) weights}
        assignments: (n_spikes,) predicted unit labels

    Returns:
        (n_spikes, K) reweighted p2p
    """
    p2p = np.asarray(p2p, dtype=np.float64)
    assignments = np.asarray(assignments).ravel()
    out = np.zeros_like(p2p)
    k_dim = p2p.shape[1]

    for u, w in cluster_weights.items():
        mask = assignments == u
        if np.any(mask):
            out[mask] = p2p[mask] * w[None, :]

    # Spikes with unknown assignment: uniform weight
    unknown = ~np.isin(assignments, list(cluster_weights.keys()))
    if np.any(unknown):
        out[unknown] = p2p[unknown] / k_dim

    return out


def softmax_temperature(
    weights: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply temperature scaling to attention weights.

    T < 1: sharper (more selective attention)
    T > 1: softer (more uniform attention)
    T = 1: identity

    Args:
        weights: (K,) non-negative weights (need not sum to 1)
        temperature: softmax temperature

    Returns:
        (K,) softmax-normalized weights
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = np.maximum(w, eps)
    log_w = np.log(w) / max(float(temperature), eps)
    log_w -= np.max(log_w)  # numerical stability
    exp_w = np.exp(log_w)
    return exp_w / np.sum(exp_w)


def _normalize_weights(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize non-negative weights to sum=1."""
    w = np.asarray(w, dtype=np.float64).ravel()
    w = np.maximum(w, 0.0)
    total = np.sum(w)
    if total < eps:
        return np.ones_like(w) / len(w)
    return w / total
