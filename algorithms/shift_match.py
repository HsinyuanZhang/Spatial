"""1D shift-min matching and CiM multi-copy pre-shift row banks.

Hardware story (Option A): store ``2S+1`` pre-shifted copies of each unit
centroid; broadcast the query once; take the min distance over copies.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

DistanceMetric = Literal["l1", "l2"]


def shift_vector(v: np.ndarray, s: int, fill: float = 0.0) -> np.ndarray:
    """Shift content by ``+s`` slots; vacated entries become ``fill``.

    Positive ``s`` moves content toward higher indices (right).
    """
    x = np.asarray(v, dtype=np.float64).ravel()
    out = np.full(x.shape, float(fill), dtype=np.float64)
    s = int(s)
    n = x.size
    if s == 0 or n == 0:
        return x.copy()
    if abs(s) >= n:
        return out
    if s > 0:
        out[s:] = x[: n - s]
    else:
        out[: n + s] = x[-s:]
    return out


def _pairwise_distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """a (n,K), b (m,K) -> (n,m)."""
    if metric == "l1":
        return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    if metric == "l2":
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    raise ValueError(f"Unknown metric: {metric}")


def shift_min_distance(
    query: np.ndarray,
    templates: np.ndarray,
    *,
    shift_radius: int,
    metric: DistanceMetric = "l1",
) -> np.ndarray:
    """Min distance over template shifts ``s ∈ [-S..S]``.

    Args:
        query: (n, K)
        templates: (m, K)
        shift_radius: S >= 0
        metric: ``l1`` or ``l2``

    Returns:
        (n, m) distances

    Notes
    -----
    For ``S > 0`` the shifted bank is built once via
    :func:`build_preshift_bank` and reduced with
    :func:`shift_min_distance_to_bank`, avoiding the per-shift Python loop and
    repeated ``np.stack`` allocations of the original implementation.
    """
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(templates, dtype=np.float64)
    if q.ndim != 2 or t.ndim != 2 or q.shape[1] != t.shape[1]:
        raise ValueError("query and templates must be 2D with matching width")
    S = int(shift_radius)
    if S < 0:
        raise ValueError(f"shift_radius must be >= 0, got {shift_radius}")
    if S == 0:
        return _pairwise_distance(q, t, metric)
    bank = build_preshift_bank(t, shift_radius=S)
    return shift_min_distance_to_bank(q, bank, metric=metric)


def build_preshift_bank(
    centroids: np.ndarray,
    shift_radius: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Pre-shifted CiM row bank.

    Returns:
        (m, 2S+1, K) where index ``S + s`` stores ``shift(centroid, s)``.
    """
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"centroids must be 2D, got {c.shape}")
    S = int(shift_radius)
    if S < 0:
        raise ValueError(f"shift_radius must be >= 0, got {shift_radius}")
    m, k = c.shape
    bank = np.empty((m, 2 * S + 1, k), dtype=np.float64)
    for s_idx, s in enumerate(range(-S, S + 1)):
        for i in range(m):
            bank[i, s_idx] = shift_vector(c[i], s, fill=fill)
    return bank


def shift_min_distance_to_bank(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    metric: DistanceMetric = "l1",
) -> np.ndarray:
    """Min distance of each query to each unit's pre-shifted copies.

    Args:
        query: (n, K)
        bank: (m, 2S+1, K)

    Returns:
        (n, m)
    """
    q = np.asarray(query, dtype=np.float64)
    b = np.asarray(bank, dtype=np.float64)
    if q.ndim != 2 or b.ndim != 3 or q.shape[1] != b.shape[2]:
        raise ValueError("query (n,K) and bank (m,2S+1,K) width must match")
    n, k = q.shape
    m, n_shifts, _ = b.shape
    flat = b.reshape(m * n_shifts, k)
    d_flat = _pairwise_distance(q, flat, metric)  # (n, m*n_shifts)
    return d_flat.reshape(n, m, n_shifts).min(axis=2)


def _pairwise_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,K), b (m,K) -> (n,m) dots."""
    return a @ b.T


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nrm, eps)


def shift_max_dot(
    query: np.ndarray,
    templates: np.ndarray,
    *,
    shift_radius: int,
    normalize: bool = False,
) -> np.ndarray:
    """Max dot-product over template shifts ``s ∈ [-S..S]`` (discrete CONV/corr).

    This is the MAC-CiM friendly score: larger is more similar.

    Args:
        query: (n, K)
        templates: (m, K)
        shift_radius: S >= 0
        normalize: if True, L2-normalize query and each shifted template
            before the dot (cosine / normalized correlation).

    Returns:
        (n, m) scores
    """
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(templates, dtype=np.float64)
    if q.ndim != 2 or t.ndim != 2 or q.shape[1] != t.shape[1]:
        raise ValueError("query and templates must be 2D with matching width")
    S = int(shift_radius)
    if S < 0:
        raise ValueError(f"shift_radius must be >= 0, got {shift_radius}")
    if normalize:
        q = _l2_normalize_rows(q)

    def _score(tt: np.ndarray) -> np.ndarray:
        if normalize:
            tt = _l2_normalize_rows(tt)
        return _pairwise_dot(q, tt)

    if S == 0:
        return _score(t)

    # Build the (m, 2S+1, K) shifted bank once and reduce via the bank path.
    bank = build_preshift_bank(t, shift_radius=S)
    return shift_max_dot_to_bank(q, bank, normalize=normalize)


def shift_max_dot_to_bank(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Max dot of each query against each unit's pre-shifted copies.

    Args:
        query: (n, K)
        bank: (m, 2S+1, K)
        normalize: L2-normalize before dots (cosine).

    Returns:
        (n, m)
    """
    q = np.asarray(query, dtype=np.float64)
    b = np.asarray(bank, dtype=np.float64)
    if q.ndim != 2 or b.ndim != 3 or q.shape[1] != b.shape[2]:
        raise ValueError("query (n,K) and bank (m,2S+1,K) width must match")
    n, k = q.shape
    m, n_shifts, _ = b.shape
    if normalize:
        q = _l2_normalize_rows(q)
        flat = b.reshape(m * n_shifts, k)
        flat = _l2_normalize_rows(flat)
    else:
        flat = b.reshape(m * n_shifts, k)
    dots = _pairwise_dot(q, flat).reshape(n, m, n_shifts)
    return dots.max(axis=2)


def assign_shift_max_dot(
    features: np.ndarray,
    centroids: dict[int, np.ndarray],
    *,
    shift_radius: int = 0,
    normalize: bool = False,
) -> np.ndarray:
    """Nearest unit by max shift-dot (CONV) score."""
    feats = np.asarray(features, dtype=np.float64)
    if len(feats) == 0:
        return np.empty((0,), dtype=np.int64)
    if not centroids:
        return np.full(len(feats), -1, dtype=np.int64)
    units = np.array(sorted(centroids.keys()), dtype=np.int64)
    mat = np.stack([centroids[int(u)] for u in units], axis=0)
    scores = shift_max_dot(
        feats, mat, shift_radius=int(shift_radius), normalize=normalize
    )
    return units[np.argmax(scores, axis=1)]


def cim_row_accounting(
    n_units: int,
    k: int,
    shift_radius: int,
    n_bits: int,
) -> dict[str, float | int]:
    """Multi-copy CiM storage accounting (Option A)."""
    S = int(shift_radius)
    if S < 0:
        raise ValueError(f"shift_radius must be >= 0, got {shift_radius}")
    rows_per_unit = 2 * S + 1
    total_rows = int(n_units) * rows_per_unit
    bits_per_row = int(k) * int(n_bits)
    return {
        "n_units": int(n_units),
        "k": int(k),
        "shift_radius": S,
        "n_bits": int(n_bits),
        "rows_per_unit": rows_per_unit,
        "total_rows": total_rows,
        "bits_per_row": bits_per_row,
        "total_bits": total_rows * bits_per_row,
        "area_multiplier_vs_s0": float(rows_per_unit),
    }


def search_ops_per_event(
    active_units: float,
    k: int,
    shift_radius: int,
) -> float:
    """Logical distance ops ≈ active_units * (2S+1) * K."""
    return float(active_units) * float(2 * int(shift_radius) + 1) * float(k)
