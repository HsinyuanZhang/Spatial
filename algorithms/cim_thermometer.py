"""Bipolar thermometer CAM encodings for DC-removed waveforms.

Design:
  x → remove DC → (x_+, x_-) → each side: n_cells thermometer bits
  Distance: Hamming over flattened bits (≈ staircased amplitude distance).

This expands *number of CAM cells* instead of relying on high per-cell precision.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ThreshMode = Literal["uniform", "quantile"]


def remove_dc(waveforms: np.ndarray) -> np.ndarray:
    w = np.asarray(waveforms, dtype=np.float64)
    return w - np.mean(w, axis=-1, keepdims=True)


def split_pos_neg(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0), np.maximum(-x, 0.0)


def fit_thermometer_thresholds(
    amplitudes: np.ndarray,
    n_cells: int = 4,
    mode: ThreshMode = "quantile",
    hi_percentile: float = 99.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Fit increasing thresholds on non-negative amplitudes.

    Returns:
        thresholds: (n_cells,) strictly increasing when data has spread.
    """
    n_cells = int(n_cells)
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")
    amp = np.asarray(amplitudes, dtype=np.float64).ravel()
    amp = amp[np.isfinite(amp) & (amp > eps)]
    if amp.size == 0:
        return np.linspace(1.0 / n_cells, 1.0, n_cells)

    if mode == "uniform":
        hi = float(np.percentile(amp, hi_percentile))
        hi = max(hi, eps)
        return np.linspace(hi / n_cells, hi, n_cells)

    if mode == "quantile":
        # Interior quantiles so the top cell is not stuck at max
        qs = np.linspace(100.0 / (n_cells + 1), 100.0 * n_cells / (n_cells + 1), n_cells)
        thr = np.percentile(amp, qs)
        # Enforce non-decreasing
        thr = np.maximum.accumulate(thr)
        thr = np.maximum(thr, eps)
        return thr.astype(np.float64)

    raise ValueError(f"Unknown threshold mode: {mode}")


def thermometer_encode(
    amp: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Encode non-negative amplitudes to thermometer bits.

    Args:
        amp: (...,) >= 0
        thresholds: (n_cells,)

    Returns:
        bits: (..., n_cells) uint8 in {0,1}
    """
    a = np.asarray(amp, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64).ravel()
    return (a[..., None] >= t).astype(np.uint8)


def bipolar_thermometer_encode(
    waveforms: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """DC-remove → pos/neg thermometer → flattened bit code.

    Returns:
        codes: (n_spikes, T * 2 * n_cells) uint8
    """
    x = remove_dc(waveforms)
    xp, xn = split_pos_neg(x)
    cp = thermometer_encode(xp, thresholds)
    cn = thermometer_encode(xn, thresholds)
    # (n, T, n_cells) each → concat on last axis → flatten
    packed = np.concatenate([cp, cn], axis=-1)
    n = packed.shape[0]
    return packed.reshape(n, -1)


def fit_bipolar_thermometer(
    waveforms_train: np.ndarray,
    n_cells: int = 4,
    mode: ThreshMode = "quantile",
) -> np.ndarray:
    """Fit thresholds on train |DC-removed| samples (pos and neg pooled)."""
    x = remove_dc(waveforms_train)
    xp, xn = split_pos_neg(x)
    pooled = np.concatenate([xp.ravel(), xn.ravel()])
    return fit_thermometer_thresholds(pooled, n_cells=n_cells, mode=mode)


def signed_uniform_quantize(
    waveforms: np.ndarray,
    n_bits: int = 2,
    scale: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Flat signed uniform quantize after DC removal.

    Maps to integer levels in ``[-(L), +(L)]`` excluding a doubled zero when
    ``L = 2**(n_bits-1) - 1`` for n_bits>=2 → e.g. 2-bit → levels {-1,0,1}
    actually use ``n_levels = 2**n_bits`` symmetric bins over [-hi, hi].

    Returns:
        codes: (n, T) int16
    """
    x = remove_dc(waveforms)
    if scale is None:
        scale = float(np.percentile(np.abs(x), 99.0))
    scale = max(float(scale), eps)
    # 2**n_bits levels from -1..+1 then round
    max_level = float((1 << (n_bits - 1)) - 1) if n_bits >= 2 else 1.0
    if n_bits == 2:
        # {-1, 0, +1} style ternary-ish flat code (classic coarse signed)
        max_level = 1.0
    y = np.clip(x / scale, -1.0, 1.0) * max_level
    return np.rint(y).astype(np.int16)


def abs_uniform_quantize(
    waveforms: np.ndarray,
    n_bits: int = 4,
    scale: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Unsigned |DC-removed| uniform quantize to ``[0, 2^n-1]``."""
    x = np.abs(remove_dc(waveforms))
    if scale is None:
        scale = float(np.percentile(x, 99.0))
    scale = max(float(scale), eps)
    max_level = float((1 << n_bits) - 1)
    y = np.clip(x / scale, 0.0, 1.0) * max_level
    return np.rint(y).astype(np.int16)


def hamming_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming between rows of a (n,d) and a single row / matrix b.

    If b is (d,) → (n,); if b is (m,d) → (n,m).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if b.ndim == 1:
        return np.sum(a != b[None, :], axis=1).astype(np.int64)
    return np.sum(a[:, None, :] != b[None, :, :], axis=2).astype(np.int64)


def l1_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        return np.sum(np.abs(a - b[None, :]), axis=1)
    return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
