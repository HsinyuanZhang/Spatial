"""Stateful forward-only SOS bandpass preprocessing for streaming experiments.

This module deliberately uses :func:`scipy.signal.sosfilt`, never ``filtfilt``.
It is a causal filtering primitive, not a detector and not a group-delay model:
applications must choose and report their own warm-up/guard policy.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.signal import butter, sosfilt


def _validate_filter_parameters(fs: float, low: float, high: float, order: int) -> tuple[float, float, float, int]:
    """Validate finite Butterworth bandpass parameters in Hz."""
    if not isinstance(fs, Real) or not np.isfinite(fs) or float(fs) <= 0.0:
        raise ValueError("fs must be a finite positive sampling rate")
    if not isinstance(low, Real) or not isinstance(high, Real) or not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("low and high must be finite cutoff frequencies")
    sample_rate, low_hz, high_hz = float(fs), float(low), float(high)
    if not 0.0 < low_hz < high_hz < sample_rate / 2.0:
        raise ValueError("cutoffs must satisfy 0 < low < high < fs / 2")
    if isinstance(order, bool) or not isinstance(order, Integral) or int(order) < 1:
        raise ValueError("order must be a positive integer")
    return sample_rate, low_hz, high_hz, int(order)


def _validate_channel_count(n_channels: int) -> int:
    if isinstance(n_channels, bool) or not isinstance(n_channels, Integral) or int(n_channels) < 1:
        raise ValueError("n_channels must be a positive integer")
    return int(n_channels)


class CausalBandpassFilter:
    """Forward-only, chunk-safe Butterworth SOS bandpass filter.

    Input chunks must have shape ``(n_channels, n_samples)``.  The filter
    stores SOS delay state with shape ``(n_sections, n_channels, 2)``; calling
    :meth:`process` on contiguous chunks is numerically identical to filtering
    their concatenation once.  No fixed group delay is implied or estimated.
    """

    def __init__(
        self,
        fs: float,
        *,
        low: float = 300.0,
        high: float = 6000.0,
        order: int = 3,
        n_channels: int | None = None,
    ) -> None:
        self.fs, self.low, self.high, self.order = _validate_filter_parameters(fs, low, high, order)
        self.sos = butter(self.order, [self.low, self.high], btype="bandpass", fs=self.fs, output="sos")
        if self.sos.ndim != 2 or self.sos.shape[1] != 6 or not np.all(np.isfinite(self.sos)):
            raise RuntimeError("Butterworth SOS construction returned invalid coefficients")
        self._n_channels: int | None = None
        self._zi: np.ndarray | None = None
        if n_channels is not None:
            self.reset(n_channels)

    @property
    def n_channels(self) -> int | None:
        """Bound channel count, or ``None`` until the first nonempty setup."""
        return self._n_channels

    @property
    def state(self) -> np.ndarray | None:
        """Copy of SOS delay state, shaped ``(n_sections, channels, 2)``."""
        return None if self._zi is None else self._zi.copy()

    def reset(self, n_channels: int | None = None) -> None:
        """Reset delay state to zero, optionally binding a channel count."""
        if n_channels is None:
            if self._n_channels is None:
                self._zi = None
                return
            count = self._n_channels
        else:
            count = _validate_channel_count(n_channels)
        self._n_channels = count
        self._zi = np.zeros((self.sos.shape[0], count, 2), dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter one forward chunk and update state without reading future data."""
        values = np.asarray(chunk, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("chunk must have shape (n_channels, n_samples)")
        if values.shape[0] < 1:
            raise ValueError("chunk must contain at least one channel")
        if not np.all(np.isfinite(values)):
            raise ValueError("chunk must contain finite values")
        if self._n_channels is None:
            self.reset(values.shape[0])
        elif values.shape[0] != self._n_channels:
            raise ValueError(f"chunk has {values.shape[0]} channels; expected {self._n_channels}")
        assert self._zi is not None
        if values.shape[1] == 0:
            return values.copy()
        filtered, next_state = sosfilt(self.sos, values, axis=1, zi=self._zi)
        if next_state.shape != self._zi.shape or not np.all(np.isfinite(next_state)):
            raise RuntimeError("SOS filtering produced invalid state")
        self._zi = next_state
        return filtered


def causal_bandpass_filter(
    data: np.ndarray,
    fs: float,
    *,
    low: float = 300.0,
    high: float = 6000.0,
    order: int = 3,
) -> np.ndarray:
    """Convenience single-pass forward causal bandpass filter for a full array."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples)")
    return CausalBandpassFilter(fs, low=low, high=high, order=order, n_channels=values.shape[0]).process(values)


def causal_event_guard_mask(
    event_times: np.ndarray,
    *,
    warmup_samples: int = 0,
    start_sample: int = 0,
    stop_sample: int | None = None,
    post_guard_samples: int = 0,
) -> np.ndarray:
    """Return event eligibility after causal warm-up and optional end guard.

    An event at ``start_sample + warmup_samples`` is eligible.  If a
    ``stop_sample`` is supplied, events must be strictly before
    ``stop_sample - post_guard_samples``.  This helper only marks rows; callers
    retaining chronological splits must keep equal timestamps together.
    """
    for name, value in (("warmup_samples", warmup_samples), ("post_guard_samples", post_guard_samples)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral):
        raise ValueError("start_sample must be an integer")
    if stop_sample is not None and (isinstance(stop_sample, bool) or not isinstance(stop_sample, Integral) or int(stop_sample) < int(start_sample)):
        raise ValueError("stop_sample must be an integer no smaller than start_sample")
    times = np.asarray(event_times)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("event_times must be a one-dimensional integer array")
    mask = times >= int(start_sample) + int(warmup_samples)
    if stop_sample is not None:
        mask &= times < int(stop_sample) - int(post_guard_samples)
    return mask.astype(bool, copy=False)
