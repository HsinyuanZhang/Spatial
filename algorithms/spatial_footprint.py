"""Spatial-only footprint / soft-localization features (no temporal waveform PCA/FSDE).

All features are derived from per-channel peak-to-peak (P2P) amplitudes in a
local neighborhood around each spike's central channel, plus electrode geometry.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


P2PEstimator = Literal[
    "raw",
    "centered_mean",
    "moving_average",
    "extreme_mean",
]


def primary_axis(geom: np.ndarray) -> int:
    """Return 0 or 1: coordinate axis with larger variance."""
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    if g.shape[0] == 0:
        return 1
    return 1 if float(np.var(g[:, 1])) >= float(np.var(g[:, 0])) else 0


def channel_rank_along_axis(geom: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stable rank of channels along ``axis`` (default: primary).

    Returns:
        order: (C,) channel indices sorted by geom[:, axis]
        rank: (C,) rank of each channel in that order
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    ax = primary_axis(g) if axis is None else int(axis)
    if ax not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {ax}")
    order = np.argsort(g[:, ax], kind="stable")
    rank = np.empty(g.shape[0], dtype=np.int64)
    rank[order] = np.arange(g.shape[0], dtype=np.int64)
    return order, rank


def build_relative_patch_table(
    geom: np.ndarray,
    half_width: int = 3,
    axis: int | None = None,
) -> np.ndarray:
    """Per-home relative-offset patch table along one probe axis.

    Slot ``j`` corresponds to rank offset ``j - half_width`` from the home
    channel. Out-of-bounds slots are ``-1``.

    Returns:
        (n_channels, 2*half_width+1) int64 channel ids.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    half = int(half_width)
    if half < 0:
        raise ValueError(f"half_width must be >= 0, got {half_width}")
    n_channels = g.shape[0]
    k = 2 * half + 1
    if n_channels == 0:
        return np.zeros((0, k), dtype=np.int64)

    order, rank = channel_rank_along_axis(g, axis=axis)
    table = np.full((n_channels, k), -1, dtype=np.int64)
    for ch in range(n_channels):
        home_rank = int(rank[ch])
        for j, delta in enumerate(range(-half, half + 1)):
            r = home_rank + delta
            if 0 <= r < n_channels:
                table[ch, j] = int(order[r])
    return table


def is_linear_probe(geom: np.ndarray, ratio: float = 0.25) -> bool:
    """True when sites occupy a single line (one axis has a unique coordinate).

    ``ratio`` is accepted for call compatibility and is not used. A 2- or
    3-column shank is a 2-D layout even if the transverse span is a small
    fraction of the long axis.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2 or g.shape[0] == 0:
        raise ValueError(f"geom must be (C,2), got {getattr(g, 'shape', None)}")
    n_x = int(np.unique(np.round(g[:, 0], 6)).size)
    n_y = int(np.unique(np.round(g[:, 1], 6)).size)
    return min(n_x, n_y) <= 1


def build_physical_stencil_table(
    geom: np.ndarray,
    k_neighbors: int = 7,
    pitch_slack: float = 0.8,
) -> np.ndarray:
    """Per-home table of fixed pitch-unit offsets (2-D stencil).

    Slot ``k // 2`` is always the home electrode. Other slots aim at a frozen
    set of physical offsets; missing sites are ``-1``. This is not event-wise
    KNN order.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    k = int(k_neighbors)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"k_neighbors must be a positive odd integer, got {k_neighbors}")
    n_channels = g.shape[0]
    table = np.full((n_channels, k), -1, dtype=np.int64)
    if n_channels == 0:
        return table
    pitch = float(median_nearest_neighbor_spacing(g)) if n_channels >= 2 else 1.0
    # Seven-slot plus + two diagonals; home at the centre index.
    offsets = np.array(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    if k != 7:
        extra = np.array([[-1.0, 1.0], [1.0, -1.0], [-2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        offsets = np.vstack([offsets, extra])[:k]
        offsets[k // 2] = 0.0
    home_slot = k // 2
    offsets[home_slot] = 0.0
    slack = float(pitch_slack) * max(pitch, 1e-8)
    for ch in range(n_channels):
        used: set[int] = set()
        table[ch, home_slot] = ch
        used.add(ch)
        targets = g[ch] + offsets * pitch
        for j in range(k):
            if j == home_slot:
                continue
            dist = np.linalg.norm(g - targets[j], axis=1)
            order = np.argsort(dist, kind="stable")
            for cand in order:
                ci = int(cand)
                if ci in used:
                    continue
                if float(dist[ci]) <= slack:
                    table[ch, j] = ci
                    used.add(ci)
                break
    return table


def relative_offset_neighbor_table(
    geom: np.ndarray,
    half_width: int = 3,
) -> np.ndarray:
    """Fixed relative-offset slots: 1-D rank along the primary axis, else 2-D stencil."""
    k = 2 * int(half_width) + 1
    if is_linear_probe(geom):
        return build_relative_patch_table(geom, half_width=half_width)
    return build_physical_stencil_table(geom, k_neighbors=k)


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


def median_nearest_neighbor_spacing(geom: np.ndarray) -> float:
    """Median Euclidean distance from each electrode to its nearest other site.

    This is the channel pitch on a 2-D MEA. It is not
    :func:`Spatial.algorithms.geometric_prefilters.median_channel_spacing`,
    which walks consecutive sites after a 1-D sort and underestimates pitch
    on a dense grid.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2 or g.shape[0] < 2:
        raise ValueError(f"geom must be (C,2) with C>=2, got {getattr(g, 'shape', None)}")
    n = g.shape[0]
    nn = np.empty(n, dtype=np.float64)
    for i in range(n):
        dists = np.linalg.norm(g - g[i], axis=1)
        dists[i] = np.inf
        nn[i] = float(np.min(dists))
    return float(np.median(nn))


def build_radius_neighbor_table(geom: np.ndarray, radius_um: float) -> np.ndarray:
    """Per-channel neighbors with Euclidean distance ``<= radius_um``.

    Self is always slot 0 (distance 0). Shorter rows on the array edge are
    padded with ``-1``, which :func:`extract_local_p2p` and
    :func:`com_features` already ignore.

    Returns:
        (n_channels, max_k) int64 neighbor indices.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    radius = float(radius_um)
    if radius < 0.0:
        raise ValueError(f"radius_um must be >= 0, got {radius_um}")
    n_channels = g.shape[0]
    if n_channels == 0:
        return np.zeros((0, 1), dtype=np.int64)

    neighbors: list[np.ndarray] = []
    max_k = 1
    for ch in range(n_channels):
        dists = np.linalg.norm(g - g[ch], axis=1)
        nbr = np.flatnonzero(dists <= radius + 1e-9)
        nbr = nbr[np.argsort(dists[nbr], kind="stable")]
        neighbors.append(nbr)
        if nbr.size > max_k:
            max_k = int(nbr.size)
    table = np.full((n_channels, max_k), -1, dtype=np.int64)
    for ch, nbr in enumerate(neighbors):
        table[ch, : nbr.size] = nbr
    return table


def local_home_channels(
    signal: np.ndarray,
    spike_times: np.ndarray,
    seed_xy: np.ndarray,
    geom: np.ndarray,
    radius_um: float,
    half_window: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Most-negative channel inside a same-spike disk around each seed.

    At event time ``t``, only electrodes within ``radius_um`` of ``seed_xy``
    are eligible. The home is the most-negative site in
    ``[t - half_window, t + half_window]``. Extrema outside the disk are a
    different spike and are ignored. If the disk is empty, snap to the
    nearest electrode (no amplitude look-up on a far site).

    Returns:
        homes: (n_spikes,) int64 channel ids.
        n_in_disk: (n_spikes,) int64 electrodes inside the disk (0 if snapped).
    """
    sig = np.asarray(signal, dtype=np.float64)
    times = np.asarray(spike_times, dtype=np.int64).ravel()
    xy = np.asarray(seed_xy, dtype=np.float64)
    g = np.asarray(geom, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f"signal must be (C, T), got {sig.shape}")
    if g.ndim != 2 or g.shape[1] != 2 or g.shape[0] != sig.shape[0]:
        raise ValueError("geom must be (n_channels, 2) matching signal")
    if xy.ndim != 2 or xy.shape != (times.size, 2):
        raise ValueError(
            f"seed_xy must be (n_spikes, 2), got {xy.shape} for n={times.size}"
        )
    half = int(half_window)
    if half < 0:
        raise ValueError(f"half_window must be >= 0, got {half_window}")
    radius = float(radius_um)
    if radius < 0.0:
        raise ValueError(f"radius_um must be >= 0, got {radius_um}")

    n_channels, n_samples = sig.shape
    n = times.size
    homes = np.empty(n, dtype=np.int64)
    n_in_disk = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists = np.linalg.norm(g - xy[i], axis=1)
        near = dists <= radius + 1e-9
        n_near = int(near.sum())
        n_in_disk[i] = n_near
        if n_near == 0:
            homes[i] = int(np.argmin(dists))
            continue
        t0 = max(int(times[i]) - half, 0)
        t1 = min(int(times[i]) + half + 1, n_samples)
        trough = sig[near, t0:t1].min(axis=1)
        homes[i] = int(np.flatnonzero(near)[int(np.argmin(trough))])
    return homes, n_in_disk


def estimate_p2p_amplitude(
    segment: np.ndarray,
    *,
    estimator: P2PEstimator = "raw",
    average_points: int = 1,
) -> float:
    """Estimate one channel's peak-to-peak amplitude.

    ``raw`` is the existing single-sample ``max(x)-min(x)`` baseline.

    ``centered_mean`` first finds the raw maximum and minimum, then averages a
    contiguous odd-width neighborhood around each selected index. This is the
    direct interpretation of "average 3--5 points around the peaks", but an
    impulsive noise sample can still choose the two neighborhood centers.

    ``moving_average`` applies an odd-width valid boxcar before taking the
    range. A streaming implementation needs a running sum and extrema
    registers; for a fixed width, division by ``average_points`` may be
    postponed or cancelled by later per-event footprint normalization.

    ``extreme_mean`` subtracts the mean of the globally smallest k samples
    from the mean of the globally largest k samples. It is a non-adjacent
    order-statistic diagnostic and is not assumed to be hardware-cheap.
    """
    values = np.asarray(segment, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("segment must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("segment must contain only finite samples")

    mode = str(estimator)
    if mode not in ("raw", "centered_mean", "moving_average", "extreme_mean"):
        raise ValueError(f"Unknown P2P estimator: {estimator}")

    width = int(average_points)
    if isinstance(average_points, bool) or width != average_points or width < 1:
        raise ValueError("average_points must be a positive integer")
    if width > values.size:
        raise ValueError("average_points cannot exceed the segment length")
    if width % 2 == 0:
        raise ValueError("average_points must be odd for a centered P2P estimator")
    if mode == "raw":
        if width != 1:
            raise ValueError("raw P2P requires average_points=1")
        return float(np.max(values) - np.min(values))

    if mode == "moving_average":
        kernel = np.full(width, 1.0 / float(width), dtype=np.float64)
        smoothed = np.convolve(values, kernel, mode="valid")
        return float(np.max(smoothed) - np.min(smoothed))

    if mode == "extreme_mean":
        if 2 * width > values.size:
            raise ValueError(
                "extreme_mean requires at least 2*average_points samples"
            )
        lower = np.partition(values, width - 1)[:width]
        upper = np.partition(values, values.size - width)[-width:]
        return float(np.mean(upper) - np.mean(lower))

    # centered_mean: keep exactly ``width`` contiguous samples even when a raw
    # extremum occurs at a segment boundary.
    half = width // 2

    def _mean_around(index: int) -> float:
        start = min(max(int(index) - half, 0), values.size - width)
        return float(np.mean(values[start : start + width]))

    return _mean_around(int(np.argmax(values))) - _mean_around(int(np.argmin(values)))


def extract_local_p2p(
    signal: np.ndarray,
    spike_times: np.ndarray,
    central_channels: np.ndarray,
    neighbor_table: np.ndarray,
    window: int = 15,
    p2p_estimator: P2PEstimator = "raw",
    average_points: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract local-neighborhood P2P amplitudes around each spike.

    Args:
        signal: (n_channels, n_samples) filtered recording.
        spike_times: (n_spikes,) sample indices.
        central_channels: (n_spikes,) home channel per spike.
        neighbor_table: (n_channels, K) including self.
        window: half-window samples on each side of the spike time.
        p2p_estimator: ``raw``, ``centered_mean``, ``moving_average``, or
            ``extreme_mean``. The default preserves the historical result.
        average_points: odd temporal support. Must be 1 for ``raw``; the new
            pilot registers only 3 and 5 for averaged estimators.

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
        for j, ch_j in enumerate(nbr):
            if int(ch_j) < 0:
                continue
            seg = sig[int(ch_j), t - window : t + window]
            p2p[i, j] = estimate_p2p_amplitude(
                seg,
                estimator=p2p_estimator,
                average_points=average_points,
            )

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


def quantize_signed_adc(
    signal: np.ndarray,
    n_bits: int,
    full_scale: float | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Uniform signed ADC: fixed analog full-scale, then ``n_bits`` two's complement.

    ``full_scale`` is analog ±Vref: both rails map to
    ``±(2**(n_bits-1) - 1)``. The extra two's-complement negative code is
    unused so the range stays symmetric. A scalar scale is one gain for all
    channels; a length-C vector is a per-channel gain. This is not per-event
    AGC and not :func:`quantize_p2p`.

    Returns the reconstructed analog values (code × LSB), same shape as
    ``signal``.
    """
    x = np.asarray(signal, dtype=np.float64)
    if isinstance(n_bits, bool) or int(n_bits) != n_bits:
        raise ValueError(f"n_bits must be an int, got {n_bits!r}")
    bits = int(n_bits)
    if bits < 2 or bits > 24:
        raise ValueError(f"n_bits must be in [2, 24], got {n_bits}")
    qpos = (1 << (bits - 1)) - 1
    fs = np.asarray(full_scale, dtype=np.float64)
    if fs.ndim == 0:
        if not np.isfinite(fs) or float(fs) <= 0.0:
            raise ValueError("full_scale must be a positive finite scalar")
        lsb = max(float(fs) / float(qpos), float(eps))
        codes = np.clip(np.rint(x / lsb), -qpos, qpos)
        return codes * lsb
    if x.ndim == 0:
        raise ValueError("vector full_scale requires an array signal")
    if fs.shape != (x.shape[0],):
        raise ValueError(
            f"vector full_scale must have shape (n_channels,) = ({x.shape[0]},), got {fs.shape}"
        )
    if np.any(~np.isfinite(fs)) or np.any(fs <= 0.0):
        raise ValueError("full_scale per channel must be positive and finite")
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    lsb = np.maximum(fs.reshape(shape) / float(qpos), float(eps))
    codes = np.clip(np.rint(x / lsb), -qpos, qpos)
    return codes * lsb


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


def quantize_com(
    com: np.ndarray,
    n_bits: int | None,
) -> np.ndarray:
    """Axis-wise unsigned integer quantizer of 2-D COM in ``[0, 1]``.

    Each coordinate uses a fixed analytic range ``[0, 1]`` mapped onto
    ``[0, 2**n_bits - 1]``. ``n_bits=None`` is a float passthrough.

    This is not :func:`quantize_p2p`. P2P quantizes a neighborhood amplitude
    vector; COM quantizes the two geometry-normalized centroid coordinates.
    """
    values = np.asarray(com, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"com must have shape (n, 2), got {values.shape}")
    if n_bits is None:
        return values.copy()
    if isinstance(n_bits, bool) or int(n_bits) != n_bits:
        raise ValueError(f"n_bits must be an int or None, got {n_bits!r}")
    bits = int(n_bits)
    if bits < 1 or bits > 62:
        raise ValueError(f"n_bits must be in [1, 62] or None, got {n_bits}")
    max_level = float((1 << bits) - 1)
    scaled = np.clip(values, 0.0, 1.0) * max_level
    return np.clip(np.rint(scaled), 0.0, max_level)


def main_channel_features(
    central_channels: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Normalized central-channel index. Shape (n_spikes, 1)."""
    ch = np.asarray(central_channels, dtype=np.float64).ravel()
    denom = float(max(int(n_channels) - 1, 1))
    return (ch / denom)[:, None]


def main_channel_xy_features(
    central_channels: np.ndarray,
    geom: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Bbox-normalized (x, y) of the home electrode. Shape (n_spikes, 2).

    On a 2-D MEA the raw channel index is not a spatial coordinate. This is
    the electrode-snapped analogue of COM using the same geom bbox as
    :func:`com_features`.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    ch = np.asarray(central_channels, dtype=np.int64).ravel()
    if ch.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if g.shape[0] == 0:
        raise ValueError("geom must contain at least one channel")
    ch = np.clip(ch, 0, g.shape[0] - 1)
    xy = g[ch]
    geom_min = np.min(g, axis=0)
    geom_span = np.max(g, axis=0) - geom_min
    out = np.zeros_like(xy)
    active = geom_span > float(eps)
    out[:, active] = (xy[:, active] - geom_min[active]) / geom_span[active]
    return np.clip(out, 0.0, 1.0)


def normalize_geom_xy(
    xy: np.ndarray,
    geom: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Bbox-normalize arbitrary (x, y) with the same geom frame as COM."""
    g = np.asarray(geom, dtype=np.float64)
    pts = np.asarray(xy, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy must be (n, 2), got {pts.shape}")
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    geom_min = np.min(g, axis=0)
    geom_span = np.max(g, axis=0) - geom_min
    out = np.zeros_like(pts)
    active = geom_span > float(eps)
    out[:, active] = (pts[:, active] - geom_min[active]) / geom_span[active]
    return np.clip(out, 0.0, 1.0)


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
    valid = nbr >= 0
    amp = np.where(valid, amp, 0.0)
    mass = np.sum(amp, axis=1, keepdims=True)
    weights = amp / np.maximum(mass, eps)

    # Safe gather: OOB slots (-1) map to a dummy coordinate then get zero weight.
    safe_nbr = np.where(valid, nbr, 0)
    xy = g[safe_nbr]
    xy = np.where(valid[:, :, None], xy, 0.0)
    centroid = np.sum(weights[:, :, None] * xy, axis=1)

    geom_min = np.min(g, axis=0)
    geom_span = np.max(g, axis=0) - geom_min
    centroid_norm = np.zeros_like(centroid)
    active = geom_span > eps
    centroid_norm[:, active] = (centroid[:, active] - geom_min[active]) / geom_span[active]

    squared_radius = np.sum((xy - centroid[:, None, :]) ** 2, axis=2)
    squared_radius = np.where(valid, squared_radius, 0.0)
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


def extract_local_extrema(
    signal: np.ndarray,
    spike_times: np.ndarray,
    central_channels: np.ndarray,
    neighbor_table: np.ndarray,
    window: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-slot local max/min values and first-occurrence indices."""
    sig = np.asarray(signal, dtype=np.float64)
    times = np.asarray(spike_times, dtype=np.int64).ravel()
    homes = np.asarray(central_channels, dtype=np.int64).ravel()
    nbr_table = np.asarray(neighbor_table, dtype=np.int64)

    if sig.ndim != 2:
        raise ValueError(f"signal must be 2D, got {sig.shape}")
    if times.size != homes.size:
        raise ValueError("spike_times and central_channels length mismatch")
    if nbr_table.ndim != 2:
        raise ValueError(f"neighbor_table must be 2D, got {nbr_table.shape}")

    n_channels, n_samples = sig.shape
    if nbr_table.shape[0] != n_channels:
        raise ValueError("neighbor_table row count must match n_channels")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"window must be an int, got {window}")
    win = int(window)
    if win < 1:
        raise ValueError(f"window must be >= 1 for a nonempty segment, got {window}")

    k = nbr_table.shape[1]
    valid_mask = (times >= win) & (times < n_samples - win)
    times = times[valid_mask]
    homes = homes[valid_mask]
    n = times.size

    vmax = np.zeros((n, k), dtype=np.float64)
    vmin = np.zeros((n, k), dtype=np.float64)
    imax = np.zeros((n, k), dtype=np.int64)
    imin = np.zeros((n, k), dtype=np.int64)
    neighbor_ids = np.zeros((n, k), dtype=np.int64)

    for i, (t, ch) in enumerate(zip(times, homes)):
        ch = int(np.clip(ch, 0, n_channels - 1))
        nbr = nbr_table[ch]
        neighbor_ids[i] = nbr
        valid_j = nbr >= 0
        if not np.any(valid_j):
            continue
        sample_idx = np.arange(int(t) - win, int(t) + win, dtype=np.int64)
        ch_ids = nbr[valid_j].astype(np.int64, copy=False)
        segs = sig[ch_ids[:, None], sample_idx[None, :]]
        vmax[i, valid_j] = np.max(segs, axis=1)
        vmin[i, valid_j] = np.min(segs, axis=1)
        imax[i, valid_j] = np.argmax(segs, axis=1)
        imin[i, valid_j] = np.argmin(segs, axis=1)

    return vmax, vmin, imax, imin, times, neighbor_ids


def _in_bounds_p2p_scale(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row P2P, in-bounds mask, and scale floored at eps."""
    if vmax.shape != vmin.shape or vmax.shape != neighbor_ids.shape:
        raise ValueError("vmax, vmin, and neighbor_ids shape mismatch")
    if vmax.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {vmax.shape}")

    p2p = vmax - vmin
    in_bounds = neighbor_ids >= 0
    masked = np.where(in_bounds, p2p, -np.inf)
    scale = np.maximum(np.max(masked, axis=1, keepdims=True), float(eps))
    return p2p, in_bounds, scale


def slot_liveness(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    amp_gate: float = 0.25,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return per-slot liveness mask from amplitude gate and per-event scale."""
    vmax = np.asarray(vmax, dtype=np.float64)
    vmin = np.asarray(vmin, dtype=np.float64)
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)

    gate = float(amp_gate)
    if not np.isfinite(gate) or gate < 0.0 or gate > 1.0:
        raise ValueError(f"amp_gate must be a finite float in [0.0, 1.0], got {amp_gate}")

    p2p, in_bounds, scale = _in_bounds_p2p_scale(vmax, vmin, neighbor_ids, eps)
    return in_bounds & (p2p >= gate * scale)


def posneg_codes(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    n_bits: int = 5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Quantize positive and negative peak amplitudes into separate code halves."""
    if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)):
        raise ValueError(f"n_bits must be an int in [1, 16], got {n_bits}")
    n_bits = int(n_bits)
    if n_bits < 1 or n_bits > 16:
        raise ValueError(f"n_bits must be an int in [1, 16], got {n_bits}")

    vmax = np.asarray(vmax, dtype=np.float64)
    vmin = np.asarray(vmin, dtype=np.float64)
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)

    _, in_bounds, scale = _in_bounds_p2p_scale(vmax, vmin, neighbor_ids, eps)
    max_level = float((1 << n_bits) - 1)

    pos = np.rint(max_level * np.maximum(vmax, 0.0) / scale)
    neg = np.rint(max_level * np.maximum(-vmin, 0.0) / scale)
    pos = np.clip(pos, 0.0, max_level)
    neg = np.clip(neg, 0.0, max_level)
    pos = np.where(in_bounds, pos, 0.0)
    neg = np.where(in_bounds, neg, 0.0)
    return np.concatenate([pos, neg], axis=1)


def latency_codes(
    imin: np.ndarray,
    live: np.ndarray,
    home_slot: int,
    n_bits: int = 4,
) -> np.ndarray:
    """Cross-channel trough-latency codes with the home slot column removed."""
    if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)):
        raise ValueError(f"n_bits must be an int in [2, 8], got {n_bits}")
    n_bits = int(n_bits)
    if n_bits < 2 or n_bits > 8:
        raise ValueError(f"n_bits must be an int in [2, 8], got {n_bits}")
    if isinstance(home_slot, bool) or not isinstance(home_slot, (int, np.integer)):
        raise ValueError(f"home_slot must be a valid column index, got {home_slot}")

    imin = np.asarray(imin, dtype=np.int64)
    live = np.asarray(live)
    if imin.ndim != 2:
        raise ValueError(f"imin must be 2D, got {imin.shape}")
    if live.shape != imin.shape:
        raise ValueError("imin and live shape mismatch")

    n, k = imin.shape
    home = int(home_slot)
    if home < 0 or home >= k:
        raise ValueError(f"home_slot must be a valid column index, got {home_slot}")

    limit = float((1 << (n_bits - 1)) - 1)
    home_imin = imin[:, home : home + 1].astype(np.float64)
    raw = imin.astype(np.float64) - home_imin
    code = np.clip(raw, -limit, limit) + limit
    code = np.where(live, code, limit)

    keep = [j for j in range(k) if j != home]
    return code[:, keep]


def width_codes(
    imax: np.ndarray,
    imin: np.ndarray,
    home_slot: int,
    n_bits: int = 5,
) -> np.ndarray:
    """Home-channel trough-to-peak width code."""
    if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)):
        raise ValueError(f"n_bits must be an int in [2, 8], got {n_bits}")
    n_bits = int(n_bits)
    if n_bits < 2 or n_bits > 8:
        raise ValueError(f"n_bits must be an int in [2, 8], got {n_bits}")
    if isinstance(home_slot, bool) or not isinstance(home_slot, (int, np.integer)):
        raise ValueError(f"home_slot must be a valid column index, got {home_slot}")

    imax = np.asarray(imax, dtype=np.int64)
    imin = np.asarray(imin, dtype=np.int64)
    if imax.shape != imin.shape:
        raise ValueError("imax and imin shape mismatch")
    if imax.ndim != 2:
        raise ValueError(f"imax must be 2D, got {imax.shape}")

    _, k = imax.shape
    home = int(home_slot)
    if home < 0 or home >= k:
        raise ValueError(f"home_slot must be a valid column index, got {home_slot}")

    off = float((1 << (n_bits - 1)) - 1)
    raw = imax[:, home].astype(np.float64) - imin[:, home].astype(np.float64)
    code = np.clip(raw, -off, off + 1.0) + off
    return code[:, None]
