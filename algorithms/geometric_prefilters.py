"""Geometric / pattern / cosine prefilters for Level-1 candidate routing.

Descriptor and quantizer stay external (D=9, B=5).  This module only builds
anchors, active-row masks, tile directories, Jaccard filters, and integer cosine
scores suitable for RRAM-style MAC reinterpretation of the same centroids.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ENERGY_SRAM_PJ = 100.0
ENERGY_CIM_MAC_PJ = 2.0


def proxy_max_channel(peak_amplitudes: np.ndarray) -> np.ndarray:
    """A1: argmax |peak|; ties → smallest channel index (NumPy default)."""
    amps = np.asarray(peak_amplitudes, dtype=np.float64)
    if amps.ndim != 2:
        raise ValueError("peak_amplitudes must have shape (n_events, n_channels)")
    return np.argmax(np.abs(amps), axis=1).astype(np.int64)


def fit_channel_mad_thresholds(
    filtered: np.ndarray,
    *,
    sample_start: int,
    sample_stop: int,
    mad_multiplier: float = 4.0,
    min_abs: float = 1e-6,
) -> np.ndarray:
    """Per-channel absolute threshold from fit-interval MAD (no test leakage)."""
    data = np.asarray(filtered, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("filtered must be (n_channels, n_samples)")
    start = max(int(sample_start), 0)
    stop = min(int(sample_stop), data.shape[1])
    if stop <= start:
        raise ValueError("fit sample interval is empty")
    segment = data[:, start:stop]
    med = np.median(segment, axis=1)
    mad = np.median(np.abs(segment - med[:, None]), axis=1)
    # Consistent with Gaussian MAD scaling; floor avoids zero thresholds.
    sigma = 1.4826 * mad
    thr = np.maximum(float(mad_multiplier) * sigma, float(min_abs))
    return thr.astype(np.float64)


def causal_first_channel(
    snippets: np.ndarray,
    thresholds: np.ndarray,
    *,
    fallback_channels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy A2: earliest *above-threshold* sample in the full snippet.

    This is **not** hardware first-detect: it fires on any sample already hot at
    the left edge of a GT-aligned window. Prefer
    :func:`rising_edge_first_channel` or :func:`match_stream_first_detect`.
    """
    windows = np.asarray(snippets, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    if windows.ndim != 3:
        raise ValueError("snippets must have shape (n_events, n_channels, n_samples)")
    n_events, n_channels, n_samples = windows.shape
    if thr.shape != (n_channels,):
        raise ValueError("thresholds must match channel count")
    over = np.abs(windows) > thr[None, :, None]
    # Flatten channel-major within each time: scan t, then ch (smallest index wins ties).
    flat = over.transpose(0, 2, 1).reshape(n_events, n_samples * n_channels)
    any_hit = flat.any(axis=1)
    first_flat = np.argmax(flat, axis=1)
    ch_idx = (first_flat % n_channels).astype(np.int64)
    anchors = ch_idx.copy()
    used_fallback = ~any_hit
    if fallback_channels is not None:
        fb = np.asarray(fallback_channels, dtype=np.int64).ravel()
        if fb.shape != (n_events,):
            raise ValueError("fallback_channels must match n_events")
        anchors = np.where(used_fallback, fb, anchors)
    elif np.any(used_fallback):
        peak = np.max(np.abs(windows), axis=2)
        anchors = np.where(used_fallback, np.argmax(peak, axis=1), anchors)
    return anchors.astype(np.int64), used_fallback.astype(bool)


def rising_edge_first_channel(
    snippets: np.ndarray,
    thresholds: np.ndarray,
    *,
    pre_samples: int,
    fallback_channels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First below→above threshold crossing in ``[t-pre, t]`` (peak-relative).

    Snippet layout must be ``[t-pre_full, t+post)`` with ``pre_samples`` equal to
    the pre-peak length used when extracting snippets (index ``pre_samples`` is
    the aligned peak sample). Only the inclusive range
    ``[0, pre_samples]`` is searched, so post-peak energy cannot win.

    Returns ``(anchors, used_fallback, rel_sample)`` where ``rel_sample`` is the
    hit time relative to the peak (negative or zero), or ``0`` on fallback.
    """
    windows = np.asarray(snippets, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    if windows.ndim != 3:
        raise ValueError("snippets must have shape (n_events, n_channels, n_samples)")
    n_events, n_channels, n_samples = windows.shape
    pre = int(pre_samples)
    if not 1 <= pre < n_samples:
        raise ValueError("pre_samples must lie inside the snippet")
    if thr.shape != (n_channels,):
        raise ValueError("thresholds must match channel count")

    # Search samples 1..pre inclusive (need a previous sample for rising edge).
    seg = windows[:, :, : pre + 1]
    abs_seg = np.abs(seg)
    over = abs_seg > thr[None, :, None]
    prev = abs_seg[:, :, :-1]
    curr = abs_seg[:, :, 1:]
    rising = (~(prev > thr[None, :, None])) & (curr > thr[None, :, None])
    # rising shape: (n_events, n_channels, pre) corresponding to absolute snippet
    # indices 1..pre → rel_to_peak = idx - pre.
    flat = rising.transpose(0, 2, 1).reshape(n_events, pre * n_channels)
    any_hit = flat.any(axis=1)
    first_flat = np.argmax(flat, axis=1)
    # time index within rising array (0 => snippet sample 1)
    rise_t = first_flat // n_channels
    ch_idx = (first_flat % n_channels).astype(np.int64)
    snippet_t = rise_t + 1
    rel = snippet_t - pre
    anchors = ch_idx.copy()
    used_fallback = ~any_hit
    rel_out = rel.astype(np.int64)
    if fallback_channels is not None:
        fb = np.asarray(fallback_channels, dtype=np.int64).ravel()
        if fb.shape != (n_events,):
            raise ValueError("fallback_channels must match n_events")
        anchors = np.where(used_fallback, fb, anchors)
        rel_out = np.where(used_fallback, 0, rel_out)
    else:
        peak = np.max(np.abs(windows[:, :, : pre + 1]), axis=2)
        anchors = np.where(used_fallback, np.argmax(peak, axis=1), anchors)
        rel_out = np.where(used_fallback, 0, rel_out)
    return anchors.astype(np.int64), used_fallback.astype(bool), rel_out


def stream_rising_edge_detections(
    filtered: np.ndarray,
    thresholds: np.ndarray,
    *,
    refractory_samples: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Hardware-style stream detections: global earliest rising edge per event.

    At each sample ``t>=1``, channels are scanned in ascending index order; the
    first channel with a below→above crossing emits a detection, then a global
    refractory of ``refractory_samples`` suppresses further emits.

    Returns ``(times, channels)`` sorted by time.
    """
    data = np.asarray(filtered, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    if data.ndim != 2:
        raise ValueError("filtered must be (n_channels, n_samples)")
    n_ch, n_samp = data.shape
    if thr.shape != (n_ch,):
        raise ValueError("thresholds must match channel count")
    refr = max(int(refractory_samples), 1)
    abs_data = np.abs(data)
    over = abs_data > thr[:, None]
    rising = (~over[:, :-1]) & over[:, 1:]  # (n_ch, n_samp-1) → sample index t=1..
    ch_idx, t_rel = np.where(rising)
    if t_rel.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # Stable sort by time, then channel (np.where already channel-major within a
    # column, but where returns in C order: all t for ch0, then ch1, ... — so
    # re-sort by (t, ch).
    order = np.lexsort((ch_idx, t_rel))
    t_sorted = t_rel[order] + 1
    ch_sorted = ch_idx[order]
    keep_t: list[int] = []
    keep_ch: list[int] = []
    next_allowed = -10**18
    for t, ch in zip(t_sorted.tolist(), ch_sorted.tolist()):
        if t < next_allowed:
            continue
        keep_t.append(t)
        keep_ch.append(ch)
        next_allowed = t + refr
    return np.asarray(keep_t, dtype=np.int64), np.asarray(keep_ch, dtype=np.int64)


def match_stream_first_detect(
    peak_times: np.ndarray,
    det_times: np.ndarray,
    det_channels: np.ndarray,
    *,
    pre_samples: int = 15,
    post_samples: int = 5,
    fallback_channels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each GT peak to the earliest stream detection in ``[t-pre, t+post]``.

    This is the hardware-matched first-detect: the detector fires on a rising
    edge, and we attribute the earliest fire in the peak's local window to that
    GT event. Returns ``(anchors, used_fallback, det_rel_to_peak)``.
    """
    peaks = np.asarray(peak_times, dtype=np.int64).ravel()
    dts = np.asarray(det_times, dtype=np.int64).ravel()
    dch = np.asarray(det_channels, dtype=np.int64).ravel()
    if dts.size != dch.size:
        raise ValueError("det_times and det_channels must align")
    pre, post = int(pre_samples), int(post_samples)
    n = peaks.size
    anchors = np.full(n, -1, dtype=np.int64)
    rel = np.zeros(n, dtype=np.int64)
    used_fallback = np.ones(n, dtype=bool)
    if dts.size:
        for i, t in enumerate(peaks):
            lo, hi = int(t) - pre, int(t) + post
            left = int(np.searchsorted(dts, lo, side="left"))
            right = int(np.searchsorted(dts, hi, side="right"))
            if right > left:
                # Earliest detection in window.
                j = left
                anchors[i] = int(dch[j])
                rel[i] = int(dts[j] - t)
                used_fallback[i] = False
    if fallback_channels is not None:
        fb = np.asarray(fallback_channels, dtype=np.int64).ravel()
        if fb.shape != (n,):
            raise ValueError("fallback_channels must match peak_times")
        anchors = np.where(used_fallback, fb, anchors)
        rel = np.where(used_fallback, 0, rel)
    elif np.any(used_fallback):
        raise ValueError("fallback_channels required when some peaks have no detection")
    return anchors, used_fallback, rel


def trigger_channel_masks(
    snippets: np.ndarray,
    thresholds: np.ndarray,
    *,
    peak_amplitudes: np.ndarray | None = None,
    top_k_fallback: int = 3,
) -> np.ndarray:
    """Boolean ``(n_events, n_channels)``: any-sample over threshold, else top-|peak|."""
    windows = np.asarray(snippets, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    n_events, n_channels, _ = windows.shape
    masks = (np.abs(windows) > thr[None, :, None]).any(axis=2)
    empty = ~masks.any(axis=1)
    if np.any(empty):
        if peak_amplitudes is not None:
            peaks = np.abs(np.asarray(peak_amplitudes, dtype=np.float64))
        else:
            peaks = np.max(np.abs(windows), axis=2)
        k = min(int(top_k_fallback), n_channels)
        for i in np.flatnonzero(empty):
            order = np.argsort(-peaks[i], kind="stable")
            masks[i, order[:k]] = True
    return masks


def index_neighborhood_unit_mask(
    home_channels: np.ndarray,
    anchors: np.ndarray,
    n_channels: int,
    radius: int,
    *,
    ensure_nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Activate units whose fit home lies in electrode-index ``[a±R]``.

    Returns ``(n_events, n_units)`` mask and the effective radius used per event
    after optional widening.
    """
    homes = np.asarray(home_channels, dtype=np.int64).ravel()
    anch = np.asarray(anchors, dtype=np.int64).ravel()
    c = int(n_channels)
    if c < 1:
        raise ValueError("n_channels must be positive")
    if np.any((homes < 0) | (homes >= c)) or np.any((anch < 0) | (anch >= c)):
        raise ValueError("channel index out of range")
    n_events = anch.size
    n_units = homes.size
    base_r = max(int(radius), 0)
    masks = np.zeros((n_events, n_units), dtype=bool)
    effective = np.full(n_events, base_r, dtype=np.int64)
    for i, a in enumerate(anch):
        r = base_r
        while True:
            lo = max(0, int(a) - r)
            hi = min(c - 1, int(a) + r)
            active = (homes >= lo) & (homes <= hi)
            if active.any() or not ensure_nonempty or r >= c:
                masks[i] = active if active.any() else np.ones(n_units, dtype=bool)
                effective[i] = r
                break
            r += 1
    return masks, effective


def median_channel_spacing(geom: np.ndarray) -> float:
    """Median adjacent Euclidean spacing along the probe."""
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 2:
        raise ValueError("geom must have at least two channels")
    # Order by primary axis (y if more variance, else x).
    axis = 1 if float(np.var(g[:, 1])) >= float(np.var(g[:, 0])) else 0
    order = np.argsort(g[:, axis], kind="stable")
    diffs = np.linalg.norm(np.diff(g[order], axis=0), axis=1)
    return float(np.median(diffs))


def tile_ids_from_positions(positions: np.ndarray, tile_pitch: float) -> np.ndarray:
    """1D tile index from probe-axis coordinate."""
    pos = np.asarray(positions, dtype=np.float64)
    pitch = float(tile_pitch)
    if pitch <= 0:
        raise ValueError("tile_pitch must be positive")
    if pos.ndim == 1:
        axis_vals = pos
    elif pos.ndim == 2:
        axis = 1 if float(np.var(pos[:, 1])) >= float(np.var(pos[:, 0])) else 0
        axis_vals = pos[:, axis]
    else:
        raise ValueError("positions must be 1D or (n, 2)")
    return np.floor(axis_vals / pitch).astype(np.int64)


def fit_unit_home_tiles(
    labels: np.ndarray,
    event_com_xy: np.ndarray,
    units: np.ndarray,
    tile_pitch: float,
) -> np.ndarray:
    """Per-unit home tile = tile of mean COM over fit events."""
    labs = np.asarray(labels)
    com = np.asarray(event_com_xy, dtype=np.float64)
    row_units = np.asarray(units)
    homes = np.empty(row_units.size, dtype=np.int64)
    for i, unit in enumerate(row_units):
        pts = com[labs == unit]
        if pts.size == 0:
            raise ValueError(f"unit {unit!r} has no fit events")
        mean_xy = pts.mean(axis=0)
        homes[i] = int(tile_ids_from_positions(mean_xy[None, :], tile_pitch)[0])
    return homes


def tile_activation_mask(
    query_tiles: np.ndarray,
    unit_home_tiles: np.ndarray,
    tile_radius: int,
) -> np.ndarray:
    """Activate units whose home tile is within ``tile_radius`` of the query tile."""
    q = np.asarray(query_tiles, dtype=np.int64).ravel()
    homes = np.asarray(unit_home_tiles, dtype=np.int64).ravel()
    r = max(int(tile_radius), 0)
    return (np.abs(homes[None, :] - q[:, None]) <= r)


def jaccard_scores(event_masks: np.ndarray, unit_support: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard between event channel masks and unit support rows."""
    ev = np.asarray(event_masks, dtype=bool)
    sup = np.asarray(unit_support, dtype=bool)
    if ev.ndim != 2 or sup.ndim != 2 or ev.shape[1] != sup.shape[1]:
        raise ValueError("event_masks and unit_support channel dims must match")
    # Broadcast: (n_events, n_units, n_channels)
    inter = np.logical_and(ev[:, None, :], sup[None, :, :]).sum(axis=2).astype(np.float64)
    union = np.logical_or(ev[:, None, :], sup[None, :, :]).sum(axis=2).astype(np.float64)
    scores = np.zeros_like(inter)
    np.divide(inter, union, out=scores, where=union > 0)
    return scores


def jaccard_filter_mask(
    event_masks: np.ndarray, unit_support: np.ndarray, threshold: float
) -> np.ndarray:
    """Keep units with Jaccard ≥ threshold; empty rows stay empty (caller may widen)."""
    return jaccard_scores(event_masks, unit_support) >= float(threshold)


def integer_cosine_scores(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine similarity in integer code space: dot / (|q||c|).

    Uses float64 only for the final normalization; dots are exact int64 products
    for 5-bit × 9-D codes.
    """
    q = np.asarray(codes, dtype=np.int64)
    c = np.asarray(centroids, dtype=np.int64)
    if q.ndim != 2 or c.ndim != 2 or q.shape[1] != c.shape[1]:
        raise ValueError("codes and centroids must be 2D with matching width")
    dots = q.astype(np.float64) @ c.astype(np.float64).T
    q_norm = np.linalg.norm(q.astype(np.float64), axis=1, keepdims=True)
    c_norm = np.linalg.norm(c.astype(np.float64), axis=1, keepdims=True).T
    denom = q_norm * c_norm
    out = np.zeros_like(dots)
    np.divide(dots, denom, out=out, where=denom > 0)
    return out


def cosine_distance(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """``1 - cosine`` in ``[0, 2]``."""
    return 1.0 - integer_cosine_scores(codes, centroids)


def fit_cosine_radii(
    codes: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    units: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Per-unit cosine-distance radius from calibration/fit codes."""
    q = np.asarray(codes, dtype=np.int64)
    labs = np.asarray(labels)
    cents = np.asarray(centroids, dtype=np.int64)
    row_units = np.asarray(units)
    radii = np.zeros(row_units.size, dtype=np.float64)
    for i, unit in enumerate(row_units):
        local = q[labs == unit]
        if local.size == 0:
            radii[i] = 2.0
            continue
        d = cosine_distance(local, cents[i : i + 1])[:, 0]
        radii[i] = float(np.percentile(d, float(percentile), method="linear"))
    return radii


def query_cosine_radius(
    codes: np.ndarray,
    centroids: np.ndarray,
    units: np.ndarray,
    radii: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
    ensure_nonempty: bool = True,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Candidate lists where cosine distance ≤ per-unit radius inside active rows."""
    q = np.asarray(codes, dtype=np.int64)
    cents = np.asarray(centroids, dtype=np.int64)
    row_units = np.asarray(units)
    r = np.asarray(radii, dtype=np.float64).ravel()
    n_q, n_u = q.shape[0], row_units.size
    if active_mask is None:
        mask = np.ones((n_q, n_u), dtype=bool)
    else:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape == (n_u,):
            mask = np.broadcast_to(mask, (n_q, n_u)).copy()
        if mask.shape != (n_q, n_u):
            raise ValueError("active_mask shape mismatch")
    dist = cosine_distance(q, cents)
    hits = (dist <= r[None, :]) & mask
    candidates: list[np.ndarray] = []
    fallback = np.zeros(n_q, dtype=bool)
    active_count = mask.sum(axis=1).astype(np.int64)
    row_comparisons = active_count.copy()
    for i in range(n_q):
        idx = np.flatnonzero(hits[i])
        if idx.size == 0 and ensure_nonempty and mask[i].any():
            # Nearest active by cosine distance.
            masked = np.where(mask[i], dist[i], np.inf)
            idx = np.asarray([int(np.argmin(masked))], dtype=np.int64)
            fallback[i] = True
        candidates.append(row_units[idx].copy())
    diag = {
        "widen_level": np.zeros(n_q, dtype=np.int64),
        "fallback": fallback,
        "active_row_count": active_count,
        "row_comparisons": row_comparisons,
        "fallback_row_comparisons": np.where(fallback, active_count, 0).astype(np.int64),
        "total_row_comparisons_including_fallback": row_comparisons
        + np.where(fallback, active_count, 0).astype(np.int64),
        "cosine_distance_to_winner": np.min(
            np.where(mask, dist, np.inf), axis=1
        ).astype(np.float64),
    }
    return candidates, diag


def rank_fusion_topk(
    l1_distances: np.ndarray,
    cosine_distances: np.ndarray,
    active_mask: np.ndarray,
    units: np.ndarray,
    top_k: int,
    *,
    cosine_weight: float = 0.7,
    l1_weight: float = 0.3,
) -> list[np.ndarray]:
    """Hybrid rank fusion on active rows; return top-K unit lists."""
    l1 = np.asarray(l1_distances, dtype=np.float64)
    cos = np.asarray(cosine_distances, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool)
    row_units = np.asarray(units)
    n_q = l1.shape[0]
    out: list[np.ndarray] = []
    cw, lw = float(cosine_weight), float(l1_weight)

    def _ranks(row: np.ndarray, m: np.ndarray) -> np.ndarray:
        ranks = np.full(row.size, row.size, dtype=np.float64)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return ranks
        order = idx[np.argsort(row[idx], kind="stable")]
        ranks[order] = np.arange(order.size, dtype=np.float64)
        return ranks

    for i in range(n_q):
        fused = cw * _ranks(cos[i], mask[i]) + lw * _ranks(l1[i], mask[i])
        fused = np.where(mask[i], fused, np.inf)
        order = np.argsort(fused, kind="stable")
        keep = [j for j in order if np.isfinite(fused[j])][: max(int(top_k), 1)]
        out.append(row_units[np.asarray(keep, dtype=np.int64)].copy())
    return out


def prune_candidates_topk(
    candidates: list[np.ndarray],
    codes: np.ndarray,
    centroids: np.ndarray,
    units: np.ndarray,
    top_k: int,
    *,
    metric: str = "l1",
) -> list[np.ndarray]:
    """Keep at most top-K candidates per event by L1 or cosine distance."""
    q = np.asarray(codes, dtype=np.int64)
    cents = np.asarray(centroids, dtype=np.int64)
    row_units = np.asarray(units)
    index = { (u.item() if isinstance(u, np.generic) else u): i for i, u in enumerate(row_units) }
    if metric == "l1":
        dist = np.abs(q[:, None, :] - cents[None, :, :]).sum(axis=2).astype(np.float64)
    elif metric == "cosine":
        dist = cosine_distance(q, cents)
    else:
        raise ValueError("metric must be 'l1' or 'cosine'")
    k = max(int(top_k), 1)
    out: list[np.ndarray] = []
    for i, cand in enumerate(candidates):
        rows = []
        for u in np.asarray(cand):
            key = u.item() if isinstance(u, np.generic) else u
            if key in index:
                rows.append(index[key])
        if not rows:
            out.append(np.asarray(cand)[:0].copy())
            continue
        rows_a = np.asarray(rows, dtype=np.int64)
        order = rows_a[np.argsort(dist[i, rows_a], kind="stable")]
        out.append(row_units[order[:k]].copy())
    return out


def extract_event_snippets(
    filtered: np.ndarray,
    times: np.ndarray,
    *,
    pre: int = 15,
    post: int = 49,
) -> np.ndarray:
    """Stack ``(n_events, n_channels, pre+post)`` windows ``[t-pre, t+post)``."""
    data = np.asarray(filtered, dtype=np.float64)
    t = np.asarray(times, dtype=np.int64).ravel()
    n_ch, n_samp = data.shape
    width = int(pre) + int(post)
    out = np.empty((t.size, n_ch, width), dtype=np.float64)
    for i, time in enumerate(t):
        start = int(time) - int(pre)
        stop = int(time) + int(post)
        if start < 0 or stop > n_samp:
            raise ValueError(f"snippet for t={time} out of bounds")
        out[i] = data[:, start:stop]
    return out


def energy_estimate(
    *,
    n_events: int,
    n_sram_lookups_per_event: float,
    n_cim_macs_per_event: float,
) -> dict[str, float]:
    """Symbolic energy model fixed in the plan doc."""
    sram = float(n_events) * float(n_sram_lookups_per_event) * ENERGY_SRAM_PJ
    cim = float(n_events) * float(n_cim_macs_per_event) * ENERGY_CIM_MAC_PJ
    return {
        "sram_energy_pj": sram,
        "cim_energy_pj": cim,
        "total_energy_pj": sram + cim,
        "energy_per_event_pj": (sram + cim) / max(float(n_events), 1.0),
        "sram_pj_per_lookup": ENERGY_SRAM_PJ,
        "cim_pj_per_mac": ENERGY_CIM_MAC_PJ,
    }


def cycles_estimate(
    *,
    n_active_rows_mean: float,
    n_dims: int = 9,
    score: str = "l1",
) -> dict[str, float]:
    """Rough cycle model: one abs/add or MAC per dimension per active row."""
    rows = float(n_active_rows_mean)
    d = int(n_dims)
    if score == "l1":
        # abs + (d-1) adds ≈ 2d-1 ops; count as d "MAC-equivalent" for apples-to-apples.
        ops = rows * d
    elif score == "cosine":
        ops = rows * d  # MAC per dim
    else:
        raise ValueError("score must be l1 or cosine")
    return {
        "mean_active_rows": rows,
        "ops_per_event": ops,
        "score": score,
        "n_dims": d,
    }
