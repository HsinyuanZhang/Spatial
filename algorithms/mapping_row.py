"""Mapping-row v0: interval / weighted-violation scores on spatial shape.

The row stores a shape centre and a per-dimension interval (or an equivalent
diagonal precision readout).  The event score ``S_u`` is a saturated integer
penalty sum.  Scale ``g`` and quality flags never enter ``S_u``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SHAPE_POSNEG_BITS = 4
SHAPE_POSNEG_MAX = (1 << SHAPE_POSNEG_BITS) - 1  # 15
COM_BITS = 5
COM_MAX = (1 << COM_BITS) - 1
G_BITS = 4
G_MAX = (1 << G_BITS) - 1
PENALTY_CAP = 3
P_MIN = 0
P_MAX = 3
QUIET_CODE = 0
FLOOR_MU = 1


def analog_gain(vmax: np.ndarray, vmin: np.ndarray, neighbor_ids: np.ndarray) -> np.ndarray:
    """Per-event local gain: max absolute amplitude on in-bounds slots."""
    pos = np.maximum(np.asarray(vmax, dtype=np.float64), 0.0)
    neg = np.maximum(-np.asarray(vmin, dtype=np.float64), 0.0)
    valid = np.asarray(neighbor_ids, dtype=np.int64) >= 0
    peak = np.maximum(pos, neg)
    peak = np.where(valid, peak, 0.0)
    return np.max(peak, axis=1)


def normalize_posneg_shape(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    analog_g: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """4-bit POSNEG codes: ``round(15 * a / max(g, eps))``, in-bounds only."""
    pos = np.maximum(np.asarray(vmax, dtype=np.float64), 0.0)
    neg = np.maximum(-np.asarray(vmin, dtype=np.float64), 0.0)
    valid = np.asarray(neighbor_ids, dtype=np.int64) >= 0
    g = np.maximum(np.asarray(analog_g, dtype=np.float64).reshape(-1, 1), float(eps))
    pos_c = np.clip(np.rint(SHAPE_POSNEG_MAX * pos / g), 0, SHAPE_POSNEG_MAX)
    neg_c = np.clip(np.rint(SHAPE_POSNEG_MAX * neg / g), 0, SHAPE_POSNEG_MAX)
    pos_c = np.where(valid, pos_c, 0.0)
    neg_c = np.where(valid, neg_c, 0.0)
    return pos_c.astype(np.int64), neg_c.astype(np.int64)


def quantize_gain(analog_g: np.ndarray, full_scale: float) -> np.ndarray:
    """Unsigned 4-bit gain codes against a fit-only full scale."""
    fs = max(float(full_scale), 1e-12)
    codes = np.clip(np.rint(G_MAX * np.asarray(analog_g, dtype=np.float64) / fs), 0, G_MAX)
    return codes.astype(np.int64)


def quality_flags(
    vmax: np.ndarray,
    vmin: np.ndarray,
    imax: np.ndarray,
    imin: np.ndarray,
    neighbor_ids: np.ndarray,
    analog_g: np.ndarray,
    *,
    window: int,
    home_slot: int,
    sat_abs: float,
    overlap_live: int = 6,
    amp_gate: float = 0.25,
) -> np.ndarray:
    """Return (n, 3) uint8 flags ``[sat, cut, overlap]``. Never used in ``S_u``."""
    n = int(np.asarray(vmax).shape[0])
    sat = analog_g >= (float(sat_abs) * (1.0 - 1e-12))
    edge = (2 * int(window)) - 1
    hs = int(home_slot)
    cut = (imax[:, hs] <= 0) | (imax[:, hs] >= edge) | (imin[:, hs] <= 0) | (
        imin[:, hs] >= edge
    )
    p2p = np.maximum(vmax, 0.0) - np.minimum(vmin, 0.0)
    valid = neighbor_ids >= 0
    live = valid & (p2p >= float(amp_gate) * analog_g[:, None])
    overlap = live.sum(axis=1) >= int(overlap_live)
    out = np.zeros((n, 3), dtype=np.uint8)
    out[:, 0] = sat.astype(np.uint8)
    out[:, 1] = cut.astype(np.uint8)
    out[:, 2] = overlap.astype(np.uint8)
    return out


def precision_from_width(width: np.ndarray) -> np.ndarray:
    """``p = clamp(ceil(log2(width)) - 1, 0, 3)`` with width floored at 1."""
    w = np.maximum(np.asarray(width, dtype=np.float64), 1.0)
    p = np.ceil(np.log2(w)) - 1.0
    return np.clip(p, P_MIN, P_MAX).astype(np.int64)


def raw_violation(query: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Inclusive interval: interior points contribute 0 regardless of midpoint."""
    q = np.asarray(query, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return np.maximum(0.0, np.maximum(lo - q, q - hi))


def penalty_from_violation(violation: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """``min(3, ceil(raw / 2^p))``."""
    step = np.left_shift(1, np.asarray(precision, dtype=np.int64))
    raw = np.asarray(violation, dtype=np.float64)
    stepped = np.ceil(raw / np.maximum(step.astype(np.float64), 1.0))
    return np.clip(stepped, 0, PENALTY_CAP).astype(np.int64)


def apply_censoring(
    query: np.ndarray,
    mu: np.ndarray,
    violation: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    """Asymmetric missing-energy rule (J5). Quiet is not don't-care.

    Template loud + observation quiet → strong mismatch.
    Template near floor + observation quiet → match (penalty 0).
    """
    if not enabled:
        return violation
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    v = np.asarray(violation, dtype=np.float64).copy()
    quiet = q <= QUIET_CODE
    loud = m > FLOOR_MU
    near_floor = m <= FLOOR_MU
    v = np.where(quiet & loud, np.maximum(v, float(PENALTY_CAP * 8)), v)
    v = np.where(quiet & near_floor, 0.0, v)
    return v


def score_rows(
    query: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    precision: np.ndarray,
    mu: np.ndarray,
    *,
    censoring: bool = False,
    active_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score every query against every row.

    ``query`` is (n, D), row arrays are (U, D). Returns ``S`` (n, U) and
    per-dimension penalties (n, U, D).
    """
    q = np.asarray(query, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    p = np.asarray(precision, dtype=np.int64)
    m = np.asarray(mu, dtype=np.float64)
    if q.ndim != 2 or lo.ndim != 2 or q.shape[1] != lo.shape[1]:
        raise ValueError("query/row feature dimensions must match")
    viol = raw_violation(q[:, None, :], lo[None, :, :], hi[None, :, :])
    viol = apply_censoring(q[:, None, :], m[None, :, :], viol, enabled=censoring)
    pen = penalty_from_violation(viol, p[None, :, :])
    scores = pen.sum(axis=2).astype(np.int64)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != scores.shape:
            raise ValueError("active_mask must have shape (n_queries, n_units)")
        scores = np.where(active, scores, np.iinfo(np.int64).max // 4)
        pen = np.where(active[:, :, None], pen, 0)
    return {"S": scores, "penalty": pen, "violation": viol}


def scale_alert(g_codes: np.ndarray, g_lo: np.ndarray, g_hi: np.ndarray) -> np.ndarray:
    """Per-query-per-unit alert. Does not change ``S_u``."""
    g = np.asarray(g_codes, dtype=np.int64).reshape(-1, 1)
    lo = np.asarray(g_lo, dtype=np.int64).reshape(1, -1)
    hi = np.asarray(g_hi, dtype=np.int64).reshape(1, -1)
    return (g < lo) | (g > hi)


def pack_shape_codes(com: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Concatenate COM (2) + POSNEG (2K) integer codes."""
    return np.concatenate(
        [
            np.asarray(com, dtype=np.int64),
            np.asarray(pos, dtype=np.int64),
            np.asarray(neg, dtype=np.int64),
        ],
        axis=1,
    )


def candidate_lists(
    scores: np.ndarray,
    tau: np.ndarray,
    units: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """``u in C iff S_u <= tau_u`` on active rows."""
    s = np.asarray(scores, dtype=np.int64)
    t = np.asarray(tau, dtype=np.int64).reshape(1, -1)
    labs = np.asarray(units)
    matched = s <= t
    if active_mask is not None:
        matched = matched & np.asarray(active_mask, dtype=bool)
    candidates = [labs[row].copy() for row in matched]
    n, u = s.shape
    order = np.argsort(s, axis=1, kind="stable")
    s1 = s[np.arange(n), order[:, 0]]
    s2 = s[np.arange(n), order[:, 1]] if u > 1 else np.full(n, np.iinfo(np.int64).max)
    top1 = labs[order[:, 0]]
    top2 = labs[order[:, 1]] if u > 1 else np.full(n, -1)
    counts = matched.sum(axis=1).astype(np.int64)
    compares = int(u * n) if active_mask is None else int(np.asarray(active_mask).sum())
    return candidates, {
        "counts": counts,
        "S1": s1,
        "S2": s2,
        "top1_id": top1,
        "top2_id": top2,
        "n_lower_upper_compares": int(2 * n * u),
        "n_shifts": int(n * u * s.shape[1] if False else n * u * (s.shape[1] if s.ndim else 0)),
        "n_active_row_comparisons": compares,
    }
