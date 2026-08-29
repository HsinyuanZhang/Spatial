"""Causal COM tracking on a 1-D physical axis.

This is not identity matching. Given a per-block GT COM trace, predictors
use only the past to forecast the next block. The question is whether the
forecast stays inside a physical range of the true COM.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def nearest_channel(com_um: np.ndarray, channel_pos_um: np.ndarray) -> np.ndarray:
    """Index of the electrode closest to each COM value. NaN COM → -1."""
    pos = np.asarray(channel_pos_um, dtype=np.float64).ravel()
    c = np.asarray(com_um, dtype=np.float64)
    out = np.full(c.shape, -1, dtype=np.int64)
    ok = np.isfinite(c)
    if not np.any(ok):
        return out
    d = np.abs(c[ok, None] - pos[None, :])
    out[ok] = np.argmin(d, axis=1)
    return out


def in_range(err_um: np.ndarray, radius_um: float) -> np.ndarray:
    e = np.asarray(err_um, dtype=np.float64)
    return np.isfinite(e) & (np.abs(e) <= float(radius_um))


def longest_false_run(mask: np.ndarray) -> int:
    """Longest consecutive False streak. NaN-as-missing is treated as False."""
    m = np.asarray(mask, dtype=bool)
    longest = cur = 0
    for v in m:
        if not v:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return int(longest)


def predict_frozen(observed: np.ndarray, centres_s: np.ndarray, t_fit_end: float) -> np.ndarray:
    """Hold the median COM of blocks whose centre is before t_fit_end."""
    v = np.asarray(observed, dtype=np.float64)
    t = np.asarray(centres_s, dtype=np.float64)
    fit = (t < float(t_fit_end)) & np.isfinite(v)
    hat = np.full(v.size, np.nan)
    if not np.any(fit):
        return hat
    level = float(np.median(v[fit]))
    # Causal: the frozen value is available after the fit window.
    hat[t >= float(t_fit_end)] = level
    return hat


def predict_hold(observed: np.ndarray) -> np.ndarray:
    """Zero-order hold: hat[i] = last finite observation with j < i."""
    v = np.asarray(observed, dtype=np.float64)
    hat = np.full(v.size, np.nan)
    last = np.nan
    for i in range(v.size):
        hat[i] = last
        if np.isfinite(v[i]):
            last = v[i]
    return hat


def predict_ema(observed: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA of observed COM. hat[i] uses updates from j < i only."""
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    v = np.asarray(observed, dtype=np.float64)
    hat = np.full(v.size, np.nan)
    state = np.nan
    for i in range(v.size):
        hat[i] = state
        if np.isfinite(v[i]):
            state = v[i] if not np.isfinite(state) else (1.0 - a) * state + a * v[i]
    return hat


def predict_velocity(observed: np.ndarray) -> np.ndarray:
    """Constant-velocity: hat[i] = y_{i-1} + (y_{i-1} - y_{i-2})."""
    v = np.asarray(observed, dtype=np.float64)
    hat = np.full(v.size, np.nan)
    prev = np.nan
    prev2 = np.nan
    for i in range(v.size):
        if np.isfinite(prev) and np.isfinite(prev2):
            hat[i] = prev + (prev - prev2)
        else:
            hat[i] = prev
        if np.isfinite(v[i]):
            prev2 = prev
            prev = v[i]
    return hat


def stream_every_n(
    observed: np.ndarray,
    n_commit: int,
    *,
    init_n: int = 10,
) -> np.ndarray:
    """Online COM: freeze state, score event, buffer; every n events commit median.

    ``hat[i]`` is the state *before* event i is observed.
    """
    n = max(int(n_commit), 1)
    init = max(int(init_n), 1)
    v = np.asarray(observed, dtype=np.float64)
    hat = np.full(v.size, np.nan)
    state = np.nan
    buf: list[float] = []
    n_seen = 0
    for i, yi in enumerate(v):
        hat[i] = state
        if not np.isfinite(yi):
            continue
        if not np.isfinite(state):
            buf.append(float(yi))
            if len(buf) >= init:
                state = float(np.median(buf))
                buf = []
            continue
        buf.append(float(yi))
        n_seen += 1
        if n_seen >= n:
            state = float(np.median(buf))
            buf = []
            n_seen = 0
    return hat


def stream_every_dt(
    observed: np.ndarray,
    times_s: np.ndarray,
    dt_s: float,
    *,
    t_init_end: float = 30.0,
) -> np.ndarray:
    """Online COM: freeze state, score event; commit median of buffer every dt_s.

    State is initialized as the median of events with t < t_init_end, then
    held until the first commit after that.
    """
    v = np.asarray(observed, dtype=np.float64)
    t = np.asarray(times_s, dtype=np.float64)
    hat = np.full(v.size, np.nan)
    init = (t < float(t_init_end)) & np.isfinite(v)
    state = float(np.median(v[init])) if np.any(init) else np.nan
    buf: list[float] = []
    t_commit = float(t_init_end)
    dt = float(dt_s)
    for i in range(v.size):
        hat[i] = state if t[i] >= float(t_init_end) else np.nan
        if t[i] < float(t_init_end) or not np.isfinite(v[i]):
            continue
        buf.append(float(v[i]))
        if t[i] - t_commit >= dt and buf:
            state = float(np.median(buf))
            buf = []
            t_commit = float(t[i])
    return hat


def stream_ema_event(
    observed: np.ndarray,
    alpha: float,
    *,
    init_n: int = 10,
) -> np.ndarray:
    """Per-event causal EMA: hat[i] uses updates from j < i only."""
    return predict_ema(np.asarray(observed, dtype=np.float64), float(alpha))


def contemporaneous_block_com(
    event_com: np.ndarray,
    event_t_s: np.ndarray,
    block_s: float = 5.0,
    *,
    leave_one_out: bool = False,
    min_events: int = 3,
    t0: float = 0.0,
) -> np.ndarray:
    """Per-event contemporaneous unit COM = median of that event's time block.

    This is not a causal predictor. It is the oracle "where was this unit at
    this time": the same 5 s block median used as GT COM, optionally leaving
    the scored event out so an outlier cannot shrink its own residual.

    Blocks with fewer than ``min_events`` finite COMs are NaN (same floor as
    the 5 s traces). Leave-one-out still requires that many events in the
    block; the event is then compared to the median of the others.
    """
    y = np.asarray(event_com, dtype=np.float64).ravel()
    t = np.asarray(event_t_s, dtype=np.float64).ravel()
    if y.size != t.size:
        raise ValueError(
            f"event_com and event_t_s length mismatch: {y.size} vs {t.size}"
        )
    hat = np.full(y.size, np.nan)
    ok = np.isfinite(y) & np.isfinite(t)
    if not np.any(ok):
        return hat
    bid = np.floor((t - float(t0)) / float(block_s)).astype(np.int64)
    need = int(min_events)
    for b in np.unique(bid[ok]):
        idx = np.flatnonzero(ok & (bid == b))
        if idx.size < need:
            continue
        vals = y[idx]
        if not leave_one_out:
            hat[idx] = float(np.median(vals))
            continue
        for k, i in enumerate(idx):
            hat[i] = float(np.median(np.delete(vals, k)))
    return hat


def tracking_errors(
    observed: np.ndarray,
    predicted: np.ndarray,
    channel_pos_um: np.ndarray,
    radii_um: Sequence[float],
) -> dict[str, float]:
    """Error stats on blocks where both GT and prediction are finite."""
    y = np.asarray(observed, dtype=np.float64)
    hat = np.asarray(predicted, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(hat)
    n_ok = int(ok.sum())
    nan = float("nan")
    if n_ok == 0:
        out = {
            "n_eval": 0,
            "median_abs_um": nan,
            "p90_abs_um": nan,
            "max_abs_um": nan,
            "frac_channel_match": nan,
            "frac_channel_miss": nan,
            "n_channel_miss": 0,
            "longest_channel_miss": 0,
        }
        for r in radii_um:
            out[f"frac_within_{r:g}um"] = nan
            out[f"frac_out_{r:g}um"] = nan
            out[f"n_out_{r:g}um"] = 0
            out[f"longest_miss_{r:g}um"] = 0
        return out
    err = hat[ok] - y[ok]
    abs_e = np.abs(err)
    ch_y = nearest_channel(y, channel_pos_um)
    ch_h = nearest_channel(hat, channel_pos_um)
    ch_ok = (ch_y == ch_h) & (ch_y >= 0) & ok
    n_ch_miss = int((~ch_ok[ok]).sum())
    out = {
        "n_eval": n_ok,
        "median_abs_um": float(np.median(abs_e)),
        "p90_abs_um": float(np.percentile(abs_e, 90)),
        "max_abs_um": float(np.max(abs_e)),
        "frac_channel_match": float(np.mean(ch_ok[ok])),
        "frac_channel_miss": float(n_ch_miss / n_ok),
        "n_channel_miss": n_ch_miss,
        "longest_channel_miss": longest_false_run(ch_ok),
    }
    for r in radii_um:
        inside = in_range(hat - y, r)
        n_out = int((ok & ~inside).sum())
        seq = []
        for i in range(y.size):
            if ok[i]:
                seq.append(bool(inside[i]))
        out[f"frac_within_{r:g}um"] = float(np.mean(inside[ok]))
        out[f"frac_out_{r:g}um"] = float(n_out / n_ok)
        out[f"n_out_{r:g}um"] = n_out
        out[f"longest_miss_{r:g}um"] = longest_false_run(np.array(seq, dtype=bool))
    return out
