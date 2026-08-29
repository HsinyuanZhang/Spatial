"""Causal shared/local drift tracking. Predict → freeze → classify → update.

Free per-candidate shift/gain on the current event is not implemented as a
deployable tracker here. Oracle free-shift scoring lives in
``spatial_shift_matching`` and is labeled DIAGNOSTIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.deformable_spatial_template import interpolation_matrix, shift_query_xy
from Spatial.algorithms.posneg_mvm_matching import (
    EPS,
    _scalar,
    as_distance,
    assignment_metrics,
    build_features,
    concat_posneg,
    score_kind,
)
from Spatial.algorithms.spatial_footprint import primary_axis
from Spatial.algorithms.spatial_shift_matching import (
    SHIFT_FRACTIONAL,
    score_shift_bank,
    shift_matrix,
    shift_posneg,
)


ETA_EMA = 0.05
ETA_EMA_GRID = (0.005, 0.01, 0.02, 0.05, 0.1)
ETA_STATE = 0.20
ETA_MU_SLOW = 0.005
BLOCK_S = 1.0
TILE_PITCHES = 3.0


def tile_ids(geom: np.ndarray, homes: np.ndarray, pitch_um: float, pitches_per_tile: float = TILE_PITCHES) -> np.ndarray:
    g = np.asarray(geom, dtype=np.float64)
    ax = int(primary_axis(g))
    h = np.asarray(homes, dtype=np.int64).ravel()
    pos = g[np.clip(h, 0, g.shape[0] - 1), ax]
    lo = float(np.min(g[:, ax]))
    width = max(float(pitches_per_tile) * float(pitch_um), EPS)
    return np.floor((pos - lo) / width).astype(np.int64)


def unit_xy(geom: np.ndarray, homes: np.ndarray) -> np.ndarray:
    g = np.asarray(geom, dtype=np.float64)
    h = np.asarray(homes, dtype=np.int64).ravel()
    return g[np.clip(h, 0, g.shape[0] - 1)]


def estimate_best_shift(
    pos: np.ndarray,
    neg: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    rel_xy: np.ndarray,
    live: np.ndarray,
    deltas: Sequence[float],
    pitch_um: float,
    axis: int,
    kind: str = "l2",
    representation: str = "posneg",
) -> tuple[float, float]:
    """Best rest-frame shift for one event vs one unit. Returns (delta, score)."""
    mats = np.stack(
        [shift_matrix(rel_xy, live, d, pitch_um, axis) for d in deltas],
        axis=0,
    )
    best, idx = score_shift_bank(
        pos, neg, mu_pos, mu_neg, mats, representation, kind, rel_xy
    )
    d = np.asarray(list(deltas), dtype=np.float64)
    return float(d[int(idx)]), float(best)


def estimate_gain(pos: np.ndarray, neg: np.ndarray, mu_pos: np.ndarray, mu_neg: np.ndarray) -> float:
    x = concat_posneg(pos, neg)
    mu = concat_posneg(mu_pos, mu_neg)
    den = float(np.dot(mu, mu)) + EPS
    return float(max(np.dot(x, mu) / den, 0.0))


def block_indices(times: np.ndarray, fs: float, block_s: float = BLOCK_S) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64).ravel() / float(fs)
    return np.floor(t / float(block_s)).astype(np.int64)


def gt_block_state(
    pos_true: np.ndarray,
    neg_true: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    rel_xy: np.ndarray,
    live: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    fs: float,
    pitch_um: float,
    axis: int,
    deltas: Sequence[float] = SHIFT_FRACTIONAL,
    block_s: float = BLOCK_S,
    unit_homes: np.ndarray | None = None,
) -> dict[str, Any]:
    """GT-only per-unit/block δ and g. Not used for classification.

    ``pos_true`` / ``neg_true`` are (n, K) already gathered on each event's
    *true* unit candidate-home slots.
    """
    y = np.asarray(labels)
    blocks = block_indices(times, fs, block_s)
    uniq_blocks = np.unique(blocks)
    n_u = int(units.size)
    n_b = int(uniq_blocks.size)
    delta = np.full((n_u, n_b), np.nan)
    gain = np.full((n_u, n_b), np.nan)
    counts = np.zeros((n_u, n_b), dtype=np.int64)
    bmap = {int(b): j for j, b in enumerate(uniq_blocks)}
    homes = None if unit_homes is None else np.asarray(unit_homes, dtype=np.int64).ravel()
    for ui, unit in enumerate(units):
        mask = np.array([_scalar(t) == _scalar(unit) for t in y], dtype=bool)
        if not np.any(mask):
            continue
        if rel_xy.ndim == 3:
            src = int(homes[ui]) if homes is not None else ui
            xy = rel_xy[src]
        else:
            xy = rel_xy
        if np.asarray(live).ndim == 2:
            src = int(homes[ui]) if homes is not None else ui
            lv = live[src]
        else:
            lv = live
        for e in np.flatnonzero(mask):
            bj = bmap[int(blocks[e])]
            d, _s = estimate_best_shift(
                pos_true[e],
                neg_true[e],
                mu_pos[ui],
                mu_neg[ui],
                xy,
                lv,
                deltas,
                pitch_um,
                axis,
            )
            g = estimate_gain(pos_true[e], neg_true[e], mu_pos[ui], mu_neg[ui])
            if counts[ui, bj] == 0:
                delta[ui, bj] = d
                gain[ui, bj] = g
            else:
                delta[ui, bj] += d
                gain[ui, bj] += g
            counts[ui, bj] += 1
    with np.errstate(invalid="ignore"):
        delta = np.where(counts > 0, delta / np.maximum(counts, 1), np.nan)
        gain = np.where(counts > 0, gain / np.maximum(counts, 1), np.nan)
    return {
        "delta": delta,
        "gain": gain,
        "counts": counts,
        "blocks": uniq_blocks,
    }


def pairwise_corr(series: np.ndarray) -> np.ndarray:
    """series (n_u, n_t). Pearson corr over time; NaN if <3 overlapping finite samples."""
    n = int(series.shape[0])
    out = np.full((n, n), np.nan)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            a = series[i]
            b = series[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if int(ok.sum()) < 3:
                continue
            aa = a[ok] - np.mean(a[ok])
            bb = b[ok] - np.mean(b[ok])
            den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if den <= EPS:
                continue
            c = float(np.dot(aa, bb) / den)
            out[i, j] = c
            out[j, i] = c
    return out


def coherence_vs_distance(
    delta_ub: np.ndarray,
    xy: np.ndarray,
    n_bins: int = 6,
) -> dict[str, Any]:
    corr = pairwise_corr(delta_ub)
    n = int(xy.shape[0])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    c = corr[iu, ju]
    ok = np.isfinite(c) & np.isfinite(d)
    d = d[ok]
    c = c[ok]
    if d.size == 0:
        return {
            "bin_centers": np.zeros(0),
            "c_d": np.zeros(0),
            "n_pairs": np.zeros(0, dtype=np.int64),
            "mean_corr": float("nan"),
            "nearest_corr": float("nan"),
            "corr_vs_dist_slope": float("nan"),
        }
    edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    centers = []
    vals = []
    ns = []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        vals.append(float(np.mean(c[m])) if np.any(m) else float("nan"))
        ns.append(int(m.sum()))
    nearest = float(vals[0]) if vals else float("nan")
    if d.size >= 3 and float(np.std(d)) > EPS:
        slope = float(np.polyfit(d, c, 1)[0])
    else:
        slope = float("nan")
    return {
        "bin_centers": np.asarray(centers, dtype=np.float64),
        "c_d": np.asarray(vals, dtype=np.float64),
        "n_pairs": np.asarray(ns, dtype=np.int64),
        "mean_corr": float(np.mean(c)),
        "nearest_corr": nearest,
        "corr_vs_dist_slope": slope,
        "pair_dist": d,
        "pair_corr": c,
    }


def apply_ema_step(
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    owner: int,
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    eta: float,
) -> None:
    """In-place mean-template step: μ ← (1-η)μ + η x on one unit."""
    i = int(owner)
    a = float(eta)
    mu_pos[i] = (1.0 - a) * mu_pos[i] + a * np.asarray(x_pos, dtype=np.float64)
    mu_neg[i] = (1.0 - a) * mu_neg[i] + a * np.asarray(x_neg, dtype=np.float64)


@dataclass
class TrackerState:
    mu_pos: np.ndarray
    mu_neg: np.ndarray
    delta: np.ndarray
    gain: np.ndarray


class CausalTracker:
    """Scores with frozen state; updates only through ``maybe_update``."""

    def __init__(
        self,
        mode: str,
        mu_pos: np.ndarray,
        mu_neg: np.ndarray,
        unit_homes: np.ndarray,
        rel_xy: np.ndarray,
        table: np.ndarray,
        geom: np.ndarray,
        pitch_um: float,
        axis: int,
        group_id: np.ndarray,
        deltas: Sequence[float] = SHIFT_FRACTIONAL,
        eta_state: float = ETA_STATE,
        eta_mu: float = 0.0,
        eta_gain: float = 0.0,
        eta_ema: float | None = None,
        kind: str = "l2",
        representation: str = "posneg",
        confidence_threshold: float = 0.0,
        update_owner: str = "self",
    ) -> None:
        allowed = (
            "frozen",
            "ema",
            "oracle_ema",
            "global_shift",
            "tile_shift",
            "smooth_field",
            "shared_shift_slow_mu",
            "oracle_shift",
        )
        if mode not in allowed:
            raise ValueError(f"unknown tracker mode {mode!r}")
        owner = str(update_owner)
        if mode == "oracle_ema":
            owner = "gt"
        if owner not in ("self", "gt"):
            raise ValueError(f"update_owner must be 'self' or 'gt', got {update_owner!r}")
        self.mode = str(mode)
        self.update_owner = owner
        self.eta_ema = float(ETA_EMA if eta_ema is None else eta_ema)
        self.mu_pos = np.asarray(mu_pos, dtype=np.float64).copy()
        self.mu_neg = np.asarray(mu_neg, dtype=np.float64).copy()
        self.mu_pos0 = self.mu_pos.copy()
        self.mu_neg0 = self.mu_neg.copy()
        self.unit_homes = np.asarray(unit_homes, dtype=np.int64).ravel()
        self.rel_xy = np.asarray(rel_xy, dtype=np.float64)
        self.table = np.asarray(table)
        self.geom = np.asarray(geom, dtype=np.float64)
        self.pitch = float(pitch_um)
        self.axis = int(axis)
        self.group_id = np.asarray(group_id, dtype=np.int64).ravel()
        self.n_group = int(self.group_id.max() + 1) if self.group_id.size else 1
        if mode == "global_shift":
            self.group_id = np.zeros(self.mu_pos.shape[0], dtype=np.int64)
            self.n_group = 1
        self.delta = np.zeros(self.n_group, dtype=np.float64)
        self.gain = np.ones(self.n_group, dtype=np.float64)
        self.deltas = tuple(float(d) for d in deltas)
        self.eta_state = float(eta_state)
        self.eta_mu = float(eta_mu)
        self.eta_gain = float(eta_gain)
        self.kind = str(kind)
        self.representation = str(representation)
        self.confidence_threshold = float(confidence_threshold)
        self._last_shift_index = None
        self._oracle_mats = None
        self.state_mu_pos = None
        self.state_mu_neg = None
        self.state_rel_xy = None
        self.state_table = None
        self.state_homes = None
        self.state_pitch = float(pitch_um)
        self.state_axis = int(axis)
        self._shift_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        if mode == "oracle_shift":
            from Spatial.algorithms.spatial_shift_matching import precompute_shift_mats

            self._oracle_mats = precompute_shift_mats(
                self.rel_xy, self.table, self.deltas, self.pitch, self.axis
            )

    def attach_state_encoder(
        self,
        mu_pos: np.ndarray,
        mu_neg: np.ndarray,
        rel_xy: np.ndarray,
        table: np.ndarray,
        homes: np.ndarray,
        pitch_um: float | None = None,
        axis: int | None = None,
    ) -> None:
        """Wider-K encoder used only to estimate shared δ after assignment."""
        self.state_mu_pos = np.asarray(mu_pos, dtype=np.float64)
        self.state_mu_neg = np.asarray(mu_neg, dtype=np.float64)
        self.state_rel_xy = np.asarray(rel_xy, dtype=np.float64)
        self.state_table = np.asarray(table)
        self.state_homes = np.asarray(homes, dtype=np.int64).ravel()
        if pitch_um is not None:
            self.state_pitch = float(pitch_um)
        if axis is not None:
            self.state_axis = int(axis)

    def snapshot(self) -> TrackerState:
        return TrackerState(
            mu_pos=self.mu_pos.copy(),
            mu_neg=self.mu_neg.copy(),
            delta=self.delta.copy(),
            gain=self.gain.copy(),
        )

    def restore(self, state: TrackerState) -> None:
        self.mu_pos = state.mu_pos.copy()
        self.mu_neg = state.mu_neg.copy()
        self.delta = state.delta.copy()
        self.gain = state.gain.copy()
        self._shift_cache.clear()

    def _group_delta(self, ui: int) -> float:
        return float(self.delta[int(self.group_id[ui])])

    def _shifted_mu(self, ui: int) -> tuple[np.ndarray, np.ndarray]:
        h = int(self.unit_homes[ui])
        d = self._group_delta(ui)
        g = float(self.gain[int(self.group_id[ui])])
        if abs(d) <= 1e-15:
            return g * self.mu_pos[ui], g * self.mu_neg[ui]
        key = (ui, round(float(d), 4))
        cached = self._shift_cache.get(key)
        if cached is None:
            xy = self.rel_xy[h]
            live = self.table[h] >= 0
            mat = shift_matrix(xy, live, d, self.pitch, self.axis)
            cached = shift_posneg(self.mu_pos[ui], self.mu_neg[ui], mat)
            self._shift_cache[key] = cached
        sp, sn = cached
        return g * sp, g * sn

    def scores_from_posneg(
        self,
        pos_eu: np.ndarray,
        neg_eu: np.ndarray,
    ) -> np.ndarray:
        """pos_eu (U, K) for one event or (n, U, K). Does not mutate state."""
        p = np.asarray(pos_eu, dtype=np.float64)
        n = np.asarray(neg_eu, dtype=np.float64)
        single = p.ndim == 2
        if single:
            p = p[None, :, :]
            n = n[None, :, :]
        n_e, n_u, _k = p.shape
        out = np.empty((n_e, n_u), dtype=np.float64)
        if self.mode == "oracle_shift":
            from Spatial.algorithms.spatial_shift_matching import moving_scores_all_units

            rec = moving_scores_all_units(
                p,
                n,
                self.mu_pos,
                self.mu_neg,
                self.unit_homes,
                self.rel_xy,
                self._oracle_mats,
                self.representation,
                self.kind,
            )
            return rec["score"][0] if single else rec["score"]
        for ui in range(n_u):
            h = int(self.unit_homes[ui])
            xy = self.rel_xy[h]
            sp, sn = self._shifted_mu(ui)
            mu = build_features(sp, sn, self.representation, xy)
            x = build_features(p[:, ui], n[:, ui], self.representation, xy)
            out[:, ui] = score_kind(x, mu, self.kind)
        return out[0] if single else out

    def maybe_update(
        self,
        pos_u: np.ndarray,
        neg_u: np.ndarray,
        pred_index: int,
        distances_row: np.ndarray,
        *,
        confident: bool | None = None,
        state_pos_u: np.ndarray | None = None,
        state_neg_u: np.ndarray | None = None,
        gt_index: int | None = None,
    ) -> bool:
        """Update state for *future* events. Returns whether an update ran."""
        if self.mode in ("frozen", "oracle_shift"):
            return False
        dist = np.asarray(distances_row, dtype=np.float64).ravel()
        owner_index = int(pred_index)
        if self.update_owner == "gt":
            if gt_index is None:
                raise ValueError("oracle/gt update requires gt_index")
            owner_index = int(gt_index)
        if confident is None:
            others = dist.copy()
            others[owner_index] = np.inf
            m_all = float(np.min(others) - dist[owner_index])
            confident = m_all > self.confidence_threshold
        if not confident:
            return False
        ui = int(owner_index)
        if (
            self.state_mu_pos is not None
            and state_pos_u is not None
            and state_neg_u is not None
        ):
            h = int(self.state_homes[ui])
            xy = self.state_rel_xy[h]
            live = self.state_table[h] >= 0
            pos = np.asarray(state_pos_u[ui], dtype=np.float64)
            neg = np.asarray(state_neg_u[ui], dtype=np.float64)
            mu_p = self.state_mu_pos[ui]
            mu_n = self.state_mu_neg[ui]
            pitch = self.state_pitch
            axis = self.state_axis
        else:
            h = int(self.unit_homes[ui])
            xy = self.rel_xy[h]
            live = self.table[h] >= 0
            pos = np.asarray(pos_u[ui], dtype=np.float64)
            neg = np.asarray(neg_u[ui], dtype=np.float64)
            mu_p = self.mu_pos[ui]
            mu_n = self.mu_neg[ui]
            pitch = self.pitch
            axis = self.axis
        if self.mode in ("ema", "oracle_ema"):
            id_pos = np.asarray(pos_u[ui], dtype=np.float64)
            id_neg = np.asarray(neg_u[ui], dtype=np.float64)
            apply_ema_step(self.mu_pos, self.mu_neg, ui, id_pos, id_neg, self.eta_ema)
            self._shift_cache.clear()
            return True
        d_hat, _ = estimate_best_shift(
            pos,
            neg,
            mu_p,
            mu_n,
            xy,
            live,
            self.deltas,
            pitch,
            axis,
            kind=self.kind,
            representation=self.representation,
        )
        gid = int(self.group_id[ui])
        if self.mode in ("global_shift", "tile_shift", "smooth_field", "shared_shift_slow_mu"):
            self.delta[gid] = (1.0 - self.eta_state) * self.delta[gid] + self.eta_state * d_hat
        if self.mode == "smooth_field":
            # Linear field along primary axis: after tile-like update, fit c0,c1 from group deltas.
            self._refit_linear_field()
        if self.mode == "shared_shift_slow_mu" and self.eta_mu > 0:
            h_id = int(self.unit_homes[ui])
            xy_id = self.rel_xy[h_id]
            live_id = self.table[h_id] >= 0
            mat = shift_matrix(xy_id, live_id, -self.delta[gid], self.pitch, self.axis)
            x_pos, x_neg = shift_posneg(
                np.asarray(pos_u[ui], dtype=np.float64),
                np.asarray(neg_u[ui], dtype=np.float64),
                mat,
            )
            self.mu_pos[ui] = self.mu_pos[ui] + self.eta_mu * (x_pos - self.mu_pos[ui])
            self.mu_neg[ui] = self.mu_neg[ui] + self.eta_mu * (x_neg - self.mu_neg[ui])
            self._shift_cache.clear()
        if self.eta_gain > 0:
            g_hat = estimate_gain(pos, neg, *self._shifted_mu(ui))
            self.gain[gid] = (1.0 - self.eta_gain) * self.gain[gid] + self.eta_gain * g_hat
        return True

    def _refit_linear_field(self) -> None:
        g = self.geom
        ax = self.axis
        homes = self.unit_homes
        r = g[np.clip(homes, 0, g.shape[0] - 1), ax]
        r = (r - np.mean(r)) / max(float(np.std(r)), EPS)
        y = self.delta[self.group_id]
        a = np.column_stack([np.ones(r.size), r])
        try:
            coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        except np.linalg.LinAlgError:
            return
        fitted = a @ coef
        for gid in range(self.n_group):
            members = self.group_id == gid
            if np.any(members):
                self.delta[gid] = float(np.mean(fitted[members]))

    def mu_movement_ratio(self) -> np.ndarray:
        """||μ(t)-μ(0)|| / nearest same-home rest-template distance."""
        n_u = int(self.mu_pos.shape[0])
        rest = np.concatenate([self.mu_pos0, self.mu_neg0], axis=1)
        now = np.concatenate([self.mu_pos, self.mu_neg], axis=1)
        move = np.linalg.norm(now - rest, axis=1)
        ratio = np.full(n_u, np.nan)
        for i in range(n_u):
            rival = (self.unit_homes == self.unit_homes[i]) & (np.arange(n_u) != i)
            if not np.any(rival):
                ratio[i] = np.nan
                continue
            dmin = min(float(np.linalg.norm(rest[i] - rest[j])) for j in np.flatnonzero(rival))
            ratio[i] = float(move[i] / max(dmin, EPS))
        return ratio


def run_causal_stream(
    tracker: CausalTracker,
    pos_eu: np.ndarray,
    neg_eu: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    state_pos_eu: np.ndarray | None = None,
    state_neg_eu: np.ndarray | None = None,
) -> dict[str, Any]:
    """Event-by-event: score with frozen state, then maybe update."""
    p = np.asarray(pos_eu, dtype=np.float64)
    n = np.asarray(neg_eu, dtype=np.float64)
    y = np.asarray(labels)
    n_e = int(p.shape[0])
    n_u = int(tracker.mu_pos.shape[0])
    scores = np.empty((n_e, n_u), dtype=np.float64)
    updated = np.zeros(n_e, dtype=bool)
    delta_trace = np.empty((n_e, tracker.n_group), dtype=np.float64)
    mu_move = np.empty((n_e, n_u), dtype=np.float64)
    for i in range(n_e):
        row = tracker.scores_from_posneg(p[i], n[i])
        scores[i] = row
        dist = as_distance(row, tracker.kind)
        if tracker.kind in ("l2", "l1", "weighted_l2"):
            pred_i = int(np.argmin(row))
        else:
            pred_i = int(np.argmax(row))
        sp = None if state_pos_eu is None else state_pos_eu[i]
        sn = None if state_neg_eu is None else state_neg_eu[i]
        gt_i = None
        if tracker.update_owner == "gt":
            lab = y[i]
            for j, u in enumerate(units):
                if _scalar(u) == _scalar(lab):
                    gt_i = j
                    break
        updated[i] = tracker.maybe_update(
            p[i], n[i], pred_i, dist, state_pos_u=sp, state_neg_u=sn, gt_index=gt_i
        )
        delta_trace[i] = tracker.delta
        mu_move[i] = np.linalg.norm(
            np.concatenate([tracker.mu_pos, tracker.mu_neg], axis=1)
            - np.concatenate([tracker.mu_pos0, tracker.mu_neg0], axis=1),
            axis=1,
        )
    metrics = assignment_metrics(scores, y, units, tracker.unit_homes, tracker.kind)
    metrics["n_updates"] = int(updated.sum())
    metrics["update_rate"] = float(updated.mean()) if n_e else float("nan")
    metrics["delta_trace"] = delta_trace
    metrics["mu_move_trace"] = mu_move
    metrics["mu_move_ratio_end"] = tracker.mu_movement_ratio()
    metrics["delta_end"] = tracker.delta.copy()
    metrics["mean_abs_delta_end"] = float(np.mean(np.abs(tracker.delta)))
    metrics["scores"] = scores
    return metrics


def apply_gain_ramp(
    vmax: np.ndarray,
    vmin: np.ndarray,
    g0: float,
    g1: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.asarray(vmax).shape[0])
    g = np.linspace(float(g0), float(g1), n)[:, None]
    return np.asarray(vmax, dtype=np.float64) * g, np.asarray(vmin, dtype=np.float64) * g


def apply_gain_step(
    vmax: np.ndarray,
    vmin: np.ndarray,
    g_lo: float,
    g_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.asarray(vmax).shape[0])
    g = np.full(n, float(g_lo))
    g[n // 2 :] = float(g_hi)
    return np.asarray(vmax, dtype=np.float64) * g[:, None], np.asarray(vmin, dtype=np.float64) * g[:, None]


def channel_shift_matrix(geom: np.ndarray, delta_pitch: float, pitch_um: float, axis: int) -> np.ndarray:
    g = np.asarray(geom, dtype=np.float64)
    live = np.ones(g.shape[0], dtype=bool)
    dst = shift_query_xy(g, float(delta_pitch), float(pitch_um), int(axis))
    return interpolation_matrix(g, dst, live)


def apply_spatial_ramp(
    vmax: np.ndarray,
    vmin: np.ndarray,
    geom: np.ndarray,
    d0: float,
    d1: float,
    pitch_um: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vmx = np.asarray(vmax, dtype=np.float64)
    vmn = np.asarray(vmin, dtype=np.float64)
    n = int(vmx.shape[0])
    deltas = np.linspace(float(d0), float(d1), n)
    out_p = np.empty_like(vmx)
    out_n = np.empty_like(vmn)
    last_d = None
    a = None
    for i, d in enumerate(deltas):
        dq = round(float(d), 3)
        if last_d is None or dq != last_d:
            a = channel_shift_matrix(geom, dq, pitch_um, axis)
            last_d = dq
        out_p[i] = a @ vmx[i]
        out_n[i] = a @ vmn[i]
    return out_p, out_n, deltas


def apply_two_tile_opposite(
    vmax: np.ndarray,
    vmin: np.ndarray,
    geom: np.ndarray,
    d_end: float,
    pitch_um: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(geom, dtype=np.float64)
    ax = int(axis)
    mid = float(np.median(g[:, ax]))
    low = g[:, ax] < mid
    high = ~low
    vmx = np.asarray(vmax, dtype=np.float64)
    vmn = np.asarray(vmin, dtype=np.float64)
    n = int(vmx.shape[0])
    mag = np.linspace(0.0, float(d_end), n)
    out_p = np.empty_like(vmx)
    out_n = np.empty_like(vmn)
    d_low = -mag
    d_high = mag
    # Apply per-channel interpolated field: channels in the low tile sample as if
    # shifted by -mag, high tile by +mag. Mix via a global interpolation using
    # the channel's own delta.
    for i in range(n):
        a_lo = channel_shift_matrix(geom, float(d_low[i]), pitch_um, ax)
        a_hi = channel_shift_matrix(geom, float(d_high[i]), pitch_um, ax)
        a = np.where(low[:, None], a_lo, a_hi)
        out_p[i] = a @ vmx[i]
        out_n[i] = a @ vmn[i]
    return out_p, out_n, d_low, d_high
