"""Identity + trackable spatial state on analog POSNEG.

Default frame is candidate-home geometry, not event-argmax recentering.
Phase 1 searches state independently per event toward a fit rest state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    median_nearest_neighbor_spacing,
    primary_axis,
)


EPS = 1e-12
DELTA_MAX_PITCH = 1.5
N_DELTA = 21
LAMBDA_GRID = (0.0, 0.3, 1.0)
K_GRID = (7, 12, 16)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def extract_all_channel_extrema(
    signal: np.ndarray,
    times: np.ndarray,
    window: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event, per-channel max and min in ±window samples."""
    sig = np.asarray(signal, dtype=np.float64)
    t = np.asarray(times, dtype=np.int64).ravel()
    n_ch, n_samp = sig.shape
    win = int(window)
    if win < 1:
        raise ValueError("window must be >= 1")
    keep = (t >= win) & (t < n_samp - win)
    t = t[keep]
    n = int(t.size)
    vmax = np.empty((n, n_ch), dtype=np.float64)
    vmin = np.empty((n, n_ch), dtype=np.float64)
    for i, ti in enumerate(t):
        seg = sig[:, int(ti) - win : int(ti) + win]
        vmax[i] = np.max(seg, axis=1)
        vmin[i] = np.min(seg, axis=1)
    return vmax, vmin


def geometry_k_table(geom: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K geometrically nearest electrodes per home, ordered by physical offset.

    Slot order is signed primary-axis offset then secondary, never event
    amplitude. If the probe has fewer than K sites, remaining slots are -1.
    """
    g = np.asarray(geom, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"geom must be (C,2), got {g.shape}")
    n = int(g.shape[0])
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be >= 1")
    take = min(kk, n)
    ax = primary_axis(g)
    other = 1 - ax
    table = np.full((n, kk), -1, dtype=np.int64)
    rel_xy = np.zeros((n, kk, 2), dtype=np.float64)
    if n == 0:
        return table, rel_xy
    for ch in range(n):
        dist = np.linalg.norm(g - g[ch], axis=1)
        nearest = np.argsort(dist, kind="stable")[:take]
        rel = g[nearest] - g[ch]
        order = np.lexsort((rel[:, other], rel[:, ax]))
        chosen = nearest[order]
        table[ch, :take] = chosen
        rel_xy[ch, :take] = g[chosen] - g[ch]
    return table, rel_xy


def gather_posneg(
    vmax: np.ndarray,
    vmin: np.ndarray,
    slot_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather analog q+, q- on a slot row. Absent slots (id < 0) are 0."""
    vmx = np.asarray(vmax, dtype=np.float64).ravel()
    vmn = np.asarray(vmin, dtype=np.float64).ravel()
    slots = np.asarray(slot_ids, dtype=np.int64).ravel()
    absent = slots < 0
    safe = np.where(absent, 0, slots)
    pos = np.maximum(vmx[safe], 0.0)
    neg = np.maximum(-vmn[safe], 0.0)
    pos = np.where(absent, 0.0, pos)
    neg = np.where(absent, 0.0, neg)
    return pos, neg, absent


def gather_posneg_batch(
    vmax: np.ndarray,
    vmin: np.ndarray,
    table_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather POSNEG for every event on one home's slot row."""
    vmx = np.asarray(vmax, dtype=np.float64)
    vmn = np.asarray(vmin, dtype=np.float64)
    slots = np.asarray(table_row, dtype=np.int64).ravel()
    n = int(vmx.shape[0])
    k = int(slots.size)
    absent = slots < 0
    safe = np.where(absent, 0, slots)
    pos = np.maximum(vmx[:, safe], 0.0)
    neg = np.maximum(-vmn[:, safe], 0.0)
    pos[:, absent] = 0.0
    neg[:, absent] = 0.0
    mask = np.repeat(absent[None, :], n, axis=0)
    return pos, neg, mask


def event_peak_channel(vmax: np.ndarray, vmin: np.ndarray) -> np.ndarray:
    """h_e = argmax_c max(q+_c, q-_c)."""
    peak = np.maximum(np.maximum(vmax, 0.0), np.maximum(-vmin, 0.0))
    return np.argmax(peak, axis=1).astype(np.int64)


def majority_home(homes: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    h = np.asarray(homes, dtype=np.int64).ravel()
    y = np.asarray(labels)
    out = np.full(int(units.size), -1, dtype=np.int64)
    n_ch = int(max(int(h.max()) + 1, 1)) if h.size else 1
    for i, unit in enumerate(units):
        ch = h[y == unit]
        if ch.size == 0:
            continue
        out[i] = int(np.bincount(np.clip(ch, 0, n_ch - 1), minlength=n_ch).argmax())
    return out


def interpolation_matrix(src_xy: np.ndarray, dst_xy: np.ndarray, live: np.ndarray) -> np.ndarray:
    """IDW weights so dest ≈ A @ values_src. Exact hits copy the source slot."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    live = np.asarray(live, dtype=bool).ravel()
    k = int(src.shape[0])
    a = np.zeros((k, k), dtype=np.float64)
    if not np.any(live):
        return a
    src_l = src[live]
    idx = np.flatnonzero(live)
    d = np.linalg.norm(dst[:, None, :] - src_l[None, :, :], axis=2)
    for i in range(k):
        if not live[i]:
            continue
        row = d[i]
        jmin = int(np.argmin(row))
        if row[jmin] <= 1e-8:
            a[i, idx[jmin]] = 1.0
            continue
        w = 1.0 / np.maximum(row**2, EPS)
        a[i, idx] = w / np.sum(w)
    return a


def shift_query_xy(slot_xy: np.ndarray, delta_pitch: float, pitch_um: float, axis: int) -> np.ndarray:
    """Move the sampling points by -delta along the primary axis (template shift +δ)."""
    shift = np.zeros(2, dtype=np.float64)
    shift[int(axis)] = float(delta_pitch) * float(pitch_um)
    return np.asarray(slot_xy, dtype=np.float64) - shift


def stack_posneg(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, neg], axis=-1)


def apply_gain_polarity(
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    a_pos: float | np.ndarray,
    a_neg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(a_pos) * mu_pos, np.asarray(a_neg) * mu_neg


def closed_form_scales(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    m_pos: np.ndarray,
    m_neg: np.ndarray,
    live: np.ndarray,
    *,
    gain: bool,
    polarity: bool,
) -> tuple[float, float]:
    """Least-squares a+, a- on live slots. polarity=False ties a+=a-."""
    lp = np.asarray(live, dtype=bool)
    xp = np.asarray(x_pos, dtype=np.float64)
    xn = np.asarray(x_neg, dtype=np.float64)
    mp = np.asarray(m_pos, dtype=np.float64)
    mn = np.asarray(m_neg, dtype=np.float64)
    if not gain:
        return 1.0, 1.0
    if polarity:
        den_p = float(np.dot(mp[lp], mp[lp])) + EPS
        den_n = float(np.dot(mn[lp], mn[lp])) + EPS
        ap = max(float(np.dot(xp[lp], mp[lp]) / den_p), 0.0)
        an = max(float(np.dot(xn[lp], mn[lp]) / den_n), 0.0)
        return ap, an
    m = np.concatenate([mp[lp], mn[lp]])
    x = np.concatenate([xp[lp], xn[lp]])
    den = float(np.dot(m, m)) + EPS
    a = max(float(np.dot(x, m) / den), 0.0)
    return a, a


def residual_ls(resid: np.ndarray, basis: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Ridge-free least squares of resid onto columns of basis, live slots only."""
    b = np.asarray(basis, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    lp = np.asarray(live, dtype=bool)
    bb = b[lp]
    rr = r[lp]
    gram = bb.T @ bb
    gram = gram + 1e-6 * np.eye(gram.shape[0])
    try:
        h = np.linalg.solve(gram, bb.T @ rr)
    except np.linalg.LinAlgError:
        h = np.linalg.pinv(gram) @ (bb.T @ rr)
    return h


@dataclass(frozen=True)
class StateEnables:
    gain: bool = False
    offset: bool = False
    polarity: bool = False
    n_h: int = 0


ENABLE_ARMS: dict[str, StateEnables] = {
    "C0": StateEnables(),
    "C1": StateEnables(gain=True),
    "C2": StateEnables(offset=True),
    "C3": StateEnables(gain=True, offset=True),
    "C4": StateEnables(gain=True, offset=True, polarity=True),
    "C5": StateEnables(gain=True, offset=True, polarity=True, n_h=1),
    "C6": StateEnables(gain=True, offset=True, polarity=True, n_h=2),
}


@dataclass
class RestState:
    alpha: float = 0.0
    delta: float = 0.0
    rho: float = 0.0
    h: np.ndarray | None = None
    sig_alpha: float = 1.0
    sig_delta: float = 1.0
    sig_rho: float = 1.0
    sig_h: float = 1.0


def dyn_penalty(alpha: float, delta: float, rho: float, h: np.ndarray, rest: RestState) -> float:
    d = ((alpha - rest.alpha) / max(rest.sig_alpha, EPS)) ** 2
    d += ((delta - rest.delta) / max(rest.sig_delta, EPS)) ** 2
    d += ((rho - rest.rho) / max(rest.sig_rho, EPS)) ** 2
    if h.size:
        hh = h - (rest.h if rest.h is not None and rest.h.size == h.size else 0.0)
        d += float(np.sum((hh / max(rest.sig_h, EPS)) ** 2))
    return float(d)


def mse_live(x: np.ndarray, recon: np.ndarray, live: np.ndarray) -> float:
    lp = np.asarray(live, dtype=bool)
    if not np.any(lp):
        return 0.0
    err = np.asarray(x, dtype=np.float64)[lp] - np.asarray(recon, dtype=np.float64)[lp]
    return float(np.mean(err**2))


def explained_r2(x: np.ndarray, recon: np.ndarray, live: np.ndarray) -> float:
    lp = np.asarray(live, dtype=bool)
    if not np.any(lp):
        return float("nan")
    y = np.asarray(x, dtype=np.float64)[lp]
    yhat = np.asarray(recon, dtype=np.float64)[lp]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return 1.0 - ss_res / ss_tot


class DeformableSpatialTemplate:
    """Per-unit μ plus optional shared B, scored with a searched state."""

    def __init__(
        self,
        geom: np.ndarray,
        k: int,
        enables: StateEnables,
        lam_state: float = 0.0,
        n_delta: int = N_DELTA,
        delta_max: float = DELTA_MAX_PITCH,
    ) -> None:
        self.geom = np.asarray(geom, dtype=np.float64)
        self.k = int(k)
        self.enables = enables
        self.lam_state = float(lam_state)
        self.n_delta = int(n_delta)
        self.delta_max = float(delta_max)
        self.table, self.rel_xy = geometry_k_table(self.geom, self.k)
        self.pitch = (
            float(median_nearest_neighbor_spacing(self.geom))
            if self.geom.shape[0] >= 2
            else 1.0
        )
        self.axis = int(primary_axis(self.geom))
        self.delta_grid = np.linspace(-self.delta_max, self.delta_max, self.n_delta)
        if not enables.offset:
            self.delta_grid = np.array([0.0])
        self.units_: np.ndarray | None = None
        self.homes_: np.ndarray | None = None
        self.mu_pos_: np.ndarray | None = None
        self.mu_neg_: np.ndarray | None = None
        self.live_: np.ndarray | None = None
        self.basis_: np.ndarray | None = None
        self.rest_: list[RestState] | None = None
        self.shift_mats_: dict[int, np.ndarray] | None = None

    def _home_shift_mats(self, home: int) -> np.ndarray:
        if self.shift_mats_ is not None and int(home) in self.shift_mats_:
            return self.shift_mats_[int(home)]
        xy = self.rel_xy[int(home)]
        live = self.table[int(home)] >= 0
        mats = np.stack(
            [
                interpolation_matrix(xy, shift_query_xy(xy, d, self.pitch, self.axis), live)
                for d in self.delta_grid
            ],
            axis=0,
        )
        if self.shift_mats_ is None:
            self.shift_mats_ = {}
        self.shift_mats_[int(home)] = mats
        return mats

    def fit(
        self,
        vmax: np.ndarray,
        vmin: np.ndarray,
        labels: np.ndarray,
        event_homes: np.ndarray,
        *,
        frame: str = "candidate_home",
    ) -> "DeformableSpatialTemplate":
        if frame not in ("candidate_home", "event_argmax"):
            raise ValueError(f"unknown frame {frame!r}")
        self.frame_ = frame
        y = np.asarray(labels)
        units = np.array([_scalar(u) for u in np.unique(y)])
        homes = majority_home(event_homes, y, units)
        n_u = int(units.size)
        mu_pos = np.zeros((n_u, self.k), dtype=np.float64)
        mu_neg = np.zeros((n_u, self.k), dtype=np.float64)
        live = np.zeros((n_u, self.k), dtype=bool)
        rest: list[RestState] = []
        residuals: list[np.ndarray] = []
        eh = np.asarray(event_homes, dtype=np.int64).ravel()
        for i, unit in enumerate(units):
            h = int(homes[i])
            if h < 0:
                rest.append(RestState())
                continue
            mask = y == unit
            if frame == "candidate_home":
                live[i] = self.table[h] >= 0
                pos, neg, _ = gather_posneg_batch(vmax[mask], vmin[mask], self.table[h])
            else:
                idx = np.flatnonzero(mask)
                pos = np.zeros((idx.size, self.k), dtype=np.float64)
                neg = np.zeros((idx.size, self.k), dtype=np.float64)
                for j, e in enumerate(idx):
                    pos[j], neg[j], _ = gather_posneg(vmax[e], vmin[e], self.table[int(eh[e])])
                live[i] = np.any(pos + neg > 0, axis=0) | (self.table[h] >= 0)
            mu_pos[i] = pos.mean(axis=0)
            mu_neg[i] = neg.mean(axis=0)
            a_pos, a_neg = [], []
            for xp, xn in zip(pos, neg):
                ap, an = closed_form_scales(
                    xp, xn, mu_pos[i], mu_neg[i], live[i],
                    gain=True, polarity=True,
                )
                a_pos.append(ap)
                a_neg.append(an)
                recon = stack_posneg(ap * mu_pos[i], an * mu_neg[i])
                residuals.append(stack_posneg(xp, xn) - recon)
            ap = np.asarray(a_pos, dtype=np.float64)
            an = np.asarray(a_neg, dtype=np.float64)
            ap = np.maximum(ap, EPS)
            an = np.maximum(an, EPS)
            alpha = np.log(np.sqrt(ap * an))
            rho = np.log(ap / an)
            rest.append(
                RestState(
                    alpha=float(np.median(alpha)) if alpha.size else 0.0,
                    delta=0.0,
                    rho=float(np.median(rho)) if rho.size else 0.0,
                    sig_alpha=float(max(np.std(alpha), 0.3)) if alpha.size > 1 else 0.3,
                    sig_delta=0.5,
                    sig_rho=float(max(np.std(rho), 0.3)) if rho.size > 1 else 0.3,
                    sig_h=1.0,
                )
            )
        basis = np.zeros((2 * self.k, 0), dtype=np.float64)
        n_h = int(self.enables.n_h)
        if n_h > 0 and residuals:
            rmat = np.stack(residuals, axis=0)
            rmat = np.nan_to_num(rmat, nan=0.0)
            rmat = rmat - rmat.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(rmat, full_matrices=False)
            n_keep = min(n_h, int(vt.shape[0]))
            basis = vt[:n_keep].T
        self.units_ = units
        self.homes_ = homes
        self.mu_pos_ = mu_pos
        self.mu_neg_ = mu_neg
        self.live_ = live
        self.basis_ = basis
        self.rest_ = rest
        self.frame_ = getattr(self, "frame_", "candidate_home")
        return self

    def _score_unit(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
        unit_index: int,
        gather_home: int,
    ) -> dict[str, Any]:
        assert self.mu_pos_ is not None and self.homes_ is not None
        live = self.table[int(gather_home)] >= 0
        live2 = np.concatenate([live, live])
        x = stack_posneg(pos, neg)
        rest = self.rest_[unit_index]
        mats = self._home_shift_mats(int(gather_home)) if self.enables.offset else np.eye(self.k)[None, :, :]
        deltas = self.delta_grid if self.enables.offset else np.array([0.0])
        best = {
            "score": np.inf,
            "d_obs": np.inf,
            "d_dyn": 0.0,
            "alpha": 0.0,
            "delta": 0.0,
            "rho": 0.0,
            "h": np.zeros(self.enables.n_h, dtype=np.float64),
            "r2": 0.0,
        }
        mu_p = self.mu_pos_[unit_index]
        mu_n = self.mu_neg_[unit_index]
        b = self.basis_ if self.basis_ is not None and self.enables.n_h > 0 else np.zeros((2 * self.k, 0))
        if b.shape[1] > self.enables.n_h:
            b = b[:, : self.enables.n_h]
        for di, delta in enumerate(deltas):
            a_mat = mats[di]
            mp = a_mat @ mu_p
            mn = a_mat @ mu_n
            ap, an = closed_form_scales(
                pos, neg, mp, mn, live,
                gain=self.enables.gain,
                polarity=self.enables.polarity,
            )
            rp = ap * mp
            rn = an * mn
            recon = stack_posneg(rp, rn)
            hh = np.zeros(b.shape[1], dtype=np.float64)
            if b.shape[1]:
                hh = residual_ls(x - recon, b, live2)
                recon = recon + b @ hh
            d_obs = mse_live(x, recon, live2)
            alpha = 0.5 * (np.log(ap + EPS) + np.log(an + EPS))
            rho = np.log((ap + EPS) / (an + EPS))
            d_dyn = dyn_penalty(alpha, float(delta), rho, hh, rest)
            score = d_obs + self.lam_state * d_dyn
            if score < best["score"]:
                best = {
                    "score": float(score),
                    "d_obs": float(d_obs),
                    "d_dyn": float(d_dyn),
                    "alpha": float(alpha),
                    "delta": float(delta),
                    "rho": float(rho),
                    "h": hh.copy(),
                    "r2": explained_r2(x, recon, live2),
                }
        return best

    def scores_for_event(
        self,
        vmax_row: np.ndarray,
        vmin_row: np.ndarray,
        *,
        frame: str = "candidate_home",
        event_home: int | None = None,
    ) -> dict[str, np.ndarray]:
        if self.units_ is None or self.homes_ is None:
            raise RuntimeError("not fitted")
        n_u = int(self.units_.size)
        out_score = np.full(n_u, np.inf)
        out_r2 = np.full(n_u, np.nan)
        out_delta = np.zeros(n_u)
        out_alpha = np.zeros(n_u)
        out_rho = np.zeros(n_u)
        for i, home in enumerate(self.homes_):
            if frame == "event_argmax":
                if event_home is None:
                    raise ValueError("event_home required")
                gather_home = int(event_home)
            else:
                gather_home = int(home)
            if gather_home < 0:
                continue
            pos, neg, _ = gather_posneg(vmax_row, vmin_row, self.table[gather_home])
            rec = self._score_unit(pos, neg, i, gather_home)
            out_score[i] = rec["score"]
            out_r2[i] = rec["r2"]
            out_delta[i] = rec["delta"]
            out_alpha[i] = rec["alpha"]
            out_rho[i] = rec["rho"]
        return {
            "score": out_score,
            "r2": out_r2,
            "delta": out_delta,
            "alpha": out_alpha,
            "rho": out_rho,
        }

    def score_matrix(
        self,
        vmax: np.ndarray,
        vmin: np.ndarray,
        event_homes: np.ndarray,
        *,
        frame: str = "candidate_home",
    ) -> dict[str, np.ndarray]:
        """Score every event against every unit. Lower is better."""
        n = int(np.asarray(vmax).shape[0])
        scores = np.full((n, int(self.units_.size)), np.inf)
        r2 = np.full((n, int(self.units_.size)), np.nan)
        delta = np.zeros((n, int(self.units_.size)))
        alpha = np.zeros((n, int(self.units_.size)))
        rho = np.zeros((n, int(self.units_.size)))
        eh = np.asarray(event_homes, dtype=np.int64).ravel()
        if frame != "candidate_home":
            for e in range(n):
                rec = self.scores_for_event(
                    vmax[e], vmin[e], frame=frame, event_home=int(eh[e])
                )
                scores[e] = rec["score"]
                r2[e] = rec["r2"]
                delta[e] = rec["delta"]
                alpha[e] = rec["alpha"]
                rho[e] = rec["rho"]
            return {"score": scores, "r2": r2, "delta": delta, "alpha": alpha, "rho": rho}
        assert self.mu_pos_ is not None and self.homes_ is not None
        n_u = int(self.units_.size)
        n_d = int(self.delta_grid.size)
        x_pos = np.zeros((n, n_u, self.k), dtype=np.float64)
        x_neg = np.zeros((n, n_u, self.k), dtype=np.float64)
        live = np.zeros((n_u, self.k), dtype=bool)
        sh_p = np.zeros((n_u, n_d, self.k), dtype=np.float64)
        sh_n = np.zeros((n_u, n_d, self.k), dtype=np.float64)
        for i, home in enumerate(self.homes_):
            h = int(home)
            if h < 0:
                continue
            live[i] = self.table[h] >= 0
            p, q, _ = gather_posneg_batch(vmax, vmin, self.table[h])
            x_pos[:, i] = p
            x_neg[:, i] = q
            mats = self._home_shift_mats(h)
            for di in range(n_d):
                sh_p[i, di] = mats[di] @ self.mu_pos_[i]
                sh_n[i, di] = mats[di] @ self.mu_neg_[i]
        if self.enables.gain and self.enables.polarity:
            num_p = np.einsum("euk,udk,uk->eud", x_pos, sh_p, live)
            den_p = np.einsum("udk,udk,uk->ud", sh_p, sh_p, live) + EPS
            num_n = np.einsum("euk,udk,uk->eud", x_neg, sh_n, live)
            den_n = np.einsum("udk,udk,uk->ud", sh_n, sh_n, live) + EPS
            ap = np.maximum(num_p / den_p[None], 0.0)
            an = np.maximum(num_n / den_n[None], 0.0)
        elif self.enables.gain:
            num = np.einsum("euk,udk,uk->eud", x_pos, sh_p, live) + np.einsum(
                "euk,udk,uk->eud", x_neg, sh_n, live
            )
            den = (
                np.einsum("udk,udk,uk->ud", sh_p, sh_p, live)
                + np.einsum("udk,udk,uk->ud", sh_n, sh_n, live)
                + EPS
            )
            a = np.maximum(num / den[None], 0.0)
            ap = an = a
        else:
            ap = np.ones((n, n_u, n_d))
            an = np.ones((n, n_u, n_d))
        recon_p = ap[:, :, :, None] * sh_p[None]
        recon_n = an[:, :, :, None] * sh_n[None]
        x_all = np.concatenate([x_pos, x_neg], axis=2)
        recon = np.concatenate([recon_p, recon_n], axis=3)
        live2 = np.concatenate([live, live], axis=1)
        if self.enables.n_h > 0 and self.basis_ is not None and self.basis_.shape[1]:
            b = self.basis_[:, : self.enables.n_h]
            resid = x_all[:, :, None, :] - recon
            for e in range(n):
                for i in range(n_u):
                    lp2 = live2[i]
                    bb_i = b[lp2]
                    gram = bb_i.T @ bb_i + 1e-6 * np.eye(bb_i.shape[1])
                    try:
                        inv = np.linalg.inv(gram)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(gram)
                    for di in range(n_d):
                        hh = inv @ (bb_i.T @ resid[e, i, di, lp2])
                        recon[e, i, di, lp2] = recon[e, i, di, lp2] + (b @ hh)[lp2]
        err = recon - x_all[:, :, None, :]
        err *= live2[None, :, None, :]
        n_live = np.maximum(live2.sum(axis=1), 1)[None, :, None]
        d_obs = np.sum(err**2, axis=3) / n_live
        alpha = 0.5 * (np.log(ap + EPS) + np.log(an + EPS))
        rho_v = np.log((ap + EPS) / (an + EPS))
        d_dyn = np.zeros_like(d_obs)
        for i, rest in enumerate(self.rest_ or []):
            d_dyn[:, i, :] = (
                ((alpha[:, i, :] - rest.alpha) / max(rest.sig_alpha, EPS)) ** 2
                + ((self.delta_grid[None, :] - rest.delta) / max(rest.sig_delta, EPS)) ** 2
                + ((rho_v[:, i, :] - rest.rho) / max(rest.sig_rho, EPS)) ** 2
            )
        total = d_obs + self.lam_state * d_dyn
        pick = np.argmin(total, axis=2)
        eu = np.arange(n)[:, None]
        uu = np.arange(n_u)[None, :]
        scores[:, :] = total[eu, uu, pick]
        delta[:, :] = self.delta_grid[pick]
        alpha_out = alpha[eu, uu, pick]
        rho_out = rho_v[eu, uu, pick]
        r2_out = np.full((n, n_u), np.nan)
        for i in range(n_u):
            lp2 = live2[i]
            y = x_all[:, i][:, lp2]
            yhat = recon[np.arange(n), i, pick[:, i]][:, lp2]
            tot = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
            res = np.sum((y - yhat) ** 2, axis=1)
            r2_out[:, i] = np.where(tot > EPS, 1.0 - res / tot, np.where(res <= EPS, 1.0, 0.0))
        return {
            "score": scores,
            "r2": r2_out,
            "delta": delta,
            "alpha": alpha_out,
            "rho": rho_out,
        }


def spatial_observables(
    pos: np.ndarray,
    neg: np.ndarray,
    rel_xy: np.ndarray,
    live: np.ndarray,
    axis: int,
    eps: float = EPS,
) -> dict[str, float]:
    """G2–G4 scalars on one event in a home frame."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    xy = np.asarray(rel_xy, dtype=np.float64)
    lp = np.asarray(live, dtype=bool)
    p = np.where(lp, p, 0.0)
    n = np.where(lp, n, 0.0)

    def _moments(mass: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        w = mass[lp]
        pts = xy[lp]
        s = float(np.sum(w))
        if s <= eps or pts.size == 0:
            z = np.zeros(2)
            return 0.0, 0.0, 0.0, 0.0, z
        c = (pts * w[:, None]).sum(axis=0) / s
        along = pts[:, int(axis)] - c[int(axis)]
        var = float(np.sum(w * along**2) / s)
        spread = float(np.sqrt(max(var, 0.0)))
        skew = float(np.sum(w * along**3) / s / (spread**3 + eps))
        sharp = float(np.max(mass) / (s + eps))
        return s, spread, skew, sharp, c

    sp, sig_p, g_p, sh_p, cp = _moments(p)
    sn, sig_n, g_n, sh_n, cn = _moments(n)
    tot = p + n
    _, sig, g, sh, c = _moments(tot)
    return {
        "alpha": float(np.log(sp + sn + eps)),
        "rho": float(np.log((sp + eps) / (sn + eps))),
        "c_primary": float(c[int(axis)]),
        "c_pos": float(cp[int(axis)]),
        "c_neg": float(cn[int(axis)]),
        "dc": float(cp[int(axis)] - cn[int(axis)]),
        "spread_pos": sig_p,
        "spread_neg": sig_n,
        "spread": sig,
        "skew_pos": g_p,
        "skew_neg": g_n,
        "sharp_pos": sh_p,
        "sharp_neg": sh_n,
        "sharp": sh,
    }


def score_margins(
    scores: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Lower score is better. M_home = D_nearest_wrong_same_home - D_true."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    ids = np.asarray(units)
    homes = np.asarray(unit_homes)
    row = {_scalar(u): i for i, u in enumerate(ids)}
    n = int(y.size)
    m_all = np.full(n, np.nan)
    m_home = np.full(n, np.nan)
    have = np.zeros(n, dtype=bool)
    pred = np.full(n, -1, dtype=object)
    for i, lab in enumerate(y):
        pred[i] = ids[int(np.argmin(s[i]))]
        ti = row.get(_scalar(lab))
        if ti is None:
            continue
        true = float(s[i, ti])
        others = s[i].copy()
        others[ti] = np.inf
        m_all[i] = float(np.min(others)) - true
        rival = (homes == homes[ti]) & (np.arange(ids.size) != ti)
        if np.any(rival):
            have[i] = True
            m_home[i] = float(np.min(s[i, rival])) - true
        else:
            m_home[i] = np.inf
    return {"m_all": m_all, "m_home": m_home, "have_home_rival": have, "pred": pred}


def summarize_margins(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "p_positive": float("nan"), "n": 0}
    return {
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p_positive": float(np.mean(x > 0)),
        "n": int(x.size),
    }
