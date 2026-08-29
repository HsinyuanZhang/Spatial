r"""Relative-offset POSNEG descriptor and regularized QDA.

Observation is analog q+/q- on a fixed physical slot frame. Scale is
shared L1. Classifiers are float; there is no bit-width or CAM path.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    com_features,
    primary_axis,
    relative_offset_neighbor_table,
)


ETA = 1e-6
EPS = 1e-12
LAMBDA_GRID = (0.25, 0.5, 0.75)
HALF_WIDTH = 3
K_NEIGHBORS = 2 * HALF_WIDTH + 1
MIN_COUNT_ABOVE_DIM = 5


def _as_labels(labels: Sequence[Hashable] | np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError("labels must be 1-D and match the feature rows")
    return array


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def posneg_amplitudes(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q+, q-, and an absent-electrode mask."""
    pos = np.maximum(np.asarray(vmax, dtype=np.float64), 0.0)
    neg = np.maximum(-np.asarray(vmin, dtype=np.float64), 0.0)
    nbr = np.asarray(neighbor_ids, dtype=np.int64)
    if pos.shape != neg.shape or pos.shape != nbr.shape or pos.ndim != 2:
        raise ValueError("vmax, vmin, and neighbor_ids must share shape (n, K)")
    absent = nbr < 0
    pos = np.where(absent, 0.0, pos)
    neg = np.where(absent, 0.0, neg)
    return pos, neg, absent


def polarity_profile(
    pos: np.ndarray,
    neg: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Per-slot polarity rho = (q+ - q-) / (q+ + q- + eps)."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    return (p - n) / (p + n + float(eps))


def stack_posneg(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)], axis=1)


def scale_l1(x: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    amp = np.asarray(x, dtype=np.float64)
    mass = np.sum(np.abs(amp), axis=1)
    u = amp / (mass[:, None] + float(eps))
    return u, mass


def scale_linf(x: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    amp = np.asarray(x, dtype=np.float64)
    mass = np.max(np.abs(amp), axis=1)
    u = amp / (mass[:, None] + float(eps))
    return u, mass


def com_from_posneg(
    pos: np.ndarray,
    neg: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    *,
    one_d: bool,
) -> np.ndarray:
    """COM of POSNEG mass. Linear probes keep the primary-axis coordinate only."""
    weights = np.asarray(pos, dtype=np.float64) + np.asarray(neg, dtype=np.float64)
    com = com_features(weights, neighbor_ids, geom)
    if one_d:
        ax = primary_axis(geom)
        return com[:, ax : ax + 1]
    return com


def com_xy_um(
    weights: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Un-normalized COM in probe micrometres (D1 polarity centroid)."""
    amp = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    nbr = np.asarray(neighbor_ids, dtype=np.int64)
    g = np.asarray(geom, dtype=np.float64)
    valid = nbr >= 0
    amp = np.where(valid, amp, 0.0)
    mass = np.sum(amp, axis=1, keepdims=True)
    w = amp / np.maximum(mass, float(eps))
    safe = np.where(valid, nbr, 0)
    xy = np.where(valid[:, :, None], g[safe], 0.0)
    return np.sum(w[:, :, None] * xy, axis=1)


def make_descriptor(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    *,
    scale: str = "l1",
    one_d: bool | None = None,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Build x, u, log m, COM, and z from analog POSNEG slots."""
    if scale not in ("l1", "inf"):
        raise ValueError(f"scale must be 'l1' or 'inf', got {scale!r}")
    pos, neg, absent = posneg_amplitudes(vmax, vmin, neighbor_ids)
    x = stack_posneg(pos, neg)
    u, mass = scale_l1(x, eps=eps) if scale == "l1" else scale_linf(x, eps=eps)
    if one_d is None:
        from Spatial.algorithms.spatial_footprint import is_linear_probe

        one_d = is_linear_probe(geom)
    com = com_from_posneg(pos, neg, neighbor_ids, geom, one_d=bool(one_d))
    log_m = np.log(mass + float(eps))[:, None]
    z = np.column_stack([u, log_m, com])
    return {
        "x": x,
        "u": u,
        "m": mass,
        "log_m": log_m,
        "com": com,
        "z": z,
        "absent": absent,
        "pos": pos,
        "neg": neg,
        "rho": polarity_profile(pos, neg, eps=eps),
        "one_d": np.array(bool(one_d)),
    }


class FitStandardizer:
    """Column z-score whose mean/std are frozen on fit."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, z: np.ndarray) -> "FitStandardizer":
        values = np.asarray(z, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("z must be a nonempty 2-D array")
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        self.std_ = np.maximum(std, 1e-8)
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("standardizer is not fitted")
        values = np.asarray(z, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0]:
            raise ValueError("z does not match the fitted dimension")
        return (values - self.mean_) / self.std_


def pooled_covariance(z: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Pooled within-class covariance on fit rows."""
    values = np.asarray(z, dtype=np.float64)
    labs = _as_labels(labels, values.shape[0])
    d = int(values.shape[1])
    num = np.zeros((d, d), dtype=np.float64)
    den = 0.0
    for unit in units:
        block = values[labs == unit]
        if block.shape[0] < 2:
            continue
        cov = np.cov(block, rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        num += (block.shape[0] - 1) * cov
        den += float(block.shape[0] - 1)
    if den <= 0:
        return np.eye(d, dtype=np.float64)
    return num / den


def _safe_cov(block: np.ndarray, d: int) -> np.ndarray:
    if block.shape[0] < 2:
        return np.zeros((d, d), dtype=np.float64)
    cov = np.cov(block, rowvar=False, ddof=1)
    if cov.ndim == 0:
        return np.array([[float(cov)]], dtype=np.float64)
    return np.asarray(cov, dtype=np.float64)


def _rank_deficient(cov: np.ndarray) -> bool:
    sign, logdet = np.linalg.slogdet(cov)
    return (not np.isfinite(logdet)) or sign <= 0


def shrink_covariance(
    cov_k: np.ndarray,
    cov_pool: np.ndarray,
    lam: float,
    eta: float = ETA,
    *,
    diagonal: bool = False,
) -> np.ndarray:
    sigma_k = np.asarray(cov_k, dtype=np.float64).copy()
    sigma_p = np.asarray(cov_pool, dtype=np.float64).copy()
    if diagonal:
        sigma_k = np.diag(np.diag(sigma_k))
        sigma_p = np.diag(np.diag(sigma_p))
    d = sigma_p.shape[0]
    return (1.0 - float(lam)) * sigma_k + float(lam) * sigma_p + float(eta) * np.eye(d)


class RegularizedQDA:
    """Equal-prior QDA with pooled shrinkage. Score includes log|Sigma|."""

    def __init__(
        self,
        lam: float,
        *,
        eta: float = ETA,
        diagonal: bool = False,
        include_logdet: bool = True,
    ) -> None:
        if lam < 0.0 or lam > 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = float(lam)
        self.eta = float(eta)
        self.diagonal = bool(diagonal)
        self.include_logdet = bool(include_logdet)
        self.units_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None
        self.logdet_: np.ndarray | None = None
        self.n_fit_: np.ndarray | None = None
        self.forced_pooled_: np.ndarray | None = None
        self.cov_pool_: np.ndarray | None = None

    def fit(self, z: np.ndarray, labels: Sequence[Hashable] | np.ndarray) -> "RegularizedQDA":
        values = np.asarray(z, dtype=np.float64)
        labs = _as_labels(labels, values.shape[0])
        units = np.array([_scalar(u) for u in np.unique(labs)])
        d = int(values.shape[1])
        n_u = int(units.size)
        means = np.zeros((n_u, d), dtype=np.float64)
        counts = np.zeros(n_u, dtype=np.int64)
        raw = np.zeros((n_u, d, d), dtype=np.float64)
        pool = pooled_covariance(values, labs, units)
        min_count = d + MIN_COUNT_ABOVE_DIM
        forced = np.zeros(n_u, dtype=bool)
        covs = np.zeros((n_u, d, d), dtype=np.float64)
        precs = np.zeros((n_u, d, d), dtype=np.float64)
        logdets = np.zeros(n_u, dtype=np.float64)
        for i, unit in enumerate(units):
            block = values[labs == unit]
            counts[i] = int(block.shape[0])
            means[i] = block.mean(axis=0) if block.size else 0.0
            raw[i] = _safe_cov(block, d)
            need_pool = counts[i] < min_count or _rank_deficient(raw[i] + self.eta * np.eye(d))
            forced[i] = bool(need_pool)
            lam_i = 1.0 if need_pool else self.lam
            covs[i] = shrink_covariance(raw[i], pool, lam_i, self.eta, diagonal=self.diagonal)
            sign, logdet = np.linalg.slogdet(covs[i])
            if sign <= 0 or not np.isfinite(logdet):
                covs[i] = pool + self.eta * np.eye(d)
                if self.diagonal:
                    covs[i] = np.diag(np.diag(covs[i])) + self.eta * np.eye(d)
                sign, logdet = np.linalg.slogdet(covs[i])
                forced[i] = True
            logdets[i] = float(logdet)
            precs[i] = np.linalg.pinv(covs[i])
        self.units_ = units
        self.mean_ = means
        self.cov_ = covs
        self.prec_ = precs
        self.logdet_ = logdets
        self.n_fit_ = counts
        self.forced_pooled_ = forced
        self.cov_pool_ = pool
        return self

    def scores(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.prec_ is None or self.logdet_ is None:
            raise RuntimeError("QDA is not fitted")
        values = np.asarray(z, dtype=np.float64)
        diff = values[:, None, :] - self.mean_[None, :, :]
        mahal = np.einsum("nui,uij,nuj->nu", diff, self.prec_, diff)
        s = -0.5 * mahal
        if self.include_logdet:
            s = s - 0.5 * self.logdet_[None, :]
        return s

    def predict(self, z: np.ndarray) -> np.ndarray:
        s = self.scores(z)
        idx = np.argmax(s, axis=1)
        return self.units_[idx]


def mean_distance_scores(
    z: np.ndarray,
    means: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Return **scores** (negative distance) so argmax is nearest mean."""
    values = np.asarray(z, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    if metric == "l1":
        dist = np.abs(values[:, None, :] - mu[None, :, :]).sum(axis=2)
    elif metric == "l2":
        dist = np.sqrt(((values[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2))
    else:
        raise ValueError(f"metric must be l1 or l2, got {metric!r}")
    return -dist


def fit_means(z: np.ndarray, labels: Sequence[Hashable] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(z, dtype=np.float64)
    labs = _as_labels(labels, values.shape[0])
    units = np.array([_scalar(u) for u in np.unique(labs)])
    means = np.stack([values[labs == u].mean(axis=0) for u in units], axis=0)
    return units, means


def score_margins(
    scores: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
) -> dict[str, np.ndarray]:
    """All-rival and same-home score margins. No co-home rival -> +inf."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    ids = np.asarray(units)
    homes = np.asarray(unit_homes)
    row = {_scalar(u): i for i, u in enumerate(ids)}
    n = int(y.size)
    m_all = np.full(n, np.nan, dtype=np.float64)
    m_home = np.full(n, np.nan, dtype=np.float64)
    have_home_rival = np.zeros(n, dtype=bool)
    for i, lab in enumerate(y):
        ti = row.get(_scalar(lab))
        if ti is None:
            continue
        true = float(s[i, ti])
        others = s[i].copy()
        others[ti] = -np.inf
        m_all[i] = true - float(np.max(others))
        home = homes[ti]
        rival = (homes == home) & (np.arange(ids.size) != ti)
        if np.any(rival):
            have_home_rival[i] = True
            m_home[i] = true - float(np.max(s[i, rival]))
        else:
            m_home[i] = np.inf
    return {"m_all": m_all, "m_home": m_home, "have_home_rival": have_home_rival}


def summarize_margins(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p10": float("nan"), "p50": float("nan"), "p_positive": float("nan"), "n": 0}
    return {
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p_positive": float(np.mean(x > 0)),
        "n": int(x.size),
    }


def majority_home(homes: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    h = np.asarray(homes, dtype=np.int64).ravel()
    y = np.asarray(labels)
    out = np.full(units.size, -1, dtype=np.int64)
    n_ch = int(max(int(h.max()) + 1, 1)) if h.size else 1
    for i, unit in enumerate(units):
        ch = h[y == unit]
        if ch.size == 0:
            continue
        out[i] = int(np.bincount(ch, minlength=n_ch).argmax())
    return out


def error_taxonomy(
    pred: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
    geom: np.ndarray,
) -> dict[str, float]:
    """Fraction of errors on same home / neighbouring home / far."""
    from Spatial.algorithms.spatial_footprint import median_nearest_neighbor_spacing

    y = np.asarray(labels)
    p = np.asarray(pred)
    ids = np.asarray(units)
    homes = np.asarray(unit_homes, dtype=np.int64)
    row = {_scalar(u): i for i, u in enumerate(ids)}
    g = np.asarray(geom, dtype=np.float64)
    pitch = float(median_nearest_neighbor_spacing(g)) if g.shape[0] >= 2 else 1.0
    wrong = p != y
    n_err = int(wrong.sum())
    if n_err == 0:
        return {"frac_same_home": 0.0, "frac_neighbor_home": 0.0, "frac_far": 0.0, "n_err": 0}
    same = 0
    neighbor = 0
    far = 0
    for lab, pr in zip(y[wrong], p[wrong]):
        ti = row.get(_scalar(lab))
        pi = row.get(_scalar(pr))
        if ti is None or pi is None or homes[ti] < 0 or homes[pi] < 0:
            far += 1
            continue
        if homes[ti] == homes[pi]:
            same += 1
            continue
        dist = float(np.linalg.norm(g[homes[ti]] - g[homes[pi]]))
        if dist <= 1.5 * pitch + 1e-8:
            neighbor += 1
        else:
            far += 1
    return {
        "frac_same_home": same / n_err,
        "frac_neighbor_home": neighbor / n_err,
        "frac_far": far / n_err,
        "n_err": n_err,
    }


def pairwise_lda_accuracy(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """Fit-set 2-class LDA accuracy (oracle upper bound, not a held-out test)."""
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("pair blocks must share a feature axis")
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    x = np.vstack([a, b])
    y = np.array([0] * a.shape[0] + [1] * b.shape[0])
    mu0, mu1 = a.mean(axis=0), b.mean(axis=0)
    cov = pooled_covariance(x, y, np.array([0, 1]))
    cov = cov + ETA * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov)
    w = prec @ (mu1 - mu0)
    # Threshold at the midpoint in the projected coordinate.
    t0, t1 = float(w @ mu0), float(w @ mu1)
    thresh = 0.5 * (t0 + t1)
    proj = x @ w
    pred = (proj >= thresh).astype(np.int64)
    if t1 < t0:
        pred = 1 - pred
    return float(np.mean(pred == y))


def d1_cohome_pairs(
    u: np.ndarray,
    rho: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    neighbor_ids: np.ndarray,
    geom: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    homes: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
) -> list[dict[str, Any]]:
    """Per co-home unit pair, shape/polarity/COM-delta plus LDA oracle."""
    labs = np.asarray(labels)
    rows: list[dict[str, Any]] = []
    by_home: dict[int, list[int]] = {}
    for i, h in enumerate(np.asarray(unit_homes, dtype=np.int64)):
        by_home.setdefault(int(h), []).append(i)
    mean_u = {int(u_id): u[labs == u_id].mean(axis=0) for u_id in units if np.any(labs == u_id)}
    mean_rho = {int(u_id): rho[labs == u_id].mean(axis=0) for u_id in units if np.any(labs == u_id)}
    delta = {}
    for u_id in units:
        mask = labs == u_id
        if not np.any(mask):
            continue
        com_pos = com_xy_um(pos[mask], neighbor_ids[mask], geom)
        com_neg = com_xy_um(neg[mask], neighbor_ids[mask], geom)
        delta[int(u_id)] = (com_pos - com_neg).mean(axis=0)
    for home, idxs in sorted(by_home.items()):
        if home < 0 or len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                ua, ub = int(units[ia]), int(units[ib])
                if ua not in mean_u or ub not in mean_u:
                    continue
                ua_vec, ub_vec = mean_u[ua], mean_u[ub]
                cos = float(
                    np.dot(ua_vec, ub_vec)
                    / (np.linalg.norm(ua_vec) * np.linalg.norm(ub_vec) + EPS)
                )
                rows.append(
                    {
                        "home": int(home),
                        "unit_a": ua,
                        "unit_b": ub,
                        "cosine_u": cos,
                        "polarity_l2": float(np.linalg.norm(mean_rho[ua] - mean_rho[ub])),
                        "com_delta_l2_um": float(np.linalg.norm(delta[ua] - delta[ub])),
                        "lda_oracle": pairwise_lda_accuracy(z[labs == ua], z[labs == ub]),
                        "n_a": int(np.sum(labs == ua)),
                        "n_b": int(np.sum(labs == ub)),
                    }
                )
    return rows


def select_lambda_global(
    per_lambda: Sequence[dict[str, float]],
) -> float:
    """One global lambda: max calibration accuracy, then home-margin p10."""
    if not per_lambda:
        raise ValueError("no lambda candidates")
    best = None
    for row in per_lambda:
        key = (-float(row["accuracy"]), -float(row.get("home_margin_p10", float("-inf"))), float(row["lam"]))
        if best is None or key < best[0]:
            best = (key, float(row["lam"]))
    assert best is not None
    return best[1]


__all__ = [
    "ETA",
    "EPS",
    "LAMBDA_GRID",
    "FitStandardizer",
    "RegularizedQDA",
    "com_from_posneg",
    "d1_cohome_pairs",
    "error_taxonomy",
    "fit_means",
    "make_descriptor",
    "majority_home",
    "mean_distance_scores",
    "polarity_profile",
    "pooled_covariance",
    "posneg_amplitudes",
    "relative_offset_neighbor_table",
    "scale_l1",
    "scale_linf",
    "score_margins",
    "select_lambda_global",
    "shrink_covariance",
    "stack_posneg",
    "summarize_margins",
]
