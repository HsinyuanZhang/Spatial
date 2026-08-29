"""Candidate-home POSNEG / P2P features and MVM-compatible scoring.

Default frame is candidate-home geometry-K, never strongest-K. Scoring does
not apply per-event global normalization unless the arm is cosine / A5.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    event_peak_channel,
    extract_all_channel_extrema,
    gather_posneg_batch,
    geometry_k_table,
    majority_home,
    score_margins,
    summarize_margins,
)
from Spatial.algorithms.spatial_footprint import (
    median_nearest_neighbor_spacing,
    primary_axis,
)


EPS = 1e-12
K_GRID = (7, 12, 16)
WINDOW = 15
SEED = 0
LAMBDA_AMP_GRID = (0.1, 0.5, 1.0, 2.0)
POLARITY_RATIOS = ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0))
DURATION_S = 60.0

SCORING_HIGHER = {
    "l1": False,
    "l2": False,
    "weighted_l2": False,
    "biased_dot": True,
    "pure_dot": True,
    "cosine": True,
    "angular_radial": True,
}

OPERATION_FORMS = {
    "l1": {
        "form": "sum_i |x_i - mu_i|",
        "mvm_writable": False,
        "n_mac_per_dim": 0,
        "n_abs_per_dim": 1,
        "n_norm": 0,
    },
    "l2": {
        "form": "||x-mu||^2  (MVM-equivalent to biased_dot when x is shared)",
        "mvm_writable": False,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 0,
    },
    "weighted_l2": {
        "form": "lam+||q+-mu+||^2 + lam-||q--mu-||^2",
        "mvm_writable": False,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 0,
    },
    "biased_dot": {
        "form": "S = M x + b,  M=2 mu,  b_u = -||mu_u||^2",
        "mvm_writable": True,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 0,
    },
    "pure_dot": {
        "form": "S = x^T mu",
        "mvm_writable": True,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 0,
    },
    "cosine": {
        "form": "(x^T mu) / (||x|| ||mu||)",
        "mvm_writable": False,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 2,
    },
    "angular_radial": {
        "form": "cos(x,mu) - lam_amp (log||x|| - log||mu||)^2",
        "mvm_writable": False,
        "n_mac_per_dim": 1,
        "n_abs_per_dim": 0,
        "n_norm": 2,
    },
}


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def rec_key(family: str, dataset_name: str) -> str:
    name = str(dataset_name)
    fam = str(family).lower()
    if fam == "hj" and "drift" in name:
        return "hj_drift"
    if fam == "hj" and "static" in name:
        return "hj_static"
    if "units10" in name:
        return "mearec_10u"
    if "units20" in name:
        return "mearec_20u"
    return f"{fam}_{name}"


def operation_count(kind: str, dim: int) -> dict[str, Any]:
    spec = OPERATION_FORMS[kind]
    d = int(dim)
    return {
        "operation_form": spec["form"],
        "mvm_writable": bool(spec["mvm_writable"]),
        "n_mac": int(spec["n_mac_per_dim"]) * d,
        "n_abs": int(spec["n_abs_per_dim"]) * d,
        "n_norm": int(spec["n_norm"]),
        "feature_dim": d,
    }


def l2_squared(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """||x - mu||^2. x (n, D) or (D,), mu (U, D) or (D,)."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1 and mu.ndim == 1:
        d = x - mu
        return np.array(np.dot(d, d), dtype=np.float64)
    if x.ndim == 1:
        d = mu - x[None, :]
        return np.sum(d * d, axis=1)
    if mu.ndim == 1:
        d = x - mu[None, :]
        return np.sum(d * d, axis=1)
    d = x[:, None, :] - mu[None, :, :]
    return np.sum(d * d, axis=2)


def biased_dot(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """S = 2 x^T mu - ||mu||^2. Does not divide by norms."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu_sq = np.sum(mu * mu, axis=-1)
    if x.ndim == 1 and mu.ndim == 1:
        return np.array(2.0 * np.dot(x, mu) - mu_sq, dtype=np.float64)
    if x.ndim == 1:
        return 2.0 * (mu @ x) - mu_sq
    if mu.ndim == 1:
        return 2.0 * (x @ mu) - mu_sq
    return 2.0 * (x @ mu.T) - mu_sq[None, :]


def biased_dot_mvm(x: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return S, M, b with S = M @ x + b, M = 2 mu, b = -||mu||^2.

    ``x`` is a single shared vector (D,). ``mu`` is (U, D).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2:
        raise ValueError(f"mu must be (U, D), got {mu.shape}")
    m = 2.0 * mu
    b = -np.sum(mu * mu, axis=1)
    s = m @ x + b
    return s, m, b


def pure_dot(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1 and mu.ndim == 1:
        return np.array(np.dot(x, mu), dtype=np.float64)
    if x.ndim == 1:
        return mu @ x
    if mu.ndim == 1:
        return x @ mu
    return x @ mu.T


def cosine_similarity(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Explicit amplitude-removal. Divides by ||x|| ||mu||."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1 and mu.ndim == 1:
        xn = float(np.linalg.norm(x))
        mn = float(np.linalg.norm(mu))
        return np.array(np.dot(x, mu) / max(xn * mn, EPS), dtype=np.float64)
    if x.ndim == 1:
        xn = float(np.linalg.norm(x))
        mn = np.linalg.norm(mu, axis=1)
        return (mu @ x) / np.maximum(xn * mn, EPS)
    if mu.ndim == 1:
        xn = np.linalg.norm(x, axis=1)
        mn = float(np.linalg.norm(mu))
        return (x @ mu) / np.maximum(xn * mn, EPS)
    xn = np.linalg.norm(x, axis=1)
    mn = np.linalg.norm(mu, axis=1)
    return (x @ mu.T) / np.maximum(xn[:, None] * mn[None, :], EPS)


def l1_distance(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1 and mu.ndim == 1:
        return np.array(np.sum(np.abs(x - mu)), dtype=np.float64)
    if x.ndim == 1:
        return np.sum(np.abs(mu - x[None, :]), axis=1)
    if mu.ndim == 1:
        return np.sum(np.abs(x - mu[None, :]), axis=1)
    return np.sum(np.abs(x[:, None, :] - mu[None, :, :]), axis=2)


def angular_radial_score(
    x: np.ndarray,
    mu: np.ndarray,
    lam_amp: float,
) -> np.ndarray:
    cos = cosine_similarity(x, mu)
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1:
        log_x = np.log(max(float(np.linalg.norm(x)), EPS))
    else:
        log_x = np.log(np.maximum(np.linalg.norm(x, axis=1), EPS))
    if mu.ndim == 1:
        log_mu = np.log(max(float(np.linalg.norm(mu)), EPS))
        damp = (log_x - log_mu) ** 2
    else:
        log_mu = np.log(np.maximum(np.linalg.norm(mu, axis=1), EPS))
        if x.ndim == 1:
            damp = (log_x - log_mu) ** 2
        else:
            damp = (log_x[:, None] - log_mu[None, :]) ** 2
    return cos - float(lam_amp) * damp


def weighted_l2(
    x: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim == 1 and mu.ndim == 1:
        d = x - mu
        return np.array(np.dot(w, d * d), dtype=np.float64)
    if x.ndim == 1:
        d = mu - x[None, :]
        return np.sum(w[None, :] * d * d, axis=1)
    if mu.ndim == 1:
        d = x - mu[None, :]
        return np.sum(w[None, :] * d * d, axis=1)
    d = x[:, None, :] - mu[None, :, :]
    return np.sum(w[None, None, :] * d * d, axis=2)


def polarity_weights(k: int, lam_pos: float, lam_neg: float) -> np.ndarray:
    return np.concatenate(
        [
            np.full(int(k), float(lam_pos), dtype=np.float64),
            np.full(int(k), float(lam_neg), dtype=np.float64),
        ]
    )


def score_kind(
    x: np.ndarray,
    mu: np.ndarray,
    kind: str,
    *,
    lam_amp: float = 1.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    kind = str(kind)
    if kind == "l1":
        return l1_distance(x, mu)
    if kind == "l2":
        return l2_squared(x, mu)
    if kind == "weighted_l2":
        if weights is None:
            raise ValueError("weighted_l2 requires weights")
        return weighted_l2(x, mu, weights)
    if kind == "biased_dot":
        return biased_dot(x, mu)
    if kind == "pure_dot":
        return pure_dot(x, mu)
    if kind == "cosine":
        return cosine_similarity(x, mu)
    if kind == "angular_radial":
        return angular_radial_score(x, mu, lam_amp)
    raise ValueError(f"unknown scoring kind {kind!r}")


def as_distance(scores: np.ndarray, kind: str) -> np.ndarray:
    """Lower-is-better distances for margin helpers."""
    if SCORING_HIGHER[str(kind)]:
        if str(kind) == "cosine":
            return 1.0 - np.asarray(scores, dtype=np.float64)
        return -np.asarray(scores, dtype=np.float64)
    return np.asarray(scores, dtype=np.float64)


def assign_from_scores(scores: np.ndarray, units: np.ndarray, kind: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(units)
    if s.ndim == 1:
        idx = int(np.argmax(s) if SCORING_HIGHER[kind] else np.argmin(s))
        return np.array([ids[idx]], dtype=object)
    if SCORING_HIGHER[kind]:
        idx = np.argmax(s, axis=1)
    else:
        idx = np.argmin(s, axis=1)
    return np.asarray([ids[int(i)] for i in idx], dtype=object)


def p2p_from_posneg(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + np.asarray(neg, dtype=np.float64)


def concat_posneg(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)],
        axis=-1,
    )


def polarity_mass_ratio(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    num = np.sum(p, axis=-1) + EPS
    den = np.sum(n, axis=-1) + EPS
    return np.log(num / den)


def polarity_spatial_delta(pos: np.ndarray, neg: np.ndarray, rel_xy: np.ndarray) -> np.ndarray:
    """Δc = c+ - c- in the slot relative frame. (n, 2) or (2,)."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    xy = np.asarray(rel_xy, dtype=np.float64)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[None, :]
        n = n[None, :]
    wp = np.sum(p, axis=1, keepdims=True) + EPS
    wn = np.sum(n, axis=1, keepdims=True) + EPS
    c_pos = (p @ xy) / wp
    c_neg = (n @ xy) / wn
    dc = c_pos - c_neg
    return dc[0] if squeeze else dc


def build_features(
    pos: np.ndarray,
    neg: np.ndarray,
    representation: str,
    rel_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Build a feature matrix from analog q+, q- on the same slots."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    name = str(representation)
    if name == "p2p":
        return p2p_from_posneg(pos, neg)
    if name == "pos":
        return pos
    if name == "neg":
        return neg
    if name in ("posneg", "posneg_weighted"):
        return concat_posneg(pos, neg)
    rpn = polarity_mass_ratio(pos, neg)
    if rel_xy is None:
        raise ValueError(f"{name} requires rel_xy")
    dc = polarity_spatial_delta(pos, neg, rel_xy)
    squeeze = pos.ndim == 1
    if squeeze:
        rpn = np.array([rpn], dtype=np.float64)
        pos = pos[None, :]
        neg = neg[None, :]
        dc = dc[None, :]
    stacked = concat_posneg(pos, neg)
    rpn_col = rpn.reshape(-1, 1)
    if name == "posneg_rpn":
        out = np.concatenate([stacked, rpn_col], axis=1)
    elif name == "posneg_dc":
        out = np.concatenate([stacked, dc], axis=1)
    elif name == "posneg_rpn_dc":
        out = np.concatenate([stacked, rpn_col, dc], axis=1)
    else:
        raise ValueError(f"unknown representation {representation!r}")
    return out[0] if squeeze else out


def fit_candidate_home_means(
    vmax: np.ndarray,
    vmin: np.ndarray,
    labels: np.ndarray,
    event_homes: np.ndarray,
    table: np.ndarray,
    *,
    frame: str = "candidate_home",
) -> dict[str, Any]:
    if frame not in ("candidate_home", "event_argmax"):
        raise ValueError(f"unknown frame {frame!r}")
    y = np.asarray(labels)
    units = np.array([_scalar(u) for u in np.unique(y)], dtype=object)
    homes = majority_home(event_homes, y, units)
    k = int(table.shape[1])
    n_u = int(units.size)
    mu_pos = np.zeros((n_u, k), dtype=np.float64)
    mu_neg = np.zeros((n_u, k), dtype=np.float64)
    eh = np.asarray(event_homes, dtype=np.int64).ravel()
    for i, unit in enumerate(units):
        mask = y == unit
        h = int(homes[i])
        if h < 0:
            continue
        if frame == "candidate_home":
            pos, neg, _ = gather_posneg_batch(vmax[mask], vmin[mask], table[h])
        else:
            idx = np.flatnonzero(mask)
            pos = np.zeros((idx.size, k), dtype=np.float64)
            neg = np.zeros((idx.size, k), dtype=np.float64)
            for j, e in enumerate(idx):
                p, n, _ = gather_posneg_batch(
                    vmax[e : e + 1], vmin[e : e + 1], table[int(eh[e])]
                )
                pos[j] = p[0]
                neg[j] = n[0]
        if pos.shape[0] == 0:
            continue
        mu_pos[i] = pos.mean(axis=0)
        mu_neg[i] = neg.mean(axis=0)
    return {
        "units": units,
        "homes": homes,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "frame": frame,
        "k": k,
    }


def gather_unit_posneg(
    vmax: np.ndarray,
    vmin: np.ndarray,
    table: np.ndarray,
    unit_homes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """POSNEG for every event in every unit's candidate-home slots.

    Returns pos, neg with shape (n_events, n_units, K).
    """
    vmx = np.asarray(vmax, dtype=np.float64)
    vmn = np.asarray(vmin, dtype=np.float64)
    homes = np.asarray(unit_homes, dtype=np.int64).ravel()
    n = int(vmx.shape[0])
    n_u = int(homes.size)
    k = int(table.shape[1])
    pos = np.zeros((n, n_u, k), dtype=np.float64)
    neg = np.zeros((n, n_u, k), dtype=np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ui, h in enumerate(homes):
        hh = int(h)
        if hh < 0:
            continue
        if hh not in cache:
            p, nneg, _ = gather_posneg_batch(vmx, vmn, table[hh])
            cache[hh] = (p, nneg)
        pos[:, ui] = cache[hh][0]
        neg[:, ui] = cache[hh][1]
    return pos, neg


def gather_event_argmax_posneg(
    vmax: np.ndarray,
    vmin: np.ndarray,
    table: np.ndarray,
    event_homes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One POSNEG row per event on that event's peak-channel slots."""
    vmx = np.asarray(vmax, dtype=np.float64)
    vmn = np.asarray(vmin, dtype=np.float64)
    eh = np.asarray(event_homes, dtype=np.int64).ravel()
    n = int(vmx.shape[0])
    k = int(table.shape[1])
    pos = np.zeros((n, k), dtype=np.float64)
    neg = np.zeros((n, k), dtype=np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h in np.unique(eh):
        hh = int(h)
        if hh < 0:
            continue
        cache[hh] = gather_posneg_batch(vmx, vmn, table[hh])[:2]
    for i, h in enumerate(eh):
        hh = int(h)
        if hh not in cache:
            continue
        pos[i] = cache[hh][0][i]
        neg[i] = cache[hh][1][i]
    return pos, neg


def score_candidate_home(
    pos_eu: np.ndarray,
    neg_eu: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    rel_xy: np.ndarray,
    unit_homes: np.ndarray,
    representation: str,
    kind: str,
    *,
    lam_amp: float = 1.0,
    lam_pos: float = 1.0,
    lam_neg: float = 1.0,
) -> np.ndarray:
    """Score every event against every unit. pos_eu is (n, U, K)."""
    pos_eu = np.asarray(pos_eu, dtype=np.float64)
    neg_eu = np.asarray(neg_eu, dtype=np.float64)
    n, n_u, k = pos_eu.shape
    out = np.empty((n, n_u), dtype=np.float64)
    weights = polarity_weights(k, lam_pos, lam_neg) if kind == "weighted_l2" else None
    homes = np.asarray(unit_homes, dtype=np.int64).ravel()
    for ui in range(n_u):
        xy = rel_xy[int(homes[ui])] if rel_xy.ndim == 3 else rel_xy
        x = build_features(pos_eu[:, ui], neg_eu[:, ui], representation, xy)
        mu = build_features(mu_pos[ui], mu_neg[ui], representation, xy)
        out[:, ui] = score_kind(x, mu, kind, lam_amp=lam_amp, weights=weights)
    return out


def score_shared_x(
    x: np.ndarray,
    mu: np.ndarray,
    kind: str,
    *,
    lam_amp: float = 1.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """(n, D) events vs (U, D) templates, shared feature frame."""
    return score_kind(x, mu, kind, lam_amp=lam_amp, weights=weights)


def assignment_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    dist = as_distance(scores, kind)
    margins = score_margins(dist, labels, units, unit_homes)
    pred = margins["pred"]
    y = np.asarray(labels)
    correct = np.array([_scalar(p) == _scalar(t) for p, t in zip(pred, y)], dtype=bool)
    same = margins["have_home_rival"]
    home_vals = margins["m_home"][same]
    sm = summarize_margins(home_vals)
    sall = summarize_margins(margins["m_all"])
    per_unit: dict[str, float] = {}
    for unit in units:
        mask = np.array([_scalar(t) == _scalar(unit) for t in y], dtype=bool)
        if not np.any(mask):
            continue
        per_unit[str(_scalar(unit))] = float(np.mean(correct[mask]))
    if np.any(same):
        same_acc = float(
            np.mean(
                [
                    _scalar(p) == _scalar(t)
                    for p, t in zip(pred[same], y[same])
                ]
            )
        )
    else:
        same_acc = float("nan")
    return {
        "pred": pred,
        "correct": correct,
        "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
        "same_home_accuracy": same_acc,
        "n_test": int(y.size),
        "n_correct": int(correct.sum()),
        "n_home_rival": int(same.sum()),
        "worst_unit_recall": (
            float(min(per_unit.values())) if per_unit else float("nan")
        ),
        "mean_unit_recall": (
            float(np.mean(list(per_unit.values()))) if per_unit else float("nan")
        ),
        "margin_home_p10": sm["p10"],
        "margin_home_p50": sm["p50"],
        "margin_home_p90": sm["p90"],
        "margin_home_ppos": sm["p_positive"],
        "margin_all_p10": sall["p10"],
        "margin_all_p50": sall["p50"],
        "margin_all_p90": sall["p90"],
        "margin_all_ppos": sall["p_positive"],
        "per_unit_recall": per_unit,
        "m_home": margins["m_home"],
        "m_all": margins["m_all"],
        "have_home_rival": same,
        "distances": dist,
    }


def same_home_pairs(units: np.ndarray, homes: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    h = np.asarray(homes)
    for i in range(int(units.size)):
        for j in range(i + 1, int(units.size)):
            if h[i] == h[j]:
                pairs.append((i, j))
    return pairs


def template_pair_distances(
    mu_a: np.ndarray,
    mu_b: np.ndarray,
) -> float:
    d = np.asarray(mu_a, dtype=np.float64) - np.asarray(mu_b, dtype=np.float64)
    return float(np.dot(d, d))


def probe_meta(geom: np.ndarray) -> dict[str, Any]:
    g = np.asarray(geom, dtype=np.float64)
    pitch = (
        float(median_nearest_neighbor_spacing(g)) if g.shape[0] >= 2 else 1.0
    )
    axis = int(primary_axis(g))
    return {"pitch": pitch, "axis": axis, "n_ch": int(g.shape[0])}


def geometry_k_is_physical(table: np.ndarray, geom: np.ndarray, home: int, k: int) -> bool:
    """True iff the home row is the K geometrically nearest sites, not amplitude-ranked."""
    g = np.asarray(geom, dtype=np.float64)
    dist = np.linalg.norm(g - g[int(home)], axis=1)
    nearest = set(np.argsort(dist, kind="stable")[: int(k)].tolist())
    row = [int(c) for c in table[int(home)] if int(c) >= 0]
    return set(row) == nearest
