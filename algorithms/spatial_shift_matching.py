"""Spatial shift-bank / moving-template matching (FREE-SHIFT DIAGNOSTIC).

This module is an upper bound: each candidate may pick its own shift on the
current event. That is not a tracker. Biased-dot moving correlation does not
divide by ||x|| ||mu||.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    interpolation_matrix,
    shift_query_xy,
)
from Spatial.algorithms.posneg_mvm_matching import (
    EPS,
    SCORING_HIGHER,
    as_distance,
    biased_dot,
    build_features,
    cosine_similarity,
    score_kind,
)


SHIFT_DISCRETE = (-1.0, 0.0, 1.0)
SHIFT_DISCRETE_WIDE = (-2.0, -1.0, 0.0, 1.0, 2.0)
SHIFT_FRACTIONAL = (-0.5, -0.25, 0.0, 0.25, 0.5)


def shift_matrix(
    rel_xy: np.ndarray,
    live: np.ndarray,
    delta_pitch: float,
    pitch_um: float,
    axis: int,
) -> np.ndarray:
    xy = np.asarray(rel_xy, dtype=np.float64)
    return interpolation_matrix(
        xy,
        shift_query_xy(xy, float(delta_pitch), float(pitch_um), int(axis)),
        np.asarray(live, dtype=bool),
    )


def precompute_shift_mats(
    rel_xy: np.ndarray,
    table: np.ndarray,
    deltas: Sequence[float],
    pitch_um: float,
    axis: int,
) -> dict[int, np.ndarray]:
    """home -> (n_delta, K, K) interpolation matrices."""
    out: dict[int, np.ndarray] = {}
    n_h = int(table.shape[0])
    for h in range(n_h):
        live = table[h] >= 0
        mats = np.stack(
            [
                shift_matrix(rel_xy[h], live, d, pitch_um, axis)
                for d in deltas
            ],
            axis=0,
        )
        out[h] = mats
    return out


def apply_shift_mat(mat: np.ndarray, values: np.ndarray) -> np.ndarray:
    a = np.asarray(mat, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        return a @ v
    return v @ a.T


def shift_posneg(
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_shift_mat(mat, mu_pos), apply_shift_mat(mat, mu_neg)


def score_shift_bank(
    pos: np.ndarray,
    neg: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    mats: np.ndarray,
    representation: str,
    kind: str,
    rel_xy: np.ndarray,
    *,
    lam_amp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Best score over shifts and the chosen delta index.

    pos/neg: (n, K) event features in the *unshifted* candidate-home frame.
    mu: (K,) rest template. mats: (n_delta, K, K).
    Returns scores_best (n,), idx_best (n,).
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    squeeze = pos.ndim == 1
    if squeeze:
        pos = pos[None, :]
        neg = neg[None, :]
    n = int(pos.shape[0])
    n_s = int(mats.shape[0])
    bank = np.empty((n, n_s), dtype=np.float64)
    x = build_features(pos, neg, representation, rel_xy)
    for s in range(n_s):
        sp, sn = shift_posneg(mu_pos, mu_neg, mats[s])
        mu = build_features(sp, sn, representation, rel_xy)
        bank[:, s] = score_kind(x, mu, kind, lam_amp=lam_amp)
    if SCORING_HIGHER[kind]:
        idx = np.argmax(bank, axis=1)
    else:
        idx = np.argmin(bank, axis=1)
    best = bank[np.arange(n), idx]
    if squeeze:
        return best[0], idx[0]
    return best, idx


def moving_scores_all_units(
    pos_eu: np.ndarray,
    neg_eu: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    unit_homes: np.ndarray,
    rel_xy: np.ndarray,
    mats_by_home: dict[int, np.ndarray],
    representation: str,
    kind: str,
) -> dict[str, np.ndarray]:
    """Free per-unit shift: each candidate picks its own best s.

    DIAGNOSTIC ONLY.
    """
    pos_eu = np.asarray(pos_eu, dtype=np.float64)
    n, n_u, _k = pos_eu.shape
    homes = np.asarray(unit_homes, dtype=np.int64).ravel()
    best = np.empty((n, n_u), dtype=np.float64)
    idx = np.empty((n, n_u), dtype=np.int64)
    for ui in range(n_u):
        h = int(homes[ui])
        mats = mats_by_home[h]
        xy = rel_xy[h]
        b, i = score_shift_bank(
            pos_eu[:, ui],
            neg_eu[:, ui],
            mu_pos[ui],
            mu_neg[ui],
            mats,
            representation,
            kind,
            xy,
        )
        best[:, ui] = b
        idx[:, ui] = i
    return {"score": best, "shift_index": idx}


def fixed_vs_shifted_true_wrong(
    pos_eu: np.ndarray,
    neg_eu: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
    rel_xy: np.ndarray,
    mats_by_home: dict[int, np.ndarray],
    representation: str,
    kind: str,
    fixed_distances: np.ndarray,
) -> dict[str, np.ndarray]:
    """I_true, I_wrong, L_flex. Nearest wrong same-home is from *fixed* scores."""
    from Spatial.algorithms.posneg_mvm_matching import _scalar, same_home_pairs

    moved = moving_scores_all_units(
        pos_eu,
        neg_eu,
        mu_pos,
        mu_neg,
        unit_homes,
        rel_xy,
        mats_by_home,
        representation,
        kind,
    )
    # Convert both to a common "higher is better" improvement scale.
    def as_sim_from_native(s: np.ndarray) -> np.ndarray:
        if SCORING_HIGHER[kind]:
            return np.asarray(s, dtype=np.float64)
        return -np.asarray(s, dtype=np.float64)

    sim_shift = as_sim_from_native(moved["score"])
    # fixed_distances is lower-better (as_distance). Invert that map.
    fd = np.asarray(fixed_distances, dtype=np.float64)
    if kind == "cosine":
        sim_fix = 1.0 - fd
    else:
        sim_fix = -fd

    y = np.asarray(labels)
    ids = np.asarray(units)
    homes = np.asarray(unit_homes)
    row = {_scalar(u): i for i, u in enumerate(ids)}
    n = int(y.size)
    i_true = np.full(n, np.nan)
    i_wrong = np.full(n, np.nan)
    l_flex = np.full(n, np.nan)
    true_shift = np.full(n, np.nan)
    wrong_shift = np.full(n, np.nan)
    have = np.zeros(n, dtype=bool)
    deltas = None
    for i, lab in enumerate(y):
        ti = row.get(_scalar(lab))
        if ti is None:
            continue
        i_true[i] = float(sim_shift[i, ti] - sim_fix[i, ti])
        true_shift[i] = float(moved["shift_index"][i, ti])
        rival = (homes == homes[ti]) & (np.arange(ids.size) != ti)
        if not np.any(rival):
            continue
        have[i] = True
        rival_idx = np.flatnonzero(rival)
        nearest = int(rival_idx[np.argmin(fixed_distances[i, rival_idx])])
        i_wrong[i] = float(sim_shift[i, nearest] - sim_fix[i, nearest])
        wrong_shift[i] = float(moved["shift_index"][i, nearest])
        l_flex[i] = float(i_wrong[i] - i_true[i])
    return {
        "score": moved["score"],
        "shift_index": moved["shift_index"],
        "i_true": i_true,
        "i_wrong": i_wrong,
        "l_flex": l_flex,
        "true_shift_index": true_shift,
        "wrong_shift_index": wrong_shift,
        "have_home_rival": have,
    }


def leakage_from_pair(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    mu_true_pos: np.ndarray,
    mu_true_neg: np.ndarray,
    mu_wrong_pos: np.ndarray,
    mu_wrong_neg: np.ndarray,
    mats: np.ndarray,
    rel_xy: np.ndarray,
    kind: str,
    representation: str = "posneg",
) -> dict[str, float]:
    """L_flex on a constructed true / impostor pair (no recordings)."""
    x = build_features(x_pos, x_neg, representation, rel_xy)
    mu_t = build_features(mu_true_pos, mu_true_neg, representation, rel_xy)
    mu_w = build_features(mu_wrong_pos, mu_wrong_neg, representation, rel_xy)
    s_fix_t = float(score_kind(x, mu_t, kind))
    s_fix_w = float(score_kind(x, mu_w, kind))
    s_sh_t, _ = score_shift_bank(
        x_pos, x_neg, mu_true_pos, mu_true_neg, mats, representation, kind, rel_xy
    )
    s_sh_w, _ = score_shift_bank(
        x_pos, x_neg, mu_wrong_pos, mu_wrong_neg, mats, representation, kind, rel_xy
    )
    if SCORING_HIGHER[kind]:
        i_true = float(s_sh_t - s_fix_t)
        i_wrong = float(s_sh_w - s_fix_w)
    else:
        i_true = float(s_fix_t - s_sh_t)
        i_wrong = float(s_fix_w - s_sh_w)
    return {
        "i_true": i_true,
        "i_wrong": i_wrong,
        "l_flex": float(i_wrong - i_true),
        "s_fix_true": s_fix_t,
        "s_fix_wrong": s_fix_w,
        "s_shift_true": float(s_sh_t),
        "s_shift_wrong": float(s_sh_w),
    }


def chosen_shift_stats(
    shift_index: np.ndarray,
    deltas: Sequence[float],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    d = np.asarray(list(deltas), dtype=np.float64)
    idx = np.asarray(shift_index)
    if mask is not None:
        idx = idx[np.asarray(mask, dtype=bool)]
    idx = idx[np.isfinite(idx.astype(np.float64))]
    if idx.size == 0:
        return {
            "mean_abs": float("nan"),
            "frac_nonzero": float("nan"),
            "n": 0,
        }
    chosen = d[idx.astype(np.int64)]
    return {
        "mean_abs": float(np.mean(np.abs(chosen))),
        "frac_nonzero": float(np.mean(np.abs(chosen) > 1e-12)),
        "n": int(chosen.size),
        "mean": float(np.mean(chosen)),
    }


def biased_dot_path_has_no_norm(x: np.ndarray, mu: np.ndarray) -> bool:
    """Contract: biased-dot(2x, mu) != cosine path, and equals 2*(2x)^T mu - ||mu||^2."""
    s = float(biased_dot(x, mu))
    expected = float(2.0 * np.dot(x, mu) - np.dot(mu, mu))
    if not np.allclose(s, expected, rtol=1e-12, atol=1e-12):
        return False
    # Cosine is invariant to scaling x; biased-dot is not.
    s2 = float(biased_dot(2.0 * x, mu))
    c1 = float(cosine_similarity(x, mu))
    c2 = float(cosine_similarity(2.0 * x, mu))
    return (not np.allclose(s, s2)) and np.allclose(c1, c2)
