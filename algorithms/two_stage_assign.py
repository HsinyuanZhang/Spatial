"""COM ∧ P2P spatial gate + waveform second-stage assignment (oracle helpers)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

DistanceMetric = Literal["l2", "l1"]


def vector_distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric = "l2") -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "l1":
        return float(np.sum(np.abs(a - b)))
    raise ValueError(f"Unknown metric: {metric}")


def fit_unit_centroids(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    feats = np.asarray(features, dtype=np.float64)
    labs = np.asarray(labels).ravel()
    cents: dict[int, np.ndarray] = {}
    for u in np.unique(labs):
        mask = labs == u
        if np.any(mask):
            cents[int(u)] = np.mean(feats[mask], axis=0)
    return cents


def assign_nearest(
    features: np.ndarray,
    centroids: dict[int, np.ndarray],
    metric: DistanceMetric = "l2",
) -> np.ndarray:
    """Nearest-centroid labels for each row of features."""
    feats = np.asarray(features, dtype=np.float64)
    if len(feats) == 0:
        return np.empty((0,), dtype=np.int64)
    if not centroids:
        return np.full(len(feats), -1, dtype=np.int64)
    units = np.array(sorted(centroids.keys()), dtype=np.int64)
    mat = np.stack([centroids[int(u)] for u in units], axis=0)
    if metric == "l2":
        dists = np.linalg.norm(feats[:, None, :] - mat[None, :, :], axis=2)
    elif metric == "l1":
        dists = np.sum(np.abs(feats[:, None, :] - mat[None, :, :]), axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return units[np.argmin(dists, axis=1)]


def _intra_unit_distances(
    features: np.ndarray,
    labels: np.ndarray,
    metric: DistanceMetric = "l2",
) -> np.ndarray:
    cents = fit_unit_centroids(features, labels)
    labs = np.asarray(labels).ravel()
    feats = np.asarray(features, dtype=np.float64)
    out = np.zeros(len(feats), dtype=np.float64)
    for i, (x, u) in enumerate(zip(feats, labs)):
        c = cents.get(int(u))
        if c is None:
            out[i] = np.nan
        else:
            out[i] = vector_distance(x, c, metric)
    return out


def calibrate_thresholds(
    com: np.ndarray,
    p2p: np.ndarray,
    labels: np.ndarray,
    percentile: float = 95.0,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> tuple[float, float]:
    """τ_com, τ_p2p from pooled intra-unit distance percentiles on train."""
    d_com = _intra_unit_distances(com, labels, metric=com_metric)
    d_p2p = _intra_unit_distances(p2p, labels, metric=p2p_metric)
    d_com = d_com[np.isfinite(d_com)]
    d_p2p = d_p2p[np.isfinite(d_p2p)]
    if d_com.size == 0 or d_p2p.size == 0:
        return 1.0, 1.0
    tau_com = float(np.percentile(d_com, percentile))
    tau_p2p = float(np.percentile(d_p2p, percentile))
    tau_com = max(tau_com, 1e-8)
    tau_p2p = max(tau_p2p, 1e-8)
    return tau_com, tau_p2p


def fit_unit_code_centroids(
    codes: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Per-unit integer code centroids: ``rint(mean)``. Keys are unit ids."""
    codes = np.asarray(codes)
    labs = np.asarray(labels).ravel()
    cents: dict[int, np.ndarray] = {}
    for u in np.unique(labs):
        mask = labs == u
        if np.any(mask):
            cents[int(u)] = np.rint(codes[mask].mean(axis=0)).astype(np.int64)
    return cents


def calibrate_thresholds_codes(
    codes: np.ndarray,
    labels: np.ndarray,
    percentile: float = 95.0,
) -> int:
    """τ_p2p as integer L1 threshold in code space. Always ``τ ≥ 1``."""
    codes = np.asarray(codes)
    labs = np.asarray(labels).ravel()
    if len(labs) == 0:
        return 1
    cents = fit_unit_code_centroids(codes, labs)
    intra = np.empty(len(labs), dtype=np.int64)
    for i in range(len(labs)):
        c = cents[int(labs[i])]
        intra[i] = int(np.sum(np.abs(codes[i].astype(np.int64) - c)))
    return max(int(np.percentile(intra, percentile)), 1)


def stage1_normalized_score(
    com_x: np.ndarray,
    p2p_x: np.ndarray,
    com_c: np.ndarray,
    p2p_c: np.ndarray,
    tau_com: float,
    tau_p2p: float,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> float:
    d_com = vector_distance(com_x, com_c, com_metric)
    d_p2p = vector_distance(p2p_x, p2p_c, p2p_metric)
    return d_com / max(float(tau_com), 1e-8) + d_p2p / max(float(tau_p2p), 1e-8)


def stage1_candidates(
    com_x: np.ndarray,
    p2p_x: np.ndarray,
    com_centroids: dict[int, np.ndarray],
    p2p_centroids: dict[int, np.ndarray],
    tau_com: float,
    tau_p2p: float,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> list[int]:
    """Units that pass both COM and P2P gates."""
    cands: list[int] = []
    for u in sorted(com_centroids.keys()):
        if u not in p2p_centroids:
            continue
        d_com = vector_distance(com_x, com_centroids[u], com_metric)
        d_p2p = vector_distance(p2p_x, p2p_centroids[u], p2p_metric)
        if d_com < tau_com and d_p2p < tau_p2p:
            cands.append(int(u))
    return cands


def two_stage_assign(
    com_train: np.ndarray,
    p2p_train: np.ndarray,
    wave_train: np.ndarray,
    labels_train: np.ndarray,
    com_test: np.ndarray,
    p2p_test: np.ndarray,
    wave_test: np.ndarray,
    percentile: float = 95.0,
    tau_com: float | None = None,
    tau_p2p: float | None = None,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> tuple[np.ndarray, dict[str, Any]]:
    """COM∧P2P gate then nearest waveform centroid; empty → stage-1 fallback."""
    com_cents = fit_unit_centroids(com_train, labels_train)
    p2p_cents = fit_unit_centroids(p2p_train, labels_train)
    wave_cents = fit_unit_centroids(wave_train, labels_train)

    if tau_com is None or tau_p2p is None:
        tau_com_c, tau_p2p_c = calibrate_thresholds(
            com_train,
            p2p_train,
            labels_train,
            percentile=percentile,
            com_metric=com_metric,
            p2p_metric=p2p_metric,
        )
        if tau_com is None:
            tau_com = tau_com_c
        if tau_p2p is None:
            tau_p2p = tau_p2p_c

    n = len(com_test)
    pred = np.full(n, -1, dtype=np.int64)
    n_fallback = 0
    n_cand_sizes = []

    for i in range(n):
        cands = stage1_candidates(
            com_test[i],
            p2p_test[i],
            com_cents,
            p2p_cents,
            float(tau_com),
            float(tau_p2p),
            com_metric=com_metric,
            p2p_metric=p2p_metric,
        )
        n_cand_sizes.append(len(cands))
        if cands:
            sub = {u: wave_cents[u] for u in cands if u in wave_cents}
            if sub:
                pred[i] = int(assign_nearest(wave_test[i : i + 1], sub)[0])
            else:
                n_fallback += 1
                pred[i] = _fallback_stage1(
                    com_test[i], p2p_test[i], com_cents, p2p_cents,
                    float(tau_com), float(tau_p2p),
                    com_metric=com_metric, p2p_metric=p2p_metric,
                )
        else:
            n_fallback += 1
            pred[i] = _fallback_stage1(
                com_test[i], p2p_test[i], com_cents, p2p_cents,
                float(tau_com), float(tau_p2p),
                com_metric=com_metric, p2p_metric=p2p_metric,
            )

    meta = {
        "tau_com": float(tau_com),
        "tau_p2p": float(tau_p2p),
        "com_metric": com_metric,
        "p2p_metric": p2p_metric,
        "n_fallback": int(n_fallback),
        "fallback_rate": float(n_fallback / max(n, 1)),
        "mean_n_candidates": float(np.mean(n_cand_sizes)) if n_cand_sizes else 0.0,
    }
    return pred, meta


def two_stage_assign_code_p2p(
    com_train: np.ndarray,
    codes_train: np.ndarray,
    wave_train: np.ndarray,
    labels_train: np.ndarray,
    com_test: np.ndarray,
    codes_test: np.ndarray,
    wave_test: np.ndarray,
    percentile: float = 95.0,
    tau_com: float | None = None,
    tau_p2p_codes: int | None = None,
    com_metric: DistanceMetric = "l2",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Thin wrapper: ``two_stage_assign`` with L1 on codes + precomputed ``τ≥1``."""
    codes_train = np.asarray(codes_train, dtype=np.float64)
    codes_test = np.asarray(codes_test, dtype=np.float64)
    if tau_com is None:
        d_com = _intra_unit_distances(com_train, labels_train, metric=com_metric)
        d_com = d_com[np.isfinite(d_com)]
        tau_com = float(np.percentile(d_com, percentile)) if d_com.size else 1.0
        tau_com = max(tau_com, 1e-8)
    if tau_p2p_codes is None:
        tau_p2p_codes = calibrate_thresholds_codes(
            codes_train, labels_train, percentile=percentile
        )
    tau_p2p_codes = max(int(tau_p2p_codes), 1)

    pred, meta = two_stage_assign(
        com_train,
        codes_train,
        wave_train,
        labels_train,
        com_test,
        codes_test,
        wave_test,
        percentile=percentile,
        tau_com=float(tau_com),
        tau_p2p=float(tau_p2p_codes),
        com_metric=com_metric,
        p2p_metric="l1",
    )
    meta = {
        **meta,
        "tau_p2p_codes": int(tau_p2p_codes),
        "p2p_metric": "code_l1",
    }
    return pred, meta


def _fallback_stage1(
    com_x: np.ndarray,
    p2p_x: np.ndarray,
    com_cents: dict[int, np.ndarray],
    p2p_cents: dict[int, np.ndarray],
    tau_com: float,
    tau_p2p: float,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> int:
    best_u = -1
    best_s = np.inf
    for u in com_cents:
        if u not in p2p_cents:
            continue
        s = stage1_normalized_score(
            com_x, p2p_x, com_cents[u], p2p_cents[u], tau_com, tau_p2p,
            com_metric=com_metric, p2p_metric=p2p_metric,
        )
        if s < best_s:
            best_s = s
            best_u = int(u)
    return best_u


def assign_stage1_normalized(
    com: np.ndarray,
    p2p: np.ndarray,
    labels_train: np.ndarray,
    com_train: np.ndarray,
    p2p_train: np.ndarray,
    percentile: float = 95.0,
    com_metric: DistanceMetric = "l2",
    p2p_metric: DistanceMetric = "l2",
) -> np.ndarray:
    """Nearest unit by normalized COM+P2P score (no waveform)."""
    com_cents = fit_unit_centroids(com_train, labels_train)
    p2p_cents = fit_unit_centroids(p2p_train, labels_train)
    tau_com, tau_p2p = calibrate_thresholds(
        com_train, p2p_train, labels_train, percentile=percentile,
        com_metric=com_metric, p2p_metric=p2p_metric,
    )
    pred = np.full(len(com), -1, dtype=np.int64)
    for i in range(len(com)):
        pred[i] = _fallback_stage1(
            com[i], p2p[i], com_cents, p2p_cents, tau_com, tau_p2p,
            com_metric=com_metric, p2p_metric=p2p_metric,
        )
    return pred


def zscore_fit(train: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(train, axis=0)
    sd = np.std(train, axis=0)
    sd = np.maximum(sd, eps)
    return mu, sd


def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def concat_zscore_blocks(
    train_blocks: list[np.ndarray],
    test_blocks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature block on train, concat train/test."""
    train_parts = []
    test_parts = []
    for tr, te in zip(train_blocks, test_blocks):
        mu, sd = zscore_fit(tr)
        train_parts.append(zscore_apply(tr, mu, sd))
        test_parts.append(zscore_apply(te, mu, sd))
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)
