"""Fit-only mapping-row initialisation (percentiles, tau, precision)."""

from __future__ import annotations

from typing import Any, Hashable, Sequence

import numpy as np

from Spatial.algorithms.adaptive_range_search import (
    AdaptiveRangeSearch,
    UniformUnsignedQuantizer,
    fit_integer_centroids,
)
from Spatial.algorithms.mapping_row import (
    COM_BITS,
    SHAPE_POSNEG_MAX,
    analog_gain,
    normalize_posneg_shape,
    pack_shape_codes,
    precision_from_width,
    quantize_gain,
    score_rows,
)


def _as_labels(labels: Sequence[Hashable] | np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError("labels must be 1-D and match the feature rows")
    return array


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def gain_full_scale(analog_g_fit: np.ndarray) -> float:
    g = np.asarray(analog_g_fit, dtype=np.float64).ravel()
    if g.size == 0:
        raise ValueError("fit gain array is empty")
    return float(max(np.max(g), 1e-12))


def fit_interval_rows(
    codes_fit: np.ndarray,
    labels_fit: Sequence[Hashable] | np.ndarray,
    *,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> dict[str, Any]:
    """Per-unit 5/95 boxes, integer mu, and width-derived precision."""
    codes = np.asarray(codes_fit, dtype=np.int64)
    labs = _as_labels(labels_fit, codes.shape[0])
    units = np.asarray([_scalar(u) for u in np.unique(labs)])
    n_u, d = units.size, codes.shape[1]
    lower = np.empty((n_u, d), dtype=np.int64)
    upper = np.empty((n_u, d), dtype=np.int64)
    mu = np.empty((n_u, d), dtype=np.int64)
    counts = np.empty(n_u, dtype=np.int64)
    for i, unit in enumerate(units):
        block = codes[labs == unit]
        counts[i] = int(block.shape[0])
        mu[i] = np.rint(block.mean(axis=0)).astype(np.int64)
        lower[i] = np.percentile(block, low_pct, axis=0).astype(np.int64)
        upper[i] = np.percentile(block, high_pct, axis=0).astype(np.int64)
        upper[i] = np.maximum(upper[i], lower[i])
    width = np.maximum(upper - lower, 0)
    precision = precision_from_width(width)
    return {
        "units": units,
        "mu": mu,
        "lower": lower,
        "upper": upper,
        "precision": precision,
        "width": width,
        "n_fit": counts,
    }


def fit_gain_ranges(
    g_fit: np.ndarray,
    labels_fit: Sequence[Hashable] | np.ndarray,
    units: np.ndarray,
    *,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g_fit, dtype=np.int64).ravel()
    labs = _as_labels(labels_fit, g.size)
    lo = np.empty(units.size, dtype=np.int64)
    hi = np.empty(units.size, dtype=np.int64)
    for i, unit in enumerate(units):
        block = g[labs == unit]
        if block.size == 0:
            lo[i], hi[i] = 0, 15
            continue
        lo[i] = int(np.percentile(block, lo_pct))
        hi[i] = int(np.percentile(block, hi_pct))
        if hi[i] < lo[i]:
            hi[i] = lo[i]
    return lo, hi


def calibrate_tau(
    codes_cal: np.ndarray,
    labels_cal: Sequence[Hashable] | np.ndarray,
    rows: dict[str, Any],
    *,
    percentile: float = 99.9,
    censoring: bool = False,
    codes_fit: np.ndarray | None = None,
    labels_fit: Sequence[Hashable] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit ceil(p99.9 of S on calibration). Fallback to fit if a unit is absent."""
    units = rows["units"]
    tau = np.zeros(units.size, dtype=np.int64)
    source = np.empty(units.size, dtype=object)
    labs_cal = _as_labels(labels_cal, np.asarray(codes_cal).shape[0])
    for i, unit in enumerate(units):
        mask = labs_cal == unit
        block = np.asarray(codes_cal, dtype=np.int64)[mask]
        used = "calibration"
        if block.shape[0] == 0:
            if codes_fit is None or labels_fit is None:
                tau[i] = 48
                source[i] = "no_rows_max"
                continue
            labs_fit = _as_labels(labels_fit, np.asarray(codes_fit).shape[0])
            block = np.asarray(codes_fit, dtype=np.int64)[labs_fit == unit]
            used = "fit_fallback"
        if block.shape[0] == 0:
            tau[i] = 48
            source[i] = "no_rows_max"
            continue
        scored = score_rows(
            block,
            rows["lower"][i : i + 1],
            rows["upper"][i : i + 1],
            rows["precision"][i : i + 1],
            rows["mu"][i : i + 1],
            censoring=censoring,
        )["S"][:, 0]
        tau[i] = int(np.ceil(np.percentile(scored, percentile)))
        source[i] = used
    return tau, source


def fit_mean_l1_p999(
    codes_fit: np.ndarray,
    labels_fit: Sequence[Hashable] | np.ndarray,
    codes_cal: np.ndarray,
    labels_cal: Sequence[Hashable] | np.ndarray,
    *,
    percentile: float = 99.9,
) -> dict[str, Any]:
    """J0/J1: integer mean centroid + calibration p99.9 L1 radius."""
    fit = np.asarray(codes_fit, dtype=np.int64)
    labs_fit = _as_labels(labels_fit, fit.shape[0])
    centroid_map = fit_integer_centroids(fit, labs_fit)
    units = np.asarray([_scalar(u) for u in np.unique(labs_fit)])
    centroids = np.stack([centroid_map[_scalar(u)] for u in units], axis=0)
    radii = np.empty(units.size, dtype=np.int64)
    source = np.empty(units.size, dtype=object)
    labs_cal = _as_labels(labels_cal, np.asarray(codes_cal).shape[0])
    cal = np.asarray(codes_cal, dtype=np.int64)
    for i, unit in enumerate(units):
        block = cal[labs_cal == unit]
        used = "calibration"
        if block.shape[0] == 0:
            block = fit[labs_fit == unit]
            used = "fit_fallback"
        dist = np.abs(block - centroids[i]).sum(axis=1)
        radii[i] = int(np.ceil(np.percentile(dist, percentile))) if dist.size else 0
        source[i] = used
    return {
        "units": units,
        "centroids": centroids,
        "radii": radii,
        "radius_source": source,
    }


def query_mean_l1(
    codes: np.ndarray,
    model: dict[str, Any],
    active_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    q = np.asarray(codes, dtype=np.int64)
    c = np.asarray(model["centroids"], dtype=np.int64)
    dist = np.abs(q[:, None, :] - c[None, :, :]).sum(axis=2)
    matched = dist <= model["radii"].reshape(1, -1)
    if active_mask is not None:
        matched = matched & np.asarray(active_mask, dtype=bool)
    units = model["units"]
    candidates = [units[row].copy() for row in matched]
    return candidates, dist, {
        "counts": matched.sum(axis=1).astype(np.int64),
        "n_active_row_comparisons": int(matched.size if active_mask is None else np.asarray(active_mask).sum()),
    }


def j0_codes_from_descriptor(descriptor: np.ndarray) -> tuple[np.ndarray, UniformUnsignedQuantizer]:
    """5-bit COM+P2P on the existing unit-interval D9 descriptor."""
    quant = UniformUnsignedQuantizer(n_bits=5, fixed_unit_interval=True).fit(descriptor)
    return quant.transform(descriptor), quant


def com_codes_unit_interval(com: np.ndarray) -> np.ndarray:
    quant = UniformUnsignedQuantizer(n_bits=COM_BITS, fixed_unit_interval=True)
    quant.fit(np.zeros((1, 2)))  # bounds [0,1]
    return quant.transform(com)


def column_abs_scales(
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit-only per-slot peak/trough scales (J2, no per-event g)."""
    pos = np.maximum(np.asarray(vmax, dtype=np.float64), 0.0)
    neg = np.maximum(-np.asarray(vmin, dtype=np.float64), 0.0)
    valid = np.asarray(neighbor_ids, dtype=np.int64) >= 0
    pos = np.where(valid, pos, 0.0)
    neg = np.where(valid, neg, 0.0)
    return np.maximum(pos.max(axis=0), 1e-12), np.maximum(neg.max(axis=0), 1e-12)


def shape_codes_from_extrema(
    com: np.ndarray,
    vmax: np.ndarray,
    vmin: np.ndarray,
    neighbor_ids: np.ndarray,
    analog_g: np.ndarray,
    *,
    normalize_by_g: bool,
    pos_scale: np.ndarray | None = None,
    neg_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Integer query vector: 5-bit COM plus 4-bit POSNEG."""
    com_c = com_codes_unit_interval(com)
    if normalize_by_g:
        pos, neg = normalize_posneg_shape(vmax, vmin, neighbor_ids, analog_g)
        return pack_shape_codes(com_c, pos, neg)
    if pos_scale is None or neg_scale is None:
        raise ValueError("J2 raw POSNEG requires fit-only pos_scale and neg_scale")
    pos = np.maximum(np.asarray(vmax, dtype=np.float64), 0.0)
    neg = np.maximum(-np.asarray(vmin, dtype=np.float64), 0.0)
    valid = np.asarray(neighbor_ids, dtype=np.int64) >= 0
    pos_c = np.clip(np.rint(SHAPE_POSNEG_MAX * pos / pos_scale.reshape(1, -1)), 0, SHAPE_POSNEG_MAX)
    neg_c = np.clip(np.rint(SHAPE_POSNEG_MAX * neg / neg_scale.reshape(1, -1)), 0, SHAPE_POSNEG_MAX)
    pos_c = np.where(valid, pos_c, 0.0)
    neg_c = np.where(valid, neg_c, 0.0)
    return pack_shape_codes(com_c, pos_c.astype(np.int64), neg_c.astype(np.int64))


def j0_search_from_descriptor(
    descriptor_fit: np.ndarray,
    labels_fit: Sequence[Hashable] | np.ndarray,
    descriptor_cal: np.ndarray,
    labels_cal: Sequence[Hashable] | np.ndarray,
) -> AdaptiveRangeSearch:
    """Existing 5-bit L1 object with calibration-only p99.9 radii."""
    model = AdaptiveRangeSearch(
        n_bits=5,
        fixed_unit_interval=True,
        radius_percentile=99.9,
        box_coverage=0.95,
    ).fit(descriptor_fit, labels_fit)
    cal_codes = model.transform(descriptor_cal)
    labs_cal = _as_labels(labels_cal, cal_codes.shape[0])
    for i, unit in enumerate(model.units_):
        block = cal_codes[labs_cal == unit]
        if block.shape[0] == 0:
            continue
        dist = np.abs(block - model.centroids_[i]).sum(axis=1)
        model.radii_[i] = int(np.ceil(np.percentile(dist, 99.9)))
    return model
