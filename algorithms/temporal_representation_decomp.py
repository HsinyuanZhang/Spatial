"""Same-candidate temporal representation decomposition.

Keeps the frozen Level-1 lists and compares dyadic vs raw-prefix vs Fisher
taps at float and signed-5, without changing the spatial source.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from Spatial.algorithms.causal_temporal_sketch import (
    DYADIC_DICTIONARY,
    SIGNED_CODE_MAX,
    SIGNED_CODE_MIN,
    CausalTemporalSketch,
    signed_rne_divide,
)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def assign_among_candidates(
    features: np.ndarray,
    templates: np.ndarray,
    units: np.ndarray,
    candidates: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-template among candidates. C=0 → None. C=1 → that id, no read."""
    feat = np.asarray(features, dtype=np.float64)
    tmpl = np.asarray(templates, dtype=np.float64)
    labs = np.asarray(units)
    lookup = {_scalar(u): i for i, u in enumerate(labs)}
    pred = np.empty(feat.shape[0], dtype=object)
    reads = np.zeros(feat.shape[0], dtype=np.int64)
    for i, row in enumerate(candidates):
        cand = list(np.asarray(row).tolist())
        if not cand:
            pred[i] = None
            continue
        if len(cand) == 1:
            pred[i] = cand[0]
            continue
        idx = np.asarray([lookup[_scalar(u)] for u in cand], dtype=np.int64)
        dist = np.abs(tmpl[idx] - feat[i]).sum(axis=1)
        pred[i] = cand[int(np.argmin(dist))]
        reads[i] = len(cand)
    return pred, reads


def float_dyadic_features(
    scaled_prefix: np.ndarray,
    filter_ids: np.ndarray,
) -> np.ndarray:
    """Apply the frozen dyadic dictionary to a float prefix in ``[-1, 1]``."""
    x = np.asarray(scaled_prefix, dtype=np.float64)
    ids = np.asarray(filter_ids, dtype=np.int64)
    out = np.empty((x.shape[0], ids.size), dtype=np.float64)
    for j, fid in enumerate(ids):
        entry = DYADIC_DICTIONARY[int(fid)]
        seg = x[:, entry.start : entry.start + entry.support]
        if entry.kind == "mean":
            out[:, j] = seg.mean(axis=1)
        else:
            half = entry.support // 2
            out[:, j] = seg[:, :half].mean(axis=1) - seg[:, half:].mean(axis=1)
    return out


def integer_templates(codes: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels)
    c = np.asarray(codes, dtype=np.int64)
    rows = []
    for unit in units:
        block = c[labs == unit]
        if block.shape[0] == 0:
            raise ValueError(f"no fit rows for unit {unit}")
        summed = block.sum(axis=0)
        rows.append(
            np.clip(
                signed_rne_divide(summed, int(block.shape[0])),
                SIGNED_CODE_MIN,
                SIGNED_CODE_MAX,
            )
        )
    return np.stack(rows, axis=0)


def float_templates(values: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    labs = np.asarray(labels)
    x = np.asarray(values, dtype=np.float64)
    return np.stack([x[labs == unit].mean(axis=0) for unit in units], axis=0)


def fisher_sample_ids(
    codes: np.ndarray,
    labels: np.ndarray,
    n_taps: int,
    eligible_units: np.ndarray,
) -> np.ndarray:
    """Shared Fisher ranking over prefix sample coordinates."""
    between, within, scores = CausalTemporalSketch._fisher_statistics(
        np.asarray(codes, dtype=np.float64),
        np.asarray(labels),
        np.asarray(eligible_units),
    )
    if not np.isfinite(scores).all():
        raise ValueError("Fisher scores must be finite")
    ids = np.arange(codes.shape[1], dtype=np.int64)
    order = np.lexsort((ids, -scores))
    n = min(int(n_taps), int(ids.size))
    return order[:n].astype(np.int64)
