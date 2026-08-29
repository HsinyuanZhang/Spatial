"""WTA among a frozen candidate list (codebook bin or COM ball).

Level-1 still only emits candidates. These helpers score a cheap spatial
descriptor inside that list. They are not a 64x5 waveform engine and they
do not early-exit on |C|=1 as a label.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def scatter_to_channels(
    values: np.ndarray,
    neighbor_ids: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Place slot values onto electrode indices. Invalid slots stay 0."""
    amp = np.asarray(values, dtype=np.float64)
    nbr = np.asarray(neighbor_ids, dtype=np.int64)
    if amp.shape != nbr.shape or amp.ndim != 2:
        raise ValueError("values and neighbor_ids must share shape (n, K)")
    n, k = amp.shape
    out = np.zeros((n, int(n_channels)), dtype=np.float64)
    valid = (nbr >= 0) & (nbr < int(n_channels))
    rows = np.repeat(np.arange(n, dtype=np.int64)[:, None], k, axis=1)
    out[rows[valid], nbr[valid]] = amp[valid]
    return out


def unit_mean_templates(
    features: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
) -> np.ndarray:
    """One mean template per ``units`` row. Missing units are zeros."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).ravel()
    ids = np.asarray(units).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("features/labels length mismatch")
    out = np.zeros((ids.size, x.shape[1]), dtype=np.float64)
    for i, u in enumerate(ids):
        mask = y == u
        if np.any(mask):
            out[i] = x[mask].mean(axis=0)
    return out


def integer_mean_templates(
    codes: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
) -> np.ndarray:
    """``rint(mean)`` templates for already-quantized codes."""
    tmpl = unit_mean_templates(codes, labels, units)
    return np.rint(tmpl).astype(np.int64)


def wta_among_lists(
    query: np.ndarray,
    templates: np.ndarray,
    units: np.ndarray,
    lists: Sequence[np.ndarray],
    *,
    metric: str = "l1",
) -> dict[str, np.ndarray]:
    """Argmin among each event's candidate ids. |C|<=1 does not read templates."""
    q = np.asarray(query)
    tmpl = np.asarray(templates)
    ids = np.asarray(units).ravel()
    if q.ndim != 2 or tmpl.ndim != 2 or q.shape[1] != tmpl.shape[1]:
        raise ValueError("query and templates must share a feature axis")
    if tmpl.shape[0] != ids.size:
        raise ValueError("templates rows must match units")
    if metric not in ("l1", "l2"):
        raise ValueError(f"metric must be l1 or l2, got {metric!r}")
    n = int(q.shape[0])
    if len(lists) != n:
        raise ValueError("one candidate list per query row")
    row = {int(u): i for i, u in enumerate(ids)}
    pred = np.full(n, -1, dtype=np.int64)
    reads = np.zeros(n, dtype=np.int64)
    n_cand = np.zeros(n, dtype=np.int64)
    groups: dict[bytes, list[int]] = defaultdict(list)
    raw_lists: dict[bytes, np.ndarray] = {}
    for i, cand in enumerate(lists):
        c = np.asarray(cand, dtype=np.int64).ravel()
        n_cand[i] = int(c.size)
        if c.size == 0:
            continue
        if c.size == 1:
            pred[i] = int(c[0])
            continue
        key = c.tobytes()
        groups[key].append(i)
        raw_lists[key] = c
        reads[i] = int(c.size)
    for key, idx in groups.items():
        c = raw_lists[key]
        js = np.array([row[int(u)] for u in c if int(u) in row], dtype=np.int64)
        if js.size == 0:
            continue
        keep = np.array([int(u) in row for u in c], dtype=bool)
        live = c[keep]
        block = q[np.asarray(idx, dtype=np.int64)]
        cent = tmpl[js]
        if metric == "l1":
            dist = np.abs(block[:, None, :] - cent[None, :, :]).sum(axis=2)
        else:
            dist = np.sqrt(((block[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2))
        pick = np.argmin(dist, axis=1)
        pred[np.asarray(idx, dtype=np.int64)] = live[pick]
    return {"pred": pred, "reads": reads, "n_cand": n_cand}


def summarize_refine(
    pred: np.ndarray,
    labels: np.ndarray,
    lists: Sequence[np.ndarray],
    *,
    n_units_train: int,
) -> dict[str, Any]:
    y = np.asarray(labels).ravel()
    p = np.asarray(pred, dtype=np.int64).ravel()
    n = int(y.size)
    true_in = np.array(
        [int(lab) in np.asarray(c, dtype=np.int64) for lab, c in zip(y, lists)],
        dtype=bool,
    )
    counts = np.array([len(c) for c in lists], dtype=np.int64)
    n_correct = int(np.sum(p == y))
    n_hit = int(true_in.sum())
    in_and_correct = int(np.sum((p == y) & true_in))
    mean_c = float(np.mean(counts)) if n else 0.0
    return {
        "n_eval": n,
        "n_correct": n_correct,
        "argmin_accuracy": float(n_correct / n) if n else float("nan"),
        "candidate_recall": float(n_hit / n) if n else float("nan"),
        "argmin_in_candidates": float(in_and_correct / max(n_hit, 1)),
        "mean_candidates": mean_c,
        "median_candidates": float(np.median(counts)) if n else 0.0,
        "frac_empty": float(np.mean(counts == 0)) if n else float("nan"),
        "frac_unique": float(np.mean(counts == 1)) if n else float("nan"),
        "traffic_reduction_x": float(n_units_train / max(mean_c, 1e-12)),
        "n_units_train": int(n_units_train),
    }


def codebook_candidate_lists(
    homes: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    train_frac: float = 0.5,
    code_delta: int = 1,
) -> dict[str, Any]:
    """Train occupancy of ``(home, residual code)`` bins, plus test lists."""
    from Spatial.experiments.analyze_np_com_optimize import _ordered_split_1d  # noqa: PLC0415

    split_h = _ordered_split_1d(homes, labels, times, train_frac)
    split_c = _ordered_split_1d(codes, labels, times, train_frac)
    y_tr = split_h["y_train"]
    y_te = split_h["y_test"]
    h_tr = np.asarray(split_h["x_train"], dtype=np.int64).ravel()
    h_te = np.asarray(split_h["x_test"], dtype=np.int64).ravel()
    c_tr = np.rint(np.asarray(split_c["x_train"], dtype=np.float64).ravel()).astype(np.int64)
    c_te = np.rint(np.asarray(split_c["x_test"], dtype=np.float64).ravel()).astype(np.int64)
    members: dict[tuple[int, int], set[int]] = defaultdict(set)
    majority: dict[tuple[int, int], int] = {}
    counts: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for h, c, u in zip(h_tr.tolist(), c_tr.tolist(), y_tr.tolist()):
        key = (int(h), int(c))
        members[key].add(int(u))
        counts[key][int(u)] += 1
    for key, hist in counts.items():
        majority[key] = max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    delta = max(int(code_delta), 0)
    lists: list[np.ndarray] = []
    maj_pred = np.full(y_te.size, -1, dtype=np.int64)
    for i, (hh, cc) in enumerate(zip(h_te.tolist(), c_te.tolist())):
        group: set[int] = set()
        for dc in range(-delta, delta + 1):
            group.update(members.get((int(hh), int(cc) + dc), ()))
        lists.append(np.array(sorted(group), dtype=np.int64))
        hit = majority.get((int(hh), int(cc)))
        if hit is not None:
            maj_pred[i] = int(hit)
    order = np.argsort(np.asarray(times, dtype=np.int64).ravel(), kind="stable")
    train_n = int(h_tr.size)
    orig_train = order[:train_n]
    orig_test = order[train_n:]
    units = np.array(sorted({int(u) for u in y_tr}), dtype=np.int64)
    return {
        "lists": lists,
        "y_train": y_tr,
        "y_test": y_te,
        "train_index": orig_train,
        "test_index": orig_test,
        "majority_pred": maj_pred,
        "units": units,
        "n_bins_train": int(len(members)),
    }
