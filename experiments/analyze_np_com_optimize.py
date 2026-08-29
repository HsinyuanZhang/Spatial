"""Optimize low-bit COM on Neuropixels ``sim_no_drift``.

The previous unique-ID mixed a channel index (0–383) with a residual code
in one L2 vector. Hardware does not do that: home is an address, the
residual is a few bits. This sweep reports:

1. Home-gated residual WTA (exact home or a µm slack).
2. ``(home, residual code)`` codebook unique-ID and candidate lists.
3. Radius gates on reconstructed quantized COM_y.
4. Optional signed-ADC → P2P → COM (absolute gain, not per-spike AGC).

    python -m Spatial.experiments.analyze_np_com_optimize --duration 60
    python -m Spatial.experiments.analyze_np_com_optimize --duration 60 --adc
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    build_radius_neighbor_table,
    com_features,
)
from Spatial.data.ks4_sim import load_ks4_sim_no_drift
from Spatial.experiments.analyze_adc_absolute_com import _event_abs_scales
from Spatial.experiments.analyze_com_radius_prefilter import (
    evaluate_com_radius_prefilter,
)
from Spatial.experiments.analyze_np_com_lowbit import quantize_span
from Spatial.experiments.run_com_ks4sim_nodrift import (
    DEFAULT_OUT,
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    prepare_ks4sim_com_events,
)
from Spatial.experiments.run_com_mainchannel_lowbit import (
    WINDOW,
    _chronological_split,
    oracle_argmin_assign,
)


BITS = (3, 4, 5, 6, 8)
LOCAL_HALF_UM = (45.0, 60.0, 90.0, 120.0)
HOME_RADIUS_UM = (0.0, 20.0, 60.0)
RADIUS_UM = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0)
ADC_BITS = (4, 5, 6)
FINE_H_BITS = ((90.0, 3), (90.0, 4), (60.0, 4), (90.0, 5), (90.0, 6))


def dequantize_span(
    codes: np.ndarray,
    n_bits: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Undo :func:`quantize_span` (bin centres at the integer levels)."""
    c = np.asarray(codes, dtype=np.float64).ravel()
    max_level = float((1 << int(n_bits)) - 1)
    span = max(float(hi) - float(lo), 1e-8)
    return float(lo) + c / max_level * span


def _ordered_split_1d(
    values: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
) -> dict[str, np.ndarray]:
    labs = np.asarray(labels).ravel()
    t = np.asarray(times, dtype=np.int64).ravel()
    x = np.asarray(values)
    n = int(labs.shape[0])
    if x.shape[0] != n or t.shape[0] != n:
        raise ValueError("values, labels, and times length mismatch")
    order = np.argsort(t, kind="stable")
    x, labs, t = x[order], labs[order], t[order]
    train_idx, test_idx = _chronological_split(n, train_frac)
    return {
        "x_train": x[train_idx],
        "x_test": x[test_idx],
        "y_train": labs[train_idx],
        "y_test": labs[test_idx],
    }


def _majority_home(homes: np.ndarray, labels: np.ndarray, n_ch: int) -> dict[int, int]:
    out: dict[int, int] = {}
    h = np.asarray(homes, dtype=np.int64).ravel()
    for u in np.unique(labels):
        ch = h[labels == u]
        if ch.size == 0:
            continue
        out[int(u)] = int(np.bincount(ch, minlength=n_ch).argmax())
    return out


def _nearby_channel_sets(geom: np.ndarray, radius_um: float) -> list[np.ndarray]:
    g = np.asarray(geom, dtype=np.float64)
    if float(radius_um) <= 0.0:
        return [np.array([i], dtype=np.int64) for i in range(g.shape[0])]
    d = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=2)
    return [np.flatnonzero(d[i] <= float(radius_um) + 1e-8) for i in range(g.shape[0])]


def home_then_residual_assign(
    homes: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    geom: np.ndarray,
    home_radius_um: float = 0.0,
    train_frac: float = 0.5,
) -> dict[str, Any]:
    """Gate by home electrode, then nearest residual among surviving units."""
    g = np.asarray(geom, dtype=np.float64)
    n_ch = int(g.shape[0])
    split_h = _ordered_split_1d(homes, labels, times, train_frac)
    split_r = _ordered_split_1d(residual, labels, times, train_frac)
    y_tr = split_h["y_train"]
    y_te = split_h["y_test"]
    h_tr = np.asarray(split_h["x_train"], dtype=np.int64).ravel()
    h_te = np.asarray(split_h["x_test"], dtype=np.int64).ravel()
    r_tr = np.asarray(split_r["x_train"], dtype=np.float64).ravel()
    r_te = np.asarray(split_r["x_test"], dtype=np.float64).ravel()
    maj = _majority_home(h_tr, y_tr, n_ch)
    units = np.array(sorted(maj.keys()), dtype=np.int64)
    if units.size == 0:
        raise ValueError("no train units")
    unit_home = np.array([maj[int(u)] for u in units], dtype=np.int64)
    unit_res = np.array(
        [float(np.mean(r_tr[y_tr == u])) for u in units], dtype=np.float64
    )
    nearby = _nearby_channel_sets(g, home_radius_um)
    units_by_event_home: list[np.ndarray] = []
    for ch in range(n_ch):
        allowed = set(int(c) for c in nearby[ch])
        js = np.array(
            [j for j, hh in enumerate(unit_home) if int(hh) in allowed],
            dtype=np.int64,
        )
        units_by_event_home.append(js)

    n = int(y_te.shape[0])
    pred = np.full(n, -1, dtype=np.int64)
    n_cand = np.zeros(n, dtype=np.int64)
    true_in = np.zeros(n, dtype=bool)
    col = {int(u): j for j, u in enumerate(units)}
    seen = np.array([int(u) in col for u in y_te], dtype=bool)
    true_col = np.array([col.get(int(u), -1) for u in y_te], dtype=np.int64)
    h_clip = np.clip(h_te, 0, n_ch - 1)
    for ch in np.unique(h_clip):
        idx = np.flatnonzero(h_clip == ch)
        js = units_by_event_home[int(ch)]
        n_cand[idx] = int(js.size)
        if js.size == 0:
            continue
        d = np.abs(r_te[idx][:, None] - unit_res[js][None, :])
        pred[idx] = units[js][np.argmin(d, axis=1)]
        tc = true_col[idx]
        true_in[idx] = (tc[:, None] == js[None, :]).any(axis=1) & (tc >= 0)

    n_eval = int(seen.sum())
    n_correct = int(np.sum(pred[seen] == y_te[seen])) if n_eval else 0
    n_hit = int(np.sum(true_in[seen])) if n_eval else 0
    in_set = seen & true_in
    n_in = int(in_set.sum())
    n_wta = int(np.sum(pred[in_set] == y_te[in_set])) if n_in else 0
    mean_c = float(np.mean(n_cand)) if n else 0.0
    return {
        "n_eval": n_eval,
        "n_correct": n_correct,
        "argmin_accuracy": float(n_correct / n_eval) if n_eval else float("nan"),
        "candidate_recall": float(n_hit / n_eval) if n_eval else float("nan"),
        "mean_candidates": mean_c,
        "median_candidates": float(np.median(n_cand)) if n else 0.0,
        "frac_empty": float(np.mean(n_cand == 0)) if n else float("nan"),
        "argmin_in_candidates": float(n_wta / n_in) if n_in else float("nan"),
        "n_units_train": int(units.size),
        "home_radius_um": float(home_radius_um),
    }


def codebook_assign(
    homes: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
) -> dict[str, Any]:
    """Map each ``(home, code)`` bin to the majority train unit."""
    split_h = _ordered_split_1d(homes, labels, times, train_frac)
    split_c = _ordered_split_1d(codes, labels, times, train_frac)
    y_tr = split_h["y_train"]
    y_te = split_h["y_test"]
    h_tr = np.asarray(split_h["x_train"], dtype=np.int64).ravel()
    h_te = np.asarray(split_h["x_test"], dtype=np.int64).ravel()
    c_tr = np.rint(np.asarray(split_c["x_train"], dtype=np.float64).ravel()).astype(
        np.int64
    )
    c_te = np.rint(np.asarray(split_c["x_test"], dtype=np.float64).ravel()).astype(
        np.int64
    )
    counts: dict[tuple[int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    train_units = set(int(u) for u in np.unique(y_tr))
    for h, c, u in zip(h_tr, c_tr, y_tr):
        counts[(int(h), int(c))][int(u)] += 1
    majority: dict[tuple[int, int], int] = {}
    for key, hist in counts.items():
        majority[key] = max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    n = int(y_te.shape[0])
    pred = np.full(n, -1, dtype=np.int64)
    unseen = np.ones(n, dtype=bool)
    c_max = 0
    if c_tr.size:
        c_max = max(c_max, int(c_tr.max()))
    if c_te.size:
        c_max = max(c_max, int(c_te.max()))
    stride = c_max + 1
    maj_key = np.array(
        [h * stride + c for (h, c) in majority.keys()], dtype=np.int64
    )
    maj_val = np.array(list(majority.values()), dtype=np.int64)
    lut = dict(zip(maj_key.tolist(), maj_val.tolist()))
    te_key = h_te.astype(np.int64) * stride + c_te.astype(np.int64)
    for i, key in enumerate(te_key.tolist()):
        hit = lut.get(int(key))
        if hit is None:
            continue
        unseen[i] = False
        pred[i] = int(hit)
    seen = np.array([int(u) in train_units for u in y_te], dtype=bool)
    n_eval = int(seen.sum())
    n_correct = int(np.sum(pred[seen] == y_te[seen])) if n_eval else 0
    return {
        "n_eval": n_eval,
        "n_correct": n_correct,
        "n_unseen_bin": int(np.sum(unseen[seen])) if n_eval else 0,
        "argmin_accuracy": float(n_correct / n_eval) if n_eval else float("nan"),
        "frac_unseen_bin": (
            float(np.mean(unseen[seen])) if n_eval else float("nan")
        ),
        "n_bins_train": int(len(majority)),
        "n_units_train": int(len(train_units)),
    }


def codebook_candidates(
    homes: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    train_frac: float = 0.5,
    code_delta: int = 0,
) -> dict[str, Any]:
    """Candidate list = train units observed in the same ``(home, code)`` bin.

    ``code_delta>0`` also unions neighbouring residual codes on the same home.
    """
    split_h = _ordered_split_1d(homes, labels, times, train_frac)
    split_c = _ordered_split_1d(codes, labels, times, train_frac)
    y_tr = split_h["y_train"]
    y_te = split_h["y_test"]
    h_tr = np.asarray(split_h["x_train"], dtype=np.int64).ravel()
    h_te = np.asarray(split_h["x_test"], dtype=np.int64).ravel()
    c_tr = np.rint(np.asarray(split_c["x_train"], dtype=np.float64).ravel()).astype(
        np.int64
    )
    c_te = np.rint(np.asarray(split_c["x_test"], dtype=np.float64).ravel()).astype(
        np.int64
    )
    members: dict[tuple[int, int], set[int]] = defaultdict(set)
    counts: dict[tuple[int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    train_units = set(int(u) for u in np.unique(y_tr))
    for h, c, u in zip(h_tr, c_tr, y_tr):
        key = (int(h), int(c))
        members[key].add(int(u))
        counts[key][int(u)] += 1
    majority: dict[tuple[int, int], int] = {}
    for key, hist in counts.items():
        majority[key] = max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    n = int(y_te.shape[0])
    n_cand = np.zeros(n, dtype=np.int64)
    true_in = np.zeros(n, dtype=bool)
    pred = np.full(n, -1, dtype=np.int64)
    delta = max(int(code_delta), 0)
    for i, (hh, cc, uu) in enumerate(
        zip(h_te.tolist(), c_te.tolist(), y_te.tolist())
    ):
        hh_i = int(hh)
        cc_i = int(cc)
        group: set[int] = set()
        for dc in range(-delta, delta + 1):
            group.update(members.get((hh_i, cc_i + dc), ()))
        n_cand[i] = int(len(group))
        true_in[i] = int(uu) in group
        key = (hh_i, cc_i)
        if key in majority:
            pred[i] = int(majority[key])
    seen = np.array([int(u) in train_units for u in y_te], dtype=bool)
    n_eval = int(seen.sum())
    n_hit = int(np.sum(true_in[seen])) if n_eval else 0
    n_correct = int(np.sum(pred[seen] == y_te[seen])) if n_eval else 0
    in_set = seen & true_in
    n_in = int(in_set.sum())
    n_wta = int(np.sum(pred[in_set] == y_te[in_set])) if n_in else 0
    mean_c = float(np.mean(n_cand)) if n else 0.0
    n_units = int(len(train_units))
    return {
        "n_eval": n_eval,
        "n_correct": n_correct,
        "argmin_accuracy": float(n_correct / n_eval) if n_eval else float("nan"),
        "candidate_recall": float(n_hit / n_eval) if n_eval else float("nan"),
        "mean_candidates": mean_c,
        "median_candidates": float(np.median(n_cand)) if n else 0.0,
        "frac_empty": float(np.mean(n_cand == 0)) if n else float("nan"),
        "argmin_in_candidates": float(n_wta / n_in) if n_in else float("nan"),
        "traffic_reduction_x": float(n_units / max(mean_c, 1e-12)),
        "n_bins_train": int(len(members)),
        "n_units_train": n_units,
    }


def cohome_residual_stats(
    homes: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    geom: np.ndarray,
) -> dict[str, float]:
    """Can residual separate units that share a majority home?"""
    split_h = _ordered_split_1d(homes, labels, times, 0.5)
    split_r = _ordered_split_1d(residual, labels, times, 0.5)
    y = split_h["y_train"]
    h = np.asarray(split_h["x_train"], dtype=np.int64).ravel()
    r = np.asarray(split_r["x_train"], dtype=np.float64).ravel()
    n_ch = int(geom.shape[0])
    maj = _majority_home(h, y, n_ch)
    mean_r = {int(u): float(np.mean(r[y == u])) for u in maj}
    intra = []
    for u in maj:
        d = r[y == u] - mean_r[int(u)]
        if d.size:
            intra.append(float(np.sqrt(np.mean(d**2))))
    by_home: dict[int, list[int]] = defaultdict(list)
    for u, ch in maj.items():
        by_home[int(ch)].append(int(u))
    gaps = []
    n_share = 0
    n_sep = 0
    med_intra = float(np.median(intra)) if intra else float("nan")
    for group in by_home.values():
        if len(group) < 2:
            continue
        n_share += len(group)
        ys = np.array([mean_r[u] for u in group], dtype=np.float64)
        for i, a in enumerate(ys):
            others = np.delete(ys, i)
            gap = float(np.min(np.abs(others - a)))
            gaps.append(gap)
            if np.isfinite(med_intra) and gap > 2.0 * med_intra:
                n_sep += 1
    occ = np.array([len(v) for v in by_home.values()], dtype=np.int64)
    return {
        "n_units_train": float(len(maj)),
        "n_occupied_homes": float(len(by_home)),
        "mean_units_per_home": float(np.mean(occ)) if occ.size else float("nan"),
        "max_units_per_home": float(np.max(occ)) if occ.size else float("nan"),
        "frac_units_sharing_home": float(n_share / max(len(maj), 1)),
        "intra_residual_rms_um_median": med_intra,
        "cohome_nn_residual_um_median": (
            float(np.median(gaps)) if gaps else float("nan")
        ),
        "frac_cohome_nn_gt_2x_intra": (
            float(n_sep / max(len(gaps), 1)) if gaps else float("nan")
        ),
    }


def extract_local_snippets(
    signal: np.ndarray,
    times: np.ndarray,
    homes: np.ndarray,
    neighbor_table: np.ndarray,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Analog neighbourhood snippets. Shape (n_events, K, 2*window)."""
    sig = np.asarray(signal, dtype=np.float32)
    t = np.asarray(times, dtype=np.int64).ravel()
    h = np.asarray(homes, dtype=np.int64).ravel()
    table = np.asarray(neighbor_table, dtype=np.int64)
    n = int(t.size)
    k = int(table.shape[1])
    w = int(window)
    snips = np.zeros((n, k, 2 * w), dtype=np.float32)
    nbr_ids = np.zeros((n, k), dtype=np.int64)
    n_ch, n_samp = sig.shape
    for i in range(n):
        ch = int(np.clip(h[i], 0, table.shape[0] - 1))
        nbr = table[ch]
        nbr_ids[i] = nbr
        ti = int(t[i])
        if ti < w or ti + w > n_samp:
            continue
        for j, ch_j in enumerate(nbr):
            cj = int(ch_j)
            if 0 <= cj < n_ch:
                snips[i, j] = sig[cj, ti - w : ti + w]
    return snips, nbr_ids


def p2p_from_adc_snippets(
    snippets: np.ndarray,
    neighbor_ids: np.ndarray,
    n_bits: int,
    full_scale: float | np.ndarray,
) -> np.ndarray:
    """Signed ADC on cached snippets, then peak-to-peak along time."""
    x = np.asarray(snippets, dtype=np.float64)
    nbr = np.asarray(neighbor_ids, dtype=np.int64)
    qpos = float((1 << (int(n_bits) - 1)) - 1)
    fs = np.asarray(full_scale, dtype=np.float64)
    valid = nbr >= 0
    if fs.ndim == 0:
        lsb = max(float(fs) / qpos, 1e-12)
        recon = np.clip(np.rint(x / lsb), -qpos, qpos) * lsb
    else:
        ch = np.clip(nbr, 0, fs.shape[0] - 1)
        vref = np.maximum(fs[ch], 1e-12)
        lsb = vref[..., None] / qpos
        recon = np.clip(np.rint(x / lsb), -qpos, qpos) * lsb
    recon[~valid] = 0.0
    return np.max(recon, axis=-1) - np.min(recon, axis=-1)


def _eval_row(name: str, rec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {"method": name}
    for key in (
        "n_eval",
        "n_correct",
        "argmin_accuracy",
        "candidate_recall",
        "mean_candidates",
        "median_candidates",
        "frac_empty",
        "argmin_in_candidates",
        "traffic_reduction_x",
        "n_unseen_bin",
        "frac_unseen_bin",
        "n_bins_train",
        "n_units_train",
        "home_radius_um",
    ):
        if key in rec:
            row[key] = rec[key]
    row.update(extra)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_cache_sweeps(prep: dict[str, Any]) -> dict[str, Any]:
    times = prep["times"]
    labels = prep["labels"]
    homes = np.asarray(prep["homes"], dtype=np.int64)
    com_um = np.asarray(prep["com_um"], dtype=np.float64)
    geom = np.asarray(prep["geom"], dtype=np.float64)
    home_y = geom[homes, 1]
    com_y = com_um[:, 1]
    rel_y = com_y - home_y

    diag = cohome_residual_stats(homes, rel_y, labels, times, geom)
    gate_rows: list[dict[str, Any]] = []
    code_rows: list[dict[str, Any]] = []

    gate_rows.append(
        _eval_row(
            "home_gate_residual_float",
            home_then_residual_assign(
                homes, rel_y, labels, times, geom, home_radius_um=0.0
            ),
            n_bits=-1,
            half_um=None,
            note="exact home then float residual",
        )
    )
    for slack in HOME_RADIUS_UM:
        rec = home_then_residual_assign(
            homes, rel_y, labels, times, geom, home_radius_um=float(slack)
        )
        gate_rows.append(
            _eval_row(
                f"home_r{slack:.0f}_residual_float",
                rec,
                n_bits=-1,
                half_um=None,
                note=f"home slack {slack:.0f} um + float residual",
            )
        )

    for bits in BITS:
        for half in LOCAL_HALF_UM:
            codes = quantize_span(rel_y, bits, -half, half).ravel()
            rec_q = home_then_residual_assign(
                homes, codes.astype(np.float64), labels, times, geom, 0.0
            )
            gate_rows.append(
                _eval_row(
                    f"home_exact_rel_pm{half:.0f}_{bits}bit",
                    rec_q,
                    n_bits=int(bits),
                    half_um=float(half),
                    bin_um=float(2 * half / ((1 << bits) - 1)),
                    note="exact home then quantized residual",
                )
            )
            rec_s = home_then_residual_assign(
                homes, codes.astype(np.float64), labels, times, geom, 60.0
            )
            gate_rows.append(
                _eval_row(
                    f"home_r60_rel_pm{half:.0f}_{bits}bit",
                    rec_s,
                    n_bits=int(bits),
                    half_um=float(half),
                    bin_um=float(2 * half / ((1 << bits) - 1)),
                    note="home slack 60 um then quantized residual",
                )
            )
            rec_cb = codebook_assign(homes, codes, labels, times)
            rec_cc = codebook_candidates(homes, codes, labels, times)
            code_rows.append(
                _eval_row(
                    f"codebook_pm{half:.0f}_{bits}bit",
                    rec_cb,
                    n_bits=int(bits),
                    half_um=float(half),
                    bin_um=float(2 * half / ((1 << bits) - 1)),
                    kind="unique_id",
                )
            )
            code_rows.append(
                _eval_row(
                    f"codebook_cand_pm{half:.0f}_{bits}bit",
                    rec_cc,
                    n_bits=int(bits),
                    half_um=float(half),
                    bin_um=float(2 * half / ((1 << bits) - 1)),
                    kind="candidates",
                    code_delta=0,
                )
            )

    for bits in (3, 4, 5):
        for half in (60.0, 90.0):
            codes = quantize_span(rel_y, bits, -half, half).ravel()
            for delta in (1, 2):
                rec_cc = codebook_candidates(
                    homes, codes, labels, times, code_delta=int(delta)
                )
                code_rows.append(
                    _eval_row(
                        f"codebook_cand_pm{half:.0f}_{bits}bit_d{delta}",
                        rec_cc,
                        n_bits=int(bits),
                        half_um=float(half),
                        bin_um=float(2 * half / ((1 << bits) - 1)),
                        kind="candidates",
                        code_delta=int(delta),
                    )
                )

    # Physical-µm L2: (home_x, home_y + residual) == COM with x snapped.
    home_xy = geom[homes]
    snapped = np.column_stack([home_xy[:, 0], home_xy[:, 1] + rel_y])
    mixed = oracle_argmin_assign(snapped, labels, times)
    gate_rows.append(
        _eval_row(
            "l2_home_xy_plus_rel_um",
            {
                "n_eval": mixed["n_eval"],
                "n_correct": mixed["n_correct"],
                "argmin_accuracy": mixed["argmin_accuracy"],
                "n_units_train": mixed["n_units_train"],
            },
            n_bits=-1,
            note="L2 on (home_x, COM_y) in um; not a discrete home gate",
        )
    )

    radius_rows: list[dict[str, Any]] = []
    for r in RADIUS_UM:
        rec = evaluate_com_radius_prefilter(com_um, labels, times, radius_um=float(r))
        radius_rows.append(
            {
                "source": "com_xy_float",
                "n_bits": -1,
                "half_um": None,
                "radius_um": float(r),
                "candidate_recall": rec["candidate_recall"],
                "mean_candidates": rec["mean_candidates"],
                "traffic_reduction_x": rec["traffic_reduction_x"],
                "argmin_in_candidates": rec["argmin_in_candidates"],
            }
        )

    for half, bits in FINE_H_BITS:
        codes = quantize_span(rel_y, bits, -half, half).ravel()
        rec_y = home_y + dequantize_span(codes, bits, -half, half)
        rec_xy = np.column_stack([home_xy[:, 0], rec_y])
        for r in RADIUS_UM:
            rec = evaluate_com_radius_prefilter(
                rec_xy, labels, times, radius_um=float(r)
            )
            radius_rows.append(
                {
                    "source": f"home_xy_plus_qrel_pm{half:.0f}_{bits}bit",
                    "n_bits": int(bits),
                    "half_um": float(half),
                    "radius_um": float(r),
                    "candidate_recall": rec["candidate_recall"],
                    "mean_candidates": rec["mean_candidates"],
                    "traffic_reduction_x": rec["traffic_reduction_x"],
                    "argmin_in_candidates": rec["argmin_in_candidates"],
                }
            )

    def _best_gate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        ok = [r for r in rows if "argmin_accuracy" in r]
        if not ok:
            return None
        return max(ok, key=lambda r: float(r["argmin_accuracy"]))

    def _pick_radius(
        rows: list[dict[str, Any]], source: str, min_recall: float
    ) -> dict[str, Any] | None:
        ok = [
            r
            for r in rows
            if r["source"] == source and r["candidate_recall"] >= min_recall
        ]
        if not ok:
            return None
        return min(ok, key=lambda r: r["mean_candidates"])

    sources = sorted({r["source"] for r in radius_rows})
    selected = {
        src: {
            "r99": _pick_radius(radius_rows, src, 0.99),
            "r95": _pick_radius(radius_rows, src, 0.95),
        }
        for src in sources
    }
    return {
        "cohome": diag,
        "gate": gate_rows,
        "codebook": code_rows,
        "radius": radius_rows,
        "best_unique_id": _best_gate(gate_rows),
        "selected_radius": selected,
    }


def run_adc_sweep(prep: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Absolute-gain signed ADC on the 60 s prefix, then 3-pitch COM."""
    ds = load_ks4_sim_no_drift(duration_s=duration_s)
    filtered = bandpass_filter(ds.raw_data, ds.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ)
    times = np.asarray(prep["times"], dtype=np.int64)
    labels = prep["labels"]
    homes = np.asarray(prep["homes"], dtype=np.int64)
    geom = np.asarray(prep["geom"], dtype=np.float64)
    analog_com = np.asarray(prep["com_um"], dtype=np.float64)
    radius = float(np.asarray(prep["radius_um"]))
    table = build_radius_neighbor_table(geom, radius)
    rng = np.random.default_rng(0)
    scales = _event_abs_scales(filtered, times, WINDOW, rng)
    snippets, nbr = extract_local_snippets(
        filtered, times, homes, table, window=WINDOW
    )
    del filtered
    geom_min = np.min(geom, axis=0)
    geom_span = np.maximum(np.max(geom, axis=0) - geom_min, 1e-8)

    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, int, float | np.ndarray]] = []
    for bits in ADC_BITS:
        configs.append(("all_channel_p999", bits, float(scales["global_p999"])))
        configs.append(("all_channel_max", bits, float(scales["global_max"])))
        configs.append(("per_channel_p999", bits, scales["channel_p999"]))

    analog_bbox = (analog_com - geom_min) / geom_span
    for name, bits, fs in configs:
        p2p = p2p_from_adc_snippets(snippets, nbr, bits, fs)
        com = com_features(p2p, nbr, geom)
        com_um = com * geom_span + geom_min
        err = com_um - analog_com
        rms = np.sqrt(np.mean(err**2, axis=0))
        home_y = geom[homes, 1]
        rel = com_um[:, 1] - home_y
        analog_rel = analog_com[:, 1] - home_y
        gate = home_then_residual_assign(homes, rel, labels, times, geom, 60.0)
        gate0 = home_then_residual_assign(homes, rel, labels, times, geom, 0.0)
        codes = quantize_span(rel, 4, -90.0, 90.0).ravel()
        cb = codebook_candidates(homes, codes, labels, times)
        rec80 = evaluate_com_radius_prefilter(
            com_um, labels, times, radius_um=80.0
        )
        rec60 = evaluate_com_radius_prefilter(
            com_um, labels, times, radius_um=60.0
        )
        assign_xy = oracle_argmin_assign(com, labels, times)
        rows.append(
            {
                "scale": name,
                "adc_bits": int(bits),
                "rms_x_um": float(rms[0]),
                "rms_y_um": float(rms[1]),
                "residual_rms_vs_analog_um": float(
                    np.sqrt(np.mean((rel - analog_rel) ** 2))
                ),
                "argmin_com_xy": float(assign_xy["argmin_accuracy"]),
                "home_exact_residual": float(gate0["argmin_accuracy"]),
                "home_r60_residual": float(gate["argmin_accuracy"]),
                "home_r60_recall": float(gate["candidate_recall"]),
                "home_r60_mean_c": float(gate["mean_candidates"]),
                "codebook_cand_4bit_pm90_recall": float(cb["candidate_recall"]),
                "codebook_cand_4bit_pm90_mean_c": float(cb["mean_candidates"]),
                "codebook_cand_4bit_pm90_id": float(cb["argmin_accuracy"]),
                "radius80_recall": float(rec80["candidate_recall"]),
                "radius80_mean_c": float(rec80["mean_candidates"]),
                "radius60_recall": float(rec60["candidate_recall"]),
                "radius60_mean_c": float(rec60["mean_candidates"]),
                "full_scale": (
                    float(fs) if np.ndim(fs) == 0 else "per_channel_vector"
                ),
            }
        )
        del p2p

    ch_max = np.asarray(scales["channel_max"])
    return {
        "adc": "signed two's-complement on local 3-pitch snippets; analog homes",
        "n_events": int(times.size),
        "fs": int(ds.fs),
        "analog_event_window": {
            "global_max_abs": float(scales["global_max"]),
            "global_p999_abs": float(scales["global_p999"]),
            "channel_max_median": float(np.median(ch_max)),
            "channel_max_hottest": float(np.max(ch_max)),
            "hottest_over_median_channel": float(
                np.max(ch_max) / max(float(np.median(ch_max)), 1e-12)
            ),
        },
        "argmin_analog_com_xy": float(
            oracle_argmin_assign(analog_bbox, labels, times)["argmin_accuracy"]
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--adc",
        action="store_true",
        help="Re-stream the 60 s zip prefix and sweep signed ADC → COM",
    )
    parser.add_argument(
        "--skip-sweeps",
        action="store_true",
        help="Do not rerun the cache unique-ID / codebook / radius grids",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    cache = out_dir / f"events_{int(args.duration)}s.npz"
    prep = prepare_ks4sim_com_events(
        float(args.duration), cache_path=cache, verbose=True
    )
    summary: dict[str, Any] = {
        "dataset": "ks4sim_no_drift",
        "duration_s": float(args.duration),
        "note": (
            "Home is a discrete address. Residual COM_y is quantized locally. "
            "The mixed L2 (home index, residual) number is not a hardware ID."
        ),
    }
    if not args.skip_sweeps:
        sweeps = run_cache_sweeps(prep)
        _write_csv(out_dir / "np_com_home_gate.csv", sweeps["gate"])
        _write_csv(out_dir / "np_com_codebook.csv", sweeps["codebook"])
        _write_csv(out_dir / "np_com_quant_radius.csv", sweeps["radius"])
        summary["cohome_residual"] = sweeps["cohome"]
        summary["best_unique_id"] = sweeps["best_unique_id"]
        summary["selected_radius"] = sweeps["selected_radius"]

        print(
            "co-home residual:",
            json.dumps(sweeps["cohome"], indent=2, sort_keys=True),
        )
        print(f"{'method':<42} {'bits':>4} {'id':>8} {'recall':>8} {'meanC':>8}")
        for r in sweeps["gate"]:
            bits = r.get("n_bits", "")
            rec = r.get("candidate_recall", float("nan"))
            mean_c = r.get("mean_candidates", float("nan"))
            acc = r.get("argmin_accuracy", float("nan"))
            print(
                f"{r['method']:<42} {bits!s:>4} {acc:8.4f} {rec:8.4f} {mean_c:8.2f}"
            )
        print("--- codebook ---")
        for r in sweeps["codebook"]:
            if r.get("kind") != "candidates":
                print(
                    f"{r['method']:<42} id={r.get('argmin_accuracy', float('nan')):.4f} "
                    f"unseen={r.get('frac_unseen_bin', float('nan')):.3f}"
                )
                continue
            print(
                f"{r['method']:<42} recall={r['candidate_recall']:.4f} "
                f"meanC={r['mean_candidates']:.2f} "
                f"x={r.get('traffic_reduction_x', float('nan')):.1f} "
                f"id={r['argmin_accuracy']:.4f}"
            )
        print("--- quantized COM radius (selected @0.99) ---")
        for src, recs in sweeps["selected_radius"].items():
            hit = recs.get("r99")
            if hit is None:
                print(f"  {src}: no R with recall>=0.99")
            else:
                print(
                    f"  {src}: R={hit['radius_um']:.0f} um  "
                    f"recall={hit['candidate_recall']:.4f}  "
                    f"meanC={hit['mean_candidates']:.2f}  "
                    f"{hit['traffic_reduction_x']:.1f}x"
                )

    if args.adc:
        print("Streaming 60 s for signed ADC → COM (CPU, HDD zip)...")
        adc = run_adc_sweep(prep, float(args.duration))
        _write_csv(out_dir / "np_adc_com.csv", adc["rows"])
        summary["adc"] = {k: v for k, v in adc.items() if k != "rows"}
        summary["adc_rows"] = adc["rows"]
        print(
            f"{'scale':<22} {'bits':>4} {'rmsY':>8} {'r60id':>8} "
            f"{'cb_rec':>8} {'R80rec':>8}"
        )
        for r in adc["rows"]:
            print(
                f"{r['scale']:<22} {r['adc_bits']:>4} "
                f"{r['rms_y_um']:8.2f} {r['home_r60_residual']:8.4f} "
                f"{r['codebook_cand_4bit_pm90_recall']:8.4f} "
                f"{r['radius80_recall']:8.4f}"
            )

    json_path = out_dir / "np_com_optimize_summary.json"
    if args.skip_sweeps and json_path.exists():
        prev = json.loads(json_path.read_text())
        prev.update({k: v for k, v in summary.items() if k in ("adc", "adc_rows")})
        summary = prev
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
