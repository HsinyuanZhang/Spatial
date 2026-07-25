"""Brainstorm+verify: CiM/CAM-friendly approximations to P2P L2 in stage-1.

Compares alternative distances on max-normalized local P2P footprints against
exact L2 under:
  1) Spearman rank correlation of spike→unit distances
  2) Stage-1 candidate-set agreement (Jaccard / exact match)
  3) End-to-end two-stage oracle accuracy (COM gate unchanged; P2P metric swapped)

Usage:
    python -m Spatial.experiments.verify_p2p_cim_approx --dataset hj --duration 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    com_features,
    footprint_p2p_features,
    quantize_p2p,
)
from Spatial.algorithms.two_stage_assign import fit_unit_centroids
from Spatial.experiments.run_two_stage_upper_bound import (
    _chronological_split,
    extract_main_channel_waveforms,
    normalize_waveforms,
    peak_fsde_batch,
)


# ---------------------------------------------------------------------------
# Distance primitives (query x vs stored centroid c)
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def dist_l2(x: np.ndarray, c: np.ndarray) -> float:
    return float(np.linalg.norm(x - c))


def dist_l1(x: np.ndarray, c: np.ndarray) -> float:
    return float(np.sum(np.abs(x - c)))


def dist_cosine(x: np.ndarray, c: np.ndarray, eps: float = 1e-8) -> float:
    """1 - cosine; smaller = more similar. Works on max-normalized footprints."""
    xn = _l2_normalize(x, eps)
    cn = _l2_normalize(c, eps)
    return float(1.0 - np.dot(xn, cn))


def dist_neg_dot(x: np.ndarray, c: np.ndarray) -> float:
    """Negative inner product (CIM MAC style). Assumes non-negative footprints."""
    return float(-np.dot(x, c))


def dist_hamming_codes(x_codes: np.ndarray, c_codes: np.ndarray) -> float:
    return float(np.sum(x_codes != c_codes))


def quantize_codes(p2p: np.ndarray, n_bits: int) -> np.ndarray:
    """Per-spike max relative codes in {0..2^n-1} as int."""
    return quantize_p2p(p2p, n_bits=n_bits, mode="per_spike_max").astype(np.int64)


METRICS = (
    "l2",
    "l1",
    "cosine",
    "neg_dot",
    "l2_on_l2norm",
    "hamming_2bit",
    "hamming_3bit",
    "hamming_4bit",
    "hamming_5bit",
    "l2_4bit",
)


def pairwise_distances_to_centroids(
    feats: np.ndarray,
    centroids: dict[int, np.ndarray],
    metric: str,
    codes: np.ndarray | None = None,
    code_centroids: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Return (n_spikes, n_units) distance matrix; columns sorted by unit id."""
    units = sorted(centroids.keys())
    n = len(feats)
    out = np.zeros((n, len(units)), dtype=np.float64)
    for j, u in enumerate(units):
        c = centroids[u]
        for i in range(n):
            if metric == "l2":
                out[i, j] = dist_l2(feats[i], c)
            elif metric == "l1":
                out[i, j] = dist_l1(feats[i], c)
            elif metric == "cosine":
                out[i, j] = dist_cosine(feats[i], c)
            elif metric == "neg_dot":
                out[i, j] = dist_neg_dot(feats[i], c)
            elif metric == "l2_on_l2norm":
                out[i, j] = dist_l2(_l2_normalize(feats[i]), _l2_normalize(c))
            elif metric.startswith("hamming_"):
                assert codes is not None and code_centroids is not None
                out[i, j] = dist_hamming_codes(codes[i], code_centroids[u])
            elif metric == "l2_4bit":
                # codes treated as float levels; L2 in code space
                assert codes is not None and code_centroids is not None
                out[i, j] = dist_l2(codes[i].astype(np.float64), code_centroids[u].astype(np.float64))
            else:
                raise ValueError(metric)
    return out, units


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman on flattened finite pairs."""
    x = a.ravel()
    y = b.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def calibrate_tau(dists_to_own: np.ndarray, percentile: float = 95.0) -> float:
    d = dists_to_own[np.isfinite(dists_to_own)]
    if d.size == 0:
        return 1.0
    return max(float(np.percentile(d, percentile)), 1e-8)


def candidate_sets_from_dists(
    d_com: np.ndarray,
    d_p2p: np.ndarray,
    units: list[int],
    tau_com: float,
    tau_p2p: float,
) -> list[set[int]]:
    """Per-spike candidate sets under COM∧P2P gates."""
    out = []
    for i in range(len(d_com)):
        s = set()
        for j, u in enumerate(units):
            if d_com[i, j] < tau_com and d_p2p[i, j] < tau_p2p:
                s.add(int(u))
        out.append(s)
    return out


def set_agreement(a: list[set[int]], b: list[set[int]]) -> dict:
    exact = 0
    jaccs = []
    for sa, sb in zip(a, b):
        if sa == sb:
            exact += 1
        union = sa | sb
        if not union:
            jaccs.append(1.0)
        else:
            jaccs.append(len(sa & sb) / len(union))
    n = max(len(a), 1)
    return {
        "exact_match_rate": exact / n,
        "mean_jaccard": float(np.mean(jaccs)),
    }


def two_stage_from_dists(
    d_com_test: np.ndarray,
    d_p2p_test: np.ndarray,
    units: list[int],
    tau_com: float,
    tau_p2p: float,
    wave_test: np.ndarray,
    wave_cents: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict]:
    n = len(d_com_test)
    pred = np.full(n, -1, dtype=np.int64)
    n_fb = 0
    n_cands = []
    for i in range(n):
        cands = [
            units[j]
            for j in range(len(units))
            if d_com_test[i, j] < tau_com and d_p2p_test[i, j] < tau_p2p
        ]
        n_cands.append(len(cands))
        if cands:
            best_u, best_d = -1, np.inf
            for u in cands:
                d = float(np.linalg.norm(wave_test[i] - wave_cents[u]))
                if d < best_d:
                    best_d, best_u = d, u
            pred[i] = best_u
        else:
            n_fb += 1
            # fallback: normalized com+p2p score
            best_u, best_s = -1, np.inf
            for j, u in enumerate(units):
                s = d_com_test[i, j] / tau_com + d_p2p_test[i, j] / tau_p2p
                if s < best_s:
                    best_s, best_u = s, u
            pred[i] = best_u
    return pred, {
        "fallback_rate": n_fb / max(n, 1),
        "mean_n_candidates": float(np.mean(n_cands)),
    }


def prepare_features(dataset, k_neighbors: int = 7, window: int = 15):
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=window)
    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    labels = np.array([time_to_unit.get(int(t), -1) for t in times], dtype=np.int64)
    keep = labels >= 0
    times, central, labels = times[keep], central[keep], labels[keep]

    k = min(k_neighbors, dataset.geom.shape[0])
    nbr = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, _, neighbor_ids = extract_local_p2p(filtered, times, central, nbr, window=window)
    com = com_features(p2p_raw, neighbor_ids, dataset.geom)
    p2p = footprint_p2p_features(p2p_raw, normalize=True)  # max-normalized
    wfs = normalize_waveforms(extract_main_channel_waveforms(filtered, times, central))

    order = np.argsort(times)
    return com[order], p2p[order], p2p_raw[order], wfs[order], labels[order]


def run_verify(dataset, percentile: float = 95.0, verbose: bool = True) -> list[dict]:
    com, p2p, p2p_raw, wave, labels = prepare_features(dataset)
    train_idx, test_idx = _chronological_split(len(labels), 0.5)
    y_tr, y_te = labels[train_idx], labels[test_idx]

    com_cents = fit_unit_centroids(com[train_idx], y_tr)
    p2p_cents = fit_unit_centroids(p2p[train_idx], y_tr)
    wave_cents = fit_unit_centroids(wave[train_idx], y_tr)
    units = sorted(com_cents.keys())

    # COM distances (fixed across metrics)
    d_com_tr, _ = pairwise_distances_to_centroids(com[train_idx], com_cents, "l2")
    d_com_te, _ = pairwise_distances_to_centroids(com[test_idx], com_cents, "l2")
    # own-unit COM distances for τ_com
    unit_to_j = {u: j for j, u in enumerate(units)}
    own_com = np.array([d_com_tr[i, unit_to_j[int(y_tr[i])]] for i in range(len(y_tr))])
    tau_com = calibrate_tau(own_com, percentile)

    # L2 reference P2P distances
    d_l2_tr, _ = pairwise_distances_to_centroids(p2p[train_idx], p2p_cents, "l2")
    d_l2_te, _ = pairwise_distances_to_centroids(p2p[test_idx], p2p_cents, "l2")
    own_l2 = np.array([d_l2_tr[i, unit_to_j[int(y_tr[i])]] for i in range(len(y_tr))])
    tau_l2 = calibrate_tau(own_l2, percentile)
    ref_cands = candidate_sets_from_dists(d_com_te, d_l2_te, units, tau_com, tau_l2)

    # Precompute bit codes from max-normalized p2p (same as footprint scale)
    code_cache = {}
    code_cent_cache = {}
    for nb in (2, 3, 4, 5):
        codes_all = quantize_codes(p2p, nb)  # quantize already max-normalized footprint
        code_cache[nb] = codes_all
        code_cent_cache[nb] = fit_unit_centroids(codes_all[train_idx].astype(np.float64), y_tr)

    rows = []
    if verbose:
        print(f"Dataset: {dataset.name} | spikes={len(labels)} | units={len(units)}")
        print(f"τ_com={tau_com:.4g}  τ_l2(p2p)={tau_l2:.4g}")
        print(f"\n{'metric':<16} {'spear':>7} {'exact':>7} {'jacc':>7} "
              f"{'oracle':>8} {'Δpp':>7} {'fb':>6} {'cands':>6}")
        print("-" * 72)

    ref_oracle = None
    for metric in METRICS:
        codes = None
        code_cents = None
        feats_tr, feats_te = p2p[train_idx], p2p[test_idx]
        cents = p2p_cents

        if metric.startswith("hamming_"):
            nb = int(metric.split("_")[1].replace("bit", ""))
            codes_all = code_cache[nb]
            # CAM stores rounded mean codeword per unit
            code_cents_f = fit_unit_centroids(codes_all[train_idx].astype(np.float64), y_tr)
            code_cents = {u: np.rint(v).astype(np.int64) for u, v in code_cents_f.items()}
            d_tr, _ = pairwise_distances_to_centroids(
                feats_tr, cents, metric,
                codes=codes_all[train_idx], code_centroids=code_cents,
            )
            d_te, _ = pairwise_distances_to_centroids(
                feats_te, cents, metric,
                codes=codes_all[test_idx], code_centroids=code_cents,
            )
        elif metric == "l2_4bit":
            codes_all = code_cache[4].astype(np.float64)
            code_cents = code_cent_cache[4]
            d_tr, _ = pairwise_distances_to_centroids(
                feats_tr, cents, metric, codes=codes_all[train_idx], code_centroids=code_cents
            )
            d_te, _ = pairwise_distances_to_centroids(
                feats_te, cents, metric, codes=codes_all[test_idx], code_centroids=code_cents
            )
        else:
            d_tr, _ = pairwise_distances_to_centroids(feats_tr, cents, metric)
            d_te, _ = pairwise_distances_to_centroids(feats_te, cents, metric)

        spear = spearman_corr(d_l2_te, d_te)
        own = np.array([d_tr[i, unit_to_j[int(y_tr[i])]] for i in range(len(y_tr))])
        tau_p = calibrate_tau(own, percentile)
        cands = candidate_sets_from_dists(d_com_te, d_te, units, tau_com, tau_p)
        agree = set_agreement(ref_cands, cands)

        pred, meta = two_stage_from_dists(
            d_com_te, d_te, units, tau_com, tau_p, wave[test_idx], wave_cents
        )
        acc = float(np.mean(pred == y_te))
        if metric == "l2":
            ref_oracle = acc
        delta = (acc - ref_oracle) * 100 if ref_oracle is not None else 0.0

        row = {
            "metric": metric,
            "spearman_vs_l2": spear,
            "cand_exact_match": agree["exact_match_rate"],
            "cand_mean_jaccard": agree["mean_jaccard"],
            "two_stage_oracle": acc,
            "delta_pp_vs_l2": delta,
            "fallback_rate": meta["fallback_rate"],
            "mean_n_candidates": meta["mean_n_candidates"],
            "tau_p2p": tau_p,
            "tau_com": tau_com,
        }
        rows.append(row)
        if verbose:
            print(
                f"{metric:<16} {spear:>7.3f} {agree['exact_match_rate']:>7.3f} "
                f"{agree['mean_jaccard']:>7.3f} {acc:>8.3f} {delta:>+7.2f} "
                f"{meta['fallback_rate']:>6.3f} {meta['mean_n_candidates']:>6.2f}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hj"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Spatial/output/p2p_cim_approx_hj60s.csv"),
    )
    args = parser.parse_args()
    ds = load_hybrid_janelia(duration_s=args.duration)
    rows = run_verify(ds, percentile=args.percentile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
