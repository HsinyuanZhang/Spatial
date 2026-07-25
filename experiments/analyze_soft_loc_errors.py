"""Analyze soft_loc misclassifications and whether Peak-FSDE (OSort) separates them.

Usage:
    python -m Spatial.experiments.analyze_soft_loc_errors --dataset hj --duration 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_cortexlab_npz
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    soft_localization_features,
)
from GeoOsort.feature_extractor import FeatureExtractor


def _chronological_split(n: int, train_frac: float = 0.5):
    n_train = max(1, int(n * train_frac))
    if n_train >= n and n > 1:
        n_train = n - 1
    return np.arange(0, n_train), np.arange(n_train, n)


def extract_main_channel_waveforms(
    signal: np.ndarray,
    times: np.ndarray,
    central_channels: np.ndarray,
    pre: int = 15,
    post: int = 49,
) -> np.ndarray:
    """Extract 1D main-channel snippets (default 64 samples, OSort-style)."""
    n = len(times)
    win = pre + post
    out = np.zeros((n, win), dtype=np.float64)
    n_ch, n_samp = signal.shape
    for i, (t, ch) in enumerate(zip(times, central_channels)):
        ch = int(np.clip(ch, 0, n_ch - 1))
        s = int(t - pre)
        e = int(t + post)
        if s < 0 or e > n_samp:
            continue
        out[i] = signal[ch, s:e]
    return out


def peak_fsde_batch(waveforms: np.ndarray, normalize: bool = True) -> np.ndarray:
    ext = FeatureExtractor(normalize=normalize)
    return ext.process_batch([w for w in waveforms])


def fit_centroids(features: np.ndarray, labels: np.ndarray):
    units = np.unique(labels)
    cents = {}
    for u in units:
        mask = labels == u
        if np.any(mask):
            cents[int(u)] = np.mean(features[mask], axis=0)
    return cents


def assign_nearest(features: np.ndarray, centroids: dict) -> np.ndarray:
    units = np.array(sorted(centroids.keys()), dtype=np.int64)
    mat = np.stack([centroids[u] for u in units], axis=0)
    dists = np.linalg.norm(features[:, None, :] - mat[None, :, :], axis=2)
    return units[np.argmin(dists, axis=1)]


def pairwise_oracle_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    unit_a: int,
    unit_b: int,
    train_frac: float = 0.5,
) -> dict:
    """2-class GT-centroid oracle on spikes from only these two units."""
    mask = (labels == unit_a) | (labels == unit_b)
    feat = features[mask]
    lab = labels[mask]
    if len(feat) < 4 or len(np.unique(lab)) < 2:
        return {"accuracy": float("nan"), "n": int(len(feat)), "n_err": -1}

    order = np.arange(len(feat))  # already chronological if parent was sorted
    n = len(feat)
    n_train = max(1, int(n * train_frac))
    if n_train >= n:
        n_train = n - 1
    train_f, train_l = feat[:n_train], lab[:n_train]
    test_f, test_l = feat[n_train:], lab[n_train:]
    if len(np.unique(train_l)) < 2 or len(test_f) == 0:
        return {"accuracy": float("nan"), "n": int(n), "n_err": -1}

    cents = fit_centroids(train_f, train_l)
    pred = assign_nearest(test_f, cents)
    n_err = int(np.sum(pred != test_l))
    acc = float(np.mean(pred == test_l))
    return {"accuracy": acc, "n": int(n), "n_test": int(len(test_l)), "n_err": n_err}


def unit_stats(features: np.ndarray, labels: np.ndarray, main_ch: np.ndarray, soft: np.ndarray):
    """Per-unit summary: count, main channel mode, mean soft_loc."""
    rows = {}
    for u in np.unique(labels):
        m = labels == u
        ch_mode = int(np.bincount(main_ch[m]).argmax())
        rows[int(u)] = {
            "n": int(np.sum(m)),
            "main_ch": ch_mode,
            "main_ch_frac": float(np.mean(main_ch[m] == ch_mode)),
            "cy_mean": float(np.mean(soft[m, 1])),
            "cy_std": float(np.std(soft[m, 1])),
            "spread_mean": float(np.mean(soft[m, 2])),
            "sharp_mean": float(np.mean(soft[m, 3])),
            "centroid": np.mean(features[m], axis=0),
        }
    return rows


def analyze(dataset, train_frac: float = 0.5, k_neighbors: int = 7, verbose: bool = True):
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    amps, times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=15)

    time_to_unit = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    labels = np.array([time_to_unit.get(int(t), -1) for t in times], dtype=np.int64)
    keep = labels >= 0
    times, central, labels, amps = times[keep], central[keep], labels[keep], amps[keep]

    nbr = build_knn_table_with_self(dataset.geom, min(k_neighbors, dataset.n_channels))
    p2p, times2, neighbor_ids = extract_local_p2p(filtered, times, central, nbr, window=15)
    assert len(times2) == len(times)
    soft = soft_localization_features(p2p, neighbor_ids, dataset.geom)

    # Chronological order
    order = np.argsort(times)
    soft, labels, times, central = soft[order], labels[order], times[order], central[order]

    train_idx, test_idx = _chronological_split(len(soft), train_frac)
    cents = fit_centroids(soft[train_idx], labels[train_idx])
    pred = assign_nearest(soft[test_idx], cents)
    y_true = labels[test_idx]
    y_pred = pred
    err_mask = y_true != y_pred
    n_test = len(y_true)
    n_err = int(np.sum(err_mask))
    oracle_acc = 1.0 - n_err / n_test

    if verbose:
        print(f"Dataset: {dataset.name} | spikes={len(soft)} | units={len(np.unique(labels))}")
        print(f"soft_loc oracle: acc={oracle_acc:.3f}  errors={n_err}/{n_test}")

    # Confusion among errors
    err_true = y_true[err_mask]
    err_pred = y_pred[err_mask]
    pair_counts = {}
    for a, b in zip(err_true, err_pred):
        key = (int(a), int(b))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # Symmetric pair totals (a<->b)
    sym = {}
    for (a, b), c in pair_counts.items():
        key = tuple(sorted((a, b)))
        sym[key] = sym.get(key, 0) + c
    ranked_pairs = sorted(sym.items(), key=lambda x: -x[1])

    stats = unit_stats(soft, labels, central, soft)

    # Peak-FSDE on main channel
    wfs = extract_main_channel_waveforms(filtered, times, central)
    fsde = peak_fsde_batch(wfs, normalize=True)
    soft_fsde = np.hstack([soft, fsde])

    # Standardize each block for fair concat distances? Use raw; report both.
    # z-score soft_fsde for combined oracle
    def zscore(x):
        mu = np.mean(x, axis=0)
        sd = np.std(x, axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (x - mu) / sd

    soft_z = zscore(soft)
    fsde_z = zscore(fsde)
    combo_z = np.hstack([soft_z, fsde_z])

    # Global oracles
    def global_oracle(feats):
        c = fit_centroids(feats[train_idx], labels[train_idx])
        p = assign_nearest(feats[test_idx], c)
        return float(np.mean(p == labels[test_idx]))

    acc_soft = global_oracle(soft)
    acc_fsde = global_oracle(fsde)
    acc_combo = global_oracle(combo_z)

    if verbose:
        print("\n=== Global GT-centroid oracle ===")
        print(f"  soft_loc only : {acc_soft:.3f}")
        print(f"  Peak-FSDE only: {acc_fsde:.3f}")
        print(f"  soft_loc+FSDE : {acc_combo:.3f}  (z-scored concat)")

    # Per-unit soft_loc stats
    if verbose:
        print("\n=== Per-unit soft_loc profile ===")
        print(f"{'unit':>6} {'n':>5} {'ch':>4} {'ch%':>6} {'cy':>7} {'cy_std':>7} {'spr':>7} {'shp':>7}")
        for u in sorted(stats):
            s = stats[u]
            print(
                f"{u:>6} {s['n']:>5} {s['main_ch']:>4} {s['main_ch_frac']:>6.2f} "
                f"{s['cy_mean']:>7.3f} {s['cy_std']:>7.3f} {s['spread_mean']:>7.3f} {s['sharp_mean']:>7.3f}"
            )

    # Top confused pairs
    if verbose:
        print("\n=== Top soft_loc confusion pairs (test-set errors, symmetric) ===")
        print(
            f"{'pair':<12} {'n_err':>5} {'same_ch':>8} {'d_cy':>7} "
            f"{'soft2':>7} {'fsde2':>7} {'combo2':>7}"
        )

    pair_rows = []
    for (ua, ub), n_pair_err in ranked_pairs[:15]:
        sa, sb = stats[ua], stats[ub]
        same_ch = sa["main_ch"] == sb["main_ch"]
        d_cy = abs(sa["cy_mean"] - sb["cy_mean"])
        # overlap heuristic: distance vs pooled std
        pooled = np.sqrt(0.5 * (sa["cy_std"] ** 2 + sb["cy_std"] ** 2)) + 1e-12
        cy_sep = d_cy / pooled

        soft2 = pairwise_oracle_accuracy(soft, labels, ua, ub, train_frac)
        fsde2 = pairwise_oracle_accuracy(fsde, labels, ua, ub, train_frac)
        combo2 = pairwise_oracle_accuracy(combo_z, labels, ua, ub, train_frac)

        row = {
            "unit_a": ua,
            "unit_b": ub,
            "n_confusion_errors": n_pair_err,
            "same_main_channel": same_ch,
            "ch_a": sa["main_ch"],
            "ch_b": sb["main_ch"],
            "d_cy": d_cy,
            "cy_sep_z": cy_sep,
            "soft_pair_acc": soft2["accuracy"],
            "fsde_pair_acc": fsde2["accuracy"],
            "combo_pair_acc": combo2["accuracy"],
            "n_pair_spikes": soft2["n"],
        }
        pair_rows.append(row)
        if verbose:
            print(
                f"{ua}-{ub:<9} {n_pair_err:>5} {str(same_ch):>8} {d_cy:>7.3f} "
                f"{soft2['accuracy']:>7.3f} {fsde2['accuracy']:>7.3f} {combo2['accuracy']:>7.3f}"
            )

    # Error attribution buckets
    n_same_ch_err = 0
    n_diff_ch_err = 0
    for a, b in zip(err_true, err_pred):
        if stats[int(a)]["main_ch"] == stats[int(b)]["main_ch"]:
            n_same_ch_err += 1
        else:
            n_diff_ch_err += 1

    # Among remaining soft_loc errors, how many does FSDE/combo fix?
    # Recompute test predictions
    pred_soft = assign_nearest(soft[test_idx], fit_centroids(soft[train_idx], labels[train_idx]))
    pred_fsde = assign_nearest(fsde[test_idx], fit_centroids(fsde[train_idx], labels[train_idx]))
    pred_combo = assign_nearest(combo_z[test_idx], fit_centroids(combo_z[train_idx], labels[train_idx]))
    yt = labels[test_idx]
    soft_err = pred_soft != yt
    fsde_fixes = soft_err & (pred_fsde == yt)
    combo_fixes = soft_err & (pred_combo == yt)
    still_hard = soft_err & (pred_combo != yt)

    if verbose:
        print("\n=== soft_loc error rescue by OSort Peak-FSDE ===")
        print(f"  soft_loc test errors     : {int(np.sum(soft_err))}")
        print(f"    same main-channel      : {n_same_ch_err} ({100*n_same_ch_err/max(n_err,1):.1f}%)")
        print(f"    different main-channel : {n_diff_ch_err} ({100*n_diff_ch_err/max(n_err,1):.1f}%)")
        print(f"  fixed by FSDE alone      : {int(np.sum(fsde_fixes))} ({100*np.mean(fsde_fixes[soft_err]) if n_err else 0:.1f}% of soft errors)")
        print(f"  fixed by soft+FSDE       : {int(np.sum(combo_fixes))} ({100*np.mean(combo_fixes[soft_err]) if n_err else 0:.1f}% of soft errors)")
        print(f"  still wrong after combo  : {int(np.sum(still_hard))}")

        # Which pairs remain hard under combo?
        rem = {}
        for a, b in zip(yt[still_hard], pred_combo[still_hard]):
            key = tuple(sorted((int(a), int(b))))
            rem[key] = rem.get(key, 0) + 1
        print("  remaining confusion pairs:")
        for (a, b), c in sorted(rem.items(), key=lambda x: -x[1])[:10]:
            print(f"    {a}-{b}: {c}")

    return {
        "oracle_soft": acc_soft,
        "oracle_fsde": acc_fsde,
        "oracle_combo": acc_combo,
        "n_err": n_err,
        "n_test": n_test,
        "pair_rows": pair_rows,
        "stats": stats,
        "n_same_ch_err": n_same_ch_err,
        "n_diff_ch_err": n_diff_ch_err,
        "n_fsde_fixes": int(np.sum(fsde_fixes)),
        "n_combo_fixes": int(np.sum(combo_fixes)),
        "n_still_hard": int(np.sum(still_hard)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hj", "cortexlab"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=Path("Spatial/docs/soft_loc_error_analysis.md"),
    )
    args = parser.parse_args()

    if args.dataset == "hj":
        ds = load_hybrid_janelia(duration_s=args.duration)
    else:
        ds = load_cortexlab_npz(duration_s=args.duration)

    result = analyze(ds, train_frac=args.train_frac, verbose=True)

    # Write markdown report
    doc = args.write_doc
    doc.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# soft_loc Error Analysis + OSort Peak-FSDE Rescue\n")
    lines.append(f"Dataset: `{ds.name}`, duration={args.duration:.0f}s, GT spike times.\n")
    lines.append("## Global ceilings\n")
    lines.append("| Features | GT-centroid oracle |")
    lines.append("|----------|--------------------|")
    lines.append(f"| soft_loc only | {result['oracle_soft']:.1%} |")
    lines.append(f"| Peak-FSDE (OSort) only | {result['oracle_fsde']:.1%} |")
    lines.append(f"| soft_loc + Peak-FSDE (z-concat) | **{result['oracle_combo']:.1%}** |")
    lines.append("")
    lines.append("## What are soft_loc errors?\n")
    lines.append(
        f"Test-set soft_loc errors: **{result['n_err']}/{result['n_test']}** "
        f"({100*result['n_err']/max(result['n_test'],1):.1f}%).\n"
    )
    lines.append(
        f"- Same main-channel confusions: **{result['n_same_ch_err']}** "
        f"({100*result['n_same_ch_err']/max(result['n_err'],1):.1f}%)\n"
        f"- Different main-channel confusions: **{result['n_diff_ch_err']}** "
        f"({100*result['n_diff_ch_err']/max(result['n_err'],1):.1f}%)\n"
    )
    lines.append("## Top confusion pairs\n")
    lines.append(
        "| Pair | #err | Same ch? | ch_a/ch_b | Δcy | soft 2-class | FSDE 2-class | soft+FSDE |"
    )
    lines.append("|------|------|----------|-----------|-----|--------------|--------------|-----------|")
    for r in result["pair_rows"][:12]:
        lines.append(
            f"| {r['unit_a']}-{r['unit_b']} | {r['n_confusion_errors']} | "
            f"{'yes' if r['same_main_channel'] else 'no'} | "
            f"{r['ch_a']}/{r['ch_b']} | {r['d_cy']:.3f} | "
            f"{r['soft_pair_acc']:.1%} | {r['fsde_pair_acc']:.1%} | "
            f"{r['combo_pair_acc']:.1%} |"
        )
    lines.append("")
    lines.append("## Does OSort information make them separable?\n")
    lines.append(
        f"- Of soft_loc errors, Peak-FSDE alone fixes **{result['n_fsde_fixes']}** "
        f"({100*result['n_fsde_fixes']/max(result['n_err'],1):.1f}%).\n"
        f"- soft_loc+FSDE fixes **{result['n_combo_fixes']}** "
        f"({100*result['n_combo_fixes']/max(result['n_err'],1):.1f}%).\n"
        f"- Still wrong after combo: **{result['n_still_hard']}**.\n"
    )
    lines.append(
        "Interpretation: if a confusion pair has low `soft 2-class` but high "
        "`FSDE 2-class` / `soft+FSDE`, those units share a spatial footprint "
        "(often same main channel + overlapping COM) but differ in main-channel "
        "waveform shape — exactly where OSort Peak-FSDE helps.\n"
    )
    lines.append("## Per-unit soft_loc profile\n")
    lines.append("| unit | n | main_ch | ch% | cy | cy_std | spread | sharp |")
    lines.append("|------|---|---------|-----|----|--------|--------|-------|")
    for u, s in sorted(result["stats"].items()):
        lines.append(
            f"| {u} | {s['n']} | {s['main_ch']} | {s['main_ch_frac']:.0%} | "
            f"{s['cy_mean']:.3f} | {s['cy_std']:.3f} | {s['spread_mean']:.3f} | "
            f"{s['sharp_mean']:.3f} |"
        )
    doc.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {doc}")


if __name__ == "__main__":
    main()
