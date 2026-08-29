"""Is the KS4 cluster count oversplit? Spatial, template, and time diagnostics.

358 in the COM study is the number of KS4 cluster ids in the 60 s train
split, not a biological cell count. This script looks at whether nearby
clusters look like one cell split in two (high template correlation, close
positions, shared refractory) versus two cells.

KS4 was run with nblocks=0 (no drift-block tracking).

    python -m Spatial.experiments.analyze_ks4_unit_splits
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from Spatial.data.yger_zenodo import DEFAULT_KS4_TH13_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "ks4_unit_splits"
REFRAC_SAMPLES = 30  # 1.5 ms at 20 kHz
FS = 20000


def load_cluster_tsv(path: Path, value_cast=str) -> dict[int, object]:
    rows = Path(path).read_text().splitlines()
    out: dict[int, object] = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        cid, val = line.split("\t", 1)
        out[int(cid)] = value_cast(val)
    return out


def count_cross_within(times_a: np.ndarray, times_b: np.ndarray, window: int) -> int:
    """Pairs with |t_a - t_b| <= window samples. Both arrays must be sorted."""
    a = np.asarray(times_a, dtype=np.int64).ravel()
    b = np.asarray(times_b, dtype=np.int64).ravel()
    if a.size == 0 or b.size == 0:
        return 0
    lo = np.searchsorted(b, a - int(window), side="left")
    hi = np.searchsorted(b, a + int(window), side="right")
    return int(np.sum(hi - lo))


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j in edges:
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[a] = b
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(v) for v in groups.values()]


def _binned_rates(times: np.ndarray, n_bins: int, t_max: int) -> np.ndarray:
    if t_max <= 0:
        return np.zeros(n_bins, dtype=np.float64)
    edges = np.linspace(0, t_max, n_bins + 1)
    hist, _ = np.histogram(times, bins=edges)
    return hist.astype(np.float64)


def analyze_splits(
    ks4_dir: Path,
    *,
    duration_s: float | None = None,
    max_dist_um: float = 30.0,
    min_template_sim: float = 0.80,
    refrac_ratio: float = 0.5,
) -> dict:
    ks4_dir = Path(ks4_dir)
    times = np.load(ks4_dir / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(ks4_dir / "spike_clusters.npy").reshape(-1).astype(np.int64)
    pos = np.load(ks4_dir / "spike_positions.npy").astype(np.float64)
    sim = np.load(ks4_dir / "similar_templates.npy").astype(np.float64)
    templates = np.load(ks4_dir / "templates.npy")
    labels = load_cluster_tsv(ks4_dir / "cluster_KSLabel.tsv", str)
    if duration_s is not None:
        keep = times < int(duration_s * FS)
        times, clusters, pos = times[keep], clusters[keep], pos[keep]
    t_max = int(times.max()) + 1 if times.size else 1
    units = np.array(sorted(np.unique(clusters)), dtype=np.int64)
    n_u = int(units.size)
    unit_index = {int(u): i for i, u in enumerate(units)}

    xy = np.zeros((n_u, 2), dtype=np.float64)
    n_sp = np.zeros(n_u, dtype=np.int64)
    t_by_unit: list[np.ndarray] = []
    for i, u in enumerate(units):
        sel = clusters == u
        n_sp[i] = int(sel.sum())
        xy[i] = np.median(pos[sel], axis=0)
        t_by_unit.append(np.sort(times[sel]))

    # Template cosine on flattened waveforms (KS similar_templates is the
    # official pairwise score; cosine is a check that the index alignment
    # cluster_id == template_id holds).
    flat = templates.reshape(templates.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    nn_dist = np.full(n_u, np.inf)
    nn_id = np.full(n_u, -1, dtype=np.int64)
    for i in range(n_u):
        d = np.linalg.norm(xy - xy[i], axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        nn_dist[i] = float(d[j])
        nn_id[i] = j

    pair_rows = []
    for i in range(n_u):
        for j in range(i + 1, n_u):
            dist = float(np.linalg.norm(xy[i] - xy[j]))
            if dist > float(max_dist_um):
                continue
            ui, uj = int(units[i]), int(units[j])
            s = float(sim[ui, uj]) if ui < sim.shape[0] and uj < sim.shape[1] else float("nan")
            ta, tb = t_by_unit[i], t_by_unit[j]
            obs = count_cross_within(ta, tb, REFRAC_SAMPLES)
            exp = float(ta.size * tb.size * (2 * REFRAC_SAMPLES) / max(t_max, 1))
            ratio = float(obs / exp) if exp > 0 else float("nan")
            mid = t_max / 2.0
            p_early_a = float(np.mean(ta < mid)) if ta.size else float("nan")
            p_early_b = float(np.mean(tb < mid)) if tb.size else float("nan")
            rates_a = _binned_rates(ta, 20, t_max)
            rates_b = _binned_rates(tb, 20, t_max)
            if rates_a.std() > 0 and rates_b.std() > 0:
                rate_corr = float(np.corrcoef(rates_a, rates_b)[0, 1])
            else:
                rate_corr = float("nan")
            lab_a = str(labels.get(ui, ""))
            lab_b = str(labels.get(uj, ""))
            mergeish = (
                np.isfinite(s)
                and s >= float(min_template_sim)
                and np.isfinite(ratio)
                and ratio < float(refrac_ratio)
            )
            pair_rows.append(
                {
                    "unit_a": ui,
                    "unit_b": uj,
                    "dist_um": dist,
                    "template_sim": s,
                    "n_a": int(ta.size),
                    "n_b": int(tb.size),
                    "cross_refrac_obs": obs,
                    "cross_refrac_exp": exp,
                    "cross_refrac_ratio": ratio,
                    "early_frac_a": p_early_a,
                    "early_frac_b": p_early_b,
                    "early_frac_absdiff": abs(p_early_a - p_early_b),
                    "rate_corr_20bins": rate_corr,
                    "ks_label_a": lab_a,
                    "ks_label_b": lab_b,
                    "both_good": int(lab_a == "good" and lab_b == "good"),
                    "merge_candidate": int(bool(mergeish)),
                    "time_split_support": int(
                        bool(mergeish and abs(p_early_a - p_early_b) >= 0.35)
                    ),
                }
            )

    merge_edges = [
        (unit_index[int(r["unit_a"])], unit_index[int(r["unit_b"])])
        for r in pair_rows
        if r["merge_candidate"]
    ]
    comps = connected_components(n_u, merge_edges)
    n_merged_groups = sum(1 for c in comps if len(c) > 1)
    n_after_merge = len(comps)

    def _pct(arr: np.ndarray) -> dict[str, float]:
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {k: float("nan") for k in ("min", "p10", "p50", "p90", "p99", "max")}
        qs = np.percentile(x, [0, 10, 50, 90, 99, 100])
        return {
            "min": float(qs[0]),
            "p10": float(qs[1]),
            "p50": float(qs[2]),
            "p90": float(qs[3]),
            "p99": float(qs[4]),
            "max": float(qs[5]),
        }

    n_good = int(sum(1 for u in units if labels.get(int(u)) == "good"))
    n_mua = int(n_u - n_good)
    nearby = pair_rows
    merge_n = int(sum(r["merge_candidate"] for r in nearby))
    time_split_n = int(sum(r["time_split_support"] for r in nearby))
    close15 = [r for r in nearby if r["dist_um"] <= 15.0]
    close15_highsim = [r for r in close15 if r["template_sim"] >= float(min_template_sim)]

    summary = {
        "ks4_dir": str(ks4_dir),
        "duration_s": None if duration_s is None else float(duration_s),
        "n_spikes": int(times.size),
        "n_clusters": n_u,
        "n_good": n_good,
        "n_mua": n_mua,
        "nblocks": 0,
        "centroid_nn_um": _pct(nn_dist),
        "frac_units_nn_lt_15um": float(np.mean(nn_dist < 15.0)),
        "frac_units_nn_lt_30um": float(np.mean(nn_dist < 30.0)),
        "n_pairs_within_30um": len(nearby),
        "n_pairs_within_15um": len(close15),
        "n_pairs_15um_and_sim_ge_thr": len(close15_highsim),
        "n_merge_candidate_pairs": merge_n,
        "n_merge_pairs_with_time_split": time_split_n,
        "merge_rule": {
            "max_dist_um": float(max_dist_um),
            "min_template_sim": float(min_template_sim),
            "max_cross_refrac_obs_over_exp": float(refrac_ratio),
            "refrac_ms": REFRAC_SAMPLES * 1000.0 / FS,
        },
        "n_connected_merge_groups": n_merged_groups,
        "n_units_after_greedy_merge": n_after_merge,
        "n_units_absorbed_in_merges": int(n_u - n_after_merge),
        "note": (
            "Merge candidates are KS4 cluster pairs that are spatially close, "
            "have high similar_templates, and fire together much less than an "
            "independent Poisson pair (shared refractory). This is not a "
            "Phy auto-merge and not biological ground truth."
        ),
    }
    return {
        "summary": summary,
        "pairs": pair_rows,
        "units": units,
        "xy": xy,
        "nn_dist": nn_dist,
        "n_sp": n_sp,
        "components": comps,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks4-dir", type=Path, default=DEFAULT_KS4_TH13_DIR)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = full recording")
    parser.add_argument("--max-dist-um", type=float, default=30.0)
    parser.add_argument("--min-template-sim", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    duration = None if float(args.duration) <= 0 else float(args.duration)
    result = analyze_splits(
        args.ks4_dir,
        duration_s=duration,
        max_dist_um=float(args.max_dist_um),
        min_template_sim=float(args.min_template_sim),
    )
    summary = result["summary"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = result["pairs"]
    csv_path = out_dir / "nearby_cluster_pairs.csv"
    if pairs:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)
    json_path = out_dir / "ks4_unit_splits_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"KS4 clusters: {summary['n_clusters']} "
        f"(good={summary['n_good']}, mua={summary['n_mua']}) | "
        f"spikes={summary['n_spikes']}"
    )
    nn = summary["centroid_nn_um"]
    print(
        f"Median position NN: {nn['p50']:.2f} um "
        f"(p10 {nn['p10']:.2f}, p90 {nn['p90']:.2f}) | "
        f"{summary['frac_units_nn_lt_30um']:.1%} have a neighbor <30 um"
    )
    print(
        f"Pairs <30 um: {summary['n_pairs_within_30um']} | "
        f"<15 um AND sim>={args.min_template_sim}: "
        f"{summary['n_pairs_15um_and_sim_ge_thr']}"
    )
    print(
        f"Merge-candidate pairs (close + sim>={args.min_template_sim} + "
        f"cross-refrac ratio<0.5): {summary['n_merge_candidate_pairs']} "
        f"({summary['n_merge_pairs_with_time_split']} also time-split)"
    )
    print(
        f"Greedy merge of those edges: {summary['n_connected_merge_groups']} groups, "
        f"{summary['n_units_absorbed_in_merges']} clusters absorbed, "
        f"{summary['n_units_after_greedy_merge']} ids left"
    )
    merge_pairs = [r for r in pairs if r["merge_candidate"]]
    merge_pairs.sort(key=lambda r: -r["template_sim"])
    print("Top merge-looking pairs:")
    for r in merge_pairs[:12]:
        print(
            f"  {r['unit_a']:4d}–{r['unit_b']:4d}  {r['dist_um']:5.1f} um  "
            f"sim={r['template_sim']:.3f}  refrac={r['cross_refrac_ratio']:.2f}  "
            f"Δearly={r['early_frac_absdiff']:.2f}  "
            f"rate_r={r['rate_corr_20bins']:.2f}  "
            f"{r['ks_label_a']}/{r['ks_label_b']}"
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
