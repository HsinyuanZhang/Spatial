"""Cross-scene validation of today's Spatial methods on all ready 16ch HJ NPZs.

Runs:
  1) Spatial-only upper bound (main_channel / akhoundi5 / soft_loc / footprint_*)
  2) Two-stage COM∧P2P → FSDE/waveform (P2P metric = L2 and L1)

Usage:
  python -m Spatial.experiments.run_hj16_method_sweep --duration 60
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia
from Spatial.experiments.run_spatial_upper_bound import run_study as run_spatial_study
from Spatial.experiments.run_two_stage_upper_bound import run_study as run_two_stage_study


def select_16ch_scenes(ready: list[dict]) -> list[dict]:
    out = [
        e for e in ready
        if int(e.get("n_channels", 0)) == 16 and int(e.get("duration_s", 0)) == 600
    ]
    out.sort(key=lambda e: e.get("short_name") or e.get("scene_key") or "")
    return out


def pivot_mean(rows: list[dict], method_key: str, value_key: str) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[r[method_key]].append(float(r[value_key]))
    return buckets


def print_spatial_summary(rows: list[dict]):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])

    print("\n=== Spatial-only oracle (GT-centroid) ===")
    header = f"{'scene':<22}" + "".join(f"{m[:10]:>11}" for m in methods)
    print(header)
    print("-" * len(header))
    for sc in scenes:
        cells = []
        for m in methods:
            hit = next(r for r in rows if r["scene"] == sc and r["method"] == m)
            cells.append(f"{hit['oracle_accuracy']:>11.3f}")
        print(f"{sc:<22}" + "".join(cells))

    print("\nMean ± std across scenes (oracle):")
    for m in methods:
        vals = [r["oracle_accuracy"] for r in rows if r["method"] == m]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        print(f"  {m:<16} {mu:.3f} ± {sd:.3f}")

    print("\nMean ± std across scenes (SOM):")
    for m in methods:
        vals = [r["som_accuracy"] for r in rows if r["method"] == m]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        print(f"  {m:<16} {mu:.3f} ± {sd:.3f}")


def print_two_stage_summary(rows: list[dict]):
    focus = [
        "com_and_p2p",
        "fsde_only",
        "waveform_only",
        "concat_com_p2p_fsde",
        "concat_com_p2p_waveform",
        "two_stage_fsde",
        "two_stage_waveform",
    ]
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])

    for metric in ("l2", "l1"):
        sub = [r for r in rows if r["p2p_metric"] == metric]
        if not sub:
            continue
        print(f"\n=== Two-stage oracle (P2P={metric}) ===")
        header = f"{'scene':<22}" + "".join(f"{m[:9]:>10}" for m in focus)
        print(header)
        print("-" * len(header))
        for sc in scenes:
            cells = []
            for m in focus:
                hit = next(
                    (r for r in sub if r["scene"] == sc and r["method"] == m),
                    None,
                )
                cells.append(f"{hit['oracle_accuracy']:>10.3f}" if hit else f"{'—':>10}")
            print(f"{sc:<22}" + "".join(cells))

        print(f"\nMean ± std (P2P={metric}):")
        for m in focus:
            vals = [r["oracle_accuracy"] for r in sub if r["method"] == m]
            if not vals:
                continue
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            print(f"  {m:<28} {mu:.3f} ± {sd:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Sweep today's methods on all 16ch HJ scenes")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--p2p-metrics",
        nargs="+",
        default=["l2", "l1"],
        choices=["l2", "l1"],
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Only run two-stage study",
    )
    parser.add_argument(
        "--skip-two-stage",
        action="store_true",
        help="Only run spatial-only study",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Spatial/output/hj16_method_sweep"),
    )
    args = parser.parse_args()

    ready = list_hybrid_janelia_scenes(ready_only=True)
    scenes = select_16ch_scenes(ready)
    if not scenes:
        raise SystemExit("No ready 16ch 600s scenes. Run prepare_hybrid_janelia first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scenes ({len(scenes)}), duration={args.duration}s")
    for e in scenes:
        print(f"  - {e.get('short_name')}  units={e.get('n_valid_units')}")

    spatial_rows: list[dict] = []
    two_stage_rows: list[dict] = []
    t0 = time.time()

    for e in scenes:
        short = e.get("short_name") or e["scene_key"]
        print(f"\n########## {short} ##########")
        ds = load_hybrid_janelia(scene=short, duration_s=args.duration)

        if not args.skip_spatial:
            print("--- spatial upper bound ---")
            rows = run_spatial_study(ds, verbose=True)
            for r in rows:
                spatial_rows.append({
                    "scene": short,
                    "drift": e.get("drift"),
                    "n_units": r["gt_units"],
                    "n_spikes": r["n_spikes"],
                    "method": r["method"],
                    "oracle_accuracy": r["oracle_accuracy"],
                    "som_accuracy": r["som_accuracy"],
                    "sep_ratio": r["sep_ratio"],
                    "som_clusters": r["som_clusters"],
                })

        if not args.skip_two_stage:
            for metric in args.p2p_metrics:
                print(f"--- two-stage (p2p={metric}) ---")
                rows = run_two_stage_study(
                    ds,
                    percentile=args.percentile,
                    p2p_metric=metric,
                    verbose=True,
                )
                for r in rows:
                    two_stage_rows.append({
                        "scene": short,
                        "drift": e.get("drift"),
                        "n_units": int(ds.unit_ids.size),
                        "n_test": r["n_test"],
                        "p2p_metric": metric,
                        "method": r["method"],
                        "oracle_accuracy": r["oracle_accuracy"],
                        "fallback_rate": r.get("fallback_rate", ""),
                        "mean_n_candidates": r.get("mean_n_candidates", ""),
                        "tau_com": r["tau_com"],
                        "tau_p2p": r["tau_p2p"],
                    })

    if spatial_rows:
        path = args.output_dir / f"spatial_upper_bound_hj16_{int(args.duration)}s.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spatial_rows[0].keys()))
            w.writeheader()
            w.writerows(spatial_rows)
        print(f"\nSaved: {path}")
        print_spatial_summary(spatial_rows)

    if two_stage_rows:
        path = args.output_dir / f"two_stage_hj16_{int(args.duration)}s.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(two_stage_rows[0].keys()))
            w.writeheader()
            w.writerows(two_stage_rows)
        print(f"\nSaved: {path}")
        print_two_stage_summary(two_stage_rows)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
