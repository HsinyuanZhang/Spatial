"""Validate today's Spatial methods on MEArec v1 GeoOsort NPZs (32ch).

Dataset root (default):
  synthetic_spike_dataset/generated/geoosort_npz_v1/

Runs spatial-only upper bound + two-stage COM∧P2P→FSDE/waveform (L2 & L1).

Usage:
  python -m Spatial.experiments.run_mearec_method_sweep --duration 60
  python -m Spatial.experiments.run_mearec_method_sweep --seeds 202600  # one seed only
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_mearec_npz
from Spatial.experiments.run_spatial_upper_bound import run_study as run_spatial_study
from Spatial.experiments.run_two_stage_upper_bound import run_study as run_two_stage_study

DEFAULT_NPZ_DIR = (
    Path(__file__).resolve().parents[2]
    / "synthetic_spike_dataset"
    / "generated"
    / "geoosort_npz_v1"
)


def parse_rec_name(stem: str) -> dict:
    """Parse rec_v1_units10_snr8_seed202600 → fields."""
    m = re.match(
        r"rec_v1_units(?P<n_units>\d+)_snr(?P<snr>\d+)_seed(?P<seed>\d+)$",
        stem,
    )
    if not m:
        return {"rec": stem, "n_units_design": None, "snr": None, "seed": None}
    return {
        "rec": stem,
        "n_units_design": int(m.group("n_units")),
        "snr": int(m.group("snr")),
        "seed": int(m.group("seed")),
    }


def discover_npz(npz_dir: Path, seeds: list[int] | None = None) -> list[Path]:
    paths = sorted(npz_dir.glob("rec_v1_*.npz"))
    if seeds is not None:
        seed_set = set(seeds)
        keep = []
        for p in paths:
            info = parse_rec_name(p.stem)
            if info["seed"] in seed_set:
                keep.append(p)
        paths = keep
    return paths


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return mu, 0.0
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return mu, sd


def print_spatial_summary(rows: list[dict]):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    print("\n=== Spatial-only oracle by SNR (mean over seeds/units) ===")
    snrs = sorted({r["snr"] for r in rows if r["snr"] is not None})
    header = f"{'snr':>5}" + "".join(f"{m[:10]:>11}" for m in methods)
    print(header)
    print("-" * len(header))
    for snr in snrs:
        cells = []
        for m in methods:
            vals = [
                r["oracle_accuracy"]
                for r in rows
                if r["method"] == m and r["snr"] == snr
            ]
            mu, _ = mean_std(vals)
            cells.append(f"{mu:>11.3f}")
        print(f"{snr:>5}" + "".join(cells))

    print("\nOverall mean ± std (oracle / SOM):")
    for m in methods:
        o_vals = [r["oracle_accuracy"] for r in rows if r["method"] == m]
        s_vals = [r["som_accuracy"] for r in rows if r["method"] == m]
        om, os_ = mean_std(o_vals)
        sm, ss = mean_std(s_vals)
        print(f"  {m:<16} oracle={om:.3f}±{os_:.3f}  SOM={sm:.3f}±{ss:.3f}")


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
    for metric in ("l2", "l1"):
        sub = [r for r in rows if r["p2p_metric"] == metric]
        if not sub:
            continue
        print(f"\n=== Two-stage oracle (P2P={metric}) by SNR ===")
        snrs = sorted({r["snr"] for r in sub if r["snr"] is not None})
        header = f"{'snr':>5}" + "".join(f"{m[:9]:>10}" for m in focus)
        print(header)
        print("-" * len(header))
        for snr in snrs:
            cells = []
            for m in focus:
                vals = [
                    r["oracle_accuracy"]
                    for r in sub
                    if r["method"] == m and r["snr"] == snr
                ]
                mu, _ = mean_std(vals)
                cells.append(f"{mu:>10.3f}")
            print(f"{snr:>5}" + "".join(cells))

        print(f"\nOverall mean ± std (P2P={metric}):")
        for m in focus:
            vals = [r["oracle_accuracy"] for r in sub if r["method"] == m]
            mu, sd = mean_std(vals)
            print(f"  {m:<28} {mu:.3f} ± {sd:.3f}")

        # Split by designed unit count
        print(f"\nBy n_units_design (P2P={metric}, two_stage_waveform / soft via com_and_p2p):")
        for n_u in sorted({r["n_units_design"] for r in sub if r["n_units_design"]}):
            for m in ("com_and_p2p", "two_stage_fsde", "two_stage_waveform"):
                vals = [
                    r["oracle_accuracy"]
                    for r in sub
                    if r["method"] == m and r["n_units_design"] == n_u
                ]
                mu, sd = mean_std(vals)
                print(f"  units={n_u:<3} {m:<28} {mu:.3f} ± {sd:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Sweep Spatial methods on MEArec v1 NPZs")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--p2p-metrics",
        nargs="+",
        default=["l2", "l1"],
        choices=["l2", "l1"],
    )
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--skip-two-stage", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Spatial/output/mearec_method_sweep"),
    )
    args = parser.parse_args()

    paths = discover_npz(args.npz_dir, seeds=args.seeds)
    if not paths:
        raise SystemExit(f"No MEArec NPZ found under {args.npz_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"MEArec NPZs: {len(paths)} | duration={args.duration}s | dir={args.npz_dir}")
    for p in paths:
        info = parse_rec_name(p.stem)
        print(
            f"  - {info['rec']}  units={info['n_units_design']} "
            f"snr={info['snr']} seed={info['seed']}"
        )

    spatial_rows: list[dict] = []
    two_stage_rows: list[dict] = []
    t0 = time.time()

    for path in paths:
        info = parse_rec_name(path.stem)
        print(f"\n########## {info['rec']} ##########")
        ds = load_mearec_npz(path, duration_s=args.duration)

        if not args.skip_spatial:
            print("--- spatial upper bound ---")
            rows = run_spatial_study(ds, verbose=True)
            for r in rows:
                spatial_rows.append({
                    **info,
                    "n_units_gt": r["gt_units"],
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
                        **info,
                        "n_units_gt": int(ds.unit_ids.size),
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
        out = args.output_dir / f"spatial_upper_bound_mearec_{int(args.duration)}s.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spatial_rows[0].keys()))
            w.writeheader()
            w.writerows(spatial_rows)
        print(f"\nSaved: {out}")
        print_spatial_summary(spatial_rows)

    if two_stage_rows:
        out = args.output_dir / f"two_stage_mearec_{int(args.duration)}s.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(two_stage_rows[0].keys()))
            w.writeheader()
            w.writerows(two_stage_rows)
        print(f"\nSaved: {out}")
        print_two_stage_summary(two_stage_rows)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
