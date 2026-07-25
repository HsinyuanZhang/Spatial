"""Sweep unsupervised two-stage (L1/L2) across HJ 16ch scenes + MEArec v1 NPZs.

Usage:
  python -m Spatial.experiments.run_unsupervised_two_stage_sweep --duration 60
  python -m Spatial.experiments.run_unsupervised_two_stage_sweep --family hj
  python -m Spatial.experiments.run_unsupervised_two_stage_sweep --family mearec --seeds 202600
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_hj16_method_sweep import select_16ch_scenes
from Spatial.experiments.run_mearec_method_sweep import (
    DEFAULT_NPZ_DIR,
    discover_npz,
    mean_std,
    parse_rec_name,
)
from Spatial.experiments.run_two_stage_unsupervised import run_study

FOCUS_METHODS = [
    "som_soft_loc",
    "som_soft_loc_full",
    "unsup_two_stage_fsde",
    "unsup_two_stage_waveform",
    "oracle_two_stage_fsde",
    "oracle_two_stage_waveform",
]


def _tag_rows(rows: list[dict], **meta) -> list[dict]:
    out = []
    for r in rows:
        out.append({**meta, **r})
    return out


def print_hj_summary(rows: list[dict]):
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])

    print("\n=== HJ unsupervised two-stage (test Hungarian acc) ===")
    for metric in ("l2", "l1", ""):
        if metric == "":
            # baselines without p2p_metric
            methods = ["som_soft_loc", "som_soft_loc_full"]
            sub = [r for r in rows if r["method"] in methods]
            if not sub:
                continue
            print("\nSpatial SOM baselines:")
            header = f"{'scene':<22}" + "".join(f"{m[:14]:>15}" for m in methods)
            print(header)
            print("-" * len(header))
            for sc in scenes:
                cells = []
                for m in methods:
                    hit = next(
                        (r for r in sub if r["scene"] == sc and r["method"] == m),
                        None,
                    )
                    cells.append(f"{hit['accuracy']:>15.3f}" if hit else f"{'—':>15}")
                print(f"{sc:<22}" + "".join(cells))
            for m in methods:
                vals = [r["accuracy"] for r in sub if r["method"] == m]
                mu, sd = mean_std(vals)
                print(f"  {m:<28} {mu:.3f} ± {sd:.3f}")
            continue

        methods = [
            "unsup_two_stage_fsde",
            "unsup_two_stage_waveform",
            "oracle_two_stage_fsde",
            "oracle_two_stage_waveform",
        ]
        sub = [r for r in rows if r.get("p2p_metric") == metric and r["method"] in methods]
        if not sub:
            continue
        print(f"\nP2P={metric}:")
        header = f"{'scene':<22}" + "".join(f"{m.replace('two_stage_', '')[:12]:>13}" for m in methods)
        print(header)
        print("-" * len(header))
        for sc in scenes:
            cells = []
            for m in methods:
                hit = next(
                    (r for r in sub if r["scene"] == sc and r["method"] == m),
                    None,
                )
                cells.append(f"{hit['accuracy']:>13.3f}" if hit else f"{'—':>13}")
            print(f"{sc:<22}" + "".join(cells))
        print(f"Mean ± std (P2P={metric}):")
        for m in methods:
            vals = [r["accuracy"] for r in sub if r["method"] == m]
            mu, sd = mean_std(vals)
            print(f"  {m:<32} {mu:.3f} ± {sd:.3f}")


def print_mearec_summary(rows: list[dict]):
    print("\n=== MEArec unsupervised two-stage ===")
    for metric in ("l2", "l1", ""):
        if metric == "":
            methods = ["som_soft_loc", "som_soft_loc_full"]
            sub = [r for r in rows if r["method"] in methods]
            if not sub:
                continue
            print("\nSpatial SOM baselines (overall / by SNR):")
            for m in methods:
                vals = [r["accuracy"] for r in sub if r["method"] == m]
                mu, sd = mean_std(vals)
                print(f"  {m:<28} {mu:.3f} ± {sd:.3f}")
            snrs = sorted({r["snr"] for r in sub if r.get("snr") is not None})
            header = f"{'snr':>5}" + "".join(f"{m[:14]:>15}" for m in methods)
            print(header)
            print("-" * len(header))
            for snr in snrs:
                cells = []
                for m in methods:
                    vals = [
                        r["accuracy"]
                        for r in sub
                        if r["method"] == m and r["snr"] == snr
                    ]
                    mu, _ = mean_std(vals)
                    cells.append(f"{mu:>15.3f}")
                print(f"{snr:>5}" + "".join(cells))
            continue

        methods = [
            "unsup_two_stage_fsde",
            "unsup_two_stage_waveform",
            "oracle_two_stage_fsde",
            "oracle_two_stage_waveform",
        ]
        sub = [r for r in rows if r.get("p2p_metric") == metric and r["method"] in methods]
        if not sub:
            continue
        print(f"\nP2P={metric} by SNR:")
        snrs = sorted({r["snr"] for r in sub if r.get("snr") is not None})
        header = f"{'snr':>5}" + "".join(
            f"{m.replace('two_stage_', '')[:12]:>13}" for m in methods
        )
        print(header)
        print("-" * len(header))
        for snr in snrs:
            cells = []
            for m in methods:
                vals = [
                    r["accuracy"]
                    for r in sub
                    if r["method"] == m and r["snr"] == snr
                ]
                mu, _ = mean_std(vals)
                cells.append(f"{mu:>13.3f}")
            print(f"{snr:>5}" + "".join(cells))

        print(f"\nOverall mean ± std (P2P={metric}):")
        for m in methods:
            vals = [r["accuracy"] for r in sub if r["method"] == m]
            mu, sd = mean_std(vals)
            print(f"  {m:<32} {mu:.3f} ± {sd:.3f}")

        print(f"\nBy n_units_design (P2P={metric}):")
        for n_u in sorted({r["n_units_design"] for r in sub if r.get("n_units_design")}):
            for m in ("unsup_two_stage_fsde", "unsup_two_stage_waveform", "oracle_two_stage_waveform"):
                vals = [
                    r["accuracy"]
                    for r in sub
                    if r["method"] == m and r["n_units_design"] == n_u
                ]
                mu, sd = mean_std(vals)
                print(f"  units={n_u:<3} {m:<32} {mu:.3f} ± {sd:.3f}")


def run_hj(
    duration: float,
    percentile: float,
    verbose: bool,
    p2p_repr: str = "maxnorm",
    cim_bits: int | None = None,
    ratio_clip: float = 2.0,
    include_oracle: bool = True,
) -> list[dict]:
    ready = list_hybrid_janelia_scenes(ready_only=True)
    scenes = select_16ch_scenes(ready)
    if not scenes:
        raise SystemExit("No ready 16ch 600s HJ scenes.")
    print(
        f"HJ scenes: {len(scenes)} | duration={duration}s | "
        f"p2p_repr={p2p_repr} cim_bits={cim_bits} clip={ratio_clip}"
    )
    all_rows: list[dict] = []
    for e in scenes:
        short = e.get("short_name") or e["scene_key"]
        print(f"\n########## HJ {short} ##########")
        ds = load_hybrid_janelia(scene=short, duration_s=duration)
        rows = run_study(
            ds,
            percentile=percentile,
            include_oracle=include_oracle,
            verbose=verbose,
            p2p_repr=p2p_repr,
            cim_bits=cim_bits,
            ratio_clip=ratio_clip,
        )
        all_rows.extend(
            _tag_rows(
                rows,
                family="hj",
                scene=short,
                drift=e.get("drift"),
                n_units_catalog=e.get("n_valid_units"),
                p2p_repr=p2p_repr,
                cim_bits=cim_bits if cim_bits is not None else "",
                ratio_clip=ratio_clip,
            )
        )
    return all_rows


def run_mearec(
    npz_dir: Path,
    duration: float,
    percentile: float,
    seeds: list[int] | None,
    verbose: bool,
    p2p_repr: str = "maxnorm",
    cim_bits: int | None = None,
    ratio_clip: float = 2.0,
    include_oracle: bool = True,
) -> list[dict]:
    paths = discover_npz(npz_dir, seeds=seeds)
    if not paths:
        raise SystemExit(f"No MEArec NPZ under {npz_dir}")
    print(
        f"MEArec NPZs: {len(paths)} | duration={duration}s | "
        f"p2p_repr={p2p_repr} cim_bits={cim_bits} clip={ratio_clip}"
    )
    all_rows: list[dict] = []
    for path in paths:
        info = parse_rec_name(path.stem)
        print(f"\n########## MEArec {info['rec']} ##########")
        ds = load_mearec_npz(path, duration_s=duration)
        rows = run_study(
            ds,
            percentile=percentile,
            include_oracle=include_oracle,
            verbose=verbose,
            p2p_repr=p2p_repr,
            cim_bits=cim_bits,
            ratio_clip=ratio_clip,
        )
        all_rows.extend(
            _tag_rows(
                rows,
                family="mearec",
                scene=info["rec"],
                n_units_design=info["n_units_design"],
                snr=info["snr"],
                seed=info["seed"],
                p2p_repr=p2p_repr,
                cim_bits=cim_bits if cim_bits is not None else "",
                ratio_clip=ratio_clip,
            )
        )
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised two-stage sweep on HJ + MEArec"
    )
    parser.add_argument(
        "--family",
        choices=["all", "hj", "mearec"],
        default="all",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--p2p-repr",
        choices=["maxnorm", "central_ratio"],
        default="maxnorm",
    )
    parser.add_argument("--cim-bits", type=int, default=None)
    parser.add_argument("--ratio-clip", type=float, default=2.0)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Spatial/output/unsupervised_two_stage_sweep"),
    )
    args = parser.parse_args()
    verbose = not args.quiet
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg_tag = args.p2p_repr
    if args.cim_bits is not None:
        cfg_tag += f"_c{args.cim_bits}"
    if args.p2p_repr == "central_ratio":
        cfg_tag += f"_clip{args.ratio_clip:g}"

    hj_rows: list[dict] = []
    me_rows: list[dict] = []

    if args.family in ("all", "hj"):
        hj_rows = run_hj(
            args.duration,
            args.percentile,
            verbose=verbose,
            p2p_repr=args.p2p_repr,
            cim_bits=args.cim_bits,
            ratio_clip=args.ratio_clip,
            include_oracle=not args.no_oracle,
        )
        out = args.output_dir / f"unsup_two_stage_hj16_{int(args.duration)}s_{cfg_tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hj_rows[0].keys()))
            w.writeheader()
            w.writerows(hj_rows)
        print(f"\nSaved: {out}")
        print_hj_summary(hj_rows)

    if args.family in ("all", "mearec"):
        me_rows = run_mearec(
            args.npz_dir,
            args.duration,
            args.percentile,
            seeds=args.seeds,
            verbose=verbose,
            p2p_repr=args.p2p_repr,
            cim_bits=args.cim_bits,
            ratio_clip=args.ratio_clip,
            include_oracle=not args.no_oracle,
        )
        out = args.output_dir / f"unsup_two_stage_mearec_{int(args.duration)}s_{cfg_tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(me_rows[0].keys()))
            w.writeheader()
            w.writerows(me_rows)
        print(f"\nSaved: {out}")
        print_mearec_summary(me_rows)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
