"""Oracle gate comparison: maxnorm vs central_ratio × float L2/L1 / 4-bit code L1.

Also reports the fraction of spikes where central (slot 0) is not the max P2P —
the only regime where the two representations diverge.

Usage:
    python -m Spatial.experiments.verify_central_ratio_gate --duration 60
    python -m Spatial.experiments.verify_central_ratio_gate --clips 1.0,1.5,2.0,4.0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import load_hybrid_janelia, load_mearec_npz
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    com_features,
    footprint_p2p_features,
    central_ratio_features,
    quantize_p2p,
)
from Spatial.algorithms.two_stage_assign import (
    two_stage_assign,
    calibrate_thresholds_codes,
    _intra_unit_distances,
)
from Spatial.experiments.run_two_stage_upper_bound import (
    _chronological_split,
    extract_main_channel_waveforms,
    normalize_waveforms,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz


def oracle_acc(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(pred == y_true)) if len(y_true) else 0.0


def two_stage_code_l1(
    com_tr,
    codes_tr,
    wave_tr,
    y_tr,
    com_te,
    codes_te,
    wave_te,
    percentile: float = 95.0,
):
    """Reuse ``two_stage_assign`` with L1 on codes + precomputed ``τ_p2p≥1``."""
    codes_tr = np.asarray(codes_tr, dtype=np.float64)
    codes_te = np.asarray(codes_te, dtype=np.float64)
    d_com = _intra_unit_distances(com_tr, y_tr, metric="l2")
    d_com = d_com[np.isfinite(d_com)]
    tau_com = float(np.percentile(d_com, percentile)) if d_com.size else 1.0
    tau_com = max(tau_com, 1e-8)
    tau_p2p = float(calibrate_thresholds_codes(codes_tr, y_tr, percentile=percentile))
    pred, meta = two_stage_assign(
        com_tr,
        codes_tr,
        wave_tr,
        y_tr,
        com_te,
        codes_te,
        wave_te,
        percentile=percentile,
        tau_com=tau_com,
        tau_p2p=tau_p2p,
        p2p_metric="l1",
    )
    meta = {**meta, "tau_p2p_codes": int(tau_p2p), "p2p_metric": "code_l1"}
    return pred, meta


def central_not_max_pct(p2p_raw: np.ndarray, eps: float = 1e-12) -> float:
    amp = np.maximum(np.asarray(p2p_raw, dtype=np.float64), 0.0)
    if len(amp) == 0:
        return 0.0
    return float(np.mean(amp[:, 0] < amp.max(axis=1) - eps))


def load_dataset(name: str, duration: float, mearec_path: Path | None = None):
    if name == "hj":
        return load_hybrid_janelia(duration_s=duration)
    if name == "mearec":
        if mearec_path is None:
            paths = discover_npz(DEFAULT_NPZ_DIR)
            if not paths:
                raise SystemExit(f"No MEArec NPZ under {DEFAULT_NPZ_DIR}")
            mearec_path = paths[0]
        return load_mearec_npz(mearec_path, duration_s=duration)
    raise SystemExit(f"Unknown dataset: {name}")


def prepare_features(dataset, k_neighbors: int = 7, window: int = 15):
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=window)
    t2u = dict(zip(dataset.spike_times.tolist(), dataset.spike_units.tolist()))
    labels = np.array([t2u.get(int(t), -1) for t in times], dtype=np.int64)
    keep = labels >= 0
    times, central, labels = times[keep], central[keep], labels[keep]

    k = min(k_neighbors, dataset.geom.shape[0])
    nbr = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, times2, neighbor_ids = extract_local_p2p(
        filtered, times, central, nbr, window=window
    )
    assert len(times2) == len(times)

    com = com_features(p2p_raw, neighbor_ids, dataset.geom)
    wfs = normalize_waveforms(extract_main_channel_waveforms(filtered, times, central))

    order = np.argsort(times)
    return {
        "com": com[order],
        "p2p_raw": p2p_raw[order],
        "wave": wfs[order],
        "labels": labels[order],
        "central_not_max_pct": central_not_max_pct(p2p_raw[order]),
        "dataset": dataset.name,
    }


def run_study(
    dataset,
    clips: list[float],
    n_bits: int = 4,
    train_frac: float = 0.5,
    percentile: float = 95.0,
    verbose: bool = True,
) -> list[dict]:
    data = prepare_features(dataset)
    com = data["com"]
    p2p_raw = data["p2p_raw"]
    wave = data["wave"]
    labels = data["labels"]
    cnot = data["central_not_max_pct"]
    tr, te = _chronological_split(len(labels), train_frac)
    y_tr, y_te = labels[tr], labels[te]

    if verbose:
        print(
            f"Dataset: {data['dataset']} | n={len(labels)} | "
            f"units={len(np.unique(labels))} | central_not_max={cnot:.3f}"
        )

    rows: list[dict] = []
    baseline_l2: float | None = None

    def add_row(
        repr_name: str,
        dist: str,
        bits: str,
        clip: float | str,
        pred: np.ndarray,
        meta: dict,
    ):
        nonlocal baseline_l2
        acc = oracle_acc(pred, y_te)
        if baseline_l2 is None and repr_name == "maxnorm" and dist == "float_l2":
            baseline_l2 = acc
        base = baseline_l2 if baseline_l2 is not None else acc
        row = {
            "dataset": data["dataset"],
            "repr": repr_name,
            "dist": dist,
            "bits": bits,
            "clip": clip,
            "oracle_acc": acc,
            "delta_pp": (acc - base) * 100.0,
            "fallback_rate": meta.get("fallback_rate", ""),
            "mean_n_candidates": meta.get("mean_n_candidates", ""),
            "tau_p2p": meta.get("tau_p2p", meta.get("tau_p2p_codes", "")),
            "tau_com": meta.get("tau_com", ""),
            "central_not_max_pct": cnot,
        }
        rows.append(row)
        if verbose:
            print(
                f"  {repr_name:<14} {dist:<10} {str(bits):<6} clip={clip!s:<4} "
                f"acc={acc:.3f}  Δ={row['delta_pp']:+.2f}pp  "
                f"fb={meta.get('fallback_rate', float('nan')):.3f}  "
                f"τ_p2p={row['tau_p2p']}"
            )

    # --- maxnorm baselines (clip-independent) ---
    p2p_max = footprint_p2p_features(p2p_raw, normalize=True)
    for metric in ("l2", "l1"):
        pred, meta = two_stage_assign(
            com[tr],
            p2p_max[tr],
            wave[tr],
            y_tr,
            com[te],
            p2p_max[te],
            wave[te],
            percentile=percentile,
            p2p_metric=metric,  # type: ignore[arg-type]
        )
        add_row("maxnorm", f"float_{metric}", "float", "", pred, meta)

    codes_max = quantize_p2p(p2p_raw, n_bits=n_bits, mode="per_spike_max")
    pred, meta = two_stage_code_l1(
        com[tr], codes_max[tr], wave[tr], y_tr,
        com[te], codes_max[te], wave[te],
        percentile=percentile,
    )
    add_row("maxnorm", "code_l1", f"{n_bits}bit", "", pred, meta)

    # --- central_ratio × clip sweep ---
    for clip in clips:
        p2p_cr = central_ratio_features(p2p_raw, clip=clip)
        for metric in ("l2", "l1"):
            pred, meta = two_stage_assign(
                com[tr], p2p_cr[tr], wave[tr], y_tr,
                com[te], p2p_cr[te], wave[te],
                percentile=percentile,
                p2p_metric=metric,  # type: ignore[arg-type]
            )
            add_row("central_ratio", f"float_{metric}", "float", clip, pred, meta)

        codes_cr = quantize_p2p(
            p2p_raw, n_bits=n_bits, mode="central_ratio", clip=clip
        )
        pred, meta = two_stage_code_l1(
            com[tr], codes_cr[tr], wave[tr], y_tr,
            com[te], codes_cr[te], wave[te],
            percentile=percentile,
        )
        add_row("central_ratio", "code_l1", f"{n_bits}bit", clip, pred, meta)

    if verbose:
        print(f"\ncentral_not_max_pct = {cnot:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Oracle: maxnorm vs central_ratio P2P gate (float + code L1)"
    )
    parser.add_argument("--dataset", choices=["hj", "mearec"], default="hj")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--clips",
        type=str,
        default="1.0,1.5,2.0,4.0",
        help="Comma-separated clip values for central_ratio",
    )
    parser.add_argument("--cim-bits", type=int, default=4)
    parser.add_argument("--mearec-npz", type=Path, default=None)
    parser.add_argument(
        "--all-mearec",
        action="store_true",
        help="Sweep all MEArec NPZs under --npz-dir (implies --dataset mearec).",
    )
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    clips = [float(x) for x in args.clips.split(",") if x.strip()]
    verbose = not args.quiet

    if args.output is None:
        tag = "mearec" if (args.dataset == "mearec" or args.all_mearec) else "hj"
        args.output = Path(f"Spatial/output/central_ratio_gate_{tag}{int(args.duration)}s.csv")

    all_rows: list[dict] = []
    if args.all_mearec or (args.dataset == "mearec" and args.mearec_npz is None):
        paths = discover_npz(args.npz_dir, seeds=args.seeds)
        if not paths:
            raise SystemExit(f"No MEArec NPZ under {args.npz_dir}")
        print(f"MEArec oracle gate sweep: {len(paths)} NPZs | clips={clips}")
        for path in paths:
            print(f"\n########## {path.stem} ##########")
            ds = load_mearec_npz(path, duration_s=args.duration)
            rows = run_study(ds, clips=clips, n_bits=args.cim_bits, verbose=verbose)
            for r in rows:
                r["scene"] = path.stem
            all_rows.extend(rows)
    else:
        ds = load_dataset(args.dataset, args.duration, mearec_path=args.mearec_npz)
        rows = run_study(ds, clips=clips, n_bits=args.cim_bits, verbose=verbose)
        for r in rows:
            r["scene"] = getattr(ds, "name", args.dataset)
        all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {args.output} ({len(all_rows)} rows)")

    # Compact summary: mean oracle by repr/dist/clip
    from collections import defaultdict

    buckets: dict[tuple, list[float]] = defaultdict(list)
    cnot: list[float] = []
    for r in all_rows:
        key = (r["repr"], r["dist"], str(r["bits"]), str(r["clip"]))
        buckets[key].append(float(r["oracle_acc"]))
        cnot.append(float(r["central_not_max_pct"]))
    print(f"\ncentral_not_max_pct: mean={sum(cnot)/len(cnot):.3f}  "
          f"min={min(cnot):.3f}  max={max(cnot):.3f}")
    print(f"{'repr':<14} {'dist':<10} {'bits':<6} {'clip':<5} {'mean':>7} {'std':>7}")
    print("-" * 60)
    for key in sorted(buckets.keys()):
        vals = buckets[key]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        print(f"{key[0]:<14} {key[1]:<10} {key[2]:<6} {key[3]:<5} {mu:>7.3f} {sd:>7.3f}")


if __name__ == "__main__":
    main()
