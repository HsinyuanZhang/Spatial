"""Stage A corpus: selected spatial WTA config across 12 HJ + 20 MEArec.

Reports per-condition (drift/static, 10/20 units, SNR) and never pools them
into a single headline number.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_causal_temporal_sketch import stable_three_way_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz
from Spatial.experiments.run_spatial_wta_pilot import _fit_predict
from Spatial.experiments.spatial_cim_common import (
    prepare_spatial_events,
    sha256_json,
    write_csv,
    write_json,
)


def _parse_mearec_condition(stem: str) -> dict[str, Any]:
    match = re.search(r"units(\d+)_snr(\d+)", stem)
    if not match:
        return {"units": None, "snr": None}
    return {"units": int(match.group(1)), "snr": int(match.group(2))}


def _parse_hj_condition(name: str) -> dict[str, Any]:
    drift = "drift" in name
    return {"condition": "drift" if drift else "static", "is_drift": drift}


def run_one(
    dataset_id: str,
    family: str,
    dataset,
    *,
    feature_set: str,
    n_bits: int,
    use_theta: bool,
    mask_mode: str,
    mask_size: int,
    score_mode: str,
    rank_profile: str,
) -> dict[str, Any]:
    prepared = prepare_spatial_events(dataset, k_neighbors=7)
    fit_idx, cal_idx, test_idx = stable_three_way_split(prepared["times"])
    values = prepared["features"][feature_set]
    labels = prepared["labels"]
    result = _fit_predict(
        values[fit_idx], labels[fit_idx],
        values[cal_idx], labels[cal_idx],
        values[test_idx], labels[test_idx],
        n_bits=n_bits,
        use_theta=use_theta,
        mask_mode=mask_mode,
        mask_size=mask_size,
        score_mode=score_mode,
        rank_profile=rank_profile,
    )
    row: dict[str, Any] = {
        "dataset": dataset_id,
        "family": family,
        "feature_set": feature_set,
        "n_bits": n_bits,
        "use_theta": use_theta,
        "mask_mode": mask_mode,
        "mask_size": mask_size,
        "score_mode": score_mode,
        "rank_profile": rank_profile,
        "accuracy": result["accuracy"],
        "n_correct": result["n_correct"],
        "n_events": result["n_events"],
        "worst_unit_accuracy": result["worst_unit_accuracy"],
        "search_payload_bits_per_row": result["search_payload_bits_per_row"],
        "n_units_model": result["n_units"],
        "per_unit": result["per_unit"],
        "memory_accounting": result["memory_accounting"],
        "logical_op_counts": result["logical_op_counts"],
    }
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial WTA Stage A corpus")
    parser.add_argument("--feature-set", default="com_and_p2p")
    parser.add_argument("--n-bits", type=int, default=5)
    parser.add_argument("--use-theta", action="store_true", default=True)
    parser.add_argument("--no-theta", action="store_true")
    parser.add_argument("--mask-mode", default="all_dims")
    parser.add_argument("--mask-size", type=int, default=9)
    parser.add_argument("--score-mode", default="uniform")
    parser.add_argument("--rank-profile", default="top3_x4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "spatial_cim_sorting_corpus",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args(argv)
    use_theta = not args.no_theta

    rows: list[dict[str, Any]] = []

    # HJ: 12 scenes
    for scene in list_hybrid_janelia_scenes(ready_only=True):
        key = scene.get("short_name") or scene.get("scene_key")
        path = Path(scene.get("npz_path", ""))
        if not path.exists():
            continue
        dataset_id = f"hybrid_janelia_{str(key).replace('_filtered_gt', '')}"
        print(f"HJ {dataset_id} ...", flush=True)
        dataset = load_hybrid_janelia(path=path, duration_s=args.duration)
        row = run_one(
            dataset_id, "hj", dataset,
            feature_set=args.feature_set,
            n_bits=args.n_bits,
            use_theta=use_theta,
            mask_mode=args.mask_mode,
            mask_size=args.mask_size,
            score_mode=args.score_mode,
            rank_profile=args.rank_profile,
        )
        row.update(_parse_hj_condition(dataset_id))
        rows.append(row)

    # MEArec: 20 NPZs
    for path in discover_npz(Path(DEFAULT_NPZ_DIR)):
        dataset_id = f"mearec_{path.stem}"
        print(f"MEArec {dataset_id} ...", flush=True)
        dataset = load_mearec_npz(path, duration_s=args.duration)
        row = run_one(
            dataset_id, "mearec", dataset,
            feature_set=args.feature_set,
            n_bits=args.n_bits,
            use_theta=use_theta,
            mask_mode=args.mask_mode,
            mask_size=args.mask_size,
            score_mode=args.score_mode,
            rank_profile=args.rank_profile,
        )
        row.update(_parse_mearec_condition(path.stem))
        rows.append(row)

    # Per-condition summaries (never a single pooled headline).
    summaries: dict[str, Any] = {"hj_by_condition": {}, "mearec_by_units_snr": {}}
    for condition in ("static", "drift"):
        subset = [r for r in rows if r["family"] == "hj" and r.get("condition") == condition]
        if subset:
            n = sum(r["n_events"] for r in subset)
            summaries["hj_by_condition"][condition] = {
                "n_recordings": len(subset),
                "n_events": n,
                "accuracy": sum(r["accuracy"] * r["n_events"] for r in subset) / max(n, 1),
                "per_recording": {
                    r["dataset"]: {"accuracy": r["accuracy"], "n_events": r["n_events"]}
                    for r in subset
                },
            }
    for units in (10, 20):
        for snr in (2, 3, 5, 8, 12):
            subset = [
                r for r in rows
                if r["family"] == "mearec" and r.get("units") == units and r.get("snr") == snr
            ]
            if not subset:
                continue
            n = sum(r["n_events"] for r in subset)
            key = f"units{units}_snr{snr}"
            summaries["mearec_by_units_snr"][key] = {
                "n_recordings": len(subset),
                "n_events": n,
                "accuracy": sum(r["accuracy"] * r["n_events"] for r in subset) / max(n, 1),
                "per_recording": {
                    r["dataset"]: {"accuracy": r["accuracy"], "n_events": r["n_events"]}
                    for r in subset
                },
            }

    write_csv(args.output_dir / "spatial_wta_corpus.csv", rows)
    write_json(args.output_dir / "spatial_wta_corpus_summary.json", {
        "config": {
            "feature_set": args.feature_set,
            "n_bits": args.n_bits,
            "use_theta": use_theta,
            "mask_mode": args.mask_mode,
            "mask_size": args.mask_size,
            "score_mode": args.score_mode,
            "rank_profile": args.rank_profile,
        },
        "summaries": summaries,
        "row_digest": sha256_json(
            [{"dataset": r["dataset"], "n_correct": r["n_correct"], "n_events": r["n_events"]}
             for r in rows]
        ),
    })
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
