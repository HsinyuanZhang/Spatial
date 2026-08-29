"""Tight L1 bulk + escalate to the frozen p99.9 source (not certified early-exit).

Inner radius = calibration p90. If the inner list is empty, replace it with the
outer p99.9 list. Unique inner hits are still counted as candidates, not labels.
Traffic uses 64×5 bits only when |C|>1.

Default arm is D9 (COM+P2P). ``--arm com_posneg`` is the LEVEL1_ROADMAP
tail-policy experiment on the tighter shape descriptor.

    python -m Spatial.experiments.run_bulk_tail_escalation --pilot --duration 60
    python -m Spatial.experiments.run_bulk_tail_escalation --pilot --duration 60 --arm com_posneg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.adaptive_range_search import (
    AdaptiveRangeSearch,
    evaluate_candidate_search,
    fit_integer_centroids,
)
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.run_adaptive_weighted_5bit import stable_chronological_three_way_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _load_locked_recording,
)
from Spatial.experiments.shape_descriptor_common import assemble_arm, build_descriptor_blocks
from Spatial.experiments.spatial_cim_common import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "bulk_tail_escalation"
WAVEFORM_TEMPLATE_BITS = 64 * 5
INNER_PCTL = 90.0
OUTER_PCTL = 99.9
FROZEN_POSNEG_BITS = 4
FROZEN_LAT_BITS = 4
FROZEN_AMP_GATE = 0.125
ARMS = ("d9", "com_posneg")


def _percentile_radii(
    codes: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    centroids: np.ndarray,
    pctl: float,
    fallback: np.ndarray,
) -> np.ndarray:
    radii = np.asarray(fallback, dtype=np.int64).copy()
    for i, unit in enumerate(units):
        block = codes[labels == unit]
        if block.shape[0] == 0:
            continue
        dist = np.abs(block - centroids[i]).sum(axis=1)
        radii[i] = int(np.ceil(np.percentile(dist, pctl)))
    return radii


def bulk_tail_from_codes(
    codes: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    inner_pctl: float = INNER_PCTL,
    outer_pctl: float = OUTER_PCTL,
    dataset: str = "",
    family: str = "",
) -> dict[str, Any]:
    """Integer-L1 bulk/tail policy on already-quantized codes."""
    fit, cal, test = stable_chronological_three_way_split(np.asarray(times, dtype=np.int64))
    y = np.asarray(labels).ravel()
    x = np.rint(np.asarray(codes)).astype(np.int64)
    centroids_map = fit_integer_centroids(x[fit], y[fit])
    units = np.array(list(centroids_map.keys()))
    centroids = np.stack([centroids_map[u] for u in units], axis=0)
    fit_fallback = _percentile_radii(x[fit], y[fit], units, centroids, outer_pctl, np.zeros(units.size, dtype=np.int64))
    inner = _percentile_radii(x[cal], y[cal], units, centroids, inner_pctl, fit_fallback)
    outer = _percentile_radii(x[cal], y[cal], units, centroids, outer_pctl, fit_fallback)
    codes_test = x[test]
    y_test = y[test]
    dist = np.abs(codes_test[:, None, :] - centroids[None, :, :]).sum(axis=2)
    match_in = dist <= inner.reshape(1, -1)
    match_out = dist <= outer.reshape(1, -1)
    inner_count = match_in.sum(axis=1)
    escalate = inner_count == 0
    final = np.where(escalate[:, None], match_out, match_in)
    candidates = [units[row].copy() for row in final]
    counts = final.sum(axis=1).astype(np.int64)
    search = evaluate_candidate_search(candidates, y_test)
    reads = np.where(counts > 1, counts, 0)
    actual = int(reads.sum()) * WAVEFORM_TEMPLATE_BITS
    glob = int(y_test.size * units.size) * WAVEFORM_TEMPLATE_BITS
    outer_cands = [units[row].copy() for row in match_out]
    outer_search = evaluate_candidate_search(outer_cands, y_test)
    outer_counts = match_out.sum(axis=1)
    outer_reads = np.where(outer_counts > 1, outer_counts, 0)
    outer_actual = int(outer_reads.sum()) * WAVEFORM_TEMPLATE_BITS
    inner_hit = np.array(
        [
            (lab in units[row]) if cnt else False
            for lab, row, cnt in zip(y_test, match_in, inner_count)
        ],
        dtype=bool,
    )
    return {
        "dataset": dataset,
        "family": family,
        "n_test": int(y_test.size),
        "n_units": int(units.size),
        "inner_percentile": float(inner_pctl),
        "outer_percentile": float(outer_pctl),
        "true_unit_candidate_recall": float(search["true_unit_candidate_recall"]),
        "outer_recall": float(outer_search["true_unit_candidate_recall"]),
        "mean_C": float(np.mean(counts)),
        "outer_mean_C": float(np.mean(outer_counts)),
        "p95_C": float(np.percentile(counts, 95)),
        "frac_C0": float(np.mean(counts == 0)),
        "frac_C1": float(np.mean(counts == 1)),
        "frac_Cgt1": float(np.mean(counts > 1)),
        "escalation_rate": float(np.mean(escalate)),
        "traffic_x": float(glob / actual) if actual else float("inf"),
        "outer_traffic_x": float(glob / outer_actual) if outer_actual else float("inf"),
        "inner_only_recall": float(np.mean(inner_hit)),
    }


def _d9_codes(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared = prepare_dataset_events(dataset, k_neighbors=7)
    fit, _, _ = stable_chronological_three_way_split(prepared["times"])
    model = AdaptiveRangeSearch(
        n_bits=5, fixed_unit_interval=True, radius_percentile=99.9
    ).fit(prepared["descriptor"][fit], prepared["labels"][fit])
    return model.transform(prepared["descriptor"]), prepared["labels"], prepared["times"]


def _composneg_codes(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = build_descriptor_blocks(dataset)
    codes = assemble_arm(
        blocks,
        "com_posneg",
        posneg_bits=FROZEN_POSNEG_BITS,
        lat_bits=FROZEN_LAT_BITS,
        amp_gate=FROZEN_AMP_GATE,
        recording_name=dataset.name,
    )
    return np.rint(codes).astype(np.int64), blocks["labels"], blocks["times"]


def run_recording(dataset, family: str, arm: str) -> dict[str, Any]:
    if arm == "d9":
        codes, labels, times = _d9_codes(dataset)
    elif arm == "com_posneg":
        codes, labels, times = _composneg_codes(dataset)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    rec = bulk_tail_from_codes(
        codes, labels, times, dataset=dataset.name, family=family
    )
    rec["arm"] = arm
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--arm", choices=ARMS, default="d9")
    parser.add_argument("--mearec-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s")
    out = Path(args.output_dir) if args.output_dir is not None else (
        DEFAULT_OUT if args.arm == "d9" else DEFAULT_OUT.parent / f"bulk_tail_escalation_{args.arm}"
    )
    rows = []
    for family, name in LOCKED_DEVELOPMENT_RECORDINGS:
        ds = _load_locked_recording(family, name, 60.0, Path(args.mearec_dir))
        rec = run_recording(ds, family, args.arm)
        rows.append(rec)
        print(
            f"{rec['arm']:<11} {rec['dataset']:<42} R={rec['true_unit_candidate_recall']:.4f} "
            f"(outer {rec['outer_recall']:.4f})  C={rec['mean_C']:.2f} "
            f"esc={rec['escalation_rate']:.3f} x={rec['traffic_x']:.2f} "
            f"(outer x={rec['outer_traffic_x']:.2f})"
        )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bulk_tail_escalation_pilot.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by_fam = {}
    for family in ("hj", "mearec"):
        sub = [r for r in rows if r["family"] == family]
        n = sum(r["n_test"] for r in sub)
        rec = sum(r["true_unit_candidate_recall"] * r["n_test"] for r in sub) / n
        x = float(np.mean([r["traffic_x"] for r in sub]))
        by_fam[family] = {
            "recall": rec,
            "mean_traffic_x": x,
            "escalation": float(np.mean([r["escalation_rate"] for r in sub])),
        }
    write_json(
        out / "bulk_tail_escalation_summary.json",
        {"arm": args.arm, "per_recording": rows, "family": by_fam},
    )
    print(f"Saved {path}")
    print("family:", by_fam)


if __name__ == "__main__":
    main()
