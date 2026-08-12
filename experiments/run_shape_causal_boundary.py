"""Phase A2 causal-filter and jitter boundary for the frozen shape descriptor.

  python -m Spatial.experiments.run_shape_causal_boundary --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split, prepare_dataset_events
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _iter_recordings,
    _mearec_n_units,
    _parse_recordings_arg,
    _scene_type,
)
from Spatial.experiments.shape_descriptor_common import (
    CSV_FIELDNAMES,
    CONTROL_ARM,
    CANDIDATE_ARM,
    K_NEIGHBORS,
    WINDOW,
    build_descriptor_blocks,
    evaluate_arm_rows,
    shared_event_guard_mask,
)
from Spatial.experiments.spatial_cim_common import event_weighted_mean, write_csv, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "shape_causal_boundary"
DEFAULT_WARMUP_SAMPLES = 1000
WARMUP_JUSTIFICATION = (
    "Order-3 SOS bandpass at 30 kHz needs roughly ten lower-band periods (~33 ms) to "
    "settle; 1000 samples exceeds that and also clears the 17-sample extrema+jitter margin."
)

FROZEN_POSNEG_BITS = 4
FROZEN_LAT_BITS = 4
FROZEN_AMP_GATE = 0.125

A2_ARMS = (CONTROL_ARM, "com_posneg", CANDIDATE_ARM)

PHASE_A_FAMILY_REFERENCE = {
    "hj": {
        CONTROL_ARM: 0.814,
        "com_posneg": 0.919,
        CANDIDATE_ARM: 0.920,
    },
    "mearec": {
        CONTROL_ARM: 0.777,
        "com_posneg": 0.858,
        CANDIDATE_ARM: 0.859,
    },
}

A2_CONDITIONS: tuple[tuple[str, str, int], ...] = (
    ("control", "zero_phase", 0),
    ("causal", "causal", 0),
    ("jitter_m2", "zero_phase", -2),
    ("jitter_m1", "zero_phase", -1),
    ("jitter_p1", "zero_phase", 1),
    ("jitter_p2", "zero_phase", 2),
)

A2_CSV_FIELDNAMES = CSV_FIELDNAMES + (
    "condition",
    "filter_mode",
    "jitter_samples",
    "warmup_samples",
    "n_events_guarded",
)

CAUSAL_RETENTION_THRESHOLD = 0.5
JITTER_MAX_LOSS_PP = 1.0


def lat_width_gain_over_d9(
    accuracies: Mapping[str, float],
    *,
    full_arm: str = CANDIDATE_ARM,
    posneg_arm: str = "com_posneg",
    control_arm: str = CONTROL_ARM,
) -> float:
    """Incremental LAT+WIDTH contribution: (full - posneg) relative to (posneg - d9)."""
    posneg_gain = float(accuracies[posneg_arm]) - float(accuracies[control_arm])
    lat_width_gain = float(accuracies[full_arm]) - float(accuracies[posneg_arm])
    if posneg_gain <= 0.0:
        return float("nan")
    return lat_width_gain / posneg_gain


def compute_a2_gate_results(
    per_family_per_condition: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    phase_a_reference: Mapping[str, Mapping[str, float]] = PHASE_A_FAMILY_REFERENCE,
    causal_condition: str = "causal",
    control_condition: str = "control",
    jitter_conditions: Sequence[str] = ("jitter_m2", "jitter_p2"),
) -> dict[str, Any]:
    """Evaluate Phase A2 retention and jitter gates from family-weighted accuracies."""
    gate: dict[str, Any] = {}
    causal_pass = True
    jitter_pass = True

    for family in ("hj", "mearec"):
        ref = phase_a_reference[family]
        phase_a_full_gain = float(ref[CANDIDATE_ARM]) - float(ref[CONTROL_ARM])
        phase_a_lat_width_incremental = float(ref[CANDIDATE_ARM]) - float(ref["com_posneg"])

        causal_accs = per_family_per_condition[family][causal_condition]
        control_accs = per_family_per_condition[family][control_condition]
        causal_full_gain = float(causal_accs[CANDIDATE_ARM]) - float(causal_accs[CONTROL_ARM])
        causal_lat_width_incremental = float(causal_accs[CANDIDATE_ARM]) - float(causal_accs["com_posneg"])

        if phase_a_full_gain > 0.0:
            full_retention = causal_full_gain / phase_a_full_gain
        else:
            full_retention = float("nan")
        if phase_a_lat_width_incremental > 0.0:
            lat_width_retention = causal_lat_width_incremental / phase_a_lat_width_incremental
        else:
            lat_width_retention = float("nan")

        family_causal_ok = (
            np.isfinite(full_retention)
            and full_retention >= CAUSAL_RETENTION_THRESHOLD
            and np.isfinite(lat_width_retention)
            and lat_width_retention >= CAUSAL_RETENTION_THRESHOLD
        )
        causal_pass &= family_causal_ok

        gate[f"phase_a_full_gain_over_d9_{family}"] = float(phase_a_full_gain)
        gate[f"phase_a_lat_width_incremental_gain_{family}"] = float(phase_a_lat_width_incremental)
        gate[f"causal_full_gain_over_d9_{family}"] = float(causal_full_gain)
        gate[f"causal_lat_width_incremental_gain_{family}"] = float(causal_lat_width_incremental)
        gate[f"causal_full_gain_retention_fraction_{family}"] = float(full_retention)
        gate[f"causal_lat_width_incremental_retention_fraction_{family}"] = float(lat_width_retention)
        gate[f"causal_retention_pass_{family}"] = bool(family_causal_ok)

        worst_jitter_loss_pp = 0.0
        for jitter_name in jitter_conditions:
            jitter_accs = per_family_per_condition[family][jitter_name]
            loss_pp = 100.0 * (
                float(control_accs[CANDIDATE_ARM]) - float(jitter_accs[CANDIDATE_ARM])
            )
            worst_jitter_loss_pp = max(worst_jitter_loss_pp, loss_pp)
        family_jitter_ok = worst_jitter_loss_pp <= JITTER_MAX_LOSS_PP
        jitter_pass &= family_jitter_ok
        gate[f"worst_jitter_loss_pp_{family}"] = float(worst_jitter_loss_pp)
        gate[f"jitter_gate_pass_{family}"] = bool(family_jitter_ok)

    gate["causal_retention_pass"] = bool(causal_pass)
    gate["jitter_gate_pass"] = bool(jitter_pass)
    gate["passed"] = bool(causal_pass and jitter_pass)
    return gate


def _family_weighted_arm_accuracies(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    condition: str,
) -> dict[str, float]:
    subset = [row for row in rows if row["family"] == family and row["condition"] == condition]
    if not subset:
        raise ValueError(f"no rows for family={family!r} condition={condition!r}")
    out: dict[str, float] = {}
    for arm in A2_ARMS:
        arm_rows = [row for row in subset if row["arm"] == arm]
        if not arm_rows:
            raise ValueError(f"missing arm {arm!r} for family={family!r} condition={condition!r}")
        out[arm] = event_weighted_mean(
            [{"n_events": row["n_test"], "accuracy": row["accuracy"]} for row in arm_rows],
            "accuracy",
        )
    return out


def evaluate_recording_boundary(
    dataset: Dataset,
    *,
    family: str,
    warmup_samples: int,
) -> tuple[list[dict[str, Any]], int]:
    prep = prepare_dataset_events(
        dataset,
        k_neighbors=K_NEIGHBORS,
        window=WINDOW,
        footprint_layout="relative",
    )
    guard = shared_event_guard_mask(
        prep["times"],
        warmup_samples=warmup_samples,
        n_samples=dataset.n_samples,
    )
    n_events_guarded = int(guard.sum())
    if n_events_guarded < 2:
        raise ValueError(
            f"warmup guard left fewer than two events for {dataset.name!r}; "
            f"warmup_samples={warmup_samples}, surviving={n_events_guarded}"
        )

    rows: list[dict[str, Any]] = []
    reference_times: np.ndarray | None = None
    reference_labels: np.ndarray | None = None

    for condition, filter_mode, jitter_samples in A2_CONDITIONS:
        blocks = build_descriptor_blocks(
            dataset,
            filter_mode=filter_mode,
            jitter_samples=jitter_samples,
            event_mask=guard,
        )
        if reference_times is None:
            reference_times = blocks["times"].copy()
            reference_labels = blocks["labels"].copy()
        else:
            if not np.array_equal(blocks["times"], reference_times):
                raise ValueError(
                    f"condition {condition!r} event times differ from the shared guard population"
                )
            if not np.array_equal(blocks["labels"], reference_labels):
                raise ValueError(
                    f"condition {condition!r} labels differ from the shared guard population"
                )

        train_idx, test_idx = _chronological_split(blocks["times"])
        for arm in A2_ARMS:
            metrics = evaluate_arm_rows(
                blocks,
                train_idx,
                test_idx,
                arm=arm,
                posneg_bits=FROZEN_POSNEG_BITS,
                lat_bits=FROZEN_LAT_BITS,
                amp_gate=FROZEN_AMP_GATE,
                recording_name=dataset.name,
            )
            rows.append(
                {
                    "dataset": dataset.name,
                    "family": family,
                    "scene_type": _scene_type(dataset.name, family),
                    "mearec_n_units": _mearec_n_units(dataset.name, family),
                    "arm": arm,
                    "posneg_bits": FROZEN_POSNEG_BITS,
                    "lat_bits": FROZEN_LAT_BITS,
                    "amp_gate": FROZEN_AMP_GATE,
                    "condition": condition,
                    "filter_mode": filter_mode,
                    "jitter_samples": int(jitter_samples),
                    "warmup_samples": int(warmup_samples),
                    "n_events_guarded": n_events_guarded,
                    **metrics,
                }
            )
    return rows, n_events_guarded


def summarize_boundary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    warmup_samples: int,
    guard_surviving_event_counts: Mapping[str, int],
) -> dict[str, Any]:
    per_family_per_condition: dict[str, dict[str, dict[str, float]]] = {}
    for family in ("hj", "mearec"):
        per_family_per_condition[family] = {}
        for condition, _, _ in A2_CONDITIONS:
            per_family_per_condition[family][condition] = _family_weighted_arm_accuracies(
                rows,
                family=family,
                condition=condition,
            )

    gate_results = compute_a2_gate_results(per_family_per_condition)
    per_block: dict[str, Any] = {}
    for family in ("hj", "mearec"):
        per_block[family] = {}
        for condition, _, _ in A2_CONDITIONS:
            accs = per_family_per_condition[family][condition]
            per_block[family][condition] = {
                "d9": accs[CONTROL_ARM],
                "com_posneg": accs["com_posneg"],
                "com_posneg_lat_width": accs[CANDIDATE_ARM],
                "lat_width_gain_over_d9_proxy": float(accs[CANDIDATE_ARM] - accs[CONTROL_ARM]),
                "lat_width_incremental_gain": float(accs[CANDIDATE_ARM] - accs["com_posneg"]),
                "lat_width_gain_ratio_vs_posneg": lat_width_gain_over_d9(accs),
            }

    return {
        "plan": "shape_orthogonal_descriptor_plan.md",
        "phase": "A2",
        "warmup_samples": int(warmup_samples),
        "warmup_justification": WARMUP_JUSTIFICATION,
        "frozen_configuration": {
            "arm": CANDIDATE_ARM,
            "posneg_bits": FROZEN_POSNEG_BITS,
            "lat_bits": FROZEN_LAT_BITS,
            "amp_gate": FROZEN_AMP_GATE,
        },
        "phase_a_reference_family_accuracy": PHASE_A_FAMILY_REFERENCE,
        "guard_surviving_event_counts": dict(guard_surviving_event_counts),
        "per_family_per_condition_accuracy": per_family_per_condition,
        "per_block_breakdown": per_block,
        "gate_results": gate_results,
        "passed": bool(gate_results["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape descriptor Phase A2 causal/jitter boundary")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Override locked development set as family:name pairs",
    )
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--warmup-samples", type=int, default=DEFAULT_WARMUP_SAMPLES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if isinstance(args.warmup_samples, bool) or not isinstance(args.warmup_samples, (int, np.integer)):
        raise ValueError("warmup_samples must be a non-negative integer")
    if int(args.warmup_samples) < 0:
        raise ValueError("warmup_samples must be non-negative")

    recordings = (
        _parse_recordings_arg(args.recordings)
        if args.recordings is not None
        else list(LOCKED_DEVELOPMENT_RECORDINGS)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    guard_counts: dict[str, int] = {}
    for family, dataset in _iter_recordings(recordings, args.duration, Path(args.mearec_npz_dir)):
        if not args.quiet:
            print(f"{family}: {dataset.name}", flush=True)
        rec_rows, n_guarded = evaluate_recording_boundary(
            dataset,
            family=family,
            warmup_samples=int(args.warmup_samples),
        )
        rows.extend(rec_rows)
        guard_counts[dataset.name] = n_guarded

    csv_path = out_dir / "shape_causal_boundary.csv"
    write_csv(csv_path, [{key: row[key] for key in A2_CSV_FIELDNAMES} for row in rows])

    summary = summarize_boundary_rows(
        rows,
        warmup_samples=int(args.warmup_samples),
        guard_surviving_event_counts=guard_counts,
    )
    json_path = out_dir / "shape_causal_boundary_summary.json"
    write_json(json_path, summary)

    if not args.quiet:
        print(f"Wrote {csv_path} ({len(rows)} rows)")
        print(f"Wrote {json_path}")
        print(f"Phase A2 passed: {summary['passed']}")


if __name__ == "__main__":
    main()
