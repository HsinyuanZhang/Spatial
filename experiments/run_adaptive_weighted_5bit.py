"""Representative fixed-profile power-of-two weighted-L1 5-bit experiment.

The run is a GT-row static representation ceiling.  A model sees fit rows for
centroids/weights, calibration rows only for its radius, and test labels only
for reported metrics.  ``--pilot`` never starts a confirmation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import Dataset, list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import (
    WAVEFORM_LENGTH,
    candidate_waveform_assign,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz


REPO_ROOT = Path(__file__).resolve().parents[1]
BITS = 5
DESCRIPTOR_DIM = 9
WAVEFORM_TEMPLATE_BITS = WAVEFORM_LENGTH * BITS
RADIUS_PERCENTILES = (95.0, 99.0, 99.9)
SCORE_MODES = ("uniform", "stability", "separation_stability")
WEIGHT_PROFILES = ("top3_x4", "tiered_421")
PILOT_HJ_SHORT_NAMES = {"drift16c_600s_11", "static16c_600s_11"}
PILOT_MEAREC_FILENAMES = {
    "rec_v1_units10_snr5_seed202601.npz",
    "rec_v1_units20_snr5_seed202601.npz",
}
PILOT_INPUT_SUBSET_SHA256 = "9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13"
BASE_REPOSITORY_REVISION = "63ff7d2b84d7b2138887714e96db91071fd1dc0c"
PILOT_DATASET_IDS = {
    "hj": ["hybrid_janelia_drift16c_600s_11", "hybrid_janelia_static16c_600s_11"],
    "mearec": ["mearec_rec_v1_units10_snr5_seed202601", "mearec_rec_v1_units20_snr5_seed202601"],
}


def stable_chronological_three_way_split(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest legal 50/75% split, never separating equal timestamps."""
    values = np.asarray(times, dtype=np.int64).ravel()
    if values.size < 3:
        raise ValueError("at least three events are required for fit/cal/test")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("times must already be stably sorted")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    if boundaries.size < 2:
        raise ValueError("cannot create three partitions without splitting equal timestamps")
    n = values.size
    target_second = 0.75 * n
    best: tuple[float, int, int] | None = None
    for first_index, first in enumerate(boundaries[:-1]):
        later = boundaries[first_index + 1 :]
        insert = int(np.searchsorted(later, target_second, side="left"))
        for index in (insert - 1, insert):
            if not 0 <= index < later.size:
                continue
            second = int(later[index])
            candidate = ((float(first) - 0.50 * n) ** 2 + (float(second) - target_second) ** 2, int(first), second)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, first, second = best
    return np.arange(first), np.arange(first, second), np.arange(second, n)


def stable_chronological_two_way_split(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-timestamp-safe early/late fit halves used only for stability audit."""
    values = np.asarray(times, dtype=np.int64).ravel()
    if values.size < 2 or np.any(values[1:] < values[:-1]):
        raise ValueError("times must be sorted and include at least two rows")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    if boundaries.size == 0:
        raise ValueError("cannot split identical timestamps")
    stop = int(boundaries[np.argmin(np.abs(boundaries - values.size / 2.0))])
    return np.arange(stop), np.arange(stop, values.size)


def frozen_weight_grid() -> list[tuple[str, str]]:
    """One uniform baseline plus the four pre-registered nonuniform profiles."""
    return [("uniform", "uniform")] + [
        (mode, profile) for mode in ("stability", "separation_stability") for profile in WEIGHT_PROFILES
    ]


def validate_hardware_contract(bits: int, descriptor_dim: int) -> None:
    if int(bits) != BITS:
        raise ValueError("weighted representative runner is frozen at B=5")
    if int(descriptor_dim) != DESCRIPTOR_DIM:
        raise ValueError("weighted representative runner is frozen at D=9")


def validate_confirmation_configuration(
    score_mode: str | None, weight_profile: str | None, radius_percentile: float | None
) -> tuple[str, str, float]:
    """Require one explicit frozen non-grid-changing configuration."""
    if score_mode is None or weight_profile is None or radius_percentile is None:
        raise ValueError("confirmation requires score mode, weight profile, and radius percentile")
    config = (str(score_mode), str(weight_profile))
    if config not in frozen_weight_grid():
        raise ValueError("confirmation configuration must be a frozen weighted grid point")
    if config == ("uniform", "uniform"):
        raise ValueError("confirmation configuration must be nonuniform")
    percentile = float(radius_percentile)
    if percentile not in RADIUS_PERCENTILES:
        raise ValueError("frozen radius percentile must be one of 95, 99, or 99.9")
    return config[0], config[1], percentile


def wilson_interval(hits: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial recall estimate."""
    if int(total) < 0 or int(hits) < 0 or int(hits) > int(total):
        raise ValueError("Wilson hits must lie in [0, total]")
    if int(total) == 0:
        return 0.0, 0.0
    n = float(total)
    p = float(hits) / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def weighted_level1_accounting(
    primary_rows: np.ndarray,
    shift1_lanes: np.ndarray,
    shift2_lanes: np.ndarray,
    shifted_lanes: np.ndarray,
    fallback_rows: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    descriptor_dim: int = DESCRIPTOR_DIM,
) -> dict[str, float | int | bool]:
    """Separate weighted primary work from full-D unweighted empty fallback."""
    rows = np.asarray(primary_rows, dtype=np.int64).ravel()
    one = np.asarray(shift1_lanes, dtype=np.int64).ravel()
    two = np.asarray(shift2_lanes, dtype=np.int64).ravel()
    shifted = np.asarray(shifted_lanes, dtype=np.int64).ravel()
    fallback = np.asarray(fallback_rows, dtype=np.int64).ravel()
    counts = np.asarray(candidate_counts, dtype=np.int64).ravel()
    if not (rows.shape == one.shape == two.shape == shifted.shape == fallback.shape == counts.shape):
        raise ValueError("weighted accounting arrays must align")
    if np.any(rows < 0) or np.any(one < 0) or np.any(two < 0) or np.any(shifted < 0) or np.any(fallback < 0):
        raise ValueError("weighted accounting must be nonnegative")
    if not np.array_equal(shifted, one + two):
        raise ValueError("logical shifted lanes must equal shift1 plus shift2 per event")
    if np.any(shifted > int(descriptor_dim) * rows):
        raise ValueError("logical shifted lanes cannot exceed nine lanes per primary row")
    expected_fallback = (counts == 0).astype(np.int64) * rows
    if not np.array_equal(fallback, expected_fallback):
        raise ValueError("fallback rows must equal (C==0) times active all-row unit count")
    primary_lanes = rows * int(descriptor_dim)
    fallback_lanes = fallback * int(descriptor_dim)
    total_rows = rows + fallback
    total_lanes = primary_lanes + fallback_lanes
    primary_additions = primary_lanes - rows
    fallback_additions = fallback_lanes - fallback
    total_additions = total_lanes - total_rows
    n = rows.size
    return {
        "primary_level1_row_comparisons_total": int(rows.sum()),
        "primary_abs_lanes_total": int(primary_lanes.sum()),
        "primary_shift1_lanes_total": int(one.sum()),
        "primary_shift2_lanes_total": int(two.sum()),
        "primary_shifted_lanes_total": int(shifted.sum()),
        "primary_accumulator_additions_total": int(primary_additions.sum()),
        "empty_fallback_row_comparisons_total": int(fallback.sum()),
        "empty_fallback_abs_lanes_total": int(fallback_lanes.sum()),
        "empty_fallback_accumulator_additions_total": int(fallback_additions.sum()),
        "total_level1_row_comparisons": int(total_rows.sum()),
        "total_abs_lanes": int(total_lanes.sum()),
        "total_accumulator_additions": int(total_additions.sum()),
        "mean_weighted_lanes_per_active_row": float(primary_lanes.sum() / rows.sum()) if rows.sum() else 0.0,
        "p95_weighted_lanes_per_event": float(np.percentile(primary_lanes, 95)) if n else 0.0,
        "max_weighted_lanes_per_event": int(primary_lanes.max()) if n else 0,
        "mean_shifted_lanes_per_active_row": float((one.sum() + two.sum()) / rows.sum()) if rows.sum() else 0.0,
        "p95_shifted_lanes_per_event": float(np.percentile(one + two, 95)) if n else 0.0,
        "max_shifted_lanes_per_event": int((one + two).max()) if n else 0,
        "level1_row_accounting_identity_holds": bool(np.array_equal(total_rows, rows + fallback)),
        "level1_lane_accounting_identity_holds": bool(np.array_equal(total_lanes, primary_lanes + fallback * int(descriptor_dim))),
        "primary_shift_accounting_identity_holds": bool(np.array_equal(shifted, one + two)),
        "fallback_expected_row_identity_holds": bool(np.array_equal(fallback, expected_fallback)),
        "total_addition_accounting_identity_holds": bool(np.array_equal(total_additions, total_lanes - total_rows)),
    }


def candidate_category_work(
    candidate_counts: np.ndarray,
    primary_rows: np.ndarray,
    shift1_lanes: np.ndarray,
    shift2_lanes: np.ndarray,
    fallback_rows: np.ndarray,
) -> dict[str, int]:
    """Report symbolic total work independently for C=0, C=1, and C>1."""
    counts = np.asarray(candidate_counts, dtype=np.int64).ravel()
    rows = np.asarray(primary_rows, dtype=np.int64).ravel()
    one = np.asarray(shift1_lanes, dtype=np.int64).ravel()
    two = np.asarray(shift2_lanes, dtype=np.int64).ravel()
    fallback = np.asarray(fallback_rows, dtype=np.int64).ravel()
    if not (counts.shape == rows.shape == one.shape == two.shape == fallback.shape):
        raise ValueError("candidate-work arrays must align")
    result: dict[str, int] = {}
    for name, selected in (("c0", counts == 0), ("c1", counts == 1), ("cgt1", counts > 1)):
        primary_lanes = rows[selected] * DESCRIPTOR_DIM
        fallback_lanes = fallback[selected] * DESCRIPTOR_DIM
        total_rows = rows[selected] + fallback[selected]
        total_lanes = primary_lanes + fallback_lanes
        result[f"{name}_events"] = int(selected.sum())
        result[f"{name}_total_level1_rows"] = int(total_rows.sum())
        result[f"{name}_total_abs_lanes"] = int(total_lanes.sum())
        result[f"{name}_shift1_lanes"] = int(one[selected].sum())
        result[f"{name}_shift2_lanes"] = int(two[selected].sum())
        result[f"{name}_total_accumulator_additions"] = int((total_lanes - total_rows).sum())
    return result


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def json_compact(value: Any) -> str:
    return json.dumps(_jsonable(value), separators=(",", ":"), sort_keys=True)


def _event_weighted_family(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, float]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["score_mode"], row["weight_profile"], float(row["radius_percentile"]))].append(row)
    output: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, bucket in groups.items():
        n = sum(int(row["n_test_gt_rows"]) for row in bucket)
        hits = sum(int(row["true_unit_candidate_hits"]) for row in bucket)
        global_bits = sum(int(row["global_all_unit_waveform_template_bits"]) for row in bucket)
        actual_bits = sum(int(row["actual_waveform_template_bits_read"]) for row in bucket)
        output[key] = {
            "recall": float(hits / n) if n else 0.0,
            "traffic_reduction": float(global_bits / actual_bits) if actual_bits else float("inf"),
            "actual_bits": int(actual_bits), "global_bits": int(global_bits),
        }
    return output


def _traffic_ratio(value: float, baseline: float) -> float:
    if math.isinf(value) and math.isinf(baseline):
        return 1.0
    if baseline == 0.0:
        return float("inf") if value > 0.0 else 1.0
    return float(value / baseline)


def _traffic_ratio_order(candidate_actual_bits: int, baseline_actual_bits: int) -> tuple[int, Fraction]:
    """Exact sortable factor ratio F(candidate)/F(baseline), including zero work."""
    candidate, baseline = int(candidate_actual_bits), int(baseline_actual_bits)
    if candidate == 0 and baseline > 0:
        return (2, Fraction(0, 1))  # +infinity
    if candidate == 0 and baseline == 0:
        return (1, Fraction(1, 1))
    if baseline == 0:
        return (0, Fraction(0, 1))
    return (1, Fraction(baseline, candidate))


def traffic_tolerance_95(candidate_actual_bits: int, baseline_actual_bits: int) -> bool:
    """Exact factor>=0.95 comparison using integer waveform-bit counts."""
    candidate, baseline = int(candidate_actual_bits), int(baseline_actual_bits)
    if baseline == 0:
        return candidate == 0
    if candidate == 0:
        return True
    return 95 * candidate <= 100 * baseline


def traffic_no_worse(candidate_actual_bits: int, baseline_actual_bits: int) -> bool:
    return int(candidate_actual_bits) <= int(baseline_actual_bits)


def traffic_branch_b_110(candidate_actual_bits: int, baseline_actual_bits: int) -> bool:
    """Exact factor>=1.10 comparison; both-zero is not a strict B benefit."""
    candidate, baseline = int(candidate_actual_bits), int(baseline_actual_bits)
    if baseline == 0:
        return False
    if candidate == 0:
        return True
    return 11 * candidate <= 10 * baseline


def weighted_pilot_gate(rows_by_family: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Apply the frozen same-branch, all-row development screen.

    The selected configuration is a report only.  This helper neither mutates
    a configuration nor authorizes a confirmation execution.
    """
    required = {"hj", "mearec"}
    if set(rows_by_family) != required:
        raise ValueError("weighted pilot gate requires exactly hj and mearec rows")
    aggregate = {family: _event_weighted_family(rows) for family, rows in rows_by_family.items()}
    shared = set.intersection(*(set(values) for values in aggregate.values()))
    passing: list[dict[str, Any]] = []
    configuration_audit: list[dict[str, Any]] = []
    for config in shared:
        mode, profile, percentile = config
        if mode == "uniform":
            continue
        baseline = ("uniform", "uniform", percentile)
        audit: dict[str, Any] = {"config": config, "baseline": baseline, "per_record": [], "family_branch": {}, "tie_break_keys": None}
        if any(baseline not in aggregate[family] for family in required):
            audit["gate_pass"] = False; audit["reason"] = "missing_uniform_baseline_configuration"; configuration_audit.append(audit); continue
        record_ok = True
        deltas: dict[str, float] = {}
        ratios: dict[str, float] = {}
        for family in required:
            candidates = {
                row["dataset"]: row for row in rows_by_family[family]
                if (row["score_mode"], row["weight_profile"], float(row["radius_percentile"])) == config
            }
            baselines = {
                row["dataset"]: row for row in rows_by_family[family]
                if (row["score_mode"], row["weight_profile"], float(row["radius_percentile"])) == baseline
            }
            if set(candidates) != set(baselines):
                record_ok = False; audit["per_record"].append({"family": family, "pass": False, "reason": "candidate_baseline_dataset_set_mismatch", "candidate_ids": sorted(candidates), "baseline_ids": sorted(baselines)}); continue
            for dataset, row in candidates.items():
                base = baselines[dataset]
                stability_pass=bool(row.get("early_late_shift_evaluable", True)) and float(row.get("early_late_shift_code_agreement", 1.0)) >= 0.75
                recall_pass=float(row["true_unit_candidate_recall"]) >= float(base["true_unit_candidate_recall"]) - 0.005
                traffic_pass=traffic_tolerance_95(int(row["actual_waveform_template_bits_read"]), int(base["actual_waveform_template_bits_read"]))
                reasons=[]
                if not stability_pass: reasons.append("early_late_shift_not_evaluable_or_below_0p75")
                if not recall_pass: reasons.append("recall_below_uniform_minus_0p005")
                if not traffic_pass: reasons.append("traffic_factor_below_0p95_uniform")
                audit["per_record"].append({"family":family,"dataset":dataset,"pass":not reasons,"reasons":reasons,"candidate_recall":float(row["true_unit_candidate_recall"]),"baseline_recall":float(base["true_unit_candidate_recall"]),"candidate_actual_bits":int(row["actual_waveform_template_bits_read"]),"baseline_actual_bits":int(base["actual_waveform_template_bits_read"]),"recall_pass":recall_pass,"traffic_tolerance_pass":traffic_pass,"stability_evaluable":bool(row.get("early_late_shift_evaluable",True)),"stability_agreement":float(row.get("early_late_shift_code_agreement",1.0)),"stability_pass":stability_pass})
                if reasons: record_ok=False
            if config not in aggregate[family] or baseline not in aggregate[family]:
                record_ok=False; continue
            current = aggregate[family][config]
            base_aggregate = aggregate[family][baseline]
            deltas[family] = current["recall"] - base_aggregate["recall"]
            ratios[family] = _traffic_ratio(current["traffic_reduction"], base_aggregate["traffic_reduction"])
        if set(deltas) != required:
            audit["gate_pass"]=False; audit["reason"]="incomplete_family_aggregate"; configuration_audit.append(audit); continue
        branch_a = all(deltas[family] >= 0.005 and traffic_no_worse(aggregate[family][config]["actual_bits"], aggregate[family][baseline]["actual_bits"]) for family in required)
        branch_b = all(traffic_branch_b_110(aggregate[family][config]["actual_bits"], aggregate[family][baseline]["actual_bits"]) and deltas[family] >= 0.0 for family in required)
        audit["family_branch"]={family:{"recall_delta":deltas[family],"traffic_ratio_display":ratios[family],"candidate_actual_bits":aggregate[family][config]["actual_bits"],"baseline_actual_bits":aggregate[family][baseline]["actual_bits"],"branch_a_condition":deltas[family]>=.005 and traffic_no_worse(aggregate[family][config]["actual_bits"],aggregate[family][baseline]["actual_bits"]),"branch_b_condition":deltas[family]>=0 and traffic_branch_b_110(aggregate[family][config]["actual_bits"],aggregate[family][baseline]["actual_bits"])} for family in required}
        exact_min=min((_traffic_ratio_order(aggregate[f][config]["actual_bits"],aggregate[f][baseline]["actual_bits"]) for f in required))
        audit["tie_break_keys"]={"min_recall_delta":min(deltas.values()),"min_traffic_ratio_order":[exact_min[0],str(exact_min[1])],"profile_priority":profile,"score_priority":mode,"radius_percentile":percentile}
        audit["gate_pass"]=bool(record_ok and (branch_a or branch_b)); audit["branch"]="A_recall" if branch_a else "B_traffic" if branch_b else None
        configuration_audit.append(audit)
        if record_ok and (branch_a or branch_b):
            passing.append({
                "config": config,
                "branch": "A_recall" if branch_a else "B_traffic",
                "family_recall_delta": deltas,
                "family_traffic_ratio": ratios,
            })
    profile_order = {"tiered_421": 0, "top3_x4": 1}
    score_order = {"separation_stability": 0, "stability": 1}
    def _selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
        order = min(
            _traffic_ratio_order(
                aggregate[family][item["config"]]["actual_bits"],
                aggregate[family][("uniform", "uniform", item["config"][2])]["actual_bits"],
            )
            for family in required
        )
        return (
            -min(item["family_recall_delta"].values()), -order[0], -order[1],
            profile_order[item["config"][1]], score_order[item["config"][0]], float(item["config"][2]),
        )
    passing.sort(key=_selection_key)
    return {
        "pilot_gate_pass": bool(passing),
        "pilot_gate_scope": "fixed_four_recording_all_rows_same_branch_development_screen",
        "pilot_gate_candidates": int(len(passing)),
        "pilot_gate_selected": passing[0] if passing else None,
        "family_event_weighted": aggregate,
        "nonuniform_configuration_audit": configuration_audit,
    }


def final_099_4x_report(rows_by_family: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Report final-target attainment without making it a pilot selection rule."""
    aggregate = {family: _event_weighted_family(rows) for family, rows in rows_by_family.items()}
    shared = set.intersection(*(set(values) for values in aggregate.values())) if aggregate else set()
    qualifying = [
        config for config in shared
        if all(aggregate[family][config]["recall"] >= 0.99 and aggregate[family][config]["global_bits"] >= 4 * aggregate[family][config]["actual_bits"]
               for family in aggregate)
    ]
    return {
        "final_0p99_4x_scope": "event_weighted_family_aggregate_report_only",
        "final_0p99_4x_pass": bool(qualifying),
        "final_0p99_4x_qualifying_configurations": int(len(qualifying)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pilot_selection_manifest(
    *,
    gate: dict[str, Any],
    final_report: dict[str, Any],
    csv_paths: dict[str, Path],
    rows_by_family: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Build deterministic, self-contained pilot selection provenance."""
    observed_ids = {family: sorted({str(row["dataset"]) for row in rows}) for family, rows in rows_by_family.items()}
    if observed_ids != PILOT_DATASET_IDS:
        raise ValueError("observed pilot dataset IDs do not exactly match locked development IDs")
    split_keys = ("fit_stop_index", "calibration_stop_index", "fit_last_timestamp", "calibration_first_timestamp", "calibration_last_timestamp", "test_first_timestamp", "n_events_gt_rows", "n_fit_gt_rows", "n_calibration_gt_rows", "n_test_gt_rows", "fit_unit_count", "calibration_unit_count", "test_unit_count", "waveform_template_supported_units")
    split_provenance = {family: {dataset: {key: next(row[key] for row in rows if row["dataset"] == dataset) for key in split_keys} for dataset in observed_ids[family]} for family, rows in rows_by_family.items()}
    return {
        "schema": "adaptive_weighted_5bit_pilot_selection_v1",
        "selection_scope": "fixed_four_recording_all_rows_same_branch_development_screen",
        "base_repository_revision": BASE_REPOSITORY_REVISION,
        "duration_seconds": 60,
        "pilot_dataset_ids": PILOT_DATASET_IDS,
        "dataset_split_provenance": split_provenance,
        "locked_input_subset_sha256": PILOT_INPUT_SUBSET_SHA256,
        "locked_input_subset_digest_algorithm": (
            "sha256 of four complete selected docs/dac_experiment_inputs.sha256 lines in "
            "docs/adaptive_masked_5bit_provenance.md path order, joined with newline and final newline"
        ),
        "fixed_grid": {
            "configurations": frozen_weight_grid(),
            "radius_percentiles": list(RADIUS_PERCENTILES),
            "rows_per_record": 15,
            "directory": "all_rows_only",
        },
        "quantile": {"method": "numpy.percentile(method=linear)", "radius_rule": "ceil_percentile_weighted_l1_zero_legal"},
        "per_record_predicates": {
            "recall": "candidate_recall >= uniform_same_percentile - 0.005",
            "traffic": "95*candidate_actual_waveform_bits <= 100*uniform_actual_waveform_bits; zero cases are deterministic",
        },
        "waveform_zero_rules": {"C_le_1": "zero waveform template reads", "C_gt_1": "read C fit-only 320-bit templates", "actual_zero": "traffic factor displayed as infinity; both-zero ratio is one; branch-B does not pass on both-zero"},
        "family_same_branch": {
            "A_recall": "event_weighted recall delta >= 0.005 and actual waveform bits no worse",
            "B_traffic": "event_weighted traffic factor >= 1.10*uniform and recall no worse",
        },
        "tie_break": [
            "largest minimum family recall delta",
            "largest minimum family traffic ratio",
            "tiered_421 before top3_x4",
            "separation_stability before stability",
            "lower radius percentile",
        ],
        "gate": _jsonable(gate),
        "final_0p99_4x_report": _jsonable(final_report),
        "selected_configuration": _jsonable(gate["pilot_gate_selected"]),
        "csv_sha256": {family: sha256_file(path) for family, path in sorted(csv_paths.items())},
    }


def write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(payload), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def load_verified_selection_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify an immutable pilot manifest before confirmation can run."""
    actual = sha256_file(path)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError("selection manifest SHA-256 mismatch")
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "adaptive_weighted_5bit_pilot_selection_v1":
        raise ValueError("selection manifest schema mismatch")
    if payload.get("selection_scope") != "fixed_four_recording_all_rows_same_branch_development_screen":
        raise ValueError("selection manifest scope mismatch")
    if payload.get("base_repository_revision") != BASE_REPOSITORY_REVISION:
        raise ValueError("selection manifest base revision mismatch")
    if payload.get("duration_seconds") != 60:
        raise ValueError("selection manifest duration mismatch")
    if payload.get("pilot_dataset_ids") != PILOT_DATASET_IDS:
        raise ValueError("selection manifest pilot dataset IDs mismatch")
    if payload.get("locked_input_subset_sha256") != PILOT_INPUT_SUBSET_SHA256:
        raise ValueError("selection manifest locked input digest mismatch")
    grid = payload.get("fixed_grid")
    expected_grid = {"configurations": [list(item) for item in frozen_weight_grid()], "radius_percentiles": list(RADIUS_PERCENTILES), "rows_per_record": 15, "directory": "all_rows_only"}
    if grid != expected_grid:
        raise ValueError("selection manifest fixed grid mismatch")
    hashes = payload.get("csv_sha256")
    hex_digits = frozenset("0123456789abcdefABCDEF")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"hj", "mearec"}
        or not all(
            isinstance(hashes[key], str)
            and len(hashes[key]) == 64
            and set(hashes[key]) <= hex_digits
            for key in hashes
        )
    ):
        raise ValueError("selection manifest CSV hash structure mismatch")
    selected = payload.get("selected_configuration")
    if not isinstance(selected, dict) or not selected.get("config"):
        raise ValueError("selection manifest has no selected configuration; confirmation is prohibited")
    config = selected["config"]
    if not isinstance(config, list) or len(config) != 3 or tuple(config[:2]) not in frozen_weight_grid() or float(config[2]) not in RADIUS_PERCENTILES:
        raise ValueError("selection manifest selected configuration is outside the frozen grid")
    if config[0] == "uniform":
        raise ValueError("selection manifest selected configuration must be nonuniform")
    gate = payload.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("pilot_gate_pass") is not True
        or not isinstance(gate.get("pilot_gate_candidates"), int)
        or gate["pilot_gate_candidates"] < 1
    ):
        raise ValueError("selection manifest pilot gate did not pass; confirmation is prohibited")
    if gate.get("pilot_gate_scope") != payload["selection_scope"]:
        raise ValueError("selection manifest pilot gate scope mismatch")
    if gate.get("pilot_gate_selected") != selected:
        raise ValueError("selection manifest selected configuration does not match the pilot gate")
    audits = gate.get("nonuniform_configuration_audit")
    expected_audit_configs = {
        (mode, profile, percentile)
        for mode, profile in frozen_weight_grid()
        if mode != "uniform"
        for percentile in RADIUS_PERCENTILES
    }
    if (
        not isinstance(audits, list)
        or len(audits) != len(expected_audit_configs)
        or {
            tuple(audit.get("config", ()))
            for audit in audits
            if isinstance(audit, dict)
        }
        != expected_audit_configs
    ):
        raise ValueError("selection manifest nonuniform gate audit grid mismatch")
    selected_audits = [
        audit for audit in audits
        if isinstance(audit, dict) and tuple(audit.get("config", ())) == tuple(config)
    ]
    if len(selected_audits) != 1 or selected_audits[0].get("gate_pass") is not True:
        raise ValueError("selection manifest selected configuration did not pass its gate audit")
    return payload


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    return {_scalar(unit): np.asarray(waveforms)[np.asarray(labels) == unit].mean(axis=0) for unit in units}


def _per_unit_quality(candidates: Sequence[np.ndarray], labels: np.ndarray) -> tuple[dict[str, float | int], str]:
    target = np.asarray(labels)
    details = []
    for unit in np.unique(target):
        event_indices = np.flatnonzero(target == unit)
        hits = int(sum(unit in candidates[index] for index in event_indices))
        lo, hi = wilson_interval(hits, event_indices.size)
        details.append({"label": _scalar(unit), "events": int(event_indices.size), "hits": hits,
                        "recall": float(hits / event_indices.size), "wilson95_low": lo, "wilson95_high": hi})
    recalls = np.sort(np.asarray([row["recall"] for row in details], dtype=float))
    if not details:
        return {"per_unit_recall_min": 0.0, "per_unit_recall_p05": 0.0, "per_unit_recall_p10": 0.0,
                "per_unit_recall_worst_decile": 0.0, "per_unit_test_events_min": 0, "per_unit_test_events_max": 0}, json_compact(details)
    return {
        "per_unit_recall_min": float(recalls[0]), "per_unit_recall_p05": float(np.percentile(recalls, 5)),
        "per_unit_recall_p10": float(np.percentile(recalls, 10)),
        "per_unit_recall_worst_decile": float(recalls[:max(1, math.ceil(recalls.size / 10))].mean()),
        "per_unit_test_events_min": int(min(row["events"] for row in details)),
        "per_unit_test_events_max": int(max(row["events"] for row in details)),
    }, json_compact(details)


def _true_distance_summary(codes: np.ndarray, labels: np.ndarray, model: Any) -> dict[str, float | int]:
    index = {_scalar(unit): row for row, unit in enumerate(model.units_)}
    distances, radii = [], []
    for code, label in zip(np.asarray(codes, dtype=np.int64), np.asarray(labels)):
        row = index.get(_scalar(label))
        if row is None:
            continue
        distances.append(int((np.abs(code - model.centroids_[row]) * model.weights_[row]).sum()))
        radii.append(int(model.radii_[row]))
    if not distances:
        return {"true_distance_evaluable_events": 0}
    distance = np.asarray(distances, dtype=float); radius = np.asarray(radii, dtype=float); margin = distance - radius
    normalized = distance / np.maximum(radius, 1.0)
    output: dict[str, float | int] = {"true_distance_evaluable_events": int(distance.size)}
    for name, values in (("true_weighted_distance", distance), ("true_radius", radius),
                         ("true_distance_minus_radius", margin), ("true_normalized_distance", normalized)):
        output[f"{name}_mean"] = float(values.mean()); output[f"{name}_p05"] = float(np.percentile(values, 5))
        output[f"{name}_p50"] = float(np.percentile(values, 50)); output[f"{name}_p95"] = float(np.percentile(values, 95))
        output[f"{name}_max"] = float(values.max())
    output["true_normalized_distance_zero_radius_denominator"] = "max(radius,1)"
    return output


def _early_late_shift_stability(features: np.ndarray, labels: np.ndarray, times: np.ndarray, *, mode: str, profile: str) -> dict[str, Any]:
    from Spatial.algorithms.adaptive_weighted_range_search import AdaptiveWeightedRangeSearch
    early, late = stable_chronological_two_way_split(times)
    core_profile = "top3_x4" if mode == "uniform" else profile
    left = AdaptiveWeightedRangeSearch(n_bits=BITS).fit(features[early], labels[early], score_mode=mode, rank_profile=core_profile)
    right = AdaptiveWeightedRangeSearch(n_bits=BITS).fit(features[late], labels[late], score_mode=mode, rank_profile=core_profile)
    a = {_scalar(unit): row for row, unit in enumerate(left.units_)}; b = {_scalar(unit): row for row, unit in enumerate(right.units_)}
    early_counts={_scalar(unit):int(np.sum(labels[early]==unit)) for unit in left.units_}; late_counts={_scalar(unit):int(np.sum(labels[late]==unit)) for unit in right.units_}
    shared_ids = sorted(set(a) & set(b), key=repr)
    common = [unit for unit in shared_ids if early_counts[unit] >= 5 and late_counts[unit] >= 5]
    excluded = [unit for unit in shared_ids if unit not in common]
    support = [{"label": unit, "early_events": early_counts[unit], "late_events": late_counts[unit], "eligible": unit in common} for unit in shared_ids]
    if not common:
        return {"early_late_shift_evaluable": False, "early_late_shift_common_units": 0,
                "early_late_shift_eligible_units":0,"early_late_shift_excluded_units":int(len(shared_ids)),"early_late_shift_denominator_codes":0,
                "early_late_shift_code_agreement": float("nan"), "early_late_shift_mean_abs_code_change": float("nan"),
                "early_late_shift_per_unit_json": "[]", "early_late_shift_transition_json": "{}",
                "early_late_shift_support_json": json_compact(support), "early_late_shift_eligible_ids_json": "[]", "early_late_shift_excluded_ids_json": json_compact(excluded)}
    per_unit = []; transitions = np.zeros((3, 3), dtype=np.int64); changes=[]; equal=0
    for unit in common:
        x=left.shift_codes_[a[unit]]; y=right.shift_codes_[b[unit]]; equal += int((x == y).sum()); changes.extend(np.abs(x-y).tolist())
        for source in range(3):
            for dest in range(3): transitions[source,dest] += int(np.sum((x == source) & (y == dest)))
        per_unit.append({"label":unit,"agreement":float(np.mean(x==y)),"mean_abs_code_change":float(np.mean(np.abs(x-y)))})
    return {"early_late_shift_evaluable": True, "early_late_shift_common_units": len(common),"early_late_shift_eligible_units":len(common),"early_late_shift_excluded_units":int(len(excluded)),"early_late_shift_denominator_codes":int(DESCRIPTOR_DIM*len(common)),
            "early_late_shift_code_agreement": float(equal/(DESCRIPTOR_DIM*len(common))),
            "early_late_shift_mean_abs_code_change": float(np.mean(changes)),
            "early_late_shift_per_unit_json": json_compact(per_unit), "early_late_shift_transition_json": json_compact(transitions),
            "early_late_shift_support_json": json_compact(support), "early_late_shift_eligible_ids_json": json_compact(common), "early_late_shift_excluded_ids_json": json_compact(excluded)}


def _candidate_change_metrics(weighted: Sequence[np.ndarray], uniform: Sequence[np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    target=np.asarray(labels); jacc=[]; transitions=np.zeros((3,3),dtype=np.int64); target_delta=[]; non_target_delta=[]
    for a,b,label in zip(weighted, uniform, target):
        sa=set(a.tolist()); sb=set(b.tolist()); union=sa|sb; jacc.append(len(sa&sb)/len(union) if union else 1.0)
        ca=0 if len(a)==0 else 1 if len(a)==1 else 2; cb=0 if len(b)==0 else 1 if len(b)==1 else 2; transitions[cb,ca]+=1
        target_delta.append(int(label in sa)-int(label in sb)); non_target_delta.append((len(sa)-int(label in sa))-(len(sb)-int(label in sb)))
    return {"weighted_vs_uniform_candidate_jaccard_mean":float(np.mean(jacc)), "weighted_vs_uniform_c_transition_json":json_compact(transitions),
            "weighted_vs_uniform_target_inclusion_change_total":int(sum(target_delta)), "weighted_vs_uniform_non_target_inclusion_change_total":int(sum(non_target_delta))}


def run_dataset_weighted(
    dataset: Dataset, *, bits: int = BITS, radius_percentiles: Sequence[float] = RADIUS_PERCENTILES,
    configurations: Sequence[tuple[str, str]] | None = None,
    selection_protocol: str = "exploratory_test_grid_no_final_configuration_selection", verbose: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate the 15-config all-row grid under the fixed three-way protocol."""
    from Spatial.algorithms.adaptive_weighted_range_search import AdaptiveWeightedRangeSearch
    prepared = prepare_dataset_events(dataset, k_neighbors=7)
    fit_idx, cal_idx, test_idx = stable_chronological_three_way_split(prepared["times"])
    x, y = prepared["descriptor"], prepared["labels"]; validate_hardware_contract(bits, x.shape[1])
    xfit,yfit=x[fit_idx],y[fit_idx]; xcal,ycal=x[cal_idx],y[cal_idx]; xtest,ytest=x[test_idx],y[test_idx]
    if verbose: print(f"{dataset.name}: fit/cal/test={fit_idx.size}/{cal_idx.size}/{test_idx.size}, D=9")
    rows=[]; config_list=frozen_weight_grid() if configurations is None else list(configurations)
    for percentile in dict.fromkeys(float(item) for item in radius_percentiles):
        uniform_candidates=None; uniform_radii_by_unit=None
        for mode,profile in config_list:
            core_profile="top3_x4" if mode=="uniform" else profile
            model=AdaptiveWeightedRangeSearch(n_bits=BITS,radius_percentile=percentile).fit(xfit,yfit,score_mode=mode,rank_profile=core_profile).calibrate(xcal,ycal)
            codes=model.transform(xtest); candidates,diag=model.query_codes(codes)
            counts=np.asarray([len(c) for c in candidates],dtype=np.int64)
            waves=_waveform_centroids(prepared["waveforms"][fit_idx],yfit,model.units_)
            pred,wave_meta=candidate_waveform_assign(candidates,prepared["waveforms"][test_idx],waves,codes,model.centroids_,model.units_,metric="l1")
            fallback=np.asarray(wave_meta["spatial_fallback_row_comparisons"],dtype=np.int64)
            expected=(counts==0).astype(np.int64)*model.units_.size
            if not np.array_equal(fallback,expected): raise AssertionError("empty fallback rows must equal (C==0)*U")
            logical=weighted_level1_accounting(diag["logical_primary_row_comparisons"],diag["logical_shift1_lanes"],diag["logical_shift2_lanes"],diag["logical_shifted_lanes"],fallback,counts)
            quality,per_unit_json=_per_unit_quality(candidates,ytest); hits=np.asarray([lab in cand for lab,cand in zip(ytest,candidates)],dtype=bool)
            lo,hi=wilson_interval(int(hits.sum()),ytest.size); reads=np.asarray(wave_meta["candidate_waveform_reads"],dtype=np.int64)
            global_bits=int(ytest.size)*int(model.units_.size)*WAVEFORM_TEMPLATE_BITS; actual_bits=int(reads.sum())*WAVEFORM_TEMPLATE_BITS
            stability=_early_late_shift_stability(xfit,yfit,prepared["times"][fit_idx],mode=mode,profile=profile)
            calibration_codes=model.transform(xcal); cal_candidates,_=model.query_codes(calibration_codes)
            cal_cover=float(np.mean([lab in cand for lab,cand in zip(ycal,cal_candidates)])) if ycal.size else 0.0
            storage=model.memory_accounting()
            if mode=="uniform":
                uniform_candidates=candidates
                uniform_radii_by_unit={_scalar(unit):int(model.radii_[index]) for index,unit in enumerate(model.units_)}
                categories=np.where(counts==0,0,np.where(counts==1,1,2))
                change={"weighted_vs_uniform_candidate_jaccard_mean":1.0,"weighted_vs_uniform_c_transition_json":json_compact(np.diag(np.bincount(categories,minlength=3))),"weighted_vs_uniform_target_inclusion_change_total":0,"weighted_vs_uniform_non_target_inclusion_change_total":0}
            else:
                if uniform_candidates is None: raise AssertionError("uniform must precede nonuniform configurations")
                change=_candidate_change_metrics(candidates,uniform_candidates,ytest)
            if uniform_radii_by_unit is None: raise AssertionError("uniform radii must precede all configurations")
            radius_ratio=[{"label":_scalar(unit),"weighted_radius":int(model.radii_[index]),"uniform_radius":int(uniform_radii_by_unit[_scalar(unit)]),"ratio":float(model.radii_[index]/uniform_radii_by_unit[_scalar(unit)]) if uniform_radii_by_unit[_scalar(unit)] else (1.0 if model.radii_[index]==0 else float("inf"))} for index,unit in enumerate(model.units_)]
            unit_audit={"unit_labels":model.units_,"centroids":model.centroids_,"within_q95":model.within_q95_,"between_median_separation":model.between_median_separation_,"separation_stability_scores":model.separation_stability_scores_,"feature_ranks":model.feature_ranks_,"weights":model.weights_,"shift_codes":model.shift_codes_,"radii":model.radii_}
            hardware={f"logical_{key}":value for key,value in logical.items()}
            rows.append({
                "dataset":dataset.name,"n_channels":int(dataset.n_channels),"n_units":int(model.units_.size),"n_events_gt_rows":int(y.size),"n_fit_gt_rows":int(fit_idx.size),"n_calibration_gt_rows":int(cal_idx.size),"n_test_gt_rows":int(test_idx.size),
                "fit_stop_index":int(fit_idx.size),"calibration_stop_index":int(cal_idx.size+fit_idx.size),"fit_last_timestamp":int(prepared["times"][fit_idx[-1]]),"calibration_first_timestamp":int(prepared["times"][cal_idx[0]]),"calibration_last_timestamp":int(prepared["times"][cal_idx[-1]]),"test_first_timestamp":int(prepared["times"][test_idx[0]]),"fit_unit_count":int(np.unique(yfit).size),"calibration_unit_count":int(np.unique(ycal).size),"test_unit_count":int(np.unique(ytest).size),"waveform_template_supported_units":int(len(waves)),
                "bits":BITS,"descriptor_dim":DESCRIPTOR_DIM,"offline_isolation":"gt_event_rows_equal_timestamp_safe_chronological_fit50_cal25_test25","preprocessing":"full_recording_zero_phase_filtfilt_offline","gt_protocol":"static_representation_ceiling_gt_fit_cal_rows_test_labels_metrics_only","selection_protocol":selection_protocol,
                "score_mode":mode,"weight_profile":profile,"radius_percentile":percentile,"radius_rule":"ceil_percentile_weighted_l1_numpy_linear_zero_legal","all_rows_active":True,
                "primary_candidate_count_semantics":"C_is_primary_weighted_spatial_candidate_count_fallback_does_not_change_C","empty_candidate_fallback":"existing_unweighted_all_centroid_l1_no_waveform_template",
                "true_unit_candidate_recall":float(hits.mean()),"true_unit_candidate_hits":int(hits.sum()),"true_unit_candidate_wilson95_low":lo,"true_unit_candidate_wilson95_high":hi,
                "zero_fraction":float(np.mean(counts==0)),"unique_fraction":float(np.mean(counts==1)),"ambiguous_fraction":float(np.mean(counts>1)),"mean_candidates":float(counts.mean()),"p95_candidates":float(np.percentile(counts,95)),"max_candidates":int(counts.max()),
                "candidate_waveform_reads_total":int(reads.sum()),"actual_waveform_template_bits_read":actual_bits,"global_all_unit_waveform_template_bits":global_bits,"waveform_bit_traffic_reduction_factor":float(global_bits/actual_bits) if actual_bits else float("inf"),"waveform_template_bits_per_candidate":WAVEFORM_TEMPLATE_BITS,"waveform_template_source":"fit_only","final_float_waveform_teacher_accuracy":float(np.mean(pred==ytest)),"waveform_fallback_fraction":float(np.mean(wave_meta["waveform_fallback"])),
                "radius_calibration_count_total":int(model.radius_calibration_counts_.sum()),"radius_calibration_missing_unit_count":int(model.radius_calibration_fallback_mask_.sum()),"calibration_true_unit_weighted_coverage":cal_cover,"radius_zero_count":int(np.sum(model.radii_==0)),"test_labels_absent_from_fit_events":int(np.sum(~np.isin(ytest,model.units_))),"test_labels_absent_from_fit_units":int(np.unique(ytest[~np.isin(ytest,model.units_)]).size),
                "logical_search_payload_bits_per_row":int(storage["logical_search_payload_bits_per_row"]),"logical_optimized_unweighted_payload_bits_per_row":int(storage["optimized_unweighted_payload_bits_per_row"]),"logical_weighted_minus_optimized_unweighted_payload_bits":20,"logical_row_bits_with_metadata_per_row":int(storage["logical_row_bits_with_metadata_per_row"]),"logical_candidate_fifo_entries_total":int(counts.sum()),"logical_initial_centroid_configuration_bits":int(model.units_.size*45),"logical_initial_shift_configuration_bits":int(model.units_.size*18),"logical_initial_radius_configuration_bits":int(model.units_.size*11),"logical_initial_configuration_bits_total":int(model.units_.size*74),"logical_accounting_scope":"algorithm_level_logical_operations_and_payload_only_excludes_decoder_sense_gating_banking_ecc_periphery",
                "weight_shift_audit_json":json_compact(unit_audit),"per_unit_quality_json":per_unit_json,"weighted_vs_uniform_radius_ratio_json":json_compact(radius_ratio),
                **{f"weight{weight}_coordinate_frequency":float(np.mean(model.weights_==weight)) for weight in (1,2,4)},
                "weight_sum_per_row_mean":float(model.weights_.sum(axis=1).mean()),"weight_sum_per_row_p95":float(np.percentile(model.weights_.sum(axis=1),95)),"weight_sum_per_row_max":int(model.weights_.sum(axis=1).max()),
                **quality,**_true_distance_summary(codes,ytest,model),**stability,**change,**hardware,
                **{f"logical_{key}":value for key,value in candidate_category_work(counts,diag["logical_primary_row_comparisons"],diag["logical_shift1_lanes"],diag["logical_shift2_lanes"],fallback).items()},
            })
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _iter_hj(duration: float, *, pilot: bool, confirmation: bool) -> Iterable[Dataset]:
    for scene in list_hybrid_janelia_scenes(ready_only=True):
        short=str(scene.get("short_name","")); development=short in PILOT_HJ_SHORT_NAMES
        if (pilot and not development) or (confirmation and development): continue
        yield load_hybrid_janelia(scene=short,duration_s=duration)


def _iter_mearec(duration: float, npz_dir: Path, *, pilot: bool, confirmation: bool) -> Iterable[Dataset]:
    for path in discover_npz(npz_dir,seeds=None):
        development=path.name in PILOT_MEAREC_FILENAMES
        if (pilot and not development) or (confirmation and development): continue
        yield load_mearec_npz(path,duration_s=duration)


def main() -> None:
    parser=argparse.ArgumentParser(description="Fixed-profile weighted-L1 5-bit representative experiment")
    parser.add_argument("--family",choices=("hj","mearec","all"),default=None)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--confirmation",action="store_true")
    parser.add_argument("--selection-manifest",type=Path,default=None)
    parser.add_argument("--selection-manifest-sha256",default=None)
    parser.add_argument("--duration",type=float,default=60.0); parser.add_argument("--bits",type=int,default=BITS); parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=REPO_ROOT/"output"/"adaptive_weighted_5bit")
    parser.add_argument("--mearec-npz-dir",type=Path,default=DEFAULT_NPZ_DIR,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.pilot==args.confirmation:
        parser.error("choose exactly one of --pilot or --confirmation; no full sweep is automatic")
    if (args.pilot or args.confirmation) and float(args.duration) != 60.0:
        parser.error("pilot and confirmation are frozen at --duration 60")
    if args.pilot and args.family not in (None,"all"): parser.error("--pilot always runs both fixed development families")
    if args.confirmation and args.family is None: parser.error("--confirmation requires explicit --family")
    if args.pilot and (args.selection_manifest or args.selection_manifest_sha256): parser.error("pilot does not accept a selection manifest")
    selected=None
    if args.confirmation:
        if args.selection_manifest is None or args.selection_manifest_sha256 is None: parser.error("--confirmation requires --selection-manifest and --selection-manifest-sha256")
        try: selected=load_verified_selection_manifest(args.selection_manifest,args.selection_manifest_sha256)
        except ValueError as exc: parser.error(str(exc))
    try: validate_hardware_contract(args.bits,DESCRIPTOR_DIM)
    except ValueError as exc: parser.error(str(exc))
    families=("hj","mearec") if args.pilot or args.family=="all" else (args.family,)
    config=None; percentiles=RADIUS_PERCENTILES; protocol="exploratory_test_grid_no_final_configuration_selection"
    if selected is not None:
        chosen=selected["selected_configuration"]["config"]; config=[(str(chosen[0]),str(chosen[1]))]; percentiles=(float(chosen[2]),)
        protocol="frozen_confirmation_manifest_selected_single_all_rows_configuration"
    rows_by_family={}; csv_paths={}
    for family in families:
        stream=_iter_hj(args.duration,pilot=args.pilot,confirmation=args.confirmation) if family=="hj" else _iter_mearec(args.duration,args.mearec_npz_dir,pilot=args.pilot,confirmation=args.confirmation)
        rows=[]
        for dataset in stream:
            rows.extend(run_dataset_weighted(dataset,bits=args.bits,radius_percentiles=percentiles,configurations=config,selection_protocol=protocol,verbose=not args.quiet)); del dataset
        rows_by_family[family]=rows; suffix="pilot" if args.pilot else "confirmation"; path=args.output_dir/f"adaptive_weighted_5bit_{suffix}_{family}_{int(args.duration)}s.csv"; _write_csv(path,rows); csv_paths[family]=path; print(f"Saved {len(rows)} rows: {path}")
    if args.pilot:
        gate=weighted_pilot_gate(rows_by_family); final=final_099_4x_report(rows_by_family)
        manifest=pilot_selection_manifest(gate=gate,final_report=final,csv_paths=csv_paths,rows_by_family=rows_by_family); manifest_path=args.output_dir/f"adaptive_weighted_5bit_pilot_selection_{int(args.duration)}s.json"; write_json_deterministic(manifest_path,manifest)
        print(f"Pilot formal gate pass: {gate['pilot_gate_pass']}; candidates={gate['pilot_gate_candidates']}")
        print(f"Pilot selection manifest: {manifest_path}; sha256={sha256_file(manifest_path)}")
        print(f"Final 0.99/4x report-only pass: {final['final_0p99_4x_pass']}; configurations={final['final_0p99_4x_qualifying_configurations']}")
        print("No confirmation sweep was started automatically.")


if __name__ == "__main__": main()
