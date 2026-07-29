"""Deterministic gate and authorization helpers for the temporal-sketch pilot.

This module intentionally has no dataset, filtering, or runner I/O.  A future
runner supplies one fully audited result row for every ``(delay, M)`` grid point
of each development recording; this module then makes the pre-registered
development decision reproducible and makes confirmation authorization depend
on a reviewed output-manifest anchor rather than on a caller supplied rehash.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "causal_temporal_sketch_pilot_selection_v1"
ANCHOR_SCHEMA = "causal_temporal_sketch_output_authorization_v1"
BASE_REPOSITORY_REVISION = "63ff7d2b84d7b2138887714e96db91071fd1dc0c"
LOCKED_INPUT_SUBSET_SHA256 = "9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13"
PILOT_DATASET_IDS: dict[str, tuple[str, ...]] = {
    "hj": (
        "hybrid_janelia_drift16c_600s_11",
        "hybrid_janelia_static16c_600s_11",
    ),
    "mearec": (
        "mearec_rec_v1_units10_snr5_seed202601",
        "mearec_rec_v1_units20_snr5_seed202601",
    ),
}
LOCKED_PILOT_SOURCE_SHA256: dict[str, str] = {
    "hybrid_janelia_drift16c_600s_11": "b8d2b3cf8ea9b9730750bffcdb3b338179ff04e826e60670e5bd758360290c72",
    "hybrid_janelia_static16c_600s_11": "537b0904b2f03643887723ef5fb53f5aaab129195f5187b454fcce9357e68644",
    "mearec_rec_v1_units10_snr5_seed202601": "53049aeae5d4aa1d67c8438a122028465926ae4b454e4babbe0eb9af92f8ff3c",
    "mearec_rec_v1_units20_snr5_seed202601": "db9e9434334942fce8ed6963cb561f5adacca4e4e38b70dd9aeccca5905100d9",
}
TEMPORAL_INPUT_CONTRACT = (
    "filtered_pre_event_normalization_main_channel|"
    "filter=full_recording_zero_phase_butterworth_order3_300_6000Hz|"
    "sampling_frequency=30000Hz|"
    "recording_input=dataset.raw_data_channel_x_sample|"
    "alignment=gt_event_timestamp|"
    "central=max_abs_peak_zero_phase_[t-15,t+15)_per_channel|"
    "window=[t-15,t+49)|"
    "legacy=report_only_peak_normalized_epsilon_1e-8"
)
TEMPORAL_INPUT_CONTRACT_SHA256 = "ac47cddda2eee622089311b669b7438954f73499d6b6b5a71eabf8ba079590e2"
# This is intentionally owned here, rather than imported from the runner:
# selection is the no-I/O authorization boundary and the runner imports it.
# A snapshot is the exact content-addressed working-tree implementation used
# for an invocation; ``BASE_REPOSITORY_REVISION`` is its declared repository
# base, not a claim that this source snapshot is already present in HEAD.
IMPLEMENTATION_SOURCE_PATHS: tuple[str, ...] = (
    "algorithms/__init__.py",
    "algorithms/adaptive_masked_range_search.py",
    "algorithms/adaptive_prototype_search.py",
    "algorithms/adaptive_range_search.py",
    "algorithms/adaptive_weighted_range_search.py",
    "algorithms/causal_preprocessing.py",
    "algorithms/causal_temporal_baselines.py",
    "algorithms/causal_temporal_sketch.py",
    "algorithms/cim_thermometer.py",
    "algorithms/detection.py",
    "algorithms/multiscale_footprint.py",
    "algorithms/reference_port.py",
    "algorithms/se_attention.py",
    "algorithms/som_clustering.py",
    "algorithms/spatial_features.py",
    "algorithms/spatial_footprint.py",
    "algorithms/two_stage_assign.py",
    "algorithms/unsupervised_two_stage.py",
    "data/__init__.py",
    "data/generate_2d_grid.py",
    "data/generate_mearec.py",
    "data/loader.py",
    "data/prepare_hybrid_janelia.py",
    "evaluation/__init__.py",
    "evaluation/metrics.py",
    "experiments/__init__.py",
    "experiments/analyze_soft_loc_errors.py",
    "experiments/causal_temporal_selection.py",
    "experiments/run_adaptive_5bit_range_search.py",
    "experiments/run_adaptive_masked_5bit.py",
    "experiments/run_adaptive_multiprototype_5bit.py",
    "experiments/run_adaptive_weighted_5bit.py",
    "experiments/run_causal_temporal_sketch.py",
    "experiments/run_certified_dual_range_pipeline.py",
    "experiments/run_combined_mscale_se.py",
    "experiments/run_compact_waveform_refinement.py",
    "experiments/run_dual_range_pipeline.py",
    "experiments/run_hj16_method_sweep.py",
    "experiments/run_indexed_confusion_candidate_search.py",
    "experiments/run_lowbit_p2p.py",
    "experiments/run_mearec_method_sweep.py",
    "experiments/run_multiscale_fpn.py",
    "experiments/run_online_confusion_graph.py",
    "experiments/run_pair_conditioned_waveform.py",
    "experiments/run_se_attention.py",
    "experiments/run_spatial_sorting.py",
    "experiments/run_spatial_upper_bound.py",
    "experiments/run_two_stage_unsupervised.py",
    "experiments/run_two_stage_upper_bound.py",
    "experiments/run_unsupervised_two_stage_sweep.py",
    "experiments/sweep_som_params.py",
    "experiments/validate_2d_grid.py",
    "experiments/verify_central_ratio_gate.py",
    "experiments/verify_p2p_cim_approx.py",
    "experiments/verify_waveform_cam_thermometer.py",
)
IMPLEMENTATION_SNAPSHOT_SCOPE = "repository_base_revision_plus_content_addressed_all_runtime_package_python_source_snapshot"
LOCKED_FULL_CORPUS_IDS: dict[str, tuple[str, ...]] = {
    "hj": (
        "hybrid_janelia_drift16c_600s_11", "hybrid_janelia_drift16c_600s_12",
        "hybrid_janelia_drift16c_600s_21", "hybrid_janelia_drift16c_600s_22",
        "hybrid_janelia_drift16c_600s_31", "hybrid_janelia_drift16c_600s_32",
        "hybrid_janelia_static16c_600s_11", "hybrid_janelia_static16c_600s_12",
        "hybrid_janelia_static16c_600s_21", "hybrid_janelia_static16c_600s_22",
        "hybrid_janelia_static16c_600s_31", "hybrid_janelia_static16c_600s_32",
    ),
    "mearec": (
        "mearec_rec_v1_units10_snr12_seed202600", "mearec_rec_v1_units10_snr12_seed202601",
        "mearec_rec_v1_units10_snr2_seed202600", "mearec_rec_v1_units10_snr2_seed202601",
        "mearec_rec_v1_units10_snr3_seed202600", "mearec_rec_v1_units10_snr3_seed202601",
        "mearec_rec_v1_units10_snr5_seed202600", "mearec_rec_v1_units10_snr5_seed202601",
        "mearec_rec_v1_units10_snr8_seed202600", "mearec_rec_v1_units10_snr8_seed202601",
        "mearec_rec_v1_units20_snr12_seed202600", "mearec_rec_v1_units20_snr12_seed202601",
        "mearec_rec_v1_units20_snr2_seed202600", "mearec_rec_v1_units20_snr2_seed202601",
        "mearec_rec_v1_units20_snr3_seed202600", "mearec_rec_v1_units20_snr3_seed202601",
        "mearec_rec_v1_units20_snr5_seed202600", "mearec_rec_v1_units20_snr5_seed202601",
        "mearec_rec_v1_units20_snr8_seed202600", "mearec_rec_v1_units20_snr8_seed202601",
    ),
}
LOCKED_FULL_CORPUS_SHA256 = "6cab753e1f145a7540b7ecdb5ac215ddd2af3d65ae0656c1b0e70b2018b16d08"
DELAYS = (0, 8, 16, 32, 48)
FEATURE_COUNTS = (8, 12, 16)
ROWS_PER_RECORD = len(DELAYS) * len(FEATURE_COUNTS)
FAMILIES = ("hj", "mearec")

_BOOLEAN_INVARIANTS = (
    "raw_input_provenance_ok",
    "row_identity_ok",
    "future_sample_isolation_ok",
    "bit_width_invariants_ok",
    "accounting_invariants_ok",
)
_AUDIT_FIELDS = (
    "fit_event_digest",
    "calibration_event_digest",
    "test_event_digest",
    "candidate_source_digest",
    "raw_input_provenance_digest",
    "teacher_contract_hash",
    "source_file_sha256",
    "source_input_path",
    "source_input_subset_digest",
    "temporal_input_contract_sha256",
    "repository_base_revision",
    "implementation_source_sha256",
    "implementation_source_digest",
    "level1_source_scan_events",
    "level1_source_unit_count",
    "level1_source_descriptor_quantizations",
    "level1_source_centroid_reads",
    "level1_source_l1_absolute_differences",
    "level1_source_l1_reduction_additions",
    "level1_source_radius_threshold_comparisons",
    "level1_source_argmin_reduction_comparisons",
    "level1_c0_reuse_events",
    "scale_audit",
    "logical_extraction_counters",
    "logical_category_decomposition",
    "scale_exponent",
    "selected_filter_ids",
    "fisher_audit",
)
_SPLIT_FIELDS = (
    "n_fit_events", "n_calibration_events", "fit_last_timestamp",
    "calibration_first_timestamp", "calibration_last_timestamp", "test_first_timestamp",
)
_HEX = frozenset("0123456789abcdefABCDEF")

GATE_CONTRACT = {
    "per_record": (
        "recall>=0.99; ambiguous>=100; reads>0; pairs>=5; overall_delta>=-0.015; "
        "ambiguous_delta>=-0.020; all provenance/row/future/bit/accounting invariants true"
    ),
    "per_family": (
        "recall>=0.99; ambiguous>=500; overall_and_ambiguous_delta>=-0.010; "
        "actual_bits>0; global_bits>=4*actual_bits"
    ),
    "tie_break": [
        "lowest delay", "smallest M", "largest minimum family overall delta",
        "largest minimum exact traffic factor", "ascending stable configuration name",
    ],
}

if set(LOCKED_PILOT_SOURCE_SHA256) != {
    dataset for family in PILOT_DATASET_IDS.values() for dataset in family
}:
    raise RuntimeError("locked pilot source-SHA mapping must exactly cover pilot dataset IDs")
if hashlib.sha256(TEMPORAL_INPUT_CONTRACT.encode("utf-8")).hexdigest() != TEMPORAL_INPUT_CONTRACT_SHA256:
    raise RuntimeError("temporal input contract hash constant is inconsistent")
if len(set(IMPLEMENTATION_SOURCE_PATHS)) != len(IMPLEMENTATION_SOURCE_PATHS):
    raise RuntimeError("implementation source paths must be unique")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of an existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def implementation_source_digest(source_sha256: Mapping[str, Any]) -> str:
    """Return the runner-compatible canonical digest of an implementation map."""
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != set(IMPLEMENTATION_SOURCE_PATHS):
        raise ValueError("implementation source SHA-256 map has an unexpected key set")
    if not all(_is_sha256(value) for value in source_sha256.values()):
        raise ValueError("implementation source SHA-256 map must contain only 64-character digests")
    return hashlib.sha256(
        json.dumps(dict(source_sha256), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert a small result structure to deterministic JSON-compatible data."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _reject_nonfinite(value: Any, *, context: str = "payload") -> None:
    """Reject JSON's non-standard NaN/Infinity spellings at every boundary."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, context=f"{context}[{index}]")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json_deterministic(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON; callers must review/track authorization separately."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))


def configuration_name(delay_samples: int, feature_count: int) -> str:
    return f"delay_{int(delay_samples):02d}_m_{int(feature_count):02d}"


def _config_from_row(row: Mapping[str, Any]) -> tuple[int, int]:
    try:
        delay, count = int(row["delay_samples"]), int(row["feature_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each row needs integer delay_samples and feature_count") from exc
    if delay not in DELAYS or count not in FEATURE_COUNTS:
        raise ValueError("row configuration is outside the frozen delay/M grid")
    return delay, count


def _expected_grid() -> set[tuple[int, int]]:
    return {(delay, count) for delay in DELAYS for count in FEATURE_COUNTS}


def _as_nonnegative_int(row: Mapping[str, Any], key: str) -> int:
    if key not in row or isinstance(row[key], bool):
        raise ValueError(f"row is missing integer {key}")
    try:
        value = int(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row has invalid integer {key}") from exc
    if value < 0 or value != row[key]:
        raise ValueError(f"row has invalid non-negative integer {key}")
    return value


def _row_counts(row: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "n_test_events",
        "candidate_recall_hits",
        "c0_events",
        "c1_events",
        "ambiguous_events",
        "candidate_template_reads",
        "distinct_ambiguous_candidate_pairs",
        "overall_correct",
        "full_teacher_overall_correct",
        "ambiguous_correct",
        "full_teacher_ambiguous_correct",
        "global_64x5_template_bits",
        "actual_template_bits",
    )
    values = {key: _as_nonnegative_int(row, key) for key in keys}
    if values["n_test_events"] == 0:
        raise ValueError("n_test_events must be positive")
    if values["candidate_recall_hits"] > values["n_test_events"]:
        raise ValueError("candidate_recall_hits exceeds n_test_events")
    if values["ambiguous_events"] > values["n_test_events"]:
        raise ValueError("ambiguous_events exceeds n_test_events")
    if values["c0_events"] + values["c1_events"] + values["ambiguous_events"] != values["n_test_events"]:
        raise ValueError("C0/C1/ambiguous event counts must partition n_test_events")
    if values["overall_correct"] > values["n_test_events"] or values["full_teacher_overall_correct"] > values["n_test_events"]:
        raise ValueError("overall accuracy hit count exceeds n_test_events")
    if values["ambiguous_correct"] > values["ambiguous_events"] or values["full_teacher_ambiguous_correct"] > values["ambiguous_events"]:
        raise ValueError("ambiguous accuracy hit count exceeds ambiguous_events")
    return values


def _validate_implementation_snapshot(row: Mapping[str, Any]) -> None:
    if row["repository_base_revision"] != BASE_REPOSITORY_REVISION:
        raise ValueError("row repository_base_revision does not match the frozen base revision")
    source_map = row["implementation_source_sha256"]
    digest = implementation_source_digest(source_map)
    if row["implementation_source_digest"] != digest:
        raise ValueError("row implementation source digest does not match its canonical source map")


def _validate_level1_accounting(row: Mapping[str, Any], counts: Mapping[str, int]) -> None:
    keys = (
        "level1_source_scan_events",
        "level1_source_unit_count",
        "level1_source_descriptor_quantizations",
        "level1_source_centroid_reads",
        "level1_source_l1_absolute_differences",
        "level1_source_l1_reduction_additions",
        "level1_source_radius_threshold_comparisons",
        "level1_source_argmin_reduction_comparisons",
        "level1_c0_reuse_events",
    )
    values = {key: _as_nonnegative_int(row, key) for key in keys}
    n_events, units = counts["n_test_events"], values["level1_source_unit_count"]
    if units <= 0:
        raise ValueError("level1_source_unit_count must be positive")
    expected = {
        "level1_source_scan_events": n_events,
        "level1_source_descriptor_quantizations": n_events * 9,
        "level1_source_centroid_reads": n_events * units,
        "level1_source_l1_absolute_differences": n_events * units * 9,
        "level1_source_l1_reduction_additions": n_events * units * 8,
        "level1_source_radius_threshold_comparisons": n_events * units,
        "level1_source_argmin_reduction_comparisons": n_events * (units - 1),
        "level1_c0_reuse_events": counts["c0_events"],
    }
    if any(values[key] != value for key, value in expected.items()):
        raise ValueError("Level-1 source accounting does not match the frozen D9/L1 arithmetic")


_EXTRACTION_KEYS = (
    "logical_raw_prefix_samples_consumed",
    "logical_sample_quantizations",
    "logical_selected_feature_accumulator_updates",
    "logical_positive_coefficient_updates",
    "logical_negative_coefficient_updates",
    "logical_normalization_shifts",
    "logical_emitted_feature_codes",
)
_CATEGORY_TEMPORAL_KEYS = (
    "template_reads",
    "template_bits",
    "sad_absolute_differences",
    "sad_reduction_additions",
)


def _mapping_nonnegative_ints(value: Any, *, keys: Sequence[str], context: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{context} must contain the exact required keys")
    return {key: _as_nonnegative_int(value, key) for key in keys}


def _validate_logical_category_accounting(row: Mapping[str, Any], counts: Mapping[str, int]) -> None:
    extraction = _mapping_nonnegative_ints(
        row["logical_extraction_counters"], keys=_EXTRACTION_KEYS, context="logical_extraction_counters"
    )
    category = row["logical_category_decomposition"]
    if not isinstance(category, Mapping) or set(category) != {"c0", "c1", "cgt1"}:
        raise ValueError("logical_category_decomposition must contain exactly c0/c1/cgt1")
    nested_keys = (*_EXTRACTION_KEYS, *_CATEGORY_TEMPORAL_KEYS)
    category_values = {
        name: _mapping_nonnegative_ints(value, keys=nested_keys, context=f"logical_category_decomposition.{name}")
        for name, value in category.items()
    }
    event_counts = {"c0": counts["c0_events"], "c1": counts["c1_events"], "cgt1": counts["ambiguous_events"]}
    n_events = counts["n_test_events"]
    for key, total in extraction.items():
        if total % n_events:
            raise ValueError("logical extraction totals must divide exactly by n_test_events")
        per_event = total // n_events
        if any(category_values[name][key] != event_counts[name] * per_event for name in event_counts):
            raise ValueError("logical category extraction decomposition is inconsistent")
        if sum(category_values[name][key] for name in event_counts) != total:
            raise ValueError("logical category extraction totals do not sum to the row total")
    primary = {
        "template_reads": _as_nonnegative_int(row, "logical_candidate_template_reads"),
        "template_bits": _as_nonnegative_int(row, "logical_candidate_template_bits"),
        "sad_absolute_differences": _as_nonnegative_int(row, "logical_sad_absolute_differences"),
        "sad_reduction_additions": _as_nonnegative_int(row, "logical_sad_reduction_additions"),
    }
    if primary["template_reads"] != counts["candidate_template_reads"] or primary["template_bits"] != counts["actual_template_bits"]:
        raise ValueError("logical primary template accounting disagrees with gate counters")
    for key, total in primary.items():
        if category_values["c0"][key] != 0 or category_values["c1"][key] != 0:
            raise ValueError("C0 and C1 must consume zero temporal templates")
        if category_values["cgt1"][key] != total or sum(category_values[name][key] for name in event_counts) != total:
            raise ValueError("logical category temporal accounting is inconsistent")


def _validate_saturation_audit(row: Mapping[str, Any]) -> None:
    audit = row["scale_audit"]
    if not isinstance(audit, Mapping):
        raise ValueError("scale_audit must be a mapping")
    n_fit = _as_nonnegative_int(row, "n_fit_events")
    _, feature_count = _config_from_row(row)
    prefix_length = 16 + int(row["delay_samples"])
    for name, samples_per_row in (("sketch_saturation", prefix_length), ("baseline_saturation", 64)):
        value = audit.get(name)
        required = {
            "fit_sample_saturation_fraction", "fit_sample_saturation_population",
            "fit_quantizer_eligible_row_count", "fit_quantizer_source_sample_count",
            "fit_all_sample_saturation_fraction", "fit_all_quantizer_row_count",
            "fit_all_quantizer_source_sample_count",
        }
        if not isinstance(value, Mapping) or not required <= set(value):
            raise ValueError(f"scale_audit.{name} is missing saturation provenance")
        if value["fit_sample_saturation_population"] != "fisher_eligible_fit_rows":
            raise ValueError(f"scale_audit.{name} has an invalid eligible saturation population")
        for fraction in ("fit_sample_saturation_fraction", "fit_all_sample_saturation_fraction"):
            if isinstance(value[fraction], bool) or not isinstance(value[fraction], (int, float)) or not math.isfinite(float(value[fraction])) or not 0.0 <= float(value[fraction]) <= 1.0:
                raise ValueError(f"scale_audit.{name}.{fraction} must be a finite fraction")
        eligible_rows = _as_nonnegative_int(value, "fit_quantizer_eligible_row_count")
        all_rows = _as_nonnegative_int(value, "fit_all_quantizer_row_count")
        if not 0 < eligible_rows <= n_fit or all_rows != n_fit:
            raise ValueError(f"scale_audit.{name} has inconsistent fit row counts")
        if _as_nonnegative_int(value, "fit_quantizer_source_sample_count") != eligible_rows * samples_per_row:
            raise ValueError(f"scale_audit.{name} eligible sample count is inconsistent")
        if _as_nonnegative_int(value, "fit_all_quantizer_source_sample_count") != n_fit * samples_per_row:
            raise ValueError(f"scale_audit.{name} all-row sample count is inconsistent")
    # This binds M to the emitted-feature counter and rejects a nested
    # decomposition that is self-consistent but for a different configuration.
    extraction = row["logical_extraction_counters"]
    if _as_nonnegative_int(extraction, "logical_normalization_shifts") != _as_nonnegative_int(row, "n_test_events") * feature_count:
        raise ValueError("logical normalization shifts must equal n_test_events times M")


def _validate_audit_fields(row: Mapping[str, Any], *, dataset_id: str) -> None:
    missing = [key for key in _AUDIT_FIELDS if key not in row]
    if missing:
        raise ValueError(f"row is missing required audit fields: {', '.join(missing)}")
    for key in (
        "fit_event_digest",
        "calibration_event_digest",
        "test_event_digest",
        "candidate_source_digest",
        "raw_input_provenance_digest",
        "teacher_contract_hash",
        "source_file_sha256",
        "temporal_input_contract_sha256",
    ):
        if not _is_sha256(row[key]):
            raise ValueError(f"row has invalid SHA-256 audit field {key}")
    if row["source_file_sha256"].lower() != LOCKED_PILOT_SOURCE_SHA256[dataset_id]:
        raise ValueError("row source_file_sha256 does not match the locked dataset source")
    if not isinstance(row["source_input_path"], str) or not row["source_input_path"].strip():
        raise ValueError("row source_input_path must be a non-empty string")
    if row["source_input_subset_digest"] != LOCKED_INPUT_SUBSET_SHA256:
        raise ValueError("row source_input_subset_digest does not match the locked pilot subset")
    if row["temporal_input_contract_sha256"].lower() != TEMPORAL_INPUT_CONTRACT_SHA256:
        raise ValueError("row temporal input contract hash does not match the frozen contract")
    counts = _row_counts(row)
    _validate_implementation_snapshot(row)
    _validate_level1_accounting(row, counts)
    _validate_logical_category_accounting(row, counts)
    _validate_saturation_audit(row)
    if isinstance(row["scale_exponent"], bool) or int(row["scale_exponent"]) != row["scale_exponent"]:
        raise ValueError("scale_exponent must be an integer")
    ids = row["selected_filter_ids"]
    if not isinstance(ids, list) or len(ids) not in FEATURE_COUNTS:
        raise ValueError("selected_filter_ids must be an M-length list")
    if any(isinstance(item, bool) or int(item) != item or not 0 <= int(item) < 93 for item in ids):
        raise ValueError("selected_filter_ids must contain dictionary IDs in [0, 93)")
    if len(set(ids)) != len(ids):
        raise ValueError("selected_filter_ids must be unique")
    _validate_fisher_audit(row["fisher_audit"], row["selected_filter_ids"])
    _validate_split_provenance(row)
    for key in _BOOLEAN_INVARIANTS:
        if row.get(key) is not True and row.get(key) is not False:
            raise ValueError(f"row invariant {key} must be an explicit boolean")


def _validate_fisher_audit(audit: Any, selected_filter_ids: Sequence[Any]) -> None:
    """Require enough raw Fisher state to make fit-only selection auditable."""
    required = {
        "eligible_filter_ids", "between", "within", "scores", "ranks",
        "fit_unit_counts", "eligible_unit_mask", "eligible_unit_ids",
    }
    if not isinstance(audit, Mapping) or set(audit) != required:
        raise ValueError("fisher_audit must contain the exact full Fisher audit fields")
    ids = audit["eligible_filter_ids"]
    if not isinstance(ids, list) or not ids or any(isinstance(value, bool) or int(value) != value or not 0 <= int(value) < 93 for value in ids):
        raise ValueError("Fisher eligible_filter_ids are invalid")
    if len(set(ids)) != len(ids) or not set(selected_filter_ids) <= set(ids):
        raise ValueError("selected filters must be unique eligible Fisher IDs")
    length = len(ids)
    for key in ("between", "within", "scores"):
        values = audit[key]
        if not isinstance(values, list) or len(values) != length or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Fisher {key} must be finite and match eligible filter IDs")
    ranks = audit["ranks"]
    valid_ranks = (
        isinstance(ranks, list) and len(ranks) == length
        and all(not isinstance(value, bool) and isinstance(value, int) for value in ranks)
        and set(ranks) == set(range(length))
    )
    if not valid_ranks:
        raise ValueError("Fisher ranks must be a deterministic permutation")
    counts, mask, eligible = audit["fit_unit_counts"], audit["eligible_unit_mask"], audit["eligible_unit_ids"]
    if not isinstance(counts, Mapping) or not counts or not isinstance(mask, Mapping) or set(mask) != set(counts):
        raise ValueError("Fisher unit-count and eligibility-mask audit is malformed")
    if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in counts.values()):
        raise ValueError("Fisher fit unit counts must be non-negative integers")
    if any(value is not True and value is not False for value in mask.values()):
        raise ValueError("Fisher eligible_unit_mask must contain booleans")
    if not isinstance(eligible, list) or len(set(map(str, eligible))) != len(eligible):
        raise ValueError("Fisher eligible_unit_ids must be a unique list")
    expected_eligible = {str(key) for key, value in mask.items() if value is True}
    if {str(value) for value in eligible} != expected_eligible or not expected_eligible:
        raise ValueError("Fisher eligible_unit_ids do not match eligible_unit_mask")
    if any(int(counts[key]) < 10 for key, value in mask.items() if value is True):
        raise ValueError("Fisher-eligible units require at least ten fit events")


def _validate_split_provenance(row: Mapping[str, Any]) -> None:
    values = {key: _as_nonnegative_int(row, key) for key in _SPLIT_FIELDS}
    if values["n_fit_events"] == 0 or values["n_calibration_events"] == 0:
        raise ValueError("fit and calibration split counts must be positive")
    if values["fit_last_timestamp"] >= values["calibration_first_timestamp"]:
        raise ValueError("fit/calibration split timestamps overlap")
    if values["calibration_first_timestamp"] > values["calibration_last_timestamp"]:
        raise ValueError("calibration timestamp bounds are invalid")
    if values["calibration_last_timestamp"] >= values["test_first_timestamp"]:
        raise ValueError("calibration/test split timestamps overlap")


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(_jsonable(row)) for row in sorted(rows, key=lambda item: (str(item["dataset"]), *_config_from_row(item)))]


def validate_development_rows(rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Validate the fixed four-recording, 15-row-per-record development input."""
    if set(rows_by_family) != set(FAMILIES):
        raise ValueError("development rows must contain exactly hj and mearec")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = _stable_rows(rows_by_family[family])
        expected_ids = set(PILOT_DATASET_IDS[family])
        observed_ids = {str(row.get("dataset")) for row in rows}
        if observed_ids != expected_ids:
            raise ValueError(f"{family} development dataset IDs do not match the locked pilot IDs")
        by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in expected_ids}
        for row in rows:
            dataset = str(row.get("dataset"))
            _config_from_row(row)
            _row_counts(row)
            _validate_audit_fields(row, dataset_id=dataset)
            if len(row["selected_filter_ids"]) != int(row["feature_count"]):
                raise ValueError("selected_filter_ids length must equal feature_count")
            by_dataset[dataset].append(row)
        for dataset, bucket in by_dataset.items():
            configurations = [_config_from_row(row) for row in bucket]
            if len(bucket) != ROWS_PER_RECORD or set(configurations) != _expected_grid() or len(set(configurations)) != ROWS_PER_RECORD:
                raise ValueError(f"{dataset} must provide every one of the 15 frozen grid rows exactly once")
            # The source is frozen before the Level-2 grid.  These fields must
            # therefore be identical for every representation of this recording.
            source_keys = (
                "n_test_events", "candidate_recall_hits", "ambiguous_events",
                "candidate_template_reads", "distinct_ambiguous_candidate_pairs",
                "global_64x5_template_bits", "candidate_source_digest",
                "fit_event_digest", "calibration_event_digest", "test_event_digest",
                "raw_input_provenance_digest", "teacher_contract_hash",
                "source_file_sha256", "source_input_path", "source_input_subset_digest",
                "temporal_input_contract_sha256",
                "repository_base_revision", "implementation_source_sha256", "implementation_source_digest",
                "level1_source_scan_events", "level1_source_unit_count",
                "level1_source_descriptor_quantizations", "level1_source_centroid_reads",
                "level1_source_l1_absolute_differences", "level1_source_l1_reduction_additions",
                "level1_source_radius_threshold_comparisons", "level1_source_argmin_reduction_comparisons",
                "level1_c0_reuse_events",
                *_SPLIT_FIELDS,
            )
            reference = {key: bucket[0][key] for key in source_keys}
            if any(any(row[key] != reference[key] for key in source_keys) for row in bucket[1:]):
                raise ValueError(f"{dataset} has a non-frozen candidate/source audit across grid rows")
        normalized[family] = rows
    all_rows = [row for family in FAMILIES for row in normalized[family]]
    snapshot_keys = ("repository_base_revision", "implementation_source_sha256", "implementation_source_digest")
    snapshot_reference = {key: all_rows[0][key] for key in snapshot_keys}
    if any(any(row[key] != snapshot_reference[key] for key in snapshot_keys) for row in all_rows[1:]):
        raise ValueError("all 60 development rows must share one implementation source snapshot")
    return normalized


def _record_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    counts = _row_counts(row)
    recall = counts["candidate_recall_hits"] / counts["n_test_events"]
    overall_delta = (counts["overall_correct"] - counts["full_teacher_overall_correct"]) / counts["n_test_events"]
    ambiguous_delta = (
        (counts["ambiguous_correct"] - counts["full_teacher_ambiguous_correct"]) / counts["ambiguous_events"]
        if counts["ambiguous_events"] else float("-inf")
    )
    predicates = {
        "candidate_recall_ge_0p99": recall >= 0.99,
        "ambiguous_events_ge_100": counts["ambiguous_events"] >= 100,
        "template_reads_positive": counts["candidate_template_reads"] > 0,
        "distinct_pairs_ge_5": counts["distinct_ambiguous_candidate_pairs"] >= 5,
        "overall_delta_ge_minus_0p015": overall_delta >= -0.015,
        "ambiguous_delta_ge_minus_0p020": ambiguous_delta >= -0.020,
        "all_invariants_true": all(row.get(key) is True for key in _BOOLEAN_INVARIANTS),
    }
    return {
        "dataset": str(row["dataset"]),
        "pass": all(predicates.values()),
        "candidate_recall": recall,
        "ambiguous_events": counts["ambiguous_events"],
        "candidate_template_reads": counts["candidate_template_reads"],
        "distinct_ambiguous_candidate_pairs": counts["distinct_ambiguous_candidate_pairs"],
        "overall_delta": overall_delta,
        "ambiguous_delta": ambiguous_delta,
        "predicates": predicates,
    }


def _family_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in _row_counts(rows[0])}
    for row in rows:
        for key, value in _row_counts(row).items():
            totals[key] += value
    n = totals["n_test_events"]
    ambiguous = totals["ambiguous_events"]
    recall = totals["candidate_recall_hits"] / n
    overall_delta = (totals["overall_correct"] - totals["full_teacher_overall_correct"]) / n
    ambiguous_delta = (totals["ambiguous_correct"] - totals["full_teacher_ambiguous_correct"]) / ambiguous if ambiguous else float("-inf")
    actual, global_bits = totals["actual_template_bits"], totals["global_64x5_template_bits"]
    factor = Fraction(global_bits, actual) if actual else None
    predicates = {
        "candidate_recall_ge_0p99": recall >= 0.99,
        "ambiguous_events_ge_500": ambiguous >= 500,
        "overall_delta_ge_minus_0p010": overall_delta >= -0.010,
        "ambiguous_delta_ge_minus_0p010": ambiguous_delta >= -0.010,
        "actual_bits_positive": actual > 0,
        "global_bits_ge_4x_actual": global_bits >= 4 * actual,
    }
    return {
        "pass": all(predicates.values()),
        "n_test_events": n,
        "candidate_recall_hits": totals["candidate_recall_hits"],
        "candidate_recall": recall,
        "ambiguous_events": ambiguous,
        "overall_delta": overall_delta,
        "ambiguous_delta": ambiguous_delta,
        "global_64x5_template_bits": global_bits,
        "actual_template_bits": actual,
        "traffic_factor": None if factor is None else str(factor),
        "predicates": predicates,
    }


def _selection_key(audit: Mapping[str, Any]) -> tuple[Any, ...]:
    config = audit["config"]
    family = audit["family"]
    minimum_delta = min(float(family[name]["overall_delta"]) for name in FAMILIES)
    factors = [Fraction(str(family[name]["traffic_factor"])) for name in FAMILIES]
    minimum_factor = min(factors)
    return (
        int(config["delay_samples"]),
        int(config["feature_count"]),
        -minimum_delta,
        -minimum_factor,
        str(config["name"]),
    )


def evaluate_development_gate(rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Apply the exact temporal-sketch development gate and deterministic tie-break."""
    normalized = validate_development_rows(rows_by_family)
    audit_rows: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    for delay, feature_count in sorted(_expected_grid()):
        config_rows = {
            family: [row for row in normalized[family] if _config_from_row(row) == (delay, feature_count)]
            for family in FAMILIES
        }
        records = {family: [_record_audit(row) for row in config_rows[family]] for family in FAMILIES}
        family = {name: _family_audit(config_rows[name]) for name in FAMILIES}
        config = {
            "delay_samples": delay,
            "feature_count": feature_count,
            "name": configuration_name(delay, feature_count),
        }
        passed = all(item["pass"] for bucket in records.values() for item in bucket) and all(item["pass"] for item in family.values())
        entry = {"config": config, "pass": passed, "records": records, "family": family, "tie_key": None}
        if passed:
            entry["tie_key"] = {
                "delay_samples": delay,
                "feature_count": feature_count,
                "minimum_family_overall_delta": min(float(family[name]["overall_delta"]) for name in FAMILIES),
                "minimum_family_traffic_factor": str(min(Fraction(str(family[name]["traffic_factor"])) for name in FAMILIES)),
                "stable_name": config["name"],
            }
            passing.append(entry)
        audit_rows.append(entry)
    passing.sort(key=_selection_key)
    selected = passing[0]["config"] if passing else None
    return {
        "scope": "fixed_four_recording_temporal_sketch_development_gate",
        "gate_pass": bool(passing),
        "complete_grid_ok": True,
        "candidate_count": len(passing),
        "configuration_audit": audit_rows,
        "selected_configuration": selected,
    }


def _csv_hashes(csv_paths: Mapping[str, Path | str]) -> dict[str, str]:
    if set(csv_paths) != set(FAMILIES):
        raise ValueError("CSV paths must contain exactly hj and mearec")
    return {family: sha256_file(csv_paths[family]) for family in FAMILIES}


def build_selection_manifest(
    *,
    rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    csv_paths: Mapping[str, Path | str],
    base_revision: str = BASE_REPOSITORY_REVISION,
) -> dict[str, Any]:
    """Build the self-contained development artifact before it is authorized.

    This function does not authorize confirmation.  A separately reviewed,
    tracked output-manifest anchor must name the resulting JSON SHA-256.
    """
    if str(base_revision) != BASE_REPOSITORY_REVISION:
        raise ValueError("base revision must match the frozen temporal-sketch revision")
    rows = validate_development_rows(rows_by_family)
    gate = evaluate_development_gate(rows)
    snapshot_row = rows[FAMILIES[0]][0]
    return {
        "schema": SCHEMA,
        "selection_scope": "fixed_four_recording_temporal_sketch_development_gate",
        "base_repository_revision": BASE_REPOSITORY_REVISION,
        "repository_base_revision": snapshot_row["repository_base_revision"],
        "implementation_snapshot_scope": IMPLEMENTATION_SNAPSHOT_SCOPE,
        "implementation_source_sha256": dict(snapshot_row["implementation_source_sha256"]),
        "implementation_source_digest": snapshot_row["implementation_source_digest"],
        "duration_seconds": 60,
        "locked_input_subset_sha256": LOCKED_INPUT_SUBSET_SHA256,
        "pilot_dataset_ids": {family: list(PILOT_DATASET_IDS[family]) for family in FAMILIES},
        "fixed_grid": {
            "delay_samples": list(DELAYS),
            "feature_counts": list(FEATURE_COUNTS),
            "rows_per_record": ROWS_PER_RECORD,
            "total_rows": ROWS_PER_RECORD * 4,
            "candidate_source": "uniform_5bit_l1_p99p9_all_rows_only",
        },
        "gate_contract": _jsonable(GATE_CONTRACT),
        "locked_pilot_source_sha256": dict(LOCKED_PILOT_SOURCE_SHA256),
        "source_input_subset_digest": LOCKED_INPUT_SUBSET_SHA256,
        "temporal_input_contract": TEMPORAL_INPUT_CONTRACT,
        "temporal_input_contract_sha256": TEMPORAL_INPUT_CONTRACT_SHA256,
        "record_audit_rows": {family: rows[family] for family in FAMILIES},
        "csv_sha256": _csv_hashes(csv_paths),
        "gate": gate,
        "selected_configuration": gate["selected_configuration"],
    }


def build_authorization_anchor(
    selection_manifest_sha256: str,
    csv_sha256: Mapping[str, str],
    *,
    implementation_source_sha256: Mapping[str, Any],
    implementation_source_digest_value: str,
    repository_base_revision: str,
) -> dict[str, Any]:
    """Return the content a reviewed, tracked output manifest must contain."""
    if not _is_sha256(selection_manifest_sha256):
        raise ValueError("selection manifest SHA-256 must be a 64-character hex digest")
    if set(csv_sha256) != set(FAMILIES) or not all(_is_sha256(csv_sha256[key]) for key in FAMILIES):
        raise ValueError("authorization anchor needs both CSV SHA-256 values")
    if repository_base_revision != BASE_REPOSITORY_REVISION:
        raise ValueError("authorization anchor repository base revision mismatch")
    if implementation_source_digest(implementation_source_sha256) != implementation_source_digest_value:
        raise ValueError("authorization anchor implementation source digest mismatch")
    return {
        "schema": ANCHOR_SCHEMA,
        "locked_input_subset_sha256": LOCKED_INPUT_SUBSET_SHA256,
        "pilot_dataset_ids": {family: list(PILOT_DATASET_IDS[family]) for family in FAMILIES},
        "locked_pilot_source_sha256": dict(LOCKED_PILOT_SOURCE_SHA256),
        "source_input_subset_digest": LOCKED_INPUT_SUBSET_SHA256,
        "temporal_input_contract_sha256": TEMPORAL_INPUT_CONTRACT_SHA256,
        "gate_contract": _jsonable(GATE_CONTRACT),
        "repository_base_revision": repository_base_revision,
        "implementation_snapshot_scope": IMPLEMENTATION_SNAPSHOT_SCOPE,
        "implementation_source_sha256": dict(implementation_source_sha256),
        "implementation_source_digest": implementation_source_digest_value,
        "selection_manifest_sha256": str(selection_manifest_sha256).lower(),
        "csv_sha256": {family: str(csv_sha256[family]).lower() for family in FAMILIES},
    }


def _anchor_is_git_tracked(anchor_path: Path, repository_root: Path) -> bool:
    try:
        relative = anchor_path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return False
    tracked = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "--error-unmatch", "--", str(relative)],
        check=False, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        return False
    # A tracked path is not an authorization anchor if a caller has edited it
    # locally (staged or unstaged) after review.
    for command in (
        ["git", "-C", str(repository_root), "diff", "--quiet", "--", str(relative)],
        ["git", "-C", str(repository_root), "diff", "--cached", "--quiet", "--", str(relative)],
    ):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return False
    return True


def _validate_manifest_structure(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _reject_nonfinite(payload, context="selection manifest")
    if payload.get("schema") != SCHEMA:
        raise ValueError("selection manifest schema mismatch")
    if payload.get("selection_scope") != "fixed_four_recording_temporal_sketch_development_gate":
        raise ValueError("selection manifest scope mismatch")
    if payload.get("duration_seconds") != 60:
        raise ValueError("selection manifest duration mismatch")
    if payload.get("locked_input_subset_sha256") != LOCKED_INPUT_SUBSET_SHA256:
        raise ValueError("selection manifest locked input digest mismatch")
    if payload.get("source_input_subset_digest") != LOCKED_INPUT_SUBSET_SHA256:
        raise ValueError("selection manifest source input subset digest mismatch")
    expected_ids = {family: list(PILOT_DATASET_IDS[family]) for family in FAMILIES}
    if payload.get("pilot_dataset_ids") != expected_ids:
        raise ValueError("selection manifest pilot dataset IDs mismatch")
    if payload.get("locked_pilot_source_sha256") != LOCKED_PILOT_SOURCE_SHA256:
        raise ValueError("selection manifest locked pilot source SHA-256 mapping mismatch")
    if payload.get("temporal_input_contract") != TEMPORAL_INPUT_CONTRACT:
        raise ValueError("selection manifest temporal input contract mismatch")
    if payload.get("temporal_input_contract_sha256") != TEMPORAL_INPUT_CONTRACT_SHA256:
        raise ValueError("selection manifest temporal input contract SHA-256 mismatch")
    if payload.get("gate_contract") != GATE_CONTRACT:
        raise ValueError("selection manifest frozen gate contract mismatch")
    expected_grid = {
        "delay_samples": list(DELAYS), "feature_counts": list(FEATURE_COUNTS),
        "rows_per_record": ROWS_PER_RECORD, "total_rows": ROWS_PER_RECORD * 4,
        "candidate_source": "uniform_5bit_l1_p99p9_all_rows_only",
    }
    if payload.get("fixed_grid") != expected_grid:
        raise ValueError("selection manifest fixed grid mismatch")
    if payload.get("base_repository_revision") != BASE_REPOSITORY_REVISION:
        raise ValueError("selection manifest base revision mismatch")
    if payload.get("repository_base_revision") != BASE_REPOSITORY_REVISION:
        raise ValueError("selection manifest repository base revision mismatch")
    if payload.get("implementation_snapshot_scope") != IMPLEMENTATION_SNAPSHOT_SCOPE:
        raise ValueError("selection manifest implementation snapshot scope mismatch")
    manifest_source_map = payload.get("implementation_source_sha256")
    try:
        manifest_source_digest = implementation_source_digest(manifest_source_map)
    except ValueError as exc:
        raise ValueError("selection manifest implementation source map is invalid") from exc
    if payload.get("implementation_source_digest") != manifest_source_digest:
        raise ValueError("selection manifest implementation source digest mismatch")
    csv_hashes = payload.get("csv_sha256")
    if not isinstance(csv_hashes, Mapping) or set(csv_hashes) != set(FAMILIES) or not all(_is_sha256(csv_hashes[key]) for key in FAMILIES):
        raise ValueError("selection manifest CSV hash structure mismatch")
    source_rows = payload.get("record_audit_rows")
    if not isinstance(source_rows, Mapping):
        raise ValueError("selection manifest record audit rows are missing")
    rows = validate_development_rows(source_rows)
    row_snapshot = rows[FAMILIES[0]][0]
    for key in ("repository_base_revision", "implementation_source_sha256", "implementation_source_digest"):
        if payload.get(key) != row_snapshot[key]:
            raise ValueError("selection manifest implementation snapshot does not match its audit rows")
    return rows


def load_authorized_selection_manifest(
    manifest_path: Path | str,
    *,
    expected_sha256: str,
    authorization_anchor_path: Path | str,
    csv_paths: Mapping[str, Path | str],
    repository_root: Path | str | None = None,
    require_tracked_anchor: bool = True,
) -> dict[str, Any]:
    """Load a positive, reviewed selection artifact for a future confirmation.

    ``expected_sha256`` must agree with both the bytes of the selection JSON and
    the separately reviewed output-manifest anchor.  Supplying the hash of a
    modified JSON alone is deliberately insufficient.
    """
    if not _is_sha256(expected_sha256):
        raise ValueError("expected selection SHA-256 is invalid")
    manifest_file, anchor_file = Path(manifest_path), Path(authorization_anchor_path)
    actual = sha256_file(manifest_file)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError("selection manifest SHA-256 mismatch")
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    if require_tracked_anchor and not _anchor_is_git_tracked(anchor_file, root):
        raise ValueError("authorization anchor must be a tracked repository output manifest")
    try:
        anchor = json.loads(anchor_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("authorization anchor is unreadable") from exc
    if not isinstance(anchor, Mapping) or anchor.get("schema") != ANCHOR_SCHEMA:
        raise ValueError("authorization anchor schema mismatch")
    if anchor.get("locked_input_subset_sha256") != LOCKED_INPUT_SUBSET_SHA256:
        raise ValueError("authorization anchor locked input digest mismatch")
    if anchor.get("source_input_subset_digest") != LOCKED_INPUT_SUBSET_SHA256:
        raise ValueError("authorization anchor source input subset digest mismatch")
    if anchor.get("pilot_dataset_ids") != {family: list(PILOT_DATASET_IDS[family]) for family in FAMILIES}:
        raise ValueError("authorization anchor pilot IDs mismatch")
    if anchor.get("locked_pilot_source_sha256") != LOCKED_PILOT_SOURCE_SHA256:
        raise ValueError("authorization anchor locked pilot source SHA-256 mapping mismatch")
    if anchor.get("temporal_input_contract_sha256") != TEMPORAL_INPUT_CONTRACT_SHA256:
        raise ValueError("authorization anchor temporal input contract SHA-256 mismatch")
    if anchor.get("gate_contract") != GATE_CONTRACT:
        raise ValueError("authorization anchor frozen gate contract mismatch")
    if anchor.get("repository_base_revision") != BASE_REPOSITORY_REVISION:
        raise ValueError("authorization anchor repository base revision mismatch")
    if anchor.get("implementation_snapshot_scope") != IMPLEMENTATION_SNAPSHOT_SCOPE:
        raise ValueError("authorization anchor implementation snapshot scope mismatch")
    try:
        anchor_source_digest = implementation_source_digest(anchor.get("implementation_source_sha256"))
    except ValueError as exc:
        raise ValueError("authorization anchor implementation source map is invalid") from exc
    if anchor.get("implementation_source_digest") != anchor_source_digest:
        raise ValueError("authorization anchor implementation source digest mismatch")
    if str(anchor.get("selection_manifest_sha256", "")).lower() != actual.lower():
        raise ValueError("authorization anchor does not authorize this selection JSON")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("selection manifest is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("selection manifest root must be an object")
    rows = _validate_manifest_structure(payload)
    for key in ("repository_base_revision", "implementation_snapshot_scope", "implementation_source_sha256", "implementation_source_digest"):
        if anchor.get(key) != payload.get(key):
            raise ValueError("authorization anchor implementation snapshot does not match selection manifest")
    actual_csv_hashes = _csv_hashes(csv_paths)
    if payload["csv_sha256"] != actual_csv_hashes:
        raise ValueError("selection manifest CSV hashes do not match referenced CSV files")
    if anchor.get("csv_sha256") != actual_csv_hashes:
        raise ValueError("authorization anchor CSV hashes do not match referenced CSV files")
    recomputed = evaluate_development_gate(rows)
    if payload.get("gate") != recomputed:
        raise ValueError("selection manifest gate/audit is not the deterministic recomputation")
    if payload.get("selected_configuration") != recomputed["selected_configuration"]:
        raise ValueError("selection manifest selected configuration is inconsistent with tie-break")
    selected = recomputed["selected_configuration"]
    if not recomputed["gate_pass"] or selected is None:
        raise ValueError("selection manifest has no positive selected configuration; confirmation is prohibited")
    return dict(payload)


def dataset_id_digest(ids_by_family: Mapping[str, Iterable[str]]) -> str:
    """Canonical digest for a locked full-corpus identity manifest."""
    if set(ids_by_family) != set(FAMILIES):
        raise ValueError("corpus IDs must contain exactly hj and mearec")
    normalized = {family: sorted(map(str, ids_by_family[family])) for family in FAMILIES}
    return _sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n")


def validate_confirmation_corpus(
    observed_ids: Mapping[str, Iterable[str]],
    *,
    locked_full_corpus_ids: Mapping[str, Iterable[str]] | None = None,
    expected_full_corpus_sha256: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Require exactly the locked 32-recording corpus minus the four pilot IDs."""
    # The module, not a caller, owns the complete corpus identity.  Optional
    # inputs exist only to prove an external manifest agrees with this frozen
    # list; a caller cannot substitute a new corpus plus a matching new hash.
    if locked_full_corpus_ids is not None and {
        family: tuple(sorted(map(str, locked_full_corpus_ids[family]))) for family in FAMILIES
    } != {family: tuple(sorted(LOCKED_FULL_CORPUS_IDS[family])) for family in FAMILIES}:
        raise ValueError("provided full corpus IDs differ from the immutable locked corpus")
    if expected_full_corpus_sha256 is not None and str(expected_full_corpus_sha256).lower() != LOCKED_FULL_CORPUS_SHA256:
        raise ValueError("provided full corpus digest differs from the immutable locked digest")
    if dataset_id_digest(LOCKED_FULL_CORPUS_IDS) != LOCKED_FULL_CORPUS_SHA256:
        raise RuntimeError("immutable full corpus digest constant is inconsistent")
    if set(observed_ids) != set(FAMILIES):
        raise ValueError("confirmation IDs must contain exactly hj and mearec")
    expected: dict[str, tuple[str, ...]] = {}
    for family, full_count, confirmation_count in (("hj", 12, 10), ("mearec", 20, 18)):
        full = set(LOCKED_FULL_CORPUS_IDS[family])
        pilot = set(PILOT_DATASET_IDS[family])
        observed = {str(item) for item in observed_ids[family]}
        if len(full) != full_count or not pilot <= full:
            raise ValueError(f"locked {family} full corpus must contain its exact full count and pilot IDs")
        allowed = full - pilot
        if len(allowed) != confirmation_count or observed != allowed:
            raise ValueError(f"confirmation {family} IDs must be exactly locked corpus minus pilot IDs")
        if observed & pilot:
            raise ValueError(f"confirmation {family} includes a forbidden pilot ID")
        expected[family] = tuple(sorted(allowed))
    return expected
