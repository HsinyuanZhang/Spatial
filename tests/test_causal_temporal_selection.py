"""Synthetic no-I/O tests for temporal-sketch gate and authorization logic."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.causal_temporal_selection import (
    BASE_REPOSITORY_REVISION,
    FEATURE_COUNTS,
    GATE_CONTRACT,
    IMPLEMENTATION_SOURCE_PATHS,
    IMPLEMENTATION_SNAPSHOT_SCOPE,
    LOCKED_INPUT_SUBSET_SHA256,
    LOCKED_FULL_CORPUS_IDS,
    LOCKED_FULL_CORPUS_SHA256,
    LOCKED_PILOT_SOURCE_SHA256,
    PILOT_DATASET_IDS,
    TEMPORAL_INPUT_CONTRACT_SHA256,
    TEMPORAL_INPUT_CONTRACT,
    build_authorization_anchor,
    build_selection_manifest,
    dataset_id_digest,
    evaluate_development_gate,
    implementation_source_digest,
    load_authorized_selection_manifest,
    sha256_file,
    validate_confirmation_corpus,
    validate_development_rows,
    write_json_deterministic,
)


def _digest(char: str) -> str:
    return char * 64


def _implementation_source_sha256() -> dict[str, str]:
    return {path: _digest(format(index % 16, "x")) for index, path in enumerate(IMPLEMENTATION_SOURCE_PATHS)}


def _anchor_for(manifest: dict, manifest_path: Path) -> dict:
    return build_authorization_anchor(
        sha256_file(manifest_path),
        manifest["csv_sha256"],
        implementation_source_sha256=manifest["implementation_source_sha256"],
        implementation_source_digest_value=manifest["implementation_source_digest"],
        repository_base_revision=manifest["repository_base_revision"],
    )


def _row(dataset: str, delay: int, feature_count: int, *, family: str, actual_bits: int | None = None) -> dict:
    """One passing, independently auditable synthetic grid row."""
    n_events, ambiguous, reads = 1_000, 300, 600
    actual = reads * 5 * feature_count if actual_bits is None else actual_bits
    c0, c1 = 100, 600
    prefix_length = 16 + delay
    source_map = _implementation_source_sha256()
    extraction = {
        "logical_raw_prefix_samples_consumed": n_events * prefix_length,
        "logical_sample_quantizations": n_events * prefix_length,
        "logical_selected_feature_accumulator_updates": n_events * feature_count * 2,
        "logical_positive_coefficient_updates": n_events * feature_count * 2,
        "logical_negative_coefficient_updates": 0,
        "logical_normalization_shifts": n_events * feature_count,
        "logical_emitted_feature_codes": n_events * feature_count,
    }
    category_counts = {"c0": c0, "c1": c1, "cgt1": ambiguous}
    category = {
        name: {key: category_counts[name] * (value // n_events) for key, value in extraction.items()}
        for name in category_counts
    }
    for name in ("c0", "c1", "cgt1"):
        category[name].update({key: 0 for key in ("template_reads", "template_bits", "sad_absolute_differences", "sad_reduction_additions")})
    category["cgt1"].update({
        "template_reads": reads,
        "template_bits": actual,
        "sad_absolute_differences": reads * feature_count,
        "sad_reduction_additions": reads * (feature_count - 1),
    })
    def saturation(samples_per_row: int) -> dict:
        return {
            "fit_sample_saturation_fraction": 0.0,
            "fit_sample_saturation_population": "fisher_eligible_fit_rows",
            "fit_quantizer_eligible_row_count": 2_000,
            "fit_quantizer_source_sample_count": 2_000 * samples_per_row,
            "fit_all_sample_saturation_fraction": 0.0,
            "fit_all_quantizer_row_count": 2_000,
            "fit_all_quantizer_source_sample_count": 2_000 * samples_per_row,
        }
    return {
        "dataset": dataset,
        "delay_samples": delay,
        "feature_count": feature_count,
        "n_test_events": n_events,
        "candidate_recall_hits": 995,
        "c0_events": c0,
        "c1_events": c1,
        "ambiguous_events": ambiguous,
        "candidate_template_reads": reads,
        "distinct_ambiguous_candidate_pairs": 5,
        "overall_correct": 940,
        "full_teacher_overall_correct": 950,
        "ambiguous_correct": 267,
        "full_teacher_ambiguous_correct": 270,
        "global_64x5_template_bits": n_events * 10 * 64 * 5,
        "actual_template_bits": actual,
        "raw_input_provenance_ok": True,
        "row_identity_ok": True,
        "future_sample_isolation_ok": True,
        "bit_width_invariants_ok": True,
        "accounting_invariants_ok": True,
        "fit_event_digest": _digest("a"),
        "calibration_event_digest": _digest("b"),
        "test_event_digest": _digest("c"),
        "candidate_source_digest": _digest("d"),
        "raw_input_provenance_digest": _digest("e"),
        "teacher_contract_hash": _digest("f"),
        "source_file_sha256": LOCKED_PILOT_SOURCE_SHA256[dataset],
        "source_input_path": f"locked-inputs/{dataset}.npz",
        "source_input_subset_digest": LOCKED_INPUT_SUBSET_SHA256,
        "temporal_input_contract_sha256": TEMPORAL_INPUT_CONTRACT_SHA256,
        "repository_base_revision": BASE_REPOSITORY_REVISION,
        "implementation_source_sha256": source_map,
        "implementation_source_digest": implementation_source_digest(source_map),
        "n_fit_events": 2_000,
        "n_calibration_events": 1_000,
        "fit_last_timestamp": 49,
        "calibration_first_timestamp": 50,
        "calibration_last_timestamp": 74,
        "test_first_timestamp": 75,
        "scale_exponent": 4 + delay // 8,
        "level1_source_scan_events": n_events,
        "level1_source_unit_count": 10,
        "level1_source_descriptor_quantizations": n_events * 9,
        "level1_source_centroid_reads": n_events * 10,
        "level1_source_l1_absolute_differences": n_events * 10 * 9,
        "level1_source_l1_reduction_additions": n_events * 10 * 8,
        "level1_source_radius_threshold_comparisons": n_events * 10,
        "level1_source_argmin_reduction_comparisons": n_events * 9,
        "level1_c0_reuse_events": c0,
        "logical_candidate_template_reads": reads,
        "logical_candidate_template_bits": actual,
        "logical_sad_absolute_differences": reads * feature_count,
        "logical_sad_reduction_additions": reads * (feature_count - 1),
        "logical_extraction_counters": extraction,
        "logical_category_decomposition": category,
        "scale_audit": {
            "sketch_saturation": saturation(prefix_length),
            "baseline_saturation": saturation(64),
        },
        "selected_filter_ids": list(range(feature_count)),
        "fisher_audit": {
            "eligible_filter_ids": list(range(20)),
            "between": [float(index + 1) for index in range(20)],
            "within": [float(index + 2) for index in range(20)],
            "scores": [float(index + 1) / float(index + 2) for index in range(20)],
            "ranks": list(range(20)),
            "fit_unit_counts": {"unit_a": 12, "unit_b": 15},
            "eligible_unit_mask": {"unit_a": True, "unit_b": True},
            "eligible_unit_ids": ["unit_a", "unit_b"],
        },
    }


def _rows() -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for family, datasets in PILOT_DATASET_IDS.items():
        output[family] = [
            _row(dataset, delay, count, family=family)
            for dataset in datasets
            for delay in (0, 8, 16, 32, 48)
            for count in FEATURE_COUNTS
        ]
    return output


def _csvs(tmp_path: Path) -> dict[str, Path]:
    paths = {family: tmp_path / f"{family}.csv" for family in PILOT_DATASET_IDS}
    for family, path in paths.items():
        path.write_text(f"synthetic,{family}\n", encoding="utf-8")
    return paths


def _manifest_and_anchor(tmp_path: Path) -> tuple[dict, Path, Path, dict[str, Path]]:
    csv_paths = _csvs(tmp_path)
    manifest = build_selection_manifest(rows_by_family=_rows(), csv_paths=csv_paths, base_revision=BASE_REPOSITORY_REVISION)
    manifest_path = tmp_path / "selection.json"
    write_json_deterministic(manifest_path, manifest)
    anchor = _anchor_for(manifest, manifest_path)
    anchor_path = tmp_path / "reviewed-output-manifest.json"
    write_json_deterministic(anchor_path, anchor)
    return manifest, manifest_path, anchor_path, csv_paths


def test_gate_requires_complete_grid_and_selects_lowest_delay_then_m_deterministically():
    rows = _rows()
    forward = evaluate_development_gate(rows)
    shuffled = {family: list(reversed(values)) for family, values in rows.items()}
    reverse = evaluate_development_gate(shuffled)
    assert forward == reverse
    assert forward["gate_pass"]
    assert forward["candidate_count"] == 15
    assert forward["selected_configuration"] == {
        "delay_samples": 0, "feature_count": 8, "name": "delay_00_m_08"
    }
    missing = _rows()
    missing["hj"].pop()
    with pytest.raises(ValueError, match="15 frozen grid rows"):
        validate_development_rows(missing)


def test_zero_traffic_or_no_temporal_invocation_cannot_pass():
    rows = _rows()
    # For one configuration all four recordings look spatial-only: neither
    # per-record reads nor family actual bits may pass the temporal gate.
    for family in rows:
        for row in rows[family]:
            if row["delay_samples"] == 0 and row["feature_count"] == 8:
                row["actual_template_bits"] = 0
                row["logical_candidate_template_bits"] = 0
                row["logical_category_decomposition"]["cgt1"]["template_bits"] = 0
    gate = evaluate_development_gate(rows)
    audited = next(item for item in gate["configuration_audit"] if item["config"]["name"] == "delay_00_m_08")
    assert not audited["pass"]
    assert not audited["family"]["hj"]["predicates"]["actual_bits_positive"]
    assert gate["selected_configuration"] == {
        "delay_samples": 0, "feature_count": 12, "name": "delay_00_m_12"
    }


def test_every_per_record_constraint_and_boolean_invariant_is_enforced():
    rows = _rows()
    target = rows["hj"][0]
    for row in rows["hj"]:
        if row["dataset"] == target["dataset"]:
            row["ambiguous_events"] = 99
            row["c1_events"] = 801
            row["ambiguous_correct"] = 90
            row["full_teacher_ambiguous_correct"] = 90
            for key, total in row["logical_extraction_counters"].items():
                per_event = total // row["n_test_events"]
                row["logical_category_decomposition"]["c1"][key] = 801 * per_event
                row["logical_category_decomposition"]["cgt1"][key] = 99 * per_event
    target["future_sample_isolation_ok"] = False
    gate = evaluate_development_gate(rows)
    first = next(item for item in gate["configuration_audit"] if item["config"]["name"] == "delay_00_m_08")
    record = next(item for item in first["records"]["hj"] if item["dataset"] == target["dataset"])
    assert not record["pass"]
    assert not record["predicates"]["ambiguous_events_ge_100"]
    assert not record["predicates"]["all_invariants_true"]


def test_gate_rejects_nonfrozen_source_across_grid_rows():
    rows = _rows()
    rows["mearec"][1]["candidate_source_digest"] = _digest("9")
    with pytest.raises(ValueError, match="non-frozen candidate/source"):
        evaluate_development_gate(rows)


def test_each_row_requires_the_exact_locked_source_sha_and_subset_digest():
    rows = _rows()
    rows["hj"][0]["source_file_sha256"] = _digest("0")
    with pytest.raises(ValueError, match="source_file_sha256"):
        validate_development_rows(rows)

    rows = _rows()
    rows["mearec"][0]["source_input_subset_digest"] = _digest("1")
    with pytest.raises(ValueError, match="source_input_subset_digest"):
        validate_development_rows(rows)


def test_source_path_and_temporal_contract_are_frozen_across_each_dataset_grid():
    rows = _rows()
    rows["hj"][1]["source_input_path"] = "locked-inputs/drift-replaced.npz"
    with pytest.raises(ValueError, match="non-frozen candidate/source"):
        validate_development_rows(rows)

    rows = _rows()
    rows["mearec"][1]["temporal_input_contract_sha256"] = _digest("2")
    with pytest.raises(ValueError, match="temporal input contract hash"):
        validate_development_rows(rows)


def test_manifest_loader_recomputes_gate_checks_csv_and_uses_anchor(tmp_path: Path):
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    loaded = load_authorized_selection_manifest(
        manifest_path,
        expected_sha256=sha256_file(manifest_path),
        authorization_anchor_path=anchor_path,
        csv_paths=csv_paths,
        require_tracked_anchor=False,
    )
    assert loaded["selected_configuration"] == manifest["selected_configuration"]
    assert loaded["locked_pilot_source_sha256"] == LOCKED_PILOT_SOURCE_SHA256
    assert loaded["source_input_subset_digest"] == LOCKED_INPUT_SUBSET_SHA256
    assert loaded["temporal_input_contract_sha256"] == TEMPORAL_INPUT_CONTRACT_SHA256
    assert loaded["gate_contract"] == GATE_CONTRACT
    assert loaded["implementation_snapshot_scope"] == IMPLEMENTATION_SNAPSHOT_SCOPE
    assert loaded["repository_base_revision"] == BASE_REPOSITORY_REVISION
    assert loaded["implementation_source_sha256"] == _implementation_source_sha256()
    assert loaded["implementation_source_digest"] == implementation_source_digest(_implementation_source_sha256())

    # Rehashing a modified JSON does not authorize it: the reviewed anchor
    # still names the original artifact hash.
    tampered = copy.deepcopy(manifest)
    tampered["selected_configuration"] = {"delay_samples": 48, "feature_count": 16, "name": "delay_48_m_16"}
    write_json_deterministic(manifest_path, tampered)
    with pytest.raises(ValueError, match="anchor does not authorize"):
        load_authorized_selection_manifest(
            manifest_path,
            expected_sha256=sha256_file(manifest_path),
            authorization_anchor_path=anchor_path,
            csv_paths=csv_paths,
            require_tracked_anchor=False,
        )


def test_loader_rejects_manual_anchor_rehash_when_gate_or_csv_audit_is_tampered(tmp_path: Path):
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["gate"]["selected_configuration"] = None
    tampered["selected_configuration"] = None
    write_json_deterministic(manifest_path, tampered)
    # Even an attacker who rewrites the untracked test anchor cannot make the
    # internally inconsistent gate artifact authorize confirmation.
    write_json_deterministic(anchor_path, _anchor_for(tampered, manifest_path))
    with pytest.raises(ValueError, match="deterministic recomputation"):
        load_authorized_selection_manifest(
            manifest_path,
            expected_sha256=sha256_file(manifest_path),
            authorization_anchor_path=anchor_path,
            csv_paths=csv_paths,
            require_tracked_anchor=False,
        )
    # A CSV modification is independently detected against both artifact and anchor.
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    csv_paths["hj"].write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV hashes"):
        load_authorized_selection_manifest(
            manifest_path,
            expected_sha256=sha256_file(manifest_path),
            authorization_anchor_path=anchor_path,
            csv_paths=csv_paths,
            require_tracked_anchor=False,
        )


def test_loader_default_requires_a_tracked_output_manifest_anchor(tmp_path: Path):
    _, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    # Supplying both a freshly rehashed JSON and a freshly written *untracked*
    # anchor does not satisfy confirmation authorization.
    with pytest.raises(ValueError, match="tracked repository"):
        load_authorized_selection_manifest(
            manifest_path,
            expected_sha256=sha256_file(manifest_path),
            authorization_anchor_path=anchor_path,
            csv_paths=csv_paths,
        )


def test_loader_rejects_frozen_base_revision_and_nonfinite_json_tampering(tmp_path: Path):
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["base_repository_revision"] = "attacker_revision"
    write_json_deterministic(manifest_path, manifest)
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="base revision mismatch"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )


def test_manifest_and_anchor_fail_closed_on_provenance_or_gate_contract_drift(tmp_path: Path):
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["locked_pilot_source_sha256"] = dict(manifest["locked_pilot_source_sha256"])
    manifest["locked_pilot_source_sha256"][PILOT_DATASET_IDS["hj"][0]] = _digest("3")
    write_json_deterministic(manifest_path, manifest)
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="source SHA-256 mapping mismatch"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["source_input_subset_digest"] = _digest("5")
    write_json_deterministic(manifest_path, manifest)
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="source input subset digest mismatch"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["gate_contract"] = dict(manifest["gate_contract"])
    manifest["gate_contract"]["per_record"] = "attacker-selected gate"
    write_json_deterministic(manifest_path, manifest)
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="frozen gate contract mismatch"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["temporal_input_contract_sha256"] = _digest("4")
    write_json_deterministic(anchor_path, anchor)
    with pytest.raises(ValueError, match="anchor temporal input contract SHA-256 mismatch"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    # Python's JSON decoder accepts NaN by default, so validation must reject
    # it even if an external producer emitted non-standard JSON.
    raw = manifest_path.read_text(encoding="utf-8").replace('"overall_correct":940', '"overall_correct":NaN', 1)
    manifest_path.write_text(raw, encoding="utf-8")
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="non-finite"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )


def test_manifest_requires_all_audit_fields_and_rejects_null_selection(tmp_path: Path):
    rows = _rows()
    del rows["hj"][0]["teacher_contract_hash"]
    with pytest.raises(ValueError, match="required audit fields"):
        build_selection_manifest(rows_by_family=rows, csv_paths=_csvs(tmp_path), base_revision=BASE_REPOSITORY_REVISION)

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["selected_configuration"] = None
    write_json_deterministic(manifest_path, manifest)
    write_json_deterministic(anchor_path, _anchor_for(manifest, manifest_path))
    with pytest.raises(ValueError, match="selected configuration is inconsistent|no positive"):
        load_authorized_selection_manifest(
            manifest_path,
            expected_sha256=sha256_file(manifest_path),
            authorization_anchor_path=anchor_path,
            csv_paths=csv_paths,
            require_tracked_anchor=False,
        )


def test_malformed_fisher_and_split_provenance_are_rejected_before_gate():
    rows = _rows()
    rows["hj"][0]["fisher_audit"]["ranks"] = [0] * 20
    with pytest.raises(ValueError, match="Fisher ranks"):
        evaluate_development_gate(rows)
    rows = _rows()
    rows["mearec"][0]["calibration_first_timestamp"] = 49
    with pytest.raises(ValueError, match="fit/calibration split timestamps overlap"):
        evaluate_development_gate(rows)


@pytest.mark.parametrize(
    "field",
    (
        "level1_source_scan_events",
        "level1_source_unit_count",
        "level1_source_descriptor_quantizations",
        "level1_source_centroid_reads",
        "level1_source_l1_absolute_differences",
        "level1_source_l1_reduction_additions",
        "level1_source_radius_threshold_comparisons",
        "level1_source_argmin_reduction_comparisons",
        "level1_c0_reuse_events",
    ),
)
def test_every_level1_source_counter_is_independently_recomputed(field: str):
    rows = _rows()
    rows["hj"][0][field] += 1
    with pytest.raises(ValueError, match="Level-1 source accounting"):
        validate_development_rows(rows)


def test_implementation_snapshot_map_digest_and_base_fail_closed():
    rows = _rows()
    del rows["hj"][0]["implementation_source_sha256"][IMPLEMENTATION_SOURCE_PATHS[0]]
    with pytest.raises(ValueError, match="unexpected key set"):
        validate_development_rows(rows)

    rows = _rows()
    rows["hj"][0]["implementation_source_sha256"][IMPLEMENTATION_SOURCE_PATHS[0]] = "not-a-sha"
    with pytest.raises(ValueError, match="only 64-character digests"):
        validate_development_rows(rows)

    rows = _rows()
    rows["hj"][0]["implementation_source_digest"] = _digest("0")
    with pytest.raises(ValueError, match="implementation source digest"):
        validate_development_rows(rows)

    rows = _rows()
    rows["hj"][0]["repository_base_revision"] = "attacker_revision"
    with pytest.raises(ValueError, match="repository_base_revision"):
        validate_development_rows(rows)


def test_saturation_and_nested_category_accounting_fail_closed():
    rows = _rows()
    rows["hj"][0]["scale_audit"]["sketch_saturation"]["fit_sample_saturation_population"] = "all_fit_rows"
    with pytest.raises(ValueError, match="eligible saturation population"):
        validate_development_rows(rows)

    rows = _rows()
    rows["hj"][0]["scale_audit"]["baseline_saturation"]["fit_quantizer_source_sample_count"] += 1
    with pytest.raises(ValueError, match="eligible sample count"):
        validate_development_rows(rows)

    rows = _rows()
    rows["hj"][0]["logical_category_decomposition"]["cgt1"]["template_reads"] += 1
    with pytest.raises(ValueError, match="temporal accounting"):
        validate_development_rows(rows)


def test_manifest_and_anchor_fail_closed_on_implementation_snapshot_drift(tmp_path: Path):
    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    manifest["implementation_source_sha256"] = dict(manifest["implementation_source_sha256"])
    manifest["implementation_source_sha256"].pop(IMPLEMENTATION_SOURCE_PATHS[0])
    write_json_deterministic(manifest_path, manifest)
    # The anchor may be freshly rehashed in this test, but the malformed
    # manifest snapshot itself must still be rejected.
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["selection_manifest_sha256"] = sha256_file(manifest_path)
    write_json_deterministic(anchor_path, anchor)
    with pytest.raises(ValueError, match="implementation source map"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )

    manifest, manifest_path, anchor_path, csv_paths = _manifest_and_anchor(tmp_path)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["implementation_source_digest"] = _digest("0")
    write_json_deterministic(anchor_path, anchor)
    with pytest.raises(ValueError, match="anchor implementation source digest"):
        load_authorized_selection_manifest(
            manifest_path, expected_sha256=sha256_file(manifest_path), authorization_anchor_path=anchor_path,
            csv_paths=csv_paths, require_tracked_anchor=False,
        )


def test_confirmation_corpus_is_exact_full_locked_corpus_minus_pilot_ids():
    full = {family: list(values) for family, values in LOCKED_FULL_CORPUS_IDS.items()}
    digest = dataset_id_digest(full)
    assert digest == LOCKED_FULL_CORPUS_SHA256
    observed = {
        family: [item for item in values if item not in PILOT_DATASET_IDS[family]]
        for family, values in full.items()
    }
    allowed = validate_confirmation_corpus(
        observed, locked_full_corpus_ids=full, expected_full_corpus_sha256=digest
    )
    assert len(allowed["hj"]) == 10 and len(allowed["mearec"]) == 18

    observed["hj"][0] = PILOT_DATASET_IDS["hj"][0]
    with pytest.raises(ValueError, match="exactly locked corpus minus pilot IDs"):
        validate_confirmation_corpus(observed, locked_full_corpus_ids=full, expected_full_corpus_sha256=digest)


def test_caller_modified_full_corpus_plus_new_digest_is_rejected():
    altered = {family: list(values) for family, values in LOCKED_FULL_CORPUS_IDS.items()}
    altered["hj"][-1] = "hj_attacker_replacement"
    observed = {
        family: [item for item in values if item not in PILOT_DATASET_IDS[family]]
        for family, values in altered.items()
    }
    with pytest.raises(ValueError, match="immutable locked corpus"):
        validate_confirmation_corpus(
            observed, locked_full_corpus_ids=altered, expected_full_corpus_sha256=dataset_id_digest(altered)
        )


def test_locked_digest_constant_is_present_and_not_callers_choice():
    assert LOCKED_INPUT_SUBSET_SHA256 == "9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13"


def test_runtime_source_snapshot_is_the_complete_sorted_runtime_python_superset():
    root = Path(__file__).resolve().parents[1]
    observed = tuple(
        sorted(
            str(path.relative_to(root))
            for directory in ("algorithms", "data", "evaluation", "experiments")
            for path in (root / directory).rglob("*.py")
        )
    )
    assert observed == IMPLEMENTATION_SOURCE_PATHS
    assert len(IMPLEMENTATION_SOURCE_PATHS) == 55


def test_temporal_input_contract_hash_binds_sampling_raw_input_and_legacy_route():
    assert "sampling_frequency=30000Hz" in TEMPORAL_INPUT_CONTRACT
    assert "recording_input=dataset.raw_data_channel_x_sample" in TEMPORAL_INPUT_CONTRACT
    assert "legacy=report_only_peak_normalized_epsilon_1e-8" in TEMPORAL_INPUT_CONTRACT
    assert hashlib.sha256(TEMPORAL_INPUT_CONTRACT.encode("utf-8")).hexdigest() == TEMPORAL_INPUT_CONTRACT_SHA256
