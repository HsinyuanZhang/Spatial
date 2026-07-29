import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_adaptive_weighted_5bit import (
    DESCRIPTOR_DIM,
    candidate_category_work,
    frozen_weight_grid,
    load_verified_selection_manifest,
    pilot_selection_manifest,
    stable_chronological_three_way_split,
    stable_chronological_two_way_split,
    validate_confirmation_configuration,
    validate_hardware_contract,
    weighted_level1_accounting,
    weighted_pilot_gate,
    write_json_deterministic,
    wilson_interval,
)


def test_equal_timestamp_safe_fit_cal_test_split_preserves_complete_timestamp_groups():
    times = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    fit, calibration, test = stable_chronological_three_way_split(times)
    np.testing.assert_array_equal(fit, [0, 1, 2, 3])
    np.testing.assert_array_equal(calibration, [4, 5])
    np.testing.assert_array_equal(test, [6, 7])
    assert set(times[fit]).isdisjoint(set(times[calibration]))
    assert set(times[calibration]).isdisjoint(set(times[test]))
    early, late = stable_chronological_two_way_split(times)
    assert set(times[early]).isdisjoint(set(times[late]))


def test_frozen_grid_contains_exactly_five_weight_configurations():
    assert frozen_weight_grid() == [
        ("uniform", "uniform"),
        ("stability", "top3_x4"),
        ("stability", "tiered_421"),
        ("separation_stability", "top3_x4"),
        ("separation_stability", "tiered_421"),
    ]


def test_wilson_interval_boundaries_and_middle_value_are_well_formed():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5
    lo, hi = wilson_interval(10, 10)
    assert 0.5 < lo < 1.0 and hi == pytest.approx(1.0)
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_weighted_level1_accounting_charges_nine_lanes_and_unweighted_empty_fallback():
    report = weighted_level1_accounting(
        np.array([2, 1]), np.array([4, 1]), np.array([2, 1]), np.array([6, 2]), np.array([2, 0]), np.array([0, 1])
    )
    assert report["primary_abs_lanes_total"] == 3 * DESCRIPTOR_DIM
    assert report["primary_shift1_lanes_total"] == 5
    assert report["primary_shift2_lanes_total"] == 3
    assert report["empty_fallback_abs_lanes_total"] == 2 * DESCRIPTOR_DIM
    assert report["total_level1_row_comparisons"] == 5
    assert report["total_abs_lanes"] == 5 * DESCRIPTOR_DIM
    assert report["total_accumulator_additions"] == 5 * (DESCRIPTOR_DIM - 1)
    assert report["level1_row_accounting_identity_holds"]
    assert report["level1_lane_accounting_identity_holds"]
    assert report["total_addition_accounting_identity_holds"]
    assert report["primary_shift_accounting_identity_holds"]
    assert report["fallback_expected_row_identity_holds"]
    with pytest.raises(ValueError, match="shifted lanes"):
        weighted_level1_accounting(np.array([1]), np.array([1]), np.array([1]), np.array([1]), np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="fallback rows"):
        weighted_level1_accounting(np.array([1]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]))


def test_candidate_category_work_partitions_c0_c1_and_ambiguous_work():
    report = candidate_category_work(
        np.array([0, 1, 2]), np.array([2, 2, 2]), np.array([2, 3, 4]),
        np.array([1, 1, 2]), np.array([2, 0, 0]),
    )
    assert report["c0_events"] == report["c1_events"] == report["cgt1_events"] == 1
    assert report["c0_total_level1_rows"] == 4
    assert report["c0_total_abs_lanes"] == 4 * DESCRIPTOR_DIM
    assert report["c0_total_accumulator_additions"] == 4 * (DESCRIPTOR_DIM - 1)
    assert report["c1_shift1_lanes"] == 3 and report["cgt1_shift2_lanes"] == 2


def test_hardware_contract_and_confirmation_freeze_are_explicit():
    validate_hardware_contract(5, DESCRIPTOR_DIM)
    with pytest.raises(ValueError, match="B=5"):
        validate_hardware_contract(4, DESCRIPTOR_DIM)
    with pytest.raises(ValueError, match="D=9"):
        validate_hardware_contract(5, 8)
    assert validate_confirmation_configuration("stability", "tiered_421", 99.0) == (
        "stability", "tiered_421", 99.0
    )
    with pytest.raises(ValueError, match="requires"):
        validate_confirmation_configuration(None, "tiered_421", 99.0)
    with pytest.raises(ValueError, match="grid"):
        validate_confirmation_configuration("uniform", "top3_x4", 99.0)
    with pytest.raises(ValueError, match="nonuniform"):
        validate_confirmation_configuration("uniform", "uniform", 99.0)


def _gate_row(dataset, mode, profile, recall, traffic):
    return {
        "dataset": dataset, "score_mode": mode, "weight_profile": profile,
        "radius_percentile": 99.0, "n_test_gt_rows": 1000,
        "true_unit_candidate_hits": int(round(1000 * recall)),
        "true_unit_candidate_recall": recall,
        "global_all_unit_waveform_template_bits": 1_000_000,
        "actual_waveform_template_bits_read": int(1_000_000 / traffic),
        "waveform_bit_traffic_reduction_factor": traffic,
    }


def test_weighted_pilot_gate_requires_same_nonuniform_config_and_branch_in_both_families():
    rows = {}
    for family in ("hj", "mearec"):
        rows[family] = []
        for record in (f"{family}_one", f"{family}_two"):
            rows[family].append(_gate_row(record, "uniform", "uniform", 0.900, 4.0))
            rows[family].append(_gate_row(record, "separation_stability", "tiered_421", 0.906, 4.0))
    report = weighted_pilot_gate(rows)
    assert report["pilot_gate_pass"]
    assert report["pilot_gate_selected"]["config"][:2] == ("separation_stability", "tiered_421")
    assert report["pilot_gate_selected"]["branch"] == "A_recall"


def test_weighted_pilot_gate_rejects_aggregate_gain_that_breaks_per_record_tolerance():
    rows = {}
    for family in ("hj", "mearec"):
        rows[family] = [
            _gate_row(f"{family}_one", "uniform", "uniform", 0.900, 4.0),
            _gate_row(f"{family}_one", "stability", "top3_x4", 0.894, 4.0),
            _gate_row(f"{family}_two", "uniform", "uniform", 0.900, 4.0),
            _gate_row(f"{family}_two", "stability", "top3_x4", 0.920, 4.0),
        ]
    assert not weighted_pilot_gate(rows)["pilot_gate_pass"]


def test_weighted_pilot_gate_reports_zero_traffic_and_stability_failure_reason():
    rows = {}
    for family in ("hj", "mearec"):
        base = _gate_row(f"{family}_base", "uniform", "uniform", .9, 4.0)
        candidate = _gate_row(f"{family}_base", "stability", "top3_x4", .91, 4.0)
        base["actual_waveform_template_bits_read"] = 0; candidate["actual_waveform_template_bits_read"] = 0
        candidate["early_late_shift_evaluable"] = False; candidate["early_late_shift_code_agreement"] = float("nan")
        rows[family] = [base, candidate]
    report = weighted_pilot_gate(rows)
    assert not report["pilot_gate_pass"]
    audit = report["nonuniform_configuration_audit"][0]["per_record"]
    assert any("early_late_shift_not_evaluable_or_below_0p75" in row["reasons"] for row in audit)


def test_failed_pilot_manifest_is_deterministic_and_cannot_authorize_confirmation(tmp_path):
    csv_paths = {}
    for family in ("hj", "mearec"):
        path = tmp_path / f"{family}.csv"
        path.write_text("x\n", encoding="utf-8")
        csv_paths[family] = path
    dataset_ids = {"hj": ["hybrid_janelia_drift16c_600s_11", "hybrid_janelia_static16c_600s_11"], "mearec": ["mearec_rec_v1_units10_snr5_seed202601", "mearec_rec_v1_units20_snr5_seed202601"]}
    rows = {}
    for family, ids in dataset_ids.items():
        rows[family] = []
        for dataset in ids:
            row = _gate_row(dataset, "uniform", "uniform", .9, 4.0)
            row.update({"fit_stop_index":10,"calibration_stop_index":15,"fit_last_timestamp":9,"calibration_first_timestamp":10,"calibration_last_timestamp":14,"test_first_timestamp":15,"n_events_gt_rows":20,"n_fit_gt_rows":10,"n_calibration_gt_rows":5,"n_test_gt_rows":5,"fit_unit_count":1,"calibration_unit_count":1,"test_unit_count":1,"waveform_template_supported_units":1})
            rows[family].append(row)
    gate = weighted_pilot_gate(rows)
    payload = pilot_selection_manifest(
        gate=gate, final_report={"final_0p99_4x_pass": False}, csv_paths=csv_paths, rows_by_family=rows
    )
    path = tmp_path / "selection.json"
    write_json_deterministic(path, payload)
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no selected"):
        load_verified_selection_manifest(path, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_selection_manifest(path, "0" * 64)


def test_confirmation_manifest_rejects_rehashed_provenance_grid_and_uniform_tampering(tmp_path):
    import copy
    import hashlib
    csv_paths = {family: tmp_path / f"{family}.csv" for family in ("hj", "mearec")}
    for path in csv_paths.values(): path.write_text("x\n", encoding="utf-8")
    ids = {"hj": ["hybrid_janelia_drift16c_600s_11", "hybrid_janelia_static16c_600s_11"], "mearec": ["mearec_rec_v1_units10_snr5_seed202601", "mearec_rec_v1_units20_snr5_seed202601"]}
    rows = {}
    for family, names in ids.items():
        rows[family] = []
        for name in names:
            row = _gate_row(name, "uniform", "uniform", .9, 4.0)
            row.update({"fit_stop_index":1,"calibration_stop_index":2,"fit_last_timestamp":0,"calibration_first_timestamp":1,"calibration_last_timestamp":1,"test_first_timestamp":2,"n_events_gt_rows":4,"n_fit_gt_rows":1,"n_calibration_gt_rows":1,"n_test_gt_rows":2,"fit_unit_count":1,"calibration_unit_count":1,"test_unit_count":1,"waveform_template_supported_units":1})
            rows[family].append(row)
    payload = pilot_selection_manifest(gate=weighted_pilot_gate(rows), final_report={}, csv_paths=csv_paths, rows_by_family=rows)
    payload["selected_configuration"] = {"config": ["stability", "top3_x4", 99.0]}
    manual = copy.deepcopy(payload)
    path = tmp_path / "manual_selection.json"
    write_json_deterministic(path, manual)
    with pytest.raises(ValueError, match="gate did not pass"):
        load_verified_selection_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
    for name, mutate, message in (
        ("ids", lambda x: x.__setitem__("pilot_dataset_ids", {}), "dataset IDs"),
        ("digest", lambda x: x.__setitem__("locked_input_subset_sha256", "0" * 64), "input digest"),
        ("grid", lambda x: x["fixed_grid"].__setitem__("directory", "support"), "fixed grid"),
        ("csv_hash", lambda x: x["csv_sha256"].__setitem__("hj", "z" * 64), "CSV hash"),
        ("uniform", lambda x: x.__setitem__("selected_configuration", {"config": ["uniform", "uniform", 99.0]}), "nonuniform"),
    ):
        altered = copy.deepcopy(payload); mutate(altered); path = tmp_path / f"{name}.json"; write_json_deterministic(path, altered)
        with pytest.raises(ValueError, match=message): load_verified_selection_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_confirmation_manifest_requires_gate_consistency_and_complete_audit_grid(tmp_path):
    import copy
    import hashlib

    csv_paths = {family: tmp_path / f"{family}.csv" for family in ("hj", "mearec")}
    for path in csv_paths.values():
        path.write_text("x\n", encoding="utf-8")
    ids = {
        "hj": ["hybrid_janelia_drift16c_600s_11", "hybrid_janelia_static16c_600s_11"],
        "mearec": ["mearec_rec_v1_units10_snr5_seed202601", "mearec_rec_v1_units20_snr5_seed202601"],
    }
    rows = {}
    for family, names in ids.items():
        rows[family] = []
        for name in names:
            row = _gate_row(name, "uniform", "uniform", .9, 4.0)
            row.update({"fit_stop_index":1,"calibration_stop_index":2,"fit_last_timestamp":0,"calibration_first_timestamp":1,"calibration_last_timestamp":1,"test_first_timestamp":2,"n_events_gt_rows":4,"n_fit_gt_rows":1,"n_calibration_gt_rows":1,"n_test_gt_rows":2,"fit_unit_count":1,"calibration_unit_count":1,"test_unit_count":1,"waveform_template_supported_units":1})
            rows[family].append(row)
    payload = pilot_selection_manifest(
        gate=weighted_pilot_gate(rows), final_report={}, csv_paths=csv_paths, rows_by_family=rows
    )
    selected = {
        "config": ["stability", "top3_x4", 99.0],
        "branch": "A_recall",
        "family_recall_delta": {"hj": 0.01, "mearec": 0.01},
        "family_traffic_ratio": {"hj": 1.0, "mearec": 1.0},
    }
    payload["selected_configuration"] = selected
    payload["gate"].update({
        "pilot_gate_pass": True,
        "pilot_gate_candidates": 1,
        "pilot_gate_selected": copy.deepcopy(selected),
    })
    payload["gate"]["nonuniform_configuration_audit"] = [
        {
            "config": [mode, profile, percentile],
            "gate_pass": [mode, profile, percentile] == selected["config"],
        }
        for mode, profile in frozen_weight_grid()
        if mode != "uniform"
        for percentile in (95.0, 99.0, 99.9)
    ]
    path = tmp_path / "valid_selection.json"
    write_json_deterministic(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_verified_selection_manifest(path, digest)["selected_configuration"] == selected

    mismatch = copy.deepcopy(payload)
    mismatch["gate"]["pilot_gate_selected"]["config"] = ["stability", "tiered_421", 99.0]
    path = tmp_path / "mismatched_selection.json"
    write_json_deterministic(path, mismatch)
    with pytest.raises(ValueError, match="does not match"):
        load_verified_selection_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())

    incomplete = copy.deepcopy(payload)
    incomplete["gate"]["nonuniform_configuration_audit"].pop()
    path = tmp_path / "incomplete_audit.json"
    write_json_deterministic(path, incomplete)
    with pytest.raises(ValueError, match="audit grid"):
        load_verified_selection_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
