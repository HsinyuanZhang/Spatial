import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_adaptive_masked_5bit import (
    DESCRIPTOR_DIM,
    candidate_category_level1_accounting,
    conditional_candidate_recall_given_true_row_active,
    directory_true_unit_recall,
    final_099_4x_report,
    level1_masked_accounting,
    mask_jaccard_summary,
    masked_grid,
    pilot_gate,
    stable_chronological_three_way_split,
    stable_chronological_two_way_split,
    true_unit_masked_distance_summary,
    unit_directory_pointer_accounting,
    validate_confirmation_configuration,
    validate_hardware_contract,
    wilson_binomial_interval,
)


def test_equal_timestamp_safe_fit_cal_test_split_keeps_timestamp_groups_whole():
    times = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    fit, calibration, test = stable_chronological_three_way_split(times)
    np.testing.assert_array_equal(fit, [0, 1, 2, 3])
    np.testing.assert_array_equal(calibration, [4, 5])
    np.testing.assert_array_equal(test, [6, 7])
    assert set(times[fit]).isdisjoint(set(times[calibration]))
    assert set(times[calibration]).isdisjoint(set(times[test]))


def test_fit_only_early_late_split_and_jaccard_exclude_missing_units_not_zero_fill():
    times = np.array([0, 0, 1, 1, 2, 2])
    early, late = stable_chronological_two_way_split(times)
    assert set(times[early]).isdisjoint(set(times[late]))
    # Unit 10 appears in both halves; 11 and 12 appear in one half only.
    features = np.array([
        [0.1] * DESCRIPTOR_DIM, [0.2] * DESCRIPTOR_DIM,
        [0.3] * DESCRIPTOR_DIM, [0.4] * DESCRIPTOR_DIM,
        [0.5] * DESCRIPTOR_DIM, [0.6] * DESCRIPTOR_DIM,
    ])
    labels = np.array([10, 11, 10, 12, 12, 12])
    summary = mask_jaccard_summary(
        features, labels, times, mask_mode="all_dims", mask_size=9
    )
    assert summary["early_late_fit_mask_jaccard_valid_units"] == 1
    assert summary["early_late_fit_mask_jaccard"] == pytest.approx(1.0)


def test_frozen_grid_has_only_all_dims_m9_and_all_nonall_sizes():
    grid = masked_grid()
    assert grid[0] == ("all_dims", 9)
    assert ("all_dims", 3) not in grid
    assert len(grid) == 9
    assert {(mode, size) for mode, size in grid if mode != "all_dims"} == {
        (mode, size) for mode in ("stability", "separation_stability") for size in (3, 5, 7, 9)
    }


def test_level1_masked_accounting_charges_actual_empty_fallback_rows_and_full_d_lanes():
    report = level1_masked_accounting(
        primary_rows=np.array([2, 1]), primary_lanes=np.array([8, 3]), fallback_rows=np.array([3, 0])
    )
    assert report["primary_level1_row_comparisons_total"] == 3
    assert report["empty_fallback_row_comparisons_total"] == 3
    assert report["empty_fallback_full_d_abs_lanes_total"] == 3 * DESCRIPTOR_DIM
    assert report["total_level1_row_comparisons"] == 6
    assert report["total_abs_lanes"] == 8 + 3 + 3 * DESCRIPTOR_DIM
    assert report["level1_row_accounting_identity_holds"]
    assert report["level1_lane_accounting_identity_holds"]
    assert report["primary_addition_accounting_identity_holds"]
    assert report["fallback_addition_accounting_identity_holds"]
    assert report["total_addition_accounting_identity_holds"]


def test_candidate_category_accounting_partitions_c0_c1_and_ambiguous_rows_lanes_additions():
    report = candidate_category_level1_accounting(
        np.array([0, 1, 3]), np.array([2, 2, 2]), np.array([4, 3, 6]), np.array([2, 0, 0])
    )
    assert report["c0_events"] == report["c1_events"] == report["cgt1_events"] == 1
    assert report["c0_total_level1_rows"] == 4
    assert report["c0_total_abs_lanes"] == 4 + 2 * DESCRIPTOR_DIM
    assert report["c0_total_abs_additions"] == 2 * DESCRIPTOR_DIM
    assert report["c1_total_abs_additions"] == 1
    assert report["cgt1_total_abs_additions"] == 4


def test_all_row_directory_does_not_mark_unseen_test_label_as_active_or_conditionally_recalled():
    labels = np.array([7, 99])
    units = np.array([7])
    candidates = [np.array([7]), np.array([7])]
    assert directory_true_unit_recall(None, labels, units) == pytest.approx(0.5)
    assert conditional_candidate_recall_given_true_row_active(candidates, labels, None, units) == pytest.approx(1.0)
    inactive = np.zeros((2, 1), dtype=bool)
    assert conditional_candidate_recall_given_true_row_active(candidates, labels, inactive, units) == 0.0


def test_fixed_hardware_contract_and_unit_directory_pointer_scope_numbers():
    validate_hardware_contract(5, 9)
    with pytest.raises(ValueError, match="B=5"):
        validate_hardware_contract(4, 9)
    with pytest.raises(ValueError, match="D=9"):
        validate_hardware_contract(5, 8)
    accounting = unit_directory_pointer_accounting(np.array([1, 2, 1]), n_units=3)
    assert accounting == {
        "directory_row_id_pointers_total": 4,
        "directory_row_id_pointer_bits_per_entry": 2,
        "directory_row_id_pointer_bits_total": 8,
    }


def test_wilson_interval_is_fixed_bounded_and_rejects_invalid_counts():
    low, high = wilson_binomial_interval(90, 100)
    assert low == pytest.approx(0.8256343384950865)
    assert high == pytest.approx(0.9447708629393249)
    assert wilson_binomial_interval(0, 10)[0] == pytest.approx(0.0)
    assert wilson_binomial_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least one"):
        wilson_binomial_interval(0, 0)
    with pytest.raises(ValueError, match=r"\[0, total\]"):
        wilson_binomial_interval(11, 10)


def test_true_unit_masked_distance_summary_tracks_radius_margin_and_unseen_labels():
    model = SimpleNamespace(
        units_=np.array([10, 20]),
        centroids_=np.array([[0] * DESCRIPTOR_DIM, [2] * DESCRIPTOR_DIM]),
        masks_=np.array([[True, True, True] + [False] * 6, [True] * DESCRIPTOR_DIM]),
        radii_=np.array([3, 9]),
    )
    report = true_unit_masked_distance_summary(
        np.array([[1] * DESCRIPTOR_DIM, [1] * DESCRIPTOR_DIM, [31] * DESCRIPTOR_DIM]),
        np.array([10, 20, 99]),
        model,
    )
    assert report["true_unit_distance_rows"] == 2
    assert report["true_unit_distance_unseen_label_rows"] == 1
    assert report["true_unit_masked_l1_distance_mean"] == pytest.approx(6.0)
    assert report["true_unit_radius_mean"] == pytest.approx(6.0)
    assert report["true_unit_distance_minus_radius_max"] == 0
    assert report["true_unit_distance_over_radius_plus_one_mean"] == pytest.approx(0.825)
    assert report["true_unit_distance_le_radius_fraction"] == pytest.approx(1.0)


def test_confirmation_configuration_requires_one_pre_registered_all_rows_grid_point():
    assert validate_confirmation_configuration("separation_stability", 3, 99.0) == (
        "separation_stability", 3, 99.0
    )
    with pytest.raises(ValueError, match="requires"):
        validate_confirmation_configuration(None, 3, 99.0)
    with pytest.raises(ValueError, match="all_dims"):
        validate_confirmation_configuration("all_dims", 3, 99.0)
    with pytest.raises(ValueError, match="95, 99, or 99.9"):
        validate_confirmation_configuration("stability", 3, 98.0)


def _gate_row(dataset, mode, size, recall, traffic, *, radius=99.0):
    n = 1000
    return {
        "dataset": dataset,
        "mask_mode": mode,
        "mask_size": size,
        "radius_percentile": radius,
        "directory_strategy": "all_rows",
        "directory_coverage": "",
        "n_test_gt_rows": n,
        "true_unit_candidate_hits": int(round(recall * n)),
        "true_unit_candidate_recall": recall,
        "actual_waveform_template_bits_read": int(1_000_000 / traffic),
        "global_all_unit_waveform_template_bits": 1_000_000,
        "waveform_bit_traffic_reduction_factor": traffic,
    }


def test_pilot_gate_requires_same_m_lt9_all_rows_config_to_pass_both_families():
    rows = {}
    for family in ("hj", "mearec"):
        family_rows = []
        for recording in (f"{family}_one", f"{family}_two"):
            family_rows.append(_gate_row(recording, "all_dims", 9, 0.900, 4.0))
            # Passes per-record tolerance and branch A in both family aggregates.
            family_rows.append(_gate_row(recording, "separation_stability", 3, 0.906, 4.0))
            # M=9 variants are deliberately not eligible even if they improve.
            family_rows.append(_gate_row(recording, "stability", 9, 0.950, 8.0))
        rows[family] = family_rows
    report = pilot_gate(rows)
    assert report["pilot_gate_pass"]
    assert report["pilot_gate_selected"]["config"][:2] == ("separation_stability", 3)
    assert report["pilot_gate_selected"]["branch"] == "A_recall"


def test_pilot_gate_rejects_cross_family_improvements_that_do_not_share_a_branch():
    rows = {
        "hj": [
            _gate_row("hj_a", "all_dims", 9, 0.900, 4.0),
            _gate_row("hj_a", "stability", 3, 0.906, 4.0),
        ],
        "mearec": [
            _gate_row("mea_a", "all_dims", 9, 0.900, 4.0),
            # Not enough recall gain for A and not enough traffic for B.
            _gate_row("mea_a", "stability", 3, 0.902, 4.1),
        ],
    }
    assert not pilot_gate(rows)["pilot_gate_pass"]


def test_pilot_gate_enforces_per_record_tolerance_before_family_aggregate_branch():
    rows = {}
    for family in ("hj", "mearec"):
        rows[family] = [
            _gate_row(f"{family}_one", "all_dims", 9, 0.900, 4.0),
            _gate_row(f"{family}_one", "stability", 3, 0.894, 4.0),  # below base - .005
            _gate_row(f"{family}_two", "all_dims", 9, 0.900, 4.0),
            _gate_row(f"{family}_two", "stability", 3, 0.920, 4.0),
        ]
    assert not pilot_gate(rows)["pilot_gate_pass"]


def test_final_099_4x_report_is_observational_not_a_pilot_selection_rule():
    rows = {
        family: [_gate_row(family, "all_dims", 9, 0.991, 4.1)]
        for family in ("hj", "mearec")
    }
    report = final_099_4x_report(rows)
    assert report["final_0p99_4x_pass"]
    assert report["final_0p99_4x_qualifying_configurations"] == 1
