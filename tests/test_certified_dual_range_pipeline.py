"""No-I/O tests for temporal calibration and certified safe exits."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.experiments.run_certified_dual_range_pipeline import (
    _calibration_summary,
    _certified_test_metrics,
    _safe_flag_bits_for_method,
    apply_certified_dual_range_policy,
    certify_safe_units,
    chronological_fraction_split,
    wilson_lower_bound,
)
from Spatial.experiments.run_dual_range_pipeline import payload_row_bits


def _diag(rows):
    values = np.asarray(rows, dtype=np.int64)
    return {"row_comparisons": values, "active_row_count": np.full(values.size, 3, dtype=np.int64)}


def test_chronological_fraction_split_never_divides_equal_timestamps():
    times = np.array([1, 2, 2, 3, 4, 4, 5])
    left, right = chronological_fraction_split(times, 0.5)
    np.testing.assert_array_equal(left, [0, 1, 2])
    np.testing.assert_array_equal(right, [3, 4, 5, 6])
    assert times[left[-1]] < times[right[0]]


def test_wilson_lower_bound_is_conservative_and_handles_no_support():
    assert wilson_lower_bound(0, 0, 1.645) == 0.0
    with np.testing.assert_raises(ValueError):
        wilson_lower_bound(0, 0, -0.1)
    with np.testing.assert_raises(ValueError):
        wilson_lower_bound(0, 0, float("nan"))
    assert 0.78 < wilson_lower_bound(10, 10, 1.645) < 0.80
    assert wilson_lower_bound(8, 10, 1.645) < 0.8
    assert wilson_lower_bound(8, 10, 0.0) == 0.8


def test_certification_counts_by_predicted_singleton_unit_not_true_unit():
    certification = certify_safe_units(
        [np.array([2]), np.array([2]), np.array([99])], np.array([99, 99, 99]), np.array([2, 99]),
        precision_target=0.5, min_support=1, z=0.0,
    )
    # The two wrong singleton outputs are evidence *against predicted unit 2*,
    # not calibration support for the true unit 99.
    np.testing.assert_array_equal(certification["support"], [2, 1])
    np.testing.assert_array_equal(certification["correct"], [0, 1])
    np.testing.assert_array_equal(certification["safe"], [False, True])


def test_unsafe_singleton_must_trigger_outer_and_accounts_rows():
    candidates, diag = apply_certified_dual_range_policy(
        [np.array([2]), np.array([99])], _diag([3, 3]),
        [np.array([99]), np.array([2, 99])], _diag([3, 3]),
        np.array([2, 99]), np.array([False, True]),
    )
    np.testing.assert_array_equal(diag["safe_singleton"], [False, True])
    np.testing.assert_array_equal(diag["outer_trigger"], [True, False])
    np.testing.assert_array_equal(candidates[0], [99, 2])
    np.testing.assert_array_equal(candidates[1], [99])
    np.testing.assert_array_equal(diag["inner_row_comparisons"], [3, 3])
    np.testing.assert_array_equal(diag["outer_row_comparisons"], [3, 0])
    np.testing.assert_array_equal(diag["row_comparisons"], [6, 3])
    np.testing.assert_array_equal(diag["inner_active_row_count"], [3, 3])
    np.testing.assert_array_equal(diag["outer_active_row_count"], [3, 0])
    np.testing.assert_array_equal(diag["total_active_row_accesses"], [6, 3])
    np.testing.assert_array_equal(diag["fallback_row_comparisons"], [0, 0])
    np.testing.assert_array_equal(diag["total_row_comparisons_including_fallback"], [6, 3])


def test_wrong_certified_singleton_really_lowers_recall_without_repair():
    candidates, diag = apply_certified_dual_range_policy(
        [np.array([2])], _diag([3]), [np.array([99])], _diag([3]),
        np.array([2, 99]), np.array([True, False]),
    )
    report = evaluate_candidate_search(candidates, np.array([99]), diag)
    assert diag["safe_singleton"][0]
    assert not diag["outer_trigger"][0]
    np.testing.assert_array_equal(candidates[0], [2])
    assert report["true_unit_candidate_recall"] == 0.0


def test_summary_calibration_fields_and_d9_b5_raw_plus_flag_payload():
    certification = {
        "safe": np.array([True, False, True]),
        "support": np.array([5, 4, 1]),
        "correct": np.array([5, 3, 1]),
        "wilson_lower_bound": np.array([0.65, 0.2, 0.4]),
    }
    summary = _calibration_summary(certification)
    for key in (
        "safe_unit_count", "safe_unit_fraction", "calibration_singleton_support",
        "calibration_singleton_correct", "calibration_singleton_empirical_precision",
    ):
        assert key in summary
    assert summary["safe_unit_count"] == 2
    assert summary["calibration_singleton_support"] == 10
    assert summary["calibration_singleton_correct"] == 9
    inner = [np.array([2]), np.array([2]), np.array([99, 2])]
    test_metrics = _certified_test_metrics(
        inner, np.array([2, 99, 99]),
        {"safe_singleton": np.array([True, False, False]), "outer_trigger": np.array([False, True, True])},
    )
    assert test_metrics["certified_early_exit_fraction"] == 1 / 3
    assert test_metrics["test_certified_singleton_precision"] == 1.0
    assert test_metrics["unsafe_singleton_outer_trigger_fraction"] == 1 / 3
    assert payload_row_bits(9, 5) == (90, 54, 144)
    assert payload_row_bits(9, 5)[2] + 1 == 145
    assert _safe_flag_bits_for_method("outer_l1_only") == 0
    assert _safe_flag_bits_for_method("uncertified_dual") == 0
    assert _safe_flag_bits_for_method("certified_dual") == 1
