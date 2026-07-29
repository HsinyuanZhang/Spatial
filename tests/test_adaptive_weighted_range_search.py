"""No-I/O semantic tests for fixed 5-bit power-of-two weighted L1 rows."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_weighted_range_search import AdaptiveWeightedRangeSearch


def _manual_l1_candidates(codes, centroids, radii, units, active):
    distance = np.abs(codes[:, None, :] - centroids[None, :, :]).sum(axis=2)
    matched = active & (distance <= radii[None, :])
    return [units[rows] for rows in matched], matched


def test_uniform_weights_match_independent_calibrated_unweighted_l1_exactly():
    train = np.vstack([np.zeros((3, 9)), np.ones((3, 9))])
    labels = np.array([7, 7, 7, 103, 103, 103])
    calibration = np.vstack([np.full((2, 9), 0.2), np.full((2, 9), 0.8)])
    calibration_labels = np.array([7, 7, 103, 103])
    model = AdaptiveWeightedRangeSearch(radius_percentile=100.0).fit(
        train, labels, score_mode="uniform"
    ).calibrate(calibration, calibration_labels)
    np.testing.assert_array_equal(model.weights_, np.ones((2, 9), dtype=np.int64))
    np.testing.assert_array_equal(model.shift_codes_, np.zeros((2, 9), dtype=np.int64))

    calibration_codes = model.transform(calibration)
    manual_radii = np.asarray([
        int(np.ceil(np.percentile(
            np.abs(calibration_codes[calibration_labels == unit] - model.centroids_[row]).sum(axis=1),
            100.0, method="linear",
        )))
        for row, unit in enumerate(model.units_)
    ])
    np.testing.assert_array_equal(model.radii_, manual_radii)
    query_codes = np.vstack([np.zeros(9, dtype=int), np.full(9, 31, dtype=int), np.full(9, 6, dtype=int)])
    active = np.array([[True, True], [False, True], [True, False]])
    expected, matched = _manual_l1_candidates(
        query_codes, model.centroids_, manual_radii, model.units_, active
    )
    actual, diagnostics = model.query_codes(query_codes, active_row_mask=active)
    assert [rows.tolist() for rows in actual] == [rows.tolist() for rows in expected]
    np.testing.assert_array_equal(diagnostics["active_row_count"], active.sum(axis=1))
    np.testing.assert_array_equal(diagnostics["logical_row_comparisons"], active.sum(axis=1))
    np.testing.assert_array_equal(diagnostics["candidate_count"], matched.sum(axis=1))


def test_shift_codes_apply_expected_power_of_two_distance_and_boundary_is_inclusive():
    # Stability ranking picks feature 0 (zero dispersion) for x4.  The other
    # eight coordinates vary, so the top3 profile also promotes 1 and 2; query
    # feature 0 has a known fourfold contribution.
    features = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    model = AdaptiveWeightedRangeSearch(radius_percentile=100.0).fit(
        features, [7, 7], score_mode="stability", rank_profile="top3_x4"
    )
    assert model.weights_[0, 0] == 4
    code = model.centroids_[0].copy()
    code[0] = min(code[0] + 1, 31)
    weighted_distance = int((np.abs(code - model.centroids_[0]) * model.weights_[0]).sum())
    assert weighted_distance == 4
    model.radii_[0] = weighted_distance
    candidates, _ = model.query_codes([code])
    assert candidates[0].tolist() == [7]


def test_stable_ties_and_single_unit_separation_are_auditable():
    model = AdaptiveWeightedRangeSearch().fit(
        np.full((3, 9), 0.5), [103, 103, 103],
        score_mode="separation_stability", rank_profile="tiered_421",
    )
    np.testing.assert_array_equal(model.between_median_separation_, np.zeros((1, 9)))
    np.testing.assert_array_equal(model.separation_stability_scores_, np.zeros((1, 9)))
    np.testing.assert_array_equal(model.feature_ranks_[0], np.arange(9))
    np.testing.assert_array_equal(model.weights_[0], [4, 4, 4, 2, 2, 2, 1, 1, 1])
    np.testing.assert_array_equal(model.shift_codes_[0], [2, 2, 2, 1, 1, 1, 0, 0, 0])


def test_fit_only_state_is_unchanged_by_calibration_or_test_perturbation():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    model = AdaptiveWeightedRangeSearch(radius_percentile=100.0).fit(
        train, [7, 7, 103, 103], score_mode="separation_stability", rank_profile="tiered_421"
    )
    frozen = tuple(
        value.copy() for value in (
            model.centroids_, model.within_q95_, model.between_median_separation_,
            model.separation_stability_scores_, model.feature_ranks_, model.weights_, model.shift_codes_,
        )
    )
    initial_radii = model.radii_.copy()
    model.calibrate(np.full((2, 9), 0.5), [103, 7])
    assert not np.array_equal(model.radii_, initial_radii)
    model.query(np.vstack([np.zeros(9), np.ones(9)]))
    for actual, expected in zip(
        (model.centroids_, model.within_q95_, model.between_median_separation_,
         model.separation_stability_scores_, model.feature_ranks_, model.weights_, model.shift_codes_),
        frozen,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_noncontiguous_units_active_masks_and_shift_operation_counters():
    features = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    model = AdaptiveWeightedRangeSearch().fit(
        features, [7, 7, 103, 103], score_mode="stability", rank_profile="tiered_421"
    )
    active = np.array([[True, False], [False, True]])
    candidates, diagnostics = model.query_codes(
        np.vstack([np.zeros(9, dtype=int), np.full(9, 31, dtype=int)]), active_row_mask=active
    )
    assert [rows.tolist() for rows in candidates] == [[7], [103]]
    np.testing.assert_array_equal(diagnostics["logical_enabled_abs_lanes"], [9, 9])
    np.testing.assert_array_equal(diagnostics["logical_shift1_lanes"], [3, 3])
    np.testing.assert_array_equal(diagnostics["logical_shift2_lanes"], [3, 3])
    np.testing.assert_array_equal(diagnostics["logical_accumulator_additions"], [8, 8])
    np.testing.assert_array_equal(
        diagnostics["logical_total_abs_lanes"],
        diagnostics["logical_total_level1_row_comparisons"] * 9,
    )
    np.testing.assert_array_equal(
        diagnostics["logical_shifted_lanes"],
        diagnostics["logical_shift1_lanes"] + diagnostics["logical_shift2_lanes"]
    )


def test_calibration_missing_unit_keeps_fit_radius_and_unseen_label_is_ignored():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    model = AdaptiveWeightedRangeSearch(radius_percentile=100.0).fit(train, [7, 7, 103, 103])
    fit_radii = model.radii_.copy()
    model.calibrate(np.vstack([np.full((1, 9), 0.5), np.ones((1, 9))]), [7, 999])
    assert model.units_.tolist() == [7, 103]
    assert model.radii_[0] != fit_radii[0]
    assert model.radii_[1] == fit_radii[1]
    np.testing.assert_array_equal(model.radius_calibration_counts_, [1, 0])
    np.testing.assert_array_equal(model.radius_calibration_fallback_mask_, [False, True])


def test_uniform_reference_parity_includes_missing_calibration_and_zero_radius_policy():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    labels = np.array([7, 7, 103, 103])
    model = AdaptiveWeightedRangeSearch(radius_percentile=100.0).fit(
        train, labels, score_mode="uniform"
    )
    # Unit 7 receives a zero-radius calibration set; unit 103 is missing and
    # must retain its zero fit radius.  The independent reference uses exactly
    # this declared missing-unit policy rather than legacy fit-only radii.
    model.calibrate(np.zeros((1, 9)), [7])
    np.testing.assert_array_equal(model.radii_, [0, 0])
    np.testing.assert_array_equal(model.radius_calibration_fallback_mask_, [False, True])
    query = np.vstack([np.zeros(9, dtype=int), np.full(9, 31, dtype=int)])
    active = np.array([[True, True], [True, True]])
    expected, matches = _manual_l1_candidates(
        query, model.centroids_, np.array([0, 0]), model.units_, active
    )
    actual, diagnostics = model.query_codes(query, active_row_mask=active)
    assert [row.tolist() for row in actual] == [row.tolist() for row in expected]
    np.testing.assert_array_equal(diagnostics["candidate_count"], matches.sum(axis=1))
    np.testing.assert_array_equal(diagnostics["active_row_count"], active.sum(axis=1))
    np.testing.assert_array_equal(diagnostics["logical_row_comparisons"], active.sum(axis=1))


def test_zero_radius_and_11_bit_maximum_distance_and_74_bit_accounting():
    model = AdaptiveWeightedRangeSearch().fit(
        np.zeros((2, 9)), [7, 7], score_mode="stability", rank_profile="top3_x4"
    )
    assert model.radii_[0] == 0
    maximum = int((np.full(9, 31) * 4).sum())
    assert maximum == 1116 <= (1 << 11) - 1
    accounting = model.memory_accounting()
    assert accounting["logical_search_payload_bits_per_row"] == 74
    assert accounting["optimized_unweighted_payload_bits_per_row"] == 54
    assert accounting["full_scale_weighted_radius_bits_per_row"] == 11
    assert accounting["maximum_weighted_distance"] == 1116
    assert accounting["shift_code_encoding"] == "00->1,01->2,10->4,11->reserved_invalid"


def test_reserved_external_shift_code_is_rejected_before_query_or_accounting():
    model = AdaptiveWeightedRangeSearch().fit(np.zeros((2, 9)), [7, 7])
    model.shift_codes_[0, 0] = 3
    with pytest.raises(ValueError):
        model.query_codes([np.zeros(9, dtype=int)])
    with pytest.raises(ValueError):
        model.memory_accounting()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: AdaptiveWeightedRangeSearch(n_bits=4),
        lambda: AdaptiveWeightedRangeSearch().fit(np.zeros((2, 8)), [7, 7]),
        lambda: AdaptiveWeightedRangeSearch().fit(np.zeros((2, 9)), [7, 7], score_mode="bad"),
        lambda: AdaptiveWeightedRangeSearch().fit(np.zeros((2, 9)), [7, 7], rank_profile="bad"),
    ],
)
def test_invalid_bits_dimensions_and_configuration_are_rejected(operation):
    with pytest.raises(ValueError):
        operation()
