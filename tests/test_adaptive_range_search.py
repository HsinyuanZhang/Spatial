"""Tests for per-unit 5-bit adaptive spatial candidate search."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import (
    AdaptiveRangeSearch,
    UniformUnsignedQuantizer,
    evaluate_candidate_search,
    fit_integer_centroids,
)


def test_unsigned_5_bit_bounds_clipping_and_train_only_fit():
    train = np.array([[2.0, 10.0], [6.0, 14.0]])
    quantizer = UniformUnsignedQuantizer(n_bits=5).fit(train)
    np.testing.assert_array_equal(quantizer.transform(train), [[0, 0], [31, 31]])
    # Held-out extrema must not alter fitted train bounds and are clipped.
    np.testing.assert_array_equal(quantizer.transform([[-9.0, 100.0]]), [[0, 31]])
    np.testing.assert_allclose(quantizer.data_min_, [2.0, 10.0])
    np.testing.assert_allclose(quantizer.data_max_, [6.0, 14.0])


def test_unit_interval_and_constant_dimensions_are_safe():
    fixed = UniformUnsignedQuantizer(fixed_unit_interval=True).fit([[42.0], [99.0]])
    np.testing.assert_array_equal(fixed.transform([[0.0], [1.0], [2.0]]).ravel(), [0, 31, 31])

    quantizer = UniformUnsignedQuantizer().fit([[3.0, 0.0], [3.0, 1.0]])
    assert quantizer.constant_mask_[0]
    np.testing.assert_array_equal(quantizer.transform([[3.0, 0.5]]), [[0, 16]])


def test_non_contiguous_labels_integer_centroids_and_per_unit_ranges():
    features = np.array([[0.0], [0.1], [0.2], [9.0], [10.0], [11.0]])
    labels = np.array([7, 7, 7, 103, 103, 103])
    search = AdaptiveRangeSearch(radius_percentile=50, box_coverage=0.5).fit(features, labels)
    assert set(search.units_) == {7, 103}
    assert search.radii_[0] != search.radii_[1]
    assert not np.array_equal(search.box_lower_[0], search.box_lower_[1])
    centroids = fit_integer_centroids(search.transform(features), labels)
    assert set(centroids) == {7, 103}
    assert np.issubdtype(centroids[7].dtype, np.integer)


def test_l1_and_box_comparisons_are_inclusive_at_boundaries():
    features = np.array([[0.0], [1.0], [10.0], [11.0]])
    labels = np.array([1, 1, 9, 9])
    search = AdaptiveRangeSearch(radius_percentile=100, box_coverage=1.0).fit(features, labels)
    codes = search.transform(features)
    # Training extrema must be admitted by their own L1 threshold and box faces.
    l1, _ = search.query_codes(codes, mode="l1")
    box, _ = search.query_codes(codes, mode="box")
    assert all(label in rows for label, rows in zip(labels, l1))
    assert all(label in rows for label, rows in zip(labels, box))


def test_box_joint_coverage_is_nested_and_full_coverage_uses_observed_bounds():
    # Diagonal and off-centre rows exercise joint coverage without relying on
    # simultaneous multi-axis widening, which used to make coverage non-monotone.
    features = np.array(
        [[0.0, 0.0], [0.0, 9.0], [1.0, 10.0], [9.0, 1.0],
         [10.0, 0.0], [10.0, 10.0], [20.0, 20.0], [21.0, 19.0]]
    )
    labels = np.full(features.shape[0], 5)
    coverages = (0.5, 0.75, 0.99, 1.0)
    fitted = [AdaptiveRangeSearch(box_coverage=value).fit(features, labels) for value in coverages]
    codes = fitted[0].transform(features)

    for coverage, search in zip(coverages, fitted):
        lower, upper = search.box_lower_[0], search.box_upper_[0]
        inside = np.all((codes >= lower) & (codes <= upper), axis=1)
        assert inside.mean() >= coverage
        assert search.box_training_coverage_[0] == inside.mean()
        assert np.all(lower <= search.centroids_[0])
        assert np.all(search.centroids_[0] <= upper)

    for narrower, wider in zip(fitted, fitted[1:]):
        assert np.all(wider.box_lower_[0] <= narrower.box_lower_[0])
        assert np.all(wider.box_upper_[0] >= narrower.box_upper_[0])

    np.testing.assert_array_equal(fitted[-1].box_lower_[0], codes.min(axis=0))
    np.testing.assert_array_equal(fitted[-1].box_upper_[0], codes.max(axis=0))


def test_active_mask_progressive_widening_and_optional_fallback():
    features = np.array([[0.0], [0.0], [10.0], [10.0]])
    labels = np.array([2, 2, 8, 8])
    search = AdaptiveRangeSearch().fit(features, labels)
    codes = search.transform([[5.0]])

    # Only the second directory row is active, and a one-code widening is
    # needed because the two fitted rows are at code 0 and 31.
    candidates, diag = search.query_codes(
        [[30]], mode="l1", active_mask=np.array([False, True]), widening_steps=1
    )
    assert candidates[0].tolist() == [8]
    assert diag["active_row_count"].tolist() == [1]
    assert diag["initial_candidate_count"].tolist() == [0]
    assert diag["final_candidate_count"].tolist() == [1]
    assert diag["widen_level"].tolist() == [1]
    assert diag["row_comparisons"].tolist() == [2]
    assert not diag["fallback"][0]

    none, missing = search.query_codes(codes, active_mask=np.array([True, False]))
    assert none[0].size == 0
    assert missing["final_candidate_count"][0] == 0
    backed, fallback = search.query_codes(
        codes, active_mask=np.array([True, False]), ensure_nonempty=True
    )
    assert backed[0].tolist() == [2]
    assert fallback["fallback"][0]
    # The ordinary failed membership test and the explicit normalized
    # outside-row fallback each inspect the active row once.
    assert fallback["row_comparisons"].tolist() == [1]
    assert fallback["fallback_row_comparisons"].tolist() == [1]
    assert fallback["total_row_comparisons_including_fallback"].tolist() == [2]


def test_metrics_and_hardware_bit_formulas():
    features = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    labels = np.array([10, 10, 30, 30])
    search = AdaptiveRangeSearch(n_bits=5).fit(features, labels)
    candidates, diag = search.query(features, mode="l1")
    report = evaluate_candidate_search(candidates, labels, diag)
    assert report["true_unit_candidate_recall"] == 1.0
    assert report["zero_fraction"] == 0.0
    assert report["stage2_activation_fraction"] == report["ambiguous_fraction"]
    assert report["p95_active_rows"] == 2.0
    assert report["max_active_rows"] == 2
    assert report["p95_row_comparisons"] == 2.0
    assert report["max_row_comparisons"] == 2
    assert report["total_row_comparisons"] == 8
    assert search.l1_row_bits == 2 * 5 + search.radius_bits
    assert search.radius_capacity_bits == 6  # ceil(log2(2 * 31 + 1))
    assert search.l1_row_bits_full_scale == 2 * 5 + 6
    assert search.box_row_bits == 2 * 2 * 5
    assert search.hardware_costs()["l1_row_bits"] == search.l1_row_bits
    assert search.hardware_costs()["default_l1_row_bits"] == search.l1_row_bits_full_scale
