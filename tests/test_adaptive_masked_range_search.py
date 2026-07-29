"""No-I/O semantics for fixed-width unit-specific masked 5-bit L1 rows."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_masked_range_search import AdaptiveMaskedRangeSearch
from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch


def test_all_dims_matches_fixed_interval_adaptive_range_search_exactly():
    rng = np.random.default_rng(20260726)
    features = rng.uniform(0.0, 1.0, size=(16, 9))
    labels = np.array([7] * 8 + [103] * 8)
    baseline = AdaptiveRangeSearch(
        n_bits=5, fixed_unit_interval=True, radius_percentile=95.0
    ).fit(features, labels)
    masked = AdaptiveMaskedRangeSearch(radius_percentile=95.0).fit(
        features, labels, mask_size=9, mask_mode="all_dims"
    )

    np.testing.assert_array_equal(masked.units_, baseline.units_)
    np.testing.assert_array_equal(masked.centroids_, baseline.centroids_)
    np.testing.assert_array_equal(masked.radii_, baseline.radii_)
    np.testing.assert_array_equal(masked.masks_, np.ones((2, 9), dtype=bool))
    codes = baseline.transform(rng.uniform(0.0, 1.0, size=(5, 9)))
    expected, _ = baseline.query_codes(codes, mode="l1")
    actual, diagnostics = masked.query_codes(codes)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(diagnostics["candidate_count"], [len(row) for row in expected])


def test_disabled_dimension_does_not_change_masked_membership():
    # Dimensions 0--2 have zero within dispersion, so stability M=3 selects
    # them.  The remaining dimensions vary strongly and must be gated off.
    features = np.array(
        [
            [0.5, 0.5, 0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.5, 0.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        ]
    )
    model = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(
        features, [41, 41], mask_size=3, mask_mode="stability"
    )
    np.testing.assert_array_equal(model.masks_[0], [True, True, True, False, False, False, False, False, False])
    centre = model.centroids_[0].copy()
    perturbed = centre.copy()
    perturbed[3] = 31 if centre[3] != 31 else 0
    before, _ = model.query_codes([centre])
    after, _ = model.query_codes([perturbed])
    assert before[0].tolist() == after[0].tolist() == [41]


def test_distance_equal_to_radius_is_inclusive():
    features = np.zeros((2, 9), dtype=float)
    features[1, 0] = 0.2  # quantizes to code 6; centroid is 3 and radius is 3.
    model = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(
        features, [7, 7], mask_size=9, mask_mode="all_dims"
    )
    assert model.radii_[0] == 3
    candidates, _ = model.query_codes([np.zeros(9, dtype=int)])
    assert candidates[0].tolist() == [7]


def test_stable_feature_id_ties_apply_to_both_mask_scores():
    features = np.full((4, 9), 0.5)
    labels = np.full(4, 103)
    stability = AdaptiveMaskedRangeSearch().fit(
        features, labels, mask_size=3, mask_mode="stability"
    )
    separation = AdaptiveMaskedRangeSearch().fit(
        features, labels, mask_size=3, mask_mode="separation_stability"
    )
    expected = np.array([True, True, True, False, False, False, False, False, False])
    np.testing.assert_array_equal(stability.masks_[0], expected)
    # With one unit separation is specified as zero, so the score ties retain
    # ascending feature IDs as well.
    np.testing.assert_array_equal(separation.masks_[0], expected)


def test_test_queries_cannot_change_fit_only_masks():
    train = np.array(
        [np.linspace(0.0, 0.8, 9), np.linspace(0.1, 0.9, 9), np.linspace(0.2, 1.0, 9)]
    )
    model = AdaptiveMaskedRangeSearch().fit(train, [7, 7, 7], mask_size=5, mask_mode="stability")
    fitted_masks = model.masks_.copy()
    model.query(np.zeros((3, 9)))
    model.query(np.ones((3, 9)))
    np.testing.assert_array_equal(model.masks_, fitted_masks)


def test_calibrate_updates_only_radii_and_reports_missing_unit_fallback():
    train = np.zeros((4, 9), dtype=float)
    train[2:] = 1.0
    labels = np.array([7, 7, 103, 103])
    model = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(train, labels)
    centroids, masks, old_radii = model.centroids_.copy(), model.masks_.copy(), model.radii_.copy()
    calibration = np.full((2, 9), 0.5)
    model.calibrate(calibration, [7, 7])

    np.testing.assert_array_equal(model.centroids_, centroids)
    np.testing.assert_array_equal(model.masks_, masks)
    assert model.radii_[0] != old_radii[0]
    assert model.radii_[1] == old_radii[1]
    np.testing.assert_array_equal(model.radius_calibration_counts_, [2, 0])
    np.testing.assert_array_equal(model.radius_calibration_fallback_mask_, [False, True])


def test_active_masks_noncontiguous_ids_and_enabled_lane_accounting():
    features = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    labels = np.array([7, 7, 103, 103])
    model = AdaptiveMaskedRangeSearch().fit(features, labels, mask_size=3, mask_mode="stability")
    candidates, diagnostics = model.query_codes(
        np.vstack([np.zeros(9, dtype=int), np.full(9, 31, dtype=int)]),
        active_row_mask=np.array([[True, False], [False, True]]),
    )
    assert [rows.tolist() for rows in candidates] == [[7], [103]]
    np.testing.assert_array_equal(diagnostics["active_row_count"], [1, 1])
    np.testing.assert_array_equal(diagnostics["row_comparisons"], [1, 1])
    np.testing.assert_array_equal(diagnostics["enabled_abs_lanes"], [3, 3])


def test_fixed_63_bit_accounting_and_enabled_dimension_summary():
    features = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    model = AdaptiveMaskedRangeSearch().fit(
        features, [7, 7, 103, 103], mask_size=5, mask_mode="stability"
    )
    accounting = model.memory_accounting()
    assert accounting["centroid_bits_per_row"] == 45
    assert accounting["mask_bits_per_row"] == 9
    assert accounting["full_scale_radius_bits_per_row"] == 9
    assert accounting["search_payload_bits_per_row"] == 63
    assert accounting["unit_id_bits_per_row"] == 1
    assert accounting["valid_bits_per_row"] == 1
    assert accounting["row_bits_with_metadata_per_row"] == 65
    assert accounting["enabled_dimensions_mean"] == 5.0
    assert accounting["enabled_dimensions_max"] == 5


def test_separation_stability_uses_exact_score_and_stable_feature_ties():
    # Each unit is internally constant, so within=0.  Feature 4 is the only
    # inter-centroid separation; all irrelevant features tie at score zero and
    # therefore use ascending feature ID to fill the remaining M=3 slots.
    features = np.vstack([np.zeros((2, 9)), np.tile(np.eye(1, 9, 4), (2, 1))])
    labels = np.array([7, 7, 103, 103])
    model = AdaptiveMaskedRangeSearch().fit(
        features, labels, mask_size=3, mask_mode="separation_stability"
    )
    np.testing.assert_array_equal(model.within_q95_, np.zeros((2, 9)))
    separation = np.abs(model.centroids_[1] - model.centroids_[0]).astype(float)
    expected_score = separation / (model.within_q95_[0] + 1.0)
    np.testing.assert_array_equal(model.between_median_separation_[0], separation)
    np.testing.assert_array_equal(model.between_median_separation_[1], separation)
    np.testing.assert_array_equal(expected_score, [0.0, 0.0, 0.0, 0.0, 31.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(model.separation_stability_scores_[0], expected_score)
    np.testing.assert_array_equal(model.separation_stability_scores_[1], expected_score)
    expected_mask = np.array([True, True, False, False, True, False, False, False, False])
    np.testing.assert_array_equal(model.masks_[0], expected_mask)
    np.testing.assert_array_equal(model.masks_[1], expected_mask)


def test_calibration_perturbations_only_change_radii_not_fit_only_state():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    model = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(
        train, [7, 7, 103, 103], mask_size=3, mask_mode="separation_stability"
    )
    baseline_calibration = np.vstack([np.zeros((1, 9)), np.ones((1, 9))])
    model.calibrate(baseline_calibration, [7, 103])
    fit_only = (
        model.centroids_.copy(),
        model.within_q95_.copy(),
        model.between_median_separation_.copy(),
        model.separation_stability_scores_.copy(),
        model.masks_.copy(),
    )
    first_radii = model.radii_.copy()

    # Reassigning the calibration rows and perturbing their feature values may
    # legitimately change radii, but it must never alter fit-only scores/masks.
    changed_calibration = np.full((2, 9), 0.5)
    model.calibrate(changed_calibration, [103, 7])
    assert not np.array_equal(model.radii_, first_radii)
    np.testing.assert_array_equal(model.centroids_, fit_only[0])
    np.testing.assert_array_equal(model.within_q95_, fit_only[1])
    np.testing.assert_array_equal(model.between_median_separation_, fit_only[2])
    np.testing.assert_array_equal(model.separation_stability_scores_, fit_only[3])
    np.testing.assert_array_equal(model.masks_, fit_only[4])


def test_zero_radius_is_legal_and_full_scale_masked_distance_fits_nine_bit_field():
    model = AdaptiveMaskedRangeSearch().fit(
        np.zeros((2, 9)), [7, 7], mask_size=9, mask_mode="all_dims"
    )
    assert model.radii_[0] == 0
    candidates, _ = model.query_codes([model.centroids_[0]])
    assert candidates[0].tolist() == [7]
    maximum_masked_distance = int((model.masks_[0] * 31).sum())
    assert maximum_masked_distance == 9 * 31 == 279
    assert maximum_masked_distance <= (1 << 9) - 1
    assert model.radii_[0] <= maximum_masked_distance


def test_all_dims_calibration_matches_manual_unmasked_l1_candidates_and_accesses():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    labels = np.array([7, 7, 103, 103])
    calibration = np.vstack([np.full((2, 9), 0.2), np.full((2, 9), 0.8)])
    calibration_labels = np.array([7, 7, 103, 103])
    model = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(
        train, labels, mask_size=9, mask_mode="all_dims"
    ).calibrate(calibration, calibration_labels)
    calibration_codes = model.transform(calibration)
    manual_radii = np.asarray(
        [
            int(np.ceil(np.percentile(np.abs(calibration_codes[calibration_labels == unit] - model.centroids_[row]).sum(axis=1), 100.0)))
            for row, unit in enumerate(model.units_)
        ]
    )
    np.testing.assert_array_equal(model.radii_, manual_radii)

    query_codes = np.vstack([np.zeros(9, dtype=int), np.full(9, 31, dtype=int), np.full(9, 6, dtype=int)])
    active = np.array([[True, True], [False, True], [True, False]])
    expected_match = active & (
        np.abs(query_codes[:, None, :] - model.centroids_[None, :, :]).sum(axis=2)
        <= manual_radii[None, :]
    )
    candidates, diagnostics = model.query_codes(query_codes, active_row_mask=active)
    assert [rows.tolist() for rows in candidates] == [model.units_[rows].tolist() for rows in expected_match]
    np.testing.assert_array_equal(diagnostics["active_row_count"], active.sum(axis=1))
    np.testing.assert_array_equal(diagnostics["row_comparisons"], active.sum(axis=1))


def test_calibration_only_unseen_labels_are_ignored_without_allocating_rows():
    train = np.vstack([np.zeros((2, 9)), np.ones((2, 9))])
    labels = np.array([7, 7, 103, 103])
    base = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(train, labels)
    calibration = np.vstack([np.zeros((1, 9)), np.ones((1, 9))])
    base.calibrate(calibration, [7, 103])
    expected_radii = base.radii_.copy()

    with_unseen = AdaptiveMaskedRangeSearch(radius_percentile=100.0).fit(train, labels)
    with_unseen.calibrate(
        np.vstack([calibration, np.full((1, 9), 0.5)]), [7, 103, 999]
    )
    np.testing.assert_array_equal(with_unseen.units_, [7, 103])
    assert with_unseen.units_.size == 2
    np.testing.assert_array_equal(with_unseen.radii_, expected_radii)
    np.testing.assert_array_equal(with_unseen.radius_calibration_counts_, [1, 1])
    np.testing.assert_array_equal(with_unseen.radius_calibration_fallback_mask_, [False, False])


@pytest.mark.parametrize(
    "constructor_or_fit",
    [
        lambda: AdaptiveMaskedRangeSearch(n_bits=4),
        lambda: AdaptiveMaskedRangeSearch().fit(np.zeros((2, 8)), [7, 7]),
        lambda: AdaptiveMaskedRangeSearch().fit(np.zeros((2, 9)), [7, 7], mask_size=1),
        lambda: AdaptiveMaskedRangeSearch().fit(
            np.zeros((2, 9)), [7, 7], mask_size=3, mask_mode="all_dims"
        ),
        lambda: AdaptiveMaskedRangeSearch().fit(
            np.zeros((2, 9)), [7, 7], mask_size=3, mask_mode="not_a_mode"
        ),
    ],
)
def test_invalid_dimension_bits_mask_size_and_mode_are_rejected(constructor_or_fit):
    with pytest.raises(ValueError):
        constructor_or_fit()
