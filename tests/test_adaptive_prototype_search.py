"""Tests for bounded deterministic 5-bit multi-prototype Level-1 search."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_prototype_search import AdaptivePrototypeSearch
from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch


def test_one_prototype_matches_adaptive_range_search_l1_centroid_and_radius():
    features = np.array(
        [[0.01, 0.10], [0.23, 0.30], [0.45, 0.50], [0.70, 0.65], [0.99, 0.90]]
    )
    labels = np.array([103, 7, 103, 7, 103])
    baseline = AdaptiveRangeSearch(
        n_bits=5, fixed_unit_interval=True, radius_percentile=75.0
    ).fit(features, labels)
    search = AdaptivePrototypeSearch(n_bits=5, radius_percentile=75.0).fit(
        features, labels, prototypes_per_unit=1, mode="farthest_l1"
    )

    np.testing.assert_array_equal(search.units_, baseline.units_)
    np.testing.assert_array_equal(search.prototype_units_, baseline.units_)
    np.testing.assert_array_equal(search.centroids_, baseline.centroids_)
    np.testing.assert_array_equal(search.radii_, baseline.radii_)

    query_codes = baseline.transform([[0.20, 0.30], [0.80, 0.80]])
    expected, _ = baseline.query_codes(query_codes, mode="l1")
    actual, diagnostics = search.query_codes(query_codes)
    for got, wanted in zip(actual, expected):
        np.testing.assert_array_equal(got, wanted)
    np.testing.assert_array_equal(
        diagnostics["prototype_hit_count"], diagnostics["distinct_candidate_count"]
    )


def test_multiple_prototypes_deduplicate_non_contiguous_units_in_row_order():
    # Unit 7 deliberately has two identical temporal prototype rows.  A single
    # query therefore creates two hits but only one waveform candidate.
    features = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    labels = np.array([7, 7, 7, 7, 103, 103])
    search = AdaptivePrototypeSearch().fit(features, labels, prototypes_per_unit=2, mode="temporal")

    candidates, diagnostics = search.query_codes([[0]])
    assert candidates[0].tolist() == [7, 103]
    assert diagnostics["prototype_hit_count"].tolist() == [4]
    assert diagnostics["distinct_candidate_count"].tolist() == [2]
    assert diagnostics["duplicates_removed"].tolist() == [2]
    assert diagnostics["active_row_count"].tolist() == [4]
    assert diagnostics["row_comparisons"].tolist() == [4]


def test_l1_membership_is_inclusive_at_the_integer_radius_boundary():
    features = np.array([[0.0], [0.2]])
    search = AdaptivePrototypeSearch(radius_percentile=100.0).fit(
        features, [41, 41], prototypes_per_unit=1
    )
    assert search.centroids_[0, 0] == 3
    assert search.radii_[0] == 3

    candidates, diagnostics = search.query_codes([[0]])
    assert candidates[0].tolist() == [41]
    assert diagnostics["prototype_hit_count"].tolist() == [1]


def test_active_row_masks_support_shared_and_per_query_prototype_rows():
    features = np.array([[0.0], [0.0], [1.0], [1.0]])
    labels = np.array([7, 7, 103, 103])
    search = AdaptivePrototypeSearch().fit(features, labels, prototypes_per_unit=2, mode="temporal")

    # Rows 0/1 belong to unit 7 and rows 2/3 to 103 in fixed row order.
    shared_candidates, shared = search.query_codes([[0]], active_row_mask=np.array([False, True, False, False]))
    assert shared_candidates[0].tolist() == [7]
    assert shared["active_row_count"].tolist() == [1]
    assert shared["row_comparisons"].tolist() == [1]

    mask = np.array([[True, False, False, False], [False, False, True, False]])
    candidates, diagnostics = search.query_codes([[0], [31]], active_row_mask=mask)
    assert [rows.tolist() for rows in candidates] == [[7], [103]]
    assert diagnostics["active_prototype_row_count"].tolist() == [1, 1]
    assert diagnostics["row_comparisons"].tolist() == [1, 1]


def test_storage_uses_54_bit_search_payload_for_nine_5_bit_features():
    features = np.vstack(
        [np.zeros((2, 9), dtype=float), np.ones((2, 9), dtype=float)]
    )
    labels = np.array([7, 7, 103, 103])
    search = AdaptivePrototypeSearch().fit(features, labels, prototypes_per_unit=2)

    accounting = search.memory_accounting()
    assert accounting["search_payload_bits_per_prototype"] == 54
    assert accounting["centroid_bits_per_prototype"] == 45
    assert accounting["full_scale_radius_bits_per_prototype"] == 9
    assert accounting["unit_id_bits_per_prototype"] == 1
    assert accounting["valid_bits_per_prototype"] == 1
    assert accounting["row_bits_with_metadata_per_prototype"] == 56
    assert accounting["n_prototype_rows"] == 4
    assert accounting["configured_prototypes_per_unit_mean"] == 2.0
    assert accounting["configured_prototypes_per_unit_max"] == 2


def test_farthest_l1_is_deterministic_and_exposes_pilot_cluster_assignments():
    features = np.array([[0.0], [0.0], [0.4], [0.5], [0.9], [1.0], [0.1], [0.8]])
    labels = np.array([7, 7, 7, 7, 7, 7, 103, 103])
    first = AdaptivePrototypeSearch().fit(
        features, labels, prototypes_per_unit=3, mode="farthest_l1"
    )
    second = AdaptivePrototypeSearch().fit(
        features, labels, prototypes_per_unit=3, mode="farthest_l1"
    )

    np.testing.assert_array_equal(first.prototype_units_, second.prototype_units_)
    np.testing.assert_array_equal(first.centroids_, second.centroids_)
    np.testing.assert_array_equal(first.radii_, second.radii_)
    np.testing.assert_array_equal(
        first.training_prototype_assignments_, second.training_prototype_assignments_
    )
    np.testing.assert_array_equal(
        first.training_cluster_assignments_, first.training_prototype_assignments_
    )
    np.testing.assert_array_equal(first.train_prototype_rows_, first.training_prototype_assignments_)
    for sample_index, prototype_row in enumerate(first.training_prototype_assignments_):
        assert first.prototype_units_[prototype_row] == labels[sample_index]
    assert all(indices.size > 0 for indices in first.prototype_training_indices_)


def test_temporal_mode_partitions_each_unit_in_input_order():
    features = np.array([[0.0], [0.1], [0.8], [0.9]])
    search = AdaptivePrototypeSearch().fit(
        features, np.full(4, 55), prototypes_per_unit=2, mode="temporal"
    )
    np.testing.assert_array_equal(search.training_prototype_assignments_, [0, 0, 1, 1])
    np.testing.assert_array_equal(search.prototype_training_indices_[0], [0, 1])
    np.testing.assert_array_equal(search.prototype_training_indices_[1], [2, 3])


@pytest.mark.parametrize(
    ("features", "labels", "kwargs"),
    [
        (np.array([[0.0]]), np.array([7]), {"prototypes_per_unit": 0}),
        (np.array([[0.0]]), np.array([7]), {"prototypes_per_unit": 1, "mode": "unknown"}),
        (np.array([[0.0], [1.0]]), np.array([7]), {"prototypes_per_unit": 1}),
    ],
)
def test_invalid_fit_arguments_are_rejected(features, labels, kwargs):
    with pytest.raises(ValueError):
        AdaptivePrototypeSearch().fit(features, labels, **kwargs)


def test_invalid_query_arguments_and_prefit_calls_are_rejected():
    unfitted = AdaptivePrototypeSearch()
    with pytest.raises(RuntimeError):
        unfitted.query_codes([[0]])

    search = AdaptivePrototypeSearch().fit([[0.0], [1.0]], [7, 103])
    with pytest.raises(ValueError):
        search.query_codes([[0.5]])
    with pytest.raises(ValueError):
        search.query_codes([[0]], active_row_mask=np.array([1, 0]))
    with pytest.raises(ValueError):
        search.query_codes([[0]], active_row_mask=np.array([True, False, True]))


def test_quantizer_is_fixed_at_five_bits():
    with pytest.raises(ValueError):
        AdaptivePrototypeSearch(n_bits=4)
