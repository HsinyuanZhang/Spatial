"""No-I/O semantic tests for bounded multi-prototype experiment helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_adaptive_multiprototype_5bit import (
    SPATIAL_PAYLOAD_BITS,
    deduplicate_prototype_hits,
    directory_true_unit_row_activation_recall,
    fit_prototype_channel_support,
    prototype_directory_mask,
    prototype_row_storage_bits,
    prototype_pointer_accounting,
    recall_and_gate,
    waveform_traffic,
    level1_comparison_accounting,
    validate_hardware_contract,
)
from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split
from Spatial.experiments.run_adaptive_5bit_range_search import candidate_waveform_assign
from Spatial.algorithms.adaptive_prototype_search import AdaptivePrototypeSearch
from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch


def test_prototype_support_is_minimal_and_channel_ties_are_stable():
    assignment = np.array([0, 0, 0, 0, 0, 1, 1])
    central = np.array([3, 3, 3, 1, 1, 2, 2])
    support = fit_prototype_channel_support(assignment, central, 3, 5, 0.6)
    np.testing.assert_array_equal(support[0], [False, False, False, True, False])
    np.testing.assert_array_equal(support[1], [False, False, True, False, False])
    assert not support[2].any()


def test_chronological_split_does_not_divide_timestamp_tie():
    train, test = _chronological_split(np.array([1, 2, 2, 2, 3, 4]))
    np.testing.assert_array_equal(train, [0, 1, 2, 3])
    np.testing.assert_array_equal(test, [4, 5])


def test_prototype_directory_and_unit_dedup_preserve_frozen_candidate_set():
    geom = np.column_stack([np.arange(5), np.zeros(5)])
    support = np.array([[False, True, False, False, False], [False, True, True, False, False], [False, False, False, True, False]])
    active = prototype_directory_mask(np.array([1]), support, geom, 1)
    np.testing.assert_array_equal(active, [[True, True, False]])
    hits = [np.array([0, 1, 0])]
    candidates, removed = deduplicate_prototype_hits(hits, np.array([10, 10, 99]))
    np.testing.assert_array_equal(candidates[0], [10])
    np.testing.assert_array_equal(removed, [2])
    np.testing.assert_array_equal(hits[0], [0, 1, 0])
    with np.testing.assert_raises(ValueError):
        deduplicate_prototype_hits([np.array([3])], np.array([10, 10, 99]))


def test_directory_true_unit_activation_recall_is_separate_from_range_misses():
    prototype_units = np.array([7, 7, 19])
    active = np.array([[True, False, False], [False, False, True], [False, False, False]])
    # Unit 7 has an active row on event 0; 19 on event 1; unseen 103 has none.
    assert directory_true_unit_row_activation_recall(active, np.array([7, 19, 103]), prototype_units) == 2 / 3
    assert directory_true_unit_row_activation_recall(None, np.array([7, 19, 103]), prototype_units) == 2 / 3


def test_storage_traffic_gate_and_p1_toy_invariant():
    storage = prototype_row_storage_bits(4)
    assert storage["prototype_row_payload_bits"] == SPATIAL_PAYLOAD_BITS == 54
    assert storage["prototype_row_total_bits"] == 57  # 54 + 2-bit unit ID + valid
    pointers = prototype_pointer_accounting(np.array([1, 2, 0, 1]), n_prototype_rows=4)
    assert pointers["prototype_directory_pointers_total"] == 4
    assert pointers["prototype_directory_pointer_row_id_bits"] == 2
    assert pointers["prototype_directory_total_pointer_bits"] == 8
    single = [np.array([7]), np.array([9])]
    traffic = waveform_traffic(single, n_units=2)
    assert traffic["candidate_waveform_reads_total"] == 0
    assert traffic["global_all_unit_waveform_template_bits"] == 2 * 2 * 64 * 5
    ambiguous_traffic = waveform_traffic([np.array([7, 9])], n_units=2)
    assert ambiguous_traffic["candidate_waveform_reads_total"] == 2
    assert ambiguous_traffic["actual_waveform_template_bits_read"] == 2 * 64 * 5
    report = recall_and_gate(single, np.array([7, 9]), n_units=2)
    assert report["true_unit_candidate_recall"] == 1.0
    assert report["meets_recall_0p99_and_waveform_reduction_4x"]
    # One prototype per unit has no duplicate prototype hit to alter its unit
    # candidate rows, the toy P=1 equivalence condition.
    dedup, removed = deduplicate_prototype_hits([np.array([0, 1])], np.array([7, 9]))
    np.testing.assert_array_equal(dedup[0], [7, 9]); np.testing.assert_array_equal(removed, [0])


def test_total_level1_accounting_adds_actual_zero_candidate_fallback_metadata():
    candidates = [np.array([], dtype=int), np.array([7])]
    _, meta = candidate_waveform_assign(
        candidates, np.zeros((2, 64)), {7: np.zeros(64), 9: np.zeros(64)},
        np.array([[0], [31]]), np.array([[0], [31], [16]]), np.array([7, 9, 11]), metric="l1",
    )
    np.testing.assert_array_equal(meta["spatial_fallback_row_comparisons"], [3, 0])
    accounting = level1_comparison_accounting(np.array([4, 4]), meta["spatial_fallback_row_comparisons"])
    assert accounting["primary_prototype_row_comparisons_total"] == 8
    assert accounting["spatial_fallback_level1_row_comparisons_total"] == 3
    assert accounting["total_level1_row_comparisons"] == 11
    assert accounting["max_total_level1_row_comparisons"] == 7


def test_frozen_hardware_contract_rejects_non_d9_or_unbounded_p():
    with np.testing.assert_raises(ValueError):
        validate_hardware_contract(5, 8, [1])
    with np.testing.assert_raises(ValueError):
        validate_hardware_contract(5, 9, [5])


def test_p1_all_row_is_identical_to_single_per_unit_l1_on_toy_codes():
    features = np.array([[0.0, 0.1], [0.1, 0.0], [0.9, 1.0], [1.0, 0.9]])
    labels = np.array([7, 7, 103, 103])
    baseline = AdaptiveRangeSearch(n_bits=5, fixed_unit_interval=True, radius_percentile=95.0).fit(features, labels)
    prototype = AdaptivePrototypeSearch(n_bits=5, radius_percentile=95.0).fit(
        features, labels, prototypes_per_unit=1, mode="temporal"
    )
    codes = baseline.transform(np.array([[0.05, 0.05], [0.95, 0.95]]))
    expected, _ = baseline.query_codes(codes, mode="l1")
    actual, diagnostics = prototype.query_codes(codes)
    for want, got in zip(expected, actual):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(diagnostics["duplicates_removed"], [0, 0])
