"""No-I/O tests for adaptive 5-bit benchmark helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_adaptive_5bit_range_search import (
    _chronological_split,
    build_directory_mask,
    build_support_directory_mask,
    candidate_waveform_assign,
    fit_unit_channel_support,
    labels_for_peak_output,
    pooled_l1_candidates,
    summarize_configuration,
)


def test_peak_label_alignment_keeps_duplicate_timestamp_rows():
    times = np.array([2, 10, 10, 98])
    labels = np.array([1, 7, 103, 9])
    got = labels_for_peak_output(times, labels, np.array([10, 10]), 100, window=5)
    np.testing.assert_array_equal(got, [7, 103])


def test_chronological_split_keeps_equal_timestamps_on_one_side():
    train, test = _chronological_split(np.array([1, 2, 2, 2, 3, 4, 4, 5]))
    np.testing.assert_array_equal(train, [0, 1, 2, 3])
    np.testing.assert_array_equal(test, [4, 5, 6, 7])


def test_directory_mask_uses_home_channels_not_unit_id_contiguity():
    geom = np.column_stack([np.arange(7), np.zeros(7)])
    units = np.array([3, 19, 101])
    homes = np.array([0, 3, 6])
    mask = build_directory_mask(np.array([0, 6]), units, homes, geom, directory_neighbors=2)
    np.testing.assert_array_equal(mask, [[True, False, False], [False, False, True]])
    all_rows = build_directory_mask(np.array([0]), units, homes, geom, directory_neighbors=0)
    assert all_rows.all()


def test_noncontiguous_units_get_minimal_frequency_ordered_channel_support():
    labels = np.array([17, 17, 17, 17, 17, 101, 101, 101, 101])
    central = np.array([4, 4, 4, 2, 2, 1, 1, 0, 0])
    units = np.array([17, 101])
    support_60 = fit_unit_channel_support(labels, central, units, n_channels=6, coverage=0.60)
    # Unit 17 needs its 3/5 channel only; unit 101 needs two tied 2/4
    # channels, selecting channel 0 before channel 1 on the tie.
    np.testing.assert_array_equal(support_60, [
        [False, False, False, False, True, False],
        [True, True, False, False, False, False],
    ])
    support_80 = fit_unit_channel_support(labels, central, units, n_channels=6, coverage=0.80)
    np.testing.assert_array_equal(support_80[0], [False, False, True, False, True, False])
    with np.testing.assert_raises(ValueError):
        fit_unit_channel_support(labels, central, units, n_channels=6, coverage=0.0)


def test_support_directory_activates_once_when_any_support_channel_hits_knn():
    geom = np.column_stack([np.arange(6), np.zeros(6)])
    support = np.array([
        [False, True, True, False, False, False],  # two pointers in query KNN
        [False, False, False, False, True, False],
        [False, False, False, True, False, False],
    ])
    mask = build_support_directory_mask(np.array([2]), support, geom, directory_neighbors=3)
    # KNN(2) is {2, 1, 3}; first row remains a single bool activation even
    # though it intersects through two support channels.
    np.testing.assert_array_equal(mask, [[True, False, True]])
    assert mask.dtype == bool and mask.shape == (1, 3)


def test_pooled_l1_candidates_include_exact_radius_boundary():
    candidates, diag = pooled_l1_candidates(
        np.array([[3, 2]]), np.array([[0, 0], [3, 3]]), np.array([7, 42]), 5,
        np.array([[True, True]]), inclusive=True,
    )
    np.testing.assert_array_equal(candidates[0], [7, 42])
    assert diag["final_candidate_count"][0] == 2


def test_candidate_waveform_assignment_reads_only_candidates_and_spatial_falls_back():
    candidates = [np.array([10]), np.array([10, 30]), np.array([], dtype=int)]
    waveforms = np.array([[0.0, 1.0], [0.9, 0.1], [0.9, 0.1]])
    wave_centroids = {10: np.array([0.0, 1.0]), 30: np.array([1.0, 0.0])}
    pred, meta = candidate_waveform_assign(
        candidates, waveforms, wave_centroids, np.array([[0, 0], [30, 0], [30, 0]]),
        np.array([[0, 0], [31, 0]]), np.array([10, 30]), metric="l1",
    )
    np.testing.assert_array_equal(pred, [10, 30, 30])
    # |C|=1 is an early exit.  The ambiguous row reads precisely its two
    # candidates; the empty row uses spatial fallback and reads none.
    np.testing.assert_array_equal(meta["candidate_waveform_reads"], [0, 2, 0])
    np.testing.assert_array_equal(meta["waveform_fallback"], [False, False, True])
    # Empty candidates perform a separate all-unit spatial-centroid fallback,
    # not a hidden waveform-template read.
    np.testing.assert_array_equal(meta["spatial_fallback_row_comparisons"], [0, 0, 2])


def test_summary_exposes_candidate_directory_and_waveform_fields():
    candidates = [np.array([4]), np.array([4, 9])]
    labels = np.array([4, 9])
    diag = {
        "widen_level": np.array([0, 1]), "fallback": np.array([False, False]),
        "active_row_count": np.array([2, 2]), "row_comparisons": np.array([2, 4]),
    }
    summary = summarize_configuration(
        candidates, labels, diag, np.ones((2, 2), dtype=bool), np.array([4, 9]),
        np.array([4, 9]), {
            "candidate_waveform_reads": np.array([1, 2]),
            "waveform_fallback": np.array([False, False]),
            "spatial_fallback_row_comparisons": np.array([0, 0]),
        },
        directory_entries_per_unit=np.array([1, 3]),
    )
    for key in (
        "true_unit_candidate_recall", "directory_true_unit_recall", "p95_active_rows",
        "row_activation_reduction", "final_waveform_accuracy", "unique_hit_accuracy",
        "candidate_waveform_reads_per_event", "waveform_read_reduction_vs_global",
        "candidate_waveform_read_fraction_vs_global", "waveform_read_reduction_factor_global_over_actual",
        "fallback_row_comparisons", "total_row_comparisons_including_fallback",
        "mean_directory_entries_per_unit", "max_directory_entries_per_unit", "total_directory_entries",
    ):
        assert key in summary
    assert summary["final_waveform_accuracy"] == 1.0
    assert summary["mean_directory_entries_per_unit"] == 2.0
    assert summary["max_directory_entries_per_unit"] == 3
    assert summary["total_directory_entries"] == 4
