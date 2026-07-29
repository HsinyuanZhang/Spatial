"""No-I/O tests for deterministic indexed calibration-confusion retrieval."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.experiments.run_certified_dual_range_pipeline import chronological_fraction_split
from Spatial.experiments.run_indexed_confusion_candidate_search import (
    build_anchor_graph,
    fixed_top_m_candidates,
    graph_storage_bits,
    rank_unit_rows,
    stable_anchor_union_candidates,
    summarize_indexed_candidates,
)


def test_ranking_ties_stably_follow_fit_unit_row_order():
    ranks = rank_unit_rows(
        np.array([[1]]), np.array([[0], [2], [6]]), np.array([1, 1, 1]), mode="raw_l1"
    )
    np.testing.assert_array_equal(ranks, [[0, 1, 2]])
    np.testing.assert_array_equal(fixed_top_m_candidates(ranks, np.array([10, 20, 30]), 8)[0], [10, 20, 30])


def test_fit_cal_split_does_not_divide_equal_timestamps():
    times = np.array([1, 2, 2, 3, 4, 4, 5])
    fit, calibration = chronological_fraction_split(times, 0.5)
    np.testing.assert_array_equal(fit, [0, 1, 2])
    np.testing.assert_array_equal(calibration, [3, 4, 5, 6])
    assert times[fit[-1]] < times[calibration[0]]


def test_graph_counts_prediction_anchor_not_true_unit_anchor():
    units = np.array([10, 20])
    graph, diagnostics = build_anchor_graph(
        np.array([[0, 1], [0, 1], [1, 0]]), np.array([20, 20, 10]), units, coverage=1.0
    )
    # The two true-20 events were predicted by anchor 10, so their pointer
    # count belongs to anchor row 0 rather than anchor row 1.
    np.testing.assert_array_equal(diagnostics["counts"], [[0, 2], [1, 0]])
    np.testing.assert_array_equal(graph[0], [20, 10])  # count order + forced anchor


def test_graph_coverage_uses_minimal_list_stable_ties_and_includes_anchor():
    units = np.array([10, 20])
    rank = np.tile(np.array([[0, 1]]), (4, 1))
    graph_half, _ = build_anchor_graph(rank, np.array([10, 20, 10, 20]), units, coverage=0.5)
    # Tied 2/4 counts choose fit row 0 first; unseen anchor row 1 falls back to itself.
    np.testing.assert_array_equal(graph_half[0], [10])
    np.testing.assert_array_equal(graph_half[1], [20])
    graph_high, _ = build_anchor_graph(rank, np.array([10, 20, 10, 20]), units, coverage=0.75)
    np.testing.assert_array_equal(graph_high[0], [10, 20])


def test_unseen_calibration_labels_are_counted_not_materialized_as_rows():
    graph, diagnostics = build_anchor_graph(
        np.array([[0, 1], [0, 1]]), np.array([10, 99]), np.array([10, 20]), coverage=1.0
    )
    assert diagnostics["unseen_calibration_label_count"] == 1
    assert diagnostics["unseen_calibration_label_rate"] == 0.5
    np.testing.assert_array_equal(graph[0], [10])


def test_top_two_anchor_union_is_stable_and_no_test_label_repairs_a_miss():
    graph = [np.array([10, 20]), np.array([20, 30])]
    candidates = stable_anchor_union_candidates(np.array([[0, 1]]), graph, 2)
    np.testing.assert_array_equal(candidates[0], [10, 20, 30])
    miss = stable_anchor_union_candidates(np.array([[0, 1]]), [np.array([10]), np.array([20])], 1)
    report = evaluate_candidate_search(miss, np.array([20]))
    assert report["true_unit_candidate_recall"] == 0.0


def test_pointer_accounting_and_candidate_waveform_read_summary():
    storage = graph_storage_bits([np.array([i]) for i in range(10)], 10)
    assert storage["graph_total_pointers"] == 10
    assert storage["graph_pointer_id_bits"] == 4
    assert storage["graph_count_field_bits"] == 4
    assert storage["graph_total_bits"] == 80
    candidates = [np.array([1]), np.array([1, 2]), np.array([], dtype=int)]
    labels = np.array([1, 2, 1])
    diagnostics = {
        "row_comparisons": np.array([3, 3, 3]), "active_row_count": np.array([3, 3, 3]),
        "widen_level": np.zeros(3, dtype=int), "fallback": np.zeros(3, dtype=bool),
    }
    summary = summarize_indexed_candidates(
        candidates, labels, diagnostics, np.array([1, 2, 2]),
        {"candidate_waveform_reads": np.array([0, 2, 0]), "waveform_fallback": np.array([False, False, True])}, n_units=3,
    )
    assert summary["candidate_waveform_reads_per_event"] == 2 / 3
    assert summary["all_row_l1_comparisons_per_query"] == 3.0
    assert summary["total_row_comparisons"] == 9
    assert summary["unique_fraction"] == 1 / 3
    assert summary["ambiguous_fraction"] == 1 / 3
