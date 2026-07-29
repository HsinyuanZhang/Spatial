"""No-I/O tests for causal graph-only online updates."""

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.experiments.run_indexed_confusion_candidate_search import stable_anchor_union_candidates, summarize_indexed_candidates
from Spatial.experiments.run_online_confusion_graph import (
    block_event_slices, graph_from_observations, pointer_delta, post_update_graph_transition,
    update_observations,
)


def test_block_boundaries_keep_identical_timestamps_together():
    blocks = block_event_slices(np.array([1, 1, 999, 1000, 1000, 2001]), 1000)
    [np.testing.assert_array_equal(block, expected) for block, expected in zip(blocks, ([0, 1, 2], [3, 4], [5]))]


def test_block_label_update_is_causal_and_only_changes_next_block_graph():
    units = np.array([10, 20])
    history = deque([(0, 10, 0)])
    before, _ = graph_from_observations(history, units, 1.0)
    current = stable_anchor_union_candidates(np.array([[0, 1]]), before, 1)
    # Current block's true 20 is absent before its delayed update; no repair occurs.
    assert evaluate_candidate_search(current, np.array([20]))["true_unit_candidate_recall"] == 0.0
    update_observations(history, np.array([0]), np.array([20]), np.array([5]), units, mode="accumulate", block_end_time=5, horizon_samples=0)
    after, _ = graph_from_observations(history, units, 1.0)
    np.testing.assert_array_equal(after[0], [10, 20])


def test_rolling_history_expires_by_event_time():
    units = np.array([10, 20])
    history = deque([(0, 10, 0), (0, 20, 10)])
    unknown, removed = update_observations(history, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), units, mode="rolling", block_end_time=20, horizon_samples=10)
    assert unknown == 0 and removed == 1
    assert list(history) == [(0, 20, 10)]


def test_oracle_and_self_update_sources_produce_distinct_graph_observations():
    units = np.array([10, 20])
    oracle = deque([(0, 10, 0)])
    self_history = deque([(0, 10, 0)])
    update_observations(oracle, np.array([0]), np.array([20]), np.array([5]), units, mode="accumulate", block_end_time=5, horizon_samples=0)
    update_observations(self_history, np.array([0]), np.array([10]), np.array([5]), units, mode="accumulate", block_end_time=5, horizon_samples=0)
    oracle_graph, _ = graph_from_observations(oracle, units, 1.0)
    self_graph, _ = graph_from_observations(self_history, units, 1.0)
    np.testing.assert_array_equal(oracle_graph[0], [10, 20])
    np.testing.assert_array_equal(self_graph[0], [10])


def test_pointer_add_remove_accounting_and_joint_metric_field():
    delta = pointer_delta([np.array([10]), np.array([20])], [np.array([10, 20]), np.array([20])], 4)
    assert delta["pointer_additions"] == 1 and delta["pointer_removals"] == 0
    assert delta["pointer_bit_rewrites"] == 4  # membership-only compatibility lower bound
    assert delta["pointer_position_content_rewrites"] == 1
    assert delta["pointer_count_field_metadata_rewrites"] == 1
    delta2 = pointer_delta([np.array([10, 20])], [np.array([10])], 4)
    assert delta2["pointer_additions"] == 0 and delta2["pointer_removals"] == 1
    assert delta2["pointer_bit_rewrites"] == 4
    candidates = [np.array([10])]
    diagnostics = {"row_comparisons": np.array([2]), "active_row_count": np.array([2])}
    summary = summarize_indexed_candidates(candidates, np.array([10]), diagnostics, np.array([10]), {"candidate_waveform_reads": np.array([0]), "waveform_fallback": np.array([False])}, n_units=2)
    assert summary["meets_recall_0p99_and_waveform_reduction_4x"]


def test_pointer_reorder_rewrites_contiguous_positions_even_without_membership_delta():
    delta = pointer_delta([np.array([10, 20])], [np.array([20, 10])], pointer_bits=4, count_field_bits=2)
    assert delta["pointer_additions"] == 0
    assert delta["pointer_removals"] == 0
    assert delta["pointer_bit_rewrites"] == 0  # legacy membership lower bound
    assert delta["pointer_position_content_rewrites"] == 2
    assert delta["pointer_position_content_bit_rewrites"] == 8
    assert delta["pointer_count_field_metadata_rewrites"] == 0
    assert delta["pointer_ordered_list_total_bit_rewrites_including_count_metadata"] == 8


def test_pointer_count_change_rewrites_count_metadata_separately():
    delta = pointer_delta([np.array([10])], [np.array([10, 20])], pointer_bits=4, count_field_bits=2)
    assert delta["pointer_position_content_rewrites"] == 1
    assert delta["pointer_count_field_metadata_rewrites"] == 1
    assert delta["pointer_count_field_metadata_bit_rewrites"] == 2
    assert delta["pointer_ordered_list_total_bit_rewrites_including_count_metadata"] == 6


def test_last_block_post_update_graph_transition_is_materialized_and_accounted():
    units = np.array([10, 20])
    history = deque([(0, 10, 0)])
    before, _ = graph_from_observations(history, units, 1.0)
    update_observations(history, np.array([0]), np.array([20]), np.array([5]), units, mode="accumulate", block_end_time=5, horizon_samples=0)
    final, info, delta = post_update_graph_transition(before, history, units, 1.0)
    np.testing.assert_array_equal(final[0], [10, 20])
    assert info["graph_total_pointers"] == 3  # [10,20] plus mandatory anchor row [20]
    assert delta["pointer_additions"] == 1
    assert delta["pointer_count_field_metadata_rewrites"] == 1
