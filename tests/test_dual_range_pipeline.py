"""No-I/O tests for the all-row dual box/L1 policy."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.experiments.run_dual_range_pipeline import (
    apply_dual_range_policy,
    ordered_unique_union,
    payload_row_bits,
    summarize_dual_range_configuration,
)


def _inner_diag(rows):
    n = len(rows)
    return {
        "row_comparisons": np.asarray(rows, dtype=np.int64),
        "active_row_count": np.full(n, 3, dtype=np.int64),
    }


def _outer_diag(rows):
    n = len(rows)
    return {
        "row_comparisons": np.asarray(rows, dtype=np.int64),
        "active_row_count": np.full(n, 3, dtype=np.int64),
    }


def test_inner_unique_is_a_true_early_exit_without_outer_access():
    candidates, diag = apply_dual_range_policy(
        [np.array([19])], _inner_diag([3]), [], _outer_diag([])
    )
    np.testing.assert_array_equal(candidates[0], [19])
    np.testing.assert_array_equal(diag["outer_trigger"], [False])
    np.testing.assert_array_equal(diag["outer_row_comparisons"], [0])
    np.testing.assert_array_equal(diag["row_comparisons"], [3])
    np.testing.assert_array_equal(diag["inner_active_row_count"], [3])
    np.testing.assert_array_equal(diag["outer_active_row_count"], [0])
    np.testing.assert_array_equal(diag["total_active_row_accesses"], [3])
    np.testing.assert_array_equal(diag["fallback_row_comparisons"], [0])
    np.testing.assert_array_equal(diag["total_row_comparisons_including_fallback"], [3])


def test_zero_and_ambiguous_inner_results_trigger_outer_and_keep_inner_misses():
    candidates, diag = apply_dual_range_policy(
        [np.array([], dtype=int), np.array([2, 7]), np.array([11, 13])], _inner_diag([3, 3, 3]),
        [np.array([5]), np.array([7, 9]), np.array([], dtype=int)], _outer_diag([3, 3, 3]),
    )
    np.testing.assert_array_equal(candidates[0], [5])
    # Outer order is retained, and the inner-only unit 2 is appended instead
    # of being silently lost because outer L1 did not return it.
    np.testing.assert_array_equal(candidates[1], [7, 9, 2])
    # An empty outer result must not erase a non-empty ambiguous inner result.
    np.testing.assert_array_equal(candidates[2], [11, 13])
    np.testing.assert_array_equal(diag["outer_trigger"], [True, True, True])
    np.testing.assert_array_equal(diag["outer_row_comparisons"], [3, 3, 3])
    np.testing.assert_array_equal(diag["row_comparisons"], [6, 6, 6])
    # Simultaneous inner and outer masks remain distinct; total accesses charge
    # both ranges only for events that actually triggered outer search.
    np.testing.assert_array_equal(diag["inner_active_row_count"], [3, 3, 3])
    np.testing.assert_array_equal(diag["outer_active_row_count"], [3, 3, 3])
    np.testing.assert_array_equal(diag["total_active_row_accesses"], [6, 6, 6])


def test_ordered_union_is_stable_and_de_duplicates_both_inputs():
    np.testing.assert_array_equal(
        ordered_unique_union(np.array([7, 3, 7]), np.array([3, 19, 7, 23])),
        [7, 3, 19, 23],
    )


def test_wrong_inner_unique_lowers_recall_without_gt_repair_or_outer_query():
    candidates, diag = apply_dual_range_policy(
        [np.array([7])], _inner_diag([4]), [], _outer_diag([])
    )
    report = evaluate_candidate_search(candidates, np.array([19]), diag)
    assert report["true_unit_candidate_recall"] == 0.0
    assert not diag["outer_trigger"][0]
    # A hypothetical outer [19] never reaches the policy for a singleton; no
    # GT/top-k mechanism is allowed to turn this into a hit.
    np.testing.assert_array_equal(candidates[0], [7])


def test_summary_exposes_dual_accounting_fields_and_fixed_payload_bits():
    candidates = [np.array([4]), np.array([4, 9]), np.array([], dtype=int)]
    labels = np.array([4, 9, 9])
    diagnostics = {
        "row_comparisons": np.array([3, 6, 6]),
        "active_row_count": np.array([3, 3, 3]),
        "inner_row_comparisons": np.array([3, 3, 3]),
        "outer_row_comparisons": np.array([0, 3, 3]),
        "inner_active_row_count": np.array([3, 3, 3]),
        "outer_active_row_count": np.array([0, 3, 3]),
        "total_active_row_accesses": np.array([3, 6, 6]),
        "outer_trigger": np.array([False, True, True]),
        "widen_level": np.zeros(3, dtype=int),
        "fallback": np.zeros(3, dtype=bool),
    }
    metrics = summarize_dual_range_configuration(
        candidates, labels, diagnostics, np.array([4, 9, 4]),
        {
            "candidate_waveform_reads": np.array([0, 2, 0]),
            "waveform_fallback": np.array([False, False, True]),
            "spatial_fallback_row_comparisons": np.array([0, 0, 3]),
        },
        inner_candidates=[np.array([4]), np.array([4, 9]), np.array([], dtype=int)],
        outer_trigger=np.array([False, True, True]), n_units=3,
    )
    for field in (
        "inner_unique_fraction", "inner_unique_precision", "outer_trigger_fraction",
        "final_true_unit_candidate_recall", "zero_fraction", "unique_fraction",
        "ambiguous_fraction", "mean_candidates", "p95_candidates", "stage2_activation",
        "mean_inner_row_comparisons", "mean_outer_row_comparisons",
        "total_level1_row_comparisons", "total_active_row_accesses",
        "fallback_row_comparisons", "total_row_comparisons_including_fallback", "candidate_waveform_reads_per_event",
        "global_waveform_reduction_factor", "final_waveform_accuracy",
    ):
        assert field in metrics
    assert metrics["inner_unique_fraction"] == 1 / 3
    assert metrics["inner_unique_precision"] == 1.0
    assert metrics["outer_trigger_fraction"] == 2 / 3
    assert metrics["mean_inner_row_comparisons"] == 3.0
    assert metrics["mean_outer_row_comparisons"] == 2.0
    assert metrics["total_level1_row_comparisons"] == 15
    assert metrics["total_active_row_accesses"] == 15
    assert metrics["spatial_fallback_row_comparisons"] == 3
    assert metrics["fallback_row_comparisons"] == 3
    assert metrics["total_row_comparisons_including_fallback"] == 18
    assert metrics["global_all_unit_waveform_reads"] == 9
    assert metrics["global_waveform_reduction_factor"] == 4.5
    assert metrics["final_waveform_accuracy"] == 2 / 3
    assert payload_row_bits(9, 5) == (90, 54, 144)
