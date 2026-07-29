"""No-I/O tests for pair-conditioned tournament helpers."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_pair_conditioned_waveform import (
    nearest_float_teacher, pair_fisher_taps, pair_metrics, pair_traffic_accounting,
    stable_spatial_candidate_order, tournament_assign, unordered_pair_key,
)


def test_pair_taps_are_train_only_with_stable_sample_ties():
    waves = np.zeros((4, 64)); waves[:2, 7] = -1; waves[2:, 7] = 1
    taps = pair_fisher_taps(waves, np.array([3, 3, 99, 99]), 3, 99, 3)
    np.testing.assert_array_equal(taps, [7, 0, 1])
    assert unordered_pair_key(99, 3) == unordered_pair_key(3, 99) == (3, 99)


def test_full_float_tournament_equals_same_candidate_nearest_teacher():
    candidates = [np.array([9, 3, 7])]
    codes = np.array([[5]]); centers = np.array([[0], [4], [8]]); units = np.array([3, 7, 9])
    ordered = stable_spatial_candidate_order(candidates, codes, centers, units)
    np.testing.assert_array_equal(ordered[0], [7, 9, 3])
    templates = {3: np.array([0.0, 0.0]), 7: np.array([1.0, 1.0]), 9: np.array([2.0, 2.0])}
    q = np.array([[0.8, 1.1]])
    teacher = nearest_float_teacher(ordered, q, templates, codes, centers, units)
    pred, _ = tournament_assign(ordered, q, templates, codes, centers, units, mode="float64")
    np.testing.assert_array_equal(pred, teacher)
    np.testing.assert_array_equal(candidates[0], [9, 3, 7])


def test_unique_zero_and_three_candidate_comparison_counts():
    ordered = [np.array([1]), np.array([1, 2, 3]), np.array([], dtype=int)]
    templates = {1: np.array([0]), 2: np.array([2]), 3: np.array([3])}
    pred, meta = tournament_assign(ordered, np.array([[0], [2], [3]]), templates, np.array([[0], [2], [3]]), np.array([[0], [2], [3]]), np.array([1, 2, 3]), mode="signed_full64")
    np.testing.assert_array_equal(pred, [1, 2, 3])
    np.testing.assert_array_equal(meta["pair_comparisons"], [0, 2, 0])
    np.testing.assert_array_equal(meta["spatial_fallback"], [False, False, True])


def test_integer_sad_tie_keeps_spatially_earlier_champion_without_changing_set():
    # Candidate 2 is spatially nearer to code 3 than candidate 1 and is
    # therefore ordered first.  Both signed templates tie at SAD=1, so the
    # challenger must not replace that spatially earlier champion.
    candidates = [np.array([1, 2])]
    ordered = stable_spatial_candidate_order(
        candidates, np.array([[3]]), np.array([[0], [4]]), np.array([1, 2])
    )
    np.testing.assert_array_equal(ordered[0], [2, 1])
    pred, _ = tournament_assign(
        ordered, np.array([[1]]), {1: np.array([0]), 2: np.array([2])},
        np.array([[3]]), np.array([[0], [4]]), np.array([1, 2]), mode="signed_full64",
    )
    np.testing.assert_array_equal(pred, [2])
    np.testing.assert_array_equal(candidates[0], [1, 2])


def test_pair_traffic_storage_and_wrong_pair_decision_delta():
    traffic = pair_traffic_accounting(np.array([2, 0]), n_units=3, bits=5, n_taps=8)
    assert traffic["pair_template_code_bits_read"] == 160
    assert traffic["pair_tap_index_bits_read_compressed"] == 96
    assert traffic["pair_tap_mask64_bits_read"] == 128
    assert traffic["total_bits_read_compressed_index"] == 256
    candidates = [np.array([1, 2]), np.array([1, 2])]
    metrics = pair_metrics(candidates, np.array([1, 2]), np.array([1, 1]), np.array([1, 2]), np.array([1, 1]), 3, 5, 8, 3, 2)
    assert metrics["accuracy_delta_vs_float_teacher"] == -0.5
    assert metrics["pair_tap_index_storage_bits_compressed"] == 3 * 8 * 6
    assert metrics["total_storage_bits_compressed_index"] == 3 * 64 * 5 + 3 * 8 * 6


def test_pair_specific_tap_can_make_a_real_wrong_decision_and_reduce_accuracy():
    ordered = [np.array([1, 2])]
    query = np.zeros((1, 64)); query[0, 0] = 1.0
    first = np.zeros(64); first[0] = 1.0; first[1] = 1.0
    second = np.zeros(64)
    templates = {1: first, 2: second}
    codes = np.array([[0]]); centers = np.array([[0], [1]]); units = np.array([1, 2])
    teacher = nearest_float_teacher(ordered, query, templates, codes, centers, units)
    pair_pred, _ = tournament_assign(
        ordered, query, templates, codes, centers, units, mode="pair_fisher",
        pair_taps={(1, 2): np.array([1])},
    )
    np.testing.assert_array_equal(teacher, [1])  # full distances tie, stable first candidate wins
    np.testing.assert_array_equal(pair_pred, [2])
    metrics = pair_metrics(ordered, np.array([1]), pair_pred, teacher, np.array([1]), 2, 5, 1, 1, 1)
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["accuracy_delta_vs_float_teacher"] == -1.0
