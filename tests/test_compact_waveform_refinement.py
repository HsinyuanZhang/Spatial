"""No-I/O tests for frozen-candidate compact waveform refinement helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_compact_waveform_refinement import (
    fisher_tap_indices,
    fit_integer_templates,
    morphology_features,
    quantize_signed_uniform,
    refine_frozen_candidates,
    refinement_metrics,
    traffic_accounting,
)


def test_signed_quantizer_clips_boundaries_and_uses_deterministic_rounding():
    values = np.array([[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]])
    # Four-bit symmetric codes have scale 7; np.rint(3.5) rounds to 4.
    np.testing.assert_array_equal(quantize_signed_uniform(values, 4), [[-7, -7, -4, 0, 4, 7, 7]])


def test_fisher_taps_are_train_only_and_ties_break_by_sample_index():
    train = np.zeros((4, 64))
    train[:2, 3] = -1
    train[2:, 3] = 1
    labels = np.array([7, 7, 19, 19])
    selected = fisher_tap_indices(train, labels, 3)
    assert selected[0] == 3
    # All zero-score remaining taps rank by sample index, independent of any
    # hypothetical held-out waveform values.
    np.testing.assert_array_equal(selected[1:], [0, 1])


def test_compact_template_is_transform_then_centroid_round_clip():
    codes = np.array([[1, -7], [2, -6], [7, 7]])
    labels = np.array([10, 10, 99])
    templates = fit_integer_templates(codes, labels, np.array([10, 99]), -7, 7)
    np.testing.assert_array_equal(templates[10], [2, -6])
    np.testing.assert_array_equal(templates[99], [7, 7])


def test_morphology_mixed_ranges_map_to_shared_unit_interval():
    waveform = np.linspace(-1.0, 1.0, 64)[None, :]
    features = morphology_features(waveform)
    assert features.shape == (1, 12)
    assert np.all((features >= 0.0) & (features <= 1.0))
    np.testing.assert_allclose(features[0, :6], [0.0, 1.0, 1.0, 1.0, 0.0, 1.0])


def test_frozen_candidates_keep_unique_zero_and_ambiguous_read_semantics():
    candidates = [np.array([10]), np.array([10, 30]), np.array([], dtype=int)]
    test = np.array([[0, 1], [1, 0], [1, 0]])
    templates = {10: np.array([0, 1]), 30: np.array([1, 0])}
    pred, meta = refine_frozen_candidates(
        candidates, test, templates, np.array([[0], [31], [31]]), np.array([[0], [31]]), np.array([10, 30]), metric="sad"
    )
    np.testing.assert_array_equal(pred, [10, 30, 30])
    np.testing.assert_array_equal(meta["candidate_template_reads"], [0, 2, 0])
    np.testing.assert_array_equal(meta["spatial_fallback"], [False, False, True])
    # Refinement did not mutate or repair the frozen candidate source.
    np.testing.assert_array_equal(candidates[1], [10, 30])


def test_two_waveform_representations_receive_the_same_frozen_candidate_arrays():
    candidates = [np.array([5, 9]), np.array([5])]
    before_ids = [id(row) for row in candidates]
    spatial_codes = np.array([[0], [31]])
    spatial_centroids = np.array([[0], [31]])
    units = np.array([5, 9])
    first, _ = refine_frozen_candidates(
        candidates, np.array([[0.0], [0.0]]), {5: np.array([0.0]), 9: np.array([1.0])},
        spatial_codes, spatial_centroids, units,
    )
    second, _ = refine_frozen_candidates(
        candidates, np.array([[0], [0]]), {5: np.array([0]), 9: np.array([1])},
        spatial_codes, spatial_centroids, units, metric="sad",
    )
    assert [id(row) for row in candidates] == before_ids
    np.testing.assert_array_equal(candidates[0], [5, 9])
    np.testing.assert_array_equal(first, second)


def test_traffic_and_delta_are_relative_to_same_candidate_teacher():
    candidates = [np.array([1, 2]), np.array([1, 2])]
    labels = np.array([1, 2])
    teacher = np.array([1, 2])
    compact = np.array([1, 1])
    metrics = refinement_metrics(
        candidates, labels, compact, teacher, np.array([2, 2]),
        template_bits_per_unit=48, n_units=2, feature_dim=12, shared_tap_index_bits=16 * 6,
    )
    assert metrics["accuracy_delta_vs_float_teacher"] == -0.5
    assert metrics["float_teacher_prediction_agreement"] == 0.5
    assert metrics["candidate_template_reads"] == 4
    assert metrics["shared_tap_index_bits"] == 96
    assert metrics["total_stored_template_bits"] == 192
    assert "excludes one-time shared tap index" in metrics["actual_template_bits_read_note"]
    traffic = traffic_accounting(np.array([2, 0]), template_bits_per_unit=48, n_units=2)
    assert traffic["actual_template_bits_read"] == 96
    assert traffic["global_64x4bit_template_bits"] == 1024
    assert traffic["global_64x4bit_over_actual_bit_traffic_factor"] == 1024 / 96
