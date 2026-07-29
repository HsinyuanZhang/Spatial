"""No-I/O tests for bit-exact alignment-conditioned temporal sketching."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.causal_temporal_sketch import (
    ALLOWED_DELAYS,
    CausalTemporalSketch,
    DYADIC_DICTIONARY,
    round_half_away_from_zero,
    signed_haz_divide,
    signed_power_of_two_scale,
    signed_rne_divide,
)


def _fit_zero_sketch(delay=48, n_features=8):
    waveforms = np.zeros((20, 64), dtype=float)
    return CausalTemporalSketch(delay_samples=delay, n_features=n_features).fit(
        waveforms, [7] * 10 + [103] * 10
    )


def test_full_dictionary_has_93_stable_ordered_ids_and_horizon_eligibility():
    assert len(DYADIC_DICTIONARY) == 93
    assert [entry.filter_id for entry in DYADIC_DICTIONARY] == list(range(93))
    assert all(entry.kind == "mean" for entry in DYADIC_DICTIONARY[:62])
    assert all(entry.kind == "haar" for entry in DYADIC_DICTIONARY[62:])
    assert [(entry.support, entry.start) for entry in DYADIC_DICTIONARY[:4]] == [
        (2, 0), (2, 2), (2, 4), (2, 6)
    ]
    assert DYADIC_DICTIONARY[-1].kind == "haar"
    assert (DYADIC_DICTIONARY[-1].support, DYADIC_DICTIONARY[-1].start) == (64, 0)

    early = _fit_zero_sketch(delay=0, n_features=8)
    assert np.all([DYADIC_DICTIONARY[i].final_index <= 15 for i in early.eligible_filter_ids_])
    assert 92 not in early.eligible_filter_ids_
    late = _fit_zero_sketch(delay=48, n_features=16)
    np.testing.assert_array_equal(late.eligible_filter_ids_, np.arange(93))


def test_signed_half_away_rounding_scale_and_mean_haar_encoding_are_exact():
    np.testing.assert_array_equal(round_half_away_from_zero([-1.5, -0.5, 0.5, 1.5]), [-2, -1, 1, 2])
    np.testing.assert_array_equal(signed_haz_divide([-6, -2, 2, 6], 4), [-2, -1, 1, 2])
    # Explicit quotient/remainder nearest-even template rounding, including
    # signed half ties: 1.5 -> 2, 2.5 -> 2, and their negative counterparts.
    np.testing.assert_array_equal(signed_rne_divide([3, 5, -3, -5], 2), [2, 2, -2, -2])
    scale, exponent, quantile = signed_power_of_two_scale(np.array([-0.75, 0.75]))
    assert quantile == 0.75 and scale == 1.0 and exponent == 0
    zero_scale, zero_exponent, _ = signed_power_of_two_scale(np.zeros(4))
    assert zero_scale == 1.0 and zero_exponent == 0

    sketch = _fit_zero_sketch(delay=48, n_features=8)
    samples = np.zeros((1, 64), dtype=np.int64)
    samples[0, :4] = [1, 2, 1, 1]
    # ID 0 is mean support 2: 3 / 2 -> HAZ 2.  ID 62 is Haar support
    # 4: (1+2)-(1+1)=1, then 1/4 -> HAZ 0.
    encoded = sketch._encode_filter_ids(samples, np.array([0, 62]))
    np.testing.assert_array_equal(encoded, [[2, 0]])
    samples[0, :4] = [2, 2, -2, -2]
    encoded = sketch._encode_filter_ids(samples, np.array([62]))
    np.testing.assert_array_equal(encoded, [[2]])


def test_signed_five_bit_quantizer_bounds_and_horizon_future_sample_isolation():
    train = np.zeros((20, 64), dtype=float)
    train[10:] = 0.25
    sketch = CausalTemporalSketch(delay_samples=0, n_features=8).fit(train, [7] * 10 + [103] * 10)
    waveform = np.zeros((1, 64), dtype=float)
    waveform[0, :16] = np.linspace(-1.0, 1.0, 16)
    changed = waveform.copy()
    changed[0, 16:] = np.linspace(-100.0, 100.0, 48)
    codes = sketch.quantize_prefix(waveform)
    assert codes.min() >= -15 and codes.max() <= 15
    np.testing.assert_array_equal(codes, sketch.quantize_prefix(changed))
    np.testing.assert_array_equal(sketch.transform(waveform), sketch.transform(changed))
    # Saturation uses the same frozen prefix/scale and therefore cannot change
    # when only later samples are perturbed.
    assert sketch.sample_saturation_fraction(waveform) == sketch.sample_saturation_fraction(changed)
    assert sketch.sample_saturation_fraction(np.full((1, 64), 100.0)) == 1.0
    pred, diag = sketch.assign_candidates(waveform, [np.array([7, 103])])
    changed_pred, changed_diag = sketch.assign_candidates(changed, [np.array([7, 103])])
    np.testing.assert_array_equal(pred, changed_pred)
    for key in diag:
        np.testing.assert_array_equal(diag[key], changed_diag[key])


def test_fit_only_fisher_selection_stable_ties_and_integer_template_round_even():
    # Every eligible feature has zero Fisher score; smallest stable IDs must be
    # selected.  All zero codes also make the stored integer templates exact.
    sketch = _fit_zero_sketch(delay=0, n_features=8)
    np.testing.assert_array_equal(sketch.eligible_fisher_scores_, np.zeros_like(sketch.eligible_fisher_scores_))
    np.testing.assert_array_equal(sketch.selected_filter_ids_, sketch.eligible_filter_ids_[:8])
    np.testing.assert_array_equal(sketch.templates_, np.zeros((2, 8), dtype=np.int64))
    assert sketch.templates_.dtype == np.int64
    np.testing.assert_array_equal(sketch.fit_unit_counts_, [10, 10])
    np.testing.assert_array_equal(sketch.fisher_eligible_unit_mask_, [True, True])
    np.testing.assert_array_equal(sketch.fisher_eligible_unit_ids_, [7, 103])
    assert np.isfinite(sketch.fisher_between_sum_squares_).all()
    assert np.isfinite(sketch.fisher_within_sum_squares_).all()
    assert np.isfinite(sketch.eligible_fisher_scores_).all()
    assert np.all(sketch.fisher_ranks_[sketch.eligible_filter_ids_] >= 0)
    # Inference waveform perturbations cannot alter fit-only selected IDs or templates.
    selected, templates = sketch.selected_filter_ids_.copy(), sketch.templates_.copy()
    sketch.transform(np.ones((1, 64)))
    np.testing.assert_array_equal(sketch.selected_filter_ids_, selected)
    np.testing.assert_array_equal(sketch.templates_, templates)


def test_candidate_sad_preserves_candidate_order_on_ties_and_handles_c0_c1():
    sketch = _fit_zero_sketch()
    waveforms = np.zeros((3, 64))
    candidates = [np.array([103, 7]), np.array([7]), np.array([], dtype=int)]
    prediction, diagnostics = sketch.assign_candidates(waveforms, candidates)
    assert prediction.tolist() == [103, 7, None]
    np.testing.assert_array_equal(diagnostics["candidate_count"], [2, 1, 0])
    np.testing.assert_array_equal(diagnostics["logical_candidate_template_reads"], [2, 0, 0])
    np.testing.assert_array_equal(diagnostics["logical_candidate_template_bits"], [2 * 40, 0, 0])
    np.testing.assert_array_equal(diagnostics["logical_sad_absolute_differences"], [16, 0, 0])
    np.testing.assert_array_equal(diagnostics["logical_sad_reduction_additions"], [14, 0, 0])
    np.testing.assert_array_equal(diagnostics["winning_sad_distance"], [0, -1, -1])


def test_noncontiguous_labels_same_horizon_float_ceiling_and_unseen_test_label_are_safe():
    waveforms = np.zeros((20, 64))
    waveforms[10:] = 0.5
    sketch = CausalTemporalSketch(delay_samples=8, n_features=8).fit(waveforms, [7] * 10 + [103] * 10)
    candidates = [np.array([103, 7]), np.array([7])]
    query = np.vstack([np.full(64, 0.5), np.zeros(64)])
    sad_prediction, _ = sketch.assign_candidates(query, candidates)
    float_prediction = sketch.same_horizon_float_assign(query, candidates)
    assert sad_prediction.tolist() == [103, 7]
    assert float_prediction.tolist() == [103, 7]
    # A test label is never an input to fit/transform; unseen evaluation labels
    # can be compared externally without allocating a template row.
    assert 999 not in sketch.units_


def test_direct_streaming_logical_accounting_and_exact_template_traffic_formula():
    sketch = _fit_zero_sketch(delay=48, n_features=12)
    accounting = sketch.logical_extraction_accounting(2)
    for values in accounting.values():
        assert values.shape == (2,)
    np.testing.assert_array_equal(
        accounting["logical_selected_feature_accumulator_updates"],
        accounting["logical_positive_coefficient_updates"] + accounting["logical_negative_coefficient_updates"],
    )
    assert accounting["logical_raw_prefix_samples_consumed"][0] == 64
    assert accounting["logical_normalization_shifts"][0] == 12

    traffic = sketch.template_traffic([np.array([7, 103]), np.array([7]), np.array([], dtype=int)])
    assert traffic["conditional_template_reads"] == 2
    assert traffic["actual_sketch_template_bits"] == 2 * 60
    assert traffic["global_full64x5_template_bits"] == 3 * 2 * 64 * 5
    assert traffic["meets_exact_4x_traffic_condition"]
    storage = sketch.memory_accounting()
    assert storage["template_bits_per_unit"] == 60
    assert storage["stored_template_bits"] == 120
    assert storage["selected_filter_id_bits"] == 84
    assert storage["dictionary_entries"] == 93
    assert storage["fit_sample_saturation_population"] == "fisher_eligible_fit_rows"
    assert storage["fit_quantizer_eligible_row_count"] == 20
    assert storage["fit_quantizer_source_sample_count"] == 20 * 64
    assert storage["fit_all_quantizer_row_count"] == 20
    assert storage["fit_all_quantizer_source_sample_count"] == 20 * 64
    assert sketch.saturation_accounting() == {
        key: storage[key]
        for key in (
            "fit_sample_saturation_fraction",
            "fit_sample_saturation_population",
            "fit_quantizer_eligible_row_count",
            "fit_quantizer_source_sample_count",
            "fit_all_sample_saturation_fraction",
            "fit_all_quantizer_row_count",
            "fit_all_quantizer_source_sample_count",
        )
    }


def test_rare_fit_units_keep_templates_but_do_not_configure_fisher_and_single_class_is_not_evaluable():
    waveforms = np.zeros((21, 64), dtype=float)
    waveforms[10:20] = 0.25
    waveforms[20] = -0.25
    sketch = CausalTemporalSketch(delay_samples=0, n_features=8).fit(
        waveforms, [7] * 10 + [103] * 10 + [999]
    )
    assert sketch.units_.tolist() == [7, 103, 999]
    np.testing.assert_array_equal(sketch.fit_unit_counts_, [10, 10, 1])
    np.testing.assert_array_equal(sketch.fisher_eligible_unit_mask_, [True, True, False])
    np.testing.assert_array_equal(sketch.fisher_eligible_unit_ids_, [7, 103])
    assert sketch.templates_.shape == (3, 8)
    with pytest.raises(ValueError):
        CausalTemporalSketch(delay_samples=0, n_features=8).fit(
            np.zeros((19, 64)), [7] * 10 + [103] * 9
        )


def test_rare_fit_row_perturbations_cannot_change_any_fisher_selection_state():
    base = np.zeros((21, 64), dtype=float)
    base[10:20, :16] = 0.25
    labels = [7] * 10 + [103] * 10 + [999]
    first = CausalTemporalSketch(delay_samples=0, n_features=8).fit(base, labels)
    perturbed = base.copy()
    perturbed[20] = np.linspace(-1_000_000.0, 1_000_000.0, 64)
    second = CausalTemporalSketch(delay_samples=0, n_features=8).fit(perturbed, labels)
    for first_value, second_value in zip(
        (
            first.scale_, first.fit_sample_saturation_fraction_,
            first.fit_sample_saturation_population_, first.fit_quantizer_eligible_row_count_,
            first.fit_quantizer_source_sample_count_, first.selected_filter_ids_, first.fisher_between_sum_squares_,
            first.fisher_within_sum_squares_, first.eligible_fisher_scores_, first.fisher_ranks_,
        ),
        (
            second.scale_, second.fit_sample_saturation_fraction_,
            second.fit_sample_saturation_population_, second.fit_quantizer_eligible_row_count_,
            second.fit_quantizer_source_sample_count_, second.selected_filter_ids_, second.fisher_between_sum_squares_,
            second.fisher_within_sum_squares_, second.eligible_fisher_scores_, second.fisher_ranks_,
        ),
    ):
        np.testing.assert_array_equal(first_value, second_value)
    assert first.fit_all_sample_saturation_fraction_ < second.fit_all_sample_saturation_fraction_


def test_signed_numerator_widths_reserved_minus16_and_max_sad_are_guarded():
    sketch = _fit_zero_sketch(delay=48, n_features=16)
    extrema = np.full((1, 64), 15, dtype=np.int64)
    # Largest mean numerator is 32*15=480, which is signed-10 safe.
    np.testing.assert_array_equal(sketch._encode_filter_ids(extrema, np.array([61])), [[15]])
    haar_extrema = extrema.copy()
    haar_extrema[0, 32:] = -15
    # Largest Haar numerator is 32*15 - 32*(-15)=960, signed-11 safe.
    np.testing.assert_array_equal(sketch._encode_filter_ids(haar_extrema, np.array([92])), [[15]])
    reserved = extrema.copy()
    reserved[0, 0] = -16
    with pytest.raises(ValueError):
        sketch._encode_filter_ids(reserved, np.array([0]))

    # The first 16 stable tie-selected filters are means.  A +15 feature row
    # against an externally supplied valid -15 template reaches exact max SAD.
    sketch.templates_[:] = -15
    prediction, diagnostics = sketch.assign_candidates(np.ones((1, 64)), [np.array([7, 103])])
    assert prediction[0] == 7
    assert diagnostics["winning_sad_distance"][0] == 30 * 16 == 480


@pytest.mark.parametrize(
    "operation",
    [
        lambda: CausalTemporalSketch(delay_samples=1, n_features=8),
        lambda: CausalTemporalSketch(delay_samples=0, n_features=9),
        lambda: CausalTemporalSketch(delay_samples=0, n_features=8).fit(np.zeros((2, 63)), [7, 7]),
        lambda: CausalTemporalSketch(delay_samples=0, n_features=8).fit(np.zeros((2, 64)), [7]),
        lambda: signed_power_of_two_scale([np.nan]),
    ],
)
def test_invalid_ranges_and_inputs_are_rejected(operation):
    with pytest.raises(ValueError):
        operation()


def test_unknown_ambiguous_candidate_label_is_rejected_and_empty_accounting_is_valid():
    sketch = _fit_zero_sketch()
    with pytest.raises(ValueError):
        sketch.assign_candidates(np.zeros((1, 64)), [np.array([7, 999])])
    empty = sketch.logical_extraction_accounting(0)
    assert all(values.shape == (0,) for values in empty.values())
    assert 0 in ALLOWED_DELAYS
