"""No-I/O tests for mandatory full-window causal temporal baselines."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.causal_temporal_baselines import (
    LEGACY_PEAK_NORMALIZATION_EPSILON,
    CausalTemporalBaselines,
    REPRESENTATIONS,
    legacy_peak_normalize,
)


def _fit_zero_baselines():
    return CausalTemporalBaselines().fit(np.zeros((20, 64)), [7] * 10 + [103] * 10)


def test_fit_scale_full64_teacher_and_signed_baselines_have_fixed_ranges_payloads_and_contract_hashes():
    train = np.zeros((20, 64))
    train[10:] = 0.5
    model = CausalTemporalBaselines().fit(train, [7] * 10 + [103] * 10)
    signed = model.transform(np.full((1, 64), 99.0), "signed_full64x5")
    assert signed.min() == signed.max() == 15
    morphology = model.transform(np.full((1, 64), -99.0), "morphology12_unsigned5")
    assert morphology.shape == (1, 12)
    assert np.all((morphology >= 0) & (morphology <= 31))
    assert model.signed_full_templates_.dtype == np.int64
    assert model.fisher48_templates_.shape == (2, 48)
    assert model.morphology12_templates_.shape == (2, 12)

    full = model.payload_accounting("signed_full64x5")
    fisher = model.payload_accounting("fisher48_signed5")
    morph = model.payload_accounting("morphology12_unsigned5")
    assert full["template_bits_per_unit"] == 64 * 5
    assert fisher["template_bits_per_unit"] == 48 * 5
    assert fisher["shared_configuration_bits"] == 48 * 6
    assert morph["template_bits_per_unit"] == 12 * 5
    assert model.payload_accounting("fit_scale_float64")["logical_encoded_template_bits"] == 0
    assert model.payload_accounting("fit_scale_float64")["report_only_float_representation"] == "true"
    for representation in REPRESENTATIONS:
        contract = model.payload_accounting(representation)["teacher_input_contract"]
        digest = model.payload_accounting(representation)["teacher_input_contract_sha256"]
        assert digest == hashlib.sha256(contract.encode("utf-8")).hexdigest()


def test_common_candidate_assignment_has_c0_c1_cgt1_reads_and_stable_ties_for_every_representation():
    model = _fit_zero_baselines()
    waveforms = np.zeros((3, 64))
    candidates = [np.array([103, 7]), np.array([7]), np.array([], dtype=int)]
    for representation in REPRESENTATIONS:
        prediction, diagnostics = model.assign(waveforms, candidates, representation)
        assert prediction.tolist() == [103, 7, None]
        np.testing.assert_array_equal(diagnostics["candidate_count"], [2, 1, 0])
        assert diagnostics["logical_candidate_template_reads"].tolist() == [2, 0, 0]
        if representation in {"fit_scale_float64", "legacy_peak_normalized_float64"}:
            assert diagnostics["logical_candidate_template_bits"].tolist() == [0, 0, 0]
        else:
            assert diagnostics["logical_candidate_template_bits"][0] > 0


def test_shared_fisher48_uses_fit_eligible_units_only_stable_sample_ties_and_rare_templates():
    waveforms = np.zeros((21, 64))
    waveforms[10:20, 9] = 0.5
    waveforms[20] = -0.25
    labels = [7] * 10 + [103] * 10 + [999]
    first = CausalTemporalBaselines().fit(waveforms, labels)
    np.testing.assert_array_equal(first.fit_unit_counts_, [10, 10, 1])
    np.testing.assert_array_equal(first.fisher_eligible_unit_mask_, [True, True, False])
    np.testing.assert_array_equal(first.fisher_eligible_unit_ids_, [7, 103])
    assert first.fisher48_templates_.shape == (3, 48)
    # The discriminative sample wins; zero-score residuals use ascending ID.
    assert first.fisher48_ids_[0] == 9

    perturbed = waveforms.copy()
    perturbed[20] = np.linspace(-1e6, 1e6, 64)
    second = CausalTemporalBaselines().fit(perturbed, labels)
    for a, b in zip(
        (first.scale_, first.fit_sample_saturation_fraction_, first.fit_quantizer_eligible_row_count_, first.fit_quantizer_source_sample_count_, first.fisher48_ids_, first.fisher_between_sum_squares_, first.fisher_within_sum_squares_, first.fisher_scores_, first.fisher_ranks_),
        (second.scale_, second.fit_sample_saturation_fraction_, second.fit_quantizer_eligible_row_count_, second.fit_quantizer_source_sample_count_, second.fisher48_ids_, second.fisher_between_sum_squares_, second.fisher_within_sum_squares_, second.fisher_scores_, second.fisher_ranks_),
    ):
        np.testing.assert_array_equal(a, b)
    assert first.fit_quantizer_eligible_row_count_ == 20
    assert first.fit_quantizer_source_sample_count_ == 20 * 64
    payload = first.payload_accounting("signed_full64x5")
    assert payload["fit_sample_saturation_population"] == "fisher_eligible_fit_rows"
    assert payload["fit_quantizer_eligible_row_count"] == 20
    assert payload["fit_quantizer_source_sample_count"] == 20 * 64
    assert payload["fit_all_quantizer_row_count"] == 21
    assert payload["fit_all_quantizer_source_sample_count"] == 21 * 64
    assert first.saturation_accounting() == {
        key: payload[key]
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


def test_fit_isolation_legacy_report_only_and_explicit_integer_rne_morphology_haz():
    train = np.zeros((20, 64))
    train[10:] = 0.25
    model = CausalTemporalBaselines().fit(train, [7] * 10 + [103] * 10)
    frozen = (model.scale_, model.fisher48_ids_.copy(), model.fisher_scores_.copy(), model.signed_full_templates_.copy())
    # Transforming arbitrary held-out inputs cannot change any fitted state.
    legacy = model.transform(np.array([[2.0] * 64, [-3.0] * 64]), "legacy_peak_normalized_float64")
    np.testing.assert_array_equal(legacy, [[1.0] * 64, [-1.0] * 64])
    model.transform(np.linspace(-10, 10, 128).reshape(2, 64), "fisher48_signed5")
    assert model.scale_ == frozen[0]
    np.testing.assert_array_equal(model.fisher48_ids_, frozen[1])
    np.testing.assert_array_equal(model.fisher_scores_, frozen[2])
    np.testing.assert_array_equal(model.signed_full_templates_, frozen[3])

    # Integer RNE template ties: 1.5 rounds to 2 and 2.5 rounds to even 2.
    codes = np.array([[1, 2], [2, 3]], dtype=np.int64)
    templates = model._integer_templates(codes, np.array([7, 7]), np.array([7]), -15, 15)
    np.testing.assert_array_equal(templates, [[2, 2]])
    # Morphology unsigned5 uses HAZ: 0.5 maps to 16, not implicit banker's 15/16.
    np.testing.assert_array_equal(model._unsigned_morphology_codes([[0.5]]), [[16]])
    assert model.test_sample_saturation_fraction(np.full((1, 64), 100.0)) == 1.0


def test_legacy_report_uses_prepared_epsilon_contract_without_changing_primary_raw_teacher():
    train = np.zeros((20, 64), dtype=float)
    train[10:] = 0.25
    model = CausalTemporalBaselines().fit(train, [7] * 10 + [103] * 10)
    raw = np.array(
        [
            np.full(64, 5e-9),
            np.full(64, -2e-9),
        ],
        dtype=float,
    )
    prepared_legacy = raw / np.maximum(
        np.abs(raw).max(axis=1, keepdims=True), 1e-8
    )
    assert LEGACY_PEAK_NORMALIZATION_EPSILON == 1e-8
    np.testing.assert_array_equal(legacy_peak_normalize(raw), prepared_legacy)
    np.testing.assert_array_equal(
        model.transform(raw, "legacy_peak_normalized_float64"), prepared_legacy
    )
    np.testing.assert_array_equal(
        model.transform(raw, "fit_scale_float64"),
        np.clip(raw / model.scale_, -1.0, 1.0),
    )
    assert "epsilon=1e-8" in model.payload_accounting(
        "legacy_peak_normalized_float64"
    )["teacher_input_contract"]


def test_integer_sad_bounds_are_declared_and_reached_without_wrap():
    model = _fit_zero_baselines()
    assert model._sad_maximum("signed_full64x5") == 1920
    assert model._sad_maximum("fisher48_signed5") == 1440
    assert model._sad_maximum("morphology12_unsigned5") == 372
    # Reach the signed full64 bound using valid externally supplied templates.
    model.signed_full_templates_[:] = -15
    prediction, diagnostics = model.assign(np.ones((1, 64)), [np.array([7, 103])], "signed_full64x5")
    assert prediction[0] == 7
    assert diagnostics["winning_distance"][0] == 1920


def test_invalid_inputs_not_evaluable_fisher_and_unknown_ambiguous_candidate_are_rejected():
    with pytest.raises(ValueError):
        CausalTemporalBaselines().fit(np.zeros((19, 64)), [7] * 10 + [103] * 9)
    model = _fit_zero_baselines()
    with pytest.raises(ValueError):
        model.assign(np.zeros((1, 64)), [np.array([7, 999])], "signed_full64x5")
    with pytest.raises(ValueError):
        model.transform(np.zeros((1, 63)), "signed_full64x5")
    with pytest.raises(ValueError):
        model.payload_accounting("unknown")
