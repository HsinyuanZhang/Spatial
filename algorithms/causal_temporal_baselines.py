"""Frozen same-candidate temporal baselines for the causal-sketch protocol.

All representations are fitted from the delay-48 aligned full window.  This
module intentionally performs no spatial fallback: ``C=0`` has no temporal
template read and returns ``None`` for the runner to combine with the frozen
Level-1 decision path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence

import numpy as np

from .causal_temporal_sketch import (
    SIGNED_CODE_MAX,
    SIGNED_CODE_MIN,
    WINDOW_LENGTH,
    round_half_away_from_zero,
    signed_power_of_two_scale,
    signed_rne_divide,
)


Representation = Literal[
    "fit_scale_float64",
    "legacy_peak_normalized_float64",
    "signed_full64x5",
    "fisher48_signed5",
    "morphology12_unsigned5",
]

REPRESENTATIONS: tuple[Representation, ...] = (
    "fit_scale_float64",
    "legacy_peak_normalized_float64",
    "signed_full64x5",
    "fisher48_signed5",
    "morphology12_unsigned5",
)
FISHER_TAP_COUNT = 48
MORPHOLOGY_DIMENSIONS = 12
LEGACY_PEAK_NORMALIZATION_EPSILON = 1e-8


def legacy_peak_normalize(
    waveforms: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the frozen per-event peak normalization used by legacy reports.

    This is intentionally separate from the primary raw/fit-scale teacher.
    Its ``1e-8`` floor is the same contract used by prepared recordings, so a
    validated prepared legacy matrix and this transform agree even for very
    small nonzero waveforms.
    """
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"waveforms must have shape (n_events, {WINDOW_LENGTH})")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("waveforms must be non-empty and finite")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    return values / np.maximum(scale, LEGACY_PEAK_NORMALIZATION_EPSILON)


class CausalTemporalBaselines:
    """Fit mandatory full-window temporal reports on a common fit partition.

    The power-of-two scale, Fisher score, and Fisher tap IDs are configuration
    state.  They are computed from fit units having at least ten examples;
    rare fit units retain templates but cannot influence selection state.
    """

    teacher_input_contract = (
        "primary_fit_scale_full64_raw_float_l1|"
        "input=filtered_pre_event_normalization_main_channel|"
        "alignment=gt_event_timestamp|delay=48|scale=fit_eligible_unit_power2|"
        "clip=[-1,1]|templates=fit_unit_mean|candidates=frozen|tie=level1_order"
    )
    legacy_input_contract = (
        "report_only_legacy_per_event_peak_normalized_full64_float_l1|"
        "input=legacy_full_window_event_peak_normalization|epsilon=1e-8|"
        "templates=fit_unit_mean|candidates=frozen|tie=level1_order"
    )
    signed_input_contract = (
        "fit_scale_full64_signed5_haz|scale=fit_eligible_unit_power2|"
        "clip=[-1,1]|templates=fit_integer_rne|candidates=frozen|tie=level1_order"
    )
    morphology_input_contract = (
        "fit_scale_full64_morphology12_unsigned5_haz|scale=fit_eligible_unit_power2|"
        "clip=[-1,1]|analytical_morphology12_[0,1]|templates=fit_integer_rne|"
        "candidates=frozen|tie=level1_order"
    )

    def fit(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
    ) -> "CausalTemporalBaselines":
        """Fit all mandatory baselines from raw aligned full-64 fit rows."""
        wave = self._validate_waveforms(waveforms, name="waveforms")
        labs = self._validate_labels(labels, wave.shape[0], name="labels")
        units = np.unique(labs)
        counts = np.asarray([np.count_nonzero(labs == unit) for unit in units], dtype=np.int64)
        eligible_mask = counts >= 10
        eligible_units = units[eligible_mask]
        if eligible_units.size < 2:
            raise ValueError("fewer than two fit units have count >= 10; Fisher selection is not evaluable")
        eligible_rows = np.isin(labs, eligible_units)
        scale, exponent, quantile = signed_power_of_two_scale(wave[eligible_rows])
        fit_scaled = np.clip(wave / scale, -1.0, 1.0)
        signed_codes, all_fit_saturation = self._signed_codes_with_saturation(wave / scale)
        _, fit_saturation = self._signed_codes_with_saturation(wave[eligible_rows] / scale)
        between, within, scores = self._fisher_statistics(signed_codes, labs, eligible_units)
        if not np.isfinite(scores).all():
            raise ValueError("Fisher scores must be finite")
        sample_ids = np.arange(WINDOW_LENGTH, dtype=np.int64)
        fisher_order = np.lexsort((sample_ids, -scores))
        fisher_ids = fisher_order[:FISHER_TAP_COUNT]

        legacy = self._legacy_peak_normalize(wave)
        morphology_float = self._morphology_features(fit_scaled)
        morphology_codes = self._unsigned_morphology_codes(morphology_float)

        self.units_ = np.asarray(units)
        self.fit_unit_counts_ = counts
        self.fisher_eligible_unit_mask_ = eligible_mask
        self.fisher_eligible_unit_ids_ = np.asarray(eligible_units)
        self.scale_ = scale
        self.scale_exponent_ = exponent
        self.fit_abs_q999_ = quantile
        self.fit_sample_saturation_fraction_ = fit_saturation
        self.fit_all_sample_saturation_fraction_ = all_fit_saturation
        self.fit_sample_saturation_population_ = "fisher_eligible_fit_rows"
        self.fit_quantizer_eligible_row_count_ = int(eligible_rows.sum())
        self.fit_quantizer_source_sample_count_ = int(eligible_rows.sum() * WINDOW_LENGTH)
        self.fit_all_quantizer_row_count_ = int(wave.shape[0])
        self.fit_all_quantizer_source_sample_count_ = int(wave.shape[0] * WINDOW_LENGTH)
        self.fit_scale_float_templates_ = self._float_templates(fit_scaled, labs, units)
        self.legacy_peak_float_templates_ = self._float_templates(legacy, labs, units)
        self.signed_full_templates_ = self._integer_templates(signed_codes, labs, units, -15, 15)
        self.fisher_between_sum_squares_ = between
        self.fisher_within_sum_squares_ = within
        self.fisher_scores_ = scores
        self.fisher_ranks_ = np.empty(WINDOW_LENGTH, dtype=np.int64)
        self.fisher_ranks_[fisher_order] = np.arange(WINDOW_LENGTH, dtype=np.int64)
        self.fisher48_ids_ = fisher_ids.astype(np.int64)
        self.fisher48_templates_ = self._integer_templates(
            signed_codes[:, fisher_ids], labs, units, -15, 15
        )
        self.morphology12_templates_ = self._integer_templates(
            morphology_codes, labs, units, 0, 31
        )
        self.contracts_ = self._contracts()
        self.is_fitted_ = True
        return self

    def transform(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        representation: Representation,
    ) -> np.ndarray:
        """Transform raw full-window inputs for one named frozen baseline."""
        self._require_fitted()
        name = self._validate_representation(representation)
        wave = self._validate_waveforms(waveforms, name="waveforms")
        scaled = np.clip(wave / self.scale_, -1.0, 1.0)
        if name == "fit_scale_float64":
            return scaled
        if name == "legacy_peak_normalized_float64":
            return self._legacy_peak_normalize(wave)
        signed = self._signed_codes(scaled)
        if name == "signed_full64x5":
            return signed
        if name == "fisher48_signed5":
            return signed[:, self.fisher48_ids_]
        return self._unsigned_morphology_codes(self._morphology_features(scaled))

    def assign(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        candidates: Sequence[np.ndarray | Sequence[Any]],
        representation: Representation,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Apply common C0/C1/C>1 candidate semantics with stable SAD/L1 ties."""
        self._require_fitted()
        name = self._validate_representation(representation)
        features = self.transform(waveforms, name)
        if len(candidates) != features.shape[0]:
            raise ValueError("candidates and waveforms must have the same length")
        templates, code_bits, metric = self._template_spec(name)
        lookup = {self._scalar(unit): row for row, unit in enumerate(self.units_)}
        prediction = np.empty(features.shape[0], dtype=object)
        counts = np.empty(features.shape[0], dtype=np.int64)
        reads = np.zeros(features.shape[0], dtype=np.int64)
        distance_out = np.full(features.shape[0], -1.0, dtype=np.float64)
        for event, values in enumerate(candidates):
            candidate = list(np.asarray(values).tolist())
            counts[event] = len(candidate)
            if not candidate:
                prediction[event] = None
                continue
            if len(candidate) == 1:
                prediction[event] = candidate[0]
                continue
            if any(self._scalar(unit) not in lookup for unit in candidate):
                raise ValueError("candidate label has no fitted temporal template")
            rows = np.asarray([lookup[self._scalar(unit)] for unit in candidate], dtype=np.int64)
            distance = np.abs(templates[rows] - features[event]).sum(axis=1)
            if metric == "sad":
                if (distance < 0).any() or (distance > self._sad_maximum(name)).any():
                    raise AssertionError("integer SAD baseline distance must not wrap or saturate")
            winner = int(np.argmin(distance))  # first Level-1 candidate wins a tie
            prediction[event] = candidate[winner]
            distance_out[event] = float(distance[winner])
            reads[event] = len(candidate)
        dim = int(templates.shape[1])
        diagnostics = {
            "candidate_count": counts,
            "logical_candidate_template_reads": reads,
            "logical_candidate_template_bits": reads * (code_bits * dim),
            "logical_sad_absolute_differences": reads * dim if metric == "sad" else np.zeros_like(reads),
            "logical_sad_reduction_additions": reads * (dim - 1) if metric == "sad" else np.zeros_like(reads),
            "winning_distance": distance_out,
            "c0_events": (counts == 0).astype(np.int64),
            "c1_events": (counts == 1).astype(np.int64),
            "cgt1_events": (counts > 1).astype(np.int64),
        }
        return prediction, diagnostics

    def test_sample_saturation_fraction(
        self, waveforms: np.ndarray | Sequence[Sequence[float]]
    ) -> float:
        """Return pre-clip signed-5 saturation fraction under frozen fit scale."""
        self._require_fitted()
        wave = self._validate_waveforms(waveforms, name="waveforms")
        _, saturation = self._signed_codes_with_saturation(wave / self.scale_)
        return saturation

    def payload_accounting(self, representation: Representation) -> dict[str, int | float | str]:
        """Return per-unit payload and shared fit configuration bits."""
        self._require_fitted()
        name = self._validate_representation(representation)
        templates, bits, _ = self._template_spec(name)
        dim = int(templates.shape[1])
        shared = 48 * 6 if name == "fisher48_signed5" else 0
        return {
            "representation": name,
            "template_bits_per_unit": bits * dim,
            "stored_template_bits": int(bits * dim * self.units_.size),
            "shared_configuration_bits": shared,
            "logical_encoded_template_bits": bits * dim if bits else 0,
            "report_only_float_representation": "true" if bits == 0 else "false",
            "fit_sample_saturation_fraction": self.fit_sample_saturation_fraction_,
            "fit_sample_saturation_population": self.fit_sample_saturation_population_,
            "fit_quantizer_eligible_row_count": self.fit_quantizer_eligible_row_count_,
            "fit_quantizer_source_sample_count": self.fit_quantizer_source_sample_count_,
            "fit_all_sample_saturation_fraction": self.fit_all_sample_saturation_fraction_,
            "fit_all_quantizer_row_count": self.fit_all_quantizer_row_count_,
            "fit_all_quantizer_source_sample_count": self.fit_all_quantizer_source_sample_count_,
            "teacher_input_contract": self.contracts_[name]["contract"],
            "teacher_input_contract_sha256": self.contracts_[name]["sha256"],
        }

    def _contracts(self) -> dict[str, dict[str, str]]:
        source = {
            "fit_scale_float64": self.teacher_input_contract,
            "legacy_peak_normalized_float64": self.legacy_input_contract,
            "signed_full64x5": self.signed_input_contract,
            "fisher48_signed5": self.signed_input_contract + "|shared_fisher48_ids=fit_only_stable_sample_order",
            "morphology12_unsigned5": self.morphology_input_contract,
        }
        return {
            name: {"contract": value, "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
            for name, value in source.items()
        }

    def _template_spec(self, representation: Representation) -> tuple[np.ndarray, int, str]:
        mapping = {
            "fit_scale_float64": (self.fit_scale_float_templates_, 0, "l1"),
            "legacy_peak_normalized_float64": (self.legacy_peak_float_templates_, 0, "l1"),
            "signed_full64x5": (self.signed_full_templates_, 5, "sad"),
            "fisher48_signed5": (self.fisher48_templates_, 5, "sad"),
            "morphology12_unsigned5": (self.morphology12_templates_, 5, "sad"),
        }
        return mapping[representation]

    @staticmethod
    def _signed_codes(scaled: np.ndarray) -> np.ndarray:
        return CausalTemporalBaselines._signed_codes_with_saturation(scaled)[0]

    @staticmethod
    def _signed_codes_with_saturation(scaled: np.ndarray) -> tuple[np.ndarray, float]:
        rounded = round_half_away_from_zero(15.0 * np.asarray(scaled, dtype=np.float64))
        saturation = float((np.abs(rounded) > 15).mean())
        return np.clip(rounded, -15, 15).astype(np.int64), saturation

    @staticmethod
    def _sad_maximum(representation: Representation) -> int:
        if representation == "signed_full64x5":
            return 30 * 64
        if representation == "fisher48_signed5":
            return 30 * 48
        if representation == "morphology12_unsigned5":
            return 31 * 12
        raise ValueError("float representations do not have integer SAD bounds")

    _legacy_peak_normalize = staticmethod(legacy_peak_normalize)

    def saturation_accounting(self) -> dict[str, int | float | str]:
        """Return JSON-serializable fit saturation population metadata."""
        self._require_fitted()
        return {
            "fit_sample_saturation_fraction": self.fit_sample_saturation_fraction_,
            "fit_sample_saturation_population": self.fit_sample_saturation_population_,
            "fit_quantizer_eligible_row_count": self.fit_quantizer_eligible_row_count_,
            "fit_quantizer_source_sample_count": self.fit_quantizer_source_sample_count_,
            "fit_all_sample_saturation_fraction": self.fit_all_sample_saturation_fraction_,
            "fit_all_quantizer_row_count": self.fit_all_quantizer_row_count_,
            "fit_all_quantizer_source_sample_count": self.fit_all_quantizer_source_sample_count_,
        }

    @staticmethod
    def _morphology_features(clipped: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(clipped, dtype=np.float64), -1.0, 1.0)
        minimum, maximum = values.min(axis=1), values.max(axis=1)
        argmin, argmax = np.argmin(values, axis=1), np.argmax(values, axis=1)
        negative = np.maximum(-values, 0.0).sum(axis=1) / WINDOW_LENGTH
        positive = np.maximum(values, 0.0).sum(axis=1) / WINDOW_LENGTH
        transitions = (np.signbit(values[:, 1:]) != np.signbit(values[:, :-1])).sum(axis=1)
        features = np.column_stack((
            (minimum + 1.0) / 2.0, (maximum + 1.0) / 2.0, (maximum - minimum) / 2.0,
            np.max(np.abs(values), axis=1), argmin / 63.0, argmax / 63.0,
            (values[:, :16].mean(axis=1) + 1.0) / 2.0,
            (values[:, 24:40].mean(axis=1) + 1.0) / 2.0,
            (values[:, 48:64].mean(axis=1) + 1.0) / 2.0,
            negative, positive, transitions / 63.0,
        ))
        return np.clip(features, 0.0, 1.0)

    @staticmethod
    def _unsigned_morphology_codes(features: np.ndarray) -> np.ndarray:
        return np.clip(
            round_half_away_from_zero(31.0 * np.asarray(features, dtype=np.float64)), 0, 31
        ).astype(np.int64)

    @staticmethod
    def _float_templates(values: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
        return np.stack([values[labels == unit].mean(axis=0) for unit in units], axis=0)

    @staticmethod
    def _integer_templates(
        codes: np.ndarray, labels: np.ndarray, units: np.ndarray, lower: int, upper: int
    ) -> np.ndarray:
        return np.stack(
            [
                np.clip(
                    signed_rne_divide(codes[labels == unit].sum(axis=0), np.count_nonzero(labels == unit)),
                    lower,
                    upper,
                ).astype(np.int64)
                for unit in units
            ],
            axis=0,
        )

    @staticmethod
    def _fisher_statistics(
        features: np.ndarray, labels: np.ndarray, eligible_units: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = np.isin(labels, eligible_units)
        overall = features[keep].mean(axis=0)
        between = np.zeros(features.shape[1], dtype=np.float64)
        within = np.zeros(features.shape[1], dtype=np.float64)
        for unit in eligible_units:
            group = features[labels == unit]
            average = group.mean(axis=0)
            between += group.shape[0] * (average - overall) ** 2
            within += ((group - average) ** 2).sum(axis=0)
        return between, within, between / (within + 1e-12)

    @staticmethod
    def _validate_waveforms(waveforms: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
        values = np.asarray(waveforms, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != WINDOW_LENGTH:
            raise ValueError(f"{name} must have shape (n_events, {WINDOW_LENGTH})")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    @staticmethod
    def _validate_labels(labels: np.ndarray | Sequence[Any], n_events: int, *, name: str) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 1 or values.shape[0] != n_events:
            raise ValueError(f"{name} must be one-dimensional and match waveforms")
        return values

    @staticmethod
    def _validate_representation(representation: str) -> Representation:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown temporal baseline representation: {representation}")
        return representation  # type: ignore[return-value]

    @staticmethod
    def _scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    def _require_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("CausalTemporalBaselines must be fitted before use")
