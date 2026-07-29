"""Bit-exact alignment-conditioned prefix-causal dyadic temporal sketches.

The component deliberately knows nothing about event detection or filtering.
Given an already aligned 64-sample waveform, it consumes only the fixed prefix
ending at ``15 + delay_samples`` and never reads a later sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


WINDOW_LENGTH = 64
PRE_ALIGNMENT_SAMPLES = 15
# Compatibility name for callers that imported the former constant.  New
# documentation and all internal timing language use "alignment".
PRE_PEAK_SAMPLES = PRE_ALIGNMENT_SAMPLES
SIGNED_CODE_MIN = -15
SIGNED_CODE_MAX = 15
ALLOWED_DELAYS = (0, 8, 16, 32, 48)
ALLOWED_FEATURE_COUNTS = (8, 12, 16)


@dataclass(frozen=True)
class DyadicFilter:
    """A stable full-window dyadic dictionary entry."""

    filter_id: int
    kind: Literal["mean", "haar"]
    support: int
    start: int

    @property
    def final_index(self) -> int:
        return self.start + self.support - 1


def build_dyadic_dictionary() -> tuple[DyadicFilter, ...]:
    """Build the frozen 93-entry full-window dictionary in stable ID order."""
    entries: list[DyadicFilter] = []
    for support in (2, 4, 8, 16, 32):
        for start in range(0, WINDOW_LENGTH, support):
            entries.append(DyadicFilter(len(entries), "mean", support, start))
    for support in (4, 8, 16, 32, 64):
        for start in range(0, WINDOW_LENGTH, support):
            entries.append(DyadicFilter(len(entries), "haar", support, start))
    if len(entries) != 93:
        raise AssertionError("the frozen full-window dyadic dictionary must have 93 entries")
    return tuple(entries)


DYADIC_DICTIONARY = build_dyadic_dictionary()


def round_half_away_from_zero(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return signed round-to-nearest integers with half ties away from zero."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return (np.sign(array) * np.floor(np.abs(array) + 0.5)).astype(np.int64)


def signed_haz_divide(numerators: np.ndarray | Sequence[int], denominator: int) -> np.ndarray:
    """Signed integer HAZ division: ``sign(n) * ((abs(n)+S/2)//S)``.

    ``denominator`` must be a positive even power of two, as required by the
    dyadic mean/Haar dictionary.  No floating-point division participates in
    feature normalization.
    """
    if isinstance(denominator, bool) or not isinstance(denominator, (int, np.integer)):
        raise ValueError("denominator must be a positive even power of two")
    divisor = int(denominator)
    if divisor < 2 or divisor & (divisor - 1):
        raise ValueError("denominator must be a positive even power of two")
    values = np.asarray(numerators)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("numerators must be integer values")
    signed = values.astype(np.int64, copy=False)
    return np.sign(signed) * ((np.abs(signed) + divisor // 2) // divisor)


def signed_rne_divide(numerators: np.ndarray | Sequence[int], denominator: int) -> np.ndarray:
    """Signed integer quotient rounded nearest-even using quotient/remainder."""
    if isinstance(denominator, bool) or not isinstance(denominator, (int, np.integer)):
        raise ValueError("denominator must be a positive integer")
    divisor = int(denominator)
    if divisor <= 0:
        raise ValueError("denominator must be a positive integer")
    values = np.asarray(numerators)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("numerators must be integer values")
    signed = values.astype(np.int64, copy=False)
    quotient, remainder = np.divmod(np.abs(signed), divisor)
    increment = (2 * remainder > divisor) | ((2 * remainder == divisor) & ((quotient & 1) == 1))
    return np.sign(signed) * (quotient + increment.astype(np.int64))


def signed_power_of_two_scale(values: np.ndarray | Sequence[float]) -> tuple[float, int, float]:
    """Return ``(scale, exponent, q99.9_abs)`` from finite fit-prefix samples."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("fit prefix samples must be non-empty and finite")
    quantile = float(np.percentile(np.abs(array), 99.9, method="linear"))
    if quantile == 0.0:
        return 1.0, 0, quantile
    exponent = int(np.ceil(np.log2(quantile)))
    scale = float(2.0**exponent)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fit-derived scale must be finite and positive")
    return scale, exponent, quantile


class CausalTemporalSketch:
    """Shared fit-selected dyadic prefix sketch with signed five-bit SAD rows.

    Parameters are intentionally restricted to the frozen pilot grid.  The
    dictionary has stable global IDs; changing a horizon merely changes its
    eligible prefix.  Mean/Haar normalization uses signed half-away-from-zero
    division by a power-of-two support.  Unit templates instead use the frozen
    nearest-even average rule required by the protocol.
    """

    def __init__(self, *, delay_samples: int, n_features: int) -> None:
        if delay_samples not in ALLOWED_DELAYS:
            raise ValueError(f"delay_samples must be one of {ALLOWED_DELAYS}")
        if n_features not in ALLOWED_FEATURE_COUNTS:
            raise ValueError(f"n_features must be one of {ALLOWED_FEATURE_COUNTS}")
        self.delay_samples = int(delay_samples)
        self.n_features = int(n_features)
        self.prefix_length = PRE_ALIGNMENT_SAMPLES + self.delay_samples + 1
        self.horizon_index = self.prefix_length - 1
        self.dictionary_ = DYADIC_DICTIONARY

    def fit(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
    ) -> "CausalTemporalSketch":
        """Fit scale, shared Fisher filter IDs, templates, and float ceiling.

        Only these fit rows configure the scale, Fisher scores, selected IDs,
        integer templates, and same-horizon float templates.
        """
        wave = self._validate_waveforms(waveforms, name="waveforms")
        labs = self._validate_labels(labels, wave.shape[0], name="labels")
        prefix = wave[:, : self.prefix_length]
        units = np.unique(labs)
        unit_counts = np.asarray([np.count_nonzero(labs == unit) for unit in units], dtype=np.int64)
        fisher_unit_mask = unit_counts >= 10
        fisher_units = units[fisher_unit_mask]
        if fisher_units.size < 2:
            raise ValueError("fewer than two fit units have count >= 10; Fisher selection is not evaluable")
        fisher_rows = np.isin(labs, fisher_units)
        # The scale feeds integer feature codes and therefore filter selection;
        # it is fitted from the same eligible population as Fisher statistics.
        # Rare units still receive templates under this frozen shared scale.
        scale, exponent, quantile = signed_power_of_two_scale(prefix[fisher_rows])
        # Selection and scale-audit saturation use exactly the same eligible
        # population.  We still encode all fit rows below because every unit,
        # including a rare one, retains a frozen template.  Keep its broader
        # saturation diagnostic under an explicitly different name so it
        # cannot be mistaken for selection configuration state.
        codes, all_fit_saturation = self._quantize_prefix_with_scale(prefix, scale)
        _, eligible_fit_saturation = self._quantize_prefix_with_scale(
            prefix[fisher_rows], scale
        )
        eligible_ids = np.asarray(
            [entry.filter_id for entry in self.dictionary_ if entry.final_index <= self.horizon_index],
            dtype=np.int64,
        )
        if eligible_ids.size < self.n_features:
            raise ValueError("horizon has fewer eligible filters than n_features")
        all_eligible = self._encode_filter_ids(codes, eligible_ids)
        fisher_between, fisher_within, fisher = self._fisher_statistics(
            all_eligible, labs, fisher_units
        )
        if not np.isfinite(fisher).all():
            raise ValueError("Fisher scores must be finite")
        order = np.lexsort((eligible_ids, -fisher))
        selected = eligible_ids[order[: self.n_features]]
        selected_codes = self._encode_filter_ids(codes, selected)
        templates = np.stack(
            [
                np.clip(
                    signed_rne_divide(selected_codes[labs == unit].sum(axis=0), np.count_nonzero(labs == unit)),
                    SIGNED_CODE_MIN,
                    SIGNED_CODE_MAX,
                ).astype(np.int64)
                for unit in units
            ],
            axis=0,
        )
        scaled_float = np.clip(prefix / scale, -1.0, 1.0)
        float_templates = np.stack([scaled_float[labs == unit].mean(axis=0) for unit in units], axis=0)

        self.units_ = np.asarray(units)
        self.fit_unit_counts_ = unit_counts
        self.fisher_eligible_unit_mask_ = fisher_unit_mask
        self.fisher_eligible_unit_ids_ = np.asarray(fisher_units)
        self.scale_ = scale
        self.scale_exponent_ = exponent
        self.fit_abs_q999_ = quantile
        self.fit_sample_saturation_fraction_ = eligible_fit_saturation
        self.fit_all_sample_saturation_fraction_ = all_fit_saturation
        self.fit_sample_saturation_population_ = "fisher_eligible_fit_rows"
        self.fit_quantizer_eligible_row_count_ = int(fisher_rows.sum())
        self.fit_quantizer_source_sample_count_ = int(
            fisher_rows.sum() * self.prefix_length
        )
        self.fit_all_quantizer_row_count_ = int(wave.shape[0])
        self.fit_all_quantizer_source_sample_count_ = int(
            wave.shape[0] * self.prefix_length
        )
        self.eligible_filter_ids_ = eligible_ids
        self.fisher_between_sum_squares_ = fisher_between
        self.fisher_within_sum_squares_ = fisher_within
        self.eligible_fisher_scores_ = fisher
        self.fisher_scores_ = self._full_score_array(eligible_ids, fisher)
        self.fisher_ranks_ = self._full_rank_array(eligible_ids, order)
        self.selected_filter_ids_ = selected
        self.templates_ = templates
        self.float_prefix_templates_ = float_templates
        self.is_fitted_ = True
        return self

    def quantize_prefix(self, waveforms: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Encode only the eligible prefix into signed 5-bit sample codes."""
        self._require_fitted()
        wave = self._validate_waveforms(waveforms, name="waveforms")
        codes, _ = self._quantize_prefix_with_scale(wave[:, : self.prefix_length], self.scale_)
        return codes

    def sample_saturation_fraction(
        self, waveforms: np.ndarray | Sequence[Sequence[float]]
    ) -> float:
        """Return rounded-preclip signed-5 saturation under frozen prefix/scale.

        This reports the same ``abs(round_HAZ(15*x/scale)) > 15`` condition
        used for the fit audit.  Samples later than the configured alignment
        prefix are deliberately neither inspected nor counted.
        """
        self._require_fitted()
        wave = self._validate_waveforms(waveforms, name="waveforms")
        _, saturation = self._quantize_prefix_with_scale(wave[:, : self.prefix_length], self.scale_)
        return saturation

    def transform(self, waveforms: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Return selected signed 5-bit dyadic feature codes for waveform rows."""
        self._require_fitted()
        return self._encode_filter_ids(self.quantize_prefix(waveforms), self.selected_filter_ids_)

    def assign_candidates(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        candidates: Sequence[np.ndarray | Sequence[Any]],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Refine frozen candidate lists with SAD while retaining input tie order.

        ``C=0`` is intentionally left as ``None``: its frozen unweighted
        spatial fallback belongs to Level 1 and performs zero temporal reads.
        ``C=1`` is a direct decision.  Only ``C>1`` reads unit templates.
        """
        self._require_fitted()
        features = self.transform(waveforms)
        if len(candidates) != features.shape[0]:
            raise ValueError("candidates and waveforms must have the same length")
        lookup = {self._scalar(unit): row for row, unit in enumerate(self.units_)}
        predictions = np.empty(features.shape[0], dtype=object)
        counts = np.empty(features.shape[0], dtype=np.int64)
        template_reads = np.zeros(features.shape[0], dtype=np.int64)
        sad_abs = np.zeros(features.shape[0], dtype=np.int64)
        sad_additions = np.zeros(features.shape[0], dtype=np.int64)
        winning_sad_distance = np.full(features.shape[0], -1, dtype=np.int64)
        for event, values in enumerate(candidates):
            candidate = list(np.asarray(values).tolist())
            counts[event] = len(candidate)
            if not candidate:
                predictions[event] = None
                continue
            if len(candidate) == 1:
                predictions[event] = candidate[0]
                continue
            missing = [unit for unit in candidate if self._scalar(unit) not in lookup]
            if missing:
                raise ValueError("candidate label has no fitted temporal template")
            rows = np.asarray([lookup[self._scalar(unit)] for unit in candidate], dtype=np.int64)
            distance = np.abs(self.templates_[rows] - features[event]).sum(axis=1)
            if (distance < 0).any() or (distance > 30 * self.n_features).any():
                raise AssertionError("signed 5-bit SAD must not wrap or saturate")
            # np.argmin keeps the first candidate under an exact SAD tie.
            winner = int(np.argmin(distance))
            predictions[event] = candidate[winner]
            winning_sad_distance[event] = int(distance[winner])
            template_reads[event] = len(candidate)
            sad_abs[event] = len(candidate) * self.n_features
            sad_additions[event] = len(candidate) * (self.n_features - 1)
        diagnostics = self._logical_event_accounting(features.shape[0])
        diagnostics.update(
            {
                "candidate_count": counts,
                "logical_candidate_template_reads": template_reads,
                "logical_candidate_template_bits": template_reads * (5 * self.n_features),
                "logical_sad_absolute_differences": sad_abs,
                "logical_sad_reduction_additions": sad_additions,
                "winning_sad_distance": winning_sad_distance,
                "c0_events": (counts == 0).astype(np.int64),
                "c1_events": (counts == 1).astype(np.int64),
                "cgt1_events": (counts > 1).astype(np.int64),
            }
        )
        return predictions, diagnostics

    def same_horizon_float_assign(
        self,
        waveforms: np.ndarray | Sequence[Sequence[float]],
        candidates: Sequence[np.ndarray | Sequence[Any]],
    ) -> np.ndarray:
        """Diagnostic float-prefix ceiling using the same fit scale and candidates."""
        self._require_fitted()
        wave = self._validate_waveforms(waveforms, name="waveforms")
        if len(candidates) != wave.shape[0]:
            raise ValueError("candidates and waveforms must have the same length")
        values = np.clip(wave[:, : self.prefix_length] / self.scale_, -1.0, 1.0)
        lookup = {self._scalar(unit): row for row, unit in enumerate(self.units_)}
        prediction = np.empty(wave.shape[0], dtype=object)
        for event, candidate_values in enumerate(candidates):
            candidate = list(np.asarray(candidate_values).tolist())
            if not candidate:
                prediction[event] = None
            elif len(candidate) == 1:
                prediction[event] = candidate[0]
            else:
                rows = np.asarray([lookup[self._scalar(unit)] for unit in candidate], dtype=np.int64)
                distance = np.abs(self.float_prefix_templates_[rows] - values[event]).sum(axis=1)
                prediction[event] = candidate[int(np.argmin(distance))]
        return prediction

    def logical_extraction_accounting(self, n_events: int) -> dict[str, np.ndarray]:
        """Return per-event direct-streaming logical extraction counters."""
        self._require_fitted()
        if isinstance(n_events, bool) or not isinstance(n_events, (int, np.integer)) or n_events < 0:
            raise ValueError("n_events must be a non-negative integer")
        return self._logical_event_accounting(int(n_events))

    def memory_accounting(self) -> dict[str, int | float | str]:
        """Return template/selected-ID storage and frozen prefix configuration facts."""
        self._require_fitted()
        template_bits_per_unit = 5 * self.n_features
        selected_id_bits = 7 * self.n_features
        return {
            "n_units": int(self.units_.size),
            "n_features": self.n_features,
            "template_bits_per_unit": template_bits_per_unit,
            "stored_template_bits": int(self.units_.size * template_bits_per_unit),
            "selected_filter_id_bits": selected_id_bits,
            "dictionary_entries": len(self.dictionary_),
            "filter_id_bits": 7,
            "prefix_length": self.prefix_length,
            "horizon_index": self.horizon_index,
            "delay_samples": self.delay_samples,
            "scale_exponent": self.scale_exponent_,
            "fit_sample_saturation_fraction": self.fit_sample_saturation_fraction_,
            "fit_sample_saturation_population": self.fit_sample_saturation_population_,
            "fit_quantizer_eligible_row_count": self.fit_quantizer_eligible_row_count_,
            "fit_quantizer_source_sample_count": self.fit_quantizer_source_sample_count_,
            "fit_all_sample_saturation_fraction": self.fit_all_sample_saturation_fraction_,
            "fit_all_quantizer_row_count": self.fit_all_quantizer_row_count_,
            "fit_all_quantizer_source_sample_count": self.fit_all_quantizer_source_sample_count_,
        }

    def saturation_accounting(self) -> dict[str, int | float | str]:
        """Return JSON-serializable fit/test saturation population metadata.

        ``fit_sample_saturation_fraction`` is the configuration-relevant
        eligible-row value.  ``fit_all_sample_saturation_fraction`` is a
        separate diagnostic over all rows that received templates.
        """
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

    def template_traffic(self, candidates: Sequence[np.ndarray | Sequence[Any]]) -> dict[str, int | float | bool]:
        """Return exact global-64x5 versus conditional sketch-template traffic."""
        self._require_fitted()
        counts = np.asarray([len(values) for values in candidates], dtype=np.int64)
        actual_reads = int(counts[counts > 1].sum())
        actual_bits = actual_reads * (5 * self.n_features)
        global_bits = int(counts.size * self.units_.size * WINDOW_LENGTH * 5)
        return {
            "global_full64x5_template_bits": global_bits,
            "actual_sketch_template_bits": actual_bits,
            "conditional_template_reads": actual_reads,
            "global_to_actual_template_traffic_factor": (
                float(global_bits / actual_bits) if actual_bits else float("inf")
            ),
            "meets_exact_4x_traffic_condition": bool(global_bits >= 4 * actual_bits),
        }

    def _logical_event_accounting(self, n_events: int) -> dict[str, np.ndarray]:
        selected = [self.dictionary_[identifier] for identifier in self.selected_filter_ids_]
        support_total = int(sum(entry.support for entry in selected))
        positive = int(sum(entry.support if entry.kind == "mean" else entry.support // 2 for entry in selected))
        negative = int(sum(0 if entry.kind == "mean" else entry.support // 2 for entry in selected))
        if positive + negative != support_total:
            raise AssertionError("direct-streaming signed updates must equal selected support total")
        constant = lambda value: np.full(n_events, value, dtype=np.int64)
        return {
            "logical_raw_prefix_samples_consumed": constant(self.prefix_length),
            "logical_sample_quantizations": constant(self.prefix_length),
            "logical_selected_feature_accumulator_updates": constant(support_total),
            "logical_positive_coefficient_updates": constant(positive),
            "logical_negative_coefficient_updates": constant(negative),
            "logical_normalization_shifts": constant(self.n_features),
            "logical_emitted_feature_codes": constant(self.n_features),
        }

    def _encode_filter_ids(self, sample_codes: np.ndarray, filter_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(sample_codes, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != self.prefix_length:
            raise ValueError("sample_codes must have the configured prefix length")
        if (values < SIGNED_CODE_MIN).any() or (values > SIGNED_CODE_MAX).any():
            raise ValueError("sample codes must lie in [-15, 15]; reserved code -16 is invalid")
        result = np.empty((values.shape[0], filter_ids.size), dtype=np.int64)
        for column, identifier in enumerate(filter_ids):
            entry = self.dictionary_[int(identifier)]
            segment = values[:, entry.start : entry.start + entry.support]
            if entry.kind == "mean":
                numerator = segment.sum(axis=1)
                if (numerator < -512).any() or (numerator > 511).any():
                    raise AssertionError("mean numerator must fit signed 10-bit range")
            else:
                half = entry.support // 2
                numerator = segment[:, :half].sum(axis=1) - segment[:, half:].sum(axis=1)
                if (numerator < -1024).any() or (numerator > 1023).any():
                    raise AssertionError("Haar numerator must fit signed 11-bit range")
            result[:, column] = np.clip(
                signed_haz_divide(numerator, entry.support), SIGNED_CODE_MIN, SIGNED_CODE_MAX
            )
        return result

    @staticmethod
    def _fisher_statistics(
        features: np.ndarray, labels: np.ndarray, units: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eligible_rows = np.isin(labels, units)
        if not np.any(eligible_rows):
            raise ValueError("Fisher statistics require eligible fit rows")
        # Rare fit units receive templates but must not influence *any*
        # selection statistic, including the Fisher global mean.
        grand_mean = features[eligible_rows].mean(axis=0)
        between = np.zeros(features.shape[1], dtype=np.float64)
        within = np.zeros(features.shape[1], dtype=np.float64)
        for unit in units:
            group = features[labels == unit]
            mean = group.mean(axis=0)
            between += group.shape[0] * (mean - grand_mean) ** 2
            within += ((group - mean) ** 2).sum(axis=0)
        return between, within, between / (within + 1e-12)

    @staticmethod
    def _full_score_array(eligible_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        full = np.full(len(DYADIC_DICTIONARY), np.nan, dtype=np.float64)
        full[eligible_ids] = scores
        return full

    @staticmethod
    def _full_rank_array(eligible_ids: np.ndarray, sorted_positions: np.ndarray) -> np.ndarray:
        full = np.full(len(DYADIC_DICTIONARY), -1, dtype=np.int64)
        full[eligible_ids[sorted_positions]] = np.arange(eligible_ids.size, dtype=np.int64)
        return full

    @staticmethod
    def _quantize_prefix_with_scale(prefix: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
        raw = np.asarray(prefix, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] == 0 or not np.isfinite(raw).all():
            raise ValueError("prefix must be a finite non-empty two-dimensional array")
        scaled = round_half_away_from_zero(15.0 * raw / float(scale))
        saturation = float((np.abs(scaled) > SIGNED_CODE_MAX).mean())
        return np.clip(scaled, SIGNED_CODE_MIN, SIGNED_CODE_MAX).astype(np.int64), saturation

    @staticmethod
    def _validate_waveforms(
        waveforms: np.ndarray | Sequence[Sequence[float]], *, name: str
    ) -> np.ndarray:
        values = np.asarray(waveforms, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != WINDOW_LENGTH:
            raise ValueError(f"{name} must have shape (n_events, {WINDOW_LENGTH})")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    @staticmethod
    def _validate_labels(
        labels: np.ndarray | Sequence[Any], n_events: int, *, name: str
    ) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 1 or values.shape[0] != n_events or n_events == 0:
            raise ValueError(f"{name} must be one-dimensional and match waveforms")
        return values

    @staticmethod
    def _scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    def _require_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("CausalTemporalSketch must be fitted before use")
