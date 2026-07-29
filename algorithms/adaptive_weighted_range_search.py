"""Fixed 5-bit, per-unit power-of-two weighted-L1 candidate search."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from .adaptive_range_search import UniformUnsignedQuantizer


ScoreMode = Literal["uniform", "stability", "separation_stability"]
RankProfile = Literal["top3_x4", "tiered_421"]
_DIMENSIONS = 9
_BITS = 5
_WEIGHTS = np.asarray((1, 2, 4), dtype=np.int64)


class AdaptiveWeightedRangeSearch:
    """One 9-D integer centroid/radius row with per-coordinate shifts.

    All rows retain nine active absolute-difference lanes.  ``stability`` ranks
    fit-only within-unit Q95 deviations ascending, and
    ``separation_stability`` ranks fit-only median centroid separation divided
    by ``within + 1`` descending.  Percentiles explicitly use NumPy's
    ``method='linear'`` interpolation.  For an even number of other units,
    ``np.median`` uses the arithmetic midpoint of the two central separations.
    Shift encodings are ``00 -> 1``, ``01 -> 2``, and ``10 -> 4``; ``11`` is
    reserved and invalid.  Distances use integer arithmetic with a declared
    11-bit maximum of 1116—there is no 10-bit wrap or saturation.
    """

    def __init__(self, n_bits: int = 5, *, radius_percentile: float = 95.0) -> None:
        if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)):
            raise ValueError("n_bits must be the fixed integer value 5")
        if int(n_bits) != _BITS:
            raise ValueError("AdaptiveWeightedRangeSearch requires fixed 5-bit codes")
        if not np.isfinite(radius_percentile) or not 0.0 <= float(radius_percentile) <= 100.0:
            raise ValueError("radius_percentile must lie in [0, 100]")
        self.n_bits = _BITS
        self.radius_percentile = float(radius_percentile)
        self.quantizer = UniformUnsignedQuantizer(_BITS, fixed_unit_interval=True)

    def fit(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
        *,
        score_mode: ScoreMode = "uniform",
        rank_profile: RankProfile = "top3_x4",
    ) -> "AdaptiveWeightedRangeSearch":
        """Fit one weighted row per unit and initialize fit-partition radii."""
        x = self._validate_features(features, name="features")
        y = self._validate_labels(labels, x.shape[0], name="labels")
        mode, profile = self._validate_configuration(score_mode, rank_profile)
        codes = self.quantizer.fit_transform(x)
        units = np.unique(y)
        centroids = np.stack(
            [np.rint(codes[y == unit].mean(axis=0)).astype(np.int64) for unit in units], axis=0
        )
        within = np.stack(
            [
                np.percentile(
                    np.abs(codes[y == unit] - centroids[row]), 95.0, axis=0, method="linear"
                )
                for row, unit in enumerate(units)
            ],
            axis=0,
        )
        between = self._between_median_separation(centroids)
        separation_scores = between / (within + 1.0)
        ranks = self._feature_ranks(within, separation_scores, mode)
        weights = self._weights_from_ranks(ranks, mode, profile)

        self.n_features_in_ = _DIMENSIONS
        self.units_ = np.asarray(units)
        self.centroids_ = centroids
        self.within_q95_ = within
        self.between_median_separation_ = between
        self.separation_stability_scores_ = separation_scores
        self.feature_ranks_ = ranks
        self.weights_ = weights
        self.shift_codes_ = np.log2(weights).astype(np.int64)
        self.score_mode_ = mode
        self.rank_profile_ = profile
        self.radii_ = self._radii_for_rows(codes, y)
        self.radius_calibration_counts_ = np.zeros(units.size, dtype=np.int64)
        self.radius_calibration_fallback_mask_ = np.ones(units.size, dtype=bool)
        self.radius_source_ = "fit"
        self.is_fitted_ = True
        return self

    def calibrate(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
    ) -> "AdaptiveWeightedRangeSearch":
        """Replace supported radii from calibration rows, retaining missing rows."""
        self._require_fitted()
        self._assert_valid_shift_codes()
        x = self._validate_features(features, name="features")
        y = self._validate_labels(labels, x.shape[0], name="labels")
        codes = self.quantizer.transform(x)
        counts = np.asarray([np.count_nonzero(y == unit) for unit in self.units_], dtype=np.int64)
        fallback = counts == 0
        radii = self.radii_.copy()
        for row, unit in enumerate(self.units_):
            if fallback[row]:
                continue
            local = codes[y == unit]
            radii[row] = self._radius(local, row)
        self.radii_ = radii
        self.radius_calibration_counts_ = counts
        self.radius_calibration_fallback_mask_ = fallback
        self.radius_source_ = "calibration"
        return self

    def transform(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Quantize D=9 raw rows using the fixed ``[0,1]`` 5-bit mapping."""
        self._require_fitted()
        return self.quantizer.transform(self._validate_features(features, name="features"))

    def query(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        *,
        active_row_mask: np.ndarray | Sequence[bool] | None = None,
        active_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        return self.query_codes(
            self.transform(features), active_row_mask=active_row_mask, active_mask=active_mask
        )

    def query_codes(
        self,
        codes: np.ndarray | Sequence[Sequence[int]],
        *,
        active_row_mask: np.ndarray | Sequence[bool] | None = None,
        active_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Return stable unit candidates from inclusive weighted integer L1 rows."""
        self._require_fitted()
        self._assert_valid_shift_codes()
        if active_row_mask is not None and active_mask is not None:
            raise ValueError("pass only one of active_row_mask and active_mask")
        values = self._validate_codes(codes)
        active = self._validate_active_mask(
            active_row_mask if active_row_mask is not None else active_mask, values.shape[0]
        )
        distances = (
            np.abs(values[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :])
            * self.weights_[np.newaxis, :, :]
        ).sum(axis=2)
        matched = active & (distances <= self.radii_[np.newaxis, :])
        candidates = [self.units_[rows].copy() for rows in matched]
        active_rows = active.sum(axis=1).astype(np.int64)
        candidate_count = matched.sum(axis=1).astype(np.int64)
        shift1_per_row = (self.shift_codes_ == 1).sum(axis=1, dtype=np.int64)
        shift2_per_row = (self.shift_codes_ == 2).sum(axis=1, dtype=np.int64)
        enabled_abs_lanes = active_rows * _DIMENSIONS
        shift1_lanes = (active.astype(np.int64) * shift1_per_row[np.newaxis, :]).sum(axis=1)
        shift2_lanes = (active.astype(np.int64) * shift2_per_row[np.newaxis, :]).sum(axis=1)
        fallback_rows = np.zeros(values.shape[0], dtype=np.int64)
        diagnostics = {
            "candidate_count": candidate_count,
            "initial_candidate_count": candidate_count.copy(),
            "final_candidate_count": candidate_count.copy(),
            "active_row_count": active_rows,
            "logical_row_comparisons": active_rows.copy(),
            "logical_primary_row_comparisons": active_rows.copy(),
            "logical_fallback_row_comparisons": fallback_rows,
            "logical_total_level1_row_comparisons": active_rows.copy(),
            "logical_enabled_abs_lanes": enabled_abs_lanes,
            "logical_primary_abs_lanes": enabled_abs_lanes.copy(),
            "logical_fallback_abs_lanes": np.zeros(values.shape[0], dtype=np.int64),
            "logical_total_abs_lanes": enabled_abs_lanes.copy(),
            "logical_shift1_lanes": shift1_lanes.astype(np.int64),
            "logical_shift2_lanes": shift2_lanes.astype(np.int64),
            "logical_shifted_lanes": (shift1_lanes + shift2_lanes).astype(np.int64),
            "logical_accumulator_additions": (enabled_abs_lanes - active_rows).astype(np.int64),
        }
        return candidates, diagnostics

    def memory_accounting(self) -> dict[str, float | int | str]:
        """Return fixed 74-bit weighted and 54-bit optimized-uniform row costs."""
        self._require_fitted()
        self._assert_valid_shift_codes()
        n_units = int(self.units_.size)
        unit_id_bits = int(np.ceil(np.log2(n_units))) if n_units > 1 else 0
        centroid_bits, shift_bits, radius_bits = 45, 18, 11
        logical_payload = centroid_bits + shift_bits + radius_bits
        optimized_uniform_payload = 45 + 9
        weight_counts = {weight: int(np.count_nonzero(self.weights_ == weight)) for weight in _WEIGHTS}
        return {
            "n_units": n_units,
            "n_features": _DIMENSIONS,
            "n_bits": _BITS,
            "centroid_bits_per_row": centroid_bits,
            "shift_code_bits_per_row": shift_bits,
            "shift_code_encoding": "00->1,01->2,10->4,11->reserved_invalid",
            "full_scale_weighted_radius_bits_per_row": radius_bits,
            "logical_search_payload_bits_per_row": logical_payload,
            "optimized_unweighted_payload_bits_per_row": optimized_uniform_payload,
            "unit_id_bits_per_row": unit_id_bits,
            "valid_bits_per_row": 1,
            "logical_row_bits_with_metadata_per_row": logical_payload + unit_id_bits + 1,
            "optimized_unweighted_row_bits_with_metadata_per_row": (
                optimized_uniform_payload + unit_id_bits + 1
            ),
            "logical_search_payload_total_bits": int(n_units * logical_payload),
            "logical_total_bits_with_metadata": int(n_units * (logical_payload + unit_id_bits + 1)),
            "logical_weight1_coordinates_total": weight_counts[1],
            "logical_weight2_coordinates_total": weight_counts[2],
            "logical_weight4_coordinates_total": weight_counts[4],
            "logical_shift1_dimensions_mean": float((self.shift_codes_ == 1).sum(axis=1).mean()),
            "logical_shift2_dimensions_mean": float((self.shift_codes_ == 2).sum(axis=1).mean()),
            "maximum_weighted_distance": 9 * 4 * 31,
        }

    def _radii_for_rows(self, codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        radii = np.empty(self.units_.size, dtype=np.int64)
        for row, unit in enumerate(self.units_):
            radii[row] = self._radius(codes[labels == unit], row)
        return radii

    def _radius(self, local_codes: np.ndarray, row: int) -> int:
        distance = (np.abs(local_codes - self.centroids_[row]) * self.weights_[row]).sum(axis=1)
        return int(np.ceil(np.percentile(distance, self.radius_percentile, method="linear")))

    @staticmethod
    def _between_median_separation(centroids: np.ndarray) -> np.ndarray:
        between = np.zeros_like(centroids, dtype=np.float64)
        for row in range(centroids.shape[0]):
            if centroids.shape[0] > 1:
                between[row] = np.median(
                    np.abs(centroids[np.arange(centroids.shape[0]) != row] - centroids[row]), axis=0
                )
        return between

    def _assert_valid_shift_codes(self) -> None:
        """Reject reserved/external shift codes before they affect accounting/search."""
        expected_shape = (self.units_.size, _DIMENSIONS)
        if self.shift_codes_.shape != expected_shape or not np.isin(self.shift_codes_, (0, 1, 2)).all():
            raise ValueError("shift_codes_ must use only 0, 1, or 2; code 3 is reserved")
        if not np.array_equal(self.weights_, (1 << self.shift_codes_)):
            raise ValueError("weights_ must exactly match shift_codes_ power-of-two encoding")

    @staticmethod
    def _feature_ranks(within: np.ndarray, separation_scores: np.ndarray, mode: ScoreMode) -> np.ndarray:
        ranks = np.empty_like(within, dtype=np.int64)
        feature_ids = np.arange(_DIMENSIONS)
        for row in range(within.shape[0]):
            if mode == "stability":
                order = np.lexsort((feature_ids, within[row]))
            elif mode == "separation_stability":
                order = np.lexsort((feature_ids, -separation_scores[row]))
            else:
                order = feature_ids
            ranks[row, order] = np.arange(_DIMENSIONS, dtype=np.int64)
        return ranks

    @staticmethod
    def _weights_from_ranks(
        ranks: np.ndarray, mode: ScoreMode, profile: RankProfile
    ) -> np.ndarray:
        weights = np.ones(ranks.shape, dtype=np.int64)
        if mode == "uniform":
            return weights
        weights[ranks <= 2] = 4
        if profile == "tiered_421":
            weights[(ranks >= 3) & (ranks <= 5)] = 2
        return weights

    @staticmethod
    def _validate_configuration(score_mode: str, rank_profile: str) -> tuple[ScoreMode, RankProfile]:
        if score_mode not in ("uniform", "stability", "separation_stability"):
            raise ValueError("score_mode must be uniform, stability, or separation_stability")
        if rank_profile not in ("top3_x4", "tiered_421"):
            raise ValueError("rank_profile must be top3_x4 or tiered_421")
        return score_mode, rank_profile  # type: ignore[return-value]

    @staticmethod
    def _validate_features(
        features: np.ndarray | Sequence[Sequence[float]], *, name: str
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != _DIMENSIONS:
            raise ValueError(f"{name} must have shape (n_samples, 9)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    @staticmethod
    def _validate_labels(
        labels: np.ndarray | Sequence[Any], n_samples: int, *, name: str
    ) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 1 or values.shape[0] != n_samples or n_samples == 0:
            raise ValueError(f"{name} must be one-dimensional and match features")
        return values

    def _validate_codes(self, codes: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        values = np.asarray(codes)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != _DIMENSIONS:
            raise ValueError("codes must have shape (n_queries, 9)")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("codes must be finite numeric values")
        return np.clip(np.rint(values), 0, self.quantizer.code_max).astype(np.int64)

    def _validate_active_mask(
        self, active_row_mask: np.ndarray | Sequence[bool] | None, n_queries: int
    ) -> np.ndarray:
        if active_row_mask is None:
            return np.ones((n_queries, self.units_.size), dtype=bool)
        mask = np.asarray(active_row_mask)
        if mask.dtype != np.bool_:
            raise ValueError("active_row_mask must have boolean dtype")
        if mask.shape == (self.units_.size,):
            return np.broadcast_to(mask, (n_queries, self.units_.size)).copy()
        if mask.shape == (n_queries, self.units_.size):
            return mask.copy()
        raise ValueError("active_row_mask must have shape (n_units,) or (n_queries, n_units)")

    def _require_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("AdaptiveWeightedRangeSearch must be fitted before use")
