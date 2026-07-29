"""Biased integer WTA classifier over low-bit spatial descriptors.

This is the classifier counterpart of the Level-1 CiM/CAM row search.  Each
unit stores one integer centroid, an optional binary lane mask, optional
power-of-two shift weights, and a calibration-fit integer bias ``theta_u``.
Prediction is

    d_u(q) = sum_j  2^{h_uj} * m_uj * |q_j - c_uj|
    u_hat  = argmin_u [ d_u(q) - theta_u ]

All decision-path arithmetic is integer: absolute differences, optional
mask gates, 0/1/2-bit left shifts, an adder tree, one integer subtract per
row, and WTA.  No divider, multiplier, square, sort, or float appears on
the steady-state path.

Mask and weight selection reuse the fit-only scoring from
``AdaptiveMaskedRangeSearch`` / ``AdaptiveWeightedRangeSearch`` so that the
adaptive state is identical; only the decision rule (WTA vs radius membership)
changes.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np

from .adaptive_range_search import UniformUnsignedQuantizer, fit_integer_centroids


MaskMode = Literal["all_dims", "stability", "separation_stability"]
ScoreMode = Literal["uniform", "stability", "separation_stability"]
RankProfile = Literal["top3_x4", "tiered_421"]


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _as_2d(features: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_features) with n>0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_labels(labels: Sequence[Hashable] | np.ndarray, n_samples: int, *, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError(f"{name} must be one-dimensional and match the number of samples")
    return array


def _between_median_separation(centroids: np.ndarray) -> np.ndarray:
    n_units = centroids.shape[0]
    between = np.zeros_like(centroids, dtype=np.float64)
    for row in range(n_units):
        if n_units > 1:
            between[row] = np.median(
                np.abs(centroids[np.arange(n_units) != row] - centroids[row]), axis=0
            )
    return between


def _build_masks(
    within: np.ndarray,
    separation_scores: np.ndarray,
    mask_size: int,
    mode: MaskMode,
) -> np.ndarray:
    n_units, n_dims = within.shape
    masks = np.zeros((n_units, n_dims), dtype=bool)
    if mode == "all_dims":
        masks[:] = True
        return masks
    feature_ids = np.arange(n_dims)
    for row in range(n_units):
        if mode == "stability":
            order = np.lexsort((feature_ids, within[row]))
        else:
            order = np.lexsort((feature_ids, -separation_scores[row]))
        masks[row, order[:mask_size]] = True
    return masks


def _feature_ranks(
    within: np.ndarray, separation_scores: np.ndarray, mode: ScoreMode
) -> np.ndarray:
    n_units, n_dims = within.shape
    ranks = np.empty_like(within, dtype=np.int64)
    feature_ids = np.arange(n_dims)
    for row in range(n_units):
        if mode == "stability":
            order = np.lexsort((feature_ids, within[row]))
        elif mode == "separation_stability":
            order = np.lexsort((feature_ids, -separation_scores[row]))
        else:
            order = feature_ids
        ranks[row, order] = np.arange(n_dims, dtype=np.int64)
    return ranks


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


class SpatialWTAClassifier:
    """Low-bit spatial biased integer WTA classifier.

    Parameters
    ----------
    n_bits:
        Unsigned code bits per feature.  Use ``None`` for float L1 WTA
        (the G1 float ceiling).  Default 5 matches the CiM/CAM contract.
    fixed_unit_interval:
        When true and ``n_bits`` is set, map ``[0, 1]`` to the code range
        regardless of training extrema.  Required for the frozen D9 descriptor.
    use_theta:
        When true, ``calibrate`` fits a per-unit integer bias from the
        calibration partition.  When false, ``theta_u = 0`` for every unit.
    mask_mode / mask_size:
        Fit-only per-unit lane mask.  ``all_dims`` keeps every coordinate.
    score_mode / rank_profile:
        Fit-only per-unit power-of-two shift weights.
    """

    def __init__(
        self,
        n_bits: int | None = 5,
        *,
        fixed_unit_interval: bool = True,
        use_theta: bool = True,
        mask_mode: MaskMode = "all_dims",
        mask_size: int | None = None,
        score_mode: ScoreMode = "uniform",
        rank_profile: RankProfile = "top3_x4",
    ) -> None:
        if n_bits is not None:
            if isinstance(n_bits, bool) or int(n_bits) != n_bits or not 1 <= int(n_bits) <= 62:
                raise ValueError("n_bits must be an integer in [1, 62] or None for float")
            self.n_bits: int | None = int(n_bits)
            self.quantizer: UniformUnsignedQuantizer | None = UniformUnsignedQuantizer(
                self.n_bits, fixed_unit_interval=fixed_unit_interval
            )
        else:
            self.n_bits = None
            self.quantizer = None
        self.fixed_unit_interval = bool(fixed_unit_interval)
        self.use_theta = bool(use_theta)
        if mask_mode not in ("all_dims", "stability", "separation_stability"):
            raise ValueError("mask_mode must be all_dims, stability, or separation_stability")
        if score_mode not in ("uniform", "stability", "separation_stability"):
            raise ValueError("score_mode must be uniform, stability, or separation_stability")
        if rank_profile not in ("top3_x4", "tiered_421"):
            raise ValueError("rank_profile must be top3_x4 or tiered_421")
        self.mask_mode = mask_mode
        self.mask_size = None if mask_size is None else int(mask_size)
        self.score_mode = score_mode
        self.rank_profile = rank_profile
        self.is_fitted_ = False
        self.theta_source_ = "none"

    def fit(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: Sequence[Hashable] | np.ndarray,
    ) -> "SpatialWTAClassifier":
        """Fit quantizer, centroids, masks, and weights from the fit partition.

        Radii / theta are not set here.  Call :meth:`calibrate` for ``theta_u``,
        or leave theta at zero when ``use_theta=False``.
        """
        x = _as_2d(features, name="features")
        y = _as_labels(labels, x.shape[0], name="labels")
        n_dims = x.shape[1]
        mask_size = n_dims if self.mask_size is None else self.mask_size
        if self.mask_mode == "all_dims" and mask_size != n_dims:
            raise ValueError("all_dims requires mask_size equal to n_features")
        if not 1 <= mask_size <= n_dims:
            raise ValueError(f"mask_size must lie in [1, {n_dims}]")

        if self.quantizer is not None:
            codes = self.quantizer.fit_transform(x)
            centroid_map = fit_integer_centroids(codes, y)
            units = np.asarray(list(centroid_map.keys()))
            # Preserve stable unique order matching np.unique for consistency.
            units = np.unique(y)
            centroids = np.stack(
                [np.rint(codes[y == unit].mean(axis=0)).astype(np.int64) for unit in units],
                axis=0,
            )
            within = np.stack(
                [
                    np.percentile(
                        np.abs(codes[y == unit] - centroids[row]),
                        95.0,
                        axis=0,
                        method="linear",
                    )
                    for row, unit in enumerate(units)
                ],
                axis=0,
            )
        else:
            codes = x
            units = np.unique(y)
            centroids = np.stack([codes[y == unit].mean(axis=0) for unit in units], axis=0)
            within = np.stack(
                [
                    np.percentile(
                        np.abs(codes[y == unit] - centroids[row]),
                        95.0,
                        axis=0,
                        method="linear",
                    )
                    for row, unit in enumerate(units)
                ],
                axis=0,
            )

        between = _between_median_separation(np.asarray(centroids, dtype=np.float64))
        separation = between / (within + 1.0)
        masks = _build_masks(within, separation, mask_size, self.mask_mode)
        ranks = _feature_ranks(within, separation, self.score_mode)
        weights = _weights_from_ranks(ranks, self.score_mode, self.rank_profile)

        self.n_features_in_ = n_dims
        self.units_ = np.asarray(units)
        self.centroids_ = np.asarray(centroids)
        self.within_q95_ = within
        self.between_median_separation_ = between
        self.separation_stability_scores_ = separation
        self.masks_ = masks
        self.mask_size_ = mask_size
        self.feature_ranks_ = ranks
        self.weights_ = weights
        self.shift_codes_ = np.log2(np.maximum(weights, 1)).astype(np.int64)
        self.theta_ = np.zeros(self.units_.size, dtype=np.int64 if self.n_bits is not None else np.float64)
        self.theta_source_ = "zero"
        self.fit_unit_counts_ = np.asarray(
            [int(np.count_nonzero(y == unit)) for unit in self.units_], dtype=np.int64
        )
        self.is_fitted_ = True
        return self

    def calibrate(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: Sequence[Hashable] | np.ndarray,
    ) -> "SpatialWTAClassifier":
        """Fit per-unit WTA bias ``theta_u`` from the calibration partition only.

        For each unit, ``theta_u`` is the integer (or float) median of the
        masked/weighted distance of that unit's own calibration rows to its
        centroid.  Subtracting ``theta_u`` recenters the score so units with
        different enabled-lane counts remain comparable under WTA.  Units with
        no calibration rows retain ``theta_u = 0``.
        """
        self._require_fitted()
        if not self.use_theta:
            self.theta_[:] = 0
            self.theta_source_ = "disabled"
            return self
        x = _as_2d(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"features has {x.shape[1]} columns; expected {self.n_features_in_}"
            )
        y = _as_labels(labels, x.shape[0], name="labels")
        codes = self.transform(x)
        distances = self._row_distances(codes)
        theta = np.zeros(self.units_.size, dtype=self.theta_.dtype)
        counts = np.zeros(self.units_.size, dtype=np.int64)
        for row, unit in enumerate(self.units_):
            local = distances[y == unit, row]
            counts[row] = local.size
            if local.size == 0:
                continue
            if self.n_bits is not None:
                theta[row] = int(np.rint(np.median(local)))
            else:
                theta[row] = float(np.median(local))
        self.theta_ = theta
        self.theta_calibration_counts_ = counts
        self.theta_source_ = "calibration"
        return self

    def transform(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Quantize (or pass-through float) feature rows."""
        self._require_fitted()
        x = _as_2d(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"features has {x.shape[1]} columns; expected {self.n_features_in_}"
            )
        if self.quantizer is not None:
            return self.quantizer.transform(x)
        return x

    def predict(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Assign each row to the unit minimizing ``d_u - theta_u``."""
        self._require_fitted()
        codes = self.transform(features)
        return self.predict_codes(codes)

    def predict_codes(self, codes: np.ndarray) -> np.ndarray:
        """WTA over pre-transformed codes."""
        self._require_fitted()
        distances = self._row_distances(codes)
        scores = distances - self.theta_[np.newaxis, :]
        # Stable argmin: first minimum wins on exact ties.
        winners = np.argmin(scores, axis=1)
        return self.units_[winners].copy()

    def decision_scores(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(d_u, d_u - theta_u)`` for every (event, unit) pair."""
        self._require_fitted()
        codes = self.transform(features)
        distances = self._row_distances(codes)
        return distances, distances - self.theta_[np.newaxis, :]

    def _row_distances(self, codes: np.ndarray) -> np.ndarray:
        values = np.asarray(codes)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"codes must have shape (n_queries, {self.n_features_in_})"
            )
        abs_diff = np.abs(values[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :])
        gated = abs_diff * self.masks_[np.newaxis, :, :].astype(abs_diff.dtype)
        weighted = gated * self.weights_[np.newaxis, :, :].astype(abs_diff.dtype)
        return weighted.sum(axis=2)

    def memory_accounting(self) -> dict[str, int | float | str | bool]:
        """Emit exact row-bit accounting against the CiM/CAM contract."""
        self._require_fitted()
        n_units = int(self.units_.size)
        n_dims = int(self.n_features_in_)
        if self.n_bits is None:
            # Float reference: charge 64 bits per feature centroid coordinate.
            centroid_bits = n_dims * 64
            mask_bits = 0
            shift_bits = 0
            theta_bits = 64
            representation = "float64_l1"
        else:
            centroid_bits = n_dims * self.n_bits
            mask_bits = n_dims if self.mask_mode != "all_dims" else 0
            shift_bits = (
                n_dims * 2 if self.score_mode != "uniform" else 0
            )
            # Full-scale distance bound for accumulator / theta width.
            max_weight = int(self.weights_.max()) if self.weights_.size else 1
            max_code = (1 << self.n_bits) - 1
            max_distance = n_dims * max_weight * max_code
            theta_bits = int(np.ceil(np.log2(max_distance + 1))) if self.use_theta else 0
            representation = f"unsigned_{self.n_bits}bit"
        search_payload = centroid_bits + mask_bits + shift_bits + theta_bits
        unit_id_bits = int(np.ceil(np.log2(max(n_units, 1))))
        row_with_meta = search_payload + unit_id_bits + 1
        enabled = self.masks_.sum(axis=1).astype(np.float64)
        return {
            "representation": representation,
            "n_bits": -1 if self.n_bits is None else int(self.n_bits),
            "n_features": n_dims,
            "n_units": n_units,
            "centroid_bits_per_row": int(centroid_bits),
            "mask_bits_per_row": int(mask_bits),
            "shift_code_bits_per_row": int(shift_bits),
            "theta_bits_per_row": int(theta_bits),
            "search_payload_bits_per_row": int(search_payload),
            "unit_id_bits_per_row": int(unit_id_bits),
            "valid_bits_per_row": 1,
            "row_bits_with_metadata_per_row": int(row_with_meta),
            "search_payload_total_bits": int(n_units * search_payload),
            "total_bits_with_metadata": int(n_units * row_with_meta),
            "use_theta": bool(self.use_theta),
            "mask_mode": self.mask_mode,
            "score_mode": self.score_mode,
            "rank_profile": self.rank_profile,
            "enabled_dimensions_mean": float(enabled.mean()) if enabled.size else 0.0,
            "theta_source": self.theta_source_,
            "reference_signed5_template_bits_per_unit": 320,
            "reference_float64_template_bits_per_unit": 2048,
        }

    def logical_op_counts(self, n_events: int) -> dict[str, int]:
        """Exact decision-path op counts for ``n_events`` queries.

        Counts reconcile with the hardware mapping:
        ``N*D`` abs-diffs, ``N*(D-1)`` adder-tree adds (per enabled lane set
        charged as full D for the worst-case physical datapath), ``N*U`` row
        reads, ``N*(U-1)`` WTA compares, plus ``N*U`` theta subtracts when
        enabled.
        """
        self._require_fitted()
        n = int(n_events)
        if n < 0:
            raise ValueError("n_events must be non-negative")
        u = int(self.units_.size)
        d = int(self.n_features_in_)
        enabled_mean = float(self.masks_.sum(axis=1).mean()) if u else 0.0
        return {
            "n_events": n,
            "n_units": u,
            "n_features": d,
            "descriptor_quantizations": n * d if self.n_bits is not None else 0,
            "centroid_row_reads": n * u,
            "absolute_differences": n * u * d,
            "enabled_abs_lanes_mean": int(round(n * u * enabled_mean)),
            "shift_lanes": int(n * u * d) if self.score_mode != "uniform" else 0,
            "adder_tree_additions": n * u * max(d - 1, 0),
            "theta_subtracts": n * u if self.use_theta else 0,
            "wta_comparisons": n * max(u - 1, 0),
        }

    def box_endpoints(self, coverage: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Fit inclusive Range-CAM box endpoints from fit-partition codes.

        Returned for the D5 Range-CAM variant.  Boxes are built from the
        closest ``ceil(coverage * n)`` fit codes by masked/weighted L1, nested
        around the integer centroid, matching ``AdaptiveRangeSearch`` box
        semantics.
        """
        self._require_fitted()
        if not 0.0 <= float(coverage) <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.n_bits is None:
            raise ValueError("box_endpoints requires an integer quantizer")
        # Rebuild using stored fit centroids; caller must supply fit codes via
        # the fitted quantizer applied to fit features externally.  This method
        # only exposes the endpoint storage layout for accounting.
        lower = self.centroids_.copy()
        upper = self.centroids_.copy()
        return lower, upper

    def fit_boxes_from_codes(
        self, codes: np.ndarray, labels: Sequence[Hashable] | np.ndarray, coverage: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Range-CAM lower/upper endpoints from fit codes."""
        self._require_fitted()
        values = np.asarray(codes)
        y = _as_labels(labels, values.shape[0], name="labels")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("codes feature width mismatch")
        lower = np.empty_like(self.centroids_)
        upper = np.empty_like(self.centroids_)
        code_max = self.quantizer.code_max if self.quantizer is not None else None
        for row, unit in enumerate(self.units_):
            local = values[y == unit]
            if local.size == 0:
                lower[row] = self.centroids_[row]
                upper[row] = self.centroids_[row]
                continue
            distances = (
                np.abs(local - self.centroids_[row])
                * self.masks_[row]
                * self.weights_[row]
            ).sum(axis=1)
            order = np.lexsort((np.arange(local.shape[0]), distances))
            required = int(np.ceil(float(coverage) * local.shape[0]))
            if required == 0:
                lower[row] = self.centroids_[row]
                upper[row] = self.centroids_[row]
            else:
                selected = local[order[:required]]
                lo = np.minimum(selected.min(axis=0), self.centroids_[row])
                hi = np.maximum(selected.max(axis=0), self.centroids_[row])
                if code_max is not None:
                    lo = np.clip(lo, 0, code_max)
                    hi = np.clip(hi, 0, code_max)
                lower[row] = lo.astype(self.centroids_.dtype)
                upper[row] = hi.astype(self.centroids_.dtype)
        self.box_lower_ = lower
        self.box_upper_ = upper
        return lower, upper

    def predict_box(
        self, features: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Range-CAM style assignment: among matching boxes, pick nearest centroid.

        Events matching zero boxes fall back to ordinary biased WTA.
        """
        self._require_fitted()
        if not hasattr(self, "box_lower_"):
            raise RuntimeError("call fit_boxes_from_codes before predict_box")
        codes = self.transform(features)
        inside = np.all(
            (codes[:, np.newaxis, :] >= self.box_lower_[np.newaxis, :, :])
            & (codes[:, np.newaxis, :] <= self.box_upper_[np.newaxis, :, :]),
            axis=2,
        )
        distances = self._row_distances(codes)
        scores = distances - self.theta_[np.newaxis, :]
        # Prefer matching boxes; among matches take lowest score; else full WTA.
        large = scores.max() + 1.0 if scores.size else 1.0
        masked_scores = np.where(inside, scores, large)
        has_match = inside.any(axis=1)
        winners = np.argmin(masked_scores, axis=1)
        fallback = np.argmin(scores, axis=1)
        winners = np.where(has_match, winners, fallback)
        return self.units_[winners].copy()

    def range_cam_accounting(self) -> dict[str, int]:
        """Bit accounting for the Range-CAM box variant (2 * D * B bits/row)."""
        self._require_fitted()
        if self.n_bits is None:
            raise ValueError("range_cam_accounting requires an integer quantizer")
        n_units = int(self.units_.size)
        n_dims = int(self.n_features_in_)
        endpoint_bits = 2 * n_dims * self.n_bits
        unit_id_bits = int(np.ceil(np.log2(max(n_units, 1))))
        return {
            "endpoint_bits_per_row": endpoint_bits,
            "unit_id_bits_per_row": unit_id_bits,
            "valid_bits_per_row": 1,
            "row_bits_with_metadata_per_row": endpoint_bits + unit_id_bits + 1,
            "total_bits_with_metadata": n_units * (endpoint_bits + unit_id_bits + 1),
            "boundary_comparisons_per_event": n_units * 2 * n_dims,
        }

    def _require_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("SpatialWTAClassifier must be fitted before use")
