"""Fixed-width 5-bit, unit-specific masked-L1 spatial candidate search.

Each fitted unit owns one 9-dimensional integer centroid, a nine-bit binary
dimension mask, and a full-scale masked-L1 radius.  Masks are constructed from
fit rows only; :meth:`calibrate` may subsequently replace radii from a
disjoint calibration partition without changing a centroid or mask.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from .adaptive_range_search import UniformUnsignedQuantizer


MaskMode = Literal["all_dims", "stability", "separation_stability"]
_DIMENSIONS = 9
_BITS = 5
_MASK_SIZES = (3, 5, 7, 9)


class AdaptiveMaskedRangeSearch:
    """One masked 5-bit integer L1 row per unit.

    ``all_dims`` is the exact all-one-mask baseline and therefore requires
    ``mask_size=9``.  ``stability`` keeps the dimensions with the lowest
    per-unit 95th-percentile absolute code deviation from that unit's integer
    centroid.  ``separation_stability`` ranks the ratio of median other-unit
    centroid separation to ``within + 1``.  All feature-score ties resolve to
    the lower feature ID.  Within-dimension and masked-radius percentiles use
    NumPy's explicit ``method='linear'`` interpolation so their values do not
    depend on a future NumPy default.
    """

    def __init__(self, n_bits: int = 5, *, radius_percentile: float = 95.0) -> None:
        if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)):
            raise ValueError("n_bits must be the fixed integer value 5")
        if int(n_bits) != _BITS:
            raise ValueError("AdaptiveMaskedRangeSearch requires fixed 5-bit codes")
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
        mask_size: int = 9,
        mask_mode: MaskMode = "all_dims",
    ) -> "AdaptiveMaskedRangeSearch":
        """Fit centroids, fit-only masks, and initial fit-partition radii.

        The initial radii are useful for no-calibration baselines.  Calling
        :meth:`calibrate` later replaces every radius using calibration rows,
        while retaining the fit-only centroid/mask bank exactly.
        """
        x = self._validate_features(features, name="features")
        y = self._validate_labels(labels, x.shape[0], name="labels")
        size, mode = self._validate_mask_configuration(mask_size, mask_mode)
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
        separation_stability_scores = between / (within + 1.0)
        masks = self._build_masks(within, separation_stability_scores, size, mode)

        self.n_features_in_ = _DIMENSIONS
        self.units_ = np.asarray(units)
        self.centroids_ = centroids
        self.within_q95_ = within
        # These are fit-only audit attributes.  They are deliberately computed
        # once and the same final score array is passed into mask selection, so
        # a runner cannot reproduce a subtly different ranking downstream.
        self.between_median_separation_ = between
        self.separation_stability_scores_ = separation_stability_scores
        self.masks_ = masks
        self.mask_size_ = size
        self.mask_mode_ = mode
        self.radii_ = self._radii_for_rows(codes, y)
        # Before a calibration pass every row deliberately uses its fit-only
        # initialization.  These fields retain a uniform reporting surface.
        self.radius_calibration_counts_ = np.zeros(self.units_.size, dtype=np.int64)
        self.radius_calibration_fallback_mask_ = np.ones(self.units_.size, dtype=bool)
        self.radius_source_ = "fit"
        self.is_fitted_ = True
        return self

    def calibrate(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
    ) -> "AdaptiveMaskedRangeSearch":
        """Replace all per-unit radii using calibration rows only.

        A fitted unit with no calibration event retains its fit-derived radius
        rather than losing its allocated row.  Per-row calibration counts and
        that explicit fallback mask are exposed for runner reporting.
        Calibration labels absent from the fitted bank are ignored, since they
        have no allocated search row.  Centroids and masks remain fit-only
        state.
        """
        self._require_fitted()
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
            distances = (np.abs(local - self.centroids_[row]) * self.masks_[row]).sum(axis=1)
            radii[row] = int(
                np.ceil(np.percentile(distances, self.radius_percentile, method="linear"))
            )
        self.radii_ = radii
        self.radius_calibration_counts_ = counts
        self.radius_calibration_fallback_mask_ = fallback
        self.radius_source_ = "calibration"
        return self

    def transform(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Return fixed-``[0,1]`` five-bit codes for nine-dimensional rows."""
        self._require_fitted()
        return self.quantizer.transform(self._validate_features(features, name="features"))

    def query(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        *,
        active_row_mask: np.ndarray | Sequence[bool] | None = None,
        active_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Quantize raw features and execute inclusive masked-L1 row search."""
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
        """Search integer codes with an inclusive masked L1 distance.

        A shared ``(U,)`` or per-query ``(N,U)`` active mask is supported,
        with ``active_mask`` retained as an alias for range-search callers.
        ``enabled_abs_lanes`` counts exactly the enabled absolute-difference
        lanes for active unit rows, independently of whether those rows match.
        """
        self._require_fitted()
        if active_row_mask is not None and active_mask is not None:
            raise ValueError("pass only one of active_row_mask and active_mask")
        values = self._validate_codes(codes)
        active = self._validate_active_mask(
            active_row_mask if active_row_mask is not None else active_mask,
            values.shape[0],
        )
        distances = (
            np.abs(values[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :])
            * self.masks_[np.newaxis, :, :]
        ).sum(axis=2)
        matched = active & (distances <= self.radii_[np.newaxis, :])
        candidates = [self.units_[row].copy() for row in matched]
        candidate_count = matched.sum(axis=1).astype(np.int64)
        active_rows = active.sum(axis=1).astype(np.int64)
        enabled_dims = self.masks_.sum(axis=1, dtype=np.int64)
        enabled_abs_lanes = (active.astype(np.int64) * enabled_dims[np.newaxis, :]).sum(axis=1)
        diagnostics = {
            "candidate_count": candidate_count,
            "initial_candidate_count": candidate_count.copy(),
            "final_candidate_count": candidate_count.copy(),
            "active_row_count": active_rows,
            "row_comparisons": active_rows.copy(),
            "enabled_abs_lanes": enabled_abs_lanes.astype(np.int64),
        }
        return candidates, diagnostics

    def memory_accounting(self) -> dict[str, float | int]:
        """Return the fixed 63-bit row payload and enabled-dimension summary."""
        self._require_fitted()
        n_units = int(self.units_.size)
        unit_id_bits = int(np.ceil(np.log2(n_units))) if n_units > 1 else 0
        enabled = self.masks_.sum(axis=1, dtype=np.int64)
        centroid_bits = _DIMENSIONS * _BITS
        mask_bits = _DIMENSIONS
        radius_bits = int(np.ceil(np.log2(_DIMENSIONS * ((1 << _BITS) - 1) + 1)))
        search_payload = centroid_bits + mask_bits + radius_bits
        row_total = search_payload + unit_id_bits + 1
        return {
            "n_units": n_units,
            "n_features": _DIMENSIONS,
            "n_bits": _BITS,
            "centroid_bits_per_row": centroid_bits,
            "mask_bits_per_row": mask_bits,
            "full_scale_radius_bits_per_row": radius_bits,
            "search_payload_bits_per_row": search_payload,
            "unit_id_bits_per_row": unit_id_bits,
            "valid_bits_per_row": 1,
            "row_bits_with_metadata_per_row": row_total,
            "search_payload_total_bits": int(n_units * search_payload),
            "total_bits_with_metadata": int(n_units * row_total),
            "enabled_dimensions_mean": float(enabled.mean()),
            "enabled_dimensions_p95": float(np.percentile(enabled, 95.0, method="linear")),
            "enabled_dimensions_max": int(enabled.max()),
        }

    def _radii_for_rows(self, codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        radii = np.empty(self.units_.size, dtype=np.int64)
        for row, unit in enumerate(self.units_):
            local = codes[labels == unit]
            distances = (np.abs(local - self.centroids_[row]) * self.masks_[row]).sum(axis=1)
            radii[row] = int(
                np.ceil(np.percentile(distances, self.radius_percentile, method="linear"))
            )
        return radii

    @staticmethod
    def _build_masks(
        within: np.ndarray,
        separation_stability_scores: np.ndarray,
        mask_size: int,
        mode: MaskMode,
    ) -> np.ndarray:
        masks = np.zeros((within.shape[0], _DIMENSIONS), dtype=bool)
        if mode == "all_dims":
            masks[:] = True
            return masks
        for row in range(within.shape[0]):
            if mode == "stability":
                score = within[row]
                order = np.lexsort((np.arange(_DIMENSIONS), score))
            else:
                score = separation_stability_scores[row]
                order = np.lexsort((np.arange(_DIMENSIONS), -score))
            masks[row, order[:mask_size]] = True
        return masks

    @staticmethod
    def _between_median_separation(centroids: np.ndarray) -> np.ndarray:
        """Per-unit median L1-coordinate separation, with zero single-unit rows."""
        n_units = centroids.shape[0]
        between = np.zeros_like(centroids, dtype=np.float64)
        for row in range(n_units):
            if n_units > 1:
                between[row] = np.median(
                    np.abs(centroids[np.arange(n_units) != row] - centroids[row]), axis=0
                )
        return between

    @staticmethod
    def _validate_mask_configuration(mask_size: int, mask_mode: str) -> tuple[int, MaskMode]:
        if isinstance(mask_size, bool) or not isinstance(mask_size, (int, np.integer)):
            raise ValueError("mask_size must be one of 3, 5, 7, or 9")
        size = int(mask_size)
        if size not in _MASK_SIZES:
            raise ValueError("mask_size must be one of 3, 5, 7, or 9")
        if mask_mode not in ("all_dims", "stability", "separation_stability"):
            raise ValueError("mask_mode must be all_dims, stability, or separation_stability")
        if mask_mode == "all_dims" and size != _DIMENSIONS:
            raise ValueError("all_dims requires mask_size=9")
        return size, mask_mode  # type: ignore[return-value]

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
            raise RuntimeError("AdaptiveMaskedRangeSearch must be fitted before use")
