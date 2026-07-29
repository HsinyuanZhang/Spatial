"""Per-unit, quantized spatial range search for a CiM/CAM first stage.

The module deliberately contains no learned model and depends only on NumPy.
``AdaptiveRangeSearch`` fits all calibration quantities on its training data:
an unsigned feature quantizer, one integer centroid per unit, and both an L1
ball and an axis-aligned box per unit.  The resulting queries model a spatial
directory by accepting a mask of active rows before doing comparisons.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np


SearchMode = Literal["l1", "box"]


def _as_2d_features(features: np.ndarray, *, name: str = "features") -> np.ndarray:
    """Return a finite two-dimensional floating point feature matrix."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features); got {array.shape}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_labels(labels: Sequence[Hashable] | np.ndarray, n_samples: int) -> np.ndarray:
    """Validate labels without requiring labels to be consecutive integers."""
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError("labels must be one-dimensional and match the number of samples")
    if n_samples == 0:
        raise ValueError("at least one training sample is required")
    return array


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_percentile(percentile: float, *, name: str) -> float:
    value = float(percentile)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be in [0, 100]")
    return value


class UniformUnsignedQuantizer:
    """Training-fitted, per-feature uniform unsigned integer quantizer.

    Parameters
    ----------
    n_bits:
        Number of unsigned code bits.  Five bits is the CiM/CAM default.
    fixed_unit_interval:
        When true, use exactly ``[0, 1]`` for every feature rather than using
        the training extrema.  This is appropriate only for known normalized
        features, and keeps inference independent of held-out values.

    Notes
    -----
    Constant training dimensions are assigned a denominator of one and hence
    map their training value to zero.  Values outside the fitted range are
    clipped to the available code range.
    """

    def __init__(self, n_bits: int = 5, *, fixed_unit_interval: bool = False) -> None:
        if isinstance(n_bits, bool) or int(n_bits) != n_bits or not 1 <= int(n_bits) <= 62:
            raise ValueError("n_bits must be an integer in [1, 62]")
        self.n_bits = int(n_bits)
        self.fixed_unit_interval = bool(fixed_unit_interval)
        self.code_max = (1 << self.n_bits) - 1

    def fit(self, features: np.ndarray) -> "UniformUnsignedQuantizer":
        """Fit per-feature bounds using *only* the supplied training rows."""
        values = _as_2d_features(features)
        if values.shape[0] == 0:
            raise ValueError("at least one training sample is required")
        if self.fixed_unit_interval:
            lower = np.zeros(values.shape[1], dtype=np.float64)
            upper = np.ones(values.shape[1], dtype=np.float64)
        else:
            lower = values.min(axis=0)
            upper = values.max(axis=0)
        width = upper - lower
        # This protects division while retaining an explicit constant-dimension
        # mask for users auditing fitted calibration values.
        self.data_min_ = lower
        self.data_max_ = upper
        self.constant_mask_ = width == 0.0
        self.scale_ = np.ones_like(width, dtype=np.float64)
        np.divide(self.code_max, width, out=self.scale_, where=~self.constant_mask_)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Quantize rows to inclusive integer codes in ``[0, 2**n_bits - 1]``."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("quantizer is not fitted")
        values = _as_2d_features(features)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"features has {values.shape[1]} columns; expected {self.n_features_in_}"
            )
        codes = np.rint((values - self.data_min_) * self.scale_)
        return np.clip(codes, 0, self.code_max).astype(np.int64)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit on training rows then return their quantized codes."""
        return self.fit(features).transform(features)


def fit_integer_centroids(codes: np.ndarray, labels: Sequence[Hashable] | np.ndarray) -> dict[Any, np.ndarray]:
    """Fit one integer ``rint(mean(code))`` centroid for every label.

    The dictionary keys preserve arbitrary (including non-contiguous) labels.
    """
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("codes must have shape (n_samples, n_features)")
    if values.shape[0] == 0:
        raise ValueError("at least one training sample is required")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("codes must be finite numeric values")
    labs = _as_labels(labels, values.shape[0])
    result: dict[Any, np.ndarray] = {}
    for unit in np.unique(labs):
        mask = labs == unit
        result[_python_scalar(unit)] = np.rint(values[mask].mean(axis=0)).astype(np.int64)
    return result


class AdaptiveRangeSearch:
    """Fitted per-unit L1-ball and box candidate search in unsigned code space.

    ``fit`` builds both search structures.  At query time ``mode='l1'`` uses
    ``sum(abs(x-centroid_u)) <= radius_u``; ``mode='box'`` uses inclusive
    coordinate comparisons ``lower_u <= x <= upper_u``.  A query never adds
    nearest rows while ordinary search has candidates.  Nearest-row fallback
    occurs only when requested by ``ensure_nonempty`` after all widening levels
    have failed.
    """

    def __init__(
        self,
        n_bits: int = 5,
        *,
        fixed_unit_interval: bool = False,
        radius_percentile: float = 95.0,
        box_coverage: float = 0.95,
    ) -> None:
        self.quantizer = UniformUnsignedQuantizer(
            n_bits=n_bits, fixed_unit_interval=fixed_unit_interval
        )
        self.radius_percentile = _validate_percentile(
            radius_percentile, name="radius_percentile"
        )
        self.box_coverage = float(box_coverage)
        if not 0.0 <= self.box_coverage <= 1.0:
            raise ValueError("box_coverage must be in [0, 1]")

    @property
    def n_bits(self) -> int:
        """Number of code bits in every feature field."""
        return self.quantizer.n_bits

    def fit(
        self, features: np.ndarray, labels: Sequence[Hashable] | np.ndarray
    ) -> "AdaptiveRangeSearch":
        """Fit quantization, integer centroids, L1 radii, and nested coverage boxes.

        For each unit, a box is formed from the closest ``ceil(coverage * n)``
        integer codes by L1 distance to its integer centroid.  The stable
        distance ordering makes selected prefixes—and consequently boxes—nested
        as ``box_coverage`` increases.
        """
        raw = _as_2d_features(features)
        labs = _as_labels(labels, raw.shape[0])
        codes = self.quantizer.fit_transform(raw)
        unique = np.unique(labs)
        self.units_ = np.asarray([_python_scalar(value) for value in unique])
        centroid_by_unit = fit_integer_centroids(codes, labs)
        self.centroids_ = np.stack(
            [centroid_by_unit[_python_scalar(unit)] for unit in unique], axis=0
        )
        n_units, n_features = self.centroids_.shape
        self.radii_ = np.empty(n_units, dtype=np.int64)
        self.box_lower_ = np.empty((n_units, n_features), dtype=np.int64)
        self.box_upper_ = np.empty((n_units, n_features), dtype=np.int64)
        self.box_training_coverage_ = np.empty(n_units, dtype=np.float64)

        for row, unit in enumerate(unique):
            unit_codes = codes[labs == unit]
            l1_distances = np.abs(unit_codes - self.centroids_[row]).sum(axis=1)
            # ``ceil`` makes a percentile threshold inclusive in integer space.
            self.radii_[row] = int(np.ceil(np.percentile(l1_distances, self.radius_percentile)))

            # Stable tie-breaking by original training order gives every unit a
            # deterministic nested sequence of selected code prefixes.  Unlike
            # widening all faces at once, adding a row can never shrink a box or
            # over-expand unrelated dimensions merely to admit a diagonal row.
            order = np.lexsort((np.arange(unit_codes.shape[0]), l1_distances))
            required = int(np.ceil(self.box_coverage * unit_codes.shape[0]))
            if required == 0:
                # A zero requested coverage is explicitly centroid-only.  This
                # remains nested in every positive-coverage construction below.
                lower = self.centroids_[row].copy()
                upper = self.centroids_[row].copy()
            else:
                selected = unit_codes[order[:required]]
                lower = np.minimum(selected.min(axis=0), self.centroids_[row])
                upper = np.maximum(selected.max(axis=0), self.centroids_[row])
            inside = np.all((unit_codes >= lower) & (unit_codes <= upper), axis=1)
            self.box_lower_[row] = lower
            self.box_upper_[row] = upper
            self.box_training_coverage_[row] = float(inside.mean())

        self.n_features_in_ = raw.shape[1]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Quantize query features with the training-fitted feature ranges."""
        self._check_fitted()
        return self.quantizer.transform(features)

    def fit_transform(
        self, features: np.ndarray, labels: Sequence[Hashable] | np.ndarray
    ) -> np.ndarray:
        """Fit search structures and return training feature codes."""
        self.fit(features, labels)
        return self.transform(features)

    @property
    def radius_bits(self) -> int:
        """Minimum unsigned bits needed to represent the largest fitted radius."""
        self._check_fitted()
        maximum = int(self.radii_.max(initial=0))
        return int(np.ceil(np.log2(maximum + 1))) if maximum else 0

    @property
    def l1_row_bits(self) -> int:
        """Fitted L1-row bits: centroid fields plus the observed-radius field.

        This reflects the smallest field that represents the current fitted
        radii.  Use :attr:`l1_row_bits_full_scale` for a fixed DAC/CAM layout.
        """
        self._check_fitted()
        return self.n_features_in_ * self.n_bits + self.radius_bits

    @property
    def radius_capacity_bits(self) -> int:
        """Bits for a radius field spanning every possible code-space L1 distance.

        A fixed hardware row must represent values through
        ``n_features * (2**n_bits - 1)``, independently of this training set's
        observed radii.
        """
        self._check_fitted()
        maximum = self.n_features_in_ * self.quantizer.code_max
        return int(np.ceil(np.log2(maximum + 1)))

    @property
    def l1_row_bits_full_scale(self) -> int:
        """Fixed full-scale L1-row bits suitable as the default DAC/CAM cost."""
        self._check_fitted()
        return self.n_features_in_ * self.n_bits + self.radius_capacity_bits

    @property
    def box_row_bits(self) -> int:
        """Box-row storage: lower and upper unsigned bound per feature."""
        self._check_fitted()
        return 2 * self.n_features_in_ * self.n_bits

    def hardware_costs(self) -> dict[str, int]:
        """Return fitted and fixed-layout per-row bit accounting.

        ``default_l1_row_bits`` / ``dac_l1_row_bits`` deliberately use the
        full-scale fixed radius field, whereas ``l1_row_bits`` records the
        smaller, training-specific value for calibration analysis.
        """
        return {
            "feature_bits": self.n_features_in_ * self.n_bits,
            "radius_bits": self.radius_bits,
            "l1_row_bits": self.l1_row_bits,
            "radius_capacity_bits": self.radius_capacity_bits,
            "l1_row_bits_full_scale": self.l1_row_bits_full_scale,
            "default_l1_row_bits": self.l1_row_bits_full_scale,
            "dac_l1_row_bits": self.l1_row_bits_full_scale,
            "box_row_bits": self.box_row_bits,
        }

    def query(
        self,
        features: np.ndarray,
        *,
        mode: SearchMode = "l1",
        active_mask: np.ndarray | None = None,
        widening_steps: int = 0,
        ensure_nonempty: bool = False,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Search quantized raw features and return candidates plus diagnostics.

        ``active_mask`` can have shape ``(n_units,)`` (shared directory rows) or
        ``(n_queries, n_units)`` (query-specific rows).  ``widening_steps`` adds
        one code to each permitted L1 radius / box face at each level.  If all
        levels are empty, ``ensure_nonempty=True`` returns the active row with
        the smallest normalized out-of-bound distance.
        """
        return self.query_codes(
            self.transform(features),
            mode=mode,
            active_mask=active_mask,
            widening_steps=widening_steps,
            ensure_nonempty=ensure_nonempty,
        )

    def query_codes(
        self,
        codes: np.ndarray,
        *,
        mode: SearchMode = "l1",
        active_mask: np.ndarray | None = None,
        widening_steps: int = 0,
        ensure_nonempty: bool = False,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Search already-quantized codes; see :meth:`query` for semantics."""
        self._check_fitted()
        if mode not in ("l1", "box"):
            raise ValueError("mode must be 'l1' or 'box'")
        if isinstance(widening_steps, bool) or int(widening_steps) != widening_steps or widening_steps < 0:
            raise ValueError("widening_steps must be a non-negative integer")
        steps = int(widening_steps)
        values = np.asarray(codes)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"codes must have shape (n_queries, {self.n_features_in_})"
            )
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError("codes must be finite numeric values")
        values = np.clip(np.rint(values), 0, self.quantizer.code_max).astype(np.int64)
        active = self._active_rows(active_mask, values.shape[0])

        candidates: list[np.ndarray] = []
        initial_count = np.zeros(values.shape[0], dtype=np.int64)
        final_count = np.zeros(values.shape[0], dtype=np.int64)
        widen_level = np.zeros(values.shape[0], dtype=np.int64)
        fallback = np.zeros(values.shape[0], dtype=bool)
        row_comparisons = np.zeros(values.shape[0], dtype=np.int64)
        # Kept separately from ordinary range membership comparisons: a
        # requested outside-row fallback evaluates every enabled row once more.
        fallback_row_comparisons = np.zeros(values.shape[0], dtype=np.int64)
        active_row_count = active.sum(axis=1).astype(np.int64)

        for query_index, code in enumerate(values):
            mask = active[query_index]
            matched = self._matches(code, mask, mode, level=0)
            row_comparisons[query_index] += int(mask.sum())
            initial_count[query_index] = int(matched.sum())
            level_found = 0
            for level in range(1, steps + 1):
                if np.any(matched):
                    break
                matched = self._matches(code, mask, mode, level=level)
                row_comparisons[query_index] += int(mask.sum())
                if np.any(matched):
                    level_found = level
                    break
            if not np.any(matched) and ensure_nonempty and np.any(mask):
                selected = self._nearest_outside_row(code, mask, mode, level=steps)
                fallback_row_comparisons[query_index] = int(mask.sum())
                matched[selected] = True
                fallback[query_index] = True
                level_found = steps
            final_count[query_index] = int(matched.sum())
            widen_level[query_index] = level_found
            candidates.append(self.units_[matched].copy())

        diagnostics = {
            "initial_candidate_count": initial_count,
            "final_candidate_count": final_count,
            "widen_level": widen_level,
            "fallback": fallback,
            "active_row_count": active_row_count,
            # Counts every enabled row comparison (including unsuccessful
            # progressive levels), but excludes an explicit outside-row
            # fallback scan for backward-compatible normal-search accounting.
            "row_comparisons": row_comparisons,
            "fallback_row_comparisons": fallback_row_comparisons,
            "total_row_comparisons_including_fallback": (
                row_comparisons + fallback_row_comparisons
            ),
        }
        return candidates, diagnostics

    def _check_fitted(self) -> None:
        if not hasattr(self, "centroids_"):
            raise RuntimeError("AdaptiveRangeSearch is not fitted")

    def _active_rows(self, active_mask: np.ndarray | None, n_queries: int) -> np.ndarray:
        n_units = self.units_.shape[0]
        if active_mask is None:
            return np.ones((n_queries, n_units), dtype=bool)
        active = np.asarray(active_mask, dtype=bool)
        if active.shape == (n_units,):
            return np.broadcast_to(active, (n_queries, n_units)).copy()
        if active.shape == (n_queries, n_units):
            return active
        raise ValueError(
            "active_mask must have shape (n_units,) or (n_queries, n_units)"
        )

    def _matches(self, code: np.ndarray, active: np.ndarray, mode: SearchMode, *, level: int) -> np.ndarray:
        if mode == "l1":
            distances = np.abs(self.centroids_ - code).sum(axis=1)
            return active & (distances <= self.radii_ + level)
        lower = np.maximum(self.box_lower_ - level, 0)
        upper = np.minimum(self.box_upper_ + level, self.quantizer.code_max)
        return active & np.all((code >= lower) & (code <= upper), axis=1)

    def _nearest_outside_row(
        self, code: np.ndarray, active: np.ndarray, mode: SearchMode, *, level: int
    ) -> int:
        """Index of an active row with minimum normalized excess distance."""
        if mode == "l1":
            distance = np.abs(self.centroids_ - code).sum(axis=1)
            excess = np.maximum(distance - (self.radii_ + level), 0)
            score = excess / np.maximum(self.radii_ + level, 1)
        else:
            lower = np.maximum(self.box_lower_ - level, 0)
            upper = np.minimum(self.box_upper_ + level, self.quantizer.code_max)
            excess = np.maximum(lower - code, 0) + np.maximum(code - upper, 0)
            widths = np.maximum(upper - lower + 1, 1)
            score = (excess / widths).sum(axis=1)
        score = np.where(active, score, np.inf)
        return int(np.argmin(score))


def evaluate_candidate_search(
    candidates: Sequence[np.ndarray | Sequence[Hashable]],
    true_labels: Sequence[Hashable] | np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Summarize recall, ambiguity, and directory activity of a candidate stage.

    ``true_unit_candidate_recall`` only tests whether each true (possibly
    non-contiguous) label appears in its candidate list.  It does not presume
    candidate lists are predictions.
    """
    labels = np.asarray(true_labels)
    if labels.ndim != 1 or len(candidates) != labels.shape[0]:
        raise ValueError("candidates and true_labels must have the same length")
    counts = np.asarray([len(rows) for rows in candidates], dtype=np.int64)
    recalled = np.asarray(
        [any(candidate == label for candidate in rows) for rows, label in zip(candidates, labels)],
        dtype=bool,
    )
    n = counts.size
    report: dict[str, float | int] = {
        "n_queries": int(n),
        "true_unit_candidate_recall": float(recalled.mean()) if n else 0.0,
        "zero_fraction": float((counts == 0).mean()) if n else 0.0,
        "unique_fraction": float((counts == 1).mean()) if n else 0.0,
        "ambiguous_fraction": float((counts > 1).mean()) if n else 0.0,
        "mean_candidates": float(counts.mean()) if n else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if n else 0.0,
        "max_candidates": int(counts.max()) if n else 0,
        "stage2_activation_fraction": float((counts > 1).mean()) if n else 0.0,
    }
    if diagnostics is not None:
        for key in (
            "widen_level", "fallback", "active_row_count", "row_comparisons",
            "fallback_row_comparisons", "total_row_comparisons_including_fallback",
        ):
            if key not in diagnostics:
                continue
            values = np.asarray(diagnostics[key])
            if values.shape != (n,):
                raise ValueError(f"diagnostics['{key}'] must have shape ({n},)")
        if "widen_level" in diagnostics:
            report["widen_rate"] = float(np.mean(np.asarray(diagnostics["widen_level"]) > 0)) if n else 0.0
        if "fallback" in diagnostics:
            report["fallback_rate"] = float(np.mean(np.asarray(diagnostics["fallback"], dtype=bool))) if n else 0.0
        if "active_row_count" in diagnostics:
            report["mean_active_rows"] = float(np.mean(diagnostics["active_row_count"])) if n else 0.0
            report["p95_active_rows"] = float(np.percentile(diagnostics["active_row_count"], 95)) if n else 0.0
            report["max_active_rows"] = int(np.max(diagnostics["active_row_count"])) if n else 0
        if "row_comparisons" in diagnostics:
            report["mean_row_comparisons"] = float(np.mean(diagnostics["row_comparisons"])) if n else 0.0
            report["p95_row_comparisons"] = float(np.percentile(diagnostics["row_comparisons"], 95)) if n else 0.0
            report["max_row_comparisons"] = int(np.max(diagnostics["row_comparisons"])) if n else 0
            report["total_row_comparisons"] = int(np.sum(diagnostics["row_comparisons"]))
        if "fallback_row_comparisons" in diagnostics:
            fallback_rows = np.asarray(diagnostics["fallback_row_comparisons"])
            report["mean_fallback_row_comparisons"] = float(np.mean(fallback_rows)) if n else 0.0
            report["total_fallback_row_comparisons"] = int(np.sum(fallback_rows))
        if "total_row_comparisons_including_fallback" in diagnostics:
            total_rows = np.asarray(diagnostics["total_row_comparisons_including_fallback"])
            report["mean_row_comparisons_including_fallback"] = float(np.mean(total_rows)) if n else 0.0
            report["total_row_comparisons_including_fallback"] = int(np.sum(total_rows))
    return report


# Concise aliases useful in experimental notebooks while retaining explicit
# class/function names above as the public API.
UnsignedFeatureQuantizer = UniformUnsignedQuantizer
AdaptiveSpatialRangeSearch = AdaptiveRangeSearch
