"""Deterministic bounded multi-prototype 5-bit Level-1 candidate search.

The search bank contains a bounded number of integer L1 prototypes for each
unit.  A prototype is a 5-bit centroid and a full-scale L1 radius.  Multiple
matching prototype rows may belong to one unit; :meth:`query_codes` removes
those duplicate unit IDs in stable prototype-row order before returning the
candidate set for waveform refinement.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from .adaptive_range_search import UniformUnsignedQuantizer


PrototypeMode = Literal["temporal", "farthest_l1"]


class AdaptivePrototypeSearch:
    """A bounded, per-unit multi-prototype integer L1 candidate search.

    Parameters
    ----------
    n_bits:
        Quantizer resolution.  The Level-1 contract fixes this at ``5`` bits;
        passing another value is rejected rather than silently changing the
        54-bit row contract.
    radius_percentile:
        Percentile of within-prototype L1 distances used for the inclusive
        integer radius.  The default matches :class:`AdaptiveRangeSearch`.

    Notes
    -----
    The quantizer has a fixed training-independent ``[0, 1]`` range.  For one
    prototype per unit, centroid and radius calculation deliberately matches
    ``AdaptiveRangeSearch``'s per-unit L1 semantics: rounded mean centroid,
    then the ceiling of the requested L1-distance percentile.
    """

    def __init__(self, n_bits: int = 5, *, radius_percentile: float = 95.0):
        if not isinstance(n_bits, (int, np.integer)) or isinstance(n_bits, bool):
            raise ValueError("n_bits must be a positive integer")
        if int(n_bits) != 5:
            raise ValueError("AdaptivePrototypeSearch requires the fixed 5-bit quantizer")
        if not np.isfinite(radius_percentile) or not 0.0 <= float(radius_percentile) <= 100.0:
            raise ValueError("radius_percentile must lie in [0, 100]")

        self.n_bits = int(n_bits)
        self.radius_percentile = float(radius_percentile)
        self.quantizer = UniformUnsignedQuantizer(
            self.n_bits,
            fixed_unit_interval=True,
        )

    def fit(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        labels: np.ndarray | Sequence[Any],
        *,
        prototypes_per_unit: int = 1,
        mode: PrototypeMode = "temporal",
    ) -> "AdaptivePrototypeSearch":
        """Fit a bounded prototype bank from labelled pilot features.

        ``temporal`` partitions each unit's supplied samples into consecutive,
        near-equal chunks.  Thus the caller controls temporal order by passing
        pilot data in chronological order.  ``farthest_l1`` uses deterministic
        farthest-first L1 seeds followed by coordinate-median Lloyd updates;
        all ties select the lower original sample/prototype index.

        Empty requested clusters are not materialised: a unit with fewer than
        ``P`` pilot samples has only as many valid prototype rows as samples.
        The resulting per-unit configured counts are available through
        :meth:`memory_accounting`.
        """
        x = self._validate_features(features, name="features")
        y = np.asarray(labels)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must be one-dimensional and match features")
        if y.size == 0:
            raise ValueError("features and labels must be non-empty")
        if not isinstance(prototypes_per_unit, (int, np.integer)) or isinstance(
            prototypes_per_unit, bool
        ):
            raise ValueError("prototypes_per_unit must be a positive integer")
        if int(prototypes_per_unit) <= 0:
            raise ValueError("prototypes_per_unit must be a positive integer")
        if mode not in ("temporal", "farthest_l1"):
            raise ValueError("mode must be 'temporal' or 'farthest_l1'")

        # ``np.unique`` establishes stable, deterministic unit row blocks even
        # for non-contiguous/non-zero unit labels.
        units = np.unique(y)
        codes = self.quantizer.fit_transform(x)
        row_units: list[Any] = []
        row_centroids: list[np.ndarray] = []
        row_radii: list[int] = []
        row_local_indices: list[int] = []
        training_assignments = np.empty(x.shape[0], dtype=np.int64)
        prototype_training_indices: list[np.ndarray] = []

        for unit in units:
            unit_indices = np.flatnonzero(y == unit)
            unit_codes = codes[unit_indices]
            effective_p = min(int(prototypes_per_unit), unit_indices.size)

            if effective_p == 1:
                local_assignment = np.zeros(unit_indices.size, dtype=np.int64)
                # Exact AdaptiveRangeSearch per-unit L1-centroid semantics.
                prototype_centroids = np.rint(unit_codes.mean(axis=0)).astype(np.int64)
            elif mode == "temporal":
                local_assignment = self._temporal_assignments(unit_indices.size, effective_p)
                prototype_centroids = None
            else:
                local_assignment, prototype_centroids = self._farthest_l1_assignments(
                    unit_codes, effective_p
                )

            # Prototype rows are sorted by their local prototype number, within
            # a unit block sorted by unit ID.  This is the query de-dup order.
            for local_prototype in range(effective_p):
                member_positions = np.flatnonzero(local_assignment == local_prototype)
                if member_positions.size == 0:
                    # The deterministic farthest-L1 repair should prevent this;
                    # keep the guard to ensure no invalid storage rows appear.
                    continue
                member_codes = unit_codes[member_positions]
                if effective_p == 1:
                    centroid = prototype_centroids
                elif mode == "temporal":
                    centroid = np.rint(member_codes.mean(axis=0)).astype(np.int64)
                else:
                    centroid = prototype_centroids[local_prototype]
                distances = np.abs(member_codes - centroid).sum(axis=1)
                radius = int(np.ceil(np.percentile(distances, self.radius_percentile)))

                row_index = len(row_units)
                training_assignments[unit_indices[member_positions]] = row_index
                row_units.append(unit)
                row_centroids.append(np.asarray(centroid, dtype=np.int64))
                row_radii.append(radius)
                row_local_indices.append(local_prototype)
                prototype_training_indices.append(unit_indices[member_positions].copy())

        self.n_features_in_ = x.shape[1]
        self.units_ = np.asarray(units)
        self.centroids_ = np.asarray(row_centroids, dtype=np.int64)
        self.radii_ = np.asarray(row_radii, dtype=np.int64)
        self.prototype_units_ = np.asarray(row_units, dtype=units.dtype)
        self.prototype_indices_within_unit_ = np.asarray(row_local_indices, dtype=np.int64)
        self.training_prototype_assignments_ = training_assignments
        # Both aliases make the pilot-to-prototype directory explicit for code
        # that calls the relation a cluster assignment instead of a row index.
        self.training_cluster_assignments_ = training_assignments
        self.train_prototype_rows_ = training_assignments
        self.prototype_training_indices_ = tuple(prototype_training_indices)
        self.prototypes_per_unit_requested_ = int(prototypes_per_unit)
        self.mode_ = mode
        self.is_fitted_ = True
        return self

    def transform(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Quantize features with the fixed ``[0, 1]`` unsigned quantizer."""
        self._require_fitted()
        x = self._validate_features(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("features have a different number of columns than fit data")
        return self.quantizer.transform(x)

    def query(
        self,
        features: np.ndarray | Sequence[Sequence[float]],
        *,
        active_row_mask: np.ndarray | Sequence[bool] | None = None,
        active_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Quantize raw features and query the prototype bank."""
        return self.query_codes(
            self.transform(features),
            active_row_mask=active_row_mask,
            active_mask=active_mask,
        )

    def query_codes(
        self,
        codes: np.ndarray | Sequence[Sequence[int]],
        *,
        active_row_mask: np.ndarray | Sequence[bool] | None = None,
        active_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Return stable de-duplicated unit candidates for integer code rows.

        Membership uses the inclusive condition ``L1(code, centroid) <=
        radius``.  ``active_row_mask`` (or the compatibility alias
        ``active_mask``) may be a shared ``(R,)`` mask or a per-query ``(N,
        R)`` mask, where ``R`` is the configured prototype-row count.
        Accounting records active *prototype rows*, so a unit reached by
        multiple matching prototypes increases ``prototype_hit_count`` but is
        returned once in the candidate array.
        """
        self._require_fitted()
        query_codes = self._validate_codes(codes)
        n_queries = query_codes.shape[0]
        if active_row_mask is not None and active_mask is not None:
            raise ValueError("pass only one of active_row_mask and active_mask")
        active = self._validate_active_mask(
            active_row_mask if active_row_mask is not None else active_mask,
            n_queries,
        )

        l1_distances = np.abs(
            query_codes[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
        ).sum(axis=2)
        matched = active & (l1_distances <= self.radii_[np.newaxis, :])

        prototype_hit_count = matched.sum(axis=1).astype(np.int64)
        active_row_count = active.sum(axis=1).astype(np.int64)
        distinct_candidate_count = np.empty(n_queries, dtype=np.int64)
        duplicates_removed = np.empty(n_queries, dtype=np.int64)
        candidates: list[np.ndarray] = []

        for query_index in range(n_queries):
            candidate_units: list[Any] = []
            # Preserve the first matching prototype's row order without
            # requiring numerical/contiguous unit IDs.
            for prototype_row in np.flatnonzero(matched[query_index]):
                unit = self.prototype_units_[prototype_row]
                if not any(unit == seen for seen in candidate_units):
                    candidate_units.append(unit)
            candidate_array = np.asarray(candidate_units, dtype=self.prototype_units_.dtype)
            candidates.append(candidate_array)
            distinct_candidate_count[query_index] = candidate_array.size
            duplicates_removed[query_index] = (
                prototype_hit_count[query_index] - candidate_array.size
            )

        diagnostics = {
            "prototype_hit_count": prototype_hit_count,
            "distinct_candidate_count": distinct_candidate_count,
            "candidate_count": distinct_candidate_count.copy(),
            "duplicates_removed": duplicates_removed,
            "active_row_count": active_row_count,
            "active_prototype_row_count": active_row_count,
            "row_comparisons": active_row_count.copy(),
        }
        return candidates, diagnostics

    def memory_accounting(self) -> dict[str, float | int]:
        """Return explicit Level-1 prototype-bank storage accounting.

        For the intended 9-dimensional, 5-bit design, the search payload is
        exactly ``9 * 5 + ceil(log2(9 * 31 + 1)) = 54`` bits: 45 centroid bits
        plus a full-scale L1 radius.  Unit ID and valid bits are listed
        separately because they depend on the fitted number of units.
        """
        self._require_fitted()
        n_units = int(self.units_.size)
        n_rows = int(self.prototype_units_.size)
        centroid_bits = int(self.n_features_in_ * self.n_bits)
        max_l1_distance = int(self.n_features_in_ * self.quantizer.code_max)
        radius_bits = int(np.ceil(np.log2(max_l1_distance + 1)))
        search_payload_row_bits = centroid_bits + radius_bits
        unit_id_bits = int(np.ceil(np.log2(n_units))) if n_units > 1 else 0
        valid_bits_per_prototype = 1
        row_bits_with_metadata = search_payload_row_bits + unit_id_bits + valid_bits_per_prototype

        counts = np.asarray(
            [np.count_nonzero(self.prototype_units_ == unit) for unit in self.units_],
            dtype=np.int64,
        )
        return {
            "n_units": n_units,
            "n_prototype_rows": n_rows,
            "n_features": int(self.n_features_in_),
            "n_bits": int(self.n_bits),
            "centroid_bits_per_prototype": centroid_bits,
            "full_scale_radius_bits_per_prototype": radius_bits,
            "search_payload_bits_per_prototype": search_payload_row_bits,
            "unit_id_bits_per_prototype": unit_id_bits,
            "valid_bits_per_prototype": valid_bits_per_prototype,
            "row_bits_with_metadata_per_prototype": row_bits_with_metadata,
            "search_payload_total_bits": int(n_rows * search_payload_row_bits),
            "total_bits_with_metadata": int(n_rows * row_bits_with_metadata),
            "configured_prototypes_per_unit_mean": float(counts.mean()),
            "configured_prototypes_per_unit_p95": float(np.percentile(counts, 95.0)),
            "configured_prototypes_per_unit_max": int(counts.max()),
            "prototypes_per_unit_requested": int(self.prototypes_per_unit_requested_),
        }

    @staticmethod
    def _temporal_assignments(n_samples: int, n_prototypes: int) -> np.ndarray:
        assignments = np.empty(n_samples, dtype=np.int64)
        for prototype, positions in enumerate(np.array_split(np.arange(n_samples), n_prototypes)):
            assignments[positions] = prototype
        return assignments

    @staticmethod
    def _farthest_l1_assignments(
        unit_codes: np.ndarray, n_prototypes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic L1 farthest-first k-medians with empty-cluster repair."""
        n_samples = unit_codes.shape[0]
        coordinate_median = np.median(unit_codes, axis=0)
        first = int(np.argmin(np.abs(unit_codes - coordinate_median).sum(axis=1)))
        seed_positions = [first]
        min_seed_distance = np.abs(unit_codes - unit_codes[first]).sum(axis=1)
        while len(seed_positions) < n_prototypes:
            # argmax preserves the lowest original input index on ties.
            next_seed = int(np.argmax(min_seed_distance))
            seed_positions.append(next_seed)
            min_seed_distance = np.minimum(
                min_seed_distance,
                np.abs(unit_codes - unit_codes[next_seed]).sum(axis=1),
            )

        centroids = unit_codes[np.asarray(seed_positions)].astype(np.int64, copy=True)
        assignments = np.zeros(n_samples, dtype=np.int64)
        for _ in range(100):
            distances = np.abs(unit_codes[:, np.newaxis, :] - centroids[np.newaxis, :, :]).sum(axis=2)
            updated_assignments = np.argmin(distances, axis=1).astype(np.int64)
            updated_assignments = AdaptivePrototypeSearch._repair_empty_clusters(
                unit_codes, updated_assignments, centroids
            )
            updated_centroids = np.empty_like(centroids)
            for prototype in range(n_prototypes):
                members = unit_codes[updated_assignments == prototype]
                updated_centroids[prototype] = np.rint(np.median(members, axis=0)).astype(np.int64)
            if np.array_equal(updated_assignments, assignments) and np.array_equal(
                updated_centroids, centroids
            ):
                assignments, centroids = updated_assignments, updated_centroids
                break
            assignments, centroids = updated_assignments, updated_centroids
        return assignments, centroids

    @staticmethod
    def _repair_empty_clusters(
        unit_codes: np.ndarray, assignments: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Fill empty clusters deterministically by splitting a populated one."""
        repaired = assignments.copy()
        counts = np.bincount(repaired, minlength=centroids.shape[0])
        for empty_cluster in np.flatnonzero(counts == 0):
            donor_candidates = np.flatnonzero(counts[repaired] > 1)
            # Pick the point farthest from its current cluster centre; ties keep
            # the lower original input index because argmax sees it first.
            donor_distances = np.abs(
                unit_codes[donor_candidates] - centroids[repaired[donor_candidates]]
            ).sum(axis=1)
            donor = int(donor_candidates[int(np.argmax(donor_distances))])
            old_cluster = int(repaired[donor])
            repaired[donor] = empty_cluster
            counts[old_cluster] -= 1
            counts[empty_cluster] += 1
        return repaired

    @staticmethod
    def _validate_features(
        features: np.ndarray | Sequence[Sequence[float]], *, name: str
    ) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty two-dimensional array")
        if not np.isfinite(x).all():
            raise ValueError(f"{name} must contain only finite values")
        return x

    def _validate_codes(self, codes: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        array = np.asarray(codes)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != self.n_features_in_:
            raise ValueError("codes must be a non-empty (n_queries, n_features) array")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("codes must be finite numeric values")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError("codes must contain integer values")
        if (rounded < 0).any() or (rounded > self.quantizer.code_max).any():
            raise ValueError("codes lie outside the quantizer code range")
        return rounded.astype(np.int64)

    def _validate_active_mask(
        self,
        active_row_mask: np.ndarray | Sequence[bool] | None,
        n_queries: int,
    ) -> np.ndarray:
        n_rows = self.prototype_units_.size
        if active_row_mask is None:
            return np.ones((n_queries, n_rows), dtype=bool)
        mask = np.asarray(active_row_mask)
        if mask.dtype != np.bool_:
            raise ValueError("active_row_mask must have boolean dtype")
        if mask.ndim == 1 and mask.shape == (n_rows,):
            return np.broadcast_to(mask, (n_queries, n_rows)).copy()
        if mask.ndim == 2 and mask.shape == (n_queries, n_rows):
            return mask.copy()
        raise ValueError(
            "active_row_mask must have shape (n_prototype_rows,) or "
            "(n_queries, n_prototype_rows)"
        )

    def _require_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("AdaptivePrototypeSearch must be fitted before use")
