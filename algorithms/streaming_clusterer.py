"""Semi-streaming cluster discovery with online updates and new-unit growth.

Lifecycle
---------
The clusterer has two phases, mirroring the deployment scenario where a device
boots with no labels and must discover units on the fly:

[Phase A — warm-up]   The first ``warmup_fraction`` of events are processed
    offline with the existing :class:`SpatialSOM` to establish initial unit
    centroids, quantization range, and per-unit calibration thresholds.  This
    is the "small non-streaming prefix" the caller authorized.

[Phase B — streaming] Each subsequent event is processed exactly once, in
    chronological order:
      1. COM coarse match — find units whose COM centroid lies within
         ``tau_com`` of the event's COM feature.
      2. P2P fine match — among COM candidates, compute integer-L1 distance to
         each P2P template; pick the nearest.
      3. If the best distance is within the unit's ``tau_p2p`` and the margin
         over the runner-up is at least ``margin_floor``, accept: assign the
         event and apply an error-feedback EMA template update (the same
         bit-exact primitive used by
         :class:`~Spatial.algorithms.online_p2p_templates.ConfidenceGatedP2PTemplateBank`).
      4. Otherwise the event enters a bounded pending pool.  When the pool
         fills, :meth:`drain_pool` runs a miniature SOM over the pooled events
         and promotes any stable sub-cluster (``|c| >= min_new_cluster_size``,
         COM-disjoint from existing units) to a new unit.

Ground-truth labels never enter :meth:`process`.  They may only be used in a
post-run audit, exactly as in the confidence-gated template bank.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from Spatial.algorithms.online_p2p_templates import signed_trunc_shift
from Spatial.algorithms.som_clustering import SpatialSOM

__all__ = ["StreamingClusterer"]


Phase = Literal["warmup", "streaming"]


class StreamingClusterer:
    """Two-phase semi-streaming cluster discovery with online growth.

    Parameters
    ----------
    n_bits:
        Unsigned integer width for the P2P descriptor.  Templates and queries
        are stored as integer codes in ``[0, 2**n_bits - 1]``.
    warmup_fraction:
        Fraction of the initial event batch reserved for offline SOM
        initialization.  Must lie in ``[0, 0.5]``.
    pool_size:
        Pending-pool capacity.  When the pool fills, :meth:`drain_pool` runs.
    min_new_cluster_size:
        Minimum number of pooled events that must coalesce before a new unit
        is promoted.  Guards against noise opening spurious clusters.
    com_grid / p2p_grid:
        SOM grid sizes for the warm-up COM pass and the drain-pool pass.
    com_quantile / p2p_quantile:
        Calibration quantile (in ``[0, 1]``) for the COM radius and the
        per-unit P2P acceptance radius.  Higher is more permissive.
    margin_floor:
        Minimum integer-L1 margin ``d2 - d1`` required to accept an assignment
        for update.  Must be at least 1 so an integer-distance tie can never
        drive an update.
    lr_shift:
        Power-of-two EMA denominator for the error-feedback update
        (``4`` → coefficient ``1/16``).
    recycle_tau_multiplier:
        During pool drain, leftover events that did not promote a new unit are
        given a second chance against existing units with the acceptance radii
        scaled by this factor (``1.5`` = 50% more permissive).
    max_units:
        Optional hard cap on the unit count.  When set and reached, no further
        units are promoted; pooled events are recycled or discarded.
    seed:
        RNG seed for the warm-up and drain-pool SOMs.
    """

    def __init__(
        self,
        *,
        n_bits: int = 8,
        warmup_fraction: float = 0.1,
        pool_size: int = 50,
        min_new_cluster_size: int = 5,
        com_grid: int = 32,
        p2p_grid: int = 8,
        com_quantile: float = 0.95,
        p2p_quantile: float = 0.95,
        margin_floor: int = 1,
        lr_shift: int = 4,
        recycle_tau_multiplier: float = 1.5,
        max_units: int | None = None,
        p2p_fallback_on_com_miss: bool = True,
        com_small_n_k: float = 2.0,
        shrinkage_k: float = 10.0,
        early_promotion_bonus: int = 3,
        seed: int = 42,
    ) -> None:
        if isinstance(n_bits, bool) or int(n_bits) != n_bits or not 1 <= int(n_bits) <= 16:
            raise ValueError("n_bits must be an integer in [1, 16]")
        wf = float(warmup_fraction)
        if not 0.0 <= wf <= 0.5:
            raise ValueError("warmup_fraction must lie in [0, 0.5]")
        if int(pool_size) < 1:
            raise ValueError("pool_size must be >= 1")
        if int(min_new_cluster_size) < 2:
            raise ValueError("min_new_cluster_size must be >= 2")
        if not 0.0 < float(com_quantile) <= 1.0:
            raise ValueError("com_quantile must lie in (0, 1]")
        if not 0.0 < float(p2p_quantile) <= 1.0:
            raise ValueError("p2p_quantile must lie in (0, 1]")
        if int(margin_floor) < 1:
            raise ValueError("margin_floor must be >= 1")
        if isinstance(lr_shift, bool) or int(lr_shift) != lr_shift or not 1 <= int(lr_shift) <= 16:
            raise ValueError("lr_shift must be an integer in [1, 16]")
        if float(recycle_tau_multiplier) < 1.0:
            raise ValueError("recycle_tau_multiplier must be >= 1.0")
        if max_units is not None and int(max_units) < 1:
            raise ValueError("max_units must be >= 1 when set")

        self.n_bits = int(n_bits)
        self.code_max = (1 << self.n_bits) - 1
        self.warmup_fraction = wf
        self.pool_size = int(pool_size)
        self.min_new_cluster_size = int(min_new_cluster_size)
        self.com_grid = int(com_grid)
        self.p2p_grid = int(p2p_grid)
        self.com_quantile = float(com_quantile)
        self.p2p_quantile = float(p2p_quantile)
        self.margin_floor = int(margin_floor)
        self.lr_shift = int(lr_shift)
        self.recycle_tau_multiplier = float(recycle_tau_multiplier)
        self.max_units = None if max_units is None else int(max_units)
        self.p2p_fallback_on_com_miss = bool(p2p_fallback_on_com_miss)
        if float(com_small_n_k) < 0.0:
            raise ValueError("com_small_n_k must be >= 0")
        self.com_small_n_k = float(com_small_n_k)
        if float(shrinkage_k) < 0.0:
            raise ValueError("shrinkage_k must be >= 0")
        self.shrinkage_k = float(shrinkage_k)
        if isinstance(early_promotion_bonus, bool) or int(early_promotion_bonus) < 0:
            raise ValueError("early_promotion_bonus must be a non-negative integer")
        self.early_promotion_bonus = int(early_promotion_bonus)
        self.seed = int(seed)

        # State — populated by warmup().
        self.units_: np.ndarray = np.empty((0,), dtype=np.int64)
        self.com_centroids_: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self.p2p_templates_: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self.p2p_accumulators_: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self.tau_com_: np.ndarray = np.empty((0,), dtype=np.float64)
        self.tau_com_global_: float = 0.0
        self.tau_p2p_: np.ndarray = np.empty((0,), dtype=np.int64)
        self.n_events_per_unit_: np.ndarray = np.empty((0,), dtype=np.int64)
        self.pending_pool_: list[dict[str, np.ndarray]] = []

        # Bookkeeping.
        self.phase_: Phase = "warmup"
        self.n_events_seen_: int = 0
        self.n_events_seen_initial_: int = 0  # set at end of warmup()
        self.n_units_promoted_: int = 0
        self.n_accepted_: int = 0
        self.n_rejected_distance_: int = 0
        self.n_rejected_margin_: int = 0
        self.n_rejected_no_com_candidate_: int = 0
        self.n_recycled_: int = 0
        self.n_pooled_: int = 0
        self.n_template_coordinate_writes_: int = 0
        self.n_accumulator_coordinate_writes_: int = 0
        self.n_units_discarded_: int = 0  # pool sub-clusters rejected as duplicates

    # ------------------------------------------------------------------
    # Phase A: warm-up
    # ------------------------------------------------------------------

    def warmup(
        self,
        com_features: np.ndarray,
        p2p_codes: np.ndarray,
    ) -> "StreamingClusterer":
        """Offline SOM initialization on the warm-up prefix.

        Parameters
        ----------
        com_features:
            ``(n, 2)`` COM feature vectors (e.g. ``soft_localization_features``
            columns 0–1).  Used both for SOM discovery and the coarse gate.
        p2p_codes:
            ``(n, K)`` unsigned integer P2P descriptor codes in
            ``[0, 2**n_bits - 1]``.  Used for the fine gate and templates.
        """
        com = np.asarray(com_features, dtype=np.float64)
        codes = np.asarray(p2p_codes, dtype=np.int64)
        if com.ndim != 2 or com.shape[1] != 2:
            raise ValueError("com_features must have shape (n, 2)")
        if codes.ndim != 2:
            raise ValueError("p2p_codes must be 2D")
        if com.shape[0] != codes.shape[0]:
            raise ValueError("com_features and p2p_codes row count mismatch")
        if codes.size == 0:
            raise ValueError("warm-up requires at least one event")
        if np.any((codes < 0) | (codes > self.code_max)):
            raise ValueError("p2p code is outside the configured bit range")

        som = SpatialSOM(
            grid_size=max(2, min(self.com_grid, len(com))),
            alpha=16,
            beta=8,
            batch_samples=min(2000, len(com)),
            max_epochs=40,
            seed=self.seed,
        )
        som.train(com, verbose=False)
        labels = som.assign(com)

        units = np.unique(labels)
        # Renumber to a contiguous range starting at 0.
        remap = {int(u): i for i, u in enumerate(units)}
        labels = np.array([remap[int(u)] for u in labels], dtype=np.int64)
        units = np.arange(len(remap), dtype=np.int64)

        com_centroids = np.stack(
            [com[labels == u].mean(axis=0) for u in units], axis=0
        )
        p2p_templates = np.stack(
            [
                np.rint(codes[labels == u].astype(np.float64).mean(axis=0)).astype(np.int64)
                for u in units
            ],
            axis=0,
        )

        # Shrinkage centroid estimation.  Units with few warm-up samples (e.g.
        # n=7) have unreliable sample means; shrink them toward the pooled
        # global mean to reduce outlier-centroid risk.  The weight is
        # n/(n+shrinkage_k), so a 7-sample unit with k=10 gets weight 0.41
        # (mostly global), while a 50-sample unit gets 0.83 (mostly own data).
        if self.shrinkage_k > 0.0 and len(units) > 1:
            global_com_mean = com.mean(axis=0)
            global_p2p_mean = np.rint(
                codes.astype(np.float64).mean(axis=0)
            ).astype(np.int64)
            for u in units:
                n_u = int(np.sum(labels == u))
                w = n_u / (n_u + self.shrinkage_k)
                idx = int(u)
                com_centroids[idx] = w * com_centroids[idx] + (1.0 - w) * global_com_mean
                p2p_templates[idx] = np.rint(
                    w * p2p_templates[idx].astype(np.float64)
                    + (1.0 - w) * global_p2p_mean.astype(np.float64)
                ).astype(np.int64)

        # Calibrate per-unit COM radius.  A global scalar tau_com under-fits
        # small / peripheral units whose intra-unit COM spread is large (e.g.
        # a unit with only 7 warm-up events).  Two corrections are applied:
        #   1. per-unit radius = max(own-spread quantile, global floor);
        #   2. small-n inflation: warm-up with n samples under-estimates the
        #      true spread by ~sqrt(n/n_true); multiply by (1 + k/sqrt(n)) so
        #      a 7-sample unit gets ~1.76× (k=2), converging to 1.0× as n→∞.
        tau_com_global = 0.0
        if len(units) > 0:
            all_com_dists = []
            for u in units:
                mask = labels == u
                if mask.sum() > 0:
                    all_com_dists.append(
                        np.linalg.norm(com[mask] - com_centroids[int(u)], axis=1)
                    )
            pooled = (
                np.concatenate(all_com_dists) if all_com_dists else np.array([0.0])
            )
            tau_com_global = max(float(np.quantile(pooled, self.com_quantile)), 1e-8)
        tau_com = np.full(len(units), tau_com_global, dtype=np.float64)
        for u in units:
            mask = labels == u
            n_u = int(mask.sum())
            if n_u < 2:
                continue
            intra = np.linalg.norm(com[mask] - com_centroids[int(u)], axis=1)
            own = float(np.quantile(intra, self.com_quantile))
            # Small-n inflation: the warm-up spread under-estimates the true
            # tail.  The factor (1 + k/sqrt(n)) corrects this; k=2 is
            # calibrated on HJ 16ch data where the stream/warmup q95 ratio
            # averages ~1.0 for n≥50 but reaches 1.8-2.1 for n<30.
            inflation = 1.0 + self.com_small_n_k / max(np.sqrt(n_u), 1.0)
            tau_com[int(u)] = max(own * inflation, tau_com_global)

        # Calibrate per-unit integer-L1 P2P radius.
        tau_p2p = np.zeros(len(units), dtype=np.int64)
        for u in units:
            mask = labels == u
            if mask.sum() == 0:
                continue
            intra = np.sum(
                np.abs(codes[mask].astype(np.int64) - p2p_templates[int(u)]),
                axis=1,
                dtype=np.int64,
            )
            tau_p2p[int(u)] = max(int(np.quantile(intra, self.p2p_quantile)), 1)

        self.units_ = units
        self.com_centroids_ = com_centroids
        self.p2p_templates_ = p2p_templates
        self.p2p_accumulators_ = np.zeros_like(p2p_templates)
        self.tau_com_ = tau_com
        self.tau_com_global_ = tau_com_global
        self.tau_p2p_ = tau_p2p
        self.n_events_per_unit_ = np.array(
            [int(np.sum(labels == u)) for u in units], dtype=np.int64
        )
        self.phase_ = "streaming"
        self.n_events_seen_ = int(len(com))
        self.n_events_seen_initial_ = int(len(com))
        return self

    # ------------------------------------------------------------------
    # Phase B: streaming
    # ------------------------------------------------------------------

    def _com_candidates(self, com_x: np.ndarray) -> np.ndarray:
        """Indices into ``self.units_`` whose COM centroid is within tau_com."""
        if self.units_.size == 0:
            return np.empty((0,), dtype=np.int64)
        d = np.linalg.norm(self.com_centroids_ - com_x[None, :], axis=1)
        return np.where(d <= self.tau_com_)[0]

    def _match_p2p(
        self, p2p_x: np.ndarray, cand: np.ndarray
    ) -> tuple[int, int, int, int]:
        """Best, second-best, winner (row index among candidates), margin."""
        if cand.size == 0:
            return -1, -1, -1, 0
        dists = np.sum(
            np.abs(
                p2p_x[None, :].astype(np.int64)
                - self.p2p_templates_[cand].astype(np.int64)
            ),
            axis=1,
            dtype=np.int64,
        )
        if cand.size == 1:
            return int(dists[0]), -1, 0, np.iinfo(np.int64).max
        order = np.argsort(dists, kind="stable")
        best_row = int(order[0])
        second_row = int(order[1])
        best = int(dists[best_row])
        second = int(dists[second_row])
        return best, second, best_row, second - best

    def _apply_update(self, winner_unit_idx: int, p2p_x: np.ndarray) -> None:
        """Bit-exact error-feedback EMA update (mirrors ConfidenceGatedP2PTemplateBank)."""
        row = int(winner_unit_idx)
        old_t = self.p2p_templates_[row]
        old_a = self.p2p_accumulators_[row]
        intermediate = old_a + (p2p_x.astype(np.int64) - old_t)
        delta = signed_trunc_shift(intermediate, self.lr_shift)
        new_t = np.clip(old_t + delta, 0, self.code_max).astype(np.int64)
        applied = new_t - old_t
        new_a = intermediate - applied * (1 << self.lr_shift)
        remainder_limit = (1 << self.lr_shift) - 1
        if np.any(np.abs(new_a) > remainder_limit):
            raise AssertionError("error-feedback remainder exceeded its exact bound")
        template_changed = np.count_nonzero(new_t != old_t)
        accumulator_changed = np.count_nonzero(new_a != old_a)
        self.p2p_templates_[row] = new_t
        self.p2p_accumulators_[row] = new_a
        self.n_template_coordinate_writes_ += int(template_changed)
        self.n_accumulator_coordinate_writes_ += int(accumulator_changed)

    def process(self, com_x: np.ndarray, p2p_x: np.ndarray) -> dict[str, Any]:
        """Process one streaming event: assign (+ update) or pool.

        No label argument is accepted.  The returned dict carries the
        assignment and the gate outcomes so an external audit can reconstruct
        the decision path without ground truth.

        Returns
        -------
        dict with keys:
            ``assigned_unit``: int unit id, or ``-1`` if pooled or unmatched.
            ``outcome``: ``"accepted"``, ``"pooled"``, or ``"recycled"``.
            ``winner_row``: row index among candidates (or ``-1``).
            ``best_distance`` / ``second_distance`` / ``margin``: int L1 values.
            ``pool_drained``: True if this event triggered a drain.
        """
        if self.phase_ != "streaming":
            raise RuntimeError("call warmup() before process()")
        com_x = np.asarray(com_x, dtype=np.float64).ravel()
        if com_x.shape != (2,):
            raise ValueError("com_x must be a length-2 vector")
        p2p_x = np.asarray(p2p_x, dtype=np.int64).ravel()
        if p2p_x.shape != (self.p2p_templates_.shape[1],):
            raise ValueError("p2p_x width mismatch")
        if np.any((p2p_x < 0) | (p2p_x > self.code_max)):
            raise ValueError("p2p code is outside the configured bit range")

        self.n_events_seen_ += 1
        result: dict[str, Any] = {
            "assigned_unit": -1,
            "outcome": "pooled",
            "winner_row": -1,
            "best_distance": -1,
            "second_distance": -1,
            "margin": 0,
            "pool_drained": False,
        }

        cand = self._com_candidates(com_x)
        if cand.size == 0:
            # Optimization C: COM gate missed entirely.  Before pooling, try a
            # full P2P nearest-neighbour match against all units — if the P2P
            # distance and margin are strong, the COM miss was a coarse-gate
            # false negative (common for under-represented units whose centroid
            # is slightly off), not a genuinely unknown unit.  Only when the
            # P2P fallback also fails does the event enter the pending pool.
            if self.p2p_fallback_on_com_miss:
                all_idx = np.arange(self.units_.size, dtype=np.int64)
                best, second, winner_row, margin = self._match_p2p(
                    p2p_x, all_idx
                )
                winner_unit_idx = int(winner_row)
                if (
                    best <= int(self.tau_p2p_[winner_unit_idx])
                    and margin >= self.margin_floor
                ):
                    self.n_accepted_ += 1
                    self.n_events_per_unit_[winner_unit_idx] += 1
                    self._apply_update(winner_unit_idx, p2p_x)
                    result["assigned_unit"] = int(self.units_[winner_unit_idx])
                    result["winner_row"] = winner_row
                    result["best_distance"] = best
                    result["second_distance"] = second
                    result["margin"] = margin
                    result["outcome"] = "accepted_p2p_fallback"
                    return result
            self.n_rejected_no_com_candidate_ += 1
            self._push_pool(com_x, p2p_x)
            drained = self._maybe_drain()
            result["pool_drained"] = drained
            return result

        best, second, winner_row, margin = self._match_p2p(p2p_x, cand)
        winner_unit_idx = int(cand[winner_row])
        result["winner_row"] = winner_unit_idx
        result["best_distance"] = best
        result["second_distance"] = second
        result["margin"] = margin

        distance_ok = best <= int(self.tau_p2p_[winner_unit_idx])
        margin_ok = margin >= self.margin_floor
        if distance_ok and margin_ok:
            self.n_accepted_ += 1
            self.n_events_per_unit_[winner_unit_idx] += 1
            self._apply_update(winner_unit_idx, p2p_x)
            result["assigned_unit"] = int(self.units_[winner_unit_idx])
            result["outcome"] = "accepted"
            return result

        if not distance_ok:
            self.n_rejected_distance_ += 1
        else:
            self.n_rejected_margin_ += 1
        self._push_pool(com_x, p2p_x)
        drained = self._maybe_drain()
        result["pool_drained"] = drained
        return result

    def _push_pool(self, com_x: np.ndarray, p2p_x: np.ndarray) -> None:
        self.pending_pool_.append({"com": com_x.copy(), "p2p": p2p_x.copy()})
        self.n_pooled_ += 1

    def _maybe_drain(self) -> bool:
        if len(self.pending_pool_) < self.pool_size:
            return False
        self.drain_pool()
        return True

    # ------------------------------------------------------------------
    # New-unit promotion from the pending pool
    # ------------------------------------------------------------------

    def drain_pool(self) -> dict[str, int]:
        """Cluster the pending pool and promote stable sub-clusters to units.

        Returns
        -------
        dict with counts: ``n_promoted``, ``n_recycled``, ``n_retained``,
        ``n_discarded_duplicate``.
        """
        if not self.pending_pool_:
            return {
                "n_promoted": 0,
                "n_recycled": 0,
                "n_retained": 0,
                "n_discarded_duplicate": 0,
            }

        pool_com = np.stack([e["com"] for e in self.pending_pool_], axis=0)
        pool_p2p = np.stack([e["p2p"] for e in self.pending_pool_], axis=0)
        self.pending_pool_.clear()

        # Mini SOM on COM features to find candidate sub-clusters.
        grid = max(2, min(self.p2p_grid, int(np.sqrt(len(pool_com)))))
        som = SpatialSOM(
            grid_size=grid,
            alpha=16,
            beta=4,
            batch_samples=min(2000, len(pool_com)),
            max_epochs=30,
            seed=self.seed + self.n_events_seen_,
        )
        som.train(pool_com, verbose=False)
        local_labels = som.assign(pool_com)

        n_promoted = 0
        n_recycled = 0
        n_discarded_duplicate = 0
        retain_indices: list[int] = []

        # Adaptive early promotion: when few events have been processed
        # relative to the warm-up, undiscovered units are the dominant loss
        # source, so lower the promotion bar.  The bonus decays as
        # min(1, warmup_n / n_events_seen) → 0 once we've seen ~1× the warm-up
        # volume in the stream, converging to the configured min_new_cluster_size.
        warmup_n = max(int(self.n_events_seen_) - self.n_events_seen_initial_, 1)
        stream_seen = max(int(self.n_events_seen_) - warmup_n, 0)
        early_factor = min(1.0, warmup_n / max(stream_seen, 1))
        early_bonus = int(np.floor(self.early_promotion_bonus * early_factor))
        effective_min_size = max(2, self.min_new_cluster_size - early_bonus)

        for sub in np.unique(local_labels):
            sub_idx = np.where(local_labels == sub)[0]
            if sub_idx.size < effective_min_size:
                retain_indices.extend(int(i) for i in sub_idx)
                continue

            sub_com = pool_com[sub_idx].mean(axis=0)
            sub_p2p = np.rint(
                pool_p2p[sub_idx].astype(np.float64).mean(axis=0)
            ).astype(np.int64)

            # Duplicate guard: reject if an existing unit is within tau_com.
            if self.units_.size > 0:
                d_existing = np.linalg.norm(
                    self.com_centroids_ - sub_com[None, :], axis=1
                )
                if np.any(d_existing <= self.tau_com_):
                    n_discarded_duplicate += 1
                    retain_indices.extend(int(i) for i in sub_idx)
                    continue

            if self.max_units is not None and self.units_.size >= self.max_units:
                n_discarded_duplicate += 1
                retain_indices.extend(int(i) for i in sub_idx)
                continue

            self._promote_unit(sub_com, sub_p2p)
            n_promoted += 1

        # Second-chance recycle for leftovers, then retain the rest.
        if retain_indices:
            recycle_idx, retained_idx = self._recycle_events(
                pool_com[retain_indices], pool_p2p[retain_indices]
            )
            n_recycled = int(recycle_idx)
            for i in retained_idx:
                self.pending_pool_.append(
                    {"com": pool_com[retain_indices[i]].copy(),
                     "p2p": pool_p2p[retain_indices[i]].copy()}
                )

        self.n_units_promoted_ += n_promoted
        self.n_units_discarded_ += n_discarded_duplicate
        self.n_recycled_ += n_recycled
        return {
            "n_promoted": n_promoted,
            "n_recycled": n_recycled,
            "n_retained": len(self.pending_pool_),
            "n_discarded_duplicate": n_discarded_duplicate,
        }

    def _promote_unit(self, com_centroid: np.ndarray, p2p_template: np.ndarray) -> None:
        """Append a new unit with calibrated thresholds."""
        new_id = int(self.units_.max() + 1) if self.units_.size else 0
        self.units_ = np.append(self.units_, np.int64(new_id))
        self.com_centroids_ = np.vstack([self.com_centroids_, com_centroid[None, :]])
        self.p2p_templates_ = np.vstack([self.p2p_templates_, p2p_template[None, :]])
        self.p2p_accumulators_ = np.vstack(
            [self.p2p_accumulators_, np.zeros_like(p2p_template[None, :])]
        )
        # Conservative initial radii: P2P full-scale / 2 and the global COM
        # floor ensure the new unit can capture its own early events; both
        # tighten as EMA updates and per-unit calibration accumulate.
        new_tau = max(self.p2p_templates_.shape[1] * self.code_max // 2, 1)
        self.tau_p2p_ = np.append(self.tau_p2p_, np.int64(new_tau))
        new_tau_com = max(float(self.tau_com_global_), 1e-8)
        self.tau_com_ = np.append(self.tau_com_, np.float64(new_tau_com))
        self.n_events_per_unit_ = np.append(self.n_events_per_unit_, np.int64(0))

    def _recycle_events(
        self, com: np.ndarray, p2p: np.ndarray
    ) -> tuple[int, list[int]]:
        """Second-chance assignment with relaxed radii. Returns (n_recycled, retained_local_idx)."""
        if self.units_.size == 0 or len(com) == 0:
            return 0, list(range(len(com)))
        n_recycled = 0
        retained: list[int] = []
        relaxed_tau = (self.tau_p2p_.astype(np.float64) * self.recycle_tau_multiplier)
        for i in range(len(com)):
            d_com = np.linalg.norm(self.com_centroids_ - com[i][None, :], axis=1)
            cand = np.where(d_com <= self.tau_com_ * self.recycle_tau_multiplier)[0]
            if cand.size == 0:
                retained.append(i)
                continue
            dists = np.sum(
                np.abs(
                    p2p[i][None, :].astype(np.int64)
                    - self.p2p_templates_[cand].astype(np.int64)
                ),
                axis=1,
                dtype=np.int64,
            )
            order = np.argsort(dists, kind="stable")
            best_row = int(order[0])
            best = int(dists[best_row])
            winner_idx = int(cand[best_row])
            if best <= int(relaxed_tau[winner_idx]):
                self.n_events_per_unit_[winner_idx] += 1
                self._apply_update(winner_idx, p2p[i])
                n_recycled += 1
            else:
                retained.append(i)
        return n_recycled, retained

    # ------------------------------------------------------------------
    # Batch convenience + accounting
    # ------------------------------------------------------------------

    def assign_all(
        self, com_features: np.ndarray, p2p_codes: np.ndarray
    ) -> np.ndarray:
        """Assign a batch of events without updating templates or pooling.

        Convenience for evaluation: COM coarse match then P2P fine match, with
        ``-1`` for events that pass neither gate.  Does not mutate state.
        """
        com = np.asarray(com_features, dtype=np.float64)
        codes = np.asarray(p2p_codes, dtype=np.int64)
        if com.ndim != 2 or com.shape[1] != 2:
            raise ValueError("com_features must have shape (n, 2)")
        if codes.ndim != 2:
            raise ValueError("p2p_codes must be 2D")
        if com.shape[0] != codes.shape[0]:
            raise ValueError("com_features and p2p_codes row count mismatch")
        out = np.full(len(com), -1, dtype=np.int64)
        for i in range(len(com)):
            cand = self._com_candidates(com[i])
            if cand.size == 0:
                continue
            best, _, winner_row, _ = self._match_p2p(codes[i], cand)
            if best <= int(self.tau_p2p_[int(cand[winner_row])]):
                out[i] = int(self.units_[int(cand[winner_row])])
        return out

    def accounting(self) -> dict[str, Any]:
        """Emit decision counts and write traffic; not PPA evidence."""
        return {
            "phase": self.phase_,
            "n_units": int(self.units_.size),
            "n_events_seen": int(self.n_events_seen_),
            "n_accepted": int(self.n_accepted_),
            "n_rejected_distance": int(self.n_rejected_distance_),
            "n_rejected_margin": int(self.n_rejected_margin_),
            "n_rejected_no_com_candidate": int(self.n_rejected_no_com_candidate_),
            "n_pooled": int(self.n_pooled_),
            "n_recycled": int(self.n_recycled_),
            "n_units_promoted": int(self.n_units_promoted_),
            "n_units_discarded_duplicate": int(self.n_units_discarded_),
            "n_pending_pool": int(len(self.pending_pool_)),
            "n_template_coordinate_writes": int(self.n_template_coordinate_writes_),
            "n_accumulator_coordinate_writes": int(self.n_accumulator_coordinate_writes_),
            "tau_com_global": float(self.tau_com_global_),
            "tau_com_mean": float(np.mean(self.tau_com_)) if self.tau_com_.size else 0.0,
            "tau_com_max": float(np.max(self.tau_com_)) if self.tau_com_.size else 0.0,
            "com_small_n_k": float(self.com_small_n_k),
            "shrinkage_k": float(self.shrinkage_k),
            "early_promotion_bonus": int(self.early_promotion_bonus),
            "lr_shift": int(self.lr_shift),
            "ema_coefficient_reciprocal": int(1 << self.lr_shift),
            "margin_floor": int(self.margin_floor),
            "n_bits": int(self.n_bits),
            "n_events_per_unit": self.n_events_per_unit_.tolist(),
        }
