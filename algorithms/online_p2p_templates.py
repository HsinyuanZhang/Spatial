"""Bit-exact confidence-gated online updates for integer P2P templates.

The bank deliberately consumes already-quantized P2P rows.  It does not use
ground-truth labels, refit confidence thresholds, or silently accept tied
matches.  Small residuals are retained in a signed error-feedback accumulator
until their sum is large enough to produce a power-of-two EMA step::

    a <- a + (q - t)
    d <- trunc_toward_zero(a / 2**s)
    t <- clip(t + d)
    a <- a - (t_new - t_old) * 2**s

The frozen anchor, adaptive SRAM shadow, and periodically committed
nonvolatile image are separate states so that rollback and write accounting
remain explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def signed_trunc_shift(values: np.ndarray | int, shift: int) -> np.ndarray:
    """Return signed ``trunc(values / 2**shift)`` using integer operations."""
    if isinstance(shift, bool) or int(shift) != shift or int(shift) < 0:
        raise ValueError("shift must be a non-negative integer")
    x = np.asarray(values, dtype=np.int64)
    magnitude = np.abs(x) >> int(shift)
    return np.where(x >= 0, magnitude, -magnitude).astype(np.int64, copy=False)


class ConfidenceGatedP2PTemplateBank:
    """All-row integer-L1 matcher with error-feedback template updates.

    Parameters
    ----------
    templates:
        Initial unsigned integer templates with shape ``(units, K)``.  These
        become immutable rollback anchors as well as the initial SRAM and
        nonvolatile images.
    bits:
        Descriptor width.  Every query and template coordinate must be in
        ``[0, 2**bits - 1]``.
    lr_shift:
        Power-of-two EMA denominator.  ``4`` represents a nominal ``1/16``
        coefficient, but error feedback prevents sub-16 residuals from being
        discarded forever.
    distance_radius:
        Frozen per-unit maximum accepted L1 distance, fitted on calibration
        data only.
    margin_floor:
        Frozen per-unit minimum ``second_best - best`` L1 margin.  Values must
        be at least one, so an integer-distance tie can never update a row.
    commit_every:
        Commit a dirty unit row to the nonvolatile image only after this many
        accepted events for that unit.

    Notes
    -----
    ``process`` intentionally has no label argument.  Ground truth belongs in
    an external post-run contamination audit, never in deployed update logic.
    A caller using shifted matching must inverse-align the winning query before
    passing it to this canonical-row bank.
    """

    def __init__(
        self,
        templates: np.ndarray,
        *,
        bits: int,
        lr_shift: int,
        distance_radius: np.ndarray,
        margin_floor: np.ndarray,
        commit_every: int = 16,
    ) -> None:
        if isinstance(bits, bool) or int(bits) != bits or not 1 <= int(bits) <= 16:
            raise ValueError("bits must be an integer in [1, 16]")
        if (
            isinstance(lr_shift, bool)
            or int(lr_shift) != lr_shift
            or not 1 <= int(lr_shift) <= 16
        ):
            raise ValueError("lr_shift must be an integer in [1, 16]")
        if (
            isinstance(commit_every, bool)
            or int(commit_every) != commit_every
            or int(commit_every) < 1
        ):
            raise ValueError("commit_every must be a positive integer")

        values = np.asarray(templates)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("templates must have shape (at least 2 units, K>=1)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("templates must contain integer quantizer codes")
        self.bits = int(bits)
        self.code_max = (1 << self.bits) - 1
        values64 = values.astype(np.int64, copy=True)
        if np.any((values64 < 0) | (values64 > self.code_max)):
            raise ValueError("template code is outside the configured bit range")

        n_units = int(values64.shape[0])
        radius = np.asarray(distance_radius, dtype=np.int64)
        margin = np.asarray(margin_floor, dtype=np.int64)
        if radius.shape != (n_units,) or np.any(radius < 0):
            raise ValueError("distance_radius must be one non-negative value per unit")
        if margin.shape != (n_units,) or np.any(margin < 1):
            raise ValueError("margin_floor must be one positive value per unit")

        self.lr_shift = int(lr_shift)
        self.commit_every = int(commit_every)
        self.distance_radius = radius.copy()
        self.margin_floor = margin.copy()
        self.anchor_templates_ = values64.copy()
        self.templates_ = values64.copy()
        self.committed_templates_ = values64.copy()
        self.accumulators_ = np.zeros_like(values64, dtype=np.int64)
        self.accepted_since_commit_ = np.zeros(n_units, dtype=np.int64)
        self.dirty_rows_ = np.zeros(n_units, dtype=bool)

        self.n_predictions_ = 0
        self.n_accepted_ = 0
        self.rejected_distance_ = 0
        self.rejected_margin_ = 0
        self.rejected_control_ = 0
        self.template_coordinate_writes_ = 0
        self.accumulator_coordinate_writes_ = 0
        self.nonvolatile_row_commits_ = 0

    @property
    def n_units(self) -> int:
        return int(self.templates_.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.templates_.shape[1])

    @property
    def accumulator_state_bits_per_coordinate(self) -> int:
        """Bits for the post-feedback remainder ``[-2**s+1, 2**s-1]``."""
        return self.lr_shift + 1

    @property
    def accumulator_intermediate_bits_per_coordinate(self) -> int:
        """Exact signed width for remainder plus one b-bit residual."""
        max_abs = ((1 << self.lr_shift) - 1) + self.code_max
        return 1 + int(np.ceil(np.log2(max_abs + 1)))

    def _validate_query(self, query: np.ndarray) -> np.ndarray:
        values = np.asarray(query)
        if values.shape != (self.n_features,):
            raise ValueError(f"query must have shape ({self.n_features},)")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("query must contain integer quantizer codes")
        q = values.astype(np.int64, copy=False)
        if np.any((q < 0) | (q > self.code_max)):
            raise ValueError("query code is outside the configured bit range")
        return q

    def match(self, query: np.ndarray) -> dict[str, int]:
        """Match one quantized row without modifying state."""
        q = self._validate_query(query)
        distances = np.sum(
            np.abs(self.templates_ - q[np.newaxis, :]), axis=1, dtype=np.int64
        )
        winner = int(np.argmin(distances))
        best = int(distances[winner])
        alternatives = distances.copy()
        alternatives[winner] = np.iinfo(np.int64).max
        second = int(np.min(alternatives))
        return {
            "winner": winner,
            "best_distance": best,
            "second_distance": second,
            "margin": second - best,
        }

    def process(
        self,
        query: np.ndarray,
        *,
        stable_path: bool = True,
        fallback_used: bool = False,
        overlap_or_low_confidence: bool = False,
    ) -> dict[str, int | bool]:
        """Predict, apply confidence/control gates, then optionally update."""
        q = self._validate_query(query)
        result: dict[str, int | bool] = dict(self.match(q))
        winner = int(result["winner"])
        self.n_predictions_ += 1

        control_ok = bool(
            stable_path and not fallback_used and not overlap_or_low_confidence
        )
        distance_ok = int(result["best_distance"]) <= int(
            self.distance_radius[winner]
        )
        margin_ok = int(result["margin"]) >= int(self.margin_floor[winner])
        accepted = bool(control_ok and distance_ok and margin_ok)
        result.update(
            {
                "control_gate_pass": control_ok,
                "distance_gate_pass": distance_ok,
                "margin_gate_pass": margin_ok,
                "accepted_update": accepted,
                "template_coordinates_changed": 0,
                "row_committed": False,
            }
        )
        if not accepted:
            self.rejected_control_ += int(not control_ok)
            self.rejected_distance_ += int(control_ok and not distance_ok)
            self.rejected_margin_ += int(control_ok and distance_ok and not margin_ok)
            return result

        old_template = self.templates_[winner].copy()
        old_accumulator = self.accumulators_[winner].copy()
        intermediate = old_accumulator + (q - old_template)
        delta = signed_trunc_shift(intermediate, self.lr_shift)
        new_template = np.clip(
            old_template + delta, 0, self.code_max
        ).astype(np.int64, copy=False)
        applied_delta = new_template - old_template
        new_accumulator = intermediate - applied_delta * (1 << self.lr_shift)
        remainder_limit = (1 << self.lr_shift) - 1
        if np.any(np.abs(new_accumulator) > remainder_limit):
            raise AssertionError("error-feedback remainder exceeded its exact bound")

        template_changed = new_template != old_template
        accumulator_changed = new_accumulator != old_accumulator
        changed_count = int(np.count_nonzero(template_changed))
        self.templates_[winner] = new_template
        self.accumulators_[winner] = new_accumulator
        self.template_coordinate_writes_ += changed_count
        self.accumulator_coordinate_writes_ += int(
            np.count_nonzero(accumulator_changed)
        )
        self.n_accepted_ += 1
        self.accepted_since_commit_[winner] += 1
        self.dirty_rows_[winner] = bool(
            np.any(self.templates_[winner] != self.committed_templates_[winner])
        )

        row_committed = False
        if self.accepted_since_commit_[winner] >= self.commit_every:
            if self.dirty_rows_[winner]:
                self.committed_templates_[winner] = self.templates_[winner]
                self.nonvolatile_row_commits_ += 1
                row_committed = True
            self.accepted_since_commit_[winner] = 0
            self.dirty_rows_[winner] = False

        result["template_coordinates_changed"] = changed_count
        result["row_committed"] = row_committed
        return result

    def rollback(self, unit: int | None = None) -> None:
        """Restore one unit, or the whole bank, to the immutable fit anchor."""
        if unit is None:
            indices = np.arange(self.n_units, dtype=np.int64)
        else:
            if isinstance(unit, bool) or int(unit) != unit or not 0 <= int(unit) < self.n_units:
                raise ValueError("unit index is outside the template bank")
            indices = np.asarray([int(unit)], dtype=np.int64)
        self.templates_[indices] = self.anchor_templates_[indices]
        self.committed_templates_[indices] = self.anchor_templates_[indices]
        self.accumulators_[indices] = 0
        self.accepted_since_commit_[indices] = 0
        self.dirty_rows_[indices] = False

    def accounting(self) -> dict[str, Any]:
        """Return logical state and write traffic; this is not PPA evidence."""
        state_bits = self.accumulator_state_bits_per_coordinate
        template_bits = self.n_units * self.n_features * self.bits
        accumulator_bits = self.n_units * self.n_features * state_bits
        return {
            "n_predictions": int(self.n_predictions_),
            "n_accepted_updates": int(self.n_accepted_),
            "accepted_update_fraction": (
                self.n_accepted_ / self.n_predictions_ if self.n_predictions_ else 0.0
            ),
            "rejected_control": int(self.rejected_control_),
            "rejected_distance": int(self.rejected_distance_),
            "rejected_margin": int(self.rejected_margin_),
            "template_coordinate_writes": int(self.template_coordinate_writes_),
            "accumulator_coordinate_writes": int(
                self.accumulator_coordinate_writes_
            ),
            "sram_bits_written": int(
                self.template_coordinate_writes_ * self.bits
                + self.accumulator_coordinate_writes_ * state_bits
            ),
            "nonvolatile_row_commits": int(self.nonvolatile_row_commits_),
            "nonvolatile_bits_written": int(
                self.nonvolatile_row_commits_ * self.n_features * self.bits
            ),
            "anchor_template_bits": int(template_bits),
            "adaptive_template_bits": int(template_bits),
            "committed_template_bits": int(template_bits),
            "residual_accumulator_bits": int(accumulator_bits),
            "accumulator_state_bits_per_coordinate": int(state_bits),
            "accumulator_intermediate_bits_per_coordinate": int(
                self.accumulator_intermediate_bits_per_coordinate
            ),
            "lr_shift": int(self.lr_shift),
            "ema_coefficient_reciprocal": int(1 << self.lr_shift),
            "commit_every": int(self.commit_every),
            "dirty_rows": int(np.count_nonzero(self.dirty_rows_)),
        }

