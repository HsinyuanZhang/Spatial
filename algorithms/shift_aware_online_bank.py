"""Shift-aware online template matching with inverse-aligned updates.

The :class:`~Spatial.algorithms.online_p2p_templates.ConfidenceGatedP2PTemplateBank`
operates on *canonical* (alignment-zero) integer rows.  Its docstring requires
that a caller using shifted matching inverse-align the winning query before it
enters the update — otherwise spatial jitter is averaged into the row and the
template drifts across offsets.  Until now that caller did not exist.

This module closes that gap.  A pre-shifted bank (``2S+1`` copies per unit) is
matched first to find the best unit *and* the best shift.  The query is then
inverse-shifted by the winning shift before the confidence-gated
error-feedback update is applied to the canonical row.  The update primitive
itself is delegated bit-exact to the underlying bank; this wrapper adds only
the alignment bookkeeping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from Spatial.algorithms.online_p2p_templates import ConfidenceGatedP2PTemplateBank
from Spatial.algorithms.shift_match import build_preshift_bank, shift_min_distance_to_bank, shift_vector

__all__ = ["ShiftAwareOnlineBank"]


class ShiftAwareOnlineBank:
    """Confidence-gated online updates with explicit shift inverse-alignment.

    Parameters
    ----------
    templates:
        ``(units, K)`` initial unsigned integer templates — the canonical
        (zero-shift) rows.  Passed straight to the underlying
        :class:`ConfidenceGatedP2PTemplateBank`.
    shift_radius:
        Maximum spatial shift ``S >= 0`` searched per query.  ``0`` reduces to
        the plain canonical bank.
    bits, lr_shift, distance_radius, margin_floor, commit_every:
        Forwarded unchanged to :class:`ConfidenceGatedP2PTemplateBank`.

    Notes
    -----
    The pre-shifted bank is materialized once at construction.  When the
    underlying adaptive templates change (after each accepted update), the
    pre-shifted copies are rebuilt lazily on the next match.  This keeps the
    steady-state match cost at one broadcast L1 over the ``(units, 2S+1, K)``
    bank while preserving the bit-exact canonical update contract.
    """

    def __init__(
        self,
        templates: np.ndarray,
        *,
        shift_radius: int,
        bits: int,
        lr_shift: int,
        distance_radius: np.ndarray,
        margin_floor: np.ndarray,
        commit_every: int = 16,
    ) -> None:
        S = int(shift_radius)
        if S < 0:
            raise ValueError(f"shift_radius must be >= 0, got {shift_radius}")
        self.shift_radius = S
        self._bank = ConfidenceGatedP2PTemplateBank(
            templates,
            bits=bits,
            lr_shift=lr_shift,
            distance_radius=distance_radius,
            margin_floor=margin_floor,
            commit_every=commit_every,
        )
        self._preshift_bank: np.ndarray | None = None
        self.n_shift_resolves: dict[int, int] = {}
        self.n_inverse_aligned_updates = 0

    # ------------------------------------------------------------------
    # Lazy pre-shift bank management
    # ------------------------------------------------------------------

    def _ensure_preshift_bank(self) -> np.ndarray:
        """Rebuild the pre-shifted copies if the canonical templates changed."""
        if self._preshift_bank is None:
            self._preshift_bank = build_preshift_bank(
                self._bank.templates_, shift_radius=self.shift_radius
            )
        return self._preshift_bank

    def _invalidate_preshift_bank(self) -> None:
        self._preshift_bank = None

    # ------------------------------------------------------------------
    # Match + update
    # ------------------------------------------------------------------

    def match(self, query: np.ndarray) -> dict[str, Any]:
        """Match a query against all units and shifts without modifying state.

        Returns the underlying bank's match dict plus the winning shift.
        """
        q = self._bank._validate_query(query)
        if self.shift_radius == 0:
            base = self._bank.match(q)
            base["best_shift"] = 0
            return base

        bank = self._ensure_preshift_bank()
        # (1, units, 2S+1) reduced over shifts → (1, units)
        d_per_shift = shift_min_distance_to_bank(
            q[np.newaxis, :], bank, metric="l1"
        )[0]  # (units,)
        winner = int(np.argmin(d_per_shift))
        best_distance = int(d_per_shift[winner])
        # Recover the winning shift by re-argmining over the shift axis.
        d_unit_shifts = np.sum(
            np.abs(
                q[None, None, :].astype(np.int64)
                - bank[winner].astype(np.int64)[None, :, :]
            ),
            axis=2,
            dtype=np.int64,
        )[0]  # (2S+1,)
        best_shift_idx = int(np.argmin(d_unit_shifts))
        best_shift = int(best_shift_idx - self.shift_radius)
        # match() is a read-only observable; record the alignment decision so
        # the accounting reflects every resolved query, not only updated ones.
        self.n_shift_resolves[best_shift] = (
            self.n_shift_resolves.get(best_shift, 0) + 1
        )

        alternatives = d_per_shift.copy()
        alternatives[winner] = np.iinfo(np.int64).max
        second = int(np.min(alternatives))
        return {
            "winner": winner,
            "best_distance": best_distance,
            "second_distance": second,
            "margin": second - best_distance,
            "best_shift": best_shift,
        }

    def process(
        self,
        query: np.ndarray,
        *,
        stable_path: bool = True,
        fallback_used: bool = False,
        overlap_or_low_confidence: bool = False,
    ) -> dict[str, Any]:
        """Shift-aware match → inverse-align → confidence-gated update.

        The query is inverse-shifted by ``-best_shift`` so that the canonical
        row sees an alignment-zero sample before the error-feedback EMA runs.
        This is the path the
        :class:`ConfidenceGatedP2PTemplateBank` docstring requires but that
        had no implementation before this module.
        """
        q = self._bank._validate_query(query)
        match_info = self.match(q)
        best_shift = int(match_info["best_shift"])

        # Inverse-align: undo the winning shift before the canonical update.
        # shift_vector moves content by +s; to undo a +best_shift alignment we
        # apply -best_shift.
        aligned_q = shift_vector(q.astype(np.float64), -best_shift)
        aligned_q = np.rint(aligned_q).astype(np.int64)
        np.clip(aligned_q, 0, self._bank.code_max, out=aligned_q)

        result = self._bank.process(
            aligned_q,
            stable_path=stable_path,
            fallback_used=fallback_used,
            overlap_or_low_confidence=overlap_or_low_confidence,
        )
        # Surface the alignment decision alongside the gate outcomes.
        result["best_shift"] = best_shift
        result["inverse_aligned"] = bool(best_shift != 0)
        if result["accepted_update"]:
            self.n_inverse_aligned_updates += 1
            # The canonical templates changed; force a rebuild on next match.
            self._invalidate_preshift_bank()
        return result

    # ------------------------------------------------------------------
    # Delegated state + accounting
    # ------------------------------------------------------------------

    def rollback(self, unit: int | None = None) -> None:
        self._bank.rollback(unit)
        self._invalidate_preshift_bank()

    @property
    def templates_(self) -> np.ndarray:
        return self._bank.templates_

    @property
    def n_units(self) -> int:
        return self._bank.n_units

    @property
    def n_features(self) -> int:
        return self._bank.n_features

    def accounting(self) -> dict[str, Any]:
        base = self._bank.accounting()
        base.update(
            {
                "shift_radius": self.shift_radius,
                "rows_per_unit": 2 * self.shift_radius + 1,
                "n_inverse_aligned_updates": int(self.n_inverse_aligned_updates),
                "shift_resolution_counts": dict(self.n_shift_resolves),
            }
        )
        return base
