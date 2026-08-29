"""Tests for the shift-aware online bank with inverse-aligned updates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.shift_aware_online_bank import ShiftAwareOnlineBank
from Spatial.algorithms.shift_match import shift_vector


def _bank(*, shift_radius: int = 1, commit_every: int = 16) -> ShiftAwareOnlineBank:
    return ShiftAwareOnlineBank(
        np.array([[100, 100, 100], [220, 220, 220]], dtype=np.int64),
        shift_radius=shift_radius,
        bits=8,
        lr_shift=4,
        distance_radius=np.array([80, 80]),
        margin_floor=np.array([10, 10]),
        commit_every=commit_every,
    )


def test_zero_shift_bank_matches_plain_confidence_gated_bank():
    """At S=0 the wrapper reduces to the underlying canonical bank."""
    bank = _bank(shift_radius=0)
    template_before = bank.templates_.copy()
    q = np.array([101, 101, 101], dtype=np.int64)
    result = bank.process(q)
    # The canonical bank's own accumulator test says 16 accepted small-residual
    # updates move the template by exactly 1.
    for _ in range(15):
        bank.process(q)
    np.testing.assert_array_equal(bank.templates_[0], [101, 101, 101])
    # Unit 1 untouched.
    np.testing.assert_array_equal(bank.templates_[1], template_before[1])
    assert result["best_shift"] == 0


def test_inverse_align_then_update_recovers_aligned_template():
    """A query that is a +1 shift of a template should resolve to shift=+1
    and, after inverse-align, feed the canonical update with a sample that
    matches the canonical row on every non-edge coordinate.

    Because a shift zeros-out vacated slots, a shifted query cannot perfectly
    reconstruct the original at the boundary.  This test therefore uses a wide
    template with a distinctive interior so the inverse-aligned query lands
    within the unit's acceptance radius and margin."""
    canonical = np.array([120, 130, 140, 150, 160], dtype=np.int64)
    far = np.array([250, 250, 250, 250, 250], dtype=np.int64)
    # distance_radius must exceed the lost-edge cost (160) so the inverse-aligned
    # query — which carries a 0 at the shifted-out slot — still passes the gate.
    bank = ShiftAwareOnlineBank(
        np.stack([canonical, far]),
        shift_radius=1,
        bits=8,
        lr_shift=4,
        distance_radius=np.array([200, 200]),
        margin_floor=np.array([10, 10]),
    )
    # +1 shift of canonical → [0, 120, 130, 140, 150]; the right edge (160) is
    # lost but the interior is preserved.
    shifted_q = shift_vector(canonical.astype(np.float64), 1).astype(np.int64)
    result = bank.match(shifted_q)
    assert result["best_shift"] == 1
    assert result["best_distance"] == 0  # perfect match at the right shift

    proc = bank.process(shifted_q)
    assert proc["accepted_update"] is True
    assert proc["best_shift"] == 1
    assert proc["inverse_aligned"] is True
    # The inverse-aligned query equals the canonical on indices 0..3 and 0 at
    # index 4 (the lost edge).  A single error-feedback step with lr_shift=4
    # truncates the (0-160)/16 = -10 residual toward zero, leaving the template
    # unchanged at the truncation grain.  Confirm the template did NOT drift
    # toward the [0,...] offset boundary.
    np.testing.assert_array_equal(bank.templates_[0, :4], canonical[:4])


def test_shifted_noise_not_averaged_into_template():
    """A shifted query that does NOT inverse-align to the canonical row must
    still respect the confidence gates — a near-tie must be rejected."""
    bank = _bank(shift_radius=1)
    template_before = bank.templates_.copy()
    # Construct a query whose every shift lands at similar distance to both
    # units → small margin → rejected by the margin floor.
    tied_q = np.array([160, 160, 160], dtype=np.int64)
    result = bank.process(tied_q)
    assert result["margin_gate_pass"] is False
    assert result["accepted_update"] is False
    np.testing.assert_array_equal(bank.templates_, template_before)
    assert bank.accounting()["n_accepted_updates"] == 0


def test_preshift_bank_rebuilt_after_accepted_update():
    """An accepted update must invalidate the cached pre-shift bank so the
    next match sees the new canonical template."""
    bank = _bank(shift_radius=1)
    # First match builds the pre-shift bank.
    bank.match(np.array([101, 101, 101], dtype=np.int64))
    assert bank._preshift_bank is not None
    snapshot = bank._preshift_bank[0, bank.shift_radius, :].copy()  # canonical copy
    # Drive a template change via accepted updates (16 small-residual steps).
    q = np.array([116, 116, 116], dtype=np.int64)
    for _ in range(16):
        bank.process(q)
    # Canonical template moved.
    assert np.any(bank.templates_[0] != snapshot)
    # Pre-shift bank was rebuilt (or invalidated) on the last accepted update.
    # After the last process() call, it is invalidated; the next match rebuilds.
    assert bank._preshift_bank is None
    bank.match(q)
    assert bank._preshift_bank is not None
    np.testing.assert_array_equal(
        bank._preshift_bank[0, bank.shift_radius, :], bank.templates_[0]
    )


def test_rollback_clears_preshift_bank():
    bank = _bank(shift_radius=1, commit_every=1)
    bank.match(np.array([101, 101, 101], dtype=np.int64))
    assert bank._preshift_bank is not None
    # Drive one accepted update so rollback has work to do.
    bank.process(np.array([116, 116, 116], dtype=np.int64))
    bank.rollback(0)
    assert bank._preshift_bank is None
    np.testing.assert_array_equal(bank.templates_[0], [100, 100, 100])


def test_accounting_reports_shift_resolution_counts():
    bank = _bank(shift_radius=1)
    # A handful of perfectly-aligned queries → shift 0 wins.
    for _ in range(3):
        bank.match(np.array([101, 101, 101], dtype=np.int64))
    # One shifted query → shift 1 wins.
    canonical = np.array([100, 100, 100], dtype=np.int64)
    shifted_q = shift_vector(canonical.astype(np.float64), 1).astype(np.int64)
    bank.match(shifted_q)
    acct = bank.accounting()
    assert acct["shift_radius"] == 1
    assert acct["rows_per_unit"] == 3
    assert acct["shift_resolution_counts"].get(0, 0) >= 3
    assert acct["shift_resolution_counts"].get(1, 0) >= 1
