"""Bit-exact tests for confidence-gated P2P template updates."""

from __future__ import annotations

import numpy as np
import pytest

from Spatial.algorithms.online_p2p_templates import (
    ConfidenceGatedP2PTemplateBank,
    signed_trunc_shift,
)


def _bank(*, commit_every: int = 16) -> ConfidenceGatedP2PTemplateBank:
    return ConfidenceGatedP2PTemplateBank(
        np.array([[100, 100], [220, 220]], dtype=np.int64),
        bits=8,
        lr_shift=4,
        distance_radius=np.array([80, 80]),
        margin_floor=np.array([10, 10]),
        commit_every=commit_every,
    )


def test_signed_trunc_shift_is_symmetric():
    np.testing.assert_array_equal(
        signed_trunc_shift(np.array([-17, -16, -15, 0, 15, 16, 17]), 4),
        [-1, -1, 0, 0, 0, 1, 1],
    )


def test_error_feedback_accumulates_small_residuals_until_one_step():
    bank = _bank()
    query = np.array([101, 101], dtype=np.int64)
    for _ in range(15):
        result = bank.process(query)
        assert result["accepted_update"] is True
        np.testing.assert_array_equal(bank.templates_[0], [100, 100])
    bank.process(query)
    np.testing.assert_array_equal(bank.templates_[0], [101, 101])
    np.testing.assert_array_equal(bank.accumulators_[0], [0, 0])


def test_opposite_small_residuals_cancel_without_ratchet():
    bank = _bank()
    for _ in range(20):
        bank.process(np.array([101, 101], dtype=np.int64))
        bank.process(np.array([99, 99], dtype=np.int64))
    np.testing.assert_array_equal(bank.templates_[0], [100, 100])
    np.testing.assert_array_equal(bank.accumulators_[0], [0, 0])


def test_tied_or_control_rejected_event_cannot_modify_state():
    bank = _bank()
    before = bank.templates_.copy()
    tied = bank.process(np.array([160, 160], dtype=np.int64))
    blocked = bank.process(
        np.array([101, 101], dtype=np.int64), fallback_used=True
    )
    assert tied["margin_gate_pass"] is False
    assert tied["accepted_update"] is False
    assert blocked["control_gate_pass"] is False
    assert blocked["accepted_update"] is False
    np.testing.assert_array_equal(bank.templates_, before)
    assert bank.accounting()["n_accepted_updates"] == 0


def test_nonvolatile_commit_is_batched_per_unit():
    bank = _bank(commit_every=16)
    query = np.array([116, 116], dtype=np.int64)
    for _ in range(15):
        result = bank.process(query)
        assert result["row_committed"] is False
    result = bank.process(query)
    assert result["row_committed"] is True
    assert bank.accounting()["nonvolatile_row_commits"] == 1
    assert bank.accounting()["nonvolatile_bits_written"] == 2 * 8
    np.testing.assert_array_equal(bank.committed_templates_[0], bank.templates_[0])


def test_exact_accumulator_width_and_write_accounting():
    bank = _bank()
    # Stored remainder needs signed 5-bit state; intermediate maximum is
    # (2**4-1)+(2**8-1)=270, which needs signed 10-bit arithmetic.
    assert bank.accumulator_state_bits_per_coordinate == 5
    assert bank.accumulator_intermediate_bits_per_coordinate == 10
    bank.process(np.array([101, 100], dtype=np.int64))
    accounting = bank.accounting()
    assert accounting["accumulator_coordinate_writes"] == 1
    assert accounting["template_coordinate_writes"] == 0
    assert accounting["sram_bits_written"] == 5


def test_rollback_restores_anchor_and_clears_adaptive_state():
    bank = _bank(commit_every=1)
    bank.process(np.array([116, 116], dtype=np.int64))
    assert np.any(bank.templates_[0] != bank.anchor_templates_[0])
    bank.rollback(0)
    np.testing.assert_array_equal(bank.templates_[0], [100, 100])
    np.testing.assert_array_equal(bank.committed_templates_[0], [100, 100])
    np.testing.assert_array_equal(bank.accumulators_[0], [0, 0])


def test_bank_rejects_float_or_zero_margin_configuration():
    with pytest.raises(TypeError, match="integer"):
        ConfidenceGatedP2PTemplateBank(
            np.array([[0.0], [1.0]]),
            bits=8,
            lr_shift=4,
            distance_radius=np.array([1, 1]),
            margin_floor=np.array([1, 1]),
        )
    with pytest.raises(ValueError, match="positive"):
        ConfidenceGatedP2PTemplateBank(
            np.array([[0], [1]], dtype=np.int64),
            bits=8,
            lr_shift=4,
            distance_radius=np.array([1, 1]),
            margin_floor=np.array([0, 1]),
        )

