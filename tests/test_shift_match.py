"""Tests for 1D shift-min matching and CiM pre-shift row banks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.shift_match import (
    build_preshift_bank,
    cim_row_accounting,
    shift_max_dot,
    shift_max_dot_to_bank,
    shift_min_distance,
    shift_min_distance_to_bank,
    shift_vector,
)


def test_shift_vector_positive_and_negative():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(shift_vector(v, 1), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(shift_vector(v, -1), [2.0, 3.0, 4.0, 0.0])
    np.testing.assert_allclose(shift_vector(v, 0), v)
    np.testing.assert_allclose(shift_vector(v, 4, fill=0.0), [0.0, 0.0, 0.0, 0.0])


def test_shift_min_distance_s0_matches_plain():
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d = shift_min_distance(q, t, shift_radius=0, metric="l1")
    expected = np.sum(np.abs(q[:, None, :] - t[None, :, :]), axis=2)
    np.testing.assert_allclose(d, expected)


def test_shift_min_recovers_aligned_template():
    """Query is a +1 shift of template → S=1 L1 distance is 0."""
    t = np.array([[0.0, 1.0, 0.5, 0.2, 0.0]])
    q = shift_vector(t[0], 1)[None, :]
    d0 = shift_min_distance(q, t, shift_radius=0, metric="l1")
    d1 = shift_min_distance(q, t, shift_radius=1, metric="l1")
    assert d0[0, 0] > 0.0
    np.testing.assert_allclose(d1[0, 0], 0.0)


def test_preshift_bank_shape_and_equiv_to_shift_min():
    rng = np.random.default_rng(0)
    templates = rng.random((5, 7))
    queries = rng.random((11, 7))
    S = 2
    bank = build_preshift_bank(templates, shift_radius=S)
    assert bank.shape == (5, 2 * S + 1, 7)
    # Center copy is identity
    np.testing.assert_allclose(bank[:, S, :], templates)

    for metric in ("l1", "l2"):
        d_shift = shift_min_distance(queries, templates, shift_radius=S, metric=metric)
        d_bank = shift_min_distance_to_bank(queries, bank, metric=metric)
        np.testing.assert_allclose(d_shift, d_bank, rtol=1e-12, atol=1e-12)


def test_cim_row_accounting():
    acc = cim_row_accounting(n_units=10, k=7, shift_radius=1, n_bits=4)
    assert acc["rows_per_unit"] == 3
    assert acc["total_rows"] == 30
    assert acc["bits_per_row"] == 28
    assert acc["total_bits"] == 30 * 28
    assert acc["area_multiplier_vs_s0"] == 3.0


def test_shift_max_dot_recovers_aligned_template():
    t = np.array([[0.0, 1.0, 0.5, 0.2, 0.0]])
    q = shift_vector(t[0], 1)[None, :]
    s0 = shift_max_dot(q, t, shift_radius=0)
    s1 = shift_max_dot(q, t, shift_radius=1)
    assert s1[0, 0] > s0[0, 0]
    np.testing.assert_allclose(s1[0, 0], float(np.dot(q[0], shift_vector(t[0], 1))))


def test_shift_max_dot_bank_equiv():
    rng = np.random.default_rng(1)
    templates = rng.random((4, 7))
    queries = rng.random((9, 7))
    S = 2
    bank = build_preshift_bank(templates, shift_radius=S)
    a = shift_max_dot(queries, templates, shift_radius=S)
    b = shift_max_dot_to_bank(queries, bank)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    ac = shift_max_dot(queries, templates, shift_radius=S, normalize=True)
    bc = shift_max_dot_to_bank(queries, bank, normalize=True)
    np.testing.assert_allclose(ac, bc, rtol=1e-12, atol=1e-12)


def test_cosine_prefers_shape_over_energy():
    """Raw dot prefers bright template; cosine prefers matching shape."""
    q = np.array([[1.0, 0.5, 0.0]])
    bright_wrong = np.array([[1.0, 1.0, 1.0]])
    dim_right = np.array([[0.5, 0.25, 0.0]])
    # templates stacked: unit0=bright_wrong, unit1=dim_right
    t = np.vstack([bright_wrong, dim_right])
    raw = shift_max_dot(q, t, shift_radius=0, normalize=False)[0]
    cos = shift_max_dot(q, t, shift_radius=0, normalize=True)[0]
    assert raw[0] > raw[1]
    assert cos[1] > cos[0]


def _reference_shift_min_loop(q, t, S, metric):
    """Original per-shift Python loop, kept as a bit-exact regression oracle."""
    best = None
    for s in range(-S, S + 1):
        shifted = np.stack([shift_vector(row, s) for row in t], axis=0)
        if metric == "l1":
            d = np.sum(np.abs(q[:, None, :] - shifted[None, :, :]), axis=2)
        else:
            d = np.linalg.norm(q[:, None, :] - shifted[None, :, :], axis=2)
        best = d if best is None else np.minimum(best, d)
    return best


def _reference_shift_max_dot_loop(q, t, S, normalize):
    best = None
    if normalize:
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    for s in range(-S, S + 1):
        shifted = np.stack([shift_vector(row, s) for row in t], axis=0)
        if normalize:
            shifted = shifted / np.maximum(
                np.linalg.norm(shifted, axis=1, keepdims=True), 1e-12
            )
        score = q @ shifted.T
        best = score if best is None else np.maximum(best, score)
    return best


def test_shift_min_distance_matches_preshift_bank():
    """Vectorized shift_min_distance must equal the build_preshift_bank path."""
    rng = np.random.default_rng(7)
    templates = rng.random((6, 7))
    queries = rng.random((13, 7))
    S = 2
    bank = build_preshift_bank(templates, shift_radius=S)
    for metric in ("l1", "l2"):
        d_direct = shift_min_distance(queries, templates, shift_radius=S, metric=metric)
        d_bank = shift_min_distance_to_bank(queries, bank, metric=metric)
        np.testing.assert_allclose(d_direct, d_bank, rtol=1e-12, atol=1e-12)


def test_vectorized_min_matches_loop_implementation():
    """Vectorized shift_min_distance must be bit-exact with the loop oracle."""
    rng = np.random.default_rng(11)
    templates = rng.random((5, 9))
    queries = rng.random((17, 9))
    S = 3
    for metric in ("l1", "l2"):
        got = shift_min_distance(queries, templates, shift_radius=S, metric=metric)
        ref = _reference_shift_min_loop(queries, templates, S, metric)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_vectorized_max_dot_matches_loop_implementation():
    """Vectorized shift_max_dot must be bit-exact with the loop oracle."""
    rng = np.random.default_rng(23)
    templates = rng.random((4, 8))
    queries = rng.random((15, 8))
    S = 2
    for normalize in (False, True):
        got = shift_max_dot(
            queries, templates, shift_radius=S, normalize=normalize
        )
        ref = _reference_shift_max_dot_loop(queries, templates, S, normalize)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
