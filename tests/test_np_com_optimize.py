"""Home-gated residual / codebook unique-ID helpers (no recording I/O)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import quantize_signed_adc
from Spatial.experiments.analyze_np_com_lowbit import quantize_span
from Spatial.experiments.analyze_np_com_optimize import (
    codebook_assign,
    codebook_candidates,
    dequantize_span,
    home_then_residual_assign,
    p2p_from_adc_snippets,
)


def _two_home_three_units():
    """Units 1,2 share home 0 and differ in residual; unit 3 is on home 4."""
    homes = np.array([0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4], dtype=np.int64)
    residual = np.array(
        [-8.0, 8.0, -8.2, 8.1, 0.0, 0.2, -7.5, 7.8, -8.4, 8.3, -0.1, 0.3],
        dtype=np.float64,
    )
    labels = np.array([1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3], dtype=np.int64)
    times = np.arange(12, dtype=np.int64)
    geom = np.zeros((5, 2), dtype=np.float64)
    geom[:, 1] = np.arange(5) * 20.0
    return homes, residual, labels, times, geom


def test_dequantize_span_roundtrip_midpoints():
    x = np.array([-90.0, 0.0, 90.0])
    codes = quantize_span(x, n_bits=4, lo=-90.0, hi=90.0)
    rec = dequantize_span(codes, n_bits=4, lo=-90.0, hi=90.0)
    np.testing.assert_allclose(rec[0], -90.0)
    np.testing.assert_allclose(rec[-1], 90.0)
    assert rec.shape == (3,)


def test_home_gate_uses_residual_not_channel_index():
    homes, residual, labels, times, geom = _two_home_three_units()
    out = home_then_residual_assign(
        homes, residual, labels, times, geom, home_radius_um=0.0
    )
    assert out["n_eval"] == 6
    assert out["argmin_accuracy"] == 1.0
    assert out["candidate_recall"] == 1.0
    # Four test events on home 0 see two units; two events on home 4 see one.
    assert abs(out["mean_candidates"] - 10.0 / 6.0) < 1e-9


def test_home_slack_includes_neighbor_channel():
    homes = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    residual = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.1, -0.1], dtype=np.float64)
    labels = np.array([7, 7, 7, 7, 7, 7, 7, 7], dtype=np.int64)
    times = np.arange(8, dtype=np.int64)
    geom = np.zeros((3, 2), dtype=np.float64)
    geom[:, 1] = [0.0, 20.0, 40.0]
    exact = home_then_residual_assign(
        homes, residual, labels, times, geom, home_radius_um=0.0
    )
    slack = home_then_residual_assign(
        homes, residual, labels, times, geom, home_radius_um=20.0
    )
    # Train majority home is 0 or 1 depending on first half (0,0,1,1) — tie
    # broken by bincount argmax = 0. Test events on home 1 miss without slack.
    assert exact["candidate_recall"] < 1.0
    assert slack["candidate_recall"] == 1.0


def test_codebook_majority_and_unseen_bin():
    homes = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    codes = np.array([1, 2, 1, 2, 1, 2, 1, 7], dtype=np.float64)
    labels = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    times = np.arange(8, dtype=np.int64)
    out = codebook_assign(homes, codes, labels, times)
    assert out["n_eval"] == 4
    # Test codes 1,2,1,7 → units 1,2,1, unseen
    assert out["n_correct"] == 3
    assert out["n_unseen_bin"] == 1
    assert out["argmin_accuracy"] == 0.75


def test_codebook_candidates_are_units_observed_in_bin():
    homes = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    codes = np.array([3, 3, 3, 3, 3, 3, 3, 4], dtype=np.float64)
    labels = np.array([8, 8, 8, 9, 8, 9, 8, 9], dtype=np.int64)
    times = np.arange(8, dtype=np.int64)
    out = codebook_candidates(homes, codes, labels, times)
    # Train bin (0,3) = {8,9}; test last event is unseen code 4.
    assert out["candidate_recall"] == 0.75
    assert out["mean_candidates"] == 1.5
    # Majority in (0,3) is 8 → test 8,9,8, unseen = 2/4 unique-ID.
    assert out["argmin_accuracy"] == 0.5
    assert out["argmin_in_candidates"] == 2.0 / 3.0


def test_codebook_delta_unions_neighbor_codes():
    homes = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    codes = np.array([3, 3, 3, 3, 4, 4, 4, 4], dtype=np.float64)
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    times = np.arange(8, dtype=np.int64)
    exact = codebook_candidates(homes, codes, labels, times, code_delta=0)
    slack = codebook_candidates(homes, codes, labels, times, code_delta=1)
    # Train occupies code 3; test occupies code 4 of the same unit.
    assert exact["candidate_recall"] == 0.0
    assert slack["candidate_recall"] == 1.0
    assert slack["mean_candidates"] == 1.0


def test_adc_snippet_p2p_matches_signed_adc():
    rng = np.random.default_rng(0)
    snips = rng.normal(0.0, 1.0, size=(5, 3, 8))
    nbr = np.array([[0, 1, 2], [0, 1, -1], [2, 2, 1], [0, -1, -1], [1, 0, 2]])
    fs = 3.0
    p2p = p2p_from_adc_snippets(snips, nbr, n_bits=5, full_scale=fs)
    for i in range(5):
        for j in range(3):
            if nbr[i, j] < 0:
                assert p2p[i, j] == 0.0
                continue
            recon = quantize_signed_adc(snips[i, j], 5, fs)
            np.testing.assert_allclose(p2p[i, j], recon.max() - recon.min())
