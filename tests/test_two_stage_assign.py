"""Unit tests for COM∧P2P → waveform two-stage assignment."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.two_stage_assign import (
    fit_unit_centroids,
    fit_unit_code_centroids,
    calibrate_thresholds,
    calibrate_thresholds_codes,
    stage1_candidates,
    assign_nearest,
    two_stage_assign,
    two_stage_assign_code_p2p,
    stage1_normalized_score,
)


def test_fit_unit_centroids():
    feats = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    cents = fit_unit_centroids(feats, labels)
    np.testing.assert_allclose(cents[0], [1.0, 0.0])
    np.testing.assert_allclose(cents[1], [1.0, 2.0])


def test_calibrate_thresholds_percentile():
    # Unit 0 around origin, unit 1 around (10,0)
    rng = np.random.default_rng(0)
    com = np.vstack(
        [
            rng.normal(0.0, 0.1, size=(50, 2)),
            rng.normal([10.0, 0.0], 0.1, size=(50, 2)),
        ]
    )
    p2p = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (50, 1)),
            np.tile([0.0, 1.0, 0.0], (50, 1)),
        ]
    ).astype(np.float64)
    # Add tiny noise to p2p for non-zero intra distances
    p2p = p2p + rng.normal(0, 0.01, size=p2p.shape)
    labels = np.array([0] * 50 + [1] * 50)
    tau_com, tau_p2p = calibrate_thresholds(com, p2p, labels, percentile=95.0)
    assert tau_com > 0
    assert tau_p2p > 0
    # Most same-unit spikes should pass their own centroid
    cents_com = fit_unit_centroids(com, labels)
    d0 = np.linalg.norm(com[:50] - cents_com[0], axis=1)
    assert np.mean(d0 < tau_com) >= 0.9


def test_stage1_requires_both_com_and_p2p():
    com_cents = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
    p2p_cents = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0])}
    tau_com, tau_p2p = 0.2, 0.2

    # Close in COM to unit 0, but P2P matches unit 1 → no candidates
    cands = stage1_candidates(
        com_x=np.array([0.05, 0.0]),
        p2p_x=np.array([0.0, 1.0, 0.0]),
        com_centroids=com_cents,
        p2p_centroids=p2p_cents,
        tau_com=tau_com,
        tau_p2p=tau_p2p,
    )
    assert cands == []

    # Passes both for unit 0
    cands2 = stage1_candidates(
        com_x=np.array([0.05, 0.0]),
        p2p_x=np.array([1.0, 0.0, 0.0]),
        com_centroids=com_cents,
        p2p_centroids=p2p_cents,
        tau_com=tau_com,
        tau_p2p=tau_p2p,
    )
    assert cands2 == [0]


def test_two_stage_uses_waveform_among_candidates():
    # Identical spatial centroids for both units; only waveform separates them.
    com = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    p2p = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    wave = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1])
    test_com = np.array([[0.0, 0.0]])
    test_p2p = np.array([[1.0, 0.0]])
    test_wave = np.array([[0.0, 0.0, 1.0]])  # matches unit 1

    pred, meta = two_stage_assign(
        com_train=com,
        p2p_train=p2p,
        wave_train=wave,
        labels_train=labels,
        com_test=test_com,
        p2p_test=test_p2p,
        wave_test=test_wave,
        percentile=95.0,
        tau_com=1.0,
        tau_p2p=1.0,
    )
    assert pred.shape == (1,)
    assert int(pred[0]) == 1
    assert meta["fallback_rate"] == 0.0
    assert meta["mean_n_candidates"] == 2.0


def test_two_stage_fallback_when_no_candidates():
    com = np.array([[0.0, 0.0], [10.0, 0.0]])
    p2p = np.array([[1.0, 0.0], [0.0, 1.0]])
    wave = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    # Tight thresholds from exact centroids → far test misses both
    test_com = np.array([[5.0, 5.0]])
    test_p2p = np.array([[0.5, 0.5]])
    test_wave = np.array([[0.0, 1.0]])  # would prefer unit 1 if used

    pred, meta = two_stage_assign(
        com_train=com,
        p2p_train=p2p,
        wave_train=wave,
        labels_train=labels,
        com_test=test_com,
        p2p_test=test_p2p,
        wave_test=test_wave,
        percentile=50.0,  # very tight
        tau_com=0.01,
        tau_p2p=0.01,
    )
    assert meta["n_fallback"] == 1
    assert meta["fallback_rate"] == 1.0
    # Fallback ignores waveform; closer in normalized stage-1 to one of the units
    assert int(pred[0]) in (0, 1)


def test_stage1_normalized_score():
    score = stage1_normalized_score(
        com_x=np.array([0.0, 0.0]),
        p2p_x=np.array([1.0, 0.0]),
        com_c=np.array([1.0, 0.0]),
        p2p_c=np.array([1.0, 0.0]),
        tau_com=1.0,
        tau_p2p=1.0,
    )
    np.testing.assert_allclose(score, 1.0)  # d_com=1, d_p2p=0


def test_p2p_l1_metric_changes_distance():
    from Spatial.algorithms.two_stage_assign import vector_distance

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(vector_distance(a, b, "l2"), np.sqrt(2.0))
    np.testing.assert_allclose(vector_distance(a, b, "l1"), 2.0)


def test_two_stage_l1_p2p_runs():
    com = np.zeros((4, 2))
    p2p = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    wave = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    pred, meta = two_stage_assign(
        com, p2p, wave, labels,
        com_test=np.zeros((1, 2)),
        p2p_test=np.array([[0.0, 1.0]]),
        wave_test=np.array([[0.0, 1.0]]),
        p2p_metric="l1",
        tau_com=1.0,
        tau_p2p=1.0,
    )
    assert int(pred[0]) == 1
    assert meta["p2p_metric"] == "l1"


def test_calibrate_thresholds_codes_tau_at_least_one():
    # Identical codes within each unit → percentile 0, but τ floored at 1
    codes = np.array([[7, 0], [7, 0], [0, 7], [0, 7]], dtype=np.int64)
    labels = np.array([10, 10, 20, 20])  # non-contiguous unit ids
    tau = calibrate_thresholds_codes(codes, labels, percentile=50.0)
    assert tau >= 1
    cents = fit_unit_code_centroids(codes, labels)
    assert set(cents.keys()) == {10, 20}
    np.testing.assert_array_equal(cents[10], [7, 0])


def test_two_stage_assign_code_p2p_uses_unit_row_map():
    com = np.zeros((4, 2))
    codes = np.array([[8, 0], [8, 0], [0, 8], [0, 8]], dtype=np.int64)
    wave = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([5, 5, 9, 9])
    pred, meta = two_stage_assign_code_p2p(
        com, codes, wave, labels,
        com_test=np.zeros((1, 2)),
        codes_test=np.array([[0, 8]], dtype=np.int64),
        wave_test=np.array([[0.0, 1.0]]),
        tau_com=1.0,
        tau_p2p_codes=2,
    )
    assert int(pred[0]) == 9
    assert meta["tau_p2p_codes"] >= 1
    assert meta["p2p_metric"] == "code_l1"


def test_stage1_gate_matrix_matches_per_event_candidates():
    """Vectorized batch gate must equal per-event stage1_candidates bit-for-bit."""
    from Spatial.algorithms.two_stage_assign import stage1_gate_matrix

    rng = np.random.default_rng(123)
    n_units = 5
    com_cents = {u: rng.uniform(0, 1, size=2) for u in range(n_units)}
    p2p_cents = {u: rng.uniform(0, 1, size=4) for u in range(n_units)}
    tau_com, tau_p2p = 0.35, 0.6
    n_events = 40
    com_test = rng.uniform(0, 1, size=(n_events, 2))
    p2p_test = rng.uniform(0, 1, size=(n_events, 4))

    for com_metric in ("l2", "l1"):
        for p2p_metric in ("l2", "l1"):
            for S in (0, 1, 2):
                gate, units = stage1_gate_matrix(
                    com_test, p2p_test, com_cents, p2p_cents,
                    tau_com, tau_p2p,
                    com_metric=com_metric, p2p_metric=p2p_metric,
                    shift_radius=S,
                )
                for i in range(n_events):
                    ref = set(stage1_candidates(
                        com_test[i], p2p_test[i], com_cents, p2p_cents,
                        tau_com, tau_p2p,
                        com_metric=com_metric, p2p_metric=p2p_metric,
                        shift_radius=S,
                    ))
                    vec = {units[j] for j in range(len(units)) if gate[i, j]}
                    assert ref == vec, (
                        f"metric={com_metric},{p2p_metric} S={S} event={i}: "
                        f"ref={sorted(ref)} vec={sorted(vec)}"
                    )
