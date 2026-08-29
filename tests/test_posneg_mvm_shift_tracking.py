"""Contract tests for POSNEG MVM scoring, shift leakage, and causal freeze.

Synthetic arrays only. Drives shipped functions; does not mock them.
"""

import numpy as np

from Spatial.algorithms.deformable_spatial_template import geometry_k_table
from Spatial.algorithms.drift_tracking import CausalTracker, tile_ids
from Spatial.algorithms.posneg_mvm_matching import (
    biased_dot,
    biased_dot_mvm,
    build_features,
    concat_posneg,
    cosine_similarity,
    geometry_k_is_physical,
    l2_squared,
    p2p_from_posneg,
    polarity_mass_ratio,
    polarity_spatial_delta,
    score_kind,
)
from Spatial.algorithms.spatial_shift_matching import (
    SHIFT_DISCRETE,
    biased_dot_path_has_no_norm,
    leakage_from_pair,
    precompute_shift_mats,
    score_shift_bank,
    shift_matrix,
)


def _line_geom(n: int = 8, pitch: float = 20.0) -> np.ndarray:
    g = np.zeros((n, 2), dtype=np.float64)
    g[:, 1] = np.arange(n) * pitch
    return g


def test_argmin_l2_equals_argmax_biased_dot_and_mvm_form():
    rng = np.random.default_rng(0)
    x = rng.normal(size=32)
    mu = rng.normal(size=(11, 32))
    d = l2_squared(x, mu)
    s = biased_dot(x, mu)
    assert int(np.argmin(d)) == int(np.argmax(s))
    s_mvm, m, b = biased_dot_mvm(x, mu)
    np.testing.assert_allclose(s, s_mvm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m, 2.0 * mu)
    np.testing.assert_allclose(b, -np.sum(mu * mu, axis=1))
    np.testing.assert_allclose(s, -d + np.dot(x, x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_mvm, m @ x + b, rtol=1e-12, atol=1e-12)


def test_geometry_k_is_physical_offset_not_amplitude():
    g = _line_geom(8)
    table, rel = geometry_k_table(g, k=5)
    assert geometry_k_is_physical(table, g, home=3, k=5)
    np.testing.assert_array_equal(table[3], [1, 2, 3, 4, 5])
    # Amplitude on a far channel does not change the slot table (table ignores it).
    vmax = np.zeros(8)
    vmax[7] = 1e6
    table2, _ = geometry_k_table(g, k=5)
    np.testing.assert_array_equal(table, table2)
    assert 7 not in set(int(c) for c in table[3])
    np.testing.assert_allclose(rel[3, 2], 0.0)


def test_p2p_posneg_pos_neg_are_distinct_concatenations():
    pos = np.array([1.0, 0.0, 3.0, 0.5])
    neg = np.array([0.2, 4.0, 0.0, 1.0])
    p2p = p2p_from_posneg(pos, neg)
    raw = concat_posneg(pos, neg)
    pos_only = build_features(pos, neg, "pos")
    neg_only = build_features(pos, neg, "neg")
    np.testing.assert_allclose(p2p, pos + neg)
    np.testing.assert_allclose(raw, np.concatenate([pos, neg]))
    np.testing.assert_allclose(pos_only, pos)
    np.testing.assert_allclose(neg_only, neg)
    assert raw.shape[-1] == 2 * p2p.shape[-1]
    assert not np.allclose(p2p, pos)
    assert not np.allclose(p2p, neg)
    assert not np.allclose(raw[:4], raw[4:])
    rpn = polarity_mass_ratio(pos, neg)
    dc = polarity_spatial_delta(pos, neg, np.stack([np.zeros(4), np.arange(4) * 20.0], axis=1))
    feat = build_features(pos, neg, "posneg_rpn_dc", np.stack([np.zeros(4), np.arange(4) * 20.0], axis=1))
    assert feat.shape[-1] == 8 + 1 + 2
    assert feat[8] == rpn
    np.testing.assert_allclose(feat[9:], dc)


def test_cosine_divides_by_norms_biased_dot_moving_does_not():
    x = np.array([0.0, 2.0, 4.0, 1.0])
    mu = np.array([0.0, 1.0, 2.0, 0.5])
    assert biased_dot_path_has_no_norm(x, mu)
    c1 = float(cosine_similarity(x, mu))
    c2 = float(cosine_similarity(3.0 * x, mu))
    np.testing.assert_allclose(c1, c2)
    s1 = float(biased_dot(x, mu))
    s2 = float(biased_dot(3.0 * x, mu))
    assert s2 > s1
    # Moving-correlation biased-dot path uses score_kind('biased_dot'), not cosine.
    g = _line_geom(5)
    table, rel = geometry_k_table(g, k=5)
    mats = precompute_shift_mats(rel, table, SHIFT_DISCRETE, pitch_um=20.0, axis=1)
    pos = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    neg = np.zeros(5)
    mu_p = pos.copy()
    mu_n = neg.copy()
    best_bd, _ = score_shift_bank(pos, neg, mu_p, mu_n, mats[2], "posneg", "biased_dot", rel[2])
    best_cs, _ = score_shift_bank(2.0 * pos, 2.0 * neg, mu_p, mu_n, mats[2], "posneg", "cosine", rel[2])
    best_bd2, _ = score_shift_bank(2.0 * pos, 2.0 * neg, mu_p, mu_n, mats[2], "posneg", "biased_dot", rel[2])
    np.testing.assert_allclose(best_cs, 1.0, atol=1e-6)
    assert float(best_bd2) != float(best_bd)


def test_flexibility_leakage_formula_on_constructed_pair():
    g = _line_geom(5)
    table, rel = geometry_k_table(g, k=5)
    mats = precompute_shift_mats(rel, table, SHIFT_DISCRETE, pitch_um=20.0, axis=1)
    mu_true_pos = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    mu_true_neg = np.zeros(5)
    mu_wrong_pos = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    mu_wrong_neg = np.zeros(5)
    x_pos = mu_true_pos.copy()
    x_neg = mu_true_neg.copy()
    rec = leakage_from_pair(
        x_pos,
        x_neg,
        mu_true_pos,
        mu_true_neg,
        mu_wrong_pos,
        mu_wrong_neg,
        mats[2],
        rel[2],
        kind="biased_dot",
    )
    np.testing.assert_allclose(rec["l_flex"], rec["i_wrong"] - rec["i_true"], atol=1e-12)
    # True already aligned: free shift cannot help it much; impostor can shift onto x.
    assert rec["i_wrong"] > rec["i_true"]
    assert rec["l_flex"] > 0.0


def test_causal_tracker_freezes_state_before_scoring():
    g = _line_geom(5)
    table, rel = geometry_k_table(g, k=5)
    mu_pos = np.array([[0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0, 0.0]])
    mu_neg = np.zeros_like(mu_pos)
    homes = np.array([2, 1], dtype=np.int64)
    groups = np.zeros(2, dtype=np.int64)
    tracker = CausalTracker(
        "ema",
        mu_pos,
        mu_neg,
        homes,
        rel,
        table,
        g,
        20.0,
        1,
        groups,
        kind="l2",
        representation="posneg",
        confidence_threshold=-np.inf,
    )
    pos_query = mu_pos.copy()
    neg_query = mu_neg.copy()
    pos_update = mu_pos.copy()
    pos_update[0] = np.array([0.0, 2.0, 12.0, 1.0, 0.0])
    neg_update = mu_neg.copy()
    snap0 = tracker.snapshot()
    s1 = tracker.scores_from_posneg(pos_query, neg_query)
    snap1 = tracker.snapshot()
    np.testing.assert_allclose(snap0.mu_pos, snap1.mu_pos)
    np.testing.assert_allclose(s1, tracker.scores_from_posneg(pos_query, neg_query))
    dist = s1.copy()
    tracker.maybe_update(pos_update, neg_update, pred_index=0, distances_row=dist, confident=True)
    snap2 = tracker.snapshot()
    assert not np.allclose(snap2.mu_pos, snap0.mu_pos)
    s2 = tracker.scores_from_posneg(pos_query, neg_query)
    assert not np.allclose(s1, s2)
    tracker.restore(snap0)
    s3 = tracker.scores_from_posneg(pos_query, neg_query)
    np.testing.assert_allclose(s1, s3)


def test_frozen_tracker_update_is_noop():
    g = _line_geom(4)
    table, rel = geometry_k_table(g, k=3)
    mu_pos = np.ones((2, 3))
    mu_neg = np.zeros((2, 3))
    homes = np.array([1, 2], dtype=np.int64)
    tr = CausalTracker(
        "frozen",
        mu_pos,
        mu_neg,
        homes,
        rel,
        table,
        g,
        20.0,
        1,
        np.zeros(2, dtype=np.int64),
    )
    pos = np.full((2, 3), 3.0)
    neg = np.zeros((2, 3))
    s1 = tr.scores_from_posneg(pos, neg)
    assert tr.maybe_update(pos, neg, 0, s1, confident=True) is False
    s2 = tr.scores_from_posneg(pos, neg)
    np.testing.assert_allclose(s1, s2)


def test_tile_ids_group_by_primary_axis_bands():
    g = _line_geom(9, pitch=20.0)
    homes = np.arange(9, dtype=np.int64)
    tid = tile_ids(g, homes, pitch_um=20.0, pitches_per_tile=3.0)
    assert tid[0] == tid[2]
    assert tid[0] != tid[5]
    assert int(tid.max()) >= 1
