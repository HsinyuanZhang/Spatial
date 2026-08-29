"""Contract tests for EMA* / Oracle-EMA mean-template updates.

Synthetic arrays only. Drives shipped CausalTracker and apply_ema_step.
"""

import numpy as np

from Spatial.algorithms.deformable_spatial_template import geometry_k_table
from Spatial.algorithms.drift_tracking import CausalTracker, apply_ema_step


def _line_geom(n: int = 5, pitch: float = 20.0) -> np.ndarray:
    g = np.zeros((n, 2), dtype=np.float64)
    g[:, 1] = np.arange(n) * pitch
    return g


def _tracker(mode, mu_pos, mu_neg, eta, owner="self"):
    g = _line_geom(5)
    table, rel = geometry_k_table(g, k=5)
    homes = np.array([2, 1], dtype=np.int64)
    return CausalTracker(
        mode,
        mu_pos,
        mu_neg,
        homes,
        rel,
        table,
        g,
        20.0,
        1,
        np.zeros(2, dtype=np.int64),
        eta_ema=eta,
        kind="l2",
        representation="posneg",
        confidence_threshold=-np.inf,
        update_owner=owner,
    )


def test_two_etas_move_mu_by_different_amounts():
    mu_p = np.array([[0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0, 0.0]])
    mu_n = np.zeros_like(mu_p)
    x_p = np.array([0.0, 2.0, 12.0, 1.0, 0.0])
    x_n = np.zeros(5)
    a = mu_p.copy()
    b = mu_p.copy()
    an = mu_n.copy()
    bn = mu_n.copy()
    apply_ema_step(a, an, 0, x_p, x_n, 0.01)
    apply_ema_step(b, bn, 0, x_p, x_n, 0.10)
    d_small = np.linalg.norm(a[0] - mu_p[0])
    d_large = np.linalg.norm(b[0] - mu_p[0])
    assert d_large > d_small > 0
    np.testing.assert_allclose(a[1], mu_p[1])
    np.testing.assert_allclose(b[1], mu_p[1])
    # Closed form: step = η (x − μ0).
    np.testing.assert_allclose(a[0], (1 - 0.01) * mu_p[0] + 0.01 * x_p)
    np.testing.assert_allclose(b[0], (1 - 0.10) * mu_p[0] + 0.10 * x_p)


def test_self_ema_updates_predicted_owner_oracle_updates_gt_only():
    mu_p = np.array([[0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0, 0.0]])
    mu_n = np.zeros_like(mu_p)
    pos_e = np.array([[0.0, 4.0, 14.0, 2.0, 0.0], [0.0, 9.0, 1.0, 0.0, 0.0]])
    neg_e = np.zeros_like(pos_e)
    dist = np.array([5.0, 1.0])  # predicted owner would be unit 1 if argmin
    self_tr = _tracker("ema", mu_p.copy(), mu_n.copy(), 0.2, owner="self")
    ora_tr = _tracker("oracle_ema", mu_p.copy(), mu_n.copy(), 0.2)
    # Wrong predicted owner = 1; GT owner = 0.
    self_tr.maybe_update(pos_e, neg_e, pred_index=1, distances_row=dist, confident=True, gt_index=0)
    ora_tr.maybe_update(pos_e, neg_e, pred_index=1, distances_row=dist, confident=True, gt_index=0)
    assert not np.allclose(self_tr.mu_pos[1], mu_p[1])
    np.testing.assert_allclose(self_tr.mu_pos[0], mu_p[0])
    assert not np.allclose(ora_tr.mu_pos[0], mu_p[0])
    np.testing.assert_allclose(ora_tr.mu_pos[1], mu_p[1])
    np.testing.assert_allclose(ora_tr.mu_pos[0], 0.8 * mu_p[0] + 0.2 * pos_e[0])


def test_ema_scores_current_event_before_update():
    mu_p = np.array([[0.0, 0.0, 10.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0, 0.0]])
    mu_n = np.zeros_like(mu_p)
    tr = _tracker("ema", mu_p.copy(), mu_n.copy(), 0.5)
    pos_q = mu_p.copy()
    neg_q = mu_n.copy()
    pos_u = mu_p.copy()
    pos_u[0] = np.array([0.0, 3.0, 14.0, 2.0, 0.0])
    snap0 = tr.snapshot()
    s1 = tr.scores_from_posneg(pos_q, neg_q)
    snap1 = tr.snapshot()
    np.testing.assert_allclose(snap0.mu_pos, snap1.mu_pos)
    tr.maybe_update(pos_u, neg_q, pred_index=0, distances_row=s1, confident=True)
    s2 = tr.scores_from_posneg(pos_q, neg_q)
    assert not np.allclose(s1, s2)
    tr.restore(snap0)
    np.testing.assert_allclose(s1, tr.scores_from_posneg(pos_q, neg_q))
