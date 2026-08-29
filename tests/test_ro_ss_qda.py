"""Contract tests for RO-SS-QDA (no recording I/O)."""

import numpy as np

from Spatial.algorithms.ro_ss_qda import (
    ETA,
    FitStandardizer,
    RegularizedQDA,
    fit_means,
    make_descriptor,
    mean_distance_scores,
    polarity_profile,
    posneg_amplitudes,
    scale_l1,
    select_lambda_global,
    shrink_covariance,
    stack_posneg,
)
from Spatial.algorithms.spatial_footprint import (
    build_relative_patch_table,
    is_linear_probe,
    relative_offset_neighbor_table,
)
from Spatial.algorithms.mapping_init import j0_search_from_descriptor


def test_shared_l1_scale_preserves_polarity_separate_does_not():
    pos = np.array([[8.0, 2.0]])
    neg = np.array([[1.0, 4.0]])
    x = stack_posneg(pos, neg)
    u, mass = scale_l1(x)
    assert mass[0] == 15.0
    rho_raw = polarity_profile(pos, neg)
    rho_u = polarity_profile(u[:, :2] * mass[0], u[:, 2:] * mass[0])
    np.testing.assert_allclose(rho_u, rho_raw)
    pos_n = pos / pos.sum()
    neg_n = neg / neg.sum()
    rho_sep = polarity_profile(pos_n, neg_n)
    assert np.max(np.abs(rho_sep - rho_raw)) > 1e-6


def test_standardizer_statistics_come_from_fit_only():
    fit = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    test = np.array([[10.0, 10.0]], dtype=np.float64)
    scaler = FitStandardizer().fit(fit)
    np.testing.assert_allclose(scaler.mean_, [1.0, 2.0])
    z_test = scaler.transform(test)
    scaler2 = FitStandardizer().fit(fit)
    np.testing.assert_allclose(scaler2.transform(test), z_test)
    assert scaler2.mean_[0] != test.mean()


def test_lambda_is_chosen_on_calibration_and_frozen():
    rows = [
        {"lam": 0.25, "accuracy": 0.40, "home_margin_p10": 0.9},
        {"lam": 0.5, "accuracy": 0.70, "home_margin_p10": 0.1},
        {"lam": 0.75, "accuracy": 0.70, "home_margin_p10": 0.4},
    ]
    chosen = select_lambda_global(rows)
    assert chosen == 0.75
    assert select_lambda_global(rows) == chosen


def test_p3_off_diagonals_are_exactly_zero():
    rng = np.random.default_rng(0)
    cov_k = rng.normal(size=(4, 4))
    cov_k = cov_k @ cov_k.T
    cov_p = rng.normal(size=(4, 4))
    cov_p = cov_p @ cov_p.T
    shrunk = shrink_covariance(cov_k, cov_p, 0.5, diagonal=True)
    off = shrunk - np.diag(np.diag(shrunk))
    np.testing.assert_allclose(off, 0.0, atol=1e-12)


def test_logdet_term_changes_the_score():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(40, 3))
    y = np.array([0] * 20 + [1] * 20)
    z[:20] += np.array([1.0, 0.0, 0.0])
    z[20:] += np.array([-1.0, 0.0, 0.0])
    with_term = RegularizedQDA(0.5, include_logdet=True).fit(z, y).scores(z[:2])
    without = RegularizedQDA(0.5, include_logdet=False).fit(z, y).scores(z[:2])
    assert np.max(np.abs(with_term - without)) > 1e-8


def test_equal_prior_ignores_class_counts():
    z = np.vstack(
        [
            np.zeros((10, 2)),
            np.ones((100, 2)),
        ]
    )
    y = np.array([0] * 10 + [1] * 100)
    model = RegularizedQDA(1.0).fit(z, y)
    query = np.array([[0.5, 0.5]])
    s = model.scores(query)[0]
    # Means are (0,0) and (1,1); query is equidistant. Equal prior → equal scores
    # up to covariance volume, which is shared at lambda=1.
    np.testing.assert_allclose(s[0], s[1], rtol=0, atol=1e-8)


def test_small_sample_unit_uses_pooled_and_does_not_singular():
    d = 6
    z0 = np.zeros((30, d))
    z1 = np.ones((3, d))  # n = 3 < d+5
    z = np.vstack([z0, z1])
    y = np.array([0] * 30 + [1] * 3)
    model = RegularizedQDA(0.25).fit(z, y)
    assert bool(model.forced_pooled_[list(model.units_).index(1)])
    s = model.scores(np.ones((1, d)))
    assert np.all(np.isfinite(s))


def test_p0_mean_l1_matches_j0_centroids_on_unit_interval():
    rng = np.random.default_rng(2)
    x_fit = np.clip(rng.random((40, 9)), 0, 1)
    y_fit = np.array([1] * 20 + [7] * 20)
    x_cal = np.clip(rng.random((20, 9)), 0, 1)
    y_cal = np.array([1] * 10 + [7] * 10)
    x_test = np.clip(rng.random((8, 9)), 0, 1)
    model = j0_search_from_descriptor(x_fit, y_fit, x_cal, y_cal)
    codes = model.transform(x_test)
    dist = np.abs(codes[:, None, :] - model.centroids_[None, :, :]).sum(axis=2)
    pred_j0 = model.units_[np.argmin(dist, axis=1)]
    scores = mean_distance_scores(codes, model.centroids_, "l1")
    pred = model.units_[np.argmax(scores, axis=1)]
    np.testing.assert_array_equal(pred, pred_j0)


def test_linear_probe_uses_rank_offset_and_2d_uses_stencil():
    linear = np.zeros((8, 2))
    linear[:, 1] = np.arange(8) * 20.0
    assert is_linear_probe(linear)
    table_1d = relative_offset_neighbor_table(linear, half_width=3)
    np.testing.assert_array_equal(table_1d, build_relative_patch_table(linear, half_width=3))
    three_col = np.array(
        [[x, y] for y in (0.0, 25.0, 50.0, 75.0) for x in (-18.0, 0.0, 18.0)],
        dtype=np.float64,
    )
    assert not is_linear_probe(three_col)
    grid = np.array([[x, y] for y in (0.0, 20.0, 40.0) for x in (0.0, 32.0)], dtype=np.float64)
    assert not is_linear_probe(grid)
    table_2d = relative_offset_neighbor_table(grid, half_width=3)
    assert table_2d.shape == (6, 7)
    for ch in range(6):
        assert table_2d[ch, 3] == ch


def test_absent_slot_is_zero_in_x_but_flagged():
    vmax = np.array([[1.0, 0.0, 0.5]])
    vmin = np.array([[-2.0, 0.0, -0.1]])
    nbr = np.array([[4, -1, 5]])
    pos, neg, absent = posneg_amplitudes(vmax, vmin, nbr)
    np.testing.assert_array_equal(absent, [[False, True, False]])
    np.testing.assert_allclose(pos[0, 1], 0.0)
    np.testing.assert_allclose(neg[0, 1], 0.0)


def test_make_descriptor_shares_one_l1_mass_across_pos_and_neg():
    vmax = np.array([[3.0, 1.0]])
    vmin = np.array([[-1.0, -0.0]])
    nbr = np.array([[0, 1]])
    geom = np.array([[0.0, 0.0], [0.0, 20.0]])
    pack = make_descriptor(vmax, vmin, nbr, geom, scale="l1", one_d=True)
    np.testing.assert_allclose(pack["m"][0], 5.0)
    np.testing.assert_allclose(pack["u"].sum(axis=1), 1.0)
    assert pack["z"].shape[1] == 2 + 2 + 1 + 1  # u=4 (K=2), log m, 1-D COM
