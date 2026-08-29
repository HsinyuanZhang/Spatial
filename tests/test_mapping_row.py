"""Contract tests for mapping-row v0 (no recording I/O)."""

import numpy as np

from Spatial.algorithms.mapping_init import (
    calibrate_tau,
    fit_gain_ranges,
    fit_interval_rows,
    fit_mean_l1_p999,
    query_mean_l1,
)
from Spatial.algorithms.mapping_row import (
    apply_censoring,
    candidate_lists,
    penalty_from_violation,
    quality_flags,
    raw_violation,
    scale_alert,
    score_rows,
)


def test_interval_interior_has_zero_violation_independent_of_midpoint():
    query = np.array([[5.0, 8.0]])
    lower = np.array([[0.0, 8.0]])
    upper = np.array([[10.0, 8.0]])
    v = raw_violation(query, lower, upper)
    np.testing.assert_array_equal(v, [[0.0, 0.0]])
    # Distance to midpoint is 0 vs 0 on dim0? midpoint of [0,10] is 5, of [8,8] is 8.
    # Move inside the first box away from the centre: still 0.
    v2 = raw_violation(np.array([[1.0, 8.0]]), lower, upper)
    np.testing.assert_array_equal(v2, [[0.0, 0.0]])


def test_inclusive_bounds_are_matches():
    v = raw_violation(np.array([[0.0, 10.0]]), np.array([[0.0, 2.0]]), np.array([[4.0, 10.0]]))
    np.testing.assert_array_equal(v, [[0.0, 0.0]])


def test_j5_loud_template_quiet_query_is_high_penalty():
    query = np.array([[0.0]])
    mu_loud = np.array([[9.0]])
    mu_floor = np.array([[0.0]])
    box_lo = np.array([[0.0]])
    box_hi = np.array([[12.0]])
    # Symmetric box would accept 0 because lo=0.
    v_sym = raw_violation(query, box_lo, box_hi)
    np.testing.assert_array_equal(v_sym, [[0.0]])
    v_loud = apply_censoring(query, mu_loud, v_sym, enabled=True)
    assert v_loud[0, 0] >= 3.0
    v_floor = apply_censoring(query, mu_floor, v_sym, enabled=True)
    np.testing.assert_array_equal(v_floor, [[0.0]])


def test_scale_alert_does_not_change_score():
    query = np.array([[4.0, 4.0]])
    lo = np.array([[3.0, 3.0]])
    hi = np.array([[5.0, 5.0]])
    p = np.array([[0, 0]])
    mu = np.array([[4.0, 4.0]])
    s = score_rows(query, lo, hi, p, mu)["S"]
    g = np.array([15])
    alert = scale_alert(g, np.array([1]), np.array([8]))
    assert bool(alert[0, 0])
    np.testing.assert_array_equal(s, [[0]])


def test_quality_flags_do_not_enter_s():
    vmax = np.array([[1.0, 0.2]])
    vmin = np.array([[-1.0, 0.0]])
    imax = np.array([[0, 3]])
    imin = np.array([[10, 4]])
    nbr = np.array([[0, 1]])
    g = np.array([1.0])
    flags = quality_flags(
        vmax, vmin, imax, imin, nbr, g, window=5, home_slot=0, sat_abs=0.5
    )
    assert flags.shape == (1, 3)
    # Score path has no quality argument by construction.
    s = score_rows(
        np.array([[1.0, 1.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[2.0, 2.0]]),
        np.array([[0, 0]]),
        np.array([[1.0, 1.0]]),
    )["S"]
    np.testing.assert_array_equal(s, [[0]])


def test_tau_uses_calibration_not_test():
    # Dim 1: unit 7 lives at 0 on fit/cal and 10 on a held-out block.
    fit = np.array([[0], [0], [1], [0]], dtype=np.int64)
    y_fit = np.array([7, 7, 7, 7])
    cal = np.array([[0], [1], [0]], dtype=np.int64)
    y_cal = np.array([7, 7, 7])
    test = np.array([[10], [10]], dtype=np.int64)
    rows = fit_interval_rows(fit, y_fit)
    tau, source = calibrate_tau(cal, y_cal, rows, percentile=99.9, codes_fit=fit, labels_fit=y_fit)
    tau_again, _ = calibrate_tau(cal, y_cal, rows, percentile=99.9, codes_fit=fit, labels_fit=y_fit)
    np.testing.assert_array_equal(tau, tau_again)
    assert source[0] == "calibration"
    scored_test = score_rows(test, rows["lower"], rows["upper"], rows["precision"], rows["mu"])["S"]
    # Recalibrating on test would inflate tau; the frozen tau must stay small.
    tau_if_test, _ = calibrate_tau(test, np.array([7, 7]), rows, percentile=99.9)
    assert int(tau[0]) < int(tau_if_test[0])
    assert int(scored_test[0, 0]) > int(tau[0])


def test_j5_score_path_matches_censoring_helper():
    q = np.array([[0.0, 8.0]])
    mu = np.array([[9.0, 8.0]])
    lo = np.array([[0.0, 7.0]])
    hi = np.array([[12.0, 9.0]])
    p = np.array([[0, 0]])
    s_sym = score_rows(q, lo, hi, p, mu, censoring=False)["S"]
    s_j5 = score_rows(q, lo, hi, p, mu, censoring=True)["S"]
    assert int(s_j5[0, 0]) > int(s_sym[0, 0])


def test_penalty_caps_at_three():
    p = penalty_from_violation(np.array([[100.0]]), np.array([[0]]))
    np.testing.assert_array_equal(p, [[3]])


def test_mean_l1_noncontiguous_unit_ids():
    fit = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.int64)
    y = np.array([20, 20, 3, 3])
    model = fit_mean_l1_p999(fit, y, fit, y)
    assert set(model["units"].tolist()) == {3, 20}
    cand, dist, meta = query_mean_l1(np.array([[0, 0]], dtype=np.int64), model)
    assert 20 in cand[0]


def test_gain_range_fit_percentiles():
    g = np.arange(100, dtype=np.int64)
    y = np.zeros(100, dtype=np.int64)
    lo, hi = fit_gain_ranges(g, y, np.array([0]))
    assert 0 <= int(lo[0]) <= int(hi[0]) <= 99


def test_candidate_lists_inclusive_tau():
    s = np.array([[2, 5], [0, 1]])
    tau = np.array([2, 1])
    units = np.array([4, 9])
    cand, diag = candidate_lists(s, tau, units)
    np.testing.assert_array_equal(cand[0], [4])
    np.testing.assert_array_equal(sorted(cand[1].tolist()), [4, 9])
    np.testing.assert_array_equal(diag["counts"], [1, 2])
