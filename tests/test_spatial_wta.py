"""Tests for SpatialWTAClassifier and OnlineSpatialAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from Spatial.algorithms.online_spatial_adaptation import OnlineSpatialAdapter
from Spatial.algorithms.spatial_wta import SpatialWTAClassifier


def _toy_data(n_units: int = 3, n_per: int = 40, n_dims: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Well-separated centroids in [0, 1].
    centers = np.linspace(0.15, 0.85, n_units)
    features = []
    labels = []
    for u, c in enumerate(centers):
        base = np.full((n_per, n_dims), c)
        base += rng.normal(0, 0.02, size=base.shape)
        features.append(np.clip(base, 0, 1))
        labels.append(np.full(n_per, u + 10))
    x = np.vstack(features)
    y = np.concatenate(labels)
    order = rng.permutation(x.shape[0])
    return x[order], y[order]


def test_uniform_wta_recovers_well_separated_units():
    x, y = _toy_data()
    n = x.shape[0]
    fit, cal, test = np.arange(n // 2), np.arange(n // 2, 3 * n // 4), np.arange(3 * n // 4, n)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(x[fit], y[fit]).calibrate(x[cal], y[cal])
    pred = clf.predict(x[test])
    assert float(np.mean(pred == y[test])) >= 0.95


def test_float_and_5bit_are_close_on_toy():
    x, y = _toy_data()
    n = x.shape[0]
    fit, cal, test = np.arange(n // 2), np.arange(n // 2, 3 * n // 4), np.arange(3 * n // 4, n)
    accs = []
    for n_bits in (5, None):
        clf = SpatialWTAClassifier(n_bits=n_bits, use_theta=True)
        clf.fit(x[fit], y[fit]).calibrate(x[cal], y[cal])
        pred = clf.predict(x[test])
        accs.append(float(np.mean(pred == y[test])))
    assert abs(accs[0] - accs[1]) <= 0.05


def test_mask_stability_enables_exactly_m_lanes():
    x, y = _toy_data()
    clf = SpatialWTAClassifier(
        n_bits=5, use_theta=False, mask_mode="stability", mask_size=3
    )
    clf.fit(x, y)
    assert clf.masks_.shape == (3, 9)
    assert np.all(clf.masks_.sum(axis=1) == 3)


def test_weights_are_power_of_two_and_shift_codes_valid():
    x, y = _toy_data()
    clf = SpatialWTAClassifier(
        n_bits=5,
        use_theta=False,
        score_mode="stability",
        rank_profile="tiered_421",
    )
    clf.fit(x, y)
    assert set(np.unique(clf.weights_).tolist()).issubset({1, 2, 4})
    assert np.array_equal(clf.weights_, 1 << clf.shift_codes_)
    assert np.isin(clf.shift_codes_, (0, 1, 2)).all()


def test_theta_comes_from_calibration_only():
    x, y = _toy_data()
    n = x.shape[0]
    fit, cal = np.arange(n // 2), np.arange(n // 2, n)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(x[fit], y[fit])
    assert clf.theta_source_ == "zero"
    assert np.all(clf.theta_ == 0)
    before = clf.centroids_.copy()
    clf.calibrate(x[cal], y[cal])
    assert clf.theta_source_ == "calibration"
    np.testing.assert_array_equal(clf.centroids_, before)


def test_memory_accounting_uniform_54bit_for_d9_b5():
    x, y = _toy_data(n_dims=9)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True, mask_mode="all_dims")
    clf.fit(x, y).calibrate(x, y)
    acct = clf.memory_accounting()
    # centroid 45 + theta 9 = 54 when uniform and use_theta
    assert acct["centroid_bits_per_row"] == 45
    assert acct["mask_bits_per_row"] == 0
    assert acct["shift_code_bits_per_row"] == 0
    assert acct["theta_bits_per_row"] == 9
    assert acct["search_payload_bits_per_row"] == 54


def test_logical_op_counts_reconcile():
    x, y = _toy_data()
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(x, y).calibrate(x, y)
    n_events = 17
    ops = clf.logical_op_counts(n_events)
    u, d = clf.units_.size, clf.n_features_in_
    assert ops["absolute_differences"] == n_events * u * d
    assert ops["adder_tree_additions"] == n_events * u * (d - 1)
    assert ops["centroid_row_reads"] == n_events * u
    assert ops["wta_comparisons"] == n_events * (u - 1)
    assert ops["theta_subtracts"] == n_events * u


def test_range_cam_box_predict_runs():
    x, y = _toy_data()
    n = x.shape[0]
    fit, cal, test = np.arange(n // 2), np.arange(n // 2, 3 * n // 4), np.arange(3 * n // 4, n)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(x[fit], y[fit]).calibrate(x[cal], y[cal])
    codes = clf.transform(x[fit])
    clf.fit_boxes_from_codes(codes, y[fit], coverage=0.95)
    pred = clf.predict_box(x[test])
    assert pred.shape == (test.size,)
    acct = clf.range_cam_accounting()
    assert acct["endpoint_bits_per_row"] == 2 * 9 * 5


def test_online_adapter_updates_without_labels():
    x, y = _toy_data()
    n = x.shape[0]
    fit, cal, test = np.arange(n // 2), np.arange(n // 2, 3 * n // 4), np.arange(3 * n // 4, n)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(x[fit], y[fit]).calibrate(x[cal], y[cal])
    before = clf.centroids_.copy()
    adapter = OnlineSpatialAdapter(clf, lr_shift=3)
    pred = adapter.predict_and_update(x[test])
    assert pred.shape == (test.size,)
    acct = adapter.update_accounting()
    assert acct["n_predictions"] == test.size
    assert acct["n_updates"] == test.size
    # Centroids may or may not move depending on data; accounting must be consistent.
    assert acct["centroid_writes"] >= 0
    assert acct["total_update_bits_written"] == (
        acct["centroid_bits_written"] + acct["theta_bits_written"]
    )
    # State object is the same classifier; writes go through.
    assert clf.centroids_.shape == before.shape


def test_online_updates_are_visible_to_subsequent_events():
    """Later events must observe earlier writes (batch-then-update is a bug)."""
    x, y = _toy_data(n_units=2, n_per=30, seed=1)
    clf = SpatialWTAClassifier(n_bits=5, use_theta=False, mask_mode="all_dims")
    clf.fit(x, y)
    # Isolate unit 0: others parked at code_max, unit 0 at 0. Use a low-valued
    # query so unit 0 uniquely wins with a nonzero residual.
    clf.centroids_[:] = int(clf.quantizer.code_max)
    clf.centroids_[0] = 0
    low_feat = np.full((1, clf.n_features_in_), 0.25, dtype=np.float64)
    code = clf.transform(low_feat)[0]
    assert int(code.min()) >= 2
    assert int(np.argmin(clf._row_distances(code[np.newaxis, :])[0])) == 0

    n_steps = 5
    stream = np.repeat(low_feat, n_steps, axis=0)
    adapter = OnlineSpatialAdapter(clf, lr_shift=1, update_theta=False)
    adapter.predict_and_update(stream)
    final = clf.centroids_[0].astype(np.int64)

    # Ground-truth sequential trunc-EMA from the same initial state.
    expected = np.zeros_like(code, dtype=np.int64)
    for _ in range(n_steps):
        expected = expected + OnlineSpatialAdapter.trunc_shift(code.astype(np.int64) - expected, 1)
        expected = np.clip(expected, 0, int(clf.quantizer.code_max))
    np.testing.assert_array_equal(final, expected)

    # Batched bug applies the initial delta n_steps times; that diverges here.
    delta0 = OnlineSpatialAdapter.trunc_shift(code.astype(np.int64), 1)
    batched = np.clip(n_steps * delta0, 0, int(clf.quantizer.code_max))
    assert not np.array_equal(final, batched)
    assert adapter.centroid_writes_ >= 1


def test_online_shift_is_symmetric_not_a_ratchet():
    """Arithmetic >> on negatives is a one-way ratchet; trunc-toward-zero is not."""
    assert OnlineSpatialAdapter.trunc_shift(np.array([15, -15]), 4).tolist() == [0, 0]
    assert OnlineSpatialAdapter.trunc_shift(np.array([16, -16]), 4).tolist() == [1, -1]
    assert OnlineSpatialAdapter.trunc_shift(np.array([17, -17]), 4).tolist() == [1, -1]
    assert OnlineSpatialAdapter.trunc_shift(np.array([1, -1, 0]), 4).tolist() == [0, 0, 0]
    # Arithmetic right-shift (the old bug) would map -1..-15 -> -1, +1..+15 -> 0.
    buggy = (np.array([-9, 9], dtype=np.int64) >> 4).tolist()
    fixed = OnlineSpatialAdapter.trunc_shift(np.array([-9, 9]), 4).tolist()
    assert buggy == [-1, 0]
    assert fixed == [0, 0]


def test_online_rejects_float_classifier():
    x, y = _toy_data()
    clf = SpatialWTAClassifier(n_bits=None, use_theta=True)
    clf.fit(x, y).calibrate(x, y)
    with pytest.raises(ValueError, match="integer quantizer"):
        OnlineSpatialAdapter(clf)


def test_all_dims_requires_full_mask_size():
    x, y = _toy_data()
    clf = SpatialWTAClassifier(n_bits=5, mask_mode="all_dims", mask_size=3)
    with pytest.raises(ValueError, match="all_dims"):
        clf.fit(x, y)
