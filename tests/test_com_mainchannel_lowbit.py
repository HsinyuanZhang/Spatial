"""COM / main-channel nearest-centroid helpers (no recording I/O)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import quantize_com
from Spatial.algorithms.two_stage_assign import assign_nearest, fit_unit_centroids
from Spatial.data.loader import Dataset
from Spatial.experiments.analyze_com_radius_prefilter import (
    evaluate_com_radius_prefilter,
)
from Spatial.experiments.run_com_mainchannel_lowbit import (
    _chronological_split,
    fill_missing_seed_xy,
    juxta_match_rate,
    oracle_argmin_assign,
    prepare_com_mainchannel_events,
    run_study,
)


def _two_unit_com(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.20, 0.20], 0.01, size=(n_per, 2))
    b = rng.normal([0.80, 0.80], 0.01, size=(n_per, 2))
    feats = np.empty((2 * n_per, 2), dtype=np.float64)
    labels = np.empty(2 * n_per, dtype=np.int64)
    times = np.arange(2 * n_per, dtype=np.int64)
    for i in range(n_per):
        feats[2 * i] = np.clip(a[i], 0.0, 1.0)
        labels[2 * i] = 10
        feats[2 * i + 1] = np.clip(b[i], 0.0, 1.0)
        labels[2 * i + 1] = 20
    return feats, labels, times


def test_chronological_split_is_prefix_half():
    train, test = _chronological_split(10, train_frac=0.5)
    np.testing.assert_array_equal(train, np.arange(5))
    np.testing.assert_array_equal(test, np.arange(5, 10))


def test_oracle_argmin_separates_two_com_clouds():
    feats, labels, times = _two_unit_com()
    out = oracle_argmin_assign(feats, labels, times, train_frac=0.5, metric="l2")
    assert out["n_train"] == 40
    assert out["n_test"] == 40
    assert out["n_units_train"] == 2
    assert out["argmin_accuracy"] == 1.0
    np.testing.assert_array_equal(out["pred"], out["y_true"])


def test_quantized_com_assignment_preserves_separated_units():
    feats, labels, times = _two_unit_com()
    codes = quantize_com(feats, 4)
    assert codes.dtype == np.float64
    assert np.all((codes >= 0.0) & (codes <= 15.0))
    out = oracle_argmin_assign(codes, labels, times, train_frac=0.5, metric="l2")
    assert out["argmin_accuracy"] == 1.0


def test_assign_nearest_matches_fit_unit_centroids():
    feats, labels, _ = _two_unit_com(n_per=8)
    cents = fit_unit_centroids(feats[:8], labels[:8])
    pred = assign_nearest(feats[8:], cents, metric="l2")
    np.testing.assert_array_equal(pred, labels[8:])


def test_unseen_unit_is_excluded_from_primary_accuracy():
    feats = np.array(
        [[0.1, 0.1], [0.12, 0.11], [0.9, 0.9], [0.91, 0.88]],
        dtype=np.float64,
    )
    labels = np.array([1, 1, 1, 7], dtype=np.int64)
    times = np.array([0, 1, 2, 3], dtype=np.int64)
    out = oracle_argmin_assign(feats, labels, times, train_frac=0.5, metric="l2")
    assert out["n_eval"] == 1
    assert out["n_units_unseen_test"] == 1
    assert out["argmin_accuracy"] == 1.0
    assert out["argmin_accuracy_all_test"] == 0.5


def test_juxta_match_rate_counts_coincident_predictions():
    pred = np.array([310, 12, 310], dtype=np.int64)
    times = np.array([100, 200, 400], dtype=np.int64)
    juxta = np.array([101, 399], dtype=np.int64)
    stats = juxta_match_rate(pred, times, juxta, juxta_cluster=310, tolerance_samples=8)
    assert stats["juxta_n_coincident"] == 2
    assert stats["juxta_n_pred_matched_cluster"] == 2
    assert stats["juxta_match_rate"] == 1.0


def _two_unit_grid_with_far_interferer():
    """7x7 / 30 um grid: two well-separated units plus a louder far trough."""
    n_side, pitch, fs = 7, 30.0, 20000
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    geom = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    home_a = 3 * n_side + 1  # (30, 90)
    home_b = 3 * n_side + 5  # (150, 90)
    interferer = 0  # (0, 0)
    n_ch = geom.shape[0]
    n_samp = 400
    raw = np.zeros((n_ch, n_samp), dtype=np.float64)
    times_a = np.array([80, 140, 200, 260], dtype=np.int64)
    times_b = np.array([100, 160, 220, 280], dtype=np.int64)
    for t in times_a:
        raw[home_a, t - 1 : t + 2] = -6.0
        raw[home_a, t] = -10.0
        # Overlap: stronger trough 4+ pitches away at the same sample.
        raw[interferer, t - 1 : t + 2] = -40.0
    for t in times_b:
        raw[home_b, t - 1 : t + 2] = -6.0
        raw[home_b, t] = -10.0
    times = np.concatenate([times_a, times_b])
    units = np.concatenate(
        [np.full(times_a.size, 10, dtype=np.int64), np.full(times_b.size, 20, dtype=np.int64)]
    )
    order = np.argsort(times, kind="stable")
    ds = Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=times[order],
        spike_units=units[order],
        name="synthetic_grid_overlap",
    )
    seed = geom[np.where(units[order] == 10, home_a, home_b)]
    return ds, seed, home_a, home_b, interferer


def test_fill_missing_seed_xy_uses_unit_median():
    labels = np.array([1, 1, 1, 2], dtype=np.int64)
    xy = np.array(
        [[10.0, 0.0], [np.nan, np.nan], [12.0, 0.0], [np.nan, np.nan]],
        dtype=np.float64,
    )
    filled = fill_missing_seed_xy(xy, labels)
    np.testing.assert_allclose(filled[1], [11.0, 0.0])
    assert not np.isfinite(filled[3]).all()


def test_radius_prepare_does_not_let_far_trough_steal_home():
    ds, seed, home_a, home_b, interferer = _two_unit_grid_with_far_interferer()
    legacy = prepare_com_mainchannel_events(ds, k_neighbors=7, window=15)
    local = prepare_com_mainchannel_events(
        ds,
        window=15,
        same_spike_max_pitches=3.0,
        home_half_window=2,
        seed_xy=seed,
    )
    # Whole-array |amp| follows the interferer on unit-10 events.
    unit10 = local["labels"] == 10
    assert np.all(legacy["central"][legacy["labels"] == 10] == interferer)
    np.testing.assert_array_equal(local["central"][unit10], home_a)
    np.testing.assert_array_equal(local["central"][local["labels"] == 20], home_b)
    assert local["meta"]["frac_global_home_outside_radius"] > 0.0
    assert interferer not in set(local["central"].tolist())


def test_radius_oracle_separates_two_grid_units():
    ds, seed, *_ = _two_unit_grid_with_far_interferer()
    rows = run_study(
        ds,
        com_bits=(4,),
        window=15,
        train_frac=0.5,
        same_spike_max_pitches=3.0,
        home_half_window=2,
        seed_xy=seed,
        verbose=False,
    )
    by_method = {r["method"]: r for r in rows}
    assert by_method["main_channel"]["argmin_accuracy"] == 1.0
    assert by_method["main_channel_xy"]["argmin_accuracy"] == 1.0
    assert by_method["com_float"]["argmin_accuracy"] == 1.0
    assert by_method["com_4bit"]["argmin_accuracy"] == 1.0


def test_com_radius_prefilter_drops_far_unit_and_keeps_true():
    rng = np.random.default_rng(0)
    n = 40
    a = rng.normal([0.0, 0.0], 2.0, size=(n, 2))
    b = rng.normal([100.0, 0.0], 2.0, size=(n, 2))
    com = np.empty((2 * n, 2), dtype=np.float64)
    labels = np.empty(2 * n, dtype=np.int64)
    times = np.arange(2 * n, dtype=np.int64)
    for i in range(n):
        com[2 * i] = a[i]
        labels[2 * i] = 1
        com[2 * i + 1] = b[i]
        labels[2 * i + 1] = 2
    tight = evaluate_com_radius_prefilter(com, labels, times, radius_um=20.0)
    wide = evaluate_com_radius_prefilter(com, labels, times, radius_um=200.0)
    assert tight["n_units_train"] == 2
    assert tight["candidate_recall"] == 1.0
    assert tight["mean_candidates"] == 1.0
    assert tight["frac_units_dropped"] == 0.5
    assert wide["candidate_recall"] == 1.0
    assert wide["mean_candidates"] == 2.0
    assert wide["frac_units_dropped"] == 0.0
    too_small = evaluate_com_radius_prefilter(com, labels, times, radius_um=0.01)
    assert too_small["candidate_recall"] < 1.0
    assert too_small["frac_empty"] > 0.0


def test_com_radius_prefilter_rejects_ambiguous_policy():
    com = np.zeros((4, 2), dtype=np.float64)
    labels = np.array([1, 1, 1, 1])
    times = np.arange(4)
    try:
        evaluate_com_radius_prefilter(com, labels, times, radius_um=10.0, per_unit_percentile=95.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both policies are set")
