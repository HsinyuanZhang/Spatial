"""Causal COM predictors: no future leak, range membership, freeze-then-update."""

import numpy as np

from Spatial.algorithms.com_tracking import (
    contemporaneous_block_com,
    in_range,
    longest_false_run,
    nearest_channel,
    predict_ema,
    predict_frozen,
    predict_hold,
    predict_velocity,
    stream_every_dt,
    stream_every_n,
    tracking_errors,
)


def test_hold_uses_only_the_past():
    y = np.array([10.0, 12.0, 20.0, 21.0])
    hat = predict_hold(y)
    assert np.isnan(hat[0])
    np.testing.assert_allclose(hat[1:], [10.0, 12.0, 20.0])
    # Mutating y[2] after predicting index 2 must not be possible: hat[2] was 12.
    assert hat[2] == 12.0


def test_frozen_ignores_post_fit_motion():
    t = np.array([2.5, 7.5, 12.5, 32.5, 37.5])
    y = np.array([0.0, 1.0, 2.0, 80.0, 90.0])
    hat = predict_frozen(y, t, t_fit_end=30.0)
    assert np.all(np.isnan(hat[:3]))
    np.testing.assert_allclose(hat[3:], [1.0, 1.0])  # median of first three


def test_ema_two_rates_differ_and_are_causal():
    y = np.array([0.0, 10.0, 20.0, 30.0])
    slow = predict_ema(y, 0.2)
    fast = predict_ema(y, 0.8)
    assert np.isnan(slow[0]) and np.isnan(fast[0])
    np.testing.assert_allclose(slow[1], 0.0)
    np.testing.assert_allclose(fast[1], 0.0)
    # After seeing 0 then 10, hat[2] = (1-a)*0 + a*10 = 10a
    np.testing.assert_allclose(slow[2], 2.0)
    np.testing.assert_allclose(fast[2], 8.0)
    assert abs(fast[3] - y[2]) < abs(slow[3] - y[2])


def test_velocity_extrapolates_previous_step():
    y = np.array([0.0, 5.0, 10.0, 10.0])
    hat = predict_velocity(y)
    assert np.isnan(hat[0])
    np.testing.assert_allclose(hat[1], 0.0)
    np.testing.assert_allclose(hat[2], 10.0)  # 5 + (5-0)
    np.testing.assert_allclose(hat[3], 15.0)  # 10 + (10-5)


def test_range_and_channel_match():
    pos = np.arange(8) * 20.0
    y = np.array([40.0, 40.0, 80.0])
    hat = np.array([41.0, 55.0, 79.0])
    assert in_range(hat - y, 10.0).tolist() == [True, False, True]
    ch_y = nearest_channel(y, pos)
    ch_h = nearest_channel(hat, pos)
    np.testing.assert_array_equal(ch_y, [2, 2, 4])
    np.testing.assert_array_equal(ch_h, [2, 3, 4])
    rec = tracking_errors(y, hat, pos, radii_um=(10.0, 20.0))
    assert rec["n_eval"] == 3
    np.testing.assert_allclose(rec["median_abs_um"], 1.0)  # |1|, 15, 1 → median 1
    assert rec["frac_channel_match"] == 2 / 3
    assert rec["frac_within_10um"] == 2 / 3
    assert rec["longest_miss_10um"] == 1


def test_stream_every_n_is_causal_and_batches():
    y = np.arange(12, dtype=np.float64)
    hat1 = stream_every_n(y, n_commit=1, init_n=2)
    assert np.isnan(hat1[0]) and np.isnan(hat1[1])
    np.testing.assert_allclose(hat1[2], 0.5)  # median of first two
    np.testing.assert_allclose(hat1[3], 2.0)  # committed y[2]
    np.testing.assert_allclose(hat1[4], 3.0)
    hat2 = stream_every_n(y, n_commit=2, init_n=2)
    np.testing.assert_allclose(hat2[2], 0.5)
    np.testing.assert_allclose(hat2[3], 0.5)  # still waiting for 2 events
    np.testing.assert_allclose(hat2[4], 2.5)  # median of y[2], y[3]
    # Current event cannot change its own score.
    y2 = y.copy()
    y2[5] = 999.0
    np.testing.assert_allclose(stream_every_n(y, 2, init_n=2)[5], stream_every_n(y2, 2, init_n=2)[5])


def test_stream_every_dt_commits_on_time_not_on_current_event():
    t = np.array([0.0, 10.0, 31.0, 31.2, 31.4, 32.1, 32.2])
    y = np.array([0.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0])
    hat = stream_every_dt(y, t, dt_s=1.0, t_init_end=30.0)
    assert np.all(np.isnan(hat[:2]))
    np.testing.assert_allclose(hat[2], 1.0)  # median of t<30; then commit y=10
    np.testing.assert_allclose(hat[3], 10.0)
    np.testing.assert_allclose(hat[4], 10.0)
    np.testing.assert_allclose(hat[5], 10.0)  # then commit median(11,12,20)=12
    np.testing.assert_allclose(hat[6], 12.0)
    y2 = y.copy()
    y2[5] = 999.0
    np.testing.assert_allclose(
        stream_every_dt(y, t, 1.0, t_init_end=30.0)[5],
        stream_every_dt(y2, t, 1.0, t_init_end=30.0)[5],
    )


def test_longest_false_run():
    assert longest_false_run(np.array([True, False, False, True, False])) == 2
    assert longest_false_run(np.array([True, True, True])) == 0


def test_contemporaneous_block_median_and_leave_one_out():
    t = np.array([0.5, 1.0, 1.5, 5.5, 6.0, 6.5])
    y = np.array([10.0, 10.0, 40.0, 20.0, 20.0, 20.0])
    hat = contemporaneous_block_com(y, t, block_s=5.0, leave_one_out=False, min_events=3)
    np.testing.assert_allclose(hat[:3], 10.0)  # median(10, 10, 40)
    np.testing.assert_allclose(hat[3:], 20.0)
    loo = contemporaneous_block_com(y, t, block_s=5.0, leave_one_out=True, min_events=3)
    np.testing.assert_allclose(loo[:3], [25.0, 25.0, 10.0])
    np.testing.assert_allclose(loo[3:], 20.0)
    # A 2-event block is below the min_events floor.
    t2 = np.array([0.1, 0.2, 5.1, 5.2])
    y2 = np.array([0.0, 1.0, 10.0, 11.0])
    empty = contemporaneous_block_com(y2, t2, block_s=5.0, min_events=3)
    assert np.all(np.isnan(empty))


def test_oracle_scatter_is_event_vs_true_com_not_a_tracker():
    pos = np.arange(8) * 20.0
    t = np.array([0.5, 1.0, 1.5, 2.0])
    y = np.array([40.0, 40.0, 40.0, 80.0])  # last spike already 40 µm away
    true_com = contemporaneous_block_com(y, t, block_s=5.0, leave_one_out=False, min_events=3)
    rec = tracking_errors(y, true_com, pos, radii_um=(20.0,))
    assert rec["n_eval"] == 4
    assert rec["n_out_20um"] == 1
    np.testing.assert_allclose(rec["frac_out_20um"], 0.25)
    assert rec["n_channel_miss"] == 1
