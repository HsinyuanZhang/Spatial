"""Contract tests for the drift window-budget position readout.

The window-budget argument rests on one property: the COM readout must respond
to spatial motion and must **not** respond to amplitude change. If it inherited
the amplitude sensitivity of ``estimate_best_shift`` (see
``drift_estimator_diagnostics.md``), the measured static noise floor would not
be a valid control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.run_drift_window_budget import (
    axis_row_step,
    block_medians,
    com_along_axis,
    home_migration,
    segment_motion,
)


def _footprint(pos: np.ndarray, centre: float, amp: float, width: float = 25.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((pos - centre) / width) ** 2)


POS = np.arange(16, dtype=np.float64) * 25.0


def test_com_is_invariant_to_global_amplitude_scale():
    """The control property: gain changes must not move the reported position."""
    base = _footprint(POS, centre=187.5, amp=100.0)[None, :]
    ref = com_along_axis(base, POS)[0]
    for g in (0.25, 0.5, 2.0, 10.0):
        scaled = com_along_axis(g * base, POS)[0]
        np.testing.assert_allclose(scaled, ref, rtol=0, atol=1e-9)


def test_com_tracks_true_position():
    centres = np.array([150.0, 175.0, 200.0, 225.0])
    p2p = np.stack([_footprint(POS, c, amp=100.0) for c in centres])
    est = com_along_axis(p2p, POS)
    # Discrete sampling and the amplitude floor bias the absolute value, so the
    # contract is on the response to motion, not on the offset.
    assert np.all(np.diff(est) > 0)
    slope = np.polyfit(centres, est, 1)[0]
    assert 0.5 < slope < 1.5


def test_com_ignores_subthreshold_channels():
    """A far-field pedestal below the amplitude floor must not pull the centre."""
    clean = _footprint(POS, centre=187.5, amp=100.0)[None, :]
    pedestal = clean + 5.0
    np.testing.assert_allclose(
        com_along_axis(pedestal, POS)[0],
        com_along_axis(clean, POS)[0],
        rtol=0,
        atol=1e-9,
    )


def test_block_medians_drops_sparse_blocks():
    times = np.array([0.1, 0.2, 0.3, 5.1, 5.2])
    com = np.array([10.0, 12.0, 11.0, 90.0, 91.0])
    centres, vals, counts = block_medians(com, times, 0.0, 10.0, 5.0)
    assert centres.size == 2
    assert counts.tolist() == [3, 2]
    assert np.isfinite(vals[0]) and vals[0] == 11.0
    assert np.isnan(vals[1])  # only 2 events, below MIN_EVENTS_PER_BLOCK


def test_segment_motion_recovers_a_known_ramp():
    centres = np.arange(2.5, 60.0, 5.0)
    vals = 0.2 * centres  # 0.2 um/s
    mot = segment_motion(centres, vals)
    np.testing.assert_allclose(mot["slope_um_per_s"], 0.2, rtol=1e-6)
    np.testing.assert_allclose(mot["net_um"], 0.2 * (centres[-1] - centres[0]), rtol=1e-6)


def test_segment_motion_is_flat_without_drift():
    centres = np.arange(2.5, 60.0, 5.0)
    mot = segment_motion(centres, np.full(centres.size, 42.0))
    np.testing.assert_allclose(mot["slope_um_per_s"], 0.0, atol=1e-9)
    np.testing.assert_allclose(mot["spread_um"], 0.0, atol=1e-9)
    np.testing.assert_allclose(mot["excursion_um"], 0.0, atol=1e-9)


def test_excursion_sees_an_out_and_back_trip_that_the_trend_misses():
    """The HJ drift profile returns toward its origin; the trend cancels."""
    centres = np.arange(2.5, 600.0, 5.0)
    # Symmetric out-and-back ramp: 0 -> 40 um -> 0, as in the HJ drift profile.
    vals = 40.0 * (1.0 - np.abs(centres - 300.0) / 300.0)
    mot = segment_motion(centres, vals)
    assert abs(mot["net_um"]) < 1.0  # symmetry cancels the trend entirely
    assert mot["excursion_um"] > 35.0  # but the unit did travel ~40 um


def test_excursion_smoothing_rejects_single_block_spikes():
    centres = np.arange(2.5, 100.0, 5.0)
    vals = np.zeros(centres.size)
    vals[7] = 50.0
    mot = segment_motion(centres, vals)
    np.testing.assert_allclose(mot["excursion_um"], 0.0, atol=1e-9)


# --- W3: channel selection -------------------------------------------------

# Staggered two-column layout, the Hybrid Janelia 16 ch siprobe geometry.
STAGGERED = np.array(
    [[43.0, 80.0], [11.0, 80.0], [57.0, 100.0], [27.0, 100.0]] * 1
    + [[43.0, 120.0], [11.0, 120.0], [57.0, 140.0], [27.0, 140.0]]
)


def test_row_step_is_not_the_nearest_neighbour_pitch_on_a_staggered_probe():
    """The load-bearing distinction behind the shift-grid range claim."""
    from Spatial.algorithms.spatial_footprint import median_nearest_neighbor_spacing

    step = axis_row_step(STAGGERED, axis=1)
    np.testing.assert_allclose(step, 20.0)
    # The NN distance is the diagonal to the other column, and is larger.
    assert median_nearest_neighbor_spacing(STAGGERED) > step


def test_home_migration_is_silent_when_the_peak_channel_never_moves():
    times = np.linspace(0.0, 599.0, 1200)
    peak = np.full(times.size, 7, dtype=np.int64)
    res = home_migration(peak, times, n_ch=16)
    assert res["fit_home"] == 7
    assert res["n_blocks_home_differs"] == 0
    assert res["n_distinct_block_homes"] == 1
    np.testing.assert_allclose(res["frac_events_off_fit_home"], 0.0)


def test_home_migration_flags_a_channel_that_moves_after_the_fit_split():
    times = np.linspace(0.0, 599.0, 1200)
    peak = np.where(times < 300.0, 7, 8).astype(np.int64)
    res = home_migration(peak, times, n_ch=16)
    assert res["fit_home"] == 7  # fit split is the first 30 s, still on 7
    assert res["n_distinct_block_homes"] == 2
    # Roughly half the recording sits on the other channel.
    assert 0.4 < res["frac_blocks_home_differs"] < 0.6
    assert 0.4 < res["frac_events_off_fit_home"] < 0.6


def test_home_migration_ignores_a_minority_of_stray_events():
    """A block keeps its majority home when only a few events stray."""
    times = np.linspace(0.0, 599.0, 1200)
    peak = np.full(times.size, 7, dtype=np.int64)
    peak[::10] = 9
    res = home_migration(peak, times, n_ch=16)
    assert res["n_blocks_home_differs"] == 0
    np.testing.assert_allclose(res["frac_events_off_fit_home"], 0.1, atol=1e-9)
