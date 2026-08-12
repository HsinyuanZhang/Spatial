"""Unit tests for shape-orthogonal descriptor feature primitives."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import (
    build_relative_patch_table,
    extract_local_extrema,
    extract_local_p2p,
    latency_codes,
    posneg_codes,
    slot_liveness,
    width_codes,
)


def _linear_geom(n_channels: int = 8) -> np.ndarray:
    return np.column_stack([np.zeros(n_channels), np.arange(n_channels) * 20.0])


def test_extract_local_extrema_first_occurrence_on_plateau():
    signal = np.zeros((1, 40), dtype=np.float64)
    # window=5 → segment length 10; plateau max at indices 2 and 3
    signal[0, 10:20] = np.array([0.0, 1.0, 5.0, 5.0, 2.0, 0.0, -1.0, -2.0, 0.0, 0.0])
    table = np.array([[0]], dtype=np.int64)
    vmax, vmin, imax, imin, _, _ = extract_local_extrema(
        signal, np.array([15]), np.array([0]), table, window=5
    )
    assert imax[0, 0] == 2
    np.testing.assert_allclose(vmax[0, 0], 5.0)
    assert imin[0, 0] == 7
    np.testing.assert_allclose(vmin[0, 0], -2.0)


def test_extract_local_extrema_aligns_with_extract_local_p2p():
    rng = np.random.default_rng(11)
    n_ch, n_samp = 8, 200
    signal = rng.normal(size=(n_ch, n_samp)) * 0.2
    signal[3, 95:105] = np.linspace(-4.0, 6.0, 10)
    signal[2, 98] = 3.0
    signal[4, 101] = -2.5

    geom = _linear_geom(n_ch)
    table = build_relative_patch_table(geom, half_width=3)
    spike_times = np.array([50, 100, 150, 5, 195])
    central_channels = np.array([0, 3, 7, 2, 4])

    p2p, p2p_times, p2p_nbr = extract_local_p2p(
        signal, spike_times, central_channels, table, window=15
    )
    vmax, vmin, _, _, ext_times, ext_nbr = extract_local_extrema(
        signal, spike_times, central_channels, table, window=15
    )

    np.testing.assert_array_equal(ext_times, p2p_times)
    np.testing.assert_array_equal(ext_nbr, p2p_nbr)
    np.testing.assert_allclose(vmax - vmin, p2p)


def test_extract_local_extrema_oob_slots_are_neutral():
    signal = np.zeros((4, 80), dtype=np.float64)
    signal[0, 40] = 8.0
    signal[0, 44] = -6.0
    geom = _linear_geom(4)
    table = build_relative_patch_table(geom, half_width=3)
    vmax, vmin, imax, imin, _, nbr = extract_local_extrema(
        signal, np.array([40]), np.array([0]), table, window=10
    )
    oob = nbr[0] < 0
    np.testing.assert_allclose(vmax[0, oob], 0.0)
    np.testing.assert_allclose(vmin[0, oob], 0.0)
    np.testing.assert_array_equal(imax[0, oob], 0)
    np.testing.assert_array_equal(imin[0, oob], 0)

    codes = posneg_codes(vmax, vmin, nbr, n_bits=5)
    k = table.shape[1]
    np.testing.assert_allclose(codes[0, :k][oob], 0.0)
    np.testing.assert_allclose(codes[0, k:][oob], 0.0)


def test_slot_liveness_boundary_and_oob():
    vmax = np.array([[10.0, 4.0, 0.0, 8.0]], dtype=np.float64)
    vmin = np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float64)
    nbr = np.array([[-1, 0, 1, 2]], dtype=np.int64)
    # in-bounds scale = max(10, 4, 6) = 10; gate 0.25 → threshold 2.5
    live = slot_liveness(vmax, vmin, nbr, amp_gate=0.25)
    assert live.shape == (1, 4)
    assert not live[0, 0]
    assert live[0, 1]
    assert not live[0, 2]
    assert live[0, 3]

    vmax_edge = np.array([[10.0, 2.5, 2.49]], dtype=np.float64)
    vmin_edge = np.zeros((1, 3), dtype=np.float64)
    nbr3 = np.array([[0, 1, 2]], dtype=np.int64)
    live_edge = slot_liveness(vmax_edge, vmin_edge, nbr3, amp_gate=0.25)
    assert live_edge[0, 1]
    assert not live_edge[0, 2]


def test_posneg_codes_reconstruction_bound_and_max_slot():
    rng = np.random.default_rng(7)
    n, k, n_bits = 40, 5, 5
    vmax = rng.uniform(0.0, 1.0, size=(n, k))
    vmin = -rng.uniform(0.0, 1.0, size=(n, k))
    nbr = np.tile(np.arange(k), (n, 1))

    codes = posneg_codes(vmax, vmin, nbr, n_bits=n_bits)
    max_level = float((1 << n_bits) - 1)
    assert codes.shape == (n, 2 * k)
    assert np.all(codes >= 0.0)
    assert np.all(codes <= max_level)

    p2p = vmax - vmin
    scale = np.maximum(np.max(p2p, axis=1, keepdims=True), 1e-12)
    target = np.rint(max_level * p2p / scale)
    err = np.abs(codes[:, :k] + codes[:, k:] - target)
    assert np.all(err <= 1.0)

    spike_max = np.argmax(p2p, axis=1)
    for i in range(n):
        j = int(spike_max[i])
        np.testing.assert_allclose(codes[i, j] + codes[i, k + j], max_level, atol=1.0)


def test_latency_codes_drops_home_and_neutral_dead():
    imin = np.array(
        [
            [4, 7, 10, 2, 5],
            [1, 3, 8, 6, 9],
        ],
        dtype=np.int64,
    )
    live = np.ones((2, 5), dtype=bool)
    live[0, 1] = False
    home = 2
    codes = latency_codes(imin, live, home_slot=home, n_bits=4)
    assert codes.shape == (2, 4)
    limit = float((1 << (4 - 1)) - 1)

    expected0 = []
    for j in range(5):
        if j == home:
            continue
        if not live[0, j]:
            expected0.append(limit)
        else:
            expected0.append(float(np.clip(imin[0, j] - imin[0, home], -limit, limit) + limit))
    np.testing.assert_allclose(codes[0], expected0)

    neutral = latency_codes(
        np.zeros((1, 5), dtype=np.int64),
        np.zeros((1, 5), dtype=bool),
        home_slot=home,
        n_bits=4,
    )
    np.testing.assert_allclose(neutral, limit)


def test_latency_codes_invariant_to_constant_time_shift():
    imin = np.array([[2, 5, 9, 4, 7]], dtype=np.int64)
    live = np.ones((1, 5), dtype=bool)
    home = 2
    base = latency_codes(imin, live, home_slot=home, n_bits=4)
    shifted = latency_codes(imin + 20, live, home_slot=home, n_bits=4)
    np.testing.assert_allclose(base, shifted)


def test_latency_codes_saturation_clips():
    limit = float((1 << (4 - 1)) - 1)
    imin = np.array([[0, 100, 0, -100]], dtype=np.int64)
    live = np.array([[True, True, True, True]])
    codes = latency_codes(imin, live, home_slot=2, n_bits=4)
    assert codes.shape == (1, 3)
    assert np.all(codes >= 0.0)
    assert np.all(codes <= 2.0 * limit)
    np.testing.assert_allclose(codes[0, 0], limit)
    np.testing.assert_allclose(codes[0, 1], 2.0 * limit)
    np.testing.assert_allclose(codes[0, 2], 0.0)


def test_width_codes_neutral_clipping_and_ordering():
    off = float((1 << (5 - 1)) - 1)
    imax = np.array([[10]], dtype=np.int64)
    imin = np.array([[10]], dtype=np.int64)
    neutral = width_codes(imax, imin, home_slot=0, n_bits=5)
    np.testing.assert_allclose(neutral, off)

    imax_wide = np.array([[25]], dtype=np.int64)
    imin_wide = np.array([[5]], dtype=np.int64)
    wide = width_codes(imax_wide, imin_wide, home_slot=0, n_bits=5)
    np.testing.assert_allclose(wide, 2.0 * off + 1.0)

    imax_clip = np.array([[100]], dtype=np.int64)
    imin_clip = np.array([[0]], dtype=np.int64)
    clipped = width_codes(imax_clip, imin_clip, home_slot=0, n_bits=5)
    np.testing.assert_allclose(clipped, 2.0 * off + 1.0)

    narrow = width_codes(np.array([[12]]), np.array([[10]]), home_slot=0, n_bits=5)
    assert narrow[0, 0] < wide[0, 0]


def test_extract_local_extrema_validation_is_fail_closed():
    signal = np.zeros((2, 40), dtype=np.float64)
    times = np.array([20], dtype=np.int64)
    homes = np.array([0], dtype=np.int64)
    table = np.array([[0, 1]], dtype=np.int64)
    base = dict(
        signal=signal,
        spike_times=times,
        central_channels=homes,
        neighbor_table=table,
        window=10,
    )
    for key, val in (
        ("signal", signal[0]),
        ("central_channels", np.array([0, 1])),
        ("neighbor_table", np.zeros((3, 2), dtype=np.int64)),
        ("window", True),
        ("window", -1),
    ):
        kwargs = dict(base)
        kwargs[key] = val
        try:
            extract_local_extrema(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for extract_local_extrema {key}={val!r}")


def test_slot_liveness_validation_is_fail_closed():
    vmax = np.array([[1.0, 2.0]])
    vmin = np.array([[0.0, 0.0]])
    nbr = np.array([[0, 1]], dtype=np.int64)
    base = dict(vmax=vmax, vmin=vmin, neighbor_ids=nbr)
    for key, val in (
        ("vmax", vmax[0]),
        ("vmin", vmin[:, :1]),
        ("neighbor_ids", nbr.T),
        ("amp_gate", -0.1),
        ("amp_gate", 1.1),
        ("amp_gate", np.nan),
    ):
        kwargs = dict(base)
        kwargs[key] = val
        try:
            slot_liveness(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for slot_liveness {key}={val!r}")


def test_posneg_codes_validation_is_fail_closed():
    vmax = np.array([[2.0, 1.0]])
    vmin = np.array([[0.0, 0.0]])
    nbr = np.array([[0, 1]], dtype=np.int64)
    base = dict(vmax=vmax, vmin=vmin, neighbor_ids=nbr, n_bits=5)
    for key, val in (
        ("n_bits", True),
        ("n_bits", 0),
        ("n_bits", 17),
        ("vmax", vmax[0]),
        ("neighbor_ids", nbr.T),
    ):
        kwargs = dict(base)
        kwargs[key] = val
        try:
            posneg_codes(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for posneg_codes {key}={val!r}")


def test_latency_codes_validation_is_fail_closed():
    imin = np.array([[1, 2, 3]], dtype=np.int64)
    live = np.ones((1, 3), dtype=bool)
    base = dict(imin=imin, live=live, home_slot=1, n_bits=4)
    for key, val in (
        ("n_bits", True),
        ("n_bits", 1),
        ("n_bits", 9),
        ("home_slot", 3),
        ("home_slot", True),
        ("live", live[:, :2]),
        ("imin", imin[0]),
    ):
        kwargs = dict(base)
        kwargs[key] = val
        try:
            latency_codes(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for latency_codes {key}={val!r}")


def test_width_codes_validation_is_fail_closed():
    imax = np.array([[4, 5]], dtype=np.int64)
    imin = np.array([[1, 2]], dtype=np.int64)
    base = dict(imax=imax, imin=imin, home_slot=0, n_bits=5)
    for key, val in (
        ("n_bits", True),
        ("n_bits", 1),
        ("n_bits", 9),
        ("home_slot", 2),
        ("home_slot", True),
        ("imax", imax[0]),
        ("imin", imin[:, :1]),
    ):
        kwargs = dict(base)
        kwargs[key] = val
        try:
            width_codes(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for width_codes {key}={val!r}")
