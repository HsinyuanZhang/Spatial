"""Unit tests for spatial-only footprint / soft-localization features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    build_radius_neighbor_table,
    build_relative_patch_table,
    estimate_p2p_amplitude,
    extract_local_p2p,
    local_home_channels,
    main_channel_features,
    main_channel_xy_features,
    median_nearest_neighbor_spacing,
    normalize_geom_xy,
    primary_axis,
    soft_localization_features,
    com_features,
    footprint_p2p_features,
    footprint_pca_features,
    central_ratio_features,
    quantize_com,
    quantize_p2p,
    quantize_signed_adc,
)


def test_p2p_estimators_have_explicit_distinct_semantics():
    """Contiguous and order-statistic averages must not be conflated."""
    segment = np.array([0.0, 6.0, 6.0, 6.0, 0.0, -3.0, -3.0, -3.0, 0.0])
    raw = estimate_p2p_amplitude(segment, estimator="raw", average_points=1)
    centered = estimate_p2p_amplitude(
        segment, estimator="centered_mean", average_points=3
    )
    moving = estimate_p2p_amplitude(
        segment, estimator="moving_average", average_points=3
    )
    extreme = estimate_p2p_amplitude(
        segment, estimator="extreme_mean", average_points=3
    )
    np.testing.assert_allclose([raw, moving, extreme], 9.0)
    # A first-hit raw extrema locator centers the neighborhood at the plateau
    # edge, showing why centered_mean and moving_average are not equivalent.
    np.testing.assert_allclose(centered, 6.0)


def test_moving_average_range_suppresses_isolated_extrema():
    segment = np.zeros(15, dtype=np.float64)
    segment[4] = 12.0
    segment[10] = -9.0
    raw = estimate_p2p_amplitude(segment, estimator="raw", average_points=1)
    moving3 = estimate_p2p_amplitude(
        segment, estimator="moving_average", average_points=3
    )
    moving5 = estimate_p2p_amplitude(
        segment, estimator="moving_average", average_points=5
    )
    np.testing.assert_allclose(raw, 21.0)
    np.testing.assert_allclose(moving3, 7.0)
    np.testing.assert_allclose(moving5, 4.2)


def test_centered_mean_can_still_be_centered_by_impulsive_noise():
    """Peak-neighborhood averaging does not fix a wrong raw peak location."""
    segment = np.array([0.0, 3.0, 3.0, 3.0, 0.0, 10.0, 0.0, -2.0, -2.0, -2.0])
    centered = estimate_p2p_amplitude(
        segment, estimator="centered_mean", average_points=3
    )
    moving = estimate_p2p_amplitude(
        segment, estimator="moving_average", average_points=3
    )
    # centered max is mean([0,10,0]); moving max is the true [3,3,3] plateau.
    assert moving > centered


def test_extreme_mean_is_nonadjacent_order_statistic_average():
    segment = np.array([9.0, 0.0, 8.0, 0.0, 7.0, -9.0, 0.0, -8.0, 0.0, -7.0])
    observed = estimate_p2p_amplitude(
        segment, estimator="extreme_mean", average_points=3
    )
    np.testing.assert_allclose(observed, 16.0)


def test_p2p_estimator_validation_is_fail_closed():
    segment = np.arange(9, dtype=np.float64)
    for kwargs in (
        {"estimator": "raw", "average_points": 3},
        {"estimator": "moving_average", "average_points": 2},
        {"estimator": "moving_average", "average_points": 11},
        {"estimator": "extreme_mean", "average_points": 5},
        {"estimator": "unknown", "average_points": 1},
    ):
        try:
            estimate_p2p_amplitude(segment, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_extract_local_p2p_moving_average_is_opt_in():
    signal = np.zeros((1, 80), dtype=np.float64)
    signal[0, 39] = 12.0
    signal[0, 43] = -9.0
    table = np.array([[0]], dtype=np.int64)
    raw, raw_times, _ = extract_local_p2p(
        signal, np.array([40]), np.array([0]), table, window=10
    )
    averaged, averaged_times, _ = extract_local_p2p(
        signal,
        np.array([40]),
        np.array([0]),
        table,
        window=10,
        p2p_estimator="moving_average",
        average_points=3,
    )
    np.testing.assert_array_equal(raw_times, averaged_times)
    np.testing.assert_allclose(raw[0, 0], 21.0)
    np.testing.assert_allclose(averaged[0, 0], 7.0)


def test_knn_table_includes_self():
    geom = np.column_stack([np.zeros(8), np.arange(8) * 25.0])
    table = build_knn_table_with_self(geom, k_neighbors=5)
    assert table.shape == (8, 5)
    for ch in range(8):
        assert table[ch, 0] == ch


def test_soft_loc_depth_shifts_with_mass():
    """Mass shifted toward deeper channels should increase y centroid."""
    geom = np.column_stack([np.zeros(5), np.arange(5) * 20.0])
    # Two spikes, same neighbor set = all channels
    neighbor_ids = np.tile(np.arange(5), (2, 1))

    # Spike 0: mass on shallow channels
    p2p_shallow = np.array([[10.0, 8.0, 1.0, 0.0, 0.0]])
    # Spike 1: mass on deep channels
    p2p_deep = np.array([[0.0, 0.0, 1.0, 8.0, 10.0]])
    p2p = np.vstack([p2p_shallow, p2p_deep])

    feats = soft_localization_features(p2p, neighbor_ids, geom)
    assert feats.shape == (2, 4)
    assert np.all(np.isfinite(feats))
    # centroid_y_norm is column 1
    assert feats[1, 1] > feats[0, 1]


def test_com_features_is_soft_loc_xy_only():
    geom = np.column_stack([np.zeros(5), np.arange(5) * 20.0])
    neighbor_ids = np.tile(np.arange(5), (2, 1))
    p2p = np.array(
        [
            [10.0, 8.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0, 10.0],
        ]
    )
    soft = soft_localization_features(p2p, neighbor_ids, geom)
    com = com_features(p2p, neighbor_ids, geom)
    assert com.shape == (2, 2)
    np.testing.assert_allclose(com, soft[:, :2])
    assert com[1, 1] > com[0, 1]


def test_footprint_p2p_dim_and_normalize():
    p2p = np.array([[2.0, 4.0, 1.0], [10.0, 0.0, 5.0]])
    fp = footprint_p2p_features(p2p, normalize=True)
    assert fp.shape == (2, 3)
    np.testing.assert_allclose(fp[0].max(), 1.0)
    np.testing.assert_allclose(fp[1].max(), 1.0)
    assert np.all(np.isfinite(fp))


def test_footprint_pca_dim():
    rng = np.random.default_rng(0)
    fp = rng.normal(size=(50, 7))
    pcs = footprint_pca_features(fp, n_components=4)
    assert pcs.shape == (50, 4)
    assert np.all(np.isfinite(pcs))


def test_main_channel_features():
    ch = np.array([0, 7, 15])
    feats = main_channel_features(ch, n_channels=16)
    assert feats.shape == (3, 1)
    np.testing.assert_allclose(feats[0, 0], 0.0)
    np.testing.assert_allclose(feats[-1, 0], 1.0)


def test_extract_local_p2p_basic():
    rng = np.random.default_rng(1)
    n_ch, n_samp = 8, 200
    signal = rng.normal(size=(n_ch, n_samp))
    # Inject a large bipolar pulse on ch3 around t=100
    signal[3, 95:105] = 0.0
    signal[3, 98] = 50.0
    signal[3, 102] = -40.0
    signal[2, 98] = 20.0
    signal[4, 98] = 15.0

    geom = np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 20.0])
    table = build_knn_table_with_self(geom, k_neighbors=5)
    p2p, times, nbr = extract_local_p2p(
        signal,
        spike_times=np.array([100]),
        central_channels=np.array([3]),
        neighbor_table=table,
        window=15,
    )
    assert p2p.shape == (1, 5)
    assert times[0] == 100
    assert nbr[0, 0] == 3
    # Home channel should have largest P2P
    assert p2p[0, 0] >= p2p[0, 1:].max() - 1e-9


def test_quantize_p2p_per_spike_max_levels():
    """Per-spike max mode maps max channel to (2^n-1) and keeps relative order."""
    p2p = np.array([[10.0, 5.0, 0.0], [8.0, 8.0, 2.0]])
    q = quantize_p2p(p2p, n_bits=3, mode="per_spike_max")
    assert q.shape == p2p.shape
    # Max level = 7 for 3 bits
    np.testing.assert_allclose(q[0, 0], 7.0)
    np.testing.assert_allclose(q[0, 1], 4.0)  # round(5/10*7)=4
    np.testing.assert_allclose(q[0, 2], 0.0)
    # Second spike: both 8s hit max
    np.testing.assert_allclose(q[1, 0], 7.0)
    np.testing.assert_allclose(q[1, 1], 7.0)
    assert q[1, 2] < q[1, 0]


def test_quantize_p2p_global_max():
    p2p = np.array([[10.0, 5.0], [2.0, 1.0]])
    q = quantize_p2p(p2p, n_bits=2, mode="global_max")
    # global max=10, levels=3 → first max=3, second row scaled vs 10
    np.testing.assert_allclose(q[0, 0], 3.0)
    np.testing.assert_allclose(q[0, 1], 2.0)  # round(5/10*3)=2
    np.testing.assert_allclose(q[1, 0], 1.0)  # round(2/10*3)=1


def test_quantize_p2p_lsb_clip():
    p2p = np.array([[12.0, 3.0, 0.5]])
    q = quantize_p2p(p2p, n_bits=3, mode="lsb", lsb=2.0)
    # codes: 12/2=6, 3/2=1.5→2, 0.5/2=0.25→0; clip to [0,7]
    np.testing.assert_allclose(q[0], [6.0, 2.0, 0.0])


def test_quantize_p2p_passthrough_none_bits():
    p2p = np.array([[1.5, 2.5]])
    q = quantize_p2p(p2p, n_bits=None, mode="per_spike_max")
    np.testing.assert_allclose(q, p2p)


def test_quantize_com_passthrough_and_fixed_unit_interval():
    com = np.array([[0.0, 1.0], [0.5, 0.25], [-0.1, 1.2]])
    raw = quantize_com(com, None)
    np.testing.assert_allclose(raw, com)
    q5 = quantize_com(com, 5)
    np.testing.assert_allclose(q5[0], [0.0, 31.0])
    np.testing.assert_allclose(q5[1], [16.0, 8.0])  # round(0.5*31), round(0.25*31)
    np.testing.assert_allclose(q5[2], [0.0, 31.0])  # clipped to [0, 1]
    q4 = quantize_com(com[:1], 4)
    np.testing.assert_allclose(q4[0], [0.0, 15.0])


def test_quantize_com_matches_uniform_unit_interval_quantizer():
    from Spatial.algorithms.adaptive_range_search import UniformUnsignedQuantizer

    rng = np.random.default_rng(0)
    com = rng.random((20, 2))
    q = UniformUnsignedQuantizer(n_bits=6, fixed_unit_interval=True)
    q.fit(np.zeros((1, 2), dtype=np.float64))
    np.testing.assert_array_equal(
        quantize_com(com, 6).astype(np.int64), q.transform(com)
    )


def test_quantize_com_rejects_p2p_shaped_input_and_bool_bits():
    try:
        quantize_com(np.array([[1.0, 2.0, 3.0]]), 5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for (n, 3) COM")
    try:
        quantize_com(np.array([[0.1, 0.2]]), True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bool n_bits")


def test_central_ratio_features_slot0_is_one_and_clip():
    p2p = np.array([[10.0, 5.0, 20.0], [4.0, 2.0, 0.0]])
    r = central_ratio_features(p2p, clip=2.0)
    np.testing.assert_allclose(r[:, 0], [1.0, 1.0])
    np.testing.assert_allclose(r[0, 1], 0.5)
    np.testing.assert_allclose(r[0, 2], 2.0)  # 20/10=2 → clip
    np.testing.assert_allclose(r[1, 1], 0.5)


def test_quantize_p2p_central_ratio_mode():
    """Central-ratio 4-bit: slot0 → mid of [0,clip] mapped to max_level/clip*1."""
    p2p = np.array([[10.0, 5.0, 0.0]])
    q = quantize_p2p(p2p, n_bits=4, mode="central_ratio", clip=2.0)
    max_level = 15.0
    # ratio [1, 0.5, 0] / 2 * 15 → [7.5→8, 3.75→4, 0]
    np.testing.assert_allclose(q[0, 0], np.rint(1.0 / 2.0 * max_level))
    np.testing.assert_allclose(q[0, 1], np.rint(0.5 / 2.0 * max_level))
    np.testing.assert_allclose(q[0, 2], 0.0)


def test_central_ratio_zero_central_is_all_zero():
    """central P2P=0 must not blow up via 1/eps."""
    p2p = np.array([[0.0, 5.0, 3.0], [0.0, 0.0, 0.0]])
    r = central_ratio_features(p2p, clip=2.0)
    np.testing.assert_allclose(r, 0.0)
    q = quantize_p2p(p2p, n_bits=4, mode="central_ratio", clip=2.0)
    np.testing.assert_allclose(q, 0.0)


def test_soft_loc_stable_under_light_quantization():
    """4-bit per-spike quant should preserve depth ordering of soft_loc."""
    geom = np.column_stack([np.zeros(5), np.arange(5) * 20.0])
    neighbor_ids = np.tile(np.arange(5), (2, 1))
    p2p = np.array(
        [
            [10.0, 8.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0, 10.0],
        ]
    )
    q = quantize_p2p(p2p, n_bits=4, mode="per_spike_max")
    feats_f = soft_localization_features(p2p, neighbor_ids, geom)
    feats_q = soft_localization_features(q, neighbor_ids, geom)
    assert feats_q[1, 1] > feats_q[0, 1]
    # Centroids should stay close after mild quantization
    assert abs(feats_q[0, 1] - feats_f[0, 1]) < 0.05
    assert abs(feats_q[1, 1] - feats_f[1, 1]) < 0.05


def test_primary_axis_picks_higher_variance():
    geom = np.column_stack([np.zeros(8), np.arange(8) * 25.0])
    assert primary_axis(geom) == 1
    geom_x = np.column_stack([np.arange(8) * 25.0, np.zeros(8)])
    assert primary_axis(geom_x) == 0


def test_relative_patch_table_linear_offsets_and_oob():
    """Linear probe: slots are [h-3..h+3] with -1 past the ends."""
    n = 8
    geom = np.column_stack([np.zeros(n), np.arange(n, dtype=np.float64) * 20.0])
    half = 3
    table = build_relative_patch_table(geom, half_width=half)
    assert table.shape == (n, 2 * half + 1)
    for h in range(n):
        assert table[h, half] == h
        expected = []
        for d in range(-half, half + 1):
            r = h + d
            expected.append(r if 0 <= r < n else -1)
        np.testing.assert_array_equal(table[h], expected)


def test_relative_patch_table_staggered_center_is_home():
    """Two-column stagger: center slot is always the home channel id."""
    # 6 channels: x alternates 0/10, y increases every pair
    xs = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    ys = np.array([0.0, 0.0, 20.0, 20.0, 40.0, 40.0])
    geom = np.column_stack([xs, ys])
    half = 2
    table = build_relative_patch_table(geom, half_width=half)
    assert table.shape == (6, 2 * half + 1)
    assert primary_axis(geom) == 1
    for h in range(6):
        assert table[h, half] == h


def test_extract_relative_patch_oob_zero_p2p():
    """Edge home: OOB slots stay -1 with P2P 0; valid slots extracted."""
    rng = np.random.default_rng(2)
    n_ch, n_samp = 8, 200
    signal = rng.normal(size=(n_ch, n_samp)) * 0.1
    signal[0, 95:105] = 0.0
    signal[0, 98] = 30.0
    signal[0, 102] = -20.0
    signal[1, 98] = 10.0
    geom = np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 20.0])
    table = build_relative_patch_table(geom, half_width=3)
    p2p, times, nbr = extract_local_p2p(
        signal,
        spike_times=np.array([100]),
        central_channels=np.array([0]),
        neighbor_table=table,
        window=15,
    )
    assert times[0] == 100
    assert nbr[0, 3] == 0  # center
    assert nbr[0, 0] == -1 and nbr[0, 1] == -1 and nbr[0, 2] == -1
    np.testing.assert_allclose(p2p[0, :3], 0.0)
    assert p2p[0, 3] > 0.0


def test_soft_loc_ignores_oob_neighbor_ids():
    """Negative neighbor ids must not be used to index geom."""
    geom = np.column_stack([np.zeros(5), np.arange(5) * 20.0])
    neighbor_ids = np.array([[-1, -1, 0, 1, 2]], dtype=np.int64)
    p2p = np.array([[0.0, 0.0, 10.0, 5.0, 1.0]])
    feats = soft_localization_features(p2p, neighbor_ids, geom)
    assert feats.shape == (1, 4)
    assert np.all(np.isfinite(feats))
    # Mass only on channels 0,1,2 → centroid near shallow end
    assert feats[0, 1] < 0.5
    com = com_features(p2p, neighbor_ids, geom)
    np.testing.assert_allclose(com, feats[:, :2])


def _grid_geom(n_side: int = 7, pitch: float = 30.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def test_median_nearest_neighbor_spacing_is_grid_pitch():
    geom = _grid_geom(5, pitch=30.0)
    np.testing.assert_allclose(median_nearest_neighbor_spacing(geom), 30.0)


def test_radius_neighbor_table_includes_self_excludes_far_and_pads():
    geom = _grid_geom(7, pitch=30.0)
    table = build_radius_neighbor_table(geom, radius_um=90.0)
    center = 3 * 7 + 3  # (90, 90)
    row = table[center]
    valid = row[row >= 0]
    assert row[0] == center
    dists = np.linalg.norm(geom[valid] - geom[center], axis=1)
    assert np.all(dists <= 90.0 + 1e-9)
    # Corner (0,0) is 3 pitches on each axis → 127 um, outside the disk.
    assert 0 not in set(valid.tolist())
    # Edge channel has fewer neighbors and is padded with -1.
    edge_valid = table[0] >= 0
    assert int(edge_valid.sum()) < int((row >= 0).sum())
    assert np.all(table[0, ~edge_valid] == -1)


def test_local_home_ignores_far_stronger_trough():
    """A louder trough more than 3 pitches away is a different spike."""
    geom = _grid_geom(7, pitch=30.0)
    n_ch = geom.shape[0]
    signal = np.zeros((n_ch, 80), dtype=np.float64)
    home = 3 * 7 + 3
    far = 0
    t = 40
    signal[home, t - 1 : t + 2] = -8.0
    signal[far, t - 1 : t + 2] = -50.0
    radius = 3.0 * median_nearest_neighbor_spacing(geom)
    assert np.linalg.norm(geom[far] - geom[home]) > radius
    homes, n_in = local_home_channels(
        signal,
        np.array([t]),
        geom[home][None, :],
        geom,
        radius,
        half_window=2,
    )
    assert homes[0] == home
    assert n_in[0] >= 1
    table = build_radius_neighbor_table(geom, radius)
    p2p, _, nbr = extract_local_p2p(
        signal, np.array([t]), homes, table, window=15
    )
    assert far not in set(nbr[0, nbr[0] >= 0].tolist())
    com = com_features(p2p, nbr, geom)
    com_xy = com[0] * (geom.max(axis=0) - geom.min(axis=0)) + geom.min(axis=0)
    assert np.linalg.norm(com_xy - geom[home]) < 30.0


def test_main_channel_xy_matches_geom_bbox_normalization():
    geom = _grid_geom(4, pitch=30.0)
    ch = np.array([0, geom.shape[0] - 1], dtype=np.int64)
    xy = main_channel_xy_features(ch, geom)
    expected = normalize_geom_xy(geom[ch], geom)
    np.testing.assert_allclose(xy, expected)
    np.testing.assert_allclose(xy[0], [0.0, 0.0])
    np.testing.assert_allclose(xy[1], [1.0, 1.0])


def test_signed_adc_maps_full_scale_and_clips():
    x = np.array([0.0, 1.0, -1.0, 1.5, -1.5])
    y = quantize_signed_adc(x, n_bits=5, full_scale=1.0)
    qpos = 15
    lsb = 1.0 / qpos
    np.testing.assert_allclose(y[0], 0.0)
    np.testing.assert_allclose(y[1], 1.0)
    np.testing.assert_allclose(y[2], -1.0)
    np.testing.assert_allclose(y[3], 1.0)
    np.testing.assert_allclose(y[4], -1.0)
    # Mid-level is a multiple of LSB.
    x2 = np.array([3.0 * lsb])
    y2 = quantize_signed_adc(x2, n_bits=5, full_scale=1.0)
    np.testing.assert_allclose(y2[0], 3.0 * lsb)


def test_signed_adc_per_channel_scale_is_independent():
    x = np.array([[1.0, -1.0], [10.0, -10.0]])
    y = quantize_signed_adc(x, n_bits=5, full_scale=np.array([1.0, 10.0]))
    np.testing.assert_allclose(y[0], [1.0, -1.0])
    np.testing.assert_allclose(y[1], [10.0, -10.0])
    # Shared all-channel scale saturates the larger channel.
    y_shared = quantize_signed_adc(x, n_bits=5, full_scale=1.0)
    np.testing.assert_allclose(y_shared[0], [1.0, -1.0])
    np.testing.assert_allclose(y_shared[1], [1.0, -1.0])
