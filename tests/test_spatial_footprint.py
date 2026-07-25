"""Unit tests for spatial-only footprint / soft-localization features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    extract_local_p2p,
    main_channel_features,
    soft_localization_features,
    com_features,
    footprint_p2p_features,
    footprint_pca_features,
    central_ratio_features,
    quantize_p2p,
)


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
