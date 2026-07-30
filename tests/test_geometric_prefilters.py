"""Unit tests for geometric prefilters and gate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from Spatial.algorithms.geometric_prefilters import (
    causal_first_channel,
    cosine_distance,
    fit_channel_mad_thresholds,
    index_neighborhood_unit_mask,
    integer_cosine_scores,
    jaccard_filter_mask,
    jaccard_scores,
    median_channel_spacing,
    proxy_max_channel,
    tile_activation_mask,
    tile_ids_from_positions,
)
from Spatial.experiments.geometric_prefilter_common import evaluate_gates


def test_proxy_max_tie_breaks_to_smallest_index():
    amps = np.array([[1.0, -1.0, 0.5], [0.2, 0.9, -0.9]])
    assert proxy_max_channel(amps).tolist() == [0, 1]


def test_causal_first_finds_earliest_over_threshold():
    # shape (events, channels, time)
    snippets = np.zeros((2, 3, 5))
    snippets[0, 2, 1] = 10.0  # ch2 at t=1
    snippets[0, 0, 3] = 10.0  # later on ch0 — should lose
    snippets[1, :, :] = 0.0  # no trigger → fallback
    thr = np.array([1.0, 1.0, 1.0])
    anchors, fb = causal_first_channel(snippets, thr, fallback_channels=np.array([1, 1]))
    assert anchors.tolist() == [2, 1]
    assert fb.tolist() == [False, True]


def test_rising_edge_ignores_already_hot_left_edge():
    from Spatial.algorithms.geometric_prefilters import rising_edge_first_channel

    # pre_samples=3 → peak at index 3. Channel 0 is hot from sample 0 (legacy trap).
    # True rising edge: channel 2 goes below→above at sample 2 (rel=-1).
    snippets = np.zeros((1, 3, 6))
    snippets[0, 0, :] = 10.0
    snippets[0, 2, 1] = 0.0
    snippets[0, 2, 2] = 10.0
    thr = np.array([1.0, 1.0, 1.0])
    anchors, fb, rel = rising_edge_first_channel(
        snippets, thr, pre_samples=3, fallback_channels=np.array([1])
    )
    assert fb.tolist() == [False]
    assert anchors.tolist() == [2]
    assert rel.tolist() == [-1]


def test_stream_first_detect_matches_earliest_in_peak_window():
    from Spatial.algorithms.geometric_prefilters import (
        match_stream_first_detect,
        stream_rising_edge_detections,
    )

    # 2 channels, synthetic: ch1 rises at t=10, ch0 rises at t=20.
    filtered = np.zeros((2, 40))
    filtered[1, 10:] = 5.0
    filtered[0, 20:] = 5.0
    thr = np.array([1.0, 1.0])
    det_t, det_ch = stream_rising_edge_detections(filtered, thr, refractory_samples=5)
    assert det_t.tolist()[0] == 10
    assert det_ch.tolist()[0] == 1
    peaks = np.array([12, 25])
    anchors, fb, rel = match_stream_first_detect(
        peaks, det_t, det_ch, pre_samples=5, post_samples=2, fallback_channels=np.array([0, 0])
    )
    assert fb.tolist() == [False, False]
    assert anchors.tolist() == [1, 0]  # peak@12→det@10 ch1; peak@25→det@20 ch0
    assert rel.tolist() == [-2, -5]


def test_index_neighborhood_clamps_and_widens():
    homes = np.array([0, 5, 9])
    anchors = np.array([0, 9])
    mask, eff = index_neighborhood_unit_mask(homes, anchors, 10, radius=1, ensure_nonempty=True)
    assert mask[0].tolist() == [True, False, False]
    assert mask[1].tolist() == [False, False, True]
    # Empty at r=0 for anchor far from all homes → widen
    homes2 = np.array([0, 1])
    mask2, eff2 = index_neighborhood_unit_mask(
        homes2, np.array([9]), 10, radius=0, ensure_nonempty=True
    )
    assert mask2[0].any()
    assert int(eff2[0]) >= 1


def test_jaccard_and_tile_helpers():
    event = np.array([[True, True, False, False]])
    support = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
            [False, False, True, True],
        ]
    )
    scores = jaccard_scores(event, support)
    assert scores.shape == (1, 3)
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] == pytest.approx(0.5)
    assert jaccard_filter_mask(event, support, 0.6)[0].tolist() == [True, False, False]

    geom = np.column_stack([np.zeros(8), np.arange(8) * 20.0])
    pitch = 2 * median_channel_spacing(geom)
    tiles = tile_ids_from_positions(geom, pitch)
    assert tiles[0] == tiles[1]  # 0 and 20 share tile when pitch=40
    mask = tile_activation_mask(np.array([tiles[0]]), np.array([tiles[0], tiles[-1]]), 0)
    assert mask[0, 0] and not mask[0, 1]


def test_cosine_ranks_agree_with_l1_on_aligned_codes():
    rng = np.random.default_rng(0)
    # Distinct non-constant patterns so cosine has discriminative power.
    centroids = np.array(
        [
            [2, 4, 8, 16, 8, 4, 2, 1, 0],
            [0, 1, 2, 4, 8, 16, 8, 4, 2],
        ],
        dtype=np.int64,
    )
    codes = np.vstack(
        [
            np.clip(centroids[i] + rng.integers(-1, 2, size=(20, 9)), 0, 31)
            for i in range(2)
        ]
    )
    cos = integer_cosine_scores(codes, centroids)
    l1 = np.abs(codes[:, None, :] - centroids[None, :, :]).sum(axis=2)
    cos_winners = np.argmax(cos, axis=1)
    l1_winners = np.argmin(l1, axis=1)
    agree = float(np.mean(cos_winners == l1_winners))
    assert agree >= 0.9
    assert cosine_distance(codes, centroids).shape == cos.shape


def test_fit_mad_thresholds_positive():
    rng = np.random.default_rng(1)
    filtered = rng.normal(0, 1, size=(4, 1000))
    thr = fit_channel_mad_thresholds(filtered, sample_start=0, sample_stop=500)
    assert thr.shape == (4,)
    assert np.all(thr > 0)


def test_evaluate_gates_smoke():
    rows = []
    for family, condition, n in (
        ("hj", "drift", 100),
        ("hj", "static", 100),
        ("mearec", "mearec", 100),
    ):
        for method, recall, mean_c, nrad, frac in (
            ("A1_proxy_max", 0.995, 8.0, 2, None),
            ("A2_causal_first", 0.997, 8.0, 2, None),
            ("baseline", 0.99, 12.0, None, None),
            ("D_cosine", 0.989, 12.0, None, None),
            ("E_power_gate", 0.995, 5.0, None, 0.2),
        ):
            row = {
                "dataset": f"{family}_{condition}",
                "family": family,
                "condition": condition,
                "method": method,
                "radius_percentile": 99.9,
                "n_test": n,
                "true_unit_candidate_recall": recall,
                "mean_candidates": mean_c,
                "cfg_directory": "mode_knn7" if method == "baseline" else None,
                "cfg_neighbor_radius": nrad,
                "fraction_tiles_active": frac,
            }
            # Add matching baseline all for G-D
            rows.append(row)
            if method == "baseline":
                rows.append({**row, "cfg_directory": "all", "mean_candidates": 12.0})
    gates = evaluate_gates(rows)
    assert "G_A" in gates and "G_D" in gates and "G_drift" in gates and "G_E" in gates
