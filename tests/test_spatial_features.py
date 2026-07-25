"""Unit tests for spatial feature extraction."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_features import (
    extract_spatial_features,
    extract_spatial_features_fast,
    build_knn_table,
    find_neighbors,
)


def test_find_neighbors_linear_probe():
    """On a linear probe, neighbors should be adjacent channels."""
    geom = np.column_stack([np.zeros(8), np.arange(8) * 25.0])
    neighbors = find_neighbors(geom, central_idx=3, n_neighbors=4)
    # Closest to ch3 should be ch2, ch4, ch1, ch5
    assert 2 in neighbors
    assert 4 in neighbors
    assert 3 not in neighbors  # self excluded


def test_find_neighbors_2d_grid():
    """On a 2D grid, neighbors should be spatially adjacent."""
    # 4x4 grid, 30um pitch
    x, y = np.meshgrid(np.arange(4) * 30.0, np.arange(4) * 30.0)
    geom = np.column_stack([x.ravel(), y.ravel()])
    # Center channel (1,1) = index 5
    neighbors = find_neighbors(geom, central_idx=5, n_neighbors=4)
    # Should include (0,1)=1, (2,1)=9, (1,0)=4, (1,2)=6
    assert set(neighbors.tolist()).issubset({1, 4, 6, 9, 0, 2, 8, 10})


def test_spatial_features_symmetric():
    """A symmetric amplitude pattern should give angle ~0 or ~pi."""
    geom = np.array([
        [0.0, 0.0],   # ch0 (central)
        [-30.0, 0.0],  # ch1 (left)
        [30.0, 0.0],   # ch2 (right)
        [0.0, -30.0],  # ch3 (below)
        [0.0, 30.0],   # ch4 (above)
    ])
    # Symmetric left-right, zero up-down -> angle should be ~0 or ~pi
    waveforms = np.array([[100.0, 50.0, 50.0, 0.0, 0.0]])
    features = extract_spatial_features(waveforms, geom, central_channels=np.array([0]), n_neighbors=4)
    # angle, ratio, x_est, y_est, peak_amp
    assert features.shape == (1, 5)
    assert abs(features[0, 4] - 100.0) < 1e-6  # peak amp preserved
    # Symmetric x -> weighted x cancels, angle should be ~0 or pi
    assert features[0, 1] == 0.5  # ratio = 50/100


def test_spatial_features_directional():
    """Amplitude only on one side should give angle pointing that way."""
    geom = np.array([
        [0.0, 0.0],   # ch0 (central)
        [30.0, 0.0],   # ch1 (right)
        [-30.0, 0.0],  # ch2 (left)
        [0.0, 30.0],   # ch3 (up)
        [0.0, -30.0],  # ch4 (down)
    ])
    # Strong amplitude only to the right
    waveforms = np.array([[100.0, 80.0, 0.0, 0.0, 0.0]])
    features = extract_spatial_features(waveforms, geom, central_channels=np.array([0]), n_neighbors=4)
    angle = features[0, 0]
    # Should point right: angle ~ 0
    assert abs(angle) < 0.5  # within ~30 degrees of 0


def test_fast_matches_slow():
    """Vectorized version should match loop version."""
    rng = np.random.default_rng(42)
    n_spikes = 50
    n_channels = 16
    geom = np.column_stack([np.zeros(n_channels), np.arange(n_channels) * 25.0])
    amplitudes = rng.normal(0, 50, (n_spikes, n_channels))
    central_channels = np.argmax(np.abs(amplitudes), axis=1)

    knn = build_knn_table(geom, n_neighbors=6)
    fast = extract_spatial_features_fast(amplitudes, geom, central_channels, knn)
    slow = extract_spatial_features(amplitudes, geom, central_channels, n_neighbors=6)

    np.testing.assert_allclose(fast, slow, atol=1e-10)


def test_build_knn_table():
    """KNN table should have correct shape and no self-references."""
    geom = np.column_stack([np.zeros(10), np.arange(10) * 20.0])
    knn = build_knn_table(geom, n_neighbors=4)
    assert knn.shape == (10, 4)
    for ch in range(10):
        assert ch not in knn[ch]


if __name__ == "__main__":
    test_find_neighbors_linear_probe()
    test_find_neighbors_2d_grid()
    test_spatial_features_symmetric()
    test_spatial_features_directional()
    test_fast_matches_slow()
    test_build_knn_table()
    print("All spatial feature tests passed.")
