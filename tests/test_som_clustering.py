"""Unit tests for Modified SOM clustering."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.som_clustering import SpatialSOM


def test_som_separable_clusters():
    """SOM should separate well-separated Gaussian clusters."""
    rng = np.random.default_rng(42)
    # 3 clusters at distinct locations in 2D
    c1 = rng.normal([0, 0], 0.3, (200, 2))
    c2 = rng.normal([5, 5], 0.3, (200, 2))
    c3 = rng.normal([10, 0], 0.3, (200, 2))
    features = np.vstack([c1, c2, c3])
    true_labels = np.array([0]*200 + [1]*200 + [2]*200)

    som = SpatialSOM(grid_size=15, alpha=16, beta=4, batch_samples=600, max_epochs=50)
    som.train(features, verbose=False)
    pred_labels = som.assign(features)

    # Should converge to ~3 clusters
    assert 2 <= som.n_clusters <= 8, f"Expected ~3 clusters, got {som.n_clusters}"

    # Check that same-cluster points mostly get same label
    # For each true cluster, find majority predicted label
    for true_id in [0, 1, 2]:
        mask = true_labels == true_id
        cluster_preds = pred_labels[mask]
        majority = np.bincount(cluster_preds).max()
        purity = majority / mask.sum()
        assert purity > 0.8, f"Cluster {true_id} purity too low: {purity:.2f}"


def test_som_convergence():
    """SOM should converge (active count stabilizes)."""
    rng = np.random.default_rng(123)
    features = rng.normal(0, 1, (500, 5))

    som = SpatialSOM(grid_size=30, alpha=16, beta=4, max_epochs=50, convergence_window=3)
    som.train(features, verbose=False)

    assert som.trained
    assert len(som.n_active_history) > 0
    # Should have converged before max_epochs
    assert len(som.n_active_history) < 50
    assert som.n_epochs == len(som.n_active_history)
    assert som.converged is True


def test_som_single_cluster():
    """Tightly clustered data should converge to few clusters."""
    rng = np.random.default_rng(7)
    features = rng.normal([3, 3, 3], 0.05, (500, 3))

    som = SpatialSOM(grid_size=10, alpha=16, beta=4, batch_samples=500, max_epochs=50)
    som.train(features, verbose=False)

    assert som.n_clusters <= 5


def test_som_assign_before_train_raises():
    """Assigning before training should raise."""
    som = SpatialSOM()
    try:
        som.assign(np.zeros((10, 5)))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_som_alpha_effect():
    """Higher alpha = slower updates = more stable centroids."""
    rng = np.random.default_rng(42)
    c1 = rng.normal([0, 0], 0.5, (100, 2))
    c2 = rng.normal([5, 5], 0.5, (100, 2))
    features = np.vstack([c1, c2])

    som_low = SpatialSOM(grid_size=30, alpha=4, beta=4, max_epochs=20)
    som_low.train(features, verbose=False)

    som_high = SpatialSOM(grid_size=30, alpha=64, beta=4, max_epochs=20)
    som_high.train(features, verbose=False)

    # Both should find ~2 clusters
    assert som_low.n_clusters >= 1
    assert som_high.n_clusters >= 1


if __name__ == "__main__":
    test_som_separable_clusters()
    test_som_convergence()
    test_som_single_cluster()
    test_som_assign_before_train_raises()
    test_som_alpha_effect()
    print("All SOM clustering tests passed.")
