"""Modified SOM clustering following Akhoundi et al. 2025.

Key differences from standard SOM:
  - Grid elements represent cluster centroids in spatial feature space
  - BMU update uses exponential moving average with factor alpha
  - Batch-based frequency pruning removes low-activity elements
  - Training converges when active cluster count stabilizes
"""

import numpy as np


class SpatialSOM:
    """Modified Self-Organizing Map for spatial spike clustering.

    Args:
        grid_size: Initial number of grid elements per axis (total = grid_size^2 for 2D).
        alpha: BMU update factor. new_pos = ((alpha-1)*old + feature) / alpha.
        beta: Frequency pruning divisor. threshold = n_spikes_batch / (n_active * beta).
        batch_samples: Number of spikes per training batch.
        max_epochs: Maximum training epochs.
        convergence_window: Number of consecutive epochs with stable cluster count to stop.
    """

    def __init__(
        self,
        grid_size: int = 64,
        alpha: int = 16,
        beta: int = 4,
        batch_samples: int = 2000,
        max_epochs: int = 50,
        convergence_window: int = 5,
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.batch_samples = batch_samples
        self.max_epochs = max_epochs
        self.convergence_window = convergence_window
        self.rng = np.random.default_rng(seed)

        self.centroids = None  # (n_active, n_features)
        self.active_mask = None
        self.frequency = None
        self.n_active_history = []
        self.n_epochs = 0
        self.converged = False
        self.trained = False

    def initialize(self, features: np.ndarray):
        """Initialize grid centroids spanning the feature space.

        Uses a uniform grid over [min, max] of each feature dimension.
        """
        n_features = features.shape[1]
        n_init = min(self.grid_size, len(features))

        # Initialize by sampling from data distribution + small jitter
        indices = self.rng.choice(len(features), size=n_init, replace=False)
        self.centroids = features[indices].copy()
        jitter = (features.max(axis=0) - features.min(axis=0)) * 0.01
        self.centroids += self.rng.normal(0, 1, self.centroids.shape) * jitter[None, :]

        self.active_mask = np.ones(n_init, dtype=bool)
        self.frequency = np.zeros(n_init, dtype=np.int64)
        self.n_active_history = []
        self.n_epochs = 0
        self.converged = False
        self.trained = False

    def _find_bmu(self, feature: np.ndarray) -> int:
        """Find Best Matching Unit — nearest active centroid."""
        active_idx = np.where(self.active_mask)[0]
        dists = np.linalg.norm(self.centroids[active_idx] - feature[None, :], axis=1)
        return active_idx[np.argmin(dists)]

    def _find_bmu_batch(self, features: np.ndarray) -> np.ndarray:
        """Vectorized BMU search for a batch of features."""
        active_idx = np.where(self.active_mask)[0]
        active_centroids = self.centroids[active_idx]
        # (n_spikes, n_active)
        dists = np.linalg.norm(
            features[:, None, :] - active_centroids[None, :, :], axis=2
        )
        bmu_local = np.argmin(dists, axis=1)
        return active_idx[bmu_local]

    def _update_bmu(self, bmu_idx: int, feature: np.ndarray):
        """Update BMU centroid: new = ((alpha-1)*old + feature) / alpha."""
        self.centroids[bmu_idx] = (
            (self.alpha - 1) * self.centroids[bmu_idx] + feature
        ) / self.alpha
        self.frequency[bmu_idx] += 1

    def _end_batch(self, n_spikes_in_batch: int):
        """Prune low-frequency elements at end of batch.

        Threshold T = n_spikes / (n_active * beta).
        Elements with frequency < T are deactivated.
        """
        n_active = np.sum(self.active_mask)
        if n_active <= 1:
            return

        threshold = n_spikes_in_batch / (n_active * self.beta)
        prune_mask = self.active_mask & (self.frequency < threshold)
        self.active_mask[prune_mask] = False

        # Reset frequency counters for next batch
        self.frequency[:] = 0

    def _merge_nearby(self, merge_dist: float):
        """Merge active centroids that are closer than merge_dist.

        The higher-frequency centroid absorbs the lower-frequency one.
        """
        active_idx = np.where(self.active_mask)[0]
        if len(active_idx) <= 1:
            return

        active_centroids = self.centroids[active_idx]
        # Pairwise distances
        dists = np.linalg.norm(
            active_centroids[:, None, :] - active_centroids[None, :, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)

        merged = set()
        for i in range(len(active_idx)):
            if i in merged:
                continue
            close = np.where(dists[i] < merge_dist)[0]
            for j in close:
                if j in merged or j <= i:
                    continue
                # Merge j into i (keep higher frequency)
                gi, gj = active_idx[i], active_idx[j]
                fi, fj = self.frequency[gi], self.frequency[gj]
                if fj > fi:
                    gi, gj = gj, gi
                # Weighted average
                total_f = self.frequency[gi] + self.frequency[gj]
                if total_f > 0:
                    self.centroids[gi] = (
                        self.frequency[gi] * self.centroids[gi]
                        + self.frequency[gj] * self.centroids[gj]
                    ) / total_f
                self.frequency[gi] = total_f
                self.active_mask[gj] = False
                merged.add(j)

    def train(self, features: np.ndarray, verbose: bool = False):
        """Train the SOM on spatial features.

        Args:
            features: (n_spikes, n_features) spatial feature vectors.
            verbose: Print convergence info.
        """
        self.initialize(features)
        n_spikes = len(features)
        stable_count = 0
        prev_n_active = np.sum(self.active_mask)

        # Merge distance: fraction of feature space extent
        feat_extent = np.linalg.norm(features.max(axis=0) - features.min(axis=0))
        merge_dist = feat_extent / self.grid_size * 2.0

        converged = False
        n_epochs = 0
        for epoch in range(self.max_epochs):
            n_epochs = epoch + 1
            # Shuffle and process in batches
            perm = self.rng.permutation(n_spikes)

            for batch_start in range(0, n_spikes, self.batch_samples):
                batch_idx = perm[batch_start:batch_start + self.batch_samples]
                batch_features = features[batch_idx]

                # Find and update BMUs
                bmu_indices = self._find_bmu_batch(batch_features)
                for i, bmu in enumerate(bmu_indices):
                    self._update_bmu(bmu, batch_features[i])

                # End of batch: prune + merge nearby
                self._end_batch(len(batch_idx))
                self._merge_nearby(merge_dist)

            n_active = int(np.sum(self.active_mask))
            self.n_active_history.append(n_active)

            if verbose:
                print(f"  Epoch {epoch+1}: {n_active} active clusters")

            # Check convergence
            if n_active == prev_n_active:
                stable_count += 1
            else:
                stable_count = 0
            prev_n_active = n_active

            if stable_count >= self.convergence_window:
                converged = True
                if verbose:
                    print(f"  Converged at epoch {epoch+1} with {n_active} clusters")
                break

        self.n_epochs = int(n_epochs)
        self.converged = bool(converged)
        self.trained = True

    def assign(self, features: np.ndarray) -> np.ndarray:
        """Assign features to nearest active cluster.

        Returns:
            labels: (n_spikes,) cluster index for each spike.
        """
        if not self.trained:
            raise RuntimeError("SOM not trained. Call train() first.")

        active_idx = np.where(self.active_mask)[0]
        active_centroids = self.centroids[active_idx]

        # (n_spikes, n_active) distances
        dists = np.linalg.norm(
            features[:, None, :] - active_centroids[None, :, :], axis=2
        )
        local_labels = np.argmin(dists, axis=1)

        # Map back to global indices
        labels = active_idx[local_labels]
        return labels

    @property
    def n_clusters(self) -> int:
        return int(np.sum(self.active_mask)) if self.active_mask is not None else 0

    @property
    def active_centroids(self) -> np.ndarray:
        if self.centroids is None:
            return np.array([])
        return self.centroids[self.active_mask]
