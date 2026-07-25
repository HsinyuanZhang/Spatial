"""Tests for unsupervised two-stage COM∧P2P → waveform clustering."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.unsupervised_two_stage import (
    discover_hierarchical_labels,
    unsupervised_two_stage_assign,
)
from Spatial.evaluation.metrics import clustering_accuracy


def _synthetic_two_units_same_com(rng, n_per=80):
    """Two units share COM location but differ in P2P footprint and waveform."""
    com0 = rng.normal([0.0, 0.0], 0.05, size=(n_per, 2))
    com1 = rng.normal([0.0, 0.0], 0.05, size=(n_per, 2))
    com = np.vstack([com0, com1])

    p2p0 = np.tile([1.0, 0.1, 0.0, 0.0], (n_per, 1)) + rng.normal(0, 0.02, (n_per, 4))
    p2p1 = np.tile([0.1, 1.0, 0.0, 0.0], (n_per, 1)) + rng.normal(0, 0.02, (n_per, 4))
    p2p = np.vstack([p2p0, p2p1])

    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    wave0 = np.sin(t)[None, :] + rng.normal(0, 0.05, (n_per, 16))
    wave1 = np.cos(t)[None, :] + rng.normal(0, 0.05, (n_per, 16))
    wave = np.vstack([wave0, wave1])

    labels = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    times = np.arange(len(labels), dtype=np.int64)
    return com, p2p, wave, labels, times


def test_hierarchical_discovery_splits_colocated_units():
    rng = np.random.default_rng(0)
    com, p2p, wave, gt, _ = _synthetic_two_units_same_com(rng, n_per=60)

    # Train on all (discovery only); spatial-only would collapse, hierarchy should split
    disc, disc_meta = discover_hierarchical_labels(
        com,
        wave,
        spatial_grid=8,
        wave_grid=8,
        min_cluster_size=15,
        spatial_beta=4,
        wave_beta=4,
        seed=0,
    )
    n_clusters = len(np.unique(disc))
    assert n_clusters >= 2
    assert "spatial_n_epochs" in disc_meta
    assert "spatial_converged" in disc_meta

    # Hungarian should recover high purity vs GT
    metrics = clustering_accuracy(disc, gt, np.arange(len(gt)), np.arange(len(gt)))
    assert metrics["accuracy"] >= 0.85


def _interleaved_split(n: int, train_frac: float = 0.5):
    """Even/odd split so both classes appear in train when data is blocked by unit."""
    idx = np.arange(n)
    tr = idx[idx % 2 == 0]
    te = idx[idx % 2 == 1]
    if train_frac != 0.5:
        n_train = max(1, int(n * train_frac))
        tr, te = idx[:n_train], idx[n_train:]
    return tr, te


def test_unsupervised_two_stage_l1_and_l2():
    rng = np.random.default_rng(1)
    com, p2p, wave, gt, times = _synthetic_two_units_same_com(rng, n_per=100)
    tr, te = _interleaved_split(len(gt))

    for metric in ("l2", "l1"):
        pred, meta = unsupervised_two_stage_assign(
            com[tr],
            p2p[tr],
            wave[tr],
            com[te],
            p2p[te],
            wave[te],
            p2p_metric=metric,
            spatial_grid=8,
            wave_grid=8,
            min_cluster_size=15,
            seed=1,
        )
        assert meta["p2p_metric"] == metric
        assert meta["n_discovered_clusters"] >= 2
        assert len(pred) == len(te)
        metrics = clustering_accuracy(pred, gt[te], times[te], times[te])
        assert metrics["accuracy"] >= 0.80, f"{metric} acc={metrics['accuracy']}"


def test_unsupervised_two_stage_uses_no_gt_labels():
    """Smoke: API does not take GT labels for fitting."""
    rng = np.random.default_rng(2)
    com, p2p, wave, gt, times = _synthetic_two_units_same_com(rng, n_per=40)
    tr, te = _interleaved_split(len(gt))
    pred, meta = unsupervised_two_stage_assign(
        com[tr], p2p[tr], wave[tr], com[te], p2p[te], wave[te],
        p2p_metric="l1",
        spatial_grid=6,
        wave_grid=6,
        min_cluster_size=10,
        seed=2,
    )
    assert "labels_train" in meta
    assert "spatial_n_epochs" in meta
    assert "spatial_converged" in meta
    assert meta["n_discovered_clusters"] >= 1
    assert len(pred) == len(te)
