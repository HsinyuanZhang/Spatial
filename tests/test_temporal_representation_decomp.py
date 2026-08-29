"""Tests for temporal decomposition helpers (no recording I/O)."""

import numpy as np

from Spatial.algorithms.causal_temporal_sketch import CausalTemporalSketch, DYADIC_DICTIONARY
from Spatial.algorithms.temporal_representation_decomp import (
    assign_among_candidates,
    float_dyadic_features,
)


def test_assign_c0_is_none_and_c1_skips_read():
    feat = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    tmpl = np.array([[0.0, 1.0], [1.0, 0.0]])
    units = np.array([3, 8])
    cand = [np.array([]), np.array([8]), np.array([3, 8])]
    pred, reads = assign_among_candidates(feat, tmpl, units, cand)
    assert pred[0] is None
    assert pred[1] == 8
    assert reads[0] == 0 and reads[1] == 0 and reads[2] == 2


def test_float_dyadic_respects_selected_ids_and_support():
    prefix = np.zeros((1, 24), dtype=np.float64)
    prefix[0, 0:4] = 1.0
    ids = np.array([0], dtype=np.int64)  # first dictionary entry
    entry = DYADIC_DICTIONARY[0]
    feat = float_dyadic_features(prefix[:, : max(entry.final_index + 1, 4)], ids)
    assert feat.shape == (1, 1)
    assert np.isfinite(feat).all()


def test_sketch_prefix_does_not_exceed_horizon():
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 1, size=(40, 64))
    labels = np.array([1] * 20 + [2] * 20)
    sketch = CausalTemporalSketch(delay_samples=8, n_features=8).fit(wave, labels)
    assert sketch.prefix_length == 24
    codes = sketch.quantize_prefix(wave)
    assert codes.shape[1] == 24
