"""Helpers for KS4 oversplit diagnostics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.analyze_ks4_unit_splits import (
    connected_components,
    count_cross_within,
)


def test_count_cross_within_counts_pairs_inside_window():
    a = np.array([100, 200, 400], dtype=np.int64)
    b = np.array([101, 250, 399], dtype=np.int64)
    assert count_cross_within(a, b, 8) == 2
    assert count_cross_within(a, b, 0) == 0


def test_connected_components_merges_pairs():
    comps = connected_components(5, [(0, 1), (1, 2), (3, 3)])
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 1, 3]
