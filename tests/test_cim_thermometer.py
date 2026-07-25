"""Tests for bipolar thermometer CAM encodings."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.cim_thermometer import (
    remove_dc,
    fit_thermometer_thresholds,
    thermometer_encode,
    bipolar_thermometer_encode,
    fit_bipolar_thermometer,
    hamming_distance_rows,
    signed_uniform_quantize,
)


def test_remove_dc_zero_mean():
    w = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 9.0]])
    x = remove_dc(w)
    np.testing.assert_allclose(x.mean(axis=1), 0.0, atol=1e-12)


def test_thermometer_monotonic_hamming():
    thr = np.array([1.0, 2.0, 3.0, 4.0])
    a = thermometer_encode(np.array([0.0, 1.5, 3.5, 5.0]), thr)
    # 0 → 0000, 1.5 → 1000, 3.5 → 1110, 5 → 1111
    np.testing.assert_array_equal(a[0], [0, 0, 0, 0])
    np.testing.assert_array_equal(a[1], [1, 0, 0, 0])
    np.testing.assert_array_equal(a[3], [1, 1, 1, 1])
    # Larger amp gap → larger Hamming
    d01 = int(np.sum(a[0] != a[1]))
    d03 = int(np.sum(a[0] != a[3]))
    assert d03 > d01


def test_bipolar_encode_shape_and_pos_neg():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 8))
    thr = fit_bipolar_thermometer(w, n_cells=4, mode="quantile")
    assert thr.shape == (4,)
    codes = bipolar_thermometer_encode(w, thr)
    assert codes.shape == (5, 8 * 2 * 4)
    assert set(np.unique(codes)).issubset({0, 1})


def test_quantile_vs_uniform_thresholds_differ():
    rng = np.random.default_rng(1)
    # Skewed amplitudes
    amp = np.concatenate([rng.exponential(0.2, 1000), rng.exponential(2.0, 50)])
    tq = fit_thermometer_thresholds(amp, n_cells=4, mode="quantile")
    tu = fit_thermometer_thresholds(amp, n_cells=4, mode="uniform")
    assert tq.shape == tu.shape == (4,)
    assert not np.allclose(tq, tu)


def test_hamming_distance_rows():
    a = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    b = np.array([0, 1, 1], dtype=np.uint8)
    d = hamming_distance_rows(a, b)
    np.testing.assert_array_equal(d, [1, 1])


def test_signed_2bit_levels():
    w = np.array([[-10.0, 0.0, 10.0]])
    q = signed_uniform_quantize(w, n_bits=2, scale=10.0)
    assert q.shape == (1, 3)
    assert q[0, 0] <= 0 and q[0, 2] >= 0
