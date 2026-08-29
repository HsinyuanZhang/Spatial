"""KS4-paper simulation helpers (no 40 GB I/O)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.ks4_sim import parse_shank_geom


def test_parse_shank_geom_two_column_np():
    # 4 sites: (col0,row0), (col1,row0), (col0,row1), (col1,row1)
    meta = (
        "~snsShankMap=(1,2,480)"
        "(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)\n"
    )
    geom = parse_shank_geom(meta, n_ap=4)
    np.testing.assert_allclose(geom[0], [0.0, 0.0])
    np.testing.assert_allclose(geom[1], [32.0, 0.0])
    np.testing.assert_allclose(geom[2], [0.0, 20.0])
    np.testing.assert_allclose(geom[3], [32.0, 20.0])


def test_quantize_span_full_scale_and_clip():
    from Spatial.experiments.analyze_np_com_lowbit import quantize_span

    x = np.array([0.0, 60.0, 120.0, -10.0, 200.0])
    codes = quantize_span(x, n_bits=5, lo=0.0, hi=120.0).ravel()
    assert codes[0] == 0
    assert codes[1] == 16 or codes[1] == 15  # 60/120 * 31
    assert codes[2] == 31
    assert codes[3] == 0
    assert codes[4] == 31
    np.testing.assert_allclose(codes[1], np.rint(60.0 / 120.0 * 31.0))
