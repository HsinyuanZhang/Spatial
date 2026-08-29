"""Unit tests for Yger / Zenodo 1205233 conversion helpers."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.yger_zenodo import (
    CORNER_CHANNELS,
    DEFAULT_ARCHIVE_DIR,
    UINT16_OFFSET,
    coincidence_counts,
    convert_mcs_uint16_to_int16,
    kilosort_probe_dict,
    parse_mea256_prb,
    parse_recording_txt,
)


def test_parse_prb_drops_four_corners_and_keeps_peak_channel():
    prb = DEFAULT_ARCHIVE_DIR / "mea_256.prb"
    if not prb.exists():
        import pytest

        pytest.skip(f"Yger probe file not mounted: {prb}")
    probe = parse_mea256_prb(prb)
    assert probe.n_extra == 252
    assert tuple(sorted(set(range(256)) - set(probe.extra_channels.tolist()))) == CORNER_CHANNELS
    assert probe.ks_index(69) == 69
    np.testing.assert_allclose(probe.geom_extra[probe.ks_index(69)], [360.0, 300.0])
    ks = kilosort_probe_dict(probe)
    assert ks["n_chan"] == 252
    assert ks["chanMap"].shape == (252,)
    assert ks["xc"].shape == (252,)


def test_parse_recording_txt(tmp_path: Path):
    path = tmp_path / "patch.txt"
    path.write_text(
        "threshold = 8\nname = 20160415/patch_2_MEA\njuxta_channel = 254\n"
        "min_peak = -41.6849\nchannel = 69\npadding = 1871\n"
    )
    meta = parse_recording_txt(path)
    assert meta["padding"] == 1871
    assert meta["peak_channel"] == 69
    assert meta["juxta_channel"] == 254


def test_uint16_conversion_subtracts_mcs_offset(tmp_path: Path):
    n_file = 8
    extra = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    padding = 7
    n_samples = 5
    raw = tmp_path / "rec.raw"
    header = b"\xab" * padding
    block = np.full((n_samples, n_file), UINT16_OFFSET, dtype="<u2")
    block[:, 0] = UINT16_OFFSET + 10
    block[:, 1] = UINT16_OFFSET - 20
    raw.write_bytes(header + block.tobytes())
    out = tmp_path / "extra.bin"
    info = convert_mcs_uint16_to_int16(raw, padding, extra, out, n_file_channels=n_file)
    converted = np.memmap(out, dtype="<i2", mode="r", shape=(n_samples, extra.size))
    np.testing.assert_array_equal(converted[:, 0], 10)
    np.testing.assert_array_equal(converted[:, 1], -20)
    assert info["n_samples"] == n_samples
    assert info["n_channels"] == extra.size
    assert info["clipped"] is False


def test_k_nearest_keeps_peak_and_is_sorted():
    prb = DEFAULT_ARCHIVE_DIR / "mea_256.prb"
    if not prb.exists():
        import pytest

        pytest.skip(f"Yger probe file not mounted: {prb}")
    probe = parse_mea256_prb(prb)
    orig = probe.k_nearest_original(69, k=64)
    assert orig.shape == (64,)
    assert 69 in orig
    assert np.all(np.diff(orig) > 0)
    peak = probe.geom_all[69]
    kept = probe.geom_all[orig]
    rest_idx = [i for i, ch in enumerate(probe.extra_channels) if ch not in set(orig.tolist())]
    rest = probe.geom_extra[rest_idx]
    assert np.max(np.linalg.norm(kept - peak, axis=1)) <= np.min(np.linalg.norm(rest - peak, axis=1)) + 1e-9


def test_load_yger_ks4_memmaps_int16_bin(tmp_path: Path):
    from Spatial.data.loader import load_yger_ks4

    n_ch, n_samp = 4, 50
    raw = (np.arange(n_samp * n_ch, dtype=np.int16).reshape(n_samp, n_ch))
    bin_path = tmp_path / "extra_int16.bin"
    raw.tofile(bin_path)
    npz = tmp_path / "pseudo.npz"
    np.savez_compressed(
        npz,
        spike_times_0based=np.array([5, 20, 40], dtype=np.int64),
        spike_unit_ids=np.array([2, 2, 7], dtype=np.int64),
        geom=np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0], [30.0, 30.0]]),
        fs=np.int64(20000),
        n_samples=np.int64(n_samp),
        n_channels=np.int64(n_ch),
        original_file_channels=np.arange(n_ch, dtype=np.int64),
        unit_ids=np.array([2, 7], dtype=np.int64),
        unit_ks_label=np.array(["good", "mua"]),
        juxta_spike_times=np.array([5], dtype=np.int64),
        juxta_matched_cluster_id=np.int64(2),
        raw_bin_path=np.array(str(bin_path)),
        raw_layout=np.array("time_major_int16"),
        provenance_json=np.array("{}"),
    )
    ds = load_yger_ks4(path=npz, duration_s=0.002)
    assert ds.raw_data.shape == (4, 40)
    np.testing.assert_array_equal(ds.spike_times, [5, 20])
    np.testing.assert_array_equal(ds.spike_units, [2, 2])
    ds_good = load_yger_ks4(path=npz, good_only=True)
    np.testing.assert_array_equal(ds_good.spike_units, [2, 2])
    np.testing.assert_array_equal(ds.raw_data[:, 0], raw[0].astype(np.float64))


def test_coincidence_greedy_one_to_one():
    gt = np.array([100, 200, 300])
    pred = np.array([102, 198, 500])
    stats = coincidence_counts(gt, pred, tol_samples=5)
    assert stats["tp"] == 2
    assert stats["fp"] == 1
    assert stats["fn"] == 1
    assert stats["n_gt"] == 3
