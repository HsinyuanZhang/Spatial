"""Kilosort4-paper Figshare simulations (sim_no_drift and siblings).

Recordings stay on the HDD as the downloaded zip / cbin. Spatial memmaps or
streams a time slice; nothing is copied to the SSD.
"""

from __future__ import annotations

import json
import re
import zipfile
import zlib
from pathlib import Path

import numpy as np

from Spatial.data.loader import Dataset

FIGSHARE_DIR = Path(
    "/mnt/data/backup_datasets/SNN_SpikeSorting/figshare_25298815_kilosort4_sims"
)
NO_DRIFT_ZIP = FIGSHARE_DIR / "sim_no_drift.zip"
NO_DRIFT_META_DIR = FIGSHARE_DIR / "extracted_no_drift"
FS = 30000
N_AP = 384


def parse_shank_geom(meta_text: str, n_ap: int = N_AP) -> np.ndarray:
    """(n_ap, 2) µm from SpikeGLX ``~snsShankMap`` (col * 32, row * 20)."""
    line = None
    for raw in meta_text.splitlines():
        if raw.startswith("~snsShankMap="):
            line = raw.split("=", 1)[1]
            break
    if line is None:
        raise ValueError("meta has no ~snsShankMap")
    sites = re.findall(r"\((\d+):(\d+):(\d+):(\d+)\)", line)
    if len(sites) < n_ap:
        raise ValueError(f"shank map has {len(sites)} sites, need {n_ap}")
    geom = np.zeros((n_ap, 2), dtype=np.float64)
    for i, (_shank, col, row, used) in enumerate(sites[:n_ap]):
        if int(used) == 0:
            continue
        geom[i, 0] = float(int(col) * 32.0)
        geom[i, 1] = float(int(row) * 20.0)
    return geom


def _read_cbin_slice_from_zip(
    zip_path: Path,
    ch_path: Path,
    n_samples: int,
    inner_name: str = "sim.imec0.ap.cbin",
) -> np.ndarray:
    """Return int16 (n_channels, n_samples) AP+SY for the first ``n_samples``."""
    header = json.loads(Path(ch_path).read_text())
    n_ch = int(header["n_channels"])
    bounds = np.asarray(header["chunk_bounds"], dtype=np.int64)
    offsets = np.asarray(header["chunk_offsets"], dtype=np.int64)
    if header.get("algorithm") != "zlib":
        raise ValueError(f"unsupported cbin algorithm {header.get('algorithm')}")
    if not header.get("do_time_diff", False):
        raise ValueError("loader expects do_time_diff=true")

    last = int(np.searchsorted(bounds, n_samples, side="left"))
    if last <= 0:
        raise ValueError("n_samples is empty")
    if last >= bounds.size:
        last = int(bounds.size - 1)

    out = np.zeros((n_ch, n_samples), dtype=np.int16)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(inner_name) as handle:
            cursor = 0
            for i in range(last):
                t0 = int(bounds[i])
                t1 = int(min(bounds[i + 1], n_samples))
                if t1 <= t0:
                    break
                start = int(offsets[i])
                stop = int(offsets[i + 1])
                if start < cursor:
                    raise ValueError("cbin chunk offsets are not prefix-sorted")
                if start > cursor:
                    handle.read(start - cursor)
                blob = handle.read(stop - start)
                cursor = stop
                raw = np.frombuffer(zlib.decompress(blob), dtype=np.int16)
                n_samp_chunk = int(bounds[i + 1] - bounds[i])
                chunk = raw.reshape((n_samp_chunk, n_ch), order="F").astype(np.int32)
                chunk = np.cumsum(chunk, axis=0)
                sl = chunk[: t1 - t0]
                out[:, t0:t1] = np.clip(sl, -32768, 32767).T.astype(np.int16)
    return out


def load_ks4_sim_no_drift(
    duration_s: float = 60.0,
    zip_path: Path | None = None,
    meta_dir: Path | None = None,
) -> Dataset:
    """Load a prefix of the Figshare ``sim_no_drift`` Neuropixels simulation.

    Ground-truth labels are the simulator cluster ids (1200 units), not a
    Kilosort sorting. The 40 GB cbin stays inside the zip on /mnt/data.
    """
    zip_path = Path(zip_path or NO_DRIFT_ZIP)
    meta_dir = Path(meta_dir or NO_DRIFT_META_DIR)
    if not zip_path.exists():
        raise FileNotFoundError(f"sim_no_drift zip missing: {zip_path}")
    meta_path = meta_dir / "sim.imec0.ap.meta"
    ch_path = meta_dir / "sim.imec0.ap.ch"
    params_path = meta_dir / "sim.imec0.ap_params.npz"
    for path in (meta_path, ch_path, params_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Extract the small files from the zip on the HDD."
            )

    n_keep = int(float(duration_s) * FS)
    raw_i16 = _read_cbin_slice_from_zip(zip_path, ch_path, n_keep)
    raw = np.asarray(raw_i16[:N_AP], dtype=np.float64)
    del raw_i16
    geom = parse_shank_geom(meta_path.read_text(), n_ap=N_AP)

    params = np.load(params_path)
    times = np.asarray(params["st"], dtype=np.int64).ravel()
    units = np.asarray(params["cl"], dtype=np.int64).ravel()
    mask = (times >= 0) & (times < n_keep)
    times = times[mask]
    units = units[mask]

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=FS,
        spike_times=times,
        spike_units=units,
        name="ks4sim_no_drift",
    )
