"""Unified dataset loader for Spatial clustering experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

HJ_NPZ_PATH = REPO_ROOT / "new_datasets" / "hybridjanelia" / "static16c_600s_11_filtered_gt.npz"
HJ_DIR = REPO_ROOT / "new_datasets" / "hybridjanelia"
HJ_CATALOG_PATH = HJ_DIR / "catalog.json"
CORTEXLAB_NPZ_PATH = (
    REPO_ROOT
    / "output"
    / "cortexlab_10pct"
    / "slices"
    / "dataset_1_20141202_228s_ksgt_int16.npz"
)
YGER_KS4_NPZ_PATH = Path(
    "/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work/"
    "spatial_npz/yger_20160415_patch2_ks4_th13.npz"
)


@dataclass
class Dataset:
    raw_data: np.ndarray  # (n_channels, n_samples) float64
    geom: np.ndarray  # (n_channels, 2) — (x, y) in um
    fs: int
    spike_times: np.ndarray  # GT spike sample indices (0-based)
    spike_units: np.ndarray  # GT unit label per spike
    name: str

    @property
    def n_channels(self) -> int:
        return self.raw_data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.raw_data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def unit_ids(self) -> np.ndarray:
        return np.unique(self.spike_units)


def resolve_hybrid_janelia_path(scene: str | None = None, path: Path | None = None) -> Path:
    """Resolve HJ NPZ by explicit path, scene_key, short_name, or default."""
    if path is not None:
        return Path(path)
    if scene is None:
        return HJ_NPZ_PATH

    # Direct filename / stem
    cand = HJ_DIR / scene
    if cand.suffix == ".npz" and cand.exists():
        return cand
    cand2 = HJ_DIR / f"{scene}_filtered_gt.npz"
    if cand2.exists():
        return cand2
    cand3 = HJ_DIR / f"{scene}.npz"
    if cand3.exists():
        return cand3

    # Catalog lookup
    if HJ_CATALOG_PATH.exists():
        import json

        with open(HJ_CATALOG_PATH) as f:
            catalog = json.load(f)
        for e in catalog.get("scenes", []):
            if scene in (e.get("scene_key"), e.get("short_name")) and e.get("npz_path"):
                p = Path(e["npz_path"])
                if p.exists():
                    return p

    raise FileNotFoundError(
        f"Hybrid Janelia scene '{scene}' not found under {HJ_DIR}. "
        f"Run: python -m Spatial.data.prepare_hybrid_janelia --only ..."
    )


def list_hybrid_janelia_scenes(ready_only: bool = True) -> list[dict]:
    """Return catalog entries (requires prepare_hybrid_janelia --list at least once)."""
    import json

    if not HJ_CATALOG_PATH.exists():
        return []
    with open(HJ_CATALOG_PATH) as f:
        catalog = json.load(f)
    scenes = catalog.get("scenes", [])
    if ready_only:
        scenes = [e for e in scenes if e.get("npz_ready")]
    return scenes


def load_hybrid_janelia(
    path: Path = None,
    duration_s: float = None,
    scene: str = None,
) -> Dataset:
    """Load a Hybrid Janelia filtered_gt NPZ.

    Args:
        path: Explicit NPZ path (overrides scene).
        duration_s: If set, truncate to this many seconds from the start.
        scene: Scene key or short name, e.g. ``static_siprobe_16c_600s_12``
            or ``static16c_600s_12``. Default: static16c_600s_11.
    """
    path = resolve_hybrid_janelia_path(scene=scene, path=path)
    f = np.load(str(path), allow_pickle=True)

    raw = f["raw_data"].astype(np.float64)
    geom = f["geom"].astype(np.float64)
    spike_times = f["spike_times_0based"].astype(np.int64)
    spike_units = f["spike_unit_ids"].astype(np.int64)
    fs = 30000

    if duration_s is not None:
        n_keep = int(duration_s * fs)
        raw = raw[:, :n_keep]
        mask = spike_times < n_keep
        spike_times = spike_times[mask]
        spike_units = spike_units[mask]

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=spike_times,
        spike_units=spike_units,
        name=f"hybrid_janelia_{path.stem.replace('_filtered_gt', '')}",
    )


def load_mearec_h5(path: Path, duration_s: float = None) -> Dataset:
    """Load a MEArec-generated .h5 recording.

    Expects standard MEArec HDF5 layout:
      /recordings  — (n_samples, n_channels) float32
      /spiketrains — group with per-unit spike time arrays (in seconds)
      /channel_positions — (n_channels, 3) or (n_channels, 2)
    """
    import h5py

    path = Path(path)
    with h5py.File(str(path), "r") as f:
        recordings = f["recordings"][:]  # (n_samples, n_channels)
        fs = int(f["recordings"].attrs.get("fs", 30000))

        if "channel_positions" in f:
            pos = f["channel_positions"][:]
            geom = pos[:, :2].astype(np.float64)
        elif "info" in f and "electrode_positions" in f["info"]:
            geom = f["info"]["electrode_positions"][:].astype(np.float64)
        else:
            n_ch = recordings.shape[1]
            geom = np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 25.0])

        # Extract spike trains
        st_group = f["spiketrains"]
        all_times = []
        all_units = []
        for unit_name in sorted(st_group.keys()):
            unit_id = int(unit_name.split("_")[-1]) if "_" in unit_name else hash(unit_name) % 10000
            times_s = st_group[unit_name][:]
            samples = (times_s * fs).astype(np.int64)
            all_times.append(samples)
            all_units.append(np.full(len(samples), unit_id, dtype=np.int64))

        spike_times = np.concatenate(all_times) if all_times else np.array([], dtype=np.int64)
        spike_units = np.concatenate(all_units) if all_units else np.array([], dtype=np.int64)

    raw = recordings.T.astype(np.float64)  # -> (n_channels, n_samples)

    if duration_s is not None:
        n_keep = int(duration_s * fs)
        raw = raw[:, :n_keep]
        mask = spike_times < n_keep
        spike_times = spike_times[mask]
        spike_units = spike_units[mask]

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=spike_times,
        spike_units=spike_units,
        name=f"mearec_{path.stem}",
    )


def load_cortexlab_npz(path: Path = None, duration_s: float = None) -> Dataset:
    """Load CortexLab Neuropixels 10% slice NPZ (128ch linear).

    Args:
        path: Override default NPZ path.
        duration_s: If set, truncate to this many seconds from the start.
    """
    path = Path(path) if path else CORTEXLAB_NPZ_PATH
    f = np.load(str(path), allow_pickle=True)

    raw = f["raw_data"].astype(np.float64)
    geom = f["geom"].astype(np.float64)
    spike_times = f["spike_times_0based"].astype(np.int64)
    spike_units = f["spike_unit_ids"].astype(np.int64)
    fs = int(f["fs"]) if "fs" in f else 30000

    if duration_s is not None:
        n_keep = int(duration_s * fs)
        raw = raw[:, :n_keep]
        mask = spike_times < n_keep
        spike_times = spike_times[mask]
        spike_units = spike_units[mask]

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=spike_times,
        spike_units=spike_units,
        name="cortexlab_np_128ch",
    )


def load_yger_ks4(
    path: Path | None = None,
    duration_s: float = None,
    good_only: bool = False,
) -> Dataset:
    """Load Kilosort4 Th=13 pseudo-GT for Yger 20160415_patch2.

    Labels are KS4 clusters, not juxta ground truth. Raw stays on the HDD as
    the existing int16 binary and is memmapped; nothing is copied to the SSD.
    See ``Spatial/docs/yger_ks4_pseudo_gt.md``.
    """
    path = Path(path) if path is not None else YGER_KS4_NPZ_PATH
    f = np.load(str(path), allow_pickle=True)
    spike_times = f["spike_times_0based"].astype(np.int64)
    spike_units = f["spike_unit_ids"].astype(np.int64)
    geom = f["geom"].astype(np.float64)
    fs = int(f["fs"])
    n_channels = int(f["n_channels"])
    n_samples = int(f["n_samples"])
    bin_path = Path(str(f["raw_bin_path"]))
    if not bin_path.exists():
        raise FileNotFoundError(
            f"Yger KS4 raw binary missing: {bin_path}. "
            "It must remain on /mnt/data; Spatial does not copy it to the SSD."
        )

    if good_only:
        unit_ids = f["unit_ids"].astype(np.int64)
        labels = np.asarray(f["unit_ks_label"]).astype(str)
        good = set(int(u) for u, lab in zip(unit_ids, labels) if lab == "good")
        keep = np.array([int(u) in good for u in spike_units])
        spike_times = spike_times[keep]
        spike_units = spike_units[keep]

    if duration_s is not None:
        n_keep = int(duration_s * fs)
        n_samples = min(n_samples, n_keep)
        mask = spike_times < n_samples
        spike_times = spike_times[mask]
        spike_units = spike_units[mask]

    raw_mm = np.memmap(
        bin_path, dtype="<i2", mode="r", shape=(int(f["n_samples"]), n_channels)
    )
    raw = np.asarray(raw_mm[:n_samples].T, dtype=np.float64)
    del raw_mm

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=spike_times,
        spike_units=spike_units,
        name="yger_20160415_patch2_ks4_th13",
    )


def load_mearec_npz(path: Path, duration_s: float = None) -> Dataset:
    """Load a GeoOsort-format NPZ export of MEArec data."""
    path = Path(path)
    f = np.load(str(path), allow_pickle=True)

    raw = f["raw_data"].astype(np.float64)
    if raw.ndim == 2 and raw.shape[0] > raw.shape[1]:
        raw = raw.T  # ensure (channels, samples)

    geom = f["geom"].astype(np.float64) if "geom" in f else None
    if geom is None:
        n_ch = raw.shape[0]
        geom = np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 25.0])

    spike_times = f["spike_times_0based"].astype(np.int64)
    spike_units = f["spike_unit_ids"].astype(np.int64)
    # Some exports store unit ids as "#k"
    if spike_units.dtype.kind in ("U", "S", "O"):
        spike_units = np.array(
            [int(str(u).lstrip("#")) for u in spike_units], dtype=np.int64
        )
    fs = int(float(np.asarray(f["fs"]).ravel()[0])) if "fs" in f else 30000

    if duration_s is not None:
        n_keep = int(duration_s * fs)
        raw = raw[:, :n_keep]
        mask = spike_times < n_keep
        spike_times = spike_times[mask]
        spike_units = spike_units[mask]

    return Dataset(
        raw_data=raw,
        geom=geom,
        fs=fs,
        spike_times=spike_times,
        spike_units=spike_units,
        name=f"mearec_{path.stem}",
    )
