"""Yger / Zenodo 1205233 in-vitro juxta-cellular MEA recordings.

Each tar.gz is one loose-patch ganglion cell plus 252 extracellular channels
on a 16x16 MEA (30 um). Ground truth is the patched cell only.

Default archive:
  /mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_ground_truth
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ARCHIVE_DIR = Path(
    "/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_ground_truth"
)
DEFAULT_WORK_DIR = Path(
    "/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work"
)

N_FILE_CHANNELS = 256
FS = 20000
CORNER_CHANNELS = (126, 127, 254, 255)
UINT16_OFFSET = 32768
CHUNK_SAMPLES = 50_000
# Garcia / SpikeInterface quality cull of author juxta traces (8 kept, 11 rejected).
GARCIA_KEPT_RECORDINGS = (
    "20160415_patch2",
    "20160426_patch2",
    "20160426_patch3",
    "20170621_patch1",
    "20170713_patch1",
    "20170725_patch1",
    "20170728_patch2",
    "20170803_patch1",
)


@dataclass(frozen=True)
class YgerProbe:
    extra_channels: np.ndarray  # original file channel ids, length 252
    geom_extra: np.ndarray  # (252, 2) um
    geom_all: np.ndarray  # (256, 2) um

    @property
    def n_extra(self) -> int:
        return int(self.extra_channels.shape[0])

    def ks_index(self, original_channel: int) -> int:
        hits = np.flatnonzero(self.extra_channels == int(original_channel))
        if hits.size != 1:
            raise KeyError(f"channel {original_channel} is not an extracellular site")
        return int(hits[0])

    def k_nearest_original(self, peak_channel: int, k: int = 64) -> np.ndarray:
        """Original-file channel ids of the k extra sites nearest the peak.

        SpikeForest PAIRED_MEA64C_YGER describes an 8x8 sub-array of the 16x16
        MEA. The published geom for 20160415_patch2 is exactly the 64 extra
        sites nearest the juxta STA peak, kept in original channel order.
        """
        if k < 1 or k > self.n_extra:
            raise ValueError(f"k={k} out of range for {self.n_extra} extra channels")
        peak_xy = self.geom_all[int(peak_channel)]
        dist = np.linalg.norm(self.geom_extra - peak_xy, axis=1)
        keep = np.argsort(dist, kind="stable")[: int(k)]
        orig = self.extra_channels[keep]
        return np.sort(orig)


@dataclass(frozen=True)
class YgerRecording:
    recording_id: str
    extracted_dir: Path
    raw_path: Path
    juxta_path: Path
    triggers_path: Path
    txt_path: Path
    padding: int
    juxta_channel: int
    peak_channel: int
    threshold: float
    min_peak: float
    n_samples: int
    spike_times: np.ndarray


def assert_on_hdd(path: Path, label: str = "path") -> Path:
    """Refuse to place recordings on the NVMe root filesystem."""
    path = Path(path).resolve()
    target = ""
    try:
        import subprocess

        proc = subprocess.run(
            ["findmnt", "-no", "TARGET", "-T", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        target = proc.stdout.strip()
    except OSError:
        target = ""
    if not str(path).startswith("/mnt/data/") and target not in ("/mnt/data",):
        raise RuntimeError(
            f"{label} {path} is not under /mnt/data (mount={target!r}). "
            "Recordings stay on the HDD; do not copy them to the SSD."
        )
    return path


def pin_tmpdir(work_dir: Path) -> Path:
    work_dir = assert_on_hdd(work_dir, "work_dir")
    tmp = work_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    return work_dir


def recording_id_from_archive(path: Path) -> str:
    name = Path(path).name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    return Path(path).stem


def list_archive_recordings(archive_dir: Path | None = None) -> list[str]:
    archive_dir = Path(archive_dir or DEFAULT_ARCHIVE_DIR)
    return [recording_id_from_archive(path) for path in sorted(archive_dir.glob("*.tar.gz"))]


def archive_path_for(recording_id: str, archive_dir: Path | None = None) -> Path:
    archive_dir = Path(archive_dir or DEFAULT_ARCHIVE_DIR)
    path = archive_dir / f"{recording_id}.tar.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing archive {path}")
    return path


def parse_mea256_prb(path: Path | None = None) -> YgerProbe:
    path = Path(path or (DEFAULT_ARCHIVE_DIR / "mea_256.prb"))
    geom_all = np.full((N_FILE_CHANNELS, 2), np.nan, dtype=np.float64)
    for match in re.finditer(
        r"(\d+)\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
        path.read_text(),
    ):
        idx = int(match.group(1))
        geom_all[idx, 0] = float(match.group(2))
        geom_all[idx, 1] = float(match.group(3))
    if np.isnan(geom_all).any():
        missing = np.flatnonzero(np.isnan(geom_all).any(axis=1))
        raise ValueError(f"incomplete probe geometry, missing channels {missing.tolist()}")
    extra = np.array(
        [i for i in range(N_FILE_CHANNELS) if i not in CORNER_CHANNELS],
        dtype=np.int64,
    )
    return YgerProbe(extra_channels=extra, geom_extra=geom_all[extra], geom_all=geom_all)


def parse_recording_txt(path: Path) -> dict:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return {
        "threshold": float(meta["threshold"]),
        "name": meta["name"],
        "juxta_channel": int(float(meta["juxta_channel"])),
        "min_peak": float(meta["min_peak"]),
        "peak_channel": int(float(meta["channel"])),
        "padding": int(float(meta["padding"])),
    }


def kilosort_probe_dict(probe: YgerProbe) -> dict:
    n = probe.n_extra
    return {
        "chanMap": np.arange(n, dtype=np.int32),
        "xc": probe.geom_extra[:, 0].astype(np.float64),
        "yc": probe.geom_extra[:, 1].astype(np.float64),
        "kcoords": np.zeros(n, dtype=np.float64),
        "n_chan": n,
    }


def extract_recording(
    recording_id: str,
    archive_dir: Path | None = None,
    work_dir: Path | None = None,
) -> Path:
    archive = archive_path_for(recording_id, archive_dir)
    dest = assert_on_hdd(Path(work_dir or DEFAULT_WORK_DIR), "work_dir") / "extracted" / recording_id
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extract_ok"
    if marker.exists() and any(dest.glob("*.raw")) and any(dest.glob("*.txt")):
        return dest
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    marker.write_text("ok\n")
    return dest


def _require_one(extracted: Path, suffix: str) -> Path:
    hits = sorted(extracted.glob(f"*{suffix}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *{suffix} in {extracted}, found {hits}")
    return hits[0]


def load_extracted_recording(extracted_dir: Path) -> YgerRecording:
    extracted_dir = Path(extracted_dir)
    txt_path = _require_one(extracted_dir, ".txt")
    # juxta is named *.juxta.raw and would also match *.raw
    raw_candidates = [p for p in extracted_dir.glob("*.raw") if "juxta" not in p.name]
    juxta_candidates = list(extracted_dir.glob("*.juxta.raw"))
    if len(raw_candidates) != 1 or len(juxta_candidates) != 1:
        raise FileNotFoundError(
            f"raw/juxta split failed in {extracted_dir}: raw={raw_candidates} juxta={juxta_candidates}"
        )
    raw_path = raw_candidates[0]
    juxta_path = juxta_candidates[0]
    triggers_path = _require_one(extracted_dir, ".triggers.npy")
    meta = parse_recording_txt(txt_path)
    n_samples = (raw_path.stat().st_size - meta["padding"]) // (N_FILE_CHANNELS * 2)
    if n_samples <= 0:
        raise ValueError(f"non-positive n_samples for {raw_path}")
    spike_times = np.load(triggers_path).astype(np.int64, copy=False)
    return YgerRecording(
        recording_id=extracted_dir.name,
        extracted_dir=extracted_dir,
        raw_path=raw_path,
        juxta_path=juxta_path,
        triggers_path=triggers_path,
        txt_path=txt_path,
        padding=int(meta["padding"]),
        juxta_channel=int(meta["juxta_channel"]),
        peak_channel=int(meta["peak_channel"]),
        threshold=float(meta["threshold"]),
        min_peak=float(meta["min_peak"]),
        n_samples=int(n_samples),
        spike_times=spike_times,
    )


def convert_mcs_uint16_to_int16(
    raw_path: Path,
    padding: int,
    extra_channels: np.ndarray,
    out_bin: Path,
    uint16_offset: int = UINT16_OFFSET,
    n_file_channels: int = N_FILE_CHANNELS,
) -> dict:
    extra = np.asarray(extra_channels, dtype=np.int64)
    file_size = Path(raw_path).stat().st_size
    payload = file_size - int(padding)
    sample_bytes = n_file_channels * 2
    if payload % sample_bytes != 0:
        raise ValueError(
            f"{raw_path} payload {payload} is not divisible by {sample_bytes} after padding={padding}"
        )
    n_samples = payload // sample_bytes
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    src = np.memmap(
        raw_path, dtype="<u2", mode="r", offset=int(padding), shape=(n_samples, n_file_channels)
    )
    dst = np.memmap(
        out_bin, dtype="<i2", mode="w+", shape=(n_samples, extra.shape[0])
    )
    abs_max = 0
    sum_abs = 0.0
    n_seen = 0
    for start in range(0, n_samples, CHUNK_SAMPLES):
        stop = min(start + CHUNK_SAMPLES, n_samples)
        block = src[start:stop][:, extra].astype(np.int32) - int(uint16_offset)
        abs_max = max(abs_max, int(np.max(np.abs(block))))
        sum_abs += float(np.mean(np.abs(block))) * (stop - start)
        n_seen += stop - start
        dst[start:stop] = np.clip(block, -32768, 32767).astype(np.int16)
    dst.flush()
    del dst, src
    return {
        "n_samples": int(n_samples),
        "n_channels": int(extra.shape[0]),
        "abs_max": int(abs_max),
        "mean_abs": float(sum_abs / max(n_seen, 1)),
        "uint16_offset": int(uint16_offset),
        "clipped": bool(abs_max > 32767),
    }


def write_probe_json(probe: YgerProbe, path: Path) -> None:
    payload = kilosort_probe_dict(probe)
    serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else int(value) if key == "n_chan" else value)
        for key, value in payload.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def prepare_recording(
    recording_id: str,
    archive_dir: Path | None = None,
    work_dir: Path | None = None,
    force: bool = False,
) -> dict:
    work_dir = pin_tmpdir(Path(work_dir or DEFAULT_WORK_DIR))
    probe = parse_mea256_prb(Path(archive_dir or DEFAULT_ARCHIVE_DIR) / "mea_256.prb")
    extracted = extract_recording(recording_id, archive_dir=archive_dir, work_dir=work_dir)
    rec = load_extracted_recording(extracted)
    prepared = work_dir / "ks_bin" / recording_id
    prepared.mkdir(parents=True, exist_ok=True)
    bin_path = prepared / "extra_int16.bin"
    probe_path = prepared / "probe.json"
    gt_path = prepared / "juxta_gt.npz"
    meta_path = prepared / "meta.json"
    if force or not bin_path.exists():
        conv = convert_mcs_uint16_to_int16(
            rec.raw_path, rec.padding, probe.extra_channels, bin_path
        )
    else:
        n_samples = rec.n_samples
        conv = {
            "n_samples": n_samples,
            "n_channels": probe.n_extra,
            "uint16_offset": UINT16_OFFSET,
            "skipped_existing": True,
        }
    write_probe_json(probe, probe_path)
    np.savez_compressed(
        gt_path,
        spike_times=rec.spike_times,
        spike_units=np.zeros(rec.spike_times.shape[0], dtype=np.int64),
        peak_channel_original=np.int64(rec.peak_channel),
        peak_channel_ks=np.int64(probe.ks_index(rec.peak_channel)),
        juxta_channel=np.int64(rec.juxta_channel),
        extra_channels=probe.extra_channels,
        geom_extra=probe.geom_extra,
        fs=np.int64(FS),
    )
    meta = {
        "recording_id": recording_id,
        "fs": FS,
        "n_file_channels": N_FILE_CHANNELS,
        "n_extra_channels": probe.n_extra,
        "n_samples": rec.n_samples,
        "duration_s": rec.n_samples / FS,
        "padding": rec.padding,
        "peak_channel_original": rec.peak_channel,
        "peak_channel_ks": probe.ks_index(rec.peak_channel),
        "juxta_channel": rec.juxta_channel,
        "n_juxta_spikes": int(rec.spike_times.size),
        "threshold": rec.threshold,
        "min_peak": rec.min_peak,
        "bin_path": str(bin_path),
        "probe_path": str(probe_path),
        "gt_path": str(gt_path),
        "extracted_dir": str(extracted),
        "conversion": conv,
        "gt_semantics": "single patched ganglion cell; not a full multi-unit label set",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def write_k_nearest_subset(
    recording_id: str,
    k: int = 64,
    archive_dir: Path | None = None,
    work_dir: Path | None = None,
) -> dict:
    """Slice an already-converted 252-ch int16 binary down to k nearest sites.

    Writes next to the 252-ch files on the HDD work dir. Does not copy the
    parent recording to the SSD.
    """
    work_dir = pin_tmpdir(Path(work_dir or DEFAULT_WORK_DIR))
    prepared = work_dir / "ks_bin" / recording_id
    src_bin = prepared / "extra_int16.bin"
    meta_path = prepared / "meta.json"
    if not src_bin.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"252-ch conversion missing under {prepared}; run prepare_recording first"
        )
    meta = json.loads(meta_path.read_text())
    probe = parse_mea256_prb(Path(archive_dir or DEFAULT_ARCHIVE_DIR) / "mea_256.prb")
    peak = int(meta["peak_channel_original"])
    orig = probe.k_nearest_original(peak, k=k)
    src_idx = np.array([probe.ks_index(int(ch)) for ch in orig], dtype=np.int64)
    n_samples = int(meta["n_samples"])
    src = np.memmap(src_bin, dtype="<i2", mode="r", shape=(n_samples, probe.n_extra))
    out_bin = prepared / f"extra_int16_k{k}.bin"
    dst = np.memmap(out_bin, dtype="<i2", mode="w+", shape=(n_samples, orig.size))
    for start in range(0, n_samples, CHUNK_SAMPLES):
        stop = min(start + CHUNK_SAMPLES, n_samples)
        dst[start:stop] = src[start:stop][:, src_idx]
    dst.flush()
    del dst, src
    subset_probe = {
        "chanMap": np.arange(orig.size, dtype=np.int32),
        "xc": probe.geom_all[orig, 0].astype(np.float64),
        "yc": probe.geom_all[orig, 1].astype(np.float64),
        "kcoords": np.zeros(orig.size, dtype=np.float64),
        "n_chan": int(orig.size),
    }
    probe_path = prepared / f"probe_k{k}.json"
    probe_path.write_text(
        json.dumps(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in subset_probe.items()
            },
            indent=2,
        )
    )
    peak_xy = probe.geom_all[peak]
    radius = float(np.max(np.linalg.norm(probe.geom_all[orig] - peak_xy, axis=1)))
    subset = {
        "recording_id": recording_id,
        "k": int(k),
        "rule": "k extra sites nearest the juxta STA peak, original channel order",
        "spikeforest_match": "PAIRED_MEA64C_YGER 8x8 language; published geom is 64-NN",
        "peak_channel_original": peak,
        "original_channels": orig.tolist(),
        "src_indices_in_252": src_idx.tolist(),
        "radius_um": radius,
        "bin_path": str(out_bin),
        "probe_path": str(probe_path),
        "n_samples": n_samples,
        "n_channels": int(orig.size),
    }
    (prepared / f"subset_k{k}.json").write_text(json.dumps(subset, indent=2))
    return subset


def coincidence_counts(
    gt_times: np.ndarray,
    pred_times: np.ndarray,
    tol_samples: int,
) -> dict:
    """Greedy one-to-one match within ±tol samples."""
    gt = np.sort(np.asarray(gt_times, dtype=np.int64))
    pred = np.sort(np.asarray(pred_times, dtype=np.int64))
    i = 0
    j = 0
    tp = 0
    while i < gt.size and j < pred.size:
        delta = int(pred[j]) - int(gt[i])
        if delta < -tol_samples:
            j += 1
        elif delta > tol_samples:
            i += 1
        else:
            tp += 1
            i += 1
            j += 1
    fn = int(gt.size - tp)
    fp = int(pred.size - tp)
    denom = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": int(gt.size),
        "n_pred": int(pred.size),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "accuracy": float(tp / denom) if denom else 0.0,
    }


DEFAULT_KS4_TH13_DIR = DEFAULT_WORK_DIR / "ks_out" / "20160415_patch2_th13"
DEFAULT_SPATIAL_NPZ = (
    DEFAULT_WORK_DIR / "spatial_npz" / "yger_20160415_patch2_ks4_th13.npz"
)


def _read_cluster_kslabel(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        labels[int(parts[0])] = parts[1]
    return labels


def export_ks4_spatial_npz(
    ks_dir: Path | None = None,
    bin_path: Path | None = None,
    probe_json: Path | None = None,
    juxta_npz: Path | None = None,
    out_path: Path | None = None,
    archive_dir: Path | None = None,
) -> dict:
    """Pack a KS4 sorting into a compact Spatial NPZ (labels+geom+pointers, no raw copy)."""
    ks_dir = Path(ks_dir or DEFAULT_KS4_TH13_DIR)
    rec_id = "20160415_patch2"
    prepared = DEFAULT_WORK_DIR / "ks_bin" / rec_id
    bin_path = Path(bin_path or prepared / "extra_int16.bin")
    probe_json = Path(probe_json or prepared / "probe.json")
    juxta_npz = Path(juxta_npz or prepared / "juxta_gt.npz")
    out_path = Path(out_path or DEFAULT_SPATIAL_NPZ)
    assert_on_hdd(out_path.parent, "spatial_npz_dir")
    assert_on_hdd(bin_path, "raw_bin")
    assert_on_hdd(ks_dir, "ks_dir")

    times = np.load(ks_dir / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    units = np.load(ks_dir / "spike_clusters.npy").astype(np.int64, copy=False).reshape(-1)
    if times.shape != units.shape:
        raise ValueError(f"spike_times {times.shape} vs spike_clusters {units.shape}")
    probe = json.loads(probe_json.read_text())
    geom = np.column_stack([probe["xc"], probe["yc"]]).astype(np.float64)
    n_channels = int(probe["n_chan"])
    n_samples = int(bin_path.stat().st_size // (n_channels * 2))
    mea = parse_mea256_prb(Path(archive_dir or DEFAULT_ARCHIVE_DIR) / "mea_256.prb")
    kslabel = _read_cluster_kslabel(ks_dir / "cluster_KSLabel.tsv")
    unique_units = np.unique(units)
    unit_ks_label = np.array([kslabel.get(int(u), "unsorted") for u in unique_units])
    score = {}
    score_path = ks_dir / "juxta_score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
    juxta_times = np.array([], dtype=np.int64)
    juxta_cluster = -1
    if juxta_npz.exists():
        juxta_times = np.load(juxta_npz)["spike_times"].astype(np.int64, copy=False).reshape(-1)
    if score.get("score", {}).get("best"):
        juxta_cluster = int(score["score"]["best"]["cluster_id"])

    provenance = {
        "kind": "kilosort4_pseudo_gt",
        "recording_id": rec_id,
        "dataset": "zenodo_1205233 / 20160415_patch2",
        "not_juxta_gt": True,
        "kilosort_version": "4.1.7",
        "env": "snn_cuda",
        "device": "cuda:0",
        "settings": {
            "fs": FS,
            "n_chan_bin": n_channels,
            "nblocks": 0,
            "nt": 61,
            "Th_learned": 13.0,
            "Th_universal": 13.0,
            "data_dtype": "int16",
        },
        "command": (
            "CUDA_VISIBLE_DEVICES=0 python -m Spatial.experiments.run_kilosort_yger "
            "--recording 20160415_patch2 --device cuda:0 --nblocks 0 --tol-ms 0.4 "
            "--tag th13 --th-learned 13 --th-universal 13"
        ),
        "ks_dir": str(ks_dir),
        "raw_bin_path": str(bin_path),
        "uint16_offset": UINT16_OFFSET,
        "raw_already_highpass_hz": 100,
        "juxta_matched_cluster_id": juxta_cluster,
        "juxta_score": score.get("score", {}).get("best"),
        "n_ks_clusters": int(unique_units.size),
        "n_ks_spikes": int(times.size),
        "n_good_units": int(np.sum(unit_ks_label == "good")),
        "doc": "Spatial/docs/yger_ks4_pseudo_gt.md",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spike_times_0based=times,
        spike_unit_ids=units,
        geom=geom,
        fs=np.int64(FS),
        n_samples=np.int64(n_samples),
        n_channels=np.int64(n_channels),
        original_file_channels=mea.extra_channels.astype(np.int64),
        unit_ids=unique_units.astype(np.int64),
        unit_ks_label=unit_ks_label,
        juxta_spike_times=juxta_times,
        juxta_matched_cluster_id=np.int64(juxta_cluster),
        raw_bin_path=np.array(str(bin_path)),
        raw_layout=np.array("time_major_int16"),
        provenance_json=np.array(json.dumps(provenance)),
    )
    sidecar = out_path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2))
    return {
        "npz_path": str(out_path),
        "provenance_path": str(sidecar),
        "n_spikes": int(times.size),
        "n_units": int(unique_units.size),
        "n_good_units": int(np.sum(unit_ks_label == "good")),
        "n_samples": n_samples,
        "n_channels": n_channels,
        "juxta_matched_cluster_id": juxta_cluster,
        "raw_bin_path": str(bin_path),
    }
