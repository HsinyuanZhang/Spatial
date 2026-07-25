"""Convert Hybrid Janelia MDA (raw+firings+geom) → filtered_gt NPZ for experiments.

MDA header (MountainLab / SpikeForest style):
  int32 dtype_code (negative)
  int32 bytes_per_entry
  int32 num_dims
  int32 dim_0 .. dim_{n-1}
  <payload>

NPZ schema matches ``static16c_600s_11_filtered_gt.npz``.

Usage:
  python -m Spatial.data.prepare_hybrid_janelia --list
  python -m Spatial.data.prepare_hybrid_janelia --only static_siprobe_16c_600s
  python -m Spatial.data.prepare_hybrid_janelia --scene-key static_siprobe_16c_600s_11
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
HJ_ROOT = REPO_ROOT / "Hybridjanelia"
RAW_DIR = HJ_ROOT / "downloads_hybrid_raw"
FIR_DIR = HJ_ROOT / "downloads"
GEOM_DIR = HJ_ROOT / "downloads_hybrid_geom"
OUT_DIR = REPO_ROOT / "new_datasets" / "hybridjanelia"
CATALOG_PATH = OUT_DIR / "catalog.json"

DTYPE_MAP = {
    -1: np.dtype("<f4"),
    -2: np.dtype("<u1"),
    -3: np.dtype("<f8"),
    -4: np.dtype("<i2"),
    -5: np.dtype("<i4"),
    -6: np.dtype("<u2"),
    -7: np.dtype("<f8"),
}


@dataclass(frozen=True)
class Scene:
    prefix: str  # full HYBRID_JANELIA_... without _raw/_firings/_geom suffix
    drift: str  # static|drift
    probe: str  # siprobe|tetrode
    n_channels: int
    duration_s: int
    rep: str

    @property
    def key(self) -> str:
        return f"{self.drift}_{self.probe}_{self.n_channels}c_{self.duration_s}s_{self.rep}"

    @property
    def short_name(self) -> str:
        return f"{self.drift}{self.n_channels}c_{self.duration_s}s_{self.rep}"


def read_mda(path: Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        code = int(np.fromfile(f, dtype="<i4", count=1)[0])
        bpe = int(np.fromfile(f, dtype="<i4", count=1)[0])
        nd = int(np.fromfile(f, dtype="<i4", count=1)[0])
        if nd < 1 or nd > 10:
            raise ValueError(f"Bad num_dims={nd} in {path}")
        dims = [int(x) for x in np.fromfile(f, dtype="<i4", count=nd)]
        dtype = DTYPE_MAP.get(code)
        if dtype is None:
            raise ValueError(f"Unsupported MDA dtype code {code} in {path}")
        if dtype.itemsize != bpe:
            # still trust dtype code; warn via exception only if product inconsistent
            pass
        data = np.fromfile(f, dtype=dtype)
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(
            f"MDA size mismatch in {path.name}: got {data.size}, expected {expected} from dims {dims}"
        )
    return data.reshape(dims, order="F")  # Fortran order like MATLAB/MDA


def read_geom_csv(path: Path) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append([float(row[0]), float(row[1])])
    return np.asarray(rows, dtype=np.float64)


def parse_scene(prefix: str) -> Scene | None:
    m = re.search(
        r"hybrid_(static|drift)_(siprobe|tetrode)_rec_(\d+)c_(\d+)s_(\d+)$",
        prefix,
    )
    if not m:
        return None
    return Scene(
        prefix=prefix,
        drift=m.group(1),
        probe=m.group(2),
        n_channels=int(m.group(3)),
        duration_s=int(m.group(4)),
        rep=m.group(5),
    )


def discover_scenes() -> list[Scene]:
    raws = {
        p.name[: -len("_raw.mda")]
        for p in RAW_DIR.glob("HYBRID_JANELIA_*_raw.mda")
        if not p.name.startswith("._")
    }
    firs = {
        p.name[: -len("_firings_true.mda")]
        for p in FIR_DIR.glob("HYBRID_JANELIA_*_firings_true.mda")
        if not p.name.startswith("._")
    }
    geoms = {
        p.name[: -len("_geom.csv")]
        for p in GEOM_DIR.glob("HYBRID_JANELIA_*_geom.csv")
        if not p.name.startswith("._")
    }
    scenes = []
    for prefix in sorted(raws & firs & geoms):
        sc = parse_scene(prefix)
        if sc is not None:
            scenes.append(sc)
    return scenes


def bandpass(x: np.ndarray, fs: float, low: float = 300.0, high: float = 6000.0, order: int = 3):
    """x: (n_channels, n_samples)"""
    b, a = butter(order, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
    # filtfilt along time
    return filtfilt(b, a, x.astype(np.float64), axis=1)


def estimate_noise_mad(filtered: np.ndarray) -> np.ndarray:
    """Per-channel noise = MAD / 0.6745. Shape (n_channels,)"""
    med = np.median(filtered, axis=1, keepdims=True)
    mad = np.median(np.abs(filtered - med), axis=1)
    return (mad / 0.6745).astype(np.float64)


def unit_snr_and_main_channel(
    filtered: np.ndarray,
    times: np.ndarray,
    noise: np.ndarray,
    max_spikes: int = 200,
    window: int = 30,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """SNR = Vpeak / noise(best_ch); main_ch = argmax |template|."""
    rng = rng or np.random.default_rng(0)
    n_ch, n_samp = filtered.shape
    times = times[(times >= window) & (times < n_samp - window)]
    if times.size == 0:
        return 0.0, 0
    if times.size > max_spikes:
        times = rng.choice(times, size=max_spikes, replace=False)
    snippets = np.stack(
        [filtered[:, t - window : t + window] for t in times], axis=0
    )  # (n, ch, w)
    template = np.mean(snippets, axis=0)  # (ch, w)
    peak_per_ch = np.max(np.abs(template), axis=1)
    main_ch = int(np.argmax(peak_per_ch))
    vpeak = float(peak_per_ch[main_ch])
    snr = vpeak / max(float(noise[main_ch]), 1e-8)
    return snr, main_ch


def per_spike_main_channels(
    filtered: np.ndarray,
    times: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    n_ch, n_samp = filtered.shape
    out = np.zeros(len(times), dtype=np.int32)
    for i, t in enumerate(times):
        t = int(t)
        if t < window or t >= n_samp - window:
            continue
        seg = filtered[:, t - window : t + window]
        out[i] = int(np.argmax(np.max(np.abs(seg), axis=1)))
    return out


def convert_scene(
    scene: Scene,
    out_dir: Path = OUT_DIR,
    snr_threshold: float = 8.0,
    fs: float = 30000.0,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scene.drift}{scene.n_channels}c_{scene.duration_s}s_{scene.rep}_filtered_gt"
    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}_meta.json"

    if npz_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] {npz_path.name} exists")
        with open(meta_path) as f:
            return json.load(f)

    raw_path = RAW_DIR / f"{scene.prefix}_raw.mda"
    fir_path = FIR_DIR / f"{scene.prefix}_firings_true.mda"
    geom_path = GEOM_DIR / f"{scene.prefix}_geom.csv"

    if verbose:
        print(f"[load] {scene.key}")
        print(f"  raw={raw_path.name}")

    raw = read_mda(raw_path)
    # MDA often (n_channels, n_samples) already matching npz
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D raw, got {raw.shape}")
    if raw.shape[0] != scene.n_channels and raw.shape[1] == scene.n_channels:
        raw = raw.T
    if raw.shape[0] != scene.n_channels:
        raise ValueError(f"Channel mismatch: {raw.shape} vs n_channels={scene.n_channels}")

    geom = read_geom_csv(geom_path)
    if geom.shape[0] != scene.n_channels:
        raise ValueError(f"Geom channels {geom.shape[0]} != {scene.n_channels}")

    firings = read_mda(fir_path)
    # (3, N): channel, time(1-based), unit
    if firings.shape[0] == 3:
        times_1b = np.rint(firings[1]).astype(np.int64)
        units = np.rint(firings[2]).astype(np.int64)
    elif firings.shape[1] == 3:
        times_1b = np.rint(firings[:, 1]).astype(np.int64)
        units = np.rint(firings[:, 2]).astype(np.int64)
    else:
        raise ValueError(f"Unexpected firings shape {firings.shape}")

    times_0b = times_1b - 1
    valid_time = (times_0b >= 0) & (times_0b < raw.shape[1])
    times_0b = times_0b[valid_time]
    times_1b = times_1b[valid_time]
    units = units[valid_time]

    if verbose:
        print(f"  raw={raw.shape} firings={len(units)} units_total={len(np.unique(units))}")
        print("  bandpass + SNR filter...")

    filtered = bandpass(raw, fs=fs)
    noise = estimate_noise_mad(filtered)

    unit_ids = np.unique(units)
    snrs = []
    main_chs = []
    keep_units = []
    rng = np.random.default_rng(0)
    for u in unit_ids:
        ut = times_0b[units == u]
        snr, mch = unit_snr_and_main_channel(filtered, ut, noise, rng=rng)
        snrs.append(snr)
        main_chs.append(mch)
        if snr >= snr_threshold:
            keep_units.append(int(u))

    snrs = np.asarray(snrs, dtype=np.float32)
    main_chs = np.asarray(main_chs, dtype=np.int32)
    keep_units = np.asarray(sorted(keep_units), dtype=np.int32)
    unit_to_snr = {int(u): float(s) for u, s in zip(unit_ids, snrs)}
    unit_to_main = {int(u): int(m) for u, m in zip(unit_ids, main_chs)}

    mask = np.isin(units, keep_units)
    times_0b_f = times_0b[mask]
    times_1b_f = times_1b[mask]
    units_f = units[mask].astype(np.int64)

    unit_main_channels = np.asarray([unit_to_main[int(u)] for u in keep_units], dtype=np.int32)
    unit_snrs = np.asarray([unit_to_snr[int(u)] for u in keep_units], dtype=np.float32)
    unit_spike_counts = np.asarray(
        [int(np.sum(units_f == u)) for u in keep_units], dtype=np.int64
    )

    spike_main_unit = np.asarray([unit_to_main[int(u)] for u in units_f], dtype=np.int32)
    if verbose:
        print("  per-spike main channels...")
    spike_main_per = per_spike_main_channels(filtered, times_0b_f)

    # Store raw as int16 like the reference package
    if raw.dtype != np.int16:
        raw_out = np.clip(np.rint(raw), -32768, 32767).astype(np.int16)
    else:
        raw_out = raw.astype(np.int16, copy=False)

    np.savez_compressed(
        npz_path,
        raw_data=raw_out,
        geom=geom,
        spike_times_1based=times_1b_f,
        spike_times_0based=times_0b_f,
        spike_unit_ids=units_f,
        spike_main_channels=spike_main_unit,
        spike_main_channels_unit_template=spike_main_unit,
        spike_main_channels_per_spike=spike_main_per,
        valid_units=keep_units,
        unit_main_channels=unit_main_channels,
        unit_spike_counts=unit_spike_counts,
        unit_snrs=unit_snrs,
    )

    meta = {
        "scene_key": scene.key,
        "scene_prefix": scene.prefix,
        "drift": scene.drift,
        "probe": scene.probe,
        "duration_s": scene.duration_s,
        "rep": scene.rep,
        "threshold": float(snr_threshold),
        "sample_rate": float(fs),
        "raw_file": str(raw_path),
        "ground_truth_file": str(fir_path),
        "geom_file": str(geom_path),
        "n_channels": int(scene.n_channels),
        "n_samples": int(raw.shape[1]),
        "n_units_total": int(len(unit_ids)),
        "n_valid_units": int(len(keep_units)),
        "n_spikes_total": int(len(units)),
        "n_spikes_filtered": int(len(units_f)),
        "valid_units": [int(u) for u in keep_units],
        "unit_spike_counts_filtered": {
            str(int(u)): int(c) for u, c in zip(keep_units, unit_spike_counts)
        },
        "unit_to_main_channel": {
            str(int(u)): int(m) for u, m in zip(keep_units, unit_main_channels)
        },
        "unit_snrs": {str(int(u)): float(s) for u, s in zip(keep_units, unit_snrs)},
        "npz_path": str(npz_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(
            f"  saved {npz_path.name} | valid_units={len(keep_units)} "
            f"spikes={len(units_f)}"
        )
    return meta


def write_catalog(scenes: list[Scene], metas: list[dict] | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_by_key = {m["scene_key"]: m for m in (metas or []) if "scene_key" in m}
    # also load existing metas
    for p in OUT_DIR.glob("*_filtered_gt_meta.json"):
        with open(p) as f:
            m = json.load(f)
        if "scene_key" in m:
            meta_by_key[m["scene_key"]] = m

    entries = []
    for sc in scenes:
        npz = OUT_DIR / f"{sc.drift}{sc.n_channels}c_{sc.duration_s}s_{sc.rep}_filtered_gt.npz"
        e = {
            "scene_key": sc.key,
            "short_name": sc.short_name,
            "prefix": sc.prefix,
            "drift": sc.drift,
            "probe": sc.probe,
            "n_channels": sc.n_channels,
            "duration_s": sc.duration_s,
            "rep": sc.rep,
            "raw_mda": str(RAW_DIR / f"{sc.prefix}_raw.mda"),
            "firings_mda": str(FIR_DIR / f"{sc.prefix}_firings_true.mda"),
            "geom_csv": str(GEOM_DIR / f"{sc.prefix}_geom.csv"),
            "npz_ready": npz.exists(),
            "npz_path": str(npz) if npz.exists() else None,
        }
        if sc.key in meta_by_key:
            m = meta_by_key[sc.key]
            e["n_valid_units"] = m.get("n_valid_units")
            e["n_spikes_filtered"] = m.get("n_spikes_filtered")
            e["valid_units"] = m.get("valid_units")
        entries.append(e)

    catalog = {
        "n_scenes_discovered": len(scenes),
        "n_npz_ready": sum(1 for e in entries if e["npz_ready"]),
        "out_dir": str(OUT_DIR),
        "scenes": entries,
    }
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
    # also CSV index
    csv_path = OUT_DIR / "catalog.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene_key", "drift", "probe", "n_channels", "duration_s", "rep",
                "npz_ready", "n_valid_units", "n_spikes_filtered", "npz_path",
            ],
        )
        w.writeheader()
        for e in entries:
            w.writerow({
                "scene_key": e["scene_key"],
                "drift": e["drift"],
                "probe": e["probe"],
                "n_channels": e["n_channels"],
                "duration_s": e["duration_s"],
                "rep": e["rep"],
                "npz_ready": e["npz_ready"],
                "n_valid_units": e.get("n_valid_units", ""),
                "n_spikes_filtered": e.get("n_spikes_filtered", ""),
                "npz_path": e.get("npz_path") or "",
            })
    return catalog


def filter_scenes(
    scenes: list[Scene],
    only: str | None = None,
    scene_key: str | None = None,
) -> list[Scene]:
    out = []
    for s in scenes:
        if scene_key and s.key != scene_key:
            continue
        if only:
            token = only
            if "static" in token and "drift" not in token and s.drift != "static":
                continue
            if "drift" in token and s.drift != "drift":
                continue
            if "siprobe" in token and s.probe != "siprobe":
                continue
            if "tetrode" in token and s.probe != "tetrode":
                continue
            m_ch = re.search(r"(\d+)c", token)
            if m_ch and s.n_channels != int(m_ch.group(1)):
                continue
            m_dur = re.search(r"(\d+)s", token)
            if m_dur and s.duration_s != int(m_dur.group(1)):
                continue
        out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Hybrid Janelia NPZ datasets")
    parser.add_argument("--list", action="store_true", help="List discovered scenes and exit")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Filter e.g. static_siprobe_16c_600s / drift_siprobe_16c_600s",
    )
    parser.add_argument("--scene-key", type=str, default=None)
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-ref",
        action="store_true",
        help="Compare static16c_600s_11 conversion to existing reference NPZ",
    )
    args = parser.parse_args()

    scenes = discover_scenes()
    if not scenes:
        print("No complete scenes found under Hybridjanelia/", file=sys.stderr)
        sys.exit(1)

    if args.list:
        catalog = write_catalog(scenes)
        print(f"Discovered {catalog['n_scenes_discovered']} scenes; "
              f"{catalog['n_npz_ready']} NPZ ready")
        print(f"Catalog: {CATALOG_PATH}")
        # summary counts
        from collections import Counter
        c = Counter((s.drift, s.probe, s.n_channels, s.duration_s) for s in scenes)
        for k, v in sorted(c.items()):
            print(f"  {k}: {v}")
        return

    selected = filter_scenes(scenes, only=args.only, scene_key=args.scene_key)
    if not selected:
        print("No scenes matched filters", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(selected)} scene(s)...")
    metas = []
    for sc in selected:
        meta = convert_scene(
            sc,
            snr_threshold=args.snr_threshold,
            overwrite=args.overwrite,
        )
        metas.append(meta)

    catalog = write_catalog(scenes, metas)
    print(f"\nDone. NPZ ready: {catalog['n_npz_ready']}/{catalog['n_scenes_discovered']}")
    print(f"Catalog: {CATALOG_PATH}")

    if args.validate_ref:
        ref = OUT_DIR / "static16c_600s_11_filtered_gt.npz"
        new = OUT_DIR / "static16c_600s_11_filtered_gt.npz"
        # After convert, our naming is static16c_600s_11_filtered_gt.npz — same as ref
        if not ref.exists():
            print("No reference NPZ to validate")
            return
        a = np.load(ref)
        # Prefer freshly written path from meta
        new_path = Path(metas[0]["npz_path"]) if metas else new
        b = np.load(new_path)
        print("\nValidation vs reference:")
        print("  valid_units ref", a["valid_units"], "new", b["valid_units"])
        print("  n_spikes ref", len(a["spike_times_0based"]), "new", len(b["spike_times_0based"]))
        print("  raw equal", np.array_equal(a["raw_data"], b["raw_data"]))
        print("  geom close", np.allclose(a["geom"], b["geom"]))


if __name__ == "__main__":
    main()
