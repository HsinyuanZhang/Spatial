"""Per-unit main-channel / KS4-position wander on Yger KS4 Th=13.

Home channel is the most-negative site in ±2 samples at the event time
(not the whole-array |amp| argmax used in the first COM eval).

    python -m Spatial.experiments.run_main_channel_drift --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.data.loader import YGER_KS4_NPZ_PATH, load_yger_ks4
from Spatial.experiments.run_adaptive_5bit_range_search import labels_for_peak_output

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "main_channel_drift"
KS4_DIR = Path(
    "/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work/"
    "ks_out/20160415_patch2_th13"
)
HOME_HALF_WIN = 2
FILTER_LOW_HZ = 300.0
FILTER_HIGH_HZ = 6000.0


def _homes_local_min(filtered: np.ndarray, times: np.ndarray, half: int = HOME_HALF_WIN) -> np.ndarray:
    n_ch, n_samp = filtered.shape
    homes = np.empty(len(times), dtype=np.int64)
    for i, t in enumerate(times):
        t0 = max(int(t) - half, 0)
        t1 = min(int(t) + half + 1, n_samp)
        homes[i] = int(np.argmin(filtered[:, t0:t1].min(axis=1)))
    return homes


def _unit_rows(
    units: np.ndarray,
    xy: np.ndarray,
    min_spikes: int,
) -> list[dict]:
    rows = []
    for u in np.unique(units):
        sel = units == u
        pts = xy[sel]
        n = int(pts.shape[0])
        if n < 1:
            continue
        # geometric median approx: coordinate-wise median
        center = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        # majority channel if xy are electrode coords snapped to geom we pass homes as xy
        rows.append(
            {
                "unit_id": int(u),
                "n_spikes": n,
                "center_x_um": float(center[0]),
                "center_y_um": float(center[1]),
                "rms_um": float(np.sqrt(np.mean(dist ** 2))),
                "median_um": float(np.median(dist)),
                "p90_um": float(np.percentile(dist, 90)) if n >= 2 else 0.0,
                "max_um": float(np.max(dist)),
                "frac_within_30um": float(np.mean(dist <= 30.0)),
                "frac_within_60um": float(np.mean(dist <= 60.0)),
                "included_in_summary": int(n >= min_spikes),
            }
        )
    return rows


def _home_unit_rows(
    units: np.ndarray,
    homes: np.ndarray,
    geom: np.ndarray,
    min_spikes: int,
) -> list[dict]:
    rows = []
    for u in np.unique(units):
        sel = units == u
        ch = homes[sel].astype(np.int64)
        n = int(ch.size)
        if n < 1:
            continue
        counts = np.bincount(ch, minlength=geom.shape[0])
        maj = int(np.argmax(counts))
        xy = geom[ch]
        center = geom[maj]
        dist = np.linalg.norm(xy - center[None, :], axis=1)
        rows.append(
            {
                "unit_id": int(u),
                "n_spikes": n,
                "majority_channel": maj,
                "majority_frac": float(counts[maj] / n),
                "n_unique_homes": int(np.unique(ch).size),
                "majority_x_um": float(center[0]),
                "majority_y_um": float(center[1]),
                "rms_um": float(np.sqrt(np.mean(dist ** 2))),
                "median_um": float(np.median(dist)),
                "p90_um": float(np.percentile(dist, 90)) if n >= 2 else 0.0,
                "max_um": float(np.max(dist)),
                "frac_within_30um": float(np.mean(dist <= 30.0)),
                "frac_within_60um": float(np.mean(dist <= 60.0)),
                "included_in_summary": int(n >= min_spikes),
            }
        )
    return rows


def _summarize(rows: list[dict], dist_keys: tuple[str, ...]) -> dict:
    kept = [r for r in rows if r.get("included_in_summary")]
    if not kept:
        return {"n_units": 0}
    out: dict = {"n_units": len(kept), "n_units_all": len(rows)}
    for key in dist_keys:
        vals = np.array([r[key] for r in kept], dtype=np.float64)
        out[key] = {
            "min": float(vals.min()),
            "p10": float(np.percentile(vals, 10)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--min-spikes", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ds = load_yger_ks4(duration_s=args.duration)
    filtered = bandpass_filter(ds.raw_data, ds.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ)
    _, peak_times, homes_global = get_peak_amplitudes(
        filtered, ds.spike_times, window=15
    )
    labels = labels_for_peak_output(
        ds.spike_times, ds.spike_units, peak_times, ds.n_samples, window=15
    )
    homes_local = _homes_local_min(filtered, peak_times, HOME_HALF_WIN)

    local_rows = _home_unit_rows(labels, homes_local, ds.geom, args.min_spikes)
    global_rows = _home_unit_rows(labels, homes_global, ds.geom, args.min_spikes)

    ks_times = np.load(KS4_DIR / "spike_times.npy").reshape(-1).astype(np.int64)
    ks_clu = np.load(KS4_DIR / "spike_clusters.npy").reshape(-1).astype(np.int64)
    ks_pos = np.load(KS4_DIR / "spike_positions.npy").astype(np.float64)
    n_keep = int(args.duration * ds.fs)
    ks_mask = ks_times < n_keep
    pos_rows = _unit_rows(ks_clu[ks_mask], ks_pos[ks_mask], args.min_spikes)

    juxta = np.load(YGER_KS4_NPZ_PATH)["juxta_spike_times"].astype(np.int64)
    juxta_half = 8  # extra peak lags juxta by about -5 samples
    juxta = juxta[(juxta >= juxta_half) & (juxta < ds.n_samples - juxta_half)]
    near69 = np.linalg.norm(ds.geom - ds.geom[69], axis=1) <= 60.0
    juxta_homes = np.empty(len(juxta), dtype=np.int64)
    near_idx = np.flatnonzero(near69)
    for i, t in enumerate(juxta):
        sl = filtered[near69, int(t) - juxta_half : int(t) + juxta_half + 1].min(axis=1)
        juxta_homes[i] = int(near_idx[int(np.argmin(sl))])
    juxta_rows = _home_unit_rows(
        np.zeros(len(juxta_homes), dtype=np.int64),
        juxta_homes,
        ds.geom,
        min_spikes=1,
    )
    if juxta_rows:
        juxta_rows[0]["unit_id"] = -310
        juxta_rows[0]["note"] = "author juxta times; home among channels within 60 um of ch69"

    # Home among electrodes within 60 um of that event's KS4 position.
    n_keep = int(args.duration * ds.fs)
    ks_mask = ks_times < n_keep
    time_to_pos = {int(t): ks_pos[i] for i, t in enumerate(ks_times) if ks_mask[i]}
    homes_ks = np.full(len(peak_times), -1, dtype=np.int64)
    for i, t in enumerate(peak_times):
        xy = time_to_pos.get(int(t))
        if xy is None:
            continue
        near = np.linalg.norm(ds.geom - xy, axis=1) <= 60.0
        if not np.any(near):
            continue
        t0 = max(int(t) - HOME_HALF_WIN, 0)
        t1 = min(int(t) + HOME_HALF_WIN + 1, ds.n_samples)
        sl = filtered[near, t0:t1].min(axis=1)
        homes_ks[i] = int(np.flatnonzero(near)[int(np.argmin(sl))])
    ks_local_rows = _home_unit_rows(labels, homes_ks, ds.geom, args.min_spikes)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(args.out_dir / "home_local_min_pm2.csv", local_rows)
    write_csv(args.out_dir / "home_global_argmax_pm15.csv", global_rows)
    write_csv(args.out_dir / "home_within_60um_of_ks4_pos.csv", ks_local_rows)
    write_csv(args.out_dir / "ks4_spike_positions.csv", pos_rows)
    write_csv(args.out_dir / "juxta_home_within_60um_of_ch69.csv", juxta_rows)

    dist_home = (
        "rms_um",
        "median_um",
        "p90_um",
        "max_um",
        "majority_frac",
        "n_unique_homes",
        "frac_within_30um",
        "frac_within_60um",
    )
    dist_pos = ("rms_um", "median_um", "p90_um", "max_um", "frac_within_30um", "frac_within_60um")
    summary = {
        "dataset": ds.name,
        "duration_s": args.duration,
        "min_spikes": args.min_spikes,
        "home_rule_primary": "most-negative channel in ±2 samples at event time",
        "home_rule_legacy": "global argmax |amp| in ±15 samples (COM-eval method)",
        "ks4_positions": "Kilosort4 spike_positions.npy, median-centered RMS",
        "local_home": _summarize(local_rows, dist_home),
        "legacy_global_home": _summarize(global_rows, dist_home),
        "home_near_ks4_pos": _summarize(ks_local_rows, dist_home),
        "ks4_positions_summary": _summarize(pos_rows, dist_pos),
        "juxta_local_home": juxta_rows[0] if juxta_rows else {},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def _line(title: str, block: dict, key: str, unit: str = "µm") -> None:
        d = block[key]
        print(
            f"  {title:28s}  med {d['median']:7.2f}  p10 {d['p10']:7.2f}  "
            f"p90 {d['p90']:7.2f}  max {d['max']:7.2f} {unit}"
        )

    loc = summary["local_home"]
    print(
        f"{ds.name}  {args.duration:.0f}s  units with ≥{args.min_spikes} spikes: "
        f"{loc['n_units']} / {loc['n_units_all']}"
    )
    print("local home (argmin ±2 samples)")
    _line("RMS from majority home", loc, "rms_um")
    _line("median |Δ|", loc, "median_um")
    _line("p90 |Δ|", loc, "p90_um")
    _line("unique homes / unit", loc, "n_unique_homes", unit="")
    _line("majority-home fraction", loc, "majority_frac", unit="")
    print(
        f"  frac spikes within 30/60 µm of majority home:  "
        f"med {loc['frac_within_30um']['median']:.2f} / "
        f"{loc['frac_within_60um']['median']:.2f}"
    )
    print("home among channels within 60 µm of KS4 spike position")
    kn = summary["home_near_ks4_pos"]
    _line("RMS from majority home", kn, "rms_um")
    _line("median |Δ|", kn, "median_um")
    _line("p90 |Δ|", kn, "p90_um")
    _line("majority-home fraction", kn, "majority_frac", unit="")
    print(
        f"  frac spikes within 30/60 µm of majority home:  "
        f"med {kn['frac_within_30um']['median']:.2f} / "
        f"{kn['frac_within_60um']['median']:.2f}"
    )
    print("KS4 spike_positions (same duration)")
    kp = summary["ks4_positions_summary"]
    _line("RMS from median x,y", kp, "rms_um")
    _line("median |Δ|", kp, "median_um")
    _line("p90 |Δ|", kp, "p90_um")
    j = summary["juxta_local_home"]
    print(
        "juxta times (local home): "
        f"n={j.get('n_spikes')} majority_ch={j.get('majority_channel')} "
        f"frac={j.get('majority_frac'):.3f} rms={j.get('rms_um'):.1f} µm "
        f"p90={j.get('p90_um'):.1f} µm unique_homes={j.get('n_unique_homes')}"
    )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
