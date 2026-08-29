"""How much drift does the locked 60 s window actually contain, and can the
deployed representation express it?

Every drift-tracking result in this repository is measured on the first 60 s
of a 600 s Hybrid Janelia recording, with the tracker evaluated on the last
25 % of that cut (15 s). This runner measures three things on the **input
recording**, not on any tracker:

W1  how much spatial motion the full recording contains, and what fraction of
    it survives inside the used window;
W2  whether the deployed +-0.5 pitch shift grid covers that motion;
W3  whether the frozen per-unit home channel, which fixes the K-slot channel
    selection of every template, stays valid over the recording.

Position is measured by the amplitude-weighted centre of mass of the P2P
footprint along the primary probe axis. COM is invariant to a global
amplitude scale, so it does **not** inherit the amplitude bias of
``estimate_best_shift`` (see ``drift_estimator_diagnostics.md``). It is a
model-free position readout, not a tracker.

The static recording is the control: its measured motion and home instability
are this readout's noise floor, not drift.

    python -m Spatial.experiments.run_drift_window_budget --pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    event_peak_channel,
    extract_all_channel_extrema,
)
from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    median_nearest_neighbor_spacing,
    primary_axis,
)
from Spatial.data.loader import load_hybrid_janelia
from Spatial.experiments.posneg_mvm_common import (
    REPO_ROOT,
    json_ready,
    rel_or_str,
    source_hashes,
    write_csv_rows,
    write_sha256_manifest,
)
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "drift_window_budget"
SOURCE_PATHS = ("Spatial/experiments/run_drift_window_budget.py",)

SCENES = ("drift16c_600s_11", "static16c_600s_11")
FULL_S = 600.0
USED_S = 60.0          # window every drift result is measured on
TEST_FRAC = 0.25       # the tracker is evaluated on the last 25 % of that
BLOCK_S = 5.0
AMP_FLOOR = 0.3        # keep channels at or above this fraction of the peak
MIN_EVENTS_PER_BLOCK = 3
WINDOW = 15
FIT_END_S = USED_S * 0.5   # the pipeline fits mu on the first 50 % of the cut
SHIFT_GRID_HALFWIDTH = 0.5  # SHIFT_FRACTIONAL half-width, in pitch units


def com_along_axis(
    p2p: np.ndarray,
    pos: np.ndarray,
    amp_floor: float = AMP_FLOOR,
) -> np.ndarray:
    """Amplitude-weighted centre of mass, one value per event, in micrometres.

    Channels below ``amp_floor`` times the event peak are dropped so the noise
    floor on distant sites does not pull the centre toward the probe middle.
    Invariant to a global amplitude scale by construction.
    """
    a = np.asarray(p2p, dtype=np.float64)
    peak = np.max(a, axis=1, keepdims=True)
    w = np.where(a >= float(amp_floor) * peak, a, 0.0)
    tot = np.sum(w, axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = tot > 0
    out[ok] = (w[ok] @ np.asarray(pos, dtype=np.float64)) / tot[ok]
    return out


def axis_row_step(geom: np.ndarray, axis: int) -> float:
    """Electrode row spacing **along the drift axis**.

    This is not ``median_nearest_neighbor_spacing``. On a staggered probe the
    nearest neighbour sits on the other column, so the NN distance is the
    diagonal and overstates the step a unit must travel to change rows. The
    whole pipeline carries the NN value as ``pitch_um``, so both are reported
    and every shift-grid statement has to say which one it uses.
    """
    u = np.unique(np.round(np.asarray(geom, dtype=np.float64)[:, int(axis)], 6))
    return float(np.median(np.diff(u))) if u.size >= 2 else float("nan")


def majority(values: np.ndarray, n_ch: int) -> int:
    if values.size == 0:
        return -1
    return int(np.bincount(np.clip(values, 0, n_ch - 1), minlength=n_ch).argmax())


def home_migration(
    peak_ch: np.ndarray,
    times_s: np.ndarray,
    n_ch: int,
    block_s: float = BLOCK_S,
    t_end: float = FULL_S,
) -> dict[str, Any]:
    """Does one unit's frozen home channel stay valid over the recording?

    ``fit_home`` reproduces ``majority_home`` on the pipeline's fit split: the
    mode of the per-event peak channel over the first 50 % of the 60 s cut.
    Nothing in ``CausalTracker`` ever updates it, so a block whose own majority
    home differs is a block where the template's channel selection is wrong.
    """
    fit_mask = times_s < FIT_END_S
    fit_home = majority(peak_ch[fit_mask], n_ch)
    edges = np.arange(0.0, float(t_end) + block_s, float(block_s))
    block_homes: list[int] = []
    for i in range(edges.size - 1):
        m = (times_s >= edges[i]) & (times_s < edges[i + 1])
        if int(m.sum()) >= MIN_EVENTS_PER_BLOCK:
            block_homes.append(majority(peak_ch[m], n_ch))
    bh = np.array(block_homes, dtype=np.int64)
    n_blocks = int(bh.size)
    n_diff = int(np.sum(bh != fit_home)) if n_blocks else 0
    return {
        "fit_home": fit_home,
        "n_blocks": n_blocks,
        "n_blocks_home_differs": n_diff,
        "frac_blocks_home_differs": (n_diff / n_blocks) if n_blocks else float("nan"),
        "n_distinct_block_homes": int(np.unique(bh).size) if n_blocks else 0,
        "frac_events_off_fit_home": float(np.mean(peak_ch != fit_home)),
    }


def block_medians(
    com: np.ndarray,
    times_s: np.ndarray,
    t0: float,
    t1: float,
    block_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median COM per time block inside [t0, t1). Returns centres, values, n."""
    edges = np.arange(float(t0), float(t1) + block_s, float(block_s))
    centres = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(centres.size, np.nan)
    counts = np.zeros(centres.size, dtype=np.int64)
    for i in range(centres.size):
        m = (times_s >= edges[i]) & (times_s < edges[i + 1]) & np.isfinite(com)
        counts[i] = int(m.sum())
        if counts[i] >= MIN_EVENTS_PER_BLOCK:
            vals[i] = float(np.median(com[m]))
    return centres, vals, counts


SMOOTH_BLOCKS = 5  # 25 s at BLOCK_S=5


def _rolling_median(v: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    return np.array(
        [np.median(v[max(0, i - half) : min(v.size, i + half + 1)]) for i in range(v.size)]
    )


def segment_motion(centres: np.ndarray, vals: np.ndarray) -> dict[str, float]:
    """Motion statistics for one COM trace segment.

    ``net_um`` is the linear trend across the segment. The HJ drift profile is
    an excursion that returns toward its origin, so the trend **understates**
    how far a unit gets from a fixed template; ``excursion_um`` is the
    statistic that answers that, smoothed so per-block sampling noise does not
    dominate it.
    """
    ok = np.isfinite(vals)
    nan = float("nan")
    if int(ok.sum()) < 2:
        return {
            "net_um": nan,
            "spread_um": nan,
            "excursion_um": nan,
            "slope_um_per_s": nan,
            "n_blocks": int(ok.sum()),
        }
    c = centres[ok]
    v = vals[ok]
    slope = float(np.polyfit(c, v, 1)[0]) if c.size >= 2 and np.std(c) > 0 else nan
    if v.size >= SMOOTH_BLOCKS:
        sm = _rolling_median(v, SMOOTH_BLOCKS)
        excursion = float(np.max(sm) - np.min(sm))
    else:
        excursion = nan
    return {
        # Net displacement is read off the fitted trend, so a single noisy
        # first or last block cannot define it.
        "net_um": float(slope * (c[-1] - c[0])) if np.isfinite(slope) else nan,
        "spread_um": float(np.percentile(v, 90) - np.percentile(v, 10)),
        "excursion_um": excursion,
        "slope_um_per_s": slope,
        "n_blocks": int(ok.sum()),
    }


def analyse_scene(scene: str, quiet: bool) -> dict[str, Any]:
    dataset = load_hybrid_janelia(scene=scene, duration_s=FULL_S)
    fs = float(dataset.fs)
    geom = np.asarray(dataset.geom, dtype=np.float64)
    axis = int(primary_axis(geom))
    pitch = float(median_nearest_neighbor_spacing(geom))
    pos = geom[:, axis]

    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    times = np.asarray(dataset.spike_times, dtype=np.int64)
    units_all = np.asarray(dataset.spike_units, dtype=np.int64)
    keep = (times >= WINDOW) & (times < filtered.shape[1] - WINDOW)
    times = times[keep]
    units_all = units_all[keep]
    vmax, vmin = extract_all_channel_extrema(filtered, times, window=WINDOW)
    del filtered, dataset

    p2p = np.maximum(vmax, 0.0) + np.maximum(-vmin, 0.0)
    com = com_along_axis(p2p, pos)
    peak_ch = event_peak_channel(vmax, vmin)
    n_ch = int(geom.shape[0])
    row_step = axis_row_step(geom, axis)
    t_s = times.astype(np.float64) / fs

    test_t0 = USED_S * (1.0 - TEST_FRAC)
    segments = {
        "full_600s": (0.0, FULL_S),
        "used_60s": (0.0, USED_S),
        "test_15s": (test_t0, USED_S),
    }

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for unit in np.unique(units_all):
        m = units_all == unit
        if int(m.sum()) < 50:
            continue
        cu = com[m]
        tu = t_s[m]
        row: dict[str, Any] = {
            "scene": scene,
            "unit": int(unit),
            "n_spikes": int(m.sum()),
            "rate_hz": float(m.sum()) / FULL_S,
            "pitch_um": pitch,
            "row_step_um": row_step,
        }
        row.update(
            {f"home_{k}": v for k, v in home_migration(peak_ch[m], tu, n_ch).items()}
        )
        for name, (t0, t1) in segments.items():
            centres, vals, counts = block_medians(cu, tu, t0, t1, BLOCK_S)
            mot = segment_motion(centres, vals)
            row[f"{name}_net_um"] = mot["net_um"]
            row[f"{name}_net_pitch"] = mot["net_um"] / pitch
            row[f"{name}_spread_um"] = mot["spread_um"]
            row[f"{name}_excursion_um"] = mot["excursion_um"]
            row[f"{name}_excursion_pitch"] = mot["excursion_um"] / pitch
            row[f"{name}_slope_um_per_s"] = mot["slope_um_per_s"]
            row[f"{name}_n_blocks"] = mot["n_blocks"]
            if name == "full_600s":
                traces[str(int(unit))] = {
                    "centres_s": centres,
                    "com_um": vals,
                    "counts": counts,
                    "events_t_s": tu,
                    "events_com_um": cu,
                }
        rows.append(row)
        if not quiet:
            print(
                f"{scene:<20} u{int(unit):<4} rate={row['rate_hz']:5.2f}Hz "
                f"600s net={row['full_600s_net_um']:+7.2f}um "
                f"60s net={row['used_60s_net_um']:+6.2f}um "
                f"15s net={row['test_15s_net_um']:+6.2f}um",
                flush=True,
            )
    return {
        "rows": rows,
        "traces": traces,
        "pitch": pitch,
        "axis": axis,
        "channel_pos_um": pos,
        "row_step_um": row_step,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scene in sorted({r["scene"] for r in rows}):
        sub = [r for r in rows if r["scene"] == scene]
        rec: dict[str, Any] = {
            "scene": scene,
            "n_units": len(sub),
            "pitch_um": sub[0]["pitch_um"],
            "row_step_um": sub[0]["row_step_um"],
            "shift_grid_halfwidth_um": SHIFT_GRID_HALFWIDTH * sub[0]["pitch_um"],
            "shift_grid_halfwidth_rows": (
                SHIFT_GRID_HALFWIDTH * sub[0]["pitch_um"] / sub[0]["row_step_um"]
            ),
            "n_units_home_migrates": sum(1 for r in sub if r["home_n_blocks_home_differs"] > 0),
            "max_blocks_home_differs": max(r["home_n_blocks_home_differs"] for r in sub),
            "median_frac_blocks_home_differs": float(
                np.median([r["home_frac_blocks_home_differs"] for r in sub])
            ),
            "max_distinct_block_homes": max(r["home_n_distinct_block_homes"] for r in sub),
        }
        for name in ("full_600s", "used_60s", "test_15s"):
            net = np.array([abs(r[f"{name}_net_um"]) for r in sub], dtype=np.float64)
            spread = np.array([r[f"{name}_spread_um"] for r in sub], dtype=np.float64)
            exc = np.array([r[f"{name}_excursion_um"] for r in sub], dtype=np.float64)
            net = net[np.isfinite(net)]
            spread = spread[np.isfinite(spread)]
            exc = exc[np.isfinite(exc)]
            rec[f"{name}_median_abs_net_um"] = float(np.median(net)) if net.size else float("nan")
            rec[f"{name}_max_abs_net_um"] = float(np.max(net)) if net.size else float("nan")
            rec[f"{name}_median_abs_net_pitch"] = (
                float(np.median(net)) / rec["pitch_um"] if net.size else float("nan")
            )
            rec[f"{name}_median_spread_um"] = (
                float(np.median(spread)) if spread.size else float("nan")
            )
            rec[f"{name}_median_excursion_um"] = (
                float(np.median(exc)) if exc.size else float("nan")
            )
            rec[f"{name}_max_excursion_um"] = float(np.max(exc)) if exc.size else float("nan")
            rec[f"{name}_median_excursion_pitch"] = (
                float(np.median(exc)) / rec["pitch_um"] if exc.size else float("nan")
            )
        for name in ("used_60s", "test_15s"):
            a = rec[f"{name}_median_abs_net_um"]
            b = rec["full_600s_median_abs_net_um"]
            rec[f"{name}_frac_of_full"] = float(a / b) if b and np.isfinite(b) and b > 0 else float("nan")
        out.append(rec)
    return out


def _plot(all_traces: dict[str, dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    scenes = sorted(all_traces)
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 3.6), squeeze=False)
    for ax, scene in zip(axes[0], scenes):
        for unit, tr in all_traces[scene].items():
            v = np.asarray(tr["com_um"], dtype=np.float64)
            c = np.asarray(tr["centres_s"], dtype=np.float64)
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            ax.plot(c[ok], v[ok] - np.nanmedian(v[ok][:3]), lw=1.0, label=f"u{unit}")
        ax.axvspan(0, USED_S, color="orange", alpha=0.18)
        ax.axvspan(USED_S * (1 - TEST_FRAC), USED_S, color="red", alpha=0.22)
        ax.set_title(f"{scene}\norange = used 60 s, red = 15 s test")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("COM shift from start (um)")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = out / "plots" / "com_drift_full_vs_window.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")

    rows: list[dict[str, Any]] = []
    all_traces: dict[str, dict[str, Any]] = {}
    for scene in args.scenes:
        res = analyse_scene(scene, bool(args.quiet))
        rows.extend(res["rows"])
        all_traces[scene] = res["traces"]

    summary_rows = summarize(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_unit = out / "drift_window_budget_per_unit.csv"
    summary_csv = out / "drift_window_budget_summary.csv"
    write_csv_rows(per_unit, rows)
    write_csv_rows(summary_csv, summary_rows)
    _plot(all_traces, out)

    json_path = out / "drift_window_budget_summary.json"
    write_json(
        json_path,
        json_ready(
            {
                "note": (
                    "Amplitude-weighted COM along the primary probe axis, "
                    "GT spike times, median per 5 s block. Static scene is the "
                    "readout noise floor, not drift. Home migration reproduces "
                    "majority_home on the pipeline fit split; nothing in "
                    "CausalTracker ever updates it."
                ),
                "full_duration_s": FULL_S,
                "used_window_s": USED_S,
                "test_window_s": USED_S * TEST_FRAC,
                "fit_split_end_s": FIT_END_S,
                "block_s": BLOCK_S,
                "amp_floor": AMP_FLOOR,
                "summary": summary_rows,
                "source_sha256": source_hashes(SOURCE_PATHS),
                "gt_used": True,
            }
        ),
    )
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            rel_or_str(p): sha256_file(p)
            for p in (per_unit, summary_csv, json_path)
            if p.exists()
        },
    )

    print("\n=== motion budget (median over units) ===")
    for r in summary_rows:
        print(f"{r['scene']}  pitch={r['pitch_um']:.2f}um  n_units={r['n_units']}")
        for name in ("full_600s", "used_60s", "test_15s"):
            print(
                f"   {name:<10} |trend net|={r[f'{name}_median_abs_net_um']:6.2f}um  "
                f"excursion(25s-smoothed)={r[f'{name}_median_excursion_um']:6.2f}um "
                f"({r[f'{name}_median_excursion_pitch']:.3f} pitch, "
                f"max {r[f'{name}_max_excursion_um']:6.2f}um)",
                flush=True,
            )

    print("\n=== W3: frozen home channel over 600 s ===")
    for r in summary_rows:
        print(
            f"{r['scene']:<20} row step={r['row_step_um']:.1f}um vs pitch_um="
            f"{r['pitch_um']:.2f}um  =>  +-0.5 pitch = "
            f"{r['shift_grid_halfwidth_rows']:.2f} rows",
            flush=True,
        )
        print(
            f"   units whose block-majority home leaves the fit home: "
            f"{r['n_units_home_migrates']}/{r['n_units']}   "
            f"worst {r['max_blocks_home_differs']} blocks   "
            f"max distinct homes visited {r['max_distinct_block_homes']}",
            flush=True,
        )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
