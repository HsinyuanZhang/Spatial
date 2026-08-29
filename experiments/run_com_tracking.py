"""Causal COM tracking: can a predictor stay in the true COM range?

No identity matching. GT event COM traces from the 600 s window-budget
readout; predictors use only the past. Static scene is the noise-floor control.

    python -m Spatial.experiments.run_com_tracking --pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np

from Spatial.algorithms.com_tracking import (
    contemporaneous_block_com,
    predict_ema,
    predict_frozen,
    predict_hold,
    predict_velocity,
    stream_ema_event,
    stream_every_dt,
    stream_every_n,
    tracking_errors,
)
from Spatial.experiments.posneg_mvm_common import (
    REPO_ROOT,
    json_ready,
    rel_or_str,
    source_hashes,
    try_savefig,
    write_csv_rows,
    write_sha256_manifest,
)
from Spatial.experiments.run_drift_window_budget import (
    BLOCK_S,
    FIT_END_S,
    FULL_S,
    MIN_EVENTS_PER_BLOCK,
    analyse_scene,
)

# Two independent HJ drift recordings (full 600 s). No classification.
DRIFT_SCENES = ("drift16c_600s_11", "drift16c_600s_12")
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "com_tracking"
SOURCE_PATHS = (
    "Spatial/algorithms/com_tracking.py",
    "Spatial/experiments/run_com_tracking.py",
    "Spatial/experiments/run_drift_window_budget.py",
    "Spatial/tests/test_com_tracking.py",
)
EMA_ALPHAS = (0.2, 0.5, 1.0)
STREAM_N = (1, 5, 10, 20)
STREAM_DT_S = (0.5, 1.0, 2.0, 5.0)
STREAM_EMA_A = (0.05, 0.2, 0.5)


def _radii(row_step: float, pitch: float) -> tuple[float, float, float]:
    return (0.5 * float(row_step), float(row_step), 0.5 * float(pitch))


def _predictors(t_fit_end: float) -> list[tuple[str, Callable[..., np.ndarray]]]:
    out: list[tuple[str, Callable[..., np.ndarray]]] = [
        ("frozen_fit", lambda y, t: predict_frozen(y, t, t_fit_end)),
        ("hold", lambda y, t: predict_hold(y)),
        ("velocity", lambda y, t: predict_velocity(y)),
    ]
    for a in EMA_ALPHAS:
        out.append((f"ema_{a:g}", lambda y, t, a=a: predict_ema(y, a)))
    return out


def eval_unit(
    observed: np.ndarray,
    centres: np.ndarray,
    channel_pos: np.ndarray,
    radii: tuple[float, ...],
    t_fit_end: float,
) -> list[dict[str, Any]]:
    rows = []
    for name, fn in _predictors(t_fit_end):
        hat = fn(observed, centres)
        rec = tracking_errors(observed, hat, channel_pos, radii)
        rec["arm"] = name
        rec["hat"] = hat
        rows.append(rec)
    return rows


def run_pilot(scenes: list[str], quiet: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces_out: dict[str, Any] = {}
    for scene in scenes:
        res = analyse_scene(scene, quiet)
        pos = np.asarray(res["channel_pos_um"], dtype=np.float64)
        radii = _radii(res["row_step_um"], res["pitch"])
        traces_out[scene] = {"channel_pos_um": pos, "units": {}}
        for uid, tr in res["traces"].items():
            y = np.asarray(tr["com_um"], dtype=np.float64)
            t = np.asarray(tr["centres_s"], dtype=np.float64)
            evals = eval_unit(y, t, pos, radii, FIT_END_S)
            traces_out[scene]["units"][uid] = {"centres_s": t, "com_um": y}
            ev_t = np.asarray(tr.get("events_t_s", []), dtype=np.float64)
            ev_y = np.asarray(tr.get("events_com_um", []), dtype=np.float64)
            if ev_y.size:
                stream_hats: list[tuple[str, np.ndarray]] = [
                    ("stream_frozen", predict_frozen(ev_y, ev_t, FIT_END_S)),
                    (
                        "oracle_block",
                        contemporaneous_block_com(
                            ev_y, ev_t, BLOCK_S, leave_one_out=False,
                            min_events=MIN_EVENTS_PER_BLOCK,
                        ),
                    ),
                    (
                        "oracle_loo",
                        contemporaneous_block_com(
                            ev_y, ev_t, BLOCK_S, leave_one_out=True,
                            min_events=MIN_EVENTS_PER_BLOCK,
                        ),
                    ),
                ]
                for n in STREAM_N:
                    stream_hats.append(
                        (f"stream_n{n}", stream_every_n(ev_y, n, init_n=10))
                    )
                for dt in STREAM_DT_S:
                    stream_hats.append(
                        (f"stream_dt{dt:g}s", stream_every_dt(ev_y, ev_t, dt, t_init_end=FIT_END_S))
                    )
                for a in STREAM_EMA_A:
                    stream_hats.append(
                        (f"stream_ema{a:g}", stream_ema_event(ev_y, a))
                    )
                for name, hat in stream_hats:
                    rec = tracking_errors(ev_y, hat, pos, radii)
                    rec["arm"] = name
                    rec["hat"] = hat
                    evals.append(rec)
            for rec in evals:
                traces_out[scene]["units"][uid][rec["arm"]] = rec["hat"]
                one_row_key = f"frac_within_{radii[1]:g}um"
                n_out_one = rec.get(f"n_out_{radii[1]:g}um", 0)
                n_eval = int(rec["n_eval"])
                frac_in = rec[one_row_key]
                frac_out = rec.get(
                    f"frac_out_{radii[1]:g}um",
                    (1.0 - frac_in) if np.isfinite(frac_in) else float("nan"),
                )
                row = {
                    "scene": scene,
                    "unit": int(uid),
                    "arm": rec["arm"],
                    "role": (
                        "ORACLE_SCATTER"
                        if rec["arm"] in ("oracle_block", "oracle_loo")
                        else "DIAGNOSTIC_GT_COM"
                    ),
                    "gt_used": True,
                    "row_step_um": float(res["row_step_um"]),
                    "pitch_um": float(res["pitch"]),
                    "half_row_um": radii[0],
                    "one_row_um": radii[1],
                    "half_pitch_um": radii[2],
                    "n_eval": n_eval,
                    "n_out_one_row": int(n_out_one),
                    "n_channel_miss": int(rec.get("n_channel_miss", 0)),
                    "frac_out_one_row": float(frac_out) if np.isfinite(frac_out) else float("nan"),
                    "frac_channel_miss": rec.get(
                        "frac_channel_miss",
                        (1.0 - rec["frac_channel_match"])
                        if np.isfinite(rec["frac_channel_match"])
                        else float("nan"),
                    ),
                    "median_abs_um": rec["median_abs_um"],
                    "p90_abs_um": rec["p90_abs_um"],
                    "max_abs_um": rec["max_abs_um"],
                    "frac_within_half_row": rec[f"frac_within_{radii[0]:g}um"],
                    "frac_within_one_row": frac_in,
                    "frac_within_half_pitch": rec[f"frac_within_{radii[2]:g}um"],
                    "frac_channel_match": rec["frac_channel_match"],
                    "longest_miss_half_row": rec[f"longest_miss_{radii[0]:g}um"],
                    "longest_miss_one_row": rec[f"longest_miss_{radii[1]:g}um"],
                    "longest_channel_miss": rec["longest_channel_miss"],
                }
                rows.append(row)
                if not quiet:
                    print(
                        f"{scene:<20} u{uid:<4} {rec['arm']:<12} "
                        f"med={rec['median_abs_um']:6.2f}um  "
                        f"in-row={row['frac_within_one_row']:.3f}  "
                        f"ch={row['frac_channel_match']:.3f}",
                        flush=True,
                    )
    return {"rows": rows, "traces": traces_out}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scene in sorted({r["scene"] for r in rows}):
        arms = sorted({r["arm"] for r in rows if r["scene"] == scene})
        for arm in arms:
            sub = [r for r in rows if r["scene"] == scene and r["arm"] == arm]
            def med(k):
                v = np.array([r[k] for r in sub], dtype=np.float64)
                v = v[np.isfinite(v)]
                return float(np.median(v)) if v.size else float("nan")
            def mx(k):
                v = np.array([r[k] for r in sub], dtype=np.float64)
                v = v[np.isfinite(v)]
                return float(np.max(v)) if v.size else float("nan")
            out.append(
                {
                    "scene": scene,
                    "arm": arm,
                    "n_units": len(sub),
                    "median_of_median_abs_um": med("median_abs_um"),
                    "median_p90_abs_um": med("p90_abs_um"),
                    "max_abs_um": mx("max_abs_um"),
                    "median_frac_half_row": med("frac_within_half_row"),
                    "median_frac_one_row": med("frac_within_one_row"),
                    "median_frac_channel_match": med("frac_channel_match"),
                    "worst_frac_one_row": float(
                        np.min([r["frac_within_one_row"] for r in sub])
                    ),
                    "worst_frac_channel_match": float(
                        np.min([r["frac_channel_match"] for r in sub])
                    ),
                    "max_longest_miss_one_row": mx("longest_miss_one_row"),
                    "n_eval_sum": int(sum(int(r["n_eval"]) for r in sub)),
                    "n_out_one_row_sum": int(
                        sum(int(r.get("n_out_one_row", 0)) for r in sub)
                    ),
                    "n_channel_miss_sum": int(
                        sum(int(r.get("n_channel_miss", 0)) for r in sub)
                    ),
                }
            )
    return out


def summarize_pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spike-weighted ceiling: sum n_out / sum n_eval over units of one scene."""
    out = []
    for scene in sorted({r["scene"] for r in rows}):
        arms = sorted({r["arm"] for r in rows if r["scene"] == scene})
        for arm in arms:
            sub = [r for r in rows if r["scene"] == scene and r["arm"] == arm]
            n_eval = int(sum(int(r["n_eval"]) for r in sub))
            n_out = int(sum(int(r.get("n_out_one_row", 0)) for r in sub))
            n_ch = int(sum(int(r.get("n_channel_miss", 0)) for r in sub))
            out.append(
                {
                    "scene": scene,
                    "arm": arm,
                    "n_units": len(sub),
                    "n_eval": n_eval,
                    "n_out_one_row": n_out,
                    "n_channel_miss": n_ch,
                    "frac_out_one_row": (
                        float(n_out / n_eval) if n_eval else float("nan")
                    ),
                    "frac_within_one_row": (
                        float(1.0 - n_out / n_eval) if n_eval else float("nan")
                    ),
                    "frac_channel_miss": (
                        float(n_ch / n_eval) if n_eval else float("nan")
                    ),
                    "worst_unit_frac_out_one_row": float(
                        np.max(
                            [
                                r.get(
                                    "frac_out_one_row",
                                    1.0 - r["frac_within_one_row"]
                                    if np.isfinite(r["frac_within_one_row"])
                                    else float("nan"),
                                )
                                for r in sub
                            ]
                        )
                    ),
                    "worst_unit": int(
                        max(sub, key=lambda r: r.get("frac_out_one_row", 0.0))["unit"]
                    ),
                }
            )
    return out


def _plot(traces: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    scenes = [s for s in DRIFT_SCENES if s in traces] or sorted(traces)
    n = len(scenes)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), sharex=True, squeeze=False)
    for ax, scene in zip(axes[:, 0], scenes):
        units = traces[scene]["units"]
        names = sorted(units, key=lambda u: int(u))
        if not names:
            continue
        # Prefer a unit Frozen cannot hold (largest frozen |err|).
        frozen_err = {
            str(r["unit"]): r["median_abs_um"]
            for r in rows
            if r["scene"] == scene and r["arm"] == "frozen_fit"
        }
        uid = max(names, key=lambda u: frozen_err.get(str(u), frozen_err.get(int(u), 0.0)))
        u = units[uid]
        t = np.asarray(u["centres_s"])
        y = np.asarray(u["com_um"])
        ax.plot(t, y, "k-", lw=1.4, label="GT COM")
        for arm, style in (("frozen_fit", "C0--"), ("hold", "C1-"), ("velocity", "C2:")):
            if arm in u:
                ax.plot(t, np.asarray(u[arm]), style, lw=1.0, label=arm)
        ax.set_ylabel("COM (um)")
        ax.set_title(f"{scene} u{uid}")
        ax.legend(fontsize=7, loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    try_savefig(out / "plots" / "com_tracking_traces.png", fig)
    plt.close(fig)

    recs = scenes
    arms = [
        "oracle_block",
        "oracle_loo",
        "stream_n10",
        "stream_dt1s",
        "hold",
        "frozen_fit",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(recs))
    summary = summarize(rows)
    for i, arm in enumerate(arms):
        vals = []
        for rec in recs:
            hit = [
                s["median_frac_one_row"]
                for s in summary
                if s["scene"] == rec and s["arm"] == arm
            ]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + i * 0.18, vals, 0.18, label=arm)
    ax.set_xticks(x + 0.27)
    ax.set_xticklabels(recs)
    ax.set_ylabel("median frac |err| ≤ 1 row (20 µm)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    try_savefig(out / "plots" / "com_tracking_in_row.png", fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=list(DRIFT_SCENES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    result = run_pilot(list(args.scenes), bool(args.quiet))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "com_tracking_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    summary = summarize(result["rows"])
    pooled = summarize_pooled(result["rows"])
    write_csv_rows(out / "com_tracking_summary.csv", summary)
    write_csv_rows(out / "com_tracking_pooled.csv", pooled)
    _plot(result["traces"], result["rows"], out)
    json_path = out / "com_tracking_summary.json"
    write_json(
        json_path,
        json_ready(
            {
                "note": (
                    "Causal predictors of next 5 s GT COM, plus an oracle "
                    "scatter ceiling: each spike vs the contemporaneous 5 s "
                    "GT-unit block median (oracle_block) or leave-one-out "
                    "median (oracle_loo). Frozen holds the t<30 s median. "
                    "Hold / EMA / velocity observe GT COM of past blocks only. "
                    "Not identity matching."
                ),
                "gt_used": True,
                "duration_s": FULL_S,
                "fit_end_s": FIT_END_S,
                "radii": "half_row=10um, one_row=20um, half_pitch=pitch/2",
                "summary": summary,
                "pooled": pooled,
                "source_sha256": source_hashes(SOURCE_PATHS),
            }
        ),
    )
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            rel_or_str(p): sha256_file(p)
            for p in (
                csv_path,
                json_path,
                out / "com_tracking_summary.csv",
                out / "com_tracking_pooled.csv",
            )
            if p.exists()
        },
    )
    print("\n=== ORACLE ceiling: spikes already outside contemporaneous unit COM ===")
    for s in pooled:
        if s["arm"] not in ("oracle_block", "oracle_loo"):
            continue
        print(
            f"{s['scene']:<20} {s['arm']:<12} "
            f"n={s['n_eval']:<7}  "
            f"out>20um={s['n_out_one_row']:<6} ({100.0 * s['frac_out_one_row']:5.2f}%)  "
            f"ch-miss={s['n_channel_miss']:<6} ({100.0 * s['frac_channel_miss']:5.2f}%)  "
            f"worst-u{s['worst_unit']} out={s['worst_unit_frac_out_one_row']:.3f}",
            flush=True,
        )
    print("\n=== median over units: frac within one row (20 um) ===")
    for s in summary:
        print(
            f"{s['scene']:<20} {s['arm']:<12} "
            f"med|err|={s['median_of_median_abs_um']:6.2f}um  "
            f"in-row={s['median_frac_one_row']:.3f}  "
            f"ch-match={s['median_frac_channel_match']:.3f}  "
            f"worst-row={s['worst_frac_one_row']:.3f}",
            flush=True,
        )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
