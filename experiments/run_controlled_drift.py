"""Controlled synthetic drift on static/no-drift locked recordings.

    python -m Spatial.experiments.run_controlled_drift --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.drift_tracking import (
    ETA_STATE,
    CausalTracker,
    apply_gain_ramp,
    apply_gain_step,
    apply_spatial_ramp,
    apply_two_tile_opposite,
    run_causal_stream,
    tile_ids,
)
from Spatial.algorithms.spatial_shift_matching import SHIFT_FRACTIONAL
from Spatial.experiments.posneg_mvm_common import (
    REPO_ROOT,
    base_row,
    eval_arm,
    fit_k_model,
    flatten_metrics,
    json_ready,
    load_pilot_packs,
    source_hashes,
    try_savefig,
    write_csv_rows,
    write_sha256_manifest,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "controlled_drift"
SOURCE_PATHS = (
    "Spatial/algorithms/drift_tracking.py",
    "Spatial/experiments/run_controlled_drift.py",
    "Spatial/docs/posneg_mvm_shift_drift_plan.md",
    "Spatial/tests/test_posneg_mvm_shift_tracking.py",
)

STATIC_KEYS = ("hj_static", "mearec_10u", "mearec_20u")


def _perturb(pack: dict[str, Any], kind: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = pack["test"]
    vmax = pack["vmax"][idx].copy()
    vmin = pack["vmin"][idx].copy()
    n = int(vmax.shape[0])
    meta: dict[str, Any] = {"perturbation": kind}
    if kind == "gain_ramp":
        vmax, vmin = apply_gain_ramp(vmax, vmin, 1.0, 1.5)
        meta["g_gt"] = np.linspace(1.0, 1.5, n)
        meta["d_gt"] = np.zeros(n)
    elif kind == "gain_step":
        vmax, vmin = apply_gain_step(vmax, vmin, 1.0, 1.5)
        g = np.ones(n)
        g[n // 2 :] = 1.5
        meta["g_gt"] = g
        meta["d_gt"] = np.zeros(n)
    elif kind == "spatial_ramp":
        vmax, vmin, d = apply_spatial_ramp(
            vmax, vmin, pack["geom"], 0.0, 0.5, pack["pitch"], pack["axis"]
        )
        meta["d_gt"] = d
        meta["g_gt"] = np.ones(n)
    elif kind == "two_tile":
        vmax, vmin, d_lo, d_hi = apply_two_tile_opposite(
            vmax, vmin, pack["geom"], 0.5, pack["pitch"], pack["axis"]
        )
        meta["d_gt_low"] = d_lo
        meta["d_gt_high"] = d_hi
        meta["d_gt"] = 0.5 * (d_hi - d_lo)
        meta["g_gt"] = np.ones(n)
    else:
        raise ValueError(kind)
    return vmax, vmin, meta


def _refit_test_features(pack, fitted, vmax_t, vmin_t):
    from Spatial.algorithms.posneg_mvm_matching import gather_unit_posneg

    pos, neg = gather_unit_posneg(vmax_t, vmin_t, fitted["table"], fitted["model"]["homes"])
    return pos, neg


def settling_time(delta_trace: np.ndarray, d_gt: np.ndarray, fs_events_per_s: float, tol: float = 0.1) -> float:
    err = np.abs(np.asarray(delta_trace)[:, 0] - np.asarray(d_gt))
    ok = err < tol
    # first index after which 3 consecutive events stay inside tol
    for i in range(0, ok.size - 2):
        if np.all(ok[i : i + 3]):
            return float(i / max(fs_events_per_s, 1e-9))
    return float("nan")


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = [p for p in load_pilot_packs(60.0, mearec_dir, quiet) if p["rec_key"] in STATIC_KEYS]
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for pack in packs:
        fitted = fit_k_model(pack, 12)
        model = fitted["model"]
        groups0 = np.zeros(model["homes"].size, dtype=np.int64)
        tiles = tile_ids(pack["geom"], model["homes"], pack["pitch"])
        frozen = eval_arm(pack, fitted, "test", "posneg", "l2")
        cal = eval_arm(pack, fitted, "cal", "posneg", "l2")
        thresh = max(0.0, float(cal["margin_all_p10"]) if np.isfinite(cal["margin_all_p10"]) else 0.0)
        dt = float(np.mean(np.diff(pack["times"][pack["test"]])) / pack["fs"]) if pack["test"].size > 1 else 1.0
        ev_per_s = 1.0 / max(dt, 1e-9)
        for pert in ("gain_ramp", "gain_step", "spatial_ramp", "two_tile"):
            vmax_t, vmin_t, meta = _perturb(pack, pert)
            pos, neg = _refit_test_features(pack, fitted, vmax_t, vmin_t)
            y = pack["labels"][pack["test"]]
            jobs = [
                ("D0_frozen", "frozen", groups0, "FROZEN"),
                ("D1_ema", "ema", groups0, "EMA"),
                ("D2_oracle_shift", "oracle_shift", groups0, "ORACLE_DIAGNOSTIC"),
                ("D3_global", "global_shift", groups0, "GLOBAL_SHARED"),
                ("D4_tile", "tile_shift", tiles, "LOCAL_TILE"),
            ]
            for arm, mode, groups, role in jobs:
                tr = CausalTracker(
                    mode,
                    model["mu_pos"],
                    model["mu_neg"],
                    model["homes"],
                    fitted["rel_xy"],
                    fitted["table"],
                    pack["geom"],
                    pack["pitch"],
                    pack["axis"],
                    groups,
                    deltas=SHIFT_FRACTIONAL,
                    eta_state=ETA_STATE,
                    kind="l2",
                    representation="posneg",
                    confidence_threshold=thresh,
                )
                met = run_causal_stream(tr, pos, neg, y, model["units"])
                row = base_row(pack, 12, "controlled", arm)
                row["split"] = "test"
                row.update(flatten_metrics(met))
                d_gt = np.asarray(meta["d_gt"], dtype=np.float64)
                d_hat = met["delta_trace"]
                if d_hat.ndim == 1:
                    d_hat = d_hat[:, None]
                # compare group-0 to global GT; for two-tile compare mean |err| over groups
                if pert == "two_tile" and d_hat.shape[1] > 1:
                    err = float(np.mean(np.abs(d_hat[:, 0] - meta["d_gt_low"]))) + float(
                        np.mean(np.abs(d_hat[:, min(1, d_hat.shape[1] - 1)] - meta["d_gt_high"]))
                    )
                    err *= 0.5
                else:
                    err = float(np.mean(np.abs(d_hat[:, 0] - d_gt)))
                settle = settling_time(d_hat, d_gt, ev_per_s) if pert == "spatial_ramp" else float("nan")
                unpert = slice(0, max(int(0.1 * d_gt.size), 1))
                false_m = float(np.mean(np.abs(d_hat[unpert, 0])))
                ratio = met["mu_move_ratio_end"]
                finite = np.asarray(ratio, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                row.update(
                    {
                        "perturbation": pert,
                        "role": role,
                        "scoring": "l2",
                        "k_state": 12,
                        "state_err_mean": err,
                        "settling_time_s": settle,
                        "false_motion_unperturbed_prefix": false_m,
                        "mu_move_ratio_mean": float(np.mean(finite)) if finite.size else float("nan"),
                        "delta_acc_vs_frozen": float(met["accuracy"] - frozen["accuracy"]),
                        "delta_home_acc_vs_frozen": (
                            float(met["same_home_accuracy"] - frozen["same_home_accuracy"])
                            if np.isfinite(met["same_home_accuracy"])
                            and np.isfinite(frozen["same_home_accuracy"])
                            else float("nan")
                        ),
                        "delta_home_p10_vs_frozen": (
                            float(met["margin_home_p10"] - frozen["margin_home_p10"])
                            if np.isfinite(met["margin_home_p10"])
                            and np.isfinite(frozen["margin_home_p10"])
                            else float("nan")
                        ),
                        "n_updates": met["n_updates"],
                        "mean_abs_delta_end": met["mean_abs_delta_end"],
                    }
                )
                rows.append(row)
                traces.append(
                    {
                        "rec_key": pack["rec_key"],
                        "arm": arm,
                        "perturbation": pert,
                        "d_hat": d_hat[:, 0],
                        "d_gt": d_gt,
                    }
                )
                if not quiet:
                    print(
                        f"{pack['rec_key']:<12} {pert:<12} {arm:<20} "
                        f"acc={met['accuracy']:.4f} err={err:.3f} mu={row['mu_move_ratio_mean']}",
                        flush=True,
                    )
    return {"rows": rows, "traces": traces}


def _plots(traces: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = [t for t in traces if t["perturbation"] == "spatial_ramp" and t["arm"] in ("D0_frozen", "D3_global", "D4_tile")]
    rec_keys = sorted({t["rec_key"] for t in recs})
    if not rec_keys:
        return
    fig, axes = plt.subplots(1, len(rec_keys), figsize=(4 * len(rec_keys), 3.2), squeeze=False)
    for ax, rec in zip(axes[0], rec_keys):
        for t in recs:
            if t["rec_key"] != rec:
                continue
            ax.plot(t["d_hat"], label=t["arm"], lw=1.2)
        gt = next(t["d_gt"] for t in recs if t["rec_key"] == rec)
        ax.plot(gt, "k--", label="GT", lw=1.0)
        ax.set_title(rec)
        ax.set_ylabel("delta (pitch)")
        ax.legend(fontsize=7)
    fig.suptitle("Controlled spatial ramp: state traces")
    fig.tight_layout()
    try_savefig(out / "plots" / "controlled_spatial_ramp_traces.png", fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    # mu movement is in rows; skip if empty
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--mearec-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s")
    result = run_pilot(Path(args.mearec_dir), bool(args.quiet))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "controlled_drift_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    _plots(result["traces"], out)
    keys = (
        "rec_key",
        "arm",
        "perturbation",
        "accuracy",
        "same_home_accuracy",
        "margin_home_p10",
        "margin_home_p50",
        "margin_home_p90",
        "margin_home_ppos",
        "worst_unit_recall",
        "state_err_mean",
        "settling_time_s",
        "false_motion_unperturbed_prefix",
        "delta_acc_vs_frozen",
        "delta_home_acc_vs_frozen",
        "delta_home_p10_vs_frozen",
        "mu_move_ratio_mean",
        "mean_abs_delta_end",
        "role",
    )
    summary = {
        "recordings": sorted({r["rec_key"] for r in result["rows"]}),
        "note": "static/no-drift recordings only; test-split perturbation; GT events",
        "test_rows": [json_ready({k: r[k] for k in keys if k in r}) for r in result["rows"]],
        "source_sha256": source_hashes(SOURCE_PATHS),
    }
    json_path = out / "controlled_drift_summary.json"
    write_json(json_path, json_ready(summary))
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            str(csv_path.relative_to(REPO_ROOT)): sha256_file(csv_path),
            str(json_path.relative_to(REPO_ROOT)): sha256_file(json_path),
        },
    )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
