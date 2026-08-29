"""Part D causal tracking vs Frozen on locked 60 s development recordings.

    python -m Spatial.experiments.run_drift_tracking_compare --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.drift_tracking import (
    ETA_MU_SLOW,
    ETA_STATE,
    CausalTracker,
    coherence_vs_distance,
    gt_block_state,
    run_causal_stream,
    tile_ids,
    unit_xy,
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
from Spatial.experiments.run_drift_coherence import true_unit_gather
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "drift_tracking_compare"
SOURCE_PATHS = (
    "Spatial/algorithms/drift_tracking.py",
    "Spatial/algorithms/spatial_shift_matching.py",
    "Spatial/experiments/run_drift_tracking_compare.py",
    "Spatial/docs/posneg_mvm_shift_drift_plan.md",
    "Spatial/tests/test_posneg_mvm_shift_tracking.py",
)


def _conf_thresh(pack, fitted) -> float:
    met = eval_arm(pack, fitted, "cal", "posneg", "l2")
    p10 = met.get("margin_all_p10", 0.0)
    if not isinstance(p10, float) or not np.isfinite(p10):
        p10 = 0.0
    return float(max(0.0, p10))


def _make_tracker(mode, pack, fitted, thresh, groups, eta_mu=0.0, eta_gain=0.0) -> CausalTracker:
    model = fitted["model"]
    return CausalTracker(
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
        eta_mu=eta_mu,
        eta_gain=eta_gain,
        kind="l2",
        representation="posneg",
        confidence_threshold=thresh,
    )


def _stream_row(pack, fitted, arm, mode, thresh, groups, role, k_state=None, state_fitted=None, eta_mu=0.0, eta_gain=0.0) -> dict[str, Any]:
    tr = _make_tracker(mode, pack, fitted, thresh, groups, eta_mu=eta_mu, eta_gain=eta_gain)
    sp = sn = None
    if state_fitted is not None:
        tr.attach_state_encoder(
            state_fitted["model"]["mu_pos"],
            state_fitted["model"]["mu_neg"],
            state_fitted["rel_xy"],
            state_fitted["table"],
            state_fitted["model"]["homes"],
            pack["pitch"],
            pack["axis"],
        )
        idx = pack["test"]
        sp = state_fitted["pos_eu"][idx]
        sn = state_fitted["neg_eu"][idx]
    idx = pack["test"]
    met = run_causal_stream(
        tr,
        fitted["pos_eu"][idx],
        fitted["neg_eu"][idx],
        pack["labels"][idx],
        fitted["model"]["units"],
        state_pos_eu=sp,
        state_neg_eu=sn,
    )
    row = base_row(pack, fitted["k"], "D", arm)
    row["split"] = "test"
    row.update(flatten_metrics(met))
    ratio = met["mu_move_ratio_end"]
    finite = np.asarray(ratio, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    row.update(
        {
            "role": role,
            "scoring": "l2",
            "representation": "posneg",
            "k_state": int(k_state) if k_state is not None else int(fitted["k"]),
            "confidence_threshold": thresh,
            "mean_abs_delta_end": met["mean_abs_delta_end"],
            "mu_move_ratio_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "n_updates": met["n_updates"],
            "update_rate": met["update_rate"],
        }
    )
    return row, met


def _coherence_gate(pack, fitted12) -> dict[str, Any]:
    model = fitted12["model"]
    pos_t, neg_t = true_unit_gather(pack, fitted12)
    rec = gt_block_state(
        pos_t,
        neg_t,
        model["mu_pos"],
        model["mu_neg"],
        fitted12["rel_xy"],
        fitted12["table"] >= 0,
        pack["times"],
        pack["labels"],
        model["units"],
        pack["fs"],
        pack["pitch"],
        pack["axis"],
        unit_homes=model["homes"],
    )
    xy = unit_xy(pack["geom"], model["homes"])
    cd = coherence_vs_distance(rec["delta"], xy)
    mean_abs = float(np.nanmean(np.abs(rec["delta"])))
    g = rec["gain"]
    from Spatial.algorithms.drift_tracking import pairwise_corr

    gc = pairwise_corr(g)
    iu, ju = np.triu_indices(gc.shape[0], 1)
    return {
        "nearest_corr": cd["nearest_corr"],
        "slope": cd["corr_vs_dist_slope"],
        "mean_abs_delta": mean_abs,
        "mean_gain_corr": float(np.nanmean(gc[iu, ju])) if iu.size else float("nan"),
        "mean_abs_log_gain": float(np.nanmean(np.abs(np.log(np.maximum(g, 1e-12))))),
        "static_quiet": mean_abs < 0.15,
    }


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = load_pilot_packs(60.0, mearec_dir, quiet)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stop: list[str] = []
    coh: dict[str, dict[str, Any]] = {}
    fitted_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for pack in packs:
        for k in (7, 12, 16):
            fitted_cache[(pack["rec_key"], k)] = fit_k_model(pack, k)
        coh[pack["rec_key"]] = _coherence_gate(pack, fitted_cache[(pack["rec_key"], 12)])

    d5_ok = (
        np.isfinite(coh["hj_drift"]["nearest_corr"])
        and coh["hj_drift"]["nearest_corr"] > 0.3
        and np.isfinite(coh["hj_drift"]["slope"])
        and coh["hj_drift"]["slope"] < 0
        and coh["hj_static"]["static_quiet"]
        and coh["mearec_20u"]["static_quiet"]
    )
    gain_ok = (
        np.isfinite(coh["hj_drift"]["mean_gain_corr"])
        and coh["hj_drift"]["mean_gain_corr"] > 0.3
        and coh["hj_static"]["mean_abs_log_gain"] < 0.05
        and coh["mearec_20u"]["mean_abs_log_gain"] < 0.05
    )
    if not d5_ok:
        skipped.append({"arm": "D5_smooth_field", "reason": f"local coherence trigger failed: {coh}"})
    if not gain_ok:
        skipped.append({"arm": "shared_gain", "reason": f"gain coherence trigger failed: { {k: v['mean_gain_corr'] for k,v in coh.items()} }"})

    frozen_test: dict[tuple[str, int], dict[str, Any]] = {}
    for pack in packs:
        for k in (7, 12, 16):
            fitted = fitted_cache[(pack["rec_key"], k)]
            met = eval_arm(pack, fitted, "test", "posneg", "l2")
            row = base_row(pack, k, "D", "D0_frozen")
            row["split"] = "test"
            row.update(flatten_metrics(met))
            row.update(
                {
                    "role": "FROZEN",
                    "scoring": "l2",
                    "representation": "posneg",
                    "k_state": k,
                    "mean_abs_delta_end": 0.0,
                    "mu_move_ratio_mean": 0.0,
                    "n_updates": 0,
                    "update_rate": 0.0,
                }
            )
            rows.append(row)
            frozen_test[(pack["rec_key"], k)] = row

        fitted7 = fitted_cache[(pack["rec_key"], 7)]
        fitted12 = fitted_cache[(pack["rec_key"], 12)]
        fitted16 = fitted_cache[(pack["rec_key"], 16)]
        thresh = _conf_thresh(pack, fitted12)
        tiles12 = tile_ids(pack["geom"], fitted12["model"]["homes"], pack["pitch"])
        tiles7 = tile_ids(pack["geom"], fitted7["model"]["homes"], pack["pitch"])
        jobs = [
            ("D1_ema", "ema", fitted12, np.zeros(fitted12["model"]["homes"].size, dtype=np.int64), "EMA", 12, None, 0.0, 0.0),
            ("D2_oracle_shift", "oracle_shift", fitted12, np.zeros(fitted12["model"]["homes"].size, dtype=np.int64), "ORACLE_DIAGNOSTIC", 12, None, 0.0, 0.0),
            ("D3_global_k12", "global_shift", fitted12, np.zeros(fitted12["model"]["homes"].size, dtype=np.int64), "GLOBAL_SHARED", 12, None, 0.0, 0.0),
            ("D4_tile_k12", "tile_shift", fitted12, tiles12, "LOCAL_TILE", 12, None, 0.0, 0.0),
            ("D3_global_k7_st12", "global_shift", fitted7, np.zeros(fitted7["model"]["homes"].size, dtype=np.int64), "GLOBAL_SHARED", 12, fitted12, 0.0, 0.0),
            ("D4_tile_k7_st12", "tile_shift", fitted7, tiles7, "LOCAL_TILE", 12, fitted12, 0.0, 0.0),
            ("D3_global_k12_st16", "global_shift", fitted12, np.zeros(fitted12["model"]["homes"].size, dtype=np.int64), "GLOBAL_SHARED", 16, fitted16, 0.0, 0.0),
            ("D4_tile_k12_st16", "tile_shift", fitted12, tiles12, "LOCAL_TILE", 16, fitted16, 0.0, 0.0),
        ]
        if d5_ok:
            jobs.append(
                ("D5_smooth_field", "smooth_field", fitted12, tiles12, "SMOOTH_FIELD", 12, None, 0.0, 0.0)
            )
        streamed = {}
        for arm, mode, fitted, groups, role, kst, st, emu, eg in jobs:
            row, met = _stream_row(
                pack, fitted, arm, mode, thresh, groups, role, k_state=kst, state_fitted=st, eta_mu=emu, eta_gain=eg
            )
            streamed[arm] = row
            rows.append(row)
            if not quiet:
                print(
                    f"{pack['rec_key']:<12} {arm:<22} acc={row['accuracy']:.4f} "
                    f"home={row.get('same_home_accuracy')} d|={row.get('mean_abs_delta_end')}",
                    flush=True,
                )

        # D6 gated after seeing D3/D4 vs Frozen on this recording is not allowed
        # (would be test-look). Use HJ drift *calibration* stream for the trigger.
        # Frozen protocol: trigger uses shared tracker vs Frozen on HJ drift TEST
        # only after all D3/D4 are computed; D6 then would see test. That's a
        # test-set look. Freeze D6 trigger on CALIBRATION stream instead.
        # We record D6 as skipped here and compute the cal-based trigger below.

    # D6 trigger from already-computed TEST numbers would leak. Use cal stream
    # of HJ drift D3 vs D0. Cheap: stream cal for HJ drift only.
    hj = next(p for p in packs if p["rec_key"] == "hj_drift")
    f12 = fitted_cache[("hj_drift", 12)]
    thresh = _conf_thresh(hj, f12)
    tr = _make_tracker("global_shift", hj, f12, thresh, np.zeros(f12["model"]["homes"].size, dtype=np.int64))
    cal_idx = hj["cal"]
    cal_d3 = run_causal_stream(tr, f12["pos_eu"][cal_idx], f12["neg_eu"][cal_idx], hj["labels"][cal_idx], f12["model"]["units"])
    cal_d0 = eval_arm(hj, f12, "cal", "posneg", "l2")
    d6_ok = (
        np.isfinite(cal_d3["same_home_accuracy"])
        and np.isfinite(cal_d0["same_home_accuracy"])
        and cal_d3["same_home_accuracy"] + 1e-12 >= cal_d0["same_home_accuracy"]
    )
    static_ok = True
    for rec in ("hj_static", "mearec_20u"):
        d3 = next(r for r in rows if r["rec_key"] == rec and r["arm"] == "D3_global_k12")
        d0 = frozen_test[(rec, 12)]
        if np.isfinite(d3.get("same_home_accuracy", np.nan)) and np.isfinite(d0.get("same_home_accuracy", np.nan)):
            if d3["same_home_accuracy"] < d0["same_home_accuracy"] - 0.02:
                static_ok = False
    # static_ok uses test — that's a stop condition after the fact, not a license
    # to run D6. License D6 only from HJ drift calibration.
    if not d6_ok:
        skipped.append(
            {
                "arm": "D6_slow_mu",
                "reason": (
                    f"D3 cal same-home {cal_d3['same_home_accuracy']} "
                    f"< Frozen cal {cal_d0['same_home_accuracy']}"
                ),
            }
        )
    else:
        for pack in packs:
            fitted12 = fitted_cache[(pack["rec_key"], 12)]
            thresh = _conf_thresh(pack, fitted12)
            groups = np.zeros(fitted12["model"]["homes"].size, dtype=np.int64)
            row, _ = _stream_row(
                pack,
                fitted12,
                "D6_shared_slow_mu",
                "shared_shift_slow_mu",
                thresh,
                groups,
                "SHARED_SHIFT_SLOW_MU",
                k_state=12,
                eta_mu=ETA_MU_SLOW,
            )
            rows.append(row)

    if gain_ok:
        for pack in packs:
            fitted12 = fitted_cache[(pack["rec_key"], 12)]
            thresh = _conf_thresh(pack, fitted12)
            groups = np.zeros(fitted12["model"]["homes"].size, dtype=np.int64)
            row, _ = _stream_row(
                pack,
                fitted12,
                "D3_global_gain",
                "global_shift",
                thresh,
                groups,
                "GLOBAL_SHARED_GAIN",
                k_state=12,
                eta_gain=ETA_STATE,
            )
            rows.append(row)

    # Deltas vs Frozen
    for r in rows:
        if r["arm"] == "D0_frozen":
            continue
        ref = frozen_test.get((r["rec_key"], int(r["k"])))
        if ref is None:
            continue
        r["delta_acc_vs_frozen"] = float(r["accuracy"] - ref["accuracy"])
        ha, hb = r.get("same_home_accuracy"), ref.get("same_home_accuracy")
        if isinstance(ha, float) and isinstance(hb, float) and np.isfinite(ha) and np.isfinite(hb):
            r["delta_home_acc_vs_frozen"] = float(ha - hb)
        else:
            r["delta_home_acc_vs_frozen"] = float("nan")
        pa, pb = r.get("margin_home_p10"), ref.get("margin_home_p10")
        if isinstance(pa, float) and isinstance(pb, float) and np.isfinite(pa) and np.isfinite(pb):
            r["delta_home_p10_vs_frozen"] = float(pa - pb)
        else:
            r["delta_home_p10_vs_frozen"] = float("nan")
        r["false_motion_mean_abs_delta"] = (
            r.get("mean_abs_delta_end") if r["rec_key"] != "hj_drift" else float("nan")
        )

    for rec in ("hj_drift",):
        for arm in ("D3_global_k12", "D4_tile_k12"):
            hit = [r for r in rows if r["rec_key"] == rec and r["arm"] == arm]
            ref = frozen_test[(rec, 12)]
            if hit:
                dacc = hit[0].get("delta_home_acc_vs_frozen", hit[0]["accuracy"] - ref["accuracy"])
                if isinstance(dacc, float) and np.isfinite(dacc) and dacc < 0:
                    stop.append(f"{arm} does not beat Frozen same-home on HJ drift")
    for rec in ("hj_static", "mearec_20u"):
        for arm in ("D3_global_k12", "D4_tile_k12"):
            hit = [r for r in rows if r["rec_key"] == rec and r["arm"] == arm]
            if hit and isinstance(hit[0].get("delta_home_acc_vs_frozen"), float):
                if hit[0]["delta_home_acc_vs_frozen"] < -0.02:
                    stop.append(f"{arm} hurts {rec} same-home")
        if not coh[rec]["static_quiet"]:
            stop.append(f"state estimator false motion on {rec}: |d|={coh[rec]['mean_abs_delta']:.3f}")

    return {
        "rows": rows,
        "skipped": skipped,
        "stop_conditions_fired": sorted(set(stop)),
        "coherence": coh,
        "d5_ok": d5_ok,
        "d6_ok": d6_ok,
        "gain_ok": gain_ok,
    }


def _plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = ["hj_drift", "hj_static", "mearec_10u", "mearec_20u"]
    arms = ["D0_frozen", "D1_ema", "D2_oracle_shift", "D3_global_k12", "D4_tile_k12"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(recs))
    for i, arm in enumerate(arms):
        vals = []
        for rec in recs:
            hit = [r.get("delta_home_acc_vs_frozen", 0.0) if arm != "D0_frozen" else 0.0 for r in rows if r["rec_key"] == rec and r["arm"] == arm]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + i * 0.15, vals, 0.15, label=arm)
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(recs, rotation=20)
    ax.set_ylabel("Δ same-home acc vs Frozen")
    ax.legend(fontsize=7)
    fig.tight_layout()
    try_savefig(out / "plots" / "tracking_delta_home_acc.png", fig)
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
    csv_path = out / "drift_tracking_compare_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    _plots(result["rows"], out)
    keys = (
        "rec_key",
        "k",
        "k_state",
        "arm",
        "accuracy",
        "same_home_accuracy",
        "margin_home_p10",
        "margin_home_p50",
        "margin_home_p90",
        "margin_home_ppos",
        "worst_unit_recall",
        "delta_acc_vs_frozen",
        "delta_home_acc_vs_frozen",
        "delta_home_p10_vs_frozen",
        "mean_abs_delta_end",
        "false_motion_mean_abs_delta",
        "mu_move_ratio_mean",
        "n_updates",
        "role",
    )
    summary = {
        "skipped": json_ready(result["skipped"]),
        "stop_conditions_fired": result["stop_conditions_fired"],
        "d5_ok": result["d5_ok"],
        "d6_ok": result["d6_ok"],
        "gain_ok": result["gain_ok"],
        "coherence": json_ready(result["coherence"]),
        "recordings": sorted({r["rec_key"] for r in result["rows"]}),
        "test_rows": [json_ready({k: r[k] for k in keys if k in r}) for r in result["rows"]],
        "source_sha256": source_hashes(SOURCE_PATHS),
    }
    json_path = out / "drift_tracking_compare_summary.json"
    write_json(json_path, json_ready(summary))
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            str(csv_path.relative_to(REPO_ROOT)): sha256_file(csv_path),
            str(json_path.relative_to(REPO_ROOT)): sha256_file(json_path),
        },
    )
    print(f"Saved {csv_path}")
    print("stops:", result["stop_conditions_fired"] or "none")
    print("skipped:", [s["arm"] for s in result["skipped"]])


if __name__ == "__main__":
    main()
