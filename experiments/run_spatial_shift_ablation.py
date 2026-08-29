"""Part C free-shift diagnostic on locked 60 s development recordings.

    python -m Spatial.experiments.run_spatial_shift_ablation --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.posneg_mvm_matching import (
    K_GRID,
    as_distance,
    assignment_metrics,
    operation_count,
    score_candidate_home,
)
from Spatial.algorithms.spatial_shift_matching import (
    SHIFT_DISCRETE,
    SHIFT_DISCRETE_WIDE,
    SHIFT_FRACTIONAL,
    chosen_shift_stats,
    fixed_vs_shifted_true_wrong,
    precompute_shift_mats,
)
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


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "spatial_shift_ablation"
SOURCE_PATHS = (
    "Spatial/algorithms/spatial_shift_matching.py",
    "Spatial/algorithms/posneg_mvm_matching.py",
    "Spatial/experiments/run_spatial_shift_ablation.py",
    "Spatial/docs/posneg_mvm_shift_drift_plan.md",
    "Spatial/tests/test_posneg_mvm_shift_tracking.py",
)


def _row(pack, k, arm, split, metrics, extra) -> dict[str, Any]:
    row = base_row(pack, k, "C", arm)
    row["split"] = split
    row.update(flatten_metrics(metrics))
    row.update(extra)
    return row


def _shift_eval(pack, fitted, split, deltas, kind, arm) -> dict[str, Any]:
    idx = pack[split]
    model = fitted["model"]
    pos = fitted["pos_eu"][idx]
    neg = fitted["neg_eu"][idx]
    mats = precompute_shift_mats(
        fitted["rel_xy"], fitted["table"], deltas, pack["pitch"], pack["axis"]
    )
    fixed_scores = score_candidate_home(
        pos,
        neg,
        model["mu_pos"],
        model["mu_neg"],
        fitted["rel_xy"],
        model["homes"],
        "posneg",
        kind,
    )
    fixed_dist = as_distance(fixed_scores, kind)
    leak = fixed_vs_shifted_true_wrong(
        pos,
        neg,
        model["mu_pos"],
        model["mu_neg"],
        pack["labels"][idx],
        model["units"],
        model["homes"],
        fitted["rel_xy"],
        mats,
        "posneg",
        kind,
        fixed_dist,
    )
    metrics = assignment_metrics(
        leak["score"], pack["labels"][idx], model["units"], model["homes"], kind
    )
    metrics.update(operation_count(kind, int(2 * fitted["k"])))
    have = leak["have_home_rival"]
    def _mean(a, mask=None):
        x = np.asarray(a, dtype=np.float64)
        if mask is not None:
            x = x[mask]
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")
    metrics["i_true_mean"] = _mean(leak["i_true"])
    metrics["i_wrong_mean"] = _mean(leak["i_wrong"], have)
    metrics["l_flex_mean"] = _mean(leak["l_flex"], have)
    lf = leak["l_flex"][have]
    lf = lf[np.isfinite(lf)]
    metrics["l_flex_p50"] = float(np.median(lf)) if lf.size else float("nan")
    st_true = chosen_shift_stats(leak["true_shift_index"], deltas, np.isfinite(leak["true_shift_index"]))
    metrics["chosen_shift_mean_abs"] = st_true["mean_abs"]
    metrics["chosen_shift_frac_nonzero"] = st_true["frac_nonzero"]
    if pack["rec_key"].endswith("static") or pack["rec_key"].startswith("mearec"):
        metrics["static_false_nonzero_shift"] = st_true["frac_nonzero"]
    else:
        metrics["static_false_nonzero_shift"] = float("nan")
    # C0 margin for delta
    c0 = assignment_metrics(fixed_scores, pack["labels"][idx], model["units"], model["homes"], kind)
    metrics["c0_same_home_accuracy"] = c0["same_home_accuracy"]
    metrics["c0_margin_home_p10"] = c0["margin_home_p10"]
    metrics["delta_home_acc_vs_c0"] = (
        float(metrics["same_home_accuracy"] - c0["same_home_accuracy"])
        if np.isfinite(metrics["same_home_accuracy"]) and np.isfinite(c0["same_home_accuracy"])
        else float("nan")
    )
    metrics["delta_home_p10_vs_c0"] = (
        float(metrics["margin_home_p10"] - c0["margin_home_p10"])
        if np.isfinite(metrics["margin_home_p10"]) and np.isfinite(c0["margin_home_p10"])
        else float("nan")
    )
    extra = {
        "scoring": kind,
        "representation": "posneg",
        "role": "FREE_SHIFT_DIAGNOSTIC",
        "shift_set": ",".join(str(d) for d in deltas),
        "n_shifts": len(tuple(deltas)),
    }
    return _row(pack, fitted["k"], arm, split, metrics, extra), leak, deltas


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = load_pilot_packs(60.0, mearec_dir, quiet)
    rows: list[dict[str, Any]] = []
    cal_lflex = []
    stop = []
    skipped = []
    for pack in packs:
        for k in K_GRID:
            fitted = fit_k_model(pack, k)
            for split in ("cal", "test"):
                met = eval_arm(pack, fitted, split, "posneg", "l2")
                rows.append(
                    _row(
                        pack,
                        k,
                        "C0_fixed",
                        split,
                        met,
                        {
                            "scoring": "l2",
                            "representation": "posneg",
                            "role": "fixed",
                            "i_true_mean": 0.0,
                            "i_wrong_mean": 0.0,
                            "l_flex_mean": 0.0,
                        },
                    )
                )
                for kind, tag in (("biased_dot", "bd"), ("cosine", "cos")):
                    row, leak, _ = _shift_eval(
                        pack, fitted, split, SHIFT_DISCRETE, kind, f"C1_discrete_{tag}"
                    )
                    rows.append(row)
                    if split == "cal" and kind == "biased_dot":
                        cal_lflex.append(row)
                    row2, _, _ = _shift_eval(
                        pack, fitted, split, SHIFT_FRACTIONAL, kind, f"C2_frac_{tag}"
                    )
                    rows.append(row2)
            if not quiet:
                r = [x for x in rows if x["rec_key"] == pack["rec_key"] and x["k"] == k and x["split"] == "test" and x["arm"] == "C1_discrete_bd"][-1]
                print(
                    f"{pack['rec_key']:<12} K={k} C1_bd home={r.get('same_home_accuracy')} "
                    f"Lflex={r.get('l_flex_mean')}",
                    flush=True,
                )

    # Extra ±2 only if C1 cal L_flex < 0 and home P10 improved vs C0, globally.
    c1_cal = [r for r in rows if r["arm"] == "C1_discrete_bd" and r["split"] == "cal" and int(r["k"]) == 12]
    c0_cal = [r for r in rows if r["arm"] == "C0_fixed" and r["split"] == "cal" and int(r["k"]) == 12]
    lflex_ok = all(isinstance(r.get("l_flex_mean"), float) and r["l_flex_mean"] < 0 for r in c1_cal if np.isfinite(r.get("l_flex_mean", np.nan)))
    p10_ok = True
    for rec in {r["rec_key"] for r in c1_cal}:
        a = next(r for r in c1_cal if r["rec_key"] == rec)
        b = next(r for r in c0_cal if r["rec_key"] == rec)
        if not (np.isfinite(a.get("margin_home_p10", np.nan)) and np.isfinite(b.get("margin_home_p10", np.nan)) and a["margin_home_p10"] > b["margin_home_p10"]):
            # MEArec 10u has NaN margin: ignore
            if a["rec_key"] == "mearec_10u":
                continue
            p10_ok = False
    if lflex_ok and p10_ok and c1_cal:
        for pack in packs:
            fitted = fit_k_model(pack, 12)
            for split in ("cal", "test"):
                row, _, _ = _shift_eval(
                    pack, fitted, split, SHIFT_DISCRETE_WIDE, "biased_dot", "C1_wide_bd"
                )
                rows.append(row)
    else:
        skipped.append(
            {
                "arm": "C1_wide_pm2",
                "reason": f"C1 cal L_flex<0={lflex_ok} and P10 improve={p10_ok}",
            }
        )

    for r in rows:
        if r["split"] == "test" and r["arm"].startswith("C1") and r.get("scoring") == "biased_dot":
            lf = r.get("l_flex_mean")
            if isinstance(lf, float) and np.isfinite(lf) and lf >= 0:
                stop.append(f"L_flex>=0 on {r['rec_key']} K={r['k']} {r['arm']}")
            dmp = r.get("delta_home_p10_vs_c0")
            if isinstance(dmp, float) and np.isfinite(dmp) and dmp < 0:
                stop.append(f"shift margin drop on {r['rec_key']} K={r['k']}")
    return {"rows": rows, "skipped": skipped, "stop_conditions_fired": sorted(set(stop))}


def _plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = ["hj_drift", "hj_static", "mearec_10u", "mearec_20u"]
    test = [r for r in rows if r["split"] == "test" and int(r["k"]) == 12]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(recs))
    for i, arm in enumerate(("C0_fixed", "C1_discrete_bd", "C2_frac_bd", "C1_discrete_cos")):
        vals = []
        for rec in recs:
            hit = [r.get("l_flex_mean", 0.0) if arm != "C0_fixed" else 0.0 for r in test if r["rec_key"] == rec and r["arm"] == arm]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + i * 0.18, vals, 0.18, label=arm)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x + 0.27)
    ax.set_xticklabels(recs, rotation=20)
    ax.set_ylabel("L_flex mean")
    ax.set_title("Flexibility leakage K=12 (positive = impostor helped more)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    try_savefig(out / "plots" / "lflex_k12.png", fig)
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
    csv_path = out / "spatial_shift_ablation_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    _plots(result["rows"], out)
    summary = {
        "skipped": result["skipped"],
        "stop_conditions_fired": result["stop_conditions_fired"],
        "n_rows": len(result["rows"]),
        "recordings": sorted({r["rec_key"] for r in result["rows"]}),
        "test_rows": [
            json_ready(
                {
                    k: r[k]
                    for k in (
                        "rec_key",
                        "k",
                        "arm",
                        "accuracy",
                        "same_home_accuracy",
                        "margin_home_p10",
                        "margin_home_p50",
                        "margin_home_p90",
                        "margin_home_ppos",
                        "worst_unit_recall",
                        "i_true_mean",
                        "i_wrong_mean",
                        "l_flex_mean",
                        "chosen_shift_mean_abs",
                        "chosen_shift_frac_nonzero",
                        "static_false_nonzero_shift",
                        "delta_home_acc_vs_c0",
                        "delta_home_p10_vs_c0",
                        "role",
                        "scoring",
                    )
                    if k in r
                }
            )
            for r in result["rows"]
            if r["split"] == "test"
        ],
        "source_sha256": source_hashes(SOURCE_PATHS),
    }
    json_path = out / "spatial_shift_ablation_summary.json"
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


if __name__ == "__main__":
    main()
