"""Part A scoring and Part B representation on locked 60 s development recordings.

    python -m Spatial.experiments.run_posneg_mvm_ablation --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.posneg_mvm_matching import (
    K_GRID,
    LAMBDA_AMP_GRID,
    POLARITY_RATIOS,
    _scalar,
    assignment_metrics,
    build_features,
    concat_posneg,
    gather_event_argmax_posneg,
    operation_count,
    p2p_from_posneg,
    same_home_pairs,
    score_kind,
    template_pair_distances,
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


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "posneg_mvm_ablation"
SOURCE_PATHS = (
    "Spatial/algorithms/posneg_mvm_matching.py",
    "Spatial/experiments/run_posneg_mvm_ablation.py",
    "Spatial/experiments/posneg_mvm_common.py",
    "Spatial/docs/posneg_mvm_shift_drift_plan.md",
    "Spatial/tests/test_posneg_mvm_shift_tracking.py",
)

A_ARMS = (
    ("A0_l1", "l1", "posneg"),
    ("A1_l2", "l2", "posneg"),
    ("A2_biased_dot", "biased_dot", "posneg"),
    ("A3_pure_dot", "pure_dot", "posneg"),
    ("A4_cosine", "cosine", "posneg"),
)
B_ARMS = (
    ("B0_p2p", "p2p", "l2", 1.0, 1.0),
    ("B1_posneg", "posneg", "l2", 1.0, 1.0),
    ("B2_pos", "pos", "l2", 1.0, 1.0),
    ("B3_neg", "neg", "l2", 1.0, 1.0),
    ("B4_1to1", "posneg_weighted", "weighted_l2", 1.0, 1.0),
    ("B4_1to2", "posneg_weighted", "weighted_l2", 1.0, 2.0),
    ("B4_2to1", "posneg_weighted", "weighted_l2", 2.0, 1.0),
    ("B5_rpn", "posneg_rpn", "l2", 1.0, 1.0),
    ("B6_dc", "posneg_dc", "l2", 1.0, 1.0),
    ("B7_rpn_dc", "posneg_rpn_dc", "l2", 1.0, 1.0),
)


def _pack_row(pack, k, part, arm, split_name, metrics, extra=None) -> dict[str, Any]:
    row = base_row(pack, k, part, arm)
    row["split"] = split_name
    row.update(flatten_metrics(metrics))
    if extra:
        row.update(extra)
    return row


def _event_weighted(rows: list[dict[str, Any]], field: str) -> float:
    n = 0
    acc = 0.0
    for r in rows:
        v = r.get(field)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        nn = int(r["n_test"])
        acc += float(v) * nn
        n += nn
    return acc / n if n else float("nan")


def a5_trigger(cal_rows: list[dict[str, Any]]) -> tuple[bool, str]:
    def pick(rec, arm):
        sub = [r for r in cal_rows if r["rec_key"] == rec and r["arm"] == arm and int(r["k"]) == 12]
        return sub[0] if sub else None

    drift_l2 = pick("hj_drift", "A1_l2")
    drift_cos = pick("hj_drift", "A4_cosine")
    m20_l2 = pick("mearec_20u", "A1_l2")
    m20_cos = pick("mearec_20u", "A4_cosine")
    if not all([drift_l2, drift_cos, m20_l2, m20_cos]):
        return False, "missing K=12 calibration rows for cosine vs L2"
    d_l2 = float(drift_l2["same_home_accuracy"])
    d_c = float(drift_cos["same_home_accuracy"])
    m_l2 = float(m20_l2["same_home_accuracy"])
    m_c = float(m20_cos["same_home_accuracy"])
    ok_drift = np.isfinite(d_c) and np.isfinite(d_l2) and d_c > d_l2
    ok_m20 = np.isfinite(m_c) and np.isfinite(m_l2) and (m_c >= m_l2 - 0.02)
    if ok_drift and ok_m20:
        return True, "cosine cal same-home > L2 on HJ drift and MEArec 20u drop <= 2 pp"
    return False, (
        f"cosine independent-shape trigger failed "
        f"(HJ drift cal home {d_c:.4f} vs L2 {d_l2:.4f}; "
        f"MEA20 {m_c:.4f} vs L2 {m_l2:.4f})"
    )


def confused_pairs(pack, fitted) -> list[dict[str, Any]]:
    model = fitted["model"]
    pairs = same_home_pairs(model["units"], model["homes"])
    rows = []
    idx = pack["test"]
    y = pack["labels"][idx]
    for i, j in pairs:
        mu_p_i, mu_n_i = model["mu_pos"][i], model["mu_neg"][i]
        mu_p_j, mu_n_j = model["mu_pos"][j], model["mu_neg"][j]
        d_p2p = template_pair_distances(p2p_from_posneg(mu_p_i, mu_n_i), p2p_from_posneg(mu_p_j, mu_n_j))
        d_pn = template_pair_distances(concat_posneg(mu_p_i, mu_n_i), concat_posneg(mu_p_j, mu_n_j))
        d_pos = template_pair_distances(mu_p_i, mu_p_j)
        d_neg = template_pair_distances(mu_n_i, mu_n_j)
        ui = _scalar(model["units"][i])
        uj = _scalar(model["units"][j])
        n_i = int(np.sum([_scalar(t) == ui for t in y]))
        n_j = int(np.sum([_scalar(t) == uj for t in y]))
        rows.append(
            {
                "rec_key": pack["rec_key"],
                "dataset": pack["dataset"],
                "k": int(fitted["k"]),
                "unit_a": ui,
                "unit_b": uj,
                "home": int(model["homes"][i]),
                "d_p2p": d_p2p,
                "d_posneg": d_pn,
                "d_pos": d_pos,
                "d_neg": d_neg,
                "ratio_p2p_over_posneg": d_p2p / d_pn if d_pn > 0 else float("nan"),
                "n_test_a": n_i,
                "n_test_b": n_j,
                "gt_used": True,
            }
        )
    return rows


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = load_pilot_packs(60.0, mearec_dir, quiet)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    l2_preds: dict[tuple[str, int, str], np.ndarray] = {}
    l2_test: dict[tuple[str, int], dict[str, Any]] = {}

    for pack in packs:
        for k in K_GRID:
            fitted = fit_k_model(pack, k, frame="candidate_home")
            for split in ("cal", "test"):
                for arm, kind, rep in A_ARMS:
                    met = eval_arm(pack, fitted, split, rep, kind)
                    extra = {"scoring": kind, "representation": rep, "role": "scoring", "frame": "candidate_home"}
                    if arm == "A1_l2":
                        l2_preds[(pack["rec_key"], k, split)] = met["pred"]
                        if split == "test":
                            l2_test[(pack["rec_key"], k)] = met
                    if arm == "A2_biased_dot":
                        l2p = l2_preds.get((pack["rec_key"], k, split))
                        if l2p is not None:
                            extra["l2_biased_agreement"] = float(
                                np.mean(
                                    [_scalar(a) == _scalar(b) for a, b in zip(l2p, met["pred"])]
                                )
                            )
                    row = _pack_row(pack, k, "A", arm, split, met, extra)
                    rows.append(row)
                    if split == "cal":
                        cal_rows.append(row)
                for arm, rep, kind, lp, ln in B_ARMS:
                    met = eval_arm(
                        pack, fitted, split, rep, kind, lam_pos=lp, lam_neg=ln
                    )
                    extra = {
                        "scoring": kind,
                        "representation": rep,
                        "role": "representation",
                        "lam_pos": lp,
                        "lam_neg": ln,
                        "frame": "candidate_home",
                    }
                    row = _pack_row(pack, k, "B", arm, split, met, extra)
                    rows.append(row)
            if k == 12:
                pair_rows.extend(confused_pairs(pack, fitted))
            if not quiet:
                tes = [r for r in rows if r["rec_key"] == pack["rec_key"] and r["k"] == k and r["split"] == "test" and r["arm"] == "A1_l2"][-1]
                print(
                    f"{pack['rec_key']:<12} K={k:<2} A1_l2 acc={tes['accuracy']:.4f} "
                    f"home={tes['same_home_accuracy'] if tes['same_home_accuracy'] == tes['same_home_accuracy'] else float('nan'):.4f}",
                    flush=True,
                )

        fitted12 = fit_k_model(pack, 12, frame="event_argmax")
        table = fitted12["table"]
        pos_e, neg_e = gather_event_argmax_posneg(
            pack["vmax"], pack["vmin"], table, pack["homes"]
        )
        model = fitted12["model"]
        mu = concat_posneg(model["mu_pos"], model["mu_neg"])
        for split in ("cal", "test"):
            idx = pack[split]
            x = concat_posneg(pos_e[idx], neg_e[idx])
            scores = score_kind(x, mu, "l2")
            met = assignment_metrics(
                scores, pack["labels"][idx], model["units"], model["homes"], "l2"
            )
            met.update(operation_count("l2", int(x.shape[1])))
            row = _pack_row(
                pack,
                12,
                "A",
                "A1_event_argmax",
                split,
                met,
                {
                    "scoring": "l2",
                    "representation": "posneg",
                    "role": "NEGATIVE_ABLATION",
                    "frame": "event_argmax",
                },
            )
            rows.append(row)

    run_a5, a5_reason = a5_trigger(cal_rows)
    skipped = []
    lam_amp = None
    if not run_a5:
        skipped.append({"arm": "A5_angular_radial", "reason": a5_reason})
    else:
        best = None
        for lam in LAMBDA_AMP_GRID:
            recs = []
            for pack in packs:
                fitted = fit_k_model(pack, 12)
                met = eval_arm(pack, fitted, "cal", "posneg", "angular_radial", lam_amp=lam)
                recs.append({"n_test": met["n_test"], "accuracy": met["accuracy"], "margin_home_p10": met["margin_home_p10"]})
            acc = _event_weighted(recs, "accuracy")
            p10 = float(np.nanmean([r["margin_home_p10"] for r in recs]))
            key = (-acc, -p10 if np.isfinite(p10) else 0.0, lam)
            if best is None or key < best[0]:
                best = (key, lam)
        lam_amp = float(best[1])
        for pack in packs:
            for k in K_GRID:
                fitted = fit_k_model(pack, k)
                for split in ("cal", "test"):
                    met = eval_arm(
                        pack, fitted, split, "posneg", "angular_radial", lam_amp=lam_amp
                    )
                    rows.append(
                        _pack_row(
                            pack,
                            k,
                            "A",
                            "A5_angular_radial",
                            split,
                            met,
                            {
                                "scoring": "angular_radial",
                                "representation": "posneg",
                                "role": "scoring",
                                "lam_amp": lam_amp,
                                "frame": "candidate_home",
                            },
                        )
                    )

    # Paired delta vs L2 on test A arms and vs B1 on test B arms.
    l2_lookup = {
        (r["rec_key"], int(r["k"])): r
        for r in rows
        if r["part"] == "A" and r["arm"] == "A1_l2" and r["split"] == "test"
    }
    b1_lookup = {
        (r["rec_key"], int(r["k"])): r
        for r in rows
        if r["part"] == "B" and r["arm"] == "B1_posneg" and r["split"] == "test"
    }
    for r in rows:
        if r["split"] != "test":
            continue
        if r["part"] == "A":
            ref = l2_lookup.get((r["rec_key"], int(r["k"])))
        else:
            ref = b1_lookup.get((r["rec_key"], int(r["k"])))
        if ref is None:
            continue
        r["delta_acc_vs_ref"] = float(r["accuracy"] - ref["accuracy"])
        ha = r.get("same_home_accuracy")
        hb = ref.get("same_home_accuracy")
        if isinstance(ha, float) and isinstance(hb, float) and np.isfinite(ha) and np.isfinite(hb):
            r["delta_home_acc_vs_ref"] = float(ha - hb)
        else:
            r["delta_home_acc_vs_ref"] = float("nan")
        mp = r.get("margin_home_p10")
        mq = ref.get("margin_home_p10")
        if isinstance(mp, float) and isinstance(mq, float) and np.isfinite(mp) and np.isfinite(mq):
            r["delta_home_p10_vs_ref"] = float(mp - mq)
        else:
            r["delta_home_p10_vs_ref"] = float("nan")

    stop = []
    for rec in ("hj_drift", "hj_static", "mearec_20u"):
        cos = next((r for r in rows if r["rec_key"] == rec and r["arm"] == "A4_cosine" and r["split"] == "test" and int(r["k"]) == 12), None)
        l2 = l2_lookup.get((rec, 12))
        if cos and l2 and np.isfinite(cos.get("same_home_accuracy", np.nan)) and np.isfinite(l2.get("same_home_accuracy", np.nan)):
            if rec != "hj_drift" and (cos["same_home_accuracy"] < l2["same_home_accuracy"] - 0.02):
                stop.append(f"cosine same-home drop on {rec}")
            if rec == "hj_drift" and cos["same_home_accuracy"] > l2["same_home_accuracy"]:
                if any("cosine same-home drop" in s for s in stop):
                    stop.append("stop-2: cosine helps drift, hurts dense/static")

    return {
        "rows": rows,
        "pairs": pair_rows,
        "skipped": skipped,
        "stop_conditions_fired": stop,
        "a5_lambda_amp": lam_amp,
        "a5_reason": a5_reason,
    }


def _plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = ["hj_drift", "hj_static", "mearec_10u", "mearec_20u"]
    test = [r for r in rows if r["split"] == "test" and int(r["k"]) == 12]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, part, arms in (
        (axes[0], "A", ["A0_l1", "A1_l2", "A2_biased_dot", "A3_pure_dot", "A4_cosine"]),
        (axes[1], "B", ["B0_p2p", "B1_posneg", "B2_pos", "B3_neg"]),
    ):
        x = np.arange(len(recs))
        width = 0.15
        for i, arm in enumerate(arms):
            vals = []
            for rec in recs:
                hit = [r["same_home_accuracy"] for r in test if r["rec_key"] == rec and r["arm"] == arm]
                v = hit[0] if hit else np.nan
                vals.append(v if isinstance(v, float) else np.nan)
            ax.bar(x + i * width, vals, width, label=arm)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(recs, rotation=20)
        ax.set_ylabel("same-home accuracy")
        ax.set_title(f"Part {part} K=12 test")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    try_savefig(out / "plots" / "same_home_acc_k12.png", fig)
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
    csv_path = out / "posneg_mvm_ablation_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    pair_path = out / "posneg_mvm_confused_pairs.csv"
    write_csv_rows(pair_path, result["pairs"] or [{"rec_key": "", "note": "no same-home pairs"}])
    _plots(result["rows"], out)
    summary = {
        "a5_lambda_amp": result["a5_lambda_amp"],
        "a5_reason": result["a5_reason"],
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
                        "part",
                        "arm",
                        "accuracy",
                        "same_home_accuracy",
                        "margin_home_p10",
                        "margin_home_p50",
                        "margin_home_p90",
                        "margin_home_ppos",
                        "worst_unit_recall",
                        "mean_unit_recall",
                        "n_home_rival",
                        "n_test",
                        "operation_form",
                        "mvm_writable",
                        "n_mac",
                        "n_abs",
                        "n_norm",
                        "delta_acc_vs_ref",
                        "delta_home_acc_vs_ref",
                        "l2_biased_agreement",
                        "role",
                        "frame",
                    )
                    if k in r
                }
            )
            for r in result["rows"]
            if r["split"] == "test"
        ],
        "source_sha256": source_hashes(SOURCE_PATHS),
    }
    json_path = out / "posneg_mvm_ablation_summary.json"
    write_json(json_path, json_ready(summary))
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            str(csv_path.relative_to(REPO_ROOT)): sha256_file(csv_path),
            str(json_path.relative_to(REPO_ROOT)): sha256_file(json_path),
            str(pair_path.relative_to(REPO_ROOT)): sha256_file(pair_path),
        },
    )
    print(f"Saved {csv_path}")
    print(f"A5: {result['a5_reason']}")
    for rec in ("hj_drift", "hj_static", "mearec_10u", "mearec_20u"):
        for arm in ("A1_l2", "A2_biased_dot", "A4_cosine", "B0_p2p", "B1_posneg"):
            hit = [
                r
                for r in result["rows"]
                if r["rec_key"] == rec and r["arm"] == arm and r["split"] == "test" and int(r["k"]) == 12
            ]
            if not hit:
                continue
            r = hit[0]
            home = r.get("same_home_accuracy")
            home_s = f"{home:.4f}" if isinstance(home, float) and np.isfinite(home) else "nan"
            print(f"  {rec:<12} {arm:<16} acc={r['accuracy']:.4f} home={home_s} Mp10={r.get('margin_home_p10')}")


if __name__ == "__main__":
    main()
