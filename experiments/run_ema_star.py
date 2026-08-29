"""EMA* and Oracle-EMA mean-template baseline on locked 60 s recordings.

    python -m Spatial.experiments.run_ema_star --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.drift_tracking import (
    ETA_EMA_GRID,
    CausalTracker,
    apply_gain_ramp,
    apply_spatial_ramp,
    run_causal_stream,
)
from Spatial.algorithms.posneg_mvm_matching import gather_unit_posneg
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
    rel_or_str,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "ema_star"
SOURCE_PATHS = (
    "Spatial/algorithms/drift_tracking.py",
    "Spatial/experiments/run_ema_star.py",
    "Spatial/docs/ema_star_identity_convergence.md",
    "Spatial/tests/test_ema_star.py",
)
K_ID = 12
STATIC_KEYS = ("hj_static", "mearec_10u", "mearec_20u")
CONTROLLED = ("spatial_ramp", "gain_ramp", "shift_gain")


def _conf_thresh(pack, fitted) -> float:
    met = eval_arm(pack, fitted, "cal", "posneg", "l2")
    p10 = met.get("margin_all_p10", 0.0)
    if not isinstance(p10, float) or not np.isfinite(p10):
        p10 = 0.0
    return float(max(0.0, p10))


def _make_ema(pack, fitted, thresh, eta: float, owner: str) -> CausalTracker:
    model = fitted["model"]
    mode = "oracle_ema" if owner == "gt" else "ema"
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
        np.zeros(model["homes"].size, dtype=np.int64),
        deltas=SHIFT_FRACTIONAL,
        eta_ema=float(eta),
        kind="l2",
        representation="posneg",
        confidence_threshold=thresh,
        update_owner=owner,
    )


def _stream(pack, fitted, split, thresh, eta, owner, pos=None, neg=None) -> dict[str, Any]:
    idx = pack[split]
    tr = _make_ema(pack, fitted, thresh, eta, owner)
    p = fitted["pos_eu"][idx] if pos is None else pos
    n = fitted["neg_eu"][idx] if neg is None else neg
    met = run_causal_stream(tr, p, n, pack["labels"][idx], fitted["model"]["units"])
    ratio = np.asarray(met["mu_move_ratio_end"], dtype=np.float64)
    finite = ratio[np.isfinite(ratio)]
    move = met["mu_move_trace"][-1] if met["mu_move_trace"].size else np.zeros(0)
    met["mu_move_ratio_mean"] = float(np.mean(finite)) if finite.size else float("nan")
    met["mu_move_l2_mean"] = float(np.mean(move)) if np.size(move) else 0.0
    met["eta"] = float(eta)
    met["update_owner"] = owner
    return met


def _pack_row(pack, arm, split, met, extra) -> dict[str, Any]:
    row = base_row(pack, K_ID, "ema_star", arm)
    row["split"] = split
    row.update(flatten_metrics(met))
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


def _select_eta(cal_by_eta: dict[float, list[dict[str, Any]]]) -> float:
    best = None
    for eta, recs in cal_by_eta.items():
        acc = _event_weighted(recs, "accuracy")
        home = float(np.nanmean([r.get("same_home_accuracy", np.nan) for r in recs]))
        worst = min(float(r["worst_unit_recall"]) for r in recs)
        key = (
            -acc if np.isfinite(acc) else 0.0,
            -home if np.isfinite(home) else 0.0,
            -worst if np.isfinite(worst) else 0.0,
            float(eta),
        )
        if best is None or key < best[0]:
            best = (key, eta)
    assert best is not None
    return float(best[1])


def _perturb(pack, fitted, kind: str):
    idx = pack["test"]
    vmax = pack["vmax"][idx].copy()
    vmin = pack["vmin"][idx].copy()
    meta: dict[str, Any] = {"perturbation": kind}
    if kind == "gain_ramp":
        vmax, vmin = apply_gain_ramp(vmax, vmin, 1.0, 1.5)
        meta["g_gt"] = np.linspace(1.0, 1.5, vmax.shape[0])
        meta["d_gt"] = np.zeros(vmax.shape[0])
    elif kind == "spatial_ramp":
        vmax, vmin, d = apply_spatial_ramp(
            vmax, vmin, pack["geom"], 0.0, 0.5, pack["pitch"], pack["axis"]
        )
        meta["d_gt"] = d
        meta["g_gt"] = np.ones(vmax.shape[0])
    elif kind == "shift_gain":
        vmax, vmin, d = apply_spatial_ramp(
            vmax, vmin, pack["geom"], 0.0, 0.5, pack["pitch"], pack["axis"]
        )
        vmax, vmin = apply_gain_ramp(vmax, vmin, 1.0, 1.5)
        meta["d_gt"] = d
        meta["g_gt"] = np.linspace(1.0, 1.5, vmax.shape[0])
    else:
        raise ValueError(kind)
    pos, neg = gather_unit_posneg(vmax, vmin, fitted["table"], fitted["model"]["homes"])
    return pos, neg, meta


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = load_pilot_packs(60.0, mearec_dir, quiet)
    fitted: dict[str, dict[str, Any]] = {}
    thresh: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    frozen_test: dict[str, dict[str, Any]] = {}

    for pack in packs:
        fit = fit_k_model(pack, K_ID)
        fitted[pack["rec_key"]] = fit
        thresh[pack["rec_key"]] = _conf_thresh(pack, fit)
        for split in ("cal", "test"):
            met = eval_arm(pack, fit, split, "posneg", "l2")
            row = _pack_row(
                pack,
                "D0_frozen",
                split,
                met,
                {
                    "role": "FROZEN",
                    "eta": 0.0,
                    "update_owner": "none",
                    "mu_move_ratio_mean": 0.0,
                    "mu_move_l2_mean": 0.0,
                    "n_updates": 0,
                    "update_rate": 0.0,
                    "scoring": "l2",
                    "k": K_ID,
                },
            )
            rows.append(row)
            if split == "test":
                frozen_test[pack["rec_key"]] = row

    cal_by_eta: dict[float, list[dict[str, Any]]] = {float(e): [] for e in ETA_EMA_GRID}
    for pack in packs:
        fit = fitted[pack["rec_key"]]
        t = thresh[pack["rec_key"]]
        for eta in ETA_EMA_GRID:
            met = _stream(pack, fit, "cal", t, eta, "self")
            rec = {
                "n_test": met["n_test"],
                "accuracy": met["accuracy"],
                "same_home_accuracy": met["same_home_accuracy"],
                "worst_unit_recall": met["worst_unit_recall"],
                "rec_key": pack["rec_key"],
                "eta": float(eta),
            }
            cal_by_eta[float(eta)].append(rec)
            rows.append(
                _pack_row(
                    pack,
                    f"ema_cal_eta{eta}",
                    "cal",
                    met,
                    {
                        "role": "SELF_EMA_CAL",
                        "eta": float(eta),
                        "update_owner": "self",
                        "scoring": "l2",
                        "k": K_ID,
                    },
                )
            )
            if not quiet:
                print(
                    f"cal {pack['rec_key']:<12} eta={eta:<5} acc={met['accuracy']:.4f} "
                    f"home={met['same_home_accuracy']}",
                    flush=True,
                )

    eta_star = _select_eta(cal_by_eta)
    if not quiet:
        print(f"frozen eta*={eta_star}", flush=True)

    for pack in packs:
        fit = fitted[pack["rec_key"]]
        t = thresh[pack["rec_key"]]
        for arm, owner, role in (
            ("EMA_star", "self", "SELF_EMA_STAR"),
            ("Oracle_EMA", "gt", "ORACLE_EMA_DIAGNOSTIC"),
        ):
            met = _stream(pack, fit, "test", t, eta_star, owner)
            ref = frozen_test[pack["rec_key"]]
            extra = {
                "role": role,
                "eta": eta_star,
                "eta_star": eta_star,
                "update_owner": owner,
                "scoring": "l2",
                "k": K_ID,
                "delta_acc_vs_frozen": float(met["accuracy"] - ref["accuracy"]),
                "delta_home_acc_vs_frozen": (
                    float(met["same_home_accuracy"] - ref["same_home_accuracy"])
                    if np.isfinite(met["same_home_accuracy"])
                    and np.isfinite(ref.get("same_home_accuracy", np.nan))
                    else float("nan")
                ),
                "delta_home_p10_vs_frozen": (
                    float(met["margin_home_p10"] - ref["margin_home_p10"])
                    if np.isfinite(met["margin_home_p10"])
                    and np.isfinite(ref.get("margin_home_p10", np.nan))
                    else float("nan")
                ),
                "delta_worst_vs_frozen": float(
                    met["worst_unit_recall"] - ref["worst_unit_recall"]
                ),
            }
            rows.append(_pack_row(pack, arm, "test", met, extra))
            if not quiet:
                print(
                    f"{pack['rec_key']:<12} {arm:<12} acc={met['accuracy']:.4f} "
                    f"home={met['same_home_accuracy']} mu={met['mu_move_ratio_mean']}",
                    flush=True,
                )

    # Deltas vs EMA* after both exist
    ema_test = {
        r["rec_key"]: r
        for r in rows
        if r["arm"] == "EMA_star" and r["split"] == "test"
    }
    for r in rows:
        if r["split"] != "test" or r["arm"] not in ("Oracle_EMA", "D0_frozen"):
            continue
        ref = ema_test.get(r["rec_key"])
        if ref is None:
            continue
        r["delta_acc_vs_ema_star"] = float(r["accuracy"] - ref["accuracy"])
        ha, hb = r.get("same_home_accuracy"), ref.get("same_home_accuracy")
        if isinstance(ha, float) and isinstance(hb, float) and np.isfinite(ha) and np.isfinite(hb):
            r["delta_home_acc_vs_ema_star"] = float(ha - hb)
        else:
            r["delta_home_acc_vs_ema_star"] = float("nan")

    for pack in packs:
        if pack["rec_key"] not in STATIC_KEYS:
            continue
        fit = fitted[pack["rec_key"]]
        t = thresh[pack["rec_key"]]
        for pert in CONTROLLED:
            pos, neg, meta = _perturb(pack, fit, pert)
            tr_f = CausalTracker(
                "frozen",
                fit["model"]["mu_pos"],
                fit["model"]["mu_neg"],
                fit["model"]["homes"],
                fit["rel_xy"],
                fit["table"],
                pack["geom"],
                pack["pitch"],
                pack["axis"],
                np.zeros(fit["model"]["homes"].size, dtype=np.int64),
                kind="l2",
                representation="posneg",
            )
            frozen_m = run_causal_stream(
                tr_f, pos, neg, pack["labels"][pack["test"]], fit["model"]["units"]
            )
            frozen_m["mu_move_ratio_mean"] = 0.0
            frozen_m["mu_move_l2_mean"] = 0.0
            rows.append(
                _pack_row(
                    pack,
                    "D0_frozen",
                    "test",
                    frozen_m,
                    {
                        "role": "FROZEN",
                        "perturbation": pert,
                        "eta": 0.0,
                        "update_owner": "none",
                        "scoring": "l2",
                        "k": K_ID,
                    },
                )
            )
            for arm, owner, role in (
                ("EMA_star", "self", "SELF_EMA_STAR"),
                ("Oracle_EMA", "gt", "ORACLE_EMA_DIAGNOSTIC"),
            ):
                met = _stream(pack, fit, "test", t, eta_star, owner, pos=pos, neg=neg)
                extra = {
                    "role": role,
                    "perturbation": pert,
                    "eta": eta_star,
                    "eta_star": eta_star,
                    "update_owner": owner,
                    "scoring": "l2",
                    "k": K_ID,
                    "delta_acc_vs_frozen": float(met["accuracy"] - frozen_m["accuracy"]),
                    "delta_home_acc_vs_frozen": (
                        float(met["same_home_accuracy"] - frozen_m["same_home_accuracy"])
                        if np.isfinite(met["same_home_accuracy"])
                        and np.isfinite(frozen_m["same_home_accuracy"])
                        else float("nan")
                    ),
                    "delta_home_p10_vs_frozen": (
                        float(met["margin_home_p10"] - frozen_m["margin_home_p10"])
                        if np.isfinite(met["margin_home_p10"])
                        and np.isfinite(frozen_m["margin_home_p10"])
                        else float("nan")
                    ),
                    "delta_worst_vs_frozen": float(
                        met["worst_unit_recall"] - frozen_m["worst_unit_recall"]
                    ),
                    "mu_move_ratio_mean": met["mu_move_ratio_mean"],
                    "mu_move_l2_mean": met["mu_move_l2_mean"],
                }
                rows.append(_pack_row(pack, arm, "test", met, extra))
                if not quiet:
                    print(
                        f"{pack['rec_key']:<12} {pert:<12} {arm:<12} "
                        f"acc={met['accuracy']:.4f} dAcc={extra['delta_acc_vs_frozen']:+.3f}",
                        flush=True,
                    )

    return {"rows": rows, "eta_star": eta_star, "cal_by_eta": cal_by_eta}


def _plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = ["hj_drift", "hj_static", "mearec_10u", "mearec_20u"]
    arms = ["D0_frozen", "EMA_star", "Oracle_EMA"]
    test = [
        r
        for r in rows
        if r["split"] == "test" and not r.get("perturbation")
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(recs))
    for i, arm in enumerate(arms):
        vals = []
        for rec in recs:
            hit = [r["accuracy"] for r in test if r["rec_key"] == rec and r["arm"] == arm]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + i * 0.25, vals, 0.25, label=arm)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(recs, rotation=15)
    ax.set_ylabel("overall accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    try_savefig(out / "plots" / "ema_star_acc.png", fig)
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
    csv_path = out / "ema_star_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    _plots(result["rows"], out)
    keys = (
        "rec_key",
        "arm",
        "split",
        "perturbation",
        "eta",
        "eta_star",
        "update_owner",
        "role",
        "accuracy",
        "same_home_accuracy",
        "margin_home_p10",
        "margin_home_p50",
        "margin_home_p90",
        "margin_home_ppos",
        "worst_unit_recall",
        "n_test",
        "n_home_rival",
        "n_updates",
        "update_rate",
        "mu_move_ratio_mean",
        "mu_move_l2_mean",
        "delta_acc_vs_frozen",
        "delta_home_acc_vs_frozen",
        "delta_home_p10_vs_frozen",
        "delta_worst_vs_frozen",
        "delta_acc_vs_ema_star",
        "delta_home_acc_vs_ema_star",
    )
    summary = {
        "eta_star": result["eta_star"],
        "eta_grid": list(ETA_EMA_GRID),
        "recordings": sorted({r["rec_key"] for r in result["rows"]}),
        "test_rows": [
            json_ready({k: r[k] for k in keys if k in r})
            for r in result["rows"]
            if r["split"] == "test"
        ],
        "cal_eta_table": json_ready(
            {
                str(eta): recs
                for eta, recs in result["cal_by_eta"].items()
            }
        ),
        "source_sha256": source_hashes(SOURCE_PATHS),
        "gt_used": True,
    }
    json_path = out / "ema_star_summary.json"
    write_json(json_path, json_ready(summary))
    write_sha256_manifest(
        out / "outputs.sha256",
        {
            rel_or_str(csv_path): sha256_file(csv_path),
            rel_or_str(json_path): sha256_file(json_path),
        },
    )
    print(f"Saved {csv_path}")
    print(f"eta*={result['eta_star']}")


if __name__ == "__main__":
    main()
