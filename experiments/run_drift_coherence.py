"""GT-only 1 s block state-ownership (not a classifier).

    python -m Spatial.experiments.run_drift_coherence --pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.drift_tracking import (
    BLOCK_S,
    coherence_vs_distance,
    gt_block_state,
    pairwise_corr,
    unit_xy,
)
from Spatial.algorithms.posneg_mvm_matching import _scalar
from Spatial.algorithms.spatial_shift_matching import SHIFT_FRACTIONAL
from Spatial.experiments.posneg_mvm_common import (
    REPO_ROOT,
    base_row,
    fit_k_model,
    json_ready,
    load_pilot_packs,
    source_hashes,
    try_savefig,
    write_csv_rows,
    write_sha256_manifest,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.spatial_cim_common import sha256_file, write_json


DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "drift_coherence"
SOURCE_PATHS = (
    "Spatial/algorithms/drift_tracking.py",
    "Spatial/experiments/run_drift_coherence.py",
    "Spatial/docs/posneg_mvm_shift_drift_plan.md",
    "Spatial/tests/test_posneg_mvm_shift_tracking.py",
)


def true_unit_gather(pack, fitted) -> tuple[np.ndarray, np.ndarray]:
    model = fitted["model"]
    row = {_scalar(u): i for i, u in enumerate(model["units"])}
    n = int(pack["times"].size)
    k = int(fitted["k"])
    pos = np.zeros((n, k))
    neg = np.zeros((n, k))
    y = pack["labels"]
    for e, lab in enumerate(y):
        ui = row.get(_scalar(lab))
        if ui is None:
            continue
        pos[e] = fitted["pos_eu"][e, ui]
        neg[e] = fitted["neg_eu"][e, ui]
    return pos, neg


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = load_pilot_packs(60.0, mearec_dir, quiet)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cd_store: dict[str, dict[str, Any]] = {}
    for pack in packs:
        for k in (7, 12, 16):
            fitted = fit_k_model(pack, k)
            model = fitted["model"]
            pos_t, neg_t = true_unit_gather(pack, fitted)
            live = fitted["table"] >= 0
            rec = gt_block_state(
                pos_t,
                neg_t,
                model["mu_pos"],
                model["mu_neg"],
                fitted["rel_xy"],
                live,
                pack["times"],
                pack["labels"],
                model["units"],
                pack["fs"],
                pack["pitch"],
                pack["axis"],
                deltas=SHIFT_FRACTIONAL,
                block_s=BLOCK_S,
                unit_homes=model["homes"],
            )
            xy = unit_xy(pack["geom"], model["homes"])
            cd = coherence_vs_distance(rec["delta"], xy)
            gcoh = coherence_vs_distance(np.log(np.maximum(rec["gain"], 1e-12)), xy)
            mean_abs_delta = float(np.nanmean(np.abs(rec["delta"])))
            mean_log_g = float(np.nanmean(np.abs(np.log(np.maximum(rec["gain"], 1e-12)))))
            gain_corr = pairwise_corr(rec["gain"])
            iu, ju = np.triu_indices(gain_corr.shape[0], 1)
            mean_gain_corr = float(np.nanmean(gain_corr[iu, ju]))
            row = base_row(pack, k, "coherence", "GT_block")
            row.update(
                {
                    "split": "all_gt",
                    "gt_used": True,
                    "n_units": int(model["units"].size),
                    "n_blocks": int(rec["blocks"].size),
                    "mean_abs_delta": mean_abs_delta,
                    "nearest_corr": cd["nearest_corr"],
                    "mean_pair_corr": cd["mean_corr"],
                    "corr_vs_dist_slope": cd["corr_vs_dist_slope"],
                    "mean_abs_log_gain": mean_log_g,
                    "mean_gain_pair_corr": mean_gain_corr,
                    "estimator_trusted": not (
                        pack["rec_key"] != "hj_drift" and mean_abs_delta > 0.15
                    ),
                }
            )
            rows.append(row)
            cd_store[f"{pack['rec_key']}_k{k}"] = cd
            d = cd.get("pair_dist")
            c = cd.get("pair_corr")
            if d is not None:
                for di, ci in zip(np.asarray(d), np.asarray(c)):
                    pair_rows.append(
                        {
                            "rec_key": pack["rec_key"],
                            "k": k,
                            "distance_um": float(di),
                            "delta_corr": float(ci),
                            "gt_used": True,
                        }
                    )
            if not quiet and k == 12:
                print(
                    f"{pack['rec_key']:<12} K=12 |d|={mean_abs_delta:.3f} "
                    f"C_nn={cd['nearest_corr']} slope={cd['corr_vs_dist_slope']} "
                    f"gainCorr={mean_gain_corr:.3f}",
                    flush=True,
                )
    return {"rows": rows, "pairs": pair_rows, "cd": cd_store}


def _plots(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    recs = ["hj_drift", "hj_static", "mearec_10u", "mearec_20u"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=True)
    for ax, rec in zip(axes.ravel(), recs):
        sub = [p for p in pairs if p["rec_key"] == rec and int(p["k"]) == 12]
        if not sub:
            ax.set_title(rec)
            continue
        ax.scatter([p["distance_um"] for p in sub], [p["delta_corr"] for p in sub], s=12, alpha=0.6)
        ax.set_title(rec)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("unit-home distance (um)")
        ax.set_ylabel("corr(delta_u, delta_v)")
    fig.suptitle("GT block drift coherence C(d) at K=12")
    fig.tight_layout()
    try_savefig(out / "plots" / "coherence_cd_k12.png", fig)
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
    csv_path = out / "drift_coherence_pilot.csv"
    write_csv_rows(csv_path, result["rows"])
    pair_path = out / "drift_coherence_pairs.csv"
    write_csv_rows(pair_path, result["pairs"] or [{"rec_key": "", "note": "empty"}])
    _plots(result["rows"], result["pairs"], out)
    summary = {
        "recordings": sorted({r["rec_key"] for r in result["rows"]}),
        "rows": json_ready(result["rows"]),
        "source_sha256": source_hashes(SOURCE_PATHS),
        "gt_used": True,
        "note": "GT block analysis; not classification",
    }
    json_path = out / "drift_coherence_summary.json"
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


if __name__ == "__main__":
    main()
