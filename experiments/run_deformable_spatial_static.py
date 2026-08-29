"""Phase-1 deformable POSNEG template: classification ceiling, not tracking.

    python -m Spatial.experiments.run_deformable_spatial_static --pilot --duration 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.deformable_spatial_template import (
    ENABLE_ARMS,
    K_GRID,
    LAMBDA_GRID,
    DeformableSpatialTemplate,
    StateEnables,
    event_peak_channel,
    extract_all_channel_extrema,
    gather_posneg,
    score_margins,
    spatial_observables,
    summarize_margins,
)
from Spatial.algorithms.detection import bandpass_filter
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.run_adaptive_weighted_5bit import stable_chronological_three_way_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _load_locked_recording,
)
from Spatial.experiments.spatial_cim_common import classification_metrics, sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "deformable_spatial_static"
WINDOW = 15
MAIN_ARMS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
ATTR_ARMS = {
    "A_polarity": StateEnables(gain=True, polarity=True),
    "A_h1": StateEnables(n_h=1),
    "L6_nogain": StateEnables(gain=False, offset=True, polarity=True, n_h=2),
    "L6_nooffset": StateEnables(gain=True, offset=False, polarity=True, n_h=2),
    "L6_nopol": StateEnables(gain=True, offset=True, polarity=False, n_h=2),
}
G2_KEYS = ("c_primary", "c_pos", "c_neg", "dc")
G4_KEYS = ("spread", "spread_pos", "spread_neg", "skew_pos", "skew_neg", "sharp", "sharp_pos", "sharp_neg")
SOURCE_PATHS = (
    "Spatial/algorithms/deformable_spatial_template.py",
    "Spatial/experiments/run_deformable_spatial_static.py",
    "Spatial/docs/deformable_spatial_template_plan.md",
)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def prepare_events(dataset) -> dict[str, Any]:
    knn = prepare_dataset_events(dataset, k_neighbors=7, footprint_layout="knn")
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    vmax, vmin = extract_all_channel_extrema(filtered, knn["times"], window=WINDOW)
    if vmax.shape[0] != knn["times"].shape[0]:
        raise AssertionError("extrema rows drifted from prepared events")
    homes = event_peak_channel(vmax, vmin)
    fit, cal, test = stable_chronological_three_way_split(knn["times"])
    return {
        "dataset": dataset.name,
        "geom": dataset.geom,
        "vmax": vmax,
        "vmin": vmin,
        "labels": knn["labels"],
        "times": knn["times"],
        "homes": homes,
        "fit": fit,
        "cal": cal,
        "test": test,
    }


def _metrics(pred: np.ndarray, labels: np.ndarray, margins: dict[str, np.ndarray], r2_true: np.ndarray) -> dict[str, Any]:
    clf = classification_metrics(pred, labels)
    home_vals = margins["m_home"][margins["have_home_rival"]]
    same = margins["have_home_rival"]
    if np.any(same):
        same_acc = float(np.mean(pred[same] == labels[same]))
    else:
        same_acc = float("nan")
    sm = summarize_margins(home_vals)
    sall = summarize_margins(margins["m_all"])
    return {
        "accuracy": float(clf["accuracy"]),
        "same_home_accuracy": same_acc,
        "worst_unit_recall": float(clf["worst_unit_accuracy"]),
        "n_test": int(clf["n_events"]),
        "n_correct": int(clf["n_correct"]),
        "n_home_rival": int(same.sum()),
        "margin_home_p10": sm["p10"],
        "margin_home_p50": sm["p50"],
        "margin_home_p90": sm["p90"],
        "margin_home_ppos": sm["p_positive"],
        "margin_all_p10": sall["p10"],
        "margin_all_ppos": sall["p_positive"],
        "r2_true_mean": float(np.nanmean(r2_true)),
    }


def evaluate_model(
    model: DeformableSpatialTemplate,
    prep: dict[str, Any],
    idx: np.ndarray,
    *,
    frame: str,
) -> dict[str, Any]:
    rec = model.score_matrix(
        prep["vmax"][idx],
        prep["vmin"][idx],
        prep["homes"][idx],
        frame=frame,
    )
    y = prep["labels"][idx]
    margins = score_margins(rec["score"], y, model.units_, model.homes_)
    pred = margins["pred"]
    true_r2 = np.array(
        [
            rec["r2"][i, list(model.units_).index(_scalar(lab))]
            if _scalar(lab) in set(model.units_)
            else np.nan
            for i, lab in enumerate(y)
        ],
        dtype=np.float64,
    )
    out = _metrics(pred, y, margins, true_r2)
    out["true_delta_std"] = float(
        np.nanstd(
            [
                rec["delta"][i, list(model.units_).index(_scalar(lab))]
                if _scalar(lab) in set(model.units_)
                else np.nan
                for i, lab in enumerate(y)
            ]
        )
    )
    return out


def _event_weighted(rows: list[dict[str, Any]], field: str) -> float:
    n = sum(int(r["n_test"]) for r in rows)
    if n == 0:
        return float("nan")
    return sum(float(r[field]) * int(r["n_test"]) for r in rows if np.isfinite(r[field])) / max(n, 1)


def scalar_add_on(
    model: DeformableSpatialTemplate,
    prep: dict[str, Any],
    fit: np.ndarray,
    eval_idx: np.ndarray,
    keys: tuple[str, ...],
    base_scores: np.ndarray,
) -> np.ndarray:
    """Add z-scored observable L2 to an existing score matrix."""
    y_fit = prep["labels"][fit]
    means = {}
    stds = {}
    for i, unit in enumerate(model.units_):
        h = int(model.homes_[i])
        idx = fit[prep["labels"][fit] == unit]
        vals = []
        for e in idx:
            pos, neg, _ = gather_posneg(prep["vmax"][e], prep["vmin"][e], model.table[h])
            live = model.table[h] >= 0
            obs = spatial_observables(pos, neg, model.rel_xy[h], live, model.axis)
            vals.append([obs[k] for k in keys])
        arr = np.asarray(vals, dtype=np.float64)
        means[i] = arr.mean(axis=0) if arr.size else np.zeros(len(keys))
        stds[i] = np.maximum(arr.std(axis=0), 1e-6) if arr.size else np.ones(len(keys))
    extra = np.zeros_like(base_scores)
    for e, ei in enumerate(eval_idx):
        for i, h in enumerate(model.homes_):
            pos, neg, _ = gather_posneg(prep["vmax"][ei], prep["vmin"][ei], model.table[int(h)])
            live = model.table[int(h)] >= 0
            obs = spatial_observables(pos, neg, model.rel_xy[int(h)], live, model.axis)
            vec = np.array([obs[k] for k in keys], dtype=np.float64)
            extra[e, i] = float(np.mean(((vec - means[i]) / stds[i]) ** 2))
    return base_scores + extra


def run_pilot(mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    packs = []
    for family, name in LOCKED_DEVELOPMENT_RECORDINGS:
        dataset = _load_locked_recording(family, name, 60.0, mearec_dir)
        prep = prepare_events(dataset)
        prep["family"] = family
        packs.append(prep)
        if not quiet:
            print(f"prepared {prep['dataset']} n={prep['times'].size} C={prep['geom'].shape[0]}")
        del dataset

    rows: list[dict[str, Any]] = []

    def fit_and_eval(prep, k, arm, enables, frame, lam):
        model = DeformableSpatialTemplate(
            prep["geom"], k=k, enables=enables, lam_state=lam
        ).fit(
            prep["vmax"][prep["fit"]],
            prep["vmin"][prep["fit"]],
            prep["labels"][prep["fit"]],
            prep["homes"][prep["fit"]],
            frame=frame,
        )
        cal = evaluate_model(model, prep, prep["cal"], frame=frame)
        tes = evaluate_model(model, prep, prep["test"], frame=frame)
        return model, cal, tes

    # λ from C4 / K=12 / candidate-home calibration, one global value.
    lam_stats = {lam: [] for lam in LAMBDA_GRID}
    for prep in packs:
        for lam in LAMBDA_GRID:
            _m, cal, _t = fit_and_eval(prep, 12, "C4", ENABLE_ARMS["C4"], "candidate_home", lam)
            lam_stats[lam].append(cal)
    def _pick(stat_lists):
        best = None
        for lam, recs in stat_lists.items():
            acc = _event_weighted(recs, "accuracy")
            p10 = float(np.nanmean([r["margin_home_p10"] for r in recs]))
            key = (-acc, -p10 if np.isfinite(p10) else 0.0, lam)
            if best is None or key < best[0]:
                best = (key, lam)
        return best[1]
    lam = _pick(lam_stats)
    if not quiet:
        print(f"frozen lambda_state={lam}")

    jobs: list[tuple[str, int, str, StateEnables, str]] = []
    for k in K_GRID:
        for arm in MAIN_ARMS:
            jobs.append((arm, k, "candidate_home", ENABLE_ARMS[arm], arm))
    jobs.append(("C0", 12, "event_argmax", ENABLE_ARMS["C0"], "C0_argmax"))
    jobs.append(("C3", 12, "event_argmax", ENABLE_ARMS["C3"], "C3_argmax"))
    for name, enables in ATTR_ARMS.items():
        jobs.append((name, 12, "candidate_home", enables, name))

    models_c3: dict[str, DeformableSpatialTemplate] = {}
    scores_c3_test: dict[str, np.ndarray] = {}

    for src_arm, k, frame, enables, tag in jobs:
        for prep in packs:
            model, cal, tes = fit_and_eval(prep, k, src_arm, enables, frame, lam if enables != ENABLE_ARMS["C0"] else 0.0)
            row = {
                "dataset": prep["dataset"],
                "family": prep["family"],
                "arm": tag,
                "k": k,
                "frame": frame,
                "lambda_state": lam,
                **tes,
                "cal_accuracy": cal["accuracy"],
            }
            rows.append(row)
            if not quiet:
                print(
                    f"{tag:<12} K={k:<2} {frame:<16} {prep['dataset']:<42} "
                    f"acc={tes['accuracy']:.4f} homeAcc={tes['same_home_accuracy']:.4f} "
                    f"Mp10={tes['margin_home_p10']:.4g} R2={tes['r2_true_mean']:.3f}"
                )
            if tag == "C3" and k == 12 and frame == "candidate_home":
                models_c3[prep["dataset"]] = model
                scores_c3_test[prep["dataset"]] = model.score_matrix(
                    prep["vmax"][prep["test"]],
                    prep["vmin"][prep["test"]],
                    prep["homes"][prep["test"]],
                    frame="candidate_home",
                )["score"]

    for prep in packs:
        model = models_c3[prep["dataset"]]
        base = scores_c3_test[prep["dataset"]]
        y = prep["labels"][prep["test"]]
        for name, keys in (("C3_G2", G2_KEYS), ("C3_G4", G4_KEYS)):
            scored = scalar_add_on(model, prep, prep["fit"], prep["test"], keys, base)
            margins = score_margins(scored, y, model.units_, model.homes_)
            r2 = np.full(y.size, np.nan)
            tes = _metrics(margins["pred"], y, margins, r2)
            rows.append(
                {
                    "dataset": prep["dataset"],
                    "family": prep["family"],
                    "arm": name,
                    "k": 12,
                    "frame": "candidate_home",
                    "lambda_state": lam,
                    **tes,
                    "cal_accuracy": float("nan"),
                }
            )
            if not quiet:
                print(
                    f"{name:<12} K=12 {prep['dataset']:<42} acc={tes['accuracy']:.4f} "
                    f"homeAcc={tes['same_home_accuracy']:.4f} Mp10={tes['margin_home_p10']:.4g}"
                )
    return {"lambda_state": lam, "rows": rows}


def _family_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    arms = sorted({r["arm"] for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    for arm in arms:
        for k in ks:
            for family in ("hj", "mearec", "all"):
                sub = [
                    r
                    for r in rows
                    if r["arm"] == arm
                    and int(r["k"]) == k
                    and (family == "all" or r["family"] == family)
                ]
                if not sub:
                    continue
                out.append(
                    {
                        "arm": arm,
                        "k": k,
                        "family": family,
                        "n_test": sum(int(r["n_test"]) for r in sub),
                        "accuracy": _event_weighted(sub, "accuracy"),
                        "same_home_accuracy": _event_weighted(sub, "same_home_accuracy"),
                        "margin_home_p10": float(
                            np.nanmean([r["margin_home_p10"] for r in sub])
                        ),
                        "r2_true_mean": float(np.nanmean([r["r2_true_mean"] for r in sub])),
                        "worst_unit_min": min(float(r["worst_unit_recall"]) for r in sub),
                    }
                )
    return out


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
    rows = result["rows"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    csv_path = out / "deformable_spatial_static_pilot.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    family = _family_table(rows)
    write_json(
        out / "deformable_spatial_static_summary.json",
        {
            "lambda_state": result["lambda_state"],
            "family": family,
            "n_rows": len(rows),
            "source_sha256": {
                p: sha256_file(REPO_ROOT / p) for p in SOURCE_PATHS if (REPO_ROOT / p).exists()
            },
        },
    )
    print(f"Saved {csv_path}")
    print(f"lambda_state={result['lambda_state']}")
    for rec in family:
        if rec["family"] in ("hj", "mearec") and rec["arm"] in MAIN_ARMS and rec["k"] in (7, 12, 16):
            print(
                f"  {rec['arm']:<4} K={rec['k']:<2} {rec['family']:<6} "
                f"acc={rec['accuracy']:.4f} home={rec['same_home_accuracy']:.4f} "
                f"Mp10={rec['margin_home_p10']:.4g} R2={rec['r2_true_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
