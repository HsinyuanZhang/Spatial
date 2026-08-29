"""RO-SS-QDA frozen contrast on locked development recordings.

    python -m Spatial.experiments.run_ro_ss_qda --pilot --duration 60 --quiet
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.mapping_init import j0_search_from_descriptor
from Spatial.algorithms.ro_ss_qda import (
    ETA,
    LAMBDA_GRID,
    FitStandardizer,
    RegularizedQDA,
    d1_cohome_pairs,
    error_taxonomy,
    fit_means,
    majority_home,
    make_descriptor,
    mean_distance_scores,
    score_margins,
    select_lambda_global,
    summarize_margins,
)
from Spatial.algorithms.spatial_footprint import (
    extract_local_extrema,
    is_linear_probe,
    relative_offset_neighbor_table,
)
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.run_adaptive_weighted_5bit import stable_chronological_three_way_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _load_locked_recording,
)
from Spatial.experiments.spatial_cim_common import classification_metrics, sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "ro_ss_qda"
WINDOW = 15
HALF_WIDTH = 3
ARMS = ("P0", "P1", "P2", "P3", "P2b", "D1")
SOURCE_PATHS = (
    "Spatial/algorithms/ro_ss_qda.py",
    "Spatial/experiments/run_ro_ss_qda.py",
    "Spatial/docs/ro_ss_qda_plan.md",
    "Spatial/docs/RO-SS-QDA_执行说明.md",
    "Spatial/tests/test_ro_ss_qda.py",
)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def prepare_ro_events(dataset) -> dict[str, Any]:
    knn = prepare_dataset_events(dataset, k_neighbors=7, footprint_layout="knn")
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    table = relative_offset_neighbor_table(dataset.geom, half_width=HALF_WIDTH)
    vmax, vmin, _imax, _imin, times, nbr = extract_local_extrema(
        filtered, knn["times"], knn["central"], table, window=WINDOW
    )
    if not np.array_equal(times, knn["times"]):
        raise AssertionError("RO extrema dropped events relative to the J0 rows")
    one_d = is_linear_probe(dataset.geom)
    pack = make_descriptor(vmax, vmin, nbr, dataset.geom, scale="l1", one_d=one_d)
    pack_inf = make_descriptor(vmax, vmin, nbr, dataset.geom, scale="inf", one_d=one_d)
    pack.update(
        {
            "times": knn["times"],
            "labels": knn["labels"],
            "homes": knn["central"],
            "neighbor_ids": nbr,
            "p0_descriptor": knn["descriptor"],
            "z_inf": pack_inf["z"],
            "geom": dataset.geom,
            "dataset": dataset.name,
            "one_d_probe": one_d,
        }
    )
    return pack


def _metrics(
    pred: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    units: np.ndarray,
    unit_homes: np.ndarray,
    geom: np.ndarray,
) -> dict[str, Any]:
    clf = classification_metrics(pred, labels)
    margins = score_margins(scores, labels, units, unit_homes)
    tax = error_taxonomy(pred, labels, units, unit_homes, geom)
    home_vals = margins["m_home"][margins["have_home_rival"]]
    return {
        "accuracy": float(clf["accuracy"]),
        "n_test": int(clf["n_events"]),
        "n_correct": int(clf["n_correct"]),
        "worst_unit_recall": float(clf["worst_unit_accuracy"]),
        "n_units": int(units.size),
        "margin_all_p10": summarize_margins(margins["m_all"])["p10"],
        "margin_all_p50": summarize_margins(margins["m_all"])["p50"],
        "margin_all_ppos": summarize_margins(margins["m_all"])["p_positive"],
        "margin_home_p10": summarize_margins(home_vals)["p10"],
        "margin_home_p50": summarize_margins(home_vals)["p50"],
        "margin_home_ppos": summarize_margins(home_vals)["p_positive"],
        "n_home_rival_events": int(home_vals.size),
        "frac_err_same_home": float(tax["frac_same_home"]),
        "frac_err_neighbor_home": float(tax["frac_neighbor_home"]),
        "frac_err_far": float(tax["frac_far"]),
        "n_err": int(tax["n_err"]),
    }


def _assign_from_scores(scores: np.ndarray, units: np.ndarray) -> np.ndarray:
    return np.asarray(units)[np.argmax(scores, axis=1)]


def _p0_bundle(prep: dict[str, Any], fit: np.ndarray, cal: np.ndarray) -> dict[str, Any]:
    y = prep["labels"]
    desc = prep["p0_descriptor"]
    model = j0_search_from_descriptor(desc[fit], y[fit], desc[cal], y[cal])
    unit_homes = majority_home(prep["homes"][fit], y[fit], model.units_)
    return {"model": model, "unit_homes": unit_homes, "units": model.units_}


def _eval_p0(bundle: dict[str, Any], prep: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    model = bundle["model"]
    codes = model.transform(prep["p0_descriptor"][idx])
    scores = mean_distance_scores(codes.astype(np.float64), model.centroids_.astype(np.float64), "l1")
    pred = _assign_from_scores(scores, model.units_)
    y = prep["labels"][idx]
    out = _metrics(pred, y, scores, model.units_, bundle["unit_homes"], prep["geom"])
    candidates, _ = model.query_codes(codes, mode="l1")
    search = evaluate_candidate_search(candidates, y)
    out["candidate_recall"] = float(search["true_unit_candidate_recall"])
    out["mean_C"] = float(np.mean([len(c) for c in candidates]))
    return out


def fit_mean_arm(
    z_fit: np.ndarray,
    y_fit: np.ndarray,
    homes_fit: np.ndarray,
) -> dict[str, Any]:
    units, means = fit_means(z_fit, y_fit)
    unit_homes = majority_home(homes_fit, y_fit, units)
    return {"units": units, "means": means, "unit_homes": unit_homes}


def eval_mean_arm(
    bundle: dict[str, Any],
    z: np.ndarray,
    y: np.ndarray,
    geom: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    scores = mean_distance_scores(z, bundle["means"], metric)
    pred = _assign_from_scores(scores, bundle["units"])
    return _metrics(pred, y, scores, bundle["units"], bundle["unit_homes"], geom)


def eval_qda(
    model: RegularizedQDA,
    z: np.ndarray,
    y: np.ndarray,
    unit_homes: np.ndarray,
    geom: np.ndarray,
) -> dict[str, Any]:
    scores = model.scores(z)
    pred = model.predict(z)
    out = _metrics(pred, y, scores, model.units_, unit_homes, geom)
    out["n_forced_pooled"] = int(model.forced_pooled_.sum())
    out["forced_pooled_units"] = ",".join(
        str(_scalar(u)) for u, f in zip(model.units_, model.forced_pooled_) if f
    )
    out["n_fit_min"] = int(model.n_fit_.min()) if model.n_fit_.size else 0
    return out


def _event_weighted(rows: list[dict[str, Any]], field: str) -> float:
    n = sum(int(r["n_test"]) for r in rows)
    if n == 0:
        return float("nan")
    return sum(float(r[field]) * int(r["n_test"]) for r in rows) / n


def run_pilot(duration_s: float, mearec_dir: Path, quiet: bool) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for family, name in LOCKED_DEVELOPMENT_RECORDINGS:
        dataset = _load_locked_recording(family, name, duration_s, mearec_dir)
        prep = prepare_ro_events(dataset)
        fit, cal, test = stable_chronological_three_way_split(prep["times"])
        y = prep["labels"]
        scaler = FitStandardizer().fit(prep["z"][fit])
        z_fit, z_cal, z_test = (scaler.transform(prep["z"][idx]) for idx in (fit, cal, test))
        scaler_inf = FitStandardizer().fit(prep["z_inf"][fit])
        z_inf_cal = scaler_inf.transform(prep["z_inf"][cal])
        z_inf_test = scaler_inf.transform(prep["z_inf"][test])
        p0 = _p0_bundle(prep, fit, cal)
        mean_l1 = fit_mean_arm(z_fit, y[fit], prep["homes"][fit])
        qda = {}
        qda_diag = {}
        for lam in LAMBDA_GRID:
            qda[lam] = RegularizedQDA(lam, eta=ETA, diagonal=False).fit(z_fit, y[fit])
            qda_diag[lam] = RegularizedQDA(lam, eta=ETA, diagonal=True).fit(z_fit, y[fit])
        qda1 = RegularizedQDA(1.0, eta=ETA, diagonal=False).fit(z_fit, y[fit])
        unit_homes_q = majority_home(prep["homes"][fit], y[fit], qda[LAMBDA_GRID[0]].units_)
        d1 = d1_cohome_pairs(
            prep["u"][fit],
            prep["rho"][fit],
            prep["pos"][fit],
            prep["neg"][fit],
            prep["neighbor_ids"][fit],
            prep["geom"],
            z_fit,
            y[fit],
            prep["homes"][fit],
            mean_l1["units"],
            mean_l1["unit_homes"],
        )
        cal_l1 = eval_mean_arm(mean_l1, z_cal, y[cal], prep["geom"], "l1")
        cal_l2 = eval_mean_arm(mean_l1, z_cal, y[cal], prep["geom"], "l2")
        cal_qda = {
            lam: eval_qda(qda[lam], z_cal, y[cal], unit_homes_q, prep["geom"])
            for lam in LAMBDA_GRID
        }
        states.append(
            {
                "family": family,
                "dataset": dataset.name,
                "prep": prep,
                "idx": (fit, cal, test),
                "y": y,
                "z_cal": z_cal,
                "z_test": z_test,
                "z_inf_cal": z_inf_cal,
                "z_inf_test": z_inf_test,
                "scaler": scaler,
                "p0": p0,
                "mean": mean_l1,
                "qda": qda,
                "qda_diag": qda_diag,
                "qda1": qda1,
                "unit_homes_q": unit_homes_q,
                "d1": d1,
                "cal_l1": cal_l1,
                "cal_l2": cal_l2,
                "cal_qda": cal_qda,
                "one_d": prep["one_d_probe"],
                "n_fit": int(fit.size),
                "n_cal": int(cal.size),
                "n_test": int(test.size),
                "z_dim": int(z_fit.shape[1]),
            }
        )
        if not quiet:
            print(f"prepared {dataset.name}: n_test={test.size} z_dim={z_fit.shape[1]} 1d={prep['one_d_probe']}")
        del dataset

    p1_l1_acc = _event_weighted(
        [{"n_test": s["n_cal"], "accuracy": s["cal_l1"]["accuracy"]} for s in states],
        "accuracy",
    )
    p1_l2_acc = _event_weighted(
        [{"n_test": s["n_cal"], "accuracy": s["cal_l2"]["accuracy"]} for s in states],
        "accuracy",
    )
    p1_metric = "l1" if p1_l1_acc >= p1_l2_acc else "l2"
    lam_rows = []
    for lam in LAMBDA_GRID:
        acc = _event_weighted(
            [{"n_test": s["n_cal"], "accuracy": s["cal_qda"][lam]["accuracy"]} for s in states],
            "accuracy",
        )
        # event-weighted home p10 is not a percentile of pooled margins; use
        # recording-size-weighted mean of p10 as the tie-break only.
        p10 = _event_weighted(
            [
                {
                    "n_test": s["n_cal"],
                    "home_margin_p10": s["cal_qda"][lam]["margin_home_p10"],
                }
                for s in states
            ],
            "home_margin_p10",
        )
        lam_rows.append({"lam": lam, "accuracy": acc, "home_margin_p10": p10})
    lam = select_lambda_global(lam_rows)

    rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    for s in states:
        prep = s["prep"]
        fit, cal, test = s["idx"]
        y = s["y"]
        geom = prep["geom"]
        p1_test = eval_mean_arm(s["mean"], s["z_test"], y[test], geom, p1_metric)
        inf_fit = FitStandardizer().fit(prep["z_inf"][fit]).transform(prep["z_inf"][fit])
        mean_inf = fit_mean_arm(inf_fit, y[fit], prep["homes"][fit])
        p1b = eval_mean_arm(mean_inf, s["z_inf_test"], y[test], geom, p1_metric)
        p0_test = _eval_p0(s["p0"], prep, test)
        p2_test = eval_qda(s["qda"][lam], s["z_test"], y[test], s["unit_homes_q"], geom)
        p3_test = eval_qda(s["qda_diag"][lam], s["z_test"], y[test], s["unit_homes_q"], geom)
        p2b_test = eval_qda(s["qda1"], s["z_test"], y[test], s["unit_homes_q"], geom)
        common = {
            "dataset": s["dataset"],
            "family": s["family"],
            "n_fit": s["n_fit"],
            "n_cal": s["n_cal"],
            "z_dim": s["z_dim"],
            "one_d_probe": bool(s["one_d"]),
            "p1_metric": p1_metric,
            "lambda": float(lam),
            "eta": ETA,
        }
        for arm, rec, extra in (
            ("P0", p0_test, {"lambda": None, "p1_metric": "l1_5bit"}),
            ("P1", p1_test, {}),
            ("P1b", p1b, {"scale": "inf"}),
            ("P2", p2_test, {}),
            ("P3", p3_test, {}),
            ("P2b", p2b_test, {"lambda": 1.0}),
        ):
            row = {**common, **rec, **extra, "arm": arm}
            rows.append(row)
            if not quiet:
                print(
                    f"{arm:<4} {s['dataset']:<42} acc={row['accuracy']:.4f} "
                    f"home_p10={row['margin_home_p10']:.4g} worst={row['worst_unit_recall']:.3f} "
                    f"sameH_err={row['frac_err_same_home']:.3f}"
                )
        for item in s["d1"]:
            d1_rows.append({"dataset": s["dataset"], "family": s["family"], **item})

    return {
        "p1_metric": p1_metric,
        "lambda": float(lam),
        "lambda_calibration": lam_rows,
        "p1_cal_accuracy": {"l1": p1_l1_acc, "l2": p1_l2_acc},
        "rows": rows,
        "d1": d1_rows,
    }


def _family_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for arm in ("P0", "P1", "P1b", "P2", "P3", "P2b"):
        for family in ("hj", "mearec", "all"):
            sub = [
                r
                for r in rows
                if r["arm"] == arm and (family == "all" or r["family"] == family)
            ]
            if not sub:
                continue
            n = sum(int(r["n_test"]) for r in sub)
            n_err = sum(int(r["n_err"]) for r in sub)
            same_err = sum(float(r["frac_err_same_home"]) * int(r["n_err"]) for r in sub)
            home_p10s = [float(r["margin_home_p10"]) for r in sub if np.isfinite(r["margin_home_p10"])]
            out.append(
                {
                    "arm": arm,
                    "family": family,
                    "n_test": n,
                    "accuracy": _event_weighted(sub, "accuracy"),
                    "worst_unit_min": min(float(r["worst_unit_recall"]) for r in sub),
                    "margin_home_p10_mean": float(np.mean(home_p10s)) if home_p10s else float("nan"),
                    "frac_err_same_home": float(same_err / n_err) if n_err else 0.0,
                    "n_err": int(n_err),
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
    parser.add_argument("--arms", type=str, default="P0,P1,P2,P3,P2b,D1")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot (locked four-recording development set)")
    if float(args.duration) != 60.0:
        parser.error("duration locked to 60 s")
    wanted = {part.strip() for part in str(args.arms).split(",") if part.strip()}
    unknown = wanted - set(ARMS)
    if unknown:
        parser.error(f"unknown arms {sorted(unknown)}")
    result = run_pilot(60.0, Path(args.mearec_dir), bool(args.quiet))
    rows = [r for r in result["rows"] if r["arm"] in wanted or r["arm"] == "P1b"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ro_ss_qda_pilot.csv"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    d1_path = out / "ro_ss_qda_d1.csv"
    if result["d1"] and "D1" in wanted:
        with d1_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["d1"][0].keys()))
            writer.writeheader()
            writer.writerows(result["d1"])
    family = _family_table(result["rows"])
    write_json(
        out / "ro_ss_qda_summary.json",
        {
            "p1_metric": result["p1_metric"],
            "lambda": result["lambda"],
            "eta": ETA,
            "lambda_calibration": result["lambda_calibration"],
            "p1_cal_accuracy": result["p1_cal_accuracy"],
            "family": family,
            "n_rows": len(rows),
            "source_sha256": {p: sha256_file(REPO_ROOT / p) for p in SOURCE_PATHS if (REPO_ROOT / p).exists()},
        },
    )
    print(f"Saved {csv_path}")
    print(f"P1 metric={result['p1_metric']}  lambda={result['lambda']}")
    for rec in family:
        if rec["family"] in ("hj", "mearec") and rec["arm"] in ("P0", "P1", "P2", "P3", "P2b"):
            print(
                f"  {rec['arm']:<4} {rec['family']:<6} acc={rec['accuracy']:.4f} "
                f"home_p10~{rec['margin_home_p10_mean']:.4g} "
                f"sameH={rec['frac_err_same_home']:.3f}"
            )


if __name__ == "__main__":
    main()
