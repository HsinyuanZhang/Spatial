"""Stage A pilot: low-bit spatial WTA on the four locked recordings.

Sweeps bit depth (A1), feature set (A2), adaptive ablation (A3), and baselines
(A4).  Reuses ``prepare_causal_events`` / ``stable_three_way_split`` from the
temporal pilot so results are directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.spatial_wta import SpatialWTAClassifier
from Spatial.experiments.run_causal_temporal_sketch import (
    discover_pilot_datasets,
    prepare_causal_events,
    stable_three_way_split,
)
from Spatial.experiments.spatial_cim_common import (
    classification_metrics,
    event_weighted_mean,
    float_template_assign,
    prepare_spatial_events,
    sha256_json,
    write_csv,
    write_json,
)


BIT_DEPTHS: list[int | None] = [2, 3, 4, 5, 6, None]
MASK_SIZES = (3, 5, 7, 9)
DEFAULT_FEATURE = "com_and_p2p"


def _bits_label(n_bits: int | None) -> str:
    return "float" if n_bits is None else f"b{n_bits}"


def _fit_predict(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_bits: int | None,
    use_theta: bool = True,
    mask_mode: str = "all_dims",
    mask_size: int | None = None,
    score_mode: str = "uniform",
    rank_profile: str = "top3_x4",
) -> dict[str, Any]:
    clf = SpatialWTAClassifier(
        n_bits=n_bits,
        fixed_unit_interval=True,
        use_theta=use_theta,
        mask_mode=mask_mode,  # type: ignore[arg-type]
        mask_size=mask_size,
        score_mode=score_mode,  # type: ignore[arg-type]
        rank_profile=rank_profile,  # type: ignore[arg-type]
    )
    clf.fit(x_fit, y_fit)
    clf.calibrate(x_cal, y_cal)
    pred = clf.predict(x_test)
    metrics = classification_metrics(pred, y_test)
    accounting = clf.memory_accounting()
    ops = clf.logical_op_counts(len(y_test))
    return {
        **metrics,
        "predictions": pred,
        "memory_accounting": accounting,
        "logical_op_counts": ops,
        "theta_source": clf.theta_source_,
        "n_units": int(clf.units_.size),
        "n_features": int(clf.n_features_in_),
        "search_payload_bits_per_row": int(accounting["search_payload_bits_per_row"]),
    }


def run_recording(dataset_id: str, family: str, dataset) -> list[dict[str, Any]]:
    """Run A1–A4 on one recording; return flat result rows."""
    # Default D9 descriptor via the temporal pilot preparation (byte-comparable).
    prepared = prepare_causal_events(dataset)
    fit_idx, cal_idx, test_idx = stable_three_way_split(prepared["times"])
    desc = prepared["descriptor"]
    labels = prepared["labels"]
    raw = prepared["raw_waveforms"]
    x_fit, y_fit = desc[fit_idx], labels[fit_idx]
    x_cal, y_cal = desc[cal_idx], labels[cal_idx]
    x_test, y_test = desc[test_idx], labels[test_idx]

    rows: list[dict[str, Any]] = []
    common = {
        "dataset": dataset_id,
        "family": family,
        "n_fit": int(fit_idx.size),
        "n_calibration": int(cal_idx.size),
        "n_test": int(test_idx.size),
        "feature_set": DEFAULT_FEATURE,
    }

    # --- A1 bit depth ---
    for n_bits in BIT_DEPTHS:
        result = _fit_predict(
            x_fit, y_fit, x_cal, y_cal, x_test, y_test,
            n_bits=n_bits, use_theta=True,
        )
        rows.append({
            **common,
            "stage": "A1_bit_depth",
            "configuration": f"uniform_theta_{_bits_label(n_bits)}",
            "n_bits": -1 if n_bits is None else int(n_bits),
            "use_theta": True,
            "mask_mode": "all_dims",
            "mask_size": 9,
            "score_mode": "uniform",
            "accuracy": result["accuracy"],
            "n_correct": result["n_correct"],
            "n_events": result["n_events"],
            "worst_unit_accuracy": result["worst_unit_accuracy"],
            "search_payload_bits_per_row": result["search_payload_bits_per_row"],
            "memory_accounting": result["memory_accounting"],
            "logical_op_counts": result["logical_op_counts"],
            "per_unit": result["per_unit"],
        })

    # --- A4 baselines (needed early for G2) ---
    # Teacher on all units (no candidate list): nearest float L1 template.
    teacher_pred, _ = float_template_assign(raw[fit_idx], labels[fit_idx], raw[test_idx])
    teacher_metrics = classification_metrics(teacher_pred, y_test)
    rows.append({
        **common,
        "stage": "A4_baseline",
        "configuration": "single_channel_float64_teacher",
        "n_bits": -1,
        "use_theta": False,
        "mask_mode": "n/a",
        "mask_size": 0,
        "score_mode": "n/a",
        "accuracy": teacher_metrics["accuracy"],
        "n_correct": teacher_metrics["n_correct"],
        "n_events": teacher_metrics["n_events"],
        "worst_unit_accuracy": teacher_metrics["worst_unit_accuracy"],
        "search_payload_bits_per_row": 2048,
        "memory_accounting": {"representation": "float64_central_64", "bits_per_unit": 2048},
        "logical_op_counts": {},
        "per_unit": teacher_metrics["per_unit"],
    })

    # Multi-channel neighborhood float template baseline.
    spatial = prepare_spatial_events(dataset, k_neighbors=7)
    # Align spatial prep to the same chronological split via times.
    # prepare_spatial_events uses the same filter/alignment; re-split by times.
    s_fit, s_cal, s_test = stable_three_way_split(spatial["times"])
    neigh = spatial["raw_neighborhood"]
    multi_pred, _ = float_template_assign(
        neigh[s_fit], spatial["labels"][s_fit], neigh[s_test]
    )
    multi_metrics = classification_metrics(multi_pred, spatial["labels"][s_test])
    rows.append({
        **common,
        "stage": "A4_baseline",
        "configuration": "multichannel_float64_neighborhood7x64",
        "n_bits": -1,
        "use_theta": False,
        "mask_mode": "n/a",
        "mask_size": 0,
        "score_mode": "n/a",
        "accuracy": multi_metrics["accuracy"],
        "n_correct": multi_metrics["n_correct"],
        "n_events": multi_metrics["n_events"],
        "worst_unit_accuracy": multi_metrics["worst_unit_accuracy"],
        "search_payload_bits_per_row": 7 * 64 * 64,
        "memory_accounting": {
            "representation": "float64_neighborhood_7x64",
            "bits_per_unit": 7 * 64 * 64,
        },
        "logical_op_counts": {},
        "per_unit": multi_metrics["per_unit"],
    })

    # --- A2 feature set (B=5, uniform, theta) ---
    for feature_set, values in spatial["features"].items():
        result = _fit_predict(
            values[s_fit], spatial["labels"][s_fit],
            values[s_cal], spatial["labels"][s_cal],
            values[s_test], spatial["labels"][s_test],
            n_bits=5, use_theta=True,
        )
        rows.append({
            **common,
            "stage": "A2_feature_set",
            "configuration": f"b5_uniform_theta_{feature_set}",
            "feature_set": feature_set,
            "n_bits": 5,
            "use_theta": True,
            "mask_mode": "all_dims",
            "mask_size": int(values.shape[1]),
            "score_mode": "uniform",
            "accuracy": result["accuracy"],
            "n_correct": result["n_correct"],
            "n_events": result["n_events"],
            "worst_unit_accuracy": result["worst_unit_accuracy"],
            "search_payload_bits_per_row": result["search_payload_bits_per_row"],
            "memory_accounting": result["memory_accounting"],
            "logical_op_counts": result["logical_op_counts"],
            "per_unit": result["per_unit"],
        })

    # --- A3 adaptive ablation on D9 ---
    adaptive_configs = [
        ("uniform", "all_dims", 9, "uniform", "top3_x4"),
        ("mask_stability_m3", "stability", 3, "uniform", "top3_x4"),
        ("mask_stability_m5", "stability", 5, "uniform", "top3_x4"),
        ("mask_stability_m7", "stability", 7, "uniform", "top3_x4"),
        ("mask_sep_m3", "separation_stability", 3, "uniform", "top3_x4"),
        ("mask_sep_m5", "separation_stability", 5, "uniform", "top3_x4"),
        ("weight_stability_top3", "all_dims", 9, "stability", "top3_x4"),
        ("weight_stability_tiered", "all_dims", 9, "stability", "tiered_421"),
        ("weight_sep_top3", "all_dims", 9, "separation_stability", "top3_x4"),
        ("mask3_weight_stability", "stability", 3, "stability", "top3_x4"),
    ]
    for name, mask_mode, mask_size, score_mode, rank_profile in adaptive_configs:
        for use_theta in (False, True):
            result = _fit_predict(
                x_fit, y_fit, x_cal, y_cal, x_test, y_test,
                n_bits=5,
                use_theta=use_theta,
                mask_mode=mask_mode,
                mask_size=mask_size,
                score_mode=score_mode,
                rank_profile=rank_profile,
            )
            theta_tag = "theta" if use_theta else "notheta"
            rows.append({
                **common,
                "stage": "A3_adaptive",
                "configuration": f"b5_{name}_{theta_tag}",
                "n_bits": 5,
                "use_theta": use_theta,
                "mask_mode": mask_mode,
                "mask_size": mask_size,
                "score_mode": score_mode,
                "rank_profile": rank_profile,
                "accuracy": result["accuracy"],
                "n_correct": result["n_correct"],
                "n_events": result["n_events"],
                "worst_unit_accuracy": result["worst_unit_accuracy"],
                "search_payload_bits_per_row": result["search_payload_bits_per_row"],
                "memory_accounting": result["memory_accounting"],
                "logical_op_counts": result["logical_op_counts"],
                "per_unit": result["per_unit"],
            })

    return rows


def evaluate_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute G1–G3 from integer hit counts (never from rounded display)."""
    by_family: dict[str, list[dict[str, Any]]] = {"hj": [], "mearec": []}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    def family_acc(family: str, stage: str, configuration: str) -> float:
        selected = [
            r for r in by_family[family]
            if r["stage"] == stage and r["configuration"] == configuration
        ]
        if not selected:
            return float("nan")
        return event_weighted_mean(selected, "accuracy")

    g1 = {}
    g2 = {}
    for family in ("hj", "mearec"):
        acc5 = family_acc(family, "A1_bit_depth", "uniform_theta_b5")
        acc_float = family_acc(family, "A1_bit_depth", "uniform_theta_float")
        teacher = family_acc(family, "A4_baseline", "single_channel_float64_teacher")
        g1[family] = {
            "acc_b5": acc5,
            "acc_float": acc_float,
            "delta_pp": (acc5 - acc_float) * 100.0,
            "pass": abs(acc5 - acc_float) <= 0.01,
        }
        g2[family] = {
            "acc_b5": acc5,
            "acc_teacher": teacher,
            "delta_pp": (acc5 - teacher) * 100.0,
            "pass": acc5 > teacher,
        }

    # G3: best adaptive vs uniform on both families.
    g3_rows = [r for r in rows if r["stage"] == "A3_adaptive"]
    g3: dict[str, Any] = {"per_family": {}, "pass": False}
    best_name = None
    best_min_gain = -1e9
    configs = sorted({r["configuration"] for r in g3_rows if r["configuration"] != "b5_uniform_theta"})
    uniform = {
        family: family_acc(family, "A3_adaptive", "b5_uniform_theta")
        for family in ("hj", "mearec")
    }
    for config in configs:
        gains = {}
        for family in ("hj", "mearec"):
            acc = family_acc(family, "A3_adaptive", config)
            gains[family] = (acc - uniform[family]) * 100.0
        min_gain = min(gains.values())
        if min_gain > best_min_gain:
            best_min_gain = min_gain
            best_name = config
            g3["per_family"] = gains
            g3["best_configuration"] = config
            g3["uniform"] = uniform
    g3["pass"] = (
        best_name is not None
        and g3["per_family"].get("hj", -1) >= 2.0
        and g3["per_family"].get("mearec", -1) >= 2.0
    )

    return {
        "G1_lowbit_sufficiency": {
            "per_family": g1,
            "pass": all(v["pass"] for v in g1.values()),
        },
        "G2_spatial_sufficiency": {
            "per_family": g2,
            "pass": all(v["pass"] for v in g2.values()),
        },
        "G3_adaptive_gain": g3,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial WTA Stage A pilot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "spatial_cim_sorting_pilot",
    )
    args = parser.parse_args(argv)

    found = discover_pilot_datasets()
    all_rows: list[dict[str, Any]] = []
    for family in ("hj", "mearec"):
        for spec in sorted(found[family], key=lambda s: s.dataset_id):
            print(f"running {spec.dataset_id} ...", flush=True)
            rows = run_recording(spec.dataset_id, family, spec.load_dataset())
            all_rows.extend(rows)

    gates = evaluate_gates(all_rows)
    # Strip bulky prediction arrays before writing (already not stored).
    write_csv(args.output_dir / "spatial_wta_pilot.csv", all_rows)
    write_json(args.output_dir / "spatial_wta_pilot_gates.json", gates)
    write_json(
        args.output_dir / "spatial_wta_pilot_summary.json",
        {
            "n_rows": len(all_rows),
            "gates": gates,
            "row_digest": sha256_json(
                [
                    {
                        "dataset": r["dataset"],
                        "configuration": r["configuration"],
                        "n_correct": r["n_correct"],
                        "n_events": r["n_events"],
                    }
                    for r in all_rows
                ]
            ),
        },
    )
    print(json_dumps_gates(gates))


def json_dumps_gates(gates: dict[str, Any]) -> str:
    import json

    return json.dumps(gates, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
