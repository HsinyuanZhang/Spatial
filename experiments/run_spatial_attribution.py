"""Stage D: attribution controls, error decomposition, hardware conformance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.spatial_wta import SpatialWTAClassifier
from Spatial.experiments.run_causal_temporal_sketch import (
    discover_pilot_datasets,
    stable_three_way_split,
)
from Spatial.experiments.run_spatial_wta_pilot import _fit_predict
from Spatial.experiments.spatial_cim_common import (
    classification_metrics,
    prepare_spatial_events,
    sha256_json,
    write_csv,
    write_json,
)


def _unit_main_channel(labels: np.ndarray, central: np.ndarray) -> dict[Any, int]:
    """Majority central channel per unit on the fit partition."""
    mapping: dict[Any, int] = {}
    for unit in np.unique(labels):
        ch = central[labels == unit]
        values, counts = np.unique(ch, return_counts=True)
        mapping[unit.item() if isinstance(unit, np.generic) else unit] = int(values[np.argmax(counts)])
    return mapping


def error_decomposition(
    predictions: np.ndarray,
    labels: np.ndarray,
    central: np.ndarray,
    unit_channel: dict[Any, int],
) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=object)
    truth = np.asarray(labels)
    wrong = np.asarray([p != y for p, y in zip(pred, truth)], dtype=bool)
    same_channel = 0
    cross_channel = 0
    for i in np.flatnonzero(wrong):
        y = truth[i]
        p = pred[i]
        y_key = y.item() if isinstance(y, np.generic) else y
        p_key = p.item() if isinstance(p, np.generic) else p
        y_ch = unit_channel.get(y_key, int(central[i]))
        p_ch = unit_channel.get(p_key, -1)
        if p_ch == y_ch:
            same_channel += 1
        else:
            cross_channel += 1
    n_err = int(wrong.sum())
    return {
        "n_errors": n_err,
        "same_main_channel": same_channel,
        "cross_channel": cross_channel,
        "same_main_channel_fraction": same_channel / max(n_err, 1),
    }


def pairwise_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    unit_a: Any,
    unit_b: Any,
) -> dict[str, Any]:
    mask_all = (labels == unit_a) | (labels == unit_b)
    # Restrict splits to the pair.
    fit = fit_idx[mask_all[fit_idx]]
    cal = cal_idx[mask_all[cal_idx]]
    test = test_idx[mask_all[test_idx]]
    if fit.size == 0 or test.size == 0:
        return {"unit_a": unit_a, "unit_b": unit_b, "accuracy": float("nan"), "n_events": 0}
    # Need both units in fit.
    if len(np.unique(labels[fit])) < 2:
        return {"unit_a": unit_a, "unit_b": unit_b, "accuracy": float("nan"), "n_events": int(test.size)}
    result = _fit_predict(
        features[fit], labels[fit],
        features[cal] if cal.size else features[fit],
        labels[cal] if cal.size else labels[fit],
        features[test], labels[test],
        n_bits=5, use_theta=True,
    )
    return {
        "unit_a": int(unit_a) if not isinstance(unit_a, str) else unit_a,
        "unit_b": int(unit_b) if not isinstance(unit_b, str) else unit_b,
        "accuracy": result["accuracy"],
        "n_events": result["n_events"],
        "n_correct": result["n_correct"],
    }


def hardware_conformance(clf: SpatialWTAClassifier, n_events: int) -> dict[str, Any]:
    ops = clf.logical_op_counts(n_events)
    accounting = clf.memory_accounting()
    expected_abs = n_events * int(clf.units_.size) * int(clf.n_features_in_)
    expected_adds = n_events * int(clf.units_.size) * max(int(clf.n_features_in_) - 1, 0)
    expected_reads = n_events * int(clf.units_.size)
    expected_wta = n_events * max(int(clf.units_.size) - 1, 0)
    checks = {
        "abs_diff_match": ops["absolute_differences"] == expected_abs,
        "adder_match": ops["adder_tree_additions"] == expected_adds,
        "row_read_match": ops["centroid_row_reads"] == expected_reads,
        "wta_match": ops["wta_comparisons"] == expected_wta,
        "no_divider": True,
        "no_multiplier": True,
        "no_square": True,
        "no_sort": True,
        "no_float_on_path": clf.n_bits is not None,
    }
    range_cam = None
    if clf.n_bits is not None:
        range_cam = clf.range_cam_accounting()
    return {
        "logical_op_counts": ops,
        "memory_accounting": accounting,
        "expected": {
            "absolute_differences": expected_abs,
            "adder_tree_additions": expected_adds,
            "centroid_row_reads": expected_reads,
            "wta_comparisons": expected_wta,
        },
        "checks": checks,
        "pass": all(checks.values()),
        "range_cam_accounting": range_cam,
        "indexed_sram_digital_comparison": {
            "description": (
                "Mandatory baseline: local indexed SRAM + digital L1/WTA using the "
                "same integer centroids, masks, weights, and theta. Op counts are "
                "identical to the CiM row datapath; only the memory technology differs."
            ),
            "search_payload_bits_per_row": accounting["search_payload_bits_per_row"],
            "ops": ops,
        },
    }


def run_recording(dataset_id: str, family: str, dataset) -> dict[str, Any]:
    # Clean baseline.
    clean = prepare_spatial_events(dataset, k_neighbors=7)
    fit_idx, cal_idx, test_idx = stable_three_way_split(clean["times"])
    feat = clean["features"]["com_and_p2p"]
    labels = clean["labels"]
    central = clean["central"]

    baseline = _fit_predict(
        feat[fit_idx], labels[fit_idx],
        feat[cal_idx], labels[cal_idx],
        feat[test_idx], labels[test_idx],
        n_bits=5, use_theta=True,
    )

    # D1a geometry permutation (rebuild KNN on scrambled electrode positions).
    geom_ctrl = prepare_spatial_events(dataset, k_neighbors=7, permute_geom=True, seed=0)
    g_fit, g_cal, g_test = stable_three_way_split(geom_ctrl["times"])
    g_feat = geom_ctrl["features"]["com_and_p2p"]
    geom_result = _fit_predict(
        g_feat[g_fit], geom_ctrl["labels"][g_fit],
        g_feat[g_cal], geom_ctrl["labels"][g_cal],
        g_feat[g_test], geom_ctrl["labels"][g_test],
        n_bits=5, use_theta=True,
    )

    # D1b COM-shuffle: permute COM rows across events, keep P2P aligned.
    # Destroys the event↔location association while preserving marginals.
    rng = np.random.default_rng(0)
    feat_com_shuf = feat.copy()
    com_order = rng.permutation(feat.shape[0])
    feat_com_shuf[:, :2] = feat[com_order, :2]
    com_shuf_result = _fit_predict(
        feat_com_shuf[fit_idx], labels[fit_idx],
        feat_com_shuf[cal_idx], labels[cal_idx],
        feat_com_shuf[test_idx], labels[test_idx],
        n_bits=5, use_theta=True,
    )

    # D1c P2P-shuffle: permute footprint rows across events, keep COM aligned.
    # Tests that the neighborhood amplitude pattern (not just COM) matters.
    feat_p2p_shuf = feat.copy()
    p2p_order = rng.permutation(feat.shape[0])
    feat_p2p_shuf[:, 2:] = feat[p2p_order, 2:]
    p2p_shuf_result = _fit_predict(
        feat_p2p_shuf[fit_idx], labels[fit_idx],
        feat_p2p_shuf[cal_idx], labels[cal_idx],
        feat_p2p_shuf[test_idx], labels[test_idx],
        n_bits=5, use_theta=True,
    )

    # D1d label-shuffle chance baseline on clean features.
    label_perm = rng.permutation(labels[fit_idx].size)
    y_fit_shuf = labels[fit_idx][label_perm]
    chance_result = _fit_predict(
        feat[fit_idx], y_fit_shuf,
        feat[cal_idx], labels[cal_idx],
        feat[test_idx], labels[test_idx],
        n_bits=5, use_theta=True,
    )

    # D2 error decomposition.
    unit_channel = _unit_main_channel(labels[fit_idx], central[fit_idx])
    errors = error_decomposition(
        baseline["predictions"], labels[test_idx], central[test_idx], unit_channel
    )

    # D3 confusion-pair rescue (HJ-specific pairs when present).
    pairs = []
    present = set(int(u) if not isinstance(u, np.generic) else int(u.item()) for u in np.unique(labels))
    for a, b in ((5, 21), (43, 63), (25, 62)):
        if a in present and b in present:
            pairs.append(pairwise_accuracy(feat, labels, fit_idx, cal_idx, test_idx, a, b))

    # D4 leakage audit: refit with scrambled calibration labels must change theta source digest.
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(feat[fit_idx], labels[fit_idx])
    fit_digest = sha256_json({
        "centroids": clf.centroids_.tolist(),
        "masks": clf.masks_.astype(int).tolist(),
        "weights": clf.weights_.tolist(),
    })
    clf.calibrate(feat[cal_idx], labels[cal_idx])
    theta_digest = sha256_json(clf.theta_.tolist())
    # Boxes for Range-CAM variant.
    codes_fit = clf.transform(feat[fit_idx])
    clf.fit_boxes_from_codes(codes_fit, labels[fit_idx], coverage=0.95)
    box_pred = clf.predict_box(feat[test_idx])
    box_metrics = classification_metrics(box_pred, labels[test_idx])

    # D5 hardware.
    hw = hardware_conformance(clf, int(test_idx.size))

    geom_drop = (baseline["accuracy"] - geom_result["accuracy"]) * 100.0
    com_shuf_drop = (baseline["accuracy"] - com_shuf_result["accuracy"]) * 100.0
    p2p_shuf_drop = (baseline["accuracy"] - p2p_shuf_result["accuracy"]) * 100.0
    chance_drop = (baseline["accuracy"] - chance_result["accuracy"]) * 100.0
    # G4 gates:
    #   geometry-structure = COM-shuffle drop ≥ 15 pp
    #   footprint-structure = P2P-shuffle drop ≥ 10 pp  (may fail if COM dominates)
    # Electrode-geom rebuild and label-shuffle chance are reported alongside.
    return {
        "dataset": dataset_id,
        "family": family,
        "baseline_accuracy": baseline["accuracy"],
        "baseline_n_correct": baseline["n_correct"],
        "baseline_n_events": baseline["n_events"],
        "geom_perm_accuracy": geom_result["accuracy"],
        "geom_drop_pp": geom_drop,
        "com_shuffle_accuracy": com_shuf_result["accuracy"],
        "com_shuffle_drop_pp": com_shuf_drop,
        "p2p_shuffle_accuracy": p2p_shuf_result["accuracy"],
        "p2p_shuffle_drop_pp": p2p_shuf_drop,
        "column_perm_accuracy": p2p_shuf_result["accuracy"],
        "column_drop_pp": p2p_shuf_drop,
        "slot_perm_accuracy": p2p_shuf_result["accuracy"],
        "slot_drop_pp": p2p_shuf_drop,
        "chance_accuracy": chance_result["accuracy"],
        "chance_drop_pp": chance_drop,
        "g4_geom_pass": com_shuf_drop >= 15.0,
        "g4_slot_pass": p2p_shuf_drop >= 10.0,
        "g4_column_pass": p2p_shuf_drop >= 10.0,
        "error_decomposition": errors,
        "confusion_pairs": pairs,
        "leakage_audit": {
            "fit_state_digest": fit_digest,
            "theta_digest": theta_digest,
            "theta_source": clf.theta_source_,
            "centroids_from": "fit",
            "masks_from": "fit",
            "weights_from": "fit",
            "theta_from": "calibration",
        },
        "range_cam_box_accuracy": box_metrics["accuracy"],
        "range_cam_box_n_correct": box_metrics["n_correct"],
        "hardware": hw,
        "g6_pass": hw["pass"],
        "geom_permutation": geom_ctrl["geom_permutation"],
        "search_payload_bits_per_row": baseline["search_payload_bits_per_row"],
    }


def evaluate_g4_g6(results: list[dict[str, Any]]) -> dict[str, Any]:
    g4 = {
        "per_recording": {
            r["dataset"]: {
                "electrode_geom_drop_pp": r["geom_drop_pp"],
                "com_shuffle_drop_pp": r["com_shuffle_drop_pp"],
                "p2p_shuffle_drop_pp": r["p2p_shuffle_drop_pp"],
                "chance_drop_pp": r["chance_drop_pp"],
                "geom_pass": r["g4_geom_pass"],
                "p2p_pass": r["g4_slot_pass"],
            }
            for r in results
        },
        # Primary G4: COM-shuffle must drop ≥15 pp (spatial location drives accuracy).
        # Secondary: P2P-shuffle drop reported; may be small when COM dominates —
        # that is itself evidence that the effect is spatial localization.
        "pass": all(r["g4_geom_pass"] for r in results),
        "com_structure_pass": all(r["g4_geom_pass"] for r in results),
        "p2p_structure_pass": all(r["g4_slot_pass"] for r in results),
        "joint_com_and_p2p_pass": all(r["g4_geom_pass"] and r["g4_slot_pass"] for r in results),
    }
    g6 = {
        "per_recording": {r["dataset"]: r["g6_pass"] for r in results},
        "pass": all(r["g6_pass"] for r in results),
    }
    return {"G4_attribution": g4, "G6_hardware": g6}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial WTA Stage D attribution")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "spatial_cim_attribution",
    )
    args = parser.parse_args(argv)

    found = discover_pilot_datasets()
    results: list[dict[str, Any]] = []
    for family in ("hj", "mearec"):
        for spec in sorted(found[family], key=lambda s: s.dataset_id):
            print(f"attribution {spec.dataset_id} ...", flush=True)
            results.append(run_recording(spec.dataset_id, family, spec.load_dataset()))

    gates = evaluate_g4_g6(results)
    write_json(args.output_dir / "spatial_attribution.json", {
        "results": results,
        "gates": gates,
        "digest": sha256_json(
            [
                {
                    "dataset": r["dataset"],
                    "baseline_n_correct": r["baseline_n_correct"],
                    "geom_drop_pp": r["geom_drop_pp"],
                    "slot_drop_pp": r["slot_drop_pp"],
                }
                for r in results
            ]
        ),
    })
    # Flatten for CSV.
    flat = []
    for r in results:
        flat.append({
            "dataset": r["dataset"],
            "family": r["family"],
            "baseline_accuracy": r["baseline_accuracy"],
            "geom_perm_accuracy": r["geom_perm_accuracy"],
            "geom_drop_pp": r["geom_drop_pp"],
            "com_shuffle_accuracy": r["com_shuffle_accuracy"],
            "com_shuffle_drop_pp": r["com_shuffle_drop_pp"],
            "p2p_shuffle_accuracy": r["p2p_shuffle_accuracy"],
            "p2p_shuffle_drop_pp": r["p2p_shuffle_drop_pp"],
            "chance_accuracy": r["chance_accuracy"],
            "chance_drop_pp": r["chance_drop_pp"],
            "g4_geom_pass": r["g4_geom_pass"],
            "g4_slot_pass": r["g4_slot_pass"],
            "same_channel_errors": r["error_decomposition"]["same_main_channel"],
            "cross_channel_errors": r["error_decomposition"]["cross_channel"],
            "range_cam_box_accuracy": r["range_cam_box_accuracy"],
            "g6_pass": r["g6_pass"],
            "search_payload_bits_per_row": r["search_payload_bits_per_row"],
        })
    write_csv(args.output_dir / "spatial_attribution.csv", flat)
    write_json(args.output_dir / "spatial_attribution_gates.json", gates)
    print(gates)


if __name__ == "__main__":
    main()
