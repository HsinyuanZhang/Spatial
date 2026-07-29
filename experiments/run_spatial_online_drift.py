"""Stage B: online spatial adaptation on HJ drift scenes.

Arms: frozen (no update), online (self-predicted EMA), oracle-refit (upper bound
using GT labels on a sliding window — not deployable, bounds recoverability).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.online_spatial_adaptation import OnlineSpatialAdapter
from Spatial.algorithms.spatial_wta import SpatialWTAClassifier
from Spatial.data.loader import list_hybrid_janelia_scenes, load_hybrid_janelia
from Spatial.experiments.run_causal_temporal_sketch import stable_three_way_split
from Spatial.experiments.spatial_cim_common import (
    classification_metrics,
    prepare_spatial_events,
    sha256_json,
    write_csv,
    write_json,
)


def _fit_calibrate(
    features: np.ndarray, labels: np.ndarray, fit_idx: np.ndarray, cal_idx: np.ndarray
) -> SpatialWTAClassifier:
    clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
    clf.fit(features[fit_idx], labels[fit_idx])
    clf.calibrate(features[cal_idx], labels[cal_idx])
    return clf


def run_scene(dataset_id: str, dataset, *, lr_shift: int = 4) -> dict[str, Any]:
    prepared = prepare_spatial_events(dataset, k_neighbors=7)
    fit_idx, cal_idx, test_idx = stable_three_way_split(prepared["times"])
    feat = prepared["features"]["com_and_p2p"]
    labels = prepared["labels"]

    # Frozen arm.
    frozen = _fit_calibrate(feat, labels, fit_idx, cal_idx)
    frozen_pred = frozen.predict(feat[test_idx])
    frozen_metrics = classification_metrics(frozen_pred, labels[test_idx])

    # Online arm: stream test events in time order, predict then update.
    online_clf = _fit_calibrate(feat, labels, fit_idx, cal_idx)
    adapter = OnlineSpatialAdapter(online_clf, lr_shift=lr_shift, update_theta=True)
    online_pred = adapter.predict_and_update(feat[test_idx])
    online_metrics = classification_metrics(online_pred, labels[test_idx])
    update_acct = adapter.update_accounting()

    # Oracle-refit upper bound: refit centroids on fit+cal+all-seen-test-so-far
    # using GT labels every block. Not deployable; bounds G5.
    block = max(int(test_idx.size // 10), 1)
    oracle_pred = np.empty(test_idx.size, dtype=object)
    seen_feat = list(feat[fit_idx])
    seen_lab = list(labels[fit_idx])
    # Include calibration in the oracle bank.
    seen_feat.extend(list(feat[cal_idx]))
    seen_lab.extend(list(labels[cal_idx]))
    for start in range(0, test_idx.size, block):
        stop = min(start + block, test_idx.size)
        bank_x = np.asarray(seen_feat, dtype=np.float64)
        bank_y = np.asarray(seen_lab)
        clf = SpatialWTAClassifier(n_bits=5, use_theta=True)
        clf.fit(bank_x, bank_y)
        # Theta from the most recent calibration-sized tail of the bank.
        cal_tail = min(cal_idx.size, bank_x.shape[0])
        clf.calibrate(bank_x[-cal_tail:], bank_y[-cal_tail:])
        block_feat = feat[test_idx[start:stop]]
        oracle_pred[start:stop] = clf.predict(block_feat)
        seen_feat.extend(list(block_feat))
        seen_lab.extend(list(labels[test_idx[start:stop]]))
    oracle_metrics = classification_metrics(oracle_pred, labels[test_idx])

    return {
        "dataset": dataset_id,
        "family": "hj",
        "condition": "drift",
        "n_test": int(test_idx.size),
        "frozen_accuracy": frozen_metrics["accuracy"],
        "frozen_n_correct": frozen_metrics["n_correct"],
        "online_accuracy": online_metrics["accuracy"],
        "online_n_correct": online_metrics["n_correct"],
        "oracle_refit_accuracy": oracle_metrics["accuracy"],
        "oracle_refit_n_correct": oracle_metrics["n_correct"],
        "online_minus_frozen_pp": (online_metrics["accuracy"] - frozen_metrics["accuracy"]) * 100.0,
        "oracle_minus_frozen_pp": (oracle_metrics["accuracy"] - frozen_metrics["accuracy"]) * 100.0,
        "update_accounting": update_acct,
        "lr_shift": lr_shift,
        "search_payload_bits_per_row": int(frozen.memory_accounting()["search_payload_bits_per_row"]),
    }


def evaluate_g5(
    drift_rows: list[dict[str, Any]], static_accuracy: float | None
) -> dict[str, Any]:
    """G5: online recovers ≥ half the static−drift gap."""
    if not drift_rows:
        return {"pass": False, "reason": "no drift rows"}
    n = sum(r["n_test"] for r in drift_rows)
    frozen = sum(r["frozen_accuracy"] * r["n_test"] for r in drift_rows) / max(n, 1)
    online = sum(r["online_accuracy"] * r["n_test"] for r in drift_rows) / max(n, 1)
    oracle = sum(r["oracle_refit_accuracy"] * r["n_test"] for r in drift_rows) / max(n, 1)
    if static_accuracy is None:
        # Without a static reference, require online > frozen and report gap to oracle.
        gap = oracle - frozen
        recovered = online - frozen
        return {
            "frozen_accuracy": frozen,
            "online_accuracy": online,
            "oracle_refit_accuracy": oracle,
            "static_accuracy": None,
            "static_minus_drift_gap": None,
            "recovered_pp": recovered * 100.0,
            "oracle_gap_pp": gap * 100.0,
            "pass": online > frozen,
            "note": "static reference unavailable; pass if online > frozen",
        }
    gap = static_accuracy - frozen
    recovered = online - frozen
    required = 0.5 * gap
    return {
        "frozen_accuracy": frozen,
        "online_accuracy": online,
        "oracle_refit_accuracy": oracle,
        "static_accuracy": static_accuracy,
        "static_minus_drift_gap": gap,
        "recovered": recovered,
        "required": required,
        "pass": gap <= 0 or recovered >= required,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial online drift Stage B")
    parser.add_argument("--lr-shift", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "spatial_cim_online_drift",
    )
    parser.add_argument(
        "--static-accuracy",
        type=float,
        default=None,
        help="Optional static family accuracy for G5 gap computation",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    for scene in list_hybrid_janelia_scenes(ready_only=True):
        key = scene.get("short_name") or scene.get("scene_key")
        path = Path(scene.get("npz_path", ""))
        if not path.exists():
            continue
        dataset_id = f"hybrid_janelia_{str(key).replace('_filtered_gt', '')}"
        is_drift = "drift" in dataset_id
        print(f"{'drift' if is_drift else 'static'} {dataset_id} ...", flush=True)
        dataset = load_hybrid_janelia(path=path, duration_s=args.duration)
        row = run_scene(dataset_id, dataset, lr_shift=args.lr_shift)
        row["condition"] = "drift" if is_drift else "static"
        if is_drift:
            rows.append(row)
        else:
            # For static, only need frozen accuracy as reference.
            static_rows.append(row)

    static_acc = args.static_accuracy
    if static_acc is None and static_rows:
        n = sum(r["n_test"] for r in static_rows)
        static_acc = sum(r["frozen_accuracy"] * r["n_test"] for r in static_rows) / max(n, 1)

    gates = {"G5_drift": evaluate_g5(rows, static_acc)}
    write_csv(args.output_dir / "spatial_online_drift.csv", rows + static_rows)
    write_json(args.output_dir / "spatial_online_drift_gates.json", gates)
    write_json(
        args.output_dir / "spatial_online_drift_summary.json",
        {
            "gates": gates,
            "n_drift": len(rows),
            "n_static": len(static_rows),
            "digest": sha256_json(
                [
                    {
                        "dataset": r["dataset"],
                        "frozen_n_correct": r["frozen_n_correct"],
                        "online_n_correct": r["online_n_correct"],
                    }
                    for r in rows
                ]
            ),
        },
    )
    print(gates)


if __name__ == "__main__":
    main()
