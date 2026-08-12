"""Phase B candidate recall and waveform-template traffic for shape descriptors.

  python -m Spatial.experiments.run_shape_candidate_traffic --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search, fit_integer_centroids
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    _chronological_split,
    candidate_waveform_assign,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import (
    LOCKED_DEVELOPMENT_RECORDINGS,
    _iter_recordings,
    _load_locked_recording,
    _mearec_n_units,
    _parse_recordings_arg,
    _scene_type,
)
from Spatial.experiments.shape_descriptor_common import assemble_arm, build_descriptor_blocks, descriptor_ledger
from Spatial.experiments.spatial_cim_common import write_csv, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "shape_candidate_traffic"

PHASE_B_ARMS = ("d9", "com_posneg", "com_posneg_lat_width")
FROZEN_POSNEG_BITS = 4
FROZEN_LAT_BITS = 4
FROZEN_AMP_GATE = 0.125
DEFAULT_PERCENTILES = (95.0, 99.0, 99.9, 100.0)
RECALL_GATE = 0.99
TRAFFIC_REDUCTION_GATE = 4.0

CSV_FIELDNAMES = (
    "dataset",
    "family",
    "scene_type",
    "mearec_n_units",
    "arm",
    "posneg_bits",
    "lat_bits",
    "amp_gate",
    "radius_percentile",
    "n_train",
    "n_test",
    "n_units",
    "n_coords",
    "row_bits",
    "true_unit_candidate_recall",
    "zero_candidate_fraction",
    "one_candidate_fraction",
    "multi_candidate_fraction",
    "mean_candidate_count",
    "mean_row_comparisons",
    "mean_row_comparisons_including_fallback",
    "total_row_comparisons_including_fallback",
    "distinct_waveform_template_reads_total",
    "distinct_waveform_template_reads_per_event",
    "global_all_unit_waveform_reads",
    "waveform_read_reduction_factor",
    "final_assignment_accuracy",
    "n_correct",
)


def _python_scalar(value: Any) -> Hashable:
    return value.item() if isinstance(value, np.generic) else value


def _validate_percentile(percentile: float, *, name: str = "radius_percentile") -> float:
    value = float(percentile)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be in [0, 100]")
    return value


def fit_per_unit_l1_from_codes(
    codes: np.ndarray,
    labels: Sequence[Hashable] | np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit integer centroids and inclusive L1 radii on training codes only."""
    values = np.rint(np.asarray(codes)).astype(np.int64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("codes must be a non-empty (n_samples, n_features) matrix")
    labs = np.asarray(labels)
    if labs.ndim != 1 or labs.shape[0] != values.shape[0]:
        raise ValueError("labels must be one-dimensional and match codes")
    percentile = _validate_percentile(percentile)
    unique = np.unique(labs)
    unit_list = [_python_scalar(unit) for unit in unique]
    centroid_by_unit = fit_integer_centroids(values, labs)
    centroids = np.stack([centroid_by_unit[unit] for unit in unit_list], axis=0)
    radii = np.empty(len(unit_list), dtype=np.int64)
    for row, unit in enumerate(unit_list):
        unit_codes = values[labs == unit]
        distances = np.abs(unit_codes - centroids[row]).sum(axis=1)
        radii[row] = int(np.ceil(np.percentile(distances, percentile)))
    return np.asarray(unit_list), centroids, radii


def query_per_unit_l1_codes(
    codes: np.ndarray,
    units: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
    *,
    widening_steps: int = 0,
    ensure_nonempty: bool = False,
    active_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Per-unit L1 candidate search on integer codes; mirrors AdaptiveRangeSearch.query_codes."""
    if isinstance(widening_steps, bool) or int(widening_steps) != widening_steps or widening_steps < 0:
        raise ValueError("widening_steps must be a non-negative integer")
    steps = int(widening_steps)
    values = np.rint(np.asarray(codes)).astype(np.int64)
    if values.ndim != 2:
        raise ValueError("codes must have shape (n_queries, n_features)")
    row_units = np.asarray(units)
    centers = np.asarray(centroids, dtype=np.int64)
    unit_radii = np.asarray(radii, dtype=np.int64)
    if centers.shape[0] != row_units.shape[0] or unit_radii.shape[0] != row_units.shape[0]:
        raise ValueError("units, centroids, and radii must align")
    n_queries = values.shape[0]
    n_units = row_units.shape[0]
    if active_mask is None:
        active = np.ones((n_queries, n_units), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape == (n_units,):
            active = np.broadcast_to(active, (n_queries, n_units)).copy()
        elif active.shape != (n_queries, n_units):
            raise ValueError("active_mask must have shape (n_units,) or (n_queries, n_units)")

    candidates: list[np.ndarray] = []
    initial_count = np.zeros(n_queries, dtype=np.int64)
    final_count = np.zeros(n_queries, dtype=np.int64)
    widen_level = np.zeros(n_queries, dtype=np.int64)
    fallback = np.zeros(n_queries, dtype=bool)
    row_comparisons = np.zeros(n_queries, dtype=np.int64)
    fallback_row_comparisons = np.zeros(n_queries, dtype=np.int64)
    active_row_count = active.sum(axis=1).astype(np.int64)

    for query_index, code in enumerate(values):
        mask = active[query_index]
        distances = np.abs(centers - code).sum(axis=1)
        matched = mask & (distances <= unit_radii)
        row_comparisons[query_index] = int(mask.sum())
        initial_count[query_index] = int(matched.sum())
        level_found = 0
        for level in range(1, steps + 1):
            if np.any(matched):
                break
            matched = mask & (distances <= unit_radii + level)
            row_comparisons[query_index] += int(mask.sum())
            if np.any(matched):
                level_found = level
                break
        if not np.any(matched) and ensure_nonempty and np.any(mask):
            excess = np.maximum(distances - (unit_radii + steps), 0)
            score = excess / np.maximum(unit_radii + steps, 1)
            score = np.where(mask, score, np.inf)
            selected = int(np.argmin(score))
            matched[selected] = True
            fallback_row_comparisons[query_index] = int(mask.sum())
            fallback[query_index] = True
            level_found = steps
        final_count[query_index] = int(matched.sum())
        widen_level[query_index] = level_found
        candidates.append(row_units[matched].copy())

    diagnostics = {
        "initial_candidate_count": initial_count,
        "final_candidate_count": final_count,
        "widen_level": widen_level,
        "fallback": fallback,
        "active_row_count": active_row_count,
        "row_comparisons": row_comparisons,
        "fallback_row_comparisons": fallback_row_comparisons,
        "total_row_comparisons_including_fallback": row_comparisons + fallback_row_comparisons,
    }
    return candidates, diagnostics


def summarize_candidate_traffic(
    candidates: Sequence[np.ndarray],
    labels: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    units: np.ndarray,
    pred: np.ndarray,
    wave_meta: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    """Aggregate candidate recall, traffic, and downstream assignment metrics."""
    report = dict(evaluate_candidate_search(candidates, labels, dict(diagnostics)))
    counts = np.asarray([len(rows) for rows in candidates], dtype=np.int64)
    correct = np.asarray(pred == labels, dtype=bool)
    reads = np.asarray(wave_meta["candidate_waveform_reads"], dtype=np.int64)
    n = labels.size
    range_fallback_rows = np.asarray(
        diagnostics.get("fallback_row_comparisons", np.zeros(n, dtype=np.int64)), dtype=np.int64
    )
    spatial_fallback_rows = np.asarray(
        wave_meta.get("spatial_fallback_row_comparisons", np.zeros(n, dtype=np.int64)), dtype=np.int64
    )
    normal_rows = np.asarray(diagnostics["row_comparisons"], dtype=np.int64)
    total_rows = normal_rows + range_fallback_rows + spatial_fallback_rows
    global_reads = int(n * len(units))
    reads_total = int(reads.sum())
    return {
        "true_unit_candidate_recall": float(report["true_unit_candidate_recall"]),
        "zero_candidate_fraction": float(report["zero_fraction"]),
        "one_candidate_fraction": float(report["unique_fraction"]),
        "multi_candidate_fraction": float(report["ambiguous_fraction"]),
        "mean_candidate_count": float(report["mean_candidates"]),
        "mean_row_comparisons": float(normal_rows.mean()) if n else 0.0,
        "mean_row_comparisons_including_fallback": float(total_rows.mean()) if n else 0.0,
        "total_row_comparisons_including_fallback": int(total_rows.sum()),
        "distinct_waveform_template_reads_total": reads_total,
        "distinct_waveform_template_reads_per_event": float(reads.mean()) if n else 0.0,
        "global_all_unit_waveform_reads": global_reads,
        "waveform_read_reduction_factor": (
            float(global_reads / reads_total) if reads_total else float("inf")
        ),
        "final_assignment_accuracy": float(correct.mean()) if n else 0.0,
        "n_correct": int(correct.sum()),
    }


def evaluate_arm_candidate_traffic(
    blocks: Mapping[str, np.ndarray],
    waveforms: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
    recording_name: str,
    radius_percentile: float,
) -> dict[str, Any]:
    """Evaluate one arm at one radius percentile on the held-out test partition."""
    codes = assemble_arm(
        blocks,
        arm,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=amp_gate,
        recording_name=recording_name,
    )
    ledger = descriptor_ledger(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    train_codes = np.rint(codes[train_idx]).astype(np.int64)
    test_codes = np.rint(codes[test_idx]).astype(np.int64)
    train_labels = blocks["labels"][train_idx]
    test_labels = blocks["labels"][test_idx]
    units, centroids, radii = fit_per_unit_l1_from_codes(
        train_codes, train_labels, radius_percentile
    )
    units_with_template = {_python_scalar(u) for u in units}
    keep = np.asarray(
        [_python_scalar(label) in units_with_template for label in test_labels],
        dtype=bool,
    )
    eval_idx = test_idx[keep]
    if eval_idx.size == 0:
        raise ValueError("no test labels have a fitted template")
    query_codes = np.rint(codes[eval_idx]).astype(np.int64)
    labels_te = blocks["labels"][eval_idx]
    candidates, diagnostics = query_per_unit_l1_codes(
        query_codes,
        units,
        centroids,
        radii,
        widening_steps=0,
        ensure_nonempty=False,
    )
    waveform_centroids = {
        _python_scalar(unit): waveforms[train_idx][train_labels == unit].mean(axis=0)
        for unit in units
    }
    pred, wave_meta = candidate_waveform_assign(
        candidates,
        waveforms[eval_idx],
        waveform_centroids,
        query_codes,
        centroids,
        units,
        metric="l1",
    )
    metrics = summarize_candidate_traffic(
        candidates, labels_te, diagnostics, units, pred, wave_meta
    )
    return {
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_eval": int(eval_idx.size),
        "n_units": int(units.size),
        "n_coords": ledger["n_coords"],
        "row_bits": ledger["row_bits"],
        **metrics,
    }


def _family_event_weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    arm: str,
    field: str,
) -> float:
    subset = [row for row in rows if row["family"] == family and row["arm"] == arm]
    if not subset:
        return float("nan")
    total = sum(int(row["n_eval"]) for row in subset)
    if total == 0:
        return float("nan")
    return sum(float(row[field]) * int(row["n_eval"]) for row in subset) / total


def _family_reduction_factor(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    arm: str,
) -> float:
    subset = [row for row in rows if row["family"] == family and row["arm"] == arm]
    if not subset:
        return float("nan")
    global_reads = sum(int(row["global_all_unit_waveform_reads"]) for row in subset)
    actual_reads = sum(int(row["distinct_waveform_template_reads_total"]) for row in subset)
    if actual_reads == 0:
        return float("inf")
    return float(global_reads / actual_reads)


def _family_point_at_percentile(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    arm: str,
    percentile: float,
) -> dict[str, float] | None:
    subset = [
        row
        for row in rows
        if row["family"] == family
        and row["arm"] == arm
        and float(row["radius_percentile"]) == float(percentile)
    ]
    if not subset:
        return None
    return {
        "radius_percentile": float(percentile),
        "recall": _family_event_weighted_mean(
            subset, family=family, arm=arm, field="true_unit_candidate_recall"
        ),
        "reduction_factor": _family_reduction_factor(subset, family=family, arm=arm),
        "mean_candidate_count": float(
            np.mean([float(row["mean_candidate_count"]) for row in subset])
        ),
    }


def summarize_joint_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build per-family/arm frontier summary and joint recall/traffic gate outcomes."""
    percentiles = sorted({float(row["radius_percentile"]) for row in rows})
    frontier: list[dict[str, Any]] = []
    for arm in PHASE_B_ARMS:
        for family in ("hj", "mearec"):
            for percentile in percentiles:
                point = _family_point_at_percentile(
                    rows, family=family, arm=arm, percentile=percentile
                )
                if point is None:
                    continue
                subset = [
                    row
                    for row in rows
                    if row["arm"] == arm
                    and row["family"] == family
                    and float(row["radius_percentile"]) == percentile
                ]
                frontier.append(
                    {
                        "family": family,
                        "arm": arm,
                        **point,
                        "recordings": [row["dataset"] for row in subset],
                    }
                )

    arm_summaries: dict[str, Any] = {}
    for arm in PHASE_B_ARMS:
        per_family: dict[str, Any] = {}
        for family in ("hj", "mearec"):
            qualifying_points = [
                _family_point_at_percentile(rows, family=family, arm=arm, percentile=p)
                for p in percentiles
            ]
            qualifying_points = [
                point
                for point in qualifying_points
                if point is not None and point["recall"] >= RECALL_GATE
            ]
            tightest_point = None
            best_reduction_point = None
            max_reduction = -1.0
            for point in qualifying_points:
                if tightest_point is None or point["radius_percentile"] < tightest_point["radius_percentile"]:
                    tightest_point = point
                if point["reduction_factor"] > max_reduction:
                    max_reduction = point["reduction_factor"]
                    best_reduction_point = point
            per_family[family] = {
                "best_operating_point_ge_recall_gate": tightest_point,
                "best_reduction_at_recall_ge_gate": best_reduction_point,
                "qualifying_percentile_count": len(qualifying_points),
            }

        joint_pass_percentiles: list[float] = []
        for percentile in percentiles:
            hj_point = _family_point_at_percentile(
                rows, family="hj", arm=arm, percentile=percentile
            )
            mearec_point = _family_point_at_percentile(
                rows, family="mearec", arm=arm, percentile=percentile
            )
            if hj_point is None or mearec_point is None:
                continue
            if (
                hj_point["recall"] >= RECALL_GATE
                and hj_point["reduction_factor"] >= TRAFFIC_REDUCTION_GATE
                and mearec_point["recall"] >= RECALL_GATE
                and mearec_point["reduction_factor"] >= TRAFFIC_REDUCTION_GATE
            ):
                joint_pass_percentiles.append(float(percentile))

        arm_summaries[arm] = {
            "per_family": per_family,
            "joint_pass_percentiles": joint_pass_percentiles,
            "joint_recall_gate": RECALL_GATE,
            "joint_traffic_reduction_gate": TRAFFIC_REDUCTION_GATE,
            "hj_passes_joint_gate": any(
                p is not None
                and p["recall"] >= RECALL_GATE
                and p["reduction_factor"] >= TRAFFIC_REDUCTION_GATE
                for p in [
                    _family_point_at_percentile(rows, family="hj", arm=arm, percentile=percentile)
                    for percentile in percentiles
                ]
            ),
            "mearec_passes_joint_gate": any(
                p is not None
                and p["recall"] >= RECALL_GATE
                and p["reduction_factor"] >= TRAFFIC_REDUCTION_GATE
                for p in [
                    _family_point_at_percentile(rows, family="mearec", arm=arm, percentile=percentile)
                    for percentile in percentiles
                ]
            ),
            "passes_joint_gate_on_both_families": bool(joint_pass_percentiles),
        }

    return {
        "plan": "shape_orthogonal_descriptor_plan.md",
        "phase": "B",
        "candidate_policy": "per_unit_l1_radius_all_rows_active",
        "c_zero_convention": (
            "empty candidate list after per-unit L1 search; zero waveform-template reads; "
            "unified spatial-centroid fallback in candidate_waveform_assign"
        ),
        "distinct_template_reads": (
            "sum over test events of distinct candidate waveform centroids read when |C|>1; "
            "|C|=1 early exit and |C|=0 spatial fallback read zero templates"
        ),
        "full_scan_baseline": "n_eval_events * n_units fitted templates per recording row",
        "recall_gate": RECALL_GATE,
        "traffic_reduction_gate": TRAFFIC_REDUCTION_GATE,
        "frontier_points": frontier,
        "arms": arm_summaries,
    }


def evaluate_recording(
    dataset: Dataset,
    *,
    family: str,
    percentiles: Sequence[float],
) -> list[dict[str, Any]]:
    blocks = build_descriptor_blocks(dataset)
    prep = prepare_dataset_events(
        dataset, k_neighbors=7, window=15, footprint_layout="relative"
    )
    if not np.array_equal(blocks["times"], prep["times"]):
        raise AssertionError("descriptor event rows must match waveform preparation")
    train_idx, test_idx = _chronological_split(blocks["times"])
    rows: list[dict[str, Any]] = []
    for arm in PHASE_B_ARMS:
        for percentile in percentiles:
            metrics = evaluate_arm_candidate_traffic(
                blocks,
                prep["waveforms"],
                train_idx,
                test_idx,
                arm=arm,
                posneg_bits=FROZEN_POSNEG_BITS,
                lat_bits=FROZEN_LAT_BITS,
                amp_gate=FROZEN_AMP_GATE,
                recording_name=dataset.name,
                radius_percentile=float(percentile),
            )
            rows.append(
                {
                    "dataset": dataset.name,
                    "family": family,
                    "scene_type": _scene_type(dataset.name, family),
                    "mearec_n_units": _mearec_n_units(dataset.name, family),
                    "arm": arm,
                    "posneg_bits": FROZEN_POSNEG_BITS,
                    "lat_bits": FROZEN_LAT_BITS,
                    "amp_gate": FROZEN_AMP_GATE,
                    "radius_percentile": float(percentile),
                    **metrics,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape descriptor Phase B candidate traffic")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Override locked development set as family:name pairs",
    )
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTILES),
        help="Per-unit L1 radius percentile sweep",
    )
    args = parser.parse_args()

    percentiles = [_validate_percentile(p) for p in args.percentiles]
    recordings = (
        _parse_recordings_arg(args.recordings)
        if args.recordings is not None
        else list(LOCKED_DEVELOPMENT_RECORDINGS)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for family, dataset in _iter_recordings(recordings, args.duration, Path(args.mearec_npz_dir)):
        if not args.quiet:
            print(f"{family}: {dataset.name}", flush=True)
        rows.extend(evaluate_recording(dataset, family=family, percentiles=percentiles))

    csv_path = out_dir / "shape_candidate_traffic.csv"
    write_csv(csv_path, [{key: row[key] for key in CSV_FIELDNAMES} for row in rows])

    summary = summarize_joint_gates(rows)
    summary["development_recordings"] = sorted({row["dataset"] for row in rows})
    json_path = out_dir / "shape_candidate_traffic_summary.json"
    write_json(json_path, summary)

    if not args.quiet:
        print(f"Wrote {csv_path} ({len(rows)} rows)")
        print(f"Wrote {json_path}")
        for arm in PHASE_B_ARMS:
            gate = summary["arms"][arm]
            print(
                f"{arm}: joint_gate={gate['passes_joint_gate_on_both_families']} "
                f"(hj={gate['hj_passes_joint_gate']}, mearec={gate['mearec_passes_joint_gate']})"
            )


if __name__ == "__main__":
    main()
