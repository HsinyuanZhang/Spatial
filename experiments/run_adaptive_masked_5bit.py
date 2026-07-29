"""Representative masked-L1, 5-bit spatial-row pilot.

This runner is deliberately a static GT-row representation ceiling.  It uses
an equal-timestamp-safe chronological fit/calibration/test split (50/25/25),
learns centroids and masks from fit rows only, and learns radii from
calibration rows only.  ``--pilot`` is a bounded development screen and never
launches a confirmation sweep by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_masked_range_search import AdaptiveMaskedRangeSearch
from Spatial.data.loader import Dataset, list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import (
    WAVEFORM_LENGTH,
    build_support_directory_mask,
    candidate_waveform_assign,
    fit_unit_channel_support,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz


REPO_ROOT = Path(__file__).resolve().parents[1]
BITS = 5
DESCRIPTOR_DIM = 9
WAVEFORM_TEMPLATE_BITS = WAVEFORM_LENGTH * BITS
MASK_MODES = ("all_dims", "stability", "separation_stability")
MASK_SIZES = (3, 5, 7, 9)
RADIUS_PERCENTILES = (95.0, 99.0, 99.9)
DIRECTORY_COVERAGES = (0.95, 0.99, 1.0)
PILOT_HJ_SHORT_NAMES = {"drift16c_600s_11", "static16c_600s_11"}
PILOT_MEAREC_TOKENS = {
    "rec_v1_units10_snr5_seed202601.npz",
    "rec_v1_units20_snr5_seed202601.npz",
}


def stable_chronological_three_way_split(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fit/cal/test indices nearest 50/25/25 without splitting ties.

    ``times`` must already be in stable chronological order.  Candidate split
    points are only between unequal timestamps.  The selected pair minimises
    squared deviation from the requested 50% and 75% cumulative boundaries;
    lexicographic ordering makes any exact tie deterministic and earlier.
    """
    values = np.asarray(times, dtype=np.int64).ravel()
    if values.size < 3:
        raise ValueError("at least three events are required for fit/cal/test")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("times must be stably sorted before chronological splitting")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    if boundaries.size < 2:
        raise ValueError("cannot form three partitions without splitting equal timestamps")
    n = values.size
    # For every legal first boundary, the best second boundary is one of the
    # two points adjacent to 75%.  That keeps this equal-timestamp-safe choice
    # O(number of timestamp groups), not O(events²).
    best: tuple[float, int, int] | None = None
    target_second = 0.75 * n
    for first_pos, first in enumerate(boundaries[:-1]):
        later = boundaries[first_pos + 1 :]
        right = int(np.searchsorted(later, target_second, side="left"))
        for second_pos in (right - 1, right):
            if not 0 <= second_pos < later.size:
                continue
            second = int(later[second_pos])
            loss = (float(first) - 0.50 * n) ** 2 + (float(second) - target_second) ** 2
            candidate = (loss, int(first), second)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, fit_stop, cal_stop = best
    return np.arange(fit_stop), np.arange(fit_stop, cal_stop), np.arange(cal_stop, n)


def stable_chronological_two_way_split(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-timestamp-safe early/late halves for a fit-only audit."""
    values = np.asarray(times, dtype=np.int64).ravel()
    if values.size < 2:
        raise ValueError("at least two rows are required for an early/late audit")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("times must be stably sorted before chronological splitting")
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    if boundaries.size == 0:
        raise ValueError("cannot split equal timestamps across early/late fit halves")
    split = int(boundaries[np.argmin(np.abs(boundaries - values.size / 2.0))])
    return np.arange(split), np.arange(split, values.size)


def masked_grid() -> list[tuple[str, int]]:
    """Frozen grid: one exact all-dimension row plus non-all fixed masks."""
    return [("all_dims", 9)] + [
        (mode, size)
        for mode in ("stability", "separation_stability")
        for size in MASK_SIZES
    ]


def validate_hardware_contract(bits: int, descriptor_dim: int) -> None:
    if int(bits) != BITS:
        raise ValueError("masked representative pilot is frozen at B=5")
    if int(descriptor_dim) != DESCRIPTOR_DIM:
        raise ValueError("masked representative pilot is frozen at D=9")


def validate_confirmation_configuration(
    mask_mode: str | None, mask_size: int | None, radius_percentile: float | None
) -> tuple[str, int, float]:
    """Require one pre-registered all-row configuration before confirmation."""
    if mask_mode is None or mask_size is None or radius_percentile is None:
        raise ValueError("confirmation requires mask mode, mask size, and radius percentile")
    config = (str(mask_mode), int(mask_size))
    if config not in masked_grid():
        raise ValueError("frozen all_dims requires M=9; masked sizes must be in the frozen grid")
    percentile = float(radius_percentile)
    if percentile not in RADIUS_PERCENTILES:
        raise ValueError("frozen radius percentile must be one of 95, 99, or 99.9")
    return config[0], config[1], percentile


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def unit_directory_pointer_accounting(pointer_counts: np.ndarray, n_units: int) -> dict[str, int]:
    """Count only row-ID payload bits, excluding periphery/valid/decode logic."""
    counts = np.asarray(pointer_counts, dtype=np.int64).ravel()
    if int(n_units) < 1 or np.any(counts < 0):
        raise ValueError("pointer counts must be nonnegative and units positive")
    row_id_bits = int(math.ceil(math.log2(int(n_units)))) if int(n_units) > 1 else 0
    total = int(counts.sum())
    return {
        "directory_row_id_pointers_total": total,
        "directory_row_id_pointer_bits_per_entry": row_id_bits,
        "directory_row_id_pointer_bits_total": total * row_id_bits,
    }


def directory_true_unit_recall(active: np.ndarray | None, labels: np.ndarray, units: np.ndarray) -> float:
    target = np.asarray(labels)
    row_units = np.asarray(units)
    if active is None:
        matrix = np.ones((target.size, row_units.size), dtype=bool)
    else:
        matrix = np.asarray(active, dtype=bool)
    if matrix.shape != (target.size, row_units.size):
        raise ValueError("active directory mask must be (events, units)")
    hits = np.asarray([np.any(matrix[i] & (row_units == label)) for i, label in enumerate(target)])
    return float(hits.mean()) if hits.size else 0.0


def conditional_candidate_recall_given_true_row_active(
    candidates: Sequence[np.ndarray], labels: np.ndarray, active: np.ndarray | None, units: np.ndarray
) -> float:
    """Candidate recall conditional on a directory having activated the true row.

    Unseen test labels do not own an all-row entry and therefore cannot be
    silently treated as directory-active.
    """
    target = np.asarray(labels)
    row_units = np.asarray(units)
    matrix = np.ones((target.size, row_units.size), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if matrix.shape != (target.size, row_units.size):
        raise ValueError("active directory mask must be (events, units)")
    true_active = np.asarray([np.any(matrix[i] & (row_units == label)) for i, label in enumerate(target)])
    if not np.any(true_active):
        return 0.0
    return float(np.mean([target[i] in candidates[i] for i in np.flatnonzero(true_active)]))


def per_unit_recall_summary(candidates: Sequence[np.ndarray], labels: np.ndarray) -> dict[str, float | int]:
    """Test-only recall distribution, retaining labels absent from candidates."""
    target = np.asarray(labels)
    if not target.size:
        return {
            "per_unit_recall_min": 0.0, "per_unit_recall_p05": 0.0,
            "per_unit_recall_p10": 0.0, "per_unit_recall_worst_decile": 0.0,
            "per_unit_recall_unit_count": 0, "per_unit_test_events_min": 0,
            "per_unit_test_events_max": 0,
        }
    unit_labels: list[Any] = []
    recalls: list[float] = []
    event_counts: list[int] = []
    for unit in np.unique(target):
        local = target == unit
        unit_labels.append(_scalar(unit))
        event_counts.append(int(local.sum()))
        recalls.append(float(np.mean([unit in candidates[i] for i in np.flatnonzero(local)])))
    ordered = np.sort(np.asarray(recalls, dtype=float))
    worst_count = max(1, int(math.ceil(ordered.size / 10.0)))
    return {
        "per_unit_recall_min": float(ordered[0]),
        "per_unit_recall_p05": float(np.percentile(ordered, 5)),
        "per_unit_recall_p10": float(np.percentile(ordered, 10)),
        "per_unit_recall_worst_decile": float(ordered[:worst_count].mean()),
        "per_unit_recall_unit_count": int(ordered.size),
        "per_unit_test_events_min": int(min(event_counts)),
        "per_unit_test_events_max": int(max(event_counts)),
        "per_unit_label_json": json.dumps(unit_labels, separators=(",", ":")),
        "per_unit_recall_json": json.dumps(recalls, separators=(",", ":")),
        "per_unit_test_event_count_json": json.dumps(event_counts, separators=(",", ":")),
    }


def true_unit_masked_distance_summary(
    codes: np.ndarray,
    labels: np.ndarray,
    model: AdaptiveMaskedRangeSearch,
) -> dict[str, float | int]:
    """Summarize test distance and radius at each fitted true-unit row."""
    values = np.asarray(codes, dtype=np.int64)
    target = np.asarray(labels)
    if values.ndim != 2 or values.shape != (target.size, DESCRIPTOR_DIM):
        raise ValueError("codes and labels must align with the fixed nine-dimensional descriptor")
    unit_rows = {_scalar(unit): row for row, unit in enumerate(model.units_)}
    distances: list[int] = []
    radii: list[int] = []
    for query, label in zip(values, target):
        row = unit_rows.get(_scalar(label))
        if row is None:
            continue
        distances.append(int((np.abs(query - model.centroids_[row]) * model.masks_[row]).sum()))
        radii.append(int(model.radii_[row]))
    unseen = int(target.size - len(distances))
    if not distances:
        return {
            "true_unit_distance_rows": 0,
            "true_unit_distance_unseen_label_rows": unseen,
            "true_unit_masked_l1_distance_mean": float("nan"),
            "true_unit_masked_l1_distance_p50": float("nan"),
            "true_unit_masked_l1_distance_p95": float("nan"),
            "true_unit_masked_l1_distance_p99": float("nan"),
            "true_unit_masked_l1_distance_max": 0,
            "true_unit_radius_mean": float("nan"),
            "true_unit_radius_p50": float("nan"),
            "true_unit_radius_p95": float("nan"),
            "true_unit_radius_max": 0,
            "true_unit_distance_minus_radius_mean": float("nan"),
            "true_unit_distance_minus_radius_p95": float("nan"),
            "true_unit_distance_minus_radius_max": 0,
            "true_unit_distance_over_radius_plus_one_mean": float("nan"),
            "true_unit_distance_over_radius_plus_one_p95": float("nan"),
            "true_unit_distance_le_radius_fraction": float("nan"),
        }
    distance_values = np.asarray(distances, dtype=np.int64)
    radius_values = np.asarray(radii, dtype=np.int64)
    margin = distance_values - radius_values
    ratio = distance_values / (radius_values + 1.0)
    return {
        "true_unit_distance_rows": int(distance_values.size),
        "true_unit_distance_unseen_label_rows": unseen,
        "true_unit_masked_l1_distance_mean": float(distance_values.mean()),
        "true_unit_masked_l1_distance_p50": float(np.percentile(distance_values, 50, method="linear")),
        "true_unit_masked_l1_distance_p95": float(np.percentile(distance_values, 95, method="linear")),
        "true_unit_masked_l1_distance_p99": float(np.percentile(distance_values, 99, method="linear")),
        "true_unit_masked_l1_distance_max": int(distance_values.max()),
        "true_unit_radius_mean": float(radius_values.mean()),
        "true_unit_radius_p50": float(np.percentile(radius_values, 50, method="linear")),
        "true_unit_radius_p95": float(np.percentile(radius_values, 95, method="linear")),
        "true_unit_radius_max": int(radius_values.max()),
        "true_unit_distance_minus_radius_mean": float(margin.mean()),
        "true_unit_distance_minus_radius_p95": float(np.percentile(margin, 95, method="linear")),
        "true_unit_distance_minus_radius_max": int(margin.max()),
        "true_unit_distance_over_radius_plus_one_mean": float(ratio.mean()),
        "true_unit_distance_over_radius_plus_one_p95": float(np.percentile(ratio, 95, method="linear")),
        "true_unit_distance_le_radius_fraction": float(np.mean(distance_values <= radius_values)),
    }


def wilson_binomial_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion.

    The default normal quantile gives a nominal 95% interval.  The method and
    quantile are fixed so every per-record candidate-recall row is directly
    reproducible without an optional statistics dependency.
    """
    hits, count = int(successes), int(total)
    if count < 1:
        raise ValueError("Wilson interval requires at least one trial")
    if hits < 0 or hits > count:
        raise ValueError("successes must lie in [0, total]")
    z_value = float(z)
    if not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z must be finite and positive")
    proportion = hits / count
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / count
    centre = (proportion + z_squared / (2.0 * count)) / denominator
    half_width = z_value * math.sqrt(
        proportion * (1.0 - proportion) / count + z_squared / (4.0 * count * count)
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def level1_masked_accounting(
    primary_rows: np.ndarray,
    primary_lanes: np.ndarray,
    fallback_rows: np.ndarray,
    *,
    descriptor_dim: int = DESCRIPTOR_DIM,
) -> dict[str, float | int | bool]:
    """Account primary masked comparisons and full-D empty-list fallback scans."""
    rows = np.asarray(primary_rows, dtype=np.int64).ravel()
    lanes = np.asarray(primary_lanes, dtype=np.int64).ravel()
    fallback = np.asarray(fallback_rows, dtype=np.int64).ravel()
    if rows.shape != lanes.shape or rows.shape != fallback.shape:
        raise ValueError("row, lane, and fallback arrays must align")
    if np.any(rows < 0) or np.any(lanes < 0) or np.any(fallback < 0):
        raise ValueError("comparison accounting must be nonnegative")
    fallback_lanes = fallback * int(descriptor_dim)
    total_rows = rows + fallback
    total_lanes = lanes + fallback_lanes
    primary_additions = lanes - rows
    fallback_additions = fallback_lanes - fallback
    total_additions = total_lanes - total_rows
    n = rows.size
    return {
        "primary_level1_row_comparisons_total": int(rows.sum()),
        "primary_enabled_abs_lanes_total": int(lanes.sum()),
        "primary_enabled_abs_additions_total": int(primary_additions.sum()),
        "empty_fallback_row_comparisons_total": int(fallback.sum()),
        "empty_fallback_full_d_abs_lanes_total": int(fallback_lanes.sum()),
        "empty_fallback_full_d_abs_additions_total": int(fallback_additions.sum()),
        "total_level1_row_comparisons": int(total_rows.sum()),
        "total_abs_lanes": int(total_lanes.sum()),
        "total_abs_additions": int(total_additions.sum()),
        "mean_active_rows": float(rows.mean()) if n else 0.0,
        "p95_active_rows": float(np.percentile(rows, 95)) if n else 0.0,
        "max_active_rows": int(rows.max()) if n else 0,
        "mean_primary_enabled_abs_lanes": float(lanes.mean()) if n else 0.0,
        "p95_primary_enabled_abs_lanes": float(np.percentile(lanes, 95)) if n else 0.0,
        "max_primary_enabled_abs_lanes": int(lanes.max()) if n else 0,
        "mean_total_level1_rows": float(total_rows.mean()) if n else 0.0,
        "p95_total_level1_rows": float(np.percentile(total_rows, 95)) if n else 0.0,
        "max_total_level1_rows": int(total_rows.max()) if n else 0,
        "mean_total_abs_lanes": float(total_lanes.mean()) if n else 0.0,
        "p95_total_abs_lanes": float(np.percentile(total_lanes, 95)) if n else 0.0,
        "max_total_abs_lanes": int(total_lanes.max()) if n else 0,
        "level1_row_accounting_identity_holds": bool(np.array_equal(total_rows, rows + fallback)),
        "level1_lane_accounting_identity_holds": bool(np.array_equal(total_lanes, lanes + fallback_lanes)),
        "primary_addition_accounting_identity_holds": bool(np.array_equal(primary_additions, lanes - rows)),
        "fallback_addition_accounting_identity_holds": bool(np.array_equal(fallback_additions, fallback * (int(descriptor_dim) - 1))),
        "total_addition_accounting_identity_holds": bool(np.array_equal(total_additions, total_lanes - total_rows)),
    }


def candidate_category_level1_accounting(
    candidate_counts: np.ndarray,
    primary_rows: np.ndarray,
    primary_lanes: np.ndarray,
    fallback_rows: np.ndarray,
    *,
    descriptor_dim: int = DESCRIPTOR_DIM,
) -> dict[str, int]:
    """Partition rows/lanes/additions by final candidate cardinality C."""
    counts = np.asarray(candidate_counts, dtype=np.int64).ravel()
    rows = np.asarray(primary_rows, dtype=np.int64).ravel()
    lanes = np.asarray(primary_lanes, dtype=np.int64).ravel()
    fallback = np.asarray(fallback_rows, dtype=np.int64).ravel()
    if not (counts.shape == rows.shape == lanes.shape == fallback.shape):
        raise ValueError("candidate and level-1 accounting arrays must align")
    output: dict[str, int] = {}
    for name, membership in (("c0", counts == 0), ("c1", counts == 1), ("cgt1", counts > 1)):
        fallback_lanes = fallback[membership] * int(descriptor_dim)
        total_rows = rows[membership] + fallback[membership]
        total_lanes = lanes[membership] + fallback_lanes
        output[f"{name}_events"] = int(membership.sum())
        output[f"{name}_total_level1_rows"] = int(total_rows.sum())
        output[f"{name}_total_abs_lanes"] = int(total_lanes.sum())
        output[f"{name}_total_abs_additions"] = int((total_lanes - total_rows).sum())
    return output


def mask_jaccard_summary(
    features: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    mask_mode: str,
    mask_size: int,
) -> dict[str, float | int]:
    """Fit-only early/late mask Jaccard, excluding units absent from either half."""
    early, late = stable_chronological_two_way_split(np.asarray(times))
    if early.size == 0 or late.size == 0:
        return {"early_late_fit_mask_jaccard": float("nan"), "early_late_fit_mask_jaccard_valid_units": 0}
    left = AdaptiveMaskedRangeSearch(n_bits=BITS).fit(
        np.asarray(features)[early], np.asarray(labels)[early], mask_size=mask_size, mask_mode=mask_mode
    )
    right = AdaptiveMaskedRangeSearch(n_bits=BITS).fit(
        np.asarray(features)[late], np.asarray(labels)[late], mask_size=mask_size, mask_mode=mask_mode
    )
    left_map = {_scalar(unit): row for row, unit in enumerate(left.units_)}
    right_map = {_scalar(unit): row for row, unit in enumerate(right.units_)}
    common = sorted(set(left_map).intersection(right_map), key=repr)
    if not common:
        return {"early_late_fit_mask_jaccard": float("nan"), "early_late_fit_mask_jaccard_valid_units": 0}
    values = []
    for unit in common:
        a = left.masks_[left_map[unit]]
        b = right.masks_[right_map[unit]]
        values.append(float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum()))
    return {
        "early_late_fit_mask_jaccard": float(np.mean(values)),
        "early_late_fit_mask_jaccard_valid_units": int(len(values)),
    }


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    return {_scalar(unit): np.asarray(waveforms)[np.asarray(labels) == unit].mean(axis=0) for unit in units}


def _mask_fields(masks: np.ndarray) -> dict[str, float | int]:
    table = np.asarray(masks, dtype=bool)
    enabled = table.sum(axis=1, dtype=np.int64)
    result: dict[str, float | int] = {
        "enabled_dimensions_mean": float(enabled.mean()) if enabled.size else 0.0,
        "enabled_dimensions_p95": float(np.percentile(enabled, 95)) if enabled.size else 0.0,
        "enabled_dimensions_max": int(enabled.max()) if enabled.size else 0,
    }
    for dim, count in enumerate(table.sum(axis=0, dtype=np.int64)):
        result[f"mask_dimension_{dim}_selected_rows"] = int(count)
        result[f"mask_dimension_{dim}_selection_frequency"] = float(count / max(table.shape[0], 1))
    return result


def _mask_score_fields(model: AdaptiveMaskedRangeSearch) -> dict[str, float | str]:
    """Report frozen fit-only score arrays directly from the search core."""
    output = {
        "fit_within_q95_mean": float(np.mean(model.within_q95_)),
        "fit_within_q95_p95": float(np.percentile(model.within_q95_, 95)),
        "fit_between_median_separation_mean": float(np.mean(model.between_median_separation_)),
        "fit_between_median_separation_p95": float(np.percentile(model.between_median_separation_, 95)),
        "fit_separation_stability_score_mean": float(np.mean(model.separation_stability_scores_)),
        "fit_separation_stability_score_p95": float(np.percentile(model.separation_stability_scores_, 95)),
        "fit_unit_labels_json": json.dumps(
            [_scalar(unit) for unit in model.units_], separators=(",", ":")
        ),
        "fit_masks_by_unit_dimension_json": json.dumps(
            model.masks_.astype(np.int64).tolist(), separators=(",", ":")
        ),
        "fit_within_q95_by_unit_dimension_json": json.dumps(
            model.within_q95_.tolist(), separators=(",", ":")
        ),
        "fit_between_median_separation_by_unit_dimension_json": json.dumps(
            model.between_median_separation_.tolist(), separators=(",", ":")
        ),
        "fit_separation_stability_score_by_unit_dimension_json": json.dumps(
            model.separation_stability_scores_.tolist(), separators=(",", ":")
        ),
    }
    for dim in range(DESCRIPTOR_DIM):
        output[f"fit_within_q95_dimension_{dim}_mean"] = float(np.mean(model.within_q95_[:, dim]))
        output[f"fit_between_median_separation_dimension_{dim}_mean"] = float(np.mean(model.between_median_separation_[:, dim]))
        output[f"fit_separation_stability_score_dimension_{dim}_mean"] = float(np.mean(model.separation_stability_scores_[:, dim]))
    return output


def _candidate_metrics(
    candidates: Sequence[np.ndarray],
    labels: np.ndarray,
    pred: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    wave_meta: dict[str, np.ndarray],
    n_units: int,
) -> dict[str, Any]:
    counts = np.asarray([len(row) for row in candidates], dtype=np.int64)
    target = np.asarray(labels)
    true_hit = np.asarray([label in row for label, row in zip(target, candidates)], dtype=bool)
    reads = np.asarray(wave_meta["candidate_waveform_reads"], dtype=np.int64)
    fallback_rows = np.asarray(wave_meta["spatial_fallback_row_comparisons"], dtype=np.int64)
    global_bits = int(target.size) * int(n_units) * WAVEFORM_TEMPLATE_BITS
    actual_bits = int(reads.sum()) * WAVEFORM_TEMPLATE_BITS
    candidate_hits = int(true_hit.sum())
    recall_low, recall_high = wilson_binomial_interval(candidate_hits, int(target.size))
    return {
        "true_unit_candidate_recall": float(true_hit.mean()) if target.size else 0.0,
        "true_unit_candidate_hits": candidate_hits,
        "candidate_recall_interval_method": "two_sided_wilson_95pct_z1p959963984540054",
        "candidate_recall_wilson95_low": recall_low,
        "candidate_recall_wilson95_high": recall_high,
        "zero_fraction": float(np.mean(counts == 0)) if target.size else 0.0,
        "unique_fraction": float(np.mean(counts == 1)) if target.size else 0.0,
        "ambiguous_fraction": float(np.mean(counts > 1)) if target.size else 0.0,
        "mean_candidates": float(counts.mean()) if target.size else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if target.size else 0.0,
        "max_candidates": int(counts.max()) if counts.size else 0,
        "candidate_waveform_reads_total": int(reads.sum()),
        "actual_waveform_template_bits_read": actual_bits,
        "global_all_unit_waveform_template_bits": global_bits,
        "waveform_bit_traffic_reduction_factor": float(global_bits / actual_bits) if actual_bits else float("inf"),
        "meets_final_candidate_recall_0p99_and_traffic_4x": bool(
            (float(true_hit.mean()) if target.size else 0.0) >= 0.99
            and (float(global_bits / actual_bits) if actual_bits else float("inf")) >= 4.0
        ),
        "waveform_fallback_fraction": float(np.mean(wave_meta["waveform_fallback"])) if target.size else 0.0,
        "final_float_waveform_teacher_accuracy": float(np.mean(np.asarray(pred) == target)) if target.size else 0.0,
        "waveform_accuracy_teacher": "float64_l1_candidate_only_unique_early_exit",
        "waveform_template_bits_per_candidate": WAVEFORM_TEMPLATE_BITS,
        **per_unit_recall_summary(candidates, target),
        **level1_masked_accounting(
            diagnostics["row_comparisons"], diagnostics["enabled_abs_lanes"], fallback_rows
        ),
        **candidate_category_level1_accounting(
            counts, diagnostics["row_comparisons"], diagnostics["enabled_abs_lanes"], fallback_rows
        ),
    }


def _directory_configs(
    *,
    include_support: bool,
    model: AdaptiveMaskedRangeSearch,
    fit_labels: np.ndarray,
    fit_central: np.ndarray,
    test_central: np.ndarray,
    dataset: Dataset,
    directory_coverages: Sequence[float],
    k_neighbors: int,
) -> list[tuple[str, float | None, np.ndarray | None, np.ndarray]]:
    all_rows = [("all_rows", None, None, np.zeros(model.units_.size, dtype=np.int64))]
    if not include_support:
        return all_rows
    output = all_rows
    for coverage in dict.fromkeys(float(value) for value in directory_coverages):
        if not 0.0 < coverage <= 1.0:
            raise ValueError("directory coverages must lie in (0, 1]")
        support = fit_unit_channel_support(
            fit_labels, fit_central, model.units_, dataset.n_channels, coverage
        )
        output.append((
            "unit_support", coverage,
            build_support_directory_mask(
                test_central, support, dataset.geom, min(int(k_neighbors), dataset.n_channels)
            ),
            support.sum(axis=1, dtype=np.int64),
        ))
    return output


def run_dataset_masked(
    dataset: Dataset,
    *,
    bits: int = BITS,
    radius_percentiles: Sequence[float] = RADIUS_PERCENTILES,
    directory_coverages: Sequence[float] = DIRECTORY_COVERAGES,
    include_support: bool = True,
    mask_configurations: Sequence[tuple[str, int]] | None = None,
    selection_protocol: str = "exploratory_test_grid_no_final_configuration_selection",
    k_neighbors: int = 7,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the frozen masked grid without allowing test rows to tune a model."""
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    fit_idx, cal_idx, test_idx = stable_chronological_three_way_split(prepared["times"])
    features, labels = prepared["descriptor"], prepared["labels"]
    validate_hardware_contract(bits, features.shape[1])
    x_fit, y_fit = features[fit_idx], labels[fit_idx]
    x_cal, y_cal = features[cal_idx], labels[cal_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    if verbose:
        print(f"{dataset.name}: fit/cal/test={fit_idx.size}/{cal_idx.size}/{test_idx.size}, D={features.shape[1]}")
    rows: list[dict[str, Any]] = []
    for mask_mode, mask_size in (masked_grid() if mask_configurations is None else list(mask_configurations)):
        jaccard = mask_jaccard_summary(
            x_fit, y_fit, prepared["times"][fit_idx], mask_mode=mask_mode, mask_size=mask_size
        )
        for percentile in dict.fromkeys(float(value) for value in radius_percentiles):
            if not 0.0 < percentile <= 100.0:
                raise ValueError("radius percentiles must lie in (0, 100]")
            model = AdaptiveMaskedRangeSearch(n_bits=BITS, radius_percentile=percentile).fit(
                x_fit, y_fit, mask_size=mask_size, mask_mode=mask_mode
            ).calibrate(x_cal, y_cal)
            codes_test = model.transform(x_test)
            true_unit_distance = true_unit_masked_distance_summary(codes_test, y_test, model)
            calibration_candidates, _ = model.query_codes(model.transform(x_cal))
            calibration_true_hits = np.asarray(
                [label in candidate for label, candidate in zip(y_cal, calibration_candidates)], dtype=bool
            )
            wave_centroids = _waveform_centroids(prepared["waveforms"][fit_idx], y_fit, model.units_)
            calibration_counts = np.asarray(model.radius_calibration_counts_, dtype=np.int64)
            calibration_fallback = np.asarray(model.radius_calibration_fallback_mask_, dtype=bool)
            radii = np.asarray(model.radii_, dtype=np.int64)
            storage = model.memory_accounting()
            for strategy, coverage, active_mask, pointers in _directory_configs(
                include_support=include_support, model=model, fit_labels=y_fit,
                fit_central=prepared["central"][fit_idx], test_central=prepared["central"][test_idx],
                dataset=dataset, directory_coverages=directory_coverages, k_neighbors=k_neighbors,
            ):
                candidates, diagnostics = model.query_codes(codes_test, active_row_mask=active_mask)
                pred, wave_meta = candidate_waveform_assign(
                    candidates, prepared["waveforms"][test_idx], wave_centroids,
                    codes_test, model.centroids_, model.units_, metric="l1",
                )
                metrics = _candidate_metrics(candidates, y_test, pred, diagnostics, wave_meta, model.units_.size)
                rows.append({
                    "dataset": dataset.name,
                    "n_channels": int(dataset.n_channels), "n_units": int(model.units_.size),
                    "n_events_gt_rows": int(labels.size), "n_fit_gt_rows": int(fit_idx.size),
                    "n_calibration_gt_rows": int(cal_idx.size), "n_test_gt_rows": int(test_idx.size),
                    "bits": BITS, "descriptor_dim": DESCRIPTOR_DIM, "k_neighbors": int(min(k_neighbors, dataset.n_channels)),
                    "offline_isolation": "gt_event_rows_equal_timestamp_safe_chronological_fit50_cal25_test25",
                    "preprocessing": "full_recording_zero_phase_filtfilt_offline",
                    "gt_protocol": "static_representation_ceiling_gt_fit_cal_rows_test_labels_metrics_only",
                    "selection_protocol": selection_protocol,
                    "mask_mode": mask_mode, "mask_size": int(mask_size),
                    "radius_percentile": float(percentile),
                    "radius_rule": "ceil_percentile_masked_l1_calibration_radius_zero_legal",
                    "directory_strategy": strategy, "directory_coverage": "" if coverage is None else float(coverage),
                    "directory_neighbors": int(min(k_neighbors, dataset.n_channels)),
                    "search_payload_bits_per_row": int(storage["search_payload_bits_per_row"]),
                    "unit_id_bits_per_row": int(storage["unit_id_bits_per_row"]),
                    "valid_bits_per_row": int(storage["valid_bits_per_row"]),
                    "row_bits_with_metadata_per_row": int(storage["row_bits_with_metadata_per_row"]),
                    "search_payload_layout": "45bit_centroid_9bit_mask_9bit_full_scale_radius",
                    "configuration_centroid_bits_written": int(model.units_.size * DESCRIPTOR_DIM * BITS),
                    "configuration_mask_bits_written": int(model.units_.size * DESCRIPTOR_DIM),
                    "configuration_radius_bits_written": int(model.units_.size * DESCRIPTOR_DIM),
                    "configuration_write_bits_total": int(model.units_.size * 63),
                    "radius_calibration_count_total": int(calibration_counts.sum()),
                    "radius_calibration_count_mean": float(calibration_counts.mean()),
                    "radius_calibration_count_min": int(calibration_counts.min()),
                    "radius_calibration_count_max": int(calibration_counts.max()),
                    "radius_calibration_missing_unit_count": int(calibration_fallback.sum()),
                    "radius_calibration_coverage_fraction": float(np.mean(~calibration_fallback)),
                    "calibration_true_unit_masked_coverage": float(calibration_true_hits.mean()) if y_cal.size else 0.0,
                    "radius_zero_count": int(np.sum(radii == 0)),
                    "test_labels_absent_from_fit_events": int(np.sum(~np.isin(y_test, model.units_))),
                    "test_labels_absent_from_fit_units": int(np.unique(y_test[~np.isin(y_test, model.units_)]).size),
                    "empty_fallback_search": "existing_unmasked_all_centroid_l1",
                    "empty_fallback_lanes_per_row": DESCRIPTOR_DIM,
                    "directory_true_unit_row_activation_recall": directory_true_unit_recall(active_mask, y_test, model.units_),
                    "conditional_masked_candidate_recall_given_true_row_active": conditional_candidate_recall_given_true_row_active(
                        candidates, y_test, active_mask, model.units_
                    ),
                    "directory_row_id_pointer_accounting_scope": "row_id_payload_only_excludes_periphery_valid_decode",
                    **unit_directory_pointer_accounting(pointers, model.units_.size),
                    **_mask_fields(model.masks_), **_mask_score_fields(model),
                    **true_unit_distance, **jaccard, **metrics,
                })
    return rows


def _event_weighted_family(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, float]]:
    """Aggregate all rows by configuration with test-event weighting."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mask_mode"], int(row["mask_size"]), float(row["radius_percentile"]),
                 row["directory_strategy"], row["directory_coverage"])].append(row)
    result: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, bucket in grouped.items():
        n = sum(int(row["n_test_gt_rows"]) for row in bucket)
        hits = sum(int(row["true_unit_candidate_hits"]) for row in bucket)
        actual = sum(int(row["actual_waveform_template_bits_read"]) for row in bucket)
        global_bits = sum(int(row["global_all_unit_waveform_template_bits"]) for row in bucket)
        result[key] = {
            "recall": float(hits / n) if n else 0.0,
            "traffic_reduction": float(global_bits / actual) if actual else float("inf"),
            "recordings": float(len(bucket)),
        }
    return result


def pilot_gate(rows_by_family: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Apply the pre-registered all-row development gate without selecting test data.

    Each candidate must first be within the per-record tolerance versus its
    same-radius all-dimension baseline.  It then needs the *same* aggregate
    branch in HJ and MEArec: either +0.005 recall at no traffic loss (A), or
    +10% traffic reduction at no recall loss (B).  The function reports the
    frozen tie-break winner but never launches another experiment.
    """
    required = {"hj", "mearec"}
    if set(rows_by_family) != required:
        raise ValueError("pilot gate requires exactly hj and mearec family rows")
    family_aggregates = {family: _event_weighted_family(rows) for family, rows in rows_by_family.items()}
    candidates: list[dict[str, Any]] = []
    configurations = set.intersection(*(set(values) for values in family_aggregates.values()))
    for config in configurations:
        mode, size, percentile, strategy, coverage = config
        if strategy != "all_rows" or mode == "all_dims" or int(size) >= 9:
            continue
        baseline = ("all_dims", 9, percentile, strategy, coverage)
        if any(baseline not in family_aggregates[family] for family in required):
            continue
        record_ok = True
        deltas: dict[str, float] = {}
        ratios: dict[str, float] = {}
        for family in required:
            for row in rows_by_family[family]:
                if (row["mask_mode"], int(row["mask_size"]), float(row["radius_percentile"]),
                    row["directory_strategy"], row["directory_coverage"]) != config:
                    continue
                match = next((base for base in rows_by_family[family] if (
                    base["mask_mode"], int(base["mask_size"]), float(base["radius_percentile"]),
                    base["directory_strategy"], base["directory_coverage"]
                ) == baseline and base["dataset"] == row["dataset"]), None)
                if match is None:
                    record_ok = False
                    break
                if float(row["true_unit_candidate_recall"]) < float(match["true_unit_candidate_recall"]) - 0.005:
                    record_ok = False
                    break
                if float(row["waveform_bit_traffic_reduction_factor"]) < 0.95 * float(match["waveform_bit_traffic_reduction_factor"]):
                    record_ok = False
                    break
            candidate = family_aggregates[family][config]
            base = family_aggregates[family][baseline]
            deltas[family] = candidate["recall"] - base["recall"]
            ratios[family] = candidate["traffic_reduction"] / base["traffic_reduction"]
        branch_a = all(deltas[family] >= 0.005 and ratios[family] >= 1.0 for family in required)
        branch_b = all(ratios[family] >= 1.10 and deltas[family] >= 0.0 for family in required)
        if record_ok and (branch_a or branch_b):
            candidates.append({
                "config": config, "branch": "A_recall" if branch_a else "B_traffic",
                "family_recall_delta": deltas, "family_traffic_ratio": ratios,
            })
    order = {"separation_stability": 0, "stability": 1}
    candidates.sort(key=lambda entry: (
        -min(entry["family_recall_delta"].values()),
        -min(entry["family_traffic_ratio"].values()),
        int(entry["config"][1]), order[entry["config"][0]], float(entry["config"][2]),
    ))
    return {
        "pilot_gate_pass": bool(candidates),
        "pilot_gate_scope": "all_rows_only_development_recordings",
        "pilot_gate_candidates": int(len(candidates)),
        "pilot_gate_selected": candidates[0] if candidates else None,
        "family_event_weighted": family_aggregates,
    }


def final_099_4x_report(rows_by_family: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Report, without selecting, configurations reaching the final joint target."""
    aggregates = {family: _event_weighted_family(rows) for family, rows in rows_by_family.items()}
    common = set.intersection(*(set(values) for values in aggregates.values())) if aggregates else set()
    qualifying = [
        config for config in common
        if all(aggregates[family][config]["recall"] >= 0.99 and aggregates[family][config]["traffic_reduction"] >= 4.0
               for family in aggregates)
    ]
    return {
        "final_0p99_4x_scope": "event_weighted_family_aggregate_report_only",
        "final_0p99_4x_pass": bool(qualifying),
        "final_0p99_4x_qualifying_configurations": int(len(qualifying)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _iter_hj(duration: float, *, pilot: bool, confirmation: bool) -> Iterable[Dataset]:
    for scene in list_hybrid_janelia_scenes(ready_only=True):
        short = str(scene.get("short_name", ""))
        if pilot and short not in PILOT_HJ_SHORT_NAMES:
            continue
        if confirmation and short in PILOT_HJ_SHORT_NAMES:
            continue
        yield load_hybrid_janelia(scene=short, duration_s=duration)


def _iter_mearec(duration: float, npz_dir: Path, *, pilot: bool, confirmation: bool) -> Iterable[Dataset]:
    for path in discover_npz(npz_dir, seeds=None):
        is_pilot = path.name in PILOT_MEAREC_TOKENS
        if pilot and not is_pilot:
            continue
        if confirmation and is_pilot:
            continue
        yield load_mearec_npz(path, duration_s=duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked-L1 representative 5-bit spatial-row experiment")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default=None)
    parser.add_argument("--pilot", action="store_true", help="run only the four fixed development recordings, all_rows only")
    parser.add_argument("--confirmation", action="store_true", help="exclude all four development recordings")
    parser.add_argument("--include-unit-support", action="store_true", help="opt in to conditional unit-support directory rows outside pilot/confirmation")
    parser.add_argument("--freeze-mask-mode", choices=MASK_MODES, default=None, help="required with --confirmation")
    parser.add_argument("--freeze-mask-size", type=int, choices=MASK_SIZES, default=None, help="required with --confirmation")
    parser.add_argument("--freeze-radius-percentile", type=float, default=None, help="required with --confirmation")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "adaptive_masked_5bit")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pilot and args.confirmation:
        parser.error("--pilot and --confirmation are mutually exclusive")
    if not args.pilot and args.family is None:
        parser.error("choose --pilot or explicitly provide --family; no full sweep is automatic")
    if args.pilot and args.family not in (None, "all"):
        parser.error("--pilot always requires both HJ and MEArec development families")
    if (args.pilot or args.confirmation) and args.include_unit_support:
        parser.error("pilot and confirmation are all_rows only; unit-support is not eligible")
    frozen_configuration: list[tuple[str, int]] | None = None
    frozen_percentiles: tuple[float, ...] = RADIUS_PERCENTILES
    if args.confirmation:
        try:
            mode, size, percentile = validate_confirmation_configuration(
                args.freeze_mask_mode, args.freeze_mask_size, args.freeze_radius_percentile
            )
        except ValueError as exc:
            parser.error(str(exc))
        frozen_configuration = [(mode, size)]
        frozen_percentiles = (percentile,)
    try:
        validate_hardware_contract(args.bits, DESCRIPTOR_DIM)
    except ValueError as exc:
        parser.error(str(exc))
    families = ("hj", "mearec") if args.pilot or args.family == "all" else (args.family,)
    all_family_rows: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        stream = _iter_hj(args.duration, pilot=args.pilot, confirmation=args.confirmation) if family == "hj" else _iter_mearec(
            args.duration, args.mearec_npz_dir, pilot=args.pilot, confirmation=args.confirmation
        )
        rows: list[dict[str, Any]] = []
        for dataset in stream:
            rows.extend(run_dataset_masked(
                dataset, bits=args.bits,
                radius_percentiles=frozen_percentiles,
                include_support=args.include_unit_support,
                mask_configurations=frozen_configuration,
                selection_protocol=(
                    "frozen_confirmation_pre_registered_pilot_selected_single_all_rows_configuration"
                    if args.confirmation else "exploratory_test_grid_no_final_configuration_selection"
                ),
                verbose=not args.quiet,
            ))
            del dataset
        all_family_rows[family] = rows
        suffix = "pilot" if args.pilot else "confirmation" if args.confirmation else "full"
        output = args.output_dir / f"adaptive_masked_5bit_{suffix}_{family}_{int(args.duration)}s.csv"
        _write_csv(output, rows)
        print(f"Saved {len(rows)} rows: {output}")
    if args.pilot:
        report = pilot_gate(all_family_rows)
        final_report = final_099_4x_report(all_family_rows)
        print(f"Pilot formal gate pass: {report['pilot_gate_pass']}; candidates={report['pilot_gate_candidates']}")
        print(f"Pilot frozen selection report: {report['pilot_gate_selected']}")
        print(f"Final 0.99/4x report-only pass: {final_report['final_0p99_4x_pass']}; configurations={final_report['final_0p99_4x_qualifying_configurations']}")
        print("No confirmation sweep was started automatically.")


if __name__ == "__main__":
    main()
