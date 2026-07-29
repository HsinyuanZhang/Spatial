"""Static bounded multi-prototype 5-bit candidate-search benchmark.

The runner is deliberately a GT-event/GT-row offline isolation experiment:
spatial prototypes, prototype-specific directory pointers, and waveform unit
templates are fitted only on the chronological training half.  Waveform
assignment always consumes de-duplicated *unit* candidates, so additional
prototypes cannot manufacture a Level-2 advantage by duplicate template reads.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.spatial_footprint import build_knn_table_with_self
from Spatial.algorithms.adaptive_prototype_search import AdaptivePrototypeSearch
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    REPO_ROOT,
    _chronological_split,
    _iter_hj_datasets,
    _iter_mearec_datasets,
    candidate_waveform_assign,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR


SPATIAL_PAYLOAD_BITS = 54  # D=9 centroid codes (45) + full-scale L1 radius (9)
# Traffic baseline is the existing signed 5-bit, 64-sample template payload.
# Accuracy below can still use the float64 L1 teacher as a software reference.
WAVEFORM_TEMPLATE_BITS = 64 * 5


def validate_hardware_contract(bits: int, descriptor_dim: int | None, prototypes: Sequence[int]) -> None:
    """Enforce frozen 5-bit, D=9, bounded-P≤4 benchmark scope."""
    if int(bits) != 5:
        raise ValueError("the cross-dataset hardware benchmark is fixed at 5 spatial bits")
    if descriptor_dim is not None and int(descriptor_dim) != 9:
        raise ValueError("the frozen hardware benchmark requires D=9 (COM2 + K7 P2P)")
    if any(int(value) < 1 or int(value) > 4 for value in prototypes):
        raise ValueError("prototypes_per_unit must be in [1, 4] for this bounded benchmark")


def prototype_row_storage_bits(n_units: int) -> dict[str, int]:
    """Storage per prototype row: fixed 54-bit search payload, unit ID, valid bit."""
    if int(n_units) < 1:
        raise ValueError("n_units must be positive")
    unit_id_bits = int(np.ceil(np.log2(int(n_units)))) if int(n_units) > 1 else 0
    return {
        "prototype_row_payload_bits": SPATIAL_PAYLOAD_BITS,
        "prototype_row_unit_id_bits": unit_id_bits,
        "prototype_row_valid_bits": 1,
        "prototype_row_total_bits": SPATIAL_PAYLOAD_BITS + unit_id_bits + 1,
    }


def fit_prototype_channel_support(
    prototype_assignment: np.ndarray,
    central_channels: np.ndarray,
    n_prototypes: int,
    n_channels: int,
    coverage: float,
) -> np.ndarray:
    """Fit a minimum channel support set for each training-assigned prototype.

    Count desc/channel-id asc ranking makes ties deterministic.  Rows with no
    train assignment remain all false and therefore own no directory pointer.
    """
    assignment = np.asarray(prototype_assignment, dtype=np.int64).ravel()
    channels = np.asarray(central_channels, dtype=np.int64).ravel()
    if assignment.shape != channels.shape:
        raise ValueError("prototype_assignment and central_channels must align")
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if int(n_prototypes) < 1 or int(n_channels) < 1:
        raise ValueError("prototype and channel counts must be positive")
    if np.any((assignment < 0) | (assignment >= int(n_prototypes))):
        raise ValueError("prototype assignment outside row range")
    if np.any((channels < 0) | (channels >= int(n_channels))):
        raise ValueError("central channel outside range")
    support = np.zeros((int(n_prototypes), int(n_channels)), dtype=bool)
    for proto in range(int(n_prototypes)):
        local = channels[assignment == proto]
        if not local.size:
            continue
        ids, counts = np.unique(local, return_counts=True)
        order = np.lexsort((ids, -counts))
        required = int(np.ceil(float(coverage) * local.size))
        take = int(np.searchsorted(np.cumsum(counts[order]), required, side="left")) + 1
        support[proto, ids[order[:take]]] = True
    return support


def prototype_directory_mask(
    query_channels: np.ndarray, support: np.ndarray, geom: np.ndarray, directory_neighbors: int
) -> np.ndarray:
    """Query-by-prototype activation from channel support ∩ geometry KNN."""
    query = np.asarray(query_channels, dtype=np.int64).ravel()
    table_support = np.asarray(support, dtype=bool)
    geometry = np.asarray(geom, dtype=np.float64)
    n_channels = geometry.shape[0]
    if table_support.ndim != 2 or table_support.shape[1] != n_channels:
        raise ValueError("support must have shape (n_prototypes, n_channels)")
    if np.any((query < 0) | (query >= n_channels)):
        raise ValueError("query channel outside geometry")
    if int(directory_neighbors) <= 0:
        return np.ones((query.size, table_support.shape[0]), dtype=bool)
    knn = build_knn_table_with_self(geometry, min(int(directory_neighbors), n_channels))
    return np.asarray([table_support[:, knn[ch]].any(axis=1) for ch in query], dtype=bool)


def directory_true_unit_row_activation_recall(
    active_row_mask: np.ndarray | None, true_labels: np.ndarray, prototype_units: np.ndarray
) -> float:
    """Fraction whose true unit owns at least one activated prototype row.

    This explicitly diagnoses directory misses before a prototype-radius range
    comparison can remove the true unit.  Unseen labels have no matching row
    and count as misses; ``None`` denotes the all-row baseline.
    """
    labels = np.asarray(true_labels)
    rows = np.asarray(prototype_units)
    if active_row_mask is None:
        active = np.ones((labels.size, rows.size), dtype=bool)
    else:
        active = np.asarray(active_row_mask, dtype=bool)
        if active.shape != (labels.size, rows.size):
            raise ValueError("active_row_mask must be (n_queries, n_prototype_rows)")
    hits = np.asarray(
        [np.any(active[index] & (rows == label)) for index, label in enumerate(labels)], dtype=bool
    )
    return float(hits.mean()) if labels.size else 0.0


def prototype_pointer_accounting(pointer_counts: np.ndarray, n_prototype_rows: int) -> dict[str, int]:
    """Directory pointer count and prototype-row-ID payload, excluding periphery."""
    counts = np.asarray(pointer_counts, dtype=np.int64).ravel()
    if np.any(counts < 0) or int(n_prototype_rows) < 1:
        raise ValueError("pointer counts must be non-negative and rows positive")
    row_id_bits = int(np.ceil(np.log2(int(n_prototype_rows)))) if int(n_prototype_rows) > 1 else 0
    total = int(counts.sum())
    return {
        "prototype_directory_pointers_total": total,
        "prototype_directory_pointer_row_id_bits": row_id_bits,
        "prototype_directory_total_pointer_bits": total * row_id_bits,
    }


def deduplicate_prototype_hits(prototype_hits: Sequence[np.ndarray], prototype_units: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Stable prototype-hit→unit candidates and number of duplicate hits removed."""
    units = np.asarray(prototype_units)
    output: list[np.ndarray] = []
    removed = np.zeros(len(prototype_hits), dtype=np.int64)
    for row, hits in enumerate(prototype_hits):
        values: list[Any] = []
        for proto in np.asarray(hits, dtype=np.int64):
            if proto < 0 or proto >= units.size:
                raise ValueError("prototype hit index outside prototype_units")
            unit = units[int(proto)]
            if not any(unit == prior for prior in values):
                values.append(unit)
            else:
                removed[row] += 1
        output.append(np.asarray(values, dtype=units.dtype))
    return output, removed


def waveform_traffic(candidate_lists: Sequence[np.ndarray], n_units: int) -> dict[str, float | int]:
    """Unique/zero rows read no teacher template; only ambiguous unit lists read C."""
    counts = np.asarray([len(row) for row in candidate_lists], dtype=np.int64)
    reads = np.where(counts > 1, counts, 0)
    actual = int(reads.sum()) * WAVEFORM_TEMPLATE_BITS
    global_bits = int(counts.size) * int(n_units) * WAVEFORM_TEMPLATE_BITS
    return {
        "candidate_waveform_reads_total": int(reads.sum()),
        "candidate_waveform_reads_per_event": float(reads.mean()) if reads.size else 0.0,
        "actual_waveform_template_bits_read": actual,
        "global_all_unit_waveform_template_bits": global_bits,
        "waveform_bit_traffic_reduction_factor": float(global_bits / actual) if actual else float("inf"),
        "waveform_template_payload": "signed_5bit_x64_hardware_traffic_baseline",
    }


def level1_comparison_accounting(
    primary_prototype_rows: np.ndarray, spatial_fallback_rows: np.ndarray
) -> dict[str, float | int]:
    """Keep primary prototype search and zero-candidate fallback scans separate.

    ``spatial_fallback_rows`` must come from ``candidate_waveform_assign``
    metadata, so an empty candidate event pays the actual configured prototype
    row scan used by its fallback rather than an inferred candidate-count cost.
    """
    primary = np.asarray(primary_prototype_rows, dtype=np.int64).ravel()
    fallback = np.asarray(spatial_fallback_rows, dtype=np.int64).ravel()
    if primary.shape != fallback.shape or np.any(primary < 0) or np.any(fallback < 0):
        raise ValueError("primary and fallback row counts must align and be non-negative")
    total = primary + fallback
    n = total.size
    return {
        "primary_prototype_row_comparisons_total": int(primary.sum()),
        "spatial_fallback_level1_row_comparisons_total": int(fallback.sum()),
        "mean_spatial_fallback_level1_row_comparisons": float(fallback.mean()) if n else 0.0,
        "p95_spatial_fallback_level1_row_comparisons": float(np.percentile(fallback, 95)) if n else 0.0,
        "max_spatial_fallback_level1_row_comparisons": int(fallback.max()) if n else 0,
        "mean_total_level1_row_comparisons": float(total.mean()) if n else 0.0,
        "p95_total_level1_row_comparisons": float(np.percentile(total, 95)) if n else 0.0,
        "max_total_level1_row_comparisons": int(total.max()) if n else 0,
        "total_level1_row_comparisons": int(total.sum()),
    }


def recall_and_gate(candidates: Sequence[np.ndarray], labels: np.ndarray, n_units: int) -> dict[str, float | int | bool]:
    """Candidate recall/C plus the explicit 99%-recall and 4×-traffic gate."""
    target = np.asarray(labels)
    counts = np.asarray([len(row) for row in candidates], dtype=np.int64)
    recall = np.asarray([label in row for label, row in zip(target, candidates)], dtype=bool)
    traffic = waveform_traffic(candidates, n_units)
    return {
        "true_unit_candidate_recall": float(recall.mean()) if target.size else 0.0,
        "mean_candidates": float(counts.mean()) if target.size else 0.0,
        "max_candidates": int(counts.max()) if counts.size else 0,
        "ambiguous_events": int((counts > 1).sum()),
        "waveform_accuracy_teacher": "float64_l1_software_reference",
        "meets_recall_0p99_and_waveform_reduction_4x": bool(
            (float(recall.mean()) if target.size else 0.0) >= 0.99
            and float(traffic["waveform_bit_traffic_reduction_factor"]) >= 4.0
        ),
        **traffic,
    }


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    """Training-only float waveform teacher centroids keyed by unit label."""
    return {
        unit.item() if isinstance(unit, np.generic) else unit: waveforms[labels == unit].mean(axis=0)
        for unit in units
    }


def _candidate_metrics(
    candidates: list[np.ndarray], labels: np.ndarray, pred: np.ndarray, diagnostics: dict[str, np.ndarray],
    wave_meta: dict[str, np.ndarray], n_units: int,
) -> dict[str, Any]:
    report = recall_and_gate(candidates, labels, n_units)
    counts = np.asarray([len(row) for row in candidates], dtype=np.int64)
    active_rows = np.asarray(diagnostics["active_prototype_row_count"], dtype=np.int64)
    comparisons = np.asarray(diagnostics["row_comparisons"], dtype=np.int64)
    fallback_rows = np.asarray(wave_meta["spatial_fallback_row_comparisons"], dtype=np.int64)
    report.update({
        "final_waveform_accuracy": float(np.mean(pred == labels)) if labels.size else 0.0,
        "prototype_hit_count_mean": float(np.mean(diagnostics["prototype_hit_count"])) if labels.size else 0.0,
        "prototype_hit_count_max": int(np.max(diagnostics["prototype_hit_count"])) if labels.size else 0,
        "duplicate_prototype_hits_removed_total": int(np.sum(diagnostics["duplicates_removed"])),
        "duplicate_prototype_hits_removed_mean": float(np.mean(diagnostics["duplicates_removed"])) if labels.size else 0.0,
        "zero_fraction": float(np.mean(counts == 0)) if labels.size else 0.0,
        "unique_fraction": float(np.mean(counts == 1)) if labels.size else 0.0,
        "ambiguous_fraction": float(np.mean(counts > 1)) if labels.size else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if labels.size else 0.0,
        "mean_active_prototype_rows": float(np.mean(active_rows)) if labels.size else 0.0,
        "p95_active_prototype_rows": float(np.percentile(active_rows, 95)) if labels.size else 0.0,
        "max_active_prototype_rows": int(np.max(active_rows)) if labels.size else 0,
        "mean_prototype_row_comparisons": float(np.mean(comparisons)) if labels.size else 0.0,
        "total_prototype_row_comparisons": int(comparisons.sum()),
        "waveform_fallback_fraction": float(np.mean(wave_meta["waveform_fallback"])) if labels.size else 0.0,
        **level1_comparison_accounting(comparisons, fallback_rows),
    })
    return report


def run_dataset_multiprototype(
    dataset: Dataset,
    *,
    bits: int = 5,
    prototypes: Sequence[int] = (1, 2, 3, 4),
    modes: Sequence[str] = ("temporal", "farthest_l1"),
    radius_percentiles: Sequence[float] = (95.0, 99.0, 99.9),
    directory_coverages: Sequence[float] = (0.95, 0.99, 1.0),
    k_neighbors: int = 7,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate frozen all-row and prototype-support directory candidate sources.

    All fitting uses chronological GT training rows only.  For a configuration,
    each candidate source is queried exactly once and the same resulting unit
    lists are passed to the common float-L1 waveform teacher.
    """
    validate_hardware_contract(bits, None, prototypes)
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    x_train, y_train = prepared["descriptor"][train_idx], prepared["labels"][train_idx]
    x_test, y_test = prepared["descriptor"][test_idx], prepared["labels"][test_idx]
    validate_hardware_contract(bits, x_train.shape[1], prototypes)
    rows: list[dict[str, Any]] = []
    if verbose:
        print(f"{dataset.name}: GT rows train/test={train_idx.size}/{test_idx.size}, D={x_train.shape[1]}")

    for requested_p in dict.fromkeys(int(value) for value in prototypes):
        for mode in dict.fromkeys(str(value) for value in modes):
            for percentile in dict.fromkeys(float(value) for value in radius_percentiles):
                model = AdaptivePrototypeSearch(n_bits=5, radius_percentile=percentile).fit(
                    x_train, y_train, prototypes_per_unit=requested_p, mode=mode
                )
                codes_test = model.transform(x_test)
                n_rows = model.prototype_units_.size
                storage = model.memory_accounting()
                wave_centroids = _waveform_centroids(prepared["waveforms"][train_idx], y_train, model.units_)
                # Directory configurations are (name, coverage, active row mask,
                # prototype pointer counts).  all-row is emitted once; support
                # pointers are derived from the core's *training* cluster rows.
                directory_configs: list[tuple[str, float | None, np.ndarray | None, np.ndarray]] = [
                    ("all_rows", None, None, np.zeros(n_rows, dtype=np.int64))
                ]
                for coverage in dict.fromkeys(float(value) for value in directory_coverages):
                    support = fit_prototype_channel_support(
                        model.training_prototype_assignments_, prepared["central"][train_idx],
                        n_rows, dataset.n_channels, coverage,
                    )
                    directory_configs.append((
                        "prototype_support", coverage,
                        prototype_directory_mask(prepared["central"][test_idx], support, dataset.geom, min(k_neighbors, dataset.n_channels)),
                        support.sum(axis=1, dtype=np.int64),
                    ))
                for strategy, coverage, active_mask, pointer_counts in directory_configs:
                    candidates, diagnostics = model.query_codes(codes_test, active_row_mask=active_mask)
                    # ``query_codes`` has already removed repeated matching
                    # prototype rows by unit, and this exact list is consumed
                    # once by the common waveform teacher.
                    pred, wave_meta = candidate_waveform_assign(
                        candidates, prepared["waveforms"][test_idx], wave_centroids,
                        codes_test, model.centroids_, model.prototype_units_, metric="l1",
                    )
                    metrics = _candidate_metrics(candidates, y_test, pred, diagnostics, wave_meta, model.units_.size)
                    pointer_accounting = prototype_pointer_accounting(pointer_counts, n_rows)
                    rows.append({
                        "dataset": dataset.name,
                        "n_events_gt_rows": int(prepared["labels"].size),
                        "n_train_gt_rows": int(train_idx.size), "n_test_gt_rows": int(test_idx.size),
                        "offline_isolation": "gt_event_rows_chronological_train_only_fit",
                        "selection_protocol": "exploratory_test_grid_no_final_configuration_selection",
                        "preprocessing": "full_recording_zero_phase_filtfilt_offline",
                        "n_channels": int(dataset.n_channels), "n_units": int(model.units_.size),
                        "bits": 5, "descriptor_dim": int(x_train.shape[1]), "k_neighbors": int(min(k_neighbors, dataset.n_channels)),
                        "prototype_mode": mode, "prototypes_per_unit_requested": int(requested_p),
                        "radius_percentile": float(percentile), "directory_strategy": strategy,
                        "directory_coverage": "" if coverage is None else float(coverage),
                        "directory_neighbors": int(min(k_neighbors, dataset.n_channels)),
                        "prototype_rows": int(storage["n_prototype_rows"]),
                        "mean_configured_prototypes_per_unit": float(storage["configured_prototypes_per_unit_mean"]),
                        "p95_configured_prototypes_per_unit": float(storage["configured_prototypes_per_unit_p95"]),
                        "max_configured_prototypes_per_unit": int(storage["configured_prototypes_per_unit_max"]),
                        **pointer_accounting,
                        "prototype_directory_pointer_accounting_scope": "prototype_row_id_payload_only_excludes_periphery_valid_decode",
                        "prototype_directory_pointers_mean_per_row": float(pointer_counts.mean()) if pointer_counts.size else 0.0,
                        "prototype_directory_pointers_max_per_row": int(pointer_counts.max()) if pointer_counts.size else 0,
                        "prototype_row_payload_bits": int(storage["search_payload_bits_per_prototype"]),
                        "prototype_row_unit_id_bits": int(storage["unit_id_bits_per_prototype"]),
                        "prototype_row_valid_bits": int(storage["valid_bits_per_prototype"]),
                        "prototype_row_total_bits": int(storage["row_bits_with_metadata_per_prototype"]),
                        "prototype_storage_total_bits": int(storage["total_bits_with_metadata"]),
                        "waveform_accuracy_teacher": "float64_l1_software_reference",
                        "directory_true_unit_row_activation_recall": directory_true_unit_row_activation_recall(
                            active_mask, y_test, model.prototype_units_
                        ),
                        **metrics,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded multi-prototype 5-bit benchmark")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--prototypes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", choices=("temporal", "farthest_l1"), default=["temporal", "farthest_l1"])
    parser.add_argument("--radius-percentiles", nargs="+", type=float, default=[95.0, 99.0, 99.9])
    parser.add_argument("--directory-coverages", nargs="+", type=float, default=[0.95, 0.99, 1.0])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "adaptive_multiprototype_5bit")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_hardware_contract(args.bits, None, args.prototypes)
    except ValueError as exc:
        parser.error(str(exc))
    if any(not 0 < c <= 1 for c in args.directory_coverages):
        parser.error("directory coverages must be in (0,1]")
    streams: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}:
        streams.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        streams.append(("mearec", _iter_mearec_datasets(args.duration, None, args.mearec_npz_dir)))
    for family, stream in streams:
        rows: list[dict[str, Any]] = []
        for dataset in stream:
            rows.extend(run_dataset_multiprototype(
                dataset, bits=args.bits, prototypes=args.prototypes, modes=args.modes,
                radius_percentiles=args.radius_percentiles, directory_coverages=args.directory_coverages,
                verbose=not args.quiet,
            ))
            del dataset
        if rows:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = args.output_dir / f"adaptive_multiprototype_5bit_{family}_{int(args.duration)}s.csv"
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            print(f"Saved {len(rows)} rows: {output}")


if __name__ == "__main__":
    main()
