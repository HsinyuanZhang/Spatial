"""Frozen-candidate pair-conditioned waveform tournaments.

Every spatial candidate set is fitted/query-generated once and is reused by
the float teacher, signed full-64 baseline, and pair-specific Fisher-tap
tournaments.  Pair models use only chronological training GT event rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.data.loader import Dataset, list_hybrid_janelia_scenes, load_hybrid_janelia
from Spatial.experiments.run_adaptive_5bit_range_search import REPO_ROOT, _chronological_split, _iter_mearec_datasets, prepare_dataset_events
from Spatial.experiments.run_compact_waveform_refinement import (
    WAVEFORM_SAMPLES,
    fit_float_templates,
    fit_integer_templates,
    frozen_candidate_sources,
    quantize_signed_uniform,
)


def _key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def unordered_pair_key(first: Any, second: Any) -> tuple[Any, Any]:
    """Canonical unordered pair key for labels with a deterministic ordering."""
    a, b = _key(first), _key(second)
    if a == b:
        raise ValueError("a pair requires two different units")
    return (a, b) if a < b else (b, a)


def pair_fisher_taps(
    waveforms_train: np.ndarray, labels_train: np.ndarray, first: Any, second: Any, n_taps: int
) -> np.ndarray:
    """Train-only pair Fisher ranking: mean-difference² / pooled within variance."""
    waves = np.asarray(waveforms_train, dtype=np.float64)
    labels = np.asarray(labels_train)
    if waves.ndim != 2 or waves.shape[0] != labels.size:
        raise ValueError("training waveform rows must align with labels")
    if not 1 <= int(n_taps) <= waves.shape[1]:
        raise ValueError("n_taps must be in [1, waveform width]")
    a = waves[labels == first]
    b = waves[labels == second]
    if not a.size or not b.size:
        raise ValueError("both pair units need training rows")
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    within = ((a - mean_a) ** 2).sum(axis=0) + ((b - mean_b) ** 2).sum(axis=0)
    score = (mean_a - mean_b) ** 2 / np.maximum(within, 1e-12)
    indices = np.arange(waves.shape[1])
    return np.lexsort((indices, -score))[: int(n_taps)].astype(np.int64)


def fit_pair_tap_tables(
    waveforms_train: np.ndarray, labels_train: np.ndarray, units: np.ndarray, n_taps: int
) -> dict[tuple[Any, Any], np.ndarray]:
    """Fit a compressed M-index table for every unordered fitted-unit pair."""
    return {
        unordered_pair_key(first, second): pair_fisher_taps(waveforms_train, labels_train, first, second, n_taps)
        for first, second in combinations(units, 2)
    }


def stable_spatial_candidate_order(
    candidates: Sequence[np.ndarray], spatial_codes: np.ndarray, spatial_centroids: np.ndarray, units: np.ndarray
) -> list[np.ndarray]:
    """Stable raw-L1 sort each candidate set without adding/removing candidates.

    This ordering is subsequently the deterministic tie policy for signed SAD:
    a tied challenger cannot displace the spatially earlier champion.
    """
    unit_index = {_key(unit): row for row, unit in enumerate(units)}
    output: list[np.ndarray] = []
    for candidate_row, code in zip(candidates, np.asarray(spatial_codes, dtype=np.int64)):
        original = list(candidate_row)
        ordered = sorted(
            enumerate(original),
            key=lambda item: (
                int(np.abs(spatial_centroids[unit_index[_key(item[1])]] - code).sum()), item[0]
            ),
        )
        output.append(np.asarray([item[1] for item in ordered], dtype=np.asarray(candidate_row).dtype))
    return output


def nearest_float_teacher(
    ordered_candidates: Sequence[np.ndarray], waveforms_test: np.ndarray, float_templates: dict[Any, np.ndarray],
    spatial_codes: np.ndarray, spatial_centroids: np.ndarray, units: np.ndarray,
) -> np.ndarray:
    """Nearest float template on the same ordered candidate lists, zero→spatial fallback."""
    values = np.asarray(waveforms_test, dtype=np.float64)
    unit_array = np.asarray(units)
    pred = np.empty(values.shape[0], dtype=unit_array.dtype)
    for row, candidate_row in enumerate(ordered_candidates):
        candidates = list(candidate_row)
        if len(candidates) == 1:
            pred[row] = candidates[0]
        elif candidates:
            distances = [np.abs(values[row] - float_templates[_key(candidate)]).sum() for candidate in candidates]
            pred[row] = candidates[int(np.argmin(distances))]
        else:
            pred[row] = unit_array[int(np.argmin(np.abs(spatial_centroids - spatial_codes[row]).sum(axis=1)))]
    return pred


def tournament_assign(
    ordered_candidates: Sequence[np.ndarray], query_values: np.ndarray, templates: dict[Any, np.ndarray],
    spatial_codes: np.ndarray, spatial_centroids: np.ndarray, units: np.ndarray, *,
    mode: str, pair_taps: dict[tuple[Any, Any], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sequential champion tournament; unique/zero make no waveform reads."""
    values = np.asarray(query_values)
    unit_array = np.asarray(units)
    known = {_key(unit) for unit in unit_array}
    pred = np.empty(values.shape[0], dtype=unit_array.dtype)
    comparisons = np.zeros(values.shape[0], dtype=np.int64)
    fallback = np.zeros(values.shape[0], dtype=bool)
    touched: set[tuple[Any, Any]] = set()
    for row, candidate_row in enumerate(ordered_candidates):
        candidates = list(candidate_row)
        if len(candidates) == 1 and _key(candidates[0]) in known:
            pred[row] = candidates[0]
            continue
        if len(candidates) > 1:
            champion = candidates[0]
            for challenger in candidates[1:]:
                pair = unordered_pair_key(champion, challenger)
                if mode == "pair_fisher":
                    if pair_taps is None:
                        raise ValueError("pair_fisher mode requires pair_taps")
                    taps = pair_taps[pair]
                    touched.add(pair)
                    dc = np.abs(values[row, taps] - templates[_key(champion)][taps]).sum()
                    dd = np.abs(values[row, taps] - templates[_key(challenger)][taps]).sum()
                else:
                    dc = np.abs(values[row] - templates[_key(champion)]).sum()
                    dd = np.abs(values[row] - templates[_key(challenger)]).sum()
                # ``dd < dc`` rather than ``<=`` is intentional: equal signed
                # SAD preserves the raw-spatial-L1-stable earlier champion.
                if dd < dc:
                    champion = challenger
                comparisons[row] += 1
            pred[row] = champion
            continue
        pred[row] = unit_array[int(np.argmin(np.abs(spatial_centroids - spatial_codes[row]).sum(axis=1)))]
        fallback[row] = True
    return pred, {"pair_comparisons": comparisons, "spatial_fallback": fallback, "touched_pairs": touched}


def pair_traffic_accounting(
    comparisons: np.ndarray, n_units: int, bits: int, n_taps: int, *, use_pair_taps: bool = True,
) -> dict[str, float | int]:
    """Traffic for pair templates plus conservative per-comparison tap-index access."""
    total_comparisons = int(np.asarray(comparisons, dtype=np.int64).sum())
    code_bits = total_comparisons * 2 * int(n_taps) * int(bits)
    compressed_index_bits = total_comparisons * int(n_taps) * 6 if use_pair_taps else 0
    mask_index_bits = total_comparisons * 64 if use_pair_taps else 0
    compressed_total = code_bits + compressed_index_bits
    mask_total = code_bits + mask_index_bits
    # The global comparison reference scans all U templates as 64 4-bit words.
    n_events = np.asarray(comparisons).size
    global_bits = int(n_events) * int(n_units) * 64 * 4
    return {
        "pair_template_code_bits_read": code_bits,
        "pair_tap_index_bits_read_compressed": compressed_index_bits,
        "pair_tap_mask64_bits_read": mask_index_bits,
        "total_bits_read_compressed_index": compressed_total,
        "total_bits_read_mask64": mask_total,
        "global_64x4bit_bits": global_bits,
        "global_64x4bit_over_total_bits_factor_compressed": float(global_bits / compressed_total) if compressed_total else float("inf"),
        "global_64x4bit_over_total_bits_factor_mask64": float(global_bits / mask_total) if mask_total else float("inf"),
    }


def pair_metrics(
    candidates: Sequence[np.ndarray], labels: np.ndarray, prediction: np.ndarray, teacher: np.ndarray,
    comparisons: np.ndarray, n_units: int, bits: int, n_taps: int, pair_count: int, touched_pair_count: int,
) -> dict[str, float | int]:
    """Accuracy, operations, traffic, and storage for one frozen-source tournament."""
    target = np.asarray(labels)
    counts = np.asarray([len(row) for row in candidates], dtype=np.int64)
    ambiguous = counts > 1
    comparisons = np.asarray(comparisons, dtype=np.int64)
    recall = np.asarray([label in row for label, row in zip(target, candidates)], dtype=bool)
    tap_storage = int(pair_count) * int(n_taps) * 6
    metrics: dict[str, float | int] = {
        "candidate_true_unit_recall": float(recall.mean()) if target.size else 0.0,
        "mean_candidates": float(counts.mean()) if target.size else 0.0,
        "ambiguous_events": int(ambiguous.sum()),
        "overall_accuracy": float(np.mean(prediction == target)) if target.size else 0.0,
        "ambiguous_only_accuracy": float(np.mean(prediction[ambiguous] == target[ambiguous])) if np.any(ambiguous) else 0.0,
        "float_teacher_prediction_agreement": float(np.mean(prediction == teacher)) if target.size else 0.0,
        "accuracy_delta_vs_float_teacher": float(np.mean(prediction == target) - np.mean(teacher == target)) if target.size else 0.0,
        "pair_comparisons_total": int(comparisons.sum()),
        "pair_comparisons_per_event": float(comparisons.mean()) if target.size else 0.0,
        "abs_diffs_total": int(comparisons.sum()) * 2 * int(n_taps),
        "adds_total": int(comparisons.sum()) * 2 * max(int(n_taps) - 1, 0),
        "pair_count_storage": int(pair_count),
        "pair_count_test_touched": int(touched_pair_count),
        "pair_tap_index_storage_bits_compressed": tap_storage,
        "pair_tap_mask64_storage_bits": int(pair_count) * 64,
        "full_per_unit_template_storage_bits": int(n_units) * 64 * int(bits),
        "total_storage_bits_compressed_index": int(n_units) * 64 * int(bits) + tap_storage,
        "total_storage_bits_mask64": int(n_units) * 64 * int(bits) + int(pair_count) * 64,
        **pair_traffic_accounting(comparisons, n_units, bits, n_taps, use_pair_taps=pair_count > 0),
    }
    return metrics


def run_dataset_pair_conditioned(
    dataset: Dataset, *, percentiles: Sequence[float] = (95.0, 99.9), bits: int = 5,
    tap_counts: Sequence[int] = (4, 8, 12, 16), include_all_units: bool = False, verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run frozen candidate sources with full and pair-conditioned tournaments."""
    prepared = prepare_dataset_events(dataset, k_neighbors=7)
    train_idx, test_idx = _chronological_split(prepared["times"])
    train_y, test_y = prepared["labels"][train_idx], prepared["labels"][test_idx]
    sources = frozen_candidate_sources(prepared["descriptor"][train_idx], train_y, prepared["descriptor"][test_idx], percentiles)
    if not include_all_units:
        sources.pop("all_units", None)
    units = next(iter(sources.values()))["units"]
    train_wave, test_wave = prepared["waveforms"][train_idx], prepared["waveforms"][test_idx]
    maximum = (1 << (int(bits) - 1)) - 1
    signed_train, signed_test = quantize_signed_uniform(train_wave, bits), quantize_signed_uniform(test_wave, bits)
    float_templates = fit_float_templates(train_wave, train_y, units)
    signed_templates = fit_integer_templates(signed_train, train_y, units, -maximum, maximum)
    pair_tables = {int(m): fit_pair_tap_tables(train_wave, train_y, units, int(m)) for m in tap_counts}
    rows: list[dict[str, Any]] = []
    for source_name, source in sources.items():
        # This is a new list-of-arrays ordering view; source['candidates'] itself
        # is untouched and reused by every tournament below.
        ordered = stable_spatial_candidate_order(source["candidates"], source["spatial_codes"], source["spatial_centroids"], units)
        teacher = nearest_float_teacher(ordered, test_wave, float_templates, source["spatial_codes"], source["spatial_centroids"], units)
        baselines: list[tuple[str, np.ndarray, dict[str, Any], int, int, int]] = []
        float_pred, float_meta = tournament_assign(ordered, test_wave, float_templates, source["spatial_codes"], source["spatial_centroids"], units, mode="float64")
        baselines.append(("float64_tournament_teacher", float_pred, float_meta, 32, 64, 0))
        signed_pred, signed_meta = tournament_assign(ordered, signed_test, signed_templates, source["spatial_codes"], source["spatial_centroids"], units, mode="signed_full64")
        baselines.append((f"signed_{bits}bit_full64_tournament", signed_pred, signed_meta, int(bits), 64, 0))
        for taps, table in pair_tables.items():
            pred, meta = tournament_assign(ordered, signed_test, signed_templates, source["spatial_codes"], source["spatial_centroids"], units, mode="pair_fisher", pair_taps=table)
            baselines.append((f"pair_fisher{taps}_signed_{bits}bit_tournament", pred, meta, int(bits), taps, len(table)))
        for name, pred, meta, representation_bits, width, pair_count in baselines:
            # Float teacher is a correctness reference; hardware storage/traffic
            # rows use 32-bit float payload, while signed pair rows use B bits.
            metrics = pair_metrics(
                source["candidates"], test_y, pred, teacher, meta["pair_comparisons"], units.size,
                representation_bits, width, pair_count, len(meta["touched_pairs"]),
            )
            rows.append({
                "dataset": dataset.name, "n_events_gt_rows": int(prepared["labels"].size),
                "n_train_gt_rows": int(train_idx.size), "n_test_gt_rows": int(test_idx.size),
                "spatial_descriptor_dim": int(prepared["descriptor"].shape[1]), "spatial_k_neighbors": 7,
                "candidate_source": source_name, "representation": name, "signed_bits": representation_bits,
                "pair_taps": width if pair_count else "",
                "candidate_order_policy": "raw_spatial_l1_stable",
                "distance_tie_break_policy": "keep_spatially_earlier_champion",
                **metrics,
            })
    if verbose:
        print(f"{dataset.name}: GT rows train/test={train_idx.size}/{test_idx.size}, sources={len(sources)}")
    return rows


def _iter_hj_datasets(duration: float):
    scenes = list_hybrid_janelia_scenes(ready_only=True)
    if not scenes:
        yield load_hybrid_janelia(duration_s=duration)
        return
    for scene in scenes:
        key = scene.get("short_name") or scene.get("scene_key")
        if key:
            yield load_hybrid_janelia(scene=key, duration_s=duration)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair-conditioned Fisher-tap waveform tournaments")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--taps", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[95.0, 99.9])
    parser.add_argument("--include-all-units", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "pair_conditioned_waveform")
    parser.add_argument("--mearec-npz-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.bits < 2 or any(tap < 1 or tap > 64 for tap in args.taps) or not args.taps:
        parser.error("--bits must be >=2 and --taps must be in [1,64]")
    if any(p < 0 or p > 100 for p in args.percentiles) or not args.percentiles:
        parser.error("--percentiles must be in [0,100]")
    streams: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}: streams.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
        streams.append(("mearec", _iter_mearec_datasets(args.duration, None, args.mearec_npz_dir or DEFAULT_NPZ_DIR)))
    start = time.time()
    for family, stream in streams:
        rows: list[dict[str, Any]] = []
        for dataset in stream:
            rows.extend(run_dataset_pair_conditioned(dataset, percentiles=args.percentiles, bits=args.bits, tap_counts=args.taps, include_all_units=args.include_all_units, verbose=not args.quiet)); del dataset
        if rows:
            path = args.output_dir / f"pair_conditioned_waveform_{family}_{int(args.duration)}s.csv"
            _write_csv(path, rows); print(f"Saved {len(rows)} rows: {path}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
