"""5-bit centroid-anchor indexed candidate retrieval with calibration confusion lists.

This is an algorithmic SRAM/near-memory retrieval study, not a CiM/WTA PPA
claim.  Fit rows determine quantization, centroids, and L1 radii; calibration
rows build anchor-to-unit pointer lists; test labels are used only for metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch, evaluate_candidate_search
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    REPO_ROOT, _chronological_split, _iter_hj_datasets, _iter_mearec_datasets,
    _write_csv, candidate_waveform_assign, prepare_dataset_events,
)
from Spatial.experiments.run_certified_dual_range_pipeline import chronological_fraction_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR


RankingMode = Literal["raw_l1", "radius_normalized_l1"]


def rank_unit_rows(
    query_codes: np.ndarray, centroids: np.ndarray, radii: np.ndarray, *, mode: RankingMode
) -> np.ndarray:
    """Rank every fit-unit row by integer L1, stable-breaking ties by row order."""
    queries = np.asarray(query_codes, dtype=np.int64)
    rows = np.asarray(centroids, dtype=np.int64)
    radius = np.asarray(radii, dtype=np.int64)
    if queries.ndim != 2 or rows.ndim != 2 or queries.shape[1] != rows.shape[1]:
        raise ValueError("query_codes and centroids must be compatible 2D arrays")
    if radius.shape != (rows.shape[0],):
        raise ValueError("radii must align with centroids")
    if mode not in {"raw_l1", "radius_normalized_l1"}:
        raise ValueError("unknown ranking mode")
    distances = np.abs(queries[:, None, :] - rows[None, :, :]).sum(axis=2)
    scores = distances if mode == "raw_l1" else distances / (radius[None, :] + 1.0)
    return np.argsort(scores, axis=1, kind="stable")


def fixed_top_m_candidates(rankings: np.ndarray, units: np.ndarray, top_m: int) -> list[np.ndarray]:
    """Return exactly the fixed number of nearest fit rows (clipped to U)."""
    rank = np.asarray(rankings, dtype=np.int64)
    if rank.ndim != 2:
        raise ValueError("rankings must be two-dimensional")
    if int(top_m) < 1:
        raise ValueError("top_m must be positive")
    count = min(int(top_m), rank.shape[1])
    return [np.asarray(units)[rows[:count]].copy() for rows in rank]


def _stable_count_order(counts: np.ndarray) -> np.ndarray:
    """Descending count, then original fit-unit row order."""
    return np.argsort(-np.asarray(counts), kind="stable")


def build_anchor_graph(
    calibration_rankings: np.ndarray,
    calibration_labels: Sequence[Any] | np.ndarray,
    units: Sequence[Any] | np.ndarray,
    *,
    coverage: float,
    include_anchor: bool = True,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Build deterministic anchor→true-unit lists from calibration outcomes.

    Labels not present in fit units are deliberately not turned into rows.  They
    are counted in diagnostics so coverage cannot be misread as an oracle table.
    """
    rank = np.asarray(calibration_rankings, dtype=np.int64)
    labels = np.asarray(calibration_labels)
    row_units = np.asarray(units)
    if rank.ndim != 2 or rank.shape[0] != labels.size or rank.shape[1] != row_units.size:
        raise ValueError("calibration rankings, labels, and units must align")
    if not 0.0 <= float(coverage) <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    lookup = {value.item() if isinstance(value, np.generic) else value: i for i, value in enumerate(row_units)}
    counts = np.zeros((row_units.size, row_units.size), dtype=np.int64)
    unseen = 0
    for anchor, label in zip(rank[:, 0], labels):
        key = label.item() if isinstance(label, np.generic) else label
        if key not in lookup:
            unseen += 1
            continue
        counts[int(anchor), lookup[key]] += 1
    lists: list[np.ndarray] = []
    known_per_anchor = counts.sum(axis=1)
    for anchor in range(row_units.size):
        total = int(known_per_anchor[anchor])
        if total == 0:
            selected: list[int] = []
        else:
            ordered = _stable_count_order(counts[anchor])
            required = float(coverage) * total
            selected = []
            cumulative = 0
            for row in ordered:
                if cumulative >= required:
                    break
                selected.append(int(row))
                cumulative += int(counts[anchor, row])
        if include_anchor and anchor not in selected:
            selected.append(anchor)
        # An unseen calibration anchor must still have a deterministic safe list.
        if not selected:
            selected = [anchor]
        # Keep every stored list count-descending with fit-row-order ties even
        # when mandatory anchor inclusion adds a zero-count pointer.
        selected_set = set(selected)
        selected = [int(row) for row in _stable_count_order(counts[anchor]) if int(row) in selected_set]
        lists.append(row_units[np.asarray(selected, dtype=np.int64)].copy())
    graph_known_hits = 0
    graph_all_hits = 0
    for anchor, label in zip(rank[:, 0], labels):
        key = label.item() if isinstance(label, np.generic) else label
        hit = any(key == candidate for candidate in lists[int(anchor)])
        graph_all_hits += int(hit)
        if key in lookup:
            graph_known_hits += int(hit)
    n_calibration = labels.size
    known_total = n_calibration - unseen
    return lists, {
        "counts": counts,
        "unseen_calibration_label_count": int(unseen),
        "unseen_calibration_label_rate": float(unseen / n_calibration) if n_calibration else 0.0,
        "calibration_graph_empirical_coverage": float(graph_all_hits / n_calibration) if n_calibration else 0.0,
        "calibration_graph_empirical_coverage_known_labels": float(graph_known_hits / known_total) if known_total else 0.0,
    }


def stable_anchor_union_candidates(
    rankings: np.ndarray, graph: Sequence[Sequence[Any] | np.ndarray], top_anchors: int
) -> list[np.ndarray]:
    """Retrieve top-L anchor lists and return their stable, de-duplicated union."""
    rank = np.asarray(rankings, dtype=np.int64)
    if rank.ndim != 2 or int(top_anchors) < 1:
        raise ValueError("rankings must be 2D and top_anchors must be positive")
    result: list[np.ndarray] = []
    for anchor_rows in rank:
        values: list[Any] = []
        for anchor in anchor_rows[: min(int(top_anchors), anchor_rows.size)]:
            for unit in graph[int(anchor)]:
                if not any(unit == prior for prior in values):
                    values.append(unit)
        result.append(np.asarray(values))
    return result


def graph_storage_bits(graph: Sequence[Sequence[Any] | np.ndarray], n_units: int) -> dict[str, int | float]:
    """Pointer-list storage accounting, excluding unit IDs and peripheral logic."""
    units = int(n_units)
    if units < 1 or len(graph) != units:
        raise ValueError("graph must contain exactly one list per positive n_units")
    pointer_count = int(sum(len(rows) for rows in graph))
    pointer_width = int(np.ceil(np.log2(units))) if units > 1 else 0
    count_width = int(np.ceil(np.log2(units + 1)))
    return {
        "graph_total_pointers": pointer_count,
        "graph_mean_pointers_per_anchor": float(pointer_count / units),
        "graph_max_pointers_per_anchor": int(max(len(rows) for rows in graph)),
        "graph_pointer_id_bits": pointer_width,
        "graph_count_field_bits": count_width,
        "graph_total_bits": int(units * count_width + pointer_count * pointer_width),
    }


def summarize_indexed_candidates(
    candidates: Sequence[np.ndarray], labels: np.ndarray, diagnostics: dict[str, np.ndarray],
    pred: np.ndarray, wave_meta: dict[str, np.ndarray], *, n_units: int,
) -> dict[str, float | int | bool]:
    """Candidate, waveform, and full all-row L1 comparison metrics."""
    report: dict[str, float | int | bool] = dict(evaluate_candidate_search(candidates, labels, diagnostics))
    n = labels.size
    reads = np.asarray(wave_meta["candidate_waveform_reads"], dtype=np.int64)
    global_reads = n * int(n_units)
    factor = float(global_reads / reads.sum()) if reads.sum() else float("inf")
    report.update({
        "candidate_waveform_reads_per_event": float(reads.mean()) if n else 0.0,
        "candidate_waveform_reads_total": int(reads.sum()),
        "waveform_fallback_rate": float(np.mean(wave_meta["waveform_fallback"])) if n else 0.0,
        "global_all_unit_waveform_reads": int(global_reads),
        "global_waveform_reduction_factor": factor,
        "waveform_read_reduction_factor_global_over_actual": factor,
        "candidate_waveform_read_fraction_vs_global": float(reads.sum() / max(global_reads, 1)),
        "waveform_read_reduction_vs_global": float(1.0 - reads.sum() / max(global_reads, 1)),
        "final_waveform_accuracy": float(np.mean(pred == labels)) if n else 0.0,
        "all_row_l1_comparisons_per_query": float(np.mean(diagnostics["row_comparisons"])) if n else 0.0,
        "meets_recall_0p99_and_waveform_reduction_4x": bool(
            report["true_unit_candidate_recall"] >= 0.99 and factor >= 4.0
        ),
    })
    return report


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    return {unit.item() if isinstance(unit, np.generic) else unit: waveforms[labels == unit].mean(axis=0) for unit in units}


def run_dataset_indexed_confusion_candidate_search(
    dataset: Dataset, *, bits: int = 5, k_neighbors: int = 7,
    ranking_modes: Sequence[RankingMode] = ("raw_l1", "radius_normalized_l1"),
    top_ms: Sequence[int] = (1, 2, 4, 8), graph_coverages: Sequence[float] = (0.95, 0.99, 1.0),
    top_anchors: Sequence[int] = (1, 2), fit_fraction: float = 2.0 / 3.0,
    waveform_metric: str = "l1", verbose: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate fixed nearest-row and learned calibration-confusion retrieval."""
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    fit_relative, calibration_relative = chronological_fraction_split(prepared["times"][train_idx], fit_fraction)
    fit_idx, calibration_idx = train_idx[fit_relative], train_idx[calibration_relative]
    fit_features, fit_labels = prepared["descriptor"][fit_idx], prepared["labels"][fit_idx]
    calibration_labels, test_labels = prepared["labels"][calibration_idx], prepared["labels"][test_idx]
    model = AdaptiveRangeSearch(n_bits=int(bits), fixed_unit_interval=True, radius_percentile=95.0, box_coverage=1.0).fit(fit_features, fit_labels)
    units = model.units_
    n_units = len(units)
    calibration_codes = model.transform(prepared["descriptor"][calibration_idx])
    test_codes = model.transform(prepared["descriptor"][test_idx])
    waveform_centroids = _waveform_centroids(prepared["waveforms"][train_idx], prepared["labels"][train_idx], units)
    if verbose:
        print(f"{dataset.name}: train/fit/cal/test={len(train_idx)}/{len(fit_idx)}/{len(calibration_idx)}/{len(test_idx)}, units={n_units}, D={fit_features.shape[1]}, all-row L1 ranking")
    rows: list[dict[str, Any]] = []

    def append(method: str, ranking: str, candidates: list[np.ndarray], *, top_m: int | None, coverage: float | None, anchors: int | None, graph_metrics: dict[str, Any] | None) -> None:
        diag = {
            "row_comparisons": np.full(len(test_idx), n_units, dtype=np.int64),
            "active_row_count": np.full(len(test_idx), n_units, dtype=np.int64),
            "widen_level": np.zeros(len(test_idx), dtype=np.int64),
            "fallback": np.zeros(len(test_idx), dtype=bool),
        }
        pred, wave_meta = candidate_waveform_assign(candidates, prepared["waveforms"][test_idx], waveform_centroids, test_codes, model.centroids_, units, metric=waveform_metric)
        metrics = summarize_indexed_candidates(candidates, test_labels, diag, pred, wave_meta, n_units=n_units)
        graph_fields: dict[str, Any] = {
            "graph_total_pointers": 0, "graph_mean_pointers_per_anchor": 0.0, "graph_max_pointers_per_anchor": 0,
            "graph_pointer_id_bits": int(np.ceil(np.log2(n_units))) if n_units > 1 else 0,
            "graph_count_field_bits": int(np.ceil(np.log2(n_units + 1))), "graph_total_bits": 0,
            "unseen_calibration_label_count": 0, "unseen_calibration_label_rate": 0.0,
            "calibration_graph_empirical_coverage": 0.0, "calibration_graph_empirical_coverage_known_labels": 0.0,
        }
        if graph_metrics is not None:
            graph_fields.update(graph_metrics)
        rows.append({
            "dataset": dataset.name, "n_channels": int(dataset.n_channels), "n_events": int(len(prepared["labels"])),
            "n_train": int(len(train_idx)), "n_fit": int(len(fit_idx)), "n_calibration": int(len(calibration_idx)), "n_test": int(len(test_idx)),
            "fit_fraction_of_train": float(fit_fraction), "waveform_centroid_training_rows": int(len(train_idx)), "waveform_centroid_source": "raw_train_fit_plus_calibration",
            "bits": int(bits), "descriptor_dim": int(fit_features.shape[1]), "k_neighbors": int(min(k_neighbors, dataset.n_channels)), "n_units_fit": int(n_units),
            "method": method, "ranking_mode": ranking, "top_m": "" if top_m is None else int(top_m), "effective_top_m": "" if top_m is None else int(min(top_m, n_units)), "graph_coverage": "" if coverage is None else float(coverage), "top_anchors": "" if anchors is None else int(anchors),
            "ranking_l1_radius_percentile": 95.0, "ranking_note": "radius_normalized_l1 uses float isolation score d/(r+1); hardware could cross-multiply, no PPA claim",
            "directory_strategy": "all", "waveform_metric": waveform_metric,
            **graph_fields, **metrics,
        })

    for mode in dict.fromkeys(ranking_modes):
        calibration_rank = rank_unit_rows(calibration_codes, model.centroids_, model.radii_, mode=mode)
        test_rank = rank_unit_rows(test_codes, model.centroids_, model.radii_, mode=mode)
        for top_m in dict.fromkeys(int(value) for value in top_ms):
            append("fixed_top_m", mode, fixed_top_m_candidates(test_rank, units, top_m), top_m=top_m, coverage=None, anchors=None, graph_metrics=None)
        for coverage in dict.fromkeys(float(value) for value in graph_coverages):
            graph, graph_diag = build_anchor_graph(calibration_rank, calibration_labels, units, coverage=coverage, include_anchor=True)
            combined_graph = {**graph_diag, **graph_storage_bits(graph, n_units)}
            for anchors in dict.fromkeys(int(value) for value in top_anchors):
                append("indexed_confusion_graph", mode, stable_anchor_union_candidates(test_rank, graph, anchors), top_m=None, coverage=coverage, anchors=anchors, graph_metrics=combined_graph)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="5-bit indexed calibration-confusion candidate retrieval")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--ranking-modes", nargs="+", choices=("raw_l1", "radius_normalized_l1"), default=["raw_l1", "radius_normalized_l1"])
    parser.add_argument("--top-ms", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--graph-coverages", nargs="+", type=float, default=[0.95, 0.99, 1.0])
    parser.add_argument("--top-anchors", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--fit-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--waveform-metric", choices=("l1", "l2"), default="l1")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "indexed_confusion_candidate_search")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.bits <= 62 or args.k < 1 or not 0.0 < args.fit_fraction < 1.0:
        parser.error("bits, k, or fit fraction out of range")
    if not args.top_ms or any(value < 1 for value in args.top_ms) or not args.top_anchors or any(value < 1 for value in args.top_anchors):
        parser.error("top-ms and top-anchors must be positive")
    if not args.graph_coverages or any(not 0.0 <= value <= 1.0 for value in args.graph_coverages):
        parser.error("graph-coverages must be in [0, 1]")
    start = time.time()
    families: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}:
        families.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        families.append(("mearec", _iter_mearec_datasets(args.duration, args.seeds, args.mearec_npz_dir)))
    for family, datasets in families:
        family_rows: list[dict[str, Any]] = []
        for dataset in datasets:
            family_rows.extend(run_dataset_indexed_confusion_candidate_search(dataset, bits=args.bits, k_neighbors=args.k, ranking_modes=args.ranking_modes, top_ms=args.top_ms, graph_coverages=args.graph_coverages, top_anchors=args.top_anchors, fit_fraction=args.fit_fraction, waveform_metric=args.waveform_metric, verbose=not args.quiet))
            del dataset
        if family_rows:
            output = args.output_dir / f"indexed_confusion_{args.bits}bit_{family}_{int(args.duration)}s.csv"
            _write_csv(output, family_rows)
            print(f"Saved {len(family_rows)} rows: {output}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
