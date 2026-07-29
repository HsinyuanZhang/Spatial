"""All-row dual-range spatial candidate-search experiment.

This benchmark isolates a two-stage *spatial* policy.  A narrow per-unit box
is queried first; exactly one box candidate is a high-confidence early exit.
Only empty or ambiguous box results invoke a wider per-unit L1 search.  The
final list is the ordered union of the outer L1 candidates followed by any box
candidates absent from that list, so an outer miss cannot erase an inner hit.

There is deliberately no directory in this first experiment: all unit rows are
enabled for both stages.  There is also no top-k addition, GT-aware repair, or
``ensure_nonempty`` search fallback.  Empty final lists reach the existing
spatial fallback in :func:`candidate_waveform_assign` unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Match the existing experiment's direct-script behavior as well as
# ``python -m Spatial.experiments.run_dual_range_pipeline`` from the parent.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch, evaluate_candidate_search
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import (
    REPO_ROOT,
    _chronological_split,
    _iter_hj_datasets,
    _iter_mearec_datasets,
    _write_csv,
    candidate_waveform_assign,
    prepare_dataset_events,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR


def ordered_unique_union(
    first: Sequence[Any] | np.ndarray, second: Sequence[Any] | np.ndarray
) -> np.ndarray:
    """Return the stable de-duplicated union, retaining ``first`` order.

    The policy puts outer candidates first, then retains any inner candidate
    which the outer L1 region unexpectedly omitted.  Unit labels need not be
    contiguous integers.
    """
    result: list[Any] = []
    for candidate in list(first) + list(second):
        if not any(candidate == previous for previous in result):
            result.append(candidate)
    return np.asarray(result)


def payload_row_bits(descriptor_dim: int, bits: int) -> tuple[int, int, int]:
    """Return raw inner-box, full-scale outer-L1, and unshared dual row bits."""
    if int(descriptor_dim) < 1 or int(bits) < 1:
        raise ValueError("descriptor_dim and bits must be positive")
    dimension = int(descriptor_dim)
    code_bits = int(bits)
    inner_box = 2 * dimension * code_bits
    radius_bits = int(np.ceil(np.log2(dimension * ((1 << code_bits) - 1) + 1)))
    outer_l1 = dimension * code_bits + radius_bits
    return inner_box, outer_l1, inner_box + outer_l1


def apply_dual_range_policy(
    inner_candidates: Sequence[Sequence[Any] | np.ndarray],
    inner_diagnostics: dict[str, np.ndarray],
    outer_trigger_candidates: Sequence[Sequence[Any] | np.ndarray],
    outer_trigger_diagnostics: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Apply the dual policy without querying outer rows for inner singletons.

    ``outer_trigger_candidates`` and its diagnostics are ordered by the subset
    of queries for which the inner count is zero or greater than one.  The
    returned comparison totals therefore expose the actual gated work rather
    than charging every query an outer all-row scan.
    """
    n_queries = len(inner_candidates)
    inner_counts = np.asarray([len(rows) for rows in inner_candidates], dtype=np.int64)
    trigger = inner_counts != 1
    n_trigger = int(trigger.sum())
    if len(outer_trigger_candidates) != n_trigger:
        raise ValueError("outer_trigger_candidates must align with triggered inner queries")
    for key in ("row_comparisons", "active_row_count"):
        values = np.asarray(outer_trigger_diagnostics[key])
        if values.shape != (n_trigger,):
            raise ValueError(f"outer_trigger_diagnostics['{key}'] must have shape ({n_trigger},)")
    inner_rows = np.asarray(inner_diagnostics["row_comparisons"], dtype=np.int64)
    inner_active = np.asarray(inner_diagnostics["active_row_count"], dtype=np.int64)
    inner_fallback_rows = np.asarray(
        inner_diagnostics.get("fallback_row_comparisons", np.zeros(n_queries, dtype=np.int64)),
        dtype=np.int64,
    )
    if inner_rows.shape != (n_queries,):
        raise ValueError(f"inner_diagnostics['row_comparisons'] must have shape ({n_queries},)")
    if inner_active.shape != (n_queries,) or inner_fallback_rows.shape != (n_queries,):
        raise ValueError("inner active-row and fallback-row diagnostics must align with queries")

    candidates: list[np.ndarray] = []
    outer_rows = np.zeros(n_queries, dtype=np.int64)
    outer_active = np.zeros(n_queries, dtype=np.int64)
    outer_fallback_rows = np.zeros(n_queries, dtype=np.int64)
    outer_fallback_source = np.asarray(
        outer_trigger_diagnostics.get("fallback_row_comparisons", np.zeros(n_trigger, dtype=np.int64)),
        dtype=np.int64,
    )
    if outer_fallback_source.shape != (n_trigger,):
        raise ValueError("outer fallback-row diagnostics must align with triggered queries")
    outer_index = 0
    for query_index, inner in enumerate(inner_candidates):
        if not trigger[query_index]:
            # A unique (even incorrect) inner candidate is an intentional
            # early exit.  Do not look at labels or invoke the outer stage.
            candidates.append(np.asarray(inner).copy())
            continue
        outer = outer_trigger_candidates[outer_index]
        candidates.append(ordered_unique_union(outer, inner))
        outer_rows[query_index] = int(outer_trigger_diagnostics["row_comparisons"][outer_index])
        outer_active[query_index] = int(outer_trigger_diagnostics["active_row_count"][outer_index])
        outer_fallback_rows[query_index] = int(outer_fallback_source[outer_index])
        outer_index += 1

    final_counts = np.asarray([len(rows) for rows in candidates], dtype=np.int64)
    diagnostics = {
        "initial_candidate_count": inner_counts,
        "final_candidate_count": final_counts,
        "widen_level": np.zeros(n_queries, dtype=np.int64),
        "fallback": np.zeros(n_queries, dtype=bool),
        # Backward-compatible simultaneous active rows for the first/inner
        # access.  Across-stage accesses are reported separately below.
        "active_row_count": inner_active,
        "row_comparisons": inner_rows + outer_rows,
        "inner_row_comparisons": inner_rows,
        "outer_row_comparisons": outer_rows,
        "fallback_row_comparisons": inner_fallback_rows + outer_fallback_rows,
        "total_row_comparisons_including_fallback": (
            inner_rows + outer_rows + inner_fallback_rows + outer_fallback_rows
        ),
        "inner_active_row_count": inner_active,
        "outer_active_row_count": outer_active,
        "total_active_row_accesses": inner_active + outer_active,
        "outer_trigger": trigger,
    }
    return candidates, diagnostics


def _waveform_centroids(
    waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray
) -> dict[Any, np.ndarray]:
    return {
        unit.item() if isinstance(unit, np.generic) else unit: waveforms[labels == unit].mean(axis=0)
        for unit in units
    }


def summarize_dual_range_configuration(
    candidates: Sequence[np.ndarray],
    labels: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    pred: np.ndarray,
    wave_meta: dict[str, np.ndarray],
    *,
    inner_candidates: Sequence[np.ndarray] | None,
    outer_trigger: np.ndarray,
    n_units: int,
) -> dict[str, float | int | None]:
    """Produce one CSV-ready report, including explicit dual-stage observability."""
    report: dict[str, float | int | None] = dict(evaluate_candidate_search(candidates, labels, diagnostics))
    n_queries = labels.size
    reads = np.asarray(wave_meta["candidate_waveform_reads"], dtype=np.int64)
    correct = np.asarray(pred == labels, dtype=bool)
    inner_rows = np.asarray(diagnostics["inner_row_comparisons"], dtype=np.int64)
    outer_rows = np.asarray(diagnostics["outer_row_comparisons"], dtype=np.int64)
    inner_active = np.asarray(
        diagnostics.get("inner_active_row_count", diagnostics["active_row_count"]), dtype=np.int64
    )
    outer_active = np.asarray(
        diagnostics.get("outer_active_row_count", np.zeros(n_queries, dtype=np.int64)), dtype=np.int64
    )
    range_fallback_rows = np.asarray(
        diagnostics.get("fallback_row_comparisons", np.zeros(n_queries, dtype=np.int64)), dtype=np.int64
    )
    spatial_fallback_rows = np.asarray(
        wave_meta.get("spatial_fallback_row_comparisons", np.zeros(n_queries, dtype=np.int64)),
        dtype=np.int64,
    )
    triggered = np.asarray(outer_trigger, dtype=bool)
    if inner_rows.shape != (n_queries,) or outer_rows.shape != (n_queries,):
        raise ValueError("dual row-comparison diagnostics must align with labels")
    if inner_candidates is None:
        inner_unique_fraction: float | None = None
        inner_unique_precision: float | None = None
    else:
        counts = np.asarray([len(rows) for rows in inner_candidates], dtype=np.int64)
        unique = counts == 1
        inner_unique_fraction = float(unique.mean()) if n_queries else 0.0
        inner_unique_precision = (
            float(np.mean([rows[0] == label for rows, label in zip(inner_candidates, labels) if len(rows) == 1]))
            if np.any(unique)
            else 0.0
        )
    global_reads = int(n_queries * int(n_units))
    report.update({
        "inner_unique_fraction": inner_unique_fraction,
        "inner_unique_precision": inner_unique_precision,
        "outer_trigger_fraction": float(triggered.mean()) if n_queries else 0.0,
        "final_true_unit_candidate_recall": report["true_unit_candidate_recall"],
        "stage2_activation": report["stage2_activation_fraction"],
        "mean_inner_row_comparisons": float(inner_rows.mean()) if n_queries else 0.0,
        "mean_outer_row_comparisons": float(outer_rows.mean()) if n_queries else 0.0,
        "total_level1_row_comparisons": int((inner_rows + outer_rows).sum()),
        "mean_inner_active_rows": float(inner_active.mean()) if n_queries else 0.0,
        "mean_outer_active_rows": float(outer_active.mean()) if n_queries else 0.0,
        "total_active_row_accesses": int((inner_active + outer_active).sum()),
        "range_fallback_row_comparisons": int(range_fallback_rows.sum()),
        "spatial_fallback_row_comparisons": int(spatial_fallback_rows.sum()),
        "fallback_row_comparisons": int(range_fallback_rows.sum() + spatial_fallback_rows.sum()),
        "total_row_comparisons_including_fallback": int(
            (inner_rows + outer_rows + range_fallback_rows + spatial_fallback_rows).sum()
        ),
        "candidate_waveform_reads_per_event": float(reads.mean()) if n_queries else 0.0,
        "candidate_waveform_reads_total": int(reads.sum()),
        "waveform_fallback_rate": float(np.mean(wave_meta["waveform_fallback"])) if n_queries else 0.0,
        "global_all_unit_waveform_reads": global_reads,
        "candidate_waveform_read_fraction_vs_global": reads.sum() / max(global_reads, 1),
        "waveform_read_reduction_vs_global": 1.0 - reads.sum() / max(global_reads, 1),
        "global_waveform_reduction_factor": float(global_reads / reads.sum()) if reads.sum() else float("inf"),
        "waveform_read_reduction_factor_global_over_actual": float(global_reads / reads.sum()) if reads.sum() else float("inf"),
        "final_waveform_accuracy": float(correct.mean()) if n_queries else 0.0,
    })
    return report
def run_dataset_dual_range_pipeline(
    dataset: Dataset,
    *,
    bits: int = 5,
    k_neighbors: int = 7,
    inner_coverages: Sequence[float] = (0.50, 0.75, 0.90, 0.95),
    outer_percentile: float = 99.9,
    waveform_metric: str = "l1",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark all-row inner-box, outer-L1, and gated dual candidates."""
    if not inner_coverages or any(not 0.0 <= float(value) <= 1.0 for value in inner_coverages):
        raise ValueError("inner_coverages must contain values in [0, 1]")
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    train_features = prepared["descriptor"][train_idx]
    test_features = prepared["descriptor"][test_idx]
    train_labels = prepared["labels"][train_idx]
    test_labels = prepared["labels"][test_idx]
    outer_model = AdaptiveRangeSearch(
        n_bits=int(bits), fixed_unit_interval=True, radius_percentile=float(outer_percentile), box_coverage=1.0,
    ).fit(train_features, train_labels)
    query_codes = outer_model.transform(test_features)
    units = outer_model.units_
    n_units = len(units)
    all_rows = np.ones((len(test_idx), n_units), dtype=bool)
    waveform_centroids = _waveform_centroids(prepared["waveforms"][train_idx], train_labels, units)
    outer_all_candidates, outer_all_diag = outer_model.query_codes(
        query_codes, mode="l1", active_mask=all_rows, ensure_nonempty=False
    )
    outer_baseline_diag = {
        **outer_all_diag,
        "inner_row_comparisons": np.zeros(len(test_idx), dtype=np.int64),
        "outer_row_comparisons": np.asarray(outer_all_diag["row_comparisons"], dtype=np.int64),
        "inner_active_row_count": np.zeros(len(test_idx), dtype=np.int64),
        "outer_active_row_count": np.asarray(outer_all_diag["active_row_count"], dtype=np.int64),
        "total_active_row_accesses": np.asarray(outer_all_diag["active_row_count"], dtype=np.int64),
    }
    descriptor_dim = int(train_features.shape[1])
    inner_bits, expected_outer_bits, dual_bits = payload_row_bits(descriptor_dim, int(bits))
    outer_bits = int(outer_model.l1_row_bits_full_scale)
    if outer_bits != expected_outer_bits:
        raise AssertionError("full-scale outer L1 payload formula disagrees with fitted search")
    payload_note = (
        "No unit id, directory, or periphery; dual payload is the raw non-shared "
        "inner-box plus outer-L1 total (no shared-field optimization assumed)."
    )
    if verbose:
        print(
            f"{dataset.name}: n={len(prepared['labels'])}, train/test={len(train_idx)}/{len(test_idx)}, "
            f"units={n_units}, descriptor_dim={descriptor_dim}, all-row only"
        )

    rows: list[dict[str, Any]] = []

    def append_row(
        method: str,
        coverage: float,
        candidates: list[np.ndarray],
        diagnostics: dict[str, np.ndarray],
        *,
        inner_candidates: Sequence[np.ndarray] | None,
        outer_trigger: np.ndarray,
        payload_bits: int,
    ) -> None:
        pred, wave_meta = candidate_waveform_assign(
            candidates, prepared["waveforms"][test_idx], waveform_centroids, query_codes,
            outer_model.centroids_, units, metric=waveform_metric,
        )
        metrics = summarize_dual_range_configuration(
            candidates, test_labels, diagnostics, pred, wave_meta,
            inner_candidates=inner_candidates, outer_trigger=outer_trigger, n_units=n_units,
        )
        rows.append({
            "dataset": dataset.name,
            "n_channels": int(dataset.n_channels),
            "n_events": int(len(prepared["labels"])),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_units_train": int(n_units),
            "bits": int(bits),
            "descriptor_dim": descriptor_dim,
            "k_neighbors": int(min(k_neighbors, dataset.n_channels)),
            "method": method,
            "directory_strategy": "all",
            "directory_neighbors": 0,
            "inner_box_coverage": float(coverage),
            "outer_l1_percentile": float(outer_percentile),
            "waveform_metric": waveform_metric,
            "ensure_nonempty": False,
            "search_payload_row_bits": int(payload_bits),
            "inner_box_row_bits": int(inner_bits),
            "outer_l1_full_scale_row_bits": int(outer_bits),
            "dual_raw_total_row_bits": int(dual_bits),
            "search_payload_row_bits_note": payload_note,
            **metrics,
        })

    for requested_coverage in dict.fromkeys(float(value) for value in inner_coverages):
        inner_model = AdaptiveRangeSearch(
            n_bits=int(bits), fixed_unit_interval=True, radius_percentile=float(outer_percentile),
            box_coverage=requested_coverage,
        ).fit(train_features, train_labels)
        # Fixed [0, 1] quantization makes the two fitted code spaces identical;
        # this assertion catches accidental calibration drift in future edits.
        np.testing.assert_array_equal(inner_model.transform(test_features), query_codes)
        inner_candidates, inner_diag = inner_model.query_codes(
            query_codes, mode="box", active_mask=all_rows, ensure_nonempty=False
        )
        triggers = np.asarray([len(rows_) != 1 for rows_ in inner_candidates], dtype=bool)
        outer_trigger_candidates, outer_trigger_diag = outer_model.query_codes(
            query_codes[triggers], mode="l1", active_mask=np.ones((int(triggers.sum()), n_units), dtype=bool),
            ensure_nonempty=False,
        )
        dual_candidates, dual_diag = apply_dual_range_policy(
            inner_candidates, inner_diag, outer_trigger_candidates, outer_trigger_diag
        )
        inner_baseline_diag = {
            **inner_diag,
            "inner_row_comparisons": np.asarray(inner_diag["row_comparisons"], dtype=np.int64),
            "outer_row_comparisons": np.zeros(len(test_idx), dtype=np.int64),
            "inner_active_row_count": np.asarray(inner_diag["active_row_count"], dtype=np.int64),
            "outer_active_row_count": np.zeros(len(test_idx), dtype=np.int64),
            "total_active_row_accesses": np.asarray(inner_diag["active_row_count"], dtype=np.int64),
        }
        append_row(
            "inner_box_only", requested_coverage, inner_candidates, inner_baseline_diag,
            inner_candidates=inner_candidates, outer_trigger=np.zeros(len(test_idx), dtype=bool),
            payload_bits=inner_bits,
        )
        append_row(
            "outer_l1_only", requested_coverage, outer_all_candidates, outer_baseline_diag,
            inner_candidates=None, outer_trigger=np.ones(len(test_idx), dtype=bool), payload_bits=outer_bits,
        )
        append_row(
            "dual_box_l1", requested_coverage, dual_candidates, dual_diag,
            inner_candidates=inner_candidates, outer_trigger=triggers, payload_bits=dual_bits,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="All-row dual box/L1 spatial candidate pipeline")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--k", type=int, default=7, help="Spatial K-neighbor footprint width")
    parser.add_argument("--inner-coverages", nargs="+", type=float, default=[0.50, 0.75, 0.90, 0.95])
    parser.add_argument("--outer-percentile", type=float, default=99.9)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--waveform-metric", choices=("l1", "l2"), default="l1")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "dual_range_pipeline")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.bits <= 62:
        parser.error("--bits must be in [1, 62]")
    if args.k < 1:
        parser.error("--k must be positive")
    if not args.inner_coverages or any(not 0.0 <= value <= 1.0 for value in args.inner_coverages):
        parser.error("--inner-coverages must contain values in [0, 1]")
    if not 0.0 <= args.outer_percentile <= 100.0:
        parser.error("--outer-percentile must be in [0, 100]")

    t0 = time.time()
    families: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}:
        families.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        families.append(("mearec", _iter_mearec_datasets(args.duration, args.seeds, args.mearec_npz_dir)))
    for family, datasets in families:
        family_rows: list[dict[str, Any]] = []
        for dataset in datasets:
            if not args.quiet:
                print(f"\n########## {family}: {dataset.name} ##########")
            family_rows.extend(run_dataset_dual_range_pipeline(
                dataset, bits=args.bits, k_neighbors=args.k, inner_coverages=args.inner_coverages,
                outer_percentile=args.outer_percentile, waveform_metric=args.waveform_metric,
                verbose=not args.quiet,
            ))
            del dataset
        if not family_rows:
            print(f"{family}: no datasets found; skipping")
            continue
        percentile_tag = str(args.outer_percentile).replace(".", "p")
        output = args.output_dir / (
            f"dual_{args.bits}bit_range_pipeline_{family}_{int(args.duration)}s_outerp{percentile_tag}.csv"
        )
        _write_csv(output, family_rows)
        print(f"Saved {len(family_rows)} rows: {output}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
