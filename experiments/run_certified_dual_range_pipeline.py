"""All-row certified dual box/L1 experiment with per-unit safe-exit flags.

The experiment is deliberately stricter than the ordinary dual pipeline:
chronological train/test separation is followed by a second chronological
fit/calibration separation inside train.  Spatial ranges are fitted only on
fit rows.  Calibration singleton outcomes certify a *predicted* unit for an
early exit with a Wilson lower confidence bound; test labels are never used to
open or close a safe flag.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch
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
from Spatial.experiments.run_dual_range_pipeline import (
    ordered_unique_union,
    payload_row_bits,
    summarize_dual_range_configuration,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR


def chronological_fraction_split(times: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split already sorted times near ``fraction`` without dividing ties."""
    event_times = np.asarray(times, dtype=np.int64).ravel()
    if event_times.size < 2:
        raise ValueError("at least two events are required for a chronological split")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    if np.any(event_times[1:] < event_times[:-1]):
        raise ValueError("times must already be stably sorted")
    boundaries = np.flatnonzero(event_times[1:] != event_times[:-1]) + 1
    if boundaries.size == 0:
        raise ValueError("cannot split identical timestamps across chronological subsets")
    target = event_times.size * float(fraction)
    n_left = int(boundaries[np.argmin(np.abs(boundaries - target))])
    return np.arange(n_left), np.arange(n_left, event_times.size)


def wilson_lower_bound(correct: int, support: int, z: float = 1.645) -> float:
    """One-sided Wilson lower confidence bound for a binomial precision."""
    n = int(support)
    k = int(correct)
    if n < 0 or k < 0 or k > n:
        raise ValueError("correct must be in [0, support]")
    score = float(z)
    if score < 0.0 or not np.isfinite(score):
        raise ValueError("z must be finite and non-negative")
    if n == 0:
        return 0.0
    proportion = k / n
    denominator = 1.0 + score * score / n
    center = (proportion + score * score / (2.0 * n)) / denominator
    margin = score * np.sqrt((proportion * (1.0 - proportion) + score * score / (4.0 * n)) / n) / denominator
    return float(max(0.0, center - margin))


def certify_safe_units(
    singleton_candidates: Sequence[Sequence[Any] | np.ndarray],
    true_labels: Sequence[Any] | np.ndarray,
    units: Sequence[Any] | np.ndarray,
    *,
    precision_target: float,
    min_support: int,
    z: float = 1.645,
) -> dict[str, np.ndarray]:
    """Certify each predicted unit using calibration singleton support only."""
    labels = np.asarray(true_labels)
    row_units = np.asarray(units)
    if len(singleton_candidates) != labels.size:
        raise ValueError("singleton_candidates and true_labels must have equal length")
    if not 0.0 <= float(precision_target) <= 1.0:
        raise ValueError("precision_target must be in [0, 1]")
    if int(min_support) < 0:
        raise ValueError("min_support must be non-negative")
    index = {
        value.item() if isinstance(value, np.generic) else value: position
        for position, value in enumerate(row_units)
    }
    support = np.zeros(row_units.size, dtype=np.int64)
    correct = np.zeros(row_units.size, dtype=np.int64)
    for candidates, label in zip(singleton_candidates, labels):
        if len(candidates) != 1:
            continue
        predicted = candidates[0]
        key = predicted.item() if isinstance(predicted, np.generic) else predicted
        if key not in index:
            continue
        row = index[key]
        support[row] += 1
        correct[row] += int(predicted == label)
    lower = np.asarray([wilson_lower_bound(k, n, z) for k, n in zip(correct, support)])
    safe = (support >= int(min_support)) & (lower >= float(precision_target))
    return {
        "safe": safe,
        "support": support,
        "correct": correct,
        "wilson_lower_bound": lower,
    }


def apply_certified_dual_range_policy(
    inner_candidates: Sequence[Sequence[Any] | np.ndarray],
    inner_diagnostics: dict[str, np.ndarray],
    outer_candidates: Sequence[Sequence[Any] | np.ndarray],
    outer_diagnostics: dict[str, np.ndarray],
    units: Sequence[Any] | np.ndarray,
    safe_flags: Sequence[bool] | np.ndarray,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Gate early exits by a per-predicted-unit calibration safe flag.

    ``outer_candidates`` may be a cached all-query result so grid configurations
    do not rerun equal outer L1 queries.  Diagnostics charge its row comparison
    only for rows that this policy actually triggers.
    """
    n_queries = len(inner_candidates)
    if len(outer_candidates) != n_queries:
        raise ValueError("outer_candidates must contain one cached result per inner query")
    row_units = np.asarray(units)
    flags = np.asarray(safe_flags, dtype=bool)
    if flags.shape != row_units.shape:
        raise ValueError("safe_flags must align with units")
    index = {
        value.item() if isinstance(value, np.generic) else value: position
        for position, value in enumerate(row_units)
    }
    inner_rows = np.asarray(inner_diagnostics["row_comparisons"], dtype=np.int64)
    inner_active = np.asarray(inner_diagnostics["active_row_count"], dtype=np.int64)
    inner_fallback_rows = np.asarray(
        inner_diagnostics.get("fallback_row_comparisons", np.zeros(n_queries, dtype=np.int64)),
        dtype=np.int64,
    )
    outer_rows_all = np.asarray(outer_diagnostics["row_comparisons"], dtype=np.int64)
    outer_active_all = np.asarray(outer_diagnostics["active_row_count"], dtype=np.int64)
    outer_fallback_all = np.asarray(
        outer_diagnostics.get("fallback_row_comparisons", np.zeros(n_queries, dtype=np.int64)),
        dtype=np.int64,
    )
    for name, values in (("inner row", inner_rows), ("inner active", inner_active), ("inner fallback", inner_fallback_rows), ("outer row", outer_rows_all), ("outer active", outer_active_all), ("outer fallback", outer_fallback_all)):
        if values.shape != (n_queries,):
            raise ValueError(f"{name} diagnostics must have shape ({n_queries},)")

    safe_singleton = np.zeros(n_queries, dtype=bool)
    for event, candidates in enumerate(inner_candidates):
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        key = candidate.item() if isinstance(candidate, np.generic) else candidate
        safe_singleton[event] = bool(flags[index[key]]) if key in index else False
    trigger = ~safe_singleton
    result: list[np.ndarray] = []
    for event, inner in enumerate(inner_candidates):
        if safe_singleton[event]:
            result.append(np.asarray(inner).copy())
        else:
            # This keeps a nonempty inner list if outer L1 unexpectedly misses
            # it, and leaves a fully empty list for the existing spatial fallback.
            result.append(ordered_unique_union(outer_candidates[event], inner))
    outer_rows = np.where(trigger, outer_rows_all, 0)
    outer_active = np.where(trigger, outer_active_all, 0)
    outer_fallback_rows = np.where(trigger, outer_fallback_all, 0)
    counts = np.asarray([len(rows) for rows in result], dtype=np.int64)
    return result, {
        "initial_candidate_count": np.asarray([len(rows) for rows in inner_candidates], dtype=np.int64),
        "final_candidate_count": counts,
        "widen_level": np.zeros(n_queries, dtype=np.int64),
        "fallback": np.zeros(n_queries, dtype=bool),
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
        "safe_singleton": safe_singleton,
    }


def _waveform_centroids(waveforms: np.ndarray, labels: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    return {
        unit.item() if isinstance(unit, np.generic) else unit: waveforms[labels == unit].mean(axis=0)
        for unit in units
    }


def _calibration_summary(certification: dict[str, np.ndarray]) -> dict[str, float | int]:
    support = np.asarray(certification["support"], dtype=np.int64)
    correct = np.asarray(certification["correct"], dtype=np.int64)
    safe = np.asarray(certification["safe"], dtype=bool)
    total = int(support.sum())
    return {
        "safe_unit_count": int(safe.sum()),
        "safe_unit_fraction": float(safe.mean()) if safe.size else 0.0,
        "calibration_singleton_support": total,
        "calibration_singleton_correct": int(correct.sum()),
        "calibration_singleton_empirical_precision": float(correct.sum() / total) if total else 0.0,
    }


def _safe_flag_bits_for_method(method: str) -> int:
    """A safe-exit flag is stored only for the certified policy's CAM rows."""
    return int(method == "certified_dual")


def _certified_test_metrics(
    inner_candidates: Sequence[np.ndarray], labels: np.ndarray, diagnostics: dict[str, np.ndarray]
) -> dict[str, float]:
    safe_singleton = np.asarray(diagnostics["safe_singleton"], dtype=bool)
    trigger = np.asarray(diagnostics["outer_trigger"], dtype=bool)
    singleton = np.asarray([len(rows) == 1 for rows in inner_candidates], dtype=bool)
    certified_precision = (
        float(np.mean([inner_candidates[i][0] == labels[i] for i in np.flatnonzero(safe_singleton)]))
        if np.any(safe_singleton)
        else 0.0
    )
    return {
        "certified_early_exit_fraction": float(safe_singleton.mean()) if labels.size else 0.0,
        "test_certified_singleton_precision": certified_precision,
        # Fraction of all test events which were singleton but unsafe and hence
        # were sent to outer L1.  The companion count makes the denominator clear.
        "unsafe_singleton_outer_trigger_fraction": float((singleton & trigger).mean()) if labels.size else 0.0,
        "unsafe_singleton_outer_trigger_count": int((singleton & trigger).sum()),
    }


def run_dataset_certified_dual_range_pipeline(
    dataset: Dataset,
    *,
    bits: int = 5,
    k_neighbors: int = 7,
    inner_coverages: Sequence[float] = (0.50, 0.75, 0.90, 0.95),
    precision_targets: Sequence[float] = (0.95, 0.99),
    min_supports: Sequence[int] = (5, 10, 20),
    outer_percentile: float = 99.9,
    safe_z: float = 1.645,
    fit_fraction: float = 2.0 / 3.0,
    waveform_metric: str = "l1",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run outer, uncertified, and Wilson-certified dual policies per recording."""
    if not inner_coverages or any(not 0.0 <= float(value) <= 1.0 for value in inner_coverages):
        raise ValueError("inner_coverages must contain values in [0, 1]")
    if not precision_targets or any(not 0.0 <= float(value) <= 1.0 for value in precision_targets):
        raise ValueError("precision_targets must contain values in [0, 1]")
    if not min_supports or any(int(value) < 0 for value in min_supports):
        raise ValueError("min_supports must contain non-negative integers")
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    fit_relative, calibration_relative = chronological_fraction_split(
        prepared["times"][train_idx], fit_fraction
    )
    fit_idx = train_idx[fit_relative]
    calibration_idx = train_idx[calibration_relative]
    fit_features = prepared["descriptor"][fit_idx]
    fit_labels = prepared["labels"][fit_idx]
    calibration_features = prepared["descriptor"][calibration_idx]
    calibration_labels = prepared["labels"][calibration_idx]
    test_features = prepared["descriptor"][test_idx]
    test_labels = prepared["labels"][test_idx]

    outer_model = AdaptiveRangeSearch(
        n_bits=int(bits), fixed_unit_interval=True, radius_percentile=float(outer_percentile), box_coverage=1.0,
    ).fit(fit_features, fit_labels)
    units = outer_model.units_
    n_units = len(units)
    calibration_codes = outer_model.transform(calibration_features)
    test_codes = outer_model.transform(test_features)
    outer_all_candidates, outer_all_diag = outer_model.query_codes(
        test_codes, mode="l1", active_mask=np.ones((len(test_idx), n_units), dtype=bool), ensure_nonempty=False
    )
    outer_baseline_diag = {
        **outer_all_diag,
        "inner_row_comparisons": np.zeros(len(test_idx), dtype=np.int64),
        "outer_row_comparisons": np.asarray(outer_all_diag["row_comparisons"], dtype=np.int64),
        "inner_active_row_count": np.zeros(len(test_idx), dtype=np.int64),
        "outer_active_row_count": np.asarray(outer_all_diag["active_row_count"], dtype=np.int64),
        "total_active_row_accesses": np.asarray(outer_all_diag["active_row_count"], dtype=np.int64),
    }
    # Waveform centroids intentionally use the complete original chronological
    # train half (fit + calibration), never test.  This does not affect flags.
    waveform_centroids = _waveform_centroids(
        prepared["waveforms"][train_idx], prepared["labels"][train_idx], units
    )
    descriptor_dim = int(fit_features.shape[1])
    inner_bits, expected_outer_bits, dual_bits = payload_row_bits(descriptor_dim, int(bits))
    outer_bits = int(outer_model.l1_row_bits_full_scale)
    if outer_bits != expected_outer_bits:
        raise AssertionError("full-scale outer L1 payload formula disagrees with fitted search")
    payload_note = (
        "Raw spatial dual is inner-box plus outer-L1 only: no unit id, directory, or periphery. "
        "Certified policy adds one per-unit/CAM-row safe-exit flag bit (reported as 145 bits/row for D=9,B=5); "
        "this is storage accounting, not a fabricated circuit-area estimate."
    )
    if verbose:
        print(
            f"{dataset.name}: n={len(prepared['labels'])}, train/fit/cal/test="
            f"{len(train_idx)}/{len(fit_idx)}/{len(calibration_idx)}/{len(test_idx)}, "
            f"units={n_units}, descriptor_dim={descriptor_dim}, all-row only"
        )

    rows: list[dict[str, Any]] = []

    def append_row(
        method: str,
        coverage: float | None,
        candidates: list[np.ndarray],
        diagnostics: dict[str, np.ndarray],
        *,
        inner_candidates: Sequence[np.ndarray] | None,
        outer_trigger: np.ndarray,
        precision_target: float | None,
        min_support: int | None,
        certification: dict[str, np.ndarray] | None,
        payload_bits: int,
    ) -> None:
        safe_flag_bits = _safe_flag_bits_for_method(method)
        pred, wave_meta = candidate_waveform_assign(
            candidates, prepared["waveforms"][test_idx], waveform_centroids, test_codes,
            outer_model.centroids_, units, metric=waveform_metric,
        )
        metrics = summarize_dual_range_configuration(
            candidates, test_labels, diagnostics, pred, wave_meta,
            inner_candidates=inner_candidates, outer_trigger=outer_trigger, n_units=n_units,
        )
        if certification is None:
            metrics.update({
                "safe_unit_count": None,
                "safe_unit_fraction": None,
                "calibration_singleton_support": None,
                "calibration_singleton_correct": None,
                "calibration_singleton_empirical_precision": None,
                "certified_early_exit_fraction": None,
                "test_certified_singleton_precision": None,
                "unsafe_singleton_outer_trigger_fraction": None,
                "unsafe_singleton_outer_trigger_count": None,
            })
        else:
            metrics.update(_calibration_summary(certification))
            metrics.update(_certified_test_metrics(inner_candidates or [], test_labels, diagnostics))
        rows.append({
            "dataset": dataset.name,
            "n_channels": int(dataset.n_channels),
            "n_events": int(len(prepared["labels"])),
            "n_train": int(len(train_idx)),
            "n_fit": int(len(fit_idx)),
            "n_calibration": int(len(calibration_idx)),
            "n_test": int(len(test_idx)),
            "fit_fraction_of_train": float(fit_fraction),
            "waveform_centroid_training_rows": int(len(train_idx)),
            "waveform_centroid_source": "raw_train_fit_plus_calibration",
            "n_units_fit": int(n_units),
            "bits": int(bits),
            "descriptor_dim": descriptor_dim,
            "k_neighbors": int(min(k_neighbors, dataset.n_channels)),
            "method": method,
            "directory_strategy": "all",
            "directory_neighbors": 0,
            "inner_box_coverage": "" if coverage is None else float(coverage),
            "outer_l1_percentile": float(outer_percentile),
            "precision_target": "" if precision_target is None else float(precision_target),
            "safe_min_support": "" if min_support is None else int(min_support),
            "safe_wilson_z": float(safe_z),
            "waveform_metric": waveform_metric,
            "ensure_nonempty": False,
            "search_payload_row_bits": int(payload_bits),
            "inner_box_row_bits": int(inner_bits),
            "outer_l1_full_scale_row_bits": int(outer_bits),
            "dual_raw_total_row_bits": int(dual_bits),
            "safe_exit_flag_bits_per_unit": safe_flag_bits,
            "safe_exit_flag_bits_per_cam_row": safe_flag_bits,
            "certified_dual_payload_row_bits_with_flag": int(dual_bits + 1),
            "outer_l1_only_payload_row_bits": int(outer_bits),
            "search_payload_row_bits_note": payload_note,
            **metrics,
        })

    # Outer has no inner coverage or safety hyperparameter and is emitted once.
    append_row(
        "outer_l1_only", None, outer_all_candidates, outer_baseline_diag,
        inner_candidates=None, outer_trigger=np.ones(len(test_idx), dtype=bool),
        precision_target=None, min_support=None, certification=None, payload_bits=outer_bits,
    )
    for coverage in dict.fromkeys(float(value) for value in inner_coverages):
        inner_model = AdaptiveRangeSearch(
            n_bits=int(bits), fixed_unit_interval=True, radius_percentile=float(outer_percentile), box_coverage=coverage,
        ).fit(fit_features, fit_labels)
        np.testing.assert_array_equal(inner_model.transform(calibration_features), calibration_codes)
        np.testing.assert_array_equal(inner_model.transform(test_features), test_codes)
        calibration_inner, _ = inner_model.query_codes(
            calibration_codes, mode="box", active_mask=np.ones((len(calibration_idx), n_units), dtype=bool), ensure_nonempty=False
        )
        test_inner, test_inner_diag = inner_model.query_codes(
            test_codes, mode="box", active_mask=np.ones((len(test_idx), n_units), dtype=bool), ensure_nonempty=False
        )
        # Uncertified dual is one cached outer result plus unconditional singleton early exits.
        uncertified_flags = np.ones(n_units, dtype=bool)
        uncertified_candidates, uncertified_diag = apply_certified_dual_range_policy(
            test_inner, test_inner_diag, outer_all_candidates, outer_all_diag, units, uncertified_flags
        )
        append_row(
            "uncertified_dual", coverage, uncertified_candidates, uncertified_diag,
            inner_candidates=test_inner, outer_trigger=uncertified_diag["outer_trigger"],
            precision_target=None, min_support=None, certification=None, payload_bits=dual_bits,
        )
        for target in dict.fromkeys(float(value) for value in precision_targets):
            for minimum in dict.fromkeys(int(value) for value in min_supports):
                certification = certify_safe_units(
                    calibration_inner, calibration_labels, units,
                    precision_target=target, min_support=minimum, z=safe_z,
                )
                certified_candidates, certified_diag = apply_certified_dual_range_policy(
                    test_inner, test_inner_diag, outer_all_candidates, outer_all_diag,
                    units, certification["safe"],
                )
                append_row(
                    "certified_dual", coverage, certified_candidates, certified_diag,
                    inner_candidates=test_inner, outer_trigger=certified_diag["outer_trigger"],
                    precision_target=target, min_support=minimum, certification=certification,
                    payload_bits=dual_bits + 1,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="All-row Wilson-certified dual box/L1 candidate pipeline")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--inner-coverages", nargs="+", type=float, default=[0.50, 0.75, 0.90, 0.95])
    parser.add_argument("--precision-targets", nargs="+", type=float, default=[0.95, 0.99])
    parser.add_argument("--min-supports", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--outer-percentile", type=float, default=99.9)
    parser.add_argument("--safe-z", type=float, default=1.645)
    parser.add_argument("--fit-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--waveform-metric", choices=("l1", "l2"), default="l1")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "certified_dual_range_pipeline")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.bits <= 62:
        parser.error("--bits must be in [1, 62]")
    if args.k < 1:
        parser.error("--k must be positive")
    if not args.inner_coverages or any(not 0.0 <= value <= 1.0 for value in args.inner_coverages):
        parser.error("--inner-coverages must contain values in [0, 1]")
    if not args.precision_targets or any(not 0.0 <= value <= 1.0 for value in args.precision_targets):
        parser.error("--precision-targets must contain values in [0, 1]")
    if not args.min_supports or any(value < 0 for value in args.min_supports):
        parser.error("--min-supports must contain non-negative integers")
    if not 0.0 <= args.outer_percentile <= 100.0:
        parser.error("--outer-percentile must be in [0, 100]")
    if args.safe_z < 0.0:
        parser.error("--safe-z must be non-negative")
    if not 0.0 < args.fit_fraction < 1.0:
        parser.error("--fit-fraction must be in (0, 1)")

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
            family_rows.extend(run_dataset_certified_dual_range_pipeline(
                dataset, bits=args.bits, k_neighbors=args.k, inner_coverages=args.inner_coverages,
                precision_targets=args.precision_targets, min_supports=args.min_supports,
                outer_percentile=args.outer_percentile, safe_z=args.safe_z,
                fit_fraction=args.fit_fraction, waveform_metric=args.waveform_metric, verbose=not args.quiet,
            ))
            del dataset
        if not family_rows:
            print(f"{family}: no datasets found; skipping")
            continue
        percentile_tag = str(args.outer_percentile).replace(".", "p")
        output = args.output_dir / (
            f"certified_dual_{args.bits}bit_range_pipeline_{family}_{int(args.duration)}s_outerp{percentile_tag}.csv"
        )
        _write_csv(output, family_rows)
        print(f"Saved {len(family_rows)} rows: {output}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
