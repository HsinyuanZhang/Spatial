"""Isolated compact digital waveform refinement on frozen spatial candidates.

Candidate lists are built once from GT event rows and then shared unchanged by
every Level-2 representation.  This makes compact-code results a refinement
comparison, rather than a different first-stage candidate-search experiment.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch
from Spatial.data.loader import Dataset, list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import (
    REPO_ROOT,
    _chronological_split,
    _iter_mearec_datasets,
    prepare_dataset_events,
)


WAVEFORM_SAMPLES = 64
MORPHOLOGY_DIM = 12


def quantize_signed_uniform(values: np.ndarray, bits: int) -> np.ndarray:
    """Symmetrically quantize normalized waveform values to signed integer codes.

    Inputs are clipped to ``[-1, 1]`` and use ``rint(x * (2**(bits-1)-1))``.
    Thus the 4-bit payload uses codes ``[-7, 7]`` (the asymmetric -8 code is
    intentionally unused), making zero exact and positive/negative scales
    identical.  NumPy's deterministic round-to-nearest-even applies at ties.
    """
    if isinstance(bits, bool) or int(bits) != bits or int(bits) < 2:
        raise ValueError("signed waveform bits must be an integer >= 2")
    maximum = (1 << (int(bits) - 1)) - 1
    return np.clip(np.rint(np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0) * maximum), -maximum, maximum).astype(np.int64)


def quantize_unit_interval(values: np.ndarray, bits: int) -> np.ndarray:
    """Quantize already-normalized morphology features to unsigned fixed codes."""
    if isinstance(bits, bool) or int(bits) != bits or int(bits) < 1:
        raise ValueError("unsigned bits must be an integer >= 1")
    maximum = (1 << int(bits)) - 1
    return np.clip(np.rint(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) * maximum), 0, maximum).astype(np.int64)


def fisher_tap_indices(waveforms_train: np.ndarray, labels_train: np.ndarray, n_taps: int) -> np.ndarray:
    """Choose training-only Fisher taps, with ascending sample index tie-break.

    The score at each sample is ``between_sum_squares / (within_sum_squares +
    eps)``.  It uses no test rows and no learned projection: selected samples
    are directly quantized and compared with SAD downstream.
    """
    waveforms = np.asarray(waveforms_train, dtype=np.float64)
    labels = np.asarray(labels_train)
    if waveforms.ndim != 2 or waveforms.shape[0] != labels.size:
        raise ValueError("waveforms_train must be 2D and align with labels_train")
    if not 1 <= int(n_taps) <= waveforms.shape[1]:
        raise ValueError("n_taps must be in [1, waveform sample count]")
    overall = waveforms.mean(axis=0)
    between = np.zeros(waveforms.shape[1], dtype=np.float64)
    within = np.zeros(waveforms.shape[1], dtype=np.float64)
    for unit in np.unique(labels):
        values = waveforms[labels == unit]
        mean = values.mean(axis=0)
        between += values.shape[0] * (mean - overall) ** 2
        within += ((values - mean) ** 2).sum(axis=0)
    score = between / np.maximum(within, 1e-12)
    indices = np.arange(waveforms.shape[1])
    order = np.lexsort((indices, -score))
    return order[: int(n_taps)].astype(np.int64)


def morphology_features(waveforms: np.ndarray) -> np.ndarray:
    """Return 12 fixed-window streaming-friendly features normalized to [0, 1].

    Features are: global min, max, P2P, abs peak, argmin, argmax, means of
    samples [0:16), [24:40), [48:64), negative area, positive area, and sign
    transition count.  Operations are fixed-window sums, min/max, argmin/
    argmax, comparisons, and absolute/subtractive accumulation—no PCA, FFT,
    learned projection, or test-fitted scaling.  Signed values/ranges are
    converted by fixed analytical normalization before one common unsigned SAD.
    """
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != WAVEFORM_SAMPLES:
        raise ValueError("waveforms must have shape (n_events, 64)")
    clipped = np.clip(values, -1.0, 1.0)
    minimum = clipped.min(axis=1)
    maximum = clipped.max(axis=1)
    p2p = maximum - minimum
    abs_peak = np.max(np.abs(clipped), axis=1)
    argmin = np.argmin(clipped, axis=1)
    argmax = np.argmax(clipped, axis=1)
    pre = clipped[:, :16].mean(axis=1)
    center = clipped[:, 24:40].mean(axis=1)
    post = clipped[:, 48:64].mean(axis=1)
    negative_area = np.maximum(-clipped, 0.0).sum(axis=1) / WAVEFORM_SAMPLES
    positive_area = np.maximum(clipped, 0.0).sum(axis=1) / WAVEFORM_SAMPLES
    transitions = (np.signbit(clipped[:, 1:]) != np.signbit(clipped[:, :-1])).sum(axis=1)
    mixed_range = np.column_stack([
        (minimum + 1.0) / 2.0, (maximum + 1.0) / 2.0, p2p / 2.0, abs_peak,
        argmin / (WAVEFORM_SAMPLES - 1), argmax / (WAVEFORM_SAMPLES - 1),
        (pre + 1.0) / 2.0, (center + 1.0) / 2.0, (post + 1.0) / 2.0,
        negative_area, positive_area, transitions / (WAVEFORM_SAMPLES - 1),
    ])
    return np.clip(mixed_range, 0.0, 1.0)


def fit_float_templates(values_train: np.ndarray, labels_train: np.ndarray, units: np.ndarray) -> dict[Any, np.ndarray]:
    """Fit float L1 teacher templates from training rows only."""
    return {
        _key(unit): np.asarray(values_train, dtype=np.float64)[np.asarray(labels_train) == unit].mean(axis=0)
        for unit in units
    }


def fit_integer_templates(
    codes_train: np.ndarray,
    labels_train: np.ndarray,
    units: np.ndarray,
    lower: int,
    upper: int,
) -> dict[Any, np.ndarray]:
    """Transform first, then compute/rint/clip each integer unit template."""
    codes = np.asarray(codes_train, dtype=np.int64)
    labels = np.asarray(labels_train)
    return {
        _key(unit): np.clip(np.rint(codes[labels == unit].mean(axis=0)), lower, upper).astype(np.int64)
        for unit in units
    }


def _key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def frozen_candidate_sources(
    descriptor_train: np.ndarray,
    labels_train: np.ndarray,
    descriptor_test: np.ndarray,
    percentiles: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Build pXX L1 and all-unit candidate sources once, with all rows enabled."""
    sources: dict[str, dict[str, Any]] = {}
    for percentile in percentiles:
        label = f"per_unit_5bit_l1_p{float(percentile):g}"
        model = AdaptiveRangeSearch(
            n_bits=5, fixed_unit_interval=True, radius_percentile=float(percentile)
        ).fit(descriptor_train, labels_train)
        candidates, diagnostics = model.query(descriptor_test, mode="l1")
        sources[label] = {
            "candidates": candidates, "diagnostics": diagnostics, "units": model.units_,
            "spatial_codes": model.transform(descriptor_test), "spatial_centroids": model.centroids_,
        }
    # All-unit stress baseline shares the fitted fixed-[0,1] spatial codes but
    # intentionally bypasses any range gate.  It never uses GT repair/top-k.
    if not sources:
        raise ValueError("at least one candidate percentile is required")
    reference = next(iter(sources.values()))
    n_test = descriptor_test.shape[0]
    n_units = reference["units"].size
    all_candidates = [reference["units"].copy() for _ in range(n_test)]
    sources["all_units"] = {
        "candidates": all_candidates,
        "diagnostics": {
            "initial_candidate_count": np.full(n_test, n_units, dtype=np.int64),
            "final_candidate_count": np.full(n_test, n_units, dtype=np.int64),
            "widen_level": np.zeros(n_test, dtype=np.int64),
            "fallback": np.zeros(n_test, dtype=bool),
            "active_row_count": np.full(n_test, n_units, dtype=np.int64),
            "row_comparisons": np.full(n_test, n_units, dtype=np.int64),
        },
        "units": reference["units"], "spatial_codes": reference["spatial_codes"],
        "spatial_centroids": reference["spatial_centroids"],
    }
    return sources


def refine_frozen_candidates(
    candidates: Sequence[np.ndarray],
    test_values: np.ndarray,
    templates: dict[Any, np.ndarray],
    spatial_codes: np.ndarray,
    spatial_centroids: np.ndarray,
    spatial_units: np.ndarray,
    *,
    metric: str = "l1",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Refine frozen lists; unique exits early and zero uses spatial fallback.

    This function never changes a candidate list, appends top-k rows, or reads
    a waveform template outside an originally ambiguous list.
    """
    values = np.asarray(test_values)
    units = np.asarray(spatial_units)
    codes = np.asarray(spatial_codes, dtype=np.int64)
    centers = np.asarray(spatial_centroids, dtype=np.int64)
    if len(candidates) != values.shape[0] or codes.shape[0] != values.shape[0]:
        raise ValueError("candidate, test-value, and spatial-code rows must align")
    if metric not in {"l1", "sad"}:
        raise ValueError("metric must be 'l1' or 'sad'")
    known = {_key(unit) for unit in units}
    predictions = np.empty(values.shape[0], dtype=units.dtype)
    reads = np.zeros(values.shape[0], dtype=np.int64)
    fallback = np.zeros(values.shape[0], dtype=bool)
    for row, candidate_row in enumerate(candidates):
        selected = list(candidate_row)
        if len(selected) == 1 and _key(selected[0]) in known:
            predictions[row] = selected[0]
            continue
        if len(selected) > 1:
            valid = [candidate for candidate in selected if _key(candidate) in templates]
            if valid:
                template_matrix = np.stack([templates[_key(candidate)] for candidate in valid])
                distance = np.abs(template_matrix - values[row]).sum(axis=1)
                predictions[row] = valid[int(np.argmin(distance))]
                reads[row] = len(valid)
                continue
        predictions[row] = units[int(np.argmin(np.abs(centers - codes[row]).sum(axis=1)))]
        fallback[row] = True
    return predictions, {"candidate_template_reads": reads, "spatial_fallback": fallback}


def traffic_accounting(candidate_reads: np.ndarray, template_bits_per_unit: int, n_units: int) -> dict[str, float | int]:
    """Report template payload traffic against a global 64-sample 4-bit scan."""
    reads = np.asarray(candidate_reads, dtype=np.int64).ravel()
    actual = int(reads.sum()) * int(template_bits_per_unit)
    global_bits = int(reads.size) * int(n_units) * WAVEFORM_SAMPLES * 4
    return {
        "candidate_template_reads": int(reads.sum()),
        "actual_template_bits_read": actual,
        "global_64x4bit_template_bits": global_bits,
        "template_bit_traffic_reduction": float(1.0 - actual / max(global_bits, 1)),
        "global_64x4bit_over_actual_bit_traffic_factor": (
            float(global_bits / actual) if actual else float("inf")
        ),
    }


def refinement_metrics(
    candidates: Sequence[np.ndarray],
    true_labels: np.ndarray,
    predictions: np.ndarray,
    teacher_predictions: np.ndarray,
    candidate_reads: np.ndarray,
    template_bits_per_unit: int,
    n_units: int,
    feature_dim: int,
    shared_tap_index_bits: int = 0,
) -> dict[str, float | int]:
    """Summarize one representation on a fixed candidate source."""
    labels = np.asarray(true_labels)
    counts = np.asarray([len(row) for row in candidates], dtype=np.int64)
    ambiguous = counts > 1
    recall = np.asarray([label in row for label, row in zip(labels, candidates)], dtype=bool)
    metrics: dict[str, float | int] = {
        "candidate_true_unit_recall": float(recall.mean()) if labels.size else 0.0,
        "mean_candidates": float(counts.mean()) if labels.size else 0.0,
        "ambiguous_events": int(ambiguous.sum()),
        "ambiguous_fraction": float(ambiguous.mean()) if labels.size else 0.0,
        "overall_accuracy": float(np.mean(predictions == labels)) if labels.size else 0.0,
        "ambiguous_only_accuracy": float(np.mean(predictions[ambiguous] == labels[ambiguous])) if np.any(ambiguous) else 0.0,
        "float_teacher_prediction_agreement": float(np.mean(predictions == teacher_predictions)) if labels.size else 0.0,
        "accuracy_delta_vs_float_teacher": (
            float(np.mean(predictions == labels) - np.mean(teacher_predictions == labels)) if labels.size else 0.0
        ),
        "template_bits_per_unit": int(template_bits_per_unit),
        "template_bits_per_candidate_read": int(template_bits_per_unit),
        # Fisher tap locations are one recording-wide configuration table, not
        # copied into each unit template or re-read per candidate template.
        "shared_tap_index_bits": int(shared_tap_index_bits),
        "total_stored_template_bits": int(n_units) * int(template_bits_per_unit) + int(shared_tap_index_bits),
        "actual_template_bits_read_note": "candidate template payload only; excludes one-time shared tap index configuration",
        "abs_diffs_per_candidate": int(feature_dim),
        "adds_per_candidate": max(int(feature_dim) - 1, 0),
        **traffic_accounting(candidate_reads, template_bits_per_unit, n_units),
    }
    return metrics


def _representation_specs(
    wave_train: np.ndarray,
    labels_train: np.ndarray,
    units: np.ndarray,
    wave_test: np.ndarray,
    bits: Sequence[int],
    tap_counts: Sequence[int],
) -> list[dict[str, Any]]:
    """Fit all transforms/templates from train rows and return test-ready specs."""
    specs: list[dict[str, Any]] = [{
        "name": "float_l1_64_teacher", "test": wave_test,
        "templates": fit_float_templates(wave_train, labels_train, units),
        "template_bits": WAVEFORM_SAMPLES * 32, "dim": WAVEFORM_SAMPLES, "metric": "l1",
    }]
    for n_bits in bits:
        maximum = (1 << (int(n_bits) - 1)) - 1
        train_codes = quantize_signed_uniform(wave_train, int(n_bits))
        specs.append({
            "name": f"signed_{int(n_bits)}bit_full64_sad",
            "test": quantize_signed_uniform(wave_test, int(n_bits)),
            "templates": fit_integer_templates(train_codes, labels_train, units, -maximum, maximum),
            "template_bits": WAVEFORM_SAMPLES * int(n_bits), "dim": WAVEFORM_SAMPLES, "metric": "sad",
        })
        for n_taps in tap_counts:
            taps = fisher_tap_indices(wave_train, labels_train, int(n_taps))
            specs.append({
                "name": f"fisher{int(n_taps)}_signed_{int(n_bits)}bit_sad",
                "test": quantize_signed_uniform(wave_test[:, taps], int(n_bits)),
                "templates": fit_integer_templates(train_codes[:, taps], labels_train, units, -maximum, maximum),
                "template_bits": int(n_taps) * int(n_bits), "dim": int(n_taps), "metric": "sad",
                "fisher_taps": taps,
                "shared_tap_index_bits": int(n_taps) * int(np.ceil(np.log2(WAVEFORM_SAMPLES))),
            })
        morph_train = quantize_unit_interval(morphology_features(wave_train), int(n_bits))
        specs.append({
            "name": f"morphology12_unsigned_{int(n_bits)}bit_sad",
            "test": quantize_unit_interval(morphology_features(wave_test), int(n_bits)),
            "templates": fit_integer_templates(morph_train, labels_train, units, 0, (1 << int(n_bits)) - 1),
            "template_bits": MORPHOLOGY_DIM * int(n_bits), "dim": MORPHOLOGY_DIM, "metric": "sad",
        })
    return specs


def run_dataset_refinement(
    dataset: Dataset,
    *,
    percentiles: Sequence[float] = (95.0, 99.9),
    bits: Sequence[int] = (4, 5),
    tap_counts: Sequence[int] = (8, 16, 24),
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run GT-event-isolated, chronological train/test compact refinement."""
    prepared = prepare_dataset_events(dataset, k_neighbors=7)
    train_idx, test_idx = _chronological_split(prepared["times"])
    train_labels, test_labels = prepared["labels"][train_idx], prepared["labels"][test_idx]
    sources = frozen_candidate_sources(
        prepared["descriptor"][train_idx], train_labels, prepared["descriptor"][test_idx], percentiles
    )
    # Every transform, Fisher ranking, quantizer, and template is fitted before
    # test refinement and only sees chronological training GT event rows.
    common_units = next(iter(sources.values()))["units"]
    specs = _representation_specs(
        prepared["waveforms"][train_idx], train_labels, common_units,
        prepared["waveforms"][test_idx], bits, tap_counts,
    )
    rows: list[dict[str, Any]] = []
    for source_name, source in sources.items():
        candidates = source["candidates"]  # frozen object shared by every spec below
        teacher = specs[0]
        teacher_pred, teacher_meta = refine_frozen_candidates(
            candidates, teacher["test"], teacher["templates"], source["spatial_codes"],
            source["spatial_centroids"], source["units"], metric=teacher["metric"],
        )
        for spec in specs:
            if spec is teacher:
                prediction, meta = teacher_pred, teacher_meta
            else:
                prediction, meta = refine_frozen_candidates(
                    candidates, spec["test"], spec["templates"], source["spatial_codes"],
                    source["spatial_centroids"], source["units"], metric=spec["metric"],
                )
            metrics = refinement_metrics(
                candidates, test_labels, prediction, teacher_pred,
                meta["candidate_template_reads"], spec["template_bits"], source["units"].size, spec["dim"],
                shared_tap_index_bits=int(spec.get("shared_tap_index_bits", 0)),
            )
            rows.append({
                "dataset": dataset.name, "n_events_gt_rows": int(prepared["labels"].size),
                "n_train_gt_rows": int(train_idx.size), "n_test_gt_rows": int(test_idx.size),
                "spatial_descriptor_dim": int(prepared["descriptor"].shape[1]), "spatial_k_neighbors": 7,
                "spatial_fixed_unit_interval": True, "candidate_source": source_name,
                "representation": spec["name"], "fisher_taps": "" if "fisher_taps" not in spec else ";".join(map(str, spec["fisher_taps"])),
                **metrics,
            })
    if verbose:
        print(f"{dataset.name}: GT rows train/test={train_idx.size}/{test_idx.size}; "
              f"sources={len(sources)}, representations={len(specs)}")
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-candidate compact waveform refinement")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 5], help="Signed/morphology waveform code widths")
    parser.add_argument("--taps", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[95.0, 99.9])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "compact_waveform_refinement")
    parser.add_argument("--mearec-npz-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(bit < 2 for bit in args.bits) or not args.bits:
        parser.error("--bits must contain signed widths >= 2")
    if any(tap < 1 or tap > WAVEFORM_SAMPLES for tap in args.taps) or not args.taps:
        parser.error("--taps must be in [1, 64]")
    if any(percentile < 0 or percentile > 100 for percentile in args.percentiles) or not args.percentiles:
        parser.error("--percentiles must be in [0, 100]")
    family_streams: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}:
        family_streams.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        if args.mearec_npz_dir is None:
            from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
            mearec_dir = DEFAULT_NPZ_DIR
        else:
            mearec_dir = args.mearec_npz_dir
        family_streams.append(("mearec", _iter_mearec_datasets(args.duration, None, mearec_dir)))
    start = time.time()
    for family, stream in family_streams:
        rows: list[dict[str, Any]] = []
        for dataset in stream:
            rows.extend(run_dataset_refinement(
                dataset, percentiles=args.percentiles, bits=args.bits, tap_counts=args.taps,
                verbose=not args.quiet,
            ))
            del dataset
        if rows:
            out = args.output_dir / f"compact_waveform_refinement_{family}_{int(args.duration)}s.csv"
            _write_csv(out, rows)
            print(f"Saved {len(rows)} rows: {out}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
