"""Cross-dataset 5-bit spatial candidate-search benchmark.

The benchmark keeps every GT event as a distinct row, including events with
identical sample timestamps.  Its first stage compares a legacy COM/P2P AND
gate with pooled and per-unit quantized range searches; its second stage reads
64-sample, normalized main-channel waveforms only for surviving candidates.

Examples
--------
python -m Spatial.experiments.run_adaptive_5bit_range_search --family all --duration 60
python -m Spatial.experiments.run_adaptive_5bit_range_search --family mearec --seeds 202600
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.algorithms.adaptive_range_search import (
    AdaptiveRangeSearch,
    evaluate_candidate_search,
)
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    com_features,
    extract_local_p2p,
    footprint_p2p_features,
)
from Spatial.data.loader import (
    Dataset,
    list_hybrid_janelia_scenes,
    load_hybrid_janelia,
    load_mearec_npz,
)
from Spatial.experiments.run_mearec_method_sweep import (
    DEFAULT_NPZ_DIR,
    discover_npz,
    parse_rec_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WAVEFORM_PRE = 15
WAVEFORM_POST = 49
WAVEFORM_LENGTH = WAVEFORM_PRE + WAVEFORM_POST


def labels_for_peak_output(
    spike_times: np.ndarray,
    spike_units: np.ndarray,
    valid_times: np.ndarray,
    n_samples: int,
    *,
    window: int,
) -> np.ndarray:
    """Align GT labels to :func:`get_peak_amplitudes` output without a time map.

    ``get_peak_amplitudes`` preserves the input event order after removing only
    its boundary-invalid rows.  Indexing its known input mask therefore keeps
    multiple events occurring at exactly the same timestamp as separate rows.
    The assertion is deliberately strong: a future change to that helper's
    ordering cannot silently corrupt labels in this benchmark.
    """
    times = np.asarray(spike_times, dtype=np.int64).ravel()
    labels = np.asarray(spike_units).ravel()
    returned = np.asarray(valid_times, dtype=np.int64).ravel()
    if times.shape != labels.shape:
        raise ValueError("spike_times and spike_units must have equal length")
    mask = (times >= int(window)) & (times < int(n_samples) - int(window))
    expected_times = times[mask]
    assert np.array_equal(returned, expected_times), (
        "get_peak_amplitudes output no longer matches its documented stable "
        "boundary-filtered input order"
    )
    aligned = labels[mask].copy()
    assert aligned.shape == returned.shape
    return aligned


def _stable_chronological_order(times: np.ndarray) -> np.ndarray:
    """Return a stable chronological order so equal timestamps retain row order."""
    return np.argsort(np.asarray(times), kind="stable")


def _mode_channel(values: np.ndarray) -> int:
    """Smallest channel wins a mode tie, matching ``np.unique`` ordering."""
    channels, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    if channels.size == 0:
        raise ValueError("cannot take a mode of an empty channel array")
    return int(channels[np.argmax(counts)])


def fit_home_channels(
    labels: Sequence[Any], central_channels: np.ndarray, units: Sequence[Any]
) -> np.ndarray:
    """Fit each unit's directory home channel as its training central-channel mode."""
    labs = np.asarray(labels)
    central = np.asarray(central_channels, dtype=np.int64).ravel()
    if labs.shape != central.shape:
        raise ValueError("labels and central_channels must have equal shape")
    return np.asarray(
        [_mode_channel(central[labs == unit]) for unit in np.asarray(units)], dtype=np.int64
    )


def fit_unit_channel_support(
    labels: Sequence[Any],
    central_channels: np.ndarray,
    units: Sequence[Any],
    n_channels: int,
    coverage: float,
) -> np.ndarray:
    """Fit the smallest per-unit central-channel support set meeting coverage.

    Rows follow ``units`` exactly, so labels need not be consecutive integers.
    Within a unit, channels are considered in descending training-event count;
    equal counts are resolved by ascending channel id.  The returned ``U x C``
    boolean matrix represents directory pointers, not duplicated CAM rows.
    """
    target_coverage = float(coverage)
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("directory coverage must be in (0, 1]")
    labs = np.asarray(labels)
    central = np.asarray(central_channels, dtype=np.int64).ravel()
    row_units = np.asarray(units)
    if labs.shape != central.shape:
        raise ValueError("labels and central_channels must have equal shape")
    if int(n_channels) < 1:
        raise ValueError("n_channels must be positive")
    if np.any((central < 0) | (central >= int(n_channels))):
        raise ValueError("central channel is outside the recording channel range")
    support = np.zeros((row_units.size, int(n_channels)), dtype=bool)
    for row, unit in enumerate(row_units):
        unit_channels = central[labs == unit]
        if unit_channels.size == 0:
            raise ValueError("every supplied unit must have at least one training event")
        channels, counts = np.unique(unit_channels, return_counts=True)
        # lexsort's last key is primary: -count descending, then channel ascending.
        order = np.lexsort((channels, -counts))
        required = int(np.ceil(target_coverage * unit_channels.size))
        selected = channels[order]
        cumulative = np.cumsum(counts[order])
        take = int(np.searchsorted(cumulative, required, side="left")) + 1
        support[row, selected[:take]] = True
    return support


def build_directory_mask(
    query_central_channels: np.ndarray,
    units: Sequence[Any],
    home_channels: np.ndarray,
    geom: np.ndarray,
    directory_neighbors: int,
) -> np.ndarray:
    """Build query-by-row directory masks, supporting arbitrary unit labels.

    ``directory_neighbors=0`` means no directory pruning.  Otherwise a row is
    active when its training home channel is in the query central channel's
    geometry KNN set (including that query channel itself).
    """
    query = np.asarray(query_central_channels, dtype=np.int64).ravel()
    row_units = np.asarray(units)
    homes = np.asarray(home_channels, dtype=np.int64).ravel()
    geometry = np.asarray(geom, dtype=np.float64)
    if homes.shape != row_units.shape:
        raise ValueError("home_channels must align with units")
    n_channels = geometry.shape[0]
    if n_channels < 1:
        raise ValueError("geometry must include at least one channel")
    if np.any((query < 0) | (query >= n_channels)):
        raise ValueError("query central channel is outside geometry")
    if np.any((homes < 0) | (homes >= n_channels)):
        raise ValueError("unit home channel is outside geometry")
    if int(directory_neighbors) <= 0:
        return np.ones((query.size, row_units.size), dtype=bool)
    table = build_knn_table_with_self(geometry, min(int(directory_neighbors), n_channels))
    return np.asarray([np.isin(homes, table[channel]) for channel in query], dtype=bool)


def build_support_directory_mask(
    query_central_channels: np.ndarray,
    support: np.ndarray,
    geom: np.ndarray,
    directory_neighbors: int,
) -> np.ndarray:
    """Activate a row once when its channel support intersects query-channel KNN.

    A support row can contribute several channel/tile directory pointers, but
    the result remains one boolean activation per CAM row and query.
    """
    query = np.asarray(query_central_channels, dtype=np.int64).ravel()
    channel_support = np.asarray(support, dtype=bool)
    geometry = np.asarray(geom, dtype=np.float64)
    if channel_support.ndim != 2:
        raise ValueError("support must have shape (n_units, n_channels)")
    n_channels = geometry.shape[0]
    if channel_support.shape[1] != n_channels:
        raise ValueError("support channel dimension must match geometry")
    if np.any((query < 0) | (query >= n_channels)):
        raise ValueError("query central channel is outside geometry")
    if int(directory_neighbors) <= 0:
        return np.ones((query.size, channel_support.shape[0]), dtype=bool)
    table = build_knn_table_with_self(geometry, min(int(directory_neighbors), n_channels))
    return np.asarray(
        [channel_support[:, table[channel]].any(axis=1) for channel in query], dtype=bool
    )


def directory_entry_statistics(entries_per_unit: np.ndarray) -> dict[str, float | int]:
    """Return transparent pointer counts; all-row bypass is represented by zeros."""
    entries = np.asarray(entries_per_unit, dtype=np.int64).ravel()
    if np.any(entries < 0):
        raise ValueError("directory entry counts must be non-negative")
    return {
        "mean_directory_entries_per_unit": float(entries.mean()) if entries.size else 0.0,
        "max_directory_entries_per_unit": int(entries.max()) if entries.size else 0,
        "total_directory_entries": int(entries.sum()),
    }


def _fit_centroids(features: np.ndarray, labels: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Fit centroid rows in exactly the supplied (possibly non-contiguous) unit order."""
    values = np.asarray(features, dtype=np.float64)
    labs = np.asarray(labels)
    return np.stack([values[labs == unit].mean(axis=0) for unit in units], axis=0)


def _pooled_radius(codes: np.ndarray, labels: np.ndarray, centroids: np.ndarray, units: np.ndarray, percentile: float) -> int:
    index = {unit.item() if isinstance(unit, np.generic) else unit: i for i, unit in enumerate(units)}
    distances = np.asarray(
        [np.abs(code.astype(np.int64) - centroids[index[label.item() if isinstance(label, np.generic) else label]]).sum()
         for code, label in zip(codes, labels)],
        dtype=np.int64,
    )
    return int(np.ceil(np.percentile(distances, float(percentile)))) if distances.size else 0


def pooled_l1_candidates(
    query_codes: np.ndarray,
    centroids: np.ndarray,
    units: Sequence[Any],
    radius: int,
    active_mask: np.ndarray,
    *,
    inclusive: bool = True,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Pooled L1 code-ball candidates with inclusive boundary support by default."""
    codes = np.asarray(query_codes, dtype=np.int64)
    rows = np.asarray(centroids, dtype=np.int64)
    active = np.asarray(active_mask, dtype=bool)
    labels = np.asarray(units)
    if codes.ndim != 2 or rows.ndim != 2 or codes.shape[1] != rows.shape[1]:
        raise ValueError("query_codes and centroids must be compatible 2D arrays")
    if active.shape != (codes.shape[0], rows.shape[0]):
        raise ValueError("active_mask must have shape (n_queries, n_units)")
    distances = np.abs(codes[:, None, :] - rows[None, :, :]).sum(axis=2)
    matched = distances <= int(radius) if inclusive else distances < int(radius)
    matched &= active
    candidates = [labels[row].copy() for row in matched]
    counts = matched.sum(axis=1).astype(np.int64)
    return candidates, {
        "initial_candidate_count": counts.copy(),
        "final_candidate_count": counts,
        "widen_level": np.zeros(codes.shape[0], dtype=np.int64),
        "fallback": np.zeros(codes.shape[0], dtype=bool),
        "active_row_count": active.sum(axis=1).astype(np.int64),
        "row_comparisons": active.sum(axis=1).astype(np.int64),
    }


def legacy_pooled_and_candidates(
    query_com: np.ndarray,
    query_p2p_codes: np.ndarray,
    com_centroids: np.ndarray,
    p2p_centroids: np.ndarray,
    units: Sequence[Any],
    tau_com: float,
    tau_p2p: int,
    active_mask: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Existing gate semantics: strict float-COM L2 AND strict code-P2P L1."""
    active = np.asarray(active_mask, dtype=bool)
    com_distance = np.linalg.norm(
        np.asarray(query_com)[:, None, :] - np.asarray(com_centroids)[None, :, :], axis=2
    )
    p2p_distance = np.abs(
        np.asarray(query_p2p_codes, dtype=np.int64)[:, None, :]
        - np.asarray(p2p_centroids, dtype=np.int64)[None, :, :]
    ).sum(axis=2)
    matched = active & (com_distance < float(tau_com)) & (p2p_distance < int(tau_p2p))
    labels = np.asarray(units)
    candidates = [labels[row].copy() for row in matched]
    counts = matched.sum(axis=1).astype(np.int64)
    return candidates, {
        "initial_candidate_count": counts.copy(),
        "final_candidate_count": counts,
        "widen_level": np.zeros(len(candidates), dtype=np.int64),
        "fallback": np.zeros(len(candidates), dtype=bool),
        "active_row_count": active.sum(axis=1).astype(np.int64),
        # A row has two gate predicates, but this is a directory row access count.
        "row_comparisons": active.sum(axis=1).astype(np.int64),
    }


def candidate_waveform_assign(
    candidates: Sequence[np.ndarray | Sequence[Any]],
    waveforms: np.ndarray,
    waveform_centroids: dict[Any, np.ndarray],
    spatial_codes: np.ndarray,
    spatial_centroids: np.ndarray,
    spatial_units: Sequence[Any],
    *,
    metric: str = "l1",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Assign candidate-only waveform centroids; empty candidates use spatial fallback.

    No waveform top-k or global waveform scan is added on ordinary non-empty
    candidate lists.  Empty lists read zero waveforms and choose the nearest
    unified 5-bit spatial centroid across all fitted units.
    """
    wave = np.asarray(waveforms, dtype=np.float64)
    codes = np.asarray(spatial_codes, dtype=np.int64)
    centers = np.asarray(spatial_centroids, dtype=np.int64)
    units = np.asarray(spatial_units)
    if metric not in {"l1", "l2"}:
        raise ValueError("metric must be 'l1' or 'l2'")
    if len(candidates) != wave.shape[0] or codes.shape[0] != wave.shape[0]:
        raise ValueError("candidates, waveforms, and spatial_codes must share row count")
    unit_index = {unit.item() if isinstance(unit, np.generic) else unit: i for i, unit in enumerate(units)}
    pred = np.empty(wave.shape[0], dtype=units.dtype)
    reads = np.zeros(wave.shape[0], dtype=np.int64)
    used_fallback = np.zeros(wave.shape[0], dtype=bool)
    # This is a separate L1 centroid scan after an empty/invalid candidate list;
    # it is not a waveform-template read and must not disappear from Level-1
    # access accounting.
    spatial_fallback_row_comparisons = np.zeros(wave.shape[0], dtype=np.int64)
    for i, candidate_rows in enumerate(candidates):
        candidate_list = list(candidate_rows)
        # The intended two-level architecture has a |C|=1 early exit: that
        # unit is already the decision, so no level-2 waveform memory read is
        # necessary.  (This is deliberately not a waveform tie-break.)
        if len(candidate_list) == 1 and candidate_list[0] in unit_index:
            pred[i] = candidate_list[0]
            continue
        if candidate_list:
            valid = [u for u in candidate_list if u in waveform_centroids]
            if valid:
                centroid_matrix = np.stack([waveform_centroids[u] for u in valid], axis=0)
                if metric == "l1":
                    distance = np.abs(centroid_matrix - wave[i]).sum(axis=1)
                else:
                    distance = np.linalg.norm(centroid_matrix - wave[i], axis=1)
                pred[i] = valid[int(np.argmin(distance))]
                reads[i] = len(valid)
                continue
        # Candidate list is empty (or unexpectedly lacks waveform centroids):
        # only the stipulated unified-spatial nearest-centroid fallback remains.
        distance = np.abs(centers - codes[i]).sum(axis=1)
        pred[i] = units[int(np.argmin(distance))]
        used_fallback[i] = True
        spatial_fallback_row_comparisons[i] = int(units.size)
    return pred, {
        "candidate_waveform_reads": reads,
        "waveform_fallback": used_fallback,
        "spatial_fallback_row_comparisons": spatial_fallback_row_comparisons,
    }


def _candidate_summary(
    candidates: Sequence[np.ndarray],
    labels: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    directory_mask: np.ndarray,
    units: np.ndarray,
) -> dict[str, float | int]:
    """Combine core evaluator output with directory and p95 observability fields."""
    report = dict(evaluate_candidate_search(candidates, labels, diagnostics))
    active = np.asarray(directory_mask, dtype=bool)
    unit_to_row = {unit.item() if isinstance(unit, np.generic) else unit: i for i, unit in enumerate(units)}
    true_rows = np.asarray([unit_to_row.get(label.item() if isinstance(label, np.generic) else label, -1) for label in labels])
    directory_true = (true_rows >= 0) & active[np.arange(labels.size), np.maximum(true_rows, 0)]
    rows = np.asarray(diagnostics["row_comparisons"], dtype=np.float64)
    active_rows = np.asarray(diagnostics["active_row_count"], dtype=np.float64)
    report.update({
        "directory_true_unit_recall": float(directory_true.mean()) if labels.size else 0.0,
        "p95_active_rows": float(np.percentile(active_rows, 95)) if labels.size else 0.0,
        "max_active_rows": int(active_rows.max()) if labels.size else 0,
        "row_activation_reduction": float(1.0 - active_rows.mean() / max(len(units), 1)) if labels.size else 0.0,
        "p95_row_comparisons": float(np.percentile(rows, 95)) if labels.size else 0.0,
        "max_row_comparisons": int(rows.max()) if labels.size else 0,
    })
    return report


def summarize_configuration(
    candidates: Sequence[np.ndarray],
    labels: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    directory_mask: np.ndarray,
    units: np.ndarray,
    pred: np.ndarray,
    wave_meta: dict[str, np.ndarray],
    directory_entries_per_unit: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return the stable CSV summary fields for one candidate-search configuration."""
    report = _candidate_summary(candidates, labels, diagnostics, directory_mask, units)
    counts = np.asarray([len(c) for c in candidates], dtype=np.int64)
    correct = np.asarray(pred == labels, dtype=bool)
    unique = counts == 1
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
    report.update({
        "final_waveform_accuracy": float(correct.mean()) if n else 0.0,
        "unique_hit_fraction": float(unique.mean()) if n else 0.0,
        "unique_hit_accuracy": float(correct[unique].mean()) if np.any(unique) else 0.0,
        "candidate_waveform_reads_per_event": float(reads.mean()) if n else 0.0,
        "candidate_waveform_reads_total": int(reads.sum()),
        "waveform_fallback_rate": float(np.mean(wave_meta["waveform_fallback"])) if n else 0.0,
        "global_all_unit_waveform_reads": int(n * len(units)),
        "candidate_waveform_read_fraction_vs_global": float(reads.sum() / max(n * len(units), 1)),
        "waveform_read_reduction_vs_global": float(1.0 - reads.sum() / max(n * len(units), 1)),
        "waveform_read_reduction_factor_global_over_actual": (
            float(n * len(units) / reads.sum()) if reads.sum() else float("inf")
        ),
        "fallback_row_comparisons": int(range_fallback_rows.sum() + spatial_fallback_rows.sum()),
        "range_fallback_row_comparisons": int(range_fallback_rows.sum()),
        "spatial_fallback_row_comparisons": int(spatial_fallback_rows.sum()),
        "total_row_comparisons_including_fallback": int(total_rows.sum()),
    })
    if directory_entries_per_unit is not None:
        report.update(directory_entry_statistics(directory_entries_per_unit))
    return report


def _extract_main_channel_waveforms(
    signal: np.ndarray, times: np.ndarray, central_channels: np.ndarray
) -> np.ndarray:
    """Extract exact 64-sample [t-15, t+49) main-channel waveforms."""
    filtered = np.asarray(signal, dtype=np.float64)
    out = np.empty((len(times), WAVEFORM_LENGTH), dtype=np.float64)
    for i, (time_index, channel) in enumerate(zip(times, central_channels)):
        out[i] = filtered[int(channel), int(time_index) - WAVEFORM_PRE : int(time_index) + WAVEFORM_POST]
    return out


def _normalize_waveforms(waveforms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scale = np.max(np.abs(waveforms), axis=1, keepdims=True)
    return waveforms / np.maximum(scale, eps)


def prepare_dataset_events(dataset: Dataset, *, k_neighbors: int = 7, window: int = 15) -> dict[str, np.ndarray]:
    """Prepare stable, GT-aligned descriptor and waveform rows for one recording."""
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, peak_times, central = get_peak_amplitudes(filtered, dataset.spike_times, window=window)
    labels = labels_for_peak_output(
        dataset.spike_times, dataset.spike_units, peak_times, dataset.n_samples, window=window
    )
    # P2P admits the same window, while 64-sample waveform reads need an extra
    # 34 samples after the event.  Filter all per-event arrays together.
    waveform_ok = peak_times + WAVEFORM_POST <= dataset.n_samples
    peak_times, central, labels = peak_times[waveform_ok], central[waveform_ok], labels[waveform_ok]
    k = min(int(k_neighbors), dataset.n_channels)
    neighbor_table = build_knn_table_with_self(dataset.geom, k)
    p2p_raw, p2p_times, neighbor_ids = extract_local_p2p(
        filtered, peak_times, central, neighbor_table, window=window
    )
    assert np.array_equal(p2p_times, peak_times), "local P2P must preserve aligned event rows"
    assert p2p_raw.shape[0] == labels.size == central.size
    com = com_features(p2p_raw, neighbor_ids, dataset.geom)
    p2p = footprint_p2p_features(p2p_raw, normalize=True)
    descriptor = np.column_stack([com, p2p])
    if descriptor.shape[1] != k + 2:
        raise AssertionError("descriptor must be COM(2) + K-neighbor P2P")
    assert np.all(np.isfinite(descriptor)) and np.all((descriptor >= 0.0) & (descriptor <= 1.0))
    waveforms = _normalize_waveforms(_extract_main_channel_waveforms(filtered, peak_times, central))
    order = _stable_chronological_order(peak_times)
    return {
        "times": peak_times[order], "labels": labels[order], "central": central[order],
        "descriptor": descriptor[order], "com": com[order], "p2p": p2p[order],
        "waveforms": waveforms[order],
    }


def _chronological_split(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose the closest 50/50 time boundary without splitting equal timestamps."""
    event_times = np.asarray(times, dtype=np.int64).ravel()
    n_events = event_times.size
    if n_events < 2:
        raise ValueError("at least two valid GT events are required")
    if np.any(event_times[1:] < event_times[:-1]):
        raise ValueError("times must already be stably sorted")
    boundaries = np.flatnonzero(event_times[1:] != event_times[:-1]) + 1
    if boundaries.size == 0:
        raise ValueError("cannot make chronological split without dividing equal timestamps")
    target = n_events / 2.0
    # ``argmin`` deterministically chooses the earlier boundary on exact ties.
    n_train = int(boundaries[np.argmin(np.abs(boundaries - target))])
    return np.arange(n_train), np.arange(n_train, n_events)


def run_dataset_benchmark(
    dataset: Dataset,
    *,
    bits: int = 5,
    percentile: float = 95.0,
    coverage: float = 0.95,
    widening_steps: Sequence[int] = (0, 1, 2, 4),
    directory_neighbors: Sequence[int] = (0, 7),
    directory_coverages: Sequence[float] = (0.95, 0.99, 1.0),
    k_neighbors: int = 7,
    waveform_metric: str = "l1",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run every requested first-stage configuration on a chronological 50/50 split."""
    prepared = prepare_dataset_events(dataset, k_neighbors=k_neighbors)
    train_idx, test_idx = _chronological_split(prepared["times"])
    train_labels, test_labels = prepared["labels"][train_idx], prepared["labels"][test_idx]
    model = AdaptiveRangeSearch(
        n_bits=int(bits), fixed_unit_interval=True,
        radius_percentile=float(percentile), box_coverage=float(coverage),
    ).fit(prepared["descriptor"][train_idx], train_labels)
    units = model.units_
    query_codes = model.transform(prepared["descriptor"][test_idx])
    train_codes = model.transform(prepared["descriptor"][train_idx])
    home_channels = fit_home_channels(train_labels, prepared["central"][train_idx], units)
    validated_directory_coverages: list[float] = []
    for requested_coverage in directory_coverages:
        directory_coverage = float(requested_coverage)
        if not 0.0 < directory_coverage <= 1.0:
            raise ValueError("directory_coverages must contain values in (0, 1]")
        if directory_coverage not in validated_directory_coverages:
            validated_directory_coverages.append(directory_coverage)
    waveform_centroids = {
        unit.item() if isinstance(unit, np.generic) else unit:
        prepared["waveforms"][train_idx][train_labels == unit].mean(axis=0)
        for unit in units
    }
    # Legacy gate: float COM L2 plus the P2P components of the unified 5-bit codes.
    com_centroids = _fit_centroids(prepared["com"][train_idx], train_labels, units)
    p2p_centroids = _fit_centroids(train_codes[:, 2:], train_labels, units).round().astype(np.int64)
    legacy_com_distances = np.asarray([
        np.linalg.norm(prepared["com"][train_idx][i] - com_centroids[np.flatnonzero(units == train_labels[i])[0]])
        for i in range(train_idx.size)
    ])
    tau_com = max(float(np.percentile(legacy_com_distances, percentile)), 1e-8)
    tau_p2p = max(_pooled_radius(train_codes[:, 2:], train_labels, p2p_centroids, units, percentile), 1)
    pooled_centroids = model.centroids_
    pooled_radius = _pooled_radius(train_codes, train_labels, pooled_centroids, units, percentile)
    rows: list[dict[str, Any]] = []
    if verbose:
        print(f"{dataset.name}: n={len(prepared['labels'])}, train/test={len(train_idx)}/{len(test_idx)}, "
              f"units={len(units)}, descriptor_dim={prepared['descriptor'].shape[1]}")

    def add_row(
        name: str,
        candidates: list[np.ndarray],
        diag: dict[str, np.ndarray],
        directory: np.ndarray,
        row_bits: int,
        note: str,
        *,
        steps: int,
        ensure: bool,
        directory_strategy: str,
        directory_coverage: float | None,
        directory_entries_per_unit: np.ndarray,
        directory_neighbor_count: int,
    ) -> None:
        pred, wave_meta = candidate_waveform_assign(
            candidates, prepared["waveforms"][test_idx], waveform_centroids, query_codes,
            model.centroids_, units, metric=waveform_metric,
        )
        metrics = summarize_configuration(
            candidates, test_labels, diag, directory, units, pred, wave_meta,
            directory_entries_per_unit=directory_entries_per_unit,
        )
        rows.append({
            "dataset": dataset.name, "n_channels": int(dataset.n_channels), "n_events": int(len(prepared["labels"])),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "n_units_train": int(len(units)),
            "bits": int(bits), "descriptor_dim": int(prepared["descriptor"].shape[1]), "k_neighbors": int(min(k_neighbors, dataset.n_channels)),
            "method": name, "widening_steps": int(steps), "ensure_nonempty": bool(ensure),
            "directory_neighbors": int(directory_neighbor_count),
            "directory_strategy": directory_strategy,
            "directory_coverage": "" if directory_coverage is None else float(directory_coverage),
            "waveform_metric": waveform_metric,
            "percentile": float(percentile), "box_coverage": float(coverage),
            "search_payload_row_bits": int(row_bits), "search_payload_row_bits_note": note,
            "legacy_tau_com": float(tau_com) if name == "legacy_pooled_and" else "",
            "legacy_tau_p2p_code_l1": int(tau_p2p) if name == "legacy_pooled_and" else "",
            "pooled_l1_radius": int(pooled_radius) if name == "pooled_l1" else "",
            "shared_pooled_l1_threshold_bits": (
                int(np.ceil(np.log2(pooled_radius + 1))) if name == "pooled_l1" and pooled_radius else 0
            ) if name == "pooled_l1" else "",
            **metrics,
        })

    clipped_directory_neighbors: list[int] = []
    for requested_directory in directory_neighbors:
        clipped = max(0, min(int(requested_directory), dataset.n_channels))
        if clipped not in clipped_directory_neighbors:
            clipped_directory_neighbors.append(clipped)
    if not clipped_directory_neighbors:
        raise ValueError("directory_neighbors must contain at least one value")

    for directory_neighbors_current in clipped_directory_neighbors:
        # With no pruning, every coverage would be the same all-row query.
        # Emit it exactly once and keep its entry counts at zero: the directory
        # is bypassed rather than materialising C pointers per unit.
        directory_configs: list[tuple[str, float | None, np.ndarray, np.ndarray]] = []
        if directory_neighbors_current == 0:
            directory_configs.append((
                "all", None,
                np.ones((len(test_idx), len(units)), dtype=bool),
                np.zeros(len(units), dtype=np.int64),
            ))
        else:
            directory_configs.append((
                "mode", None,
                build_directory_mask(
                    prepared["central"][test_idx], units, home_channels, dataset.geom,
                    directory_neighbors_current,
                ),
                np.ones(len(units), dtype=np.int64),
            ))
            for directory_coverage in validated_directory_coverages:
                support = fit_unit_channel_support(
                    train_labels, prepared["central"][train_idx], units,
                    dataset.n_channels, directory_coverage,
                )
                directory_configs.append((
                    "support", directory_coverage,
                    build_support_directory_mask(
                        prepared["central"][test_idx], support, dataset.geom,
                        directory_neighbors_current,
                    ),
                    support.sum(axis=1, dtype=np.int64),
                ))

        for directory_strategy, directory_coverage, directory, entries_per_unit in directory_configs:
            legacy_candidates, legacy_diag = legacy_pooled_and_candidates(
                prepared["com"][test_idx], query_codes[:, 2:], com_centroids, p2p_centroids, units,
                tau_com, tau_p2p, directory,
            )
            add_row(
                "legacy_pooled_and", legacy_candidates, legacy_diag, directory,
                2 * 32 + (prepared["descriptor"].shape[1] - 2) * int(bits),
                "2x float32 COM centroid + Kx unsigned P2P code centroid; shared thresholds excluded",
                steps=0, ensure=False, directory_strategy=directory_strategy,
                directory_coverage=directory_coverage, directory_entries_per_unit=entries_per_unit,
                directory_neighbor_count=directory_neighbors_current,
            )
            pooled_candidates, pooled_diag = pooled_l1_candidates(
                query_codes, pooled_centroids, units, pooled_radius, directory, inclusive=True
            )
            add_row(
                "pooled_l1", pooled_candidates, pooled_diag, directory,
                model.hardware_costs()["feature_bits"],
                "unified D*B centroid only; pooled L1 threshold is one shared calibration field",
                steps=0, ensure=False, directory_strategy=directory_strategy,
                directory_coverage=directory_coverage, directory_entries_per_unit=entries_per_unit,
                directory_neighbor_count=directory_neighbors_current,
            )
            for mode, name, row_bits in (
                ("l1", "adaptive_l1", model.hardware_costs()["dac_l1_row_bits"]),
                ("box", "adaptive_box", model.hardware_costs()["box_row_bits"]),
            ):
                for steps in widening_steps:
                    for ensure in (False, True):
                        candidates, diag = model.query_codes(
                            query_codes, mode=mode, active_mask=directory,
                            widening_steps=int(steps), ensure_nonempty=ensure,
                        )
                        add_row(
                            name, candidates, diag, directory, int(row_bits),
                            "AdaptiveRangeSearch fixed DAC/CAM hardware_costs payload only (no unit id/directory/periphery)",
                            steps=int(steps), ensure=ensure, directory_strategy=directory_strategy,
                            directory_coverage=directory_coverage, directory_entries_per_unit=entries_per_unit,
                            directory_neighbor_count=directory_neighbors_current,
                        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_family_summary(rows: list[dict[str, Any]]) -> None:
    """Print family-level means, retaining widening/fallback as distinct configurations."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            row["method"], row["directory_neighbors"], row["directory_strategy"],
            row["directory_coverage"], row["widening_steps"], row["ensure_nonempty"],
        )].append(row)
    print("\n=== family aggregate (mean across datasets) ===")
    print(f"{'method':<18} {'dir':>4} {'strategy':<7} {'cover':>5} {'wide':>5} {'fallback':>8} {'dir-rec':>8} {'recall':>8} {'cand':>7} {'final':>7} {'row-red':>8}")
    for key, bucket in groups.items():
        method, directory, strategy, directory_coverage, steps, ensure = key
        mean = lambda name: float(np.mean([float(r[name]) for r in bucket]))
        cover = "—" if directory_coverage == "" else f"{float(directory_coverage):.2f}"
        print(f"{method:<18} {directory:>4} {strategy:<7} {cover:>5} {steps:>5} {str(ensure):>8} "
              f"{mean('directory_true_unit_recall'):>8.3f} {mean('true_unit_candidate_recall'):>8.3f} {mean('mean_candidates'):>7.2f} "
              f"{mean('final_waveform_accuracy'):>7.3f} {mean('row_activation_reduction'):>8.3f}")


def _iter_hj_datasets(duration: float):
    """Yield HJ recordings one at a time, never retaining all raw arrays."""
    scenes = list_hybrid_janelia_scenes(ready_only=True)
    if not scenes:
        yield load_hybrid_janelia(duration_s=duration)
        return
    for scene in scenes:
        key = scene.get("short_name") or scene.get("scene_key")
        if key:
            yield load_hybrid_janelia(scene=key, duration_s=duration)


def _iter_mearec_datasets(duration: float, seeds: list[int] | None, npz_dir: Path):
    """Yield MEArec recordings one at a time, keeping only NPZ paths in memory."""
    for path in discover_npz(npz_dir, seeds=seeds):
        yield load_mearec_npz(path, duration_s=duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="HJ/MEArec 5-bit adaptive range-search benchmark")
    parser.add_argument("--family", choices=("hj", "mearec", "all"), default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=95.0, help="L1 radius / legacy pooled percentile")
    parser.add_argument("--coverage", type=float, default=0.95, help="Adaptive box joint training coverage")
    parser.add_argument("--widening-steps", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--directory-neighbors", nargs="+", type=int, default=[0, 7])
    parser.add_argument(
        "--directory-coverages", nargs="+", type=float, default=[0.95, 0.99, 1.0],
        help="Per-unit central-channel support coverage for support directory rows.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional MEArec seeds")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--waveform-metric", choices=("l1", "l2"), default="l1")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "adaptive_5bit_range_search")
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.bits <= 62:
        parser.error("--bits must be in [1, 62]")
    if not args.widening_steps or any(step < 0 for step in args.widening_steps):
        parser.error("--widening-steps must contain non-negative integers")
    if not args.directory_coverages or any(not 0.0 < value <= 1.0 for value in args.directory_coverages):
        parser.error("--directory-coverages must contain values in (0, 1]")
    t0 = time.time()
    families: list[tuple[str, Any]] = []
    if args.family in {"hj", "all"}:
        families.append(("hj", _iter_hj_datasets(args.duration)))
    if args.family in {"mearec", "all"}:
        families.append(("mearec", _iter_mearec_datasets(args.duration, args.seeds, args.mearec_npz_dir)))
    for family, datasets in families:
        family_rows: list[dict[str, Any]] = []
        n_datasets = 0
        for dataset in datasets:
            n_datasets += 1
            if not args.quiet:
                print(f"\n########## {family}: {dataset.name} ##########")
            family_rows.extend(run_dataset_benchmark(
                dataset, bits=args.bits, percentile=args.percentile, coverage=args.coverage,
                widening_steps=args.widening_steps, directory_neighbors=args.directory_neighbors,
                directory_coverages=args.directory_coverages,
                waveform_metric=args.waveform_metric, verbose=not args.quiet,
            ))
            # The generator will load the next raw recording only after this
            # reference is released; family_rows contain just scalar summaries.
            del dataset
        if not n_datasets:
            print(f"{family}: no datasets found; skipping")
            continue
        pct_tag = str(args.percentile).replace(".", "p")
        coverage_tag = str(args.coverage).replace(".", "p")
        out = args.output_dir / (
            f"adaptive_{args.bits}bit_range_search_{family}_{int(args.duration)}s_"
            f"p{pct_tag}_c{coverage_tag}.csv"
        )
        _write_csv(out, family_rows)
        print(f"Saved {len(family_rows)} rows: {out}")
        print_family_summary(family_rows)
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
