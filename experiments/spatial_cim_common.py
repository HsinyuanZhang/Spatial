"""Shared helpers for the spatial CiM/CAM sorting evidence runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from Spatial.algorithms.spatial_footprint import (
    build_knn_table_with_self,
    build_relative_patch_table,
    central_ratio_features,
    com_features,
    extract_local_p2p,
    footprint_p2p_features,
    soft_localization_features,
)
from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.experiments.run_adaptive_5bit_range_search import labels_for_peak_output
from Spatial.experiments.run_causal_temporal_sketch import (
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    FILTER_ORDER,
    POST_ALIGNMENT_SAMPLES,
    PRE_ALIGNMENT_SAMPLES,
    WINDOW,
)


FEATURE_SETS = (
    "com_only",
    "p2p_only",
    "com_and_p2p",
    "soft_loc",
    "soft_loc_p2p",
    "central_ratio",
)

FootprintLayout = Literal["knn", "relative"]


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Plain GT-aligned accuracy (no Hungarian). Predictions share GT unit IDs."""
    pred = np.asarray(predictions, dtype=object)
    truth = np.asarray(labels)
    correct = np.asarray([p == y for p, y in zip(pred, truth)], dtype=bool)
    per_unit: dict[str, dict[str, float | int]] = {}
    for unit in np.unique(truth):
        mask = truth == unit
        per_unit[str(_scalar(unit))] = {
            "events": int(mask.sum()),
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
        }
    return {
        "n_events": int(truth.size),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if truth.size else 0.0,
        "per_unit": per_unit,
        "worst_unit_accuracy": (
            min(v["accuracy"] for v in per_unit.values()) if per_unit else 0.0
        ),
    }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def prepare_spatial_events(
    dataset,
    *,
    k_neighbors: int = 7,
    footprint_layout: FootprintLayout = "knn",
    permute_geom: bool | np.ndarray | None = None,
    permute_slots: bool | np.ndarray | None = None,
    seed: int = 0,
    p2p_estimator: str = "raw",
    p2p_average_points: int = 1,
) -> dict[str, Any]:
    """Prepare multi-feature spatial rows plus central and neighborhood waveforms.

    Attribution controls:
    - ``permute_geom``: shuffle electrode geometry so COM is meaningless.
    - ``permute_slots``: apply a fixed permutation to P2P neighborhood slots so
      the footprint layout is meaningless while the value multiset is identical.

    ``footprint_layout``:
    - ``knn``: distance-ranked neighbors (slot 0 = home); enables ``central_ratio``.
    - ``relative``: fixed primary-axis offsets (center slot = home); OOB = -1 / P2P 0.
      ``central_ratio`` is omitted (it assumes KNN slot 0).
    """
    layout = str(footprint_layout)
    if layout not in ("knn", "relative"):
        raise ValueError(
            f"footprint_layout must be 'knn' or 'relative', got {footprint_layout!r}"
        )

    filtered = bandpass_filter(
        dataset.raw_data, dataset.fs, FILTER_LOW_HZ, FILTER_HIGH_HZ, FILTER_ORDER
    )
    _, times, central = get_peak_amplitudes(
        filtered, dataset.spike_times, window=PRE_ALIGNMENT_SAMPLES
    )
    labels = labels_for_peak_output(
        dataset.spike_times, dataset.spike_units, times, dataset.n_samples,
        window=PRE_ALIGNMENT_SAMPLES,
    )
    original = np.flatnonzero(
        (dataset.spike_times >= PRE_ALIGNMENT_SAMPLES)
        & (dataset.spike_times < dataset.n_samples - PRE_ALIGNMENT_SAMPLES)
    )
    joint = times + POST_ALIGNMENT_SAMPLES < dataset.n_samples
    times, central, labels, original = (
        times[joint], central[joint], labels[joint], original[joint]
    )
    if times.size == 0:
        raise ValueError("no events remain after boundary filtering")

    geom = np.asarray(dataset.geom, dtype=np.float64).copy()
    geom_perm = None
    if permute_geom is not None and permute_geom is not False:
        rng = np.random.default_rng(seed)
        if permute_geom is True:
            geom_perm = rng.permutation(geom.shape[0])
        else:
            geom_perm = np.asarray(permute_geom, dtype=np.int64)
        geom = geom[geom_perm]

    k = min(int(k_neighbors), dataset.n_channels)
    if layout == "knn":
        table = build_knn_table_with_self(geom, k)
    else:
        half_width = max(k // 2, 0)
        table = build_relative_patch_table(geom, half_width=half_width)
        k = int(table.shape[1])

    p2p, returned, neighbors = extract_local_p2p(
        filtered,
        times,
        central,
        table,
        window=PRE_ALIGNMENT_SAMPLES,
        p2p_estimator=p2p_estimator,
        average_points=p2p_average_points,
    )
    assert np.array_equal(returned, times)

    slot_perm = None
    if permute_slots is not None and permute_slots is not False:
        if layout == "relative":
            raise ValueError("permute_slots is only supported with footprint_layout='knn'")
        rng = np.random.default_rng(seed + 1)
        if permute_slots is True:
            # Keep slot 0 (self/central) fixed; permute the rest.
            rest = rng.permutation(np.arange(1, p2p.shape[1]))
            slot_perm = np.concatenate([[0], rest])
        else:
            slot_perm = np.asarray(permute_slots, dtype=np.int64)
        p2p = p2p[:, slot_perm]
        neighbors = neighbors[:, slot_perm]

    com = com_features(p2p, neighbors, geom)
    soft = soft_localization_features(p2p, neighbors, geom)
    p2p_norm = footprint_p2p_features(p2p, normalize=True)
    features = {
        "com_only": com,
        "p2p_only": p2p_norm,
        "com_and_p2p": np.column_stack([com, p2p_norm]),
        "soft_loc": soft,
        "soft_loc_p2p": np.column_stack([soft, p2p_norm]),
    }
    if layout == "knn":
        features["central_ratio"] = central_ratio_features(p2p)

    # Central-channel 64-sample waveforms (existing temporal teacher input).
    raw_central = np.stack(
        [
            filtered[int(ch), int(t) - PRE_ALIGNMENT_SAMPLES : int(t) + POST_ALIGNMENT_SAMPLES + 1]
            for t, ch in zip(times, central)
        ]
    )
    # Multi-channel neighborhood waveforms: K channels x 64 samples, flattened.
    # OOB neighbor ids (-1) contribute zeros.
    neigh_wave = np.zeros((times.size, k * WINDOW), dtype=np.float64)
    for i, (t, nbrs) in enumerate(zip(times, neighbors)):
        start = int(t) - PRE_ALIGNMENT_SAMPLES
        stop = int(t) + POST_ALIGNMENT_SAMPLES + 1
        for j, ch_j in enumerate(nbrs):
            if int(ch_j) < 0:
                continue
            neigh_wave[i, j * WINDOW : (j + 1) * WINDOW] = filtered[int(ch_j), start:stop]

    order = np.argsort(times, kind="stable")
    result = {
        "event_row_id": original[order],
        "times": times[order],
        "labels": labels[order],
        "central": central[order],
        "neighbors": neighbors[order],
        "p2p": p2p[order],
        "features": {name: values[order] for name, values in features.items()},
        "raw_central": raw_central[order],
        "raw_neighborhood": neigh_wave[order],
        "geom": geom,
        "geom_permutation": None if geom_perm is None else geom_perm.tolist(),
        "slot_permutation": None if slot_perm is None else slot_perm.tolist(),
        "k_neighbors": int(k_neighbors),
        "footprint_layout": layout,
    }
    return result


def float_template_assign(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    units: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest float L1 mean-template assignment (strong temporal baseline)."""
    values = np.asarray(train_values, dtype=np.float64)
    labels = np.asarray(train_labels)
    test = np.asarray(test_values, dtype=np.float64)
    if units is None:
        units = np.unique(labels)
    templates = np.stack([values[labels == unit].mean(axis=0) for unit in units], axis=0)
    distances = np.abs(test[:, np.newaxis, :] - templates[np.newaxis, :, :]).sum(axis=2)
    winners = np.argmin(distances, axis=1)
    return units[winners].copy(), templates


def event_weighted_mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    total_n = sum(int(r["n_events"]) for r in rows)
    if total_n == 0:
        return 0.0
    return sum(float(r[key]) * int(r["n_events"]) for r in rows) / total_n


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    # Flatten nested dicts to JSON strings for CSV.
    flat_rows = []
    fieldnames: list[str] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                flat[key] = json.dumps(value, sort_keys=True, default=_json_default)
            elif isinstance(value, (np.floating, float)):
                flat[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                flat[key] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                flat[key] = bool(value)
            else:
                flat[key] = value
            if key not in fieldnames:
                fieldnames.append(key)
        flat_rows.append(flat)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-serialize {type(value)}")
