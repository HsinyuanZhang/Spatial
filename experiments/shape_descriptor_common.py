"""Shared block assembly for the shape-orthogonal descriptor Phase A pilot."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from Spatial.algorithms.adaptive_range_search import UniformUnsignedQuantizer
from Spatial.algorithms.causal_preprocessing import causal_bandpass_filter, causal_event_guard_mask
from Spatial.algorithms.detection import bandpass_filter
from Spatial.algorithms.spatial_footprint import (
    build_relative_patch_table,
    com_features,
    extract_local_extrema,
    extract_local_p2p,
    footprint_p2p_features,
    latency_codes,
    posneg_codes,
    slot_liveness,
    width_codes,
)
from Spatial.data.loader import Dataset
from Spatial.experiments.run_adaptive_5bit_range_search import prepare_dataset_events
from Spatial.experiments.spatial_cim_common import event_weighted_mean

K_NEIGHBORS = 7
HALF_WIDTH = 3
HOME_SLOT = 3
WINDOW = 15
COM_BITS = 5
P2P_BITS = 5
WIDTH_BITS = 5

ARM_IDS = (
    "d9",
    "d9_lat",
    "d9_width",
    "d9_lat_width",
    "d9_rand",
    "com_posneg",
    "com_posneg_lat_width",
)

CANDIDATE_ARM = "com_posneg_lat_width"
CONTROL_ARM = "d9"
RAND_ARM = "d9_rand"

POSNEG_BIT_CHOICES = (4, 5)
LAT_BIT_CHOICES = (3, 4)
AMP_GATE_CHOICES = (0.125, 0.25, 0.375)

CSV_FIELDNAMES = (
    "dataset",
    "family",
    "scene_type",
    "mearec_n_units",
    "arm",
    "posneg_bits",
    "lat_bits",
    "amp_gate",
    "n_train",
    "n_test",
    "n_units",
    "n_coords",
    "row_bits",
    "lane_count",
    "accum_bits",
    "accuracy",
    "n_correct",
    "worst_unit_accuracy",
    "margin_mean",
    "positive_margin_fraction",
    "true_template_distance_norm",
    "hard_quartile_accuracy",
    "nonconforming_slot_fraction",
    "posneg_reconstruction_max_error",
)


def _reject_bool(name: str, value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int, not bool")


def quantize_fixed_unit_interval(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Quantize features in [0, 1] with a fixed analytic range (never fitted)."""
    _reject_bool("n_bits", n_bits)
    bits = int(n_bits)
    if bits < 1 or bits > 62:
        raise ValueError(f"n_bits must be in [1, 62], got {n_bits}")
    quantizer = UniformUnsignedQuantizer(bits, fixed_unit_interval=True)
    quantizer.fit(np.zeros((1, np.asarray(values).shape[1]), dtype=np.float64))
    return quantizer.transform(values).astype(np.float64)


def _lat_max_code(lat_bits: int) -> int:
    _reject_bool("lat_bits", lat_bits)
    limit = (1 << (int(lat_bits) - 1)) - 1
    return int(2 * limit)


def _block_max_code(block: str, *, posneg_bits: int, lat_bits: int) -> int:
    if block in ("COM", "P2P", "POSNEG", "WIDTH", "RAND_WIDTH"):
        if block == "POSNEG":
            return (1 << int(posneg_bits)) - 1
        if block == "RAND_WIDTH":
            return (1 << WIDTH_BITS) - 1
        return (1 << COM_BITS) - 1 if block == "COM" else (1 << P2P_BITS) - 1
    if block in ("LAT", "RAND_LAT"):
        return _lat_max_code(lat_bits)
    raise ValueError(f"unknown block {block!r}")


def arm_block_layout(
    arm: str,
    *,
    posneg_bits: int,
    lat_bits: int,
) -> tuple[tuple[str, int, int], ...]:
    """Return ordered (block_name, n_cols, bits) for one arm."""
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}")
    k = K_NEIGHBORS
    layout: list[tuple[str, int, int]] = [("COM", 2, COM_BITS)]
    if arm in ("d9", "d9_lat", "d9_width", "d9_lat_width", "d9_rand"):
        layout.append(("P2P", k, P2P_BITS))
    elif arm in ("com_posneg", "com_posneg_lat_width"):
        layout.append(("POSNEG", 2 * k, int(posneg_bits)))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if arm in ("d9_lat", "d9_lat_width", "com_posneg_lat_width"):
        layout.append(("LAT", k - 1, int(lat_bits)))
    if arm in ("d9_width", "d9_lat_width", "com_posneg_lat_width"):
        layout.append(("WIDTH", 1, WIDTH_BITS))
    if arm == "d9_rand":
        layout.append(("RAND_LAT", k - 1, int(lat_bits)))
        layout.append(("RAND_WIDTH", 1, WIDTH_BITS))
    return tuple(layout)


def descriptor_ledger(
    arm: str,
    *,
    posneg_bits: int,
    lat_bits: int,
) -> dict[str, int]:
    layout = arm_block_layout(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    n_coords = sum(n_cols for _, n_cols, _ in layout)
    row_bits = sum(n_cols * bits for _, n_cols, bits in layout)
    lane_count = n_coords
    worst_case_sum = 0
    for block_name, n_cols, bits in layout:
        worst_case_sum += n_cols * _block_max_code(
            block_name, posneg_bits=posneg_bits, lat_bits=lat_bits
        )
    accum_bits = int(math.ceil(math.log2(max(worst_case_sum, 1)))) + 1
    return {
        "n_coords": int(n_coords),
        "row_bits": int(row_bits),
        "lane_count": int(lane_count),
        "accum_bits": int(accum_bits),
        "max_code": int(
            max(
                _block_max_code(name, posneg_bits=posneg_bits, lat_bits=lat_bits)
                for name, _, _ in layout
            )
        ),
    }


def build_rand_block(
    n_events: int,
    *,
    lat_bits: int,
    recording_name: str,
) -> np.ndarray:
    """Seeded uniform integers matching LAT+WIDTH coordinate widths."""
    _reject_bool("lat_bits", lat_bits)
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    digest = hashlib.sha256(recording_name.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.default_rng(seed)
    lat_max = _lat_max_code(lat_bits)
    width_max = (1 << WIDTH_BITS) - 1
    lat_part = rng.integers(0, lat_max + 1, size=(n_events, K_NEIGHBORS - 1), dtype=np.int64)
    width_part = rng.integers(0, width_max + 1, size=(n_events, 1), dtype=np.int64)
    return np.concatenate([lat_part, width_part], axis=1).astype(np.float64)


def _filter_recording_signal(dataset: Dataset, filter_mode: str) -> np.ndarray:
    if filter_mode == "zero_phase":
        return bandpass_filter(dataset.raw_data, dataset.fs)
    if filter_mode == "causal":
        return causal_bandpass_filter(dataset.raw_data, dataset.fs)
    raise ValueError(f"filter_mode must be 'zero_phase' or 'causal', got {filter_mode!r}")


def shared_event_guard_mask(
    event_times: np.ndarray,
    *,
    warmup_samples: int,
    n_samples: int,
    window: int = WINDOW,
    max_jitter: int = 2,
) -> np.ndarray:
    """Warm-up plus window/jitter bounds shared by every A2 condition."""
    _reject_bool("warmup_samples", warmup_samples)
    _reject_bool("max_jitter", max_jitter)
    _reject_bool("window", window)
    warm = int(warmup_samples)
    jitter = int(max_jitter)
    win = int(window)
    if warm < 0:
        raise ValueError(f"warmup_samples must be non-negative, got {warmup_samples}")
    if jitter < 0:
        raise ValueError(f"max_jitter must be non-negative, got {max_jitter}")
    mask = causal_event_guard_mask(
        np.asarray(event_times, dtype=np.int64),
        warmup_samples=warm,
        start_sample=0,
        stop_sample=int(n_samples),
    )
    mask &= np.asarray(event_times, dtype=np.int64) >= win + jitter
    mask &= np.asarray(event_times, dtype=np.int64) + win + jitter < int(n_samples)
    return mask


def _subset_prep(prep: Mapping[str, np.ndarray], event_mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(event_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("event_mask must be one-dimensional")
    return {key: np.asarray(value)[mask] for key, value in prep.items()}


def build_descriptor_blocks(
    dataset: Dataset,
    *,
    filter_mode: str = "zero_phase",
    jitter_samples: int = 0,
    event_mask: np.ndarray | None = None,
    frozen_prep: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build every per-event block once on the relative K=7 patch."""
    if filter_mode not in ("zero_phase", "causal"):
        raise ValueError(f"filter_mode must be 'zero_phase' or 'causal', got {filter_mode!r}")
    _reject_bool("jitter_samples", jitter_samples)
    jitter = int(jitter_samples)
    if frozen_prep is not None and event_mask is not None:
        raise ValueError("event_mask must be omitted when frozen_prep is supplied")

    if frozen_prep is None:
        prep = prepare_dataset_events(
            dataset,
            k_neighbors=K_NEIGHBORS,
            window=WINDOW,
            footprint_layout="relative",
        )
        if event_mask is not None:
            prep = _subset_prep(prep, event_mask)
    else:
        prep = {key: np.asarray(value) for key, value in frozen_prep.items()}

    filtered = _filter_recording_signal(dataset, filter_mode)
    neighbor_table = build_relative_patch_table(dataset.geom, half_width=HALF_WIDTH)
    read_times = np.asarray(prep["times"], dtype=np.int64) + jitter
    vmax, vmin, imax, imin, valid_times, neighbor_ids = extract_local_extrema(
        filtered,
        read_times,
        prep["central"],
        neighbor_table,
        window=WINDOW,
    )
    if not np.array_equal(valid_times, read_times):
        raise ValueError(
            "extrema extraction dropped events; tighten shared_event_guard_mask before evaluation"
        )
    if not np.array_equal(neighbor_ids, prep["neighbor_ids"]):
        raise AssertionError("neighbor_ids must match prepare_dataset_events")

    if filter_mode == "zero_phase" and jitter == 0:
        com_codes = quantize_fixed_unit_interval(prep["com"], COM_BITS)
        p2p_codes = quantize_fixed_unit_interval(prep["p2p"], P2P_BITS)
    else:
        p2p_raw, p2p_times, _ = extract_local_p2p(
            filtered,
            prep["times"],
            prep["central"],
            neighbor_table,
            window=WINDOW,
        )
        if not np.array_equal(p2p_times, prep["times"]):
            raise AssertionError("local P2P must preserve aligned event rows")
        com_codes = quantize_fixed_unit_interval(
            com_features(p2p_raw, prep["neighbor_ids"], dataset.geom),
            COM_BITS,
        )
        p2p_codes = quantize_fixed_unit_interval(
            footprint_p2p_features(p2p_raw, normalize=True),
            P2P_BITS,
        )
    return {
        "times": np.asarray(prep["times"], dtype=np.int64),
        "labels": np.asarray(prep["labels"]),
        "central": np.asarray(prep["central"], dtype=np.int64),
        "com": com_codes,
        "p2p": p2p_codes,
        "vmax": vmax,
        "vmin": vmin,
        "imax": imax,
        "imin": imin,
        "neighbor_ids": neighbor_ids,
    }


def assemble_arm(
    blocks: Mapping[str, np.ndarray],
    arm: str,
    *,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
    recording_name: str,
) -> np.ndarray:
    """Assemble one named arm from pre-built blocks."""
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}")
    _reject_bool("posneg_bits", posneg_bits)
    _reject_bool("lat_bits", lat_bits)
    gate = float(amp_gate)
    if not np.isfinite(gate) or gate < 0.0 or gate > 1.0:
        raise ValueError(f"amp_gate must be a finite float in [0.0, 1.0], got {amp_gate}")

    parts: list[np.ndarray] = [np.asarray(blocks["com"], dtype=np.float64)]
    if arm in ("d9", "d9_lat", "d9_width", "d9_lat_width", "d9_rand"):
        parts.append(np.asarray(blocks["p2p"], dtype=np.float64))
    elif arm in ("com_posneg", "com_posneg_lat_width"):
        parts.append(
            posneg_codes(
                blocks["vmax"],
                blocks["vmin"],
                blocks["neighbor_ids"],
                n_bits=int(posneg_bits),
            )
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    live = slot_liveness(
        blocks["vmax"], blocks["vmin"], blocks["neighbor_ids"], amp_gate=gate
    )
    if arm in ("d9_lat", "d9_lat_width", "com_posneg_lat_width"):
        parts.append(
            latency_codes(blocks["imin"], live, HOME_SLOT, n_bits=int(lat_bits))
        )
    if arm in ("d9_width", "d9_lat_width", "com_posneg_lat_width"):
        parts.append(width_codes(blocks["imax"], blocks["imin"], HOME_SLOT, n_bits=WIDTH_BITS))
    if arm == "d9_rand":
        parts.append(
            build_rand_block(
                blocks["com"].shape[0],
                lat_bits=int(lat_bits),
                recording_name=recording_name,
            )
        )
    return np.column_stack(parts)


def arm_uses_posneg(arm: str) -> bool:
    return arm in ("com_posneg", "com_posneg_lat_width")


def arm_uses_lat(arm: str) -> bool:
    return arm in ("d9_lat", "d9_lat_width", "com_posneg_lat_width")


def arm_uses_amp_gate(arm: str) -> bool:
    return arm in ("d9_lat", "d9_lat_width", "com_posneg_lat_width")


def arm_uses_lat_bits(arm: str) -> bool:
    return arm in ("d9_lat", "d9_lat_width", "d9_rand", "com_posneg_lat_width")


def evaluation_cache_key(
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
) -> tuple[Any, ...]:
    key: list[Any] = [arm]
    if arm_uses_posneg(arm):
        key.append(int(posneg_bits))
    if arm_uses_lat_bits(arm):
        key.append(int(lat_bits))
    if arm_uses_amp_gate(arm):
        key.append(float(amp_gate))
    return tuple(key)


def posneg_diagnostics(
    blocks: Mapping[str, np.ndarray],
    *,
    posneg_bits: int,
) -> tuple[float, int]:
    """Return nonconforming fraction and max reconstruction error on conforming slots."""
    codes = posneg_codes(
        blocks["vmax"],
        blocks["vmin"],
        blocks["neighbor_ids"],
        n_bits=int(posneg_bits),
    )
    k = K_NEIGHBORS
    pos = codes[:, :k]
    neg = codes[:, k : 2 * k]
    in_bounds = blocks["neighbor_ids"] >= 0
    conforming = in_bounds & (blocks["vmax"] >= 0.0) & (blocks["vmin"] <= 0.0)
    n_in_bounds = int(in_bounds.sum())
    nonconforming_fraction = (
        float(np.count_nonzero(in_bounds & ~conforming)) / n_in_bounds
        if n_in_bounds
        else 0.0
    )
    if not np.any(conforming):
        return nonconforming_fraction, 0

    # The scale must be the one posneg_codes used, which is the per-event max
    # over in-bounds slots. Masking to conforming slots here would compare the
    # codes against a different denominator.
    p2p = blocks["vmax"] - blocks["vmin"]
    masked = np.where(in_bounds, p2p, -np.inf)
    scale = np.maximum(np.max(masked, axis=1, keepdims=True), 1e-12)
    max_level = float((1 << int(posneg_bits)) - 1)
    target = np.rint(max_level * p2p / scale)
    err = np.abs(pos + neg - target)[conforming]
    return nonconforming_fraction, int(np.max(err))


def fit_integer_templates(
    codes: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(codes, dtype=np.float64)
    labs = np.asarray(labels)
    units = np.unique(labs)
    templates = np.stack(
        [np.rint(values[labs == unit].mean(axis=0)).astype(np.int64) for unit in units],
        axis=0,
    )
    return units, templates


def assign_nearest_l1(
    query: np.ndarray,
    templates: np.ndarray,
    units: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-template integer L1 with first-minimum tie-break."""
    q = np.asarray(query, dtype=np.int64)
    refs = np.asarray(templates, dtype=np.int64)
    distances = np.abs(q[:, None, :] - refs[None, :, :]).sum(axis=2, dtype=np.int64)
    order = np.argsort(distances, axis=1, kind="stable")
    pred_cols = order[:, 0]
    sorted_d = np.take_along_axis(distances, order, axis=1)
    top2_gap = sorted_d[:, 1] - sorted_d[:, 0] if distances.shape[1] > 1 else np.zeros(q.shape[0], dtype=np.int64)
    return units[pred_cols].copy(), distances, top2_gap


def evaluate_arm_rows(
    blocks: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
    recording_name: str,
) -> dict[str, Any]:
    codes = assemble_arm(
        blocks,
        arm,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=amp_gate,
        recording_name=recording_name,
    )
    ledger = descriptor_ledger(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    units, templates = fit_integer_templates(codes[train_idx], blocks["labels"][train_idx])
    unit_to_col = {
        unit.item() if isinstance(unit, np.generic) else unit: i
        for i, unit in enumerate(units)
    }
    labels_te = blocks["labels"][test_idx]
    keep = np.asarray(
        [
            (label.item() if isinstance(label, np.generic) else label) in unit_to_col
            for label in labels_te
        ],
        dtype=bool,
    )
    eval_idx = test_idx[keep]
    if eval_idx.size == 0:
        raise ValueError("no test labels have a fitted template")

    query = np.rint(codes[eval_idx]).astype(np.int64)
    truth = blocks["labels"][eval_idx]
    pred, distances, top2_gap = assign_nearest_l1(query, templates, units)
    true_cols = np.asarray(
        [
            unit_to_col[label.item() if isinstance(label, np.generic) else label]
            for label in truth
        ],
        dtype=np.int64,
    )
    pred_cols = np.asarray(
        [unit_to_col[p.item() if isinstance(p, np.generic) else p] for p in pred],
        dtype=np.int64,
    )
    correct = pred_cols == true_cols
    true_dist = distances[np.arange(distances.shape[0]), true_cols].astype(np.float64)
    wrong = distances.astype(np.float64, copy=True)
    wrong[np.arange(wrong.shape[0]), true_cols] = np.inf
    nearest_wrong = np.min(wrong, axis=1)
    margins = nearest_wrong - true_dist

    per_unit_acc = []
    for col in np.unique(true_cols):
        mask = true_cols == col
        per_unit_acc.append(float(np.mean(correct[mask])))

    hard_mask = np.zeros(eval_idx.size, dtype=bool)
    if eval_idx.size:
        n_hard = max(1, int(math.ceil(0.25 * eval_idx.size)))
        hard_order = np.argsort(top2_gap, kind="stable")
        hard_mask[hard_order[:n_hard]] = True

    normalizer = float(ledger["lane_count"] * ledger["max_code"])
    return {
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_units": int(units.size),
        "n_coords": ledger["n_coords"],
        "row_bits": ledger["row_bits"],
        "lane_count": ledger["lane_count"],
        "accum_bits": ledger["accum_bits"],
        "accuracy": float(np.mean(correct)),
        "n_correct": int(np.sum(correct)),
        "worst_unit_accuracy": float(min(per_unit_acc)) if per_unit_acc else 0.0,
        "margin_mean": float(np.mean(margins)),
        "positive_margin_fraction": float(np.mean(margins > 0.0)),
        "true_template_distance_norm": float(np.mean(true_dist) / max(normalizer, 1.0)),
        "hard_quartile_accuracy": float(np.mean(correct[hard_mask])) if hard_mask.any() else 0.0,
        **dict(
            zip(
                ("nonconforming_slot_fraction", "posneg_reconstruction_max_error"),
                posneg_diagnostics(blocks, posneg_bits=posneg_bits),
            )
        ),
    }


def _scene_type(dataset_name: str, family: str) -> str:
    if family == "mearec":
        return "mearec"
    if "drift" in dataset_name:
        return "drift"
    if "static" in dataset_name:
        return "static"
    return "other"


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
) -> Mapping[str, Any] | None:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["arm"] == arm
            and int(row["posneg_bits"]) == int(posneg_bits)
            and int(row["lat_bits"]) == int(lat_bits)
            and float(row["amp_gate"]) == float(amp_gate)
        ):
            return row
    return None


def _family_weighted_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    family: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
) -> float:
    subset = [
        row
        for row in rows
        if row["family"] == family
        and row["arm"] == arm
        and int(row["posneg_bits"]) == int(posneg_bits)
        and int(row["lat_bits"]) == int(lat_bits)
        and float(row["amp_gate"]) == float(amp_gate)
    ]
    if not subset:
        return float("nan")
    total = sum(int(row["n_test"]) for row in subset)
    if total == 0:
        return float("nan")
    return sum(float(row["accuracy"]) * int(row["n_test"]) for row in subset) / total


def _family_weighted_margin_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    family: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
) -> float:
    subset = [
        row
        for row in rows
        if row["family"] == family
        and row["arm"] == arm
        and int(row["posneg_bits"]) == int(posneg_bits)
        and int(row["lat_bits"]) == int(lat_bits)
        and float(row["amp_gate"]) == float(amp_gate)
    ]
    if not subset:
        return float("nan")
    total = sum(int(row["n_test"]) for row in subset)
    if total == 0:
        return float("nan")
    return (
        sum(float(row["positive_margin_fraction"]) * int(row["n_test"]) for row in subset)
        / total
    )


def evaluate_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_recordings: Sequence[str],
) -> dict[str, Any]:
    """Rank configurations and evaluate the frozen Phase A gate.

    Configurations are taken from the rows actually present so that a reduced
    sweep stays valid input, while a recording missing inside a present
    configuration remains a hard error.
    """
    present = {
        (int(row["posneg_bits"]), int(row["lat_bits"]), float(row["amp_gate"]))
        for row in rows
    }
    configs = sorted(
        config
        for config in present
        if config[0] in POSNEG_BIT_CHOICES
        and config[1] in LAT_BIT_CHOICES
        and config[2] in AMP_GATE_CHOICES
    )
    if not configs:
        raise ValueError("no rows match the registered Phase A sweep values")

    best: tuple[Any, ...] | None = None
    best_rank: tuple[Any, ...] | None = None
    best_gate: dict[str, Any] | None = None

    for posneg_bits, lat_bits, amp_gate in configs:
        gate = _gate_for_config(
            rows,
            posneg_bits=posneg_bits,
            lat_bits=lat_bits,
            amp_gate=amp_gate,
            development_recordings=development_recordings,
        )
        if not gate["passed"]:
            continue
        hj_acc = _family_weighted_accuracy(
            rows,
            arm=CANDIDATE_ARM,
            family="hj",
            posneg_bits=posneg_bits,
            lat_bits=lat_bits,
            amp_gate=amp_gate,
        )
        mearec_acc = _family_weighted_accuracy(
            rows,
            arm=CANDIDATE_ARM,
            family="mearec",
            posneg_bits=posneg_bits,
            lat_bits=lat_bits,
            amp_gate=amp_gate,
        )
        two_family_mean = (hj_acc + mearec_acc) / 2.0
        candidate_rows = [
            row
            for row in rows
            if row["arm"] == CANDIDATE_ARM
            and int(row["posneg_bits"]) == int(posneg_bits)
            and int(row["lat_bits"]) == int(lat_bits)
            and float(row["amp_gate"]) == float(amp_gate)
        ]
        hard_q = event_weighted_mean(
            [{"n_events": row["n_test"], "hard_quartile_accuracy": row["hard_quartile_accuracy"]} for row in candidate_rows],
            "hard_quartile_accuracy",
        )
        pos_margin = event_weighted_mean(
            [{"n_events": row["n_test"], "positive_margin_fraction": row["positive_margin_fraction"]} for row in candidate_rows],
            "positive_margin_fraction",
        )
        row_bits = int(candidate_rows[0]["row_bits"]) if candidate_rows else 10**9
        rank = (
            int(sum(1 for key in (
                "condition_1_family_gain",
                "condition_2_no_recording_loss",
                "condition_3_positive_margin",
                "condition_4_reconstruction",
                "condition_5_rand_control_fails",
            ) if gate["gate_results"][key])),
            two_family_mean,
            hard_q,
            pos_margin,
            -row_bits,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (posneg_bits, lat_bits, amp_gate)
            best_gate = gate

    if best is None or best_gate is None:
        failing = _gate_for_config(
            rows,
            posneg_bits=5,
            lat_bits=4,
            amp_gate=0.25,
            development_recordings=development_recordings,
        )
        return {
            "plan": "shape_orthogonal_descriptor_plan.md",
            "phase": "A",
            "development_recordings": list(development_recordings),
            "selected_arm": None,
            "posneg_bits": None,
            "lat_bits": None,
            "amp_gate": None,
            "gate_results": failing["gate_results"],
            "passed": False,
            "null_reason": failing["null_reason"],
        }

    posneg_bits, lat_bits, amp_gate = best
    return {
        "plan": "shape_orthogonal_descriptor_plan.md",
        "phase": "A",
        "development_recordings": list(development_recordings),
        "selected_arm": CANDIDATE_ARM,
        "posneg_bits": int(posneg_bits),
        "lat_bits": int(lat_bits),
        "amp_gate": float(amp_gate),
        "gate_results": best_gate["gate_results"],
        "passed": True,
        "null_reason": None,
    }


def _gate_for_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
    development_recordings: Sequence[str],
) -> dict[str, Any]:
    hj_candidate = _family_weighted_accuracy(
        rows, arm=CANDIDATE_ARM, family="hj", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )
    mearec_candidate = _family_weighted_accuracy(
        rows, arm=CANDIDATE_ARM, family="mearec", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )
    hj_control = _family_weighted_accuracy(
        rows, arm=CONTROL_ARM, family="hj", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )
    mearec_control = _family_weighted_accuracy(
        rows, arm=CONTROL_ARM, family="mearec", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )
    hj_rand = _family_weighted_accuracy(
        rows, arm=RAND_ARM, family="hj", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )
    mearec_rand = _family_weighted_accuracy(
        rows, arm=RAND_ARM, family="mearec", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
    )

    family_gain_hj_pp = 100.0 * (hj_candidate - hj_control)
    family_gain_mearec_pp = 100.0 * (mearec_candidate - mearec_control)
    rand_gain_hj_pp = 100.0 * (hj_rand - hj_control)
    rand_gain_mearec_pp = 100.0 * (mearec_rand - mearec_control)

    per_recording_deltas: list[float] = []
    for dataset in development_recordings:
        cand = _row_lookup(
            rows,
            dataset=dataset,
            arm=CANDIDATE_ARM,
            posneg_bits=posneg_bits,
            lat_bits=lat_bits,
            amp_gate=amp_gate,
        )
        ctrl = _row_lookup(
            rows,
            dataset=dataset,
            arm=CONTROL_ARM,
            posneg_bits=posneg_bits,
            lat_bits=lat_bits,
            amp_gate=amp_gate,
        )
        # A silently missed join would make condition 2 vacuously true, so an
        # unmatched development recording is an error rather than a skip.
        if cand is None or ctrl is None:
            raise ValueError(
                f"no {CANDIDATE_ARM}/{CONTROL_ARM} row pair for dataset {dataset!r} at "
                f"posneg_bits={posneg_bits}, lat_bits={lat_bits}, amp_gate={amp_gate}; "
                "development_recordings must use the same names as the CSV dataset column"
            )
        per_recording_deltas.append(
            100.0 * (float(cand["accuracy"]) - float(ctrl["accuracy"]))
        )
    if not per_recording_deltas:
        raise ValueError("development_recordings must not be empty")
    worst_recording_delta_pp = min(per_recording_deltas)

    hj_margin_delta_pp = 100.0 * (
        _family_weighted_margin_fraction(
            rows, arm=CANDIDATE_ARM, family="hj", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
        )
        - _family_weighted_margin_fraction(
            rows, arm=CONTROL_ARM, family="hj", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
        )
    )
    mearec_margin_delta_pp = 100.0 * (
        _family_weighted_margin_fraction(
            rows, arm=CANDIDATE_ARM, family="mearec", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
        )
        - _family_weighted_margin_fraction(
            rows, arm=CONTROL_ARM, family="mearec", posneg_bits=posneg_bits, lat_bits=lat_bits, amp_gate=amp_gate
        )
    )

    candidate_rows = [
        row
        for row in rows
        if row["arm"] == CANDIDATE_ARM
        and int(row["posneg_bits"]) == int(posneg_bits)
        and int(row["lat_bits"]) == int(lat_bits)
        and float(row["amp_gate"]) == float(amp_gate)
    ]
    max_nonconforming = max(
        (float(row["nonconforming_slot_fraction"]) for row in candidate_rows),
        default=0.0,
    )
    max_posneg_error = max(
        (int(row["posneg_reconstruction_max_error"]) for row in candidate_rows),
        default=0,
    )

    condition_1 = family_gain_hj_pp >= 2.0 and family_gain_mearec_pp >= 2.0
    condition_2 = worst_recording_delta_pp >= -0.5
    condition_3 = hj_margin_delta_pp >= 0.0 and mearec_margin_delta_pp >= 0.0
    condition_4 = max_posneg_error <= 1 and max_nonconforming < 0.005
    rand_satisfies_1 = rand_gain_hj_pp >= 2.0 and rand_gain_mearec_pp >= 2.0
    condition_5 = not rand_satisfies_1

    gate_results = {
        "family_gain_hj_pp": float(family_gain_hj_pp),
        "family_gain_mearec_pp": float(family_gain_mearec_pp),
        "worst_recording_delta_pp": float(worst_recording_delta_pp),
        "positive_margin_delta_hj_pp": float(hj_margin_delta_pp),
        "positive_margin_delta_mearec_pp": float(mearec_margin_delta_pp),
        "max_nonconforming_slot_fraction": float(max_nonconforming),
        "max_posneg_reconstruction_error": int(max_posneg_error),
        "rand_control_gain_hj_pp": float(rand_gain_hj_pp),
        "rand_control_gain_mearec_pp": float(rand_gain_mearec_pp),
        "condition_1_family_gain": bool(condition_1),
        "condition_2_no_recording_loss": bool(condition_2),
        "condition_3_positive_margin": bool(condition_3),
        "condition_4_reconstruction": bool(condition_4),
        "condition_5_rand_control_fails": bool(condition_5),
    }
    passed = all(
        gate_results[key]
        for key in (
            "condition_1_family_gain",
            "condition_2_no_recording_loss",
            "condition_3_positive_margin",
            "condition_4_reconstruction",
            "condition_5_rand_control_fails",
        )
    )
    null_reason = None
    if not passed:
        failed = [
            name
            for name, ok in (
                ("condition_1_family_gain", condition_1),
                ("condition_2_no_recording_loss", condition_2),
                ("condition_3_positive_margin", condition_3),
                ("condition_4_reconstruction", condition_4),
                ("condition_5_rand_control_fails", condition_5),
            )
            if not ok
        ]
        null_reason = "failed: " + ", ".join(failed)
    return {"gate_results": gate_results, "passed": passed, "null_reason": null_reason}
