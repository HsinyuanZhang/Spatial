"""Pre-registered robust-P2P estimator x 4--8-bit template pilot.

This is GT-event / GT-row component isolation. It holds the relative K=7
patch and S=0 matcher fixed while crossing explicit temporal P2P estimators
with integer descriptor widths. See ``docs/robust_p2p_template_plan.md``.

Run from the parent workspace root:

    python -m Spatial.experiments.run_robust_p2p_template_pilot --pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.algorithms.detection import bandpass_filter, get_peak_amplitudes
from Spatial.algorithms.shift_match import shift_max_dot
from Spatial.algorithms.spatial_footprint import (
    build_relative_patch_table,
    extract_local_p2p,
    footprint_p2p_features,
)
from Spatial.data.loader import (
    Dataset,
    list_hybrid_janelia_scenes,
    load_hybrid_janelia,
    load_mearec_npz,
)
from Spatial.experiments.run_adaptive_5bit_range_search import (
    WAVEFORM_POST,
    labels_for_peak_output,
)
from Spatial.experiments.run_adaptive_masked_5bit import (
    stable_chronological_three_way_split,
)
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "output" / "robust_p2p_template_pilot"
PILOT_HJ_SCENES = ("drift16c_600s_11", "static16c_600s_11")
PILOT_MEAREC_FILES = (
    "rec_v1_units10_snr5_seed202601.npz",
    "rec_v1_units20_snr5_seed202601.npz",
)
CONFIRMATION_MEAREC_FILES = (
    "rec_v1_units10_snr5_seed202600.npz",
    "rec_v1_units20_snr5_seed202600.npz",
)
FROZEN_ESTIMATOR = "centered_mean_k5"
FROZEN_BITS = 8
ESTIMATOR_GRID = (
    ("raw_range", "raw", 1, True),
    ("centered_mean_k3", "centered_mean", 3, True),
    ("centered_mean_k5", "centered_mean", 5, True),
    ("moving_average_k3", "moving_average", 3, True),
    ("moving_average_k5", "moving_average", 5, True),
    ("extreme_mean_k3", "extreme_mean", 3, False),
    ("extreme_mean_k5", "extreme_mean", 5, False),
)
BIT_MODES = ("float", "4", "5", "6", "7", "8")
MATCHERS = ("l1", "cosine")


def _condition(name: str) -> str:
    if "drift" in name:
        return "drift"
    if "static" in name:
        return "static"
    return "other"


def _pilot_datasets(duration: float, mearec_dir: Path):
    for scene in PILOT_HJ_SCENES:
        yield "hj", load_hybrid_janelia(scene=scene, duration_s=duration)
    for filename in PILOT_MEAREC_FILES:
        path = Path(mearec_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"locked MEArec pilot input is missing: {path}")
        yield "mearec", load_mearec_npz(path, duration_s=duration)


def _confirmation_datasets(duration: float, mearec_dir: Path):
    ready = [
        entry
        for entry in list_hybrid_janelia_scenes(ready_only=True)
        if entry.get("short_name", "").startswith(("drift16c_600s_", "static16c_600s_"))
        and entry.get("short_name") not in PILOT_HJ_SCENES
    ]
    ready.sort(key=lambda entry: str(entry["short_name"]))
    if len(ready) != 10:
        names = [str(entry.get("short_name")) for entry in ready]
        raise ValueError(f"expected exactly 10 held-out HJ 16c scenes, found {names}")
    for entry in ready:
        yield "hj", load_hybrid_janelia(
            scene=str(entry["short_name"]), duration_s=duration
        )
    for filename in CONFIRMATION_MEAREC_FILES:
        path = Path(mearec_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"locked MEArec confirmation input is missing: {path}")
        yield "mearec", load_mearec_npz(path, duration_s=duration)


def prepare_event_context(
    dataset: Dataset,
    *,
    k_neighbors: int = 7,
    window: int = 15,
) -> dict[str, Any]:
    """Filter once and freeze aligned rows shared by every estimator."""
    filtered = bandpass_filter(dataset.raw_data, dataset.fs)
    _, times, central = get_peak_amplitudes(
        filtered, dataset.spike_times, window=window
    )
    labels = labels_for_peak_output(
        dataset.spike_times,
        dataset.spike_units,
        times,
        dataset.n_samples,
        window=window,
    )
    # Match the existing relative/shift runners' event-row boundary even though
    # this isolation runner does not read Level-2 waveforms.
    waveform_ok = times + WAVEFORM_POST <= dataset.n_samples
    times = times[waveform_ok]
    central = central[waveform_ok]
    labels = labels[waveform_ok]
    if times.size < 3:
        raise ValueError("fewer than three aligned event rows remain")

    k = min(int(k_neighbors), dataset.n_channels)
    table = build_relative_patch_table(
        dataset.geom, half_width=max(k // 2, 0)
    )
    order = np.argsort(times, kind="stable")
    times = times[order]
    central = central[order]
    labels = labels[order]
    if np.any(times[1:] < times[:-1]):
        raise AssertionError("event rows must be stably chronological")
    return {
        "filtered": filtered,
        "times": times,
        "central": central,
        "labels": labels,
        "neighbor_table": table,
        "geom": np.asarray(dataset.geom, dtype=np.float64),
        "window": int(window),
    }


def extract_estimator_footprint(
    context: dict[str, Any],
    *,
    estimator: str,
    average_points: int,
) -> np.ndarray:
    """Extract one estimator on the exact frozen event-row identity."""
    raw, returned, _ = extract_local_p2p(
        context["filtered"],
        context["times"],
        context["central"],
        context["neighbor_table"],
        window=int(context["window"]),
        p2p_estimator=estimator,
        average_points=int(average_points),
    )
    if not np.array_equal(returned, context["times"]):
        raise AssertionError("P2P estimator changed the frozen event rows")
    footprint = footprint_p2p_features(raw, normalize=True)
    if (
        footprint.shape[0] != context["times"].size
        or not np.all(np.isfinite(footprint))
        or np.any((footprint < 0.0) | (footprint > 1.0))
    ):
        raise ValueError("P2P estimator produced an invalid normalized footprint")
    return footprint


def quantize_footprint(
    footprint: np.ndarray, bit_mode: str
) -> tuple[np.ndarray, int, int, bool]:
    """Return float reference rows or exact unsigned integer descriptor rows."""
    values = np.asarray(footprint, dtype=np.float64)
    if bit_mode == "float":
        return values.copy(), 32, 1, False
    bits = int(bit_mode)
    if not 1 <= bits <= 16:
        raise ValueError("descriptor bits must be in [1,16]")
    code_max = (1 << bits) - 1
    codes = np.rint(values * code_max).astype(np.int64)
    np.clip(codes, 0, code_max, out=codes)
    return codes, bits, code_max, True


def fit_templates(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    integer: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one template per GT unit; integer rows use rint(mean(code))."""
    values = np.asarray(rows)
    labs = np.asarray(labels)
    units = np.unique(labs)
    templates = np.stack([values[labs == unit].mean(axis=0) for unit in units])
    if integer:
        templates = np.rint(templates).astype(np.int64)
    else:
        templates = templates.astype(np.float64)
    return units, templates


def _l1_distances(rows: np.ndarray, templates: np.ndarray) -> np.ndarray:
    values = np.asarray(rows)
    refs = np.asarray(templates)
    if np.issubdtype(values.dtype, np.integer) and np.issubdtype(
        refs.dtype, np.integer
    ):
        return np.sum(
            np.abs(values[:, None, :].astype(np.int64) - refs[None, :, :].astype(np.int64)),
            axis=2,
            dtype=np.int64,
        )
    return np.sum(
        np.abs(values[:, None, :].astype(np.float64) - refs[None, :, :].astype(np.float64)),
        axis=2,
    )


def evaluate_rows(
    footprint: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    dataset_name: str,
    family: str,
    estimator_name: str,
    estimator: str,
    average_points: int,
    hardware_eligible: bool,
    bit_mode: str,
    matcher: str,
) -> dict[str, Any]:
    fit_idx, cal_idx, test_idx = stable_chronological_three_way_split(times)
    rows, nominal_bits, code_max, integer = quantize_footprint(
        footprint, bit_mode
    )
    units, templates = fit_templates(
        rows[fit_idx], labels[fit_idx], integer=integer
    )
    unit_to_col = {
        unit.item() if isinstance(unit, np.generic) else unit: i
        for i, unit in enumerate(units)
    }
    keep = np.asarray(
        [
            (label.item() if isinstance(label, np.generic) else label) in unit_to_col
            for label in labels[test_idx]
        ],
        dtype=bool,
    )
    eval_idx = test_idx[keep]
    if eval_idx.size == 0:
        raise ValueError("no test labels have a fitted template")

    query = rows[eval_idx]
    truth = labels[eval_idx]
    true_cols = np.asarray(
        [
            unit_to_col[label.item() if isinstance(label, np.generic) else label]
            for label in truth
        ],
        dtype=np.int64,
    )

    if matcher == "l1":
        values = _l1_distances(query, templates)
        pred_cols = np.argmin(values, axis=1)
        true_values = values[np.arange(values.shape[0]), true_cols].astype(np.float64)
        wrong = values.astype(np.float64, copy=True)
        wrong[np.arange(wrong.shape[0]), true_cols] = np.inf
        margins = np.min(wrong, axis=1) - true_values
        normalizer = float(query.shape[1] * code_max)
        true_values_normalized = true_values / max(normalizer, 1.0)
        margins_normalized = margins / max(normalizer, 1.0)
    elif matcher == "cosine":
        values = shift_max_dot(
            query.astype(np.float64),
            templates.astype(np.float64),
            shift_radius=0,
            normalize=True,
        )
        pred_cols = np.argmax(values, axis=1)
        true_values = values[np.arange(values.shape[0]), true_cols]
        wrong = values.copy()
        wrong[np.arange(wrong.shape[0]), true_cols] = -np.inf
        margins = true_values - np.max(wrong, axis=1)
        true_values_normalized = true_values
        margins_normalized = margins
    else:
        raise ValueError(f"unknown matcher: {matcher}")

    correct = pred_cols == true_cols
    per_unit_accuracy = []
    for col in np.unique(true_cols):
        mask = true_cols == col
        per_unit_accuracy.append(float(np.mean(correct[mask])))

    guard_bits = (
        0
        if estimator == "raw"
        else int(math.ceil(math.log2(max(int(average_points), 1))))
    )
    return {
        "dataset": dataset_name,
        "family": family,
        "condition": _condition(dataset_name),
        "estimator_name": estimator_name,
        "estimator": estimator,
        "average_points": int(average_points),
        "hardware_eligible": bool(hardware_eligible),
        "bit_mode": bit_mode,
        "nominal_bits": int(nominal_bits),
        "integer_template": bool(integer),
        "matcher": matcher,
        "shift_radius": 0,
        "k": int(query.shape[1]),
        "n_fit": int(fit_idx.size),
        "n_cal": int(cal_idx.size),
        "n_test": int(test_idx.size),
        "n_eval": int(eval_idx.size),
        "n_units": int(units.size),
        "n_correct": int(np.sum(correct)),
        "assignment_accuracy": float(np.mean(correct)),
        "worst_unit_accuracy": float(min(per_unit_accuracy)),
        "median_true_value": float(np.median(true_values)),
        "median_true_value_normalized": float(np.median(true_values_normalized)),
        "median_margin": float(np.median(margins)),
        "median_margin_normalized": float(np.median(margins_normalized)),
        "positive_margin_fraction": float(np.mean(margins > 0.0)),
        "template_payload_bits": int(units.size * query.shape[1] * nominal_bits),
        "estimator_sum_guard_bits": int(guard_bits),
    }


def _subset(
    rows: list[dict[str, Any]],
    *,
    estimator_name: str,
    bit_mode: str,
    matcher: str,
    family: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["estimator_name"] == estimator_name
        and row["bit_mode"] == bit_mode
        and row["matcher"] == matcher
        and (family is None or row["family"] == family)
    ]


def _event_accuracy(rows: list[dict[str, Any]]) -> float:
    total = sum(int(row["n_eval"]) for row in rows)
    return (
        sum(int(row["n_correct"]) for row in rows) / total
        if total
        else float("nan")
    )


def _event_mean(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["n_eval"]) for row in rows)
    return (
        sum(float(row[key]) * int(row["n_eval"]) for row in rows) / total
        if total
        else float("nan")
    )


def summarize_and_select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the registered estimator and bit-width gates deterministically."""
    grids: dict[str, Any] = {}
    for estimator_name, _, _, _ in ESTIMATOR_GRID:
        grids[estimator_name] = {}
        for bit_mode in BIT_MODES:
            for matcher in MATCHERS:
                key = f"b{bit_mode}_{matcher}"
                grids[estimator_name][key] = {}
                for family in ("hj", "mearec"):
                    sub = _subset(
                        rows,
                        estimator_name=estimator_name,
                        bit_mode=bit_mode,
                        matcher=matcher,
                        family=family,
                    )
                    grids[estimator_name][key][family] = {
                        "event_accuracy": _event_accuracy(sub),
                        "recording_mean_accuracy": float(
                            np.mean([row["assignment_accuracy"] for row in sub])
                        ),
                        "positive_margin_fraction": _event_mean(
                            sub, "positive_margin_fraction"
                        ),
                        "median_true_value_normalized_recording_mean": float(
                            np.mean(
                                [row["median_true_value_normalized"] for row in sub]
                            )
                        ),
                        "n_recordings": len(sub),
                    }

    raw_l1 = {
        family: grids["raw_range"]["bfloat_l1"][family]
        for family in ("hj", "mearec")
    }
    raw_cos = {
        family: grids["raw_range"]["bfloat_cosine"][family]
        for family in ("hj", "mearec")
    }
    raw_record = {
        row["dataset"]: row
        for row in _subset(
            rows,
            estimator_name="raw_range",
            bit_mode="float",
            matcher="l1",
        )
    }

    estimator_audit: dict[str, Any] = {}
    eligible: list[tuple[tuple[float, ...], str]] = []
    for estimator_name, _, average_points, hardware_eligible in ESTIMATOR_GRID:
        if estimator_name == "raw_range" or not hardware_eligible:
            continue
        current_l1 = {
            family: grids[estimator_name]["bfloat_l1"][family]
            for family in ("hj", "mearec")
        }
        current_cos = {
            family: grids[estimator_name]["bfloat_cosine"][family]
            for family in ("hj", "mearec")
        }
        family_l1_ok = all(
            current_l1[family]["event_accuracy"]
            >= raw_l1[family]["event_accuracy"] - 0.005
            for family in ("hj", "mearec")
        )
        family_cos_ok = all(
            current_cos[family]["event_accuracy"]
            >= raw_cos[family]["event_accuracy"] - 0.005
            for family in ("hj", "mearec")
        )
        candidate_records = _subset(
            rows,
            estimator_name=estimator_name,
            bit_mode="float",
            matcher="l1",
        )
        record_safeguard = all(
            row["assignment_accuracy"]
            >= raw_record[row["dataset"]]["assignment_accuracy"] - 0.01
            for row in candidate_records
        )
        drift = next(
            row
            for row in candidate_records
            if row["family"] == "hj" and row["condition"] == "drift"
        )
        raw_drift = raw_record[drift["dataset"]]
        drift_accuracy_gain = (
            drift["assignment_accuracy"]
            >= raw_drift["assignment_accuracy"] + 0.005
        )
        drift_distance_gain = (
            drift["median_true_value_normalized"]
            <= 0.98 * raw_drift["median_true_value_normalized"]
            and drift["positive_margin_fraction"]
            >= raw_drift["positive_margin_fraction"] - 1e-12
        )
        drift_ok = bool(drift_accuracy_gain or drift_distance_gain)
        passes = bool(family_l1_ok and family_cos_ok and record_safeguard and drift_ok)
        audit = {
            "family_l1_safeguard": bool(family_l1_ok),
            "family_cosine_safeguard": bool(family_cos_ok),
            "per_record_safeguard": bool(record_safeguard),
            "hj_drift_mechanism": bool(drift_ok),
            "hj_drift_accuracy_gain": bool(drift_accuracy_gain),
            "hj_drift_distance_gain": bool(drift_distance_gain),
            "passes": passes,
        }
        estimator_audit[estimator_name] = audit
        if passes:
            two_family_acc = 0.5 * sum(
                current_l1[family]["event_accuracy"]
                for family in ("hj", "mearec")
            )
            two_family_margin = 0.5 * sum(
                current_l1[family]["positive_margin_fraction"]
                for family in ("hj", "mearec")
            )
            rank = (
                float(drift["assignment_accuracy"]),
                float(two_family_acc),
                float(two_family_margin),
                float(-average_points),
            )
            eligible.append((rank, estimator_name))

    if eligible:
        eligible.sort(reverse=True)
        selected_estimator = eligible[0][1]
        estimator_gate_pass = True
    else:
        selected_estimator = "raw_range"
        estimator_gate_pass = False

    bit_audit: dict[str, Any] = {}
    selected_bits: int | None = None
    float_rows = {
        row["dataset"]: row
        for row in _subset(
            rows,
            estimator_name=selected_estimator,
            bit_mode="float",
            matcher="l1",
        )
    }
    for bit_mode in BIT_MODES[1:]:
        quant_rows = _subset(
            rows,
            estimator_name=selected_estimator,
            bit_mode=bit_mode,
            matcher="l1",
        )
        family_ok = True
        margin_ok = True
        for family in ("hj", "mearec"):
            q_family = [row for row in quant_rows if row["family"] == family]
            f_family = [
                row for row in float_rows.values() if row["family"] == family
            ]
            family_ok = family_ok and (
                _event_accuracy(q_family) >= _event_accuracy(f_family) - 0.005
            )
            margin_ok = margin_ok and (
                _event_mean(q_family, "positive_margin_fraction")
                >= _event_mean(f_family, "positive_margin_fraction") - 0.005
            )
        record_ok = all(
            row["assignment_accuracy"]
            >= float_rows[row["dataset"]]["assignment_accuracy"] - 0.01
            for row in quant_rows
        )
        passes = bool(family_ok and margin_ok and record_ok)
        bit_audit[bit_mode] = {
            "family_accuracy_safeguard": bool(family_ok),
            "family_margin_safeguard": bool(margin_ok),
            "per_record_safeguard": bool(record_ok),
            "passes": passes,
        }
        if passes and selected_bits is None:
            selected_bits = int(bit_mode)

    bit_gate_pass = selected_bits is not None
    selection = {
        "estimator_gate_pass": bool(estimator_gate_pass),
        "selected_estimator": selected_estimator,
        "estimator_audit": estimator_audit,
        "bit_gate_pass": bool(bit_gate_pass),
        "selected_bits": selected_bits,
        "bit_audit": bit_audit,
        "advance_to_online_phase": bool(estimator_gate_pass and bit_gate_pass),
        "confirmation_authorized": bool(estimator_gate_pass and bit_gate_pass),
    }
    return {"grids": grids, "selection": selection, "n_rows": len(rows)}


def summarize_confirmation(
    rows: list[dict[str, Any]],
    *,
    selected_estimator: str = FROZEN_ESTIMATOR,
    selected_bits: int = FROZEN_BITS,
) -> dict[str, Any]:
    """Apply the frozen independent-confirmation gates without reselection."""
    selected_bit_mode = str(int(selected_bits))
    raw_l1_rows = _subset(
        rows, estimator_name="raw_range", bit_mode="float", matcher="l1"
    )
    selected_float_l1_rows = _subset(
        rows,
        estimator_name=selected_estimator,
        bit_mode="float",
        matcher="l1",
    )
    selected_quant_l1_rows = _subset(
        rows,
        estimator_name=selected_estimator,
        bit_mode=selected_bit_mode,
        matcher="l1",
    )
    if not raw_l1_rows or not selected_float_l1_rows or not selected_quant_l1_rows:
        raise ValueError("confirmation rows are incomplete")

    raw_by_dataset = {row["dataset"]: row for row in raw_l1_rows}
    float_by_dataset = {row["dataset"]: row for row in selected_float_l1_rows}
    quant_by_dataset = {row["dataset"]: row for row in selected_quant_l1_rows}
    if not (
        raw_by_dataset.keys() == float_by_dataset.keys() == quant_by_dataset.keys()
    ):
        raise ValueError("confirmation controls do not share recording identity")

    estimator_family: dict[str, Any] = {}
    quant_family: dict[str, Any] = {}
    family_l1_ok = True
    family_cosine_ok = True
    quant_accuracy_ok = True
    quant_margin_ok = True
    for family in ("hj", "mearec"):
        raw_family = [row for row in raw_l1_rows if row["family"] == family]
        float_family = [
            row for row in selected_float_l1_rows if row["family"] == family
        ]
        quant_rows = [
            row for row in selected_quant_l1_rows if row["family"] == family
        ]
        raw_cos = _subset(
            rows,
            estimator_name="raw_range",
            bit_mode="float",
            matcher="cosine",
            family=family,
        )
        selected_cos = _subset(
            rows,
            estimator_name=selected_estimator,
            bit_mode="float",
            matcher="cosine",
            family=family,
        )
        raw_accuracy = _event_accuracy(raw_family)
        float_accuracy = _event_accuracy(float_family)
        quant_accuracy = _event_accuracy(quant_rows)
        float_margin = _event_mean(float_family, "positive_margin_fraction")
        quant_margin = _event_mean(quant_rows, "positive_margin_fraction")
        raw_cosine_accuracy = _event_accuracy(raw_cos)
        selected_cosine_accuracy = _event_accuracy(selected_cos)

        this_l1_ok = float_accuracy >= raw_accuracy - 0.005
        this_cosine_ok = selected_cosine_accuracy >= raw_cosine_accuracy - 0.005
        this_quant_accuracy_ok = quant_accuracy >= float_accuracy - 0.005
        this_quant_margin_ok = quant_margin >= float_margin - 0.005
        family_l1_ok = family_l1_ok and this_l1_ok
        family_cosine_ok = family_cosine_ok and this_cosine_ok
        quant_accuracy_ok = quant_accuracy_ok and this_quant_accuracy_ok
        quant_margin_ok = quant_margin_ok and this_quant_margin_ok
        estimator_family[family] = {
            "raw_float_l1_accuracy": raw_accuracy,
            "selected_float_l1_accuracy": float_accuracy,
            "selected_minus_raw_accuracy": float_accuracy - raw_accuracy,
            "raw_float_cosine_accuracy": raw_cosine_accuracy,
            "selected_float_cosine_accuracy": selected_cosine_accuracy,
            "selected_minus_raw_cosine_accuracy": (
                selected_cosine_accuracy - raw_cosine_accuracy
            ),
            "l1_safeguard_pass": bool(this_l1_ok),
            "cosine_safeguard_pass": bool(this_cosine_ok),
        }
        quant_family[family] = {
            "selected_float_l1_accuracy": float_accuracy,
            "selected_quant_l1_accuracy": quant_accuracy,
            "quant_minus_float_accuracy": quant_accuracy - float_accuracy,
            "selected_float_positive_margin_fraction": float_margin,
            "selected_quant_positive_margin_fraction": quant_margin,
            "quant_minus_float_positive_margin_fraction": quant_margin - float_margin,
            "accuracy_safeguard_pass": bool(this_quant_accuracy_ok),
            "margin_safeguard_pass": bool(this_quant_margin_ok),
        }

    estimator_record_ok = all(
        row["assignment_accuracy"]
        >= raw_by_dataset[row["dataset"]]["assignment_accuracy"] - 0.01
        for row in selected_float_l1_rows
    )
    quant_record_ok = all(
        row["assignment_accuracy"]
        >= float_by_dataset[row["dataset"]]["assignment_accuracy"] - 0.01
        for row in selected_quant_l1_rows
    )

    raw_drift = [
        row
        for row in raw_l1_rows
        if row["family"] == "hj" and row["condition"] == "drift"
    ]
    selected_drift = [
        row
        for row in selected_float_l1_rows
        if row["family"] == "hj" and row["condition"] == "drift"
    ]
    if len(raw_drift) != 5 or len(selected_drift) != 5:
        raise ValueError("confirmation requires exactly five held-out HJ drift scenes")
    raw_drift_accuracy = _event_accuracy(raw_drift)
    selected_drift_accuracy = _event_accuracy(selected_drift)
    raw_drift_distance = _event_mean(raw_drift, "median_true_value_normalized")
    selected_drift_distance = _event_mean(
        selected_drift, "median_true_value_normalized"
    )
    raw_drift_margin = _event_mean(raw_drift, "positive_margin_fraction")
    selected_drift_margin = _event_mean(
        selected_drift, "positive_margin_fraction"
    )
    drift_accuracy_gain = selected_drift_accuracy >= raw_drift_accuracy + 0.005
    drift_distance_gain = (
        selected_drift_distance <= 0.98 * raw_drift_distance
        and selected_drift_margin >= raw_drift_margin - 1e-12
    )
    drift_ok = bool(drift_accuracy_gain or drift_distance_gain)

    estimator_gate_pass = bool(
        family_l1_ok and family_cosine_ok and estimator_record_ok and drift_ok
    )
    quant_gate_pass = bool(quant_accuracy_ok and quant_margin_ok and quant_record_ok)
    per_record = {
        dataset: {
            "family": raw_by_dataset[dataset]["family"],
            "condition": raw_by_dataset[dataset]["condition"],
            "raw_float_accuracy": raw_by_dataset[dataset]["assignment_accuracy"],
            "selected_float_accuracy": float_by_dataset[dataset]["assignment_accuracy"],
            "selected_quant_accuracy": quant_by_dataset[dataset]["assignment_accuracy"],
            "selected_float_minus_raw": (
                float_by_dataset[dataset]["assignment_accuracy"]
                - raw_by_dataset[dataset]["assignment_accuracy"]
            ),
            "selected_quant_minus_float": (
                quant_by_dataset[dataset]["assignment_accuracy"]
                - float_by_dataset[dataset]["assignment_accuracy"]
            ),
        }
        for dataset in sorted(raw_by_dataset)
    }
    return {
        "frozen_selection": {
            "selected_estimator": selected_estimator,
            "selected_bits": int(selected_bits),
            "matcher": "l1",
            "shift_radius": 0,
        },
        "estimator_family_audit": estimator_family,
        "estimator_per_record_safeguard": bool(estimator_record_ok),
        "hj_drift_mechanism": {
            "raw_float_accuracy": raw_drift_accuracy,
            "selected_float_accuracy": selected_drift_accuracy,
            "selected_minus_raw_accuracy": selected_drift_accuracy - raw_drift_accuracy,
            "raw_normalized_true_l1": raw_drift_distance,
            "selected_normalized_true_l1": selected_drift_distance,
            "relative_true_l1_change": (
                selected_drift_distance / raw_drift_distance - 1.0
            ),
            "raw_positive_margin_fraction": raw_drift_margin,
            "selected_positive_margin_fraction": selected_drift_margin,
            "accuracy_gain_pass": bool(drift_accuracy_gain),
            "distance_gain_pass": bool(drift_distance_gain),
            "passes": bool(drift_ok),
        },
        "estimator_gate_pass": estimator_gate_pass,
        "quantization_family_audit": quant_family,
        "quantization_per_record_safeguard": bool(quant_record_ok),
        "quantization_gate_pass": quant_gate_pass,
        "confirmation_pass": bool(estimator_gate_pass and quant_gate_pass),
        "per_record": per_record,
        "n_rows": len(rows),
        "n_recordings": len(raw_by_dataset),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Robust P2P estimator x 4--8-bit template pilot"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="run the frozen four-recording development pilot",
    )
    mode.add_argument(
        "--confirmation",
        action="store_true",
        help="run the frozen 12-recording independent confirmation",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.k != 7:
        parser.error("the frozen protocol requires --k 7")
    if args.duration != 60.0:
        parser.error("the frozen protocol requires --duration 60")

    output_dir = Path(args.output_dir)
    if args.confirmation:
        selection_path = output_dir / "robust_p2p_template_pilot_selection.json"
        if not selection_path.exists():
            parser.error(f"missing frozen Phase A selection: {selection_path}")
        selection = json.loads(selection_path.read_text())
        expected = {
            "selected_estimator": FROZEN_ESTIMATOR,
            "selected_bits": FROZEN_BITS,
            "confirmation_authorized": True,
        }
        actual = {key: selection.get(key) for key in expected}
        if actual != expected:
            parser.error(
                f"frozen Phase A selection mismatch: expected {expected}, got {actual}"
            )

    rows: list[dict[str, Any]] = []
    datasets = (
        _pilot_datasets(args.duration, Path(args.mearec_npz_dir))
        if args.pilot
        else _confirmation_datasets(args.duration, Path(args.mearec_npz_dir))
    )
    estimator_grid = (
        ESTIMATOR_GRID
        if args.pilot
        else tuple(
            config
            for config in ESTIMATOR_GRID
            if config[0] in ("raw_range", FROZEN_ESTIMATOR)
        )
    )
    for family, dataset in datasets:
        if not args.quiet:
            print(f"prepare {dataset.name}", flush=True)
        context = prepare_event_context(dataset, k_neighbors=args.k)
        for estimator_name, estimator, average_points, hardware_eligible in estimator_grid:
            if not args.quiet:
                print(f"  estimator={estimator_name}", flush=True)
            footprint = extract_estimator_footprint(
                context,
                estimator=estimator,
                average_points=average_points,
            )
            bit_modes = (
                BIT_MODES
                if args.pilot
                else (("float",) if estimator_name == "raw_range" else ("float", str(FROZEN_BITS)))
            )
            for bit_mode in bit_modes:
                for matcher in MATCHERS:
                    rows.append(
                        evaluate_rows(
                            footprint,
                            context["labels"],
                            context["times"],
                            dataset_name=dataset.name,
                            family=family,
                            estimator_name=estimator_name,
                            estimator=estimator,
                            average_points=average_points,
                            hardware_eligible=hardware_eligible,
                            bit_mode=bit_mode,
                            matcher=matcher,
                        )
                    )

    stem = "robust_p2p_template_pilot" if args.pilot else "robust_p2p_template_confirmation"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    _write_csv(csv_path, rows)
    result = (
        summarize_and_select(rows)
        if args.pilot
        else summarize_confirmation(rows)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.pilot:
        selection_path = output_dir / "robust_p2p_template_pilot_selection.json"
        selection_path.write_text(
            json.dumps(result["selection"], indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
    if not args.quiet:
        print(json.dumps(result.get("selection", result), indent=2, sort_keys=True))
        print(f"wrote {csv_path}")
        print(f"wrote {summary_path}")
        if args.pilot:
            print(f"wrote {selection_path}")


if __name__ == "__main__":
    main()
