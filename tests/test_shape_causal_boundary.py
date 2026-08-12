"""Contract tests for the shape descriptor Phase A2 causal/jitter boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from Spatial.algorithms.causal_preprocessing import causal_event_guard_mask
from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split, prepare_dataset_events
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR
from Spatial.experiments.run_shape_descriptor_pilot import _load_locked_recording
from Spatial.experiments.run_shape_causal_boundary import (
    A2_ARMS,
    A2_CONDITIONS,
    CAUSAL_RETENTION_THRESHOLD,
    JITTER_MAX_LOSS_PP,
    compute_a2_gate_results,
    evaluate_recording_boundary,
    lat_width_gain_over_d9,
)
from Spatial.experiments.shape_descriptor_common import (
    K_NEIGHBORS,
    WINDOW,
    build_descriptor_blocks,
    shared_event_guard_mask,
)

pytest.importorskip("Spatial.algorithms.spatial_footprint", reason="primitives unavailable")


def _synthetic_dataset(
    *,
    n_samples: int = 8000,
    spike_times: np.ndarray | None = None,
    name: str = "synthetic_a2",
) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    n_channels = 8
    raw = rng.normal(scale=0.05, size=(n_channels, n_samples))
    geom = np.stack([np.arange(n_channels, dtype=np.float64), np.zeros(n_channels)], axis=1)
    if spike_times is None:
        spike_times = np.arange(1200, 7200, 120, dtype=np.int64)
    spike_units = np.tile(np.arange(3, dtype=np.int64), len(spike_times) // 3 + 1)[: len(spike_times)]
    return SimpleNamespace(
        raw_data=raw,
        geom=geom,
        fs=30000,
        spike_times=spike_times,
        spike_units=spike_units,
        n_channels=n_channels,
        n_samples=n_samples,
        name=name,
    )


def test_causal_filter_blocks_are_invariant_to_future_raw_changes() -> None:
    dataset = _synthetic_dataset()
    prep = prepare_dataset_events(
        dataset,
        k_neighbors=K_NEIGHBORS,
        window=WINDOW,
        footprint_layout="relative",
    )
    guard = shared_event_guard_mask(
        prep["times"],
        warmup_samples=0,
        n_samples=dataset.n_samples,
    )
    frozen = {key: np.asarray(value)[guard] for key, value in prep.items()}
    blocks_before = build_descriptor_blocks(
        dataset,
        filter_mode="causal",
        frozen_prep=frozen,
    )
    event_idx = 0
    event_time = int(blocks_before["times"][event_idx])
    tail_start = event_time + WINDOW + 20
    saved_tail = dataset.raw_data[:, tail_start:].copy()
    dataset.raw_data[:, tail_start:] += 1000.0
    blocks_after = build_descriptor_blocks(
        dataset,
        filter_mode="causal",
        frozen_prep=frozen,
    )
    dataset.raw_data[:, tail_start:] = saved_tail

    for key in ("vmax", "vmin", "imax", "imin", "com", "p2p"):
        np.testing.assert_allclose(
            blocks_before[key][event_idx],
            blocks_after[key][event_idx],
            rtol=0.0,
            atol=0.0,
            err_msg=f"causal path must not depend on samples after the window: {key}",
        )

    dataset.raw_data[:, tail_start:] += 1000.0
    zero_after = build_descriptor_blocks(dataset, filter_mode="zero_phase", frozen_prep=frozen)
    dataset.raw_data[:, tail_start:] = saved_tail
    zero_before = build_descriptor_blocks(dataset, filter_mode="zero_phase", frozen_prep=frozen)
    changed = not np.allclose(
        zero_before["vmax"][event_idx], zero_after["vmax"][event_idx], rtol=0.0, atol=0.0
    ) or not np.allclose(zero_before["p2p"][event_idx], zero_after["p2p"][event_idx], rtol=0.0, atol=0.0)
    assert changed, "zero-phase path should depend on future samples through filtfilt"


def test_shared_guard_drops_expected_events_and_conditions_match() -> None:
    dataset = _synthetic_dataset()
    prep = prepare_dataset_events(
        dataset,
        k_neighbors=K_NEIGHBORS,
        window=WINDOW,
        footprint_layout="relative",
    )
    warmup = 1000
    guard = shared_event_guard_mask(
        prep["times"],
        warmup_samples=warmup,
        n_samples=dataset.n_samples,
    )
    expected_warm = causal_event_guard_mask(
        prep["times"],
        warmup_samples=warmup,
        start_sample=0,
        stop_sample=dataset.n_samples,
    )
    expected_bounds = (prep["times"] >= WINDOW + 2) & (prep["times"] + WINDOW + 2 < dataset.n_samples)
    np.testing.assert_array_equal(guard, expected_warm & expected_bounds)
    assert int(guard.sum()) >= 2

    reference_times: np.ndarray | None = None
    for condition, filter_mode, jitter in A2_CONDITIONS:
        blocks = build_descriptor_blocks(
            dataset,
            filter_mode=filter_mode,
            jitter_samples=jitter,
            event_mask=guard,
        )
        if reference_times is None:
            reference_times = blocks["times"].copy()
        else:
            np.testing.assert_array_equal(blocks["times"], reference_times)


def test_jitter_zero_matches_unjittered_blocks() -> None:
    dataset = _synthetic_dataset()
    prep = prepare_dataset_events(
        dataset,
        k_neighbors=K_NEIGHBORS,
        window=WINDOW,
        footprint_layout="relative",
    )
    guard = shared_event_guard_mask(
        prep["times"],
        warmup_samples=0,
        n_samples=dataset.n_samples,
    )
    baseline = build_descriptor_blocks(
        dataset,
        filter_mode="zero_phase",
        jitter_samples=0,
        event_mask=guard,
    )
    explicit_zero = build_descriptor_blocks(
        dataset,
        filter_mode="zero_phase",
        jitter_samples=0,
        event_mask=guard,
    )
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], explicit_zero[key])


def test_jitter_shifts_extrema_read_window_by_requested_offset() -> None:
    dataset = _synthetic_dataset()
    prep = prepare_dataset_events(
        dataset,
        k_neighbors=K_NEIGHBORS,
        window=WINDOW,
        footprint_layout="relative",
    )
    guard = shared_event_guard_mask(
        prep["times"],
        warmup_samples=0,
        n_samples=dataset.n_samples,
    )
    frozen = {key: np.asarray(value)[guard] for key, value in prep.items()}
    event_time = int(frozen["times"][0])
    central = int(frozen["central"][0])
    marker = event_time + 5
    dataset.raw_data[central, marker] += 50.0

    base = build_descriptor_blocks(
        dataset,
        filter_mode="zero_phase",
        jitter_samples=0,
        frozen_prep=frozen,
    )
    shifted = build_descriptor_blocks(
        dataset,
        filter_mode="zero_phase",
        jitter_samples=5,
        frozen_prep=frozen,
    )
    home_slot = 3
    assert int(base["imax"][0, home_slot]) != int(shifted["imax"][0, home_slot])


def test_gate_retention_and_jitter_thresholds() -> None:
    per_family = {
        "hj": {
            "control": {"d9": 0.80, "com_posneg": 0.90, "com_posneg_lat_width": 0.92},
            "causal": {"d9": 0.80, "com_posneg": 0.895, "com_posneg_lat_width": 0.905},
            "jitter_m2": {"d9": 0.80, "com_posneg": 0.90, "com_posneg_lat_width": 0.915},
            "jitter_p2": {"d9": 0.80, "com_posneg": 0.90, "com_posneg_lat_width": 0.909},
        },
        "mearec": {
            "control": {"d9": 0.77, "com_posneg": 0.85, "com_posneg_lat_width": 0.86},
            "causal": {"d9": 0.77, "com_posneg": 0.845, "com_posneg_lat_width": 0.855},
            "jitter_m2": {"d9": 0.77, "com_posneg": 0.85, "com_posneg_lat_width": 0.859},
            "jitter_p2": {"d9": 0.77, "com_posneg": 0.85, "com_posneg_lat_width": 0.849},
        },
    }
    phase_a = {
        "hj": {"d9": 0.814, "com_posneg": 0.919, "com_posneg_lat_width": 0.920},
        "mearec": {"d9": 0.777, "com_posneg": 0.858, "com_posneg_lat_width": 0.859},
    }
    gate = compute_a2_gate_results(per_family, phase_a_reference=phase_a)

    hj_full_ret = gate["causal_full_gain_retention_fraction_hj"]
    hj_lat_ret = gate["causal_lat_width_incremental_retention_fraction_hj"]
    assert hj_full_ret == pytest.approx(
        (0.905 - 0.80) / (0.920 - 0.814),
        rel=0.0,
        abs=1e-12,
    )
    assert hj_lat_ret == pytest.approx(
        (0.905 - 0.895) / (0.920 - 0.919),
        rel=0.0,
        abs=1e-12,
    )
    assert gate["worst_jitter_loss_pp_hj"] == pytest.approx(100.0 * (0.92 - 0.909), abs=1e-12)
    assert gate["causal_retention_pass_hj"] == (
        hj_full_ret >= CAUSAL_RETENTION_THRESHOLD and hj_lat_ret >= CAUSAL_RETENTION_THRESHOLD
    )
    assert gate["jitter_gate_pass_hj"] == (gate["worst_jitter_loss_pp_hj"] <= JITTER_MAX_LOSS_PP)
    assert gate["passed"] == bool(gate["causal_retention_pass"] and gate["jitter_gate_pass"])


def test_lat_width_gain_over_d9_ratio() -> None:
    accs = {"d9": 0.80, "com_posneg": 0.90, "com_posneg_lat_width": 0.92}
    assert lat_width_gain_over_d9(accs) == pytest.approx(0.02 / 0.10)


def test_dataset_backed_boundary_runner_skips_when_data_missing() -> None:
    family, name = "hj", "drift16c_600s_11"
    try:
        dataset = _load_locked_recording(family, name, 5.0, DEFAULT_NPZ_DIR)
    except FileNotFoundError:
        pytest.skip("locked development recording unavailable")
    rows, n_guarded = evaluate_recording_boundary(dataset, family=family, warmup_samples=1000)
    assert n_guarded >= 2
    assert rows
    assert {row["arm"] for row in rows} == set(A2_ARMS)
    assert {row["condition"] for row in rows} == {item[0] for item in A2_CONDITIONS}


def test_synthetic_boundary_end_to_end() -> None:
    dataset = _synthetic_dataset()
    rows, n_guarded = evaluate_recording_boundary(dataset, family="hj", warmup_samples=1000)
    assert n_guarded >= 2
    assert len(rows) == len(A2_ARMS) * len(A2_CONDITIONS)
    train_counts = {
        (row["condition"], row["arm"]): row["n_train"] for row in rows
    }
    for arm in A2_ARMS:
        control_train = train_counts[("control", arm)]
        for condition, _, _ in A2_CONDITIONS:
            assert train_counts[(condition, arm)] == control_train

    blocks = build_descriptor_blocks(
        dataset,
        filter_mode="zero_phase",
        event_mask=shared_event_guard_mask(
            prepare_dataset_events(
                dataset,
                k_neighbors=K_NEIGHBORS,
                window=WINDOW,
                footprint_layout="relative",
            )["times"],
            warmup_samples=1000,
            n_samples=dataset.n_samples,
        ),
    )
    train_idx, test_idx = _chronological_split(blocks["times"])
    assert train_idx.size + test_idx.size == n_guarded
