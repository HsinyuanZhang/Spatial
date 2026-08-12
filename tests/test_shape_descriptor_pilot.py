"""Contract tests for the shape-orthogonal descriptor Phase A pilot."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split
from Spatial.experiments.shape_descriptor_common import CSV_FIELDNAMES
from Spatial.experiments.run_shape_descriptor_pilot import evaluate_recording
from Spatial.experiments.shape_descriptor_common import (
    AMP_GATE_CHOICES,
    ARM_IDS,
    CANDIDATE_ARM,
    CONTROL_ARM,
    K_NEIGHBORS,
    LAT_BIT_CHOICES,
    POSNEG_BIT_CHOICES,
    arm_block_layout,
    assemble_arm,
    build_rand_block,
    descriptor_ledger,
    evaluate_selection,
    evaluation_cache_key,
    fit_integer_templates,
    quantize_fixed_unit_interval,
)

pytest.importorskip("Spatial.algorithms.spatial_footprint", reason="primitives unavailable")


def _tiny_blocks(n_events: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    k = K_NEIGHBORS
    com_float = rng.random((n_events, 2))
    p2p_float = rng.random((n_events, k))
    return {
        "times": np.arange(n_events, dtype=np.int64) * 100,
        "labels": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int64)[:n_events],
        "central": np.full(n_events, 4, dtype=np.int64),
        "com": quantize_fixed_unit_interval(com_float, 5),
        "p2p": quantize_fixed_unit_interval(p2p_float, 5),
        "vmax": rng.random((n_events, k)),
        "vmin": -rng.random((n_events, k)),
        "imax": rng.integers(0, 30, size=(n_events, k)),
        "imin": rng.integers(0, 30, size=(n_events, k)),
        "neighbor_ids": np.tile(np.arange(k, dtype=np.int64), (n_events, 1)),
    }


@pytest.mark.parametrize("posneg_bits", POSNEG_BIT_CHOICES)
@pytest.mark.parametrize("lat_bits", LAT_BIT_CHOICES)
@pytest.mark.parametrize("arm", ARM_IDS)
def test_arm_layout_coordinate_counts(arm: str, posneg_bits: int, lat_bits: int) -> None:
    layout = arm_block_layout(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    n_coords = sum(n_cols for _, n_cols, _ in layout)
    ledger = descriptor_ledger(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    assert ledger["n_coords"] == n_coords
    if arm == "d9" and posneg_bits == 5 and lat_bits == 4:
        assert n_coords == 9
    if arm == "com_posneg_lat_width" and posneg_bits == 5 and lat_bits == 4:
        assert n_coords == 23
    blocks = _tiny_blocks()
    assembled = assemble_arm(
        blocks,
        arm,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=0.25,
        recording_name="synthetic",
    )
    assert assembled.shape == (blocks["com"].shape[0], n_coords)


def test_arm_column_order_matches_layout() -> None:
    blocks = _tiny_blocks(4)
    layout = arm_block_layout("com_posneg_lat_width", posneg_bits=5, lat_bits=4)
    assembled = assemble_arm(
        blocks,
        "com_posneg_lat_width",
        posneg_bits=5,
        lat_bits=4,
        amp_gate=0.25,
        recording_name="synthetic",
    )
    offset = 0
    for block_name, n_cols, _ in layout:
        chunk = assembled[:, offset : offset + n_cols]
        if block_name == "COM":
            np.testing.assert_array_equal(chunk, blocks["com"])
        elif block_name == "POSNEG":
            from Spatial.algorithms.spatial_footprint import posneg_codes

            expected = posneg_codes(
                blocks["vmax"], blocks["vmin"], blocks["neighbor_ids"], n_bits=5
            )
            np.testing.assert_array_equal(chunk, expected)
        offset += n_cols


@pytest.mark.parametrize(
    ("arm", "posneg_bits", "lat_bits", "expected_row_bits", "expected_accum"),
    [
        ("d9", 5, 4, 45, 10),
        ("d9_lat", 5, 4, 69, 10),
        ("d9_lat", 5, 3, 63, 10),
        ("com_posneg_lat_width", 5, 4, 109, 11),
        ("com_posneg_lat_width", 4, 3, 89, 10),
    ],
)
def test_row_bits_lane_count_accum_bits(
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    expected_row_bits: int,
    expected_accum: int,
) -> None:
    ledger = descriptor_ledger(arm, posneg_bits=posneg_bits, lat_bits=lat_bits)
    assert ledger["row_bits"] == expected_row_bits
    assert ledger["lane_count"] == ledger["n_coords"]
    assert ledger["accum_bits"] == expected_accum


def test_rand_block_reproducible_and_recording_specific() -> None:
    a1 = build_rand_block(6, lat_bits=4, recording_name="rec_a")
    a2 = build_rand_block(6, lat_bits=4, recording_name="rec_a")
    b = build_rand_block(6, lat_bits=4, recording_name="rec_b")
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    assert a1.shape == (6, K_NEIGHBORS)


def test_rand_block_matches_lat_width_geometry() -> None:
    for lat_bits in LAT_BIT_CHOICES:
        rand = build_rand_block(3, lat_bits=lat_bits, recording_name="x")
        lat_layout = arm_block_layout("d9_lat", posneg_bits=5, lat_bits=lat_bits)
        width_layout = arm_block_layout("d9_width", posneg_bits=5, lat_bits=lat_bits)
        lat_cols = next(n for name, n, _ in lat_layout if name == "LAT")
        width_cols = next(n for name, n, _ in width_layout if name == "WIDTH")
        assert rand.shape[1] == lat_cols + width_cols


def test_synthetic_end_to_end_csv_schema() -> None:
    geom = np.stack([np.arange(8, dtype=np.float64), np.zeros(8)], axis=1)
    raw = np.random.default_rng(1).normal(size=(8, 5000))
    dataset = SimpleNamespace(
        raw_data=raw,
        geom=geom,
        fs=30000,
        spike_times=np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], dtype=np.int64),
        spike_units=np.array([0, 0, 1, 1, 0, 1, 2, 2, 0, 1], dtype=np.int64),
        n_channels=8,
        n_samples=5000,
        name="synthetic_tiny",
    )
    rows = evaluate_recording(dataset, family="hj")
    assert rows
    assert set(rows[0].keys()) == set(CSV_FIELDNAMES)
    for row in rows:
        assert set(row.keys()) == set(CSV_FIELDNAMES)


def test_templates_ignore_test_split_perturbation() -> None:
    blocks = _tiny_blocks(12)
    train_idx, test_idx = _chronological_split(blocks["times"])
    codes = assemble_arm(
        blocks,
        "d9",
        posneg_bits=5,
        lat_bits=4,
        amp_gate=0.25,
        recording_name="leakage",
    )
    _, templates_before = fit_integer_templates(codes[train_idx], blocks["labels"][train_idx])
    perturbed = codes.copy()
    perturbed[test_idx] = 0
    _, templates_after = fit_integer_templates(perturbed[train_idx], blocks["labels"][train_idx])
    np.testing.assert_array_equal(templates_before, templates_after)


def _synthetic_gate_rows() -> list[dict[str, object]]:
    recordings = ("hj_drift", "hj_static", "mearec_10", "mearec_20")
    rows: list[dict[str, object]] = []
    for dataset in recordings:
        family = "hj" if dataset.startswith("hj") else "mearec"
        for arm in (CONTROL_ARM, CANDIDATE_ARM, "d9_rand"):
            accuracy = {
                CONTROL_ARM: 0.80,
                CANDIDATE_ARM: 0.83,
                "d9_rand": 0.79,
            }[arm]
            margin = {
                CONTROL_ARM: 0.60,
                CANDIDATE_ARM: 0.62,
                "d9_rand": 0.59,
            }[arm]
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "scene_type": "drift" if "drift" in dataset else "static",
                    "mearec_n_units": 10 if family == "mearec" else -1,
                    "arm": arm,
                    "posneg_bits": 5,
                    "lat_bits": 4,
                    "amp_gate": 0.25,
                    "n_train": 100,
                    "n_test": 100,
                    "n_units": 3,
                    "n_coords": 23,
                    "row_bits": 109,
                    "lane_count": 23,
                    "accum_bits": 12,
                    "accuracy": accuracy,
                    "n_correct": int(accuracy * 100),
                    "worst_unit_accuracy": accuracy - 0.05,
                    "margin_mean": 1.0,
                    "positive_margin_fraction": margin,
                    "true_template_distance_norm": 0.1,
                    "hard_quartile_accuracy": accuracy - 0.02,
                    "nonconforming_slot_fraction": 0.0,
                    "posneg_reconstruction_max_error": 1,
                }
            )
    return rows


def test_gate_passes_on_qualifying_rows() -> None:
    selection = evaluate_selection(
        _synthetic_gate_rows(),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["passed"] is True
    assert selection["selected_arm"] == CANDIDATE_ARM
    assert selection["null_reason"] is None
    for key in (
        "condition_1_family_gain",
        "condition_2_no_recording_loss",
        "condition_3_positive_margin",
        "condition_4_reconstruction",
        "condition_5_rand_control_fails",
    ):
        assert selection["gate_results"][key] is True


def _mutate_gate_rows(mutator) -> list[dict[str, object]]:
    rows = _synthetic_gate_rows()
    mutator(rows)
    return rows


def test_gate_fails_condition_1() -> None:
    def mutate(rows):
        for row in rows:
            if row["arm"] == CANDIDATE_ARM and row["family"] == "hj":
                row["accuracy"] = 0.805

    selection = evaluate_selection(
        _mutate_gate_rows(mutate),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["passed"] is False
    assert selection["gate_results"]["condition_1_family_gain"] is False
    assert "condition_1_family_gain" in selection["null_reason"]


def test_gate_fails_condition_2() -> None:
    def mutate(rows):
        for row in rows:
            if row["arm"] == CANDIDATE_ARM and row["dataset"] == "hj_static":
                row["accuracy"] = 0.79

    selection = evaluate_selection(
        _mutate_gate_rows(mutate),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["gate_results"]["condition_2_no_recording_loss"] is False


def test_gate_fails_condition_3() -> None:
    def mutate(rows):
        for row in rows:
            if row["arm"] == CANDIDATE_ARM and row["family"] == "mearec":
                row["positive_margin_fraction"] = 0.55

    selection = evaluate_selection(
        _mutate_gate_rows(mutate),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["gate_results"]["condition_3_positive_margin"] is False


def test_gate_fails_condition_4() -> None:
    def mutate(rows):
        for row in rows:
            if row["arm"] == CANDIDATE_ARM:
                row["nonconforming_slot_fraction"] = 0.01

    selection = evaluate_selection(
        _mutate_gate_rows(mutate),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["gate_results"]["condition_4_reconstruction"] is False


def test_gate_fails_condition_5_rand_control() -> None:
    def mutate(rows):
        for row in rows:
            if row["arm"] == "d9_rand":
                row["accuracy"] = 0.83

    selection = evaluate_selection(
        _mutate_gate_rows(mutate),
        development_recordings=["hj_drift", "hj_static", "mearec_10", "mearec_20"],
    )
    assert selection["gate_results"]["condition_5_rand_control_fails"] is False


def test_evaluation_cache_key_deduplication_contract() -> None:
    assert evaluation_cache_key("d9", 4, 3, 0.125) == evaluation_cache_key("d9", 5, 4, 0.375)
    assert evaluation_cache_key("d9_lat", 4, 3, 0.125) != evaluation_cache_key("d9_lat", 4, 4, 0.125)
    assert evaluation_cache_key("com_posneg", 4, 3, 0.125) != evaluation_cache_key("com_posneg", 5, 3, 0.125)


def test_locked_hj_recording_smoke() -> None:
    """Guard on the loader's own resolution, not on a hardcoded absolute path."""
    from pathlib import Path

    from Spatial.data.loader import load_hybrid_janelia, resolve_hybrid_janelia_path

    try:
        resolved = Path(resolve_hybrid_janelia_path(scene="static16c_600s_11"))
    except Exception as error:
        pytest.skip(f"HJ development data unavailable: {error}")
    if not resolved.exists():
        pytest.skip(f"HJ development data unavailable: {resolved}")

    dataset = load_hybrid_janelia(scene="static16c_600s_11", duration_s=1.0)
    rows = evaluate_recording(dataset, family="hj")
    assert len(rows) == len(ARM_IDS) * len(POSNEG_BIT_CHOICES) * len(LAT_BIT_CHOICES) * len(AMP_GATE_CHOICES)
    assert set(rows[0].keys()) == set(CSV_FIELDNAMES)


def test_locked_mearec_recording_smoke() -> None:
    from pathlib import Path

    from Spatial.data.loader import load_mearec_npz
    from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR

    path = Path(DEFAULT_NPZ_DIR) / "rec_v1_units10_snr5_seed202601.npz"
    if not path.exists():
        pytest.skip(f"MEArec development data unavailable: {path}")

    dataset = load_mearec_npz(path, duration_s=1.0)
    rows = evaluate_recording(dataset, family="mearec")
    assert rows[0]["mearec_n_units"] == 10
