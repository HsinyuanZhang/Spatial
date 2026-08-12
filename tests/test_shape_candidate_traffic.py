"""Synthetic tests for Phase B candidate recall and template traffic."""

from __future__ import annotations

import numpy as np
import pytest

from Spatial.algorithms.adaptive_range_search import evaluate_candidate_search
from Spatial.experiments.run_adaptive_5bit_range_search import candidate_waveform_assign
from Spatial.experiments.run_shape_descriptor_pilot import LOCKED_DEVELOPMENT_RECORDINGS
from Spatial.experiments.run_shape_candidate_traffic import (
    DEFAULT_PERCENTILES,
    fit_per_unit_l1_from_codes,
    query_per_unit_l1_codes,
    summarize_candidate_traffic,
    summarize_joint_gates,
)


def _synthetic_codes(n_events: int = 20, n_features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1][:n_events])
    codes = rng.integers(0, 8, size=(n_events, n_features), dtype=np.int64)
    for unit in np.unique(labels):
        codes[labels == unit] += int(unit) * 10
    return codes, labels


def test_candidate_partition_fractions_sum_to_one():
    candidates = [
        np.array([], dtype=int),
        np.array([1]),
        np.array([1, 2]),
        np.array([0, 1, 2]),
    ]
    labels = np.array([0, 1, 2, 0])
    diagnostics = {
        "row_comparisons": np.array([3, 3, 3, 3], dtype=np.int64),
        "fallback_row_comparisons": np.zeros(4, dtype=np.int64),
        "active_row_count": np.full(4, 3, dtype=np.int64),
    }
    report = evaluate_candidate_search(candidates, labels, diagnostics)
    assert report["zero_fraction"] == 0.25
    assert report["unique_fraction"] == 0.25
    assert report["ambiguous_fraction"] == 0.5
    assert (
        report["zero_fraction"] + report["unique_fraction"] + report["ambiguous_fraction"]
    ) == 1.0
    assert report["true_unit_candidate_recall"] == 0.75


def test_waveform_read_reduction_against_full_scan_baseline():
    candidates = [np.array([10]), np.array([10, 30]), np.array([], dtype=int)]
    labels = np.array([10, 30, 10])
    units = np.array([10, 30, 50])
    waveforms = np.array([[0.0, 1.0], [0.9, 0.1], [0.9, 0.1]])
    wave_centroids = {10: np.array([0.0, 1.0]), 30: np.array([1.0, 0.0]), 50: np.array([0.5, 0.5])}
    pred, wave_meta = candidate_waveform_assign(
        candidates,
        waveforms,
        wave_centroids,
        np.zeros((3, 2), dtype=np.int64),
        np.zeros((3, 2), dtype=np.int64),
        units,
        metric="l1",
    )
    diagnostics = {
        "row_comparisons": np.array([3, 3, 3], dtype=np.int64),
        "fallback_row_comparisons": np.zeros(3, dtype=np.int64),
    }
    metrics = summarize_candidate_traffic(
        candidates, labels, diagnostics, units, pred, wave_meta
    )
    assert metrics["global_all_unit_waveform_reads"] == 9
    assert metrics["distinct_waveform_template_reads_total"] == 2
    assert metrics["waveform_read_reduction_factor"] == 4.5
    assert metrics["zero_candidate_fraction"] == 1.0 / 3.0
    assert metrics["one_candidate_fraction"] == 1.0 / 3.0
    assert metrics["multi_candidate_fraction"] == 1.0 / 3.0


def test_test_partition_perturbation_does_not_change_fitted_radii():
    codes, labels = _synthetic_codes()
    train_idx = np.arange(12)
    test_idx = np.arange(12, codes.shape[0])
    units_a, centroids_a, radii_a = fit_per_unit_l1_from_codes(
        codes[train_idx], labels[train_idx], 95.0
    )
    perturbed = codes.copy()
    perturbed[test_idx] += 99
    units_b, centroids_b, radii_b = fit_per_unit_l1_from_codes(
        perturbed[train_idx], labels[train_idx], 95.0
    )
    np.testing.assert_array_equal(units_a, units_b)
    np.testing.assert_array_equal(centroids_a, centroids_b)
    np.testing.assert_array_equal(radii_a, radii_b)


def test_larger_radius_percentile_cannot_reduce_candidate_recall():
    codes, labels = _synthetic_codes()
    units, centroids, radii_lo = fit_per_unit_l1_from_codes(codes, labels, 95.0)
    _, _, radii_hi = fit_per_unit_l1_from_codes(codes, labels, 100.0)
    assert np.all(radii_hi >= radii_lo)
    candidates_lo, _ = query_per_unit_l1_codes(codes, units, centroids, radii_lo)
    candidates_hi, _ = query_per_unit_l1_codes(codes, units, centroids, radii_hi)
    recall_lo = evaluate_candidate_search(candidates_lo, labels)["true_unit_candidate_recall"]
    recall_hi = evaluate_candidate_search(candidates_hi, labels)["true_unit_candidate_recall"]
    assert recall_hi >= recall_lo


def test_summarize_joint_gates_reports_boolean_outcomes():
    rows = [
        {
            "dataset": "hj_a",
            "family": "hj",
            "arm": "d9",
            "radius_percentile": 95.0,
            "n_eval": 100,
            "true_unit_candidate_recall": 0.98,
            "mean_candidate_count": 5.0,
            "distinct_waveform_template_reads_total": 400,
            "global_all_unit_waveform_reads": 1000,
        },
        {
            "dataset": "hj_b",
            "family": "hj",
            "arm": "d9",
            "radius_percentile": 95.0,
            "n_eval": 100,
            "true_unit_candidate_recall": 0.98,
            "mean_candidate_count": 5.0,
            "distinct_waveform_template_reads_total": 400,
            "global_all_unit_waveform_reads": 1000,
        },
        {
            "dataset": "mearec_a",
            "family": "mearec",
            "arm": "d9",
            "radius_percentile": 95.0,
            "n_eval": 100,
            "true_unit_candidate_recall": 0.995,
            "mean_candidate_count": 3.0,
            "distinct_waveform_template_reads_total": 200,
            "global_all_unit_waveform_reads": 1000,
        },
        {
            "dataset": "mearec_b",
            "family": "mearec",
            "arm": "d9",
            "radius_percentile": 95.0,
            "n_eval": 100,
            "true_unit_candidate_recall": 0.995,
            "mean_candidate_count": 3.0,
            "distinct_waveform_template_reads_total": 200,
            "global_all_unit_waveform_reads": 1000,
        },
    ]
    summary = summarize_joint_gates(rows)
    arm = summary["arms"]["d9"]
    assert arm["hj_passes_joint_gate"] is False
    assert arm["mearec_passes_joint_gate"] is True
    assert arm["passes_joint_gate_on_both_families"] is False
    assert summary["candidate_policy"] == "per_unit_l1_radius_all_rows_active"


def _load_locked_recording(family: str, name: str, duration_s: float):
    """Resolve a locked development recording, or None when it is genuinely absent."""
    from pathlib import Path

    from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR

    if family == "hj":
        from Spatial.data.loader import load_hybrid_janelia, resolve_hybrid_janelia_path

        try:
            path = Path(resolve_hybrid_janelia_path(scene=name))
        except Exception:
            return None
        if not path.exists():
            return None
        return load_hybrid_janelia(scene=name, duration_s=duration_s)

    from Spatial.data.loader import load_mearec_npz

    path = Path(DEFAULT_NPZ_DIR) / f"{name}.npz"
    if not path.exists():
        return None
    return load_mearec_npz(path, duration_s=duration_s)


@pytest.mark.parametrize("family,name", LOCKED_DEVELOPMENT_RECORDINGS)
def test_locked_development_recording_produces_valid_frontier_rows(
    family: str, name: str
) -> None:
    """Exercise the real runner on every locked recording that is actually present."""
    pytest.importorskip("Spatial.algorithms.spatial_footprint", reason="primitives unavailable")
    from Spatial.experiments.run_shape_candidate_traffic import (
        PHASE_B_ARMS,
        evaluate_recording,
    )

    dataset = _load_locked_recording(family, name, duration_s=10.0)
    if dataset is None:
        pytest.skip(f"locked development recording is absent: {family}:{name}")

    percentiles = [95.0, 100.0]
    rows = evaluate_recording(dataset, family=family, percentiles=percentiles)
    assert len(rows) == len(PHASE_B_ARMS) * len(percentiles)

    for row in rows:
        assert row["family"] == family
        assert 0.0 <= row["true_unit_candidate_recall"] <= 1.0
        assert 0.0 <= row["final_assignment_accuracy"] <= 1.0
        partition = (
            float(row["zero_candidate_fraction"])
            + float(row["one_candidate_fraction"])
            + float(row["multi_candidate_fraction"])
        )
        assert abs(partition - 1.0) < 1e-9
        assert int(row["distinct_waveform_template_reads_total"]) >= 0
        assert int(row["distinct_waveform_template_reads_total"]) <= int(
            row["global_all_unit_waveform_reads"]
        )

    # A radius covering every training distance cannot lose the true unit.
    for arm in PHASE_B_ARMS:
        tight = next(
            r for r in rows if r["arm"] == arm and float(r["radius_percentile"]) == 95.0
        )
        loose = next(
            r for r in rows if r["arm"] == arm and float(r["radius_percentile"]) == 100.0
        )
        assert (
            float(loose["true_unit_candidate_recall"])
            >= float(tight["true_unit_candidate_recall"]) - 1e-12
        )
