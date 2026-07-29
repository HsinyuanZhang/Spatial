"""Smoke tests for spatial CiM experiment helpers and gate evaluation."""

from __future__ import annotations

import numpy as np

from Spatial.experiments.run_spatial_wta_pilot import evaluate_gates
from Spatial.experiments.spatial_cim_common import classification_metrics


def test_classification_metrics_plain_equality():
    pred = np.array([1, 2, 1, 3])
    truth = np.array([1, 2, 2, 3])
    m = classification_metrics(pred, truth)
    assert m["n_correct"] == 3
    assert abs(m["accuracy"] - 0.75) < 1e-12


def test_evaluate_gates_from_integer_counts():
    rows = []
    for family, n, correct_b5, correct_float, correct_teacher in (
        ("hj", 100, 80, 81, 70),
        ("hj", 50, 40, 40, 35),
        ("mearec", 200, 160, 162, 140),
        ("mearec", 100, 80, 80, 70),
    ):
        for stage, config, correct in (
            ("A1_bit_depth", "uniform_theta_b5", correct_b5),
            ("A1_bit_depth", "uniform_theta_float", correct_float),
            ("A4_baseline", "single_channel_float64_teacher", correct_teacher),
            ("A3_adaptive", "b5_uniform_theta", correct_b5),
            ("A3_adaptive", "b5_mask_stability_m3_theta", correct_b5 + 5),
        ):
            rows.append({
                "family": family,
                "stage": stage,
                "configuration": config,
                "n_events": n,
                "n_correct": correct,
                "accuracy": correct / n,
            })
    gates = evaluate_gates(rows)
    assert "G1_lowbit_sufficiency" in gates
    assert "G2_spatial_sufficiency" in gates
    assert "G3_adaptive_gain" in gates
    # G2 should pass: b5 > teacher on both families.
    assert gates["G2_spatial_sufficiency"]["pass"] is True
