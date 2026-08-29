"""Contract tests for the robust P2P estimator/bit-width pilot."""

from __future__ import annotations

import numpy as np

from Spatial.experiments.run_robust_p2p_template_pilot import (
    BIT_MODES,
    ESTIMATOR_GRID,
    MATCHERS,
    evaluate_rows,
    fit_templates,
    quantize_footprint,
    summarize_confirmation,
    summarize_and_select,
)


def _separable_rows(n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.arange(n, dtype=np.int64) % 2
    footprint = np.zeros((n, 3), dtype=np.float64)
    footprint[labels == 0] = [1.0, 0.25, 0.0]
    footprint[labels == 1] = [0.0, 0.25, 1.0]
    times = np.arange(n, dtype=np.int64) * 10
    return footprint, labels, times


def test_quantize_footprint_preserves_exact_integer_codes():
    footprint = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    codes, bits, code_max, integer = quantize_footprint(footprint, "4")
    assert bits == 4 and code_max == 15 and integer
    assert np.issubdtype(codes.dtype, np.integer)
    np.testing.assert_array_equal(codes, [[0, 8, 15]])


def test_integer_templates_use_rounded_code_mean():
    rows = np.array([[0, 2], [1, 3], [15, 14], [14, 13]], dtype=np.int64)
    labels = np.array([2, 2, 9, 9], dtype=np.int64)
    units, templates = fit_templates(rows, labels, integer=True)
    np.testing.assert_array_equal(units, [2, 9])
    # np.rint uses deterministic ties-to-even: [0.5,2.5] -> [0,2].
    np.testing.assert_array_equal(templates, [[0, 2], [14, 14]])


def test_evaluate_rows_matches_in_integer_domain():
    footprint, labels, times = _separable_rows()
    row = evaluate_rows(
        footprint,
        labels,
        times,
        dataset_name="synthetic_static",
        family="hj",
        estimator_name="moving_average_k3",
        estimator="moving_average",
        average_points=3,
        hardware_eligible=True,
        bit_mode="4",
        matcher="l1",
    )
    assert row["integer_template"] is True
    assert row["nominal_bits"] == 4
    assert row["n_correct"] == row["n_eval"]
    assert row["assignment_accuracy"] == 1.0
    assert row["positive_margin_fraction"] == 1.0
    assert row["template_payload_bits"] == 2 * 3 * 4
    assert row["estimator_sum_guard_bits"] == 2


def test_cosine_and_l1_share_the_exact_test_rows():
    footprint, labels, times = _separable_rows()
    common = dict(
        dataset_name="synthetic_static",
        family="hj",
        estimator_name="raw_range",
        estimator="raw",
        average_points=1,
        hardware_eligible=True,
        bit_mode="5",
    )
    l1 = evaluate_rows(footprint, labels, times, matcher="l1", **common)
    cosine = evaluate_rows(footprint, labels, times, matcher="cosine", **common)
    assert l1["n_fit"] == cosine["n_fit"]
    assert l1["n_cal"] == cosine["n_cal"]
    assert l1["n_eval"] == cosine["n_eval"]
    assert l1["n_correct"] == cosine["n_correct"] == l1["n_eval"]


def _fake_gate_rows() -> list[dict[str, object]]:
    recordings = (
        ("hj_drift", "hj", "drift"),
        ("hj_static", "hj", "static"),
        ("mearec_10", "mearec", "other"),
        ("mearec_20", "mearec", "other"),
    )
    rows: list[dict[str, object]] = []
    for estimator_name, _, _, _ in ESTIMATOR_GRID:
        for bit_mode in BIT_MODES:
            for matcher in MATCHERS:
                for dataset, family, condition in recordings:
                    accuracy = 0.80
                    true_l1 = 0.20
                    if estimator_name == "moving_average_k3" and dataset == "hj_drift":
                        accuracy = 0.81
                        true_l1 = 0.19
                    rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "condition": condition,
                            "estimator_name": estimator_name,
                            "bit_mode": bit_mode,
                            "matcher": matcher,
                            "n_eval": 100,
                            "n_correct": int(round(accuracy * 100)),
                            "assignment_accuracy": accuracy,
                            "positive_margin_fraction": accuracy,
                            "median_true_value_normalized": true_l1,
                        }
                    )
    return rows


def test_registered_selection_prefers_passing_k3_then_smallest_bits():
    result = summarize_and_select(_fake_gate_rows())
    selection = result["selection"]
    assert selection["estimator_gate_pass"] is True
    assert selection["selected_estimator"] == "moving_average_k3"
    assert selection["bit_gate_pass"] is True
    assert selection["selected_bits"] == 4
    assert selection["advance_to_online_phase"] is True


def _fake_confirmation_rows(
    *, selected_gain: float = 0.01, quant_loss: float = 0.002
) -> list[dict[str, object]]:
    recordings = [
        (f"hj_drift_{i}", "hj", "drift") for i in range(5)
    ] + [
        (f"hj_static_{i}", "hj", "static") for i in range(5)
    ] + [
        ("mearec_10", "mearec", "other"),
        ("mearec_20", "mearec", "other"),
    ]
    rows: list[dict[str, object]] = []
    for dataset, family, condition in recordings:
        for estimator, bit_mode in (
            ("raw_range", "float"),
            ("centered_mean_k5", "float"),
            ("centered_mean_k5", "8"),
        ):
            for matcher in MATCHERS:
                accuracy = 0.80
                if estimator == "centered_mean_k5":
                    accuracy += selected_gain
                if bit_mode == "8":
                    accuracy -= quant_loss
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "condition": condition,
                        "estimator_name": estimator,
                        "bit_mode": bit_mode,
                        "matcher": matcher,
                        "n_eval": 1000,
                        "n_correct": int(round(accuracy * 1000)),
                        "assignment_accuracy": accuracy,
                        "positive_margin_fraction": accuracy,
                        "median_true_value_normalized": 0.20,
                    }
                )
    return rows


def test_confirmation_gate_passes_frozen_estimator_and_bits():
    result = summarize_confirmation(_fake_confirmation_rows())
    assert result["n_recordings"] == 12
    assert result["estimator_gate_pass"] is True
    assert result["quantization_gate_pass"] is True
    assert result["confirmation_pass"] is True


def test_confirmation_gate_rejects_quantized_margin_ties():
    result = summarize_confirmation(
        _fake_confirmation_rows(selected_gain=0.01, quant_loss=0.006)
    )
    assert result["estimator_gate_pass"] is True
    assert result["quantization_gate_pass"] is False
    assert result["confirmation_pass"] is False
