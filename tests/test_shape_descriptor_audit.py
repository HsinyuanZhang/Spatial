from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from Spatial.experiments.audit_shape_descriptor_gates import (
    PHASE_A_COLUMNS,
    AuditError,
    audit,
    load_phase_a_csv,
    main,
    recompute_gate_results,
)

DEV_RECORDINGS = [
    "hj_static",
    "hj_drift",
    "mea_sparse",
    "mea_dense",
]

POSNEG_BITS = 5
LAT_BITS = 4
AMP_GATE = 0.25


def _base_row(
    *,
    dataset: str,
    family: str,
    arm: str,
    accuracy: float,
    n_test: int = 1000,
    positive_margin_fraction: float = 0.80,
    nonconforming_slot_fraction: float = 0.001,
    posneg_reconstruction_max_error: int = 1,
) -> dict[str, object]:
    scene_type = "mearec" if family == "mearec" else ("static" if "static" in dataset else "drift")
    mearec_n_units = 10 if family == "mearec" else -1
    return {
        "dataset": dataset,
        "family": family,
        "scene_type": scene_type,
        "mearec_n_units": mearec_n_units,
        "arm": arm,
        "posneg_bits": POSNEG_BITS,
        "lat_bits": LAT_BITS,
        "amp_gate": AMP_GATE,
        "n_train": 1000,
        "n_test": n_test,
        "n_units": 16,
        "n_coords": 23 if arm == "com_posneg_lat_width" else 9,
        "row_bits": 109 if arm == "com_posneg_lat_width" else 45,
        "lane_count": 23 if arm == "com_posneg_lat_width" else 9,
        "accum_bits": 8,
        "accuracy": accuracy,
        "n_correct": int(round(accuracy * n_test)),
        "worst_unit_accuracy": accuracy - 0.05,
        "margin_mean": 1.0,
        "positive_margin_fraction": positive_margin_fraction,
        "true_template_distance_norm": 0.5,
        "hard_quartile_accuracy": accuracy - 0.1,
        "nonconforming_slot_fraction": nonconforming_slot_fraction,
        "posneg_reconstruction_max_error": posneg_reconstruction_max_error,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PHASE_A_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in PHASE_A_COLUMNS})


def _selection_from_results(
    gate_results: dict[str, object],
    *,
    passed: bool,
) -> dict[str, object]:
    return {
        "plan": "shape_orthogonal_descriptor_plan.md",
        "phase": "A",
        "development_recordings": list(DEV_RECORDINGS),
        "selected_arm": "com_posneg_lat_width" if passed else None,
        "posneg_bits": POSNEG_BITS,
        "lat_bits": LAT_BITS,
        "amp_gate": AMP_GATE,
        "gate_results": gate_results,
        "passed": passed,
        "null_reason": None if passed else "synthetic test",
    }


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = [
        ("hj_static", "hj", 0.75, 0.80),
        ("hj_drift", "hj", 0.76, 0.82),
        ("mea_sparse", "mearec", 0.78, 0.84),
        ("mea_dense", "mearec", 0.79, 0.86),
    ]
    for dataset, family, d9_acc, cand_acc in specs:
        rows.append(_base_row(dataset=dataset, family=family, arm="d9", accuracy=d9_acc))
        rows.append(
            _base_row(
                dataset=dataset,
                family=family,
                arm="com_posneg_lat_width",
                accuracy=cand_acc,
                positive_margin_fraction=0.82,
            )
        )
        rows.append(
            _base_row(
                dataset=dataset,
                family=family,
                arm="d9_rand",
                accuracy=d9_acc + 0.005,
            )
        )
    return rows


def _passing_gate_results() -> dict[str, object]:
    selection = {
        "development_recordings": DEV_RECORDINGS,
        "posneg_bits": POSNEG_BITS,
        "lat_bits": LAT_BITS,
        "amp_gate": AMP_GATE,
    }
    return recompute_gate_results(_passing_rows(), selection)


def test_event_weighted_family_mean_hand_check(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for dataset, family, cand_acc, d9_acc, n_test in [
        ("hj_a", "hj", 0.90, 0.85, 100),
        ("hj_b", "hj", 0.70, 0.60, 300),
        ("mea_a", "mearec", 0.88, 0.80, 200),
        ("mea_b", "mearec", 0.76, 0.72, 200),
    ]:
        rows.append(
            _base_row(
                dataset=dataset,
                family=family,
                arm="d9",
                accuracy=d9_acc,
                n_test=n_test,
            )
        )
        rows.append(
            _base_row(
                dataset=dataset,
                family=family,
                arm="com_posneg_lat_width",
                accuracy=cand_acc,
                n_test=n_test,
            )
        )
        rows.append(
            _base_row(
                dataset=dataset,
                family=family,
                arm="d9_rand",
                accuracy=d9_acc,
                n_test=n_test,
            )
        )

    selection = {
        "development_recordings": ["hj_a", "hj_b", "mea_a", "mea_b"],
        "posneg_bits": POSNEG_BITS,
        "lat_bits": LAT_BITS,
        "amp_gate": AMP_GATE,
    }
    results = recompute_gate_results(rows, selection)

    hj_gain = ((0.90 * 100 + 0.70 * 300) - (0.85 * 100 + 0.60 * 300)) / 400 * 100.0
    mea_gain = ((0.88 * 200 + 0.76 * 200) - (0.80 * 200 + 0.72 * 200)) / 400 * 100.0
    assert results["family_gain_hj_pp"] == pytest.approx(hj_gain)
    assert results["family_gain_mearec_pp"] == pytest.approx(mea_gain)
    assert hj_gain == pytest.approx(8.75)
    assert mea_gain == pytest.approx(6.0)


def test_condition_1_family_gain_failure(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "hj_static":
            row["accuracy"] = 0.755
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "hj_drift":
            row["accuracy"] = 0.765
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is False
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is True
    assert results["condition_5_rand_control_fails"] is True


def test_condition_2_recording_loss_failure(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "mea_dense":
            row["accuracy"] = 0.78
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is False
    assert results["worst_recording_delta_pp"] == pytest.approx(-1.0)
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is True
    assert results["condition_5_rand_control_fails"] is True


def test_condition_3_positive_margin_failure(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["family"] == "mearec":
            row["positive_margin_fraction"] = 0.70
        if row["arm"] == "d9" and row["family"] == "mearec":
            row["positive_margin_fraction"] = 0.75
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is False
    assert results["positive_margin_delta_mearec_pp"] == pytest.approx(-5.0)
    assert results["condition_4_reconstruction"] is True
    assert results["condition_5_rand_control_fails"] is True


def test_condition_4_reconstruction_failure(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "hj_drift":
            row["posneg_reconstruction_max_error"] = 2
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is False
    assert results["max_posneg_reconstruction_error"] == 2
    assert results["condition_5_rand_control_fails"] is True


def test_condition_4_nonconforming_failure(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "mea_sparse":
            row["nonconforming_slot_fraction"] = 0.006
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is False
    assert results["condition_5_rand_control_fails"] is True


def test_condition_5_rand_control_passes_gate(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "d9_rand" and row["family"] == "hj":
            row["accuracy"] = float(row["accuracy"]) + 0.03
        if row["arm"] == "d9_rand" and row["family"] == "mearec":
            row["accuracy"] = float(row["accuracy"]) + 0.03
    results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is True
    assert results["condition_5_rand_control_fails"] is False


def test_disagrees_when_runner_claims_pass_on_failing_csv(tmp_path: Path) -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "hj_static":
            row["accuracy"] = 0.755
        if row["arm"] == "com_posneg_lat_width" and row["dataset"] == "hj_drift":
            row["accuracy"] = 0.765

    csv_path = tmp_path / "phase_a.csv"
    selection_path = tmp_path / "selection.json"
    _write_csv(csv_path, rows)

    auditor_results = recompute_gate_results(
        rows,
        {
            "development_recordings": DEV_RECORDINGS,
            "posneg_bits": POSNEG_BITS,
            "lat_bits": LAT_BITS,
            "amp_gate": AMP_GATE,
        },
    )
    lying_results = dict(auditor_results)
    lying_results["condition_1_family_gain"] = True
    selection_path.write_text(
        json.dumps(_selection_from_results(lying_results, passed=True), indent=2)
    )

    report = audit(csv_path, selection_path)
    assert report["auditor_passed"] is False
    assert report["runner_passed"] is True
    assert report["agrees_with_runner"] is False
    assert main(["--csv", str(csv_path), "--selection", str(selection_path), "--quiet"]) == 1
    assert (
        main(
            [
                "--csv",
                str(csv_path),
                "--selection",
                str(selection_path),
                "--quiet",
                "--exit-zero",
            ]
        )
        == 0
    )


def test_schema_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    columns = [name for name in PHASE_A_COLUMNS if name != "accuracy"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: 0 for column in columns})
    with pytest.raises(AuditError, match="missing columns"):
        load_phase_a_csv(path)


def test_schema_extra_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    columns = list(PHASE_A_COLUMNS) + ["extra_column"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: 0 for column in columns})
    with pytest.raises(AuditError, match="unexpected columns"):
        load_phase_a_csv(path)


def test_missing_d9_control_raises(tmp_path: Path) -> None:
    rows = _passing_rows()
    rows = [row for row in rows if not (row["arm"] == "d9" and row["dataset"] == "hj_static")]
    with pytest.raises(AuditError, match="missing 'd9' control row"):
        recompute_gate_results(
            rows,
            {
                "development_recordings": DEV_RECORDINGS,
                "posneg_bits": POSNEG_BITS,
                "lat_bits": LAT_BITS,
                "amp_gate": AMP_GATE,
            },
        )


def test_passing_baseline_all_conditions_true(tmp_path: Path) -> None:
    results = _passing_gate_results()
    assert results["condition_1_family_gain"] is True
    assert results["condition_2_no_recording_loss"] is True
    assert results["condition_3_positive_margin"] is True
    assert results["condition_4_reconstruction"] is True
    assert results["condition_5_rand_control_fails"] is True
