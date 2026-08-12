"""Independent Phase A gate auditor for the shape-orthogonal descriptor pilot.

Recomputes every development-gate quantity from the Phase A CSV and compares
against the runner's selection JSON. All gains and deltas are candidate minus
control (d9): a positive gain means the candidate beats d9.

Usage:
  python -m Spatial.experiments.audit_shape_descriptor_gates
  python -m Spatial.experiments.audit_shape_descriptor_gates --exit-zero
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from Spatial.experiments.spatial_cim_common import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "Spatial" / "output" / "shape_descriptor_pilot"
DEFAULT_CSV = DEFAULT_DIR / "shape_descriptor_pilot.csv"
DEFAULT_SELECTION = DEFAULT_DIR / "shape_descriptor_selection.json"
DEFAULT_OUTPUT = DEFAULT_DIR / "shape_descriptor_audit.json"

CANDIDATE_ARM = "com_posneg_lat_width"
CONTROL_ARM = "d9"
RAND_ARM = "d9_rand"

PHASE_A_COLUMNS: tuple[str, ...] = (
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

INT_COLUMNS = frozenset(
    {
        "mearec_n_units",
        "posneg_bits",
        "lat_bits",
        "n_train",
        "n_test",
        "n_units",
        "n_coords",
        "row_bits",
        "lane_count",
        "accum_bits",
        "n_correct",
        "posneg_reconstruction_max_error",
    }
)
FLOAT_COLUMNS = frozenset(
    {
        "amp_gate",
        "accuracy",
        "worst_unit_accuracy",
        "margin_mean",
        "positive_margin_fraction",
        "true_template_distance_norm",
        "hard_quartile_accuracy",
        "nonconforming_slot_fraction",
    }
)

JOIN_KEY = ("dataset", "posneg_bits", "lat_bits", "amp_gate")

GATE_RESULT_KEYS: tuple[str, ...] = (
    "family_gain_hj_pp",
    "family_gain_mearec_pp",
    "worst_recording_delta_pp",
    "positive_margin_delta_hj_pp",
    "positive_margin_delta_mearec_pp",
    "max_nonconforming_slot_fraction",
    "max_posneg_reconstruction_error",
    "rand_control_gain_hj_pp",
    "rand_control_gain_mearec_pp",
    "condition_1_family_gain",
    "condition_2_no_recording_loss",
    "condition_3_positive_margin",
    "condition_4_reconstruction",
    "condition_5_rand_control_fails",
)

CONDITION_KEYS: tuple[str, ...] = (
    "condition_1_family_gain",
    "condition_2_no_recording_loss",
    "condition_3_positive_margin",
    "condition_4_reconstruction",
    "condition_5_rand_control_fails",
)

FAMILY_GAIN_THRESHOLD_PP = 2.0
WORST_RECORDING_LOSS_LIMIT_PP = 0.5
NONCONFORMING_LIMIT = 0.005
RECONSTRUCTION_ERROR_LIMIT = 1


class AuditError(Exception):
    pass


def _parse_cell(column: str, raw: str) -> Any:
    if raw == "":
        raise AuditError(f"empty value for column {column!r}")
    if column in INT_COLUMNS:
        try:
            return int(raw)
        except ValueError as exc:
            raise AuditError(f"cannot parse int for column {column!r}: {raw!r}") from exc
    if column in FLOAT_COLUMNS:
        try:
            value = float(raw)
        except ValueError as exc:
            raise AuditError(f"cannot parse float for column {column!r}: {raw!r}") from exc
        if not np.isfinite(value):
            raise AuditError(f"non-finite float for column {column!r}: {raw!r}")
        return value
    return raw


def _validate_schema(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise AuditError("CSV is missing a header row")
    expected = set(PHASE_A_COLUMNS)
    observed = set(fieldnames)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unexpected columns: {extra}")
        raise AuditError("CSV schema mismatch: " + "; ".join(parts))


def load_phase_a_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_schema(reader.fieldnames)
        rows: list[dict[str, Any]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            row: dict[str, Any] = {}
            for column in PHASE_A_COLUMNS:
                if column not in raw_row:
                    raise AuditError(
                        f"row {line_number}: missing value for column {column!r}"
                    )
                row[column] = _parse_cell(column, raw_row[column])
            rows.append(row)
    return rows


def load_selection_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for key in ("gate_results", "passed", "development_recordings"):
        if key not in payload:
            raise AuditError(f"selection JSON missing required key {key!r}")
    gate_results = payload["gate_results"]
    if not isinstance(gate_results, dict):
        raise AuditError("selection JSON gate_results must be an object")
    for key in GATE_RESULT_KEYS:
        if key not in gate_results:
            raise AuditError(f"selection JSON gate_results missing key {key!r}")
    return payload


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in JOIN_KEY)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    posneg_bits: int,
    lat_bits: int,
    amp_gate: float,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm
        and row["posneg_bits"] == posneg_bits
        and row["lat_bits"] == lat_bits
        and row["amp_gate"] == amp_gate
    ]
    if not selected:
        raise AuditError(
            f"no CSV rows for arm={arm!r}, posneg_bits={posneg_bits}, "
            f"lat_bits={lat_bits}, amp_gate={amp_gate}"
        )
    return selected


def _join_against_control(
    arm_rows: list[dict[str, Any]],
    control_by_key: dict[tuple[Any, ...], dict[str, Any]],
    *,
    arm_name: str,
    control_name: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for arm_row in arm_rows:
        key = _row_key(arm_row)
        control_row = control_by_key.get(key)
        if control_row is None:
            raise AuditError(
                f"missing {control_name!r} control row for join key {dict(zip(JOIN_KEY, key))!r} "
                f"required by {arm_name!r}"
            )
        joined.append((arm_row, control_row))
    return joined


def _paired_delta_pp(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    family: str,
    field: str,
) -> float:
    total_weight = 0
    candidate_sum = 0.0
    control_sum = 0.0
    for arm_row, control_row in pairs:
        if arm_row["family"] != family:
            continue
        weight = int(arm_row["n_test"])
        if weight < 0:
            raise AuditError(f"negative n_test for dataset {arm_row['dataset']!r}")
        total_weight += weight
        candidate_sum += float(arm_row[field]) * weight
        control_sum += float(control_row[field]) * weight
    if total_weight == 0:
        raise AuditError(
            f"no {family!r} rows with positive n_test for paired delta of {field!r}"
        )
    return (candidate_sum - control_sum) / total_weight * 100.0


def _accuracy_gain_pp(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    family: str,
) -> float:
    return _paired_delta_pp(pairs, family=family, field="accuracy")


def _positive_margin_delta_pp(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    family: str,
) -> float:
    return _paired_delta_pp(
        pairs, family=family, field="positive_margin_fraction"
    )


def _worst_recording_delta_pp(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    development_recordings: list[str],
) -> float:
    dev_set = set(development_recordings)
    deltas: list[float] = []
    for arm_row, control_row in pairs:
        if arm_row["dataset"] not in dev_set:
            continue
        delta_pp = (float(arm_row["accuracy"]) - float(control_row["accuracy"])) * 100.0
        deltas.append(delta_pp)
    if len(deltas) != len(development_recordings):
        seen = {arm_row["dataset"] for arm_row, _ in pairs if arm_row["dataset"] in dev_set}
        missing = sorted(dev_set - seen)
        if missing:
            raise AuditError(
                f"missing candidate rows for development recordings: {missing}"
            )
        extra = sorted(seen - dev_set)
        if extra:
            raise AuditError(
                f"candidate rows reference unknown development recordings: {extra}"
            )
        raise AuditError("development recording count mismatch against joined candidate rows")
    return float(min(deltas))


def _condition_1(gain_hj_pp: float, gain_mearec_pp: float) -> bool:
    return gain_hj_pp >= FAMILY_GAIN_THRESHOLD_PP and gain_mearec_pp >= FAMILY_GAIN_THRESHOLD_PP


def _condition_2(worst_recording_delta_pp: float) -> bool:
    return worst_recording_delta_pp >= -WORST_RECORDING_LOSS_LIMIT_PP


def _condition_3(delta_hj_pp: float, delta_mearec_pp: float) -> bool:
    return delta_hj_pp >= 0.0 and delta_mearec_pp >= 0.0


def _reconstruction_metrics(
    candidate_rows: list[dict[str, Any]],
    *,
    development_recordings: list[str],
) -> tuple[float, int, bool, bool]:
    dev_set = set(development_recordings)
    max_nonconforming = 0.0
    max_reconstruction = 0
    reconstruction_ok = True
    nonconforming_ok = True
    for dataset in development_recordings:
        dev_rows = [row for row in candidate_rows if row["dataset"] == dataset]
        if not dev_rows:
            raise AuditError(
                f"missing candidate row for development recording {dataset!r}"
            )
    for row in candidate_rows:
        reconstruction = int(row["posneg_reconstruction_max_error"])
        max_reconstruction = max(max_reconstruction, reconstruction)
        if reconstruction > RECONSTRUCTION_ERROR_LIMIT:
            reconstruction_ok = False
        if row["dataset"] in dev_set:
            nonconforming = float(row["nonconforming_slot_fraction"])
            max_nonconforming = max(max_nonconforming, nonconforming)
            if nonconforming >= NONCONFORMING_LIMIT:
                nonconforming_ok = False
    return max_nonconforming, max_reconstruction, reconstruction_ok, nonconforming_ok


def _condition_5(rand_gain_hj_pp: float, rand_gain_mearec_pp: float) -> bool:
    return not _condition_1(rand_gain_hj_pp, rand_gain_mearec_pp)


def recompute_gate_results(
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    development_recordings = list(selection["development_recordings"])
    if not development_recordings:
        raise AuditError("development_recordings must be non-empty")

    posneg_bits = int(selection["posneg_bits"])
    lat_bits = int(selection["lat_bits"])
    amp_gate = float(selection["amp_gate"])
    if not np.isfinite(amp_gate):
        raise AuditError(f"non-finite amp_gate in selection JSON: {amp_gate!r}")

    control_rows = _filter_rows(
        rows,
        arm=CONTROL_ARM,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=amp_gate,
    )
    control_by_key = {_row_key(row): row for row in control_rows}

    candidate_rows = _filter_rows(
        rows,
        arm=CANDIDATE_ARM,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=amp_gate,
    )
    candidate_pairs = _join_against_control(
        candidate_rows,
        control_by_key,
        arm_name=CANDIDATE_ARM,
        control_name=CONTROL_ARM,
    )

    rand_rows = _filter_rows(
        rows,
        arm=RAND_ARM,
        posneg_bits=posneg_bits,
        lat_bits=lat_bits,
        amp_gate=amp_gate,
    )
    rand_pairs = _join_against_control(
        rand_rows,
        control_by_key,
        arm_name=RAND_ARM,
        control_name=CONTROL_ARM,
    )

    family_gain_hj_pp = _accuracy_gain_pp(candidate_pairs, family="hj")
    family_gain_mearec_pp = _accuracy_gain_pp(candidate_pairs, family="mearec")
    worst_recording_delta_pp = _worst_recording_delta_pp(
        candidate_pairs,
        development_recordings=development_recordings,
    )
    positive_margin_delta_hj_pp = _positive_margin_delta_pp(
        candidate_pairs, family="hj"
    )
    positive_margin_delta_mearec_pp = _positive_margin_delta_pp(
        candidate_pairs, family="mearec"
    )
    max_nonconforming, max_reconstruction, reconstruction_ok, nonconforming_ok = (
        _reconstruction_metrics(
            candidate_rows,
            development_recordings=development_recordings,
        )
    )

    rand_control_gain_hj_pp = _accuracy_gain_pp(rand_pairs, family="hj")
    rand_control_gain_mearec_pp = _accuracy_gain_pp(rand_pairs, family="mearec")

    condition_1_family_gain = _condition_1(family_gain_hj_pp, family_gain_mearec_pp)
    condition_2_no_recording_loss = _condition_2(worst_recording_delta_pp)
    condition_3_positive_margin = _condition_3(
        positive_margin_delta_hj_pp,
        positive_margin_delta_mearec_pp,
    )
    condition_4_reconstruction = reconstruction_ok and nonconforming_ok
    condition_5_rand_control_fails = _condition_5(
        rand_control_gain_hj_pp,
        rand_control_gain_mearec_pp,
    )

    return {
        "family_gain_hj_pp": family_gain_hj_pp,
        "family_gain_mearec_pp": family_gain_mearec_pp,
        "worst_recording_delta_pp": worst_recording_delta_pp,
        "positive_margin_delta_hj_pp": positive_margin_delta_hj_pp,
        "positive_margin_delta_mearec_pp": positive_margin_delta_mearec_pp,
        "max_nonconforming_slot_fraction": max_nonconforming,
        "max_posneg_reconstruction_error": max_reconstruction,
        "rand_control_gain_hj_pp": rand_control_gain_hj_pp,
        "rand_control_gain_mearec_pp": rand_control_gain_mearec_pp,
        "condition_1_family_gain": condition_1_family_gain,
        "condition_2_no_recording_loss": condition_2_no_recording_loss,
        "condition_3_positive_margin": condition_3_positive_margin,
        "condition_4_reconstruction": condition_4_reconstruction,
        "condition_5_rand_control_fails": condition_5_rand_control_fails,
    }


def _runner_passed(gate_results: dict[str, Any]) -> bool:
    return all(bool(gate_results[key]) for key in CONDITION_KEYS)


def _floats_close(left: float, right: float, *, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=rtol, atol=atol))


def _gate_results_agree(auditor: dict[str, Any], runner: dict[str, Any]) -> bool:
    float_keys = (
        "family_gain_hj_pp",
        "family_gain_mearec_pp",
        "worst_recording_delta_pp",
        "positive_margin_delta_hj_pp",
        "positive_margin_delta_mearec_pp",
        "max_nonconforming_slot_fraction",
        "rand_control_gain_hj_pp",
        "rand_control_gain_mearec_pp",
    )
    for key in float_keys:
        if not _floats_close(float(auditor[key]), float(runner[key])):
            return False
    if int(auditor["max_posneg_reconstruction_error"]) != int(
        runner["max_posneg_reconstruction_error"]
    ):
        return False
    for key in CONDITION_KEYS:
        if bool(auditor[key]) != bool(runner[key]):
            return False
    return True


def audit(
    csv_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    rows = load_phase_a_csv(csv_path)
    selection = load_selection_json(selection_path)
    auditor_gate_results = recompute_gate_results(rows, selection)
    runner_gate_results = {
        key: selection["gate_results"][key] for key in GATE_RESULT_KEYS
    }
    auditor_passed = _runner_passed(auditor_gate_results)
    runner_passed = bool(selection["passed"])
    agrees_with_runner = (
        auditor_passed == runner_passed
        and _gate_results_agree(auditor_gate_results, runner_gate_results)
    )
    return {
        "plan": selection.get("plan", "shape_orthogonal_descriptor_plan.md"),
        "phase": selection.get("phase", "A"),
        "csv": str(csv_path),
        "selection": str(selection_path),
        "development_recordings": list(selection["development_recordings"]),
        "posneg_bits": int(selection["posneg_bits"]),
        "lat_bits": int(selection["lat_bits"]),
        "amp_gate": float(selection["amp_gate"]),
        "candidate_arm": CANDIDATE_ARM,
        "control_arm": CONTROL_ARM,
        "rand_arm": RAND_ARM,
        "auditor_gate_results": auditor_gate_results,
        "runner_gate_results": runner_gate_results,
        "auditor_passed": auditor_passed,
        "runner_passed": runner_passed,
        "agrees_with_runner": agrees_with_runner,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently audit Phase A shape-descriptor development gates."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit with status 0 even when the auditor disagrees with the runner",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args.csv, args.selection)
    write_json(args.output, report)
    if not args.quiet:
        status = "AGREE" if report["agrees_with_runner"] else "DISAGREE"
        print(
            f"{status}: auditor_passed={report['auditor_passed']} "
            f"runner_passed={report['runner_passed']} -> {args.output}"
        )
    if report["agrees_with_runner"] or args.exit_zero:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
