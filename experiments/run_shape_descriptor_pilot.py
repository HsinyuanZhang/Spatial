"""Phase A shape-orthogonal descriptor pilot on locked development recordings.

  python -m Spatial.experiments.run_shape_descriptor_pilot --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

from Spatial.data.loader import Dataset, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import _chronological_split
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, parse_rec_name
from Spatial.experiments.shape_descriptor_common import (
    AMP_GATE_CHOICES,
    ARM_IDS,
    CSV_FIELDNAMES,
    LAT_BIT_CHOICES,
    POSNEG_BIT_CHOICES,
    _scene_type,
    build_descriptor_blocks,
    evaluate_arm_rows,
    evaluate_selection,
    evaluation_cache_key,
)
from Spatial.experiments.spatial_cim_common import write_csv, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Spatial" / "output" / "shape_descriptor_pilot"

LOCKED_DEVELOPMENT_RECORDINGS = (
    ("hj", "drift16c_600s_11"),
    ("hj", "static16c_600s_11"),
    ("mearec", "rec_v1_units10_snr5_seed202601"),
    ("mearec", "rec_v1_units20_snr5_seed202601"),
)

DEDUP_NOTES = (
    "Distinct descriptors are evaluated once per recording using evaluation_cache_key: "
    "posneg_bits is ignored unless the arm includes POSNEG; lat_bits is ignored unless "
    "the arm includes LAT or RAND; amp_gate is ignored unless the arm includes LAT. "
    "Every grid row is still emitted with its sweep keys so auditors can join on "
    "(dataset, arm, posneg_bits, lat_bits, amp_gate)."
)


def _mearec_n_units(dataset_name: str, family: str) -> int:
    if family != "mearec":
        return -1
    stem = dataset_name.removeprefix("mearec_")
    info = parse_rec_name(stem)
    value = info.get("n_units_design")
    return int(value) if value is not None else -1


def _load_locked_recording(family: str, name: str, duration: float, mearec_dir: Path) -> Dataset:
    if family == "hj":
        return load_hybrid_janelia(scene=name, duration_s=duration)
    path = mearec_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"locked MEArec development input is missing: {path}")
    return load_mearec_npz(path, duration_s=duration)


def _iter_recordings(
    recordings: list[tuple[str, str]],
    duration: float,
    mearec_dir: Path,
) -> Iterator[tuple[str, Dataset]]:
    for family, name in recordings:
        yield family, _load_locked_recording(family, name, duration, mearec_dir)


def evaluate_recording(
    dataset: Dataset,
    *,
    family: str,
) -> list[dict[str, Any]]:
    blocks = build_descriptor_blocks(dataset)
    train_idx, test_idx = _chronological_split(blocks["times"])
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for arm in ARM_IDS:
        for posneg_bits in POSNEG_BIT_CHOICES:
            for lat_bits in LAT_BIT_CHOICES:
                for amp_gate in AMP_GATE_CHOICES:
                    key = evaluation_cache_key(arm, posneg_bits, lat_bits, amp_gate)
                    if key not in cache:
                        cache[key] = evaluate_arm_rows(
                            blocks,
                            train_idx,
                            test_idx,
                            arm=arm,
                            posneg_bits=posneg_bits,
                            lat_bits=lat_bits,
                            amp_gate=amp_gate,
                            recording_name=dataset.name,
                        )
                    metrics = cache[key]
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "family": family,
                            "scene_type": _scene_type(dataset.name, family),
                            "mearec_n_units": _mearec_n_units(dataset.name, family),
                            "arm": arm,
                            "posneg_bits": int(posneg_bits),
                            "lat_bits": int(lat_bits),
                            "amp_gate": float(amp_gate),
                            **metrics,
                        }
                    )
    return rows


def _parse_recordings_arg(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in values:
        if ":" not in item:
            raise ValueError(
                f"--recordings entries must look like hj:scene or mearec:rec_stem, got {item!r}"
            )
        family, name = item.split(":", 1)
        family = family.strip().lower()
        name = name.strip()
        if family not in ("hj", "mearec"):
            raise ValueError(f"unknown family {family!r} in --recordings entry {item!r}")
        if not name:
            raise ValueError(f"missing recording name in --recordings entry {item!r}")
        parsed.append((family, name))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape-orthogonal descriptor Phase A pilot")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Override locked development set as family:name pairs",
    )
    parser.add_argument("--mearec-npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    recordings = (
        _parse_recordings_arg(args.recordings)
        if args.recordings is not None
        else list(LOCKED_DEVELOPMENT_RECORDINGS)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    # The gate joins per-recording rows on the CSV `dataset` column, so the
    # selection must carry resolved loader names, not the short scene keys.
    resolved_names: list[str] = []
    for family, dataset in _iter_recordings(recordings, args.duration, Path(args.mearec_npz_dir)):
        if not args.quiet:
            print(f"{family}: {dataset.name}", flush=True)
        resolved_names.append(dataset.name)
        rows.extend(evaluate_recording(dataset, family=family))

    csv_path = out_dir / "shape_descriptor_pilot.csv"
    write_csv(csv_path, [{key: row[key] for key in CSV_FIELDNAMES} for row in rows])

    selection = evaluate_selection(
        rows,
        development_recordings=resolved_names,
    )
    selection["deduplication"] = DEDUP_NOTES
    json_path = out_dir / "shape_descriptor_selection.json"
    write_json(json_path, selection)

    if not args.quiet:
        print(DEDUP_NOTES)
        print(f"Wrote {csv_path} ({len(rows)} rows)")
        print(f"Wrote {json_path}")
        print(f"Phase A passed: {selection['passed']}")


if __name__ == "__main__":
    main()
