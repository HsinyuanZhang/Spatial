"""Geometric / CiM prefilter multi-method development pilot.

Runs schemes A–E (and optional hybrid) on the four locked recordings with a
chronological 50/25/25 split.  Confirmation is not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Spatial.experiments.causal_temporal_selection import (
    LOCKED_PILOT_SOURCE_SHA256,
    PILOT_DATASET_IDS,
)
from Spatial.experiments.geometric_prefilter_common import (
    evaluate_gates,
    prepare_pilot_bundle,
    run_dataset_sweep,
    write_csv,
    write_json,
)
from Spatial.experiments.run_causal_temporal_sketch import discover_pilot_datasets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Geometric prefilter multi-method pilot")
    parser.add_argument(
        "--methods",
        type=str,
        default="ALL",
        help="Comma list: BASELINE,A,B,C,D,E,HYBRID,ALL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "geometric_prefilter_sweep",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.duration != 60.0:
        parser.error("duration is locked to 60.0 for the development pilot")

    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    discovered = discover_pilot_datasets(duration_s=args.duration)
    rows: list[dict[str, Any]] = []

    for family in ("hj", "mearec"):
        for record in discovered[family]:
            if record.dataset_id not in PILOT_DATASET_IDS[family]:
                raise ValueError(f"unexpected dataset {record.dataset_id}")
            expected = LOCKED_PILOT_SOURCE_SHA256[record.dataset_id]
            if record.source_file_sha256.lower() != expected.lower():
                raise ValueError(f"SHA mismatch for {record.dataset_id}")
            condition = (
                "drift"
                if "drift" in record.dataset_id
                else ("static" if "static" in record.dataset_id else "synthetic")
            )
            if family == "mearec":
                condition = "mearec"
            print(f"running {record.dataset_id} methods={methods} ...", flush=True)
            dataset = record.load_dataset()
            try:
                bundle = prepare_pilot_bundle(dataset)
                part = run_dataset_sweep(
                    record.dataset_id,
                    family,
                    condition,
                    bundle,
                    methods=methods,
                )
                rows.extend(part)
            finally:
                del dataset

    gates = evaluate_gates(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "geometric_prefilter_sweep.csv"
    gates_path = args.output_dir / "geometric_prefilter_gates.json"
    summary_path = args.output_dir / "geometric_prefilter_summary.json"
    write_csv(csv_path, rows)
    write_json(gates_path, gates)
    write_json(
        summary_path,
        {
            "n_rows": len(rows),
            "methods": methods,
            "gates": gates,
            "datasets": sorted({r["dataset"] for r in rows}),
            "csv_sha256": _sha256_file(csv_path),
        },
    )
    print(json.dumps(gates, indent=2, sort_keys=True))
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
