"""Extract a Yger Zenodo recording and write a Kilosort4 int16 binary.

    python -m Spatial.data.prepare_yger_zenodo --list
    python -m Spatial.data.prepare_yger_zenodo --recording 20160415_patch2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Spatial.data.yger_zenodo import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_WORK_DIR,
    list_archive_recordings,
    prepare_recording,
    write_k_nearest_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--recording", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--k-nearest",
        type=int,
        default=0,
        help="If >0, also write a k-nearest-site int16 subset (SpikeForest uses 64).",
    )
    args = parser.parse_args()

    if args.list or args.recording is None:
        recs = list_archive_recordings(args.archive_dir)
        for rec in recs:
            print(rec)
        if args.list:
            return 0
        if not recs:
            print(f"no archives under {args.archive_dir}", flush=True)
            return 1
        print("pass --recording <id> to extract and convert", flush=True)
        return 0

    meta = prepare_recording(
        args.recording,
        archive_dir=args.archive_dir,
        work_dir=args.work_dir,
        force=args.force,
    )
    print(json.dumps(meta, indent=2))
    if args.k_nearest:
        subset = write_k_nearest_subset(
            args.recording,
            k=args.k_nearest,
            archive_dir=args.archive_dir,
            work_dir=args.work_dir,
        )
        print(json.dumps(subset, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
