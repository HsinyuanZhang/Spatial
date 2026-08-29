"""Export the selected Kilosort4 Th=13 sorting as Spatial pseudo-GT.

    python -m Spatial.data.export_yger_ks4_pseudo_gt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Spatial.data.yger_zenodo import (
    DEFAULT_KS4_TH13_DIR,
    DEFAULT_SPATIAL_NPZ,
    export_ks4_spatial_npz,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks-dir", type=Path, default=DEFAULT_KS4_TH13_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPATIAL_NPZ)
    args = parser.parse_args()
    info = export_ks4_spatial_npz(ks_dir=args.ks_dir, out_path=args.out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
