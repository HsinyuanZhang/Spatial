"""Phase-2 causal tracking. Gated on Phase-1.

Phase 1 showed per-event gain/offset/polarity/h raise R^2 and cut same-home
assignment. This entry point refuses to run T1/T2/T3 on those states.

    python -m Spatial.experiments.run_deformable_spatial_tracking --pilot
"""

from __future__ import annotations

import argparse

STOP = """Phase 2 is not licensed.

Phase 1 (run_deformable_spatial_static) killed free per-event gain, fractional
offset, polarity, and residual h: they improve reconstruction and worsen
same-home classification, including on HJ static and MEArec. Spec stop:
do not start self-tracking to rescue a state that already fails the
held-out ceiling, and do not use slow identity update (T3) to save T2.

See Spatial/docs/deformable_spatial_template_results.md.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true", help="reserved; still refused")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    raise SystemExit(STOP)


if __name__ == "__main__":
    main()
