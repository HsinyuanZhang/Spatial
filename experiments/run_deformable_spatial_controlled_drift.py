"""Controlled synthetic drift (Pool C). Gated on Phase-1.

    python -m Spatial.experiments.run_deformable_spatial_controlled_drift --pilot
"""

from __future__ import annotations

import argparse

STOP = """Pool C controlled perturbations are not licensed.

Phase 1 already showed that the proposed state coordinates (gain, fractional
offset, polarity, h) let the wrong unit fit the event. Measuring how well
those coordinates recover injected GT would answer a tracking question for
a representation that failed sorting. Spec: if R^2 up and Acc_home down,
drop the model; do not continue hyperparameter or perturbation work.

See Spatial/docs/deformable_spatial_template_results.md.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("pass --pilot")
    raise SystemExit(STOP)


if __name__ == "__main__":
    main()
