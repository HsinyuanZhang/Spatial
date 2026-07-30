# Geometric prefilter pilot — provenance

Date: 2026-07-30.

## Source digests

See [`geometric_prefilter_source.sha256`](geometric_prefilter_source.sha256).

## Output digests

See [`geometric_prefilter_outputs.sha256`](geometric_prefilter_outputs.sha256).

## Locked inputs

Pilot IDs and file SHA256: `LOCKED_PILOT_SOURCE_SHA256` in
`experiments/causal_temporal_selection.py` (same four-recording contract as the
causal temporal / masked / weighted pilots).

## Gate recomputation

Gates recomputed by `experiments.geometric_prefilter_common.evaluate_gates`
from integer `n_test`-weighted rows in
`output/geometric_prefilter_sweep/geometric_prefilter_sweep.csv`.
Never from rounded display values alone.
