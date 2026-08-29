# Temporal representation decomposition provenance

Date: 2026-08-25.
Isolation: GT-timed events, GT unit rows, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25 with equal timestamps unsplit, frozen D9/B5 p99.9
candidate source from `frozen_uniform_candidate_source`.

## Inputs (full-file SHA-256)

Same four locked development recordings as the mapping-row and temporal-sketch
pilots. See [`mapping_row_v0_provenance.md`](mapping_row_v0_provenance.md).

## Implementation sources

| File | SHA-256 |
|---|---|
| `Spatial/algorithms/temporal_representation_decomp.py` | `114da42de2c3aab46da1e528e32369d3bee834c15ca856be7aba50a9456c7dfe` |
| `Spatial/experiments/run_temporal_representation_decomp.py` | `395033abcf0cec579f7fdf01f69aed4aef0c849c4c096fc65df9c605cba50e1f` |

The runner also imports the frozen sketch source, dictionary, and candidate
helper. Those files are the same objects hashed in
[`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md).

## Outputs

| File | SHA-256 |
|---|---|
| `Spatial/output/temporal_representation_decomp/temporal_representation_decomp_pilot.csv` | `5ea74fe23cfd888270f129f671923620d071a9d1f50850ef3bb463fcfb70939a` |
| `Spatial/output/temporal_representation_decomp/temporal_representation_decomp_summary.json` | `8c2f9af5319522b0ba7855f97c35b07e965fc2395011dbe3218d4110a2e5f6dc` |

Family means in [`temporal_representation_decomp_results.md`](temporal_representation_decomp_results.md)
are event-weighted from the CSV, not from a self-reported pass flag.
