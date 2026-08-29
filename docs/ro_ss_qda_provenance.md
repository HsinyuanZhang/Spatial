# RO-SS-QDA provenance

Date: 2026-08-25.
Isolation: GT-timed events, GT unit rows, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25 with equal timestamps unsplit.

## Inputs (full-file SHA-256)

Same four locked development recordings as mapping-row v0.
See [`mapping_row_v0_provenance.md`](mapping_row_v0_provenance.md).

## Implementation sources

See [`ro_ss_qda_source.sha256`](ro_ss_qda_source.sha256).

## Outputs

See [`ro_ss_qda_outputs.sha256`](ro_ss_qda_outputs.sha256).

```text
Spatial/output/ro_ss_qda/ro_ss_qda_pilot.csv
Spatial/output/ro_ss_qda/ro_ss_qda_summary.json
Spatial/output/ro_ss_qda/ro_ss_qda_d1.csv
```

Gate decisions in [`ro_ss_qda_results.md`](ro_ss_qda_results.md) are
recomputed from the CSV, not from a self-reported pass flag.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_ro_ss_qda.py -q
```

11 passed (shared \(\ell_1\) polarity, fit-only z-score, frozen \(\lambda\),
P3 off-diagonals, \(\log|\Sigma|\), equal prior, small-sample pooled, P0
centroid L1 alignment, 1-D vs 2-D table, absent slots).
