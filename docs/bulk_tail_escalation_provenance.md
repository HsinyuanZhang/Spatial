# Bulk / tail escalation provenance

Date: 2026-08-25.
Isolation: GT-timed events, GT unit rows, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25 with equal timestamps unsplit.

## Inputs

Same four locked development recordings as mapping-row v0.
See [`mapping_row_v0_provenance.md`](mapping_row_v0_provenance.md).

## Implementation sources

| File | SHA-256 |
|---|---|
| `Spatial/experiments/run_bulk_tail_escalation.py` | `0ad2493ff650e79a05148bd1d47f65a902c7b8d2e7b4ca62d2538c3ca505bc59` |

D9 codes come from `AdaptiveRangeSearch` (5-bit, fixed unit interval) on
`prepare_dataset_events` KNN D9. `com_posneg` codes come from
`assemble_arm(..., "com_posneg", posneg_bits=4)` on the relative K=7 patch.

## Outputs

| File | SHA-256 |
|---|---|
| `Spatial/output/bulk_tail_escalation/bulk_tail_escalation_pilot.csv` | `53b5a73b85c24c17d70a072e60d06295fa7bc2d2c9addeff6e1f9a699c2bf8f1` |
| `Spatial/output/bulk_tail_escalation/bulk_tail_escalation_summary.json` | `f952f3f2ca268d535be4b061f4ac20d992c83677e5f4260b55e9015998b9c13c` |
| `Spatial/output/bulk_tail_escalation_com_posneg/bulk_tail_escalation_pilot.csv` | `b1c91c75fc8cfa9bd294800646b20d3502925f7c081c66fa33bc36eee8de186a` |
| `Spatial/output/bulk_tail_escalation_com_posneg/bulk_tail_escalation_summary.json` | `4ff05d059e626e630826542dca01bcd50f30f9344759fbf40d1a8eeaa7061e73` |
