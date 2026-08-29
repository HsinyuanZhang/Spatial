# Mapping-row v0 provenance

Date: 2026-08-25.
Isolation: GT-timed events, GT unit rows, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25 with equal timestamps unsplit.

## Inputs (full-file SHA-256)

| File | SHA-256 |
|---|---|
| `new_datasets/hybridjanelia/drift16c_600s_11_filtered_gt.npz` | `b8d2b3cf8ea9b9730750bffcdb3b338179ff04e826e60670e5bd758360290c72` |
| `new_datasets/hybridjanelia/static16c_600s_11_filtered_gt.npz` | `537b0904b2f03643887723ef5fb53f5aaab129195f5187b454fcce9357e68644` |
| `synthetic_spike_dataset/generated/geoosort_npz_v1/rec_v1_units10_snr5_seed202601.npz` | `53049aeae5d4aa1d67c8438a122028465926ae4b454e4babbe0eb9af92f8ff3c` |
| `synthetic_spike_dataset/generated/geoosort_npz_v1/rec_v1_units20_snr5_seed202601.npz` | `db9e9434334942fce8ed6963cb561f5adacca4e4e38b70dd9aeccca5905100d9` |

The two HJ hashes match the locked temporal-sketch pilot list in
`experiments/causal_temporal_selection.py`.

## Implementation sources

See [`mapping_row_v0_source.sha256`](mapping_row_v0_source.sha256).

## Outputs

See [`mapping_row_v0_outputs.sha256`](mapping_row_v0_outputs.sha256).

```text
Spatial/output/mapping_row_v0/mapping_row_v0_pilot.csv
Spatial/output/mapping_row_v0/mapping_row_v0_pilot_summary.json
```

Gate decisions in [`mapping_row_v0_results.md`](mapping_row_v0_results.md)
are recomputed from the CSV, not from a self-reported pass flag.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_mapping_row.py -q
```

11 passed (interval interior, J5 censoring, `g`/quality excluded from `S`,
calibration-only `tau`, non-contiguous unit ids).
