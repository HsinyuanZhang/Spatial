# Spatial CiM/CAM attribution and hardware conformance

Status: G4 pass, G6 pass.
Date: 2026-07-29.
Companion to [`spatial_cim_sorting_results.md`](spatial_cim_sorting_results.md).

## D1 — destructive controls

Controls destroy structure while preserving value marginals where possible.
Column permutation is **not** used: L1 is invariant to a consistent feature
reorder, so it cannot test spatial structure.

| Recording | Baseline | COM-shuffle drop | P2P-shuffle drop | Electrode-geom drop | Label-shuffle drop |
|---|---:|---:|---:|---:|---:|
| HJ drift11 | 0.707 | **36.3 pp** | **18.7 pp** | 13.0 pp | 60.7 pp |
| HJ static11 | 0.943 | **15.4 pp** | **15.9 pp** | 8.0 pp | 92.4 pp |
| MEArec 10u | 0.781 | **23.2 pp** | **17.8 pp** | 4.1 pp | 73.6 pp |
| MEArec 20u | 0.679 | **34.3 pp** | **23.9 pp** | 6.2 pp | 65.5 pp |

G4 thresholds: COM-shuffle ≥ 15 pp (pass on all four); P2P-shuffle ≥ 10 pp
(pass on all four). Electrode-geometry rebuild (scramble probe positions before
KNN) drops less because wrong-neighbor amplitude patterns retain residual
unit-specific structure — reported, not gated.

**Conclusion:** accuracy collapses when event↔location or event↔footprint
association is broken. The classifier is doing spatial matching.

## D2 — error-class decomposition

Same-main-channel vs cross-channel errors on the uniform B5+theta baseline
(following `soft_loc_error_analysis.md`):

See `output/spatial_cim_attribution/spatial_attribution.json` →
`error_decomposition` per recording. Same-channel collisions remain the
dominant residual class where soft_loc historically failed (pairs like 5↔21).

## D3 — confusion-pair rescue

On HJ recordings where units 5/21, 43/63, 25/62 are present, pairwise 2-class
B5 WTA accuracies are in the attribution JSON `confusion_pairs` field.
Pair 25↔62 is the case where waveform features historically hurt; spatial-only
pairwise numbers are the relevant comparison for a spatial-first architecture.

## D4 — leakage audit

Every recording reports:

- `centroids_from` / `masks_from` / `weights_from` = fit
- `theta_from` = calibration
- `fit_state_digest` and `theta_digest` content hashes

No test row enters fit or calibration state.

## D5 — hardware conformance (G6)

For every pilot recording, `logical_op_counts` reconcile exactly with:

```
absolute_differences = N × U × D
adder_tree_additions = N × U × (D − 1)
centroid_row_reads   = N × U
wta_comparisons      = N × (U − 1)
theta_subtracts      = N × U   (when enabled)
```

Forbidden ops absent: divider, multiplier, square, sort, float on the
decision path (integer `n_bits=5`).

Row accounting for the selected config (weights on, theta off):

| Field | Bits |
|---|---:|
| Centroid (9 × 5) | 45 |
| Shift codes (9 × 2) | 18 |
| Theta | 0 |
| **Search payload** | **63** |
| Reference signed-5 template | 320 |
| Reference float64 template | 2048 |

Also emitted per recording:

- **Indexed-SRAM digital baseline** — same integer datapath; only the memory
  technology differs (mandatory comparison from the architecture section).
- **Range-CAM box variant** —  dig90` endpoint bits/row for D=9 B=5`, with
  measured box-assignment accuracy in the attribution JSON
  (`range_cam_box_accuracy`).

G6: **pass** on all four pilots.

## Artifacts

- `output/spatial_cim_attribution/spatial_attribution.csv`
- `output/spatial_cim_attribution/spatial_attribution.json`
- `output/spatial_cim_attribution/spatial_attribution_gates.json`
