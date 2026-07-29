# Spatial CiM/CAM sorting — provenance

Date: 2026-07-29.

## Source digests

See [`spatial_cim_sorting_source.sha256`](spatial_cim_sorting_source.sha256).

## Output digests

See [`spatial_cim_sorting_outputs.sha256`](spatial_cim_sorting_outputs.sha256).

## Gate decisions

Recomputed from integer `n_correct` / `n_events` in:

- `output/spatial_cim_sorting_pilot/spatial_wta_pilot_gates.json` (G1–G3)
- `output/spatial_cim_attribution/spatial_attribution_gates.json` (G4, G6)
- `output/spatial_cim_online_drift/spatial_online_drift_gates.json` (G5)

Never from rounded display values or a self-reported pass flag.

## Methodology note

`DAC2027/sections/06_methodology.tex` previously claimed Hungarian-matched
accuracy for the temporal pilot; the implementation computes plain GT-aligned
equality. The text was corrected to match the code as part of this work.

## Online adapter correction (G5 re-run)

`algorithms/online_spatial_adaptation.py` had two defects that made the first
G5 online arm identical to frozen:

1. Batch scoring: all test distances/winners were computed before any centroid
   write, so updates never affected predictions in the same call.
2. Signed arithmetic right-shift: small negative residuals always produced
   `delta=-1` while small positive residuals produced `delta=0` (one-way ratchet).

Fix: sequential predict-then-update per event, and trunc-toward-zero shifts via
`OnlineSpatialAdapter.trunc_shift`. Stage B was re-run; G5 remains FAIL with
online **below** frozen (−3.0 pp). See `spatial_cim_sorting_results.md` Stage B.

