# Low-bit spatial CiM/CAM sorting — pre-registered plan

Status: gates registered before any corpus run.
Date: 2026-07-29.

## Claim

Low-bit spatial features with per-unit adaptive adjustment are sufficient as a
spike-sorting **classifier** under the declared CiM/CAM primitive set, and the
accuracy comes from spatial matching rather than protocol or leakage.

This reframes Level 1 from a prefilter (scored on candidate recall / traffic)
to a classifier (scored on unit accuracy). Prior masked / weighted pilots
measured recall only; they are not accuracy evidence.

## Primitive: biased integer WTA

```
d_u(q) = sum_j  2^{h_uj} * m_uj * |q_j - c_uj|     h_uj in {0,1,2}, m_uj in {0,1}
u_hat  = argmin_u [ d_u(q) - theta_u ]
```

- Datapath: D abs-diff lanes, optional mask gate, 0/1/2-bit left shift, adder
  tree, one integer subtract, WTA.
- Forbidden on the decision path: divider, multiplier, square, sort, float.
- `theta_u` from the calibration split only; centroids / masks / weights from
  the fit split only.
- Row cost: 54 bit uniform, 63 masked, 74 weighted, 83 masked+weighted, versus
  320 bit signed-5 template and 2048 bit float template.

Implementation: `algorithms/spatial_wta.py` (`SpatialWTAClassifier`).

## Pre-registered gates

| Gate | Statement | Threshold | Kill condition |
|---|---|---|---|
| **G1** low-bit sufficiency | B=5 spatial WTA within 1 pp of float spatial WTA | `|acc_5bit - acc_float| ≤ 0.01` on both HJ and MEArec family event-weighted means | Either family exceeds 1 pp |
| **G2** spatial sufficiency | B=5 spatial WTA beats single-channel float64 temporal teacher | `acc_spatial_5bit > acc_temporal_teacher` on both families | Either family loses |
| **G3** adaptive gain | Best adaptive variant beats uniform by ≥ 2 pp on **both** families | Cross-family sign agreement required | Opposite signs, or either family gain < 2 pp |
| **G4** attribution | Geometry-permutation drops ≥ 15 pp; channel-slot permutation drops ≥ 10 pp | Both controls on the pilot pair | Either control fails → headline void |
| **G5** drift | Online adaptation recovers ≥ half the static−drift gap | On HJ drift scenes vs frozen | Recovery < 50% of gap |
| **G6** hardware | Decision-path ops ⊆ allowed set; op counts reconcile exactly | `N*D` abs, `N*(D-1)` adds, `N*U` reads, `N*(U-1)` WTA | Any forbidden op or count mismatch |

Failures are reported as failures. A negative G3 still leaves G1/G2/G4 as a
publishable spatial-sufficiency result.

## Stages

### Stage A — fit-time adaptive, GT-aligned

- `experiments/run_spatial_wta_pilot.py` on the four locked pilot recordings.
- A1 bit depth `{2,3,4,5,6,float}`, fixed D=9.
- A2 feature set: `com_only`, `p2p_only`, `com_and_p2p`, `soft_loc`,
  `soft_loc+p2p`, `central_ratio`, multi-scale K∈{5,7,13}.
- A3 adaptive ablation: uniform / mask M∈{3,5,7,9} / weights / mask+weights,
  each with and without `theta_u`.
- A4 baselines: single-channel float64 teacher + strong multi-channel float
  template over the 7-neighborhood.
- Then `experiments/run_spatial_wta_corpus.py` on 12 HJ + 20 MEArec, reported
  per-condition (never pooled).

### Stage B — online adaptation

- `algorithms/online_spatial_adaptation.py` + `experiments/run_spatial_online_drift.py`.
- Integer shift-based streaming update of centroid and theta; no test labels.
- Arms: frozen / online / oracle-refit. Charge update writes.

### Stage C — unsupervised end-to-end

- `experiments/run_spatial_unsupervised.py`.
- SOM on the quantized descriptor + Hungarian protocol.
- Hold the float unsupervised line (≈0.752 HJ / 0.754 MEArec).

### Stage D — attribution and hardware

- `experiments/run_spatial_attribution.py`.
- D1 geometry / channel-slot permutation controls.
- D2 same-channel vs cross-channel error decomposition.
- D3 confusion-pair rescue (including 5↔21 and 25↔62).
- D4 leakage audit with source digests.
- D5 hardware accounting: indexed-SRAM digital comparison + Range-CAM box
  variant.

## Honest risks (stated before running)

1. Absolute target is the float spatial ceiling (≈0.871 HJ / 0.856 MEArec),
   not modern multi-channel sorter SOTA. Claim is matching that ceiling at
   54–83 bit/unit and beating the temporal pipeline.
2. Dense units (MEArec 20) are the weak regime; a 1024-channel claim needs
   tiling and no dense 2D recording exists yet.
3. G3 is the most likely gate to fail (same masks/weights already failed
   cross-family sign agreement under the recall objective).
4. Methodology text previously claimed Hungarian matching for a plain-equality
   metric; that doc bug is fixed alongside this work.

## Provenance contract

Every runner emits CSV/JSON with input hashes, source digests, and exact
integer hit/bit counts. Gate decisions are recomputed from those integers,
never from rounded display values or a self-reported pass flag.
