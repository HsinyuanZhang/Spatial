# Low-bit spatial CiM/CAM sorting — results

Status: development evidence complete; gates evaluated from integer hit counts.
Date: 2026-07-29.
Plan: [`spatial_cim_sorting_plan.md`](spatial_cim_sorting_plan.md).

Selected configuration (from Stage A3, G3 winner):
`B=5`, `com_and_p2p` D=9, stability power-of-two weights (`top3_x4`), **no theta**,
all lanes enabled. Row payload = 45 centroid + 18 shift = **63 bits/unit**.

## Gate scorecard

| Gate | Result | Evidence |
|---|---|---|
| **G1** low-bit sufficiency | **FAIL** | HJ: B5 0.8136 vs float 0.8117 (+0.19 pp) pass; MEArec: 0.7129 vs 0.7265 (−1.36 pp) fail |
| **G2** spatial sufficiency (uniform) | **FAIL** | HJ: +4.68 pp over single-channel teacher pass; MEArec: −4.22 pp fail |
| **G3** adaptive gain | **PASS** | Best `weight_stability_top3_notheta`: HJ +10.19 pp, MEArec +12.00 pp |
| **G4** attribution | **PASS** | COM-shuffle drop 15–36 pp; P2P-shuffle drop 16–24 pp on all four pilots |
| **G5** drift online | **FAIL** | Fixed adapter: online −3.0 pp vs frozen; recovers none of the 15.8 pp gap; oracle-refit still only +4.3 pp |
| **G6** hardware | **PASS** | Op counts reconcile exactly; no forbidden ops; Range-CAM + indexed-SRAM baselines emitted |

Honest reading: the headline that survives is **G3 + G4 + G6**. Uniform 5-bit spatial
WTA is already competitive on HJ and loses on MEArec to a single-channel float
teacher; **per-unit power-of-two weights without theta** close that gap and more.
Attribution controls confirm the accuracy is spatial structure, not leakage.
Online drift adaptation as implemented does not help (and after correcting
two adapter bugs, self-predicted EMA is net harmful on the drift pool).

With the selected adaptive config, both families beat the single-channel teacher
(HJ 0.916 / MEArec 0.833 vs teacher 0.767 / 0.755) — but that comparison was
not the pre-registered G2 statement (G2 was against uniform B5).

## Stage A1 — bit depth (uniform, theta on, D9)

Event-weighted family means from the four-recording pilot:

| Bits | HJ acc | MEArec acc | Bits/row |
|---:|---:|---:|---:|
| 2 | 0.518 | 0.370 | 23 |
| 3 | 0.733 | 0.620 | 33 |
| 4 | 0.784 | 0.689 | 44 |
| **5** | **0.814** | **0.713** | **54** |
| 6 | 0.808 | 0.717 | 64 |
| float | 0.812 | 0.727 | 640 |

Cliff is between 2 and 3 bits. B=5 is within 1 pp of float on HJ; MEArec misses
by 1.36 pp (G1 fail). B=4 is already close on HJ static (0.943 = B5).

## Stage A2 — feature set (B=5, uniform, theta)

Per-recording (not pooled):

| Feature | HJ drift | HJ static | MEA 10u | MEA 20u | Bits/row |
|---|---:|---:|---:|---:|---:|
| com_only | 0.762 | 0.875 | 0.664 | 0.600 | 16 |
| soft_loc | 0.833 | 0.905 | 0.686 | 0.701 | 27 |
| com_and_p2p | 0.707 | 0.943 | 0.781 | 0.679 | 54 |
| soft_loc_p2p | 0.692 | 0.937 | 0.777 | 0.653 | 64 |
| p2p_only | 0.407 | 0.773 | 0.475 | 0.341 | 43 |
| central_ratio | 0.388 | 0.776 | 0.541 | 0.354 | 43 |

COM / soft_loc carry most of the signal. P2P alone is weak. Adding P2P to COM
helps MEArec-10 and HJ-static, hurts HJ-drift under uniform+theta.

## Stage A3 — adaptive ablation (the measurement that did not exist before)

Event-weighted family means, B=5, D9:

| Configuration | HJ | MEArec | Δ vs uniform_theta |
|---|---:|---:|---:|
| uniform_theta | 0.814 | 0.713 | — |
| uniform_notheta | 0.806 | 0.833 | −0.7 / **+12.0** |
| **weight_stability_top3_notheta** | **0.916** | **0.833** | **+10.2 / +12.0** |
| weight_stability_tiered_notheta | 0.908 | 0.841 | +9.4 / +12.8 |
| weight_sep_top3_notheta | 0.884 | 0.818 | +7.1 / +10.5 |
| mask_sep_m5_notheta | 0.858 | 0.799 | +4.4 / +8.6 |
| weight_stability_top3_theta | 0.905 | 0.738 | +9.1 / +2.5 |

Key finding: **theta hurts MEArec**. The best points disable it. Power-of-two
per-unit weights with stability ranking are the cross-family winner (G3 pass).

## Stage A4 — baselines

| Baseline | HJ drift | HJ static | MEA 10u | MEA 20u | Bits/unit |
|---|---:|---:|---:|---:|---:|
| Single-channel float64 teacher | 0.649 | 0.910 | 0.853 | 0.706 | 2048 |
| Multi-channel float 7×64 | 0.860 | 0.966 | 0.937 | 0.862 | 28672 |
| Selected spatial WTA (63 bit) | 0.884* | — | — | — | 63 |

\*Corpus drift mean under selected config; see Stage A corpus.

The multi-channel float ceiling is high. The spatial claim is matching a large
fraction of that ceiling at ~450× fewer bits than the neighborhood float
template, and beating the single-channel teacher under the selected config.

## Stage A corpus (selected config, 12 HJ + 20 MEArec)

Never pooled into one number:

### HJ by condition

| Condition | Acc | Events | Recordings |
|---|---:|---:|---:|
| static | 0.855 | 5950 | 6 |
| drift | 0.813 | 8251 | 6 |

### MEArec by units × SNR

| | SNR2 | SNR3 | SNR5 | SNR8 | SNR12 |
|---|---:|---:|---:|---:|---:|
| 10 units | 0.784 | 0.890 | 0.940 | 0.951 | 0.945 |
| 20 units | 0.541 | 0.667 | 0.787 | 0.821 | 0.851 |

Dense low-SNR is the weak regime (as predicted). High-SNR 10-unit MEArec
reaches 0.94–0.95 at 63 bits/unit.

## Stage B — online drift (G5)

| Arm | Accuracy (6 drift scenes) |
|---|---:|
| Frozen | 0.689 |
| Online EMA (lr_shift=4, trunc-toward-zero, sequential) | 0.660 |
| Oracle-refit upper bound | 0.732 |
| Static reference | 0.847 |

**Bug fix (2026-07-29).** The first G5 run reported online ≡ frozen to machine
precision (recovered 0.0 pp). That was not a scientific negative: (1) all
test events were scored against the initial centroids in one vectorized pass
before any write applied, so the online arm was identical to frozen by
construction; (2) arithmetic `>>` on signed residuals is a one-way ratchet
(small positive errors truncate to 0, small negative errors to −1). Both are
fixed: predict-then-update is strictly sequential, and deltas use
trunc-toward-zero (`sign(e)·(|e|>>s)`).

**Re-run.** Online is **−3.0 pp** vs frozen (0.660 vs 0.689). Per-scene
`online−frozen` ranges from −8.0 pp to +3.0 pp; centroid writes remain rare
under `lr_shift=4` (most residuals still under the trunc grain). Oracle-refit
still recovers only 4.3 of the 15.8 pp static−drift gap, so the registered
half-gap target (7.9 pp) remains above the oracle ceiling under this protocol.
Self-predicted integer EMA does not solve drift here; the corrected run shows
it can actively hurt via wrong-label feedback.


## Stage C — unsupervised (quantized SOM + Hungarian)

| Family | Acc | Reference float unsup | Holds line (−2 pp)? |
|---|---:|---:|---|
| HJ (2 pilots) | 0.756 | 0.752 | Yes |
| MEArec (2 pilots) | 0.660 | 0.754 | No |

HJ holds the unsupervised reference. MEArec does not — consistent with the
dense-unit weakness under GT-aligned protocol as well.

## Artifacts

| Path | Role |
|---|---|
| `output/spatial_cim_sorting_pilot/` | Stage A pilot CSV + G1–G3 gates |
| `output/spatial_cim_sorting_corpus/` | Full corpus, per-condition |
| `output/spatial_cim_attribution/` | G4/G6 (see attribution doc) |
| `output/spatial_cim_online_drift/` | G5 |
| `output/spatial_cim_unsupervised/` | Stage C |
| `docs/spatial_cim_sorting_source.sha256` | Source digests |
| `docs/spatial_cim_sorting_outputs.sha256` | Output digests |

## What this supports for the paper

1. **Spatial matching is real** (G4): destroying COM or P2P event-alignment
   collapses accuracy; chance baseline drops 61–92 pp.
2. **Adaptive low-bit weights work as a classifier** (G3): the same machinery
   that failed as a traffic prefilter gains +10–12 pp when scored on accuracy.
3. **CiM/CAM primitive set is sufficient** (G6): 63-bit rows, shift-add only.
4. **Do not claim** uniform 5-bit matches float on MEArec (G1 fail), uniform
   beats the temporal teacher on MEArec (G2 fail), or that online EMA fixes
   drift (G5 fail; corrected adapter is −3.0 pp vs frozen on the drift pool).
