# Deformable spatial template — Phase 1 results

The combined deployment + result note is
[`deformable_spatial_template.md`](deformable_spatial_template.md).
This file is the result-only copy.

Status: **Phase-1 stop. Per-event gain / fractional offset / polarity / residual
\(h\) raise reconstruction \(R^2\) and lower same-home assignment. Causal
tracking (Phase 2) is not licensed.**
Date: 2026-08-27.
Isolation: GT events, GT rows, zero-phase bandpass, chronological 50/25/25.
Analog POSNEG only. Default frame = **candidate-home** geometry, \(K\in\{7,12,16\}\).
Not online, not a memory row, not 0.99/4× traffic.

Command:

```bash
python -m Spatial.experiments.run_deformable_spatial_static --pilot --duration 60
```

CSV / JSON: `Spatial/output/deformable_spatial_static/`.
Plan: [`deformable_spatial_template_plan.md`](deformable_spatial_template_plan.md).
Frozen \(\lambda_{\mathrm{state}}=1.0\) from C4/K=12 calibration.

MEA 10-unit has **no co-home rivals**; same-home numbers below for MEArec are
the 20-unit recording. Family MEA `same_home_accuracy` in the JSON mixes that
NaN with the 20-unit file — use the per-recording table.

## Stop conditions that fired

1. Gain, offset, polarity, and \(h\) all **raise true-unit \(R^2\)** and
   **cut assignment / same-home margin** on MEArec, and offset does the same
   on HJ at K=7. That is “better fit, worse sorting”.
2. Gain helps HJ and **hurts MEArec** (K=7: 0.913→0.619 on the 20-unit file).
3. C5/C6 vs C4: \(R^2\) up, assignment flat or down; extra latent dimension is
   nuisance.
4. Event-argmax recentering is **worse** than candidate-home on HJ C0
   (K=12: 0.821 vs 0.936). The default frame is correct; it was not erasing
   the identity we care about.

Phase 2 (T0–T3, controlled perturbations) is **not run**. A tracker on a
state that already fails held-out classification would only fit wrong units
faster.

## Main ladder, event-weighted assignment (candidate-home)

| Arm | K | HJ acc | HJ home acc | HJ \(M_{\mathrm{home}}\) p10 | HJ \(R^2\) | MEA acc | MEA 20u home acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C0 static** | 7 | 0.931 | 0.925 | 3.70 | 0.67 | 0.925 | 0.937 |
| C0 | 12 | 0.946 | 0.918 | 1.90 | 0.70 | 0.923 | 0.940 |
| **C0** | **16** | **0.954** | **0.931** | 1.47 | 0.70 | **0.923** | **0.937** |
| C1 gain | 7 | 0.934 | 0.937 | **6.13** | 0.83 | **0.677** | 0.546 |
| C1 | 16 | **0.969** | **0.968** | 3.02 | 0.79 | 0.860 | 0.776 |
| C2 offset | 7 | **0.659** | 0.661 | 2.84 | 0.82 | 0.834 | 0.837 |
| C2 | 16 | 0.943 | 0.912 | 1.07 | 0.79 | 0.828 | 0.866 |
| C3 gain+offset | 16 | 0.957 | 0.950 | 1.75 | 0.84 | 0.656 | 0.613 |
| C4 +polarity | 16 | 0.956 | 0.950 | 1.80 | 0.85 | 0.654 | 0.597 |
| C6 +\(h\in\mathbb{R}^2\) | 16 | 0.954 | 0.945 | 1.10 | **0.87** | 0.585 | 0.557 |

C0 in the candidate-home frame is the classification object. Every added
state degree of freedom trades identity for reconstructability, hardest on
MEArec.

## K = 7 / 12 / 16

- **C0 accuracy vs K (HJ):** 0.931 → 0.946 → 0.954. More geometry helps the
  static template on the 16-site probe (K=16 is the whole probe).
- **C0 accuracy vs K (MEA):** 0.925 → 0.923 → 0.923. Extra channels past 7
  do not separate 10–20 synthetic units on 32 sites.
- **C0 \(M_{\mathrm{home}}\) p10 vs K (HJ):** 3.70 → 1.90 → 1.47. Larger K
  **shrinks** the p10 margin even while accuracy rises — peripheral slots
  add variance, not a safer co-home gap.
- Deformable arms become less catastrophic as K grows on HJ (C2 0.66→0.94)
  because a 16-slot 1-D shift cannot hide as easily when every electrode is
  present. They still fail MEArec.

Do not select K=16 as a deformable-state win. Select C0; report K as a
static-template width, with HJ preferring 16 and MEArec saturating at 7.

## Attribution (K=12, candidate-home)

Add-one from C0 / leave-one from C6, event-weighted HJ / MEA acc:

| Group | Add-one arm | HJ Δacc vs C0 | MEA Δacc vs C0 | HJ Δ \(M_{\mathrm{home}}\) p10 | \(R^2\) |
|---|---|---:|---:|---:|---|
| gain | C1 | **+1.6 pp** | **−11.6 pp** | +1.85 | up |
| offset | C2 | −4.2 pp | −8.8 pp | −0.45 | up |
| polarity (no offset) | A_polarity | +1.6 pp | −12.2 pp | +1.85 | up |
| \(h_1\) only | A_h1 | −0.7 pp | −0.4 pp | ~0 | slight up |
| C6 full | C6 | −3.2 pp | −39.3 pp | −0.7 | up |
| drop offset from C6 | L6_nooffset | +1.8 vs C6 | +18 pp vs C6 | recovers | still < C0 on MEA |
| drop gain from C6 | L6_nogain | −8.0 vs C0 | −9 pp vs C0 | down | |
| C3 + COM/ΔCOM | C3_G2 | ~0 vs C3 | +3 pp vs C3 | ~0 | — |
| C3 + spread/skew/sharp | C3_G4 | +2.4 pp vs C3 (HJ) | +5.6 pp vs C3 | +0.4 | still < C0 |

Leave-one-out: removing **offset** is the largest rescue of C6. Removing
gain or polarity does not restore MEArec. Shape statistics on top of C3
cannot beat C0.

Event-argmax C0 is −12.5 pp HJ vs candidate-home C0. Do not recenter on
the current peak channel.

## Identity vs state (Phase-1 recommendation)

| Quantity | Identity discrimination | Same-home separation | Reconstruction / would-be tracking | Static stability | Recommendation |
|---|---|---|---|---|---|
| raw POSNEG in \(h_u\) frame (C0) | **high** | **high** | low \(R^2\) is expected | high | **identity-dominant; keep** |
| global gain \(a_t\) | HJ yes, MEA no | HJ p10 up, MEA collapse | \(R^2\) up | hurts MEA static | **drop as free per-event state**; bake mean scale into \(\mu_u\) |
| fractional offset \(\delta_t\) | low | p10 down | \(R^2\) up | hurts HJ static too | **harmful** (wrong units shift onto the event) |
| polarity \(\rho_t\) | same pattern as gain | MEA down | \(R^2\) up | hurts MEA | **drop as free state** |
| COM / ΔCOM | weak | weak | — | — | **redundant** |
| spread / skew / sharp | small HJ help on a bad C3 | small | — | — | **redundant**; not a tracking substitute |
| \(h_1,h_2\) | none / negative | p10 down | \(R^2\) up | — | **harmful nuisance** |
| event-argmax frame | lower | mixed | lower \(R^2\) | — | **do not use** |

Four-way split:

- **Identity-dominant:** analog POSNEG gathered on the candidate’s reference
  home, K=7 (MEA) to K=16 (HJ).
- **State-dominant:** none licensed. Nothing raised \(R^2\) without paying
  same-home assignment.
- **Useful for both:** none jointly across HJ and MEArec.
- **Redundant / harmful:** per-event \(\delta\), \(h\), free \(a_t\) on
  MEArec, event-argmax recentering, extra COM/shape scalars.

The hypothesis

\[
x_{u,t}=F(\mu_u,s_{u,t})
\]

is **not supported** as a classification model on this development set:
the \(s\) that explain within-unit variance are exactly the \(s\) that let
a co-home impostor fit the event. Continuity of state was not even needed
for that failure — independent per-event state search already does it.

## What this does not show

- Causal tracking error, identity-drift ratio, or controlled gain/shift GT
  (Phase 2 not licensed).
- Online \(\mu_u\) updates.
- Dense NP / Yger occupancy (different packing).
- That POSNEG is a bad observation: C0 assignment ~0.93–0.95 is strong.
  The failure is **free deformation**, not the features.

## Next

Do not tune \(\lambda\), \(\delta\) grids, or \(\dim h\). Do not start T2/T3
to “save” C6. If a later knife wants trackable state, it needs a constraint
that a wrong unit **cannot** use \(\delta\) or \(a\) — Phase 1 shows
score-level \(\lambda_{\mathrm{state}}\) at 1.0 is not that constraint.
