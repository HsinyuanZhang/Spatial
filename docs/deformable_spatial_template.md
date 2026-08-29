# Deformable spatial template — deployment and Phase-1 results

Status: **deployed; Phase-1 stop.** Per-event gain, fractional offset,
polarity, and residual \(h\) raise reconstruction \(R^2\) and lower
same-home assignment. Causal tracking is not licensed.
Date: 2026-08-27.

This note is the combined **deployment record** and **result**. Frozen
fields live in
[`deformable_spatial_template_plan.md`](deformable_spatial_template_plan.md).
Hashes live in
[`deformable_spatial_template_provenance.md`](deformable_spatial_template_provenance.md).
A shorter result-only copy is
[`deformable_spatial_template_results.md`](deformable_spatial_template_results.md).

Isolation: GT events, GT rows, zero-phase 300–6000 Hz, chronological
50/25/25, analog POSNEG only. Not online, not a memory row, not 0.99/4×
traffic, not a reopening of Mapping Row or interval QDA.

---

## 1. Hypothesis

A neuron’s POSNEG footprint is not one fixed mean. Write it as

\[
x_{u,t}
=
F\big(
\underbrace{\mu_u}_{\text{identity}},
\underbrace{s_{u,t}}_{\text{trackable state}}
\big)
=
a_t\,
P(\rho_t)\,
S(\delta_t)
\big(\mu_u + B h_t\big)
+\epsilon_t.
\]

The model must do both:

1. Different neurons on the same home electrode stay separable in
   **identity**.
2. Gain, anchor shift, probe drift, polarity change, and mild shape
   change of the *same* neuron go into a continuous, online-updatable
   **state**, not into a rewrite of \(\mu_u\).

This is not a drift-invariant embedding. Drift should remain visible in
\(s_t\). The question is not “which feature has the highest accuracy”,
but which POSNEG-derived quantities carry identity, which describe
time-varying state, which do both, and which only help the wrong unit
fit the event.

---

## 2. What was deployed

New files only. Mapping Row, interval QDA, and other killed runners were
not rewritten.

| Path | Role |
|---|---|
| `algorithms/deformable_spatial_template.py` | Geometry-K gather, \(S(\delta)\), polarity scales, shared \(B\), scoring |
| `experiments/run_deformable_spatial_static.py` | Phase-1 C0–C6, \(K\in\{7,12,16\}\), add-one / leave-one |
| `experiments/run_deformable_spatial_tracking.py` | Phase-2 T0–T3 **refuses** (gated) |
| `experiments/run_deformable_spatial_controlled_drift.py` | Pool C perturbations **refuses** (gated) |
| `tests/test_deformable_spatial_template.py` | 10 contract tests, no recording I/O |
| `docs/deformable_spatial_template_plan.md` | Frozen fields |
| `docs/deformable_spatial_template_results.md` | Result-only copy |
| `docs/deformable_spatial_tracking_results.md` | Phase 2 not run |
| `docs/deformable_spatial_template_provenance.md` | SHA-256 |

Output (gitignored CSV/JSON):

```text
Spatial/output/deformable_spatial_static/deformable_spatial_static_pilot.csv
Spatial/output/deformable_spatial_static/deformable_spatial_static_summary.json
```

### Model pieces (as implemented)

**Observation.** Analog \(q^+,q^-\) only. No waveform, dyadic, or pair-tap.

**K from geometry, not amplitude.** For each reference home, take the K
nearest electrodes by Euclidean distance, then order slots by signed
primary-axis offset then secondary. \(K\in\{7,12,16\}\). On the 16-site
HJ probe, K=16 is the whole probe.

**Default frame: candidate-home.** Fit majority home \(h_u\) is frozen.
Scoring unit \(u\) gathers POSNEG on the physical neighbourhood of
\(h_u\), so a 0.3-pitch peak walk stays in the observation and can be
represented by \(\delta_t\). Event-argmax recentering is an ablation
only (C0 and C3 at K=12).

**Identity \(\mu_u\).** Mean analog POSNEG of that unit’s fit events in
the \(h_u\) frame. Not updated per spike.

**Gain.** \(a_t=\mathrm{e}^{\alpha_t}\). Closed-form least squares on
live slots. A global scale \(x\to 1.2x\) should move \(a_t\), not
\(\mu_u\).

**Fractional offset.** \(\delta_t\) along the primary probe axis, search
range \(\pm 1.5\) pitch, 21-point grid. \(S(\delta)\) is inverse-distance
interpolation of the template field sampled at slot coordinates shifted
by \(-\delta\) (fractional, not integer electrode hops).

**Polarity.** \(\alpha=\log a\) common; \(\rho=\log(a_+/a_-)\);
\(a_+=\mathrm{e}^{\alpha+\rho/2}\), \(a_-=\mathrm{e}^{\alpha-\rho/2}\).

**Residual shape.** Shared \(B\) from SVD of fit residuals *after*
per-event gain/offset/polarity. \(\dim h\in\{0,1,2\}\). No unit-specific
\(B_u\) in this knife.

**Score (Phase 1).** For each test event, independently

\[
D_u(x)
=
\min_s
\big[
D_{\mathrm{obs}}\big(x,F(\mu_u,s)\big)
+
\lambda_{\mathrm{state}}
D_{\mathrm{dyn}}(s,s_u^{\mathrm{rest}})
\big],
\]

with \(s=[\alpha,\delta,\rho,h]\). \(D_{\mathrm{obs}}\) is mean squared
error on live slots. \(D_{\mathrm{dyn}}\) is a quadratic penalty toward
the **fit rest state**, not a causal \(s_{t-1}\). One global
\(\lambda_{\mathrm{state}}\) from \(\{0,0.3,1\}\) on C4 / K=12
calibration (assignment accuracy, then home-margin p10). Frozen value:
**1.0**.

Lower \(D_u\) wins. Same-home margin is
\(M_{\mathrm{home}}=D_{\mathrm{nearest\ wrong,\ same-home}}-D_{\mathrm{true}}\).

### Phase-1 arms

| ID | Object |
|---|---|
| C0 | static \(\|x-\mu_u\|_2\) (identity only) |
| C1 | gain \(a\) |
| C2 | offset \(\delta\) |
| C3 | gain + offset |
| C4 | + polarity \(\rho\) |
| C5 | + shared \(h\in\mathbb{R}^1\) |
| C6 | + shared \(h\in\mathbb{R}^2\) |
| C0_argmax / C3_argmax | event-argmax frame, K=12 only |
| A_polarity, A_h1 | add-one |
| L6_nogain, L6_nooffset, L6_nopol | leave-one from C6 |
| C3_G2, C3_G4 | C3 plus COM/ΔCOM or spread/skew/sharp distances |

### Data and split

Locked four 60 s development recordings (same as Mapping Row / QDA):

- HJ `drift16c_600s_11`, `static16c_600s_11`
- MEArec `rec_v1_units10_snr5_seed202601`, `rec_v1_units20_snr5_seed202601`

Chronological 50% fit / 25% calibration / 25% test; equal timestamps
unsplit. No `--confirm`. CPU only.

### How to rerun

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  Spatial/tests/test_deformable_spatial_template.py -q
python -m Spatial.experiments.run_deformable_spatial_static \
  --pilot --duration 60
```

Phase 2 entry points exit nonzero with the stop text:

```bash
python -m Spatial.experiments.run_deformable_spatial_tracking --pilot
python -m Spatial.experiments.run_deformable_spatial_controlled_drift --pilot
```

---

## 3. What ran vs what was gated

| Stage | Status |
|---|---|
| Tests | 10 passed |
| Phase 1 static ceiling (C0–C6, K=7/12/16, attribution) | **ran** (~3.5 min CPU) |
| Phase 2 T0 frozen / T1 oracle / T2 self-track / T3 slow \(\mu\) | **not run** (gated) |
| Pool C controlled gain/shift/polarity/\(h\) | **not run** (gated) |
| Confirmation set | not opened |
| Overleaf / abstract numbers | not changed |

A tracker on a state that already fails held-out classification would
only fit the wrong unit faster. Spec: do not start T2, and do not use T3
to rescue T2.

---

## 4. Phase-1 results

MEA 10-unit has **no co-home rivals**. MEA same-home numbers are the
20-unit recording. Do not use the JSON family `same_home_accuracy` for
MEA (it mixes the empty 10-unit file).

### Stop conditions that fired

1. Gain, offset, polarity, and \(h\) all **raise true-unit \(R^2\)** and
   **cut assignment / same-home margin** on MEArec; offset does the same
   on HJ at K=7. Better fit, worse sorting.
2. Gain helps HJ and **hurts MEArec** (K=7, 20-unit: 0.913 → 0.619).
3. C5/C6 vs C4: \(R^2\) up, assignment flat or down. Extra latent
   dimension is nuisance.
4. Event-argmax recentering is **worse** than candidate-home on HJ C0
   (K=12: 0.821 vs 0.936). The default frame is correct.

### Main ladder (candidate-home, event-weighted)

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

C0 in the candidate-home frame is the classification object. Added state
trades identity for reconstructability, hardest on MEArec.

### Per recording, C0 vs the worst deformable miss (K=7)

| Recording | C0 acc | C1 acc | C2 acc | C6 acc | C0 \(R^2\) | C6 \(R^2\) |
|---|---:|---:|---:|---:|---:|---:|
| HJ drift11 | 0.913 | 0.915 | **0.577** | 0.676 | 0.50 | 0.91 |
| HJ static11 | 0.951 | 0.957 | 0.759 | 0.828 | 0.85 | 0.95 |
| MEA 10u | 0.949 | 0.792 | 0.904 | 0.598 | 0.83 | 0.93 |
| MEA 20u | 0.913 | **0.619** | 0.798 | 0.413 | 0.80 | 0.92 |

### K = 7 / 12 / 16

- C0 accuracy vs K, HJ: 0.931 → 0.946 → 0.954. K=16 is the whole 16-site
  probe; more geometry helps the **static** template.
- C0 accuracy vs K, MEA: 0.925 → 0.923 → 0.923. Extra channels past 7 do
  not separate 10–20 synthetic units on 32 sites.
- C0 \(M_{\mathrm{home}}\) p10 vs K, HJ: 3.70 → 1.90 → 1.47. Larger K
  **shrinks** the p10 margin even while accuracy rises. Peripheral slots
  add variance, not a safer co-home gap.
- Deformable arms become less catastrophic as K grows on HJ (C2
  0.66 → 0.94) because a full-probe 1-D shift cannot hide as easily.
  They still fail MEArec.

Do not select K=16 as a deformable-state win. Select C0. Report K as
static-template width: HJ prefers 16, MEArec saturates at 7.

### Attribution (K=12, candidate-home)

| Group | Arm | HJ Δacc vs C0 | MEA Δacc vs C0 | HJ Δ \(M_{\mathrm{home}}\) p10 | \(R^2\) |
|---|---|---:|---:|---:|---|
| gain | C1 | **+1.6 pp** | **−11.6 pp** | +1.85 | up |
| offset | C2 | −4.2 pp | −8.8 pp | −0.45 | up |
| polarity (no offset) | A_polarity | +1.6 pp | −12.2 pp | +1.85 | up |
| \(h_1\) only | A_h1 | −0.7 pp | −0.4 pp | ~0 | slight up |
| C6 full | C6 | −3.2 pp | −39.3 pp | −0.7 | up |
| drop offset from C6 | L6_nooffset | +1.8 vs C6 | +18 pp vs C6 | recovers | still &lt; C0 on MEA |
| drop gain from C6 | L6_nogain | −8.0 vs C0 | −9 pp vs C0 | down | |
| C3 + COM/ΔCOM | C3_G2 | ~0 vs C3 | +3 pp vs C3 | ~0 | — |
| C3 + spread/skew/sharp | C3_G4 | +2.4 pp vs C3 (HJ) | +5.6 pp vs C3 | +0.4 | still &lt; C0 |

Removing **offset** is the largest rescue of C6. Shape statistics on top
of C3 cannot beat C0. Event-argmax C0 is −12.5 pp HJ versus
candidate-home C0.

---

## 5. Identity vs state (the table this knife was for)

| Quantity | Identity | Same-home | Reconstruction / would-be tracking | Static stability | Recommendation |
|---|---|---|---|---|---|
| raw POSNEG in \(h_u\) frame (C0) | **high** | **high** | low \(R^2\) is expected | high | **identity-dominant; keep** |
| global gain \(a_t\) | HJ yes, MEA no | HJ p10 up, MEA collapse | \(R^2\) up | hurts MEA static | **drop as free per-event state**; bake mean scale into \(\mu_u\) |
| fractional offset \(\delta_t\) | low | p10 down | \(R^2\) up | hurts HJ static too | **harmful** (wrong units shift onto the event) |
| polarity \(\rho_t\) | same pattern as gain | MEA down | \(R^2\) up | hurts MEA | **drop as free state** |
| COM / ΔCOM | weak | weak | — | — | **redundant** |
| spread / skew / sharp | small HJ help on a bad C3 | small | — | — | **redundant** |
| \(h_1,h_2\) | none / negative | p10 down | \(R^2\) up | — | **harmful nuisance** |
| event-argmax frame | lower | mixed | lower \(R^2\) | — | **do not use** |

Four-way split:

- **Identity-dominant:** analog POSNEG gathered on the candidate’s
  reference home; K=7 (MEA) to K=16 (HJ).
- **State-dominant:** none licensed. Nothing raised \(R^2\) without
  paying same-home assignment.
- **Useful for both:** none jointly across HJ and MEArec.
- **Redundant / harmful:** per-event \(\delta\), \(h\), free \(a_t\) on
  MEArec, event-argmax recentering, extra COM/shape scalars.

The hypothesis \(x_{u,t}=F(\mu_u,s_{u,t})\) is **not supported** as a
classification model on this development set. The \(s\) that explain
within-unit variance are exactly the \(s\) that let a co-home impostor
fit the event. Continuity of state was not required for that failure —
independent per-event state search already does it.

POSNEG **does** carry unit identity: C0 assignment is ~0.93–0.95. The
failure is **free deformation**, not the observation.

---

## 6. What this does not show

- Causal tracking error, identity-drift ratio, or controlled gain/shift
  GT (Phase 2 not licensed).
- Online \(\mu_u\) updates.
- Dense NP / Yger occupancy (different packing).
- That a later knife with a constraint a **wrong** unit cannot use
  \(\delta\) or \(a\) is impossible. Phase 1 only shows that
  score-level \(\lambda_{\mathrm{state}}=1.0\) is not that constraint.

---

## 7. Next (stop, do not retune)

- Do not sweep \(\lambda\), \(\delta\) grids, or \(\dim h\).
- Do not start T2/T3 to “save” C6.
- If a later knife wants trackable state, it needs a hard constraint
  that an impostor cannot spend \(\delta\) or \(a\).
- For a memory row, this knife still supports **reference POSNEG
  \(\mu_u\)**, not \(\mu_u+s_t\) as a free state object.
