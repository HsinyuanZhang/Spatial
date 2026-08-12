# Shape-Orthogonal Spatial Descriptor — Pre-Registered Plan

Status: **Closed for this configuration. Phase A passes, Phase A2 fails on
MEArec alignment jitter, Phase B is negative. No configuration is selected and
confirmation was never authorized.**

Date: 2026-08-12.

Results: [`shape_orthogonal_descriptor_results.md`](shape_orthogonal_descriptor_results.md).
The Phase A selection `com_posneg_lat_width` at `posneg_bits=4`, `lat_bits=4`,
`amp_gate=0.125` did not survive Phase A2 and must not be cited as a selected
configuration.

Interfaces frozen before implementation in
[`shape_descriptor_interfaces.md`](shape_descriptor_interfaces.md): primitive
signatures, arm definitions, CSV schema, selection JSON schema, and an
authoritative restatement of the development gate. Changing anything there is a
protocol change and must be recorded.

## Research question

The normalized relative-offset P2P footprint appears to have saturated: four
independent knobs (bit width, L1 vs cosine, shift radius, temporal averaging)
each move nearest-template assignment by less than 1.5 percentage points. Is
this an information ceiling of the amplitude footprint itself, and if so, can
columns that are **physically orthogonal to distance-driven amplitude decay**
raise the ceiling while keeping one fixed-column integer row per unit, integer
L1 arithmetic, and the existing error-feedback update primitive?

The study is decomposed so that a representation result cannot be confused with
a system result:

1. establish whether the ceiling is a representation limit or a model limit;
2. choose the descriptor blocks and their bit contract while holding the patch,
   matcher, and shift radius fixed;
3. evaluate the candidate-recall/template-traffic system gate separately;
4. evaluate causal-filter and jitter robustness before any online phase.

No stage may retroactively reselect a block or bit width chosen in an earlier
stage.

## Why the current footprint saturates

The existing per-channel feature discards one of its own two operands:

```text
P2P_raw(x) = max_t x[t] - min_t x[t]
```

`max` and `min` are both computed and then collapsed into a single scalar. For
a point-like current source, the resulting normalized K-vector is close to a
monotone decay curve parameterized by axial sub-channel position, lateral
distance, and a small tilt. Per-event max normalization removes the amplitude
degree of freedom. Two units at the same location therefore produce nearly the
same footprint regardless of K or bit width, which is exactly the failure
structure recorded in [`soft_loc_error_analysis.md`](soft_loc_error_analysis.md)
for the HJ 5–21 and 43–63 pairs.

The saturation signature across the existing branch:

| Knob varied | Movement | Source |
|---|---:|---|
| 4-bit to float | < 0.4 pp | [`shift_conv_bit_sweep_results.md`](shift_conv_bit_sweep_results.md) |
| L1 to cosine | +1.1 pp HJ, ±0.2 pp MEArec | same |
| Shift S=0 to S=2 | +0.4 pp | [`shift_cim_results.md`](shift_cim_results.md) |
| Raw to centered-k5 estimator | ±1 pp, sign flips per recording | [`robust_p2p_template_results.md`](robust_p2p_template_results.md) |

## Starting evidence

### Unregistered exploratory diagnostic (2026-08-12)

Scripts: `output/_diag_p2p_ceiling.py`, `output/_diag_rows_per_unit.py`
(gitignored). Contract: GT events, GT unit rows, full-recording zero-phase
filtering, relative primary-axis patch K=7, chronological 50/50, first 60 s of
each recording, fixed analytic quantization ranges (no fitted scale).

**This run touched all 12 HJ scenes and all 4 MEArec SNR5 files.** It is
hypothesis generation only. Those 16 recording/time-block pairs are therefore
disqualified as confirmation data for this plan, and the confirmation design
below is built around that contamination rather than ignoring it.

Representation vs model:

| Observable | HJ (12) | MEArec SNR5 (4) |
|---|---:|---:|
| Normalized P2P participation ratio | 2.81 | 3.18 |
| Dimensions holding 95% of P2P variance (of 7) | 4.67 | 4.75 |
| P2P nearest-centroid assignment | 0.719 | 0.703 |
| P2P kNN-15 assignment, same features | 0.872 | 0.852 |

Candidate blocks, event-weighted nearest-centroid assignment:

| Descriptor | HJ | MEArec SNR5 |
|---|---:|---:|
| COM + P2P (current D9) | 0.803 | 0.806 |
| COM + split pos/neg peaks | 0.827 | 0.863 |
| COM + pos/neg + trough latency + width | **0.850** | **0.872** |
| COM + P2P + same-width random columns (control) | 0.781 | 0.795 |

Supporting observations: 16/16 recordings improved, worst case +1.43 pp; the
random-width control lost accuracy on 16/16, so the gain is not an artifact of
adding L1 mass; on the hardest quartile by top-2 margin, HJ moved 0.551 to
0.703 and MEArec 0.558 to 0.751; drift scenes gained most.

Row-budget comparison at S=0 versus the existing multi-copy bank:

| Row content | bits/unit | HJ | MEArec SNR5 |
|---|---:|---:|---:|
| D9/B5, one row | 45 | 0.803 | 0.806 |
| D9/B5, S=2 pre-shifted bank, five rows | 225 | 0.812 | 0.809 |
| COM + pos/neg + latency + width, one row | 109 | 0.850 | 0.872 |

The exploratory run carried a redundant home-slot latency column, so it stored
113 bits rather than 109. That column holds the neutral code on every event and
every template, contributes exactly zero to every L1 distance, and is dropped by
the frozen contract below; the accuracy figures are unaffected.

A k-means multi-prototype arm was also run and degraded accuracy, but its
seeding was farthest-point and therefore outlier-driven. **That arm is void and
is not evidence that multiple rows per unit fail.** The multi-row question
stays with [`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md)
and is out of scope here.

### Prior repository evidence

- Same-home-channel, near-identical-COM unit pairs dominate spatial errors, and
  main-channel waveform shape recovers about 61% of them, while waveform shape
  alone is a weak global classifier at roughly 0.48
  ([`soft_loc_error_analysis.md`](soft_loc_error_analysis.md)).
- A fixed morphology-12 vector failed badly as a Level-2 **replacement** for the
  full waveform ([`compact_waveform_refinement.md`](compact_waveform_refinement.md)).
  It was never tested as a Level-1 **addition**, which is what this plan tests.
- Unit-specific masks and power-of-two weights both produced HJ movement with no
  MEArec movement and were correctly recorded as null. Any new relation must
  therefore move both families in the same direction.

## Structured divergence

| ID | Candidate | Orthogonal to amplitude decay? | Main problem |
|---|---|---|---|
| B1 | Keep `max` and `min` as separate columns | Yes, encodes phase balance | Doubles footprint columns |
| B2 | Per-channel ratio `\|max\|/(\|max\|+\|min\|)` | Yes | Needs a per-channel divider |
| B3 | Cross-channel trough-time offsets | Yes, propagation geometry | Index registers; filter-delay sensitive |
| B4 | Home-channel trough-to-peak width | Yes, cell-type axis | One column; low sample resolution at 30 kHz |
| B5 | Absolute (un-normalized) event amplitude | Partly | Drift and impedance sensitive |
| B6 | Per-channel zero-crossing count | Yes | Noise sensitive at low SNR |
| B7 | Cross-channel peak-time offsets (`argmax`) | Yes | Positive phase is often noisier than the trough |
| B8 | Per-unit refractory / last-spike timestamp | Fully, non-waveform | Self-label feedback can cascade |
| B9 | Rise/fall slope per channel | Yes | Needs differences plus scaling |
| B10 | Larger K or finer P2P bits | **No** | Already falsified by the bit and shift sweeps |
| B11 | Per-unit learned metric (mask/weight) | No | Already null on cross-family gates |

## Convergence

Filters: physical orthogonality to distance decay, no new sample acquisition
beyond the window already read, no divider or multiplier on the decision path,
fixed column semantics, per-unit mean row updatable by the existing
error-feedback primitive, and falsifiable on both families with a built-in
random-column control.

Retained, in order:

1. **B1 split pos/neg columns** — selected. Recovers the current P2P footprint
   by addition to within one LSB (see the frozen contract), so it cannot lose
   footprint information by construction, and it costs zero additional
   acquisition because both operands already exist inside
   `estimate_p2p_amplitude`.
2. **B3 trough latency** — selected. Integer index differences only.
3. **B4 home width** — selected. One integer index difference.
4. **B2 per-channel ratio** — rejected in favor of B1: same information, but B1
   avoids a per-channel division. Retained only as a software cross-check.
5. **B8 refractory veto** — deferred to a separate study. It is a controller
   policy with self-label feedback risk, not a descriptor column, and mixing it
   in would confound this representation gate.

### Two-sentence pitch

The current descriptor throws away one of the two operands its own P2P
estimator already computes, and it never records when each channel reached its
trough, so it can only measure how far a unit is, not what shape it has or which
way it propagates. Keep the two extrema as separate columns and add integer
index differences for cross-channel trough latency and home-channel width, so
that one fixed-column integer row per unit gains a cell-type axis and a
propagation axis without a divider, a multiplier, or one extra sample read.

### Strongest objection

Latency and width are sample-index features measured after **full-recording
zero-phase** filtering. A deployable causal filter has a frequency-dependent
group delay that will distort both, and at 30 kHz a trough-to-peak duration of
0.2–0.8 ms spans only 6–24 samples, so the usable resolution may be small
enough that the exploratory gain is a zero-phase-filter artifact.

Response: this is the registered kill condition, not a caveat. Phase A2 repeats
the frozen descriptor under `causal_preprocessing.py` and under deterministic
±1/±2-sample alignment jitter, on the same event rows. Latency and width are
index *differences* inside one window, so they are invariant to a constant shift
of the event time but not to group-delay dispersion; the experiment must
separate those two effects. If the causal path removes most of the gain, the
selection is null and the branch stops before any online phase.

## Frozen feature contract

Window and patch are unchanged from the current relative/shift experiments:
`seg[j] = filtered[ch_j, t-15 : t+15)`, relative primary-axis patch `K=7`, home
at slot 3, out-of-bounds slots encoded as neutral.

Per slot `j`, with first-occurrence tie-breaking on both extrema:

```text
vmax_j = max_t seg[j,t]      imax_j = argmax_t seg[j,t]
vmin_j = min_t seg[j,t]      imin_j = argmin_t seg[j,t]
scale  = max_j (vmax_j - vmin_j)          # identical to per_spike_max today
live_j = (vmax_j - vmin_j) >= g * scale   # g frozen in Phase A
```

Descriptor blocks and their integer codes:

| Block | Coords | Bits | Definition | Neutral code |
|---|---:|---:|---|---|
| `COM` | 2 | 5 | unchanged geometry-weighted centroid | n/a |
| `POSNEG` | 2K = 14 | `b` in {4,5} | `clip(round((2^b-1)*vmax_j/scale), 0, 2^b-1)` and the same expression for `-vmin_j` | 0 |
| `LAT` | K-1 = 6 | `c` in {3,4} | `clip(imin_j - imin_home, -L, L) + L` with `L = 2^(c-1) - 1`, home slot omitted | `L` |
| `WIDTH` | 1 | 5 | `clip(imax_home - imin_home, -15, 16) + 15` | 15 |

The bit column states the sweep range; the candidate configuration carried into
Phase A2 and beyond is `b=5, c=4`.

Dead slots (`live_j` false) and out-of-bounds slots take the neutral code.
Quantization ranges are fixed analytic constants, not fitted, so no test data
can influence a scale. At the candidate `b=5, c=4` the row payload is
`2*5 + 14*5 + 6*4 + 1*5 = 109` bits per unit before row metadata.

`POSNEG` reconstructs the current footprint by addition, because
`vmax_j + (-vmin_j) = p2p_j` exactly before quantization. Independent rounding
of the two halves makes the code-domain identity hold to within one LSB:

```text
| pos_j + neg_j - round((2^b - 1) * p2p_j / scale) | <= 1
```

This holds on any slot where `vmax_j >= 0 >= vmin_j`, which is the normal case
for a 300–6000 Hz filtered 30-sample window but is not guaranteed. Both halves
clip at zero, `scale` is floored by the existing epsilon guard, and the runner
must assert the bound on conforming slots while counting and reporting the
non-conforming fraction.

A spot check across HJ static, HJ drift, MEArec SNR5, and MEArec SNR2 measured
a maximum code-domain error of exactly 1 with zero slots exceeding it, and a
non-conforming fraction of 0.0000% on HJ and at most 0.0924% on MEArec. The
registered threshold is therefore **0.5%**: above that on any development
recording, the reconstruction argument is invalid and must be resolved before
Phase A is scored.

Subject to that, the current D9 footprint is a linear function of the new block
up to one LSB, so any measured loss against D9 is attributable to rounding or
to the distance metric rather than to discarded information.

Matching is all-row integer L1 at shift radius `S=0` throughout. Shift is a
separate, already-measured axis and is held fixed so it cannot absorb credit.

## Phase A — descriptor and bit-width pilot

### Locked development recordings

| Family | Recording | Time block | Role |
|---|---|---|---|
| HJ | `drift16c_600s_11` | 0–60 s | Real drift development |
| HJ | `static16c_600s_11` | 0–60 s | Static negative control |
| MEArec | `rec_v1_units10_snr5_seed202601` | 0–60 s | Moderate-density synthetic |
| MEArec | `rec_v1_units20_snr5_seed202601` | 0–60 s | Dense synthetic |

These are the same four recordings the estimator branch used, so the
development boundary is unchanged and already hash-locked in
`dac_experiment_inputs.sha256`.

### Development grid

```text
arms   : D9                                  # control
         D9 + LAT
         D9 + WIDTH
         D9 + LAT + WIDTH                    # matched partner for the control
         D9 + RAND                           # falsification control
         COM + POSNEG
         COM + POSNEG + LAT + WIDTH          # selected candidate
bits   : POSNEG in {4,5}; LAT in {3,4}; COM and WIDTH fixed at 5
gate g : {0.125, 0.25, 0.375}
shift  : S=0 only
```

`RAND` is a seeded uniform integer block with exactly the same coordinate count
and per-coordinate code width as `LAT + WIDTH` (6 columns at `c` bits plus one
at 5 bits), carrying no unit information. It is not a baseline; it is the
built-in falsifier for "more columns means more L1 mass," and its matched
partner is the `D9 + LAT + WIDTH` arm.

### Required metrics

Per recording and configuration:

1. closed-set nearest-template assignment accuracy and integer correct count;
2. true-versus-nearest-wrong L1 margin and positive-margin fraction;
3. per-unit worst assignment accuracy;
4. normalized true-unit template distance;
5. assignment accuracy restricted to the hardest quartile by top-2 margin;
6. row payload bits, absolute-difference lane count, and accumulator width;
7. static and drift rows kept separate, and MEArec unit count kept separate;
8. delta against the D9 control on identical event rows.

### Development gate

The candidate arm qualifies only if all hold:

1. event-weighted family-mean assignment beats D9 by at least 2.0 percentage
   points on **both** HJ and MEArec;
2. no development recording loses more than 0.5 percentage points against D9;
3. positive-margin fraction does not fall on either family;
4. the `POSNEG` reconstruction bound holds on every conforming slot and the
   non-conforming fraction stays below 0.5% on every development recording;
5. the `D9 + RAND` control does **not** satisfy condition 1.

If condition 5 fails, the entire measurement is invalid and the branch stops
for re-instrumentation rather than proceeding. Among qualifying configurations,
rank by: number of safeguards passed, two-family mean assignment, hardest-
quartile accuracy, positive-margin fraction, then fewer row bits. If none
qualifies, the selection is null and shape-orthogonal columns are closed as a
Level-1 branch.

## Phase A2 — causal and jitter boundary

Run only if Phase A produces a non-null selection. Using frozen Phase A event
rows and the frozen descriptor:

1. replace full-recording `filtfilt` with `causal_preprocessing.py` streaming
   SOS filtering, with a declared warm-up guard;
2. apply deterministic ±1 and ±2-sample event-alignment jitter;
3. no-perturbation control.

Report per-block accuracy degradation. `LAT` and `WIDTH` must retain at least
half of their Phase A gain over D9 under the causal filter on both families,
and must not lose more than 1.0 percentage point under ±2-sample jitter. A
configuration that only works under zero-phase filtering does not advance, and
that outcome is recorded as the branch result rather than repaired by
reselecting `g` or the bit widths.

## Phase B — candidate recall and template traffic

Representation accuracy is not the standing system requirement. Phase B reuses
the frozen descriptor inside the existing Level-1 candidate policy and reports
the metrics the system gate is actually written against:

- true-unit candidate recall;
- zero-, one-, and multi-candidate event fractions;
- mean candidate count and its aggregation mode;
- row comparisons including fallback;
- distinct waveform-template reads and the reduction factor against full scan.

The system gate remains 0.99 candidate recall with at least 4x template-read
reduction on both families, evaluated jointly. Phase B is expected to be
informative even if it fails; a Phase A pass must never be reported as progress
on this gate.

## Phase C — online update

Run only if Phase A and A2 pass. Every block is a per-unit unsigned integer
mean row, so `online_p2p_templates.py` applies unchanged: same error-feedback
accumulator, same confidence gate, same rollback anchor, same batched commit.

The new, falsifiable hardware hypothesis is that the index-difference columns
`LAT` and `WIDTH` are pure sample-index differences, so they are invariant to
gain drift, impedance change, and amplitude adaptation, and are affected by
positional drift only when it changes propagation geometry. `POSNEG` is
amplitude-derived and inherits the positional sensitivity of the current
footprint. If so, `LAT` and `WIDTH` should tolerate a longer commit period than
`POSNEG` at equal accuracy.

Measure per-block template displacement from the frozen anchor over time and
per-block committed nonvolatile row writes, and test that longer-period claim
directly. No GT label may enter the update API; GT audits accepted updates only
after the run.

## Confirmation boundary

The exploratory diagnostic already read the first 60 s of all 12 HJ scenes and
all 4 MEArec SNR5 files, so confirmation must use data that run never saw:

| Confirmation set | Content | Independence |
|---|---|---|
| A: HJ held-out scenes and time | the 10 non-development HJ scenes, 300–360 s block | New recordings **and** a disjoint time block; fit and test partitions both unseen |
| B: MEArec SNR holdout | the 16 non-SNR5 files (SNR 2, 3, 8, 12) | Untouched files, new noise regimes |
| C: HJ development scenes, late block | `drift16c_600s_11` and `static16c_600s_11`, 300–360 s | Temporal holdout only; same units, reported separately |

Set A is the primary confirmation because it varies both recording and time on
the realistic linear-probe family. Set B is secondary: MEArec SNR renderings
share spike-train seeds with the development SNR5 files, so they are **not**
independent temporal or biological replicates and must be reported as a noise-
regime stress test rather than as independent evidence. Set C is a diagnostic
only; because its units were seen during development it may not contribute to
any pass decision.

Confirmation reads the frozen selection JSON and may not reselect blocks, bit
widths, `g`, the matcher, the scale, or thresholds. It passes only if:

1. the selected arm beats D9 by at least 1.0 percentage point on event-weighted
   family means for both HJ set A and MEArec set B;
2. no confirmation recording loses more than 1.0 percentage point against D9;
3. positive-margin fraction does not fall by more than 0.5 percentage points on
   either family;
4. the Phase A2 causal-path retention holds on set A.

A failure records the boundary and stops Phase C. It cannot be repaired by
selecting another block combination on confirmation data.

## Three validation experiments

1. **Block isolation with falsification control:** the frozen four-recording
   development grid, including the `D9 + RAND` arm and the `POSNEG`
   reconstruction assertion.
2. **Causal and jitter boundary:** whether the gain survives a deployable
   signal path, which is the branch's most likely failure mode.
3. **Recall/traffic and online commit rate:** whether the representation gain
   converts into candidate-traffic movement, and whether shape columns really
   need fewer nonvolatile writes than amplitude columns.

## Two-week feasibility pilot

### Week 1

1. Extend `spatial_footprint.py` with bit-exact `POSNEG`, `LAT`, and `WIDTH`
   extraction plus no-I/O unit tests for tie-breaking, neutral codes, OOB
   slots, clipping, and the reconstruction bound.
2. Add the frozen four-recording runner with block/bit/gate sweeps and the
   `D9 + RAND` control.
3. Run Phase A, recompute gates independently, and freeze the selection or
   record an explicit null.
4. If non-null, run Phase A2 causal and jitter boundary tests.

### Week 2

1. Run Phase B candidate recall and template-traffic accounting.
2. Run confirmation sets A and B against the frozen selection.
3. If all pass, run Phase C per-block commit-rate measurement on one HJ
   drift/static development pair.
4. Write the result and provenance documents with source, input, and output
   SHA-256 manifests.

## Claim boundary

A passing Phase A supports only a more discriminative low-bit Level-1
descriptor under GT-event/GT-row component isolation with full-recording
zero-phase filtering. It does not establish candidate recall, template traffic,
causal operation, detected-event sorting, discovered unit rows, dense-2D
validity, RRAM endurance, or any macro/silicon PPA claim. Row payload bits and
absolute-difference lane counts are an algorithmic ledger and must not be
rewritten as area or energy.

The exploratory numbers in the starting-evidence section are unregistered
hypothesis generation on contaminated recordings. They may motivate this plan;
they may not be cited as a result.
