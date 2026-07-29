# Post-Spatial Temporal Representation Candidates

Status: **structured divergence completed; dyadic prefix sketch executed and
rejected by the bounded development gate**

## Starting failure

The completed spatial experiments expose a stable boundary:

- tight 5-bit spatial regions can reduce waveform-template traffic but miss
  the true unit;
- p99.9 spatial regions recover roughly 0.99 candidate recall but leave many
  ambiguous candidates;
- static per-unit masks and `{1,2,4}` coordinate weights change HJ geometry
  but do not resolve MEArec collisions; and
- full 5-bit waveform SAD is accurate enough only conditionally, while shared
  taps, morphology-12, and pair-conditioned taps fail accuracy, traffic, or
  storage jointly.

The next experiment must therefore introduce temporal information that is not
already present in one static spatial row, and must reduce the **bits read per
candidate**, not merely retune the spatial acceptance region.

## Divergence: 18 raw candidates

| ID | Candidate | New information | Hardware shape | Primary concern |
|---:|---|---|---|---|
| 1 | Earlier raw sample taps | waveform prefix | selected 5-bit samples | repeats Fisher taps |
| 2 | Uniformly spaced prefix taps | waveform prefix | fixed sample reads | weak adaptation |
| 3 | Dyadic interval means | multiscale local level | sums + shifts | may blur sharp peaks |
| 4 | Dyadic Haar contrasts | multiscale slope/polarity | add/sub + shifts | selected filters need IDs |
| 5 | Sparse random ternary projections | distributed shape | `{−1,0,1}` accumulators | random seed/interpretability |
| 6 | Learned integer LDA projections | supervised separation | matrix-vector MAC | coefficient and update cost |
| 7 | Walsh-Hadamard coefficients | global frequency-like shape | butterflies | full transform work |
| 8 | Low-order integer DCT | smooth temporal shape | shift-add approximation | coefficient approximation |
| 9 | Piecewise-linear segment slopes | rise/fall morphology | window sums/subtract | window choice |
| 10 | Threshold-crossing bitmap | timing relative to levels | compares + bitset | amplitude-scale sensitivity |
| 11 | Time-to-first-threshold | onset/decay timing | counters/comparators | low feature capacity |
| 12 | Sign-run sketch | polarity sequence | XOR/counters | discards magnitude |
| 13 | Positive/negative area pyramid | multiscale energy | rectification + sums | morphology-12 overlap |
| 14 | Extremum magnitude/time pyramid | local peak structure | min/max/arg logic | noisy time indices |
| 15 | First-difference histogram | slope distribution | subtract/bin counters | order information lost |
| 16 | AR residual coefficients | local dynamics | recurrence/multiply | fit and arithmetic cost |
| 17 | Small matched-filter bank | waveform modes | correlation bank | prototype duplication |
| 18 | Integrate-and-fire temporal events | sparse shape crossings | counters/events | encoding calibration |

## Convergence filters

Candidates were filtered by:

1. **New-information test:** must use waveform evolution, not another static
   spatial mask/range.
2. **Non-repetition test:** must not be equivalent to selecting individual
   Fisher taps or the rejected fixed morphology-12 vector.
3. **5-bit mapping:** event representation and per-unit template must have a
   precise low-bit contract.
4. **Runtime simplicity:** candidate comparison must use absolute difference
   and addition; feature extraction should use compare/add/sub/shift.
5. **Shared state:** no `O(U²)` pair tables.
6. **Acquisition visibility:** the last waveform sample consumed must be
   explicit, allowing accuracy–latency measurement.
7. **Fit isolation:** feature choice and scale must be fit-only; test rows may
   not configure the sketch.
8. **Falsifiability:** a fixed four-recording pilot must be able to stop the
   branch before a disjoint confirmation run.

## Five strongest candidates

### 1. Dyadic interval means plus Haar contrasts

A fixed dictionary contains aligned power-of-two interval means and
adjacent-half contrasts. Each selected feature is a normalized sum with
coefficients in `{−1,0,1}` and a power-of-two denominator, so runtime uses
accumulation and shifting. It aggregates samples rather than copying taps and
can be evaluated at explicit prefix horizons.

### 2. Sparse ternary projections

Fit-only Fisher or margin selection could choose sparse `{−1,0,1}`
projections. This is flexible and shift-free, but a learned/random coefficient
table is larger and less structured than a fixed dyadic decoder.

### 3. Piecewise-linear segment slopes

Fixed segment means and adjacent differences directly encode rise, trough,
rebound, and decay. This is interpretable and cheap, but a single segmentation
is brittle to alignment and resembles a narrow subset of the dyadic
dictionary.

### 4. Threshold-crossing time bitmap

Crossing times could compactly encode waveform morphology with compare/counter
logic. Its thresholds depend strongly on amplitude normalization, and a
bitmap/Hamming metric is less directly comparable with the existing 5-bit SAD
engine.

### 5. Fixed Walsh-Hadamard prefix coefficients

Butterfly computation is regular and multiplier-free. However, computing the
whole transform before retaining only a small coefficient subset can cost
more than accumulating only selected dyadic filters.

## Selected hypothesis

**Winner:** fit-selected dyadic prefix means and Haar contrasts.

Two-sentence pitch:

> Compute a small shared bank of multiscale interval/contrast features once
> per aligned event, quantize each to a signed 5-bit code, and compare only
> ambiguous spatial candidates against `5M`-bit unit templates. Because every
> filter has `{−1,0,1}` coefficients and a power-of-two support, feature
> extraction maps to streaming accumulators and shifts while exposing an
> explicit post-alignment acquisition delay.

This is not described as fully causal spike sorting. The current recordings
still use offline zero-phase filtering, GT event rows, and GT-time alignment.
The bounded claim is an **alignment-conditioned prefix-causal
temporal sketch**: after a fixed aligned event time, a configuration may not
read a sample later than its declared horizon.

## Three validation experiments

1. **Information/latency ceiling:** compare same-horizon raw-prefix float L1
   with the full normalized 64-sample float teacher at delays
   `0/8/16/32/48` samples.
2. **Bit-exact sketch Pareto:** evaluate `M∈{8,12,16}` signed 5-bit selected
   dyadic features on one frozen p99.9 5-bit spatial candidate source.
3. **Architecture comparison:** report accuracy, global-5-bit/template traffic,
   feature extraction operations, configuration bits, acquisition delay, and
   storage against full-64×5, shared Fisher-48×5, and morphology-12×5.

## Two-week feasibility pilot

### Week 1

- freeze dictionary order, prefix eligibility, signed rounding, scale, row
  payload, candidate source, and gate;
- implement no-I/O core tests for future-sample isolation, bit bounds,
  deterministic selection, and operation accounting;
- implement one fixed four-recording runner and deterministic selection
  artifact.

### Week 2

- execute only the four development recordings;
- independently recompute family estimators and the gate;
- stop if no common configuration passes;
- only a non-null, hash-locked selection may authorize the remaining
  recordings.

## Strongest objection

**Objection:** Fisher-selected Haar filters are merely “selected taps with
extra additions,” so the experiment is incremental and may only trade sample
noise for blur.

**Response:** the pilot explicitly compares against Fisher taps under the same
candidate source and bit width. A dyadic feature combines multiple samples,
has a fixed structured decoder, is computed once per event, and exposes a
prefix latency; it passes only if one shared `M≤16` configuration preserves the
full teacher while meeting the bit-traffic gate on both families. Failure
closes structured shared projections before introducing a learned matrix.

## Completed outcome

The experiment reached the pre-registered failure condition. All 15
delay/feature-count configurations pass the `4×` family template-bit
condition, but none preserves temporal accuracy. The least-bad shared
diagnostic point, delay 8 with `M=16`, is down `6.10pp` on HJ and `12.08pp` on
MEArec relative to the full fit-scale float teacher.

The same-horizon raw-prefix float ceiling is already within one point of the
full teacher on both families by delay 16. This means the selected dictionary
and `M≤16` code—not a lack of later waveform samples—is the principal failure.
The formal selection is `null`; no confirmation was run.

This closes candidates 3+4 as currently combined: shared Fisher-selected
dyadic means/Haar contrasts with one signed-5-bit code per selected filter.
The next diagnostic must decompose same-ID float projection, signed-5 raw
prefix taps, and feature quantization before selecting a new learned
projection or memory organization. See
[`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md).
