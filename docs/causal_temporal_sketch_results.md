# Alignment-Conditioned 5-bit Temporal-Sketch Pilot Results

Status: **completed negative four-recording development pilot; formal gate
failed; selected configuration is null; confirmation prohibited**

## Decision

The fit-selected dyadic interval-mean/Haar sketch does **not** solve the
Level-2 representation problem behind the frozen high-recall spatial source.
All 15 configurations meet the exact `4×` template-bit condition on both
families, and the Level-1 source reaches at least `0.99` true-unit candidate
recall on every recording. Every configuration nevertheless misses the
pre-registered temporal-accuracy margins by a large amount.

The deterministic selection artifact reports:

```text
gate_pass: false
candidate_count: 0
selected_configuration: null
confirmation: not run and not implemented by the pilot entry point
```

The least-bad observed point, `delay=8, M=16`, is a **diagnostic only**. It is
not a selected configuration and cannot authorize confirmation:

| Family | Sketch accuracy | Full fit-scale teacher | Overall delta | Ambiguous delta | Full64×5 / sketch template bits |
|---|---:|---:|---:|---:|---:|
| HJ | 0.71494 | 0.77597 | **−6.104 pp** | −6.116 pp | **5.920×** |
| MEArec | 0.64788 | 0.76872 | **−12.084 pp** | −12.084 pp | **5.018×** |

The negative result is therefore not a marginal threshold miss. It closes
this particular combination of:

- one recording-wide Fisher-selected dyadic dictionary;
- `M≤16` signed 5-bit features;
- per-unit integer-mean templates; and
- candidate-only L1/SAD comparison.

It does not close all low-bit temporal refinement. In particular, it does not
test a hardware-constrained learned projection, unit-conditioned feature bank,
or a two-step coarse/fine temporal code.

## Claim boundary

This is a component-isolation experiment, not end-to-end causal spike sorting.
The rows use:

- first 60 seconds of four fixed development recordings;
- full-recording zero-phase Butterworth filtering;
- ground-truth event timestamps;
- a central channel selected with a `[t−15,t+15)` filtered search window;
- a 64-sample `[t−15,t+49)` central-channel waveform; and
- chronological `50/25/25` fit/calibration/test splits that do not separate
  equal timestamps.

“Delay” refers only to the last aligned waveform sample visible to the
temporal component. It excludes detection, alignment, central-channel
selection, filter delay, and digital compute latency.

## Frozen gate

The evaluated grid was:

```text
delay samples: 0, 8, 16, 32, 48
feature count: 8, 12, 16
total:         15 configurations × 4 recordings = 60 rows
```

Every recording/configuration had to satisfy:

```text
candidate recall >= 0.99
ambiguous events >= 100
candidate template reads > 0
distinct ambiguous pairs >= 5
overall accuracy delta versus teacher >= -1.5 pp
ambiguous accuracy delta versus teacher >= -2.0 pp
all provenance, row, future-sample, bit-width, and accounting invariants true
```

Each event-weighted family aggregation also had to satisfy:

```text
candidate recall >= 0.99
ambiguous events >= 500
overall and ambiguous accuracy delta >= -1.0 pp
actual template bits > 0
global full64x5 bits >= 4 × actual sketch bits
```

The selection module recomputes these predicates from integer hit/bit counts.
It does not trust displayed rounded factors or the runner’s invariant
booleans.

## Level-1 source result

The frozen source is one all-row, uniform `D=9`, `B=5`, per-unit integer-L1
search with calibration-only p99.9 radii.

| Recording | Test events | Units | Candidate recall | C=0 | C=1 | C>1 | Candidate-template reads | Distinct pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HJ drift11 | 843 | 11 | 0.99051 | 0 | 0 | 843 | 6,123 | 55 |
| HJ static11 | 697 | 10 | 0.99283 | 0 | 3 | 694 | 4,851 | 45 |
| MEArec units10 | 1,153 | 10 | 0.99566 | 0 | 0 | 1,153 | 8,290 | 44 |
| MEArec units20 | 2,306 | 20 | 0.99350 | 0 | 0 | 2,306 | 37,668 | 190 |

The source passes the recall and support preconditions. It also shows why
Level 2 remains necessary: `4,996/4,999` test events are ambiguous. The
p99.9 source is a high-recall retrieval stage, not an early-exit classifier.

Candidate membership and the C0 fallback cache are derived from the same
single integer distance matrix:

```text
one D9 quantization
  -> one N_test × U L1 distance matrix
     -> distance <= calibrated radius: candidate lists
     -> argmin over the same row: frozen C0 fallback state
```

The artifact separately counts descriptor quantizations, centroid reads,
absolute differences, reduction additions, radius comparisons, and argmin
comparisons. Selection independently verifies:

```text
quantizations       = N × 9
centroid reads      = N × U
absolute differences= N × U × 9
L1 additions        = N × U × 8
radius comparisons  = N × U
argmin comparisons  = N × (U - 1)
C0 cache reuse      = C0 events
```

## Complete 15-point result

Accuracy deltas are event-weighted relative to the primary full-64 fit-scale
float teacher. Traffic is global full64×5 template bits divided by actual
candidate sketch-template bits.

| Delay | M | HJ overall delta | MEArec overall delta | HJ traffic | MEArec traffic | Result |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 8 | −33.896 pp | −24.631 pp | 11.841× | 10.035× | fail accuracy |
| 0 | 12 | −26.299 pp | −16.739 pp | 7.894× | 6.690× | fail accuracy |
| 0 | 16 | −24.286 pp | −15.467 pp | 5.920× | 5.018× | fail accuracy |
| 8 | 8 | −13.701 pp | −21.538 pp | 11.841× | 10.035× | fail accuracy |
| 8 | 12 | −13.701 pp | −14.397 pp | 7.894× | 6.690× | fail accuracy |
| 8 | 16 | **−6.104 pp** | **−12.084 pp** | 5.920× | 5.018× | fail accuracy |
| 16 | 8 | −17.338 pp | −21.538 pp | 11.841× | 10.035× | fail accuracy |
| 16 | 12 | −15.779 pp | −14.397 pp | 7.894× | 6.690× | fail accuracy |
| 16 | 16 | −14.545 pp | −13.472 pp | 5.920× | 5.018× | fail accuracy |
| 32 | 8 | −17.338 pp | −20.584 pp | 11.841× | 10.035× | fail accuracy |
| 32 | 12 | −15.779 pp | −16.305 pp | 7.894× | 6.690× | fail accuracy |
| 32 | 16 | −14.545 pp | −14.571 pp | 5.920× | 5.018× | fail accuracy |
| 48 | 8 | −17.338 pp | −20.584 pp | 11.841× | 10.035× | fail accuracy |
| 48 | 12 | −15.779 pp | −16.305 pp | 7.894× | 6.690× | fail accuracy |
| 48 | 16 | −14.545 pp | −13.906 pp | 5.920× | 5.018× | fail accuracy |

Every family traffic predicate passes, including the widest 80-bit template.
Every family accuracy predicate fails. At the recording level, all four
recordings also fail both required accuracy-delta predicates for all 15
configurations.

## Acquisition information is not the principal failure

The same-candidate float raw-prefix ceiling separates lack of waveform
samples from loss introduced by the representation:

| Delay | HJ ceiling delta | MEArec ceiling delta | HJ ceiling accuracy | MEArec ceiling accuracy |
|---:|---:|---:|---:|---:|
| 0 | −16.818 pp | −5.348 pp | 0.60779 | 0.71524 |
| 8 | −1.883 pp | +0.087 pp | 0.75714 | 0.76959 |
| 16 | **−0.325 pp** | **+0.665 pp** | 0.77273 | 0.77537 |
| 32 | +0.260 pp | +0.578 pp | 0.77857 | 0.77450 |
| 48 | 0.000 pp | 0.000 pp | 0.77597 | 0.76872 |

At delay 16, the raw-prefix float ceiling is already within one percentage
point of the full teacher on both families. The corresponding `M=16` dyadic
sketch is still down `14.545/13.472` points. The dominant failure is therefore
the shared compressed representation and metric, not a need to wait for the
last waveform samples.

The delay-8 diagnostic makes the same point:

```text
HJ:     float raw-prefix 0.75714 -> dyadic 5-bit M16 0.71494
MEArec: float raw-prefix 0.76959 -> dyadic 5-bit M16 0.64788
```

## Mandatory baseline context

All baselines use the same frozen candidate lists and C0/C1/C>1 decision
semantics.

| Representation | Template bits/unit | HJ accuracy | HJ delta | MEArec accuracy | MEArec delta |
|---|---:|---:|---:|---:|---:|
| Full64 fit-scale float teacher | 2,048 | 0.77597 | — | 0.76872 | — |
| Legacy per-event-normalized float64 | 2,048 | 0.78182 | +0.584 pp | 0.68141 | −8.731 pp |
| Signed full64×5 | 320 | 0.75325 | −2.273 pp | 0.70512 | −6.360 pp |
| Shared Fisher48×5 | 240 + shared indices | 0.75325 | −2.273 pp | 0.70252 | −6.620 pp |
| Fixed morphology12×5 | 60 | 0.50974 | −26.623 pp | 0.58919 | −17.953 pp |

This also tightens the interpretation. Five-bit quantization alone is not
free, especially on MEArec, but the `M≤16` dyadic code loses substantially
more than full64×5. The new branch is worse than the safe 320-bit digital
baseline on accuracy despite meeting the traffic target.

## Hardware interpretation

### Stored state

The frozen sketch uses:

```text
signed template payload per unit = 5M bits
selected dictionary ID state     = 7M bits per recording
dictionary size                  = 93 fixed filters
```

| M | Template bits/unit | Shared selected-ID bits | Family traffic range |
|---:|---:|---:|---:|
| 8 | 40 | 56 | 10.04–11.84× |
| 12 | 60 | 84 | 6.69–7.89× |
| 16 | 80 | 112 | 5.02–5.92× |

For diagnostic `delay=8, M=16`:

```text
HJ:
  global full64x5 template bits = 5,197,760
  actual sketch template bits   =   877,920
  temporal template reads       =    10,974

MEArec:
  global full64x5 template bits = 18,448,000
  actual sketch template bits   =  3,676,640
  temporal template reads       =     45,958
```

The dyadic extractor is structurally hardware-friendly: selected aligned
means and Haar contrasts use additions/subtractions plus power-of-two
normalization. That mapping is not enough for a paper contribution when the
classification margin collapses. Logical bit/op accounting is valid evidence
for an architecture model, not synthesized macro/peripheral PPA.

### Latency

At the frozen 30 kHz input rate:

| Delay samples | Component acquisition delay |
|---:|---:|
| 0 | 0 ms after alignment |
| 8 | 0.267 ms |
| 16 | 0.533 ms |
| 32 | 1.067 ms |
| 48 | 1.600 ms |

These figures exclude preprocessing and alignment. The promising information
boundary is around delay 16, but the tested feature bank cannot exploit it.

## Failure diagnosis and next experiment

The result supports three conclusions:

1. **Stop extending the same dyadic Fisher grid.** Increasing horizon beyond
   delay 8 or retaining 16 features does not approach the teacher, so a wider
   sweep over the same dictionary is not justified.
2. **Do not return to more spatial mask/range tuning as the main branch.** The
   fixed spatial source already clears the 0.99 recall precondition; the
   failure is after retrieval.
3. **Run one representation-decomposition experiment before choosing new
   hardware.** Under the same candidate source and horizons, compare:
   same-ID float dyadic features, signed-5 raw-prefix Fisher taps, and the
   existing signed full-prefix code. This will identify projection loss versus
   feature quantization loss without adding a new learned model.

Only if raw-prefix/tap `5M` remains accuracy-positive should the next design
consider a banked sample SRAM plus selected-tap SAD. If float dyadic features
already fail, the fixed dictionary is information-limited. If float dyadic
passes but signed-5 dyadic fails, the next work should target per-feature
scale/code utilization rather than a new candidate policy.

## Reproduction

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
MPLCONFIGDIR=/tmp \
/home/xinyuan/miniconda3/envs/spint/bin/python \
  -m Spatial.experiments.run_causal_temporal_sketch --pilot
```

The completed command reported:

```text
pilot gate_pass=False
manifest_sha256=64766e86d8f142119be2ff4d0cef6398d21e01d52f43d096aa96145c6c5dfe4b
confirmation was not run and is not implemented by this entry point
```

The exact source, inputs, outputs, environment, and independent recomputation
are recorded in
[`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md).

