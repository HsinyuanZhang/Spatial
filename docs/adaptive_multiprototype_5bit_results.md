# Bounded Multi-Prototype 5-bit Spatial Search

Status: **completed negative result; stopped before online self-updates by the
pre-registered static gate**

## Decision

Allowing each unit to own up to four local 5-bit centroid/radius rows does not
break the observed candidate-recall versus waveform-memory-traffic tradeoff.
Across the complete 60-second corpus:

- HJ has no configuration that simultaneously reaches 0.99 true-unit
  candidate recall and 4× global waveform-template traffic reduction;
- MEArec has no configuration that reaches 4× traffic reduction at all; and
- the only HJ 4×-class point needs four rows for every unit and reaches only
  0.898 candidate recall.

This fires kill criterion 1 in
[`adaptive_multiprototype_5bit_plan.md`](adaptive_multiprototype_5bit_plan.md).
The HJ traffic-compliant point also violates the intended mean-`P≤2` storage
bound in criterion 2. The planned self-predicted/oracle update study was
therefore not executed: an online updater cannot establish the required
operating point after the more favorable GT-row static representation study
has already failed it.

## Implemented artifacts

- `algorithms/adaptive_prototype_search.py`
  - fixed unsigned `[0,1]→[0,31]` quantization;
  - deterministic temporal and farthest-L1 prototype construction;
  - inclusive integer L1 membership;
  - stable prototype-hit to unit-ID de-duplication;
  - shared or per-query row-enable masks; and
  - row/storage diagnostics.
- `experiments/run_adaptive_multiprototype_5bit.py`
  - chronological train/test component-isolation sweep;
  - all-row and prototype-specific channel-support directories;
  - candidate, waveform-read, row, fallback, storage, and pointer accounting;
  - fixed 320-bit signed-5-bit×64 waveform-template traffic baseline.
- `algorithms/causal_preprocessing.py`
  - independent forward-only stateful Butterworth SOS primitive;
  - chunk-equivalence, reset, and no-future-leakage unit tests.
- `tests/test_adaptive_prototype_search.py`,
  `tests/test_adaptive_multiprototype_experiment.py`, and
  `tests/test_causal_preprocessing.py`.

The causal primitive is infrastructure, not evidence that this experiment is a
causal pipeline. The static experiment deliberately retains the historical
full-recording zero-phase preprocessing so that the representation change is
isolated against the existing one-row result.

## Protocol

### Dataset and split

| Family | Recordings | Analyzed duration | Generated configurations |
|---|---:|---:|---:|
| Hybrid Janelia | 12 | first 60 s/recording | 1,152 rows |
| MEArec | 20 | 60 s/recording | 1,920 rows |

The event schedule and unit rows are ground truth. Events are stably split by
time into chronological train/test halves, and equal timestamps are kept on
the same side. Quantized prototypes, per-prototype radii, channel supports,
and waveform teacher centroids are fitted from the training half only. Test
labels are used for metrics.

The raw recording is filtered before this split with full-recording
zero-phase `filtfilt`. Consequently this is an offline GT-event/GT-row
component-isolation grid, not an end-to-end detected-event, discovered-row, or
causal-stream result.

### Frozen grid

```text
descriptor                COM_x, COM_y, seven local P2P values
D, B                      9 dimensions, 5 bits/dimension
P                         1, 2, 3, 4 rows/unit
prototype construction    temporal, farthest_l1
radius percentile         95, 99, 99.9
directory                 all rows or prototype support
support coverage          0.95, 0.99, 1.00
membership                integer L1 <= integer radius
```

This gives `4×2×3×(1+3)=96` rows per recording. It is an exploratory test grid
with no final configuration selected from the test half. The complete grid is
reported; no grid winner is presented as an independently validated,
test-untuned operating point.

### Hardware accounting boundary

For `D=9`, `B=5`, one prototype row stores:

```text
centroid             9 × 5 = 45 bits
full-scale radius              9 bits
search payload                 54 bits
unit ID             ceil(log2 U) bits
valid flag                     1 bit
```

Multiple matching prototype rows from the same unit generate one candidate ID.
They may increase search-row work, but they do not duplicate waveform-template
reads. Waveform traffic is:

```text
|C| = 0       zero waveform-template reads; spatial fallback is counted
|C| = 1       zero waveform-template reads; direct label
|C| > 1       |C| × 320 template bits
global base   N_test × U × 320 template bits
```

The prototype-directory bit count includes prototype-row-ID payload only. It
does not include directory-entry valid bits, base addresses, banking,
decoders, sensing, or other periphery. Primary prototype comparisons and the
empty-list spatial fallback scan are reported separately and summed as total
Level-1 row comparisons.

## Main results

Family recalls, candidate counts, active-row counts, and accuracies below are
event-weighted. Traffic reduction is the ratio of summed global bits to summed
actual bits. Per-recording row/storage/pointer figures are explicitly marked
as recording-unweighted means.

| Family | Reported boundary | Configuration | Candidate recall | Mean C | Waveform traffic reduction | Gate |
|---|---|---|---:|---:|---:|---|
| HJ | highest recall | temporal, P=2, p99.9, all rows | **0.993336** | 14.422 | 1.253× | fail traffic |
| HJ | highest recall with ≥4× traffic | farthest-L1, P=4, p95, support-0.99 | 0.897681 | 4.540 | **4.071×** | fail recall and mean P |
| HJ | maximum traffic reduction | farthest-L1, P=4, p95, support-0.95 | 0.897470 | 4.507 | **4.102×** | fail recall and mean P |
| MEArec | highest recall | temporal, P=3, p99.9, all rows | **0.994705** | 13.352 | 1.253× | fail traffic |
| MEArec | maximum traffic reduction | temporal/farthest-L1, P=1, p95, support-0.95 | 0.942077 | 5.529 | **3.075×** | fail both |

There are zero joint-gate configurations in either family. HJ has three
directory variants just above 4×, all from the same `P=4`, farthest-L1, p95
configuration family and all below 0.899 recall. MEArec's entire grid remains
below 4×.

### Why the HJ 4× point fails

For the HJ `P=4`, p95, support-0.99 point:

- directory true-unit row activation recall is 0.999125;
- candidate recall is only 0.897681;
- mean active prototype rows/event is 41.405;
- the recording-unweighted prototype-row count is 66.33
  (`40–92` across recordings);
- prototype storage averages 3,960.7 bits/recording before directory
  periphery;
- prototype-row-ID pointer payload averages 823.7 bits/recording;
- the primary search performs 1,373,029 row comparisons over 33,161 test
  events; and
- 520 empty-list events add 31,588 fallback comparisons, for 1,404,617 total
  Level-1 row comparisons, or 42.357/event.

The directory almost always activates a row belonging to the true unit.
Therefore the approximately ten-point recall deficit is a prototype/radius
membership failure, not evidence that a larger directory would solve the
problem.

### Cost of the MEArec high-recall point

For the MEArec temporal `P=3`, p99.9, all-row point:

- candidate recall is 0.994705 but mean `C` is 13.352;
- waveform-template traffic reduction is only 1.253×;
- the recording-unweighted row count is 45 rows
  (`30` for ten-unit and `60` for twenty-unit recordings);
- prototype storage averages 2,685 bits/recording;
- mean active rows are 50.174/event after event weighting; and
- the search performs 3,269,100 Level-1 row comparisons over 65,155 test
  events.

There are no empty candidate lists at this point, so primary and total
Level-1 comparisons are identical.

## P=1 invariant

`P=1` all-row temporal and farthest-L1 configurations are identical to each
other and to the earlier single per-unit L1 implementation for every recording
and each tested percentile. Event-weighted family results are:

| Family | Radius percentile | Candidate recall | Mean C | Waveform traffic reduction |
|---|---:|---:|---:|---:|
| HJ | 95 | 0.935617 | 5.974 | 3.044× |
| HJ | 99 | 0.983475 | 12.256 | 1.474× |
| HJ | 99.9 | 0.993004 | 14.423 | 1.253× |
| MEArec | 95 | 0.947172 | 6.437 | 2.618× |
| MEArec | 99 | 0.985680 | 11.318 | 1.478× |
| MEArec | 99.9 | 0.994613 | 13.327 | 1.255× |

This rules out an implementation, quantizer, inequality, or traffic-accounting
change as the explanation for the multi-row result.

## What multiple rows actually change

At p95 with all rows active, increasing temporal `P` improves candidate recall
but broadens the union and reduces traffic benefit:

| Family | P=1 recall / reduction | P=4 recall / reduction |
|---|---:|---:|
| HJ | 0.935617 / 3.044× | 0.960074 / 2.082× |
| MEArec | 0.947172 / 2.618× | 0.964792 / 2.034× |

At p99.9, multiple rows add almost no recall because the one-row union is
already broad:

- HJ: 0.993004 at `P=1` versus 0.993305 at `P=4`;
- MEArec: 0.994613 at `P=1` versus 0.994521 at `P=4`.

The representation therefore interpolates along the existing frontier rather
than moving it: more local modes recover some tight-radius misses by admitting
more units, while high-percentile unions remain close to a global candidate
set.

## Causal-preprocessing disposition

`CausalBandpassFilter` provides a tested forward-only SOS state machine:
contiguous chunk processing equals one-pass processing, reset is deterministic,
and future-input perturbations do not change prior outputs. It does not by
itself make the candidate experiment causal because:

- event times are still GT;
- descriptor windows need samples after the event;
- the 64-sample waveform `[t-15,t+49)` contains the aligned sample plus 48
  strictly future samples;
- discovered-unit creation/deletion and non-GT confidence updates do not
  exist; and
- warm-up, detector, checkpoint, and train/test filter-state policies are not
  integrated.

The pre-registered workflow required the static representation to pass before
spending a full cross-dataset run on causal filtering and online row
lifecycle. Since the static gate failed, the causal primitive is retained for
the next genuinely viable candidate policy rather than being used to extend
this rejected branch.

## Reproduction

From the parent `SNN_SpikeSorting` directory:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_multiprototype_5bit \
  --family all --duration 60 --bits 5 \
  --prototypes 1 2 3 4 \
  --modes temporal farthest_l1 \
  --radius-percentiles 95 99 99.9 \
  --directory-coverages 0.95 0.99 1.0 \
  --quiet
```

Outputs:

```text
output/adaptive_multiprototype_5bit/
├── adaptive_multiprototype_5bit_hj_60s.csv
└── adaptive_multiprototype_5bit_mearec_60s.csv
```

The dedicated source/output hash manifests and their relationship to the
existing 32-recording input manifest are recorded in
[`adaptive_multiprototype_5bit_provenance.md`](adaptive_multiprototype_5bit_provenance.md).

## Claim boundary

This result supports the following claim:

> Under an offline, GT-event/GT-row, chronological-train-fit protocol, bounded
> `P≤4` 5-bit local prototype rows do not achieve the pre-registered
> 0.99-recall/4×-waveform-traffic operating point on either HJ or MEArec.

It does not support claims of causal sorting, online discovered-row adaptation,
total chip energy, area, latency, or a finalized hardware macro. The negative
result is useful precisely because row payload, duplicate removal, directory
activation, empty-list fallback, and waveform traffic are exposed instead of
being hidden behind final classification accuracy.
