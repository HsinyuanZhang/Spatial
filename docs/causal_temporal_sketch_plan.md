# Alignment-Conditioned Causal Temporal Sketch Pilot

Status: **protocol executed on the four locked development recordings; formal
gate failed with zero candidates; confirmation prohibited**

## Scope and claim boundary

This experiment asks whether a compact temporal representation can solve the
second-level memory-width problem after a frozen high-recall 5-bit spatial
search.

The allowed claim is:

> Given an offline-filtered, GT-time-aligned event row, a fixed prefix sketch
> never consumes a waveform sample after its declared post-alignment horizon.

It is **not** an end-to-end causal sorting claim. Full-recording `filtfilt`,
GT event times, the central-channel search window, and discovered-row lifecycle
remain outside this pilot. “Causal” below always means prefix-causal relative
to the GT timestamp supplied to this component.

## Frozen data protocol

Development recordings:

```text
HJ      drift16c_600s_11
HJ      static16c_600s_11
MEArec  units10_snr5_seed202601
MEArec  units20_snr5_seed202601
```

Each contributes its first 60 seconds. Events are stably sorted and split
without separating equal timestamps:

```text
fit          first 50%
calibration  next 25%
test         final 25%
```

- spatial centroids and temporal sketch selection/templates use fit rows;
- the spatial radius is replaced from calibration rows; a fitted unit absent
  from calibration deterministically retains its fit-derived p99.9 radius and
  is flagged as a calibration fallback;
- test labels are metrics-only;
- the raw waveform quantizer scale is fitted at the same prefix horizon from
  fit rows belonging to Fisher-eligible units (at least ten fit events);
- no calibration/test waveform, label, prediction, or distribution updates a
  temporal feature, scale, or template.

The existing offline zero-phase filtering path is retained for component
isolation and must be named in every output row.

### Mandatory temporal preparation contract

The new runner performs one filter/preparation pass locally and does not
modify or feed the legacy normalized waveform field to the sketch:

```text
temporal input =
  zero-phase-filtered main-channel samples
  before every per-event waveform normalization
```

`prepare_dataset_events()["waveforms"]` is forbidden as sketch,
same-horizon-ceiling, signed-5-bit, or template input because that field is
normalized by the maximum magnitude over all 64 samples. It may be reproduced
only as an explicitly named legacy full-window report-only baseline.

Every surviving event carries an immutable integer `event_row_id`: the stable
ordinal of the original GT input event before chronological sorting. The
runner constructs `event_row_id`, timestamp, unit label, central channel,
descriptor, raw temporal waveform, and normalized report-only waveform in one
pipeline, applies every boundary mask jointly, and then performs one stable
chronological permutation. No table is joined by timestamp or unit label.
Equal timestamps retain original ordinal order.

The runner asserts one-to-one shape/order identity for all event fields and
stores a SHA-256 digest over `(event_row_id,timestamp,central_channel,label)`
for fit/calibration/test. It also records:

```text
temporal_input = filtered_pre_event_normalization_main_channel
filter = full_recording_zero_phase_butterworth_order3_300_6000Hz
alignment_source = gt_event_timestamp
central_channel_source =
  max_abs_peak_over_zero_phase_filtered_[t-15,t+15)_per_channel
central_channel_search_future_lookahead = 14 samples
```

The central-channel search does not move the timestamp returned by
`get_peak_amplitudes`. Index 15 is the GT event time, not a measured
central-channel peak. Filter, GT alignment, and central-channel selection are
outside the temporal-component delay.

## Frozen Level-1 candidate source

Use exactly one all-row source:

```text
D = 9
B = 5
uniform per-unit integer L1
calibration radius percentile = 99.9
inclusive distance <= radius
optimized logical payload = 54 bits/unit
```

It uses the same fit/calibration/test indices as Level 2. Candidate lists are
constructed once and shared by the float teacher, baselines, and every sketch
configuration. Level 2 may not append a GT row, top-k row, directory row, or
fallback candidate.

Candidate semantics:

```text
C = 0  unweighted spatial-centroid fallback; zero temporal-template reads
C = 1  direct candidate label; zero temporal-template reads
C > 1  read one temporal template for every distinct candidate
```

The development source must reach candidate recall at least `0.99` on every
recording and on both event-weighted families. Otherwise the Level-2 pilot is
invalid rather than repaired.

## Waveform timing

The reference waveform is:

```text
[t-15, t+49)
64 samples total
15 strictly pre-alignment samples
sample index 15 is aligned time t
last sample is t+48
```

The sketch grid uses acquisition delays:

```text
delay_samples       0    8    16    32    48
post_alignment_count 1    9    17    33    49
prefix_length      16   24    32    48    64
```

A delay-`h` sketch may read only indices `[0, 15+h]`. Perturbing any later test
sample must leave sample codes, selected-feature outputs, distances, and
predictions unchanged.

At 30 kHz the delays correspond to approximately:

```text
0, 0.267, 0.533, 1.067, and 1.600 ms
```

These values exclude detector/alignment delay and compute latency.

Two separate invariants are retained:

1. at fixed filtered aligned component input, perturbing samples after the
   declared prefix must not change prefix codes, sketch features, distances,
   or predictions; and
2. this test does not rerun `filtfilt`, because future raw-recording samples
   can affect earlier zero-phase-filtered samples by construction.

## Signed 5-bit sample interface

Raw main-channel samples are mapped to symmetric codes `[-15,15]`; the
two's-complement `-16` code is unused.

For each horizon, fit the scale on exactly the Fisher-eligible fit rows:

```text
q = percentile(|eligible-unit fit prefix samples|, 99.9, method="linear")
scale = smallest power of two >= q
```

If `q=0`, use scale one. Scale must be finite and positive. Its power-of-two
exponent is one shared configuration parameter.

The code is:

\[
x_5=\operatorname{clip}_{[-15,15]}
\left(
\operatorname{round}_{\text{half-away-from-zero}}
\frac{15x}{\mathrm{scale}}
\right).
\]

The reference quantizer consumes floating filtered dataset samples. A future
fixed-point ADC could conditionally map multiplication by 15 to `16x−x` and a
power-of-two scale to a shift, but no bit-exact ADC/fixed-point interface is
defined here. This observation is therefore not a hardware implementation or
PPA claim. The pilot reports the scale exponent and fit/test saturation
fractions; ADC format, exponent field width, conversion rounding, and analog
front-end remain outside the logical evidence.

## Fixed dyadic dictionary

Construct one ordered dictionary over the full 64-sample window. Every feature
has a stable integer ID independent of horizon.

### Interval mean

For support length `S∈{2,4,8,16,32}`, and starts aligned to `S`:

\[
z=\operatorname{round}_{HAZ}
\frac{\sum_{i=s}^{s+S-1}x_5[i]}{S}.
\]

### Haar contrast

For support length `S∈{4,8,16,32,64}`, and starts aligned to `S`:

\[
z=\operatorname{round}_{HAZ}
\frac{
\sum_{i=s}^{s+S/2-1}x_5[i]
-
\sum_{i=s+S/2}^{s+S-1}x_5[i]
}{S}.
\]

Here `round_HAZ` is signed round-to-nearest, ties away from zero. For integer
numerator `n` and power-of-two support `S`, the exact rule is:

```text
if n == 0: return 0
return sign(n) * ((abs(n) + S/2) // S)
```

The correction precedes the arithmetic shift; an ordinary signed right shift
alone is not HAZ for negative values. Outputs are clipped to `[-15,15]`.

Dictionary order is:

1. interval means before Haar contrasts;
2. ascending support length; and
3. ascending start index.

There are 93 full-window filters, so one selected-filter ID needs seven bits.
A filter is horizon-eligible only if its final sample is in the declared
prefix. Length-one means are intentionally absent to prevent equivalence to
raw tap selection.

## Fit-only shared filter selection

For every horizon:

1. encode all eligible dictionary features on fit rows;
2. compute the same multi-class Fisher
   `between_sum_squares/(within_sum_squares+1e-12)` score on the integer
   feature codes;
3. rank descending score, then ascending stable filter ID;
4. select the first `M∈{8,12,16}` IDs.

Selection is recording-wide and shared by all units. No per-unit or pairwise
filter table is allowed. Test/calibration perturbations must not change
selected IDs.

Units with at least ten fit events participate in the scale and Fisher
between/within statistics. Rarer fitted units still receive a template under
that shared scale but do not configure the scale or shared ranking. At least
two eligible units are required; otherwise the recording is `not_evaluable`
and cannot pass. Scores remain float64, NaN/Inf is rejected, all-zero scores
use stable filter-ID order, and between/within/score/rank/eligibility are
serialized.

For each unit, average its fit feature codes, apply deterministic
round-to-nearest-even, clip to `[-15,15]`, and store the resulting template.
For signed integer sum `a` and positive event count `n`, the
language-independent rule is:

```text
q, r = divmod(abs(a), n)
if 2*r > n or (2*r == n and q is odd): q += 1
return sign(a) * q
```

Runtime candidate distance is ordinary integer SAD over the `M` codes. Distance
ties retain the frozen Level-1 candidate order.

Intermediate bounds are asserted:

```text
mean S=32 numerator       [-480,480]   signed 10-bit
Haar S=64 numerator       [-960,960]   signed 11-bit
M=16 SAD                  [0,480]      unsigned 9-bit
feature/template code     [-15,15]     signed 5-bit; -16 invalid
```

No intermediate wraps or saturates. Clipping occurs only at the declared
sample-code and output feature/template boundaries.

## Grid and payload

```text
5 acquisition delays × 3 feature counts = 15 sketch configurations/record
M ∈ {8,12,16}
template payload = 5M bits/unit = {40,60,80}
shared selected-ID table = 7M bits = {56,84,112}
```

The shared power-of-two scale exponent and delay/profile control are reported
as configuration parameters but are not assigned a physical field width in
the algorithmic payload claim.

Mandatory baselines on the identical candidate lists:

- primary fit-scale full-64 raw float L1 teacher;
- report-only legacy per-event-peak-normalized full-64 float teacher;
- fit-scale signed full-64×5-bit SAD;
- fit-scale shared Fisher-48×5 SAD;
- fit-scale morphology-12×5 SAD; and
- same-horizon raw-prefix float L1 information ceiling.

Baselines are reports only and cannot be selected as the sketch.

For the same-horizon raw-prefix float ceiling, divide raw prefix samples by
the same fit-only power-of-two scale, clip to `[-1,1]`, and use float L1 on
fit-only unit means. It therefore changes only numeric precision/projection,
not the available samples or amplitude scale.

The primary full-64 teacher uses the delay-48 fit-only power-of-two scale:

```text
clip(raw filtered full64 / fit scale_48, -1, 1)
fit-only unit means
candidate-only float L1
stable Level-1 candidate order on ties
```

It is the only gate teacher. Full5 and Fisher48 use delay-48 signed HAZ sample
codes and the same integer-template rule as the sketch. Fisher48 ranks the 64
raw sample coordinates from fit integer codes and stores one shared
`48×6`-bit index table. Morphology12 applies the existing fixed definitions to
the clipped fit-scale full64 values, then uses its analytical `[0,1]` mapping
and unsigned 5-bit codes
`clip(round_half_away_from_zero(31 f), 0, 31)`. Its fit-only unit templates
use the same explicit integer round-to-nearest-even average rule as the signed
representations. Each output stores an immutable
`teacher_input_contract` string and SHA-256. Historical compact-runner numbers
are not substituted because they use a different split, radius source,
normalization, rounding, and traffic numerator.

## Logical operation accounting

Feature extraction is unconditional for every eligible event. This avoids a
hidden prefix buffer and prevents combining conditional work savings with an
unbuffered early horizon. For each event/configuration report:

- raw prefix samples consumed, exactly `16+delay`;
- sample quantizations, exactly `16+delay`;
- selected feature accumulator updates, equal to the sum of selected supports;
- positive coefficient updates and negative coefficient updates separately;
- normalization shifts;
- emitted 5-bit feature codes;
- candidate-template reads;
- candidate template bits;
- SAD absolute differences and reduction additions;
- stored per-unit template bits;
- shared selected-ID configuration bits; and
- acquisition delay in samples and microseconds.

These are logical counts. They exclude raw-signal filtering, detector,
alignment, ADC, clocking, accumulator gating, SRAM decoder/sense, routing,
controller, PVT, and physical macro overhead.

Under the frozen direct streaming schedule, a selected support-`S` mean
performs `S` positive accumulator updates. A support-`S` Haar feature performs
`S/2` positive and `S/2` negative accumulator updates. Their sum is exactly
the reported selected-support total; no second set of reduction additions may
be charged for the same accumulation. Each emitted feature performs one
power-of-two normalization shift.

SAD work occurs only for ambiguous candidates:

\[
N_{\mathrm{abs}}=\sum_e\mathbf{1}[C_e>1]C_eM,
\qquad
N_{\mathrm{add}}=\sum_e\mathbf{1}[C_e>1]C_e(M-1).
\]

Every counter is also decomposed by primary `C=0`, `C=1`, and `C>1`.

## Teachers and estimators

The primary accuracy teacher is the fit-scale full-64 raw float L1
template on the same frozen candidates. The same-horizon raw-prefix float
teacher is a diagnostic information ceiling, not the gate reference.

For each representation report:

- candidate recall;
- overall and ambiguous-only accuracy;
- ambiguous-event count, distinct unordered ambiguous candidate-pair count,
  and ambiguous-only teacher/sketch correct counts;
- teacher prediction agreement;
- accuracy delta versus the full-64 float teacher;
- per-unit accuracy and worst-unit accuracy;
- `C=0/1/>1` event counts;
- template reads and bits;
- global full-64×5-bit template bits;
- exact global-5-bit/actual traffic factor;
- storage/configuration bits; and
- all feature-extraction logical counters.

Family recall and accuracy use event-weighted hits. Family traffic is the ratio
of summed global bits to summed actual bits.

For `N` test events, `U` fitted units, and `M` features:

\[
B_{\mathrm{global5}}=N U(64\times5),
\qquad
B_{\mathrm{actual}}=
\left(\sum_e\mathbf{1}[C_e>1]C_e\right)(5M).
\]

`C=0` and `C=1` read zero temporal templates. If actual bits are zero, the
displayed factor is positive infinity; every gate uses the exact integer
condition `global_bits >= 4*actual_bits`. Candidate lists and `U` are identical
for all paired representations.

Positive-infinity traffic is reporting only. A recording or family with zero
ambiguous temporal invocations is `spatial_only_no_temporal_invocation` and
cannot pass or win a tie.

## Development gate

A single sketch configuration is eligible only if:

1. the frozen spatial source has candidate recall at least `0.99` on every
   development recording and both family aggregates;
2. every recording has at least 100 `C>1` test events, positive template
   reads, and at least five distinct unordered ambiguous candidate pairs;
3. each family aggregate has at least 500 `C>1` test events;
4. on every recording, sketch overall accuracy is at least the full-64 float teacher
   minus `0.015`;
5. on every recording, ambiguous-only sketch accuracy is at least the
   ambiguous-only teacher minus `0.020`;
6. on both event-weighted families, overall and ambiguous-only sketch accuracy
   are each at least the corresponding teacher minus `0.010`;
7. on both families, actual sketch-template bits are positive and exact
   global-64×5/actual sketch-template traffic is at least `4.0×`;
8. all raw-input provenance, row-identity, future-sample isolation, bit-width,
   and accounting invariants hold; and
9. the same delay and `M` pass both families. Selected filter IDs may differ by
   recording because they are fit-only configuration state; the structural
   dictionary, selection rule, delay, and `M` are shared.

If several pass, freeze exactly one by:

1. lowest acquisition delay;
2. smallest `M`;
3. largest minimum family accuracy delta;
4. largest minimum family traffic factor; and
5. ascending stable configuration name.

The pilot writes a deterministic selection JSON even on failure. It must
contain base revision, four input IDs/digest, split provenance, complete grid,
candidate-source audit, every per-record/family predicate, tie keys, CSV
hashes, event-row digests, scale exponents, selected IDs, full Fisher audit,
teacher-contract hash, and selected configuration or `null`.

The fixed development input-subset digest is
`9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13`.
Before loading a development recording, the runner also streams and verifies
the complete source-file SHA-256 recorded in
[`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256); matching a
dataset name alone is insufficient. Discovery freezes one resolved path; the
same path is rehashed before and after its lazy load. Recordings are loaded,
evaluated, and released sequentially so the protocol does not require all four
raw arrays in memory at once.

The runner also verifies the live Git base revision and writes a
content-addressed snapshot of every Python source under `algorithms/`, `data/`,
`evaluation/`, and `experiments/`. The frozen set contains 55 files and is
shared with the selection validator; a regression test fails if any runtime
package `.py` file is outside the snapshot.
The complete 32-recording identity remains
[`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256); confirmation is
exactly that locked corpus minus the four named development paths, requiring
ten HJ and eighteen MEArec IDs with no omissions or additions.

After the pilot, a tracked dedicated output manifest records both CSV hashes
and the selection-JSON hash. A confirmation command must supply the expected
JSON hash from that reviewed manifest. The loader also validates the complete
15-configuration audit, passing predicates, deterministic tie-break, selected
configuration, source/split/input digests, and referenced CSV hashes. A
caller-provided rehash of a modified JSON is not sufficient authorization.

## Disjoint confirmation

Only a non-null pilot JSON, locked by an externally supplied SHA-256 and
verified against its internal passing gate/audit, may authorize:

```text
HJ      remaining 10 recordings
MEArec remaining 18 recordings
```

Run only the selected horizon and `M`. Require separately on both families:

```text
candidate recall >= 0.99
accuracy delta vs full float teacher >= -0.01
ambiguous-only accuracy delta >= -0.01
global-64×5/actual template traffic >= 4.0×
no recording accuracy delta < -0.03
```

No confirmation is launched automatically. Pilot and confirmation use
different filenames/namespaces; confirmation rejects every development ID and
requires exactly the locked ten/eighteen family IDs.

## Kill criteria

Stop the branch if any holds:

1. no one configuration passes the fixed development gate;
2. a positive result needs a per-unit/pair-specific filter table;
3. a positive result needs test-selected filters, scale, horizon, or `M`;
4. the same-horizon float ceiling already misses the full teacher by more than
   one point on either family at every tested delay;
5. a selected configuration fails either confirmation family; or
6. a later forward-filtered/detected-event run invalidates the component
   result.

Failure closes fixed shared dyadic projections before considering a learned
integer matrix. Passing does not establish PPA; it authorizes the next RTL and
causal-filter integration work package.

## Completed disposition

The pilot was executed on 2026-07-26 with all 60 expected rows. Its
deterministic selection artifact reports:

```text
gate_pass: false
candidate_count: 0
selected_configuration: null
selection manifest SHA-256:
64766e86d8f142119be2ff4d0cef6398d21e01d52f43d096aa96145c6c5dfe4b
```

Kill criterion 1 fired. The remaining ten HJ and eighteen MEArec recordings
were not run, and no authorization anchor was generated.

The spatial source itself passed: candidate recall is `0.99051–0.99566` over
the four recordings. All 15 configurations also pass the exact `4×` family
template-bit condition. The failure is temporal accuracy. The least-bad
diagnostic point, delay 8 with `M=16`, is down `6.104` percentage points on HJ
and `12.084` points on MEArec relative to the full fit-scale float teacher.

The same-horizon raw-prefix float ceiling is already within one point of the
full teacher on both families at delay 16. This distinguishes representation
loss from insufficient acquisition horizon and closes further sweeps over the
same shared dyadic/Fisher dictionary.

Full numeric results and the hardware interpretation are in
[`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md).
Exact source, input, output, environment, and recomputation evidence are in
[`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md).
