# Unit-Specific Power-of-Two Weighted-L1 5-bit Rows

Status: **completed development pilot; formal gate failed; confirmation
prohibited and not run**

## Why this branch follows masked L1

The binary-mask pilot establishes two facts:

- unit-specific coordinate relevance can strongly improve HJ candidate recall
  and waveform traffic; and
- assigning zero weight to six coordinates can increase MEArec collisions.

The next bounded question is:

> Can small, nonzero, power-of-two weights preserve weak evidence from every
> coordinate while emphasizing each unit's stable or discriminative
> coordinates?

This is the selected candidate from
[`next_spatial_relation_candidates.md`](next_spatial_relation_candidates.md).
It changes the distance relation without adding prototype rows, hard
don't-care dimensions, a divider, or a multiplier.

## Two-sentence pitch

Each unit stores a two-bit shift code beside each 5-bit centroid coordinate.
The search computes a weighted integer L1 distance using only absolute
differences, shifts by zero/one/two, an 11-bit adder tree, and an inclusive
calibrated-radius comparison.

## Alternatives explicitly not reopened

This experiment does not add:

- zero-valued weights or another binary-mask profile;
- more than one row per unit;
- learned continuous weights or score temperatures;
- test-selected profile thresholds;
- a normalized divider;
- a directory in the development or confirmation gate;
- another inner/outer cascade; or
- online updates before a static representation passes.

## Bit-exact row contract

For fixed `D=9`, `B=5`, and weights `a_j ∈ {1,2,4}`:

```text
centroid[9]                         45 bits
shift code[9], code in {0,1,2}      18 bits
full-scale weighted-L1 radius       11 bits
logical search payload              74 bits/row
unit ID                    ceil(log2 U) bits
valid flag                            1 bit
```

The weighted distance is:

\[
d_u(q)=\sum_{j=0}^{8} 2^{h_{u,j}}|q_j-c_{u,j}|,
\qquad h_{u,j}\in\{0,1,2\},
\]

\[
\mathrm{match}(u)\iff d_u(q)\le r_u.
\]

The maximum distance is:

\[
9\times 4\times 31=1116,
\qquad
\lceil\log_2(1116+1)\rceil=11.
\]

All comparisons are inclusive. No coordinate has zero weight. Candidate unit
IDs are stable local row indices in `[0,U)` in hardware; external/GT labels
are evaluation metadata. Non-contiguous labels remain supported in software.

The uniform baseline uses shift code zero in all nine coordinates. It has
candidate semantics identical to an unweighted L1 row fitted and calibrated
on the same partitions. For a uniform physical implementation, the weighted
format remains 74 bits; an optimized unweighted baseline remains 54 bits.
Both payloads must be reported.

The 2-bit encoding is frozen:

| Shift code | Multiplier | Status |
|---:|---:|---|
| `00` | 1 | valid |
| `01` | 2 | valid |
| `10` | 4 | valid |
| `11` | — | reserved; decoder/model must reject |

Dimension order is descriptor order `COM_x, COM_y, P2P_0..P2P_6`. The logical
payload lists centroid fields in dimension order, then shift codes in the same
order, then the unsigned radius. Physical byte/word alignment, endian choice,
ECC, tag, and valid metadata are not included in the 74 bits.

### Executable uniform semantic reference

“Uniform equivalence” does not refer to the historical 50/50 or fit-radius
CSV. The unique calibrated unweighted semantic reference uses:

- the identical fit/calibration/test indices and fixed quantizer;
- the same integer centroids and unit ordering;
- unweighted nine-dimensional L1;
- the same `ceil(Q_p(..., method="linear"))` radius on calibration rows;
- the same zero-radius and missing-calibration-unit behavior;
- the same active rows, inclusive comparison, candidate ordering, fallback
  trigger, unweighted fallback distance, and fallback tie behavior.

For every tested percentile, parity tests must compare per-unit radii,
per-event primary candidate lists in order, primary `C`, fallback flags/final
labels, primary/fallback rows, absolute-difference lanes, additions, and
aggregate totals. Candidate semantics must be event-exact. Storage is
deliberately not equal: a general uniform weighted row is 74 logical bits,
whereas an optimized unweighted row is 54.

## Fit-only weight construction

Use the same frozen fit-only coordinate statistics as masked L1:

\[
w_{u,j}=Q_{0.95}(|x_{u,j}-c_{u,j}|),
\qquad
s_{u,j}=\frac{\operatorname{median}_{v\ne u}|c_{v,j}-c_{u,j}|}
{w_{u,j}+1}.
\]

The score modes are:

1. `uniform`: every shift code is zero;
2. `stability`: rank ascending `w`, then ascending feature ID; and
3. `separation_stability`: rank descending `s`, then ascending feature ID.

Two nonuniform rank profiles are frozen:

```text
top3_x4
  best-ranked dimensions 0..2   weight 4
  remaining dimensions 3..8     weight 1

tiered_421
  best-ranked dimensions 0..2   weight 4
  ranks 3..5                     weight 2
  ranks 6..8                     weight 1
```

There is no learned scale, threshold, temperature, adaptive profile, or zero
weight. For a single-unit fit, the between-unit term is zero and feature-ID
ties determine `separation_stability`.

Both the `Q_0.95` within dispersion and radius quantiles use
`method="linear"`. `np.median` semantics are frozen: an even number of
competing centroids uses the arithmetic mean of the two middle values. Scores
are retained in full float64 precision through ranking; there is no
intermediate rounding. The `+1` denominator is one unsigned code-bin unit.
NaN/Inf inputs are rejected.

## Split and calibration

Use the equal-timestamp-safe chronological split:

```text
fit          first 50%
calibration  next 25%
test         final 25%
```

- quantization is fixed `[0,1]→[0,31]`;
- centroids, within dispersion, between separation, ranks, and weights use fit
  rows only;
- each weight configuration is calibrated independently;
- per-unit radius is

\[
r_u=\left\lceil Q_p(d_u(x))\right\rceil,
\qquad p\in\{95,99,99.9\};
\]

- percentile interpolation is explicitly `linear`;
- radius zero is legal and equality matches;
- a fitted unit absent from calibration keeps its fit-derived radius and is
  flagged;
- calibration labels absent from fit are ignored; and
- test labels appear only in metrics.

Waveform teacher centroids/templates use **fit rows only**. Calibration or test
waveforms, labels, normalizers, and predictions may not update a template or
unit bank. Outputs include fit/cal/test event counts, unit coverage, template
support, calibration-absent fit units, and test labels absent from fit.

The raw recording remains full-recording zero-phase filtered for component
isolation. GT event rows and GT unit labels are allowed only for this static
representation ceiling.

## Frozen grid

```text
uniform/all-one weights              1 configuration
2 score modes × 2 rank profiles      4 configurations
3 radius percentiles                 ×3
total                                15 rows/recording
```

The complete pilot grid is exploratory. At most one nonuniform configuration
may be selected by the formal development gate for disjoint confirmation.

## Required invariants

1. Uniform weights match an independently calibrated unweighted integer-L1
   semantic reference exactly under the same active rows, including ordered
   candidates, fallback, diagnostics, and totals.
2. Every weight is in `{1,2,4}` and every row uses all nine dimensions.
3. Exact distance/radius equality matches.
4. A shift-code change changes the weighted distance by the expected
   power-of-two factor.
5. Test and calibration perturbations do not change fit-derived centroid,
   score, rank, or shift code.
6. Calibration perturbations may change radii only.
7. Maximum distance `1116` fits in 11 bits; code/radius boundaries are tested.
   Intermediate or final arithmetic may not wrap or saturate to ten bits.
8. Feature-score ties, all-identical features, one unit, rare units,
   non-contiguous labels, and unseen calibration/test labels are tested.
9. Equal timestamps are never split across fit/calibration/test.
10. Candidate ordering and active-row diagnostics are deterministic.
11. Empty-list fallback and row/lane/shift/addition accounting identities
    hold per event.

## Required measurements

### Candidate and unit quality

- true-unit candidate recall, hits, and two-sided 95% Wilson interval;
- zero/unique/ambiguous fractions;
- mean/p95/max distinct candidates;
- per-unit label, event count, recall, minimum, p05, p10, and worst decile;
- final float waveform-teacher accuracy; and
- true-unit weighted distance, radius, distance-minus-radius, and normalized
  distance distributions.

Wilson intervals are reporting diagnostics only. Pilot and confirmation gates
use the pre-registered point estimators and never select a profile from an
interval.

### Weight behavior

- per-unit centroid, within dispersion, between separation, final score,
  rank, weight, and shift-code arrays;
- per-dimension weight-1/2/4 frequency;
- mean/p95/max weight sum and logical shifted lanes per active row;
- early/late fit shift-code agreement, per-unit agreement, Hamming changes,
  mean absolute code change, and weight-tier transition matrix; and
- calibration support/missing-unit/zero-radius fields.

Early/late agreement uses only equal-timestamp-safe halves of the fit
partition. A unit enters `U_both` only if it has at least
`N_min_fit_half=5` events in each half. For eligible units:

\[
A_{\mathrm{code}}=
\frac{\sum_{u\in U_{\mathrm{both}}}\sum_{j=0}^{8}
\mathbf{1}[h^{early}_{u,j}=h^{late}_{u,j}]}
{9|U_{\mathrm{both}}|}.
\]

A recording with no eligible common unit is `not_evaluable` and cannot pass
an agreement-based continuation check. The audit records early/late event
counts by unit, eligible and excluded unit IDs, the denominator
`9|U_both|`, and the agreement. Units are equally weighted rather than
event-weighted. Kill criterion 5 uses the minimum recording agreement in each
family, not a pooled event-weighted value.

### Hardware and memory

- 74-bit logical weighted row plus unit-ID/valid metadata;
- 54-bit optimized unweighted baseline;
- primary and fallback row comparisons;
- nine absolute-difference lanes per active weighted row;
- logical shift-by-one and shift-by-two lanes, reported separately;
- logical reduction additions;
- `C=0`, `C=1`, and `C>1` conditional work;
- initial centroid/weight/radius configuration bits;
- candidate FIFO entries;
- actual 320-bit waveform-template bits; and
- explicit exclusions for decoder, sense, gating, banking, ECC, and periphery.

All hardware fields are algorithm-level logical operation or stored-payload
counters. They are not CiM/CAM physical feasibility, PPA, energy, area, cycle,
or latency evidence. A physical work package must separately freeze row
alignment, shift implementation, 11-bit accumulator topology, overflow
behavior, access schedule, routing/control overhead, macro, node, PVT, and
synthesis constraints.

The empty-candidate fallback is frozen to the existing **unweighted
all-centroid L1** scan. It reads no waveform template. Per event:

```text
primary rows + fallback rows = total Level-1 rows
primary abs lanes + fallback rows×9 = total abs lanes
primary shifted lanes = shift1 lanes + shift2 lanes
total additions = total abs lanes - total Level-1 rows
```

Here `C` always means the **primary weighted spatial candidate count**.
Fallback does not rewrite the `C=0` category. With all rows active,
`primary_rows=U` on every event and
`fallback_rows=1[C=0]×U`. Uniform rows have zero shifted lanes. Code `11`
cannot enter a counter. Weighted lane counts are logical operations, not
energy; all nine absolute differences remain active unless later RTL
demonstrates a different implementation. The identities are asserted per
event before totals are emitted.

Each nonuniform row also reports paired shape-change diagnostics against its
same-percentile uniform result:

- event-wise candidate-set Jaccard;
- uniform-to-weighted `C=0/1/>1` transition matrix;
- true-unit lost/gained events;
- non-target candidate additions/removals; and
- per-unit weighted/uniform radius ratios.

These are descriptive. A test-matched radius or candidate-rate control is not
introduced because it would add a test-selected configuration.

## Development pilot

Use exactly:

```text
HJ      drift16c_600s_11
HJ      static16c_600s_11
MEArec  units10_snr5_seed202601
MEArec  units20_snr5_seed202601
```

All rows are active. For each nonuniform configuration, compare against
`uniform` at the same radius percentile.

Define exact integer traffic totals:

\[
B_e=320R_e,\qquad
R_e=
\begin{cases}
0,&C_e\le 1\\
C_e,&C_e>1,
\end{cases}
\]

\[
B_{\mathrm{actual}}=\sum_e B_e,\qquad
B_{\mathrm{global}}=\sum_e 320U_e,\qquad
F_{\mathrm{traffic}}=\frac{B_{\mathrm{global}}}{B_{\mathrm{actual}}}.
\]

Every paired comparison uses the same events, fitted units, and 320-bit
template width. `C=0` invokes the spatial fallback but no waveform read;
`C=1` directly emits the sole candidate and also reads no waveform template;
`C>1` reads one 320-bit template per distinct candidate. `U_e` is the common
fitted-unit count for the paired baseline/candidate event. Per-record
tolerance is
`F_candidate >= 0.95 F_baseline`; “no worse” is evaluated by the exact integer
condition `actual_bits_candidate <= actual_bits_baseline`; branch B uses
`F_candidate >= 1.10 F_baseline`. Family aggregates use summed integer bits,
not an arithmetic mean of per-record factors.

Zero-traffic rules are frozen:

- if baseline and candidate actual bits are both zero, the per-record traffic
  tolerance and “no worse” predicate pass, their traffic ratio is exactly
  one, and branch B does not pass from traffic;
- if baseline is zero and candidate is positive, traffic predicates fail;
- if baseline is positive and candidate is zero, candidate traffic is
  strictly better and its ratio is positive infinity;
- displayed factors may be infinity, but gates/ties use these integer cases
  and exact rational cross-products, never `inf/inf` floating arithmetic.

It is pilot-eligible only if:

1. on every recording, recall is at least `baseline−0.005` and traffic
   reduction is at least `0.95×baseline`; and
2. on both event-weighted family aggregates it satisfies the same branch:
   - **A, recall-led:** recall delta at least `+0.005` and traffic no worse; or
   - **B, traffic-led:** traffic reduction at least `1.10×` and recall no
     worse.

If several pass, freeze exactly one by:

1. largest minimum recall delta across HJ and MEArec;
2. largest minimum traffic ratio;
3. `tiered_421` before `top3_x4`;
4. `separation_stability` before `stability`; and
5. lower radius percentile.

The pilot does not need to meet the final 0.99/4× gate, but it must satisfy
this same-direction development rule.

### Immutable pilot selection artifact

The pilot writes a deterministic selection JSON even when no configuration
passes. It contains:

- schema/protocol version and base code revision;
- four development dataset IDs and the locked four-line input-subset digest
  `9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13`;
- fixed grid, quantile/median semantics, split indices/timestamps, quantizer,
  event/unit coverage, and CSV SHA-256 values;
- all per-record predicates and family aggregates;
- every eligibility result and tie-break key;
- selected score mode/profile/percentile or explicit `null`; and
- pilot per-unit rank/weight/shift/radius state through the referenced CSV
  audit fields.

The JSON is locked by a separate SHA-256 manifest. Re-running the pilot may
create a new artifact, but it cannot silently replace the artifact authorized
for confirmation.

## Disjoint confirmation

Only if the pilot selection JSON contains one configuration:

```text
HJ       remaining 10 recordings
MEArec   remaining 18 recordings
```

Run only the frozen score mode/profile/percentile with all rows active.
Require separately in both confirmation families:

```text
candidate recall >= 0.99
waveform-template traffic reduction >= 4.0×
```

Development recordings must not enter the confirmation estimator. A later
12-HJ/20-MEArec table may be descriptive only. Directory support, causal
preprocessing, and online updates are considered only after all-row
confirmation passes.

The confirmation CLI requires both the selection-JSON path and its expected
SHA-256. It verifies the digest, schema, development IDs, input-subset digest,
grid, and non-null selection; it rejects a manual freeze, re-selection, hash
mismatch, or any development recording in confirmation. Confirmation outputs
use a separate filename and never overwrite pilot artifacts.

## Kill criteria

Stop this branch if any holds:

1. no nonuniform configuration passes the four-recording development gate;
2. the frozen configuration fails 0.99/4× in either confirmation family;
3. a positive result requires test-selected scores, profiles, weights,
   percentiles, or directories;
4. uniform-weight equivalence fails;
5. early/late shift-code agreement is below 0.75 in either family;
6. the result disappears under causal preprocessing before an online claim.

Criterion 1 prevents another full-corpus negative sweep. Whether a passing
algorithmic point justifies 74 versus 54 logical row bits and shift/11-bit
accumulation is an **advisory RTL/PPA decision**, not a pilot-selection,
confirmation, kill, or paper-pass rule. Without RTL/macro coefficients, the
experiment reports symbolic work and leaves that physical decision open.

## Strongest objection

**Objection:** the weights can be absorbed into a recalibrated radius, leaving
the same recall/traffic frontier while adding 20 row bits and shift logic.

**Response:** coordinate weighting changes the shape, not only the scale, of
each unit's acceptance region. The uniform invariant, fixed rank profiles,
paired same-percentile baseline, per-unit distance/radius audit, and immediate
pilot stop make the objection directly testable.

## Completed disposition

The fixed pilot was executed once the implementation, uniform semantic
reference, selection artifact, logical counters, and confirmation boundary
passed review. It produced 30 rows per family and a deterministic selection
JSON. The formal outcome is:

```text
pilot_gate_pass = false
pilot_gate_candidates = 0
selected_configuration = null
final_0p99_4x_pass = false
confirmation = prohibited and not run
```

At p95, HJ `stability/top3_x4` improved event-weighted recall from `0.90649`
to `0.90844` and traffic reduction from `3.9036×` to `7.9041×`, but missed the
recall-led `+0.005` condition. On MEArec the same relation moved recall
backward from `0.95606` to `0.95548` and improved traffic by only `1.042×`
relative to uniform. Higher radii recovered recall but remained far below
`4×` traffic.

Criterion 1 therefore fired. The disjoint confirmation set remains unused by
this branch, and static mask/weight profile tuning is closed as the next main
architecture direction. Detailed tables and the exact decision audit are in
[`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md);
source/input/output provenance is in
[`adaptive_weighted_5bit_provenance.md`](adaptive_weighted_5bit_provenance.md).
