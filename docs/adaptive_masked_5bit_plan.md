# Unit-Specific Masked-L1 5-bit Spatial Rows

Status: **development pilot completed; formal cross-family gate failed and
disjoint confirmation was not run**

The frozen implementation and four-recording falsification pilot are complete.
The formal pilot gate returned zero eligible configurations, so kill criterion
1 fired. The quantitative result, per-record intervals, hardware counters, and
reproduction command are in
[`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md).

## Why this branch follows the multi-prototype result

The completed `P≤4` study rejects the idea that a unit's broad 5-bit range can
be repaired by taking the union of a few narrower local modes. At p95, more
rows recover some true-unit events by admitting more candidate units; at
p99.9, they add row/storage cost without material recall. The failure is
consistent with unit-specific noisy or drift-sensitive descriptor coordinates,
not merely an insufficient number of centroids.

The next bounded question is therefore:

> Can one row per unit retain only that unit's stable, discriminative 5-bit
> coordinates, so the true-unit radius ignores drift-prone dimensions without
> widening every other dimension?

This is candidate 9 from the prior divergent search (“per-unit dimension masks
or bit allocation”). It changes the distance relation while keeping a bounded,
one-row-per-unit memory organization.

## Two-sentence pitch

Different neurons need different spatial evidence: a P2P coordinate that is
stable and discriminative for one footprint may be drift-dominated or
uninformative for another. We store one 9-bit binary mask beside each 5-bit
centroid/radius row, gate the corresponding absolute-difference lanes, and
calibrate the masked integer L1 radius without allocating more prototype rows.

## Alternatives explicitly not reopened

This pilot will not add:

- another unmasked radius percentile beyond the existing baseline grid;
- more prototype rows;
- a singleton-confidence flag;
- a larger static confusion graph;
- test-selected feature masks; or
- unbounded per-unit bit widths or learned floating-point weights.

Those choices would repeat completed negative branches or make the hardware
contract incomparable.

## Bit-exact row contract

For the fixed `D=9`, `B=5` descriptor:

```text
centroid[9]                 45 bits
binary dimension mask       9 bits
full-scale L1 radius        9 bits
search payload             63 bits/row
unit ID          ceil(log2 U) bits
valid flag                   1 bit
```

The radius remains a uniform 9-bit field even when fewer dimensions are
enabled. This avoids a variable row layout and supports an exact all-dimension
baseline. The masked distance is:

\[
d_u(q)=\sum_{j=0}^{8}m_{u,j}|q_j-c_{u,j}|,\qquad
\mathrm{match}(u)\iff d_u(q)\le r_u.
\]

All comparisons are inclusive. Every row must enable at least one dimension.
Ties in feature scoring use ascending feature index. Candidate unit IDs remain
stable and non-contiguous labels are supported.

### Intended mapping

The 63-bit figure is a **logical search-state payload** for a masked digital
near-memory row or a hypothetical mask-capable multi-bit L1-CiM row. It is not
a demonstrated single-macro mapping or a PPA result. The reference operation
is:

1. read/activate one 63-bit search row;
2. gate nine 5-bit absolute-difference lanes with the row mask;
3. sum only enabled lanes into the fixed 9-bit accumulator;
4. compare against the 9-bit radius; and
5. de-duplicate/emit the unit ID.

The mask may be implemented as lane clock/operand isolation in a digital
near-memory datapath or as per-dimension row/column enable metadata in a CiM
macro. No energy advantage is claimed until gated and ungated lane activity is
counted and later mapped to RTL/macro cost. Logical row bits, mask-control bits
read, full row activations, enabled lane operations, and unmodeled
gating/periphery remain separate quantities.

Candidate IDs in hardware are local row indices in `[0,U)`. Non-contiguous
external/GT unit labels are software evaluation metadata and do not set the
`ceil(log2 U)` FIFO or directory-pointer width.

## Train-only mask construction

The pilot evaluates deterministic integer-code mask builders. For unit `u`
and dimension `j`, let:

\[
w_{u,j}=Q_{0.95}(|x_{u,j}-c_{u,j}|),
\qquad
s_{u,j}=\frac{\operatorname{median}_{v\ne u}|c_{v,j}-c_{u,j}|}
{w_{u,j}+1}.
\]

The selected builders are:

1. `all_dims`: all nine dimensions, exact one-row baseline;
2. `stability`: ascending `w`, then ascending feature ID; and
3. `separation_stability`: descending `s`, then ascending feature ID.

For a one-unit toy fit, the separation term is defined as zero and stable
feature-ID ties determine the mask. These formulas use only fit-partition codes
and labels. No test event, test label, test error pair, or test candidate count
may change a mask.

Fixed mask sizes are:

```text
M ∈ {3, 5, 7, 9}
```

Adaptive mask size is out of scope for this pilot. Every emitted configuration
uses one fixed `M` from the set above; a future adaptive-size rule would need a
separate fit/calibration-only selection contract.

## Split and calibration protocol

Use a stable, equal-timestamp-safe chronological three-way split:

```text
fit          first 50%
calibration  next 25%
test         final 25%
```

- fixed `[0,1]→[0,31]` quantization is not fitted from data;
- centroids, dispersion, and mask ranking use fit rows only;
- per-unit masked radii use calibration rows only;
- if a fitted unit has no calibration event, retain its fit-derived radius,
  flag the fallback, and report the calibration count rather than silently
  deleting the row;
- test labels appear only in metrics; and
- the full configuration grid is exploratory unless a configuration is
  selected by a separately declared fit/calibration rule.

For each mask configuration and unit, the frozen calibration rule is:

\[
r_u=\left\lceil Q_p\left(
\sum_jm_{u,j}|x_{u,j}-c_{u,j}|
\right)\right\rceil,\qquad
p\in\{95,99,99.9\}.
\]

Radius zero is legal and equality is accepted. Integer ties are handled by the
inclusive comparison after the `ceil` operation. Each mask configuration is
calibrated independently. Calibration rows never change centroids, feature
scores, or mask bits. Required output includes `radius_rule`, percentile,
calibration count per unit, zero-calibration unit count, zero-radius unit
count, and calibration true-unit coverage.

As with prior component-isolation experiments, GT event times and GT-derived
fit/calibration unit rows are permitted only for the static representation
ceiling. Full-recording zero-phase filtering must remain explicit in every
output row.

## Required baselines and invariants

1. `all_dims`, `M=9` must match a masked-model reference using all-one masks.
2. A single-row `M=9` model fitted/calibrated on the same partitions must
   match the corresponding unmasked integer L1 candidate lists exactly.
3. Distance equality with the radius must match.
4. A disabled dimension must not affect the match result.
5. Fixed mask construction must be invariant to test-set perturbations.
6. Stable ties, non-contiguous unit IDs, and equal timestamps need direct
   tests.
7. Calibration-feature or calibration-label changes may change radii, but may
   not change fit-derived centroids, scores, or masks.
8. Candidate lists, fallback choices, and row comparisons for `all_dims/M=9`
   must equal an unmasked reference under the same active-row mask and
   calibration rule; its hardware payload remains 63 rather than 54 bits.

The three-way split means aggregate values need not equal the historical 50/50
one-row CSV; equivalence is evaluated under identical rows and partitions.

## Required measurements

### Candidate quality

- true-unit candidate recall;
- zero/unique/ambiguous fractions;
- mean, p95, and maximum distinct candidates;
- final candidate-only waveform teacher accuracy; and
- global-to-actual 320-bit waveform-template traffic reduction.

### Mask behavior

- enabled dimensions per row: mean/p95/max;
- per-dimension selection frequency;
- per-unit within dispersion, between separation, and final selection score;
- fit-versus-calibration or early-versus-late mask Jaccard stability;
- true-unit masked-distance/radius distribution; and
- per-unit recall, especially worst-decile unit recall.

### Hardware cost

- 63-bit search rows plus unit-ID/valid metadata;
- active rows/event;
- directory true-unit row activation recall;
- masked candidate recall conditioned on true-row activation;
- enabled absolute-difference lanes/event;
- primary and fallback Level-1 row comparisons;
- fallback absolute-difference lanes;
- directory pointer row-ID payload and declared exclusions;
- waveform-template bits read; and
- configuration writes for centroid, mask, and radius.

An apparent saving that counts fewer enabled lanes but still activates a
global row bank is not a complete energy result. Rows, lanes, pointers, and
waveform bits are reported separately.

The empty-candidate policy is frozen to the existing **unmasked all-centroid
spatial fallback**. A fallback event activates all fitted unit rows and all
nine dimensions per row; it reads no waveform template. Per event and in
aggregate, accounting must satisfy:

```text
primary rows + fallback rows = total Level-1 rows
primary enabled lanes + fallback full-D lanes = total Level-1 lanes
```

Primary/fallback rows, absolute differences, and accumulator additions must
also be summarized separately for `C=0`, `C=1`, and `C>1`.

## Validation sequence

### Phase A: no-I/O semantics

- implement the masked search core and exact memory accounting;
- freeze score formulas and stable selection;
- prove the invariants above with toy integer-boundary tests.

**Disposition:** complete. The fixed `D=9`, `B=5` core, all-dimension
equivalence, mask/radius separation, inclusive boundary behavior, stable ties,
non-contiguous labels, and exact row/lane/fallback accounting are covered by
unit tests.

### Phase B: representative cross-family pilot

The development set is frozen to four recordings:

```text
HJ      drift16c_600s_11
HJ      static16c_600s_11
MEArec  units10_snr5_seed202601
MEArec  units20_snr5_seed202601
```

The pilot is a falsification and configuration-development screen, not a paper
result. Its first gate uses **all rows active only**; directory coverage is not
allowed to select or rescue a mask.

For each non-all, `M<9` configuration, compare against `all_dims/M=9` at the
same radius percentile. Family aggregates are event-weighted and waveform
traffic is `sum(global 320-bit template bits)/sum(actual 320-bit template
bits)`. A configuration is pilot-eligible only if:

1. on every one of the four recordings, recall decreases by no more than
   0.005 and traffic reduction decreases by no more than 5%; and
2. on both family aggregates it satisfies the same one of:
   - recall-led: recall improves by at least 0.005 and traffic is no worse; or
   - traffic-led: traffic reduction improves by at least 10% and recall is no
     worse.

This prevents floating-point noise or one easy recording from being called a
Pareto movement. If several configurations pass, freeze exactly one by:

1. largest minimum recall delta across HJ and MEArec;
2. then largest minimum traffic-reduction ratio;
3. then smaller `M`;
4. then `separation_stability` before `stability`; and
5. then lower radius percentile.

Continue to confirmation only if one frozen non-all configuration passes.
In addition to the gate, report each recording's paired delta, event count,
minimum/p05/p10 per-unit recall, and a recording-level interval; do not hide a
drift or unit-density regression inside a family mean.

The pilot does not need to pass the final 0.99/4× gate, but it must pass the
formal same-direction development rule above.

**Disposition:** complete and failed. `stability/M=3/p95` moves HJ strongly in
the desired direction (recall `0.906494→0.952597`, traffic
`3.903629×→6.040536×`), but worsens MEArec traffic
(`2.485771×→2.285250×`). The closest MEArec-neutral alternative,
`stability/M=5/p95`, improves MEArec recall by only `0.002024`, below the
frozen `0.005` recall-led threshold. No single non-all configuration passes
both families and all four per-record safeguards.

### Phase C: disjoint cross-dataset confirmation

If Phase B passes, run only the one frozen configuration on the disjoint
confirmation set:

```text
HJ       remaining 10 recordings
MEArec   remaining 18 recordings
```

The four development recordings must not enter the confirmatory estimator.
Apply the final pre-registered 0.99/4× gate separately to the HJ and MEArec
confirmation families, using event-weighted candidate metrics and summed
traffic bits. A later 12-HJ/20-MEArec table may be reported only as a
descriptive full-corpus view clearly marked as including development data.

Prototype/channel directory support is evaluated only after an all-row masked
configuration passes confirmation. Its coverage/configuration must be selected
from fit/calibration data, and directory activation recall must be separated
from conditional masked-search recall.

**Disposition:** not run, as pre-registered. No configuration was frozen from
Phase B; there are no confirmation CSVs and no directory-assisted masked result.

## Kill criteria

Stop the masked-row branch if any of the following holds:

1. no single non-all configuration passes the formal four-recording
   development gate;
2. the frozen configuration fails to reach at least 0.99 candidate recall and
   4× waveform-template traffic reduction on either disjoint confirmation
   family;
3. the result depends on test-selected mask size, score mode, radius, or
   directory coverage;
4. the all-dimension invariant fails under identical partitions;
5. early/late fit mask Jaccard is below 0.5 on either family; this is an oracle
   instability diagnostic, not online write-cost evidence;
6. row/lane accounting removes the advantage relative to the
   54-bit unmasked local-SRAM/digital-L1 baseline; or
7. the positive result disappears under causal preprocessing before any
   online/discovered-row claim.

Criterion 2 is the final architecture gate; criterion 1 is the efficient
development gate. Failure of the pilot prevents a confirmation run, and
failure on the disjoint confirmation recordings prevents non-GT online
mask/radius updates. Static early/late Jaccard reports oracle mask disagreement
only; physical mask-word rewrites, radius/centroid/version writes, block
duration, and bits/event remain unmeasured until an online policy exists.

## Strongest objection

**Objection:** masking noisy dimensions also creates don't-care directions,
which can increase collisions and merely exchange true-unit misses for more
false candidates.

**Response:** that is the central falsifiable risk. The experiment compares
candidate recall and traffic jointly, includes `M=9` as an exact baseline,
reports per-unit worst-decile recall and enabled-lane cost, and stops before
online work unless the same-direction Pareto movement appears in both
families.
