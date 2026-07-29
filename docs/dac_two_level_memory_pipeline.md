# DAC Direction: Two-Level Memory Pipeline for Spike Sorting

Status: **active implementation and validation**  
Working name: **STAR-Mem — Spatial-Temporal Ambiguity-Resolved Memory**

## One-sentence claim

Use a local, 5-bit, per-unit adaptive spatial search to suppress global
template-memory traffic, then activate a compact digital waveform engine only
for spatially ambiguous spikes.

The intended contribution is not merely “CAM for spike sorting.” The base
spatial chip already uses structured local SRAM access, and prior work has used
CiM or associative memories for template matching. The defensible contribution
must be the complete co-design:

1. per-unit, recall-controlled spatial ranges rather than one pooled threshold;
2. coarse locality masking before associative search;
3. ambiguity-triggered second-level waveform access;
4. online-compatible template/range update; and
5. an accuracy–candidate-recall–memory-traffic–PPA Pareto comparison against
   the indexed-SRAM baseline.

## Current DAC readiness verdict

The repository is now a solid, auditable **algorithmic component-isolation and
negative-ablation package**. It is not yet a solid positive DAC architecture
submission because no tested configuration passes the end-to-end acceptance
gates.

| Evidence dimension | Current strength | Submission consequence |
|---|---|---|
| 5-bit row semantics and payload | Strong: bit-exact quantizer, inclusive comparisons, 54/63/74/90/144/145-bit payloads, tests | Usable as architecture specification and baseline |
| Cross-dataset Level-1 Pareto | Strong negative evidence on 12 HJ + 20 MEArec recordings | Rules out single range, singleton cascade, certified exit, and graph-only update as the main claim |
| Digital Level-2 | Strong negative boundary: full64×5 is the mandatory low-bit baseline but its accuracy depends on input scaling; pair-16 recovers accuracy but not traffic; shared dyadic M≤16 passes traffic but fails accuracy | No compact representation passes accuracy and traffic together |
| Protocol/reproducibility | Strong for GT-event/GT-row offline isolation; source/input/output hashes recorded | Defensible only with the oracle/offline boundary stated |
| Online/end-to-end sorting | Weak/incomplete: graph state only; GT events/rows; no discovered-row lifecycle | Cannot claim deployable online sorting |
| Hardware evidence | Specification only; no macro, RTL equivalence, synthesis, P&R, or PVT | Cannot claim measured/synthesized energy, area, or latency |
| Scale/generalization | Two 1D families plus a legacy 1024-channel/15-unit geometry result | New memory pipeline still lacks realistic high-local-unit-density 2D validation |

The bounded multi-prototype hypothesis is also complete and negative. `P≤4`
temporal/farthest-L1 rows do not produce a joint 0.99-recall/4× point on
either family. HJ's 4×-class result reaches only 0.898 recall and requires four
rows/unit; MEArec has no 4× point. This closes “different fixed ranges for
different local modes” as the next online branch rather than leaving it as an
untested rescue argument.

The subsequent masked and power-of-two weighted single-row pilots are also
complete. They demonstrate that per-unit coordinate relevance materially
changes HJ candidate geometry, but neither supplies a frozen cross-family
policy: both formal development gates select zero configurations and therefore
preserve the disjoint confirmation recordings.

The completed alignment-conditioned temporal-sketch pilot isolates Level 2
behind a fixed p99.9 source that does clear `0.99` candidate recall. Structured
signed-5-bit dyadic templates of 40/60/80 bits per unit all exceed the exact
`4×` template-bit target, but every one of the 15 delay/width configurations
fails temporal accuracy. The least-bad common point, delay 8 and `M=16`, loses
`6.10/12.08` percentage points on HJ/MEArec. A raw-prefix float ceiling is
already within one point on both families at delay 16, so the failure is the
compressed shared representation rather than insufficient waveform horizon.

The paper should therefore not use “Range-CAM singleton early exit” as its
headline. The defensible next claim, if the remaining experiments succeed, is
**failure-guided local near-memory retrieval with ambiguity-conditioned digital
refinement**. CAM may remain one implementation point, but indexed SRAM plus
digital L1/WTA is the mandatory baseline and may be the final macro choice.

## Problem exposed by the current repository

The current two-stage implementation describes Stage 1 as using “per-unit
thresholds,” but `calibrate_thresholds` and `calibrate_thresholds_codes` pool
all within-unit distances into one global percentile. A high-variance unit can
therefore widen the candidate region for every low-variance unit. Conversely,
the logical AND of two separately calibrated pooled gates can exclude the true
unit more often than the nominal percentile suggests.

This is both an algorithmic weakness and a hardware opportunity: a CAM/CiM row
naturally stores row-specific state, so each unit should store its own search
range.

## Proposed pipeline

```text
detected/aligned event
        │
        ▼
K-neighbor P2P + COM ──► 5-bit spatial descriptor
        │
        ▼
central-channel/tile directory (coarse SRAM index)
        │                       activates local rows only
        ▼
Level 1: per-unit Range-CAM or local multi-bit L1 CiM
        │
        ├── |C| = 1 ───────────► early label, no waveform read
        │
        ├── |C| > 1 ───────────► Level 2 digital waveform refinement
        │                         (candidate template SRAM + SAD/features)
        │
        └── |C| = 0 ───────────► progressive widening, then spatial fallback
```

The associative array must not search all units globally. The coarse directory
maps the event's central channel/tile to a small row mask. Only those match
lines or CiM rows are activated.

The first cross-dataset sweep shows that one range is not sufficient for both
early exit and high recall. A subsequent nested-range experiment also shows
that an unqualified unique inner match is not safe enough. The following
calibrated variant was therefore evaluated as a bounded branch:

```text
tight inner box ── unique + certified row ──► early exit
        │
        └── zero/ambiguous/uncertified ─────► wide outer L1 search
                                                   │
                                                   ▼
                                           waveform candidates
```

This branch is still a two-level *template-memory* pipeline: the inner/outer
searches are spatial row operations, while waveform SRAM remains the second
template level. It also fails the joint gate: strict certification produces no
early exits, while relaxed certification is too rare and insufficiently
precise. The next evaluated Level-1 branch was a deterministic local candidate
fetch—directory-indexed SRAM/CiM or near-memory distance—using the same 5-bit
per-unit descriptors without treating a unique match as a confidence proof.
That branch also failed and is retained as the mandatory hardware baseline,
not as a passing candidate policy.

## Level 1: 5-bit adaptive spatial search

### Descriptor

For `K` local electrodes:

\[
q = Q_5([\mathrm{COM}_x,\mathrm{COM}_y,
         \hat p_0,\ldots,\hat p_{K-1}]),
\qquad q_d\in[0,31].
\]

`COM` and max-normalized P2P are both bounded in `[0,1]`. The quantizer is
fixed or fitted on the training partition only; test statistics must never set
the code range.

For the default `K=7`, the descriptor has `D=9` dimensions and needs 45 input
bits.

### Mapping A: per-row L1 radius

Each unit row stores an integer centroid `c_u` and its own radius `r_u`:

\[
u\in C(q) \iff \sum_d |q_d-c_{u,d}| \le r_u.
\]

`r_u` is calibrated from that unit's training distribution. This maps to a
multi-bit associative/CiM distance row plus a row-local threshold. Ignoring row
ID and update metadata, storage is

\[
D B + \lceil\log_2(D(2^B-1)+1)\rceil
\]

bits per row: 54 bits for `D=9`, `B=5`.

### Mapping B: per-row axis-aligned range

Each row stores `[l_{u,d}, h_{u,d}]` and matches when every feature is in its
inclusive interval:

\[
u\in C(q) \iff \bigwedge_d l_{u,d}\le q_d\le h_{u,d}.
\]

This is directly compatible with Range-CAM semantics and supports a different
range in every feature and unit. Its raw row storage is `2DB`: 90 bits for
`D=9`, `B=5`.

The range builder must target *joint* per-unit coverage, not independently
apply the same percentile to each dimension and assume the product remains at
the target.

### Progressive widening and fallback

An empty candidate set triggers small integer widening steps. The first
non-empty level terminates the search. If all configured levels miss, the
controller selects the row with the smallest normalized range violation or
nearest integer centroid. Widening/fallback is reported separately; it must not
be hidden inside “candidate recall.”

The initial and widened candidate sets must both use inclusive comparisons.
This matters at 5-bit boundaries, where an exclusive comparison can reject
samples identical to a stored radius or endpoint.

Measured result: zero-only widening is only a safety mechanism. It improves
cross-dataset recall by at most a few tenths of a percentage point because many
true-row misses already contain a wrong candidate and therefore never enter
the empty-set path. The main recall mechanism must use an observable
confidence/margin trigger or a fixed outer range.

## Level 2: digital waveform refinement

The implemented baseline is candidate-only signed/absolute 4–5-bit SAD over
the full waveform or a training-selected set of taps. It is a stronger
reference than the current five-dimensional Peak-FSDE feature, while avoiding
the storage expansion of a full bipolar thermometer CAM.

Candidate representations to compare:

| Representation | Example template bits | Main operations |
|---|---:|---|
| Full 64-sample, 5-bit waveform | 320 | 64 abs + accumulation |
| 48 selected taps, 5-bit | 240 + one shared 288-bit index table | 48 abs + accumulation |
| 12 morphology features, 5-bit | 60 | streaming min/max/sums |
| Dyadic prefix sketch, signed 5-bit | 40/60/80 + shared 56/84/112-bit ID table | streaming add/sub/shift + 8/12/16-code SAD |
| 64-sample, 4+4 thermometer cells | 512 | binary CAM/Hamming |

On a tight p95 spatial candidate source, Fisher-48×5 remains within 0.52pp
(HJ) and 0.66pp (MEArec) of the 64-sample float teacher. That candidate source
does not meet the 99% recall gate. At the p99.9 high-recall source, no shared
tap count from 8 through 56 meets the one-point target on both families; the
best HJ result is Fisher-40 at −1.59pp. Full 5-bit×64 SAD is −2.64pp on HJ but
only −0.49pp on MEArec. The fixed morphology-12 feature is substantially worse
and is rejected.

Pair-conditioned taps repair the accuracy side but not the memory side.
Pair-16×5 is +0.04pp/−0.97pp versus the teacher on HJ/MEArec at p99.9 spatial
recall, but conservative bit-traffic reduction is only 1.35×/1.36× and the
all-pair tap-index tables increase storage to about 18.8/16.1 kbit per
recording. Pair-4 reaches about 5.4× traffic reduction but loses
3.11pp/6.64pp. No tested pair width passes both gates.

The shared dyadic prefix sketch repairs the traffic side but not the accuracy
side. On the four locked development recordings, all `M∈{8,12,16}` widths
reach at least `5.02×` family template-bit reduction. The least-bad shared
configuration still loses `6.10/12.08pp`, and the formal selection is null.
This branch is retained as a structured shift/add extractor cost point and a
negative representation result, not as the selected Level-2 engine.

Only persistent local ambiguities should allocate waveform child templates.
Because singleton confidence has failed, this allocation decision must come
from a deterministic candidate graph/directory or a future discovered-cluster
policy, not from a unique Range-CAM match.

## Required evaluation protocol

### Isolation experiment: candidate search

Use chronological 50/50 train/test and ground-truth unit IDs only to isolate
the memory-search quality. Compare:

1. current pooled COM-and-P2P gate;
2. pooled 5-bit L1 radius;
3. per-unit 5-bit L1 radius;
4. per-unit 5-bit range box;
5. the two adaptive variants with progressive widening; and
6. all variants with and without a local row mask.

Report:

- true-unit candidate recall;
- zero, unique, and ambiguous fractions;
- mean, p95, and maximum candidate count;
- Stage-2 activation fraction (`|C| > 1`);
- mean/p95 active rows and row-comparison reduction;
- widening and fallback rates;
- row bits and candidate waveform-template reads; and
- final candidate-only waveform accuracy.

The target is not the smallest candidate set in isolation. It is the Pareto
frontier at **at least 99% true-unit candidate recall**.

The current isolation runners apply a zero-phase Butterworth `filtfilt` to the
complete recording before the event split. Their chronology prevents
test-label/state configuration leakage, but the filtered samples are not a
causal streaming input: samples on either side of the split boundary influence
the filter output. Therefore these results must be called **offline,
chronologically fitted component isolation**, not a deployable causal signal
path. A causal implementation needs a forward filter with declared state and
warm-up, or split-local filtering with a guard interval and discarded boundary
events.

A forward-only stateful SOS filter primitive is now implemented and verified
for chunk equivalence, reset, finite output, and no future-sample leakage. It
has not been integrated into the GT-event candidate runners because the
pre-registered workflow stops a branch when its more favorable static
representation ceiling fails. Centered descriptor windows, 49 post-event
waveform samples, detector state, warm-up, and discovered-row lifecycle remain
separate requirements.

### End-to-end experiment

After the oracle isolation study, repeat with discovered/online unit rows and
detected events. Report detection precision/recall/F1 and per-unit sorting
precision/recall in addition to Hungarian clustering accuracy. Exact
simultaneous timestamps must remain separate events; do not recover labels via
a `time -> unit` dictionary.

### Hardware comparison

The minimum credible comparison set is:

- original coarse-addressed local SRAM scan;
- global digital waveform-template scan;
- global CAM/CiM spatial search;
- coarse-directory + local adaptive Range-CAM;
- coarse-directory + local L1 CiM; and
- local adaptive search + ambiguity-triggered digital waveform SRAM.

Cell-only arithmetic is insufficient for DAC. Include encoders, subtract/abs or
range-compare logic, match-line sensing, ADC/WTA where applicable, directory,
FIFO/controller, update writes, and waveform SRAM peripherals.

## Acceptance and kill criteria

Continue the architecture only if the checked experiments show:

1. test candidate recall of at least 99% on both HJ and MEArec families;
2. at least 4× fewer global waveform-template reads than scanning all units;
3. a compact digital waveform representation within 1 percentage point of the
   64-sample waveform teacher;
4. a credible energy/area advantage over local indexed SRAM plus digital
   distance, not only over a global brute-force baseline;
5. results on a realistic high-density 2D probe; and
6. no test-set or ground-truth dependence in the final online configuration.

If Range-CAM loses to the indexed-SRAM implementation after peripheral costs,
retain the two-level algorithm but use local SRAM-CiM/near-memory search. Do not
force CAM into the paper title.

Current gate status:

- The single-range design does **not** pass criteria 1 and 2 jointly.
- A 99.9th-percentile per-unit L1 radius reaches mean candidate recall
  0.993/0.995 on HJ/MEArec, but only about 1.25× waveform-read reduction.
- Tight boxes provide larger read reduction but remain below 99% recall.
- The uncertified nested inner-box/outer-L1 policy also fails the joint gate.
  Its best 4×-class point is HJ coverage 0.75 at 0.956 recall and 4.14× read
  reduction; its highest-recall MEArec point is 0.987 recall at only 1.33×.
- Inner singleton precision is only 0.88–0.96 on HJ and 0.90–0.92 on MEArec,
  so unique match alone cannot authorize a safe early exit.
- The disjoint-calibration, per-unit safe-exit experiment also fails the joint
  gate. A certified unit costs one row-local flag bit; uncertified singletons
  invoke outer L1. A 0.99 Wilson
  target certifies no row on either family. Relaxing the target to 0.95
  produces only 5.8% HJ and 5.2% MEArec test early exits, with 1.38× and 1.36×
  read reduction respectively.
- Closed Level-1 accounting makes the cascade less attractive, not more. The
  HJ inner-0.75 dual point performs 23.51 row accesses/event versus 18.07 for
  one all-row stage; its certified counterpart performs 35.17. The analogous
  MEArec inner-0.50 dual/certified points perform 24.29/32.78 accesses/event
  versus 16.72 for one stage. These counts exclude circuit peripherals.
- Criterion 3 is only conditionally passed: Fisher-48×5 is within one point of
  the float teacher behind a p95 spatial gate, but no shared-tap configuration
  passes on both families behind the p99.9 high-recall gate.
- The Range-CAM singleton-exit policy is therefore no longer the assumed main
  Level-1 mechanism. A deterministic indexed local SRAM/near-memory candidate
  engine using the same 5-bit descriptors was evaluated next.
- The first deterministic indexed branch is also negative. At 4×-class traffic,
  its learned graph reaches only 0.932 HJ and 0.938 MEArec recall. Its
  highest-recall point reaches 0.980/0.986 but only 1.85×/1.53× reduction.
- Graph-state block-causal updates over the offline zero-phase-preprocessed
  stream also fail. Even delayed-GT updates reach only
  0.947/0.934 HJ/MEArec recall at 4×-class traffic; their highest-recall points
  reach 0.991/0.994 but only 1.42×/1.28×.
- Another static range, adjacency table, or graph-only online rule is not
  justified. A future online experiment must update centroid/radius/directory
  state using an observable non-GT confidence rule and count those writes.
- Bounded local-mode rows do not rescue the range formulation. The completed
  `P=1..4` sweep finds HJ recall 0.898 at its best ≥4× point and no ≥4×
  MEArec point. The highest-recall HJ/MEArec points are 0.993/0.995 but remain
  at about 1.25× traffic reduction. The HJ traffic point also needs mean
  `P=4`, violating the pre-registered mean-`P≤2` target.
- Unit-specific binary feature masks reveal a useful HJ-only effect but do not
  supply a cross-family policy. On the two HJ development recordings,
  `stability/M=3/p95` raises recall from 0.906 to 0.953, raises waveform
  reduction from 3.90× to 6.04×, and cuts enabled lanes/event from 96.23 to
  33.14. On two MEArec development recordings, the same configuration raises
  recall only from 0.956 to 0.961 while worsening reduction from 2.49× to
  2.29×. The formal pilot gate has zero candidates, so no disjoint confirmation
  or directory-assisted masked sweep was run.
- Nonzero power-of-two weights also change the HJ candidate geometry but fail
  as a cross-family policy. At p95, `stability/top3_x4` moves HJ recall from
  0.9065 to 0.9084 and traffic reduction from 3.90× to 7.90×, while MEArec
  recall moves from 0.9561 to 0.9555 and traffic improves only from 2.49× to
  2.59×. The formal four-recording gate has zero candidates; its deterministic
  selection artifact is `null`, so no confirmation or directory-assisted
  weighted run was authorized.
- The fit-selected signed-5-bit dyadic temporal sketch also fails its frozen
  four-recording gate. Its p99.9 Level-1 source passes recall
  (`0.9916/0.9942` event-weighted on HJ/MEArec), and `M=16` provides
  `5.92×/5.02×` template-bit reduction, but the least-bad shared delay-8 point
  loses `6.10/12.08pp` relative to the full fit-scale float teacher. The
  selection contains zero candidates, so confirmation is prohibited.

## Current evidence and open gaps

- The existing low-bit sweep shows that 5-bit P2P preserves the HJ
  `soft_loc` oracle closely (about 0.914 versus 0.916 for float under
  per-spike-max quantization), so 5-bit is a reasonable starting point.
- Existing cross-dataset studies show 4-bit code-L1 can match the float
  unsupervised pipeline, but they use a pooled radius.
- The implemented 5-bit per-unit L1 row uses 54 payload bits for the default
  nine-dimensional descriptor; the per-feature Range-CAM box uses 90 bits.
- At 95% calibration, per-unit L1 reduces mean candidates relative to pooled
  L1 from 6.49 to 5.37 on HJ and from 6.02 to 5.52 on MEArec, with a small
  recall loss.
- High-recall single-range operation is too broad: 99.9% L1 produces 13.14 HJ
  candidates and 11.91 MEArec candidates on average.
- One home-channel pointer is too restrictive. Per-unit multi-channel support
  raises directory recall to 0.998 on HJ and 0.996 on MEArec at 99% training
  coverage, but reduces only 9.0% and 24.3% of active rows, respectively.
- A raw unshared inner-box plus outer-L1 row needs 144 payload bits for `D=9`,
  `B=5` (90 + 54), before unit ID, directory, flag, sensing, or peripherals.
- The uncertified nested policy was evaluated on all 12 HJ scenes and 20
  MEArec files. No tested inner coverage reaches both 0.99 event-weighted
  candidate recall and 4× event-weighted waveform-read reduction.
- The current waveform thermometer representation is large; candidate-only
  digital refinement is therefore the preferred Level-2 baseline.
- The compact Level-2 sweep is complete. Fisher-48×5 passes the teacher-within-
  1pp target only with the p95 gate, whose candidate recall is 0.936/0.947.
  No tested shared-tap representation passes on both families with p99.9
  high-recall candidates; morphology-12 is a negative result.
- Pair-conditioned taps are also complete. They recover high-recall teacher
  accuracy at 16 taps but fail traffic and O(U²) storage; 4 taps passes traffic
  but fails accuracy.
- The alignment-conditioned dyadic prefix pilot is complete. Its 60 rows bind
  the exact four input files, 55-file runtime source snapshot, one-scan
  Level-1 arithmetic, C0/C1/C>1 temporal accounting, and deterministic gate.
  The raw-prefix float ceiling reaches the teacher by delay 16, but no
  40/60/80-bit shared dyadic template preserves accuracy.
- Per-unit range search, local row activation, candidate/read Pareto curves,
  and collision-safe GT event alignment are now scripted and tested.
- Bounded multi-prototype rows, prototype-specific directory supports, stable
  unit de-duplication, and fallback-inclusive Level-1 accounting are scripted,
  tested, and rejected by their pre-registered static gate.
- One-row/unit masked-L1 search, deterministic fit-only dimension masks,
  calibration-only radii, fixed Wilson intervals, and fallback-inclusive
  row/lane/addition accounting are scripted and tested. The four-recording
  development pilot is HJ-positive but fails its pre-registered cross-family
  gate; it is not a full-corpus result.
- The completed nonzero `{1,2,4}` power-of-two weighted relation keeps one
  74-bit logical row/unit and all nine coordinates. It is HJ-positive for
  traffic but fails the same-branch cross-family development gate; static
  mask/weight profile tuning is no longer the next main branch.
- Per-unit safe-exit certification and compact digital waveform baselines are
  complete. Deterministic indexed candidate retrieval is also complete and
  negative. Graph-only state updates, bounded multi-prototype ranges,
  cross-family masked/weighted rows, and pair-conditioned waveform features
  are complete and negative under their gates. A causal SOS primitive exists,
  but
  integrated causal event
  preprocessing, centroid/radius/directory update, discovered-row integration,
  realistic 1024-channel data, and synthesized hardware cost remain open.

Detailed results and commands:
[`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md) and
[`compact_waveform_refinement.md`](compact_waveform_refinement.md), plus
[`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md)
and
[`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md).
The later development-only feature-mask result is in
[`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md).
The following nonzero-weight result and stop decision are in
[`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md).
The completed temporal-sketch gate and representation diagnosis are in
[`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md), with
the exact snapshot in
[`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md).
The bit-exact memory/controller contract is in
[`hardware_mapping_spec.md`](hardware_mapping_spec.md).
The current source, input, and generated-CSV hashes are in
[`dac_experiment_provenance.md`](dac_experiment_provenance.md). The later
bounded-row snapshot is kept separately in
[`adaptive_multiprototype_5bit_provenance.md`](adaptive_multiprototype_5bit_provenance.md)
so the earlier result manifest is not silently replaced. The four-recording
masked pilot is locked separately in
[`adaptive_masked_5bit_provenance.md`](adaptive_masked_5bit_provenance.md).
The weighted pilot and null confirmation authority are locked separately in
[`adaptive_weighted_5bit_provenance.md`](adaptive_weighted_5bit_provenance.md).

## Implementation checklist

- [x] 5-bit train-fit spatial quantizer.
- [x] Per-unit L1-radius and range-box search primitives.
- [x] Inclusive comparisons, local row masks, progressive widening, fallback.
- [x] Monotonic joint-coverage box construction.
- [x] Unit tests for non-contiguous IDs, joint coverage, and bit accounting.
- [x] Collision-safe event labels and same-timestamp split handling.
- [x] HJ and MEArec candidate-search sweep.
- [x] Candidate-only waveform accuracy and memory-read accounting.
- [x] Single-home and multi-channel-support directory comparison.
- [x] Uncertified nested inner-box/outer-L1 trigger experiment.
- [x] Disjoint-calibration per-unit safe-exit experiment.
- [x] Compact selected-tap/morphology waveform sweep.
- [x] Deterministic indexed/confusion-graph candidate retrieval.
- [x] Block-causal graph-state rolling update ceiling on offline features.
- [x] Bounded `P=1..4` multi-prototype spatial-row static upper bound.
- [x] Unit-specific masked-L1 5-bit development pilot and formal stop decision.
- [x] Unit-specific nonzero power-of-two weighted-L1 development pilot and
  immutable no-confirmation decision.
- [x] Alignment-conditioned signed-5-bit dyadic temporal-sketch pilot,
  deterministic null selection, and no-confirmation decision.
- [x] Stateful forward-only SOS filter primitive and leakage tests.
- [ ] Non-GT centroid/radius/directory online update for a static policy that
  first passes the representation gate.
- [x] Pair-conditioned digital waveform feature sweep.
- [ ] Integrated causal raw-signal filtering and boundary-state protocol.
- [ ] Unsupervised/discovered-row integration.
- [ ] Online per-unit range and directory update.
- [ ] New-pipeline 1024-channel, high-local-unit-density 2D evaluation.
- [ ] RTL, SRAM/CAM macro model, synthesis, and PVT analysis.

## Relevant repository studies

- [`p2p_cim_approx.md`](p2p_cim_approx.md)
- [`central_ratio_vs_maxnorm_benchmark.md`](central_ratio_vs_maxnorm_benchmark.md)
- [`waveform_cam_thermometer.md`](waveform_cam_thermometer.md)
- [`soft_loc_error_analysis.md`](soft_loc_error_analysis.md)
