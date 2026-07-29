# Adaptive Multi-Prototype 5-bit Spatial Rows

Status: **implemented, evaluated, and rejected by the pre-registered static
kill criteria**

The completed result record is
[`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md).
The bounded row bank remains a tested baseline and reusable primitive, but the
online pilot below was intentionally not started: the GT-row static upper bound
did not pass the recall/traffic gate.

## Starting point

The current single-row descriptor is hardware-clean but fails the joint gate:
a tight row is traffic-efficient and misses true units, while a high-recall row
becomes nearly global. Drift scenes and high unit density expose the failure
most strongly. More static thresholds, singleton flags, or graph-only pointer
rules are not justified by the completed negative results.

The next question is whether the broad per-unit distribution is genuinely one
large range or a small number of local modes caused by drift, channel handoff,
or footprint state. If a unit can be represented by a bounded number of narrow
5-bit rows, the memory can search local modes while the controller de-duplicates
their common unit ID before waveform access.

## Divergent candidates considered

The following candidates are retained so rejected directions do not silently
reappear:

1. one wider static per-unit L1 radius;
2. another marginal-quantile Range-CAM box;
3. more singleton certification thresholds;
4. a larger static confusion-pointer graph;
5. bounded multi-prototype L1 rows per unit;
6. temporal-epoch prototypes for drift segments;
7. one base centroid plus a low-bit signed drift residual;
8. event-quality/SNR-conditioned radii;
9. per-unit dimension masks or bit allocation;
10. coarse 3-bit search followed by local 5-bit re-ranking;
11. multi-scale `K=3/K=7` spatial rows;
12. bounded streaming k-median row allocation/merge;
13. prototype-specific channel/tile directory support;
14. margin-triggered search expansion;
15. risk-controlled/conformal candidate widening;
16. spatial LSH or code-prefix buckets;
17. ambiguity-edge-specific waveform child templates; and
18. shared tap dictionaries referenced by ambiguity edges.

## Convergence

| Candidate | Main benefit | Main risk | Disposition |
|---|---|---|---|
| Bounded multi-prototype rows | Directly attacks multimodal/drift-induced broad ranges; maps to repeated 54-bit rows | Row/storage growth and duplicate unit hits | **Selected pilot** |
| Base row + drift residual | Low update bandwidth and compact temporal state | Residual range may still be unimodal and broad | Integrate as an update option after pilot |
| Coarse 3-bit + local 5-bit | Reduces input/search switching | Coarse collisions may erase candidate benefit | Retain as hardware follow-up |
| Prototype-specific directory | Makes physical locality first-class | Pointer growth and channel handoff misses | Required companion measurement |
| Ambiguity-edge waveform child | Avoids O(U²) all-pair tables | Depends on a stable non-GT ambiguity graph | Defer until Level 1 passes |

The first four static variants above are rejected because completed experiments
already show that widening a fixed relation trades traffic directly for recall.
The selected pilot changes the representation class rather than adding another
threshold.

## Two-sentence pitch

Per-unit spatial footprints are not always one stationary range: drift and
channel handoff can form a few local 5-bit modes whose union is much tighter
than one high-recall radius. We represent each discovered unit with a bounded
bank of local centroid/radius rows, search those rows in local memory, and
de-duplicate matching row IDs into unit candidates before
ambiguity-conditioned waveform refinement.

## Bit-exact row contract

For `D=9`, `B=5`, each prototype row initially stores:

```text
prototype_centroid[9]    45 bits
prototype_radius          9 bits
unit_id                    ceil(log2 U) bits
prototype_valid            1 bit
```

The search payload remains 54 bits/row before the ID/valid metadata. A unit may
own at most `P_max` rows. Multiple matching prototypes from one unit produce
one candidate unit and therefore must not duplicate waveform-template reads.

Required counters:

- configured prototypes/unit: mean, p95, and maximum;
- prototype row storage and unit-ID/valid metadata;
- active prototype rows/event;
- distinct candidate units/event;
- duplicate prototype hits removed/event;
- prototype create, centroid update, radius update, merge, and eviction writes;
- directory pointers/prototype and pointer rewrites;
- waveform-template bits read; and
- update rejection/acceptance by confidence reason.

## Static pilot

Fit-only labels are allowed only for the first component-isolation upper bound.
For each unit:

1. stably sort its training events chronologically;
2. fit `P=1,2,3,4` deterministic low-bit prototypes using train rows only;
3. calibrate one integer radius per prototype;
4. query all enabled prototype rows;
5. union/de-duplicate unit IDs; and
6. use the same candidate-only waveform teacher for every configuration.

Compare:

- single per-unit L1 row;
- fixed `P` rows/unit;
- adaptive `P≤P_max` selected by a train-only split/merge criterion; and
- the same rows with prototype-specific directory masks.

Report both prototype-row comparisons and distinct waveform candidate reads.
An apparent traffic gain caused only by counting duplicate prototype hits as
one row access is invalid.

### Completed disposition

The full 12-recording HJ and 20-recording MEArec sweep completed on the frozen
grid. No family has a point that simultaneously reaches 0.99 candidate recall
and 4× waveform-template traffic reduction. The HJ 4×-class point also needs
four prototypes for every unit; MEArec has no 4× point at all. This triggers
kill criterion 1, with the HJ point also violating the intended mean-`P≤2`
storage bound in criterion 2.

## Online pilot

The deployable branch may not use held-out GT labels or test-set hyperparameter
selection. Query state is read before the current event and may be updated only
after its prediction.

Observable update gates must include:

- predicted-unit membership in the fetched candidate list;
- integer spatial distance/radius ratio;
- best-versus-second waveform distance margin when at least two candidates are
  present;
- agreement between the spatial anchor and waveform decision; and
- a refractory/minimum-support rule before row creation.

Evaluate three update sources under otherwise identical state:

1. `no_update`;
2. `oracle_delayed_label`, explicitly non-deployable and used only as a ceiling;
3. `self_predicted_confident`, with no held-out GT access.

The online row bank must have bounded create/merge/evict behavior. A successful
result must count every state and directory write; silently allocating an
unbounded prototype per event is a failure.

**Disposition:** not executed. Implementing a self-updating row lifecycle
cannot establish the required operating point after the non-deployable static
GT-row representation ceiling has already failed it.

## Validation experiments

1. **Static representation test:** Does `P≤4` move the HJ and MEArec
   candidate-recall/waveform-read Pareto beyond the single-row frontier?
2. **Temporal update test:** Does a bounded self-predicted prototype bank retain
   the static gain under later held-out time blocks without GT configuration?
3. **Hardware feasibility test:** At the best algorithmic point, does total
   prototype-row, directory, update, selection, and waveform traffic retain an
   advantage over one-row indexed SRAM plus digital L1?

## Kill criteria

Stop the multi-prototype branch if any of the following holds:

1. no HJ and MEArec point simultaneously reaches `≥0.99` candidate recall and
   `≥4×` global waveform-template bit-traffic reduction;
2. the gain requires more than two prototype rows/unit on average or more than
   four for any unit;
3. prototype-row comparisons or directory traffic erase the waveform-memory
   advantage under the symbolic energy model;
4. self-predicted updates lose more than one percentage point of candidate
   recall relative to the delayed-oracle ceiling;
5. the result depends on full-recording zero-phase filtering, test labels, or
   test-selected thresholds; or
6. row allocation grows monotonically without stable merge/eviction.

## Two-week feasibility pilot

### Week 1

- implement deterministic fixed/adaptive prototype fitting and de-duplicated
  query semantics;
- add exact row/storage/access accounting and unit tests;
- run static `P=1..4` HJ/MEArec sweeps;
- stop immediately if the static upper bound cannot move the joint frontier.

Completed. The stop condition fired.

### Week 2

- add block-causal bounded centroid/radius/create/merge updates;
- add prototype-specific directory support and write counters;
- compare no-update, delayed-oracle, and confident-self update sources;
- repeat with causal-filter/guarded preprocessing before interpreting an online
  result as deployable.

Not entered because the Week-1 static gate failed. A forward-only stateful SOS
filter primitive was implemented and unit-tested independently, but it is not
presented as an integrated causal pipeline.

## Strongest objection

**Objection:** multiple rows merely duplicate each unit until search cost and
storage approach a global scan; self updates will also poison the bank.

**Response:** the experiment treats these as kill conditions rather than hidden
costs. The row bank is bounded, candidate IDs are de-duplicated, every row and
pointer access/write is counted, and self updates are compared against both a
frozen baseline and a delayed-oracle ceiling. If the static upper bound needs
an unbounded bank or the self branch cannot retain it, the hypothesis is
rejected rather than rescued with looser reporting.
