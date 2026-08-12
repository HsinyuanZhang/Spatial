# Level-1 Candidate Search — Improvement Roadmap

Status: **Active planning document.** Written 2026-08-12 after the
shape-orthogonal descriptor branch closed with no selected configuration.

This is a decision document, not a result record. It exists because roughly ten
distinct Level-1 approaches have now failed the same joint gate, and the
accumulated negative evidence has become specific enough to redirect the work
rather than to suggest another variation.

## The gate that keeps failing

> True-unit candidate recall at or above 0.99 **and** at least 4x reduction in
> waveform-template reads, on both the HJ and MEArec families, at the same
> operating point.

Every branch below cleared some local gate and none cleared this one.

| Branch | Best observation | Why it failed |
|---|---|---|
| Per-unit L1 / range | recall 0.993 HJ / 0.995 MEArec | only about 1.25x traffic |
| Inner-box plus outer-L1 | a 4x-class point on HJ | recall 0.956 there |
| Wilson-certified singleton exit | clean safe-exit semantics | strict target certifies nothing; relaxed gives 1.36–1.38x |
| Static indexed adjacency | 5.31x HJ / 4.08x MEArec | recall 0.932 / 0.938 |
| Block-causal graph update | explicit online accounting | delayed-GT upper bound only 0.947 / 0.934 |
| Up to four prototypes per unit | represents local modes | HJ best 4x-class recall 0.898 |
| Masked L1 | strong HJ movement | does not generalize to MEArec; null |
| Power-of-two weighted L1 | large HJ traffic movement | MEArec flat or negative; null |
| Tile activation / cosine | tile routing clears local gates | does not address candidate traffic |
| Shift-min multi-copy rows | +0.4 pp argmin | 5 rows per unit for nothing on the frontier |
| **Shape-orthogonal descriptor** | **+10.65 / +8.21 pp assignment** | **no frontier movement at all** |

Ten approaches, one common structure: each changed the **scoring function** or
the **row format**, and none changed the fact that a single acceptance region
per unit must cover that unit's entire intra-unit spread.

## The sharpest diagnostic we have

The shape-descriptor branch produced the most informative negative result in
the series, because it decoupled two things that were previously confounded.

| Arm | Normalized intra-unit distance | Positive margin | Mean candidates at ≥0.99 recall | Traffic |
|---|---:|---:|---:|---:|
| `d9`, HJ | 0.0670 | 0.798 | 8.07 | 1.30x |
| `com_posneg`, HJ | **0.0232** | **0.917** | **7.68** | 1.37x |
| `d9`, MEArec | 0.0803 | 0.773 | 12.06 | 1.24x |
| `com_posneg`, MEArec | **0.0308** | **0.849** | **12.84** | 1.16x |

The unit clouds became about **2.9x tighter** and the true-versus-wrong margin
improved by 12 and 8 points. The candidate set at 0.99 recall did not shrink,
and on MEArec it grew.

That result rules out the most natural hypothesis in the whole programme:
**intra-unit variation is not the binding constraint.** Making each unit's
typical spike sit closer to its own template does not reduce how many units
must be admitted to retain the true one 99% of the time.

### What this implies mechanically

If the bulk of each unit's distribution tightens by 3x and the radius needed
for 0.99 recall does not, then the radius is not set by the bulk. It is set by
the **tail**: the small fraction of events that sit far from their own template
regardless of representation. Those are the overlapping spikes, the low-SNR
events, the drift excursions, and the misaligned detections.

A single acceptance region must be large enough to contain that tail, and a
region that large admits many foreign units. Improving the representation
compresses the bulk and leaves the tail roughly where it was.

This reframes the problem. Level-1 is not a representation problem. It is a
**tail-handling and policy problem**.

## Directions, in priority order

### 1. Separate the tail instead of covering it

**Idea.** Stop asking one acceptance region to deliver 0.99 recall. Use a tight
radius sized for the well-behaved bulk, accept that some events return an empty
or wrong candidate set, and give those events an explicit second path.

The average cost is then dominated by the bulk. If 90% of events resolve with
two or three candidates and 10% escalate, the mean template read count can fall
far below what a single 0.99-recall radius requires.

**Why it is worth trying now and was not before.** The dual-range pipeline
tested inner-box plus outer-L1 on the `d9` descriptor and reached recall 0.956.
It was evaluated on a descriptor whose bulk and tail were only 0.067 apart in
normalized distance. On `com_posneg` the bulk is at 0.023 while the tail still
governs the radius, so the *separation* between well-behaved and pathological
events is now roughly three times larger. A tight-radius plus escalation policy
is a genuinely different experiment on this descriptor than it was on `d9`.

**What to measure.** Escalation rate, per-event cost split by path, and total
template reads including the escalation path. The honest metric is the mean over
all events, not the mean over the fast path.

**Kill criterion.** If total traffic including escalation does not beat 2x at
0.99 end-to-end recall on both families, the escalation architecture is not the
answer either, and that should be recorded as strongly as the ten entries above.

### 2. Ask whether the gate is well posed

**Idea.** With 10 to 20 units and a 0.99 recall requirement, 4x reduction means
admitting roughly 2.5 to 5 units out of 10 to 20 while missing the true unit at
most once in a hundred events. It is not obvious that any single-region policy
can do this, and ten failures are weak evidence that it cannot.

Two specific questions deserve a written answer before more engineering:

- **Is 0.99 the right recall target?** It is only meaningful if a Level-1 miss
  is unrecoverable. If a cheap Level-2 or a fallback can recover a fraction of
  misses, the system-level requirement is lower than 0.99 and the whole frontier
  moves.
- **Is template count the right cost unit?** The gate counts distinct template
  reads. If templates are small and the real cost is energy per row comparison
  or per activated tile, then a policy that reads more templates but activates
  fewer rows could win while scoring badly on the current metric.

**What to produce.** Not an experiment. A short analysis note that either
re-derives the gate from a stated cost model or states plainly that the gate is
a target inherited without derivation. Either outcome is progress; the current
situation, where every branch is measured against a threshold nobody has
justified, is not.

### 3. Multiple acceptance regions per unit, tested properly

**Idea.** A single region per unit is an assumption, not a requirement. CAM
arrays store rows; multiple rows per unit is native.

**Status.** Genuinely unresolved, and the existing evidence is weaker than it
looks. The multiprototype study tested P ≤ 4 and was negative. A separate
exploratory check during this branch found kNN-15 reaching 0.872 against 0.719
for a single centroid on identical P2P features, which suggests roughly 15
points of headroom that a single row cannot capture. An attempt to recover it
with k-means prototypes degraded accuracy, but that attempt used farthest-point
seeding, which selects outliers, so it is void rather than negative.

Note also that the shift branch already spends `2S+1 = 5` rows per unit on
pre-shifted copies for +0.4 pp. The same row budget spent on data-driven modes
has never been tested with a correct clusterer.

**What to measure.** Frontier position, not argmin accuracy. That is the lesson
of Phase B.

**Kill criterion.** If P rows per unit does not move the recall/traffic frontier
by more than the row cost, close the multi-row question permanently.

### 4. Salvage `com_posneg` on its own merits

The shape branch produced no selected configuration, but `com_posneg` is a
registered arm that is cheaper than the current descriptor is expensive: 66
bits per row, +10.5 pp HJ assignment, 2.9x tighter unit clouds, survives a
causal filter, and robust to ±2-sample jitter where the full arm was not.

It should be re-registered on the untouched confirmation sets A, B, and C with
the causal and jitter conditions included from the start rather than as a later
boundary check. It will not fix the frontier, and the protocol must say so.

The `LAT` and `WIDTH` columns should be dropped unless a specific
alignment-invariant reformulation is proposed: they cost 29 bits, add at most
0.4 pp over `com_posneg`, and caused the Phase A2 jitter failure.

## What not to repeat

These are closed. Reopening one requires a new mechanism, not a new parameter.

- Another per-unit scoring variation on a fixed single-region policy: masks,
  weights, bit widths, shift radii, and estimator smoothing have all been tested
  and none moved both families.
- Raw unnormalized dot product for template identity; amplitude dominates shape.
- A fixed globally-applied P2P averaging window; rejected on independent
  confirmation.
- Morphology-12 or a dyadic prefix sketch as a Level-2 replacement.
- Reporting an argmin improvement as progress on the recall/traffic gate. This
  branch demonstrated the two are close to independent.

## Sequencing

Direction 2 is cheap and unblocks the interpretation of everything else, so it
should be written first even though it produces no data. Direction 1 is the
strongest experimental bet. Direction 3 is worth one clean attempt. Direction 4
is cleanup that makes an existing gain durable.

None of these is a hardware claim. All of them inherit the standing boundary
that logical row, bit, and access counts are not PPA, and that the current
paper-facing isolation still uses GT events, GT rows, and full-recording
zero-phase filtering outside the Phase A2 causal arm.

## Sources

- [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) — current disposition of every branch
- [`shape_orthogonal_descriptor_results.md`](shape_orthogonal_descriptor_results.md) — the diagnostic this roadmap is built on
- [`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md) — the recall/traffic frontier definition
- [`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md) — deterministic top-M retrieval evidence
- [`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md) — the prior multi-row negative
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — evidence levels and system model
