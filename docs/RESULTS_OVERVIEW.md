# Results Overview

Last reviewed: **2026-08-11**.

This page is the decision-oriented map of Spatial's results. It distinguishes
local pilot gates from the system gate and points to the detailed source
documents. It is not a replacement for a result document or provenance
snapshot.

## Executive verdict

Spatial has credible component-level evidence and several useful negative
results, but **no current configuration is an end-to-end, DAC-ready STAR-Mem
solution**.

The recent relative-offset and shifted-similarity work improves the semantic
quality of the P2P representation and establishes low-bit L1/cosine options.
Its improvement is modest, its current patch is 1D, and it has not resolved
the system-wide recall/traffic/drift/causality/hardware gaps.

The subsequent robust-P2P study rejects a globally fixed 5-point averaged
estimator on independent HJ confirmation. Eight-bit integer L1 reproduces
float for that frozen estimator, but the estimator failure stops its online
data phase and makes the bit result conditional.

## System gate

| Requirement | Target | Current status | Main evidence |
|---|---:|---|---|
| True-unit candidate recall | at least 0.99 on HJ and MEArec | **Open jointly with traffic** | `adaptive_5bit_range_search.md` and later Level-1 branches |
| Waveform-template reads | at least 4x reduction from full scan | **Open jointly with recall** | `adaptive_5bit_range_search.md`, `indexed_confusion_candidate_search.md` |
| Compact temporal accuracy | within 1 percentage point of the matched teacher | **Conditional only** | `compact_waveform_refinement.md`, `causal_temporal_sketch_results.md` |
| Physical implementation advantage | beat indexed SRAM/digital baseline after peripherals | **Open** | logical mapping only in `hardware_mapping_spec.md` |
| Realistic dense-2D support | new pipeline validated on suitable dense 2D data | **Open** | legacy 1024-channel result is not the new pipeline |
| Online operation | causal signal path and no GT-dependent deployed state | **Open / blocked as a current claim** | current paper-facing isolation uses GT events/rows and zero-phase filtering |

Passing an experiment-specific gate below does not change this system verdict
unless all coupled requirements are evaluated together.

## Latest relative-offset and shifted-similarity branch

### P0: fixed relative-offset footprint

The P0 pilot replaces distance-ranked KNN slots with a fixed 1D primary-axis
patch of seven relative offsets.

| Family | KNN argmin | Relative argmin | KNN true-unit median L1 | Relative true-unit median L1 | Local gate |
|---|---:|---:|---:|---:|---|
| HJ, 12 recordings | 0.664 | **0.717** | 0.434 | **0.422** | **Pass** |
| MEArec SNR5, 4 files | 0.668 | **0.736** | **0.436** | 0.440 | Mixed: argmin passes, distance has a small regression |

The HJ hypothesis passes. MEArec also improves nearest-centroid assignment,
but it is still being projected to one primary axis; this is not evidence for
a final 2D patch-grid implementation.

Source: [`relative_offset_footprint_results.md`](relative_offset_footprint_results.md).

### P1: shift-min and multi-copy rows

| Family | Best observation | Cost / limitation | Local gate |
|---|---|---|---|
| HJ | L1 argmin 0.717 at S=0 to 0.721 at S=2; true-unit distance drops slightly | S=2 stores 5 rows/unit | **Pass** |
| MEArec SNR5 | S=0 remains best at 0.736 | S=1/S=2 do not improve argmin | Quantization/accounting pass; no shift gain |

Four-bit shifted L1 stays within 0.4 percentage points of float at the selected
family operating points. This establishes a viable low-bit primitive, but the
HJ shift gain is too small to call drift solved and storage scales by `2S+1`.

Source: [`shift_cim_results.md`](shift_cim_results.md).

### Shifted L1, raw dot product, and cosine

| Family | Best L1 | Best cosine | Raw dot product | Interpretation |
|---|---:|---:|---:|---|
| HJ | 0.721, S=2 | **0.733, S=2** | about 0.142 | Cosine is 1.1 points above matched 5-bit L1 at S=2 |
| MEArec SNR5 | **0.736, S=0** | about 0.735-0.736, S=0/1 | about 0.098 | Cosine and L1 are effectively tied |

Four-, five-, and six-bit cosine track float closely. Unnormalized maximum dot
product collapses because footprint energy dominates shape. The meaningful
MAC-CiM story is normalized correlation, not raw convolution.

Source: [`shift_conv_bit_sweep_results.md`](shift_conv_bit_sweep_results.md).

### Robust P2P averaging, 4–8 bits, and online updates

The four-recording development screen selected centered five-point averaging
and 8-bit L1. On 12 independent recordings, centered k=5 changed all-HJ L1 by
-0.21 percentage points and MEArec by +0.43 points, but held-out HJ drift fell
by 1.05 points and one drift scene fell by 6.65 points. The estimator and
overall gates therefore fail.

Eight-bit matching remains within 0.10 points of float on both confirmation
families and passes the positive-margin/per-recording guards. That only
establishes conditional quantization equivalence. The bit-exact
error-feedback update primitive is unit tested, but no self-predicted data run
is claimed after the estimator stop rule fired.

Source: [`robust_p2p_template_results.md`](robust_p2p_template_results.md).

### Shape-orthogonal descriptor, Phase A

The amplitude footprint saturates because four independent knobs (bit width,
L1 versus cosine, shift radius, temporal averaging) each move assignment by
less than 1.5 points. Keeping both operands of `max - min` as separate columns,
plus cross-channel trough latency and home-channel spike width, breaks that
ceiling on the development set.

| Arm | HJ | MEArec | bits/unit row |
|---|---:|---:|---:|
| `d9` current descriptor | 0.814 | 0.777 | 45 |
| `d9_rand` falsifier | 0.805 | 0.776 | 74 |
| `com_posneg` | 0.919 | 0.858 | 66 |
| `com_posneg_lat_width` selected | **0.920** | **0.859** | 95 |

All five pre-registered conditions pass, the random-column control correctly
fails to gain, and an independent gate recomputation that shares no code with
the runner agrees. Splitting the P2P operands does nearly all of the work;
latency and width add only 0.1 pp on top of `com_posneg`.

**Phase A2 fails, so nothing is selected.** The registered arm survives a
forward-only causal filter (retaining 103% of its HJ gain and 66% on MEArec),
which was the predicted kill condition, but it loses 1.40 pp under
deterministic ±2-sample alignment jitter on MEArec against a 1.0 pp limit. The
per-block breakdown localizes the damage to the `LAT`/`WIDTH` columns, whose
incremental contribution over `com_posneg` is sign-unstable everywhere and
reaches −1.03 pp at that operating point. `com_posneg` alone loses only
0.43 pp, but promoting it after seeing this data is post-hoc reselection and is
not done.

**Phase B is negative and bounds the claim further.** Inside the same per-unit L1
radius candidate policy, the new descriptor does not move the recall/traffic
frontier: holding recall at or above 0.99, the selected arm reaches only 1.30x
HJ and 1.11x MEArec template-read reduction, against a 4x requirement, and
`d9` behaves the same. At HJ p95 the selected arm has higher recall than `d9`
but worse traffic, so it moves along the existing frontier rather than shifting
it. Improving nearest-template discrimination does not by itself make a unit's
acceptance region more selective.

The Level-1 recall/traffic bottleneck therefore remains the system-blocking
issue, and this branch must not be reported as progress against it. No
configuration from this branch is promoted to the hardware mainline.

Two findings survive as design evidence. Keeping both operands of `max - min`
recovers about 10 pp of HJ assignment accuracy that the current descriptor
discards for free, and it survives a causal filter. And assignment accuracy and
candidate-set selectivity are close to independent here, so future Level-1 work
should be judged on the recall/traffic frontier directly rather than on argmin.

Source: [`shape_orthogonal_descriptor_results.md`](shape_orthogonal_descriptor_results.md).

### What this branch supports

- Fixed relative P2P columns are better behaved than KNN-ranked slots on the
  current HJ and MEArec SNR5 pilot.
- Low-bit L1 and normalized correlation are both viable component-level
  similarity relations.
- A small `S=1` shifted bank can be studied as home-channel-jitter insurance.

### What this branch does not support

- shift-min has solved neural drift;
- a 2D relative patch has been implemented and validated;
- P0/P1 meets 0.99 candidate recall and 4x waveform traffic;
- multi-copy CiM has a physical area or energy advantage;
- the experiment is causal, detected-event, or label-free online sorting.

## Reference and two-stage baselines

| Study | Key result | Evidence boundary |
|---|---|---|
| Legacy 1024-channel reference | Akhoundi spatial SOM 92.1% vs 81.2% CC baseline | Reproduction on the generated 2D grid, not the new memory hierarchy |
| HJ scene 11 `soft_loc` | 0.916 oracle and 0.833 SOM | Spatial-only, known event rows |
| HJ scene 11 two-stage waveform | 0.968 oracle | GT unit centroids; information ceiling |
| Cross-scene unsupervised two-stage | roughly 0.81 HJ and 0.79 MEArec | Discovered rows; lower than oracle ceiling |
| SOM-discovered per-cluster attention | 0.913 SOM on HJ scene 11 | Useful exploratory result; not the active low-bit mainline |

Sources: [`spatial_upper_bound.md`](spatial_upper_bound.md),
[`two_stage_upper_bound.md`](two_stage_upper_bound.md),
[`two_stage_unsupervised.md`](two_stage_unsupervised.md), and
[`combined_mscale_se.md`](combined_mscale_se.md).

## Level-1 memory-search disposition

The central trade-off is stable: high-recall rows are broad, while tight
candidate lists lose the true unit.

| Branch | Strongest positive observation | Why it is not selected |
|---|---|---|
| Per-unit L1/range | At p99.9, candidate recall is about 0.993 HJ / 0.995 MEArec | Only about 1.25x waveform-read reduction |
| Inner-box / outer-L1 | HJ has a 4x-class operating point | Recall is 0.956 there; MEArec best tested point is only 1.33x |
| Wilson-certified singleton exit | Clear safe-exit semantics | Strict target certifies no rows; relaxed exits are rare and yield about 1.36-1.38x |
| Static indexed adjacency | About 5.31x HJ / 4.08x MEArec at one operating class | Recall only 0.932 / 0.938 |
| Block-causal graph update | Explicit online state/accounting protocol | Even delayed-GT upper bound is 0.947 / 0.934 at 4x-class traffic |
| Up to four prototypes/unit | Represents multiple local modes | HJ best 4x-class recall 0.898; MEArec has no 4x point |
| Masked L1 | Strong HJ development improvement and fewer active lanes | Frozen configuration does not generalize to MEArec; formal gate is null |
| Power-of-two weighted L1 | Large HJ traffic improvement | MEArec movement is insufficient/slightly negative; formal gate is null |
| Spatial WTA classifier | Selected 5-bit stability-weighted L1 passes attribution and bit-conformance gates | Accuracy, online drift, and unsupervised gates do not all pass |
| Tile activation / cosine | Tile routing clears high-recall local gates; cosine stays near L1 | First-detect routing fails and this does not by itself solve candidate traffic |

The shape-descriptor branch sharpened this into a specific diagnosis. Its
`com_posneg` arm made unit clouds about 2.9x tighter in normalized intra-unit
distance and improved the positive margin by 8 to 12 points, yet the candidate
count at 0.99 recall did not shrink. The radius is therefore set by the tail of
the intra-unit distribution, not its bulk, so a better representation cannot fix
this. See [`LEVEL1_ROADMAP.md`](LEVEL1_ROADMAP.md) for the resulting plan.

Detailed sources:

- [`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md)
- [`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md)
- [`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md)
- [`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md)
- [`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md)
- [`spatial_cim_sorting_results.md`](spatial_cim_sorting_results.md)
- [`geometric_prefilter_results.md`](geometric_prefilter_results.md)

## Level-2 temporal disposition

| Representation | Accuracy behavior | Traffic / storage behavior | Disposition |
|---|---|---|---|
| Float full64 teacher | Reference | 2048 bits/unit in the reported float accounting | Teacher only |
| Signed full64x5 | Contract-dependent; can lose more than one point | 320 bits/unit | Mandatory low-bit baseline, not a guaranteed solution |
| Shared Fisher taps | Within one point behind the tighter p95 spatial source | The p95 source itself misses the 0.99 recall requirement | Conditional |
| Pair-conditioned 16 taps | +0.04 / -0.97 points vs teacher on HJ/MEArec at p99.9 source | Only 1.35x / 1.36x traffic and O(U^2) tap indices | Accuracy evidence, not memory solution |
| Dyadic 5-bit prefix sketch | All tested points fail temporal accuracy; least-bad common point loses 6.10 / 12.08 points | Every point exceeds 4x template-bit reduction | Negative result; formal selection is null |

The raw-prefix float ceiling is within one point by delay 16 samples in the
reported pilot, which localizes the dyadic failure to representation/scale/
metric rather than insufficient future waveform horizon.

Sources: [`compact_waveform_refinement.md`](compact_waveform_refinement.md) and
[`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md).

## Hardware-evidence boundary

The repository has bit-exact descriptor and row definitions, candidate/FIFO
semantics, pointer and template accounting, and symbolic operation/access
counts. That supports architectural interface and logical-cost claims.

The repository does **not** currently have:

- an RTL implementation of the full path;
- a characterized RRAM/CAM/SRAM macro under equivalent behavior;
- ADC, sense-amplifier, WTA, decoder, control, or update-write PPA;
- P&R/PVT or silicon measurements.

Therefore, statements such as “5x fewer logical rows” or “4x fewer template
bits read” must not be rewritten as “5x less area” or “4x lower energy.” See
[`hardware_mapping_spec.md`](hardware_mapping_spec.md).

## How to use this overview

For a paper or report:

1. start from the system verdict here;
2. open the linked result document and confirm its dataset, split, aggregation,
   GT/discovered boundary, and command;
3. verify any associated SHA-256 manifest for a frozen claim;
4. report local gate status and system implication separately;
5. update the DAC2027 evidence ledger before promoting a new paper claim.

The parent `DAC2027/notes/EVIDENCE_LEDGER.md` is the paper claim ledger, but its
dated snapshot may lag the newest Spatial pilots. Spatial result/provenance
documents remain the primary source for newly completed experiments until the
paper ledger is explicitly updated.
