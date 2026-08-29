# Spatial Architecture Guide

This guide explains what the Spatial repository implements, how the major
algorithm families relate, and where the current experiments stop short of a
deployable STAR-Mem sorter. It is a code-and-system map; individual result
claims remain owned by their protocol, result, and provenance documents.

## Scope

Spatial serves three related purposes:

1. reproduce the Akhoundi et al. 1024-channel spatial-clustering pipeline;
2. study spatial features and two-stage spike assignment across linear and 2D
   probes;
3. explore a hardware-oriented memory hierarchy in which a cheap spatial
   search limits expensive waveform-template accesses.

The parent `SNN_SpikeSorting` workspace owns shared datasets and the DAC2027
paper project. Spatial owns the Python algorithms, experiment runners, tests,
and primary result records.

## Target system

```text
                         fit / update state
                 +------------------------------+
                 | unit rows, radii, templates, |
                 | directories, online metadata |
                 +---------------+--------------+
                                 |
raw signal                       v
  -> detection -> alignment -> spatial descriptor
  -> locality/directory activation
  -> Level 1 low-bit candidate retrieval
  -> candidate stream {unit_id, score, last}
  -> ambiguity controller
       C = 0: spatial fallback
       C = 1: direct unit assignment
       C > 1: read only candidate waveform templates
  -> Level 2 digital temporal distance / tournament
  -> output unit ID and, eventually, confidence-gated online updates
```

The intended hardware comparison is broader than “RRAM is better.” The same
candidate behavior and access trace should be compared across RRAM-CiM/CAM,
indexed SRAM plus digital distance, and hybrid near-memory designs. Logical
bits, row comparisons, and operation counts are algorithmic accounting; they
are not synthesized macro or silicon PPA.

## Current experimental path

The most recent relative-offset/shift branch follows this component-isolation
path:

```text
Dataset loader
  -> full-recording bandpass filtering
  -> GT event times and local waveform windows
  -> per-event home/peak channel
  -> relative primary-axis P2P patch, K=7
  -> COM and normalized/quantized P2P
  -> chronological train/test split
  -> train-only unit centroids and pooled thresholds
  -> COM gate AND shifted P2P L1/L2, or shifted dot/cosine score
  -> candidate recall, candidate load, and nearest-centroid diagnostics
```

Important differences from the target system:

- event times and unit rows are often ground-truth supplied;
- most DAC isolation runners use full-recording zero-phase filtering;
- the current relative patch is a 1D primary-axis approximation, including on
  the 32-channel MEArec geometry;
- confidence-gated, label-free online row/template updates are not integrated;
- no current result is an RTL, memory-macro, P&R, PVT, or silicon result.

## Algorithm generations

### 1. Akhoundi reference reproduction

```text
raw -> bandpass -> NEO/VC-SPD detection -> Eq. 3-5 spatial features
    -> modified SOM -> clustering evaluation
```

The faithful Python path is concentrated in `algorithms/reference_port.py`.
The decomposed implementation uses `algorithms/detection.py`,
`algorithms/spatial_features.py`, and `algorithms/som_clustering.py`. The
original MATLAB code remains in `reference_code/`.

This path works best on the high-density 2D geometry for which the spatial
angles and neighborhood relations were designed. On linear probes, the angle
features degenerate.

### 2. Geometry-aware spatial footprints

`algorithms/spatial_footprint.py` provides the local feature family used by
most later experiments:

- KNN tables with the home channel in slot 0;
- local peak-to-peak amplitude extraction;
- opt-in raw, centered 3/5-point, moving-average 3/5-point, and diagnostic
  extreme-mean P2P estimators;
- `soft_loc`: geometry-weighted COM, spread, and sharpness;
- normalized P2P footprints;
- fixed relative-offset patches along the probe's primary axis;
- 4–8-bit integer P2P template evaluation in the robust-estimator runner.

The relative-offset representation changes slot semantics. Instead of “the
j-th nearest channel,” each slot represents a stable signed offset from the
event home channel. Out-of-bounds slots are encoded as zero. This removes KNN
slot scrambling on linear probes, but a canonical 2D patch-grid mapping is
still open.

### 3. Two-stage assignment

`algorithms/two_stage_assign.py` implements the principal oracle helper:

```text
COM distance <= tau_com AND P2P distance <= tau_p2p
  -> candidate units
  -> nearest waveform centroid among candidates
  -> normalized spatial fallback when no row passes
```

`algorithms/unsupervised_two_stage.py` replaces GT unit rows with hierarchical
spatial and waveform clusters. This is closer to a deployable labeling path,
although its evaluation still uses GT labels after prediction.

The two paths answer different questions. Oracle assignment measures whether
the chosen features contain discriminative information when unit identities
are known. The unsupervised path measures whether the repository can discover
and use those rows without GT labels.

### 4. Low-bit Level-1 memory search

The D9/B5 descriptor used in the main hardware studies is:

```text
[COM_x, COM_y, P2P_0, ..., P2P_6]
  9 coordinates x 5 bits = 45 input bits/event
```

The search modules deliberately separate the feature code from candidate
policy:

| Module | Search relation or state |
|---|---|
| `adaptive_range_search.py` | Per-unit integer centroid with L1 radius or axis-aligned range |
| `adaptive_prototype_search.py` | Up to four rows per unit |
| `adaptive_masked_range_search.py` | Unit-specific enabled coordinates |
| `adaptive_weighted_range_search.py` | Unit-specific power-of-two coordinate weights |
| `spatial_wta.py` | Biased integer winner-take-all classifier |
| `geometric_prefilters.py` | Channel/tile activation, Jaccard, and integer cosine |
| `online_spatial_adaptation.py` | Shift-EMA centroid and threshold updates |
| `online_p2p_templates.py` | Confidence-gated error-feedback P2P templates, rollback anchors, and batched nonvolatile commits |

Experiment-specific directory, graph, fallback, and traffic logic is kept in
`experiments/`. The search relation and the memory organization are not the
same thing: an L1 row can be scanned globally, activated by a tile directory,
or addressed through an SRAM pointer list.

The P2P update bank fixes a numerical limitation but does not establish an
online result. Small residuals are retained in signed accumulators rather than
discarded by truncation, integer ties cannot update, and distance/margin/control
gates are mandatory. The current averaged-P2P estimator failed independent
confirmation, so no self-predicted data run is authorized for that selection.

### 5. Relative-offset and shifted matching

`algorithms/shift_match.py` implements:

- zero-filled integer shifts of a K-wide template;
- minimum shifted L1/L2 distance;
- pre-shifted `(2S+1)` row banks for a multi-copy CiM mapping;
- maximum shifted raw dot product or normalized correlation;
- row, bit, and operation accounting.

The current shift operation moves **template content within the relative
patch**. For the multi-copy hardware model, a unit stores one pre-shifted row
per shift, so storage scales by `2S+1`. At `K=7` and 4 bits, each copy is 28
logical bits before row metadata.

Raw maximum dot product is amplitude-biased and fails the current pilot.
L2-normalized dot product (cosine/normalized correlation) preserves the
shape-based comparison and is the meaningful MAC-CiM alternative to L1.

### 6. Level-2 temporal refinement

Level 2 is intended to preserve fine waveform ordering only within the Level-1
candidate set. The repository includes:

- full 4/5-bit waveform SAD;
- shared Fisher-selected taps;
- fixed morphology samples;
- pair-conditioned tap tournaments;
- causal-prefix dyadic mean/Haar sketches.

These implementations live primarily in the experiment runners because their
candidate-list freezing, traffic estimators, and selection rules are part of
the protocol. See `compact_waveform_refinement.md` and
`causal_temporal_sketch_results.md` for the current result boundaries.

### 7. Exploratory feature branches

`multiscale_footprint.py` and `se_attention.py` explore FPN-like multi-K
features and channel attention. They are retained as useful ablations and
oracle/discovered-row studies, not as the current hardware mainline.

## Data and feature contracts

### Dataset object

`data/loader.py` normalizes the main input families into:

| Field | Shape / meaning |
|---|---|
| `raw_data` | `(n_channels, n_samples)` float array |
| `geom` | `(n_channels, 2)` electrode coordinates in micrometers |
| `fs` | sampling rate in Hz |
| `spike_times` | zero-based GT sample indices |
| `spike_units` | one GT unit ID per event row |
| `name` | stable recording label used in outputs |

Some prepared Hybrid Janelia NPZ files expose equivalent fields under older
names. Always use the loader or the experiment's shared preparation helper
rather than assuming an NPZ schema from one result document.

### Event rows and simultaneous spikes

Every event is a row. Simultaneous spikes must remain distinct rows; a single
`timestamp -> label` dictionary loses collisions. Chronological split helpers
must keep equal timestamps on the same side of a split.

### Quantization

Quantizers, integer centroids, thresholds, ranges, templates, feature scales,
and selected taps must be fitted on the training partition only. A 5-bit code
does not by itself define a hardware contract: the normalization mode, scale,
signedness, saturation, integer tie rule, radius width, and fallback behavior
must also be stated.

## Evaluation and control semantics

### Clustering metrics

`evaluation/metrics.py` first matches predicted event times to GT times within
a tolerance, then uses Hungarian label mapping for the matched events.
Clustering accuracy is the number of correctly clustered events divided by the
number of GT events, so detection misses remain in the denominator.

Many component experiments bypass detection and classify known event rows.
Their assignment accuracy must not be compared directly with detected-event
clustering accuracy without stating that boundary.

### Candidate metrics

Level-1 reports should include:

- true-unit candidate recall;
- zero-, one-, and multi-candidate event fractions;
- candidate count and its aggregation mode;
- row comparisons including fallback;
- distinct waveform-template reads;
- final assignment accuracy only as a separate downstream metric.

`C=0` means zero temporal-template reads only when the controller reuses a
Level-1 score/argmin. It does not mean zero total work. `C=1` can assign
without a temporal read; `C>1` reads distinct candidate templates.

## Evidence levels

Use these labels when reading or writing Spatial documentation:

| Evidence level | What is known | What is not implied |
|---|---|---|
| Oracle / GT-row | Feature or policy ceiling with true unit rows | Deployable cluster discovery |
| GT-event component isolation | Candidate/assignment behavior at known events | Detection accuracy or causal signal path |
| Discovered-row | Unit state learned without GT labels | Causal filtering or stable online adaptation |
| Causal component | One primitive respects forward-only timing | End-to-end causal sorter |
| Detected-event end-to-end | Detection and assignment evaluated together | Hardware PPA |
| Logical hardware model | Bit-exact rows, accesses, operations, interfaces | Macro/peripheral/silicon energy or area |

A local experiment gate applies only to its registered hypothesis. For
example, P0 can pass “relative footprint improves argmin” while the whole
pipeline still fails the 0.99-recall/4x-traffic system gate.

## Code ownership map

| Concern | Primary files |
|---|---|
| Loading and dataset identity | `data/loader.py`, `data/prepare_hybrid_janelia.py` |
| Detection and windows | `algorithms/detection.py`, `algorithms/reference_port.py` |
| Spatial features | `algorithms/spatial_features.py`, `algorithms/spatial_footprint.py` |
| Shifted distance/correlation | `algorithms/shift_match.py` |
| Two-stage assignment | `algorithms/two_stage_assign.py`, `algorithms/unsupervised_two_stage.py` |
| SOM clustering | `algorithms/som_clustering.py` |
| Low-bit search | `algorithms/adaptive_*_search.py`, `algorithms/spatial_wta.py` |
| Locality and tile routing | `algorithms/geometric_prefilters.py` |
| Online state | `algorithms/online_spatial_adaptation.py`, `algorithms/online_p2p_templates.py` |
| Shared paper-facing feature preparation | `experiments/spatial_cim_common.py` |
| Evaluation | `evaluation/metrics.py` and experiment-specific candidate accounting |
| Protocol and claim records | `docs/` |
| Generated evidence artifacts | `output/` |

## Output and provenance boundary

Generated CSV, JSON, and log files normally live under `output/` and are
gitignored. A paper-facing claim must therefore have a tracked document that
records the exact command, input set, split, estimator, output paths, and
decision. The most mature branches also include source/input/output SHA-256
manifests and independent gate recomputation.

Do not overwrite a frozen result artifact and continue citing its old digest.
Create a new result snapshot or explicitly update the provenance chain.

## Open architecture gaps

The shortest list of system-level gaps is:

1. a fixed semantic **2D** relative patch for staggered/hexagonal geometries;
2. a Level-1 policy that jointly reaches at least 0.99 candidate recall and 4x
   template-read reduction across both main dataset families;
3. a compact Level-2 representation that stays within one percentage point of
   its teacher behind that high-recall source;
4. an integrated causal detector/alignment path;
5. discovered-row, confidence-gated online updates without GT labels;
6. equivalent-trace RRAM/CAM versus SRAM/digital physical evaluation;
7. realistic dense-2D validation of the new pipeline rather than only the
   legacy Akhoundi reproduction.

Use [Results overview](RESULTS_OVERVIEW.md) for the current disposition of each
branch and [Experiment index](../experiments/README.md) to locate the runner.
