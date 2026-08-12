# Spatial Experiment Index

This directory contains runnable studies and verification entry points. The
runner produces generated artifacts; the corresponding file under `../docs/`
owns the protocol, checked result, decision, and claim boundary.

Run modules from the parent workspace root:

```bash
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.<module> --help
```

Use [Reproducing Spatial Experiments](../docs/REPRODUCING.md) for environment,
dataset, split, filtering, aggregation, and provenance rules.

## Choose a runner

### Current relative-offset / shifted-similarity branch

| Question | Runner | Result record |
|---|---|---|
| Do fixed physical-offset P2P slots beat KNN-ranked slots? | `run_relative_footprint_pilot.py` | [`relative_offset_footprint_results.md`](../docs/relative_offset_footprint_results.md) |
| Does shift-min reduce same-unit distance, and what does row replication cost? | `run_shift_cim_pilot.py` | [`shift_cim_results.md`](../docs/shift_cim_results.md) |
| Should a shifted MAC path use raw dot product or normalized correlation, and how many bits? | `run_shift_conv_bit_sweep.py` | [`shift_conv_bit_sweep_results.md`](../docs/shift_conv_bit_sweep_results.md) |
| Do 3/5-point P2P estimators generalize, which of 4–8 bits preserves integer-L1 confidence, and may online adaptation proceed? | `run_robust_p2p_template_pilot.py` | [`robust_p2p_template_results.md`](../docs/robust_p2p_template_results.md) |
| Has the amplitude footprint hit an information ceiling, and do split pos/neg peaks, trough latency, and spike width raise it? | `run_shape_descriptor_pilot.py` | [`shape_orthogonal_descriptor_results.md`](../docs/shape_orthogonal_descriptor_results.md) |
| Does an independent recomputation agree that the Phase A gate passed? | `audit_shape_descriptor_gates.py` | same result record, "Independent verification" |
| Does the descriptor survive a causal filter and ±2-sample alignment jitter? | `run_shape_causal_boundary.py` | **Causal passes, jitter fails**; same record, "Phase A2" |
| Does the shape descriptor move the 0.99-recall/4x-traffic frontier? | `run_shape_candidate_traffic.py` | **Negative**; same record, "Phase B" |
| Where is the POSNEG code-width optimum, and does more precision help? | `run_posneg_bitwidth_sweep.py` | [`LEVEL1_ROADMAP.md`](../docs/LEVEL1_ROADMAP.md), "noise-limited, not capacity-limited" |

The shape-descriptor branch is closed with no selected configuration. Its
runners are retained because the negative results are design evidence.

Typical commands:

```bash
python -m Spatial.experiments.run_relative_footprint_pilot \
  --duration 60 --quiet
python -m Spatial.experiments.run_shift_cim_pilot \
  --duration 60 --quiet
python -m Spatial.experiments.run_shift_conv_bit_sweep \
  --duration 60 --quiet
python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --pilot --quiet
python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --confirmation --quiet

# Phase A, then an independent recomputation of the same gate.
python -m Spatial.experiments.run_shape_descriptor_pilot --quiet
python -m Spatial.experiments.audit_shape_descriptor_gates
```

`audit_shape_descriptor_gates.py` deliberately shares no code with the runner
that produced the CSV. It exits nonzero when its own recomputation disagrees
with the runner's claimed gate outcome.

These are component pilots. Their local P0/P1 passes do not imply that the
full 0.99-recall/4x-traffic system gate passes.

The robust-P2P confirmation rejects the fixed averaged estimator. Its 8-bit
float-equivalence result is conditional, and its failed gate prohibits the
self-predicted online data run under that selection.

### Reference, feature, and two-stage baselines

| Question | Runner | Documentation |
|---|---|---|
| Can the detection-to-SOM reference pipeline be reproduced? | `run_spatial_sorting.py` | root README and reference code notes |
| Which spatial feature has the best oracle/SOM ceiling? | `run_spatial_upper_bound.py` | [`spatial_upper_bound.md`](../docs/spatial_upper_bound.md) |
| What is the GT-row ceiling of COM/P2P candidates plus waveform refinement? | `run_two_stage_upper_bound.py` | [`two_stage_upper_bound.md`](../docs/two_stage_upper_bound.md) |
| What survives when unit rows are discovered rather than GT supplied? | `run_two_stage_unsupervised.py` | [`two_stage_unsupervised.md`](../docs/two_stage_unsupervised.md) |
| How stable is the unsupervised pipeline across a corpus/grid? | `run_unsupervised_two_stage_sweep.py` | [`two_stage_unsupervised.md`](../docs/two_stage_unsupervised.md) |
| How do methods compare across all HJ scenes? | `run_hj16_method_sweep.py` | [`hj16_method_sweep.md`](../docs/hj16_method_sweep.md) |
| How do methods compare across MEArec recordings? | `run_mearec_method_sweep.py` | [`mearec_method_sweep.md`](../docs/mearec_method_sweep.md) |
| How sensitive is P2P assignment to bit depth? | `run_lowbit_p2p.py` | [`central_ratio_vs_maxnorm_benchmark.md`](../docs/central_ratio_vs_maxnorm_benchmark.md) and encoding records |
| Which SOM hyperparameters matter? | `sweep_som_params.py` | baseline result documents |

### Low-bit spatial WTA classifier

This group treats spatial WTA as a classifier rather than only a prefilter.

| Stage | Runner | Role |
|---|---|---|
| G1-G3 development grid | `run_spatial_wta_pilot.py` | Bit depth, feature set, adaptive score, and baselines |
| Selected corpus run | `run_spatial_wta_corpus.py` | 12 HJ plus 20 MEArec selected configuration |
| G4/G6 controls | `run_spatial_attribution.py` | Feature shuffles and hardware-conformance checks |
| G5 drift | `run_spatial_online_drift.py` | Frozen versus integer online adaptation |
| Stage C | `run_spatial_unsupervised.py` | Quantized SOM and Hungarian evaluation |

Protocol and result chain:

- [`spatial_cim_sorting_plan.md`](../docs/spatial_cim_sorting_plan.md)
- [`spatial_cim_sorting_results.md`](../docs/spatial_cim_sorting_results.md)
- [`spatial_cim_attribution.md`](../docs/spatial_cim_attribution.md)
- [`spatial_cim_sorting_provenance.md`](../docs/spatial_cim_sorting_provenance.md)

The selected 5-bit classifier passes some component gates, but the complete
accuracy/online/unsupervised gate set does not pass.

### Level-1 candidate search and routing

| Runner | Search question | Disposition / document |
|---|---|---|
| `run_adaptive_5bit_range_search.py` | Pooled vs per-unit L1/range rows; all-row and channel-support directories | High-recall baseline is too broad; [`adaptive_5bit_range_search.md`](../docs/adaptive_5bit_range_search.md) |
| `run_dual_range_pipeline.py` | Tight inner box plus ambiguity-triggered outer L1 | Negative joint gate; summarized in the adaptive range record |
| `run_certified_dual_range_pipeline.py` | Can Wilson-certified singleton rows exit safely? | Strict exits vanish; relaxed exits save too little |
| `run_indexed_confusion_candidate_search.py` | Deterministic top-M plus calibration anchor adjacency | Tight lists lose recall; [`indexed_confusion_candidate_search.md`](../docs/indexed_confusion_candidate_search.md) |
| `run_online_confusion_graph.py` | Can block-causal graph state repair indexed retrieval? | Even delayed-GT upper bound fails the joint gate |
| `run_adaptive_multiprototype_5bit.py` | Do P=1..4 rows capture local modes/drift? | Negative; [`adaptive_multiprototype_5bit_results.md`](../docs/adaptive_multiprototype_5bit_results.md) |
| `run_adaptive_masked_5bit.py` | Do unit-specific lane masks improve the frontier? | HJ-positive, cross-family-negative; [`adaptive_masked_5bit_results.md`](../docs/adaptive_masked_5bit_results.md) |
| `run_adaptive_weighted_5bit.py` | Do `{1,2,4}` coordinate weights improve the frontier? | HJ-positive, cross-family-negative; [`adaptive_weighted_5bit_results.md`](../docs/adaptive_weighted_5bit_results.md) |
| `run_geometric_prefilter_sweep.py` | First-trigger, tile, Jaccard, cosine, and tile power-gating | Tile/cosine pass local gates; first-trigger/drift fail; [`geometric_prefilter_results.md`](../docs/geometric_prefilter_results.md) |

Representative primary search command:

```bash
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 95 --coverage 0.95 \
  --directory-neighbors 0 7 \
  --directory-coverages 0.95 0.99 1.0
```

Before running the masked, weighted, multi-prototype, or temporal pilots, read
their plan documents. Those runners enforce registered development grids and
fail-closed confirmation boundaries.

### Level-2 waveform refinement

| Runner | Temporal question | Disposition / document |
|---|---|---|
| `run_compact_waveform_refinement.py` | Full 4/5-bit SAD, shared Fisher taps, and fixed morphology behind frozen candidates | Shared taps are conditional; [`compact_waveform_refinement.md`](../docs/compact_waveform_refinement.md) |
| `run_pair_conditioned_waveform.py` | Can pair-specific Fisher taps preserve candidate-internal ordering? | Accuracy recovers, traffic/storage do not |
| `run_causal_temporal_sketch.py` | Can a signed-5-bit dyadic prefix sketch meet accuracy and bit-traffic gates? | Negative; formal selection null; [`causal_temporal_sketch_results.md`](../docs/causal_temporal_sketch_results.md) |

The temporal sketch runner's recorded pilot does not authorize confirmation.
A best observed failed point must not be called selected.

### Exploratory feature branches and diagnostics

| Runner | Purpose | Documentation |
|---|---|---|
| `run_multiscale_fpn.py` | Multi-K footprint extraction and FPN-like fusion | [`multiscale_fpn.md`](../docs/multiscale_fpn.md) |
| `run_se_attention.py` | Fisher, variance, and per-cluster channel attention | [`se_attention.md`](../docs/se_attention.md) |
| `run_combined_mscale_se.py` | Combined multi-scale and iterative discovered attention | [`combined_mscale_se.md`](../docs/combined_mscale_se.md) |
| `analyze_soft_loc_errors.py` | Error-pair taxonomy and FSDE rescue analysis | [`soft_loc_error_analysis.md`](../docs/soft_loc_error_analysis.md) |

These studies include oracle/label-conditioned variants and should not be
mixed into an online hardware claim without their evidence boundary.

### Verification and data checks

| Runner | Checks | Documentation |
|---|---|---|
| `verify_p2p_cim_approx.py` | L1/Hamming-like P2P distance approximations | [`p2p_cim_approx.md`](../docs/p2p_cim_approx.md) |
| `verify_central_ratio_gate.py` | Central-ratio vs max-normalized P2P gate | [`central_ratio_gate.md`](../docs/central_ratio_gate.md) |
| `verify_waveform_cam_thermometer.py` | Bipolar thermometer waveform CAM | [`waveform_cam_thermometer.md`](../docs/waveform_cam_thermometer.md) |
| `validate_2d_grid.py` | Generated 1024-channel grid and reference path | root README / reference records |

These scripts estimate algorithmic tolerance or dataset validity. They are not
RTL, memory-macro, peripheral, or silicon evidence.

## Shared helpers, not primary entry points

- `spatial_cim_common.py` centralizes event preparation, feature construction,
  truth-row alignment, quantization inputs, and shared metrics for newer
  paper-facing runners.
- `shape_descriptor_common.py` assembles the COM/P2P/POSNEG/LAT/WIDTH/RAND
  descriptor arms against the frozen interfaces in
  [`shape_descriptor_interfaces.md`](../docs/shape_descriptor_interfaces.md).
  The gate auditor intentionally does not import it.
- `geometric_prefilter_common.py` centralizes the registered geometric
  prefilter corpus, split, gates, and aggregation.
- `causal_temporal_selection.py` implements strict selection/provenance and
  confirmation authorization for the temporal pilot.
- `__init__.py` marks the experiment package.

Changes to a shared helper can invalidate multiple result snapshots. Review
the relevant source manifests and result contracts before rerunning only one
downstream table.

## Evaluation rules

1. Fit every scale, quantizer, centroid, radius, threshold, directory, graph,
   template, mask, weight, and tap set only on its registered partition.
2. Keep simultaneous events as separate rows and equal timestamps on one side
   of chronological boundaries.
3. State whether events and unit rows are GT, detected, discovered, delayed GT,
   or self-predicted.
4. Candidate studies report true-unit recall and `C=0/C=1/C>1`, not only final
   accuracy.
5. Distinguish active rows from total row comparisons including widening and
   fallback.
6. Count distinct waveform-template reads and state the full-scan baseline.
7. Do not treat repeated-SNR renderings of one MEArec seed as independent
   temporal or biological replicates.
8. Separate future-sample acquisition latency from compute latency.
9. State whether filtering is full-recording zero-phase, split-local, or
   causal, including warm-up and guard handling.
10. Logical bit/access/op accounting does not establish physical energy or
    area advantage.
11. Paper-facing runs record source/input/output identity and the aggregation
    estimator.
12. A failed formal gate or null selection prohibits confirmation unless a new
    protocol explicitly reopens the question.

Generated CSV/JSON/log artifacts belong under `../output/` and are normally
gitignored. Promote a result to tracked evidence through the matching document
and provenance chain, not by committing ad hoc console output.
