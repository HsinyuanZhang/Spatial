# Spatial Documentation Index

This directory is the tracked evidence and design record for Spatial. Use the
navigation pages first; open individual protocol/result/provenance files when
you need exact evidence.

## Primary navigation

| Document | Purpose |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System model, algorithm generations, data contracts, code ownership, and evidence levels |
| [`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) | Current verdict and passed/failed/open branch disposition |
| [`REPRODUCING.md`](REPRODUCING.md) | Environment, datasets, commands, splits, metrics, and artifact discipline |
| [`../experiments/README.md`](../experiments/README.md) | Runner-to-question and runner-to-document map |
| [`COM_P2P_HANDOFF.md`](COM_P2P_HANDOFF.md) | Chinese project handoff and motivation for the relative-offset/ShiftCAM branch |

The result records remain in a flat directory on purpose. Historical commands,
relative links, paper references, and SHA-256 manifests depend on their current
paths. Reorganization happens through this index rather than by moving
provenance-sensitive files.

## Status vocabulary

| Status | Meaning |
|---|---|
| **Active question** | Current direction or unresolved design question; not a successful claim |
| **Local pass** | A pre-registered component hypothesis passed its own gate |
| **Conditional** | Holds only behind a stated source, split, dataset, or representation contract |
| **Negative** | A tested branch failed its gate and is retained as design-space evidence |
| **Baseline** | Reference, upper bound, or mandatory comparison; not necessarily deployable |
| **Open** | Missing evidence; must not be written as a completed result |

A local pass is not a system pass. The system still lacks a configuration that
jointly clears high candidate recall, 4x template-read reduction, compact
temporal accuracy, causal/no-GT operation, realistic dense-2D validation, and
physical implementation comparison.

## Current relative-offset / shifted-similarity branch

Read in this order:

1. [`COM_P2P_HANDOFF.md`](COM_P2P_HANDOFF.md) — project constraints, failure
   analysis of KNN P2P slots and drift, and the ShiftCAM-inspired hypothesis.
2. [`relative_offset_footprint_plan.md`](relative_offset_footprint_plan.md) —
   P0 fixed-slot protocol.
3. [`relative_offset_footprint_results.md`](relative_offset_footprint_results.md)
   — **local pass** on HJ; MEArec argmin improves with a small true-distance
   regression.
4. [`shift_cim_plan.md`](shift_cim_plan.md) — P1 shift-min and multi-copy row
   accounting protocol.
5. [`shift_cim_results.md`](shift_cim_results.md) — **local pass**; the HJ gain
   is small, MEArec prefers no shift, and row cost scales by `2S+1`.
6. [`shift_conv_bit_sweep_results.md`](shift_conv_bit_sweep_results.md) —
   normalized correlation is the viable MAC relation; raw dot product fails.
7. [`robust_p2p_template_plan.md`](robust_p2p_template_plan.md) — registered
   3/5-point estimators, 4–8-bit grid, confirmation, and online-update gates.
8. [`robust_p2p_template_results.md`](robust_p2p_template_results.md) —
   **negative estimator confirmation**; conditional 8-bit equivalence passes,
   and the self-predicted online data run is correctly stopped.
9. [`shape_orthogonal_descriptor_plan.md`](shape_orthogonal_descriptor_plan.md)
   — registered protocol for whether the amplitude footprint has reached an
   information ceiling, and whether split pos/neg peaks, cross-channel trough
   latency, and home-channel width raise it.
10. [`shape_orthogonal_descriptor_results.md`](shape_orthogonal_descriptor_results.md)
    — **Phase A pass, Phase A2 fail, Phase B negative; no configuration
    selected.** The representation gains +10.65 pp HJ and +8.21 pp MEArec over
    D9/B5 and survives a causal filter, but loses 1.40 pp under ±2-sample
    alignment jitter on MEArec, and moves the candidate recall/traffic frontier
    by nothing. Retained for its two durable findings and as a negative result.
11. [`shape_descriptor_interfaces.md`](shape_descriptor_interfaces.md) —
    frozen primitive signatures, descriptor arms, CSV schema, selection JSON
    schema, and the authoritative gate restatement for that plan. Frozen before
    implementation so the runner and its independent gate auditor could be
    built without agreeing at implementation time.

Related routing evidence:

- [`geometric_prefilter_plan.md`](geometric_prefilter_plan.md) — registered
  first-trigger, tile, Jaccard, cosine, and power-gate pilot.
- [`geometric_prefilter_results.md`](geometric_prefilter_results.md) — tile and
  cosine gates pass; first-trigger and drift gates fail.
- [`geometric_prefilter_provenance.md`](geometric_prefilter_provenance.md) —
  source/output digest record.

## System architecture and hardware contract

- [`dac_two_level_memory_pipeline.md`](dac_two_level_memory_pipeline.md) —
  foundational two-level architecture, evaluation gates, implementation
  checklist, and open gaps. It predates the newest relative-offset branch; use
  `RESULTS_OVERVIEW.md` for the current branch status.
- [`hardware_mapping_spec.md`](hardware_mapping_spec.md) — bit-exact descriptor,
  row, pointer, FIFO, template, and RTL-boundary specification. This is logical
  mapping evidence, not macro or silicon PPA.
- [`dac_experiment_provenance.md`](dac_experiment_provenance.md) — source,
  input, output, environment, aggregation, and verification snapshot for the
  mature DAC experiment stack.

Companion machine-checkable manifests:

```text
dac_experiment_source.sha256
dac_experiment_inputs.sha256
dac_experiment_outputs.sha256
```

## Level-1 classifier evidence

- [`spatial_cim_sorting_plan.md`](spatial_cim_sorting_plan.md) — registered G1-G6
  low-bit spatial WTA classifier protocol.
- [`spatial_cim_sorting_results.md`](spatial_cim_sorting_results.md) — G3/G4/G6
  pass; accuracy, online drift, and unsupervised gates do not all pass.
- [`spatial_cim_attribution.md`](spatial_cim_attribution.md) — COM/P2P shuffle
  controls, error decomposition, and Range-CAM/indexed-SRAM conformance.
- [`spatial_cim_sorting_provenance.md`](spatial_cim_sorting_provenance.md) —
  digest and gate-recomputation pointers.

Companion manifests:

```text
spatial_cim_sorting_source.sha256
spatial_cim_sorting_outputs.sha256
```

## Level-1 candidate-search design history

### Per-unit range/L1, directories, and indexed retrieval

- [`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md) — principal
  cross-dataset D9/B5 range/L1 result, directory mappings, nested/safe-exit
  failures, and recall/traffic frontier.
- [`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md)
  — deterministic top-M retrieval, calibration anchor adjacency, and causal
  graph-only updates; negative joint-gate result.

### Bounded multiple rows per unit

- [`adaptive_multiprototype_5bit_plan.md`](adaptive_multiprototype_5bit_plan.md)
  — pre-registered P=1..4 prototype-row protocol.
- [`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md)
  — **negative** cross-dataset result and stop decision.
- [`adaptive_multiprototype_5bit_provenance.md`](adaptive_multiprototype_5bit_provenance.md)
  — dedicated source/output snapshot.

Companion manifests:

```text
adaptive_multiprototype_5bit_source.sha256
adaptive_multiprototype_5bit_outputs.sha256
```

### Unit-specific coordinate masks

- [`adaptive_masked_5bit_plan.md`](adaptive_masked_5bit_plan.md) — frozen
  one-row/unit mask protocol and confirmation boundary.
- [`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md) — HJ
  development movement but **negative** cross-family gate; confirmation was
  not run.
- [`adaptive_masked_5bit_provenance.md`](adaptive_masked_5bit_provenance.md) —
  exact pilot snapshot and accounting audit.

Companion manifests:

```text
adaptive_masked_5bit_source.sha256
adaptive_masked_5bit_outputs.sha256
```

### Unit-specific power-of-two weights

- [`next_spatial_relation_candidates.md`](next_spatial_relation_candidates.md)
  — post-mask idea divergence and selection rationale.
- [`adaptive_weighted_5bit_plan.md`](adaptive_weighted_5bit_plan.md) — frozen
  `{1,2,4}` shift-weighted L1 protocol.
- [`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md) — HJ
  traffic movement but **negative** cross-family gate; formal selection is
  null.
- [`adaptive_weighted_5bit_provenance.md`](adaptive_weighted_5bit_provenance.md)
  — exact source/output/selection snapshot.

Companion manifests:

```text
adaptive_weighted_5bit_source.sha256
adaptive_weighted_5bit_outputs.sha256
```

## Level-2 waveform refinement

### Compact samples and pair-conditioned taps

- [`compact_waveform_refinement.md`](compact_waveform_refinement.md) — full
  4/5-bit SAD, shared Fisher taps, morphology, pair-conditioned taps, template
  payload, and bit traffic. Shared taps are conditional; pair taps recover
  accuracy but not the required traffic/storage efficiency.

### Causal-prefix temporal sketches

- [`causal_temporal_sketch_candidates.md`](causal_temporal_sketch_candidates.md)
  — candidate representations and selection of dyadic means/Haar contrasts.
- [`causal_temporal_sketch_plan.md`](causal_temporal_sketch_plan.md) — exact
  15-point development grid, input contract, gate, and stop rule.
- [`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md) —
  **negative** accuracy result; template-bit gate passes, selection is null,
  and confirmation is prohibited.
- [`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md)
  — source, input, output, environment, and independent gate audit.

Companion manifests:

```text
causal_temporal_sketch_source.sha256
causal_temporal_sketch_inputs.sha256
causal_temporal_sketch_outputs.sha256
```

## Reproduced baselines and algorithm limits

- [`spatial_upper_bound.md`](spatial_upper_bound.md) — spatial-only feature
  ceiling and why `soft_loc` helps on 1D probes.
- [`hj16_method_sweep.md`](hj16_method_sweep.md) — Hybrid Janelia cross-scene
  spatial and two-stage comparison.
- [`mearec_method_sweep.md`](mearec_method_sweep.md) — MEArec cross-recording
  comparison and repeated-SNR caveat.
- [`two_stage_upper_bound.md`](two_stage_upper_bound.md) — GT-row COM/P2P to
  waveform upper bound.
- [`two_stage_unsupervised.md`](two_stage_unsupervised.md) — discovered
  hierarchical rows across HJ and MEArec.
- [`soft_loc_error_analysis.md`](soft_loc_error_analysis.md) — same-channel and
  nearby-location failure-pair analysis.

## Encoding and CiM/CAM approximation studies

- [`p2p_cim_approx.md`](p2p_cim_approx.md) — P2P distance approximations for
  the initial two-stage gate.
- [`central_ratio_gate.md`](central_ratio_gate.md) — initial single-scene
  central-ratio encoding study.
- [`central_ratio_vs_maxnorm_benchmark.md`](central_ratio_vs_maxnorm_benchmark.md)
  — HJ/MEArec follow-up comparing central ratio and per-spike max normalization.
- [`waveform_cam_thermometer.md`](waveform_cam_thermometer.md) — bipolar
  thermometer CAM encoding, accuracy, and storage expansion.

These files report algorithmic tolerance to encodings. They are not physical
memory-array or peripheral evidence.

## Exploratory feature branches

These studies are retained as ablations and conditional evidence rather than
the current hardware mainline:

- [`multiscale_fpn.md`](multiscale_fpn.md) — multi-K FPN-like footprint fusion;
- [`se_attention.md`](se_attention.md) — Fisher/variance/per-cluster channel
  weighting;
- [`combined_mscale_se.md`](combined_mscale_se.md) — iterative
  SOM-discovered per-cluster attention and combined variants.

## Documentation contract

Every paper-facing result document must make the following explicit:

1. **Truth boundary:** event times and unit rows are GT, detected, discovered,
   delayed GT, or self-predicted.
2. **Split:** chronological boundaries and equal-timestamp handling.
3. **Fit boundary:** where quantizers, scales, centroids, radii, directories,
   templates, masks, weights, and taps are fitted.
4. **Signal path:** full-recording zero-phase, split-local, or causal filtering,
   including guard/warm-up handling.
5. **Metrics:** candidate recall, candidate count distribution, final accuracy,
   row comparisons including fallback, and distinct template reads.
6. **Aggregation:** event-weighted, recording-unweighted, seed-weighted, or
   another named estimator.
7. **Replicate identity:** repeated SNR renderings of one MEArec spike-train seed
   are not independent temporal/biological replicates.
8. **Latency:** acquisition of future waveform samples is separate from compute
   latency.
9. **Hardware boundary:** logical bits/operations/accesses are separate from
   macro/peripheral/silicon PPA.
10. **Reproduction:** exact command, output path, source revision or hash,
    dataset manifest, and builder/contract version.
11. **Decision:** local gate, system implication, selected configuration or
    explicit null, and confirmation authorization.
12. **Negative evidence:** a failed gate is retained; an attractive unregistered
    point is not renamed as the selected configuration.

Generated CSV/JSON/log files remain under `output/` and are normally
gitignored. A claim is not durable until its protocol, result, and evidence
boundary are represented in tracked documentation here.
