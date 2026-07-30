# Documentation Index

This directory separates the active DAC paper direction from reproduced
baselines and exploratory branches. Generated CSV/log files remain under
`output/` and are intentionally gitignored; every result used in a paper claim
must also be summarized in a tracked document here with its command and
evaluation protocol.

## Active DAC direction

- [`dac_two_level_memory_pipeline.md`](dac_two_level_memory_pipeline.md) —
  source of truth for the two-level spatial-search/waveform-refinement
  architecture, contribution boundary, experiment gates, and implementation
  status.
- [`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md) —
  cross-dataset results for per-unit 5-bit L1-CiM/Range-CAM search, local
  directory mappings, nested/safe-exit failure analysis, and the resulting
  indexed-memory pivot.
- [`compact_waveform_refinement.md`](compact_waveform_refinement.md) —
  frozen-candidate comparison of 4/5-bit full-waveform SAD, training-selected
  taps, fixed morphology, template payload, and bit traffic.
- [`causal_temporal_sketch_plan.md`](causal_temporal_sketch_plan.md) —
  executed alignment-conditioned protocol for a signed-5-bit dyadic
  mean/Haar prefix sketch, including the exact 15-point gate and stop rule.
- [`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md) —
  completed negative four-recording pilot: Level-1 recall and Level-2
  template-bit traffic pass, but all shared dyadic configurations fail
  temporal accuracy; selection is null and confirmation is prohibited.
- [`causal_temporal_sketch_provenance.md`](causal_temporal_sketch_provenance.md)
  — 55-file content-addressed runtime source snapshot, exact four inputs,
  three output artifacts, environment, load-time SHA closure, and independent
  gate recomputation. Companion
  `causal_temporal_sketch_{source,inputs,outputs}.sha256` manifests are
  machine-checkable.
- [`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md)
  — deterministic 5-bit top-M, calibration anchor adjacency, causal graph-only
  updates, pointer writes/storage, and their negative joint-gate results.
- [`hardware_mapping_spec.md`](hardware_mapping_spec.md) — bit-exact
  descriptor/row/FIFO/waveform interfaces, pointer and template layouts,
  symbolic energy counters, RTL module boundaries, and claim limits.
- [`spatial_cim_sorting_plan.md`](spatial_cim_sorting_plan.md) —
  pre-registered G1–G6 gates for low-bit spatial WTA as a **classifier**
  (not a prefilter), with biased integer WTA primitive.
- [`spatial_cim_sorting_results.md`](spatial_cim_sorting_results.md) —
  Stage A–C results: G3/G4/G6 pass; G1/G2/G5 fail; selected config is
  5-bit stability-weighted L1 without theta (63 bit/unit).
- [`spatial_cim_attribution.md`](spatial_cim_attribution.md) —
  COM/P2P shuffle controls, error decomposition, Range-CAM and indexed-SRAM
  hardware conformance.
- [`spatial_cim_sorting_provenance.md`](spatial_cim_sorting_provenance.md)
  — source/output digests and gate recomputation pointers.
- [`geometric_prefilter_plan.md`](geometric_prefilter_plan.md) —
  multi-method Level-1 routing pilot (A1/A2 first-trigger ±N, tile, Jaccard,
  cosine, power-gate) with gates G-A / G-D / G-drift / G-E.
- [`geometric_prefilter_results.md`](geometric_prefilter_results.md) —
  G-D/G-E pass; G-A/G-drift fail; tile activation beats first-trigger ±N;
  cosine ≈ L1 within 1 pp.
- [`geometric_prefilter_provenance.md`](geometric_prefilter_provenance.md)
  — source/output digests for the geometric prefilter sweep.
- [`adaptive_multiprototype_5bit_plan.md`](adaptive_multiprototype_5bit_plan.md)
  — pre-registered hypothesis, alternatives considered, bit-exact row
  contract, static/online pilots, and explicit kill criteria.
- [`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md)
  — completed 12-HJ/20-MEArec bounded multi-prototype sweep, P=1 invariant,
  hardware accounting, and the pre-registered decision to stop before online
  self-updates.
- [`adaptive_multiprototype_5bit_provenance.md`](adaptive_multiprototype_5bit_provenance.md)
  — dedicated source/output snapshot for the bounded-row result; companion
  `adaptive_multiprototype_5bit_{source,outputs}.sha256` manifests reuse the
  existing exact 32-recording input manifest.
- [`adaptive_masked_5bit_plan.md`](adaptive_masked_5bit_plan.md) — frozen
  protocol and completed disposition for one 63-bit row/unit with a
  deterministic 9-bit feature mask, gated L1 lanes, three-way chronology, and
  explicit stop criteria.
- [`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md) —
  completed four-recording development pilot: strong HJ movement, MEArec
  cross-family failure, per-record Wilson intervals, lane/row accounting, and
  the pre-registered decision not to run confirmation.
- [`adaptive_masked_5bit_provenance.md`](adaptive_masked_5bit_provenance.md) —
  exact four-file source and two-CSV snapshot; companion
  `adaptive_masked_5bit_{source,outputs}.sha256` manifests reuse the four
  relevant entries in the locked 32-recording input manifest.
- [`next_spatial_relation_candidates.md`](next_spatial_relation_candidates.md)
  — structured divergence over 18 post-mask relations, convergence filters,
  the five strongest candidates, and the rationale for selecting nonzero
  power-of-two weights.
- [`adaptive_weighted_5bit_plan.md`](adaptive_weighted_5bit_plan.md) — frozen
  and completed one-row/unit protocol: 74-bit logical rows, `{1,2,4}` shift
  weights, four-recording gate, immutable selection boundary, and stop rule.
- [`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md) —
  completed development pilot: strong HJ candidate narrowing, insufficient or
  negative MEArec movement, zero formal candidates, and no confirmation.
- [`adaptive_weighted_5bit_provenance.md`](adaptive_weighted_5bit_provenance.md)
  — exact four-file source and three-artifact snapshot; companion
  `adaptive_weighted_5bit_{source,outputs}.sha256` manifests reuse the locked
  four-recording input subset.
- [`dac_experiment_provenance.md`](dac_experiment_provenance.md) — exact code,
  input-corpus, environment, output-CSV, aggregation, and verification
  snapshot. Companion `dac_experiment_{source,inputs,outputs}.sha256` files are
  machine-checkable manifests.

## Reproduced baselines and algorithm limits

- [`spatial_upper_bound.md`](spatial_upper_bound.md) — spatial-only feature
  upper bound.
- [`hj16_method_sweep.md`](hj16_method_sweep.md) — Hybrid Janelia cross-scene
  comparison.
- [`mearec_method_sweep.md`](mearec_method_sweep.md) — MEArec cross-recording
  comparison.
- [`two_stage_upper_bound.md`](two_stage_upper_bound.md) — oracle two-stage
  study.
- [`two_stage_unsupervised.md`](two_stage_unsupervised.md) — unsupervised
  hierarchical sorter.
- [`soft_loc_error_analysis.md`](soft_loc_error_analysis.md) — failure pairs
  that spatial features alone cannot separate.

## CiM/CAM and low-bit studies

- [`p2p_cim_approx.md`](p2p_cim_approx.md) — distance approximations for the
  spatial gate.
- [`central_ratio_gate.md`](central_ratio_gate.md) — central-ratio encoding on
  the initial HJ scene.
- [`central_ratio_vs_maxnorm_benchmark.md`](central_ratio_vs_maxnorm_benchmark.md)
  — cross-dataset encoding comparison.
- [`waveform_cam_thermometer.md`](waveform_cam_thermometer.md) — waveform CAM
  encoding and storage expansion.

## Exploratory branches

These are retained as negative or conditional evidence, not as the main DAC
contribution.

- [`multiscale_fpn.md`](multiscale_fpn.md)
- [`se_attention.md`](se_attention.md)
- [`combined_mscale_se.md`](combined_mscale_se.md)

## Documentation rules

1. State whether spike times and unit labels are ground truth, discovered, or
   detected.
2. State the split policy and fit every quantizer, threshold, centroid, and
   template on the training partition only.
3. Report candidate recall and candidate-set size in addition to final sorting
   accuracy.
4. Do not treat multiple SNR renderings of one spike-train seed as independent
   biological/temporal replicates.
5. Separate acquisition latency (future waveform samples) from compute latency.
6. Distinguish modeled memory-cell cost from synthesized macro and peripheral
   cost.
7. Record the exact command and output path for every table.
8. State whether filtering is full-recording zero-phase, split-local, or causal,
   including guard/warm-up handling.
9. Record code revision/source hashes, dataset manifest, box-builder version,
   and aggregation mode for paper-facing reruns.
