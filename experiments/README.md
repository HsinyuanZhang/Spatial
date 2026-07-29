# Experiment Index

Generated CSVs and logs belong under `../output/` and are gitignored. Protocols,
commands, checked result tables, and negative findings belong under `../docs/`.

## Active DAC pipeline

- `run_adaptive_5bit_range_search.py`
  - 5-bit COM+P2P descriptor;
  - pooled versus per-unit L1/box search;
  - all-row, single-home, and multi-channel-support directories;
  - candidate recall, row activity, waveform reads, and final assignment.
- `run_dual_range_pipeline.py`
  - tight inner box;
  - ambiguity-triggered outer L1 search;
  - early-exit precision and combined traffic accounting.
- `run_certified_dual_range_pipeline.py`
  - disjoint fit/calibration/test chronology;
  - Wilson-certified one-bit safe-exit flag per predicted unit;
  - conservative unsafe-singleton outer trigger.
- `run_compact_waveform_refinement.py`
  - frozen spatial candidate lists shared across all Level-2 representations;
  - 4/5-bit full waveform, Fisher-selected taps, and fixed morphology SAD;
  - template storage, shared tap-index storage, operations, and bit traffic.
- `run_indexed_confusion_candidate_search.py`
  - deterministic 5-bit top-M centroid retrieval;
  - disjoint-calibration anchor-to-confusable-unit SRAM pointer lists;
  - graph storage, all-row comparisons, recall, and waveform-read Pareto.
- `run_online_confusion_graph.py`
  - block-causal graph-state update with static fit-only centroids/radii and
    offline zero-phase input features;
  - delayed-GT upper bound versus self-predicted updates;
  - rolling/accumulating history, pointer rewrites, and observation writes.
- `run_pair_conditioned_waveform.py`
  - frozen spatial candidates and spatially ordered pair tournaments;
  - train-only pair Fisher taps with explicit integer-SAD tie behavior;
  - O(U²) tap-index storage and conservative per-pair bit traffic.
- `run_adaptive_multiprototype_5bit.py`
  - bounded `P=1..4` temporal/farthest-L1 rows per unit;
  - stable prototype-hit to unit-ID de-duplication and prototype-specific
    channel-support directories;
  - 54-bit search-row payload, row/pointer/fallback accounting, and the
    completed negative static upper-bound sweep.
- `run_adaptive_masked_5bit.py`
  - one 63-bit logical row/unit: 45-bit centroid, 9-bit feature mask, and
    9-bit calibrated radius;
  - fit-only deterministic mask selection, calibration-only radii, and
    equal-timestamp-safe 50/25/25 chronology;
  - frozen four-recording all-row development gate with per-record Wilson
    intervals and fallback-inclusive row/lane/addition accounting;
  - completed negative cross-family pilot; confirmation requires an explicit
    frozen configuration and was not run.
- `run_adaptive_weighted_5bit.py`
  - one 74-bit logical row/unit: 45-bit centroid, eighteen shift-code bits, and
    an 11-bit calibrated weighted-L1 radius;
  - fit-only `{1,2,4}` coordinate profiles, calibration-only radii, and the
    same equal-timestamp-safe 50/25/25 chronology;
  - deterministic selection JSON with CSV hashes, exact traffic predicates,
    per-record gate audits, and a confirmation authorization boundary;
  - completed negative cross-family pilot; the null selection prohibits
    confirmation.
- `run_causal_temporal_sketch.py`
  - one locked four-recording, 60-second, `5 delays × 3 M` development pilot;
  - raw filtered full-64 input before per-event normalization, fit-only
    signed-5-bit scale, Fisher-selected dyadic means/Haar contrasts, and
    same-horizon float ceilings;
  - one all-row D9/B5 Level-1 distance matrix shared by candidate membership
    and the C0 argmin cache, with complete Level-1 and C0/C1/C>1 accounting;
  - exact source-path SHA checks, 55-file working-tree source snapshot,
    deterministic CSV/selection artifacts, and a fail-closed confirmation
    boundary;
  - completed negative pilot: all configurations pass `4×` template bits but
    fail accuracy, so selection is null and confirmation was not run.
- `run_two_stage_unsupervised.py`
  - discovered spatial/waveform clusters and held-out assignment.
- `run_unsupervised_two_stage_sweep.py`
  - cross-dataset wrapper for the unsupervised pipeline.

Tracked design/results:

- `../docs/dac_two_level_memory_pipeline.md`
- `../docs/adaptive_5bit_range_search.md`
- `../docs/compact_waveform_refinement.md`
- `../docs/indexed_confusion_candidate_search.md`
- `../docs/hardware_mapping_spec.md`
- `../docs/adaptive_multiprototype_5bit_results.md`
- `../docs/adaptive_multiprototype_5bit_provenance.md`
- `../docs/adaptive_masked_5bit_results.md`
- `../docs/adaptive_masked_5bit_provenance.md`
- `../docs/adaptive_weighted_5bit_results.md`
- `../docs/adaptive_weighted_5bit_provenance.md`
- `../docs/causal_temporal_sketch_plan.md`
- `../docs/causal_temporal_sketch_results.md`
- `../docs/causal_temporal_sketch_provenance.md`
- `../docs/dac_experiment_provenance.md`

## Reproduced baselines and upper bounds

- `run_spatial_sorting.py` — detection-to-SOM reference path.
- `run_spatial_upper_bound.py` — spatial feature comparison.
- `run_two_stage_upper_bound.py` — GT-row two-stage comparison.
- `run_hj16_method_sweep.py` — HJ cross-scene sweep.
- `run_mearec_method_sweep.py` — MEArec cross-recording sweep.
- `run_lowbit_p2p.py` — P2P bit-width sweep.

## Hardware-encoding checks

- `verify_p2p_cim_approx.py`
- `verify_central_ratio_gate.py`
- `verify_waveform_cam_thermometer.py`

These scripts estimate algorithmic tolerance to encodings; they are not RTL,
memory-macro, or peripheral PPA evidence.

## Exploratory/negative branches

- `run_multiscale_fpn.py`
- `run_se_attention.py`
- `run_combined_mscale_se.py`
- `analyze_soft_loc_errors.py`

Retain these for ablation and failure evidence, but do not mix their
oracle/label-conditioned results into the active online hardware claim.

## Evaluation rules

1. Fit quantizers, centroids, ranges, thresholds, and templates on the training
   partition only.
2. Keep simultaneous events as separate rows; never recover labels with one
   `timestamp -> unit` dictionary.
3. Keep equal timestamps on the same side of a chronological split.
4. State whether events/rows are GT, detected, discovered, or oracle.
5. For candidate search, report true-unit recall and zero/one/multiple
   candidate fractions, not only final classification accuracy.
6. Count unique-hit early exits as zero waveform reads.
7. Separate directory pointer storage, active rows, CAM/CiM row payload, and
   waveform-template reads.
8. Do not count repeated-SNR renderings of one MEArec spike-train seed as
   independent temporal replicates.
9. Separate future-sample acquisition latency from compute latency.
10. Current DAC isolation runners use full-recording zero-phase `filtfilt`;
    chronological fitting is not equivalent to a causal raw-signal path.
11. Paper-facing reruns must record code/source version, input manifest,
    algorithm-builder version, and the aggregation estimator.
12. Use `total_row_comparisons_including_fallback` for Level-1 access energy;
    `active_row_count` alone does not include triggered later stages or
    empty-list centroid fallback scans.
13. For online graphs, use ordered pointer-slot plus count-metadata rewrite
    fields; membership additions/removals are only a lower bound.
