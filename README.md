# Spatial Spike Sorting

Reproduction and extension of the spatial clustering spike sorter from:

> **Akhoundi et al., "A Scalable 1024-Channel Ultra-Low-Power Spike Sorting Chip With Event-Driven Detection and Spatial Clustering", IEEE JSSC, 2025.**

Starting from the TU Delft MATLAB reference, we ported to Python, validated across datasets, and developed improved spatial-only features, an unsupervised two-stage pipeline, CiM/CAM hardware approximations, and CV-inspired enhancements (SE-Net attention, multi-scale FPN).

---

## Directory Structure

```
Spatial/
├── README.md
├── .gitignore
├── papers/                              # Reference paper PDF
│
├── reference_code/                      # TU Delft MATLAB implementation
│   ├── Classification/                  #   SOM clustering
│   ├── FeatureExtraction/               #   Spatial features (Eq. 3-5)
│   ├── SpikeDetection/                  #   NEO + VC-SPD detection
│   ├── WiredOR/                         #   Wired-OR analog compression
│   └── MEArec/                          #   32x32 2D grid probe configs
│
├── configs/
│   └── experiment.yaml                  # Per-dataset optimal SOM + detection params
│
├── data/
│   ├── loader.py                        # Unified loader (HJ / MEArec / CortexLab NP)
│   ├── generate_2d_grid.py              # 32x32 staggered honeycomb grid generator
│   ├── generate_mearec.py               # Legacy 32ch 1D generator (deprecated)
│   └── prepare_hybrid_janelia.py        # HJ dataset preparation
│
├── algorithms/
│   ├── detection.py                     # NEO detection + bandpass filter
│   ├── spatial_features.py              # Akhoundi Eq. 3-5 (angle / ratio / position)
│   ├── spatial_footprint.py             # soft_loc (COM + spread + sharpness), footprint P2P/PCA
│   ├── som_clustering.py                # Modified SOM (BMU + pruning + merge)
│   ├── reference_port.py                # Faithful Python port of all 5 MATLAB modules
│   ├── two_stage_assign.py              # COM∧P2P gate + waveform 2nd-stage assignment
│   ├── unsupervised_two_stage.py        # Fully unsupervised hierarchical 2-stage sorter
│   ├── adaptive_range_search.py         # 5-bit per-unit L1/Range-CAM candidate search
│   ├── adaptive_prototype_search.py     # Bounded multi-row 5-bit candidate search
│   ├── adaptive_masked_range_search.py  # Unit-specific masked-L1 5-bit rows
│   ├── adaptive_weighted_range_search.py # Unit-specific shift-weighted 5-bit rows
│   ├── causal_preprocessing.py          # Stateful forward-only SOS filter primitive
│   ├── causal_temporal_sketch.py        # Signed-5-bit dyadic prefix features + accounting
│   ├── causal_temporal_baselines.py     # Frozen full64/tap/morphology teachers
│   ├── spatial_wta.py                   # Biased integer WTA spatial classifier
│   ├── online_spatial_adaptation.py     # Shift-EMA online centroid/theta updates
│   ├── cim_thermometer.py               # Bipolar thermometer CAM encoding (hardware approx)
│   ├── multiscale_footprint.py          # Multi-K (FPN-style) footprint extraction + fusion
│   └── se_attention.py                  # SE-Net channel attention (Fisher / variance / per-cluster)
│
├── experiments/
│   ├── README.md                         # Experiment index and evaluation rules
│   ├── run_spatial_sorting.py           # Full pipeline: detect → features → SOM → eval
│   ├── run_spatial_upper_bound.py       # Spatial feature comparison (oracle + SOM)
│   ├── run_two_stage_upper_bound.py     # Two-stage oracle (GT centroids)
│   ├── run_two_stage_unsupervised.py    # Unsupervised two-stage (no GT)
│   ├── run_unsupervised_two_stage_sweep.py  # Cross-dataset unsupervised sweep
│   ├── run_hj16_method_sweep.py         # Cross-scene HJ validation
│   ├── run_mearec_method_sweep.py       # Cross-recording MEArec validation
│   ├── run_lowbit_p2p.py                # P2P quantization bit-width sweep
│   ├── sweep_som_params.py              # SOM hyperparameter grid search
│   ├── validate_2d_grid.py              # 2D grid reference pipeline validation
│   ├── verify_central_ratio_gate.py     # P2P representation comparison
│   ├── verify_p2p_cim_approx.py         # CiM distance approximation verification
│   ├── verify_waveform_cam_thermometer.py  # Waveform CAM encoding comparison
│   ├── run_adaptive_5bit_range_search.py   # Per-unit range + directory Pareto sweep
│   ├── run_adaptive_multiprototype_5bit.py # Bounded P=1..4 row-bank sweep
│   ├── run_adaptive_masked_5bit.py       # Frozen four-recording masked-L1 pilot
│   ├── run_adaptive_weighted_5bit.py     # Frozen four-recording weighted-L1 pilot
│   ├── causal_temporal_selection.py      # Strict 15-grid gate + provenance/authorization
│   ├── run_causal_temporal_sketch.py     # Completed four-recording dyadic Level-2 pilot
│   ├── run_spatial_wta_pilot.py          # Spatial WTA classifier Stage A pilot (G1–G3)
│   ├── run_spatial_wta_corpus.py         # Selected config on 12 HJ + 20 MEArec
│   ├── run_spatial_attribution.py        # G4 attribution + G6 hardware conformance
│   ├── run_spatial_online_drift.py       # G5 online EMA on HJ drift
│   ├── run_spatial_unsupervised.py       # Quantized SOM + Hungarian Stage C
│   ├── run_dual_range_pipeline.py          # Inner-box / outer-L1 trigger study
│   ├── run_certified_dual_range_pipeline.py # Per-unit Wilson safe-exit study
│   ├── run_compact_waveform_refinement.py   # Frozen-candidate digital Level-2 sweep
│   ├── run_indexed_confusion_candidate_search.py # 5-bit anchor adjacency study
│   ├── run_online_confusion_graph.py     # Block-causal graph-state ceiling
│   ├── run_pair_conditioned_waveform.py  # Pair-specific 5-bit tap tournaments
│   ├── analyze_soft_loc_errors.py       # Error-pair analysis + FSDE rescue
│   ├── run_multiscale_fpn.py            # Multi-scale FPN footprint fusion experiment
│   ├── run_se_attention.py              # SE-Net channel attention experiment
│   └── run_combined_mscale_se.py        # Combined multi-scale + SE-Net experiment
│
├── evaluation/
│   └── metrics.py                       # Clustering accuracy (Eq. 10), detection P/R/F1
│
├── tests/                               # pytest unit tests per algorithm module
│
├── docs/                                # Tracked protocols, results, and paper direction
│   ├── README.md                        # Documentation index and result-record rules
│   ├── dac_two_level_memory_pipeline.md # Active DAC architecture and acceptance gates
│   ├── adaptive_5bit_range_search.md    # 5-bit per-unit search results and limits
│   ├── compact_waveform_refinement.md   # 4/5-bit SAD, selected taps, and traffic
│   ├── indexed_confusion_candidate_search.md # Indexed candidate graph results
│   ├── hardware_mapping_spec.md         # Bit-exact memories, interfaces, and RTL boundary
│   ├── adaptive_multiprototype_5bit_plan.md # Pre-registered bounded-row hypothesis
│   ├── adaptive_multiprototype_5bit_results.md # Completed negative bounded-row sweep
│   ├── adaptive_multiprototype_5bit_provenance.md # Dedicated result snapshot
│   ├── adaptive_multiprototype_5bit_source.sha256 # Bounded-row code/test hashes
│   ├── adaptive_multiprototype_5bit_outputs.sha256 # Final bounded-row CSV hashes
│   ├── adaptive_masked_5bit_plan.md     # Frozen unit-specific mask protocol/disposition
│   ├── adaptive_masked_5bit_results.md  # Completed cross-family development pilot
│   ├── adaptive_masked_5bit_provenance.md # Dedicated pilot snapshot
│   ├── adaptive_masked_5bit_source.sha256 # Masked-row code/test hashes
│   ├── adaptive_masked_5bit_outputs.sha256 # Final pilot CSV hashes
│   ├── next_spatial_relation_candidates.md # Post-mask candidate convergence
│   ├── adaptive_weighted_5bit_plan.md   # Frozen power-of-two weight protocol
│   ├── adaptive_weighted_5bit_results.md # Completed negative development pilot
│   ├── adaptive_weighted_5bit_provenance.md # Weighted-pilot snapshot
│   ├── adaptive_weighted_5bit_source.sha256 # Weighted-row code/test hashes
│   ├── adaptive_weighted_5bit_outputs.sha256 # Pilot CSV/selection hashes
│   ├── causal_temporal_sketch_plan.md    # Executed dyadic-prefix protocol and stop rule
│   ├── causal_temporal_sketch_results.md # Negative 5-bit Level-2 gate and diagnosis
│   ├── causal_temporal_sketch_provenance.md # 55-source/input/output snapshot
│   ├── causal_temporal_sketch_source.sha256 # Runtime package source hashes
│   ├── causal_temporal_sketch_inputs.sha256 # Exact four pilot input hashes
│   ├── causal_temporal_sketch_outputs.sha256 # Two CSV + selection JSON hashes
│   ├── dac_experiment_provenance.md     # Source/input/output hash snapshot
│   ├── dac_experiment_source.sha256     # Exact new DAC source hashes
│   ├── dac_experiment_inputs.sha256     # Exact 32-recording input hashes
│   ├── dac_experiment_outputs.sha256    # Exact paper-facing CSV hashes
│   ├── spatial_upper_bound.md
│   ├── two_stage_upper_bound.md
│   ├── two_stage_unsupervised.md
│   ├── soft_loc_error_analysis.md
│   ├── hj16_method_sweep.md
│   ├── mearec_method_sweep.md
│   ├── central_ratio_gate.md
│   ├── central_ratio_vs_maxnorm_benchmark.md
│   ├── p2p_cim_approx.md
│   ├── waveform_cam_thermometer.md
│   ├── multiscale_fpn.md                # Multi-scale FPN results
│   ├── se_attention.md                  # SE-Net attention results
│   └── combined_mscale_se.md            # Combined experiment results
│
└── output/                              # Generated CSVs and logs (gitignored)
```

---

## Datasets

| Dataset | Probe | Channels | Duration | Units | SNR | Location |
|---------|-------|----------|----------|-------|-----|----------|
| Hybrid Janelia | 16ch linear, 20µm | 16 | 600s @ 30kHz | 10 | 8–16 | `../new_datasets/hybridjanelia/` |
| MEArec v1 | 32ch linear | 32 | 60s @ 30kHz | 10/20 | 2–12 | `../synthetic_spike_dataset/generated/geoosort_npz_v1/` |
| 2D Grid | 32×32 staggered, 36µm | 1024 | 60s @ 20kHz | 15 | 3–12 | `output/mearec_2d_grid_v2/` |
| CortexLab NP | NP1 64×2, 20µm | 128 | 228s @ 30kHz | 10 | — | `../output/cortexlab_10pct/` |

---

## Algorithm Pipeline

### Generation 1: Akhoundi Reference

```
Raw → Bandpass(300–6000Hz) → NEO Detection → Eq.3-5 Features (5D) → Modified SOM
```

Works on 2D high-density probes; degenerates on 1D linear probes (angle collapses to ±π/2).

### Generation 2: soft_loc Spatial-Only

```
Raw → Bandpass → GT/Detected Spikes → Local P2P (K neighbors) → soft_loc (4D) → SOM
```

`soft_loc` = geometry-weighted center-of-mass (x,y) + spread + sharpness. Best spatial-only representation on 1D probes. Oracle 91.6%, SOM 83.3% on HJ 16ch.

### Generation 3: Unsupervised Two-Stage

```
Stage 0: Spatial SOM (soft_loc) → within-cluster waveform SOM → hierarchical labels
Stage 1: COM∧P2P gate (legacy implementation uses pooled thresholds) → candidate units
Stage 2: Nearest waveform centroid among candidates
Fallback: Normalized COM+P2P score
```

Reaches 96.8% oracle on HJ. Fully unsupervised variant: ~81% HJ, ~79% MEArec.

### Generation 4: CV-Inspired Enhancements (this work)

**SE-Net per-cluster attention**: Weight each channel by the unit's time-averaged footprint profile (denoised Bayesian prior on the per-spike amplitude weights).

| Method (HJ scene 11) | sep_ratio | Oracle | SOM |
|----------------------|-----------|--------|-----|
| baseline soft_loc | 0.336 | 0.916 | 0.833 |
| oracle per-cluster attention | **0.229** | **0.966** | **0.962** |
| SOM-discovered per-cluster (unsupervised) | 0.255 | 0.922 | **0.913** |

**Multi-scale FPN**: Extract soft_loc at multiple K values and fuse. Marginal gains — best single-K (K=5) already matches fusion for SOM; high-dimensional concat hurts clustering.

See `docs/se_attention.md`, `docs/multiscale_fpn.md`, `docs/combined_mscale_se.md`.

### Generation 5: 5-bit Per-Unit Memory Search (active DAC direction)

```text
COM + K-neighbor P2P
    → 5-bit D-dimensional descriptor
    → channel/tile directory row mask
    → local 5-bit CiM/near-memory candidate retrieval
    → candidate-only 5-bit digital waveform SAD
```

The implementation supports a different integer radius or feature box for
every unit, inclusive comparisons, progressive empty-set widening, multi-channel
directory pointers, and fixed full-scale hardware bit accounting.

Cross-dataset results show a real but limited per-unit benefit: at 95%
calibration, mean candidates fall from pooled-L1 6.49 to 5.37 on HJ and from
6.02 to 5.52 on MEArec. A single high-recall range is still too broad:
99.9%-calibrated per-unit L1 reaches 0.993/0.995 candidate recall but only about
1.25× waveform-read reduction. A tight-inner/outer-L1 cascade also fails:
uncertified singleton exits reach only 0.956 HJ recall at the 4.14× traffic
point, while the best tested MEArec recall is 0.987 at 1.33×. Per-unit Wilson
certification does not rescue it: a strict 0.99 target certifies no rows, and a
0.95 target yields only about 5% early exits and 1.38×/1.36× read reduction.

Range-CAM singleton exit is therefore retained as a negative ablation rather
than the main mechanism. The next Level-1 branch tested deterministic indexed
local SRAM/CiM or near-memory retrieval using the same 5-bit descriptors; it
is now a required baseline rather than a passing policy. The first static
anchor-adjacency implementation is insufficient: its best
4×-class recall is 0.932/0.938, while its highest-recall point reaches only
0.980/0.986 at 1.85×/1.53×. Graph-state block-causal updates also fail: even
delayed-GT updates reach only 0.947/0.934 recall at 4×-class traffic. These
experiments use offline full-recording zero-phase filtering; they are not an
end-to-end causal signal path.

A bounded multi-prototype study then tested whether unit-specific drift or
channel handoff could be represented by `P≤4` narrow 54-bit rows. It also
fails the pre-registered gate: HJ's best ≥4× point reaches only 0.898 recall
and requires `P=4`, while MEArec has no ≥4× point; the highest-recall points
remain at about 1.25× traffic reduction. A stateful forward-only SOS filter
primitive is implemented and tested, but causal event preparation and
centroid/radius/directory updates with non-GT confidence remain open.

A subsequent one-row/unit masked-L1 pilot changes the relation rather than
adding rows: each logical 63-bit row stores a 9-bit feature mask beside the
45-bit centroid and 9-bit radius. On the two HJ development recordings,
stability/M3/p95 improves recall from 0.906 to 0.953 and waveform reduction
from 3.90× to 6.04× while cutting enabled lanes by about 66%. It does not
generalize to the two MEArec development recordings, where the same point
worsens traffic from 2.49× to 2.29×. Zero configurations pass the
pre-registered cross-family pilot gate, so no disjoint confirmation or
directory-assisted masked sweep was run.

A follow-on one-row/unit weighted-L1 pilot preserves all nine coordinates and
uses only `{1,2,4}` shifts. The 74-bit logical row changes the acceptance
shape without a multiplier. At p95, `stability/top3_x4` improves HJ recall
from 0.906 to 0.908 and traffic reduction from 3.90× to 7.90×, but MEArec
recall moves from 0.956 to 0.955 and traffic improves only from 2.49× to
2.59×. No frozen nonuniform profile passes the same-branch cross-family gate;
the immutable selection artifact contains `null`, so confirmation was
prohibited and not run. Static mask/weight micro-tuning is therefore closed
as the next main branch.

For Level 2, full 5-bit×64 SAD is the mandatory digital baseline, not an
input-contract-independent accuracy guarantee. A shared
Fisher-48×5 template is within one point of the float teacher only behind the
tight p95 gate; no shared-tap design passes both datasets behind the high-recall
p99.9 gate.
Pair-conditioned taps recover the accuracy side at high recall:
pair-16×5 is +0.04pp/−0.97pp versus the teacher on HJ/MEArec. It provides only
1.35×/1.36× conservative bit-traffic reduction and adds O(U²) tap-index
storage, so it is not yet the memory solution.

The completed alignment-conditioned dyadic-prefix pilot tests the opposite
side of that trade-off. Signed-5-bit `M=8/12/16` templates all exceed 4×
template-bit reduction behind a p99.9 spatial source, but all 15 configurations
fail temporal accuracy. The least-bad delay-8/M16 point loses
6.10pp/12.08pp on HJ/MEArec, so the deterministic selection is `null` and no
confirmation was run. The raw-prefix float ceiling is already within one point
by delay 16; the missing mechanism is the representation, not more waveform
latency.

See `docs/dac_two_level_memory_pipeline.md` and
`docs/adaptive_5bit_range_search.md`, plus
`docs/compact_waveform_refinement.md` and
`docs/indexed_confusion_candidate_search.md`. The bounded-row protocol and
result are in `docs/adaptive_multiprototype_5bit_plan.md` and
`docs/adaptive_multiprototype_5bit_results.md`. The later unit-mask protocol,
pilot result, and exact snapshot are in `docs/adaptive_masked_5bit_plan.md`,
`docs/adaptive_masked_5bit_results.md`, and
`docs/adaptive_masked_5bit_provenance.md`. The selected post-mask hypothesis,
completed weighted pilot, and immutable snapshot are in
`docs/next_spatial_relation_candidates.md`,
`docs/adaptive_weighted_5bit_results.md`, and
`docs/adaptive_weighted_5bit_provenance.md`. The temporal sketch protocol,
negative result, and exact snapshot are in
`docs/causal_temporal_sketch_plan.md`,
`docs/causal_temporal_sketch_results.md`, and
`docs/causal_temporal_sketch_provenance.md`.

---

## CiM/CAM and Low-Bit Hardware Studies

The repository now separates encodings that preserve a useful algorithmic
ceiling from branches that expose a recall/traffic limit. None of these rows is
silicon PPA evidence:

| Component | Float | CiM Approximation | Verified In |
|-----------|-------|--------------------|-------------|
| P2P distance | L2 | L1 on n-bit codes | `verify_p2p_cim_approx.py` |
| P2P encoding | float | per-spike-max / central-ratio quantization | `run_lowbit_p2p.py` |
| Spatial candidate rows | pooled float/code thresholds | 5-bit per-unit L1/range rows; bounded multi-row, masked, and weighted rows are negative branches | `adaptive_range_search.py`, `adaptive_prototype_search.py`, `adaptive_masked_range_search.py`, `adaptive_weighted_range_search.py` |
| Unit-specific lane gating | all nine descriptor coordinates | 9-bit row mask over 5-bit absolute-difference lanes; HJ-positive development ablation only | `run_adaptive_masked_5bit.py` |
| Unit-specific coordinate weighting | float/continuous weights | `{1,2,4}` shifts, nine active lanes, 11-bit sum, 74-bit logical row; HJ-positive but cross-family-negative pilot | `run_adaptive_weighted_5bit.py` |
| Spatial WTA classifier | float L1 argmin | 5-bit biased/weighted integer WTA; G3/G4/G6 pass at 63 bit/unit | `spatial_wta.py`, `run_spatial_wta_pilot.py` |
| Coarse locality | one home row | per-unit multi-channel directory pointers | `run_adaptive_5bit_range_search.py` |
| Waveform distance | L2 | Hamming on bipolar thermometer codes | `verify_waveform_cam_thermometer.py` |
| Digital waveform refinement | float L1 | signed 5-bit SAD, shared/pair-conditioned taps | `run_compact_waveform_refinement.py`, `run_pair_conditioned_waveform.py` |
| Prefix temporal sketch | raw-prefix float L1 | 40/60/80-bit signed-5-bit dyadic mean/Haar templates; traffic-positive, accuracy-negative | `run_causal_temporal_sketch.py` |
| Channel attention | float weights | Per-CAM-row conductance scaling | `se_attention.py` |

---

## Quick Start

```bash
# Activate environment (requires numpy, scipy, h5py, scikit-learn)
conda activate spint   # or: source .venv/bin/activate

# Run unit tests
python -m pytest Spatial/tests/ -v

# Spatial-only upper bound on HJ
python -m Spatial.experiments.run_spatial_upper_bound --dataset hj --duration 60

# Unsupervised two-stage
python -m Spatial.experiments.run_two_stage_unsupervised --duration 60

# Active DAC experiment: per-unit 5-bit spatial search + directory Pareto
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 95 --coverage 0.95 \
  --directory-neighbors 0 7 \
  --directory-coverages 0.95 0.99 1.0

# Spatial WTA classifier evidence (G1–G6)
python -m Spatial.experiments.run_spatial_wta_pilot
python -m Spatial.experiments.run_spatial_attribution
python -m Spatial.experiments.run_spatial_wta_corpus \
  --score-mode stability --rank-profile top3_x4 --no-theta

# Nested inner-box / outer-L1 go/no-go
python -m Spatial.experiments.run_dual_range_pipeline \
  --family all --duration 60 --bits 5 --k 7 \
  --inner-coverages 0.50 0.75 0.90 0.95 \
  --outer-percentile 99.9

# Disjoint-calibration per-unit safe-exit go/no-go
python -m Spatial.experiments.run_certified_dual_range_pipeline \
  --family all --duration 60 --bits 5 \
  --inner-coverages 0.50 0.75 0.90 0.95 \
  --precision-targets 0.95 0.99 --min-supports 5 10 20

# Frozen-candidate compact digital waveform refinement
python -m Spatial.experiments.run_compact_waveform_refinement \
  --family all --duration 60 \
  --bits 4 5 --taps 8 16 24 --percentiles 95 99.9

# Deterministic 5-bit top-M and calibration-confusion pointer retrieval
python -m Spatial.experiments.run_indexed_confusion_candidate_search \
  --family all --duration 60 --bits 5 \
  --ranking-modes raw_l1 radius_normalized_l1 \
  --top-ms 1 2 4 8 --graph-coverages 0.95 0.99 1.0 \
  --top-anchors 1 2

# Block-causal graph-state update ceiling
python -m Spatial.experiments.run_online_confusion_graph \
  --family all --duration 60 --bits 5 \
  --graph-coverages 0.95 0.99 1.0 --top-anchors 1 2 \
  --block-seconds 1 5 \
  --update-sources oracle_delayed_label self_predicted_label \
  --histories accumulate rolling

# Pair-conditioned waveform taps
python -m Spatial.experiments.run_pair_conditioned_waveform \
  --family all --duration 60 --bits 5 \
  --taps 4 8 12 16 --percentiles 95 99.9

# Bounded P=1..4 5-bit prototype rows (completed negative static upper bound)
python -m Spatial.experiments.run_adaptive_multiprototype_5bit \
  --family all --duration 60 --bits 5 \
  --prototypes 1 2 3 4 \
  --modes temporal farthest_l1 \
  --radius-percentiles 95 99 99.9 \
  --directory-coverages 0.95 0.99 1.0

# Frozen one-row/unit masked-L1 development pilot (completed; gate fails)
python -m Spatial.experiments.run_adaptive_masked_5bit \
  --pilot --duration 60 --quiet

# Frozen one-row/unit weighted-L1 development pilot (completed; gate fails)
python -m Spatial.experiments.run_adaptive_weighted_5bit \
  --pilot --duration 60 --quiet

# Frozen dyadic-prefix 5-bit temporal pilot (completed; gate fails)
python -m Spatial.experiments.run_causal_temporal_sketch --pilot

# Multi-scale FPN experiment
python -m Spatial.experiments.run_multiscale_fpn --duration 60 --sweep

# SE-Net attention experiment
python -m Spatial.experiments.run_se_attention --duration 60

# Combined multi-scale + SE-Net
python -m Spatial.experiments.run_combined_mscale_se --duration 60
```

---

## Key Results Summary

### HJ 16ch (scene 11, 10 units, 60s)

| Method | Oracle | SOM | Note |
|--------|--------|-----|------|
| main_channel | 0.811 | 0.804 | trivial baseline |
| Akhoundi Eq.3-5 | 0.763 | 0.626 | degenerate on 1D |
| **soft_loc** | **0.916** | **0.833** | best spatial-only |
| soft_loc K=5 | 0.919 | **0.912** | best single-K for SOM |
| two-stage oracle (COM∧P2P→waveform) | **0.968** | — | GT centroids |
| unsupervised two-stage | — | ~0.81 | no GT labels |
| **per-cluster attention + soft_loc** | **0.966** | **0.962** | oracle attention |
| **SOM-discovered per-cluster** | 0.922 | **0.913** | unsupervised attention |

### MEArec 32ch (SNR=12, 10 units, 60s)

All spatial methods near ceiling (~0.963 oracle, ~0.837 SOM). No significant differentiation between variants.

### 2D Grid 1024ch (SNR=12, 15 neurons)

| Method | Accuracy |
|--------|----------|
| CC baseline | 81.2% |
| Akhoundi spatial SOM | **92.1%** (+10.9pp) |

This verifies the legacy spatial pipeline on a 1024-channel geometry; it is
not yet a high-local-unit-density validation of the new 5-bit two-level memory
pipeline. That experiment remains open.

### Adaptive 5-bit candidate search (GT-event/GT-row isolation)

| Family | Operating point | Aggregation | Candidate recall | Mean candidates | Waveform-read reduction |
|---|---|---|---:|---:|---:|
| HJ (12 scenes) | per-unit L1, 95% | recording-unweighted | 0.935 | 5.37 | 3.04× |
| HJ (12 scenes) | per-unit L1, 99.9% | recording-unweighted | 0.993 | 13.14 | 1.25× |
| MEArec (20 files) | per-unit L1, 95% | recording-unweighted | 0.947 | 5.52 | 2.62× |
| MEArec (20 files) | per-unit L1, 99.9% | recording-unweighted | 0.995 | 11.91 | 1.25× |
| HJ (12 scenes) | nested box/L1, inner 75% | event-weighted | 0.956 | 5.07 | 4.14× |
| MEArec (20 files) | nested box/L1, inner 95% | event-weighted | 0.987 | 12.65 | 1.33× |
| HJ (12 scenes) | certified, inner 75%, target 95% | event-weighted | 0.989 | 13.17 | 1.38× |
| MEArec (20 files) | certified, inner 50%, target 95% | event-weighted | 0.992 | 12.38 | 1.36× |
| HJ (12 scenes) | indexed graph, normalized, cov95/L1 | event-weighted | 0.932 | 3.61 | 5.31× |
| MEArec (20 files) | indexed graph, normalized, cov95/L1 | event-weighted | 0.938 | 4.29 | 4.08× |
| HJ (12 scenes) | online graph, delayed GT upper bound | event-weighted | 0.947 | 4.02 | 4.72× |
| MEArec (20 files) | online graph, delayed GT upper bound | event-weighted | 0.934 | 4.10 | 4.24× |
| HJ (12 scenes) | multi-row, best recall at ≥4× | event-weighted | 0.898 | 4.54 | 4.07× |
| MEArec (20 files) | multi-row, maximum reduction | event-weighted | 0.942 | 5.53 | 3.07× |
| HJ (2 development recordings) | masked stability/M3, p95 | event-weighted | 0.953 | 1.75 | 6.04× |
| MEArec (2 development recordings) | masked stability/M3, p95 | event-weighted | 0.961 | 7.29 | 2.29× |
| HJ (2 development recordings) | weighted stability/top3_x4, p95 | event-weighted | 0.908 | 1.69 | 7.90× |
| MEArec (2 development recordings) | weighted stability/top3_x4, p95 | event-weighted | 0.955 | 6.47 | 2.59× |

These are memory-search isolation results, not end-to-end detected-event
sorting accuracy. The aggregation estimator is explicit per row.
Read-reduction factors use event-weighted total reads throughout. Static
ranges, nested/certified exits, static indexed adjacency, graph-only online
updates, and bounded multi-prototype rows all fail the joint target of ≥99%
candidate recall and ≥4× waveform-read reduction. The masked and weighted rows
are
development-only results: the HJ movement is positive, but the same
masked configuration worsens MEArec traffic, the weighted configuration loses
slight MEArec recall with insufficient traffic improvement, and both formal
pilot gates fail.

### Compact digital waveform refinement

| Family | Spatial source | Representation | Accuracy | Delta vs float teacher | Template payload |
|---|---|---|---:|---:|---:|
| HJ | L1 p95 | Fisher-48×5 SAD | 0.840 | −0.52pp | 240 bits/unit |
| MEArec | L1 p95 | Fisher-48×5 SAD | 0.771 | −0.66pp | 240 bits/unit |
| HJ | L1 p99.9 | best shared taps: Fisher-40×5 | 0.745 | −1.59pp | 200 bits/unit |
| MEArec | L1 p99.9 | Fisher-48×5 SAD | 0.672 | −0.68pp | 240 bits/unit |
| HJ | L1 p99.9 | pair-16×5 tournament | 0.761 | +0.04pp | 18.8 kbit total/recording |
| MEArec | L1 p99.9 | pair-16×5 tournament | 0.669 | −0.97pp | 16.1 kbit total/recording |

The p95 tap result passes the Level-2 one-point criterion but is paired with
only 0.936/0.947 spatial candidate recall. At the p99.9 source, no shared-tap
count passes both families. Shared tap-index configuration storage is accounted
separately in the experiment output.
Pair conditioning passes the accuracy criterion at p99.9 but not traffic:
conservative bit-traffic reduction is only 1.35×/1.36× and storage grows with
the number of unit pairs.

---

## Dependencies

```
numpy, scipy, h5py, scikit-learn, pytest
```

Optional (data generation only): `MEArec, spikeinterface, probeinterface`
