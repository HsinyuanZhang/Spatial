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
│   ├── cim_thermometer.py               # Bipolar thermometer CAM encoding (hardware approx)
│   ├── multiscale_footprint.py          # Multi-K (FPN-style) footprint extraction + fusion
│   └── se_attention.py                  # SE-Net channel attention (Fisher / variance / per-cluster)
│
├── experiments/
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
├── docs/                                # Per-study write-ups with results tables
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
| MEArec v1 | 32ch linear | 32 | 60s @ 32kHz | 10 | 2–12 | `../synthetic_spike_dataset/generated/geoosort_npz_v1/` |
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
Stage 1: COM∧P2P gate (per-unit thresholds) → candidate units
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

---

## CiM/CAM Hardware Approximations

Verified that the float pipeline can be replaced by hardware-friendly encodings without losing the oracle ceiling:

| Component | Float | CiM Approximation | Verified In |
|-----------|-------|--------------------|-------------|
| P2P distance | L2 | L1 on n-bit codes | `verify_p2p_cim_approx.py` |
| P2P encoding | float | per-spike-max / central-ratio quantization | `run_lowbit_p2p.py` |
| Waveform distance | L2 | Hamming on bipolar thermometer codes | `verify_waveform_cam_thermometer.py` |
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

---

## Dependencies

```
numpy, scipy, h5py, scikit-learn, pytest
```

Optional (data generation only): `MEArec, spikeinterface, probeinterface`
