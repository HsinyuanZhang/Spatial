# Spatial Clustering — Akhoundi et al. 2025 Reproduction

Reproduction and validation of the spatial feature extraction + modified SOM clustering method from:

> **Akhoundi et al., "A Scalable 1024-Channel Ultra-Low-Power Spike Sorting Chip With Event-Driven Detection and Spatial Clustering", IEEE JSSC, 2025.**

Validated on three datasets covering 2D high-density and 1D linear probes.

---

## Directory Structure

```
Spatial/
├── README.md                        # This file
├── .gitignore
├── papers/                          # Reference paper (PDF)
│
├── reference_code/                  # TU Delft official MATLAB implementation
│   ├── Classification/              #   SOM clustering (Classification.m)
│   ├── FeatureExtraction/           #   Spatial features (FeatureExrtaction.m)
│   ├── SpikeDetection/              #   NEO + VC-SPD detection
│   ├── WiredOR/                     #   Wired-OR compression
│   └── MEArec/                      #   32x32 2D grid probe configs
│
├── configs/
│   └── experiment.yaml              # SOM + detection parameters (per-dataset optimal)
│
├── data/
│   ├── loader.py                    # Unified dataset loader (HJ + MEArec + NP)
│   ├── generate_2d_grid.py          # 32x32 staggered grid generator (current)
│   └── generate_mearec.py           # Legacy 32ch 1D generator (deprecated)
│
├── algorithms/
│   ├── spatial_features.py          # Our impl: Eq.3-5 direction angle + distance ratio
│   ├── spatial_footprint.py         # Soft-loc / footprint / PCA (spatial-only ceiling)
│   ├── som_clustering.py            # Our impl: modified SOM (BMU + pruning + merge)
│   ├── detection.py                 # Our impl: NEO detection + window extraction
│   └── reference_port.py            # Python port of MATLAB reference (all 5 modules)
│
├── experiments/
│   ├── run_spatial_sorting.py       # Main pipeline: detect -> features -> SOM -> eval
│   ├── run_spatial_upper_bound.py   # Spatial-only upper bound (HJ / CortexLab)
│   ├── validate_2d_grid.py          # 2D grid validation script
│   └── sweep_som_params.py          # Hyperparameter grid search
│
├── evaluation/
│   └── metrics.py                   # Clustering accuracy (Eq.10), ARI, P/R/F1
│
├── tests/
│   ├── test_spatial_features.py     # Spatial feature unit tests (6 tests)
│   ├── test_spatial_footprint.py    # Soft-loc / footprint unit tests
│   ├── test_som_clustering.py       # SOM unit tests (5 tests)
│   └── test_reference_port.py       # MATLAB port tests (18 tests)
│
└── output/                          # Generated datasets + results (gitignored)
    └── mearec_2d_grid_v2/           # Current 2D grid dataset (9.2 GB)
```

---

## Datasets

### 1. 2D Grid (Synthetic, Generated) — Primary Validation

| Property | Value |
|----------|-------|
| Grid | 32x32 staggered (honeycomb), 36um pitch, 18um stagger |
| Channels | 1024 |
| Duration | 60s @ 20kHz per recording |
| Neurons | 15 (min spacing 250um) |
| SNR levels | 3, 5, 8, 12 |
| Templates | Synthetic Gaussian EAP, sigma=30um, amp 300-800uV |
| Size | 9.2 GB (4 recordings) |
| Location | `output/mearec_2d_grid_v2/` |

### 2. Hybrid Janelia (Semi-Synthetic)

| Property | Value |
|----------|-------|
| Probe | 16ch linear, 20um spacing |
| Duration | 600s @ 30kHz |
| Units | 10 valid (SNR 8-16) |
| Known issue | Units 5/21 share ch14, units 43/63 share ch7 |
| Location | `../new_datasets/hybridjanelia/static16c_600s_11_filtered_gt.npz` |

### 3. CortexLab Neuropixels (Experimental)

| Property | Value |
|----------|-------|
| Probe | NP1, 128ch (64x2), 20um spacing |
| Duration | 228.4s @ 30kHz (10% slice) |
| Units | 10 (KS4 "good" units); ~6 unique spatial locations |
| Note | Paper describes NP as "inadequate for high-density evaluation" |
| Location | `../output/cortexlab_10pct/slices/dataset_1_20141202_228s_ksgt_int16.npz` |

---

## Algorithm Summary

### Pipeline Overview

```
Raw Data -> Bandpass -> Spike Detection (NEO) -> Spatial Features -> SOM Clustering
```

### Spatial Feature Extraction (Eq. 3-5)

For each spike with central channel `c` and neighbor amplitudes `A_i`:

- **Direction angle** (Eq. 3): `angle = atan2(sum(A_i * y_i), sum(A_i * x_i))`
- **Distance ratio** (Eq. 4): `ratio = max(A_i) / A_c`
- **Position estimate** (Eq. 5): `(x_c, y_c) + delta`

Output: 5D feature vector `[angle, ratio, x_est, y_est, peak_amp]`

### Modified SOM Clustering

- **BMU update**: `centroid_new = ((alpha-1)*centroid_old + feature) / alpha` (alpha=16)
- **Batch pruning**: remove elements with frequency < `N_spikes / (N_active * beta)`
- **Merge**: combine centroids closer than `feature_extent / grid_size * 2`
- Converge when active cluster count stabilizes

---

## Quick Start

```bash
cd /data/xinyuan_work/SNN_SpikeSorting
source .venv/bin/activate

# Run all unit tests
python -m pytest Spatial/tests/ -v

# Spatial-only upper bound on Hybrid Janelia (GT spikes, no FSDE/waveform PCA)
python -m Spatial.experiments.run_spatial_upper_bound --dataset hj --duration 60 --use-gt

# Same study on CortexLab NP slice
python -m Spatial.experiments.run_spatial_upper_bound --dataset cortexlab --duration 60 --use-gt

# Validate 2D grid dataset (SNR=12, 10s)
python -m Spatial.experiments.validate_2d_grid --snr 12 --duration 10

# Run spatial sorting on Hybrid Janelia
python -m Spatial.experiments.run_spatial_sorting --dataset hj --duration 60 --use-gt

# Generate new 2D grid dataset
python -m Spatial.data.generate_2d_grid --output-dir Spatial/output/mearec_2d_grid_v2
```

### Spatial-Only Upper Bound

`experiments/run_spatial_upper_bound.py` measures how much unit identity is
recoverable from **spatial information alone** on 1D probes (HJ, CortexLab).
No Peak-FSDE or temporal waveform PCA is used.

| Method | Features |
|--------|----------|
| `main_channel` | normalized central-channel index |
| `akhoundi5` | paper Eq.3-5 angle / ratio / position |
| `soft_loc` | geom-weighted COM + spread + sharpness (**best so far**) |
| `footprint_p2p` | local K-neighbor P2P amplitude vector |
| `footprint_pca` | PCA of the P2P footprint |

For each method the script reports (1) **GT-centroid oracle** accuracy
(chronological 50/50 train/test split) and (2) unsupervised **SpatialSOM** accuracy.
See [`docs/spatial_upper_bound.md`](docs/spatial_upper_bound.md) for the detailed write-up.

---

## Results

### 2D Grid (15 neurons, sigma=30um, amp=300-800uV)

Optimal params: grid=64, beta=4, alpha=16

| SNR | CC Baseline | Spatial SOM | Clusters | Sep. Ratio |
|-----|-------------|-------------|----------|------------|
| 3   | 11.4%       | 17.2%       | 25/15    | 30.06      |
| 5   | 44.2%       | 58.5%       | 19/15    | 3.06       |
| 8   | 72.0%       | **89.2%**   | 16/15    | 0.77       |
| 12  | 81.2%       | **92.1%**   | 15/15    | 0.61       |

### Hybrid Janelia — Akhoundi pipeline (16ch linear)

| Method | Accuracy | Note |
|--------|----------|------|
| CC Baseline | **79.9%** | Best method under Akhoundi features |
| Spatial SOM (akhoundi5) | 67.5% | Degenerate: angle collapses to +/-pi/2 |

### Hybrid Janelia — Spatial-only upper bound (60s, GT spikes)

Protocol: bandpass → GT spike times → local P2P amplitudes (K=7) → feature variants →
GT-centroid oracle + SpatialSOM (grid=32, beta=8). Raw table:
`output/spatial_upper_bound_hj60s.csv`.

| Method | Dim | Sep. Ratio | Oracle Acc | SOM Acc | Clusters |
|--------|-----|------------|------------|---------|----------|
| `main_channel` | 1 | 0.60 | 81.1% | 80.4% | 8/10 |
| `akhoundi5` | 5 | 0.92 | 76.3% | 62.6% | 9/10 |
| **`soft_loc`** | **4** | **0.34** | **91.6%** | **83.3%** | **10/10** |
| `footprint_p2p` | 7 | 0.84 | 80.6% | 64.2% | 11/10 |
| `footprint_pca` | 4 | 0.78 | 78.7% | 63.8% | 11/10 |

**Winner: `soft_loc`.** On this 1D probe it raises the spatial-only oracle ceiling by
~10.5 pp over main-channel and ~15.3 pp over Akhoundi Eq.3-5. Unsupervised SOM
also peaks here (83.3%) and recovers the correct cluster count (10/10).

### CortexLab NP (128ch linear)

| Method | Accuracy | Note |
|--------|----------|------|
| CC Baseline | **86.3%** | Theoretical ceiling = 86.5% |
| Spatial SOM | 79.0% | Collapsed to 1 cluster |

### Cross-Dataset Comparison

| Dataset | Probe Type | Channels | SOM Acc | CC Baseline | Spatial Value |
|---------|-----------|----------|---------|-------------|---------------|
| 2D Grid | 32x32 2D  | 1024     | 92.1%   | 81.2%       | **+10.9%**    |
| HJ (Akhoundi) | 16ch 1D | 16     | 67.5%   | 79.9%       | -12.4%        |
| HJ (`soft_loc`) | 16ch 1D | 16   | **83.3%** | 81.1% (main-ch oracle) | **+2.2% SOM / +10.5% oracle** |
| NP      | 64x2 1D   | 128      | 79.0%   | 86.3%       | -7.3%         |

### Key Findings

1. **Spatial features require 2D high-density probes (Akhoundi features)**: On 2D grids (36um pitch), Akhoundi features add +10.9% over main-channel baseline at SNR=12.
2. **Akhoundi features are degenerate on linear probes**: Direction angle collapses to +/-pi/2; on HJ they underperform even `main_channel` (oracle 76.3% vs 81.1%).
3. **1D spatial information is not empty — use soft localization**: `soft_loc` (COM + spread + sharpness) is currently the best spatial-only representation on HJ (oracle 91.6%, SOM 83.3%).
4. **Separability ratio is predictive**: When within-unit scatter / inter-unit distance < 1, clustering is strong (`soft_loc` sep=0.34). Akhoundi sep=0.92 correlates with its weaker accuracy.
5. **Raw footprint vectors are not automatically better**: `footprint_p2p` / `footprint_pca` match main-channel at best; the compact COM/shape summary generalizes better under SOM.
6. **Wired-OR is a hardware optimization, not an algorithm advantage**: In software, skipping Wired-OR and using direct thresholding yields 46.9% vs 21.5% (with Wired-OR) on the same data.
7. **NP dataset confirms paper's own assessment**: Paper states NP is "inadequate for assessing the accuracy of a spike sorter designed for high-density recordings" — our Akhoundi results agree.

---

## Reference Code Port

`algorithms/reference_port.py` is a faithful Python port of the TU Delft MATLAB implementation:

| Module | MATLAB Source | Python Function |
|--------|--------------|-----------------|
| Wired-OR | `WiredOR.m` | `wired_or_fast()` |
| Hexagonal neighbors | `FindNeighborElec.m` | `find_neighbor_elec()`, `build_neighbor_table()` |
| Feature extraction | `FeatureExrtaction.m` | `extract_features_reference()` |
| Spike detection | `SpikeDetection.m` | `detect_spikes_reference()` |
| SOM clustering | `Classification.m` | `ReferenceSOM` class |

Key differences from our software implementation (`som_clustering.py`):
- Reference uses 5-bit integer features clamped to [-31, 31] (hardware constraint)
- Reference uses 5x5 local BMU search (hardware memory constraint)
- Reference includes cluster migration (centroid relocates between grid cells)
- Our version uses float features + global BMU search (software-optimized)

---

## Dependencies

```
numpy, scipy          # core computation
h5py                  # HDF5 dataset I/O
scikit-learn          # KMeans (comparison only)
pytest                # unit tests
```

For data generation only (not required for running experiments):
```
MEArec, spikeinterface, probeinterface  # optional, for NEURON-based templates
```

---

## Disk Space Management

| Item | Size | Status |
|------|------|--------|
| `output/mearec_2d_grid_v2/` | 9.2 GB | Current dataset |
| `output/mearec_generated/` | 660 MB | **Obsolete** (delete) |
| `output/mearec_2d_grid_test/` | 79 MB | **Obsolete** (delete) |

Cleanup commands:
```bash
rm -rf Spatial/output/mearec_generated Spatial/output/mearec_2d_grid_test
```
