# Multi-Scale FPN Footprint Fusion — Experiment Results

**Date**: 2026-07-25
**Datasets**: HJ 16ch (scene 11, 10 units, 2788 spikes), MEArec 32ch (10 units, 3923 spikes)
**Duration**: 60s, GT spike times, chronological 50/50 train/test

## Hypothesis

Extracting soft_loc features at multiple neighborhood sizes K and fusing them
(FPN-style) should improve discrimination of near vs far neurons:
- Small K → sharp/local footprint (near neurons)
- Large K → diffuse footprint (far neurons)

## Results — HJ Scene 11

### Single-K Baselines

| Method | dim | sep_ratio | oracle | SOM | clusters |
|--------|-----|-----------|--------|-----|----------|
| soft_loc_K3 | 4 | 0.438 | 0.928 | 0.852 | 11/10 |
| **soft_loc_K5** | 4 | **0.334** | 0.919 | **0.912** | 10/10 |
| soft_loc_K7 (default) | 4 | 0.336 | 0.916 | 0.833 | 10/10 |
| soft_loc_K9 | 4 | 0.351 | 0.909 | 0.836 | 10/10 |
| soft_loc_K13 | 4 | 0.401 | **0.939** | 0.792 | 11/10 |
| soft_loc_K16 | 4 | 0.401 | 0.930 | 0.712 | 15/10 |

**Finding**: K=5 gives best SOM (0.912), K=13 gives best oracle (0.939).
The default K=7 is suboptimal for both. Smaller K → better SOM clustering
(lower dimensionality, less noise). Larger K → better oracle (more spatial
information for centroid matching).

### Multi-Scale Fusions

| Method | dim | sep_ratio | oracle | SOM | clusters |
|--------|-----|-----------|--------|-----|----------|
| ms_concat (all K) | 24 | 0.390 | 0.934 | 0.844 | 11/10 |
| ms_mean | 4 | 0.355 | 0.923 | 0.831 | 12/10 |
| ms_pca4 | 4 | 0.364 | 0.931 | 0.843 | 12/10 |
| ms_pca8 | 8 | 0.380 | 0.935 | 0.851 | 11/10 |
| K3+K7+K16 concat | 12 | 0.415 | **0.940** | 0.850 | 12/10 |
| K5+K7+K9 concat | 12 | 0.346 | 0.925 | **0.914** | 11/10 |
| K5+K7+K13 concat | 12 | 0.350 | 0.928 | 0.906 | 10/10 |

**Finding**: Multi-scale concat does NOT beat the best single-K for SOM.
K5+K7+K9 concat (0.914) matches K=5 single (0.912). The 24D full concat
hurts SOM (curse of dimensionality). Sub-band K3+K7+K16 gives best oracle
(0.940) but SOM only 0.850.

### SOM Hyperparameter Sweep (ms_concat)

| grid | beta | accuracy | clusters |
|------|------|----------|----------|
| 32 | 16 | **0.845** | 12 |
| 32 | 8 | 0.844 | 11 |
| 32 | 4 | 0.843 | 11 |
| 16 | 4 | 0.783 | 8 |
| 48 | 4 | 0.718 | 13 |
| 64 | 4 | 0.673 | 15 |

**Finding**: grid=32 is optimal. Larger grids over-split. Beta has minimal effect.

## Results — MEArec 32ch SNR12

| Method | dim | sep_ratio | oracle | SOM |
|--------|-----|-----------|--------|-----|
| soft_loc_K3 | 4 | 0.269 | 0.963 | **0.889** |
| soft_loc_K7 | 4 | 0.263 | 0.963 | 0.837 |
| soft_loc_K9 | 4 | 0.241 | **0.964** | 0.838 |
| ms_concat | 24 | 0.265 | 0.963 | 0.837 |
| ms_mean | 4 | 0.251 | 0.963 | 0.838 |

**Finding**: All methods near ceiling (~0.963 oracle). K=3 gives best SOM
(0.889). Multi-scale fusion provides no benefit — the baseline is already
saturated.

## Conclusions

1. **Multi-scale FPN alone is not a significant improvement** over single-K
   soft_loc on either dataset.
2. **K=5 is better than K=7 for SOM** on HJ 16ch — the default K should be
   reconsidered.
3. **High-dimensional concat hurts SOM** — the SOM's fixed grid struggles
   with >12D features.
4. **Sub-band fusion (K3+K7+K16)** improves oracle ceiling (+2.4pp) but
   doesn't translate to unsupervised SOM gains.
5. On MEArec 32ch, the problem is already near-saturated; no spatial feature
   enhancement helps.

## Output Files

- `Spatial/output/multiscale_fpn/multiscale_fpn_hj_60s.csv`
- `Spatial/output/multiscale_fpn/multiscale_fpn_mearec_60s.csv`
- `Spatial/output/multiscale_fpn/som_sweep_hj_60s.csv`
