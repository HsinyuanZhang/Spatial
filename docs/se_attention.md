# SE-Net Channel Attention — Experiment Results

**Date**: 2026-07-25
**Datasets**: HJ 16ch (scenes 11 & 12, 10 units each), MEArec 32ch (10 units)
**Duration**: 60s, GT spike times, chronological 50/50 train/test, K=7

## Hypothesis

Adaptive per-channel weighting (SE-Net style) can improve spatial feature
discriminability by emphasizing channels that best distinguish units.

## Attention Variants Tested

| Variant | Supervision | Mechanism |
|---------|-------------|-----------|
| Fisher global | Oracle (GT labels) | inter-cluster var / intra-cluster var per channel |
| Variance | Unsupervised | per-channel variance across all spikes |
| Inverse-CV | Unsupervised | 1/(1+CV) — reliable channels get higher weight |
| Per-cluster profile | Oracle (GT labels) | per-unit mean footprint as weight |
| SOM-Fisher | Unsupervised | Fisher weights from SOM-discovered clusters |
| Fisher + temperature | Oracle | softmax temperature on Fisher weights |

## Results — HJ Scene 11

### Soft_loc Feature Variants

| Method | sep_ratio | oracle | SOM | clusters |
|--------|-----------|--------|-----|----------|
| baseline_soft_loc | 0.336 | 0.916 | 0.833 | 10/10 |
| fisher_soft_loc | 0.439 | 0.882 | 0.763 | 11/10 |
| variance_soft_loc | 0.425 | 0.874 | 0.781 | 9/10 |
| inv_cv_soft_loc | 0.337 | 0.909 | 0.825 | 10/10 |
| **percluster_soft_loc** | **0.229** | **0.966** | **0.962** | **10/10** |
| som_fisher_soft_loc | 0.424 | 0.848 | 0.737 | 11/10 |

### Fisher Temperature Sweep

| Temperature | sep_ratio | oracle | SOM |
|-------------|-----------|--------|-----|
| T=0.1 (sharp) | 0.629 | 0.778 | 0.642 |
| T=0.5 | 0.516 | 0.849 | 0.764 |
| T=1.0 | 0.439 | 0.882 | 0.763 |
| T=2.0 | 0.396 | 0.902 | 0.800 |
| T=5.0 (soft) | 0.350 | 0.913 | 0.830 |

**Finding**: Lower temperature (sharper attention) hurts — it over-emphasizes
a few channels and destroys the geometric COM information. T→∞ recovers
baseline. Fisher attention is too aggressive for soft_loc.

### Two-Stage Gate (stage-1 only, oracle)

| Gate P2P weighting | accuracy | fallback | mean_candidates |
|--------------------|----------|----------|-----------------|
| none (baseline) | 0.915 | 0.027 | 1.21 |
| fisher | 0.917 | 0.026 | 1.24 |
| **variance** | **0.940** | **0.024** | 1.24 |
| som_fisher | 0.929 | 0.021 | 1.24 |

**Finding**: Variance attention improves the two-stage gate by +2.5pp
(unsupervised!). It sharpens the P2P representation for candidate filtering.

## Results — HJ Scene 12 (Cross-Scene Validation)

| Method | sep_ratio | oracle | SOM |
|--------|-----------|--------|-----|
| baseline_soft_loc | 0.285 | 0.925 | 0.850 |
| **percluster_soft_loc** | **0.198** | **0.967** | **0.880** |
| fisher_T5.0 | 0.299 | 0.922 | 0.847 |
| gate_variance | — | 0.950 | — |

**Finding**: Per-cluster attention generalizes across HJ scenes:
+4.2pp oracle, +3.0pp SOM on scene 12.

## Results — MEArec 32ch SNR12

| Method | sep_ratio | oracle | SOM |
|--------|-----------|--------|-----|
| baseline_soft_loc | 0.263 | 0.963 | 0.837 |
| percluster_soft_loc | 0.215 | 0.964 | 0.837 |
| variance_soft_loc | 0.251 | 0.963 | 0.837 |

**Finding**: MEArec is near ceiling. Per-cluster attention improves
separability (0.215 vs 0.263) but SOM accuracy is unchanged — the
bottleneck is not feature quality but SOM capacity on this dataset.

## Key Insight: Why Per-Cluster Attention Works

Per-cluster profile attention weights each channel by the unit's own mean
footprint. This has a geometric interpretation:

- For a neuron directly above channel 7: channels 6,7,8 get high weight
  → COM stays centered, spread tightens
- For a neuron between channels 3 and 4: channels 3,4 get high weight
  → COM resolves the ambiguity between the two positions

Effectively, it **adapts the receptive field to each neuron's location**,
suppressing irrelevant channels that add noise to the COM calculation.
This is why sep_ratio drops from 0.336 to 0.229 (32% tighter clusters).

## Conclusions

1. **Per-cluster profile attention is the most effective variant**:
   +5pp oracle, +13pp SOM on HJ scene 11. Consistent across scenes.
2. **Fisher/variance attention on soft_loc hurts** — they distort the
   geometric COM that soft_loc relies on.
3. **Variance attention helps the two-stage gate** (+2.5pp, unsupervised).
4. **Temperature control matters**: sharp attention (low T) destroys
   geometric information; soft attention (high T) recovers baseline.
5. The practical unsupervised version (SOM-discovered per-cluster) is
   tested in the combined experiment → see `combined_mscale_se.md`.

## Output Files

- `Spatial/output/se_attention/se_attention_hj_60s.csv`
- `Spatial/output/se_attention/se_attention_mearec_60s.csv`
