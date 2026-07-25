# Spatial-Only Upper Bound Study

Status: HJ 60s results in; CortexLab pending re-run with the same script.
Scope: **spatial information only** — no Peak-FSDE, no temporal waveform PCA.

## Verdict

**Best scheme so far: `soft_loc`**
(geometry-weighted center of mass + spatial spread + sharpness).

On Hybrid Janelia 60s with GT spike times:

| Metric | `soft_loc` | Runner-up |
|--------|------------|-----------|
| GT-centroid oracle | **91.6%** | `main_channel` 81.1% |
| Unsupervised SpatialSOM | **83.3%** | `main_channel` 80.4% |
| Separability ratio (lower=better) | **0.34** | `main_channel` 0.60 |
| Recovered clusters | **10/10** | `main_channel` 8/10 |

Akhoundi Eq.3-5 (`akhoundi5`) is **not** competitive on this 1D probe
(oracle 76.3%, SOM 62.6%) — worse than using only the main-channel index.

---

## What `soft_loc` computes

For each spike, take the local neighborhood (default K=7 channels including home)
and the peak-to-peak (P2P) amplitude on each neighbor. Let \(A_i\) be P2P and
\((x_i, y_i)\) the electrode coordinates:

1. **Weights**: \(w_i = A_i / \sum_j A_j\)
2. **COM (centroid)**: \(\mathbf{c} = \sum_i w_i (x_i, y_i)\), then normalized to probe bbox → `[cx, cy]`
3. **Spread**: RMS distance of mass around \(\mathbf{c}\), normalized by probe diagonal
4. **Sharpness**: \(\max_i w_i\) (how concentrated the footprint is)

Output 4D vector: `[cx_norm, cy_norm, spread, sharpness]` ∈ [0, 1].

Implementation: [`algorithms/spatial_footprint.py`](../algorithms/spatial_footprint.py) → `soft_localization_features`.

### Why this wins on 1D probes

- Linear probes have essentially one informative spatial axis (depth along the shank).
- Akhoundi's **direction angle** collapses to \(\pm\pi/2\); **x_est** is discrete. Those dims add noise, not signal.
- **Main channel** only gives integer depth bins. Two units on the same channel (common on HJ) are indistinguishable.
- **COM** gives **sub-channel** depth: amplitude leaking to neighbors continuously shifts the centroid between electrodes.
- **Spread / sharpness** describe footprint shape (on-axis vs off-axis / broad vs focused), which helps when COM alone overlaps.

So `soft_loc` is a 1D-appropriate compression of the same P2P neighborhood that
`footprint_p2p` keeps as a raw vector — but the summary is lower-dimensional and
better conditioned for both oracle nearest-centroid and SOM.

---

## Full HJ 60s table

Protocol:

```text
raw → bandpass(300–6000 Hz) → GT spike times
    → local P2P (K=7, ±15 samples)
    → feature method
    → (A) GT-centroid oracle, chronological 50/50 split
       (B) SpatialSOM (grid=32, beta=8, alpha=16)
```

Command:

```bash
python -m Spatial.experiments.run_spatial_upper_bound \
  --dataset hj --duration 60 --use-gt \
  --output Spatial/output/spatial_upper_bound_hj60s.csv
```

| Method | Dim | Sep. Ratio | Oracle | SOM | Clusters |
|--------|-----|------------|--------|-----|----------|
| `main_channel` | 1 | 0.596 | 81.1% | 80.4% | 8/10 |
| `akhoundi5` | 5 | 0.922 | 76.3% | 62.6% | 9/10 |
| **`soft_loc`** | **4** | **0.336** | **91.6%** | **83.3%** | **10/10** |
| `footprint_p2p` | 7 | 0.836 | 80.6% | 64.2% | 11/10 |
| `footprint_pca` | 4 | 0.776 | 78.7% | 63.8% | 11/10 |

### Reading the columns

- **Oracle**: supervised ceiling for that feature space (fit unit means on first half of spikes, assign second half by nearest centroid). Measures *how much identity is in the features*, independent of clustering algorithm quality.
- **SOM**: same unsupervised pipeline used elsewhere in this folder — closer to an online sorter.
- **Sep. ratio**: mean within-unit scatter / mean nearest-other-unit centroid distance. Values ≪ 1 indicate easy separation; `soft_loc` is the only method well below 0.5 here.

### Method notes

| Method | Role |
|--------|------|
| `main_channel` | Strong 1D baseline; already ~81% oracle because many HJ units have unique home channels. |
| `akhoundi5` | Hardware-oriented paper features; angle/ratio largely degenerate on linear geometry. |
| `soft_loc` | Best spatial-only representation so far. |
| `footprint_p2p` | Full local amplitude vector; more dims, noisier for SOM; does not beat COM summary. |
| `footprint_pca` | PCA of footprint; similar to raw footprint, still behind `soft_loc`. |

---

## Implications

1. On **1D** data, do **not** use Akhoundi angle/ratio as the spatial upper-bound reference.
2. Prefer **`soft_loc`** (or equivalent COM + shape stats) when probing spatial information limits without waveform morphology features.
3. There is still headroom above SOM (`91.6%` oracle vs `83.3%` SOM): clustering / online assignment can improve without changing the feature definition.
4. Remaining errors under the oracle (~8%) are likely **intrinsically hard pairs** (same channel + overlapping COM), where spatial-only features are insufficient — waveform features would be needed next, outside this study's scope.

Detailed error-pair analysis and Peak-FSDE rescue numbers:
[`soft_loc_error_analysis.md`](soft_loc_error_analysis.md).

---

## Reproduce / extend

```bash
# HJ
python -m Spatial.experiments.run_spatial_upper_bound --dataset hj --duration 60 --use-gt

# CortexLab NP slice
python -m Spatial.experiments.run_spatial_upper_bound --dataset cortexlab --duration 60 --use-gt

# Unit tests for soft_loc / footprint
python -m pytest Spatial/tests/test_spatial_footprint.py -v
```
