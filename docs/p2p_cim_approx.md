# P2P L2 ↔ CiM/CAM Approximations (Stage-1)

Question: can the **normalized P2P L2** gate be replaced by hardware-friendly
CiM/CAM distances without losing the two-stage oracle ceiling?

Data: HJ 60s, GT spikes; COM gate fixed (L2); only the P2P metric changes.
Stage-2: full waveform. Script: `Spatial/experiments/verify_p2p_cim_approx.py`.

## Brainstorm map

| Approx | Hardware story | Relation to L2 |
|--------|----------------|----------------|
| **L1 / Manhattan** | Analog CAM current sum of \|Δ\| | Same topology as many FeFET/ReRAM AMs |
| **Cosine / unit-L2** | COSIME-style associative memory | Exact monotone to L2 **after** L2-normalize |
| **Neg dot (MAC)** | Classic CiM crossbar \(x^\top W\) | Needs similar vector norms; else not ≅ L2 |
| **Hamming on n-bit codes** | Binary / ternary CAM match-line | Classic CAM; coarsens continuous L2 |
| **L2 on n-bit codes** | Multi-bit FeReX-style Euclidean AM | Direct quantized L2 |

Our footprints are **max-normalized** (÷max), not L2-normalized → raw cosine ≠ L2 ranking, but still often close for sparse positive vectors.

## HJ 60s verification

| Metric | Spearman vs L2 | Cand Jaccard | Two-stage oracle | Δ vs L2 |
|--------|----------------|--------------|------------------|---------|
| **l2** (baseline) | 1.000 | 1.000 | **96.8%** | — |
| **l1** | 0.990 | 0.990 | **96.8%** | +0.1 pp |
| **cosine** | 0.974 | 0.976 | 96.7% | −0.1 pp |
| **l2_on_l2norm** | 0.974 | 0.976 | 96.6% | −0.1 pp |
| **l2_4bit** | 0.998 | 0.994 | 96.6% | −0.1 pp |
| **hamming_3bit** | 0.607 | 0.885 | 96.3% | −0.4 pp |
| **hamming_4bit** | 0.584 | 0.882 | 96.3% | −0.5 pp |
| **hamming_2bit** | 0.241 | 0.864 | 93.8% | −2.9 pp |
| **neg_dot** | −0.21 | 0.894 | 96.3% | −0.4 pp |
| hamming_5bit | 0.419 | 0.722 | 90.7% | −6.1 pp |

## Conclusions

1. **Best CiM drop-ins for this gate: L1 or cosine / unit-norm L2** — essentially lossless on HJ.
2. **4-bit Euclidean on codes** is also nearly lossless → matches multi-bit AM literature (FeReX).
3. **Hamming CAM** works from ~3–4 bits (≤0.5 pp drop); 2-bit is the cliff.
4. **Bare MAC (neg_dot)** does **not** track L2 ranks (Spearman negative!) but COM∧gate still keeps oracle high — do **not** treat it as an L2 substitute; the COM gate is carrying it.
5. Practical hardware recipe for stage-1 P2P: **max-normalize → 4-bit → L1 or L2 in CAM**; keep COM as a cheap 2D digital/analog compare.

CSV: `Spatial/output/p2p_cim_approx_hj60s.csv`
