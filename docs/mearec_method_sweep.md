# MEArec v1 method sweep (same protocol as HJ)

Date: 2026-07-24  
Data: `synthetic_spike_dataset/generated/geoosort_npz_v1/` — 20 NPZs  
(32ch Neuronexus, 120s @ 30 kHz; units∈{10,20} × SNR∈{2,3,5,8,12} × 2 seeds)  
Protocol: first **60s**, chronological 50/50 GT-centroid oracle.

```bash
python -m Spatial.experiments.run_mearec_method_sweep --duration 60
```

CSVs: `Spatial/output/mearec_method_sweep/`

## Spatial-only (overall mean ± std)

| Method | Oracle | SOM |
|--------|--------|-----|
| `main_channel` | 0.721 ± 0.105 | 0.602 ± 0.117 |
| `akhoundi5` | 0.747 ± 0.116 | 0.683 ± 0.111 |
| **`soft_loc`** | **0.826 ± 0.113** | **0.729 ± 0.089** |
| `footprint_p2p` | 0.644 ± 0.186 | 0.484 ± 0.199 |
| `footprint_pca` | 0.618 ± 0.182 | 0.498 ± 0.222 |

Oracle rises with SNR (soft_loc: 0.69 @ SNR2 → 0.89 @ SNR12). On this **linear 32ch** probe Akhoundi is closer to soft_loc than on HJ, but soft_loc still wins.

## Two-stage (P2P=L2)

| Method | Oracle |
|--------|--------|
| **`com_and_p2p`** | **0.856 ± 0.102** |
| `fsde_only` | 0.412 ± 0.154 |
| `waveform_only` | 0.679 ± 0.135 |
| `concat_com_p2p_waveform` | 0.834 ± 0.089 |
| `two_stage_fsde` | 0.731 ± 0.202 |
| `two_stage_waveform` | 0.837 ± 0.114 |

P2P **L1** ≈ L2 (within ~0.5 pp).

### By designed unit count (L2)

| units | com_and_p2p | two_stage_fsde | two_stage_waveform |
|------:|------------:|---------------:|-------------------:|
| 10 | 0.921 ± 0.056 | 0.894 ± 0.095 | 0.919 ± 0.062 |
| 20 | 0.791 ± 0.097 | 0.568 ± 0.136 | 0.755 ± 0.091 |

## vs Hybrid Janelia 16ch (same 60s protocol)

| Finding | HJ 16ch | MEArec 32ch |
|---------|---------|-------------|
| Best spatial-only | soft_loc 0.84 | soft_loc 0.83 |
| Best two-stage style | **two_stage_waveform 0.90** | **com_and_p2p 0.86** (waveform stage-2 0.84) |
| FSDE alone | ~0.45 | ~0.41 |
| L1 ≈ L2 for P2P gate | yes | yes |
| Hard regime | denser unit reps | **20-unit** grids; FSDE stage-2 collapses |

**Takeaway:** spatial COM∧P2P / soft_loc transfer well to MEArec. On denser synthetic probes, stage-2 Peak-FSDE is fragile; full waveform helps but does not beat a strong spatial gate alone when units heavily overlap in space.
