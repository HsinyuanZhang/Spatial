# HJ 16ch cross-scene method sweep (60s GT oracle)

Date: 2026-07-23  
Protocol: first 60s, chronological 50/50 GT-centroid oracle (same as today’s studies).  
Scenes: all 12 ready `*16c_600s_*` NPZs (6 static + 6 drift).

```bash
python -m Spatial.experiments.run_hj16_method_sweep --duration 60
```

CSVs:

- `Spatial/output/hj16_method_sweep/spatial_upper_bound_hj16_60s.csv`
- `Spatial/output/hj16_method_sweep/two_stage_hj16_60s.csv`

## Spatial-only (mean ± std over 12 scenes)

| Method | Oracle | SOM |
|--------|--------|-----|
| `main_channel` | 0.646 ± 0.101 | 0.645 ± 0.103 |
| `akhoundi5` | 0.627 ± 0.070 | 0.535 ± 0.062 |
| **`soft_loc`** | **0.844 ± 0.040** | **0.722 ± 0.070** |
| `footprint_p2p` | 0.700 ± 0.075 | 0.634 ± 0.069 |
| `footprint_pca` | 0.660 ± 0.072 | 0.624 ± 0.060 |

`soft_loc` remains best spatial-only on every scene; hardest reps (`*21/22/31/32`, more SNR≥8 units) pull main_channel / Akhoundi down into the 0.55–0.63 range while soft_loc stays ~0.79–0.92.

## Two-stage (P2P = L2; mean ± std)

| Method | Oracle |
|--------|--------|
| `com_and_p2p` | 0.871 ± 0.034 |
| `fsde_only` | 0.445 ± 0.044 |
| `waveform_only` | 0.744 ± 0.056 |
| `concat_com_p2p_fsde` | 0.867 ± 0.037 |
| `concat_com_p2p_waveform` | 0.884 ± 0.037 |
| `two_stage_fsde` | 0.810 ± 0.076 |
| **`two_stage_waveform`** | **0.903 ± 0.038** |

P2P **L1** matches L2 within ~0.2 pp on the two-stage targets.

### Notes

1. On easy static reps 11/12, `two_stage_fsde` ≈ 0.94 and `two_stage_waveform` ≈ 0.97 (today’s numbers).
2. On denser unit sets (`*21–32*`), FSDE stage-2 drops more (`~0.74–0.78`) while full-waveform stage-2 stays ~0.85–0.91.
3. Flat concat of COM+P2P+waveform (0.884) is slightly below two-stage waveform (0.903) on average — sequential gating still helps.
4. Drift scenes are a bit harder than static, but the method ranking is unchanged.
