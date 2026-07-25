# Waveform stage-2: bipolar thermometer CAM (4×cell Hamming)

**Question:** Can we replace float waveform L2 with a Hamming-friendly encoding that expands **cell count** (pos/neg × 4 thermometer cells ≈ “1.5bit CAM cells”) instead of needing high per-cell precision?

**Protocol (oracle, GT spikes, chronological 50/50):**
- Stage-1 fixed: COM float L2 ∧ P2P **4-bit maxnorm L1** (τ≥1)
- Stage-2: main-channel waveform encodings below
- Datasets: **12× HJ 16ch** + **20× MEArec 32ch**, 60s

```bash
python -m Spatial.experiments.verify_waveform_cam_thermometer --family all --duration 60
```

CSVs: `Spatial/output/waveform_cam_thermometer/`

## Encoding

1. Remove DC per spike  
2. \(x_+=\max(x,0)\), \(x_-=\max(-x,0)\)  
3. Each side → `n_cells` thermometer bits (threshold ≥)  
4. Flatten → Hamming (≡ L1 on {0,1} bits)  
5. Unit centroid = encode(**mean raw waveform**), then match

Threshold modes: **uniform** (p99 ladder) vs **quantile**.

## Results (mean oracle ± std, Δ vs `wave_float_l2`)

### Hybrid Janelia (12 scenes)

| Method | Acc | Δpp |
|--------|-----|-----|
| wave_float_l2 | **0.903 ± 0.036** | 0 |
| wave_abs4bit_l1 | 0.897 ± 0.038 | −0.57 |
| **wave_therm4_quantile_hamming** | **0.869 ± 0.043** | **−3.37** |
| wave_therm4_uniform_hamming | 0.864 ± 0.053 | −3.92 |
| wave_therm2_quantile_hamming | 0.853 ± 0.050 | −5.05 |
| wave_signed2bit_hamming | 0.531 ± 0.161 | **−37.2** |

### MEArec (20 NPZs)

| Method | Acc | Δpp |
|--------|-----|-----|
| wave_float_l2 | 0.834 ± 0.115 | 0 |
| wave_abs4bit_l1 \|DC\| | **0.871 ± 0.092** | **+3.74** |
| **wave_therm4_quantile_hamming** | **0.842 ± 0.113** | **+0.76** |
| wave_therm4_uniform_hamming | 0.813 ± 0.145 | −2.13 |
| wave_therm2_quantile_hamming | 0.824 ± 0.132 | −1.04 |
| wave_signed2bit_hamming | 0.494 ± 0.318 | **−34.0** |

(On MEArec, `abs4bit_l1` uses DC-removed |x| while `float_l2` uses peak-normalized bipolar waveforms — not an apples-to-apples “same feature, fewer bits” comparison; thermometer vs float is the fair CAM story.)

## Takeaways

1. **Flat signed 2-bit Hamming collapses** (−34 to −37 pp). Confirms: coarse per-sample codes + Hamming are not enough.
2. **4-cell bipolar thermometer Hamming recovers most accuracy**: HJ −3.4 pp vs float; MEArec **≈ float / slightly above** with quantile thresholds.
3. **Quantile thresholds > uniform** (especially MEArec).
4. **4 cells > 2 cells** on HJ; MEArec 2-cell is closer but still behind quantile-4.
5. Binary L1 ≡ Hamming on thermometer bits (as expected) — hardware can use either match-line style.
6. Fits the “don’t sparsify; expand CAM cells” story: **~T×8 binary cells/spike** (T≈64 → ~512 bits) with Hamming, without needing 4-bit analog weights per tap.

## Hardware mapping (as discussed)

Per time sample: **pos 4 cells + neg 4 cells** (thermometer / ~1.5bit MLC each).  
Query vs stored template → cell Hamming sum. No sparsity required.

## Code

- Encoding: `Spatial/algorithms/cim_thermometer.py`
- Experiment: `Spatial/experiments/verify_waveform_cam_thermometer.py`
- Unit tests: `Spatial/tests/test_cim_thermometer.py`
