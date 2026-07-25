# Unsupervised two-stage — multi-dataset sweep

**Oracle ≠ unsupervised.** Numbers below are Hungarian accuracy on GT-matched spikes unless labeled oracle (exact GT-id match after GT-centroid fit).

## Setup

- Chronological 50/50 train/test; duration **60s**
- Unsupervised: soft_loc SOM → within-cluster wave SOM → COM∧P2P gate (τ from discovered clusters)
- P2P gate: **L2** and **L1** (COM stays L2)
- Script: `python -m Spatial.experiments.run_unsupervised_two_stage_sweep --duration 60`
- CSVs: `Spatial/output/unsupervised_two_stage_sweep/`

## Hybrid Janelia — 12× 16ch 600s scenes

| Method | P2P L2 | P2P L1 |
|--------|--------|--------|
| som_soft_loc (train→test) | 0.752 ± 0.059 | — |
| **unsup_two_stage_fsde** | **0.795 ± 0.050** | **0.797 ± 0.048** |
| **unsup_two_stage_waveform** | **0.812 ± 0.043** | **0.814 ± 0.041** |
| oracle_two_stage_fsde | 0.810 ± 0.076 | 0.808 ± 0.078 |
| oracle_two_stage_waveform | **0.903 ± 0.038** | **0.903 ± 0.036** |

Per-scene (L2, waveform): best static `*11/12` ~0.84–0.87; harder drift/high-unit scenes ~0.72–0.80. Oracle waveform stays ~0.85–0.97.

## MEArec v1 — 20× 32ch NPZs (units∈{10,20} × SNR∈{2,3,5,8,12} × 2 seeds)

| Method | P2P L2 | P2P L1 |
|--------|--------|--------|
| som_soft_loc | 0.754 ± 0.120 | — |
| unsup_two_stage_fsde | 0.750 ± 0.117 | 0.750 ± 0.117 |
| **unsup_two_stage_waveform** | **0.789 ± 0.105** | **0.789 ± 0.105** |
| oracle_two_stage_fsde | 0.731 ± 0.202 | 0.727 ± 0.204 |
| oracle_two_stage_waveform | **0.837 ± 0.114** | **0.834 ± 0.114** |

By designed unit count (L2, waveform):

| Units | Unsup waveform | Oracle waveform |
|------:|----------------|-----------------|
| 10 | 0.854 ± 0.087 | 0.919 ± 0.062 |
| 20 | 0.724 ± 0.078 | 0.755 ± 0.091 |

By SNR (L2 mean over seeds/units): unsupervised waveform rises 0.68 @ SNR2 → 0.85 @ SNR12; tracks soft_loc but slightly higher overall.

## Takeaways

1. **L1 ≈ L2** on both families for unsupervised and oracle gates.
2. On **HJ**, unsupervised two-stage waveform (**~81%**) beats soft_loc SOM (**~75%**) and sits below oracle waveform (**~90%**). Gap to oracle is smaller than the old single-scene 84% vs 97% story once averaged over harder scenes.
3. On **MEArec**, unsupervised waveform (**~79%**) is close to oracle waveform (**~84%**); dense 20-unit recordings are the bottleneck for both.
4. Oracle FSDE is unstable on MEArec (high variance); unsupervised FSDE can look competitive via Hungarian remapping but waveform remains the safer stage-2 feature.

## Reproduce

```bash
python -m Spatial.experiments.run_unsupervised_two_stage_sweep --duration 60
# family subset:
python -m Spatial.experiments.run_unsupervised_two_stage_sweep --family hj
python -m Spatial.experiments.run_unsupervised_two_stage_sweep --family mearec
```
