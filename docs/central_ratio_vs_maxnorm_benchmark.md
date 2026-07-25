# Central-ratio vs max-norm — HJ + MEArec

## 1) Oracle gate clip sweep (MEArec 32ch, 20 NPZs, 60s)

```bash
python -m Spatial.experiments.verify_central_ratio_gate --all-mearec --duration 60 \
  --clips 1.0,1.5,2.0,4.0 --quiet
```

CSV: `Spatial/output/central_ratio_gate_mearec60s.csv`

| Stat | Value |
|------|-------|
| **central_not_max_pct** | **mean 0.125** (min 0.084, max 0.195) |
| HJ reference (static11) | 0.084 |

MEArec 2D probe has **higher** central≠max than HJ 1D (~12.5% vs 8.4%), so the two encodings can diverge more often.

| repr | dist | bits | clip | mean oracle | std |
|------|------|------|------|-------------|-----|
| **maxnorm** | float_l2 | float | — | **0.837** | 0.114 |
| maxnorm | float_l1 | float | — | 0.834 | 0.114 |
| maxnorm | code_l1 | 4bit | — | 0.834 | 0.115 |
| central_ratio | float_l2 | float | 1.0 | 0.835 | 0.115 |
| central_ratio | code_l1 | 4bit | **1.0** | **0.834** | 0.116 |
| central_ratio | code_l1 | 4bit | 1.5 | 0.832 | 0.117 |
| central_ratio | code_l1 | 4bit | 2.0 | 0.831 | 0.117 |
| central_ratio | code_l1 | 4bit | 4.0 | 0.829 | 0.118 |

**Takeaway:** higher `central_not_max` does **not** yield a clear central_ratio win on this MEArec set. Best CR code clip remains **1.0**; all variants within ~0.8 pp of maxnorm float L2. Clip=4 still worst among CR codes.

HJ single-scene oracle (prior): same story — CR ≈ maxnorm at clip 1.0–1.5.

---

## 2) Unsupervised pipeline: maxnorm vs central_ratio codes

```bash
# maxnorm float / maxnorm 4-bit / central_ratio 4-bit clip=1.0
python -m Spatial.experiments.run_unsupervised_two_stage_sweep \
  --family all --duration 60 --quiet --no-oracle --p2p-repr maxnorm
python -m Spatial.experiments.run_unsupervised_two_stage_sweep \
  --family all --duration 60 --quiet --no-oracle --p2p-repr maxnorm --cim-bits 4
python -m Spatial.experiments.run_unsupervised_two_stage_sweep \
  --family all --duration 60 --quiet --no-oracle \
  --p2p-repr central_ratio --cim-bits 4 --ratio-clip 1.0
```

CSVs under `Spatial/output/unsupervised_two_stage_sweep/`  
(`*_maxnorm.csv`, `*_maxnorm_c4.csv`, `*_central_ratio_c4_clip1.csv`).

### Mean Hungarian accuracy — `unsup_two_stage_waveform`

| Config | HJ (12) | MEArec (20) |
|--------|---------|-------------|
| maxnorm float L2 | 0.812 ± 0.043 | 0.789 ± 0.105 |
| maxnorm 4-bit L1 | 0.813 ± 0.042 | 0.789 ± 0.105 |
| **central_ratio 4-bit L1 clip=1** | **0.813 ± 0.041** | **0.789 ± 0.105** |

Per-scene Δ (CR 4-bit − maxnorm 4-bit): HJ **−0.01 pp**, MEArec **+0.02 pp** (noise-level).

FSDE stage-2 and soft_loc SOM baselines are likewise unchanged across P2P encodings.

---

## Conclusions

1. **CiM-ready path is safe:** 4-bit code L1 (maxnorm or central_ratio @ clip=1) matches float maxnorm for both oracle and unsupervised on HJ + MEArec.
2. **central_ratio is not a free accuracy upgrade** here — even with MEArec `central_not_max≈12.5%`. Prefer it for **hardware reference stability** (detection-defined central), not for expected pp gains.
3. **Default clip for codes: 1.0** (HJ and MEArec). Avoid clip=4.
4. Probe note: MEArec v1 geom is a **32-channel 2D shank** (3×~11 staggered), not a 32×32 grid.

Related: [`central_ratio_gate.md`](central_ratio_gate.md) (HJ single-scene detail).
