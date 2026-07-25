# Central-ratio vs max-norm P2P gate (oracle)

Minimal A/B for CiM-friendly stage-1 P2P encoding.

## Setup

- HJ `static16c_600s_11`, 60s, GT spikes, chronological 50/50
- COM gate unchanged (float L2); stage-2 full waveform
- Representations: **maxnorm** (`÷max`) vs **central_ratio** (`÷slot0`, clip)
- Distances: float L2/L1; 4-bit integer L1 in code space (`rint(mean)` centroids, `τ≥1`)

```bash
python -m Spatial.experiments.verify_central_ratio_gate --duration 60 \
  --clips 1.0,1.5,2.0,4.0
```

## HJ 60s result

**`central_not_max_pct = 8.4%`** (>5% → representations can diverge; not a no-op).

| repr | dist | bits | clip | oracle | Δ vs maxnorm float L2 |
|------|------|------|------|--------|------------------------|
| maxnorm | float_l2 | float | — | **0.968** | 0 |
| maxnorm | float_l1 | float | — | 0.968 | +0.07 pp |
| maxnorm | code_l1 | 4bit | — | **0.968** | 0 |
| central_ratio | float_l2/l1 | float | 1.0–4.0 | 0.967–0.968 | ≤0.07 pp |
| central_ratio | code_l1 | 4bit | **1.0** | **0.968** | +0.07 pp |
| central_ratio | code_l1 | 4bit | **1.5** | **0.968** | 0 |
| central_ratio | code_l1 | 4bit | 2.0 | 0.964 | −0.36 pp |
| central_ratio | code_l1 | 4bit | 4.0 | 0.966 | −0.14 pp |

## Takeaways

1. Float central_ratio ≈ maxnorm on this scene (as expected when central≈max most of the time).
2. **4-bit code L1 best at clip 1.0–1.5** (matches the “don’t use clip=4” warning); clip=2.0 loses ~0.4 pp.
3. Code-L1 path reuses ``two_stage_assign`` + ``calibrate_thresholds_codes`` (τ≥1); no separate assigner.
4. Unsupervised hook: `--p2p-repr central_ratio --cim-bits 4 --ratio-clip 1.0` (via optional `p2p_raw_*`).
5. Multi-dataset follow-up (MEArec 20 NPZs + unsup A/B): [`central_ratio_vs_maxnorm_benchmark.md`](central_ratio_vs_maxnorm_benchmark.md).

CSV: `Spatial/output/central_ratio_gate_hj60s.csv`
