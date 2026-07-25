# Two-Stage COM∧P2P → Waveform Upper Bound

Status: HJ 60s GT-spike oracle complete (2026-07-23).  
Spec: [`docs/superpowers/specs/2026-07-23-two-stage-spatial-waveform-design.md`](../../docs/superpowers/specs/2026-07-23-two-stage-spatial-waveform-design.md)

## Verdict

**Two-stage (COM∧P2P gate → full main-channel waveform) reaches 96.8% oracle** on Hybrid Janelia 60s — above soft_loc alone (91.6%) and soft_loc+FSDE concat (93.0%) from the prior study.

Peak-FSDE as stage-2 is also strong (**94.0%**), close to concat COM+P2P+FSDE (93.6%).

## Protocol

```text
raw → bandpass → GT spike times
  → local P2P (K=7)
  → COM (2D) + max-normalized P2P footprint
  → main-channel 64-sample waveform → Peak-FSDE 5D + full waveform
  → chronological 50/50
  → Stage1: admit unit if d_com < τ_com AND d_p2p < τ_p2p
       (τ = 95th percentile of train intra-unit distances)
  → Stage2: among candidates, nearest waveform/FSDE centroid
  → empty candidates → normalized COM+P2P sum fallback
```

Command:

```bash
python -m Spatial.experiments.run_two_stage_upper_bound \
  --dataset hj --duration 60 \
  --output Spatial/output/two_stage_upper_bound_hj60s.csv
```

Calibrated gates (this run): `τ_com≈0.082`, `τ_p2p≈0.50`.  
Two-stage fallback rate: **2.7%**; mean candidates per spike: **1.21**.

## HJ 60s results

| Method | Oracle | Notes |
|--------|--------|-------|
| `com_only` | 86.2% | Position alone |
| `p2p_only` | 80.6% | Footprint alone |
| `com_and_p2p` | 91.5% | Spatial stage without waveform |
| `fsde_only` | 47.6% | Waveform shape without space |
| `waveform_only` | 83.5% | Full snippet without space |
| `concat_com_p2p_fsde` | 93.6% | Flat concat (not two-stage) |
| `concat_com_p2p_waveform` | 94.3% | Flat concat |
| **`two_stage_fsde`** | **94.0%** | Target pipeline, FSDE stage-2 |
| **`two_stage_waveform`** | **96.8%** | Target pipeline, full waveform stage-2 |

## Reading

1. **Order matters:** two-stage waveform (96.8%) beats flat concat of the same ingredients (94.3%) — space as a hard gate then morphology is better than one joint distance.
2. **Stage-1 alone is already strong** (`com_and_p2p` 91.5%), close to prior soft_loc (91.6%) without needing spread/sharpness.
3. **FSDE alone is useless globally** (47.6%) but useful **inside** a spatial candidate set (→ 94.0%).
4. Full waveform still has headroom over FSDE in stage-2 (+2.8 pp).

## L1 substitution (P2P gate)

Stage-1 P2P distance can be switched with `--p2p-metric l1` (COM stays L2).
HJ 60s side-by-side:

| Method | P2P=L2 | P2P=L1 |
|--------|--------|--------|
| `p2p_only` | 80.6% | 77.6% |
| `com_and_p2p` | 91.5% | 91.2% |
| `two_stage_fsde` | 94.0% | **94.0%** |
| `two_stage_waveform` | **96.8%** | **96.8%** |
| fallback / mean cands | 2.7% / 1.21 | 2.9% / 1.20 |

**L1 is a drop-in for the two-stage ceiling** (identical FSDE/waveform oracles).
Only pure `p2p_only` nearest-centroid loses ~3 pp under L1.

```bash
python -m Spatial.experiments.run_two_stage_upper_bound --p2p-metric l1
```

CSV: `Spatial/output/two_stage_upper_bound_hj60s_p2pl1.csv`
