# Shift-CONV / Cosine vs L1 — 4/5/6-bit Sweep

Status: **complete** (2026-08-10).
Experiment: `Spatial/experiments/run_shift_conv_bit_sweep.py`.

## Protocol

| Item | Value |
|---|---|
| Features | Relative 1D P2P, K=7 |
| Methods | `conv` (raw max-dot), `cosine` (L2-norm then max-dot), `l1` (shift-min) |
| Bits | float, 4, 5, 6 (`per_spike_max` → [0,1]) |
| Shift radius S | 0, 1, 2 |
| Data | HJ 12 scenes + MEArec SNR5 (4 files), 60 s, chrono 50/50 |
| Gate | COM L2 ∧ (score ≥ τ for CONV family / L1 dist ≤ τ) |

Grid size: **576** rows (16 datasets × 3 methods × 4 bits × 3 S).

## Headline

| Finding | Detail |
|---|---|
| Raw `conv` collapses | ~14% HJ / ~10% MEArec — energy bias to bright templates |
| `cosine` is the MAC story | Matches or beats L1 on HJ; ≈ L1 on MEArec |
| 4/5/6-bit ≈ float | Cosine 4-bit within **&lt;0.4 pp** of float at best S |
| Best family picks | HJ: `cosine` float S=2; MEArec: `l1` 5-bit S=0 |

## Bit curves (best S by assign accuracy)

### Hybrid Janelia (n=12)

| Method | 4-bit | 5-bit | 6-bit | float | best S |
|---|---:|---:|---:|---:|---:|
| cosine | 0.729 | 0.732 | 0.733 | **0.733** | 2 |
| l1 | 0.718 | 0.721 | 0.721 | 0.721 | 2 |
| conv | 0.142 | 0.141 | 0.142 | 0.142 | 0 |

Cosine − L1 at matched S=2, 5-bit: **+1.1 pp**.

### MEArec SNR5 (n=4)

| Method | 4-bit | 5-bit | 6-bit | float | best S |
|---|---:|---:|---:|---:|---:|
| l1 | 0.735 | **0.736** | 0.736 | 0.736 | 0 |
| cosine | 0.733 | 0.735 | 0.736 | 0.735 | 1 |
| conv | 0.097 | 0.097 | 0.097 | 0.098 | 0 |

Cosine ≈ L1 within **±0.2 pp** at S∈{0,1}; S=2 hurts both.

## Gate (family means, representative)

| Family | Method | bits | S | gate recall | mean \|C\| |
|---|---|---:|---:|---:|---:|
| HJ | cosine | 5 | 2 | 0.911 | 2.23 |
| HJ | l1 | 5 | 2 | 0.905 | 2.35 |
| MEArec | cosine | 5 | 1 | 0.927 | 3.28 |
| MEArec | l1 | 5 | 0 | 0.925 | 3.31 |

Cosine gate is slightly tighter / higher recall than L1 on the same COM L2 front-end.

## Interpretation

1. **Do not use raw MAC (unnormalized CONV)** for template ID — amplitude dominates shape.
2. **Normalized correlation (cosine)** is the correct MAC-CiM classifier: same crossbar MAC after unit-norm (or online normalize-then-dot).
3. **5-bit is enough**; 4-bit loses &lt;0.4 pp vs float; 6-bit ≈ float.
4. **Hardware pick for MAC Stage-1b**: relative K=7 + cosine + **5-bit** + **S=1** (HJ insurance / MEArec-safe). Prefer L1 only if the array is distance-CAM rather than MAC.
5. Multi-copy area still scales as (2S+1)× vs S=0 (same as P1 accounting).

## Artifacts

| Path | Role |
|---|---|
| `output/shift_conv_bit_sweep/shift_conv_bit_sweep.csv` | Full grid |
| `output/shift_conv_bit_sweep/shift_conv_bit_sweep_summary.json` | Grids, bit curves, head-to-head |

## Reproduce

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_shift_match.py -q
python -m Spatial.experiments.run_shift_conv_bit_sweep --duration 60 --quiet
```
