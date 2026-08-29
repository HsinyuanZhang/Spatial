# Shift-Min + CiM Multi-Copy (P1) — Results

Status: **pilot complete** (2026-08-09).
Plan: [`shift_cim_plan.md`](shift_cim_plan.md).

## Gate scorecard

| Gate | Result |
|---|---|
| HJ drift: S≥1 L1 lowers median true-unit P2P vs S=0 | **PASS** (best S=2, drop ≈ 0.0014) |
| L1 4-bit within 1 pp of float at best S | **PASS** (HJ Δ0.32 pp; MEArec Δ0.12 pp) |
| CiM accounting table S=0/1/2 published | **PASS** |
| **P1 pass** | **PASS** |

## Family means — L1 float (relative K=7)

| Family | S | argmin | median true dist | gate recall | mean \|C\| | area × |
|---|---:|---:|---:|---:|---:|---:|
| HJ | 0 | 0.717 | 0.422 | 0.906 | 2.39 | 1 |
| HJ | 1 | 0.720 | 0.421 | 0.906 | 2.39 | 3 |
| HJ | 2 | **0.721** | **0.421** | 0.905 | 2.35 | 5 |
| MEArec snr5 | 0 | **0.736** | 0.440 | 0.925 | 3.31 | 1 |
| MEArec snr5 | 1 | 0.735 | 0.440 | 0.926 | 3.22 | 3 |
| MEArec snr5 | 2 | 0.727 | 0.440 | 0.920 | 3.10 | 5 |

## L1 vs L2 (HJ, float, S=0)

| Metric | argmin | median true dist | gate recall |
|---|---:|---:|---:|
| L1 | 0.717 | 0.422 | 0.906 |
| L2 | **0.724** | 0.205 | 0.907 |

L2 slightly higher argmin on this grid; L1 remains the CiM-default (matches 4-bit / Hamming path). 4-bit L1 tracks float within &lt;0.4 pp.

## CiM Option A accounting (4-bit × K=7)

| S | rows / unit | bits / row | area vs S=0 |
|---:|---:|---:|---:|
| 0 | 1 | 28 | 1× |
| 1 | 3 | 28 | 3× |
| 2 | 5 | 28 | 5× |

## Interpretation

1. After P0 relative columns, residual shift-min gains on HJ are **small but positive** (drift true-distance ↓, argmin +0.4 pp at S=2).
2. MEArec prefers **S=0** for argmin; larger S slightly hurts — do not default S=2 on 2D without COM pruning / 2D shift design.
3. Practical P1 recipe: **relative + L1 + S=1** (area ×3) as insurance for ±1 home jitter; keep S=0 as MEArec baseline.
4. 4-bit multi-copy bank is enough for Stage-1b CiM story.

## Artifacts

| Path | Role |
|---|---|
| `output/shift_cim_pilot/shift_cim_pilot.csv` | Full grid rows |
| `output/shift_cim_pilot/shift_cim_pilot_summary.json` | Gates + family grids |

## Reproduce

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_shift_match.py -q
python -m Spatial.experiments.run_shift_cim_pilot --duration 60 --quiet
```
