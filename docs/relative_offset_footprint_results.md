# Relative-Offset Footprint (P0) — Results

Status: **pilot complete** (2026-08-08).
Plan: [`relative_offset_footprint_plan.md`](relative_offset_footprint_plan.md).

## Gate scorecard

| Check | Result |
|---|---|
| HJ: relative P2P argmin ≥ KNN | **PASS** (0.717 vs 0.664) |
| HJ: relative median true-unit P2P L1 ≤ KNN | **PASS** (0.422 vs 0.434) |
| **P0 HJ pass** | **PASS** |
| MEArec SNR5: argmin ≥ KNN | PASS (0.736 vs 0.668) |
| MEArec SNR5: true L1 ≤ KNN | FAIL (0.440 vs 0.436; tiny regression) |

## Family means (60s, chrono 50/50, K=7)

| Family | Layout | P2P argmin | median true P2P L1 | Gate recall @95 | mean \|C\| |
|---|---|---:|---:|---:|---:|
| HJ (12) | knn | 0.664 | 0.434 | 0.907 | 2.32 |
| HJ (12) | relative | **0.717** | **0.422** | 0.906 | 2.39 |
| MEArec snr5 (4) | knn | 0.668 | 0.436 | 0.915 | 3.06 |
| MEArec snr5 (4) | relative | **0.736** | 0.440 | **0.925** | 3.31 |

## Interpretation

1. Relative-offset columns fix the KNN slot-scramble on HJ: argmin +5.3 pp and
   lower same-unit L1, matching the P0 hypothesis.
2. MEArec (1D primary-axis approximation) still gains argmin; true-unit L1 is
   essentially flat. Full 2D patch_grid remains future work.
3. COM∧P2P gate recall is unchanged on HJ and slightly up on MEArec; candidate
   load stays ~2–3. Safe to keep as Stage-1b representation before shift-min (P1).

## Artifacts

| Path | Role |
|---|---|
| `output/relative_footprint_pilot/relative_footprint_pilot.csv` | Per-recording rows |
| `output/relative_footprint_pilot/relative_footprint_pilot_summary.json` | Family means + gates |

## Reproduce

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_spatial_footprint.py -q
python -m Spatial.experiments.run_relative_footprint_pilot --duration 60 --quiet
```
