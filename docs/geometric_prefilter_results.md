# Geometric / CiM Prefilter Multi-Method Pilot — Results

Status: **development pilot complete** (2026-07-30).
Plan: [`geometric_prefilter_plan.md`](geometric_prefilter_plan.md).
Descriptor locked: D=9, B=5. Split: chronological 50/25/25 on four locked recordings.

## Gate scorecard

| Gate | Result | Evidence |
|---|---|---|
| **G-A** first-trigger ±N | **FAIL** | HJ passes via Pareto vs `mode_knn7` at p99.9 (2 dominate rows); MEArec best A recall only **0.878**, no dominate |
| **G-D** cosine ≈ L1 | **PASS** | Worst matched-Δ recall −0.47 pp (HJ), −0.78 pp (MEArec); both ≥ −1 pp |
| **G-drift** A2 helps drift | **FAIL** | Max A2−A1 on drift = **−5.7 pp**; min on static = **−15.8 pp** (causal-first hurts) |
| **G-E** tile power-gate | **PASS** | 2 HJ rows with tile fraction ≤ 25% and recall ≥ 0.99 (best fraction ≈ 0.024) |

Confirmation corpus: **not run** (G-A failed; hybrid not authorized).

## True first-detect follow-up (required)

The original `A2_causal_first` scanned the full GT window for any sample above
threshold, so it often locked onto channels that were **already hot at
`t−15`**. That is **not** hardware first-detect. Two corrected arms were added
and re-run on the same four pilots:

| Arm | Definition |
|---|---|
| `A2_rising_edge` | First **below→above** crossing in `[t−15, t]` only (peak-relative rising edge) |
| `A2_stream_detect` | Continuous stream rising-edge detector + global refractory 15; each GT peak takes the **earliest** detection in `[t−15, t+5]` |
| `A2_legacy_window_hot` | Old broken A2 (kept as ablation) |

### Level-1 recall at N=3, p99.9

| Condition | A1 peak-max | Stream first-detect | Rising-edge | Legacy window-hot |
|---|---:|---:|---:|---:|
| HJ drift | **0.967** | 0.907 | 0.900 | 0.896 |
| HJ static | **0.981** | 0.898 | 0.888 | 0.885 |
| MEArec | **0.871** | 0.570 | 0.530 | 0.553 |

Fallback→A1 rates are all ≤1.3% — the gap is **not** “no detection, fall back”.

### Anchor quality (all events, vs fit home channel)

| Recording | A1=home | Stream=home | Rising=home | Legacy=home |
|---|---:|---:|---:|---:|
| HJ drift | 0.848 | 0.654 | 0.649 | 0.639 |
| HJ static | 0.872 | 0.593 | 0.586 | 0.578 |
| MEArec 10u | 0.797 | 0.212 | 0.160 | 0.207 |
| MEArec 20u | 0.754 | 0.315 | 0.270 | 0.340 |

Median stream/rising hit time is still **before** the peak (HJ ≈ −5 samp,
MEArec ≈ −7…−12 samp), as expected for a rising-edge detector — but the
**channel** that crosses first is systematically farther from the unit home
than peak-max (`mean|anchor−home|` ≈ 5–6 ch on MEArec vs ≈ 1.5–1.9 for A1).

### Interpretation

1. **Cross-spike hijack was never the main story**; fixing the definition does
   not rescue MEArec (~0.55→~0.57).
2. **True first-detect is a real, weaker spatial cue than peak-max** for
   electrode-index ±N routing: the first channel to cross threshold is often a
   peripheral / noisy electrode, not the eventual peak/home channel.
3. **G-drift still fails** for stream and rising (max drift A2−A1 ≈ −4.6 /
   −5.1 pp; static also worse). First-detect does **not** help drift relative
   to peak-max under this protocol.
4. For hardware narrative: peak-max (or a short post-trigger argmax in a small
   align window) remains the better directory key; first-detect alone is the
   wrong index for ±N template pools.

## Answers to registered questions

1. **Can A get mean |C| < 10 at recall > 0.99?**  
   No on both families. HJ A tops out at recall **0.981** (mean |C| ≈ 3.8). MEArec A tops at **0.878**. Baseline `all` already has HJ recall 0.992 / mean |C| 7.1 at p99.9 without neighborhood pruning.

2. **A1 (proxy-max) vs A2 (first-detect)?**  
   After correcting A2 to true rising-edge / stream first-detect, A1 still
   dominates on every condition (see **True first-detect follow-up**). Legacy
   window-hot was slightly worse still, but not the cause of the MEArec collapse.

3. **Does B (centroid tile) beat A?**  
   Yes for recall. Best `B_tile` (tile_radius=3, p99.9-class): HJ recall **0.993** / mean |C| **7.0**; MEArec **0.996** / **7.2**. Tile activation fraction 0.25 (HJ) / 0.14 (MEArec).

4. **Does D (cosine) match L1?**  
   Yes within the 1 pp gate. At p99.9 full-array:

   | Family | L1 recall / mean\|C\| | Cosine recall / mean\|C\| |
   |---|---|---|
   | HJ | 0.9916 / 7.13 | 0.9883 / 6.68 |
   | MEArec | 0.9942 / 13.29 | 0.9922 / 13.76 |

5. **E power savings?**  
   Same activation as B. Active-tile fraction down to **~2–25%** of tiles while keeping HJ recall ≥ 0.99 on passing rows. Symbolic energy uses 100 pJ SRAM + 2 pJ/MAC; absolute pJ are plan constants, not silicon.

6. **C Jaccard?**  
   Best τ=0.5: HJ recall 0.989 / mean |C| 4.9; MEArec 0.975 / 6.5. Useful middle ground, but does not clear G-A’s MEArec bar and is weaker than B on recall.

7. **Hybrid A2+D?**  
   Computed for completeness under `--methods ALL`, but **not authorized** (G-A fail). Best hybrid MEArec recall only 0.62 — bottleneck is the A2 pool.

## Baseline reference (p99.9)

| Family | Directory | Recall | Mean \|C\| | Waveform traffic × |
|---|---|---:|---:|---:|
| HJ | all | 0.992 | 7.13 | 1.48 |
| HJ | mode_knn7 | 0.980 | 3.87 | 2.73 |
| MEArec | all | 0.994 | 13.29 | 1.28 |
| MEArec | mode_knn7 | 0.924 | 3.49 | 4.69 |

## Method frontiers (event-weighted family means at best recall config)

| Method | HJ recall | HJ mean\|C\| | MEArec recall | MEArec mean\|C\| |
|---|---:|---:|---:|---:|
| A1 best | 0.981 | 3.77 | 0.878 | 2.29 |
| A2 best | < A1 | — | ≪ A1 | — |
| B_tile r=3 | 0.993 | 6.96 | 0.996 | 7.19 |
| C_jaccard τ=0.5 | 0.989 | 4.93 | 0.975 | 6.45 |
| D_cosine p99.9 | 0.988 | 6.68 | 0.992 | 13.76 |
| E (same as B) | 0.993 | 6.96 | 0.996 | 7.19 |

## Memory / cycles (representative)

| Method | Extra bits / unit | Ops / event (mean active × 9) |
|---|---:|---:|
| Baseline L1 | 0 (54-bit row) | ~U×9 full, or active×9 with directory |
| A index ±N | 0 (routing only) | mean_active×9 |
| B/E tile id | +8 | mean_active×9; tiles powered ≪ all |
| C support mask | +C (16/32) | mean_active×9 |
| D ‖c‖² | +16 | mean_active×9 MACs |

## What this supports

1. **First-detected ±N (A) is not the winning Level-1 story on these pilots.** It trims candidates but loses MEArec recall; causal-first is worse than peak-max.
2. **Centroid-tile activation (B/E) is the strongest geometric prefilter here**: high recall, mean |C| ≈ 7, and large tile power-gating headroom.
3. **Cosine/dot-product (D) is a viable RRAM-native drop-in for L1** within 1 pp recall at matched candidate load — important for hardware narrative without changing D/B.
4. **Do not claim** causal-first helps drift (G-drift fail) or that A alone hits 0.99 / |C|<10 on both families (G-A fail).

## Artifacts

| Path | Role |
|---|---|
| `output/geometric_prefilter_sweep/geometric_prefilter_sweep.csv` | Per-config rows (456) |
| `output/geometric_prefilter_sweep/geometric_prefilter_gates.json` | G-A/D/drift/E |
| `docs/geometric_prefilter_source.sha256` | Source digests |
| `docs/geometric_prefilter_outputs.sha256` | Output digests |

## Reproduce

```bash
PYTHONPATH=.. python -m Spatial.experiments.run_geometric_prefilter_sweep --methods ALL
pytest Spatial/tests/test_geometric_prefilters.py -q
```
