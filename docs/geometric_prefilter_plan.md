# Geometric / CiM Prefilter Multi-Method Pilot — Plan

Status: **registered development pilot** (2026-07-30).
Descriptor / quantizer locked: **D=9, B=5**, fixed unit-interval unsigned codes.
Confirmation corpus is **not** authorized until a method passes the gates below.

## Goal

Compare Level-1 **candidate prefilters** (routing / scoring / activation only)
on recall, traffic, and mean candidates. Do not change the COM+P2P descriptor
or the 5-bit quantizer.

## Locked recordings

| Family | Dataset ID |
|---|---|
| HJ | `hybrid_janelia_drift16c_600s_11` |
| HJ | `hybrid_janelia_static16c_600s_11` |
| MEArec | `mearec_rec_v1_units10_snr5_seed202601` |
| MEArec | `mearec_rec_v1_units20_snr5_seed202601` |

Source digests: `LOCKED_PILOT_SOURCE_SHA256` in
`experiments/causal_temporal_selection.py`.
Split: `stable_three_way_split` chronological **50 / 25 / 25**.
Events: `prepare_causal_events` (same D9 contract).

## Baseline

`AdaptiveRangeSearch` uniform L1 with radius percentiles `{95, 99, 99.9}`:

- `directory=all` (full array)
- `directory=mode` (fit home + geom-KNN=7) as a reference arm only

Waveform traffic uses candidate-only 64×5-bit templates vs full `U×320` bits.

## Methods

### A — First-detected-channel ±N (two anchors)

| Arm | Anchor |
|---|---|
| **A1 `proxy_max`** | `argmax(|peak|)` (tie → smallest index) |
| **A2 `causal_first`** | Earliest sample in the event snippet where any channel exceeds a **fit-only** per-channel MAD×threshold; if none fire → fall back to A1 and count `fallback_to_max` |

Neighborhood: electrode-index contiguous window `[ch−N, ch+N]` clamped to
`[0, C−1]`. Unit pool: fit home channel inside the window; empty → widen N by 1
until nonempty or full array.

Grid: `N ∈ {1,2,3,4}`.

### B — Centroid-of-triggers tile prefilter

Trigger set = channels over the fit MAD threshold (else top-3 by |peak|).
Centroid = mean geom of triggers. Tile = `floor(y / tile_pitch)` with
`tile_pitch = 2 × median adjacent-channel spacing`. Activate home tile
± `tile_radius`. Grid: `tile_radius ∈ {1,2,3}`; post-L1 Top-K
`K ∈ {3,5,7,10}` reported separately from unpruned pool recall.

### C — Jaccard spatial-support filter

Fit `fit_unit_channel_support` (coverage 0.95). Event mask = B’s trigger set.
Keep units with Jaccard ≥ `τ ∈ {0.2, 0.3, 0.4, 0.5}`, then L1 radius.

### D — Dot-product / cosine instead of L1

Same codes/centroids. Score `1 − cosine` with integer dot and precomputed
`‖c‖²`. Pure cosine arm + secondary rank-fusion `0.7·cosine_rank + 0.3·L1_rank`.

### E — Hierarchical tile activation (power model)

Same tiles as B. Report `fraction_tiles_active` and

```text
E = n_sram_lookups · 100 pJ + n_cim_macs · 2 pJ
```

(symbolic constants; not a silicon claim).

### Hybrid A+D

Only if G-A passes: best A pool → D cosine radius inside the pool.

## Pre-registered gates

Families are **never** pooled for pass/fail.

| Gate | Pass if |
|---|---|
| **G-A** | Some A config: recall ≥ 0.99 **and** mean \|C\| < 10 on **both** families, **or** Pareto-dominates `mode` directory at p99.9 on both (higher/equal recall and lower/equal mean \|C\|, strict on at least one) |
| **G-D** | Cosine recall within 1 pp of L1 at matched mean \|C\| (±10% relative) on both families |
| **G-drift** | On HJ: `(A2−A1)_drift ≥ +2 pp` and `(A2−A1)_static ≥ −1 pp` at the same N and percentile |
| **G-E** | Some E config: active-tile fraction ≤ 25% and recall ≥ 0.99 on HJ |

Fail → document; no confirmation sweep for that branch.

## Energy / cycle accounting (symbolic)

| Op | Cycle model | Energy |
|---|---|---|
| SRAM directory lookup | 1 / event | 100 pJ |
| Active-row L1 (D=9) | 9 abs + 8 add / row | 2 pJ / MAC-equivalent (abs counts as one) |
| Cosine MAC | 9 mul-acc / row | 2 pJ / MAC |
| Waveform SAD read | 320 bit / candidate | counted in traffic, not in CiM pJ |

## Run order

A1+A2 → D → B → C → E → (optional A+D hybrid).

## Artifacts

| Path | Role |
|---|---|
| `experiments/run_geometric_prefilter_sweep.py` | Runner |
| `experiments/geometric_prefilter_common.py` | Shared eval / gates |
| `algorithms/geometric_prefilters.py` | Anchors, tiles, Jaccard, cosine |
| `output/geometric_prefilter_sweep/` | CSV + gates JSON |
| `docs/geometric_prefilter_results.md` | Frontiers + answers |
| `docs/geometric_prefilter_{source,outputs}.sha256` | Provenance |
