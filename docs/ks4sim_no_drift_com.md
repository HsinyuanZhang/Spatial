# COM / main-channel on KS4-paper `sim_no_drift`

Status: **Baseline / local diagnostic.** 60 s numbers are in.
Source: Figshare [25298815](https://janelia.figshare.com/articles/dataset/Simulations_from_kilosort4_paper/25298815)
`sim_no_drift.zip`. Simulator **ground-truth** cluster ids (1200 units), not
a Kilosort sorting.

The 40 GB `cbin` stays in the zip on `/mnt/data`. Spatial streams a time
prefix; nothing is copied to the SSD.

## Recording

| | |
|---|---|
| Probe | Neuropixels-style, 384 AP + 1 sync, 30 kHz |
| Geometry | 2-column shank map, 32 µm × 20 µm (`~snsShankMap`) |
| Length | 45 min (81e6 samples); this eval uses the first **60 s** |
| Drift | `drift` array is constant 100 (no motion) |
| GT | 18.8 M spikes, **1200** units; all 1200 fire in 60 s (428 k events) |
| Density | 1200 / 384 ≈ **3.1 units per electrode** |

Chance among 1200 train units is `1/1200 ≈ 0.083%`.

## Protocol (aligned with Yger 3-pitch COM)

```text
zip cbin on HDD (zlib 1 s chunks) → first 60 s, 384 AP
  -> zero-phase bandpass 300–6000 Hz
  -> unit location = electrode of the most-negative mean trough
     (80-spike subsample)
  -> event home = most-negative site in a 3-pitch (60 µm) disk
  -> COM = P2P-weighted centroid on that disk (median 12 electrodes)
  -> chronological 50/50, train-only centroids, L2 assign_nearest
```

Home wander after the disk: majority fraction 0.70, median 8 unique homes,
RMS 18 µm (not the 1 mm theft seen if the whole 384-ch trough is used).
Majority-home nearest-neighbor is **0 µm**: many units snap to the same
electrode, as expected at 3.1 units/site. 99.7% of units have another
unit’s majority home within 30 µm.

## 60 s result

| Method | Dim | n correct / n eval | Oracle argmin | vs chance (`1/1200`) |
|---|---:|---:|---:|---:|
| `main_channel` (channel index) | 1 | 40981 / 214131 | **19.14%** | ~230× |
| `main_channel_xy` (electrode x,y) | 2 | 23107 / 214131 | **10.79%** | ~130× |
| COM float | 2 | 36445 / 214131 | **17.02%** | ~204× |
| COM 4-bit | 2 | 14698 / 214131 | 6.86% | |
| COM 5-bit | 2 | 23150 / 214131 | 10.81% | |
| COM 6-bit | 2 | 33120 / 214131 | 15.47% | |
| COM 8-bit | 2 | 36114 / 214131 | 16.87% | ≈ float |

On this **linear** probe the raw channel index is ordered along the shank,
so 1-D `main_channel` beats electrode-snapped 2-D xy (the latter collapses
every unit that shares a site to the same point). COM is a continuous
coordinate along the probe and sits near the 1-D index, not above it:
overlap in a 60 µm disk still mixes several of the 3+ units per site.

## Why main_channel is 19%: crowding, not far-spike theft

Split of the 19.1% 3-pitch `main_channel` number:

| Mechanism | Evidence |
|---|---|
| Same-electrode crowding | 338 occupied sites, **mean 3.55 units/site** (median 3, max 40). **93.6%** of units share an electrode with another unit. |
| Far-spike home theft (>60 µm) | **1.3%** of 3-pitch events (median home error 0 µm). Not the 19% story. |
| If home is forced to each unit’s majority channel | nearest-centroid **28.0%**; spike-weighted majority lookup on that electrode **56.4%**. |
| 3-pitch detected home (the 19.1% run) | 67.8% of events already sit on the majority channel; the rest jitter inside the 60 µm disk. |
| Global trough, no 3-pitch disk | **60.7%** of homes are >60 µm away (median 421 µm) and argmin collapses to **0.48%**. That *is* far-spike theft — and it is already turned off in the 19% protocol. |

So the 19% is mostly **too many GT units on the same channel**, plus some
**nearby** (≤60 µm) overlap. It is not Yger-style theft from the other
end of the probe. Even a perfect home-channel lookup cannot unique-ID
1200 units on 338 sites; 56% is the occupancy ceiling, 28% is what
nearest-centroid on the channel index actually gets, 19% is that plus
in-disk jitter.

## Low-bit COM on the 1-D shank

COM_y intra-unit RMS is 12.5 µm; nearest COM_y centroids are 0.94 µm
apart (99.9% closer than one pitch). Unique-ID is occupancy-limited.
Spending 4–6 bits on the **full 3820 µm** probe makes y-bins of 255 / 123 /
61 µm — coarser than the pitch — so full-probe 1-D 4/5/6-bit COM_y is
1.0 / 1.8 / 2.7%, worse than the old 2-D bbox codes.

The bit budget belongs in a **local residual**: `home_channel` plus
`COM_y − y_home` quantized over ±H around the home electrode.

| Feature | Bits on residual | ±H | y-bin | Unique-ID |
|---|---:|---:|---:|---:|
| main_channel | — | — | 20 µm sites | 19.14% |
| COM float (x,y) | float | disk | — | 17.02% |
| COM_y float | float | — | — | 18.72% |
| 2-D bbox COM | 4 / 6 / 8 | full probe | 255 / 61 / 15 µm | 6.9 / 15.5 / 16.9% |
| COM_y full probe | 4 / 6 / 8 | full probe | 255 / 61 / 15 µm | 1.0 / 2.7 / 13.5% |
| **home + local COM_y** | **4** | **±90 µm** | **12 µm** | **25.14%** |
| home + local COM_y | 4 | ±60 µm | 8 µm | 24.91% |
| home + local COM_y | 5 | ±60 µm | 3.9 µm | 24.18% |
| home + local COM_y | 6 | ±90 µm | 2.9 µm | 23.02% |
| home + local COM_y | 8 | ±90 µm | 0.71 µm | 15.68% |

The mixed-L2 row **home index + 4-bit residual = 25.14%** is a **scale
artifact**, not a hardware unique-ID. The channel index lives in `{0,…,383}`
and the residual code in `{0,…,15}`; L2 on that pair is not an address
lookup. Keep it only as a comparison to the earlier sweep.

Residual-only codes without the home index are useless (~0.3%): they
are a local offset, not a shank coordinate.

```bash
python -m Spatial.experiments.analyze_np_com_lowbit --duration 60
```

## Home as address, residual as bits

Hardware treats home as a discrete electrode address. Residual
`COM_y − y_home` is a few bits on top. Train majority-home + nearest
residual:

| Gate | Residual | Unique-ID | True unit in list | mean \|C\| |
|---|---|---:|---:|---:|
| Exact home | float | 28.26% | 0.674 | 5.09 |
| Exact home | 4-bit ±90 µm | 28.50% | 0.674 | 5.09 |
| Exact home | 3-bit ±60 µm | 28.27% | 0.674 | 5.09 |
| Home slack 20 µm | float | 16.84% | 0.789 | 11.3 |
| Home slack 60 µm | float | 6.70% | 0.988 | 40.5 |
| L2 on `(home_x, COM_y)` µm | float | 29.17% | — | — |

Exact-home + residual **28.5%** is the same number as oracle-majority-home
nearest-centroid (28.0%). Residual does not unique-ID stacked neurons:

| Co-home diagnostic (train) | Value |
|---|---:|
| Units sharing a majority home | 93.4% |
| Mean units / occupied home | 3.52 (max 39) |
| Intra-unit residual RMS (median) | **4.26 µm** |
| Nearest co-home residual centroid (median) | **0.51 µm** |
| Fraction of co-home pairs with gap `> 2 ×` intra RMS | **0.27%** |

The residual cloud of units on the same electrode is narrower than the
gap between those units. More bits on residual overfit jitter; they do
not invent a new axis.

## `(home, residual code)` codebook

The useful low-bit object is a hash table, not nearest-centroid. Train
records which units appeared in each `(home, n-bit residual)` bin.

Unique-ID = majority unit of that bin (unseen bin = miss):

| Bits | ±H | Unique-ID | Unseen bin |
|---:|---:|---:|---:|
| 3 | 90 µm | 46.27% | 0.04% |
| 4 | 45 µm | 47.29% | 0.14% |
| 5 | 45 µm | **48.18%** | 0.27% |
| 6 | 90 µm | 48.11% | 0.27% |

That 48% is the occupancy ceiling (56.4% spike-weighted majority home)
minus visitors that land in the same bin. Extra residual bits buy about
two percentage points of unique-ID, not a new classifier.

Candidate list = every train unit observed in the bin (and, optionally,
neighbouring codes on the same home):

| Key | Δ code | Recall | mean \|C\| / 1200 | vs full scan |
|---|---:|---:|---:|---:|
| 3-bit ±90 µm | 0 | 0.9838 | 21.8 | 55× |
| **3-bit ±90 µm** | **1** | **0.9918** | **25.8** | **46.5×** |
| 4-bit ±90 µm | 1 | 0.9914 | 25.6 | 47× |
| 5-bit ±60 µm | 0 | 0.9501 | 16.3 | 74× |
| 6-bit ±90 µm | 0 | 0.9379 | 14.8 | 81× |
| 2-bit ±60 µm | 1 | 0.9918 | 25.8 | 46.5× |

Selected Level-1 key on this probe: **`(home, 3-bit residual)` with
±1 neighbouring code**, recall 0.992, 26 / 1200 units, **47×**. Two bits
plus a neighbour already saturates the 0.99 recall bar because the
address is the home electrode; the residual only drops occasional
visitors. Finer than 5–6 bits shrinks `|C|` but misses bins.

## COM radius, including reconstructed 4-bit residual

A Euclidean ball on analog COM, and on `home_xy + dequantized residual`:

| Source | R | Recall | mean \|C\| | vs full scan | WTA in ball |
|---|---:|---:|---:|---:|---:|
| COM float | 60 µm | 0.9884 | 41.3 | 29× | 0.322 |
| **COM float** | **65 µm** | **0.9935** | **44.9** | **27×** | 0.322 |
| COM float | 80 µm | 0.9996 | 54.4 | 22× | 0.322 |
| 4-bit residual ±90 µm, reconstructed | 65 µm | 0.9926 | 42.4 | 28× | — |
| 3-bit residual ±90 µm, reconstructed | 70 µm | 0.9932 | 46.6 | 26× | — |

The 65 µm row was not in the first radius grid (60 / 80 only). For
≥0.99 recall the geometric ball still needs **~43–45 units**, worse than
the 26-unit codebook. Use the codebook as the 1-D Level-1 gate; keep
R = 65 µm as the geometry-only fallback. R = 80 µm remains the
near-zero-miss ball if the second stage cannot tolerate 0.7% misses.

```bash
python -m Spatial.experiments.analyze_np_com_optimize --duration 60
python -m Spatial.experiments.analyze_np_com_optimize --duration 60 --adc --skip-sweeps
```

## P2P / POSNEG inside the codebook list

The useful Level-1 object on this shank is already
`(home, 3-bit residual ±90 µm, Δ=1)`: recall **0.9918**, mean `|C|` **25.81**,
**46.5×** vs 1200 units. This diagnostic freezes that list and scores a
cheap spatial descriptor inside it. Same 60 s events as the COM cache;
P2P/POSNEG are extracted from a streamed zip prefix and cached as
`events_60s_footprint.npz` (not copied from the 40 GB cbin).

| Method inside the codebook list | Unique-ID | Accuracy \| true in C |
|---|---:|---:|
| Occupancy majority of the bin | **0.4627** | 0.4666 |
| COM xy L2 | 0.3268 | 0.3295 |
| P2P float L1, electrode-indexed | **0.5676** | 0.5722 |
| P2P signed-5 L1, per-event max | 0.5111 | 0.5153 |
| POSNEG signed-4 L1 | 0.5070 | 0.5112 |
| COM z + max-norm P2P L1 | 0.5056 | 0.5098 |

Recall of the list is unchanged (0.9918). COM inside the list is *worse*
than the bin majority: residual location does not unique-ID stacked
neurons (median co-home centroid gap 0.51 µm vs intra RMS 4.3 µm).

Float P2P reaches **56.8%**. That is the same number as the
spike-weighted majority-home occupancy ceiling (**56.4%**): P2P is
essentially picking the commonest unit on that electrode, plus a small
visitor correction. Co-home footprints on this 1-D 12-site disk are not
a new identity axis. 5-bit P2P / 4-bit POSNEG sit ~5–6 pp behind float
and tell the same story.

Contrast with Yger 252-ch 2-D, where P2P inside a 70 µm COM ball moves
unique-ID from 56% to **71%**. Packed neighbours on a plane have distinct
footprints; stacked neighbours on a shank do not.

Level-1 on this probe remains the codebook candidate list, not a
classifier. P2P is not a unique-ID rescue for the 3.55 units/site
occupancy limit. Not the HJ/MEArec joint gate.

```bash
python -m Spatial.experiments.run_np_codebook_p2p --duration 60
```

CSV: `Spatial/output/np_codebook_p2p/np_codebook_p2p.csv`.

## Versus Yger 252-ch MEA (KS4 pseudo-GT, 3-pitch)

| | Yger `20160415_patch2` | `sim_no_drift` |
|---|---:|---:|
| Layout | 2-D 30 µm MEA | 1-D NP, 20 µm y |
| Labels | KS4 clusters (358 train) | simulator GT (1200) |
| Units / site | ~1.4 | ~3.1 |
| COM float argmin | **56.0%** | **17.0%** |
| vs chance | ~200× | ~204× |
| `main_channel_xy` | 45.6% | 10.8% |

The times-chance ratio is the same order. Absolute unique-ID is worse here
because three simulator neurons routinely share one electrode, whereas the
Yger 2-D grid spreads ~360 clusters over 252 sites. Direct COM WTA is still
the wrong use; a spatial **candidate list** along the probe is the
appropriate Level-1 role, same conclusion as Yger.

## How to rerun

CPU only (do not use GPU 1). From the parent workspace:

```bash
conda activate spint
# small files already extracted on the HDD:
#   .../figshare_25298815_kilosort4_sims/extracted_no_drift/
python -m Spatial.experiments.run_com_ks4sim_nodrift --duration 60
python -m Spatial.experiments.analyze_np_com_lowbit --duration 60
python -m Spatial.experiments.analyze_np_com_optimize --duration 60
# HDD zip stream; CPU only:
python -m Spatial.experiments.analyze_np_com_optimize --duration 60 --adc --skip-sweeps
python -m Spatial.experiments.run_np_codebook_p2p --duration 60
```

CSV / JSON:

```text
Spatial/output/ks4sim_no_drift_com/com_ks4sim_no_drift.csv
Spatial/output/ks4sim_no_drift_com/com_ks4sim_no_drift_summary.json
Spatial/output/ks4sim_no_drift_com/np_com_lowbit.csv
Spatial/output/ks4sim_no_drift_com/np_com_home_gate.csv
Spatial/output/ks4sim_no_drift_com/np_com_codebook.csv
Spatial/output/ks4sim_no_drift_com/np_com_codebook_delta.csv
Spatial/output/ks4sim_no_drift_com/np_com_quant_radius.csv
Spatial/output/ks4sim_no_drift_com/np_com_optimize_summary.json
Spatial/output/ks4sim_no_drift_com/np_adc_com.csv
Spatial/output/np_codebook_p2p/np_codebook_p2p.csv
```

## Signed ADC → COM (hardware bits)

Isolation homes stay analog 3-pitch troughs. Samples on the disk are a
fixed-gain signed ADC (all-channel or per-channel p99.9 / max), then P2P
→ COM. This is the Yger absolute-ADC protocol, not per-spike AGC.
Hottest-channel / median-channel peak is only **3.4×**; global max is
4005 vs p99.9 = 721.

| Scale | ADC bits | COM_y RMS vs analog | Codebook 4-bit ±90 recall | mean \|C\| | R=80 recall |
|---|---:|---:|---:|---:|---:|
| Analog P2P | — | 0 | 0.9771 | 21.1 | 0.9996 |
| **all-ch p99.9** | **4** | **1.25 µm** | **0.9777** | **21.3** | **0.9997** |
| all-ch p99.9 | 5 | 0.74 µm | 0.9783 | 21.3 | 0.9998 |
| all-ch p99.9 | 6 | 0.57 µm | 0.9782 | 21.3 | 0.9997 |
| per-ch p99.9 | 4 | 1.02 µm | 0.9775 | 21.2 | 0.9997 |
| all-ch max | 4 | 986 µm | 0.947 | 16.1 | 0.656 |
| all-ch max | 6 | 1.46 µm | 0.9772 | 21.1 | 0.9996 |

**4-bit all-channel p99.9 ADC is enough** for the codebook and for the
R=80 µm ball. The 1.25 µm COM_y error is far below the 12 µm 4-bit
residual bin and the 4.3 µm intra residual. Using the recording-wide
max as Vref wastes the code range (same trap as Yger); p99.9 is the
lock. Per-channel p99.9 is slightly cleaner and not required.

So the 1-D hardware stack is: **4–6-bit signed ADC → 3-pitch P2P COM →
`(home, 3-bit residual)` codebook**, not 6–8-bit bbox COM as unique-ID.

## What this does not show

- It does not run Kilosort on the simulation (labels are simulator GT).
- `sim_fast_drift` was still downloading; this file is the no-drift arm only.
- Codebook unique-ID is majority-in-bin, not a trained waveform second stage.
- Online GeoOsort (detect → Peak-FSDE → OSort) lives in
  [`../../GeoOsort/docs/KS4SIM_ONLINE.md`](../../GeoOsort/docs/KS4SIM_ONLINE.md),
  not in this COM isolation study.
