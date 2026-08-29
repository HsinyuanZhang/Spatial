# Direct COM / main-channel assignment on Yger KS4 Th=13

Status: **Baseline / local diagnostic.** 60 s numbers are in, including the
3-pitch same-spike exclusion.
Scope: nearest-centroid assignment using only a 1-D home-channel index or a
2-D geometry-weighted COM (float and 4/5/6/8-bit). Not two-stage waveform
refinement, not P2P-only, not CiM candidate recall/traffic.

This is a **local component** measurement. It is **not** a STAR-Mem system-gate
result. Do not mix these oracle argmin numbers with candidate-recall or
template-read traffic.

## Verdict

KS4 `spike_positions` on this recording jitter **3.5 µm RMS per unit**, below
the **30 µm** channel pitch. Apparent home-channel wander in the first COM
eval was overlap: a louder trough on a distant RGC was treated as the same
spike. Re-running with the rule **extrema more than 3 pitches (90 µm) away
are a different spike** changes the picture.

| Feature | Isolation | Dim | Oracle argmin | vs chance (`1/358`) |
|---|---|---:|---:|---|
| `main_channel` (channel index) | whole-array `argmax \|amp\|` | 1 | **1.33%** | ~4.8× |
| COM float | whole-array + KNN K=7 | 2 | **4.06%** | ~14.5× |
| `main_channel` (channel index) | 3-pitch disk | 1 | **10.75%** | ~39× |
| `main_channel_xy` (home electrode) | 3-pitch disk | 2 | **45.55%** | ~163× |
| COM float | 3-pitch disk | 2 | **55.99%** | ~200× |
| COM 8-bit | 3-pitch disk | 2 | **56.06%** | same order as float |
| KS4 `spike_positions` (not COM) | oracle location prior | 2 | **80.65%** | location ceiling |

The 4% figure was overlap contamination, not unit wander. With a 90 µm
same-spike disk, COM is a real 2-D location estimate (~56%). It is still
**not** a unique-unit classifier: 358 train centroids sit on a 30 µm MEA
with ~7.5 µm nearest-centroid spacing, so even KS4's own positions only
reach 81%. Direct COM WTA remains the wrong Level-1 use; a spatial
**candidate list** around that coordinate is the appropriate one.

On the 3-pitch path, bit width starts to matter (4-bit 35%, 8-bit ≈ float)
because the feature now carries micrometre-scale location. Under whole-array
isolation it did not, because the coordinate was already wrong.

## Evidence boundary

Labels are **Kilosort4 Th=13 clusters** (362 units in the NPZ, 314 `good`),
not juxta-cellular ground truth. The 60 s slice used here contains 361 of
those units; the train split sees 358. Oracle argmin accuracy is
KS4-consistency: predicted unit id must equal the KS4 cluster id of that
GT-isolated event. The patched-cell juxta train is a side diagnostic only
(cluster **310**). See [`yger_ks4_pseudo_gt.md`](yger_ks4_pseudo_gt.md).

358 is not a biological cell count. Cluster centroids sit a median **5.2 µm**
apart; 98.6% have a neighbor inside one channel pitch. A merge heuristic
(close + `similar_templates` ≥ 0.8 + shared refractory) finds **14** pairs
on the full file and **no** early-vs-late occupancy split. KS4 used
`nblocks=0`. Oversplit exists at the margin; it does not collapse 358
ids to a small number of drifting cells. Details in
[`yger_ks4_pseudo_gt.md`](yger_ks4_pseudo_gt.md#are-362-clusters-362-cells).

## Protocol

Shared prefix for both isolation arms:

```text
HDD NPZ + memmapped int16 bin (not copied to SSD)
  -> load_yger_ks4(duration_s=60)          # 252 ch, fs=20000, KS4 labels
  -> zero-phase bandpass 300–6000 Hz      # full loaded slice; non-causal
  -> chronological 50/50 split, train-only centroids, L2 assign_nearest
  -> accuracy = mean(pred == KS4 id) on test events whose unit is in train
```

**Legacy isolation** (whole-array; first 60 s table):

```text
  -> home = argmax |amp| over all 252 channels in ±15 samples
  -> KNN K=7 including self, local P2P ±15 samples
  -> main_channel = normalized home-channel index (1-D)
  -> COM = P2P-weighted centroid on that KNN
```

**3-pitch same-spike isolation** (`--same-spike-max-pitches 3`):

```text
  -> seed = that event's KS4 spike_positions (x, y)
  -> pitch = median nearest-neighbor electrode spacing = 30 µm
  -> radius = 3 × pitch = 90 µm
  -> home = most-negative channel in ±2 samples among electrodes in the disk
     (a stronger trough outside the disk is a different spike)
  -> COM = P2P-weighted centroid on every electrode within 90 µm of that home
     (median 24 sites; table padded to 29 with -1)
  -> extra features:
       main_channel_xy = bbox-normalized (x, y) of the home electrode
       ks4_spike_xy    = bbox-normalized KS4 position (ceiling, not COM)
```

The 3-pitch seed is an **oracle same-spike location prior**, not a deployed
detector. It answers: if we do not let a far overlapping RGC steal the
extremum, how unique are home-channel and COM? A channel-local threshold
detector implements the same spatial support without KS4 (it never sees the
far trough as part of this event). Using the whole-array argmax on GT times
does not.

COM quantization is **not** `quantize_p2p`. Each centroid coordinate in
`[0, 1]` is independently mapped onto `{0, …, 2^N−1}` with a fixed analytic
range (same style as D9/B5 `UniformUnsignedQuantizer(fixed_unit_interval=True)`).
Assignment of quantized COM is L2 in that integer code space. Ranking is
identical to L2 on dequantized `[0, 1]` coordinates.

Primary metric is **oracle argmin** (`pred == true`). Hungarian
`clustering_accuracy` is not applied. There is no unsupervised arm.

Isolation notes, matching other Spatial DAC pilots:

- Event times come from the KS4 pseudo-GT, not an online detector.
- `scipy.signal.filtfilt` is zero-phase on the loaded slice (60 s here).
  That is non-causal. It is documented as an isolation choice, not a
  deployed filter.
- The 1-D relative-offset patch is not used; this recording is a 2-D MEA.

## 60 s result: legacy whole-array isolation

Command:

```bash
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 --quiet
```

Default NPZ (HDD):

```text
/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work/spatial_npz/yger_20160415_patch2_ks4_th13.npz
```

Recording slice: 252 channels, 20 kHz, 60 s, 60 600 valid GT events, 361 KS4
units in-slice. Train/test 30 300 / 30 300. Evaluation set: 30 258 test
events whose unit appeared in train (358 train centroids; 3 unseen test
units). `--good-only` was **not** set (all KS4 clusters, including MUA).

| Method | Bits | Dim | n correct / n eval | Oracle argmin | All-test argmin | Unique codes (x, y) |
|---|---|---:|---:|---:|---:|---|
| `main_channel` | — | 1 | 402 / 30258 | 0.0133 | 0.0133 | — |
| `com_float` | float | 2 | 1228 / 30258 | 0.0406 | 0.0405 | 56548, 56548 |
| `com_4bit` | 4 | 2 | 1289 / 30258 | 0.0426 | 0.0425 | 16, 16 |
| `com_5bit` | 5 | 2 | 1218 / 30258 | 0.0403 | 0.0402 | 32, 32 |
| `com_6bit` | 6 | 2 | 1356 / 30258 | 0.0448 | 0.0448 | 60, 60 |
| `com_8bit` | 8 | 2 | 1224 / 30258 | 0.0405 | 0.0404 | 225, 222 |

Chance among 358 train units is `1/358 ≈ 0.0028`. 4–8-bit COM stays within
about 0.5 percentage points of float. The 6-bit point being slightly above
float is not interpreted as a bit-width win; the absolute accuracy is too
low for that difference to matter.

CSV / JSON:

```text
Spatial/output/com_mainchannel_lowbit/com_mainchannel_lowbit.csv
Spatial/output/com_mainchannel_lowbit/com_mainchannel_lowbit_summary.json
```

## 60 s result: 3-pitch same-spike exclusion

Command:

```bash
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 \
  --same-spike-max-pitches 3 \
  --output-dir Spatial/output/com_mainchannel_lowbit_local3pitch
```

Same 60 s slice: 252 channels, 20 kHz, 60 600 valid GT events, 361 KS4 units,
30 300 / 30 300 split, 30 258 eval events, 358 train centroids. Pitch from
the geom is 30.0 µm; radius is 90.0 µm. Median 24 electrodes fall in the
disk (max 29). On **21.1%** of events the array-wide most-negative trough
lies outside that disk — those are the overlap thefts that produced the
legacy 4% COM number.

| Method | Bits | Dim | n correct / n eval | Oracle argmin | All-test argmin | Unique codes (x, y) |
|---|---|---:|---:|---:|---:|---|
| `main_channel` | — | 1 | 3252 / 30258 | 0.1075 | 0.1073 | — |
| `main_channel_xy` | — | 2 | 13782 / 30258 | 0.4555 | 0.4549 | 16, 16 |
| `com_float` | float | 2 | 16941 / 30258 | 0.5599 | 0.5591 | 60335, 60335 |
| `com_4bit` | 4 | 2 | 10621 / 30258 | 0.3510 | 0.3505 | 14, 15 |
| `com_5bit` | 5 | 2 | 14863 / 30258 | 0.4912 | 0.4905 | 30, 30 |
| `com_6bit` | 6 | 2 | 16345 / 30258 | 0.5402 | 0.5394 | 60, 60 |
| `com_8bit` | 8 | 2 | 16962 / 30258 | 0.5606 | 0.5598 | 237, 238 |
| `ks4_spike_xy` | — | 2 | 24403 / 30258 | 0.8065 | 0.8054 | 60398, 60421 |

`main_channel_xy` is the fair 2-D home-channel feature on this MEA: the
bbox-normalized electrode coordinate, not the integer channel index. The
1-D index still under-states the home electrode because adjacent ids are
not spatially adjacent on the 16×16 grid. Unique `(x, y)` codes of 16, 16
are the discrete electrode columns/rows (252 sites on a 16×16 layout with
four corners missing).

COM float at 56% sits between home-electrode snap (46%) and KS4's own
positions (81%). The remaining 56% → 81% gap is P2P COM being a noisier
location estimator than KS4's template-based `spike_positions`. The 81% →
100% gap is packing: within-unit RMS ~3.5 µm versus nearest-unit centroid
spacing ~7.5 µm, so location alone cannot unique-ID every event among 358
centroids.

4-bit COM drops to 35% because each axis is a 16-level code over the full
probe bbox (bin ≳ 50 µm), coarser than unit spacing. 8-bit tracks float.
That bit-width curve is only meaningful after the same-spike disk is
applied.

## 5-bit P2P → 6-bit COM

These are different quantizers. P2P bits are **per-channel weights** in the
3-pitch disk. COM bits are **per-axis codes of the bbox-normalized
centroid over the full 450 µm probe**. COM is

```text
COM = Σ (P2P_i × electrode_xy_i) / Σ P2P_i
```

so the output is a ratio of sums of known coordinates, not a 5-bit copy of
the weights. Integer 5-bit P2P (`per_spike_max`, codes 0..31 as weights)
on this 60 s slice:

| Quantity | Value |
|---|---|
| Probe bbox | 450 × 450 µm |
| 6-bit COM bin | 450 / 63 = **7.14 µm** per axis |
| COM error vs float P2P (RMS) | **0.21, 0.22 µm** (euclid 0.31 µm) |
| Equivalent bits vs probe bbox | **9.2 bit / axis** |
| Same 6-bit code as float-P2P COM | **95.5%** of events |
| Error < half of a 6-bit bin | **100%** |
| Oracle argmin, float COM from 5-bit P2P | 0.5599 (same as float P2P) |
| Oracle argmin, 6-bit COM from 5-bit P2P | 0.5415 (float-P2P 6-bit was 0.5402) |

**Yes: 5-bit P2P is enough for 6-bit COM**, and the weight quantization is
not the 6-bit bottleneck. The 6-bit loss (56% → 54%) is the 7.14 µm bin on
the full probe, which is already comparable to nearest-unit centroid
spacing (~7.5 µm). Hardware that MACs 5-bit P2P against electrode
coordinates and rounds the ratio to 6 bits per axis matches analog COM
to well under one COM LSB.

The 5-bit in that measurement is **`per_spike_max`**: each event is divided
by its own neighborhood max, then mapped to 0..31. That is per-event AGC,
**not** a 5-bit ADC. The hardware-like path is below.

On the same 60 s disk, event-max P2P is heavy-tailed but not pathological
(median 491, p99 1490, max 2601 ≈ **5.3×** median, **1.75×** p99). One
recording-wide 5-bit scale therefore wastes codes on typical events
(median peak code **6/31**; 72% of events peak at ≤7/31; almost none
saturate). Per-electrode max varies by about **2.9×** (hottest = ch 168).

| 5-bit scale | Used as COM weights | COM RMS vs float | ~eq. bits | 6-bit code match | argmin |
|---|---|---:|---:|---:|---:|
| per-event max (`per_spike_max`) | integer codes | 0.31 µm | 9.2 | 0.955 | 0.5599 |
| recording-wide max | integer codes | 1.56 µm | 6.9 | 0.774 | 0.5521 |
| recording-wide p99 LSB | integer codes | 0.75 µm | 7.9 | 0.912 | 0.5552 |
| per-channel max | codes, **no** dequant | 5.78 µm | 5.0 | 0.283 | 0.5558 |
| per-channel max | dequantized `code×LSB_ch` | 0.62 µm | 8.2 | 0.905 | 0.5583 |
| per-channel p99 | dequantized | 0.49 µm | 8.5 | 0.931 | 0.5598 |

COM needs **commensurate physical amplitudes**. A per-channel 5-bit ADC is
fine if each code is multiplied by that channel's LSB before the MAC. Using
the raw per-channel codes as weights mixes different scales and blows the
centroid (5.8 µm RMS). A single global 5-bit scale still fits under a
7.14 µm 6-bit COM bin, but typical spikes only use ~3 extra bits of the
5-bit range because of the 5× event-max tail.

Command:

```bash
python -m Spatial.experiments.analyze_com_from_p2p_bits --duration 60
```

JSON: `Spatial/output/com_from_5bit_p2p/com_from_p2p_bits.json`.

## Absolute 5/6-bit ADC (fixed gain, then signed ADC)

Hardware path: one analog scale for all channels, or one scale per
channel, then a uniform **signed** ADC on the bandpass waveform. P2P and
COM are computed from the reconstructed samples. Homes stay the analog
3-pitch trough (isolation is not re-detected). Full-scale is taken from
`|sample|` in the same ±15-sample event windows.

Event-window `|x|` has a tail: **max = 1951**, **p99.9 = 345** (5.7×).
Hottest channel max is 3.1× the median channel. A single Vref set to the
absolute max therefore spends most of a 5-bit ADC on rare peaks.

5-bit signed ADC has ±15 codes; 6-bit has ±31. Analog COM argmin is
0.5599. 6-bit COM bins are 7.14 µm.

| Scale | ADC bits | Clip frac | COM RMS | ~eq. bits | Same 6-bit COM code | COM argmin | Then 6-bit COM argmin |
|---|---:|---:|---:|---:|---:|---:|---:|
| analog P2P (no ADC) | — | 0 | 0 | — | 1.000 | 0.5599 | 0.5402 |
| all-channel max | 5 | 0 | **4.43 µm** | 5.4 | 0.472 | **0.5088** | 0.4903 |
| all-channel max | 6 | 0 | 1.56 µm | 6.9 | 0.773 | 0.5572 | 0.5345 |
| all-channel p99.9 | 5 | 0.001 | 1.19 µm | 7.3 | 0.859 | 0.5512 | 0.5382 |
| all-channel p99.9 | 6 | 0.001 | 1.07 µm | 7.4 | 0.898 | 0.5513 | 0.5318 |
| per-channel max | 5 | 0 | 1.29 µm | 7.2 | 0.814 | 0.5571 | 0.5411 |
| per-channel max | 6 | 0 | 0.62 µm | 8.2 | 0.905 | 0.5601 | 0.5419 |
| per-channel p99.9 | 5 | 0.001 | 0.92 µm | 7.7 | 0.880 | 0.5548 | 0.5369 |
| per-channel p99.9 | 6 | 0.001 | 0.75 µm | 7.9 | 0.918 | 0.5549 | 0.5366 |

**If every channel shares one Vref = recording max, 5-bit ADC does not
cleanly support 6-bit COM.** RMS 4.4 µm is a large fraction of the 7.14 µm
COM bin; unique 6-bit codes match analog COM on only 47% of events;
nearest-centroid drops from 56% to 51%. A 6-bit ADC with that same shared
Vref is enough (RMS 1.56 µm, argmin 55.7%).

**5-bit ADC is enough if the scale is not set by the 5.7× tail:** either
one Vref at p99.9 (0.1% clips, RMS 1.2 µm) or a **per-channel** Vref at
that channel's own max (RMS 1.3 µm, argmin 55.7%). Per-channel 6-bit ADC
is essentially analog COM (0.62 µm RMS).

Command:

```bash
python -m Spatial.experiments.analyze_adc_absolute_com --duration 60
```

JSON: `Spatial/output/adc_absolute_com/adc_absolute_com.json`.

## COM radius prefilter (6-bit all-channel p99.9 ADC)

Locked ADC: signed 6-bit, one Vref for all channels, Vref = p99.9 of
`|sample|` in the event windows (FS = 344.5). COM is the P2P-weighted
centroid of the reconstructed waveform on the 3-pitch disk, in µm.
Train (chronological first half) fits one COM centroid per KS4 unit.
Test keeps a unit iff `‖COM_event − centroid_u‖ ≤ R`.

This is a **candidate list**, not unique-ID. Unrestricted ADC COM
nearest-centroid is still 55.13%. Intra-unit ADC COM RMS is 14.0 µm
(median); nearest train-centroid spacing is 7.9 µm. Direct WTA cannot
separate those clouds; a ball around the event COM can still drop units
that sit on the other side of the MEA.

| R (µm) | True-unit recall | mean \|C\| / 358 | Units dropped | Traffic vs full U | argmin in list |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.9430 | 7.0 | 98.1% | 51.4× | 0.5512 |
| 40 | 0.9598 | 11.7 | 96.7% | 30.6× | 0.5513 |
| 60 | 0.9821 | 21.6 | 94.0% | 16.6× | 0.5513 |
| **70** | **0.9904** | **28.9** | **91.9%** | **12.4×** | 0.5513 |
| 75 | 0.9933 | 32.0 | 91.1% | 11.2× | 0.5513 |
| 90 | 0.9988 | 41.6 | 88.4% | 8.6× | 0.5513 |
| 120 | 1.0000 | 64.6 | 82.0% | 5.5× | 0.5513 |

Selected global radii (smallest R that meets the recall floor):

- **R = 70 µm** for recall ≥ 0.99: keep 29 of 358 units, drop **329
  classes / event**, 12.4× fewer unit rows than a full scan.
- **R = 40 µm** for recall ≥ 0.95: keep 12 units, drop 96.7%, 30.6×.

Analog (no ADC) COM traces the same curve within ~0.2 pp. 6-bit
all-channel p99.9 is not the limiter; COM scatter is.

Per-unit train-distance percentiles do **not** reach 0.99 (p99.9 →
recall 0.9857, mean \|C\| 21.7). Train p99.9 does not cover the test
COM tail (intra p90 RMS 23.7 µm). A single global R is the more robust
rule on this recording.

argmin among survivors saturates at the unrestricted 55.13% as soon as
recall is high. The radius **does not** improve unique classification;
it only removes far units before a later stage.

This is a **Yger 252-ch 2-D** local diagnostic (KS4 pseudo-GT, 60 s,
oracle 3-pitch isolation). It is not the HJ/MEArec Level-1 joint gate.
On this dense 2-D MEA a 70 µm ball is a small patch of the 450 µm
probe, which is why 0.99 recall and >4× row reduction happen together
here — that joint pair failed on 16-ch linear probes for a different
packing reason.

Command:

```bash
python -m Spatial.experiments.analyze_com_radius_prefilter --duration 60
```

CSV / JSON:

```text
Spatial/output/com_radius_prefilter/com_radius_prefilter.csv
Spatial/output/com_radius_prefilter/com_radius_prefilter_summary.json
```

CSV / JSON:

```text
Spatial/output/com_mainchannel_lowbit_local3pitch/com_mainchannel_lowbit.csv
Spatial/output/com_mainchannel_lowbit_local3pitch/com_mainchannel_lowbit_summary.json
```

## P2P / POSNEG inside the 70 µm COM ball

COM unique-ID saturates at 56%. The 70 µm ball keeps recall 0.99 at 29
candidates. This diagnostic freezes that list and asks whether a cheap
spatial descriptor can unique-ID the remaining packed neighbours.

Isolation matches the 3-pitch analog-COM path (not the 6-bit ADC COM used
to *select* R=70 µm). Analog COM at 70 µm on this 60 s split: recall
**0.9888**, mean `|C|` **28.75**, 12.5×. Train-only templates, 50/50
chronological split. `|C|=1` does not skip scoring for the unique-ID
number below; empty lists are misses.

| Method inside the ball | Unique-ID | Accuracy \| true in C | Feature |
|---|---:|---:|---|
| COM xy L2 (location only) | **0.5591** | 0.5654 | 2-D µm |
| P2P float L1, electrode-indexed | **0.7145** | 0.7226 | ~12 live / 252 |
| P2P signed-5 L1, per-event max | 0.6879 | 0.6956 | 5-bit codes |
| POSNEG signed-4 L1 | 0.6898 | 0.6976 | pos+neg |
| COM z + max-norm P2P L1 | 0.6830 | 0.6907 | concatenated |

P2P adds **+15.5 pp** over COM location among the same 29 units. 5-bit P2P
and 4-bit POSNEG sit 2–3 pp behind float P2P. Concatenating COM does not
help: location and shape should not share one L1 after an ad hoc z-score.

This is still not unique-ID of 358 clusters (71% vs chance 0.28%). Level-1
on this MEA remains a candidate ball. P2P is the right cheap object *inside*
that ball, not a replacement for it. Not the HJ/MEArec joint 0.99/4× gate.

```bash
python -m Spatial.experiments.run_yger_com_ball_p2p --duration 60
```

CSV: `Spatial/output/yger_com_ball_p2p/yger_com_ball_p2p.csv`.

## Juxta side diagnostic

Not the assignment target. On the 60 s test split, 303 events coincide with
author juxta times within 0.4 ms (8 samples at 20 kHz). Cluster 310 is
present in train and test.

| Isolation | Feature | juxta match rate (pred == 310) |
|---|---|---:|
| whole-array | main-channel or any COM | **0 / 303** |
| 3-pitch disk | `main_channel` (1-D index) | **0 / 303** |
| 3-pitch disk | `main_channel_xy` | 191 / 303 = 0.630 |
| 3-pitch disk | COM float | 144 / 303 = 0.475 |
| 3-pitch disk | COM 4-bit | 189 / 303 = 0.624 |
| 3-pitch disk | KS4 `spike_xy` | 178 / 303 = 0.587 |

The 1-D channel index still never assigns the patched cell, because index
is not geometry. Once the home is restricted to the 90 µm disk around the
event, 2-D home / COM recover cluster 310 on about half to two-thirds of
juxta-coincident test events. The rest lose nearest-centroid to a nearby
RGC — the same packing limit as the 81% KS4-position ceiling.

## How to rerun

From the parent workspace root, CPU / numpy only (do not use GPU 1; this
runner does not import CUDA):

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting

# Legacy whole-array argmax + KNN K=7
python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 --quiet

# 3-pitch same-spike disk (90 µm on this MEA)
python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 \
  --same-spike-max-pitches 3 \
  --output-dir Spatial/output/com_mainchannel_lowbit_local3pitch

# 6-bit all-channel p99.9 ADC, then COM radius candidate gate
python -m Spatial.experiments.analyze_com_radius_prefilter --duration 60

# P2P / POSNEG WTA inside the frozen 70 µm analog-COM ball
python -m Spatial.experiments.run_yger_com_ball_p2p --duration 60
```

Optional:

```bash
# KSLabel == "good" only (314 units in the NPZ)
python -m Spatial.experiments.run_com_mainchannel_lowbit --duration 60 --good-only --quiet

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  Spatial/tests/test_com_mainchannel_lowbit.py \
  Spatial/tests/test_spatial_footprint.py -q
```

Do not copy the int16 recording onto `/home`. `load_yger_ks4` memmaps the
HDD binary. `spike_positions.npy` stays in the KS4 output directory on
`/mnt/data`.

## What this does not show

- The COM radius table is a Yger-local candidate list (KS4 labels, 60 s).
  It is not the HJ/MEArec 0.99-recall / 4×-traffic system gate.
- It does not evaluate P2P footprints, shift-min, or waveform refinement
  on the surviving list.
- It does not claim a causal online sorter (GT times + zero-phase filter).
- The 3-pitch path seeds the disk with KS4 `spike_positions`. That is
  oracle same-spike isolation, not a proof that a detector will pick the
  same home.
- Unique-unit COM assignment is still failed (~55% vs 358 centroids).
  The useful product of COM on this MEA is the **radius candidate list**,
  not WTA. The 4% number was overlap theft, not that ceiling.
