# Yger 20160415_patch2 — Kilosort4 Th=13 pseudo-GT

Status: **Selected Spatial input for this recording.** Labels are a Kilosort4
sorting, not juxta-cellular ground truth.

Date: 2026-08-20.

## What this file is

Dense 2D MEA (252 extracellular channels, 16x16 at 30 µm, 20 kHz, 300 s)
from Zenodo [10.5281/zenodo.1205233](https://doi.org/10.5281/zenodo.1205233)
(`20160415_patch2`). Event times and unit IDs come from **Kilosort 4.1.7**
with `Th_learned = Th_universal = 13`. Spatial should treat them as
**pseudo-GT**: useful for training spatial descriptors and two-stage
assignment, **not** as exhaustive multi-unit biological truth.

The patched-cell juxta train is stored alongside for scoring. It is **not**
mixed into `spike_unit_ids`.

## How to load in Spatial

Run from the parent workspace root. The NPZ and the int16 binary stay on
the HDD (`/mnt/data`); the loader memmaps the binary and does not copy it
to the SSD.

```bash
cd /home/xinyuan/SNN_SpikeSorting
python - <<'PY'
from Spatial.data.loader import load_yger_ks4

# All KS4 clusters (362 units, including MUA).
ds = load_yger_ks4(duration_s=60)

# KSLabel == "good" only (314 units at export time).
ds_good = load_yger_ks4(duration_s=60, good_only=True)

print(ds.name, ds.n_channels, ds.fs, ds.raw_data.shape, ds.unit_ids.size)
PY
```

Default path:

```text
/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work/spatial_npz/yger_20160415_patch2_ks4_th13.npz
```

The NPZ schema matches the HJ / CortexLab contract:

| Key | Role |
|---|---|
| `spike_times_0based` | Event sample indices |
| `spike_unit_ids` | KS4 cluster ids (not remapped) |
| `geom` | `(252, 2)` µm |
| `fs` | 20000 |
| `unit_ks_label` | `good` / `mua` per `unit_ids` |
| `raw_bin_path` | HDD pointer to `extra_int16.bin` (time-major int16) |
| `juxta_spike_times` | Author juxta times (evaluation only) |
| `juxta_matched_cluster_id` | KS cluster 310 |

`Dataset.raw_data` is `(n_channels, n_samples)` float64 of the int16 ADC
values after `uint16 - 32768`. The MCS file was already Butterworth-HP at
100 Hz; Spatial experiments that bandpass 300–6000 Hz should still do so.

## How the KS4 sorting was obtained

Environment: conda `snn_cuda`, Python 3.11, torch 2.3.1+cu121, **Kilosort
4.1.7**. GPU **0** only (`CUDA_VISIBLE_DEVICES=0`). GPU 1 was occupied and
was not used.

Conversion (already on HDD; do not recopy to SSD):

```bash
python -m Spatial.data.prepare_yger_zenodo --recording 20160415_patch2
```

That writes a 252-channel int16 binary from the MCS `uint16` file
(`padding` from the recording `.txt`, corners 126/127/254/255 dropped).

Selected sort:

```bash
CUDA_VISIBLE_DEVICES=0 python -m Spatial.experiments.run_kilosort_yger \
  --recording 20160415_patch2 --device cuda:0 --nblocks 0 --tol-ms 0.4 \
  --tag th13 --th-learned 13 --th-universal 13
```

Frozen settings:

| Setting | Value | Why |
|---|---|---|
| `nblocks` | 0 | In-vitro retina; KS4 docs: skip drift on this geometry |
| `nt` | 61 | ~3 ms at 20 kHz |
| `Th_learned` | 13 | Selected after 8–14 sweep; accuracy vs juxta peaked here |
| `Th_universal` | 13 | Matched `Th_learned` |
| `data_dtype` | int16 | Converted MCS file |
| `fs` | 20000 | Native |
| `n_chan_bin` | 252 | Extra-cellular sites only |

Export the Spatial NPZ (labels only; raw stays as the bin):

```bash
python -m Spatial.data.export_yger_ks4_pseudo_gt
```

KS4 results directory:

```text
/mnt/data/backup_datasets/SNN_SpikeSorting/zenodo_1205233_work/ks_out/20160415_patch2_th13/
```

## Why Th=13, and what it is not

A threshold sweep on this recording, scored against the juxta cell at
0.4 ms (Garcia / SpikeInterface window), selected Th=13:

| Th | accuracy | recall | precision | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 8 (KS4 default) | 0.821 | 0.996 | 0.824 | 750 | 13 |
| 10 | 0.894 | 0.995 | 0.898 | 398 | 18 |
| 12 | 0.932 | 0.991 | 0.940 | 224 | 30 |
| **13 selected** | **0.933** | 0.990 | **0.942** | **213** | 36 |
| 14 | 0.926 | 0.984 | 0.940 | 221 | 57 |
| Garcia KS2 (MATLAB, 2019) | 0.974 | 0.996 | 0.978 | ~80 | ~15 |

Best-matching KS4 cluster is **310** (3691 spikes vs 3514 juxta). Th=14
lost recall without cutting FP. `ccg_threshold` and 2D `x_centers` did
not beat Th=13 on the full 252-channel array.

This is **not**:

- juxta / author ground truth for every unit;
- a Kilosort2 reproduction (no MATLAB on this machine);
- a 64-channel SpikeForest subset;
- a causal online sort (full-file KS4, then Spatial may still zero-phase
  filter);
- evidence that Level-1 recall/traffic is solved.

## Evidence boundary for Spatial experiments

Oracle / GT-row experiments that call `load_yger_ks4()` are measuring
whether spatial features recover **KS4 cluster identities**. Hungarian
accuracy against these labels is a KS4-consistency number. The juxta cell
can be checked separately via `juxta_spike_times` and cluster 310.
Direct nearest-centroid COM / main-channel / low-bit COM on this input:
[`yger_com_direct_assign.md`](yger_com_direct_assign.md).

Garcia kept this recording in the 8/19 juxta-quality set
(`20160415_patch2`). That validates using the juxta train as a probe of
one cell, not the rest of the KS4 labels.

## Are 362 clusters 362 cells?

No. They are Kilosort4 cluster ids. The COM 60 s train split sees **358**
of the 361 clusters that fire in that slice (362 in the full 300 s NPZ;
314 `good`, 48 `mua`). That is a sorting, not a census.

KS4 was run with **`nblocks=0`**, so it did not track probe drift with
rigid blocks. This is an in-vitro retina MEA; large electrode motion is
not expected. A same-cell split by drift would show two nearby,
high-similarity clusters that occupy complementary halves of the
recording. That signature is **absent**.

Full 300 s diagnostic (`analyze_ks4_unit_splits`):

| Check | Result |
|---|---|
| Median KS4-position NN of cluster centroids | **5.2 µm** (p10 1.8, p90 24.7) |
| Clusters with a neighbor inside 30 µm | **98.6%** |
| Pairs inside 30 µm | 875 |
| Of those, `similar_templates` ≥ 0.8 | 90 |
| Of those 90, cross-refractory count ≪ independent (obs/exp < 0.5) | **14** |
| Those 14 also time-split (‖early-frac A − B‖ ≥ 0.35) | **0** |
| Greedy merge of the 14 edges | 14 clusters absorbed → **348 ids** |

Most nearby high-similarity pairs **do** fire within 1.5 ms as often as
(or more than) independent trains (median obs/exp **1.74** among the 90).
That is neighboring RGCs with similar templates, not one cell cut in two.
A handful of pairs (14) look mergeable (close + similar + shared
refractory). Even after absorbing them you still have ~350 ids packed at
~5 µm. COM unique-ID fails because of that packing, not because 358 is
an obvious double-count of ~100 drifting cells.

```bash
python -m Spatial.experiments.analyze_ks4_unit_splits
python -m Spatial.experiments.analyze_ks4_unit_splits --duration 60 \
  --output-dir Spatial/output/ks4_unit_splits_60s
```

The 60 s slice yields more “merge” pairs (33) because coincidence
statistics are noisier in a shorter window. Prefer the full-file 14-pair
count for this question.

## How other papers score this dataset

Nobody treats the hundreds of KS / Circus clusters as exhaustive biological
GT. Each recording has **one** juxta-cellular (loose-patch) cell. Published
accuracy is almost always: run the sorter on the MEA, pick the cluster that
best coincides with that juxta train, then report accuracy / precision /
recall (or error = mean of false-positive and false-negative rates) on
**that one cell**.

| Work | What they actually score | What they do *not* claim |
|---|---|---|
| [Yger et al. 2018](https://doi.org/10.7554/eLife.34518) (SpyKING CIRCUS; this Zenodo set) | 18 in-vitro juxta cells / 14 retinas. Error vs juxta; also vs a supervised “BEER” classifier that is allowed to see the true juxta times. Hybrid injection of real templates for 4225-ch scale. | That every other cluster on the 252-ch array is a verified neuron |
| [Magland et al. 2020](https://doi.org/10.7554/eLife.55167) SpikeForest `PAIRED_MEA64C_YGER` | Same juxta matching, but on the **64 extra sites nearest the juxta peak**, not 252 ch | Population-level GT; full-array census |
| [Buccino / Garcia et al. 2020](https://doi.org/10.7554/eLife.61834) SpikeInterface | After a juxta-quality cull (**8 / 19** kept; `20160415_patch2` is in that set), KS2 accuracy **0.974** on this file vs juxta (0.4 ms). Also: sorter-vs-sorter agreement, and **MEArec hybrids** where *all* units have labels | That KS2’s other ~hundreds of units are GT |
| Later papers using SpikeForest Yger | Inherit the paired-64ch juxta protocol | Full multi-unit GT |

Typical match window is ~0.4–1 ms. The sorter may output hundreds of units;
only the Hungarian / best-coincidence cluster against juxta enters the
accuracy table. The rest of the sorting is judged, if at all, by
**internal** metrics (refractory contamination → `KSLabel good/mua`),
**cross-sorter agreement**, or **simulations** (hybrid / MEArec) where every
unit is planted.

That is why Spatial’s COM numbers on KS4 cluster ids are
**KS4-consistency**, not “accuracy vs 314 real neurons.” The juxta cell
(cluster 310 here) remains the only biological probe, same as in those
papers.
