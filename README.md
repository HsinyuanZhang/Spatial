# Spatial Spike Sorting

Spatial is the algorithm and experiment repository behind the **STAR-Mem**
spike-sorting study. It started as a Python reproduction of the spatial
clustering sorter from Akhoundi et al. (JSSC 2025), then grew into a research
testbed for low-bit spatial descriptors, CiM/CAM candidate retrieval, and
candidate-only waveform refinement.

The current research question is:

> Can a hardware-friendly spatial stage retain the true unit while shrinking
> the unit search space enough that an expensive waveform stage only reads a
> few templates?

As of **2026-08-11**, the answer is not yet end-to-end. Several component
results are useful, including the recent relative-offset footprint and
shift/cosine pilots, but no configuration jointly satisfies high candidate
recall, at least 4x waveform-read reduction, compact temporal accuracy,
causal/no-GT operation, realistic dense-2D validation, and physical hardware
evidence. See [Results overview](docs/RESULTS_OVERVIEW.md) for the exact status
and claim boundaries.

## Start here

| If you want to... | Read or run |
|---|---|
| Understand the system and the code | [Architecture guide](docs/ARCHITECTURE.md) |
| See what passed, failed, or remains open | [Results overview](docs/RESULTS_OVERVIEW.md) |
| Set up data and reproduce experiments | [Reproduction guide](docs/REPRODUCING.md) |
| Find a protocol, result, or provenance record | [Documentation index](docs/README.md) |
| Choose an experiment runner | [Experiment index](experiments/README.md) |
| Get the ShiftCAM/COM-P2P research handoff | [COM-P2P handoff](docs/COM_P2P_HANDOFF.md) |

The detailed files under `docs/` remain deliberately flat. Historical
commands, cross-links, paper references, and SHA-256 manifests depend on those
paths, so the documentation is reorganized through stable navigation pages
instead of moving provenance-sensitive records.

## System in one picture

```text
Target online path

raw channels
  -> causal detection and alignment
  -> local spatial patch
  -> COM + relative-offset P2P descriptor
  -> 4/5-bit spatial candidate retrieval (CiM/CAM or indexed SRAM)
  -> candidate set C
       C=0: spatial fallback
       C=1: direct assignment
       C>1: candidate-only waveform-template reads and digital refinement
  -> unit ID + online state update
```

Most current paper-facing experiments isolate one or more components using GT
event times and/or GT-aligned unit rows, chronological splits, and
full-recording zero-phase filtering. Those results measure an algorithmic
ceiling; they are not yet a causal deployed sorter. The distinction is
documented in [Architecture guide](docs/ARCHITECTURE.md#evidence-levels).

## Where the project is now

The project has five useful layers of work:

1. **Reference reproduction.** The TU Delft MATLAB pipeline was ported to
   Python: NEO detection, Akhoundi Eq. 3-5 features, wired-OR compression, and
   modified SOM clustering.
2. **Probe-robust spatial features.** `soft_loc` and local P2P footprints
   replace features that degenerate on 1D probes.
3. **Two-stage sorting.** COM and P2P first select candidate units; waveform
   centroids resolve ambiguous candidates. Oracle and fully unsupervised
   variants expose the gap between feature information and deployable labels.
4. **Low-bit memory search.** Per-unit range/L1 rows, directories, indexed
   graphs, masks, weights, multiple prototypes, and compact temporal templates
   explore recall/traffic/storage trade-offs. These branches provide both
   positive component evidence and strong negative results.
5. **Relative-offset + shift/cosine branch.** Fixed physical-offset P2P slots
   remove KNN slot scrambling. Shift-min gives a small HJ drift benefit;
   normalized correlation is a viable MAC-CiM alternative to L1. The current
   implementation is still a **1D primary-axis pilot**, not the final 2D patch
   mapping.
6. **Robust P2P and online-update branch.** Three-/five-point averages and
   4–8-bit integer templates are now tested with an independent confirmation
   boundary. A fixed five-point estimator fails held-out HJ drift; 8-bit
   float-equivalence passes conditionally, and a confidence-gated
   error-feedback updater is implemented but not promoted to a data result.

The active research direction should therefore be read as a hypothesis under
evaluation, not as a completed STAR-Mem implementation.

## Repository map

```text
Spatial/
├── algorithms/       Reusable detection, feature, clustering, search, and adaptation code
├── configs/          Reference experiment parameters
├── data/             Dataset loaders and data-generation/preparation tools
├── docs/             Architecture, protocols, results, provenance, and navigation
├── evaluation/       Detection and Hungarian-mapped clustering metrics
├── experiments/      Runnable experiment and verification entry points
├── output/           Generated CSV/JSON/log artifacts; mostly gitignored
├── papers/           Akhoundi et al. reference paper
├── reference_code/   TU Delft MATLAB implementation and notes
└── tests/            Unit and experiment-contract tests
```

The repository normally lives inside the parent workspace:

```text
SNN_SpikeSorting/
├── Spatial/                     this repository
├── new_datasets/                Hybrid Janelia inputs and catalog
├── synthetic_spike_dataset/     MEArec/GeoOsort-format inputs
└── DAC2027/                     paper source and claim ledger
```

## Quick start

Run modules from the **parent workspace root**, so the `Spatial` package and
sibling datasets resolve consistently:

```bash
cd /home/xinyuan/SNN_SpikeSorting
conda activate spint

# Current relative/shift branch; no recording corpus required.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  Spatial/tests/test_spatial_footprint.py \
  Spatial/tests/test_shift_match.py \
  Spatial/tests/test_two_stage_assign.py -q

# Full suite. Frozen provenance tests intentionally require their recorded
# repository revision and runtime-source inventory; see the reproduction guide.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests -q
```

Core runtime dependencies are `numpy`, `scipy`, `h5py`, `matplotlib`, and
`PyYAML`; tests require `pytest`. MEArec generation additionally uses
`MEArec`/`MEAutility`. There is currently no packaged environment lockfile, so
record the active environment for paper-facing reruns.

Representative data-backed commands:

```bash
# Spatial-only baseline / upper bound.
python -m Spatial.experiments.run_spatial_upper_bound \
  --dataset hj --duration 60

# Fully unsupervised two-stage sorter.
python -m Spatial.experiments.run_two_stage_unsupervised --duration 60

# Latest P0 representation comparison.
python -m Spatial.experiments.run_relative_footprint_pilot \
  --duration 60 --quiet

# P1 shift-min and 4-bit multi-copy accounting.
python -m Spatial.experiments.run_shift_cim_pilot \
  --duration 60 --quiet

# Shifted L1 vs raw dot product vs normalized correlation.
python -m Spatial.experiments.run_shift_conv_bit_sweep \
  --duration 60 --quiet

# Robust P2P development selection and independent confirmation.
python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --pilot --quiet
python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --confirmation --quiet
```

The commands above expect the sibling dataset paths described in
[Reproduction guide](docs/REPRODUCING.md). Exact frozen commands for a
paper-facing result belong in that result's document, not in this landing
page.

## Current result snapshot

| Question | Current evidence | System implication |
|---|---|---|
| Does the Akhoundi spatial idea work on a high-density 2D grid? | Legacy 1024-channel reproduction reaches 92.1% vs 81.2% CC baseline | Reference path reproduced; not validation of the new D9/B5 memory pipeline |
| Do fixed relative-offset P2P slots help? | P0 improves P2P argmin from 0.664 to 0.717 on HJ and 0.668 to 0.736 on MEArec SNR5 | Useful representation change; P0 passes its local gate |
| Does shift-min solve drift? | HJ gains are small; S=2 changes argmin 0.717 to 0.721 and costs 5x rows; MEArec prefers S=0 | Promising jitter insurance, not a drift solution |
| Is low-bit similarity viable? | 4/5/6-bit cosine is close to float; raw unnormalized dot product collapses | Normalized correlation or L1 remains viable at component level |
| Does a fixed 3/5-point P2P average generalize? | Centered k=5 wins development but held-out HJ drift falls 1.05 points; one scene falls 6.65 points | **No** as a global default; study a fit/calibration-only conditional selector next |
| Which bit width is safe for the averaged template? | Only 8 bit passes development confidence gates and stays within 0.10 points of float on confirmation | Conditional 8-bit equivalence; raw-P2P minimum still needs independent confirmation |
| Is online P2P updating validated? | Error-feedback arithmetic, confidence gates, rollback, and write accounting are tested; the data run is stopped after estimator failure | Implementation foundation only, not an online accuracy claim |
| Does existing Level 1 meet 0.99 recall and 4x traffic jointly? | No tested branch does so across HJ and MEArec | Main system bottleneck remains open |
| Is compact Level 2 solved behind a high-recall source? | Pair-specific taps recover accuracy but not traffic/storage; dyadic sketch saves bits but loses accuracy | No selected compact temporal representation |
| Is there physical RRAM/CiM superiority evidence? | No RTL/macro/PVT/P&R comparison | Logical bit/op counts must not be reported as PPA |

For the source tables, aggregation mode, and local-vs-system gate distinction,
use [Results overview](docs/RESULTS_OVERVIEW.md).

## Datasets

| Dataset family | Geometry | Main role | Default location from parent root |
|---|---|---|---|
| Hybrid Janelia | 16-channel linear, 20 um pitch; static and drift scenes | Realistic linear-probe development and drift evidence | `new_datasets/hybridjanelia/` |
| MEArec v1 | 32-channel Neuronexus-style 2D shank | Synthetic cross-family and higher-unit-density evidence | `synthetic_spike_dataset/generated/geoosort_npz_v1/` |
| Legacy 2D grid | 32x32 staggered, 1024 channels | Akhoundi reference validation | generated under `Spatial/output/` |
| CortexLab NP | 128-channel Neuropixels slice | Secondary/pending validation | `output/cortexlab_10pct/` |
| Yger / Zenodo 1205233 | 252-ch 16x16 MEA, 30 µm, 20 kHz | Dense-2D with **KS4 Th=13 pseudo-GT** (not juxta labels) | HDD `zenodo_1205233_work/spatial_npz/`; see [KS4 pseudo-GT](docs/yger_ks4_pseudo_gt.md) |

MEArec files that share one spike-train seed but differ in SNR rendering are not
independent temporal or biological replicates. Dataset-specific caveats and
schemas are summarized in [Reproduction guide](docs/REPRODUCING.md#datasets).

## Result semantics

Spatial uses several kinds of accuracy that must not be mixed:

- **Oracle / GT-row** fits per-unit state with ground-truth labels and measures
  a representation or assignment ceiling.
- **SOM / discovered-row** learns clusters without GT unit rows; Hungarian
  mapping is used only for evaluation.
- **GT-event isolation** uses known event times but may still evaluate a
  low-bit candidate policy independently of detection.
- **Detected-event end-to-end** includes detection misses and false positives;
  few current DAC isolation experiments are in this category.

Likewise, `candidate recall`, final assignment accuracy, row comparisons,
template reads, logical payload bits, and physical energy/area are different
quantities. The required reporting rules are listed in
[Documentation index](docs/README.md#documentation-contract).

## Citation

The reference reproduction is based on:

> Akhoundi et al., “A Scalable 1024-Channel Ultra-Low-Power Spike Sorting Chip
> With Event-Driven Detection and Spatial Clustering,” IEEE JSSC, 2025.

The STAR-Mem extensions in this repository are ongoing research. Cite the
specific protocol/result document and its evidence boundary when reporting a
new branch; do not cite a best observed point as a selected configuration when
its formal gate failed.
