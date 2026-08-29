# Reproducing Spatial Experiments

This guide covers workspace layout, environment checks, datasets, test tiers,
artifact handling, and the reporting rules shared by Spatial experiments.
Exact commands for a frozen result remain in that result's document.

## Working-directory convention

Run Python modules from the parent `SNN_SpikeSorting` root:

```bash
cd /home/xinyuan/SNN_SpikeSorting
```

The code imports modules as `Spatial.*`, while data loaders resolve sibling
directories such as `new_datasets/` and `synthetic_spike_dataset/` relative to
that parent workspace. Running a script by filename from inside `Spatial/` can
produce different import and path behavior; prefer `python -m` commands.

## Environment

The repository currently has no `pyproject.toml`, `requirements.txt`, or
locked environment file. The core code uses:

- Python with modern type-hint support;
- `numpy`;
- `scipy`;
- `h5py`;
- `matplotlib` for plotting/generation paths;
- `PyYAML` for configuration paths;
- `pytest` for validation.

MEArec data generation additionally imports `MEArec` and `MEAutility`. Some
reference or sibling-workspace paths may have their own dependencies.

On the current workstation the usual environment is:

```bash
conda activate spint
```

Validate the core imports before a data-backed run:

```bash
python -c "import h5py, matplotlib, numpy, scipy, yaml; print('core imports OK')"
python -m pytest --version
```

For a paper-facing rerun, record at least the Python version and package
versions in the provenance document. Do not infer reproducibility from the
environment name alone.

## Datasets

### Hybrid Janelia

Expected prepared location:

```text
new_datasets/hybridjanelia/
├── catalog.json
├── static16c_600s_11_filtered_gt.npz
├── ... other static scenes ...
└── ... drift scenes ...
```

`Spatial.data.loader.load_hybrid_janelia()` defaults to
`static16c_600s_11_filtered_gt.npz`. Corpus runners use `catalog.json` when it
exists and fall back to the default scene otherwise.

The preparation/catalog entry point is:

```bash
python -m Spatial.data.prepare_hybrid_janelia --help
```

Prepared files contain the raw or filtered recording, electrode geometry, GT
event times, and GT unit IDs. Historical files may expose different key names;
use `Spatial.data.loader` or an experiment's shared preparation helper.

### MEArec v1 / GeoOsort-format NPZ

Default corpus location:

```text
synthetic_spike_dataset/generated/geoosort_npz_v1/
```

The default suite spans unit count, SNR, temporal/drift configuration, and
seed. Multiple SNR renderings can share the same underlying spike-train seed.
They must not be counted as independent temporal or biological replicates.
State whether an aggregate is file-weighted, event-weighted, seed-weighted, or
recording-unweighted.

### Legacy 1024-channel grid

`Spatial.data.generate_2d_grid` creates the staggered 32x32 reference dataset.
Generation is storage- and compute-intensive and has optional simulator
dependencies; it is not part of the default test suite. The legacy result
validates the Akhoundi reference path, not the current relative D9/B5 pipeline.

### CortexLab Neuropixels slice

The loader's default path is:

```text
output/cortexlab_10pct/slices/dataset_1_20141202_228s_ksgt_int16.npz
```

This family is secondary in current paper-facing experiments and should not be
silently substituted for the missing dense-2D validation.

### Yger 252-ch MEA, Kilosort4 Th=13 pseudo-GT

HDD-only Spatial input for `20160415_patch2`. Labels are KS4 clusters, not
juxta. Protocol, command, and evidence boundary:
[`yger_ks4_pseudo_gt.md`](yger_ks4_pseudo_gt.md).

```bash
python -m Spatial.data.export_yger_ks4_pseudo_gt
# then
python -c "from Spatial.data.loader import load_yger_ks4; ds=load_yger_ks4(duration_s=60); print(ds.name, ds.n_channels, ds.unit_ids.size)"
```

## Test tiers

### Focused tests for the newest branch

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  Spatial/tests/test_spatial_footprint.py \
  Spatial/tests/test_shift_match.py \
  Spatial/tests/test_two_stage_assign.py -q
```

### Full repository tests

These tests do not require the full recording corpus unless a test explicitly
constructs a temporary fixture:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests -q
```

Disabling auto-loaded external pytest plugins makes the test environment less
dependent on unrelated packages installed in the active environment.

The temporal-sketch provenance tests are intentionally fail-closed. They check
the exact frozen base revision and the complete sorted runtime Python source
inventory. An active branch that adds a new algorithm/runner or advances HEAD
can therefore pass its focused algorithm tests while those frozen-snapshot
tests fail. Treat that as a provenance mismatch to explain or re-freeze under
an authorized rerun; do not weaken the checks merely to make the active working
tree green.

### Syntax validation

When a change touches many runners but a full corpus rerun is not warranted:

```bash
python -m compileall -q \
  Spatial/algorithms Spatial/data Spatial/evaluation Spatial/experiments
```

Syntax validation is not a substitute for unit tests or result-gate
recomputation.

## Common experiment entry points

### Reference and feature baselines

```bash
python -m Spatial.experiments.run_spatial_sorting --help
python -m Spatial.experiments.run_spatial_upper_bound \
  --dataset hj --duration 60
python -m Spatial.experiments.run_two_stage_upper_bound --duration 60
python -m Spatial.experiments.run_two_stage_unsupervised --duration 60
```

### Relative-offset and shifted matching

```bash
python -m Spatial.experiments.run_relative_footprint_pilot \
  --duration 60 --quiet
python -m Spatial.experiments.run_shift_cim_pilot \
  --duration 60 --quiet
python -m Spatial.experiments.run_shift_conv_bit_sweep \
  --duration 60 --quiet
```

Expected result locations:

```text
Spatial/output/relative_footprint_pilot/
Spatial/output/shift_cim_pilot/
Spatial/output/shift_conv_bit_sweep/
```

### D9/B5 Level-1 search

```bash
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 95 --coverage 0.95 \
  --directory-neighbors 0 7 \
  --directory-coverages 0.95 0.99 1.0

python -m Spatial.experiments.run_geometric_prefilter_sweep --methods ALL
```

Many older branches intentionally encode a frozen grid, development corpus,
selection rule, and confirmation authorization boundary. Consult the matching
plan/result document before changing command-line parameters.

### Level-2 refinement

```bash
python -m Spatial.experiments.run_compact_waveform_refinement \
  --family all --duration 60 \
  --bits 4 5 --taps 8 16 24 --percentiles 95 99.9

python -m Spatial.experiments.run_pair_conditioned_waveform \
  --family all --duration 60 --bits 5 \
  --taps 4 8 12 16 --percentiles 95 99.9

python -m Spatial.experiments.run_causal_temporal_sketch --pilot
```

The temporal pilot has a fail-closed selection boundary. Its recorded
selection is null; do not run or describe confirmation as authorized without a
new pre-registered protocol.

See [Experiment index](../experiments/README.md) for the full runner catalog.

## Split and leakage rules

1. Keep simultaneous events as separate rows.
2. Keep equal timestamps on the same side of every chronological boundary.
3. Fit quantizers, normalization scales, centroids, ranges, thresholds,
   templates, masks, weights, selected taps, and graph edges only on their
   registered partition.
4. Use calibration only for registered calibration quantities and selection;
   do not refit representation state on test.
5. Do not tune on a confirmation corpus after a formal development gate has
   failed or selected `null`.
6. State whether row labels are GT, oracle, discovered, delayed GT, or
   self-predicted.
7. State whether events are GT supplied or detected by the evaluated signal
   path.

## Filtering, alignment, and latency

Most paper-facing DAC isolation runners call the shared preparation path in
`experiments/spatial_cim_common.py`, which performs full-recording bandpass
filtering before chronological feature fitting. This is zero-phase/offline
component isolation, not a causal streaming front end.

When a causal filter primitive is tested, record:

- filter state initialization;
- warm-up/guard handling;
- event-alignment policy;
- pre- and post-event samples used;
- future-sample acquisition latency separately from compute latency.

Implementing one forward-only filter function does not make downstream
centroids, templates, directories, or updates causal.

## Metrics and aggregation

For candidate search, report at minimum:

- integer true-unit hit count and denominator;
- candidate recall;
- `C=0`, `C=1`, and `C>1` counts/fractions;
- mean or distribution of candidate count;
- total row comparisons including fallback;
- distinct template reads and full-scan baseline;
- aggregation estimator.

Use event-weighted total reads for a corpus-wide traffic-reduction factor unless
the result explicitly registers another estimator. A mean of per-recording
reduction factors is a different statistic and must be labeled.

For clustering with detected events, `evaluation/metrics.py` performs temporal
matching and Hungarian label mapping. Its accuracy denominator is the number
of GT spikes. Plain GT-row assignment accuracy from component experiments is a
different metric.

## Output and provenance

Generated artifacts belong under `Spatial/output/`; most are gitignored. For
every result used in a paper claim, add or update a tracked document under
`Spatial/docs/` containing:

1. hypothesis and status;
2. exact dataset identities and input hashes when frozen;
3. event/row truth boundary;
4. split, filtering, and aggregation policy;
5. exact command and code/source revision or source hashes;
6. output paths and output hashes;
7. integer-count gate recomputation;
8. selected configuration or an explicit `null`;
9. confirmation authorization/disposition;
10. limitations and forbidden claims.

Verify an existing manifest from the parent workspace root, for example:

```bash
sha256sum -c Spatial/docs/dac_experiment_source.sha256
sha256sum -c Spatial/docs/dac_experiment_inputs.sha256
```

Only run a manifest command whose companion provenance document describes the
expected working-tree state. A source manifest may intentionally lock an older
snapshot and fail after legitimate code changes.

## Before reporting a rerun

Use this checklist:

- tests for the touched algorithm and accounting path pass;
- command, working directory, environment, and output path are recorded;
- no train/calibration/test state leaks across the registered boundary;
- aggregate metrics can be reconstructed from integer per-record counts;
- local gate and system gate are stated separately;
- logical bits/accesses are not translated into physical PPA;
- a failed formal gate remains failed even if one unregistered point looks
  attractive;
- the parent DAC2027 evidence ledger and paper numbers are updated only after
  the Spatial evidence record is frozen.
