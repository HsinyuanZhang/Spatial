# DAC Experiment Provenance Snapshot

Status: **captured and verified on 2026-07-26 (Asia/Hong_Kong)**

This snapshot locks the source, inputs, and generated CSVs behind the current
5-bit two-level-memory evidence. It exists because `output/` is intentionally
gitignored and because the new DAC files are still in an uncommitted working
tree. Git `HEAD` alone is therefore not sufficient provenance.

## 1. Code state

Base repository revision:

```text
63ff7d2b84d7b2138887714e96db91071fd1dc0c
```

The unchanged tracked dependencies are identified by that revision. The eight
new or materially changed DAC algorithm/runner files are locked separately by:

```text
manifest: docs/dac_experiment_source.sha256
files: 8
total bytes: 194,792
manifest SHA-256:
99ce67871e089dcff1554b5b4f087a263385899c393795b6e1191ee3786815a5
```

The source manifest includes:

- the 5-bit adaptive range-search core;
- adaptive, dual, and certified Level-1 runners;
- compact and pair-conditioned waveform runners; and
- static and online indexed-confusion runners.

The current Range-box builder is the nested-prefix implementation: each unit's
training codes are stably ordered by integer centroid L1 distance, and each
requested coverage uses the bounds of a nested prefix. All box-derived
adaptive, dual, and certified CSVs were rerun from this source snapshot after
the access-accounting patch.

Verify the source snapshot from the repository root:

```bash
sha256sum -c docs/dac_experiment_source.sha256
```

## 2. Environment

```text
Python 3.10.15
NumPy 2.2.6
SciPy 1.14.1
scikit-learn 1.5.2
h5py 3.16.0
environment: /home/xinyuan/miniconda3/envs/spint
```

The current signal preparation performs full-recording, zero-phase Butterworth
filtering with SciPy `filtfilt`. Chronological splits isolate fitted labels and
state, but this is an offline preprocessing path and is not a causal RTL input
pipeline.

## 3. Input corpus

The exact 32 NPZ inputs and their per-file hashes are stored in
[`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256).

| Family | Files | Total bytes | Manifest SHA-256 |
|---|---:|---:|---|
| Hybrid Janelia | 12 | 2,625,565,809 | `4a223ad5b81e3aaa576bef0a12eb733381ba69c5773b9a3d3178cc92016dc230` |
| MEArec v1 geoosort | 20 | 8,559,434,076 | `25aabaff0daa603a2522d658e5240bf7c5f0d3b289f2c27bc1662885ded37efd` |
| Combined | 32 | 11,184,999,885 | `eb21821769246ff2110197e393b8817418ac910048e78c908c7525645eb37d61` |

The combined digest is the SHA-256 of the path-sorted standard
`sha256sum` lines in `dac_experiment_inputs.sha256`. Each experiment uses the
first 60 seconds of the corresponding input recording. The MEArec set contains
two temporal seeds for each unit-count design rendered at several SNRs; those
renderings are not treated as independent spike-train replicates.

## 4. Generated-result snapshot

The exact 20 paper-facing CSVs are locked by
[`dac_experiment_outputs.sha256`](dac_experiment_outputs.sha256):

```text
files: 20
manifest SHA-256:
8a3e36b8cbf4c8143e3772505d0184dd4ea42050699b77f14cfb8325f6dbfd47
```

This includes:

- adaptive p95, p99, and p99.9 HJ/MEArec CSVs;
- uncertified and certified dual-range CSVs;
- compact and extended shared-tap waveform CSVs;
- static indexed-confusion CSVs;
- block-causal graph-state CSVs with ordered pointer/count rewrite fields; and
- pair-conditioned waveform CSVs.

Verify the generated files while `output/` is present:

```bash
sha256sum -c docs/dac_experiment_outputs.sha256
```

The final monotonic-box/access-accounting rerun produced:

| Experiment | HJ CSV rows | MEArec CSV rows |
|---|---:|---:|
| Adaptive p95 | 1,080 | 1,800 |
| Adaptive p99 | 1,080 | 1,800 |
| Adaptive p99.9 | 72 | 120 |
| Dual | 144 | 240 |
| Certified dual | 348 | 580 |
| Online confusion graph | 1,296 | 2,160 |

The candidate, waveform-traffic, final-accuracy, nested, and certification
metrics were unchanged by the accounting patch. The new CSV columns expose
previously omitted Level-1 fallback/outer accesses and ordered pointer-list
writes.

## 5. Commands and estimators

Exact commands are maintained next to the interpreted results:

- [`adaptive_5bit_range_search.md`](adaptive_5bit_range_search.md)
- [`compact_waveform_refinement.md`](compact_waveform_refinement.md)
- [`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md)

Aggregation is not silently mixed:

- the primary single-range table reports recording-unweighted means for recall
  and mean candidate count;
- its global waveform-read factor is the ratio of event-weighted total global
  reads to total actual reads;
- dual, certified, indexed, online, compact, and pair-conditioned summaries are
  event-weighted unless a table explicitly says otherwise; and
- singleton precision is weighted by the number of singleton decisions, not by
  the number of recordings or all queries.

No paper-facing result should be regenerated after changing a source/input
hash without replacing this snapshot and rerunning the affected cross-dataset
experiments.

## 6. Verification boundary

At snapshot time:

```text
pytest: 117 passed
py_compile: passed for all eight DAC source files
git diff --check: passed
```

This provenance upgrades the current evidence to auditable algorithmic
component-isolation evidence. It does not turn it into:

- detected-event or discovered-row end-to-end sorting;
- a causal raw-signal implementation;
- a realistic high-local-unit-density 1024-channel evaluation; or
- synthesized SRAM/CAM/CiM area, latency, or energy evidence.
