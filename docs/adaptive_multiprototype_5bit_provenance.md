# Adaptive Multi-Prototype 5-bit Provenance Snapshot

Status: **captured and verified on 2026-07-26 (Asia/Hong_Kong)**

This snapshot is separate from
[`dac_experiment_provenance.md`](dac_experiment_provenance.md). It locks the
later bounded multi-prototype experiment without changing the manifests behind
the earlier single-range, dual, indexed, online-graph, and waveform studies.
This separation matters because `output/` is gitignored and the new source is
still in an uncommitted working tree.

## 1. Code state

Base repository revision:

```text
63ff7d2b84d7b2138887714e96db91071fd1dc0c
```

Six new algorithm, runner, and test files are locked by:

```text
manifest: docs/adaptive_multiprototype_5bit_source.sha256
files: 6
total bytes: 67,533
manifest SHA-256:
80ec8ce75248701f1a2bea6ae76f8d87a23f23860edad88274349e5e9b650cc2
```

The manifest contains:

- the fixed-5-bit bounded multi-prototype search core;
- the forward-only stateful SOS filter primitive;
- the cross-dataset static runner; and
- the three associated test modules.

The shared `algorithms/__init__.py` export file is intentionally excluded: the
runner imports its core module directly, and the export surface may grow as
later research branches are added without changing this result.

Verify from the `Spatial` repository root:

```bash
sha256sum -c docs/adaptive_multiprototype_5bit_source.sha256
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

The result runner uses the historical full-recording zero-phase filtering
path. The separately tested SOS primitive is not silently substituted into
these CSVs.

## 3. Input corpus

This experiment uses exactly the same 12 HJ and 20 MEArec NPZ files already
locked by [`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256):

```text
files: 32
total bytes: 11,184,999,885
manifest SHA-256:
eb21821769246ff2110197e393b8817418ac910048e78c908c7525645eb37d61
```

The family sub-manifest digests remain:

| Family | Files | Bytes | Path-sorted line digest |
|---|---:|---:|---|
| Hybrid Janelia | 12 | 2,625,565,809 | `4a223ad5b81e3aaa576bef0a12eb733381ba69c5773b9a3d3178cc92016dc230` |
| MEArec v1 geoosort | 20 | 8,559,434,076 | `25aabaff0daa603a2522d658e5240bf7c5f0d3b289f2c27bc1662885ded37efd` |

Every recording contributes its first 60 seconds. Repeated-SNR MEArec
renderings share temporal spike-train seeds and are not interpreted as
independent biological or temporal replicates.

Verify the inputs:

```bash
sha256sum -c docs/dac_experiment_inputs.sha256
```

## 4. Generated outputs

The two final fallback-inclusive CSVs are locked by:

```text
manifest: docs/adaptive_multiprototype_5bit_outputs.sha256
files: 2
total bytes: 2,377,072
manifest SHA-256:
fc2dee025a377dea58ae7adfc8e572f54867e4b246ff0ff03ea9a55ac146420a
```

| Family | Data rows | Bytes | CSV SHA-256 |
|---|---:|---:|---|
| HJ | 1,152 | 893,124 | `8e92a5aace19a1de4a71c739c680bca4526b7e651af2b4b4fa0c9460b7bd3f78` |
| MEArec | 1,920 | 1,483,948 | `c05626527c30ad58446a8473c927478fdafda46c22c1b501a56cb8e17ae0a45e` |

Verify while the gitignored `output/` tree is present:

```bash
sha256sum -c docs/adaptive_multiprototype_5bit_outputs.sha256
```

## 5. Exact command

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_multiprototype_5bit \
  --family all --duration 60 --bits 5 \
  --prototypes 1 2 3 4 \
  --modes temporal farthest_l1 \
  --radius-percentiles 95 99 99.9 \
  --directory-coverages 0.95 0.99 1.0 \
  --quiet
```

The runner fixes `D=9`, `B=5`, and `1≤P≤4`. It rejects a different descriptor
width, bit width, or unbounded prototype count.

## 6. Estimator and selection boundary

For family summaries:

- candidate recall, mean candidate count, active rows, and final accuracy are
  weighted by `n_test_gt_rows`;
- waveform traffic reduction is
  `sum(global_all_unit_waveform_template_bits) /
  sum(actual_waveform_template_bits_read)`;
- row/storage/pointer values labeled “per recording” use an unweighted mean
  over recordings;
- total Level-1 comparisons are summed across recordings; and
- the complete 96-configuration grid is reported without selecting a final
  test-tuned configuration.

Every output row contains:

```text
selection_protocol =
  exploratory_test_grid_no_final_configuration_selection

prototype_directory_pointer_accounting_scope =
  prototype_row_id_payload_only_excludes_periphery_valid_decode
```

The experiment is an exploratory static representation test. A displayed
frontier point is not claimed as a separately validated operating point.

## 7. Bookkeeping correction and invariants

The first complete run established that no joint recall/traffic point exists.
A read-only audit then found that the final-accuracy teacher's empty-list
spatial fallback was not included in total Level-1 comparisons. The runner was
corrected and both families were rerun from the source snapshot above.

The corrected CSVs expose:

```text
primary_prototype_row_comparisons_total
spatial_fallback_level1_row_comparisons_total
total_level1_row_comparisons
```

Every row satisfies the exact identity:

```text
primary + spatial fallback = total Level-1 comparisons
```

Across the entire sweep grid:

| Family | Primary comparisons | Fallback comparisons | Total comparisons | Fallback share |
|---|---:|---:|---:|---:|
| HJ | 110,938,824 | 305,532 | 111,244,356 | 0.2747% |
| MEArec | 178,073,349 | 318,060 | 178,391,409 | 0.1783% |

The fixed candidate/traffic column sequences were hashed before and after the
correction and were identical:

```text
HJ:     b8dd4a8687de388623dfc5cbef705a8c76b341c0270d12f6ed002e3c181262bb
MEArec: cf3219c745b59887d65a3c5cb782cbf05a7315d3ea4e9dcbe7cb7aa89c8a7517
```

Thus the correction changes hardware comparison accounting, not candidate
recall, unit de-duplication, final candidate sets, or 320-bit waveform traffic.

The `P=1` all-row results also match the earlier one-row per-unit L1 outputs
for every recording at p95, p99, and p99.9. This is checked both by unit tests
and by the cross-dataset result audit.

## 8. Verification boundary

At snapshot time:

```text
pytest: 143 passed
targeted multi-prototype/core tests: 20 passed
py_compile: passed
git diff --check: passed
source/output/input checksum verification: passed
```

This snapshot makes the bounded-row negative result auditable. It does not
turn the experiment into:

- detected-event or discovered-row end-to-end sorting;
- a causal raw-signal candidate pipeline;
- a train-only-selected positive configuration;
- a high-local-unit-density 1024-channel validation; or
- synthesized/macro-measured SRAM, CAM, CiM, area, latency, or energy evidence.
