# Causal Temporal-Sketch Pilot Provenance

Status: **captured and independently verified on 2026-07-26
(Asia/Hong_Kong)**

This document binds the completed four-recording temporal-sketch development
pilot to its exact working-tree implementation, input bytes, generated
artifacts, protocol contracts, and recomputed negative gate. The implementation
files are not claimed to be present in the base Git commit. Reproducibility is
provided by:

```text
verified repository base revision
+ complete content-addressed runtime Python source snapshot
+ exact four-file input manifest
+ deterministic output hashes
```

## 1. Repository and implementation snapshot

The runner verified the live repository base before loading data:

```text
repository base revision:
63ff7d2b84d7b2138887714e96db91071fd1dc0c
```

The actual implementation is an uncommitted research working tree. The
selection artifact therefore records every Python source file under:

```text
algorithms/
data/
evaluation/
experiments/
```

The snapshot is a static, exact superset of all runtime package sources,
including every package `__init__.py`. A regression test recursively
enumerates these directories and fails if the frozen set omits or adds a
`.py` file.

```text
runtime source files: 55
runtime source bytes: 883,034
canonical source-map SHA-256:
3edbc821097d1d7bb8f86d34293c90015a31d4be47cb15bdbd838a25fe558072

manifest: docs/causal_temporal_sketch_source.sha256
manifest SHA-256:
99ccd1d5035f363dfd14ce96ec133dd7405004278ac0cb3914915e2eb942ec79
```

The canonical source-map digest hashes sorted compact JSON over the
`relative path -> file SHA-256` map. The manifest digest hashes the textual
`sha256sum` file, so the two values are intentionally different while
describing the same 55 file digests.

Verify from the `Spatial` root:

```bash
sha256sum -c docs/causal_temporal_sketch_source.sha256
```

The four focused test files at final validation were:

| Test file | Bytes | SHA-256 |
|---|---:|---|
| `tests/test_causal_temporal_sketch.py` | 13,712 | `97ca3cd12a7137513bfc1523df1d42c019c3d353f8c3dc2660ffffa4dc643abe` |
| `tests/test_causal_temporal_baselines.py` | 8,950 | `3c461f082577c2ae9ee6a40ceb62a94b89c8728af5951f351a73a7f2d854bfe5` |
| `tests/test_causal_temporal_selection.py` | 28,049 | `e4b0e8f038869d258da7b5360ea0e6e1cd9e76c7f80697195e4d1233f054d7b6` |
| `tests/test_causal_temporal_sketch_experiment.py` | 21,303 | `eef65d47b48d1b983fae3f2c87f6761f3a713ea6153e263ae1ac6d5ec73bca9f` |

They contain 67 focused tests. The final full repository result is:

```text
267 passed in 3.43 s
py_compile: passed
git diff --check: passed
```

## 2. Environment

```text
Python:       3.10.15
NumPy:        2.2.6
SciPy:        1.14.1
scikit-learn: 1.5.2
h5py:         3.16.0
environment:  /home/xinyuan/miniconda3/envs/spint
```

The input contract fixes:

```text
sampling frequency: 30,000 Hz
recording input: dataset.raw_data, channel × sample
filter: full-recording zero-phase Butterworth, order 3, 300–6000 Hz
alignment: ground-truth event timestamp
central channel:
  maximum absolute filtered peak over [t-15,t+15), smallest channel on tie
waveform: filtered central channel [t-15,t+49), 64 samples
sketch/teacher input: before per-event waveform normalization
legacy report input: per-event peak normalization, epsilon 1e-8
```

The frozen contract SHA-256 is:

```text
ac47cddda2eee622089311b669b7438954f73499d6b6b5a71eabf8ba079590e2
```

The runner validates the complete preparation provenance, event count,
recording input, sampling frequency, filter, alignment, central-channel rule,
window, raw route, and legacy route before fitting any model.

## 3. Exact inputs

The dedicated input manifest is:

```text
manifest: docs/causal_temporal_sketch_inputs.sha256
files: 4
total bytes: 1,289,593,272
manifest SHA-256:
9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13
```

| Family | Input | Bytes | SHA-256 |
|---|---|---:|---|
| HJ | `drift16c_600s_11_filtered_gt.npz` | 216,390,108 | `b8d2b3cf8ea9b9730750bffcdb3b338179ff04e826e60670e5bd758360290c72` |
| HJ | `static16c_600s_11_filtered_gt.npz` | 216,263,256 | `537b0904b2f03643887723ef5fb53f5aaab129195f5187b454fcce9357e68644` |
| MEArec | `rec_v1_units10_snr5_seed202601.npz` | 428,329,398 | `53049aeae5d4aa1d67c8438a122028465926ae4b454e4babbe0eb9af92f8ff3c` |
| MEArec | `rec_v1_units20_snr5_seed202601.npz` | 428,610,510 | `db9e9434334942fce8ed6963cb561f5adacca4e4e38b70dd9aeccca5905100d9` |

The manifest hash is also the frozen four-line subset digest embedded in every
result row and in the selection artifact. Each file contributes exactly its
first 60 seconds.

Input loading is fail-closed:

1. discovery resolves the input with `resolve(strict=True)`;
2. discovery checks the locked SHA-256;
3. the lazy loader captures that same resolved path;
4. the resolved file is hashed immediately before load;
5. the same path is hashed again after load; and
6. mutation before or during loading aborts the pilot.

Tests cover catalog-path binding, symlink retargeting, mutation before load,
and mutation during load.

Verify:

```bash
sha256sum -c docs/causal_temporal_sketch_inputs.sha256
```

## 4. Exact command

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
MPLCONFIGDIR=/tmp \
/home/xinyuan/miniconda3/envs/spint/bin/python \
  -m Spatial.experiments.run_causal_temporal_sketch --pilot
```

The command completed in approximately 15.5 seconds and printed:

```text
pilot gate_pass=False
manifest_sha256=64766e86d8f142119be2ff4d0cef6398d21e01d52f43d096aa96145c6c5dfe4b
confirmation was not run and is not implemented by this entry point
```

The entry point accepts only `--pilot`; duration is fixed to exactly 60
seconds. It has no confirmation execution path.

## 5. Exact outputs

```text
manifest: docs/causal_temporal_sketch_outputs.sha256
files: 3
total bytes: 3,435,634
manifest SHA-256:
c23ed809830fe1f755fcd431dffec10455a8f790ed39b6abb28eb4e3dffd3dd3
```

| Artifact | Data rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `causal_temporal_sketch_hj_pilot.csv` | 30 | 775,121 | `a1bd98b4c2ef408d0436e3191440aac6419e9f94b1364496827e658be62a6432` |
| `causal_temporal_sketch_mearec_pilot.csv` | 30 | 840,805 | `82f5123ee1232c32caf4b34756e5166ea5f4921265aa26c790bb294d6fe761bd` |
| `causal_temporal_sketch_selection.json` | — | 1,819,708 | `64766e86d8f142119be2ff4d0cef6398d21e01d52f43d096aa96145c6c5dfe4b` |

The two CSVs have one header plus:

```text
2 recordings × 5 delays × 3 feature counts = 30 rows/family
```

The selection JSON contains all 60 rich audit rows, per-record and family gate
recomputations, CSV hashes, input source map, 55-file implementation snapshot,
temporal contract, full Fisher state, saturation populations, Level-1
arithmetic, Level-2 category accounting, and a null selection.

CSV writing performs canonical per-cell read-back before the manifest hashes
the files. Verify:

```bash
sha256sum -c docs/causal_temporal_sketch_outputs.sha256
```

## 6. Split, fitting, and memory contracts

For each recording:

- equal timestamps are kept on the same side of the chronological split;
- fit is approximately the first 50%, calibration the next 25%, and test the
  final 25%;
- the spatial quantizer, centroids, and initial radii use fit rows;
- p99.9 spatial radii are replaced from calibration rows;
- a fitted unit absent from calibration retains its fit-derived radius and is
  flagged;
- signed waveform scale, Fisher statistics, selected dyadic filters, and unit
  templates use fit rows only;
- only fit units with at least ten rows configure the scale and Fisher
  selection;
- rare units still receive templates but cannot configure scale, Fisher
  statistics, or the configuration-relevant saturation audit; and
- test labels are used only for metrics and gate computation.

The signed sample code is `[-15,15]`; two’s-complement `-16` is unused. Scale
is the smallest power of two no smaller than the fit-eligible absolute p99.9.
Sample rounding is half away from zero, dyadic normalization is integer
half-away-from-zero, and integer templates use signed round-to-nearest-even.

The Level-1 source is:

```text
D=9, B=5, all rows, integer L1, calibration p99.9 radius
logical row payload: 45-bit centroid + 9-bit radius = 54 bits/unit
```

Level-2 candidate semantics are:

```text
C=0: reuse the Level-1 all-row argmin; zero temporal-template reads
C=1: direct candidate decision; zero temporal-template reads
C>1: read every distinct candidate's temporal template and compute SAD
```

The source builds one integer `N×U` distance matrix. Candidate membership and
the all-event argmin cache use that same state. Selection independently
recomputes every Level-1 arithmetic identity and the C0/C1/C>1 nested temporal
accounting; it does not accept a self-reported boolean as sufficient evidence.

## 7. Independent gate recomputation

After the run, the main agent independently loaded the deterministic JSON and
executed:

```text
validate_development_rows(record_audit_rows)
evaluate_development_gate(record_audit_rows)
```

Observed audit:

```text
HJ rows: 30
MEArec rows: 30
four locked dataset IDs exact: true
recorded CSV hashes equal actual files: true
recorded gate equals recomputed gate: true
gate_pass: false
candidate_count: 0
selected_configuration: null
implementation source files: 55
implementation source digest:
  3edbc821097d1d7bb8f86d34293c90015a31d4be47cb15bdbd838a25fe558072
temporal input contract:
  ac47cddda2eee622089311b669b7438954f73499d6b6b5a71eabf8ba079590e2
```

Every input file was also hashed against the locked dataset-ID map before and
around loading. All 15 configurations pass the family template-bit condition,
but all fail the temporal accuracy predicates.

## 8. Authorization boundary

No output authorization anchor was generated. This is intentional:

- the selection is null;
- the gate failed;
- the pilot CLI cannot run confirmation;
- a future confirmation loader requires a positive recomputed selection;
- it also requires a separately reviewed, Git-tracked, clean anchor matching
  the selection JSON, both CSV hashes, the four input hashes, the temporal
  contract, gate contract, repository base, and 55-file implementation
  snapshot.

The current negative artifact cannot authorize the remaining 10 HJ and 18
MEArec recordings.

