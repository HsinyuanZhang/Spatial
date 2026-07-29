# Adaptive Weighted-L1 5-bit Pilot Provenance

Status: **captured and verified on 2026-07-26 (Asia/Hong_Kong)**

This snapshot locks the four-recording power-of-two weighted-L1 development
pilot separately from the earlier full-corpus and masked-L1 result manifests.
The output directory is gitignored, and the code is in the current uncommitted
research worktree, so hashes—not an implied clean checkout—identify the exact
files behind this result.

## 1. Code state

Base repository revision:

```text
63ff7d2b84d7b2138887714e96db91071fd1dc0c
```

The four algorithm, runner, and test files are locked by:

```text
manifest: docs/adaptive_weighted_5bit_source.sha256
files: 4
total bytes: 92,721
manifest SHA-256:
3f8f1baa41daae0ff5d0e9d5e6a60abae8f5677d7f5b182110492abf91326f1e
```

The manifest contains:

- `algorithms/adaptive_weighted_range_search.py`;
- `experiments/run_adaptive_weighted_5bit.py`;
- `tests/test_adaptive_weighted_range_search.py`; and
- `tests/test_adaptive_weighted_5bit_experiment.py`.

The shared `algorithms/__init__.py` export is intentionally excluded. The
runner imports the concrete module directly, so unrelated future package
exports do not silently invalidate this snapshot.

Verify from the `Spatial` repository root:

```bash
sha256sum -c docs/adaptive_weighted_5bit_source.sha256
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

The pilot retains the existing full-recording zero-phase filtering path.
Chronological model fitting does not make the signal preprocessing causal.

## 3. Exact development inputs

The four recordings are a fixed subset of the 32-file corpus locked by
[`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256):

```text
complete manifest files: 32
complete manifest bytes: 11,184,999,885
complete manifest SHA-256:
eb21821769246ff2110197e393b8817418ac910048e78c908c7525645eb37d61
```

Only these entries were loaded:

| Family | Input | Bytes | SHA-256 |
|---|---|---:|---|
| HJ | `drift16c_600s_11_filtered_gt.npz` | 216,390,108 | `b8d2b3cf8ea9b9730750bffcdb3b338179ff04e826e60670e5bd758360290c72` |
| HJ | `static16c_600s_11_filtered_gt.npz` | 216,263,256 | `537b0904b2f03643887723ef5fb53f5aaab129195f5187b454fcce9357e68644` |
| MEArec | `rec_v1_units10_snr5_seed202601.npz` | 428,329,398 | `53049aeae5d4aa1d67c8438a122028465926ae4b454e4babbe0eb9af92f8ff3c` |
| MEArec | `rec_v1_units20_snr5_seed202601.npz` | 428,610,510 | `db9e9434334942fce8ed6963cb561f5adacca4e4e38b70dd9aeccca5905100d9` |

They total `1,289,593,272` bytes. The SHA-256 of the four complete input
manifest lines, in the path order above and with a final newline, is:

```text
9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13
```

Every file contributes its first 60 seconds. These recordings are development
data and are excluded from any authorized confirmation estimator.

## 4. Generated outputs

The two CSVs and deterministic selection JSON are locked by:

```text
manifest: docs/adaptive_weighted_5bit_outputs.sha256
files: 3
total bytes: 671,577
manifest SHA-256:
34d4be9cfcda4d4816a64fcf5ba0b75e93589151cb891bbe16c486168e7e00a5
```

| Artifact | Data rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `adaptive_weighted_5bit_pilot_hj_60s.csv` | 30 | 272,630 | `0d1d098c0e0c2c39cdec625255d12be99409bff949d1e4a3378fea1359bd0f2b` |
| `adaptive_weighted_5bit_pilot_mearec_60s.csv` | 30 | 362,485 | `1c4d81c7cf82cf8877d830a79cea290f6f0964a0b9d6031342ffb58e9b6eddad` |
| `adaptive_weighted_5bit_pilot_selection_60s.json` | — | 36,462 | `abfbf3118b6e37de95946bb2d7b66964bdc4a060d3968fdea3a820b892518a37` |

Each CSV contains one header plus 30 data rows: two recordings × five fixed
weight configurations × three radius percentiles. The JSON contains the
complete gate audit and CSV hashes. No confirmation artifact exists.

Verify while the gitignored output tree is present:

```bash
sha256sum -c docs/adaptive_weighted_5bit_outputs.sha256
```

## 5. Exact command

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_weighted_5bit \
  --pilot --duration 60 --quiet
```

The completed runner reported:

```text
Pilot formal gate pass: False; candidates=0
Pilot selection manifest:
  output/adaptive_weighted_5bit/adaptive_weighted_5bit_pilot_selection_60s.json
Final 0.99/4x report-only pass: False; configurations=0
No confirmation sweep was started automatically.
```

The `--pilot` mode fixes both families, the four development IDs, 60-second
duration, `D=9`, `B=5`, all-row activation, grid, split, and gate. It cannot
accept a selection manifest and never invokes confirmation.

## 6. Estimator and artifact boundary

For every recording:

- event order is split 50/25/25 without separating equal timestamps;
- the quantizer is the fixed unit-interval 5-bit mapping;
- centroid, rank, weight, shift code, and waveform template state use fit rows;
- only the radius uses calibration rows;
- test labels are metrics-only;
- the candidate interval is a two-sided 95% Wilson interval with
  `z=1.959963984540054`;
- `C=0` invokes an unweighted all-centroid spatial fallback and reads no
  waveform template;
- `C=1` reads no waveform template; and
- `C>1` reads `C` distinct fit-only 320-bit templates.

Family candidate recall is the ratio of summed hits to summed test events.
Family traffic reduction is the ratio of summed global bits to summed actual
bits. Gate traffic predicates use integer bit counts and exact cross-products,
not rounded display factors.

The JSON schema is:

```text
adaptive_weighted_5bit_pilot_selection_v1
```

It records the base revision, duration, development IDs, split
indices/timestamps, fixed input-subset digest, fixed grid, gate rules, all
twelve nonuniform audit records, final-target report, selected configuration,
and both CSV hashes.

The JSON's input-digest description refers to the same four complete
`dac_experiment_inputs.sha256` lines in the order previously recorded by the
masked-pilot provenance. This is only a description of line order; the locked
digest and the four source lines above are identical and independently
verified here.

## 7. Independent final audit

After the final runner hardening, the main-agent audit verified:

```text
HJ data rows: 30
MEArec data rows: 30
datasets per family: 2
fixed grid exact: true
all rows active: true
logical identity fields per row: 5
all 60 rows satisfy all identities: true
selection JSON CSV hashes match files: true
nonuniform configuration audits: 12
recorded/recomputed pilot gate: false / false
recomputed pilot candidates: 0
recorded/recomputed final 0.99/4x gate: false / false
selected configuration: null
confirmation files: none
```

Authorization-only checks were strengthened after the pilot values were
generated:

- direct confirmation configuration validation now rejects the uniform
  reference; and
- CSV hashes in a supplied selection manifest must be 64 hexadecimal
  characters, not merely strings of length 64;
- the embedded pilot gate must have passed with at least one candidate;
- the top-level selection must exactly equal the gate's selected
  configuration; and
- all twelve fixed nonuniform configuration audits must be present, with the
  selected audit marked passing.

These checks do not change any fitted state, candidates, metrics, CSV, or
selection JSON. The source manifest locks the hardened code; the output
manifest locks the previously generated numeric artifacts.

Targeted weighted tests pass `25/25`, including bit boundaries, uniform
semantic parity, split isolation, accounting, gate behavior, manifest
tampering, manual-selection injection, full audit-grid validation, and the
no-selection confirmation block. The repository-wide test count is recorded
in the final workspace audit rather than embedded in these output hashes.

## 8. Confirmation disposition

The protocol requires one nonuniform selected configuration before the
remaining ten HJ and eighteen MEArec recordings can be loaded. Since
`selected_configuration` is `null`:

- the selection manifest cannot authorize confirmation;
- no confirmation configuration was manually substituted;
- no confirmation command was run;
- no confirmation CSV was written; and
- no directory-assisted weighted search was launched.
