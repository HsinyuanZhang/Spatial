# Adaptive Masked-L1 5-bit Pilot Provenance

Status: **captured and verified on 2026-07-26 (Asia/Hong_Kong)**

This snapshot locks the later four-recording masked-L1 development pilot
without modifying the manifests behind the earlier full-corpus single-range,
dual, indexed, online-graph, waveform, or bounded multi-prototype results.
This separation is necessary because `output/` is gitignored and the new
source remains in an uncommitted working tree.

## 1. Code state

Base repository revision:

```text
63ff7d2b84d7b2138887714e96db91071fd1dc0c
```

Four new algorithm, runner, and test files are locked by:

```text
manifest: docs/adaptive_masked_5bit_source.sha256
files: 4
total bytes: 84,521
manifest SHA-256:
6b025f061911e49b84e345ffa48f6ebe05220d4908b24d65bad799a294fa5963
```

The manifest contains:

- `algorithms/adaptive_masked_range_search.py`;
- `experiments/run_adaptive_masked_5bit.py`;
- `tests/test_adaptive_masked_range_search.py`; and
- `tests/test_adaptive_masked_5bit_experiment.py`.

The shared `algorithms/__init__.py` export is intentionally excluded. The
runner imports the core module directly, and a later repository export should
not silently invalidate this result snapshot.

Verify from the `Spatial` repository root:

```bash
sha256sum -c docs/adaptive_masked_5bit_source.sha256
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

The pilot uses the historical full-recording zero-phase filtering path. The
separately implemented causal SOS primitive is not substituted into these
CSVs.

## 3. Exact development inputs

The four pilot recordings are a fixed subset of the 12-HJ/20-MEArec corpus
locked by [`dac_experiment_inputs.sha256`](dac_experiment_inputs.sha256).
The complete 32-file manifest remains:

```text
files: 32
total bytes: 11,184,999,885
manifest SHA-256:
eb21821769246ff2110197e393b8817418ac910048e78c908c7525645eb37d61
```

Only these four entries were loaded:

| Family | Input | Bytes | SHA-256 |
|---|---|---:|---|
| HJ | `drift16c_600s_11_filtered_gt.npz` | 216,390,108 | `b8d2b3cf8ea9b9730750bffcdb3b338179ff04e826e60670e5bd758360290c72` |
| HJ | `static16c_600s_11_filtered_gt.npz` | 216,263,256 | `537b0904b2f03643887723ef5fb53f5aaab129195f5187b454fcce9357e68644` |
| MEArec | `rec_v1_units10_snr5_seed202601.npz` | 428,329,398 | `53049aeae5d4aa1d67c8438a122028465926ae4b454e4babbe0eb9af92f8ff3c` |
| MEArec | `rec_v1_units20_snr5_seed202601.npz` | 428,610,510 | `db9e9434334942fce8ed6963cb561f5adacca4e4e38b70dd9aeccca5905100d9` |

The four files total `1,289,593,272` bytes. The SHA-256 of their four complete
manifest lines in the path order shown above, including the final newline, is:

```text
9890283ef7e499f5848c5207e2455de403c0da63b93038aefddeb17ffa498e13
```

Every input contributes its first 60 seconds. These are development
recordings, not the disjoint confirmation set.

Verify the complete locked input corpus:

```bash
sha256sum -c docs/dac_experiment_inputs.sha256
```

## 4. Generated outputs

The two final fallback-inclusive pilot CSVs are locked by:

```text
manifest: docs/adaptive_masked_5bit_outputs.sha256
files: 2
total bytes: 564,591
manifest SHA-256:
3e727c53bbf1378e2af328b8b5a6a7b23efa2a62228ee8114fc4470322a5c59a
```

| Family | Data rows | Bytes | CSV SHA-256 |
|---|---:|---:|---|
| HJ | 54 | 255,380 | `3001effe29bca40e947a8ec0ab9a60333699fec80cd175ee8e0536bc0be9bc87` |
| MEArec | 54 | 309,211 | `e40e5e2dfe9bbd62bffe58c8d31047b0d00791157ecbf07d3daf0286334541a5` |

Each family has two recordings × 27 frozen all-row configurations. The files
contain one header plus 54 data rows. No `confirmation` output exists.

Verify while the gitignored output tree is present:

```bash
sha256sum -c docs/adaptive_masked_5bit_outputs.sha256
```

## 5. Exact command

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_masked_5bit \
  --pilot --duration 60 --quiet
```

The runner fixes `D=9`, `B=5`, the four named development recordings, all-row
activation, mask modes/sizes, radius percentiles, and the formal pilot gate.
`--pilot` never launches `--confirmation`.

The completed command printed:

```text
Pilot formal gate pass: False; candidates=0
Pilot frozen selection report: None
Final 0.99/4x report-only pass: False; configurations=0
No confirmation sweep was started automatically.
```

## 6. Estimator and selection boundary

For every recording:

- event order is split 50/25/25 into fit/calibration/test without splitting
  equal timestamps;
- centroids, masks, feature scores, and waveform teacher centroids use fit
  rows only;
- masked radii use calibration rows only;
- test labels are used only for metrics;
- candidate recall is candidate hits divided by test events;
- the recorded recall interval is a two-sided 95% Wilson interval with fixed
  `z=1.959963984540054`;
- `C=0` uses the existing unmasked all-centroid spatial fallback and reads no
  waveform template;
- `C=1` directly emits the candidate and reads no waveform template; and
- `C>1` reads one 320-bit waveform template per distinct candidate.

For family summaries:

- candidate recall is
  `sum(true_unit_candidate_hits)/sum(n_test_gt_rows)`;
- traffic reduction is
  `sum(global_all_unit_waveform_template_bits) /
  sum(actual_waveform_template_bits_read)`; and
- mean row/lane/candidate costs are event-weighted by their summed counts.

The pilot grid is explicitly marked:

```text
selection_protocol =
  exploratory_test_grid_no_final_configuration_selection
```

The formal gate may select one configuration only if the same non-all
mode/size/percentile satisfies every per-record tolerance and the same
recall-led or traffic-led branch in both families. It selected none.

## 7. Accounting invariants and final audit correction

The initial pilot contained all selection and hardware counters but omitted
the pre-registered per-record interval and did not serialize the complete
per-unit fit-score/recall arrays or true-unit masked-distance/radius
distribution. Fixed Wilson-interval and true-row-distance helpers, serialized
unit audit fields, and direct tests were added; the full four-recording pilot
was then rerun. No masks, radii, candidate lists, gate values, or selection
results changed. The hashes above refer to the corrected final source and
CSVs.

Every CSV row satisfies:

```text
primary rows + fallback rows = total Level-1 rows
primary enabled lanes + fallback rows × 9 = total Level-1 lanes
total additions = total lanes - total rows
```

The CSVs also expose primary/fallback row, lane, and addition counts separately
for final candidate cardinality `C=0`, `C=1`, and `C>1`.

## 8. Confirmation disposition

The frozen protocol requires one pilot-eligible non-all configuration before
running the remaining ten HJ and eighteen MEArec recordings. Because the
formal gate produced zero candidates:

- no confirmation configuration exists;
- no confirmation command was run;
- no confirmation CSV was created;
- no directory-supported masked variant was evaluated; and
- the development recordings are not presented as confirmatory evidence.
