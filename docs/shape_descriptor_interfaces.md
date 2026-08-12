# Shape-Orthogonal Descriptor — Frozen Interfaces

Status: **Frozen 2026-08-12 before implementation.** Companion to
[`shape_orthogonal_descriptor_plan.md`](shape_orthogonal_descriptor_plan.md).

This file exists so that the feature primitives, the Phase A runner, and the
independent gate auditor can be implemented separately without agreeing on
anything at implementation time. Changing a signature, a column name, or a
neutral code here is a protocol change and must be recorded, not done silently.

## 1. Feature primitives

All live in `Spatial/algorithms/spatial_footprint.py` and are exported from
`Spatial/algorithms/__init__.py`. Every returned code array is `float64`
holding integer values, matching the existing `quantize_p2p` convention so that
downstream integer-L1 code paths are unchanged.

**Scalar argument type policy** (clarified 2026-08-12 during review, after the
first implementation rejected numpy scalars). Integer scalar arguments accept
`int` and `np.integer` and reject `bool`, matching the existing house guard
`isinstance(x, bool) or not isinstance(x, (int, np.integer))` used in
`causal_temporal_sketch.py`. Float scalar arguments accept any real that is
finite. Callers sweeping bit widths out of a numpy array must not have to cast.

### `extract_local_extrema`

```python
extract_local_extrema(
    signal: np.ndarray,            # (n_channels, n_samples)
    spike_times: np.ndarray,       # (n_spikes,)
    central_channels: np.ndarray,  # (n_spikes,)
    neighbor_table: np.ndarray,    # (n_channels, K), -1 marks out of bounds
    window: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Reads exactly the segment `signal[ch, t - window : t + window]`, identical to
`extract_local_p2p`. Boundary filtering is identical, so returned rows align
one-to-one with `extract_local_p2p` output on the same inputs.

Returns `(vmax, vmin, imax, imin, valid_times, neighbor_ids)`:

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| `vmax` | `(n_valid, K)` | float64 | per-slot maximum sample value |
| `vmin` | `(n_valid, K)` | float64 | per-slot minimum sample value |
| `imax` | `(n_valid, K)` | int64 | index of the **first** occurrence of `vmax` in `[0, 2*window)` |
| `imin` | `(n_valid, K)` | int64 | index of the **first** occurrence of `vmin` in `[0, 2*window)` |
| `valid_times` | `(n_valid,)` | int64 | kept spike times |
| `neighbor_ids` | `(n_valid, K)` | int64 | slot channel ids, `-1` out of bounds |

Out-of-bounds slots produce `vmax = vmin = 0.0` and `imax = imin = 0`.

### `slot_liveness`

```python
slot_liveness(vmax, vmin, neighbor_ids, amp_gate: float = 0.25, eps: float = 1e-12) -> np.ndarray
```

Returns `(n, K)` bool. With `scale_i = max over in-bounds j of (vmax[i,j] - vmin[i,j])`
floored at `eps`:

```text
live[i,j] = (neighbor_ids[i,j] >= 0) and ((vmax[i,j] - vmin[i,j]) >= amp_gate * scale_i)
```

`amp_gate` must be a finite float in `[0.0, 1.0]`.

### `posneg_codes`

```python
posneg_codes(vmax, vmin, neighbor_ids, n_bits: int = 5, eps: float = 1e-12) -> np.ndarray
```

Returns `(n, 2K)` float64 codes in `[0, 2**n_bits - 1]`. Columns `0..K-1` are
positive-peak codes, columns `K..2K-1` are negative-peak codes.

```text
max_level = 2**n_bits - 1
scale_i   = max over in-bounds j of (vmax[i,j] - vmin[i,j]), floored at eps
pos[i,j]  = clip(rint(max_level * max(vmax[i,j], 0.0) / scale_i), 0, max_level)
neg[i,j]  = clip(rint(max_level * max(-vmin[i,j], 0.0) / scale_i), 0, max_level)
```

Out-of-bounds slots produce code `0` in both halves. `n_bits` must be an `int`
in `[1, 16]`; `bool` is rejected.

**Reconstruction bound.** On any slot where `vmax >= 0 >= vmin` and the slot is
in bounds:

```text
| pos[i,j] + neg[i,j] - rint(max_level * (vmax[i,j] - vmin[i,j]) / scale_i) | <= 1
```

### `latency_codes`

```python
latency_codes(imin, live, home_slot: int, n_bits: int = 4) -> np.ndarray
```

Returns `(n, K-1)` float64 codes. The home-slot column is **removed**; the
remaining columns keep their original slot order with index `home_slot` deleted.

```text
limit    = 2**(n_bits - 1) - 1                  # 7 at 4 bits, 3 at 3 bits
raw[i,j] = imin[i,j] - imin[i, home_slot]
code     = clip(raw, -limit, limit) + limit     # in [0, 2*limit]
```

Slots with `live[i,j]` false take the neutral code `limit`. `n_bits` must be an
`int` in `[2, 8]`; `bool` is rejected. `home_slot` must be a valid column index.

### `width_codes`

```python
width_codes(imax, imin, home_slot: int, n_bits: int = 5) -> np.ndarray
```

Returns `(n, 1)` float64 codes in `[0, 2**n_bits - 1]`.

```text
off  = 2**(n_bits - 1) - 1                              # 15 at 5 bits
code = clip(imax[i, home_slot] - imin[i, home_slot], -off, off + 1) + off
```

Neutral code is `off`. `n_bits` must be an `int` in `[2, 8]`; `bool` is rejected.

## 2. Descriptor arms

Assembled in `Spatial/experiments/shape_descriptor_common.py`. `COM` and the
existing normalized `P2P` block reuse the current 5-bit `[0, 1]` quantization so
that the `d9` arm is bit-identical to today's D9/B5 descriptor.

| Arm id | Blocks, in column order | Coordinates at `posneg_bits=5, lat_bits=4` |
|---|---|---:|
| `d9` | COM, P2P | 9 |
| `d9_lat` | COM, P2P, LAT | 15 |
| `d9_width` | COM, P2P, WIDTH | 10 |
| `d9_lat_width` | COM, P2P, LAT, WIDTH | 16 |
| `d9_rand` | COM, P2P, RAND | 16 |
| `com_posneg` | COM, POSNEG | 16 |
| `com_posneg_lat_width` | COM, POSNEG, LAT, WIDTH | 23 |

`RAND` is a seeded uniform integer block with exactly the coordinate count and
per-coordinate code width of `LAT + WIDTH`: `K-1` columns at `lat_bits` plus one
column at 5 bits. Its seed is derived only from the recording name and must not
depend on labels, split, or arm.

Matching is all-row integer L1 with `rint` integer templates fitted on the
training partition, nearest-template argmin, first-minimum tie-break.

## 3. Phase A CSV schema

Written to `Spatial/output/shape_descriptor_pilot/shape_descriptor_pilot.csv`.
One row per `(recording, arm, posneg_bits, lat_bits, amp_gate)`.

The `dataset` column holds the **resolved loader name**, for example
`hybrid_janelia_drift16c_600s_11` or `mearec_rec_v1_units10_snr5_seed202601`,
not the short scene key used to request the recording. Any join key or
`development_recordings` list must use that resolved form. Column order is not
frozen; consumers must address columns by name.

| Column | Type | Meaning |
|---|---|---|
| `dataset` | str | recording name from the loader |
| `family` | str | `hj` or `mearec` |
| `scene_type` | str | `static`, `drift`, or `mearec` |
| `mearec_n_units` | int | configured MEArec unit count, `-1` for HJ |
| `arm` | str | arm id from the table above |
| `posneg_bits` | int | POSNEG code width |
| `lat_bits` | int | LAT code width |
| `amp_gate` | float | liveness gate `g` |
| `n_train` | int | training events |
| `n_test` | int | test events |
| `n_units` | int | units with a fitted template |
| `n_coords` | int | descriptor coordinate count |
| `row_bits` | int | per-unit row payload bits |
| `lane_count` | int | absolute-difference lanes per row comparison |
| `accum_bits` | int | bits needed for the worst-case L1 accumulator |
| `accuracy` | float | closed-set nearest-template assignment accuracy |
| `n_correct` | int | correct test events |
| `worst_unit_accuracy` | float | minimum per-unit accuracy |
| `margin_mean` | float | mean of `d_nearest_wrong - d_true` |
| `positive_margin_fraction` | float | fraction of test events with that margin `> 0` |
| `true_template_distance_norm` | float | mean true-template L1 divided by `lane_count * max_code` |
| `hard_quartile_accuracy` | float | accuracy on the 25% of test events with the smallest top-2 distance gap |
| `nonconforming_slot_fraction` | float | fraction of in-bounds slots violating `vmax >= 0 >= vmin` |
| `posneg_reconstruction_max_error` | int | max `\|pos + neg - rint(...)\|` over conforming in-bounds slots |

`accuracy` is over test events whose true unit has a fitted template. Deltas
against the `d9` control are **not** stored; the auditor computes them by
joining rows on `(dataset, posneg_bits, lat_bits, amp_gate)`.

## 4. Selection JSON schema

Written to `Spatial/output/shape_descriptor_pilot/shape_descriptor_selection.json`.

```json
{
  "plan": "shape_orthogonal_descriptor_plan.md",
  "phase": "A",
  "development_recordings": ["..."],
  "selected_arm": "com_posneg_lat_width",
  "posneg_bits": 5,
  "lat_bits": 4,
  "amp_gate": 0.25,
  "gate_results": {
    "family_gain_hj_pp": 0.0,
    "family_gain_mearec_pp": 0.0,
    "worst_recording_delta_pp": 0.0,
    "positive_margin_delta_hj_pp": 0.0,
    "positive_margin_delta_mearec_pp": 0.0,
    "max_nonconforming_slot_fraction": 0.0,
    "max_posneg_reconstruction_error": 0,
    "rand_control_gain_hj_pp": 0.0,
    "rand_control_gain_mearec_pp": 0.0,
    "condition_1_family_gain": false,
    "condition_2_no_recording_loss": false,
    "condition_3_positive_margin": false,
    "condition_4_reconstruction": false,
    "condition_5_rand_control_fails": false
  },
  "passed": false,
  "null_reason": "text or null"
}
```

`selected_arm` is `null` when the gate fails, and `null_reason` states which
condition failed. Event-weighted family means use `n_test` as the weight.

## 5. Gate conditions (authoritative restatement)

From the plan's development gate, evaluated for the candidate arm
`com_posneg_lat_width` against the `d9` control on identical event rows:

1. event-weighted family-mean accuracy beats `d9` by at least **2.0 pp** on
   **both** HJ and MEArec;
2. no development recording loses more than **0.5 pp** against `d9`;
3. `positive_margin_fraction` does not fall on either family;
4. `posneg_reconstruction_max_error <= 1` on every row and
   `nonconforming_slot_fraction < 0.005` on every development recording;
5. the `d9_rand` control does **not** satisfy condition 1.

Condition 5 failing invalidates the measurement; the branch stops for
re-instrumentation rather than proceeding. Among qualifying configurations,
rank by: conditions passed, two-family mean accuracy, `hard_quartile_accuracy`,
`positive_margin_fraction`, then fewer `row_bits`.
