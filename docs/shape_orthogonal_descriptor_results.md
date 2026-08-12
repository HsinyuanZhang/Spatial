# Shape-Orthogonal Spatial Descriptor — Phase A Results

Status: **Phase A development gate PASSES. Phase A2 FAILS on MEArec alignment
jitter, so the registered arm does not advance. Phase B is NEGATIVE: no
movement on the candidate recall/traffic frontier. No confirmation was run and
no configuration is promoted.**

Date: 2026-08-12.
Plan: [`shape_orthogonal_descriptor_plan.md`](shape_orthogonal_descriptor_plan.md).
Interfaces: [`shape_descriptor_interfaces.md`](shape_descriptor_interfaces.md).

## Executive result

On the four locked development recordings, replacing the nine-coordinate D9/B5
descriptor with `COM + POSNEG + LAT + WIDTH` raises nearest-template assignment
by **10.65 pp on HJ** and **8.21 pp on MEArec**, at 95 bits per unit row instead
of 45. All five pre-registered development conditions pass, including the
built-in random-column falsifier, which loses accuracy as required.

Two results matter more than the headline:

1. **Splitting the P2P into its two operands does almost all of the work.**
   `COM + POSNEG` alone reaches 0.919 / 0.859 at 66 bits. Adding `LAT` and
   `WIDTH` on top moves it to 0.920 / 0.859 for 29 more bits. The index-based
   shape columns are largely redundant *given* POSNEG, even though they add
   7.4 pp on top of plain D9.
2. **Four-bit POSNEG beats five-bit.** Two-family mean 0.8894 at 95 bits versus
   0.8835 at 109 bits, so the selected configuration is both smaller and more
   accurate.

This is a development gate on four recordings. It is not confirmation, it does
not touch candidate recall or template traffic, and it was measured under
full-recording zero-phase filtering, which Phase A2 exists to challenge.

## Evidence boundary

| Item | Value |
|---|---|
| Truth boundary | GT event times and GT unit rows; component isolation |
| Split | Equal-timestamp-safe chronological 50/50 via `_chronological_split` |
| Fit boundary | Integer templates from the training half only; quantization ranges are fixed analytic constants, never fitted |
| Signal path | Full-recording zero-phase Butterworth 300–6000 Hz; **not causal** |
| Aggregation | Event-weighted by `n_test`; verified `n_correct / n_test == accuracy` on all 336 rows |
| Replicates | The two MEArec files share a spike-train seed family and are not independent biological replicates |
| Hardware boundary | `row_bits`, `lane_count`, `accum_bits` are a logical ledger, not PPA |

## Arm comparison at the selected configuration

`posneg_bits=4`, `lat_bits=4`, `amp_gate=0.125`. Accuracy is event-weighted
within each family. `hardQ` is accuracy on the 25% of test events with the
smallest top-2 distance gap; `posmrg` is the positive-margin fraction.

| Arm | HJ | MEArec | row bits | lanes | hardQ | posmrg |
|---|---:|---:|---:|---:|---:|---:|
| `d9` (current descriptor) | 0.814 | 0.777 | 45 | 9 | 0.507 | 0.781 |
| `d9_rand` (falsifier) | 0.805 | 0.776 | 74 | 16 | 0.504 | 0.782 |
| `d9_lat` | 0.831 | 0.798 | 69 | 15 | 0.522 | 0.801 |
| `d9_width` | 0.876 | 0.779 | 50 | 10 | 0.535 | 0.803 |
| `d9_lat_width` | 0.888 | 0.800 | 74 | 16 | 0.575 | 0.825 |
| `com_posneg` | 0.919 | 0.858 | 66 | 16 | 0.688 | 0.870 |
| **`com_posneg_lat_width`** | **0.920** | **0.859** | 95 | 23 | 0.700 | 0.873 |

The random-column control loses 0.84 pp on HJ and 0.03 pp on MEArec. Adding
sixteen coordinates of noise does not help, so the candidate gain is not an
artifact of a wider L1 sum. This is the single most important row in the table.

## Per-recording deltas

| Recording | `d9` | candidate | delta | test events | units |
|---|---:|---:|---:|---:|---:|
| `hybrid_janelia_drift16c_600s_11` | 0.777 | 0.889 | **+11.21 pp** | 1686 | 11 |
| `hybrid_janelia_static16c_600s_11` | 0.858 | 0.958 | **+9.97 pp** | 1394 | 10 |
| `mearec_rec_v1_units10_snr5_seed202601` | 0.849 | 0.934 | **+8.54 pp** | 2306 | 10 |
| `mearec_rec_v1_units20_snr5_seed202601` | 0.740 | 0.821 | **+8.04 pp** | 4612 | 20 |

Every recording improves and the worst case is +8.04 pp, so condition 2 passes
with a wide margin. The largest gain is on the drift scene, consistent with the
plan's hypothesis that index-difference and phase-balance columns are less
sensitive to amplitude drift than the normalized amplitude footprint.

## Gate scorecard

| Condition | Requirement | Observed | Result |
|---|---|---|---|
| 1 family gain | ≥ 2.0 pp on both families | HJ +10.65, MEArec +8.21 | **Pass** |
| 2 no recording loss | no recording worse than −0.5 pp | worst +8.04 | **Pass** |
| 3 positive margin | must not fall on either family | HJ +11.69, MEArec +8.18 | **Pass** |
| 4 reconstruction | max error ≤ 1 and nonconforming < 0.005 | error 1, nonconforming 0.00092 | **Pass** |
| 5 random control | `d9_rand` must not satisfy condition 1 | −0.84 / −0.03 | **Pass** |

Frozen selection: arm `com_posneg_lat_width`, `posneg_bits=4`, `lat_bits=4`,
`amp_gate=0.125`, 95 bits per unit row, 23 lanes, 10-bit L1 accumulator.

## Independent verification

Three checks were run against results the implementing code did not produce.

1. **Independent gate recomputation.** `audit_shape_descriptor_gates.py` shares
   no code with the runner, derives every quantity from the CSV and the plan
   alone, and exits nonzero on disagreement. It reproduces every gate
   intermediate to about 1e-13 and reports `agrees_with_runner: true`.
2. **Independent API conformance.** A reviewer-side checker written against the
   frozen interface before implementation agrees with the primitive test suite
   on every contract point, including the one-LSB reconstruction bound.
3. **Agreement with the prior exploratory diagnostic.** The `d9` control
   reproduces the earlier exploratory `COM + P2P` baseline exactly on all four
   recordings: 0.777, 0.858, 0.849, 0.740. The candidate arm lands within
   0.6 pp of the exploratory estimate on the two HJ recordings.

## Defects found and fixed during review

Recorded because each of them would have produced a wrong or unjustified pass.

| Defect | Effect | Fix |
|---|---|---|
| Per-recording join used short scene keys while the CSV holds resolved loader names | Every per-recording lookup missed, so condition 2 was vacuously true and reported `worst_recording_delta_pp: Infinity` | Selection now carries resolved `dataset.name`; an unmatched recording raises |
| `if x is float("inf")` identity check | Never fired, so the sentinel leaked into the output instead of being handled | Replaced by an explicit list and a hard error on an empty or unmatched set |
| `posneg_diagnostics` divided 1-D masked arrays by a 2-D per-event scale | Broadcast to an `(n_events, n_conforming)` matrix and reported a bogus max reconstruction error of 270, failing condition 4 for the wrong reason | Compute the target on the full 2-D grid, then mask; use the in-bounds scale that `posneg_codes` actually used |
| Gate swept a hardcoded configuration grid | A reduced or partial sweep raised instead of evaluating the configurations present | Configurations are derived from the rows present; a recording missing inside a present configuration is still a hard error |
| Primitives rejected `np.integer` scalars | Any caller sweeping bit widths from a numpy array would raise | Adopted the house guard `isinstance(x, bool) or not isinstance(x, (int, np.integer))` |
| `extract_local_extrema` accepted `window=0` | Passed validation, then crashed inside `np.max` on a zero-size axis | Requires `window >= 1` with a clear message |

The first three were caught because the gate was implemented twice and the
first Phase A run was inspected rather than trusted.

## Registered-but-unselected observation

`com_posneg` is a registered arm that reaches 0.919 / 0.859 at 66 bits, within
0.1 pp of the 95-bit selected arm. The plan pre-registered
`com_posneg_lat_width` as the candidate, so the selection stands as written and
`com_posneg` is **not** renamed as the selected configuration.

It is, however, the better bits-per-accuracy operating point on this
development set, and Phase A2 and Phase B should carry both arms so the
comparison is made on confirmation data rather than here. Treating LAT and
WIDTH as conditionally redundant given POSNEG is a hypothesis for those phases,
not a Phase A result.

## What this does not establish

- No confirmation has been run. Sets A, B, and C in the plan are untouched.
- No causal-filter or jitter evidence exists. Phase A2 is the registered kill
  condition, and `LAT`/`WIDTH` are precisely the columns most exposed to
  group-delay dispersion.
- Phase B measured candidate recall and template traffic and found **no
  frontier movement**; the 0.99-recall/4x-traffic system gate still fails for
  every arm, so Phase A must not be reported as progress against it.
- Four development recordings, GT events, GT rows, zero-phase filtering.
- `row_bits` and `lane_count` are logical accounting, not area or energy.

## Reproduction

```bash
cd /home/xinyuan/SNN_SpikeSorting

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  Spatial/tests/test_shape_descriptor.py \
  Spatial/tests/test_shape_descriptor_pilot.py \
  Spatial/tests/test_shape_descriptor_audit.py

python -m Spatial.experiments.run_shape_descriptor_pilot --quiet
python -m Spatial.experiments.audit_shape_descriptor_gates
```

The auditor exits nonzero if its recomputation disagrees with the runner.

Full-suite status at the time of this result: **417 passed, 5 failed**. The five
failures are the pre-existing frozen-provenance tests of the causal-temporal
branch (`test_causal_temporal_selection.py`,
`test_causal_temporal_sketch_experiment.py`), which by design require the
working tree to sit at their recorded base revision and source inventory. They
already failed before this branch: the first extra file reported by the
inventory assertion is `experiments/run_relative_footprint_pilot.py`, which
predates this work. No test that passed before this branch fails now.

Artifacts, all gitignored under `Spatial/output/shape_descriptor_pilot/`:

| File | Role |
|---|---|
| `shape_descriptor_pilot.csv` | 336 rows, one per (recording, arm, posneg_bits, lat_bits, amp_gate) |
| `shape_descriptor_selection.json` | Frozen selection and the runner's gate evaluation |
| `shape_descriptor_audit.json` | Independent recomputation and agreement flag |

## Phase A2 — causal and jitter boundary: FAILS on jitter

Status: **complete. The causal filter does NOT kill the branch. Deterministic
±2-sample alignment jitter DOES, on MEArec, and the fragile part is exactly the
`LAT`/`WIDTH` columns.**

Warm-up guard 1000 samples (≈33 ms at 30 kHz, about ten periods of the 300 Hz
lower band edge, and clear of the 15-sample window plus 2-sample jitter). All
six conditions are evaluated on an identical guarded event population, asserted
row-by-row before evaluation, so a filtering or jitter effect cannot be
confounded with a change in which events survive.

Accuracy per condition and block:

| Family | Condition | `d9` | `com_posneg` | full arm | full − `d9` | full − `com_posneg` |
|---|---|---:|---:|---:|---:|---:|
| HJ | control | 0.8136 | 0.9185 | 0.9201 | +10.65 pp | +0.16 pp |
| HJ | **causal** | 0.8117 | 0.9172 | 0.9214 | **+10.97 pp** | +0.42 pp |
| HJ | jitter −2 | 0.8136 | 0.9192 | 0.9214 | +10.78 pp | +0.23 pp |
| HJ | jitter +2 | 0.8136 | 0.9218 | 0.9182 | +10.45 pp | −0.36 pp |
| MEArec | control | 0.7760 | 0.8583 | 0.8577 | +8.17 pp | −0.06 pp |
| MEArec | **causal** | 0.7817 | 0.8329 | 0.8356 | **+5.39 pp** | +0.27 pp |
| MEArec | jitter −2 | 0.7760 | 0.8602 | 0.8664 | +9.04 pp | +0.62 pp |
| MEArec | jitter +2 | 0.7760 | 0.8540 | **0.8437** | +6.77 pp | **−1.03 pp** |

| Gate | Requirement | Observed | Result |
|---|---|---|---|
| Causal retention, HJ | ≥ 50% of Phase A gain | 103.5% | **Pass** |
| Causal retention, MEArec | ≥ 50% of Phase A gain | 65.8% | **Pass** |
| Jitter, HJ | worst ±2 loss ≤ 1.0 pp | 0.19 pp | **Pass** |
| Jitter, MEArec | worst ±2 loss ≤ 1.0 pp | **1.40 pp** | **Fail** |
| **Phase A2 overall** | | | **Fail** |

### Interpretation

The predicted failure mode did not happen. Group-delay dispersion from a
forward-only SOS filter was the registered kill condition, and the descriptor
survives it: HJ retains all of its gain and MEArec retains two thirds. A
deployable causal signal path is therefore not what breaks this representation.

What breaks it is alignment jitter, and the per-block breakdown localizes the
damage precisely. The `LAT`/`WIDTH` incremental contribution over `com_posneg`
is small and sign-unstable in every condition (−0.36 to +0.62 pp), and at
MEArec jitter +2 it goes to −1.03 pp — the index-based columns actively hurt.
`com_posneg` at the same operating point loses only 0.43 pp. The effect is
asymmetric between −2 and +2, which is consistent with a window shift
truncating the repolarization phase differently in each direction rather than
with symmetric noise.

This is the mechanism the plan anticipated for these columns, arriving through
jitter rather than through the filter.

### Protocol consequence

The registered candidate arm `com_posneg_lat_width` **fails Phase A2**. Under
the plan, that outcome is recorded as the branch result and may not be repaired
by reselecting the gate, the bit widths, or the arm after seeing this data.

`com_posneg` is a registered arm that would satisfy both A2 gates on these
recordings (causal retention 100% HJ / 62% MEArec, worst ±2 jitter loss 0.00 pp
HJ / 0.43 pp MEArec). It is **not** promoted to the selection here. That
observation was made after seeing Phase A2 data, so acting on it requires a new
pre-registered protocol validated on the untouched confirmation sets, which is
exactly what confirmation sets A, B, and C exist for.

Artifacts under `Spatial/output/shape_causal_boundary/`.

```bash
python -m Spatial.experiments.run_shape_causal_boundary --quiet
```

## Phase B — candidate recall and template traffic: NEGATIVE

Status: **complete. The descriptor does not move the recall/traffic frontier.
No arm passes the joint system gate.**

Phase A measured assignment accuracy. The standing system requirement is
different: 0.99 true-unit candidate recall together with at least 4x
waveform-template-read reduction, on both families. Phase B puts all three arms
inside the *same* per-unit L1 radius candidate policy (all rows active, radii
fitted on the training half only) and sweeps the radius percentile.

Family event-weighted frontier on the four development recordings:

| Arm | Family | p95 recall / reduction | p99.9 recall / reduction |
|---|---|---|---|
| `d9` | HJ | 0.9237 / 4.18x | 0.9935 / 1.30x |
| `com_posneg` | HJ | 0.9231 / 5.15x | 0.9919 / 1.37x |
| `com_posneg_lat_width` | HJ | 0.9396 / 3.81x | 0.9906 / 1.30x |
| `d9` | MEArec | 0.9485 / 2.48x | 0.9923 / 1.24x |
| `com_posneg` | MEArec | 0.9481 / 2.63x | 0.9961 / 1.16x |
| `com_posneg_lat_width` | MEArec | 0.9465 / 2.26x | 0.9964 / 1.11x |

Best reduction achieved while still holding recall ≥ 0.99, for the selected
arm: **1.30x on HJ** and **1.11x on MEArec**, both at the p99.9 radius. The 4x
target is not approached by any arm at any tested percentile that also meets
recall.

| Arm | HJ joint gate | MEArec joint gate | Both |
|---|---|---|---|
| `d9` | Fail | Fail | **Fail** |
| `com_posneg` | Fail | Fail | **Fail** |
| `com_posneg_lat_width` | Fail | Fail | **Fail** |

### Interpretation

The new descriptor buys assignment accuracy and buys nothing on the candidate
frontier. Both observations are visible in the same table: at HJ p95 the
selected arm has higher recall than `d9` (0.9396 versus 0.9237) but *worse*
traffic (3.81x versus 4.18x). It is moving along the existing frontier, not
shifting it.

The mechanism is that a per-unit radius must be wide enough to contain roughly
99% of that unit's intra-unit variation, and a radius that wide admits about as
many foreign units in the new space as in the old one. Making a descriptor more
discriminative for nearest-template argmin does not by itself make each unit's
acceptance region more selective. Phase A and Phase B are measuring genuinely
different properties, and this branch improves only the first.

The one positive signal is small: `com_posneg` reaches 5.15x on HJ at p95, the
best traffic point in the table, but at 0.9231 recall it is far from the gate.

### What this closes and what it leaves open

Closed: the claim that a better Level-1 descriptor alone would fix the
recall/traffic bottleneck. It does not, on this development set, under this
candidate policy. The Level-1 bottleneck recorded in
[`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md) remains the system-blocking issue,
and the shape descriptor must not be reported as progress against it.

Open: whether a per-unit L1 radius is the right acceptance policy for a
23-coordinate mixed-width space at all. The radius was inherited from the
9-coordinate D9/B5 study. That is a hypothesis for a separate registered
experiment, not a rescue of this result.

Artifacts under `Spatial/output/shape_candidate_traffic/`:
`shape_candidate_traffic.csv` and `shape_candidate_traffic_summary.json`.

```bash
python -m Spatial.experiments.run_shape_candidate_traffic --quiet
```

## Branch disposition

Three phases are complete and they do not all point the same way.

| Phase | Question | Outcome |
|---|---|---|
| A | Does the representation beat D9/B5? | **Pass**, +10.65 / +8.21 pp |
| A2 | Does it survive a causal filter and alignment jitter? | **Fail** on MEArec ±2 jitter |
| B | Does it move the 0.99-recall/4x-traffic frontier? | **Negative**, no movement |

The registered configuration `com_posneg_lat_width` is therefore **not
selected**, and this branch produces no configuration for the hardware
mainline. Two durable findings survive:

1. The normalized amplitude footprint is not information-limited in the way the
   bit/shift/estimator sweeps suggested. Keeping both operands of `max - min`
   recovers roughly 10 pp of assignment accuracy on HJ that the current
   descriptor discards for free, and it survives a causal filter.
2. Assignment accuracy and candidate-set selectivity are close to independent
   here. A descriptor can gain 10 pp of argmin accuracy and move the
   recall/traffic frontier by nothing at all. Future Level-1 work should be
   evaluated on the frontier directly, not on argmin.

The `LAT` and `WIDTH` columns are the fragile part and are the natural thing to
drop: they cost 29 bits, contribute at most 0.4 pp over `com_posneg`, and are
the source of the jitter failure.

## Next step

Two candidates, in priority order.

1. **Register `com_posneg` on the confirmation sets.** It is a registered arm
   that appears both cheaper (66 bits) and more robust than the selected arm,
   but that judgement used Phase A2 data, so it needs a fresh pre-registered
   protocol evaluated on confirmation sets A, B, and C, which remain untouched.
   Include the ±2 jitter and causal conditions in that protocol from the start
   rather than as a later boundary check.
2. **Attack the acceptance policy, not the descriptor.** Phase B suggests the
   per-unit L1 radius inherited from the 9-coordinate D9/B5 study may be the
   real limiter. Testing a different acceptance region on a fixed descriptor
   would separate policy from representation, which this branch could not do.
