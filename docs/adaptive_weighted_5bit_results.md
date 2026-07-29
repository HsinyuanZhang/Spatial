# Adaptive Power-of-Two Weighted-L1 5-bit Pilot Results

Status: **completed development pilot; formal gate failed; confirmation
prohibited and not run**

## Decision

The unit-specific `{1,2,4}` power-of-two weighted-L1 relation is a useful
hardware-feasible shape ablation, but it is **not** a cross-family solution to
the current spatial-retrieval bottleneck. It can sharply reduce HJ candidate
collisions, yet the corresponding MEArec gain is small or comes with true-row
recall loss. The fixed four-recording development gate selected zero
configurations, so the remaining ten HJ and eighteen MEArec recordings were
not evaluated.

This closes repeated static per-coordinate weight tuning as the next main DAC
branch. The result supports a narrower architectural conclusion: retain the
5-bit row and power-of-two weighted relation as an optional near-memory
primitive or ablation, but move the next algorithmic effort to information not
contained in one static spatial centroid row—causal temporal sketches,
observable confidence/state, or realistic high-density 2D locality.

## Frozen experiment

The protocol was fixed before running the real-data pilot:

- four development recordings:
  `drift16c_600s_11`, `static16c_600s_11`,
  `units10_snr5_seed202601`, and `units20_snr5_seed202601`;
- first 60 seconds of each recording;
- equal-timestamp-safe chronological `50% fit / 25% calibration / 25% test`;
- `D=9`, `B=5`, fixed `[0,1]→[0,31]` quantization;
- one row per unit and all fitted rows active;
- one calibrated uniform reference plus four fit-only nonuniform
  score/profile combinations;
- radii at `p∈{95,99,99.9}`, giving 15 rows per recording;
- fit-only centroids, weights, and waveform templates;
- calibration-only radii; and
- test labels used only for metrics.

The two nonuniform profiles are:

```text
top3_x4:    three best-ranked dimensions ×4, remaining six ×1
tiered_421: best three ×4, next three ×2, remaining three ×1
```

Weights are ranked either by low within-unit fit dispersion (`stability`) or
by fit-only between-centroid separation divided by within dispersion plus one
(`separation_stability`). No coordinate is disabled.

The full bit-exact protocol is in
[`adaptive_weighted_5bit_plan.md`](adaptive_weighted_5bit_plan.md).

## Formal development gate

Each nonuniform row was paired with the calibrated uniform row at the same
percentile. A configuration first had to satisfy every recording:

```text
candidate recall >= uniform recall - 0.005
traffic reduction >= 0.95 × uniform traffic reduction
early/late fit shift-code agreement >= 0.75
```

It then had to take the same branch on both event-weighted families:

```text
A: recall delta >= +0.005 and waveform traffic no worse
or
B: traffic reduction >= 1.10 × uniform and recall no worse
```

Waveform reads use the frozen semantics `C<=1 → 0 reads`,
`C>1 → C×320 bits`. Gates use summed integer bit counts and exact
cross-products, including the pre-registered zero-traffic cases.

The result was:

```text
pilot_gate_pass: false
pilot_gate_candidates: 0
selected_configuration: null
final_0p99_4x_pass: false
final_0p99_4x_qualifying_configurations: 0
confirmation outputs: none
```

The deterministic selection JSON contains twelve complete nonuniform
configuration audits (`4 profiles × 3 percentiles`). Loading this failed
artifact as confirmation authority raises:

```text
selection manifest has no selected configuration; confirmation is prohibited
```

## Event-weighted p95 results

Recall is `sum(hits)/sum(test events)`. Traffic reduction is
`sum(global U×320 bits)/sum(actual candidate-template bits)`. `Mean C`,
row/event, lane/event, and shifted-lane/event values are also event-weighted.
The Wilson interval is the two-sided 95% interval for the family hit total.

### HJ development recordings

| Score/profile | Recall | 95% Wilson | Δ recall | Traffic | Traffic ratio vs uniform | Mean C | Rows/event | Lanes/event | Shifted lanes/event | Min code agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.90649 | [0.89092, 0.92004] | — | 3.9036× | 1.000× | 2.841 | 10.692 | 96.230 | 0.000 | 1.000 |
| stability/top3_x4 | **0.90844** | [0.89300, 0.92185] | **+0.00195** | **7.9041×** | **2.025×** | **1.692** | 10.983 | 98.848 | 31.642 | 0.899 |
| stability/tiered_421 | 0.90390 | [0.88815, 0.91763] | −0.00260 | 6.9237× | 1.774× | 1.842 | 10.961 | 98.649 | 63.284 | 0.667 |
| separation/top3_x4 | 0.90519 | [0.88954, 0.91883] | −0.00130 | 7.1935× | 1.843× | 1.843 | 10.838 | 97.545 | 31.642 | 0.778 |
| separation/tiered_421 | 0.90519 | [0.88954, 0.91883] | −0.00130 | 6.3375× | 1.623× | 1.990 | 10.850 | 97.650 | 63.284 | 0.697 |

HJ shows a real geometric effect. `stability/top3_x4` reduces mean candidates
from 2.841 to 1.692 and more than doubles traffic reduction relative to the
same-percentile uniform reference. However, its recall improvement is only
`+0.00195`, below the pre-registered `+0.005` recall branch. Two tiered
variants also fail the minimum `0.75` early/late code-agreement rule on at
least one HJ recording.

### MEArec development recordings

| Score/profile | Recall | 95% Wilson | Δ recall | Traffic | Traffic ratio vs uniform | Mean C | Rows/event | Lanes/event | Shifted lanes/event | Min code agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | **0.95606** | [0.94871, 0.96240] | — | 2.4858× | 1.000× | 6.748 | 16.670 | 150.026 | 0.000 | 1.000 |
| stability/top3_x4 | 0.95548 | [0.94809, 0.96186] | −0.00058 | 2.5904× | 1.042× | 6.470 | 16.672 | 150.052 | 50.000 | 0.822 |
| stability/tiered_421 | 0.95519 | [0.94778, 0.96159] | −0.00087 | 2.5972× | 1.045× | 6.461 | 16.672 | 150.052 | 100.000 | 0.656 |
| separation/top3_x4 | 0.94796 | [0.94005, 0.95488] | −0.00809 | **2.7476×** | **1.105×** | **6.130** | 16.670 | 150.026 | 50.000 | 0.844 |
| separation/tiered_421 | 0.94912 | [0.94128, 0.95596] | −0.00694 | 2.7086× | 1.090× | 6.219 | 16.670 | 150.026 | 100.000 | 0.700 |

MEArec exposes the cross-family failure. The stable top-three profile preserves
recall within the per-record tolerance, but its traffic improvement is only
`1.042×` relative to uniform and recall moves slightly backward. The only
profile exceeding the family `1.10×` traffic threshold loses `0.00809` recall
and also violates at least one per-record recall tolerance. The 20-unit
recording remains almost entirely ambiguous at p95: for
`stability/top3_x4`, `C>1` occurs on 99.87% of test events.

## High-radius operating points

Raising the calibrated radius recovers recall but collapses memory-traffic
reduction. The most informative points are:

| Family | Configuration | Radius | Recall | Traffic | Δ recall vs uniform | Traffic ratio vs uniform |
|---|---|---:|---:|---:|---:|---:|
| HJ | uniform | p99 | 0.97338 | 1.9175× | — | 1.000× |
| HJ | stability/top3_x4 | p99 | 0.97922 | 2.2849× | +0.00584 | 1.192× |
| MEArec | uniform | p99 | 0.98930 | 1.3886× | — | 1.000× |
| MEArec | stability/top3_x4 | p99 | 0.98757 | 1.4089× | −0.00173 | 1.015× |
| HJ | uniform | p99.9 | 0.99156 | 1.4801× | — | 1.000× |
| HJ | stability/top3_x4 | p99.9 | 0.99156 | 1.6062× | +0.00000 | 1.085× |
| MEArec | uniform | p99.9 | 0.99422 | 1.2544× | — | 1.000× |
| MEArec | best weighted recall, separation/tiered | p99.9 | 0.99335 | 1.2666× | −0.00087 | 1.010× |

The p99 HJ point is the strongest recall-led weighted result, but the same
frozen configuration loses recall on MEArec. At p99.9 both families can exceed
0.99 recall, yet traffic is only about `1.26–1.61×`, far below the final
`4×` target. The core collision-versus-coverage frontier therefore remains.

## What changed geometrically

The paired same-percentile candidate-set diagnostics show that weighting is
not merely a scalar-radius rewrite:

- HJ p95 `stability/top3_x4` has event-weighted candidate Jaccard `0.614`
  against uniform and reduces mean `C` by about 40.5%;
- MEArec p95 for the same profile has Jaccard `0.745` but reduces mean `C` by
  only about 4.1%;
- HJ p95 fallback-inclusive Level-1 rows rise from `10.692` to `10.983`
  per event because more empty candidate sets trigger the unweighted
  all-centroid fallback; and
- MEArec Level-1 rows remain essentially unchanged because the candidate sets
  are almost never empty.

Thus the hypothesis “weights are fully absorbed by radius calibration” is
false in a narrow mathematical sense, but the resulting shape change is not
the cross-family Pareto improvement needed by the architecture.

## Hardware interpretation

The implemented row stores:

```text
9 × 5-bit centroid                 45 bits
9 × 2-bit shift code              18 bits
11-bit weighted-L1 radius         11 bits
logical search payload            74 bits/row
```

The optimized uniform reference is 54 bits/row, so the weighted format adds
20 logical payload bits per unit before alignment, ECC, tags, valid bits, and
periphery. Every search row still activates nine absolute-difference lanes.
Weights use fixed shifts by zero, one, or two and an 11-bit accumulation range
up to `1116`; no multiplier or divider is required.

The pilot records primary and fallback rows, nine-lane work, shift-by-one and
shift-by-two lanes, accumulator additions, candidate FIFO entries, and
`C=0/1/>1` conditional work. All five emitted accounting identities hold in
all 60 CSV rows.

These are **logical** operation and storage counters. They do not demonstrate
that a CAM/CiM macro is faster or more energy-efficient than indexed SRAM plus
digital distance. The result does not authorize cell-only PPA claims. A
physical comparison still needs macro choice, decoder/sense/periphery,
alignment, banking, accumulator timing, candidate routing, node/PVT, and RTL
or post-layout evidence.

## Reproducibility and audit

Run from `/home/xinyuan/SNN_SpikeSorting`:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_weighted_5bit \
  --pilot --duration 60 --quiet
```

The final artifacts are:

```text
output/adaptive_weighted_5bit/adaptive_weighted_5bit_pilot_hj_60s.csv
output/adaptive_weighted_5bit/adaptive_weighted_5bit_pilot_mearec_60s.csv
output/adaptive_weighted_5bit/adaptive_weighted_5bit_pilot_selection_60s.json
```

Their exact hashes, inputs, source state, environment, and selection boundary
are recorded in
[`adaptive_weighted_5bit_provenance.md`](adaptive_weighted_5bit_provenance.md).

The independent final audit verified:

- 30 data rows per family, exactly two recordings × 15 configurations;
- exact frozen grid and all-row activation;
- all logical accounting identities;
- manifest-to-CSV SHA-256 agreement;
- twelve nonuniform configuration audits;
- gate and final-target recomputation;
- `selected_configuration=null`; and
- no file whose name contains `confirmation`.

## Paper-facing use

This is development-only negative evidence, not a full-corpus estimator. It
may support:

1. the ablation that per-unit low-bit relation shape matters on HJ;
2. the argument that static spatial reweighting alone does not resolve dense
   local-unit ambiguity across probe families;
3. the 74-bit shift-add row as a bounded optional hardware point; and
4. the decision to stop static mask/weight micro-tuning before spending the
   disjoint confirmation set.

It must not support claims of cross-dataset improvement, confirmed
generalization, online sorting, causal preprocessing, CiM feasibility, or
measured energy/area/latency.
