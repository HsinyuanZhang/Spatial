# Unit-Specific Masked-L1 5-bit Pilot

Status: **completed development-only negative result; stopped before disjoint
confirmation by the pre-registered cross-family gate**

## Decision

One binary dimension mask per unit is useful on the two selected HJ
recordings, but it is not a defensible cross-family Level-1 policy.
`stability/M=3/p95` simultaneously improves HJ candidate recall from
`0.906494` to `0.952597`, improves waveform-template traffic reduction from
`3.903629×` to `6.040536×`, and reduces total enabled absolute-difference
lanes from `96.230` to `33.144` per event. On MEArec, however, the same mask
increases recall only from `0.956057` to `0.961260` while *reducing* traffic
benefit from `2.485771×` to `2.285250×`.

No single non-all configuration:

1. stays within the frozen recall and traffic tolerances on all four
   development recordings; and
2. takes the same recall-led or traffic-led aggregate branch on both HJ and
   MEArec.

The formal pilot returned:

```text
pilot_gate_pass       False
eligible candidates   0
frozen configuration  None
joint 0.99/4x points  0
```

Kill criterion 1 in
[`adaptive_masked_5bit_plan.md`](adaptive_masked_5bit_plan.md) therefore
fires. The disjoint ten-HJ/eighteen-MEArec confirmation sweep, directory
variants, causal preprocessing integration, and online mask/radius updates
were deliberately not run.

## Implemented artifacts

- `algorithms/adaptive_masked_range_search.py`
  - fixed unsigned `[0,1]→[0,31]` quantization;
  - one integer centroid, binary mask, and calibrated integer radius per unit;
  - deterministic stability and separation/stability feature ranking;
  - inclusive masked-L1 comparisons;
  - shared or per-query active-row masks; and
  - exact 63-bit logical search-payload accounting.
- `experiments/run_adaptive_masked_5bit.py`
  - equal-timestamp-safe chronological fit/calibration/test partitions;
  - the frozen four-recording all-row development pilot;
  - an explicit, non-automatic disjoint-confirmation mode;
  - candidate, waveform, per-unit, mask, calibration, row, lane, addition,
    fallback, pointer, and configuration-write counters;
  - fixed 95% Wilson intervals for per-record candidate recall; and
  - the pre-registered cross-family gate and tie-break.
- `tests/test_adaptive_masked_range_search.py` and
  `tests/test_adaptive_masked_5bit_experiment.py`
  - bit-boundary, leakage, split, equivalence, gate, and accounting tests.

The runner imports the core module directly. The shared
`algorithms/__init__.py` export is convenient for repository users but is not
needed to reproduce the CSVs.

## Protocol

### Frozen representation

```text
descriptor             COM_x, COM_y, seven local P2P coordinates
descriptor width       D=9
code width             B=5, unsigned [0,31]
row count              one row per fitted unit
mask sizes             M ∈ {3,5,7,9}
mask modes             all_dims, stability, separation_stability
radius percentiles     95, 99, 99.9
membership             sum(mask * |q-centroid|) <= radius
directory              all rows only in the pilot
waveform template      signed 5-bit × 64 = 320 bits/candidate
```

For fitted unit `u` and coordinate `j`:

\[
w_{u,j}=Q_{0.95}(|x_{u,j}-c_{u,j}|),\qquad
s_{u,j}=\frac{\operatorname{median}_{v\ne u}|c_{v,j}-c_{u,j}|}
{w_{u,j}+1}.
\]

`stability` selects the `M` smallest `w` values.
`separation_stability` selects the `M` largest `s` values. Ties use ascending
dimension index. Both scores and centroids use fit rows only.

### Chronology and leakage boundary

Every recording is stably sorted and split without separating equal
timestamps:

```text
fit          first 50%
calibration  next 25%
test         final 25%
```

Centroids, within-unit dispersion, between-unit separation, feature masks, and
waveform-teacher centroids use fit rows. Each mask configuration's radius is
independently computed from calibration rows as the ceiling of the selected
masked-L1 percentile. Test labels are metrics only.

The experiment is still an offline component-isolation ceiling:

- spike events and unit rows are ground truth;
- the recording is filtered with full-recording zero-phase `filtfilt` before
  the chronological partitions;
- the 64-sample waveform contains future samples; and
- there is no discovered-unit lifecycle or non-GT online update.

It is not a causal detected-event sorting result.

### Development recordings

| Family | Recording | Full GT rows | Test rows | Fitted units |
|---|---|---:|---:|---:|
| HJ | `drift16c_600s_11` | 3,371 | 843 | 11 |
| HJ | `static16c_600s_11` | 2,788 | 697 | 10 |
| MEArec | `units10_snr5_seed202601` | 4,611 | 1,153 | 10 |
| MEArec | `units20_snr5_seed202601` | 9,224 | 2,306 | 20 |

The two family aggregates contain 1,540 HJ and 3,459 MEArec test events.
Across the four recordings there are no fitted units missing from calibration,
no zero radii, no test-only unit labels, and no calibration-radius fallbacks.

The pilot produces 27 configurations per recording:

```text
(one all_dims/M9 + two mask modes × four M values)
× three radius percentiles
```

The grid is exploratory. It may falsify the representation and select at most
one pre-registered configuration for a disjoint confirmation run; its best
row is not itself an independently validated paper operating point.

## Formal gate

Each non-all `M<9` point is compared with `all_dims/M=9` at the same radius
percentile. It must first satisfy on every development recording:

```text
candidate recall >= baseline - 0.005
traffic reduction >= 0.95 × baseline
```

It must then take the same aggregate branch in both families:

```text
A, recall-led:  recall delta >= 0.005 and traffic no worse
B, traffic-led: traffic ratio >= 1.10 and recall no worse
```

Family recall is `sum(candidate hits)/sum(test events)`. Family traffic
reduction is `sum(global 320-bit template bits)/sum(actual template bits)`.
The result contains zero eligible configurations.

## Main p95 results

The tight p95 radius is where masking changes the frontier most. The following
values are event-weighted within each family. `Lanes/event` includes full-nine-
dimension empty-list fallback work.

### HJ development aggregate

| Mask | M | Candidate recall | Δ recall | Waveform reduction | Traffic ratio | Mean C | Lanes/event |
|---|---:|---:|---:|---:|---:|---:|---:|
| all dimensions | 9 | 0.906494 | — | 3.903629× | — | 2.702 | 96.230 |
| stability | 3 | **0.952597** | **+0.046104** | 6.040536× | **1.547×** | 1.746 | **33.144** |
| stability | 5 | 0.926623 | +0.020130 | 5.182833× | 1.328× | 2.035 | 54.245 |
| stability | 7 | 0.911688 | +0.005195 | 4.566489× | 1.170× | 2.310 | 75.369 |
| separation/stability | 3 | 0.935714 | +0.029221 | **7.346450×** | **1.882×** | **1.436** | 33.407 |
| separation/stability | 5 | 0.932468 | +0.025974 | 5.140190× | 1.317× | 2.052 | 54.157 |
| separation/stability | 7 | 0.918182 | +0.011688 | 4.325699× | 1.108× | 2.438 | 74.732 |

HJ supplies a real unit-specific-coordinate signal: several masks improve
recall and reduce candidate traffic simultaneously. This is useful as a
family-specific ablation and as evidence that the nine coordinates are not
equally stable for every unit.

### MEArec development aggregate

| Mask | M | Candidate recall | Δ recall | Waveform reduction | Traffic ratio | Mean C | Lanes/event |
|---|---:|---:|---:|---:|---:|---:|---:|
| all dimensions | 9 | 0.956057 | — | 2.485771× | — | 6.705 | 150.026 |
| stability | 3 | **0.961260** | **+0.005204** | 2.285250× | **0.919×** | 7.293 | **50.078** |
| stability | 5 | 0.958080 | +0.002024 | 2.486629× | 1.000× | 6.703 | 83.359 |
| stability | 7 | 0.956057 | +0.000000 | 2.550208× | 1.026× | 6.535 | 116.719 |
| separation/stability | 3 | 0.953166 | −0.002891 | **2.602591×** | 1.047× | **6.404** | 50.052 |
| separation/stability | 5 | 0.951720 | −0.004337 | 2.534512× | 1.020× | 6.576 | 83.411 |
| separation/stability | 7 | 0.954322 | −0.001735 | 2.480530× | 0.998× | 6.719 | 116.693 |

MEArec exposes the collision cost of don't-care coordinates. The mask with a
material recall gain (`stability/M=3`) admits more competing units, while the
mask with the largest traffic gain (`separation_stability/M=3`) loses recall
and improves traffic by only 4.7%, below the 10% traffic-led gate.
`stability/M=5` is nearly traffic-neutral, but its `+0.002024` recall movement
is below the deliberately material `+0.005` threshold.

## Per-record safeguards and intervals

Candidate-recall intervals are two-sided 95% Wilson intervals using the fixed
normal quantile `1.959963984540054`. They quantify recording-level binomial
uncertainty; they do not turn the four development recordings into independent
confirmation data.

| Recording | Configuration, p95 | Hits / test | Recall [Wilson 95%] | Traffic reduction | Minimum unit recall |
|---|---|---:|---:|---:|---:|
| HJ drift | all/M9 | 745/843 | 0.8837 [0.8603, 0.9037] | 3.272× | 0.625 |
| HJ drift | stability/M3 | 804/843 | 0.9537 [0.9374, 0.9660] | 4.873× | 0.913 |
| HJ static | all/M9 | 651/697 | 0.9340 [0.9131, 0.9502] | 5.252× | 0.845 |
| HJ static | stability/M3 | 663/697 | 0.9512 [0.9326, 0.9649] | 8.868× | 0.873 |
| MEArec 10-unit | all/M9 | 1,102/1,153 | 0.9558 [0.9423, 0.9662] | 3.860× | 0.897 |
| MEArec 10-unit | stability/M3 | 1,110/1,153 | 0.9627 [0.9501, 0.9722] | 3.109× | 0.851 |
| MEArec 20-unit | all/M9 | 2,205/2,306 | 0.9562 [0.9471, 0.9638] | 2.283× | 0.909 |
| MEArec 20-unit | stability/M3 | 2,215/2,306 | 0.9605 [0.9518, 0.9677] | 2.143× | 0.922 |

`stability/M=3` fails the per-record traffic safeguard on both MEArec files:
its traffic ratios are approximately `0.805` and `0.939`, respectively.
Thus it would be ineligible even before the family-level same-branch rule.
The unit-level view also shows why aggregate recall alone is inadequate: on
the ten-unit MEArec recording, minimum unit recall falls from `0.897` to
`0.851` even though aggregate recall rises.

## High-radius boundary

Masking does not create a hidden high-recall/high-traffic solution:

| Family | Percentile | Highest-recall mask | Recall | Traffic reduction |
|---|---:|---|---:|---:|
| HJ | 99 | stability/M3 | 0.983766 | 2.111× |
| HJ | 99.9 | all/M9 | 0.991558 | 1.480× |
| MEArec | 99 | stability/M3 | 0.991038 | 1.335× |
| MEArec | 99.9 | separation/stability/M7 | 0.995374 | 1.255× |

The full pilot grid contains no configuration that reaches 0.99 candidate
recall and 4× waveform traffic reduction in both families. As with the
single-range and bounded multi-prototype studies, high-percentile radii recover
the true row by making too many other unit rows plausible.

## What the masks select

Across the 21 fitted HJ rows, `stability/M=3` selects dimensions 0 and 1 for
every unit, dimension 2 for 19 units, and dimension 7 for the remaining two.
Across 30 MEArec rows it selects dimensions 0/1/2 with frequencies
`0.900/0.567/1.000`; the remaining selections are spread over dimensions
3–8. This difference is consistent with the cross-family behavior: a compact
set of stable coordinates is much more consistent on the two HJ recordings
than on the selected MEArec geometries.

Mean early/late fit-mask Jaccard values at p95 are:

| Family | stability/M3 | separation/stability/M3 | stability/M5 |
|---|---:|---:|---:|
| HJ | 0.820 | 0.593 | 0.616 |
| MEArec | 0.740 | 0.665 | 0.675 |

The HJ `separation_stability/M=3` result is close to the pre-registered 0.5
instability boundary on the drift recording (`0.536`). This does not fire the
family-mean diagnostic, but it is another warning against interpreting a
fit-derived mask as an inexpensive online state: physical mask-word rewrite
frequency was not measured.

## Hardware accounting and CiM/CAM interpretation

For `D=9`, `B=5`, one logical masked row stores:

```text
centroid codes        9 × 5 = 45 bits
binary lane mask                9 bits
full-scale radius               9 bits
logical search payload         63 bits
unit ID              ceil(log2 U) bits
valid flag                      1 bit
```

This is nine payload bits more than the 54-bit unmasked centroid/radius row.
It is a credible digital near-memory contract:

1. read or activate the row;
2. form nine 5-bit absolute differences;
3. isolate disabled operands/lanes;
4. sum enabled values into a 9-bit accumulator;
5. compare inclusively with the 9-bit radius; and
6. emit a stable unit ID.

It is not evidence for a specific SRAM-CiM, TCAM, memristive CAM, or analog
multi-bit macro. A mask-capable CiM array needs explicit per-dimension
row/column enables or a ternary/don't-care encoding, and its match-line,
sensing, mask-read, gating, and peripheral costs have not been synthesized.
The pilot therefore separates:

- logical row bits;
- active row comparisons;
- enabled absolute-difference lanes and additions;
- configuration bits written;
- directory row-ID payload; and
- 320-bit waveform-template traffic.

All-row search still evaluates approximately the fitted unit count on every
event: at p95, HJ `stability/M=3` performs `10.714` total Level-1 row
comparisons/event versus `10.692` for all dimensions, and MEArec performs
`16.675` versus `16.670`. The small increases come from empty-candidate
fallback scans. The benefit is lane isolation and candidate reduction, not
fewer activated rows. Any energy claim must show that disabled-lane activity
actually saves more than the extra mask storage/read and gating control.

Every CSV row satisfies:

```text
primary rows + fallback rows = total Level-1 rows
primary enabled lanes + fallback rows × 9 = total Level-1 lanes
total additions = total lanes - total rows
```

Primary/fallback costs are also partitioned by `C=0`, `C=1`, and `C>1`.
Candidate de-duplication is trivial because there is only one row per unit.
`C=0` invokes the existing unmasked all-centroid spatial fallback and reads no
waveform template; `C=1` directly emits the unit; `C>1` reads `C` candidate
templates.

## Diagnosis and next architecture implication

The pilot distinguishes two claims:

- **supported:** unit-specific low-bit coordinate relevance exists, and a
  three-lane HJ mask can move both recall and waveform traffic while reducing
  absolute-difference activity;
- **not supported:** one global mask-builder/mask-size/radius configuration is
  a robust HJ+MEArec candidate engine, or a 63-bit masked row meets the DAC
  0.99/4× target.

The failure is not a reason to tune `M`, score mode, or percentile on the test
recordings. Doing so would violate the selection boundary. A future branch
must change the relation more fundamentally and pre-register its evidence
path—for example, a fit/calibration-selected temporal sketch or a
tile-local high-density-2D policy—rather than add another mask-grid sweep.

## Reproduction

From `/home/xinyuan/SNN_SpikeSorting`:

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -m \
  Spatial.experiments.run_adaptive_masked_5bit \
  --pilot --duration 60 --quiet
```

Outputs:

```text
Spatial/output/adaptive_masked_5bit/
├── adaptive_masked_5bit_pilot_hj_60s.csv
└── adaptive_masked_5bit_pilot_mearec_60s.csv
```

The runner prints the gate result and explicitly states that no confirmation
sweep was started. Reproduction hashes, exact development inputs, environment,
and estimator definitions are recorded in
[`adaptive_masked_5bit_provenance.md`](adaptive_masked_5bit_provenance.md).

## Claim boundary

This pilot supports the following bounded statement:

> In four predeclared offline GT-event/GT-row development recordings, one
> deterministic unit-specific 5-bit feature mask produces a strong HJ
> improvement but no single configuration passes the pre-registered
> cross-family development gate; disjoint confirmation is therefore not
> justified.

It does not support a full-corpus generalization claim, causal sorting,
detected/discovered-row operation, a finalized CiM/CAM macro, or silicon
energy/area/latency numbers. The HJ result is an informative development
ablation, not a selected DAC headline.
