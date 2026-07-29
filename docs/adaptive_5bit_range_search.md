# Adaptive 5-bit Spatial Range Search

Status: **implemented and validated on HJ + MEArec; single-range,
uncertified-cascade, and per-unit certified-exit variants all fail the joint
recall/traffic target**

This study evaluates the first memory level of the proposed two-level spike
sorting pipeline:

```text
5-bit COM + local P2P
        │
        ▼
coarse channel directory
        │
        ▼
per-unit L1-CiM radius or per-feature Range-CAM box
        │
        ├── one candidate: early exit, zero waveform reads
        ├── multiple candidates: read only those waveform templates
        └── zero candidates: spatial fallback
```

Implementation:

- `algorithms/adaptive_range_search.py`
- `experiments/run_adaptive_5bit_range_search.py`
- `experiments/run_dual_range_pipeline.py`
- `experiments/run_certified_dual_range_pipeline.py`
- `tests/test_adaptive_range_search.py`
- `tests/test_adaptive_range_experiment.py`
- `tests/test_dual_range_pipeline.py`
- `tests/test_certified_dual_range_pipeline.py`

## Evaluation protocol

- Descriptor: normalized `COM_x, COM_y` plus max-normalized local P2P from
  `K=7` electrodes, for `D=9` features.
- Quantization: unsigned 5-bit fixed `[0,1]` mapping; every feature code is in
  `[0,31]`.
- Split: stable chronological 50/50. Events with the same timestamp stay on
  the same side of the split.
- Preprocessing: the complete recording is zero-phase Butterworth filtered
  with `filtfilt` before the event split. The split is label/state-isolated but
  the signal path is offline and non-causal across the boundary.
- Event source: ground-truth spike times.
- Row/unit source: ground-truth training unit labels. This is an oracle
  isolation experiment for the memory search, not an end-to-end sorter.
- Level 2: 64-sample, peak-normalized main-channel waveform centroid with L1
  distance.
- Memory accounting:
  - `|C|=1`: direct label, zero waveform reads;
  - `|C|>1`: read `|C|` waveform templates;
  - `|C|=0`: unified 5-bit spatial fallback, zero waveform reads.
  - `active_row_count` means rows enabled in one normal stage;
    `total_active_row_accesses` sums inner plus actually triggered outer
    accesses;
  - `fallback_row_comparisons` separately counts normalized-outside scans and
    empty-list all-centroid spatial fallback;
  - `total_row_comparisons_including_fallback` is the normal-search plus
    fallback total. It is the access field to use for energy modeling.
- Reported family means are unweighted means over 12 HJ scenes or 20 MEArec
  recordings unless stated otherwise.

The MEArec set contains two temporal seeds for each unit-count design rendered
at five SNRs. The 20 files are therefore not 20 independent spike-train
replicates; SNR variants of one seed must not be treated as independent
biological evidence.

## Reproduction commands

Run from the parent of the `Spatial` package:

```bash
cd /home/xinyuan/SNN_SpikeSorting

# Current-gate operating point
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 95 --coverage 0.95 \
  --widening-steps 0 1 2 4 \
  --directory-neighbors 0 7 \
  --directory-coverages 0.95 0.99 1.0 \
  --waveform-metric l1 --quiet

# High-recall operating point
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 99 --coverage 0.99 \
  --widening-steps 0 1 2 4 \
  --directory-neighbors 0 7 \
  --directory-coverages 0.95 0.99 1.0 \
  --waveform-metric l1 --quiet

# Near-max training coverage; all rows only
python -m Spatial.experiments.run_adaptive_5bit_range_search \
  --family all --duration 60 --bits 5 \
  --percentile 99.9 --coverage 0.999 \
  --widening-steps 0 --directory-neighbors 0 \
  --directory-coverages 1.0 \
  --waveform-metric l1 --quiet

# Uncertified nested-range go/no-go experiment
python -m Spatial.experiments.run_dual_range_pipeline \
  --family all --duration 60 --bits 5 --k 7 \
  --inner-coverages 0.50 0.75 0.90 0.95 \
  --outer-percentile 99.9 --waveform-metric l1 --quiet

# Disjoint-calibration, one-bit per-unit safe-exit experiment
python -m Spatial.experiments.run_certified_dual_range_pipeline \
  --family all --duration 60 --bits 5 --k 7 \
  --inner-coverages 0.50 0.75 0.90 0.95 \
  --precision-targets 0.95 0.99 --min-supports 5 10 20 \
  --outer-percentile 99.9 --safe-z 1.645 \
  --fit-fraction 0.6666666666666666 --waveform-metric l1 --quiet
```

Generated CSVs:

```text
output/adaptive_5bit_range_search/
├── adaptive_5bit_range_search_hj_60s_p95p0_c0p95.csv
├── adaptive_5bit_range_search_hj_60s_p99p0_c0p99.csv
├── adaptive_5bit_range_search_hj_60s_p99p9_c0p999.csv
├── adaptive_5bit_range_search_mearec_60s_p95p0_c0p95.csv
├── adaptive_5bit_range_search_mearec_60s_p99p0_c0p99.csv
└── adaptive_5bit_range_search_mearec_60s_p99p9_c0p999.csv

output/dual_range_pipeline/
├── dual_5bit_range_pipeline_hj_60s_outerp99p9.csv
└── dual_5bit_range_pipeline_mearec_60s_outerp99p9.csv

output/certified_dual_range_pipeline/
├── certified_dual_5bit_range_pipeline_hj_60s_outerp99p9.csv
└── certified_dual_5bit_range_pipeline_mearec_60s_outerp99p9.csv
```

## 1. Does a different range per unit help?

Yes, but the gain is modest for an L1 ball. Relative to one pooled radius,
per-unit radii reduce the mean candidate set while losing a small amount of
test recall.

| Family | Calibration | Pooled L1 recall / candidates | Per-unit L1 recall / candidates | Candidate reduction |
|---|---:|---:|---:|---:|
| HJ | 95% | 0.942 / 6.49 | 0.935 / 5.37 | 17.2% |
| HJ | 99% | 0.990 / 13.22 | 0.984 / 11.09 | 16.1% |
| MEArec | 95% | 0.949 / 6.02 | 0.947 / 5.52 | 8.4% |
| MEArec | 99% | 0.989 / 11.45 | 0.987 / 9.90 | 13.5% |

The result supports row-specific radii as a more efficient representation than
one global threshold. It does **not** show that per-unit radii alone solve the
memory-traffic problem.

## 2. Recall–candidate Pareto

### All rows active

| Family | Search | Calibration | Candidate recall | Mean candidates | Stage-2 activation | Waveform reads/event | Global-read reduction | Final accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HJ | legacy COM∧P2P | 95% | 0.905 | 2.30 | 0.575 | 1.91 | 8.15× | 0.889 |
| HJ | per-unit L1 | 95% | 0.935 | 5.37 | 0.946 | 5.33 | 3.04× | 0.853 |
| HJ | per-unit box | 95% | 0.886 | 2.78 | 0.656 | 2.48 | 6.35× | 0.878 |
| HJ | per-unit L1 | 99% | 0.984 | 11.09 | 0.999 | 11.09 | 1.47× | 0.794 |
| HJ | per-unit L1 | 99.9% | **0.993** | **13.14** | 1.000 | 13.14 | **1.25×** | 0.774 |
| MEArec | legacy COM∧P2P | 95% | 0.911 | 3.39 | 0.647 | 3.06 | 4.37× | 0.822 |
| MEArec | per-unit L1 | 95% | 0.947 | 5.52 | 0.928 | 5.45 | 2.62× | 0.805 |
| MEArec | per-unit box | 95% | 0.935 | 4.95 | 0.850 | 4.82 | 2.94× | 0.811 |
| MEArec | per-unit L1 | 99% | 0.987 | 9.90 | 0.999 | 9.90 | 1.48× | 0.741 |
| MEArec | per-unit L1 | 99.9% | **0.995** | **11.91** | 1.000 | 11.91 | **1.25×** | 0.712 |

“Global-read reduction” is computed from the event-weighted sum of all-unit
template reads divided by the event-weighted sum of actual candidate reads. It
is not the arithmetic mean of per-recording ratios.

### Main conclusion

A single range cannot satisfy both acceptance criteria:

- Near-99% recall requires an L1 radius so wide that almost every event
  activates Level 2 and searches most units.
- A tight box substantially reduces candidates and can improve final accuracy,
  but it excludes too many true rows.

At 99.9% calibration the per-unit L1 search clears 99% mean recall on both
families, but provides only about **1.25×** waveform-read reduction, far below
the 4× target. The current single-range design therefore fails the joint
recall/traffic gate.

## 3. Why final accuracy falls as candidate recall rises

The 64-sample waveform-centroid L1 stage is not a sufficiently strong global
classifier. As the spatial candidate set widens, Level 2 sees more confusing
waveform templates:

- HJ per-unit L1 final accuracy falls from 0.853 at 95% calibration to 0.774 at
  99.9%.
- MEArec falls from 0.805 to 0.712.

This is not evidence that lower candidate recall is desirable. It means the
current waveform engine benefits from the spatial gate acting as a prior and
cannot safely replace it with a near-global scan. The completed Level-2 study
compared selected taps, fixed streaming morphology, and pair-conditioned taps.
Shared taps fail the high-recall cross-family gate; pair-conditioned 16-tap
decisions recover accuracy but fail traffic and O(U²) storage. See
[`compact_waveform_refinement.md`](compact_waveform_refinement.md).

## 4. Directory mapping

Three directory strategies were evaluated with a seven-channel geometry KNN:

- `mode`: one pointer per unit, at its most frequent training central channel;
- `support-95`: the smallest channel set covering at least 95% of that unit's
  training events;
- `support-99`: the corresponding 99% set.

The support matrix represents pointers in an inverted channel/tile directory.
Multiple pointers activate one CAM row; they do not duplicate the row itself.

| Family | Directory | Directory recall | Mean active rows | Row reduction | Mean pointers/unit |
|---|---|---:|---:|---:|---:|
| HJ | mode | 0.969 | 7.10 | 56.6% | 1.00 |
| HJ | support-95 | 0.983 | 10.32 | 38.7% | 2.27 |
| HJ | support-99 | **0.998** | 15.30 | 9.0% | 5.05 |
| MEArec | mode | 0.927 | 3.61 | 76.0% | 1.00 |
| MEArec | support-95 | 0.981 | 8.50 | 45.9% | 4.67 |
| MEArec | support-99 | **0.996** | 11.59 | 24.3% | 7.89 |

Multiple channel pointers repair much of the single-home-channel recall loss,
but a 99% support target consumes most of the locality advantage, especially
on HJ. A publishable implementation should therefore use a tile-aware compact
support/range representation rather than blindly registering every observed
central channel.

The directory must be treated as a first-class memory structure. Its pointer
storage and access energy are not included in the CAM row bits below.

## 5. Hardware payload

For `D=9`, `B=5`:

| Search row | Payload | Bits/row | Interpretation |
|---|---|---:|---|
| Pooled L1 | `D×B` centroid | 45 + one shared threshold | global threshold |
| Per-unit L1 | `D×B` centroid + full-scale L1 radius | **54** | multi-bit L1 CiM/AM |
| Per-unit box | lower + upper for every feature | **90** | Range-CAM |
| Raw inner box + outer L1 | unshared 90-bit box + 54-bit L1 row | **144** | nested search before any shared-field optimization |
| Legacy software gate | 2 float32 COM + 7×5-bit P2P centroid | 99 | software comparison only |

The fixed L1 radius field is sized for the complete distance range:

\[
\lceil\log_2(D(2^B-1)+1)\rceil
=\lceil\log_2(9\cdot31+1)\rceil=9\text{ bits}.
\]

Thus the fixed row is `9×5 + 9 = 54` bits. The implementation also reports a
data-specific fitted minimum width, but that optimistic width is not used for
DAC area claims.

These figures exclude unit ID, directory pointers, match-line sensing, input
encoding, controller, update metadata, SRAM/CAM peripherals, and PVT margin.

The current runners retain the older `active_row_count` and `row_comparisons`
fields for compatibility. They now additionally emit
`inner_active_row_count`, `outer_active_row_count`,
`total_active_row_accesses`, `range_fallback_row_comparisons`,
`spatial_fallback_row_comparisons`, and
`total_row_comparisons_including_fallback`. This prevents a triggered outer
stage or an empty-list centroid fallback from disappearing from hardware
access accounting.

## 6. Boundary analysis

### Drift

HJ drift scenes are materially harder at the tight operating point:

| Search @95% | HJ drift recall | HJ static recall |
|---|---:|---:|
| Per-unit L1 | 0.920 | 0.950 |
| Per-unit box | 0.856 | 0.916 |

This identifies drift as a static-range failure mode, but does not by itself
show that an online update will solve it. A later bounded `P≤4` local-mode
study explicitly tested temporal/farthest-L1 rows and still failed the
0.99-recall/4× gate; HJ's 4×-class point reached only 0.898 recall with four
rows/unit. Simply calibrating a higher fixed percentile or taking the union of
more fixed local ranges widens the candidate set toward a global scan.

### Unit density

MEArec candidate count scales sharply with unit density even when candidate
recall is similar:

| Calibration | Units | L1 recall | Mean candidates | Final accuracy |
|---:|---:|---:|---:|---:|
| 95% | 10 | 0.948 | 2.88 | 0.880 |
| 95% | 20 | 0.947 | 8.16 | 0.730 |
| 99% | 10 | 0.989 | 5.83 | 0.837 |
| 99% | 20 | 0.984 | 13.98 | 0.644 |

The design must therefore scale with *local unit density*, not only channel
count.

### SNR

At 95% L1 calibration, MEArec mean candidates fall from 7.10 at SNR 2 to about
4.83 at SNR 12; final accuracy rises from 0.666 to 0.870. Search cost and
sorting quality must both be characterized versus SNR/event quality.

## 7. Negative result: zero-only widening

Widening only when the candidate set is empty changes recall by at most a few
tenths of a percentage point in these sweeps. Most excluded true units occur
when an incorrect row already matches, so “first non-empty” widening stops too
early.

Keep zero-only widening as a safety fallback, not as the main recall mechanism.
The next design must trigger a wider search from an observable confidence or
margin condition.

## 8. Negative result: uncertified nested ranges

The all-row go/no-go experiment implements:

```text
tight per-unit box
        │
        ├── exactly one hit ──────────────► early exit
        └── zero or multiple hits ───────► 99.9%-radius outer L1 search
                                              │
                                              ▼
                                     candidate waveform engine
```

There is no top-k addition, test-label repair, or forced nonempty candidate.
An incorrect inner singleton is a real early exit. The final list for triggered
queries is the ordered union of outer candidates and any inner candidates that
the outer range omitted.

The table below uses event-weighted sums across 33,161 held-out HJ events from
12 scenes or 65,155 held-out MEArec events from 20 files. `Reads reduction` is
the all-unit global waveform-read count divided by actual candidate-template
reads.

| Family | Inner box coverage | Inner unique precision | Outer trigger | Final candidate recall | Mean candidates | Reads reduction | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| HJ | 0.50 | 0.962 | 0.395 | 0.972 | 6.27 | 3.19× | 0.844 |
| HJ | 0.75 | 0.944 | 0.300 | 0.956 | 5.07 | **4.14×** | 0.856 |
| HJ | 0.90 | 0.890 | 0.535 | 0.944 | 8.96 | 2.13× | 0.808 |
| HJ | 0.95 | 0.882 | 0.759 | 0.967 | 12.09 | 1.52× | 0.782 |
| MEArec | 0.50 | 0.904 | 0.432 | 0.943 | 6.63 | 2.76× | 0.764 |
| MEArec | 0.75 | 0.908 | 0.530 | 0.954 | 8.06 | 2.20× | 0.745 |
| MEArec | 0.90 | 0.916 | 0.787 | 0.978 | 11.57 | 1.47× | 0.704 |
| MEArec | 0.95 | 0.909 | 0.900 | 0.987 | 12.65 | 1.33× | 0.689 |

The event-weighted outer-L1 baseline reaches 0.993 HJ and 0.995 MEArec
candidate recall, but only 1.25× read reduction. No nested operating point
reaches both 0.99 recall and 4× read reduction. The closest HJ traffic point
(`coverage=0.75`) loses 3.68 percentage points of recall relative to the outer
search; the highest-recall MEArec point still provides only 1.33× reduction.

The sequential Level-1 cost is also explicit after closing access accounting.
HJ `coverage=0.75` evaluates 23.51 inner-plus-triggered-outer rows/event, versus
18.07 for a single all-row stage. MEArec `coverage=0.50` evaluates 24.29,
versus 16.72. Thus the waveform-read reduction does not imply a Level-1 energy
reduction.

The mechanism is identifiable rather than statistical noise: the recall loss
tracks the fraction of inner singletons that point to the wrong unit. For
example, HJ `coverage=0.50` early-exits 60.5% of events at 96.18% singleton
precision; MEArec early-exits 56.8% at only 90.40% precision. A unique Range-CAM
match is therefore not a sufficient confidence signal.

This result motivated the next bounded experiment: a calibration-derived
**one-bit safe-exit flag per unit**. Search ranges are fit on a fit subset,
flags are certified on a disjoint calibration subset, and test singletons may
bypass outer L1 only when their predicted row is certified.

## 9. Negative result: per-unit safe-exit certification

The original train half is split chronologically into `2/3 fit + 1/3
calibration`; tied timestamps remain together. The 5-bit quantizer, inner box,
and outer L1 ranges see only fit rows. On calibration rows, singleton results
are counted against the **predicted unit**. A unit receives one safe-exit flag
only when:

\[
n_u \ge n_{\min}
\quad\land\quad
\operatorname{WilsonLCB}(k_u,n_u;z=1.645)\ge p_{\min}.
\]

At test time, only a singleton pointing to a flagged unit may exit. Zero,
ambiguous, and unsafe-singleton events all invoke the cached outer L1 result.
Test labels never set a flag, trigger a query, or repair a candidate list.
Waveform centroids may use the complete original train half, but do not
participate in certification.

The strongest traffic points from the full grid are:

| Family | Inner coverage | Wilson target | Safe units | Certified exits | Test exit precision | Candidate recall | Reads reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| HJ | 0.75 | 0.95 | 5.5% | 5.8% | 0.975 | 0.989 | **1.38×** |
| MEArec | 0.50 | 0.95 | 3.0% | 5.2% | 0.985 | 0.992 | **1.36×** |
| HJ | any tested | 0.99 | 0% | 0% | — | 0.990 or lower | 1.31× |
| MEArec | any tested | 0.99 | 0% | 0% | — | about 0.993 | 1.29–1.30× |

These are event-weighted results. Minimum-support values 5, 10, and 20 select
the same few rows in this dataset; every selected row already has at least 20
calibration singleton observations. The limiting factor is precision, not the
support threshold.

Certification is conservative in the desired direction, but it does not move
the Pareto frontier enough. At a 0.99 Wilson target no unit is certified, so
the design pays the 145-bit inner-box/outer-L1/flag payload and roughly two
all-row searches without meaningful early exits. Relaxing the target to 0.95
certifies only 3–6% of rows, and their held-out singleton precision is
97.5–98.5%, still below the reliability needed for a 99%-recall architecture.
At the strongest traffic points, the completed inner-plus-triggered-outer
accounting is 35.17 row accesses/event on HJ and 32.78 on MEArec.

This closes the current Range-CAM singleton-exit branch as the main DAC
mechanism. Keep the 5-bit per-unit search, directories, and certification as
well-controlled ablations. A follow-on indexed local SRAM/near-memory experiment
retrieves deterministic row sets without treating match uniqueness as
confidence; it also fails the static cross-time joint gate. See
[`indexed_confusion_candidate_search.md`](indexed_confusion_candidate_search.md).
The subsequent bounded multi-prototype range branch also fails and is recorded
in
[`adaptive_multiprototype_5bit_results.md`](adaptive_multiprototype_5bit_results.md).
The later one-row/unit feature-mask pilot finds a strong HJ development
movement but fails its pre-registered cross-family gate; it is recorded in
[`adaptive_masked_5bit_results.md`](adaptive_masked_5bit_results.md).
The following nonzero `{1,2,4}` weighted-L1 pilot also changes HJ candidate
shape but fails its frozen cross-family gate; it is recorded in
[`adaptive_weighted_5bit_results.md`](adaptive_weighted_5bit_results.md).
The 145-bit figure is raw payload only; it still excludes unit ID, directory,
sensing, control, and update peripherals.

## 10. Remaining limitations

- The study uses GT spike times and GT-derived rows.
- Full-recording zero-phase filtering is not a deployable causal preprocessing
  path. A forward-only stateful SOS primitive is tested, but experiment-level
  warm-up, guard, detector, and boundary-state integration remain required.
- It does not test discovered/online clusters or unit creation/deletion.
- It does not include online radius/bound updates.
- The 20 MEArec files contain only four independent unit-count/seed temporal
  configurations crossed with SNR.
- No realistic 1024-channel, high-unit-count 2D recording is currently present
  in the workspace.
- No RTL, memory macro, match-line, ADC/WTA, synthesis, place-and-route, or PVT
  result exists yet.
- The 64-sample waveform contains `t` plus 48 strictly future samples; reaching
  `t+48` requires 1.60 ms at 30 kHz, separate from compute latency.
