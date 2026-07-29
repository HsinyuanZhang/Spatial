# Indexed 5-bit Confusion-Graph Candidate Retrieval

Status: **implemented and validated on HJ + MEArec; static and block-causal
graph-state-only variants both fail the joint 99%-recall / 4×-read gate**

This experiment removes Range-CAM match uniqueness from the decision path.
Instead, it computes a deterministic 5-bit nearest-centroid anchor and fetches
a compact candidate pointer list from SRAM:

```text
5-bit D=9 query
        │
        ▼
all-row L1 ranking / WTA
        │
        ├── fixed top-M rows
        │
        └── top-L anchor IDs ──► calibration confusion pointer lists
                                      │
                                      ▼
                              deterministic candidate union
```

This is an algorithmic indexed-SRAM/near-memory study. It is not a synthesized
CiM, WTA, SRAM-macro, or PPA result.

Implementation:

- `experiments/run_indexed_confusion_candidate_search.py`
- `experiments/run_online_confusion_graph.py`
- `tests/test_indexed_confusion_candidate_search.py`
- `tests/test_online_confusion_graph.py`

## Protocol

- Ground-truth event times and unit rows isolate candidate retrieval.
- Stable chronological 50/50 train/test; equal timestamps remain together.
- The complete raw recording is zero-phase Butterworth filtered before this
  split. Consequently “causal” below refers only to graph-state update timing,
  not to the signal preprocessing path.
- The train half is split chronologically into `2/3 fit + 1/3 calibration`.
- Fit-only state: fixed `[0,1]` 5-bit quantizer, integer centroids, and p95
  per-unit L1 radii.
- Calibration-only state: predicted-anchor to true-unit occurrence counts and
  the smallest pointer list reaching a requested 0.95, 0.99, or 1.0 empirical
  coverage.
- Test labels never select an anchor, add a pointer, repair a candidate set, or
  change a configuration.
- Every graph list includes its anchor. An anchor with no calibration event
  falls back to `[anchor]`. Calibration labels absent from fit rows are counted
  but never materialized as nonexistent rows.
- Rankings:
  - raw integer L1;
  - radius-normalized `d/(r+1)` for algorithm isolation. A hardware
    implementation could compare ratios by cross multiplication, but this
    study does not assign it PPA.
- Test retrieval uses one or two highest-ranked anchors and a stable,
  de-duplicated union of their lists.
- Unique candidate sets read zero waveform templates; ambiguous sets read
  exactly `C`; empty sets use the existing spatial fallback with zero waveform
  reads.
- Results below are event-weighted across 12 HJ scenes or 20 MEArec files.

## Reproduction

Run from the parent of the `Spatial` package:

```bash
cd /home/xinyuan/SNN_SpikeSorting

python -m Spatial.experiments.run_indexed_confusion_candidate_search \
  --family all --duration 60 --bits 5 --k 7 \
  --ranking-modes raw_l1 radius_normalized_l1 \
  --top-ms 1 2 4 8 \
  --graph-coverages 0.95 0.99 1.0 \
  --top-anchors 1 2 \
  --fit-fraction 0.6666666666666666 \
  --waveform-metric l1 --quiet

# Block-causal graph-state update ceiling
python -m Spatial.experiments.run_online_confusion_graph \
  --family all --duration 60 --bits 5 --k 7 \
  --ranking-modes raw_l1 radius_normalized_l1 \
  --graph-coverages 0.95 0.99 1.0 --top-anchors 1 2 \
  --block-seconds 1 5 \
  --update-sources oracle_delayed_label self_predicted_label \
  --histories accumulate rolling \
  --rolling-horizon-seconds 10 \
  --fit-fraction 0.6666666666666666 \
  --waveform-metric l1 --quiet
```

Generated files:

```text
output/indexed_confusion_candidate_search/
├── indexed_confusion_5bit_hj_60s.csv
└── indexed_confusion_5bit_mearec_60s.csv

output/online_confusion_graph/
├── online_confusion_5bit_hj_60s.csv
└── online_confusion_5bit_mearec_60s.csv
```

## Fixed top-M result

`M=4` is the only tested fixed-size point near the 4× traffic threshold:

| Family | Ranking | Candidate recall | Mean candidates | Waveform-read reduction | Final waveform accuracy |
|---|---|---:|---:|---:|---:|
| HJ | raw L1, top-4 | 0.931 | 4.00 | **4.52×** | 0.853 |
| HJ | radius-normalized, top-4 | 0.909 | 4.00 | **4.52×** | 0.852 |
| MEArec | raw L1, top-4 | 0.917 | 4.00 | **4.18×** | 0.784 |
| MEArec | radius-normalized, top-4 | 0.907 | 4.00 | **4.18×** | 0.790 |

Expanding to top-8 reaches only 0.962 HJ and 0.960–0.961 MEArec recall while
reducing traffic by about 2.1–2.3×. Fixed nearest-centroid retrieval therefore
does not approach the 99% recall target at the allowable candidate count.

## Learned adjacency result

The best graph points that retain at least 4× event-weighted waveform-read
reduction use 0.95 calibration coverage and one anchor:

| Family | Ranking | Candidate recall | Mean candidates | Read reduction | Mean graph storage/recording | Final accuracy |
|---|---|---:|---:|---:|---:|---:|
| HJ | raw L1 | 0.923 | 2.73 | **7.27×** | 305 bits | 0.879 |
| HJ | radius-normalized | 0.932 | 3.61 | **5.31×** | 323 bits | 0.869 |
| MEArec | raw L1 | 0.934 | 4.07 | **4.26×** | 365 bits | 0.813 |
| MEArec | radius-normalized | 0.938 | 4.29 | **4.08×** | 349 bits | 0.788 |

The graph is compact and traffic-efficient, but its calibration confusion
support does not transfer with sufficient recall to the later test half.

The highest-recall tested graph uses radius normalization, 1.0 calibration
coverage, and two anchors:

| Family | Candidate recall | Mean candidates | Read reduction | Mean graph storage/recording |
|---|---:|---:|---:|---:|
| HJ | **0.980** | 9.75 | 1.85× | 453 bits |
| MEArec | **0.986** | 10.94 | 1.53× | 505 bits |

Even storing every calibration-observed true unit for each anchor does not
reach 99% held-out recall. Increasing to two anchors consumes most of the
candidate-size advantage. No evaluated configuration has both aggregate
candidate recall at least 0.99 and waveform-read reduction at least 4×.

## Block-causal graph-state update ceiling

The follow-up stream experiment freezes the fit-only 5-bit quantizer,
centroids, and radii. It processes the held-out half in 1 s or 5 s blocks:

1. build the graph from history available at block start;
2. predict the complete block and record metrics;
3. only then append new anchor→label observations; and
4. let the updated graph affect the next block.

Two label sources bound deployment:

- `oracle_delayed_label` uses true labels only after the current block has been
  scored. It is an explicitly non-deployable upper bound.
- `self_predicted_label` uses the candidate-only waveform prediction and never
  reads test labels to configure the graph.

Histories either accumulate or retain a rolling 10 s event-time window.
Pointer additions/removals, rewritten pointer bits, and one label-observation
write per streamed event are counted.

The legacy `pointer_bit_rewrites` field is a membership-only lower bound.
Physical contiguous-list accounting additionally records changed pointer
positions, count-field metadata writes, and their combined bit rewrites. A
pure reorder such as `[10,20] → [20,10]` therefore writes two pointer words
even though membership additions/removals are zero. The final block's
post-update graph is materialized for write accounting even though no later
query consumes it. Base addresses, allocator/free-list metadata, banking, and
SRAM peripherals remain excluded.

The best traffic-compliant points remain far below 99% recall:

| Family | Update source | Configuration summary | Candidate recall | Mean candidates | Read reduction |
|---|---|---|---:|---:|---:|
| HJ | oracle delayed | normalized, cov95, L1, 1 s, accumulate | **0.947** | 4.02 | **4.72×** |
| HJ | self predicted | normalized, cov100, L1, 5 s, rolling | **0.940** | 4.60 | **4.03×** |
| MEArec | oracle delayed | raw, cov95, L1, 1 s, rolling | **0.934** | 4.10 | **4.24×** |
| MEArec | self predicted | normalized, cov95, L1, 5 s, accumulate | **0.930** | 3.71 | **4.84×** |

The highest-recall online points again consume most unit rows:

| Family | Update source | Candidate recall | Mean candidates | Read reduction |
|---|---|---:|---:|---:|
| HJ | oracle delayed | **0.991** | 12.73 | 1.42× |
| HJ | self predicted | **0.988** | 12.30 | 1.47× |
| MEArec | oracle delayed | **0.994** | 13.10 | 1.28× |
| MEArec | self predicted | **0.994** | 14.11 | 1.19× |

Even the delayed-GT graph upper bound has no joint operating point. Online
observations can recover recall only by widening pointer lists toward a global
scan. Self updates are not consistently equivalent to the oracle and can
increase graph size through prediction errors.

This result closes **graph-only block-causal state** updates, not every online
adaptation:
centroids, radii, quantizer, and directory support remain static. A future
online study must justify their update confidence and count their write
bandwidth; it cannot cite this oracle graph result as a deployable algorithm.

## Storage accounting

For `U` anchor rows:

\[
b_{\text{ptr}}=\lceil\log_2 U\rceil,\qquad
b_{\text{count}}=\lceil\log_2(U+1)\rceil,
\]

\[
B_{\text{graph}}
=U\,b_{\text{count}}+N_{\text{ptr}}\,b_{\text{ptr}}.
\]

This includes one count field per anchor and all unit-ID pointers. It excludes
the 5-bit centroid/radius rows, directory, WTA/ranking logic, SRAM decoders,
unit metadata, controller, and update writes. The search also compares against
all fit-unit centroids on every query; graph compactness does not erase that
Level-1 compute cost.

## Interpretation

This negative result is stronger than the Range-CAM failure:

- The problem is not only threshold calibration or singleton confidence.
- Under the current `COM_x, COM_y + K=7 P2P` descriptor, a static candidate
  relation learned in the earlier time segment does not retain 99% true-unit
  coverage at `U/4`-class list sizes.
- Radius normalization improves some learned-anchor recall but does not close
  the gap.
- A larger static graph converges toward a broad template scan.

Do not present a static or graph-only online confusion table as the main DAC
solution. The remaining useful tests and the disposition of the completed
waveform branch are:

1. a causal centroid/radius/directory update policy with an observable,
   non-GT confidence rule and update writes counted;
2. pair-conditioned waveform taps for persistent ambiguity have been tested:
   16 taps recover high-recall accuracy but fail traffic and O(U²) index
   storage, so this branch is an ablation rather than the solution;
3. richer but still low-bit spatial state if online updates do not close the
   recall gap; and
4. a synthesized comparison of all-row L1/WTA versus directory-indexed local
   SRAM distance.

## Limitations

- GT event times and GT unit labels are used to build the calibration graph.
- Full-recording zero-phase filtering makes the feature stream an offline
  input; the block protocol is state-causal but not an end-to-end causal
  signal path.
- The online system has not yet shown how discovered labels update the graph.
- All fit-unit rows are ranked; no high-recall coarse directory is combined
  with this branch.
- Only graph pointers update; centroids, radii, quantizer, and directory remain
  static.
- There is no 1024-channel high-unit-count validation or circuit PPA.
