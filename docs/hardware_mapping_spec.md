# Hardware Mapping Specification for the Two-Level Memory Pipeline

Status: **algorithm-to-hardware contract; payload and access accounting are
implemented, circuit PPA is not**

This document fixes the data layout, controller semantics, and accounting
boundary for the DAC direction. It deliberately separates measured algorithmic
results from future RTL/macro evidence.

## 1. Current architectural decision

The following Level-1 branches have been implemented and rejected as the main
policy because none reaches both 99% held-out candidate recall and 4× fewer
waveform-template reads:

- one per-unit L1 radius;
- one per-unit Range-CAM box;
- inner box plus outer L1 with singleton exit;
- calibration-certified singleton exit;
- static anchor-to-confusion-list indexed retrieval;
- graph-only block-causal pointer-state updates;
- bounded `P≤4` local prototype rows per unit; and
- one-row/unit masked-L1 search as a cross-family policy; and
- one-row/unit `{1,2,4}` weighted-L1 search as a cross-family policy.

They remain required baselines and hardware cost points. The current reference
path for further development is:

```text
event descriptor
    │
    ▼
fixed 5-bit quantizer
    │
    ▼
channel/tile directory ──► local row-enable mask
    │
    ▼
local centroid L1 / near-memory ranking
    │
    ▼
causally updated deterministic candidate IDs
    │
    ├── one candidate ──► direct label
    └── multiple ──────► candidate waveform SRAM + 5-bit SAD
```

This diagram is the interface contract for the next branch, not evidence that
the current candidate policy passes its gate. Graph-only state updates have
already failed: at 4×-class traffic, even the delayed-GT upper bound reaches
only 0.947/0.934 candidate recall on HJ/MEArec. Pair-conditioned waveform taps
have also been evaluated: 16 taps recover teacher-level accuracy but provide
only 1.35×/1.36× conservative bit-traffic reduction and require O(U²) index
state. The next algorithmic policy must first pass the static
recall/traffic representation gate; only then is a non-GT causal update of
centroid/radius/directory state on discovered rows justified. A final macro
choice must follow that policy's measured access and write pattern.

The multi-prototype branch confirms that duplicate row hits must be separated
from distinct unit reads. At its best HJ 4×-class point, four rows/unit produce
0.898 recall, 41.405 primary row comparisons/event, and 42.357 total Level-1
comparisons/event after empty-list fallback. The directory activates a
true-unit row for 0.999 of events, so wider directory activation is not the
missing mechanism. MEArec has no 4× point in the tested multi-row grid.

The masked-L1 development pilot confirms a different distinction: fewer
enabled arithmetic lanes do not imply fewer row activations. A logical
63-bit row (`45` centroid + `9` mask + `9` radius) produces a strong HJ
development point at p95, reducing total lanes/event from 96.23 to 33.14 while
improving candidate recall and waveform traffic. The same
`stability/M=3/p95` policy worsens MEArec waveform traffic, however, and zero
configurations pass the pre-registered cross-family pilot gate. It remains an
HJ-positive ablation and a gated-datapath cost point, not the selected Level-1
policy. The disjoint confirmation and directory-supported masked experiments
were not run.

The weighted-L1 pilot then preserved all nine lanes and replaced hard masks
with two-bit shift codes. At p95, `stability/top3_x4` substantially reduces HJ
ambiguity and moves traffic reduction from 3.90× to 7.90×, but improves recall
by only 0.00195. On MEArec it changes traffic by just 1.042× relative to
uniform and moves recall slightly backward. No configuration passes the frozen
cross-family gate, so this is a shift-add row cost point—not the selected
Level-1 policy.

The completed staged-access counters show why waveform traffic alone is not an
energy result. One all-row stage costs 18.07/16.72 row evaluations per event
on HJ/MEArec. The selected dual traffic points cost 23.51/24.29, while the
relaxed certified points cost 35.17/32.78. Any CAM/CiM comparison must charge
these repeated Level-1 activations in addition to waveform SRAM traffic.

The alignment-conditioned temporal-sketch pilot then held the high-recall
Level-1 source fixed and tested a structured shift/add Level-2 code. Its
40/60/80-bit signed-5-bit templates all exceed the `4×` template-bit target,
but every configuration fails accuracy. Delay-8/M16 is the least-bad common
diagnostic point and still loses 6.10/12.08pp on HJ/MEArec. It is a valid
extractor/storage/accounting cost point, not a selected refinement engine.

## 2. Descriptor interface

For `K=7` local electrodes:

```text
q = [COM_x, COM_y, P2P_0, ..., P2P_6]
D = 9
B = 5
input payload = 45 bits/event
```

All fields use an unsigned fixed `[0,1] → [0,31]` code. Saturation and
round-to-nearest must match `UniformUnsignedQuantizer`. Quantizer bounds are
fixed or learned from training only; no test-stream min/max may change them.

Interface sketch:

```text
descriptor_valid
descriptor_code[44:0]
central_channel_or_tile_id
event_timestamp
```

The Python reference preserves simultaneous events as separate rows. RTL must
not use timestamp as a unique key.

## 3. Level-1 row formats

For `D=9`, `B=5`:

| Row type | Fields | Raw bits/row | Status |
|---|---|---:|---|
| L1 centroid | `c[8:0]`, each 5 bit | 45 | common base state |
| Per-unit L1 | centroid + 9-bit radius | **54** | implemented baseline |
| Multi-prototype L1 | `P≤4` independent L1 rows + common unit ID | **54P/unit** before ID/valid | rejected static branch |
| Masked per-unit L1 | centroid + 9-bit lane mask + 9-bit radius | **63** | HJ-positive development ablation; rejected cross-family policy |
| Weighted per-unit L1 | centroid + nine 2-bit shift codes + 11-bit radius | **74** | HJ-positive development ablation; rejected cross-family policy |
| Range-CAM | 9 lower + 9 upper codes | **90** | implemented baseline |
| Inner box + outer L1 | 90 + 54, no sharing assumed | **144** | rejected nested branch |
| Certified dual | raw dual + safe flag | **145** | rejected certified branch |

The 9-bit L1 accumulator/radius width follows:

\[
\lceil\log_2(9(2^5-1)+1)\rceil=9.
\]

These values exclude unit ID, valid/deleted bit, directory links, update
version, ECC, match-line sensing, ADC/WTA, decoder, and controller.

Multiple matching prototype rows from one unit must be de-duplicated before
candidate FIFO insertion. Prototype row comparisons and distinct waveform
candidate reads are different counters; waveform-template traffic must never
be multiplied by the number of matching rows carrying the same unit ID.

### L1 datapath

Per active row:

1. nine unsigned 5-bit absolute differences;
2. an adder tree producing a 9-bit distance;
3. either a 9-bit radius compare or participation in top-L selection.

For a masked row, the nine stored mask bits gate the corresponding operands or
absolute-difference lanes before accumulation. The accumulator and radius
remain fixed at nine bits because the all-dimension maximum is still
`9×31=279`. A software count of three enabled lanes is not automatically a
three-lane physical datapath: an implementation may instantiate nine lanes and
isolate six, serialize enabled coordinates through fewer lanes, or encode
don't-care fields in a mask-capable CiM/CAM macro. Area, latency, match-line,
sensing, and gating energy differ across those choices and are unmodeled.

For the implemented weighted row, all coordinates remain active and each absolute
difference is shifted left by zero, one, or two bits before the 11-bit
accumulation. This avoids multipliers and hard don't-care coordinates but does
not claim a lane-activity saving. Shift-control reads, wider adders, and the
20-bit payload increase relative to the optimized 54-bit unweighted row must
be charged separately from any reduction in waveform SRAM reads. The
development CSVs expose shift-by-one and shift-by-two lanes separately and
verify row/lane/shift/addition identities for every tested configuration.

`radius_normalized_l1 = d/(r+1)` is only an algorithmic experiment. A hardware
implementation must avoid a divider, for example by cross multiplying two row
scores during selection. Its added multiplier/comparator cost has not been
modeled.

### Range-CAM datapath

Per active row:

- 18 unsigned 5-bit boundary comparisons; and
- an AND reduction to one match bit.

Match uniqueness is not a confidence proof. The controller must not directly
label an event merely because exactly one Range-CAM row matches unless a future
validated policy explicitly reinstates that behavior.

## 4. Directory and indexed candidate memory

The channel/tile directory activates rows; multiple pointers to the same unit
must not duplicate the unit row.

Measured pointer support:

| Family | Training support target | Directory recall | Mean active rows | Mean pointers/unit |
|---|---:|---:|---:|---:|
| HJ | 99% | 0.998 | 15.30 | 5.05 |
| MEArec | 99% | 0.996 | 11.59 | 7.89 |

This is not yet an adequate locality result on HJ. A 1024-channel design must
use physical tiles and local unit populations rather than a global unit mask.

For an anchor-confusion pointer graph with `U` local rows:

\[
b_{\mathrm{unit}}=\lceil\log_2 U\rceil,\qquad
b_{\mathrm{count}}=\lceil\log_2(U+1)\rceil,
\]

\[
B_{\mathrm{graph}}=U b_{\mathrm{count}}+
N_{\mathrm{pointer}}b_{\mathrm{unit}}.
\]

The count field gives the number of contiguous IDs for an anchor. The reference
layout assumes:

```text
anchor_index[U] -> {base_address, candidate_count}
candidate_id_sram[N_pointer]
```

The current software accounting includes count and ID bits but not base
addresses, bank alignment, free-list/update metadata, or SRAM peripherals.

For a prototype-specific channel-support directory with `R` physical rows,
the currently reported pointer payload is:

\[
B_{\mathrm{prototype\ directory}}=
N_{\mathrm{pointer}}\lceil\log_2 R\rceil.
\]

This count intentionally excludes entry-valid bits, per-channel base/count
metadata, banking, decode, and periphery. A unit with several prototype rows
may have several row pointers, but matching rows still collapse to one unit ID
at the candidate FIFO.

## 5. Candidate FIFO contract

The candidate generator emits a stable, de-duplicated list:

```text
candidate_valid
candidate_id[ceil(log2 U)-1:0]
candidate_last
candidate_count[ceil(log2(U+1))-1:0]
candidate_source/ranking metadata
```

Semantics:

- `C=0`: do not read waveform SRAM; invoke the declared spatial fallback;
- `C=1`: do not read waveform SRAM; emit the sole unit ID;
- `C>1`: issue exactly `C` template reads for the conventional centroid-SAD
  engine.

Any alternative pairwise engine must separately count pair comparisons,
template-code reads, and tap-index/mask reads.

## 6. Level-2 waveform memory and datapath

Current measured representations:

| Template | Unit payload | Compute/candidate | Result boundary |
|---|---:|---|---|
| Float teacher | 2048 bits | 64 float L1 | software reference only |
| Signed 4-bit×64 | 256 bits | 64-code SAD | loses >1pp |
| Signed 5-bit×64 | **320 bits** | 64-code SAD | mandatory baseline; accuracy depends on normalization/teacher contract |
| Fisher-48×5 | 240 bits | 48-code SAD | conditional compact option |
| Morphology-12×5 | 60 bits | 12-code SAD | rejected for accuracy |
| Dyadic prefix M×5 | 40/60/80 bits + shared 7M-bit ID table | streaming add/sub/shift + M-code SAD | passes ≥4× bits; rejected for accuracy |
| Pair-conditioned M×5 | 320 bits/unit + O(U²) indices | `(C-1)M` pairwise SAD | accuracy-positive, traffic/storage-negative |

Fisher-48 uses one recording-wide `48×6=288`-bit tap-index table in addition to
per-unit templates. Candidate template traffic does not repeatedly charge this
one global table after configuration. Pair-specific tap tables, by contrast,
must count pair-index access on every invoked pair unless the hardware
demonstrates a cache.

The signed 5-bit waveform code uses `[-15,+15]`; the asymmetric `-16` code is
unused. Templates are formed by quantizing training events, averaging codes per
unit, and rounding/clipping back to integers.

SAD datapath for `M` samples:

1. `M` signed code reads for the query buffer;
2. `M` signed code reads from one candidate template;
3. `M` absolute differences;
4. `M-1` accumulator additions;
5. minimum-distance compare across candidates.

The query waveform is acquired once per event and must not be charged once per
candidate. Template SRAM reads are candidate-dependent.

### Dyadic prefix cost point

The fixed dictionary contains 93 aligned interval means and Haar contrasts.
One selected filter ID is seven bits. For `M∈{8,12,16}`:

```text
per-unit template payload = 5M = 40/60/80 bits
shared ID configuration   = 7M = 56/84/112 bits per recording
```

Means use positive accumulator updates followed by a power-of-two
normalization. Haar contrasts use positive and negative accumulator updates
followed by the same type of normalization. Sample codes, output features, and
unit templates are signed `[-15,+15]`.

The completed four-recording result reaches:

| M | HJ full64×5/sketch bits | MEArec full64×5/sketch bits | Accuracy disposition |
|---:|---:|---:|---|
| 8 | 11.84× | 10.04× | fail |
| 12 | 7.89× | 6.69× | fail |
| 16 | 5.92× | 5.02× | fail |

These are template-bit ratios, not SRAM energy ratios. Query buffering,
feature extraction, selected-ID access, decoder/sense cost, and controller
energy remain separate terms.

## 7. Access and energy model

Until memory macros and RTL exist, report symbolic energy rather than a
fabricated pJ value:

\[
E_{\mathrm{event}} =
E_{\mathrm{encode}}(D,B)
+N_{\mathrm{L1,row}}E_{\mathrm{L1,row}}
+E_{\mathrm{select}}(U,L)
+N_{\mathrm{ptr}}E_{\mathrm{ptr}}
+N_{\mathrm{wf,bit}}E_{\mathrm{wf,bit}}
+N_{\mathrm{SAD}}E_{\mathrm{abs/add}}
+E_{\mathrm{control}}.
\]

Required counters per configuration:

- descriptor encodes;
- active/search row comparisons;
- row-mask bits read and enabled versus physically toggling L1 lanes;
- top-L/WTA operations;
- pointer SRAM bits read and rewritten;
- candidate FIFO entries;
- waveform template bits read;
- absolute differences and accumulator additions;
- template/range/graph update writes; and
- fallback events.

Do not substitute “candidate count” for bit traffic when template widths differ.
Report both.

For staged searches, distinguish:

- simultaneously enabled rows in the inner or outer stage;
- across-stage active-row accesses (`inner + triggered outer`);
- ordinary membership/distance comparisons;
- normalized-outside fallback scans; and
- empty-list all-centroid spatial fallback scans.

For the frozen all-row D9/B5 source, candidate membership and C0 fallback use
one distance scan. The same `N×U` integer L1 matrix feeds the radius compares
and the argmin cache. Required source counters are:

```text
descriptor quantizations = N × 9
centroid-row reads        = N × U
absolute differences     = N × U × 9
L1 reduction additions   = N × U × 8
radius comparisons       = N × U
argmin comparisons       = N × (U - 1)
C0 cache reads/uses      = number of C0 events
```

Do not add a second all-row scan for C0 in this schedule, and do not call C0
“zero cost.” It has zero **Level-2 temporal-template** cost while reusing
already-computed Level-1 state.

For an online contiguous pointer list, count both pointer-slot content rewrites
and count-field metadata rewrites. Set-membership additions/removals alone are
only a lower bound because a frequency-induced reorder can rewrite slots
without changing list membership. The Python runners expose both the legacy
lower-bound fields and the ordered-list-plus-count payload total.

## 8. Latency boundary

The 64-sample window is `[t-15,t+49)`. It contains the aligned sample at `t`
plus 48 strictly later samples through `t+48`. At 30 kHz:

| Last visible aligned sample | Prefix samples | Acquisition delay after alignment |
|---:|---:|---:|
| `t+0` | 16 | 0 ms |
| `t+8` | 24 | 0.267 ms |
| `t+16` | 32 | 0.533 ms |
| `t+32` | 48 | 1.067 ms |
| `t+48` | 64 | 1.600 ms |

The completed raw-prefix float ceiling is within one point of the full teacher
on both families by `t+16`. The tested dyadic code remains far outside the
accuracy gate, so more acquisition delay is not the missing hardware
mechanism for that representation.

\[
48/30000 = 1.60\ \mathrm{ms}.
\]

This is acquisition latency, not compute latency. Report separately:

- event detection/alignment latency;
- future-sample acquisition;
- descriptor encoding;
- Level-1 row evaluation/selection;
- pointer/FIFO access;
- waveform SRAM reads; and
- SAD/reduction cycles.

An “event finishes in N cycles” statement must state whether samples are
already buffered.

The current Python evidence also applies zero-phase Butterworth `filtfilt` to
the full recording before any chronological split. That preprocessing is
non-causal and is not part of a realizable RTL latency claim. Hardware
evaluation must replace it with a forward filter and account for filter state,
group delay, and warm-up, or use a split-local guard interval for offline
equivalence studies.

`algorithms/causal_preprocessing.py` now supplies a forward-only stateful SOS
reference whose chunked and one-pass outputs agree and whose past output is
invariant to future-input perturbations. This closes the primitive-level
filter-state test only. The experiment-level contract still needs an online
detector, a predeclared warm-up/guard rule, continuous-versus-reset boundary
state, delayed descriptor/waveform availability, and discovered-row updates.

## 9. RTL module boundary

Recommended verification modules:

```text
spatial_quantizer
tile_directory
l1_row_array_or_sram_reader
masked_l1_lane_gate            # development ablation
weighted_l1_shift_add          # development ablation
top_l_selector
range_cam_array                 # baseline only
candidate_graph_sram
candidate_union_fifo
waveform_query_buffer
waveform_template_sram
waveform_sad_engine
online_state_updater
event_pipeline_controller
```

Each module needs bit-exact vectors exported from the Python runners. Minimum
corner cases:

- descriptor code 0 and 31;
- distance equal to radius;
- disabled masked-L1 coordinate changed without changing the match;
- all-one mask bit-exact equivalence to the unmasked L1 row;
- weighted shift codes `00/01/10` map to `×1/×2/×4`, while `11` is rejected;
- weighted distance maximum `1116` does not wrap or saturate;
- all-one weighted rows match the independently calibrated unweighted
  semantic reference event-exactly;
- range endpoint equality;
- non-contiguous unit IDs;
- zero, one, and multiple candidates;
- duplicate pointers across two anchors;
- simultaneous timestamps;
- row create/delete/update; and
- accumulator maximum.

## 10. Evidence still required for DAC

1. Causal raw-signal filtering, state initialization, and boundary protocol.
2. Causal centroid/radius/directory update without test-label configuration;
   graph-only updates have already failed the joint gate.
3. High-density 2D evaluation with realistic local unit density and drift.
4. Bit-exact RTL equivalence tests against Python vectors.
5. SRAM/CAM macro choice, banking, and peripheral area.
6. Synthesis, place-and-route, clock/voltage corner, and PVT results.
7. Energy per event versus the original indexed-SRAM baseline, not only versus
   a global waveform brute-force scan.
8. Update bandwidth/endurance and worst-case candidate FIFO sizing.

Until those exist, claims must be phrased as algorithmic payload/access
reductions, not silicon energy or area improvements.
