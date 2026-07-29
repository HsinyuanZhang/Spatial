# Compact Digital Waveform Refinement

Status: **implemented and validated on HJ + MEArec; pair-conditioned 16-tap
5-bit tournaments recover teacher-level accuracy at the high-recall spatial
point, but fail the 4× bit-traffic target and add O(U²) index storage**

This study isolates Level 2 of the proposed memory pipeline. Every waveform
representation receives the exact same frozen spatial candidate list:

```text
frozen 5-bit spatial candidates
        │
        ├── zero candidates ──► spatial fallback, zero waveform reads
        ├── one candidate ────► direct label, zero waveform reads
        └── multiple ─────────► candidate-only digital waveform SAD
```

Implementation:

- `experiments/run_compact_waveform_refinement.py`
- `experiments/run_pair_conditioned_waveform.py`
- `tests/test_compact_waveform_refinement.py`
- `tests/test_pair_conditioned_waveform.py`

## Evaluation protocol

- Event and row source: ground-truth spike times and training unit IDs. This is
  an oracle component-isolation experiment, not end-to-end detected-event
  sorting.
- Split: stable chronological 50/50; equal timestamps stay on the same side.
- Preprocessing: full-recording zero-phase Butterworth filtering precedes the
  split. Candidate/template fitting is chronologically isolated, but the raw
  signal path is offline rather than causal.
- Waveform: 64 main-channel samples `[t-15,t+49)`, peak-normalized per event to
  `[-1,1]`.
- Spatial candidate sources:
  - per-unit 5-bit L1 at the 95th training percentile;
  - per-unit 5-bit L1 at the 99.9th percentile; and
  - all units, as a waveform-only stress case.
- Spatial configuration: fixed `[0,1]` 5-bit quantization, `K=7`, `D=9`, all
  rows active, no top-k, no test-label repair, and no forced nonempty list.
- Every candidate source is constructed once per recording and reused by every
  waveform representation.
- Fisher taps, quantizers, and templates use training rows only.
- Integer templates follow `transform/quantize → per-unit mean → rint/clip`;
  the test set never determines a tap, scale, code, or template.
- All summary numbers below are event-weighted across 12 HJ scenes or 20
  MEArec files. MEArec SNR renderings that share one seed are not independent
  temporal replicates.

## Representations

| Representation | Per-unit payload | Candidate operation | Extra configuration |
|---|---:|---|---:|
| Float teacher | 64×32 = 2048 bits | 64-value L1 | none |
| Full waveform, signed 4-bit | 256 bits | 64-code SAD | none |
| Full waveform, signed 5-bit | 320 bits | 64-code SAD | none |
| Shared Fisher taps, signed 5-bit | `5M` bits | `M`-code SAD | `6M`-bit recording-wide tap-index table |
| Morphology-12, unsigned 5-bit | 60 bits | 12-code SAD | fixed feature definition |

Signed waveform codes use the symmetric range
`[-(2^(B-1)-1), +(2^(B-1)-1)]`; the asymmetric most-negative code is unused so
zero and positive/negative scale are exact. Fisher scores are computed from
training between-unit and within-unit sums of squares, with sample index as the
deterministic tie-break.

The morphology vector contains global min/max, peak-to-peak, absolute peak,
argmin/argmax, three fixed-window means, positive/negative area, and sign
transition count. Its raw mixed ranges are analytically mapped to `[0,1]`
before unsigned quantization. It uses fixed windows, comparisons, and
accumulations—no PCA, FFT, or learned projection.

Per-candidate bit traffic counts template payload only. A Fisher tap-index
table is configuration state loaded once per recording and is not repeatedly
charged on every template read. The CSV records both the shared index bits and
total stored template bits. Unit IDs, waveform-buffer cost, feature-extraction
energy, SRAM peripherals, controller, and PVT margin are not modeled.

## Reproduction

Run from the parent of the `Spatial` package:

```bash
cd /home/xinyuan/SNN_SpikeSorting

# Primary low-bit/compact sweep
python -m Spatial.experiments.run_compact_waveform_refinement \
  --family all --duration 60 \
  --bits 4 5 --taps 8 16 24 \
  --percentiles 95 99.9 --quiet

# Stop-condition extension for larger 5-bit shared tap sets
python -m Spatial.experiments.run_compact_waveform_refinement \
  --family all --duration 60 \
  --bits 5 --taps 32 40 48 56 \
  --percentiles 95 99.9 --quiet \
  --output-dir Spatial/output/compact_waveform_refinement_extended

# Pair-specific Fisher taps with spatially ordered sequential tournaments
python -m Spatial.experiments.run_pair_conditioned_waveform \
  --family all --duration 60 --bits 5 \
  --taps 4 8 12 16 --percentiles 95 99.9 --quiet
```

Generated files:

```text
output/compact_waveform_refinement/
├── compact_waveform_refinement_hj_60s.csv
└── compact_waveform_refinement_mearec_60s.csv

output/compact_waveform_refinement_extended/
├── compact_waveform_refinement_hj_60s.csv
└── compact_waveform_refinement_mearec_60s.csv

output/pair_conditioned_waveform/
├── pair_conditioned_waveform_hj_60s.csv
└── pair_conditioned_waveform_mearec_60s.csv
```

## Result 1: low-bit full waveform

| Family | Spatial source | Candidate recall | Float teacher | 5-bit×64 SAD | Delta | 4-bit×64 SAD | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| HJ | L1 p95 | 0.936 | 0.845 | 0.837 | **−0.82pp** | 0.819 | −2.62pp |
| MEArec | L1 p95 | 0.947 | 0.778 | 0.776 | **−0.22pp** | 0.765 | −1.34pp |
| HJ | L1 p99.9 | 0.993 | 0.761 | 0.734 | **−2.64pp** | 0.704 | −5.63pp |
| MEArec | L1 p99.9 | 0.995 | 0.679 | 0.674 | **−0.49pp** | 0.653 | −2.60pp |

Five bits are sufficient within one percentage point on both families only
when the spatial gate is tight. At the high-recall p99.9 source, HJ remains
2.64pp below the teacher even with all 64 samples. This is a useful boundary:
the HJ loss is caused by a highly ambiguous candidate population whose
decisions are sensitive to small distance perturbations, not only by deleting
waveform samples.

The ambiguous-only conclusions are the same. At p95, HJ teacher/5-bit
ambiguous accuracy is 0.852/0.843 and MEArec is 0.775/0.772. At p99.9 almost
every event is ambiguous, so overall and ambiguous-only accuracies coincide.

## Result 2: selected-tap Pareto

At the tight p95 spatial source, a shared 48-tap 5-bit representation satisfies
the Level-2 teacher-within-1pp target on both families:

| Family | Candidate recall | Teacher | Fisher-48×5 | Delta | Template bits/unit | Global-64×4 bit-traffic factor |
|---|---:|---:|---:|---:|---:|---:|
| HJ | 0.936 | 0.845 | 0.840 | **−0.52pp** | 240 | 3.25× |
| MEArec | 0.947 | 0.778 | 0.771 | **−0.66pp** | 240 | 2.79× |

This operating point is not an architecture pass because its spatial candidate
recall is below 99%.

At the high-recall p99.9 source:

| Family | Teacher | Best tested shared taps | Accuracy | Delta | Payload | Bit-traffic factor |
|---|---:|---|---:|---:|---:|---:|
| HJ | 0.761 | Fisher-40×5 | 0.745 | **−1.59pp** | 200 bits/unit + 240 shared index bits | 1.60× |
| MEArec | 0.679 | Fisher-48×5 | 0.672 | **−0.68pp** | 240 bits/unit + 288 shared index bits | 1.34× |

No single tested tap count meets the one-percentage-point target on both
families at high candidate recall. Adding taps is not monotonic on HJ:
Fisher-40 is better than 48 or 56, indicating that lower-ranked global taps
add noisy distance dimensions. A 56-tap design would also retain 87.5% of the
full 5-bit waveform payload, so it would be a weak hardware contribution even
if it passed.

## Result 3: morphology-12 is not competitive

The current fixed morphology vector is a clear negative result:

| Family | Spatial source | Teacher | Morphology-12×5 | Delta |
|---|---|---:|---:|---:|
| HJ | L1 p95 | 0.845 | 0.648 | −19.72pp |
| MEArec | L1 p95 | 0.778 | 0.688 | −9.05pp |
| HJ | L1 p99.9 | 0.761 | 0.477 | −28.42pp |
| MEArec | L1 p99.9 | 0.679 | 0.538 | −14.07pp |

Do not use this feature set as the paper's digital Level-2 claim. Its compact
payload does not compensate for the separation loss.

## Result 4: pair-conditioned taps recover accuracy, not traffic

The pair-conditioned experiment stores one Fisher tap-index set for every
unordered fitted-unit pair. A test candidate set is stably ordered by raw
5-bit spatial L1. The spatially earliest candidate becomes champion and is
compared sequentially with every challenger. Each binary decision uses only
that pair's tap set and the two ordinary per-unit 5-bit waveform templates.

Integer SAD ties retain the spatially earlier champion. This deterministic
tie-break explains why the full signed-5-bit tournament can differ from the
unit-row-order tie-break in the earlier compact runner even though the frozen
candidate set is identical.

At the p99.9 high-recall candidate source, accuracy and traffic are
event-weighted; storage is the arithmetic mean of per-recording configured
bits:

| Family | Pair taps | Candidate recall | Accuracy | Delta vs teacher | Conservative bit-traffic factor | Total pair+template storage/recording |
|---|---:|---:|---:|---:|---:|---:|
| HJ | 4×5 | 0.993 | 0.730 | −3.11pp | **5.38×** | 8.7 kbit |
| MEArec | 4×5 | 0.995 | 0.612 | −6.64pp | **5.43×** | 7.6 kbit |
| HJ | 8×5 | 0.993 | 0.755 | −0.52pp | 2.69× | 12.1 kbit |
| MEArec | 8×5 | 0.995 | 0.652 | −2.68pp | 2.71× | 10.4 kbit |
| HJ | 12×5 | 0.993 | 0.755 | −0.54pp | 1.79× | 15.5 kbit |
| MEArec | 12×5 | 0.995 | 0.666 | −1.32pp | 1.81× | 13.3 kbit |
| HJ | 16×5 | 0.993 | 0.761 | **+0.04pp** | 1.35× | 18.8 kbit |
| MEArec | 16×5 | 0.995 | 0.669 | **−0.97pp** | 1.36× | 16.1 kbit |

The conservative traffic count includes, for every pair comparison:

- two `M×5`-bit template-code reads; and
- one compressed `M×6`-bit tap-index read.

A fixed 64-bit pair mask is also reported. It gives only 1.54×/1.55× at
`M=16`, so the conclusion does not depend on the compressed-index assumption.

Pair conditioning is algorithmically useful: 16 taps meets the one-point
teacher target on both families at high spatial candidate recall. It is not
yet an efficient memory solution. Every event with `C>1` performs `C-1`
pair comparisons, and all-pair tap tables add O(U²) storage on top of the full
`U×320`-bit unit templates. The 4-tap point passes 4× bit traffic but loses too
much accuracy; the 16-tap point reverses that tradeoff. No tested pair-tap
count passes both accuracy and conservative bit-traffic gates.

At p95, pair-12 reaches −0.12pp/−1.02pp with 4.84×/4.10× traffic on HJ/MEArec,
but that spatial source has only 0.936/0.947 candidate recall. It does not
rescue the full architecture.

## Design decision

- Keep signed 5-bit full-waveform SAD as the mandatory digital baseline, but
  qualify its accuracy by input contract. Under this document's earlier
  per-event-normalized contract it is the strongest simple low-bit point; the
  later raw fit-scale temporal pilot measures −2.27pp/−6.36pp on
  HJ/MEArec versus its primary teacher.
- Keep Fisher-48 as a conditional compact option when a future Level-1 engine
  can provide a small, high-recall candidate set.
- Do not claim that 8/16/24 global taps or morphology-12 preserve accuracy.
- Pair-conditioned taps are an accuracy-positive but storage/traffic-negative
  ablation. Do not omit the pair-index reads or all-pair table.
- The completed signed-5-bit dyadic-prefix branch passes bit traffic but loses
  `6.10/12.08pp` at its least-bad point. It does not replace full64×5 or
  pair-conditioned taps. See
  [`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md).
- The next diagnostic should isolate same-horizon raw-tap, dyadic-projection,
  and feature-quantization losses before choosing a new Level-2 memory.

## Remaining limitations

- GT event times and GT training rows are used.
- Full-recording `filtfilt` uses future raw samples; no deployable causal
  filtering/warm-up boundary has been validated.
- Fisher selection uses GT training labels; an online system must repeat it on
  discovered unit IDs.
- Per-event peak normalization, waveform buffering, and feature extraction are
  not assigned synthesized energy or latency.
- The window contains `t` plus 48 strictly future samples; reaching `t+48`
  requires 1.60 ms of acquisition at 30 kHz, independent of SAD compute
  latency.
- There is no RTL, SRAM macro, synthesis, place-and-route, or PVT evidence.
