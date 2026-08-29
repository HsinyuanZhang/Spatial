# Robust P2P Template Matching and Online Update — Pre-Registered Plan

Status: **Phase A complete; independent confirmation failed the estimator gate;
8-bit quantization equivalence passed conditionally**.

Date: 2026-08-11.

## Research question

Can a temporally robust estimate of each channel's peak-to-peak amplitude,
combined with 4–8-bit relative-footprint templates, improve P2P template
matching without increasing the spatial descriptor width, and can the selected
integer template be updated online without the self-label feedback failure of
the existing unconditional EMA?

The study is deliberately decomposed:

1. choose the P2P estimator while holding the patch and matcher fixed;
2. choose the smallest safe descriptor bit width while holding the estimator
   fixed;
3. evaluate online updates only after a static estimator/bit configuration has
   passed.

No online result may be used to retroactively choose the static estimator or
bit width.

## Clarifying the P2P operation

The current implementation is not “one maximum sample.” It is the difference
between two single-sample extrema in an aligned channel window:

```text
P2P_raw(x) = max_t x[t] - min_t x[t]
```

Both extrema are susceptible to extreme-value noise. “Average 3–5 points” has
at least three non-equivalent meanings:

1. average contiguous samples around the raw maximum and minimum;
2. average first, then take the maximum and minimum of the moving average;
3. average the globally largest and smallest order statistics, regardless of
   temporal adjacency.

They require different hardware and have different failure modes, so Phase A
tests all three rather than silently choosing one interpretation.

## Starting evidence

- The current relative-offset P2P pilot improves nearest-template assignment,
  but still uses single-sample extrema.
- The current 4/5/6-bit shift sweep shows low-bit L1 and normalized correlation
  can track float closely, but estimator and bit depth were not crossed.
- The existing self-predicted integer EMA loses 3.0 percentage points against
  frozen templates on the six HJ drift scenes. Most `lr_shift=4` residuals are
  also below the truncation grain.
- Navajas et al. found that at least 6-bit signal resolution preserved template
  matching in their particular waveform/input-range protocol. That motivates
  measuring 4–8 bits here, but does **not** establish that the Spatial P2P
  descriptor needs or only needs 6 bits.
- OSort and spatial-window OSort demonstrate online mean-template updates, but
  direct self-assignment updates can pollute a template. Spatial therefore
  requires an explicit confidence gate and write ledger.

Primary references:

- [Minimum requirements for accurate and efficient real-time on-chip spike sorting](https://pmc.ncbi.nlm.nih.gov/articles/PMC4151286/)
- [Online detection and sorting of extracellular action potentials (OSort)](https://pubmed.ncbi.nlm.nih.gov/16488479/)
- [Spatial Information Based OSort for Real-Time Spike Sorting Using FPGA](https://real.mtak.hu/119039/1/Schaffer_et_al_2021_IEEE_TBME.pdf)
- [Spike sorting with Kilosort4](https://www.nature.com/articles/s41592-024-02232-7)
- [Normalized template matching for extracellular spike detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6700190/)

## Structured divergence

The following candidates were generated before filtering.

| ID | Candidate | Potential benefit | Main problem |
|---|---|---|---|
| A1 | Raw `max-min` | Zero new cost; mandatory baseline | Extreme-value noise |
| A2 | Mean around raw max/min, k=3 | Direct reading of “average the peak” | A noise sample can still choose the center |
| A3 | Mean around raw max/min, k=5 | More variance reduction | Blurs narrow phases more strongly |
| A4 | Moving-average range, k=3 | Running sum plus min/max; suppresses isolated noise before selection | Attenuates very narrow spikes |
| A5 | Moving-average range, k=5 | Stronger noise suppression | Greater temporal smoothing and accumulator width |
| A6 | Top/bottom-k order-statistic mean, k=3 | Robust to any one extreme | Needs selection/sorting hardware; samples may be disjoint |
| A7 | Top/bottom-k order-statistic mean, k=5 | Stronger extreme robustness | Can average unrelated phases and noise |
| A8 | Winsorized range | Limits outliers | Requires quantile/order-statistic estimation |
| A9 | Quantile range | Statistically robust | Expensive and window-size dependent |
| A10 | Negative trough only | Simple for negative-going spikes | Loses positive phase and polarity generality |
| A11 | Local RMS/energy | Smooth and shift tolerant | Not P2P; amplitude energy can erase footprint shape |
| A12 | Fixed phase-region means | Very cheap after alignment | Depends strongly on filter and alignment contract |
| A13 | Adaptive k from spike width | Preserves narrow and broad spikes | Control/state complexity and new tuning leakage |
| A14 | Joint `[P2P_raw, P2P_avg]` per channel | Keeps sharp and robust cues | Doubles descriptor width and memory rows |
| A15 | Median of several short-block P2Ps | Rejects local outliers | Multiple peak finders and median network |
| A16 | Average across neighboring channels before P2P | Spatial denoising | Mixes nearby units; damages the fixed patch semantics |

## Convergence

The filters are: temporal adjacency, hardware simplicity, no descriptor-width
growth, clear causal implementation, and a falsifiable benefit on both probe
families.

The five retained ideas are:

1. **Moving-average range k=3** — primary candidate.
2. **Moving-average range k=5** — stronger smoothing boundary probe.
3. **Centered peak/trough mean k=3** — direct interpretation control.
4. **Centered peak/trough mean k=5** — direct interpretation boundary.
5. **Top/bottom-k mean k=3/5** — software diagnostic for whether adjacency is
   essential; not the preferred hardware path.

The selected first hypothesis is moving-average range:

```text
y_k[t] = (1/k) * sum_{i=0}^{k-1} x[t+i]
P2P_MA,k(x) = max_t y_k[t] - min_t y_k[t]
```

For a fixed k and per-event footprint normalization, hardware may retain the
running **sum** rather than divide by k; the common factor cancels. The running
sum needs add/subtract, k-sample state, and approximately `ceil(log2(k))`
guard bits before the extrema registers.

### Two-sentence pitch

Single-sample P2P is a noisy extreme statistic, but replacing it with an
unstructured average can erase the spatial shape that makes the footprint
useful. We test a streaming boxcar-before-range estimator that suppresses
isolated temporal noise with one running sum while preserving the same K=7
relative descriptor and exact low-bit template interface.

### Strongest objection

The apparent gain may come only from low-pass smoothing that attenuates narrow
but discriminative spike phases; on clean or low-SNR-cross-family data it can
reduce inter-unit separation more than intra-unit variance.

Response: compare every estimator against raw P2P on identical event rows,
report both within-unit distance and true-vs-wrong margin, retain k=3 and k=5
as separate boundary points, and require no material loss on either HJ or
MEArec before confirmation.

## Phase A — estimator and bit-width pilot

### Locked development recordings

| Family | Recording | Role |
|---|---|---|
| HJ | `drift16c_600s_11` | Real drift development |
| HJ | `static16c_600s_11` | Static negative control |
| MEArec | `rec_v1_units10_snr5_seed202601` | Moderate-density synthetic development |
| MEArec | `rec_v1_units20_snr5_seed202601` | Dense synthetic development |

Duration is 60 seconds. These inputs already have locked hashes in
`dac_experiment_inputs.sha256`.

### Signal and feature contract

- Existing full-recording 300–6000 Hz filtering for comparability; this is not
  a causal end-to-end result.
- GT event times and event labels for component isolation.
- Existing event peak/home-channel alignment.
- Relative 1D primary-axis patch, K=7, out-of-bounds slots zero.
- P2P temporal window identical to the current relative/shift experiments.
- Per-event max-normalized footprint after P2P estimation.
- Equal-timestamp-safe chronological 50/25/25 fit/calibration/test split.
- Unit templates fit on the first 50% only.

### Estimator grid

```text
raw_range, k=1
centered_peak_mean, k in {3,5}
moving_average_range, k in {3,5}
extreme_mean_range, k in {3,5}
```

`extreme_mean_range` is a diagnostic upper bound on non-adjacent order-statistic
averaging. It is not eligible as the hardware winner unless a later protocol
adds an explicit selection-network cost.

### Quantization grid

```text
float, 4, 5, 6, 7, 8 bits
```

For b-bit rows:

```text
q = round((2^b - 1) * normalized_P2P)
template_u = round(mean_fit(q | unit=u))
```

Integer L1 uses the stored integer query and integer template directly. The
runner must not divide codes back to float before integer-L1 matching.

### Matcher grid

- Primary: all-row L1, shift radius S=0.
- Secondary: normalized correlation, S=0.
- Shift is held at zero to isolate the estimator and bit-width effects.
- The selected estimator/bit pair may enter a later S=0/1 confirmation; Phase A
  cannot choose k or b from shifted results.

### Required metrics

Per recording and configuration:

1. closed-set P2P template assignment accuracy;
2. integer correct count and event denominator;
3. normalized true-unit L1 distance;
4. true-vs-nearest-wrong margin and positive-margin fraction;
5. per-unit worst assignment accuracy;
6. template bits (`U*K*b`) and arithmetic width;
7. delta against the matched estimator's float result;
8. delta against raw float P2P;
9. static and drift rows kept separate.

### Estimator gate

An averaging estimator qualifies only if, at float primary L1:

1. HJ and MEArec family-mean assignment are each no worse than raw by more
   than 0.5 percentage points;
2. no development recording loses more than 1.0 percentage point;
3. HJ drift either improves assignment by at least 0.5 percentage points or
   reduces normalized true-unit L1 by at least 2% while not reducing the
   positive-margin fraction;
4. the secondary cosine result is no worse than raw cosine by more than 0.5
   percentage points on either family.

Eligible estimators are ranked by: number of safeguards passed, HJ-drift
assignment, two-family mean assignment, positive-margin fraction, then lower k.
If none qualifies, selection is `raw_range` and temporal averaging is closed as
the main branch.

### Bit-width gate

For the selected estimator, choose the smallest b in 4–8 such that primary L1:

1. family-mean assignment is within 0.5 percentage points of float on HJ and
   MEArec;
2. every development recording is within 1.0 percentage point of float;
3. positive-margin fraction loses at most 0.5 percentage points on each family.

Ties go to fewer bits. If no b qualifies, selection is null and the estimator
does not advance to online updating.

### Confirmation boundary

If and only if both gates produce a non-null estimator/bit pair, freeze that
pair and confirm on the remaining 10 HJ scenes plus the disjoint MEArec seed
202600 SNR5 recordings. Confirmation may not retune k, b, matcher, scale, or
thresholds.

Phase A selected `centered_mean_k5` and 8-bit integer L1. The confirmation
runner must read and verify that frozen selection file rather than recompute a
winner from the confirmation data. It evaluates only these six controls per
recording:

```text
raw float       x {L1, cosine}
centered k=5    x {float, 8-bit} x {L1, cosine}
```

The confirmation result passes only if all of the following pre-run gates
hold:

1. selected-estimator float L1 is no worse than raw float L1 by more than 0.5
   percentage points on either HJ or MEArec event-weighted family accuracy;
2. no individual recording loses more than 1.0 percentage point;
3. across the five held-out HJ drift recordings, selected-estimator float L1
   either gains at least 0.5 percentage points in event-weighted accuracy, or
   reduces event-weighted normalized true-template distance by at least 2%
   without reducing positive-margin fraction;
4. selected-estimator float cosine is no worse than raw float cosine by more
   than 0.5 percentage points on either family;
5. selected 8-bit L1 is within 0.5 percentage points of selected float L1 on
   each family, within 1.0 percentage point on every recording, and within 0.5
   percentage points in positive-margin fraction on each family.

These are confirmation gates, not a second model-selection opportunity. A
failure records the boundary and stops self-predicted Phase B; it cannot be
repaired by selecting another estimator or bit width on these recordings.

## Phase A2 — causal corruption boundary

This experiment tests the reason for the expected gain rather than only the
aggregate outcome.

Use frozen event windows from Phase A and apply deterministic test-only
perturbations after the common filter:

1. isolated one-sample impulses with fixed seed and amplitude in noise-sigma
   units;
2. ±1/±2-sample event-alignment jitter;
3. no perturbation control.

Report accuracy and margin degradation relative to each clean configuration.
Moving-average range should specifically resist impulses; it is not expected to
solve misalignment. A method that gains only under synthetic corruption but
loses clean cross-family accuracy does not advance.

## Phase B — online integer template update

Phase B was registered before the static results, but a self-predicted data run
requires the independent estimator confirmation to pass. That gate failed, so
only the bit-exact update primitive and its synthetic contract tests are
implemented in this study; the self-predicted dataset run is stopped.

### Why the existing EMA is insufficient

The existing update is:

```text
T_u <- T_u + trunc_toward_zero((q - T_u) / 2^s)
```

At 4 bits, every residual has magnitude at most 15. With `s=4`, every direct
residual update is zero. At 5 bits, only residuals of at least 16 change a row.
Reducing s increases write activity but also amplifies wrong-label feedback.

### Proposed primary update: confidence-gated error-feedback EMA

For winner u and integer query q:

```text
e      = q - T_u
A_u    = A_u + e
delta  = trunc_toward_zero(A_u / 2^s)
T_u    = clip(T_u + delta, 0, 2^b - 1)
A_u    = A_u - delta * 2^s
```

`A_u` is a small signed residual accumulator per coordinate. It preserves
sub-LSB evidence until it is large enough to cause an integer template change,
removing the deterministic dead zone without random rounding or a divider.

If matching uses a nonzero spatial shift later, the query must first be inverse
shifted by the winning alignment before it enters the canonical template
update; otherwise the online template will blur across offsets.

### Confidence gate

Update only when all conditions pass:

1. nearest-template distance `d1` is within a per-unit calibration radius;
2. L1 margin `d2-d1` is above a calibration-only lower bound;
3. no empty/fallback path was used;
4. the predicted row is stable under the registered directory/shift path;
5. the event is not marked as an overlap/low-confidence event by the available
   controller.

No GT label is consulted during deployed updates. GT is used after the run only
to audit accepted-update contamination.

### Memory and write policy

- Keep a frozen anchor template for rollback and audit.
- Keep the adaptive working template and residual accumulator in SRAM/register
  state.
- Do not program an RRAM row on every accepted spike.
- Commit changed rows only every B accepted events, with `B in {8,16}` fixed
  before confirmation.
- Count accepted events, changed coordinates, SRAM bits written, RRAM row
  commits, and accumulator state bits separately.

For residual range after each feedback step, the remainder magnitude is below
`2^s`; after adding a new b-bit residual, roughly `max(b,s)+2` signed bits per
accumulator coordinate are sufficient. The executable implementation must
derive and assert its exact bound.

### Online comparison arms

1. frozen template;
2. existing unconditional direct EMA;
3. confidence-gated direct EMA;
4. confidence-gated error-feedback EMA;
5. power-of-two block-mean update, B=8 or 16;
6. oracle-labelled error-feedback update as a diagnostic upper bound only.

### Online metrics and kill criteria

- sequential accuracy in fixed time blocks;
- online minus frozen accuracy;
- static-scene regression;
- accepted-update fraction;
- GT-audited wrong-update fraction;
- per-unit update coverage;
- centroid coordinate writes and committed row writes;
- recovery relative to the oracle-update ceiling;
- template distance from frozen anchor over time.

The branch passes development only if it improves HJ drift by at least 1.0
percentage point over frozen, loses no more than 0.5 percentage points on HJ
static, has GT-audited wrong updates below 0.5%, updates at least 5% of events,
and reduces committed nonvolatile row writes by at least 8x relative to
per-accepted-event commits. If the oracle-update arm cannot improve frozen by
1.0 point, self-updating is stopped under this representation.

## Three validation experiments

1. **Estimator x bit-width isolation:** the frozen four-recording grid above.
2. **Impulse and alignment boundary:** test whether improvement is caused by
   the claimed temporal-noise mechanism.
3. **Sequential drift update:** frozen vs direct EMA vs confidence/error
   feedback vs oracle, with static negative control and full write accounting.

## Two-week feasibility pilot

### Week 1

1. Implement bit-exact `raw`, centered-mean, moving-average-range, and
   extreme-mean estimators with synthetic unit tests.
2. Add the frozen four-recording runner and integer-template invariants.
3. Run Phase A, independently recompute gates, and freeze estimator/bit or
   record null.
4. If non-null, run deterministic impulse/jitter boundary tests.

### Week 2

1. Implement error-feedback accumulator and exact state/write accounting.
2. Fit confidence radii/margins on calibration only.
3. Run online arms on one HJ drift/static development pair.
4. Audit wrong updates with GT after the run.
5. Continue to the remaining HJ drift scenes only if the oracle and confidence
   gates make improvement feasible.

## Claim boundary

Even a passing Phase A supports only a more stable low-bit P2P template feature
under GT-event/GT-row component isolation. A passing Phase B would support an
online integer template-update component, not a causal detected-event sorter or
an RRAM endurance/PPA claim. Those require the full signal path and physical
memory evaluation separately.

## Frozen outcome

Phase A selected `centered_mean_k5` and 8 bits. On the independent 12-recording
confirmation, the estimator gate failed: held-out HJ drift accuracy decreased
by 1.05 percentage points, one drift recording decreased by 6.65 points, and
the HJ cosine safeguard missed its boundary. The 8-bit representation itself
passed every float-equivalence safeguard for the selected estimator, but that
does not rescue the rejected estimator or prove that 8 bits is the minimum for
raw P2P.

The complete tables, hashes, and online-update implementation boundary are in
[`robust_p2p_template_results.md`](robust_p2p_template_results.md).
