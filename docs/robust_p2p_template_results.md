# Robust P2P Template Matching and Online Update — Results

Status: **averaged-estimator hypothesis rejected on independent confirmation;
8-bit equivalence passes for the frozen estimator; online data run stopped**.

Date: 2026-08-11.

## Executive result

Three conclusions survive the registered experiment:

1. Averaging 3–5 samples is mathematically and computationally feasible, but
   **a fixed five-point average is not a reliable global replacement for raw
   P2P** on the current corpus. The development winner failed on held-out HJ
   drift recordings.
2. For the frozen `centered_mean_k5` estimator, only **8-bit integer L1** met
   all development accuracy, per-recording, and positive-margin safeguards;
   its float-equivalence then passed on 12 held-out recordings. Because the
   estimator itself failed, this is a conditional quantization result, not an
   endorsement of the full averaged-P2P configuration.
3. A bit-exact confidence-gated error-feedback update is implemented and unit
   tested, but the self-predicted online data experiment was **not run** after
   confirmation failed. This preserves the pre-registered stop rule and avoids
   turning a rejected static representation into an online claim.

## What “average P2P” meant

The historical feature is already a range of two extrema, not one maximum:

```text
P2P_raw(x) = max(x) - min(x)
```

The pilot separated three non-equivalent averages:

| Estimator | Definition | Streaming implication |
|---|---|---|
| Centered mean | Find the raw max/min indices, then average a contiguous k-point neighborhood around each | Two extrema indices plus two short sums; an impulse can still choose the centers |
| Moving-average range | Run a k-point boxcar first, then take its max-min | Running add/subtract, k-sample delay state, and extrema; strongest hardware candidate before results |
| Extreme mean | Mean the global top-k minus bottom-k samples | Non-adjacent selection/sorting network; software diagnostic only |

The implementation requires odd `k`, keeps exactly k samples at a window
edge, and leaves the historical raw path unchanged by default.

## Experimental contract

- GT event times and GT unit rows: representation/template component isolation.
- Full-recording 300–6000 Hz filtering: not a causal end-to-end path.
- Relative primary-axis patch K=7, shift S=0, per-event max normalization.
- Equal-timestamp-safe chronological 50/25/25 fit/calibration/test partition.
- Templates fitted on the first 50% only.
- Primary integer L1 uses integer queries and rounded integer template means
  directly; codes are not converted back to float before matching.
- Secondary normalized correlation protects against an L1-only artifact.
- Every estimator uses the identical event-row identity within a recording.

Development inputs were HJ `drift/static16c_600s_11` and MEArec SNR5 seed
202601 with 10/20 units. Confirmation used the other ten ready HJ 16-channel
600-second scenes and MEArec SNR5 seed 202600 with 10/20 units. Every run used
the first 60 seconds of its recording.

## Phase A: estimator screening

Event-weighted float assignment accuracy on the four development recordings:

| P2P estimator | HJ L1 | MEArec L1 | HJ cosine | MEArec cosine | Registered disposition |
|---|---:|---:|---:|---:|---|
| Raw range | 0.6305 | 0.6727 | 0.6221 | 0.6774 | Baseline |
| Centered k=3 | 0.6338 | 0.6808 | 0.6227 | 0.6901 | Fails HJ-drift mechanism gate |
| **Centered k=5** | **0.6565** | 0.6895 | **0.6416** | 0.6933 | Pass; frozen development winner |
| Moving average k=3 | 0.6344 | 0.6797 | 0.6221 | 0.6938 | Fails HJ-drift mechanism gate |
| Moving average k=5 | 0.6435 | **0.6938** | 0.6338 | **0.6970** | Pass; ranked second |
| Extreme mean k=3 | 0.6312 | 0.6805 | 0.6227 | 0.6883 | Hardware-ineligible diagnostic |
| Extreme mean k=5 | 0.6338 | 0.6895 | 0.6240 | 0.6947 | Hardware-ineligible diagnostic |

The centered k=5 winner gained 3.68 percentage points over raw on the single
development HJ drift scene. Moving-average k=5 had lower normalized true-unit
L1 on every development recording and sometimes higher static/MEArec accuracy,
but the frozen ranking prioritized HJ-drift assignment, so it was not selected
after seeing confirmation data.

## Phase A: 4–8-bit screening

For the selected centered k=5 estimator:

| Codes | HJ accuracy | MEArec accuracy | HJ positive margin | MEArec positive margin | Gate |
|---|---:|---:|---:|---:|---|
| Float | 0.6565 | 0.6895 | 0.6565 | 0.6895 | Reference |
| 4 bit | 0.6539 | 0.6722 | 0.6487 | 0.6421 | Fail |
| 5 bit | 0.6416 | 0.6716 | 0.6364 | 0.6609 | Fail |
| 6 bit | 0.6487 | 0.6898 | 0.6448 | 0.6855 | Fail |
| 7 bit | 0.6526 | 0.6892 | 0.6500 | 0.6872 | Fail: HJ margin misses beyond the tolerance |
| **8 bit** | **0.6565** | **0.6892** | **0.6552** | **0.6881** | **Pass** |

This table explains why classification accuracy alone is insufficient for an
online decision. Six- and seven-bit assignment can look close to float while
integer ties reduce the strictly positive true-vs-wrong margin; a tied winner
selected by array order is not trustworthy enough to update a template.

Raw P2P showed a different, exploratory development curve: 4 bit lost heavily
on MEArec, 5 bit lost on HJ, while 6–8 bit were close to float at family level.
Raw bit width was not the registered selection and was not included in the
independent confirmation, so the minimum safe width for a raw-P2P mainline is
still open.

## Independent confirmation

The confirmation runner read the frozen selection JSON and evaluated only raw
float, centered-k5 float, and centered-k5 8-bit controls. It did not reselect
k, bits, matcher, scale, or thresholds.

### Estimator result

| Family / condition | Raw float L1 | Centered-k5 float L1 | Delta | Result |
|---|---:|---:|---:|---|
| HJ, all 10 held-out scenes | 0.7240 | 0.7219 | -0.21 pp | Family L1 safeguard passes |
| HJ, five drift scenes | 0.6814 | 0.6709 | **-1.05 pp** | Mechanism gate fails |
| MEArec, seed 202600 | 0.7448 | 0.7491 | +0.43 pp | Pass |

The HJ cosine control changed from 0.7390 to 0.7334, a loss of 0.56 percentage
points against a 0.50-point limit. The per-recording guard also failed:
`drift16c_600s_21` fell from 0.7531 to 0.6866 (-6.65 points), and
`drift16c_600s_22` fell by 2.29 points. Other recordings improved, which is
exactly why an aggregate-only conclusion would be misleading.

The drift normalized true-template L1 became 3.96% worse rather than 2% better.
Consequently the estimator gate and the overall confirmation gate are false.

### Conditional 8-bit result

| Family | Centered-k5 float | Centered-k5 8-bit | Accuracy delta | Positive-margin delta | Quantization gate |
|---|---:|---:|---:|---:|---|
| HJ | 0.721893 | 0.721827 | -0.007 pp | -0.166 pp | Pass |
| MEArec | 0.749100 | 0.748119 | -0.098 pp | -0.196 pp | Pass |

All per-recording 8-bit safeguards also pass. This supports the statement
“8-bit integer matching reproduces float for centered-k5 under this contract,”
not “centered-k5 is better” and not “8 bit is universally required for P2P.”

## Online update calculation

The old direct integer EMA is:

```text
delta = trunc_toward_zero((q - T[u]) / 2^s)
T[u]  = clip(T[u] + delta)
```

It has a deterministic dead zone: at `s=4`, every coordinate residual with
magnitude below 16 produces zero. Lowering s raises write activity and makes a
wrong self-label more destructive.

The implemented alternative retains the discarded evidence:

```text
e       = q - T[u]
A[u]    = A[u] + e
delta   = trunc_toward_zero(A[u] / 2^s)
T_new   = clip(T[u] + delta, 0, 2^b - 1)
A[u]    = A[u] - (T_new - T[u]) * 2^s
T[u]    = T_new
```

For b=8, s=4, K=7:

- adaptive template: 56 bits/unit;
- frozen rollback anchor: another 56 bits/unit;
- post-feedback accumulator state: `s+1 = 5` signed bits/coordinate, or 35
  bits/unit;
- intermediate accumulator addition: 10 signed bits/coordinate because the
  exact magnitude bound is `(2^4-1) + (2^8-1) = 270`;
- L1 range: 0–1785, requiring 11 unsigned bits;
- a commit period B=16 permits at most one dirty nonvolatile-row commit per 16
  accepted updates for that unit, while every event matches against the SRAM
  shadow.

An event may update only if all of these are true:

```text
d1 <= frozen per-unit calibration radius
d2 - d1 >= frozen per-unit margin floor >= 1
stable registered routing/alignment path
no fallback, overlap, or controller low-confidence flag
```

No GT label enters the update API. GT may only audit the wrong-update fraction
afterward. A shifted matcher must inverse-align the winning query before this
canonical template update, or spatial jitter will be averaged into the row.

The module keeps three explicit images: immutable fit anchor, adaptive SRAM
template, and periodically committed nonvolatile template. It counts template
coordinate writes, accumulator writes, dirty rows, row commits, and logical
bits written. These counts are an algorithmic ledger, not energy/endurance/PPA
evidence.

Because independent estimator confirmation failed, only synthetic bit-exact
contract tests were run. Confidence thresholds were not fitted on the failed
descriptor, and no self-predicted HJ result is claimed.

## Reproduction and identity

Run from the parent workspace root:

```bash
MPLCONFIGDIR=/tmp/robust-p2p-mpl \
  python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --pilot --quiet

MPLCONFIGDIR=/tmp/robust-p2p-confirm-mpl \
  python -m Spatial.experiments.run_robust_p2p_template_pilot \
  --confirmation --quiet

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  Spatial/tests/test_spatial_footprint.py \
  Spatial/tests/test_robust_p2p_template_pilot.py \
  Spatial/tests/test_online_p2p_templates.py
```

Recorded artifact hashes:

```text
84b0ba849c5225927908839cbdf54838e49f0d35b62e5ad892ee296453f82258  robust_p2p_template_pilot.csv
b5ed6f7a6538fb83ba84ced3e4ebcfd7fcec91c527a8d839aa24ce5fe822cb6a  robust_p2p_template_pilot_summary.json
4c67d6415e6fb04e12693ffe75e249a240b8e53f6596a7aa2ec7e5be4b8c2399  robust_p2p_template_pilot_selection.json
9d7635d28e07792d70db8f3a1c0e2936e444dfb620ac15a240c25bac84acb1aa  robust_p2p_template_confirmation.csv
0d97d3f71c8fb50958a573aea1d53caa3370289392d348b9ca84c21538ec62f3  robust_p2p_template_confirmation_summary.json
```

Generated artifacts live under
`output/robust_p2p_template_pilot/` and are gitignored.

## Research interpretation and next experiment

The strongest objection to averaging was that temporal smoothing could erase
narrow, unit-specific phases. Confirmation supports that objection: effects
vary sharply across HJ drift scenes, while MEArec improves modestly.

The next defensible hypothesis is therefore **conditional robustness**, not a
different globally fixed k chosen after the fact. Pre-register a selector from
fit/calibration-only observables such as spike width, peak-to-noise ratio, and
raw-versus-averaged template disagreement; allow it to choose raw or k=3/5 per
unit, then validate on a new temporal block or genuinely new recording. In
parallel, confirm raw-P2P 6/7/8-bit positive-margin behavior independently
before choosing the bit width for an online mainline.

## Claim boundary

This is GT-event/GT-row component isolation with full-recording filtering. It
does not validate detection, discovered unit rows, causal filtering, a
confidence-calibrated self-update loop, RRAM endurance, or physical PPA. The
negative estimator result and the conditional 8-bit result must remain
separate.
