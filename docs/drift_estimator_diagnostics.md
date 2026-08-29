# Shift-estimator amplitude bias and the EMA dimension law — results

Status: **two diagnostics, both positive; two earlier conclusions are
narrowed.** Nothing is selected and no tracker is promoted. Date: 2026-08-28.

Companion protocol/provenance:
[`drift_estimator_diagnostics_provenance.md`](drift_estimator_diagnostics_provenance.md).
Documents amended by this record:
[`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md) §8.1 and
[`drift_tracking_results.md`](drift_tracking_results.md) (Q5 reopened), and
[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md)
(the scope of \(\eta^*\)). No number in those documents is rewritten.

This is a **measurement of two existing estimators**, not a new algorithm.
No file under `algorithms/` was modified; `drift_tracking.py` is still at the
digest recorded by the EMA\* knife.

---

## 1. What was asked

Both questions come from a mathematical reading of the deployed code, and both
were predicted before the runs.

| ID | Question | Prediction |
|---|---|---|
| E1 | `estimate_best_shift` minimizes \(\|x-S(\delta)\mu\|^2\) with **no gain profiled out**. Does it convert amplitude error into false motion? | Yes, and asymmetrically: false \(\delta\) for \(\|x\|<\|\mu\|\), little for \(\|x\|>\|\mu\|\) |
| E2 | An EMA with rate \(\eta\) has effective sample size \(N_{\mathrm{eff}}=(2-\eta)/\eta\) against a \(D=2K\)-dimensional template. Is the MEArec 20u collapse a variance effect? | Yes; the collapse rate should move with \(1/D\), so \(K=7\) tolerates a larger \(\eta\) than \(K=12\), and \(K=16\) a smaller one |

### Why E1 was predicted

\(S(\delta)\) is built from `interpolation_matrix`, whose rows are IDW convex
weights. It is therefore an **averaging operator**: a non-lattice \(\delta\)
shrinks a peaked footprint, so \(\|S(\delta)\mu\|<\|\mu\|\). Writing
\(c(\delta)=\|\mu\|^2-\langle\mu,S\mu\rangle\) and
\(s(\delta)=\|\mu\|^2-\|S\mu\|^2\) for an event \(x=g\mu\),

\[
\|g\mu-S(\delta)\mu\|^2=(g-1)^2\|\mu\|^2+2g\,c(\delta)-s(\delta).
\]

For a symmetric contraction \(c\le s\le 2c\), so the objective **decreases**
with \(|\delta|\) exactly when \(g<s/(2c)\), i.e. for under-amplitude events
only. A separate first-order term,
\(f'(0)=(g-1)[\mu^2(r_{\max})-\mu^2(r_{\min})]\), adds a footprint-truncation
contribution that is symmetric in \(g-1\). The prediction is therefore a
**one-sided** dependence with a weaker two-sided component.

---

## 2. Isolation and protocol

| Field | Value |
|---|---|
| Recordings | HJ `drift16c_600s_11`, HJ `static16c_600s_11`, MEArec `rec_v1_units10_snr5_seed202601`, `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s (locked, same four development recordings as the POSNEG/MVM knife) |
| Split | chronological 50 % fit / 25 % cal / 25 % test; equal timestamps unsplit |
| Truth boundary | **GT events, GT unit rows**; \(\mu\) fitted on the fit split only |
| Signal path | full-recording zero-phase Butterworth 300–6000 Hz, order 3 |
| Observation | analog POSNEG extrema in \(\pm15\) samples, candidate-home geometry-\(K\) slots |
| E1 events | **cal + test only.** Fit-split amplitude ratios are centered by construction and would hide the effect |
| E1 \(K\) | 12 |
| E2 split | calibration only. Test is not touched; this is not a selection |
| E2 \(K\) | 7 / 12 / 16 |
| Aggregation | event-weighted within a recording; recordings never averaged together |
| Confirmation | not opened |

E1 evaluates \(\hat\delta\) **against each event's own GT unit**, so it measures
estimator behaviour, not assignment. E2 reuses the deployed `CausalTracker`
in `ema` mode unchanged, including the Frozen-calibration confidence gate.

### Estimator arms in E1

| Arm | Objective minimized over the deployed grid \(\{-0.5,-0.25,0,0.25,0.5\}\) |
|---|---|
| `plain` | \(\|x-S(\delta)\mu\|^2\) — **reproduces the deployed `estimate_best_shift`** |
| `gain` | \(\min_{a\ge0}\|x-a\,S(\delta)\mu\|^2\) — closed-form gain profiled out at each \(\delta\) |
| `gain_sub` | `gain` plus three-point parabolic sub-grid refinement |

`gain_sub` never returns exactly zero, so its `frac_nonzero` is 1.0 by
construction and is not comparable to the other arms. Only its
\(\mathrm{mean}\,|\hat\delta|\) is meaningful.

---

## 3. E1 — the deployed estimator converts amplitude into motion

### Headline

| rec | arm | mean \(\lvert\hat\delta\rvert\) (pitch) | frac nonzero | Spearman \(\rho(\lvert\hat\delta\rvert,\ \|x\|/\|\mu\|)\) |
|---|---|---:|---:|---:|
| HJ drift | `plain` | 0.1569 | 0.509 | **−0.713** |
| HJ drift | `gain` | 0.1167 | 0.399 | −0.429 |
| HJ drift | `gain_sub` | 0.1150 | — | −0.518 |
| **HJ static** | `plain` | 0.0678 | 0.251 | **−0.489** |
| **HJ static** | `gain` | **0.0393** | 0.136 | **−0.058** |
| HJ static | `gain_sub` | 0.0443 | — | −0.092 |
| MEArec 10u | `plain` | 0.0579 | 0.205 | −0.167 |
| MEArec 10u | `gain` | 0.0521 | 0.180 | +0.062 |
| MEArec 20u | `plain` | 0.0651 | 0.212 | −0.066 |
| MEArec 20u | `gain` | 0.0579 | 0.178 | +0.202 |

**HJ static is the clean control**: it has no real drift, so any dependence on
amplitude is artefact. There \(\rho\) goes from −0.489 to −0.058 once a gain is
profiled out, and the mean \(\lvert\hat\delta\rvert\) falls 42 %.

HJ drift keeps \(\rho=-0.43\) after the correction, which is expected and is
**not** evidence of a remaining bug: on a drifting probe amplitude and position
are physically coupled, because a neuron that moves away both attenuates and
shifts.

### Amplitude quintiles (HJ static, the no-drift control)

| \(\|x\|/\|\mu\|\) quintile mean | 0.87 | 0.94 | 1.00 | 1.05 | 1.21 |
|---|---:|---:|---:|---:|---:|
| `plain` mean \(\lvert\hat\delta\rvert\) | **0.192** | 0.076 | 0.014 | 0.010 | 0.047 |
| `plain` frac nonzero | 0.742 | 0.305 | 0.054 | 0.032 | 0.122 |
| `gain` mean \(\lvert\hat\delta\rvert\) | 0.059 | 0.035 | 0.017 | 0.017 | 0.068 |

This is the predicted **asymmetric** shape: the under-amplitude end is 4x the
over-amplitude end under `plain` (0.192 vs 0.047), and the correction flattens
the under-amplitude arm while leaving a small symmetric residual at both
extremes. The residual high-ratio bin is consistent with the
footprint-truncation term and with amplitude-contaminated (overlapping) events;
it was not separated further here.

HJ drift is more extreme: the lowest-amplitude quintile receives
\(\mathrm{mean}\,\lvert\hat\delta\rvert=0.360\) pitch with **99.7 % of events
declared to be in motion**, against 0.041 in the highest quintile.

### How much of the drift-vs-static gap survives amplitude matching

HJ drift events are reweighted so their \(\|x\|/\|\mu\|\) histogram matches HJ
static (10 static-defined bins), then \(\mathrm{mean}\,\lvert\hat\delta\rvert\)
is recomputed.

| arm | HJ drift | HJ static | raw gap | drift after matching | matched gap |
|---|---:|---:|---:|---:|---:|
| `plain` | 0.1569 | 0.0678 | +0.0891 | 0.1138 | **+0.0460** |
| `gain` | 0.1167 | 0.0393 | +0.0774 | 0.0928 | +0.0535 |
| `gain_sub` | 0.1150 | 0.0443 | +0.0706 | 0.0925 | +0.0482 |

**A real motion signal exists** — roughly 0.05 pitch of excess survives
amplitude matching on every arm — but the deployed estimator reports it at
about twice its size.

### Consequence for Q5

[`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md) §8.1 states that the
static \(\lvert\hat\delta\rvert\approx0.05\) pitch shows the estimator "is
trusted enough not to invent large false motion". **That reading must be
narrowed: about half of the static floor is amplitude bias, not noise, and it
is removable.**

The bias is per-unit and per-event, so in a one-factor model
\(\delta_{u,b}=\lambda_u f_b+\epsilon_{u,b}\) it lands entirely in
\(\sigma_\epsilon\), where the observed pairwise correlation is
\(C=\lambda^2/(\lambda^2+\sigma^2)\). Taking the measured
\(C_0=0.153\) gives \(\lambda^2=0.181\sigma^2\); cutting the noise standard
deviation by 42 % (variance to 0.34) would give \(C\approx0.35\).

That extrapolation assumes the drift recording's noise floor scales like the
static one and that block averaging does not change the structure, so it is an
estimate and **not** a measured \(C(d)\). Its only role is to show that the
\(C(d)=0.153<0.3\) verdict sits within the correctable error budget.

**Disposition: Q5 is reopened.** The D5 (smooth field) and shared-gain skips
were licensed by that \(C(d)\) trigger and can no longer be treated as settled.
The \(C(d)\) measurement must be rerun with a gain-profiled, sub-grid estimator
before any conclusion about the spatial structure of drift is reported. The
block design should also be revisited independently: at ~56 events/s spread
over the unit population, a 1 s block gives only a handful of events per
unit-block against a 0.25-pitch grid step.

**Q6 is not overturned by this.** The surviving amplitude-matched motion is
~0.05 pitch, still small, so "Frozen ties the shared-shift trackers on natural
HJ" most likely stands. What changes is the *reason*: not "there is no drift to
track" but "the drift is small and the estimator resolution is comparable to
it".

> **Superseded 2026-08-28 (same day).** The paragraph above is wrong about the
> test window. Direct measurement of the recording — with an amplitude-invariant
> COM readout and the static scene as control — gives a drift-to-noise ratio of
> **0.64** in the 15 s test window: the drift scene shows *less* apparent motion
> there than the no-drift control. The correct statement is the one this
> paragraph rejected: **there is no drift to track in that window**, so Q6's
> natural-HJ arm is untested rather than tied. The ~0.05 pitch of
> amplitude-matched motion measured in §3 is over the **cal+test** events of the
> 60 s window, which is a longer span than the 15 s tracker test. See
> [`drift_window_budget.md`](drift_window_budget.md).

---

## 4. E2 — the EMA collapse rate is set by template dimension

Calibration split, self-EMA, deployed `CausalTracker` unchanged. Collapse is
defined as cal accuracy falling more than 5 pp below the matched Frozen arm at
the same \(K\).

### MEArec 20u (the recording that collapses)

| \(\eta\) | \(N_{\mathrm{eff}}/D\), K=7 | acc K=7 | \(N_{\mathrm{eff}}/D\), K=12 | acc K=12 | \(N_{\mathrm{eff}}/D\), K=16 | acc K=16 |
|---:|---:|---:|---:|---:|---:|---:|
| Frozen | — | 0.9094 | — | 0.9037 | — | 0.9011 |
| 0.005 | 28.5 | 0.9046 | 16.6 | 0.8990 | 12.5 | 0.8951 |
| 0.01 | 14.2 | 0.9042 | 8.3 | 0.8959 | 6.2 | 0.8894 |
| 0.02 | 7.07 | 0.8968 | 4.12 | 0.8877 | 3.09 | **0.8257** |
| 0.03 | 4.69 | 0.8925 | 2.74 | **0.6964** | 2.05 | 0.6587 |
| 0.05 | 2.79 | **0.5872** | 1.62 | 0.6448 | 1.22 | 0.5980 |
| 0.07 | 1.97 | 0.5373 | 1.15 | 0.6067 | 0.86 | 0.5260 |
| 0.1 | 1.36 | 0.4423 | 0.79 | 0.5867 | 0.59 | 0.5152 |

| K | \(D=2K\) | \(\eta_{\mathrm{collapse}}\) | \(N_{\mathrm{eff}}/D\) at collapse | last safe \(N_{\mathrm{eff}}/D\) |
|---:|---:|---:|---:|---:|
| 7 | 14 | 0.05 | **2.79** | 4.69 |
| 12 | 24 | 0.03 | **2.74** | 4.12 |
| 16 | 32 | 0.02 | **3.09** | 6.22 |

Across a 2.3x range of \(D\), the invariant is not \(\eta\) but the effective
sample size **per template dimension**, which sits at 2.7–3.1 at the collapse
point. That gives

\[
\boxed{\;\eta_{\mathrm{crit}}\;\approx\;\frac{2}{3D}\;=\;\frac{1}{3K}\;}
\]

predicting 0.048 / 0.028 / 0.021 for \(K=7/12/16\) against measured
0.05 / 0.03 / 0.02.

A safety margin of \(N_{\mathrm{eff}}/D\ge5\), i.e. \(\eta\le 1/(5K)\), bounds
**catastrophic** failure: no tested point in that region falls more than
2.85 pp below its matched Frozen arm, and on MEArec 20u no more than 1.3 pp.
It does not bound gradual loss — HJ drift at \(K=7\) degrades monotonically
with \(\eta\) from the first grid point (−0.95 pp already at
\(N_{\mathrm{eff}}/D=14\)).

### HJ drift

| \(\eta\) | acc K=7 | acc K=12 | acc K=16 |
|---:|---:|---:|---:|
| Frozen | 0.9419 | 0.9549 | 0.9609 |
| 0.005 | 0.9419 | 0.9561 | 0.9620 |
| 0.02 | 0.9134 | 0.9573 | 0.9620 |
| 0.03 | 0.8956 | 0.9609 | 0.9656 |
| 0.05 | **0.8731** | 0.9585 | 0.9632 |
| 0.1 | 0.8197 | 0.9561 | 0.9644 |

HJ drift never collapses at \(K=12\) or \(K=16\) inside this grid: it has few,
well-separated units and real drift for the EMA to absorb. At \(K=7\) it does
collapse, at \(\eta=0.05\), i.e. \(N_{\mathrm{eff}}/D=2.79\) — **the same
constant as MEArec 20u**.

The two effects are therefore separable and should not be discussed together:

- **template dimension \(D\) sets where the collapse boundary is**;
- **unit density / co-home overlap sets whether a recording ever enters the
  collapse regime at all.**

The variance argument gives a necessary condition and predicts the boundary
location. It does not by itself explain why the failure is catastrophic rather
than gradual; that is the closed loop, since the EMA fixed point is
\(\mathbb E[x\mid x\in R_u(\mu)]\) and the assignment region \(R_u\) is itself a
function of \(\mu\).

### Consequence for \(\eta^*\)

[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md) selects
one global \(\eta^*=0.005\). That selection is **not wrong**, but its scope must
be stated:

1. \(\eta^*\) was selected at \(K=12\) and **must not be carried to another
   \(K\), template dimension, or probe density without rechecking.** At
   \(K=16\), \(\eta=0.02\) already costs 7.5 pp on MEArec 20u.
2. \(\eta^*=0.005\) is about 6x below the \(K=12\) collapse boundary and about
   3x below the \(1/(5K)\) safe bound. HJ drift is measurably better at
   \(\eta=0.02\)–\(0.03\) (0.9573 / 0.9609 vs 0.9561 on cal), so the global
   event-weighted rule may be leaving accuracy unused. **This is an
   observation on calibration, not a reselection**; \(\eta^*\) stays 0.005.
3. The rule \(\eta\le1/(5K)\) is the extrapolation instrument for future
   probes, including the Yger 252-channel and dense NP inputs, where a
   grid search per dataset is not a satisfying answer.

---

## 5. What this does not show

- It does not measure a corrected \(C(d)\). The \(C\approx0.35\) figure in §3 is
  an extrapolation under stated assumptions, not a measurement.
- It does not propose or select a corrected estimator. `gain` and `gain_sub`
  are **diagnostic arms**, evaluated on GT-unit assignment only; neither was
  run through classification, margins, or a causal stream.
- It does not touch the free-shift \(L_{\mathrm{flex}}\) result. That failure is
  structural (a per-candidate nuisance parameter aligned with the
  discriminant direction) and is unaffected by gain profiling.
- It does not revisit the controlled-drift results, which have a separate
  problem: `apply_spatial_ramp` perturbs with the same interpolation operator
  the tracker inverts, and on already-extracted extrema, so those runs cannot
  detect misspecification of \(S(\delta)\).
- No test-split number is used for any decision here.

## 6. Next

1. Rerun the Q5 coherence measurement with a gain-profiled, sub-grid estimator,
   precision-weighted by per-unit-block event counts, and with a block length
   chosen so the per-unit-block standard error is below the drift amplitude
   under test.
2. Only if that \(C(d)\) clears its pre-registered trigger, reconsider D5 and
   shared gain.
3. Keep \(\eta^*=0.005\) at \(K=12\); record \(\eta\le1/(5K)\) as the scope rule
   rather than re-tuning on this data.

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.run_drift_estimator_diagnostics --pilot --quiet
```

Outputs and digests:
[`drift_estimator_diagnostics_provenance.md`](drift_estimator_diagnostics_provenance.md).
