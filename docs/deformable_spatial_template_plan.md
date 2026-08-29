# Deformable spatial template — plan

Frozen Phase-1 protocol. This is not a result, not a memory-row claim, and
not a reopening of Mapping Row / interval QDA.

## Question

Can a neuron’s POSNEG footprint be written as a stable identity plus a
small, continuous, trackable spatial state, such that co-home units stay
separable while gain, fractional offset, polarity, and mild shape change
are explained by state rather than by rewriting \(\mu_u\)?

Not a drift-invariant embedding: identity stays put, state stays observable.

## Locked fields (Phase 1)

| Field | Value |
|---|---|
| Recordings | locked four: HJ `drift16c_600s_11`, `static16c_600s_11`; MEArec `rec_v1_units10_snr5_seed202601`, `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s |
| Split | chronological 50 / 25 / 25, equal timestamps unsplit |
| Observation | analog POSNEG only; no waveform, dyadic, or pair-tap |
| K | `{7, 12, 16}` from **geometry** around a reference home, not event-argmax channels |
| Default frame | **candidate-home**: slots are physical offsets from fit majority home \(h_u\) |
| Ablation frame | event-argmax recentering, C0 and C3 at K=12 only |
| Model | \(x = a\,P(\rho)\,S(\delta)\,(\mu_u + B h)+\epsilon\) |
| \(\delta\) search | 1-D along the primary probe axis, ±1.5 pitch, 21-point grid |
| \(\dim h\) | 0 (off), 1 (C5), 2 (C6); **shared** \(B\) from fit residuals after gain/offset/polarity |
| \(D_{\mathrm{obs}}\) | mean squared residual on live slots |
| Phase-1 \(D_{\mathrm{dyn}}\) | penalty toward rest state from fit (not causal \(s_{t-1}\)) |
| \(\lambda_{\mathrm{state}}\) | one global value from `{0, 0.3, 1}` on calibration (accuracy, then home-margin p10) |
| Confirm | off |

## Phase-1 arms

C0 static mean L2; C1 gain; C2 offset; C3 gain+offset; C4 +polarity;
C5 +\(h\in\mathbb{R}^1\); C6 +\(h\in\mathbb{R}^2\).

Attribution at **K=12**, candidate-home: add-one group from C0, and
leave-one-group from C6.

## Stop (Phase 1)

If gain/offset/\(h\) raise reconstruction but **lower** same-home margin,
kill that state variable. If K=16 only helps \(R^2\), keep reporting it
and do not select it. Phase 2 (causal tracking, controlled drift, T3)
runs only for state variables that pass Phase 1.

## Out of scope this knife

Online identity update, unit-specific \(B_u\), neural nets, Mapping Row
interval QDA, 0.99/4× traffic as the classifier criterion, confirmation
sets, Overleaf abstract numbers.
