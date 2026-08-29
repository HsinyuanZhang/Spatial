# RO-SS-QDA results

Status: **development gate FAIL. Do not replace the template with
per-unit \((\mu,\Sigma)\). Do not write “cross-channel correlation”.**
Date: 2026-08-25.
Isolation: GT events, GT rows, zero-phase bandpass, chronological 50/25/25.
Float. Not online, not CAM/RTL, not a 0.99/4× traffic result.

Command:

```bash
python -m Spatial.experiments.run_ro_ss_qda --pilot --duration 60 --quiet
```

CSV / JSON: `Spatial/output/ro_ss_qda/`.
Provenance: [`ro_ss_qda_provenance.md`](ro_ss_qda_provenance.md).
Frozen \(\lambda=0.5\) (one global value from calibration accuracy, then
home-margin p10). Frozen P1 metric = **L1**. \(\eta=10^{-6}\). No unit was
forced onto pooled covariance (`n_fit` 40–102, \(d=17\)).

P0 candidate recall 0.9905 / 0.9928 / 0.9957 / 0.9935 matches mapping-row
J0 (HJ 0.9916 / MEArec 0.9942). Split and events did not drift.

## Joint gate

| Criterion | Result |
|---|---|
| HJ and MEA assignment ≥ P1 + 2 pp, **or** same-home errors −20% | **accuracy yes** (HJ **+4.16 pp**, MEA **+8.88 pp**). Same-home *count* does not drop on HJ (30 → 33). |
| Both families \(m_e^{\mathrm{home}}\) p10 up | **yes** where defined (HJ drift 0.20→0.27, static 0.98→1.57; MEA 20-unit 2.03→9.46). MEA 10-unit has **zero** co-home rivals. |
| No recording −1 pp vs P1; worst-unit −2 pp | **yes**. Worst drop is HJ drift **−0.36 pp**. Every worst-unit recall is ≥ P1. |
| P2 better than P3 on same-home, both families | **no**. HJ: P3 accuracy 0.9136 > P2 0.9045, home p10 1.55 > 0.92. |
| Co-home subset itself improves (not only far COM rejects) | **no on HJ**. Same-home error counts 30 → 33. MEA 20-unit 5 → 1. |

Pattern (spec §5): **P2 > P1, but P2 ≉ better than P3/P2b on HJ, and the HJ
gain is not co-home.** Write **Gaussian / volume / shared covariance**, not
cross-dimension correlation. Do not change the stored object to
\((\mu,\widetilde{\Sigma},\log\|\Sigma\|)\). Do not confirm. Do not sweep
\(\lambda\).

## Family assignment accuracy (event-weighted)

| Arm | HJ | MEArec | HJ home p10 (mean of files) | MEA home p10 | HJ same-home errors | MEA same-home errors |
|---|---:|---:|---:|---:|---:|---:|
| P0 5-bit COM+P2P L1 | 0.8065 | 0.8335 | 1.0 / 2.0 | 21.0 | 46 / 298 | 5 / 576 |
| **P1** mean L1 on \(z\) | **0.8630** | **0.8309** | 0.59 | 2.03 | 30 / 211 | 5 / 585 |
| P1b inf-norm scale | 0.7773 | 0.8485 | 0.41 | 2.88 | 33 / 343 | 2 / 524 |
| **P2** shrunk QDA \(\lambda=0.5\) | **0.9045** | **0.9196** | 0.92 | 9.46 | 33 / 147 | 1 / 278 |
| P3 diagonal QDA | 0.9136 | 0.8742 | 1.55 | 7.19 | 33 / 133 | 5 / 435 |
| P2b \(\lambda=1\) shared \(\Sigma\) | 0.9195 | 0.9110 | 2.80 | 5.95 | 26 / 124 | 1 / 308 |

P1b is worse than P1 on HJ, so shared \(\ell_1\) stays the scale.

## Per recording (test)

| Arm | HJ drift | HJ static | MEA 10u | MEA 20u |
|---|---:|---:|---:|---:|
| P0 acc (worst unit) | 0.695 (0.340) | 0.941 (0.861) | 0.902 (0.831) | 0.799 (0.500) |
| P1 acc (worst) | 0.858 (0.587) | 0.869 (0.611) | 0.875 (0.510) | 0.809 (0.058) |
| P2 acc (worst) | 0.854 (0.587) | 0.966 (0.915) | 0.951 (0.881) | 0.904 (0.733) |
| P3 acc (worst) | 0.878 (0.587) | 0.957 (0.901) | 0.921 (0.830) | 0.851 (0.326) |
| P2b acc (worst) | 0.890 (0.587) | 0.956 (0.915) | 0.952 (0.881) | 0.890 (0.670) |

The MEA 20-unit weakest unit is the headline local win: mean L1 at 0.058,
QDA at 0.733. That is volume / covariance, and P3 only reaches 0.326, so
*some* off-diagonal structure helps that recording — not both families.

## D1 (fit co-home pairs; does not authorize P2)

Eight pairs (MEA 10-unit has none). Mean \(u\) cosine is **0.95–0.99** on
every pair: amplitude shape is almost the same neuron. Polarity L2 is
0.12–0.47. COM\(q^+\)−COM\(q^-\) gap is 0.12–4.2 µm. Fit-set LDA oracle is
0.86–0.997, so \(z\) is linearly separable *on the data used to fit it*.
The hard pair is HJ drift units 4/43 at home 7 (cosine 0.989, \(\Delta\)COM
0.61 µm, LDA 0.86). High oracle + failed co-home error drop on HJ means
the classifier, not the observation, is the remaining gap for that pair —
and a new shared LDA is **out of scope** for this knife (spec §5 row 3).

## What this does not show

- Confirmation recordings were not touched.
- 0.99 candidate lists and 4× waveform traffic are not a pass, even though
  P0 still reports the J0 recall numbers as a drift check.
- PPA, online \((N,\sum z,\sum zz^\top)\), or a CAM row format.

## Next

Stop this mapping of “per-unit full \(\Sigma\) as the identity object”.
The defensible leftover is **shared (or diagonal) covariance on RO-POSNEG
\(z\)** as a float classifier, not a new memory row. Spec forbids opening
restricted shared LDA, HDC, or a time branch until POSNEG is shown to be
an insufficient observation; D1 LDA oracles are high, so a time branch is
not licensed here.
