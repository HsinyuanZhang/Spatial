# RO-SS-QDA plan

Frozen static contrast. Spec:
[`RO-SS-QDA_执行说明.md`](RO-SS-QDA_执行说明.md).

This plan copies the locked fields. It is not a result.

## Question

On 7-electrode POSNEG in a relative-offset frame, does regularized full
covariance QDA separate co-home units better than mean+L1/L2 on the same
\(z\)? If yes, is the gain from off-diagonal correlation or only from
variance / volume?

Offline GT-event / GT-row isolation. Float. No quantisation, no online
update, no CAM/RTL, no 0.99/4× traffic gate as the classifier criterion.

## Locked protocol

| Field | Value |
|---|---|
| Recordings | `drift16c_600s_11`, `static16c_600s_11`, `rec_v1_units10_snr5_seed202601`, `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s |
| Split | chronological 50 / 25 / 25, equal timestamps unsplit |
| Neighborhood | K=7 relative-offset (primary-axis rank for linear probes; fixed physical pitch stencil for 2-D) |
| Observation | analog \(q^+,q^-\) on those slots; absent electrode ≠ quiet (mask recorded, classifier sees 0) |
| Scale | shared \(\ell_1\): \(m=\|x\|_1\), \(u=x/(m+\varepsilon)\); P1b reports \(\|x\|_\infty\) only |
| \(z\) | \([u,\;\log(m+\varepsilon),\;\mathrm{COM}]\); COM from POSNEG mass; 1-D keeps the primary COM axis |
| Standardize | fit mean/std only |
| P0 | existing 5-bit COM+P2P mean+L1 (mapping-row J0 object, KNN D9) |
| P1 | float mean+L1 and mean+L2 on \(z\); pick the stronger on calibration |
| P2 | per-unit shrunk QDA, \(\lambda\in\{0.25,0.5,0.75\}\) chosen once globally on calibration |
| P3 | same as P2 with off-diagonals zeroed |
| P2b | \(\lambda=1\) shared \(\Sigma_{\mathrm{pool}}\) |
| \(\eta\) | \(10^{-6}\) |
| Prior | equal; no firing-rate term |
| Small-sample | fit count \(< d+5\) or rank-deficient \(\to\) force \(\lambda=1\) |
| Confirm | off unless `--confirm` |

Home of a unit is the majority event home on **fit**. Same-home margin and
error taxonomy use that map.

## Stop rules

- P2 must beat frozen P1 by ≥2 pp assignment on both families, **or** cut
  same-home errors ≥20% on both, **and** raise \(m_e^{\mathrm{home}}\) p10
  on both, with no recording drop >1 pp and no worst-unit drop >2 pp.
- If P2 is not better than P3 on same-home metrics, do not claim
  cross-dimension correlation.
- Do not sweep percentiles, masks, \(P\), HDC, or waveforms.
- Do not confirm if the development gate fails.
