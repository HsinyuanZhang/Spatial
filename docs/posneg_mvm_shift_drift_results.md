# POSNEG MVM matching, polarity ablation, and shift leakage — results

Frozen protocol: [`posneg_mvm_shift_drift_plan.md`](posneg_mvm_shift_drift_plan.md).
Tracking / coherence / controlled drift: [`drift_tracking_results.md`](drift_tracking_results.md).
Provenance: [`posneg_mvm_shift_drift_provenance.md`](posneg_mvm_shift_drift_provenance.md).

GT events and GT units are used throughout (component isolation). Every number
below is per recording. MEArec 10u same-home fields are undefined (no co-home
rivals) and are written `nan`, never averaged with MEArec 20u.

Split: chronological 50/25/25 on 60 s. Decision numbers are **test**.
Candidate-home analog POSNEG, geometry-\(K\), not strongest-\(K\).

## Verdict

| Question | Answer |
|---|---|
| Q1 | **Yes, among co-home units.** \(\mathrm{argmin}\) L2 \(=\) \(\mathrm{argmax}\) biased-dot on a shared \(x\) (unit test + identical same-home \(M_{\mathrm{home}}\) P10/P50/\(P>0\)). Overall assignment agrees on 91–98% of events because candidate-home \(x_u\) depends on \(h_u\). Biased-dot is \(S=Mx+b\) with \(b_u=-\|\mu_u\|^2\). |
| Q2 | Pure-dot **deletes the \(\|\mu\|^2\) amplitude penalty** and collapses (HJ drift acc 0.28, same-home 0.00). Cosine deletes \(\|x\|\) as well: it **raises mean same-home** on HJ drift/static and MEArec 20u, but **drops MEArec 20u worst-unit recall** 0.809 → 0.677. Cosine is not MVM-simple. It is **not selected**. |
| Q3 | POSNEG beats P2P on HJ same-home (~+3 pp). The extra identity is mostly the **negative trough**. POS-only is weak. \(r_{pn}\) and \(\Delta c\) do not beat raw concat. |
| Q4 | Free shift is **impostor flexibility**, not robustness. \(L_{\mathrm{flex}}>0\) on every recording that has co-home rivals. Same-home margin P10 collapses on HJ drift. Branch **stopped**. \(\{\pm 2\}\) not opened. |

Success metric is sorting, not reconstruction.

## Part A — scoring (raw POSNEG concat, candidate-home)

Test, \(K=12\). \(n_{\mathrm{test}}\): HJ drift 843, HJ static 697, MEArec 10u 1154, MEArec 20u 2303.

| rec | arm | acc | same-home | \(M\) P10 | \(P(M>0)\) | worst-unit | \(\Delta\)acc vs L2 | MVM? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HJ drift | A0 L1 | 0.9158 | 0.8828 | 0.65 | 0.914 | 0.708 | −2.02 pp | no |
| HJ drift | A1 L2 | 0.9359 | 0.9141 | 48.45 | 0.935 | 0.792 | 0 | equiv. |
| HJ drift | A2 biased-dot | 0.9383 | 0.9245 | 48.45 | 0.935 | 0.808 | +0.24 pp | **yes** |
| HJ drift | A3 pure-dot | 0.2847 | 0.0000 | −2656 | 0.372 | 0.000 | −65.1 pp | yes, no \(\|\mu\|^2\) |
| HJ drift | A4 cosine | 0.9644 | 0.9688 | 0.013 | 0.979 | 0.913 | +2.85 pp | no (2 norms) |
| HJ drift | A5 \(\lambda=0.1\) | 0.9668 | 0.9714 | 0.014 | 0.982 | 0.913 | +3.08 pp | no |
| HJ drift | A1 event-argmax | 0.8209 | 0.8854 | 54.48 | 0.938 | 0.261 | −11.5 pp | negative ablation |
| HJ static | A0 L1 | 0.9469 | 0.8918 | 3.84 | 0.935 | 0.812 | −1.15 pp | no |
| HJ static | A1 L2 | 0.9584 | 0.9221 | 42.60 | 0.957 | 0.869 | 0 | equiv. |
| HJ static | A2 biased-dot | 0.9627 | 0.9351 | 42.60 | 0.957 | 0.893 | +0.43 pp | **yes** |
| HJ static | A3 pure-dot | 0.3429 | 0.0130 | −2483 | 0.628 | 0.000 | −61.6 pp | yes, no \(\|\mu\|^2\) |
| HJ static | A4 cosine | 0.9742 | 0.9740 | 0.010 | 0.996 | 0.917 | +1.58 pp | no |
| HJ static | A5 \(\lambda=0.1\) | 0.9727 | 0.9697 | 0.011 | 0.991 | 0.917 | +1.43 pp | no |
| HJ static | A1 event-argmax | 0.8737 | 0.9221 | 41.43 | 0.948 | 0.572 | −8.46 pp | negative ablation |
| MEArec 10u | A1 L2 | 0.9514 | nan | nan | nan | 0.898 | 0 | |
| MEArec 10u | A2 biased-dot | 0.9636 | nan | nan | nan | 0.896 | +1.21 pp | |
| MEArec 10u | A3 pure-dot | 0.6904 | nan | nan | nan | 0.000 | −26.1 pp | |
| MEArec 10u | A4 cosine | 0.9636 | nan | nan | nan | 0.857 | +1.21 pp | |
| MEArec 20u | A0 L1 | 0.9020 | 0.9306 | 110 | 0.980 | 0.807 | −0.74 pp | |
| MEArec 20u | A1 L2 | 0.9094 | 0.9396 | 3101 | 0.991 | 0.809 | 0 | |
| MEArec 20u | A2 biased-dot | 0.9224 | 0.9687 | 3101 | 0.991 | 0.721 | +1.30 pp | |
| MEArec 20u | A3 pure-dot | 0.2632 | 0.7047 | −1.2e4 | 0.709 | 0.000 | −64.6 pp | |
| MEArec 20u | A4 cosine | 0.9215 | 0.9597 | 0.030 | 0.980 | **0.677** | +1.21 pp | |
| MEArec 20u | A5 \(\lambda=0.1\) | 0.9258 | 0.9597 | 0.033 | 0.982 | 0.721 | +1.65 pp | |
| MEArec 20u | A1 event-argmax | 0.8760 | 0.9463 | 3052 | 0.987 | 0.714 | −3.34 pp | |

L2 / biased-dot overall assignment agreement (test, \(K=12\)): HJ drift 0.977, HJ static 0.980, MEArec 10u 0.944, MEArec 20u 0.911. Same-home \(M\) P10/P50/\(P>0\) match to float noise; same-home *accuracy* can differ because a far-home unit may win the all-unit argmin when \(\|x_u\|^2\) is dropped.

A5 ran: calibration cosine same-home exceeded L2 on HJ drift and MEArec 20u drop ≤ 2 pp. One global \(\lambda_{\mathrm{amp}}=0.1\). A5 ≈ cosine. Stop-2 (cosine collapses dense/static same-home) **did not fire**. Cosine is still not deployable: it needs two norms, and it is the only A-arm that **cuts MEArec 20u worst-unit recall** by 13 pp.

Operation counts per candidate, \(D=2K=24\) at \(K=12\): L1 24 abs; L2 24 MAC; biased-dot 24 MAC + 1 add of \(b_u\); pure-dot 24 MAC; cosine 24 MAC + 2 norms.

## Part B — representation (L2, candidate-home, \(K=12\))

| rec | arm | acc | same-home | \(M\) P10 | worst | \(\Delta\)home vs B1 |
|---|---|---:|---:|---:|---:|---:|
| HJ drift | B0 P2P | 0.9229 | 0.8854 | 4.35 | 0.708 | −2.86 pp |
| HJ drift | B1 POSNEG | 0.9359 | 0.9141 | 48.45 | 0.792 | 0 |
| HJ drift | B2 POS-only | 0.8422 | 0.7943 | −18.6 | 0.500 | −12.0 pp |
| HJ drift | B3 NEG-only | 0.9348 | **0.9323** | 57.23 | 0.825 | **+1.82 pp** |
| HJ drift | B4 1:1 | 0.9359 | 0.9141 | 48.45 | 0.792 | 0 |
| HJ drift | B4 1:2 (neg×2) | 0.9431 | 0.9349 | 122 | 0.833 | +2.08 pp |
| HJ drift | B4 2:1 | 0.9241 | 0.8906 | 13.0 | 0.708 | −2.34 pp |
| HJ drift | B5 +\(r_{pn}\) | 0.9359 | 0.9141 | 48.46 | 0.792 | 0 |
| HJ drift | B6 +\(\Delta c\) | 0.9324 | 0.9141 | 61.2 | 0.783 | 0 |
| HJ drift | B7 both | 0.9324 | 0.9141 | 61.3 | 0.783 | 0 |
| HJ static | B0 P2P | 0.9469 | 0.8874 | 28.7 | 0.812 | −3.46 pp |
| HJ static | B1 POSNEG | 0.9584 | 0.9221 | 42.6 | 0.869 | 0 |
| HJ static | B2 POS-only | 0.8522 | 0.7186 | −4.3 | 0.541 | −20.4 pp |
| HJ static | B3 NEG-only | 0.9541 | 0.9134 | 40.3 | 0.844 | −0.87 pp |
| HJ static | B5/B6/B7 | ≤ B1 | ≤ B1 | — | — | ≤ 0 |
| MEArec 20u | B0 P2P | 0.9098 | 0.9329 | 3573 | 0.794 | −0.67 pp |
| MEArec 20u | B1 POSNEG | 0.9094 | 0.9396 | 3101 | 0.809 | 0 |
| MEArec 20u | B2 POS-only | 0.6201 | 0.6577 | 369 | 0.385 | −28.2 pp |
| MEArec 20u | B3 NEG-only | 0.8985 | 0.9060 | 2387 | 0.794 | −3.36 pp |
| MEArec 20u | B5/B6/B7 | ≈ B1 | ≈ B1 | — | — | ≈ 0 |

POSNEG vs P2P: not merging polarity is worth ~3 pp HJ same-home. The trough carries most of that. Derived \(r_{pn}\) / \(\Delta c\) are already in the concat; **do not store them**. B4 1:2 is a test-set observation, not a new default (the three ratios were a frozen report grid).

Closest same-home template pair (HJ units 4 and 43 at home 7): P2P L2 134, POSNEG 73, POS 34, NEG 39. NEG-only is what keeps that pair’s test same-home from collapsing.

## \(K\) sweep, A1 L2 POSNEG

| rec | \(K=7\) acc / home / P10 / worst | \(K=12\) | \(K=16\) |
|---|---|---|---|
| HJ drift | 0.913 / 0.919 / 51.8 / 0.638 | 0.936 / 0.914 / 48.5 / 0.792 | 0.945 / 0.927 / 48.4 / 0.817 |
| HJ static | 0.951 / 0.931 / 51.6 / 0.877 | 0.958 / 0.922 / 42.6 / 0.869 | 0.964 / 0.935 / 45.5 / 0.893 |
| MEArec 10u | 0.949 / nan / nan / 0.857 | 0.951 / nan / nan / 0.898 | 0.952 / nan / nan / 0.898 |
| MEArec 20u | 0.913 / 0.937 / 2642 / **0.820** | 0.909 / 0.940 / 3101 / 0.809 | 0.909 / 0.937 / 3539 / **0.779** |

Larger \(K\) lifts HJ overall accuracy; on MEArec 20u it does **not** lift overall accuracy and it **lowers worst-unit recall**. Stop-7 caution: do not select \(K=16\) as the identity width.

## Part C — free shift (DIAGNOSTIC only, \(K=12\))

\(L_{\mathrm{flex}}\) is the mean of \(I_{\mathrm{wrong}}-I_{\mathrm{true}}\) on events that have a co-home rival. Positive = the nearest impostor gained more from its private shift than the true unit did.

| rec | arm | acc | same-home | \(M\) P10 | \(I_{\mathrm{true}}\) | \(I_{\mathrm{wrong}}\) | \(L_{\mathrm{flex}}\) | frac nonzero \(s\) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HJ drift | C0 fixed L2 | 0.9359 | 0.9141 | 48.45 | 0 | 0 | 0 | — |
| HJ drift | C1 \(\{\pm1\}\) biased-dot | 0.8671 | 0.8698 | 1.67 | 48.3 | 46.8 | **+21.7** | 0.148 |
| HJ drift | C2 frac biased-dot | 0.9253 | 0.8984 | 12.54 | 99.6 | 88.2 | **+43.9** | 0.558 |
| HJ drift | C1 cosine | 0.8671 | 0.9375 | 0.01 | 0.0027 | 0.0034 | +0.0017 | 0.059 |
| HJ static | C1 biased-dot | 0.9555 | 0.9221 | 42.60 | 29.1 | **310.6** | **+274** | 0.017 |
| HJ static | C2 biased-dot | 0.9627 | 0.9437 | 45.43 | 37.7 | 268 | **+232** | 0.254 |
| MEArec 20u | C1 biased-dot | 0.9167 | 0.9642 | 3070 | 54.0 | 42.8 | **+33.9** | 0.029 |
| MEArec 20u | C2 biased-dot | 0.9128 | 0.9642 | 1233 | 62.5 | 1568 | **+1513** | 0.211 |

Static recordings pick a large false improvement for the impostor even when the chosen-shift rate is small. Reconstruction/best-match distance would have called this a win. Same-home margin P10 on HJ drift falls 48 → 1.7. **Stop-1 and stop-3 fire. \(\{\pm2\}\) skipped.** Free per-candidate shift is not a tracker and is not a robustness mechanism.

## Stops (this knife)

1. Free-shift alignment improves while same-home margin falls — **fired**, Part C.
2. Cosine same-home collapse on dense/static — **not fired** (cosine raises mean same-home). Worst-unit drop on MEArec 20u is reported; cosine still not selected.
3. \(L_{\mathrm{flex}}\ge 0\) — **fired** on every co-home recording.
7. \(K=16\) only helps HJ overall / hurts MEArec 20u worst-unit — **do not select \(K=16\)**.

Confirmation corpus stays closed.

## Deployable object after A–C

POSNEG event in candidate-home physical slots → L2 / biased-dot \(S=Mx+b\) against absolute \(\mu_u\).
Do not per-event cosine-normalize for deployment. Do not give each candidate a private shift.
Troughs are the identity-bearing half of POSNEG; peaks are not a substitute.

Commands:

```bash
python -m Spatial.experiments.run_posneg_mvm_ablation --pilot --duration 60
python -m Spatial.experiments.run_spatial_shift_ablation --pilot --duration 60
```
