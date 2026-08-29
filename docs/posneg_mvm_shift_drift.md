# POSNEG MVM matching, polarity, shift leakage, and causal tracking

One-document summary of this independent knife: **what was deployed**, **how
to rerun it**, and **what the frozen 60 s pilots showed**. It does not revive
Mapping Row, interval QDA, or free per-event deformable templates.

Success is **better sorting** (overall accuracy, same-home accuracy, same-home
margin), not reconstruction or best-alignment distance.

Companion files (protocol / split result tables / hashes):

- [`posneg_mvm_shift_drift_plan.md`](posneg_mvm_shift_drift_plan.md)
- [`posneg_mvm_shift_drift_results.md`](posneg_mvm_shift_drift_results.md)
- [`drift_tracking_results.md`](drift_tracking_results.md)
- [`posneg_mvm_shift_drift_provenance.md`](posneg_mvm_shift_drift_provenance.md)

---

## 1. What this knife asked

Eight questions, answered from the same locked development set. No test-set
retuning.

| ID | Question |
|---|---|
| Q1 | Can biased-dot \(S=2x^\top\mu-\|\mu\|^2\) replace squared L2 without changing assignment? Is that the MVM form \(S=Mx+b\), \(b_u=-\|\mu_u\|^2\)? |
| Q2 | What amplitude information do pure-dot and cosine delete, and is that deletion helpful on HJ drift vs harmful on HJ static / MEArec 20u? |
| Q3 | Where does POSNEG beat P2P: positive peak, negative trough, polarity mass ratio \(r_{pn}\), spatial displacement \(\Delta c\), or just higher dimension? |
| Q4 | Is moving-template / shift-min robustness, or impostor flexibility? (\(L_{\mathrm{flex}}\), same-home margin) |
| Q5 | Is inferred drift global, local/tile, a smooth field, or unit-specific noise? Static recordings must stay quiet. |
| Q6 | Which **causal** tracker actually improves classification vs Frozen? |
| Q7 | Can state move while identity \(\mu_u\) stays put? |
| Q8 | Do static / no-drift trackers stay quiet? |

---

## 2. Deployment

### 2.1 New code (do not edit killed runners)

| Role | Path |
|---|---|
| Scoring + POSNEG/P2P features | `Spatial/algorithms/posneg_mvm_matching.py` |
| Free-shift diagnostic / \(L_{\mathrm{flex}}\) | `Spatial/algorithms/spatial_shift_matching.py` |
| Causal trackers, coherence, controlled drift | `Spatial/algorithms/drift_tracking.py` |
| Shared event prep / CSV I/O | `Spatial/experiments/posneg_mvm_common.py` |
| Part A/B runner | `Spatial/experiments/run_posneg_mvm_ablation.py` |
| Part C runner | `Spatial/experiments/run_spatial_shift_ablation.py` |
| GT block \(C(d)\) | `Spatial/experiments/run_drift_coherence.py` |
| Part D runner | `Spatial/experiments/run_drift_tracking_compare.py` |
| Controlled synthetic drift | `Spatial/experiments/run_controlled_drift.py` |
| Contract tests | `Spatial/tests/test_posneg_mvm_shift_tracking.py` |

Geometry-\(K\) gather is **imported** from `deformable_spatial_template.py`
(`geometry_k_table`, `gather_posneg_batch`, `extract_all_channel_extrema`).
That module is not modified. Mapping Row and deformable Phase-2 runners stay
gated / untouched.

### 2.2 Locked protocol

| Field | Value |
|---|---|
| Recordings | HJ `drift16c_600s_11`, HJ `static16c_600s_11`, MEArec `rec_v1_units10_snr5_seed202601`, MEArec `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s |
| Split | chronological 50% fit / 25% calibration / 25% test; equal timestamps unsplit |
| Isolation | **GT events, GT units** (component isolation; labeled in every table) |
| Filter | zero-phase Butterworth 300–6000 Hz, order 3 |
| Observation | analog POSNEG extrema in \(\pm 15\) samples; P2P is \(p_i=q_i^++q_i^-\) from the same extrema |
| \(K\) | \(\{7,12,16\}\) geometrically nearest electrodes around a **fit home**, ordered by physical offset, never strongest-\(K\) |
| Default frame | **candidate-home**: unit \(u\) lives in physical-offset slots of its majority home \(h_u\) |
| Ablation frame | event-argmax recentering, A1 L2 POSNEG at \(K=12\) only |
| Confirmation | closed |
| Seed | 0 |

MEArec 10-unit has no co-home rivals. Same-home fields are `nan`, never 0, and
are never averaged with MEArec 20u.

Event counts after isolation: HJ drift 3371, HJ static 2788, MEArec 10u 4611,
MEArec 20u 9224. Test \(n\): 843 / 697 / 1154 / 2303.

### 2.3 Identity object

For each unit \(u\), fit \(\mu_u\) is the mean analog POSNEG on \(h_u\)'s
geometry-\(K\) slots:

\[
\mu_u = \bigl[\mu^+_{u,1:K};\;\mu^-_{u,1:K}\bigr].
\]

Absolute amplitude stays in \(\mu_u\). There is no per-event global
normalization unless the arm is explicitly cosine / A5.

When scoring event \(e\) against unit \(u\), the event is gathered in **that
unit's** candidate-home slots, so \(x_u\) depends on \(h_u\). Units that share
a home share \(x\).

### 2.4 Scoring (Part A)

| Arm | Formula | Higher better? | MVM? |
|---|---|---|---|
| A0 L1 | \(\sum_i\lvert x_i-\mu_i\rvert\) | no | no |
| A1 squared L2 | \(\|x-\mu\|^2\) | no | equivalent to A2 when \(x\) is shared |
| A2 biased-dot | \(S=2x^\top\mu-\|\mu\|^2\) | yes | **yes**, \(S=Mx+b\), \(M=2\mu\), \(b_u=-\|\mu_u\|^2\) |
| A3 pure-dot | \(S=x^\top\mu\) | yes | yes, no \(\|\mu\|^2\) penalty |
| A4 cosine | \((x^\top\mu)/(\|x\|\|\mu\|)\) | yes | no (two norms) |
| A5 | \(\cos(x,\mu)-\lambda_{\mathrm{amp}}(\log\|x\|-\log\|\mu\|)^2\) | yes | gated; \(\lambda\) from cal, global |

Identity used in unit tests: \(\|x-\mu\|^2 = \|x\|^2 + \|\mu\|^2 - 2x^\top\mu\),
so \(\mathrm{argmin}\) L2 \(=\mathrm{argmax}\) biased-dot when \(x\) is shared,
and \(S=-\|x-\mu\|^2+\|x\|^2\).

At \(K=12\), \(D=24\): L1 24 abs; L2 / biased-dot / pure-dot 24 MAC; cosine 24
MAC + 2 norms.

### 2.5 Representations (Part B; scoring held at L2)

| Arm | Feature |
|---|---|
| B0 | P2P \(p=q^++q^-\) |
| B1 | raw POSNEG \([q^+;q^-]\) |
| B2 | POS-only |
| B3 | NEG-only |
| B4 | weighted L2 with global \(\lambda_+\!:\!\lambda_-\in\{1:1,1:2,2:1\}\) |
| B5 | B1 \(+r_{pn}\) |
| B6 | B1 \(+\Delta c=c_+-c_-\) |
| B7 | B1 \(+r_{pn}+\Delta c\) |

### 2.6 Free shift (Part C) — diagnostic only

Not a tracker. Each candidate may pick its own shift on the **current** event.

- C0: fixed template
- C1: \(s\in\{-1,0,+1\}\) pitch (extra \(\{\pm2\}\) only if cal \(L_{\mathrm{flex}}<0\) and margin P10 improves)
- C2: \(\delta\in\{-0.5,-0.25,0,+0.25,+0.5\}\) via physical interpolation

Score paths: biased-dot moving correlation **and** cosine moving correlation.
The biased-dot path does **not** divide by \(\|x\|\|\mu\|\).

\[
I_{\mathrm{true}} = \mathrm{Score}_{\mathrm{shifted,true}}-\mathrm{Score}_{\mathrm{fixed,true}},
\quad
L_{\mathrm{flex}} = I_{\mathrm{wrong}}-I_{\mathrm{true}}
\]

Nearest wrong same-home is taken under the **fixed** score, then that unit's
own free-shift improvement is \(I_{\mathrm{wrong}}\). \(L_{\mathrm{flex}}>0\)
means the impostor gained more.

### 2.7 Causal tracking (Part D)

Order: **predict state → freeze → classify → update only if confident**.

Confidence: \(M_{\mathrm{all}}>\max(0,\) calibration Frozen \(M_{\mathrm{all}}\) P10\()\).
Stream is the **test** split only; \(\mu\) comes from fit.

| Arm | Behaviour | Role |
|---|---|---|
| D0 Frozen | \(\mu_u\) fixed, \(\delta=0\) | deployable baseline |
| D1 EMA | \(\mu\leftarrow(1-0.05)\mu+0.05 x\) after high-confidence assignment | online-template control |
| D2 free shift | Part C bank, per-candidate | **ORACLE / DIAGNOSTIC** |
| D3 global \(\delta(t)\) | every unit uses the same predicted shift | shared state |
| D4 tile \(\delta_g(t)\) | 3-pitch primary-axis tiles | local shared state |
| D5 linear field | \(c_0+c_1 r\) | gated on local coherence |
| D6 shared shift + slow \(\mu\) | \(\eta_\mu=0.005\ll\eta_{\mathrm{state}}=0.2\) | gated; never to rescue a failed tracker |

### 2.8 How to rerun

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_posneg_mvm_shift_tracking.py -q
python -m Spatial.experiments.run_posneg_mvm_ablation --pilot --duration 60
python -m Spatial.experiments.run_spatial_shift_ablation --pilot --duration 60
python -m Spatial.experiments.run_drift_coherence --pilot --duration 60
python -m Spatial.experiments.run_drift_tracking_compare --pilot --duration 60
python -m Spatial.experiments.run_controlled_drift --pilot --duration 60
```

Outputs live under `Spatial/output/{posneg_mvm_ablation,spatial_shift_ablation,drift_coherence,drift_tracking_compare,controlled_drift}/`.

---

## 3. Headline answers

| ID | Answer |
|---|---|
| Q1 | **Yes among co-home units.** Same-home \(M\) P10 / \(P(M>0)\) of L2 and biased-dot match. Overall assignment agrees 91–98% because \(x_u\) depends on home. |
| Q2 | Pure-dot deletes the \(\|\mu\|^2\) penalty and **collapses**. Cosine also deletes \(\|x\|\): it **raises mean same-home** on HJ drift, HJ static, and MEArec 20u, but **cuts MEArec 20u worst-unit recall** 0.809 → 0.677. Not MVM-simple. **Not selected.** |
| Q3 | POSNEG beats P2P on HJ same-home by ~3 pp. The extra identity is the **negative trough**. POS-only is weak. \(r_{pn}\) and \(\Delta c\) do not beat raw concat. |
| Q4 | Free shift is **impostor flexibility**. \(L_{\mathrm{flex}}>0\) on every recording with co-home rivals. HJ drift margin P10 48 → 1.7. Branch **stopped**. |
| Q5 | Inferred 1 s \(\delta\) is **weak / unit-specific**, not a strong global or tile field. HJ drift nearest \(C(d)=0.15<0.3\). Static \(\lvert\hat\delta\rvert\approx0.05\) pitch — the estimator is not hallucinating large motion. |
| Q6 | **Frozen stays the deployable tracker on this 15 s test window.** D3/D4 tie Frozen same-home on HJ drift and stay quiet on static. EMA helps HJ drift (+1.4 pp) but **collapses MEArec 20u (−31 pp)** while rewriting \(\mu\). Oracle free-shift hurts. D5 skipped. D6 does not beat Frozen same-home. |
| Q7 | **Yes, if state is shared \(\delta\) and \(\mu\) is frozen** (D3/D4: \(\|\Delta\mu\|=0\)). EMA and D6 move identity. |
| Q8 | **Mostly yes for shared shift** (\(\lvert\hat\delta_{\mathrm{end}}\rvert=0.015\)–0.03 pitch on static). EMA is not quiet: it always writes \(\mu\). |

---

## 4. Part A — scoring, raw POSNEG, candidate-home, \(K=12\) test

| rec | arm | acc | same-home | \(M\) P10 | \(P(M>0)\) | worst-unit | \(\Delta\)acc vs L2 | form |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HJ drift | A0 L1 | 0.9158 | 0.8828 | 0.65 | 0.914 | 0.708 | −2.02 pp | 24 abs |
| HJ drift | A1 L2 | 0.9359 | 0.9141 | 48.45 | 0.935 | 0.792 | 0 | 24 MAC |
| HJ drift | A2 biased-dot | 0.9383 | 0.9245 | 48.45 | 0.935 | 0.808 | +0.24 pp | **MVM** |
| HJ drift | A3 pure-dot | 0.2847 | 0.0000 | −2656 | 0.372 | 0.000 | −65.1 pp | MVM, no \(\|\mu\|^2\) |
| HJ drift | A4 cosine | 0.9644 | 0.9688 | 0.013 | 0.979 | 0.913 | +2.85 pp | +2 norms |
| HJ drift | A5 \(\lambda=0.1\) | 0.9668 | 0.9714 | 0.014 | 0.982 | 0.913 | +3.08 pp | gated |
| HJ drift | A1 event-argmax | 0.8209 | 0.8854 | 54.48 | 0.938 | 0.261 | −11.5 pp | **negative ablation** |
| HJ static | A0 L1 | 0.9469 | 0.8918 | 3.84 | 0.935 | 0.812 | −1.15 pp | |
| HJ static | A1 L2 | 0.9584 | 0.9221 | 42.60 | 0.957 | 0.869 | 0 | |
| HJ static | A2 biased-dot | 0.9627 | 0.9351 | 42.60 | 0.957 | 0.893 | +0.43 pp | |
| HJ static | A3 pure-dot | 0.3429 | 0.0130 | −2483 | 0.628 | 0.000 | −61.6 pp | |
| HJ static | A4 cosine | 0.9742 | 0.9740 | 0.010 | 0.996 | 0.917 | +1.58 pp | |
| HJ static | A5 \(\lambda=0.1\) | 0.9727 | 0.9697 | 0.011 | 0.991 | 0.917 | +1.43 pp | |
| HJ static | A1 event-argmax | 0.8737 | 0.9221 | 41.43 | 0.948 | 0.572 | −8.46 pp | |
| MEArec 10u | A1 L2 | 0.9514 | nan | nan | nan | 0.898 | 0 | no co-home |
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

L2 vs biased-dot **overall** agreement: HJ drift 0.977, HJ static 0.980, MEArec
10u 0.944, MEArec 20u 0.911. Same-home *ranking* is identical; same-home
*accuracy* can still move because a far-home unit may win the all-unit argmin
once \(\|x_u\|^2\) is dropped.

A5 ran because calibration cosine same-home beat L2 on HJ drift and MEArec 20u
drop was ≤ 2 pp. One global \(\lambda_{\mathrm{amp}}=0.1\). A5 ≈ cosine.
Stop-2 (cosine collapses dense/static same-home) **did not fire**. Cosine is
still not deployable: two norms, and it is the only A-arm that **cuts MEArec
20u worst-unit recall by 13 pp**.

Event-argmax recentering (known negative ablation) costs 8–12 pp overall on HJ
and worst-unit 0.792 → 0.261 on HJ drift. Candidate-home stays the default frame.

### Reading Q1–Q2 from this table

- **Keep L2 / biased-dot as the identity match.** They are the MVM pair.
- **Do not drop \(\|\mu\|^2\)** (pure-dot). Absolute template energy is identity.
- **Do not deploy cosine** even though mean same-home goes up. The dense-probe
  tail (worst unit) pays for it, and hardware would have to compute norms.
- L1 is strictly worse than L2 on every recording.

---

## 5. Part B — representation, L2, candidate-home, \(K=12\) test

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
| HJ static | B5 / B6 / B7 | ≤ B1 | ≤ B1 | — | — | ≤ 0 |
| MEArec 20u | B0 P2P | 0.9098 | 0.9329 | 3573 | 0.794 | −0.67 pp |
| MEArec 20u | B1 POSNEG | 0.9094 | 0.9396 | 3101 | 0.809 | 0 |
| MEArec 20u | B2 POS-only | 0.6201 | 0.6577 | 369 | 0.385 | −28.2 pp |
| MEArec 20u | B3 NEG-only | 0.8985 | 0.9060 | 2387 | 0.794 | −3.36 pp |
| MEArec 20u | B5 / B6 / B7 | ≈ B1 | ≈ B1 | — | — | ≈ 0 |

Not merging polarity is worth ~3 pp HJ same-home and a large HJ-drift margin
gap (P2P P10 4.4 vs POSNEG 48). The trough is the identity-bearing half.
Peaks are not a substitute. \(r_{pn}\) and \(\Delta c\) are already in the
concat; **do not extra-store them**.

B4 1:2 (weight the trough more) helps HJ. That is a frozen-grid observation,
not a new default chosen from the test set.

Closest same-home template pair (HJ units 4 and 43, home 7): P2P L2 134,
POSNEG 73, POS 34, NEG 39. NEG-only is what keeps that pair's test same-home
from collapsing.

---

## 6. \(K\) sweep, A1 L2 POSNEG

| rec | \(K=7\) acc / home / P10 / worst | \(K=12\) | \(K=16\) |
|---|---|---|---|
| HJ drift | 0.913 / 0.919 / 51.8 / 0.638 | 0.936 / 0.914 / 48.5 / 0.792 | 0.945 / 0.927 / 48.4 / 0.817 |
| HJ static | 0.951 / 0.931 / 51.6 / 0.877 | 0.958 / 0.922 / 42.6 / 0.869 | 0.964 / 0.935 / 45.5 / 0.893 |
| MEArec 10u | 0.949 / nan / nan / 0.857 | 0.951 / nan / nan / 0.898 | 0.952 / nan / nan / 0.898 |
| MEArec 20u | 0.913 / 0.937 / 2642 / **0.820** | 0.909 / 0.940 / 3101 / 0.809 | 0.909 / 0.937 / 3539 / **0.779** |

Larger \(K\) lifts HJ overall accuracy. On MEArec 20u it does **not** lift
overall accuracy and it **lowers worst-unit recall**. Do not select \(K=16\)
as the identity width. Reporting default remains \(K=12\); \(K=7\) is the
more local core.

Mixed \(K_{\mathrm{id}}=7\), \(K_{\mathrm{st}}=12\) for tracking **hurt** HJ
drift (see §8). Identity matching and state estimation should not be split
that way on these probes.

---

## 7. Part C — free shift, DIAGNOSTIC, \(K=12\) test

\(L_{\mathrm{flex}}\) is the mean of \(I_{\mathrm{wrong}}-I_{\mathrm{true}}\)
**only on events that have a co-home rival**. Positive means the nearest
impostor gained more from its private shift than the true unit did.

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

Static recordings give the impostor a huge score gift even when the chosen
nonzero-shift rate is small (HJ static C1: 1.7% nonzero, \(L_{\mathrm{flex}}=+274\)).
A reconstruction / best-match metric would have called this a win. Same-home
margin P10 on HJ drift falls 48 → 1.7.

**Stop-1 and stop-3 fire.** Extra \(\{\pm2\}\) skipped. Free per-candidate
shift is not robustness and is not a tracker.

This is the same failure mode as free per-event gain/offset/\(h\) in the
deformable Phase-1 knife: extra per-query freedom fits the true unit *and*
the co-home impostor, usually the impostor more.

---

## 8. Drift structure and causal tracking

### 8.1 GT block coherence (1 s, all events, fit \(\mu\), \(K=12\))

Not classification. Per-unit/block \(\delta_{u,b}\) and \(g_{u,b}\) from GT
ownership, then \(C(d)=\mathrm{corr}(\delta_u,\delta_v)\) vs home distance.

| rec | mean \(\lvert\hat\delta\rvert\) (pitch) | nearest \(C(d)\) | mean pair corr | slope vs µm | gain pair corr | mean \(\lvert\log g\rvert\) |
|---|---:|---:|---:|---:|---:|---:|
| HJ drift | 0.088 | 0.153 | 0.121 | −0.0014 | 0.048 | 0.078 |
| HJ static | 0.051 | 0.040 | −0.015 | +0.0003 | 0.023 | 0.050 |
| MEArec 10u | 0.045 | −0.074 | 0.005 | +0.0008 | 0.035 | 0.030 |
| MEArec 20u | 0.050 | 0.042 | 0.001 | −0.0002 | 0.014 | 0.036 |

Nearest-bin \(C(d)=0.15<0.3\) on HJ drift → **not a local field**. Static
\(\lvert\hat\delta\rvert\approx0.05\) pitch → estimator is trusted enough not
to invent large false motion, but the residual “drift” is mostly noise.
D5 (smooth field) and shared-gain **skipped** (triggers recorded, not omitted).

> **Amended 2026-08-28 — Q5 is reopened.** About half of that static
> \(\lvert\hat\delta\rvert\) floor is amplitude bias, not noise:
> `estimate_best_shift` does not profile out a gain, so under-amplitude events
> lower the loss by moving \(\delta\). On HJ static the correlation between
> \(\lvert\hat\delta\rvert\) and \(\|x\|/\|\mu\|\) is −0.489 and drops to −0.058
> once a gain is profiled out. \(C(d)=0.153\) therefore sits inside the
> correctable error budget and the D5 / shared-gain skips are no longer
> settled. The numbers in this section are not rewritten; see
> [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md).

> **Amended 2026-08-28 (second) — the block design is underpowered and the
> window carries no drift.** \(C(d)\) is computed on 1 s blocks against a
> 0.25-pitch grid step; at this recording's rates that is ~5 events per
> unit-block and **1 event** for u62 (1.11 Hz). Independently, the 60 s window
> itself was measured against the static control: the 15 s test sub-window has
> a drift-to-noise ratio of 0.64. A correlation computed there is a
> correlation of noise with noise. See
> [`drift_window_budget.md`](drift_window_budget.md).

### 8.2 Causal trackers, test stream, \(K=12\) identity unless noted

Frozen \(K=12\) is the Part A A1 L2 row: HJ drift 0.9359 / 0.9141, HJ static
0.9584 / 0.9221, MEArec 10u 0.9514 / nan, MEArec 20u 0.9094 / 0.9396.

| rec | arm | acc | same-home | \(\Delta\)acc | \(\Delta\)home | \(\lvert\hat\delta_{\mathrm{end}}\rvert\) | \(\|\Delta\mu\|/d_{\mathrm{home}}\) | role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HJ drift | D0 Frozen | 0.9359 | 0.9141 | 0 | 0 | 0 | 0 | FROZEN |
| HJ drift | D1 EMA | 0.9502 | 0.9271 | **+1.42 pp** | +1.30 pp | 0 | **0.86** | EMA |
| HJ drift | D2 oracle shift | 0.9027 | 0.8750 | −3.32 pp | −3.91 pp | 0 | 0 | ORACLE |
| HJ drift | D3 global | 0.9383 | 0.9141 | +0.24 pp | 0.00 pp | 0.171 | 0 | GLOBAL |
| HJ drift | D4 tile | 0.9371 | 0.9141 | +0.12 pp | 0.00 pp | 0.221 | 0 | TILE |
| HJ drift | D3 \(K_{\mathrm{id}}=7,K_{\mathrm{st}}=12\) | 0.8766 | 0.8594 | −3.68 pp vs \(K=7\) Frozen | −6.0 pp | 0.170 | 0 | mixed \(K\) |
| HJ drift | D6 slow \(\mu\) | 0.9407 | 0.9141 | +0.47 pp | 0.00 pp | 0.101 | 0.26 | gated |
| HJ static | D1 EMA | 0.9555 | 0.9134 | −0.29 pp | −0.87 pp | 0 | 0.14 | EMA |
| HJ static | D2 oracle | 0.9254 | 0.8442 | −3.30 pp | −7.79 pp | 0 | 0 | ORACLE |
| HJ static | D3 global | 0.9584 | 0.9221 | 0.00 pp | 0.00 pp | **0.015** | 0 | GLOBAL |
| HJ static | D4 tile | 0.9584 | 0.9221 | 0.00 pp | 0.00 pp | 0.031 | 0 | TILE |
| HJ static | D6 | 0.9570 | 0.9177 | −0.14 pp | −0.43 pp | 0.015 | 0.03 | gated |
| MEArec 20u | D1 EMA | 0.5967 | 0.7427 | **−31.3 pp** | **−19.7 pp** | 0 | 0.06 | EMA |
| MEArec 20u | D2 oracle | 0.8768 | 0.9060 | −3.25 pp | −3.36 pp | 0 | 0 | ORACLE |
| MEArec 20u | D3 global | 0.9089 | 0.9351 | −0.04 pp | −0.45 pp | 0.026 | 0 | GLOBAL |
| MEArec 20u | D4 tile | 0.9098 | 0.9396 | +0.04 pp | 0.00 pp | 0.033 | 0 | TILE |
| MEArec 10u | D3 / D4 | 0.951–0.952 | nan | ~0 | nan | ≤0.018 | nan | quiet |

Stop-4 (shared tracker *worse* than Frozen on HJ drift, or static drop > 2 pp)
did not fire. Shared shift also **does not beat** Frozen same-home on HJ drift:
it ties. That is not a license to replace Frozen.

D6 was licensed from HJ-drift **calibration** (D3 cal same-home ≥ Frozen). Test
same-home is still a tie and identity moves. Do not keep D6. It was not used to
rescue a failed tracker.

EMA is the textbook warning: it can look good on HJ drift by absorbing residual
into \(\mu\), then it eats MEArec 20u because errors become identity.

> **Amended 2026-08-28 — the HJ-drift comparisons in this table are untested,
> not tied.** The 15 s test window has a measured drift-to-noise ratio of
> **0.64** against the static control, so it contains no drift signal. No
> shared-shift tracker could have separated from Frozen there, and the "ties"
> above are therefore uninformative about drift tracking rather than evidence
> against it. Separately, the full 600 s recording has 0.822 pitch of median
> motion (25 s-smoothed excursion; the profile is an out-and-back trip, not a
> ramp), with 9 of 11 units exceeding the \(\pm0.5\) pitch representable range
> and 4 uncoverable by any fixed reference point, so a 600 s rerun needs a
> wider or integrating state before it is meaningful.
> The MEArec 20u EMA collapse and the static-quiet rows are **not** affected:
> neither depends on drift being present. Numbers are not rewritten; see
> [`drift_window_budget.md`](drift_window_budget.md).

### 8.3 Controlled synthetic drift (static recordings, test-only)

Fit stays unperturbed. Acc is on the **perturbed** test set. Comparisons below
are vs the perturbed D0 Frozen on the **same** perturbation.

**Spatial ramp \(0\to 0.5\) pitch**

| rec | D0 acc (err 0.25) | D3 acc (err) | D4 acc (err) | D1 EMA acc (\(\|\Delta\mu\|\) ratio) |
|---|---|---|---|---|
| HJ static | 0.905 | **0.934 (0.075)** | **0.938 (0.112)** | 0.921 (0.25) |
| MEArec 10u | 0.890 | **0.946 (0.094)** | 0.913 (0.355) | 0.930 (nan) |
| MEArec 20u | 0.835 | **0.894 (0.114)** | 0.874 (0.231) | 0.637 (0.32) |

Global shared shift recovers the known ramp and **raises classification vs
perturbed Frozen**. Identity \(\mu\) stays put. Tile shift is better than Frozen
on HJ static, worse on MEArec 10u (too many groups for 10 well-separated units).

**Two-tile opposite \(\pm 0.5\) pitch.** D4 beats D3 on state error and on
MEArec 20u acc (0.896 vs 0.850 vs D0 0.829). A single global \(\delta\) cannot
represent opposite motion.

**Gain ramp \(1\to 1.5\) and gain step.** Shift trackers cannot represent gain;
they add false \(\delta\) and usually lose to Frozen. EMA helps HJ static
gain-ramp (0.944 vs D0 0.868) by writing gain into \(\mu\) (ratio 0.54) and
destroys MEArec 20u. Oracle free-shift is the worst gain method: it spends
spatial degrees of freedom on an amplitude problem. Shared gain was not
licensed (population gain corr 0.05).

Natural HJ drift in the last 15 s of this 60 s cut is small (~0.09 pitch
inferred). Frozen is enough. When a **known 0.5-pitch ramp** is injected,
shared \(\delta\) is the right object.

---

## 9. Stop conditions

| # | Condition | Outcome |
|---|---|---|
| 1 | Alignment / reconstruction up, same-home margin down | **Fired** on free shift (HJ drift P10 48 → 1.7) |
| 2 | Cosine helps drift, collapses dense/static same-home | **Not fired** (cosine raises mean same-home). Worst-unit drop on MEArec 20u still blocks cosine. |
| 3 | \(L_{\mathrm{flex}}\ge 0\) | **Fired** on every co-home recording |
| 4 | Shared/local tracker worse than Frozen on HJ drift, or hurts static | **Not fired** at the coded threshold (home delta \(<0\) or static drop \(>2\) pp). Shared shift **ties** Frozen same-home — not a replace. |
| 5 | Lots of false motion on no-drift data | **Not fired** (\(\lvert\hat\delta\rvert\approx0.05\) pitch) |
| 6 | Slow \(\mu\) used to rescue a failed state tracker | **Not used that way.** D6 ran from cal license; test same-home still a tie; drop D6. |
| 7 | Larger \(K\) only lifts overall / reconstruction, hardest same-home / worst-unit stays down | **Do not select \(K=16\)** (MEArec 20u worst-unit 0.820 → 0.779) |

Confirmation corpus stays closed. No test-set retune.

Conditional arms: A5 ran (\(\lambda=0.1\)); C1 \(\{\pm2\}\) skipped; D5 skipped;
D6 ran then discarded; shared gain skipped.

---

## 10. Architecture judgment

Supported:

- biased-dot ≡ L2 on co-home ranking, and is the MVM form \(S=Mx+b\)
- POSNEG > P2P; the trough is the identity-bearing half
- absolute amplitude in \(\mu_u\) has identity value (pure-dot collapses)
- free per-candidate shift/gain is harmful (\(L_{\mathrm{flex}}>0\), EMA eats dense MEArec)
- small spatial motion **can** be explained by a shared/local \(\delta\) that
  does not rewrite \(\mu\) (controlled ramp; static stays quiet)

Not supported on this 60 s window:

- replacing Frozen with a tracker for HJ-drift test same-home
- cosine as the deployable score
- unit-specific free \(\delta_u\), per-event gain, mixed \(K_{\mathrm{id}}\neq K_{\mathrm{state}}\)
- storing \(r_{pn}\) / \(\Delta c\) on top of raw POSNEG
- \(K=16\) as the identity stencil

**Deployable object after this knife:**

```
POSNEG event
  → candidate-home physical-offset slots (geometry-K, default K=12)
  → optional tracked shared/local spatial frame  (safe, not yet a HJ-test win)
  → MVM identity match  S = M x + b ,  b_u = -||μ_u||²
```

Each unit memory keeps an **absolute POSNEG template** \(\mu_u\). Drift is not
a row-local freedom the current query may spend. If it is used at all, it is
a historical shared/local state applied to the reference frame **before**
scoring, then frozen for that event.

Better fit is not better sorting. This round only the latter counts.

---

## 11. Plots and hashes

Plots:

- `Spatial/output/posneg_mvm_ablation/plots/same_home_acc_k12.png`
- `Spatial/output/spatial_shift_ablation/plots/lflex_k12.png`
- `Spatial/output/drift_coherence/plots/coherence_cd_k12.png`
- `Spatial/output/drift_tracking_compare/plots/tracking_delta_home_acc.png`
- `Spatial/output/controlled_drift/plots/controlled_spatial_ramp_traces.png`

Matching CSV rerun hash:
`1a2564e5e9d949347e386dacabef5cf70b8fa36cad8ccfa649ef4f46b1497430`.

Contract tests: `Spatial/tests/test_posneg_mvm_shift_tracking.py`, 8 passed
twice. Source/output SHA-256:
[`posneg_mvm_shift_drift_provenance.md`](posneg_mvm_shift_drift_provenance.md).
