# Identity freeze and a fair EMA* drift baseline

English analysis for STAR-Mem after the POSNEG / MVM / free-shift / causal-tracking
knife. Hardware is out of scope. Success is **sorting**: overall accuracy,
same-home accuracy, same-home margin P10, worst-unit recall, error accumulation
(\(\|\mu(t)-\mu(0)\|\)), and controlled \(|\hat\theta-\theta_{\mathrm{GT}}|\).
Reconstruction / best-alignment is not success.

This document freezes identity, defines **EMA\*** and **Oracle-EMA**, states
pass/kill rules for any later tracker, and answers A–G. Numbers for A and the
killed-branch list come from [`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md).
EMA\* / Oracle-EMA numbers come from
`Spatial/output/ema_star/` after the frozen 60 s pilot.

Mapping Row, free deformable templates, free per-candidate shift/gain, QDA,
NCA, autoencoders, extra \(r_{pn}\)/\(\Delta c\), \(K\) sweeps, and cosine-as-default
are **not** reopened.

---

## Frozen identity (from the previous knife)

| Decision | Evidence | Action |
|---|---|---|
| Candidate-home physical-offset frame | Event-argmax: HJ overall −8 to −12 pp; HJ drift worst-unit 0.79 → 0.26 | **Freeze** |
| Absolute POSNEG concat, default \(K=12\) | POSNEG vs P2P ~+3 pp HJ same-home; trough is the identity-bearing half; \(r_{pn}\)/\(\Delta c\) add nothing; \(K=16\) hurts MEArec 20u worst-unit | **Freeze** |
| L2 / biased-dot \(S=2x^\top\mu-\|\mu\|^2\) | L1 worse on all four recordings; co-home ranking identical for L2 and biased-dot | **Freeze** |
| Keep \(\|\mu\|^2\) (no pure-dot) | Pure-dot HJ drift acc 0.28, same-home 0.00 | **Stop** |
| Cosine is not the default score | Mean same-home up; MEArec 20u worst-unit 0.809 → 0.677 | **Stop as default** |
| No per-event / per-candidate free shift, gain, \(h\) | \(L_{\mathrm{flex}}>0\) everywhere with co-home rivals; HJ drift margin P10 48 → 1.7 | **Stop** |

Identity object:

```
event → candidate-home geometry-K slots (K=12)
      → absolute POSNEG mean μ_u
      → d_u = ||x_u − μ_u||²   (biased-dot twin for MVM later; not a hardware topic here)
```

Do not continue QDA / NCA / AE / skewness-sharpness / extra polarity features /
\(K\) sweeps unless a **new** information source is proposed that is not already
in raw POSNEG. The previous knife already asked those representation questions.

---

## EMA* protocol (this knife)

Mean-template update after high-confidence assignment:

\[
\mu_{\hat u}(t+1)=(1-\eta)\mu_{\hat u}(t)+\eta x_t.
\]

Causal order is unchanged: **score with frozen \(\mu\) → assign → maybe update
for future events**.

| Field | Frozen value |
|---|---|
| \(\eta\) grid | \(\{0.005,0.01,0.02,0.05,0.1\}\) |
| \(\eta^*\) | **One** global value from **calibration** (not per recording, not test) |
| Cal selection | event-weighted accuracy on the four cal splits, then nanmean same-home, then min worst-unit, then smaller \(\eta\) |
| Confidence | same as last knife: \(M_{\mathrm{all}}>\max(0,\) Frozen cal \(M_{\mathrm{all}}\) P10\()\) |
| Frame / \(K\) / score | candidate-home POSNEG L2, \(K=12\) |
| Recordings | HJ drift, HJ static, MEArec 10u, MEArec 20u; 60 s; 50/25/25 |
| Controlled | spatial ramp \(0\to0.5\) pitch; gain ramp \(1\to1.5\); simultaneous shift+gain (both ramps on the same test events) |
| Self-EMA* | update owner = **predicted** unit |
| Oracle-EMA | same \(\eta^*\), same formula, same confidence on the **GT** unit’s margin; update owner = **GT** unit. Label **DIAGNOSTIC** |

Not in this baseline (would inflate it): confidence-threshold sweep, normalized
EMA, per-unit count-normalized update, robust/clipped EMA. Those are optional
later **updater** knives if Oracle-EMA ≫ self-EMA* and the gap is confidence,
not \(\eta\).

### Pass / kill vs EMA* (any later explicit tracker)

“Better than Frozen” is **not** success.

| Setting | Pass | Kill |
|---|---|---|
| Natural HJ drift | same-home and overall **≥ EMA\***; a ~1 pp same-home lift is a real win | below EMA* same-home, or worst-unit drop vs EMA* |
| HJ static / MEArec static | stay within 1 pp overall of Frozen/EMA* (quiet) | >2 pp same-home or overall drop vs Frozen |
| Dense MEArec 20u | no EMA-style collapse (overall drop ≳ 5 pp vs Frozen, or worst-unit collapse) | collapse |
| Controlled spatial ramp | recover \(\delta\); sorting ≥ perturbed Frozen and **≥ EMA\*** | false motion without sorting gain |
| Controlled gain ramp | must **not** explain amplitude as spatial shift; sorting ≥ EMA* without rewriting \(\mu\) if claiming identity/state split | hallucinated \(\delta\) as the gain mechanism |
| Simultaneous shift+gain | both states identifiable, or an honest failure | one state used as a dump for the other |
| Identity/state split | \(\|\mu(t)-\mu(0)\|\) stays small **and** sorting ≥ EMA* | \(\mu\) walks like EMA while claiming “state” |

When to **stop drift modeling and keep EMA\***: EMA* ≥ every explicit tracker
on natural HJ same-home **and** does not collapse MEArec 20u **and** Oracle-EMA
is not substantially above self-EMA* on HJ drift (no unused ownership headroom).

When **identity/state separation is proven**: a frozen-\(\mu\) tracker with
historical \([\delta]\) and/or \([\gamma]\) beats EMA* on natural or controlled
sorting **without** MEArec collapse and **without** \(\mu\) walking.

---

## A–G

### A. Is identity frozen?

**Yes.** Candidate-home physical-offset + absolute POSNEG + L2 / biased-dot is
the identity object. Do **not** continue QDA, NCA, AE, extra \(r_{pn}\)/\(\Delta c\),
skewness/sharpness, or \(K\) sweeps on this observation. Event-argmax, cosine-as-default,
pure-dot, and free per-candidate shift/gain stay dead.

The previous knife already asked those representation questions. The remaining
uncertainty is **online update**, not the static identity feature.

### B. Transform vs ownership?

At the **fair** \(\eta^*=0.005\), Oracle-EMA **equals** self-EMA* on HJ drift
test (both 0.9466 overall, 0.9245 same-home). Oracle is not a hidden upper bound
here.

That is the opposite of “fix ownership first.” The \(\eta=0.05\) collapse on
MEArec 20u (cal acc 0.899 → 0.645) was **too-fast self-update**, not a missing
deformation model. Slowing \(\eta\) removes the collapse without GT ownership.

On **controlled gain** in dense MEArec 20u, ownership *does* matter: self-EMA*
−0.7 pp vs perturbed Frozen, Oracle-EMA **+7.7 pp**. So “who may update”
becomes the issue **when the residual is large and the recording is dense**,
not on this natural 15 s HJ cut.

**Natural drift next step: safer updater (already EMA\*), not a new transform.**
Explicit spatial/gain state is reserved for large coherent perturbations that
slow EMA cannot follow (see E).

### C. Fair EMA baseline

This round’s baseline **is** EMA*: \(\eta\in\{0.005,0.01,0.02,0.05,0.1\}\), one
global \(\eta^*\) from calibration (event-weighted acc, then same-home, then
worst-unit, then smaller \(\eta\)). Result: \(\eta^*=0.005\).

Do **not** add, in this baseline:

- confidence-threshold sweep (keep Frozen cal \(M_{\mathrm{all}}\) P10)
- normalized EMA
- per-unit count-normalized update
- robust/clipped EMA

Those would make the “simple strong baseline” as complex as a tracker. They are
optional **later updater** knives only if a future large-residual setting shows
Oracle-EMA ≫ self-EMA* **and** the gap is confidence, not \(\eta\).

### D. Pass / kill vs EMA* (with this CSV)

| Setting | EMA* vs Frozen | What an advanced tracker must do |
|---|---|---|
| Natural HJ drift | **+1.07 pp overall, +1.04 pp same-home, worst-unit 0.792→0.825** | ≥ EMA* on overall **and** same-home; ~+1 pp more is a real win |
| HJ static | −0.14 pp overall, −0.43 pp same-home (quiet) | must not drop >2 pp vs Frozen |
| MEArec 20u | −0.04 pp overall, same-home tie, worst-unit tie | **no collapse** (η=0.05 would have failed this) |
| Controlled spatial 0.5 pitch | HJ static +0.29 pp; MEA 20u +1.39 pp (small) | must beat EMA* **and** recover \(\delta\); previous global-shift +2.9 pp on HJ static ramp is the bar |
| Controlled gain 1→1.5 | HJ static +0.43 pp; MEA 20u **−0.69 pp** | must not hallucinate spatial shift; must not collapse MEA 20u |
| Simultaneous shift+gain | HJ static +1.00 pp; MEA 20u **−2.04 pp** | both states identifiable, or an honest failure; self-EMA* fails dense MEA |

Oracle-EMA on natural data ≈ EMA*. On MEArec 20u **gain** / **shift+gain**,
Oracle-EMA is +7.7 / +6.9 pp vs Frozen while self-EMA* is negative: that is
ownership headroom **only under large gain**, not a license to complicate
natural-HJ tracking.

### E. Two-state \([\delta_g,\gamma_g]\) with frozen \(\mu_u\)?

**Not this round.** Calibration EMA* **does** dominate Frozen once \(\eta\) is
fair (event-weighted cal is dominated by not collapsing MEArec 20u). Criterion 4
therefore defers implementation.

The **named next knife**, if we leave natural-HJ, is exactly that two-state
tracker with **frozen \(\mu\)**:

- Prior global-shift recovered a 0.5-pitch ramp better than this EMA* (+2.9 pp
  vs +0.3 pp on HJ static).
- Spatial shift cannot explain gain (prior knife); EMA writes gain into \(\mu\).
- Population gain corr was 0.05 — so \(\gamma_g\) is only licensed on
  **controlled** or future recordings that actually show coherent gain.

It is **not** licensed as a replacement for EMA* on this 60 s natural HJ cut.

### F. Frozen → global → tile, never unit-free \(\delta_u\)

**Keep.** HJ drift nearest \(C(d)=0.15<0.3\). Weak coherence must not create
per-unit free state (that is the free-shift \(L_{\mathrm{flex}}\) failure).
No population coherence → Frozen or slow EMA*. Global coherence → global
\([\delta]\) (and \(\gamma\) only if gain is coherent). Tile only if global
fails and tile \(C(d)\) is real.

### G. Small next matrix and the boxed question

\[
\boxed{\text{On this development window we need a safer template updater, not a more complex drift model.}}
\]

That updater **is EMA\*** with \(\eta^*=0.005\).

1. **Baselines that must stay in every later table:** Frozen; EMA* \(\eta^*=0.005\);
   Oracle-EMA (diagnostic only).
2. **One advanced knife later (not run here):** frozen-\(\mu\) shared
   \([\delta_g,\gamma_g]\) on **controlled spatial + gain + shift+gain**, then
   natural HJ only if it passes controlled pass/kill vs EMA*.
3. **Stop now:** Mapping Row; free shift/gain/\(h\); cosine default; extra
   \(r_{pn}\)/\(\Delta c\); \(K\) sweeps; mixed \(K_{\mathrm{id}}\); QDA/NCA/AE;
   \(\eta=0.05\) EMA as if it were *the* baseline; “beats Frozen” as a tracker win.
4. **Stop drift modeling and keep EMA\* when:** a candidate tracker is ≤ EMA*
   on HJ drift same-home **and** no safer than EMA* on MEArec 20u **and**
   Oracle-EMA is not ≫ self-EMA* on that setting. **That already holds for
   natural 60 s HJ.**
5. **Identity/state separation is proven only when:** frozen \(\mu\) + historical
   \([\delta]\) and/or \([\gamma]\) **beats EMA\*** on sorting (natural or
   controlled) **without** MEArec collapse and **without** \(\mu\) walking.

This round ran Frozen, self-EMA*, and Oracle-EMA only. The named advanced
knife was **not** implemented: cal EMA* dominates Frozen once \(\eta\) is fair.

---

## EMA* / Oracle-EMA results

Isolation: GT events/units, 60 s, chronological 50/25/25, candidate-home POSNEG
L2, \(K=12\). \(\eta^*=0.005\) from calibration (event-weighted accuracy).
Command: `python -m Spatial.experiments.run_ema_star --pilot --duration 60`.

### Calibration \(\eta\) grid (self-EMA overall acc)

| \(\eta\) | HJ drift | HJ static | MEArec 10u | MEArec 20u |
|---:|---:|---:|---:|---:|
| 0.005 | 0.9561 | 0.9570 | 0.9618 | **0.8990** |
| 0.01 | 0.9585 | 0.9555 | 0.9618 | 0.8959 |
| 0.02 | 0.9573 | 0.9555 | 0.9610 | 0.8877 |
| 0.05 | 0.9585 | 0.9584 | 0.9601 | **0.6448** |
| 0.1 | 0.9561 | 0.9570 | 0.9601 | **0.5867** |

\(\eta=0.05\) — the only rate in the previous knife — **collapses MEArec 20u on
cal**. A global event-weighted choice therefore selects \(\eta^*=0.005\).

> **Scope amendment 2026-08-28.** This grid was run at \(K=12\) only, and the
> collapse boundary is a function of template dimension \(D=2K\). On MEArec 20u
> it sits at \(\eta=0.05/0.03/0.02\) for \(K=7/12/16\), i.e. at an effective
> sample size per dimension of 2.7–3.1 in every case, giving
> \(\eta_{\mathrm{crit}}\approx1/(3K)\) and a scope rule \(\eta\le1/(5K)\).
> \(\eta^*=0.005\) is unchanged and is **not** reselected, but it must not be
> carried to another \(K\), template dimension, or denser probe without
> rechecking. See
> [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md).

### Natural test

| rec | arm | acc | same-home | \(M\) P10 | worst-unit | \(\Delta\)acc vs Frozen | \(\|\Delta\mu\|/d_{\mathrm{home}}\) |
|---|---|---:|---:|---:|---:|---:|---:|
| HJ drift | Frozen | 0.9359 | 0.9141 | 48.45 | 0.792 | 0 | 0 |
| HJ drift | EMA* | **0.9466** | **0.9245** | 65.60 | **0.825** | **+1.07 pp** | 0.289 |
| HJ drift | Oracle-EMA | 0.9466 | 0.9245 | 67.50 | 0.825 | +1.07 pp | 0.276 |
| HJ static | Frozen | 0.9584 | 0.9221 | 42.60 | 0.869 | 0 | 0 |
| HJ static | EMA* | 0.9570 | 0.9177 | 42.75 | 0.861 | −0.14 pp | 0.032 |
| HJ static | Oracle-EMA | 0.9584 | 0.9221 | 42.71 | 0.869 | 0.00 pp | 0.030 |
| MEArec 10u | Frozen | 0.9514 | nan | nan | 0.898 | 0 | 0 |
| MEArec 10u | EMA* | 0.9523 | nan | nan | 0.898 | +0.09 pp | nan |
| MEArec 10u | Oracle-EMA | 0.9523 | nan | nan | 0.898 | +0.09 pp | nan |
| MEArec 20u | Frozen | 0.9094 | 0.9396 | 3101 | 0.809 | 0 | 0 |
| MEArec 20u | EMA* | 0.9089 | 0.9396 | 3087 | 0.809 | −0.04 pp | 0.029 |
| MEArec 20u | Oracle-EMA | 0.9094 | 0.9396 | 3102 | 0.809 | 0.00 pp | 0.021 |

Oracle-EMA − self-EMA* = **0** on HJ drift. Slow EMA is quiet on static and
dense MEArec. \(\mu\) still walks on HJ drift (ratio 0.29): this is residual
absorption, not a separated state.

> **Attribution amendment 2026-08-28.** The \(+1.07\) pp on HJ drift **stands
> as a number**, but it must not be described as drift tracking. That test
> window was measured directly against the static control and has a
> drift-to-noise ratio of **0.64** — there is no ongoing motion in it for an
> EMA to follow. The available explanation is a **stale-template offset**:
> \(\mu\) is fitted on the first 30 s, which does move, so the fitted mean is a
> time-average of a moving target and is displaced from the test window's true
> footprint; an EMA removes a constant offset of that kind. That also explains
> the drift/static asymmetry, since HJ static has no such offset. **This is a
> hypothesis, not a measurement** — it is separable by a Frozen arm refitted on
> the last block of the fit split, which has not been run. The controlled-ramp
> rows below are unaffected. See
> [`drift_window_budget.md`](drift_window_budget.md).

### Controlled test (static recordings; \(\Delta\) vs **perturbed** Frozen)

| rec | perturbation | Frozen acc | EMA* acc (\(\Delta\)) | Oracle acc (\(\Delta\)) |
|---|---|---:|---|---|
| HJ static | spatial ramp | 0.9053 | 0.9082 (+0.29 pp) | 0.9125 (+0.72 pp) |
| HJ static | gain ramp | 0.8680 | 0.8723 (+0.43 pp) | 0.8938 (+2.58 pp) |
| HJ static | shift+gain | 0.9383 | 0.9484 (+1.00 pp) | 0.9455 (+0.72 pp) |
| MEArec 10u | spatial ramp | 0.8899 | 0.8942 (+0.43 pp) | 0.8942 (+0.43 pp) |
| MEArec 10u | gain ramp | 0.6522 | 0.6947 (+4.25 pp) | 0.7042 (+5.20 pp) |
| MEArec 10u | shift+gain | 0.7285 | 0.7580 (+2.95 pp) | 0.7589 (+3.04 pp) |
| MEArec 20u | spatial ramp | 0.8348 | 0.8487 (+1.39 pp) | 0.8651 (+3.04 pp) |
| MEArec 20u | gain ramp | 0.5360 | 0.5291 (**−0.69 pp**) | **0.6132 (+7.72 pp)** |
| MEArec 20u | shift+gain | 0.6440 | 0.6236 (**−2.04 pp**) | **0.7129 (+6.90 pp)** |

Slow EMA follows a 0.5-pitch ramp only weakly (prior global-shift was +2.9 pp
on HJ static spatial ramp). On dense gain, self-EMA* **loses** to Frozen while
Oracle-EMA **gains**: ownership, not \(\eta\), is the dense-gain issue.

CSV hash (two runs):
`b05bbcc7b7c66dd433401e93b856e5cee4f88679513b8e44fbb6639812d85e00`.

Plot: `Spatial/output/ema_star/plots/ema_star_acc.png`.

## Provenance

```text
e7415fc35d1468a369397d347df1563efa47441fa125803762655a0ea348fdea  Spatial/algorithms/drift_tracking.py
7f67a810fd918c00010313743d0b91caa939ce9b9df17d339e4487cadf22b1ce  Spatial/experiments/run_ema_star.py
a474f44d748aa184dc7ad7cce6d7ff5abd657f383eecef926fbda18a47816711  Spatial/tests/test_ema_star.py
b05bbcc7b7c66dd433401e93b856e5cee4f88679513b8e44fbb6639812d85e00  Spatial/output/ema_star/ema_star_pilot.csv
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_ema_star.py -q`: 3 passed, twice.

