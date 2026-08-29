# Causal drift tracking, coherence, and controlled drift — results

Companion to [`posneg_mvm_shift_drift_results.md`](posneg_mvm_shift_drift_results.md).
Protocol: [`posneg_mvm_shift_drift_plan.md`](posneg_mvm_shift_drift_plan.md).
Free per-event shift is **not** a tracker; D2 is labeled ORACLE / DIAGNOSTIC.

GT events/units. Test stream only (last 25% of 60 s), \(\mu\) from fit.
Identity scoring is L2 on candidate-home POSNEG. Shared \(\delta\) is in pitch units
along the primary axis. Confidence: \(M_{\mathrm{all}} > \max(0,\) cal Frozen P10\()\).

## Verdict

| Question | Answer |
|---|---|
| Q5 | Inferred 1 s-block \(\delta\) is **weak, mostly unit-specific / noisy**, not a strong global or tile field. HJ drift nearest-bin \(C(d)=0.15<0.3\). Static recordings are quieter (\(\lvert\hat\delta\rvert\approx0.05\) pitch) — the estimator is not hallucinating large motion. |
| Q6 | **Frozen remains the deployable tracker on this 15 s test window.** D3/D4 match Frozen same-home on HJ drift (0.914) and stay quiet on static. D1 EMA helps HJ drift (+1.4 pp) but **collapses MEArec 20u (−31 pp)** while rewriting \(\mu\). D2 oracle free-shift **hurts**. D5 skipped. D6 does not beat Frozen same-home and moves \(\mu\). |
| Q7 | **Yes, if the state is shared \(\delta\) and \(\mu\) is frozen** (D3/D4: \(\|\mu(t)-\mu(0)\|=0\)). EMA and D6 move identity (HJ drift EMA ratio 0.86 of nearest same-home template distance). |
| Q8 | **Mostly yes for shared shift.** HJ static / MEArec D3 \(\lvert\hat\delta_{\mathrm{end}}\rvert=0.015\) / \(0.026\) pitch. EMA is not quiet on identity: it always writes \(\mu\). |

> **Amended 2026-08-28 — Q5 is reopened.** The static
> \(\lvert\hat\delta\rvert\) floor used to justify "the estimator is not
> hallucinating large motion" is about half amplitude bias:
> `estimate_best_shift` profiles out no gain, so under-amplitude events buy a
> lower loss by moving \(\delta\). \(C(d)=0.153\) is inside the correctable
> error budget, so the D5 and shared-gain skips are not settled. Roughly
> 0.05 pitch of real motion does survive amplitude matching, so the Q6 verdict
> is expected to stand — ~~this last clause is superseded by the amendment
> below~~. Numbers below are unchanged; see
> [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md).

> **Amended 2026-08-28 (second) — Q6's natural-HJ arm is untested, not tied.**
> The 15 s test window was measured directly: its drift-to-noise ratio is
> **0.64**, i.e. the drift recording shows *less* apparent motion there
> (1.01 µm) than the no-drift static control does (1.57 µm). No tracker could
> have won on that window and Frozen could not have lost, so "Frozen remains
> the deployable tracker" carries no information about drift tracking. The
> full 600 s recording does contain 0.822 pitch of median motion (25 s-smoothed
> excursion; the profile is an out-and-back trip, not a ramp), and **9 of 11
> units exceed the \(\pm0.5\) pitch representable range** of `SHIFT_FRACTIONAL`
> with 4 uncoverable by any fixed reference point. The static-quiet arms (Q8)
> and the controlled-ramp responses are unaffected. Numbers below are
> unchanged; see [`drift_window_budget.md`](drift_window_budget.md).

> **Amended 2026-08-28 (third) — Q7 has a third object that is neither state
> nor identity.** Q7 asks whether state can move while identity \(\mu_u\)
> stays put, and D3/D4 answer yes. But every template is indexed on a **frozen
> channel selection**: `table[fit_home]`, where the home is the mode of the
> per-event peak channel over the fit split. `CausalTracker.unit_homes` is
> assigned once and read thereafter — **no arm writes it, including EMA**, and
> the shift bank is a closed \(K\to K\) map that cannot bring in an outside
> channel. Over 600 s, **8 of 11 units have blocks whose own majority home
> differs from the fit home** (worst 48/120 blocks, 3 distinct channels
> visited); six of the eight have a static control of zero or one block, which
> rules out readout instability. The three
> that never migrate are the array-edge units, whose home pins while the
> footprint keeps moving. Along the drift axis the electrode rows step 20 µm
> while `pitch_um` is the staggered-diagonal 25.01 µm, so \(\pm0.5\) pitch is
> only **±0.63 rows** — one row of drift is already out of range, and no grid
> point lands on the lattice. Widening the grid does not fix the slot
> misalignment. See [`drift_window_budget.md`](drift_window_budget.md) §5.

## GT block coherence (1 s, all events, fit \(\mu\), \(K=12\))

| rec | mean \(\lvert\hat\delta\rvert\) (pitch) | nearest \(C(d)\) | mean pair corr | slope vs distance | gain pair corr | mean \(\lvert\log g\rvert\) |
|---|---:|---:|---:|---:|---:|---:|
| HJ drift | 0.088 | 0.153 | 0.121 | −0.0014 / µm | 0.048 | 0.078 |
| HJ static | 0.051 | 0.040 | −0.015 | +0.0003 | 0.023 | 0.050 |
| MEArec 10u | 0.045 | −0.074 | 0.005 | +0.0008 | 0.035 | 0.030 |
| MEArec 20u | 0.050 | 0.042 | 0.001 | −0.0002 | 0.014 | 0.036 |

D5 trigger (nearest \(C>0.3\), falling with distance, static \(\lvert\hat\delta\rvert<0.15\)): **failed** on \(C(d)\). Shared-gain trigger (gain corr \(>0.3\)): **failed**. Both skipped, not silently omitted.

## Part D — causal tracking (test)

Frozen \(K=12\) baselines (same as Part A A1): HJ drift acc 0.9359 / home 0.9141; HJ static 0.9584 / 0.9221; MEArec 10u 0.9514 / nan; MEArec 20u 0.9094 / 0.9396.

| rec | arm | acc | same-home | \(\Delta\)acc vs Frozen | \(\Delta\)home | \(\lvert\hat\delta_{\mathrm{end}}\rvert\) | \(\|\Delta\mu\|/d_{\mathrm{home}}\) | role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HJ drift | D0 Frozen \(K=12\) | 0.9359 | 0.9141 | 0 | 0 | 0 | 0 | FROZEN |
| HJ drift | D1 EMA | 0.9502 | 0.9271 | **+1.42 pp** | +1.30 pp | 0 | **0.86** | EMA |
| HJ drift | D2 oracle shift | 0.9027 | 0.8750 | −3.32 pp | −3.91 pp | 0 | 0 | ORACLE |
| HJ drift | D3 global \(K=12\) | 0.9383 | 0.9141 | +0.24 pp | 0.00 pp | 0.171 | 0 | GLOBAL |
| HJ drift | D4 tile \(K=12\) | 0.9371 | 0.9141 | +0.12 pp | 0.00 pp | 0.221 | 0 | TILE |
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

Stop-4 (shared tracker worse than Frozen on HJ drift, or hurts static) **did not fire** at the coded threshold (home delta \(<0\) or static drop \(>2\) pp). Shared shift also **does not beat** Frozen same-home on HJ drift: it ties. That is not a license to replace Frozen.

D6 was licensed from HJ-drift **calibration** (D3 cal same-home ≥ Frozen). It is not used to rescue a failed tracker. Test same-home is still a tie, and identity moves. Do not keep D6.

Mixed \(K_{\mathrm{id}}=7,K_{\mathrm{st}}=12\) **hurts** HJ drift. Identity matching and state estimation should not be split that way on this probe.

## Controlled synthetic drift (static recordings, test-only perturbation)

Fit stays unperturbed. Acc below is on the **perturbed** test set. \(\Delta\) vs the perturbed D0 Frozen on the same perturbation (not vs the unperturbed recording).

### Spatial ramp \(0\to 0.5\) pitch

| rec | D0 acc | D3 acc (\(\lvert\hat\delta-\delta_{\mathrm{GT}}\rvert\)) | D4 acc (err) | D1 EMA acc (\(\|\Delta\mu\|\) ratio) |
|---|---:|---|---|---|
| HJ static | 0.905 | **0.934 (0.075)** | **0.938 (0.112)** | 0.921 (0.25) |
| MEArec 10u | 0.890 | **0.946 (0.094)** | 0.913 (0.355) | 0.930 (nan) |
| MEArec 20u | 0.835 | **0.894 (0.114)** | 0.874 (0.231) | 0.637 (0.32) |

Global shared shift recovers the known ramp (err 0.07–0.11 vs Frozen’s 0.25, the mean of a 0–0.5 ramp) and **raises classification vs perturbed Frozen**. Identity \(\mu\) stays put. EMA also lifts some recordings but by rewriting \(\mu\). Tile shift is better than Frozen on HJ static / MEArec 20u, worse on MEArec 10u (over-parameterized for 10 well-separated units).

### Two-tile opposite \(\pm 0.5\) pitch

D4 beats D3 on state error and on MEArec 20u acc (0.896 vs 0.850 vs D0 0.829). Global cannot represent opposite motion.

### Gain ramp \(1\to 1.5\) and gain step \(1\to 1.5\)

Shift trackers cannot represent gain; they add false \(\delta\) and usually **lose** to Frozen. EMA helps HJ static gain-ramp (0.944 vs D0 0.868) by absorbing gain into \(\mu\) (ratio 0.54) and **destroys** MEArec 20u. Oracle free-shift is the worst gain method (it spends spatial degrees of freedom on an amplitude problem). Shared gain was **not** licensed (no population gain coherence).

## Architecture after D

On the locked 60 s development window, natural drift in the last 15 s is small. Frozen candidate-home POSNEG L2 / biased-dot is the classifier. A shared/local \(\delta\) is **safe** (static stays quiet, \(\mu\) does not walk) and is the right object when a larger, known spatial ramp is present. It is **not** yet a classification win on HJ drift test. Free per-candidate shift/gain and naive EMA are harmful.

Plot: `Spatial/output/drift_tracking_compare/plots/tracking_delta_home_acc.png`,
`Spatial/output/drift_coherence/plots/coherence_cd_k12.png`,
`Spatial/output/controlled_drift/plots/controlled_spatial_ramp_traces.png`.

```bash
python -m Spatial.experiments.run_drift_coherence --pilot --duration 60
python -m Spatial.experiments.run_drift_tracking_compare --pilot --duration 60
python -m Spatial.experiments.run_controlled_drift --pilot --duration 60
```
