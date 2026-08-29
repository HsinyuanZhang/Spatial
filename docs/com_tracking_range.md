# COM tracking, isolated from identity — can a causal predictor stay in range?

Status: **diagnostic, not a selected tracker.** Date: 2026-08-29.
**No classification.** GT events, GT unit rows, full **600 s**. Two independent
Hybrid Janelia **drift** recordings: `drift16c_600s_11` and `drift16c_600s_12`.

Companion: [`tracked_home_and_com_state.md`](tracked_home_and_com_state.md),
[`drift_window_budget.md`](drift_window_budget.md).

```bash
python -m Spatial.experiments.run_com_tracking --pilot
# defaults to drift16c_600s_11 drift16c_600s_12
```

---

## 1. Why this is a separate question

Two things were being mixed:

1. **Where is the unit?** — COM along the probe, in micrometres.
2. **Who is the unit?** — absolute POSNEG L2 in a physical slot frame.

(1) can fail while (2) is still the right identity object, and the 60 s
knives never tested (1) because that window has no COM motion above noise.
This knife asks only (1), on the **full 600 s** of two drift recordings.

Truth is the same amplitude-weighted COM used for the window budget (gain
invariant, unbounded). Time is 5 s blocks; a block needs ≥3 spikes.

**Causal rule:** the prediction for block \(i\) may use COM observations with
index \(< i\) only. Frozen may use only blocks whose centre is before 30 s
(the pipeline fit split of the old 60 s cut).

This is a **GT-COM dynamics diagnostic**. Hold / EMA observe the **true**
previous-block COM. They answer: *if the last block’s COM were known, would
a 5 s lag still sit in the correct range?* They do **not** yet answer
online COM estimation under self-assignment.

---

## 2. What “in the correct COM range” means

Row step along the drift axis is **20 µm**. Pipeline `pitch_um` is 25.01 µm
(staggered diagonal). Three radii are reported; the operational one is
**one row**.

| Name | Radius | Meaning |
|---|---:|---|
| half-row | 10 µm | still on the same side of a row boundary |
| **one row** | **20 µm** | predicted COM still names the true peak’s neighbourhood |
| half-pitch | 12.5 µm | half of the pipeline pitch, for comparison with \(\pm0.5\) grids |
| channel match | — | nearest electrode to \(\widehat{\mathrm{COM}}\) equals nearest to GT COM |

Channel match is stricter than one-row: COM sitting between two sites flips
the integer channel even when \(|\mathrm{err}|\) is 1–2 µm.

Streaming predictors (event order, freeze then maybe commit):

| Arm | Rule |
|---|---|
| `stream_frozen` | median COM of \(t<30\) s, held for every later spike |
| `stream_nN` | after freeze, buffer \(N\) spikes, commit their median |
| `stream_dtτ` | after freeze, buffer until \(\tau\) seconds elapsed, commit median |
| `stream_emaα` | per-spike EMA of COM, \(\hat c_i\) uses only \(j<i\) |

Block predictors (previous section):

| Arm | Rule |
|---|---|
| `frozen_fit` | median COM of \(t<30\) s, held forever |
| `hold` | last finite observation (5 s zero-order hold) |
| `ema_0.2` / `0.5` / `1` | causal EMA; \(\alpha=1\) ≡ hold |
| `velocity` | \(\hat y_i = y_{i-1}+(y_{i-1}-y_{i-2})\) |

---

## 3. Results — two drift recordings, 600 s, no classification

Both scenes have 11 GT units. Same protocol; numbers are **not** pooled.

### Population (median over units)

| scene | arm | median \|err\| | frac ≤ 20 µm (median / **worst**) | channel match (median / worst) |
|---|---|---:|---|---|
| `drift16c_600s_11` | frozen_fit | 6.31 µm | 1.00 / **0.088** | 0.658 / **0.018** |
| `drift16c_600s_11` | **hold** | **1.40 µm** | **1.00 / 0.950** | **0.891 / 0.689** |
| `drift16c_600s_11` | ema_0.5 | 1.32 µm | 1.00 / 0.983 | 0.874 / 0.664 |
| `drift16c_600s_11` | velocity | 2.44 µm | 1.00 / 0.798 | 0.832 / 0.597 |
| `drift16c_600s_12` | frozen_fit | 7.43 µm | 1.00 / **0.149** | 0.658 / **0.018** |
| `drift16c_600s_12` | **hold** | **1.12 µm** | **1.00 / 0.908** | **0.933 / 0.622** |
| `drift16c_600s_12` | ema_0.5 | 1.23 µm | 1.00 / 0.966 | 0.916 / 0.664 |
| `drift16c_600s_12` | velocity | 2.42 µm | 1.00 / 0.790 | 0.882 / 0.538 |

The two recordings agree: Frozen leaves the large movers; a 5 s GT hold
stays inside one row on ≥91 % of blocks even for the worst unit. Velocity is
worse than hold on both.

### Per-unit, Frozen vs Hold (in-row / channel match)

Same unit IDs appear in both hybrids (shared GT catalogue). The movers are
the same three: **u5, u11, u51**.

| unit | 11 Frozen | 11 Hold \|err\| / in-row / ch | 12 Frozen | 12 Hold \|err\| / in-row / ch |
|---|---|---|---|---|
| u4 | 4.4 µm / 1.00 / 0.99 | 0.26 / 1.00 / 0.98 | 3.7 / 1.00 / 1.00 | 0.17 / 1.00 / 1.00 |
| u5 | **41.8 / 0.21 / 0.10** | 4.81 / **0.95 / 0.69** | **39.1 / 0.24 / 0.11** | 5.46 / **0.91 / 0.62** |
| u11 | **34.9 / 0.36 / 0.31** | 1.79 / 0.99 / 0.89 | **28.7 / 0.40 / 0.28** | 2.20 / 0.99 / 0.88 |
| u20 | 3.9 / 1.00 / 0.71 | 0.45 / 1.00 / 0.97 | 3.7 / 1.00 / 0.75 | 0.35 / 1.00 / 0.98 |
| u21 | 6.3 / 0.87 / 0.31 | 2.00 / 1.00 / 0.86 | 10.1 / 0.90 / 0.28 | 1.12 / 1.00 / 0.84 |
| u25 | 4.8 / 1.00 / 0.66 | 0.92 / 1.00 / 0.88 | 5.2 / 1.00 / 0.66 | 0.94 / 1.00 / 0.93 |
| u43 | 3.3 / 1.00 / 0.49 | 1.40 / 1.00 / 0.85 | 2.9 / 1.00 / 0.49 | 1.17 / 1.00 / 0.87 |
| u51 | **37.9 / 0.09 / 0.02** | 0.04 / 0.99 / 0.94 | **43.3 / 0.15 / 0.02** | 0.01 / 0.98 / 0.96 |
| u62 | 7.4 / 1.00 / 0.88 | 1.86 / 1.00 / 0.80 | 7.4 / 1.00 / 0.78 | 1.77 / 1.00 / 0.80 |
| u63 | 6.4 / 1.00 / 0.68 | 1.59 / 1.00 / 0.90 | 7.5 / 1.00 / 0.68 | 1.41 / 1.00 / 0.95 |
| u68 | 3.6 / 1.00 / 0.76 | 0.52 / 1.00 / 0.97 | 3.8 / 1.00 / 0.76 | 0.47 / 1.00 / 0.93 |

u5 is the remaining hard channel-match case on both recordings (Hold ~0.62–0.69).
That unit’s COM readout is already unstable on the static control (window-budget
excursion 27 µm). Treat it as readout, not a failed 5 s hold.

---

## 4. What this says

1. **On both drift recordings, 600 s COM is slow enough for a 5 s GT hold.**
   Median lag error 1.1–1.4 µm vs a 20 µm row; worst in-row occupancy 0.91–0.95.
   Frozen cannot hold the movers (u51 in-row 0.09 / 0.15). The two scenes agree.
2. **“In range” for gathering is one row, not channel match.** Channel match
   of 0.89 under hold is mostly COM sitting between two staggered sites, not
   a 20 µm miss.
3. **Do not add velocity.** It helps nobody on drift and hurts static.
4. **This is not yet an online COM tracker.** Hold here sees **GT** COM of
   the last block. The next question — still COM-only, still no POSNEG — is
   whether a **self-assigned** COM (events classified by the current frame)
   stays in the same range, or whether wrong ownership pulls the estimate
   off the row. Static must remain Frozen-quiet under that test.

## 4b. Online streaming: accumulate a few events, then commit

Same two drift recordings, still **no classification**. Events are processed
in time order. For each spike the tracker **freezes** \(\widehat{\mathrm{COM}}\),
scores that spike’s GT COM against it, then pushes the COM into a buffer.
The state updates only when the buffer is full (every \(N\) events) or when
\(\Delta t\) since the last commit reaches a threshold (every \(\tau\) seconds).
Commit value = median of the buffer.

This is what “allow accumulating a few, then update once” means: not a 5 s
clock by obligation, and **not** an update on every spike.

| scene | arm | median \|err\| | worst unit in 20 µm | notes |
|---|---|---:|---:|---|
| 11 | stream_frozen | 6.8 µm | **0.22** (u51) | no online update |
| 11 | stream_n1 (every spike) | 3.6 µm | 0.67 | follows single-spike jitter; p90 ~25 µm |
| 11 | **stream_n10** | **2.5 µm** | **0.77** | ~0.2–9 s of data depending on rate |
| 11 | stream_n20 | 2.6 µm | 0.78 | |
| 11 | stream_dt1s | 3.1 µm | 0.73 | wall-clock 1 s |
| 11 | stream_dt5s | 2.6 µm | 0.77 | same order as the old 5 s hold |
| 11 | stream_ema 0.05 | 3.0 µm | 0.77 | per-spike but very slow |
| 12 | stream_frozen | 7.3 µm | **0.26** | |
| 12 | stream_n1 | 3.7 µm | 0.67 | |
| 12 | **stream_n10** | **2.6 µm** | **0.75** | |
| 12 | stream_dt1s | 3.0 µm | 0.74 | |
| 12 | stream_ema 0.05 | 3.1 µm | 0.79 | |

Movers (u5 / u11 / u51) on both scenes: Frozen in-row 0.22–0.40; n10 / 1 s
commit 0.73–0.90. u5 remains the noisy readout (channel match ~0.47).

**Every-spike update (n=1) is worse than batching 5–20 events**, because
single-spike COM jitters by more than the 5 s drift. Accumulating ~10 spikes
or ~1–5 s is enough; faster is not more accurate.

Still GT-unit COM (the tracker is told which spikes belong to the unit).
Self-assigned streaming COM is not in this table.

Until a self-assigned COM test is run, the licensed statement is:

> On both 600 s drift recordings, a causal streaming COM that commits every
> ~10 events (or every 1–5 s) stays inside a 20 µm row for ~75–93 % of
> spikes, including the movers. Frozen fit COM does not. Updating every
> spike is unnecessary and noisier. The oracle ceiling on that same 20 µm
> test is ~93 % of spikes (see §4c): about 7 % already sit outside the
> contemporaneous true unit COM, so a perfect tracker cannot do better.

---

## 4c. Oracle scatter: how many spikes already leave the true COM range

This is the **upper bound**, not a tracker. Each GT spike is compared to the
**contemporaneous** COM of its own unit — the 5 s block median of that unit's
events (`oracle_block`). `oracle_loo` leaves the scored spike out of the
median so an outlier cannot shrink its own residual. No prediction, no
classification: if the tracker knew the true unit COM at that instant, this
is still the fraction of spikes whose own COM already sits outside the 20 µm
row (or names a different nearest channel).

Pooled over all GT spikes that fall in a valid block (≥3 events):

| scene | arm | n spikes | out of 20 µm | channel mismatch | worst unit out of 20 µm |
|---|---|---:|---|---|---|
| `drift16c_600s_11` | **oracle_block** | 33 602 | **2 328 (6.93 %)** | 8 748 (26.0 %) | u5 19.5 % |
| `drift16c_600s_11` | oracle_loo | 33 602 | 2 451 (7.29 %) | 9 265 (27.6 %) | u5 24.0 % |
| `drift16c_600s_12` | **oracle_block** | 33 660 | **2 347 (6.97 %)** | 8 703 (25.9 %) | u5 18.4 % |
| `drift16c_600s_12` | oracle_loo | 33 660 | 2 510 (7.46 %) | 9 062 (26.9 %) | u5 24.0 % |

The two recordings agree. Leave-one-out is only ~0.4 pp stricter. The
operational ceiling is therefore:

> **~7 % of GT spikes are already outside a 20 µm row of the contemporaneous
> true unit COM.** A COM-range gate cannot keep more than ~93 % of spikes
> "in range", even with a perfect tracker.

Channel mismatch at ~26 % is **not** that 26 % have left the unit. COM sitting
between two staggered sites flips the integer channel while remaining well
inside one row. Use the 20 µm test as the range bound; channel match is
stricter than the physics of gathering.

Per-unit `oracle_block` (out of 20 µm / n spikes):

| unit | scene 11 | scene 12 |
|---|---|---|
| u4 | 7.2 % (413 / 5728) | 7.2 % (403 / 5621) |
| u5 | **19.5 % (233 / 1196)** | **18.4 % (218 / 1186)** |
| u11 | 14.0 % (308 / 2195) | 14.7 % (336 / 2290) |
| u20 | 1.0 % | 1.0 % |
| u21 | 11.7 % (550 / 4705) | 12.0 % (577 / 4825) |
| u25 | 2.5 % | 2.8 % |
| u43 | 7.1 % | 7.1 % |
| u51 | 10.3 % (549 / 5321) | 10.2 % (534 / 5234) |
| u62 | 1.5 % | 1.6 % |
| u63 | 1.4 % | 0.9 % |
| u68 | 0.3 % | 0.2 % |

Quiet units (u20, u25, u62, u63, u68) have almost no scatter. The 7 % pooled
rate is carried by u5 / u11 / u21 / u51 — the same noisy or moving units as
the window budget. u5's 18–20 % is readout, not missed drift: that unit's
static-control excursion is already 27 µm.

What this does **not** say: it is not a classification ceiling, and it does
not bound POSNEG L2. It only bounds the claim "this spike's COM is still in
the unit's current COM range." Streaming n=10 already sits at median in-row
~0.925 against an oracle median of 0.929; the remaining gap is the worst unit
(oracle 0.80–0.82 vs stream 0.75–0.77), not a missing 20 pp of trackable
spikes.

---

## 5. What stays out of this knife

- POSNEG / L2 identity, EMA* of \(\mu\), free shift, QDA.
- Promoting hold to a deployable tracker (it is GT-lag).
- Claiming Q5 field structure from COM signs (edge truncation still applies).

Outputs: `Spatial/output/com_tracking/`.
