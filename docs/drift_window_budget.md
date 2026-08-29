# The 60 s window contains no measurable drift, and the representation cannot express the drift that exists — results

Status: **the natural-drift half of Q6 is reclassified from "tested, tied" to
"not testable on this window", and the deployed drift representation is shown
to be structurally unable to express the motion in the full recording.**
Nothing is selected and no tracker is promoted. Date: 2026-08-28.

Companion protocol/provenance:
[`drift_window_budget_provenance.md`](drift_window_budget_provenance.md).
Documents amended by this record:
[`drift_tracking_results.md`](drift_tracking_results.md) (Q6 scope),
[`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md) §8.1–8.2 (Q5, Q6),
[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md) (what
the HJ-drift gain is attributed to), and
[`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md) §3 (its
"Q6 is not overturned" paragraph). **No number in those documents is
rewritten**; the amendments are scope and attribution changes.

This is a **measurement of the input data**, not of an algorithm. No file under
`algorithms/` was modified. W3 reads two deployed functions
(`event_peak_channel`, and the geometry) to reproduce the pipeline's own home
selection; it does not run a tracker.

---

## 1. What was asked

Every drift result in this repository is measured on the **first 60 s** of a
600 s Hybrid Janelia recording, split chronologically 50 / 25 / 25, with the
causal tracker evaluated on the final 25 % — a **15 s test window**.

| ID | Question | Prediction |
|---|---|---|
| W1 | How much spatial motion does the 600 s recording contain, and what fraction of it survives inside the 60 s window and the 15 s test window? | The used window is a small fraction; the test window is likely at or below the measurement noise floor |
| W2 | Is the deployed shift grid \(\{-0.5,\dots,0.5\}\) pitch wide enough for the motion that is actually present? | Unknown before the run |
| W3 | Every template's channel selection is fixed by a frozen per-unit **home channel**. Does that home stay valid over the recording? | If W1 finds motion of order one electrode row, no |

W1 matters because a tie between Frozen and a drift tracker has two
incompatible readings — "tracking does not help" and "there was nothing to
track" — and the existing experiments cannot tell them apart.

W3 was added after W1 and W2 returned. It asks whether the defect is a
parameter range at all, or whether the representation itself cannot express
drift.

---

## 2. Instrument and why it is the right one

Position is read out as the **amplitude-weighted centre of mass (COM) of the
P2P footprint along the primary probe axis**, at GT spike times, taken as a
median per 5 s block.

COM is used here **instead of** `estimate_best_shift` for three reasons:

1. **It is invariant to a global amplitude scale by construction.** A weighted
   mean of positions does not change when all weights are multiplied by \(g\).
   [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md) showed
   that the deployed estimator converts amplitude error into false motion, so
   using it here would measure its own artefact.
2. **It is unbounded.** The deployed estimator saturates at \(\pm0.5\) pitch and
   therefore cannot report the quantity W2 asks about.
3. **It is model-free.** No template, no fit split, no interpolation operator —
   so this measurement is not exposed to the inverse-crime problem of the
   controlled-drift runs.

Channels below 0.3x the event peak are dropped so the noise floor on distant
sites does not pull the centre toward the probe middle. The scale invariance
and the floor behaviour are pinned by
`tests/test_drift_window_budget.py` (8 tests).

**The static recording is the control.** `static16c_600s_11` has no drift, so
whatever motion this instrument reports there is its noise floor, measured
through the identical path — same units, same firing rates, same block design.
Every drift number below is read against it.

### Protocol

| Field | Value |
|---|---|
| Recordings | HJ `drift16c_600s_11` (signal), HJ `static16c_600s_11` (control) |
| Duration | **600 s — the full recording**, not the locked 60 s cut |
| Truth boundary | GT event times, GT unit rows. No assignment, no classification |
| Signal path | full-recording zero-phase Butterworth 300–6000 Hz, order 3 |
| Observation | analog POSNEG extrema in \(\pm15\) samples, all 16 channels |
| Block | 5 s, median COM per unit-block, blocks with <3 events dropped |
| Units | 11 valid GT units (drift), 10 (static); \(\ge50\) spikes required |
| Statistic | **trend net** and **excursion**, defined below |
| Windows | full 600 s / used 60 s / test 15 s (t ∈ [45, 60) s) |
| Aggregation | median over units; scenes never pooled |
| Decision | **none.** No configuration selected, no test split of any prior experiment consulted |

Probe pitch is 25.01 µm.

### Two statistics, because the drift is not monotone

The HJ drift profile is an **out-and-back excursion**, not a ramp: units travel
away from their starting position over ~100–250 s and return toward it near
t≈300 s, then repeat. This is visible directly in
`output/drift_window_budget/plots/com_drift_full_vs_window.png`, which plots
every unit's COM trajectory over the full 600 s with the used 60 s and the 15 s
test window shaded. Two statistics are therefore reported and they answer
different questions.

| Statistic | Definition | Answers | Noise floor at 600 s (static) |
|---|---|---|---:|
| **trend net** | slope of a least-squares line over the segment, times the segment span | *Is there drift at all?* | **0.33 µm** |
| **excursion** | max − min of the block trace after a centred 25 s rolling median | *How far does a unit get from a fixed template?* | 4.47 µm |

The trend has a very low noise floor because the line fit averages over all
blocks, which makes it the sensitive detector. But it **cancels on an
out-and-back trip** and therefore understates the operationally relevant
distance — on u5 the trend reads 27 µm while the excursion is 59 µm. The
excursion measures that distance directly, at the cost of a ~13x higher noise
floor, and is smoothed over 25 s so a single noisy block cannot create one.

Both are quoted against the static control throughout. Over a short window a
noisy slope is still extrapolated across the window, which **inflates** short
window trend values; the control absorbs exactly this artefact. Excursion needs
\(\ge5\) blocks and is therefore undefined on the 15 s window (3 blocks).

---

## 3. W1 — the motion budget

### Headline — trend net (the sensitive detector)

| window | drift scene, median \(\lvert\text{net}\rvert\) | static control (noise floor) | ratio |
|---|---:|---:|---:|
| full 600 s | **18.39 µm** (0.735 pitch) | 0.33 µm | **55.9** |
| used 60 s | 4.16 µm (0.166 pitch) | 1.46 µm | 2.85 |
| **test 15 s** | **1.01 µm** (0.040 pitch) | **1.57 µm** | **0.64** |

**In the 15 s window on which every causal-tracker verdict was decided, the
drift recording reports less apparent motion than the no-drift control.** The
ratio is below 1. There is no drift signal above the measurement noise there.

### Excursion (how far a unit actually gets)

| window | drift scene, median | max | static control | ratio |
|---|---:|---:|---:|---:|
| full 600 s | **20.55 µm** (0.822 pitch) | 66.97 µm | 4.47 µm | 4.6 |
| used 60 s | 4.84 µm (0.194 pitch) | 34.54 µm | 2.38 µm | 2.0 |
| test 15 s | — (needs \(\ge5\) blocks) | — | — | — |

The two statistics agree on the 600 s recording: 0.735 pitch by trend, 0.822
pitch by excursion. The excursion is the larger of the two, as expected for a
profile that doubles back.

For the 15 s window, where excursion is undefined, the third statistic points
the same way: block spread (p90–p10) is **1.57 µm on the drift scene against
3.75 µm on the static control** — again below 1. All computable statistics
agree that the test window carries no drift signal.

### Consistency check against a linear-drift expectation

If drift were uniform in time, a 60 s window would hold 10 % of the total
(1.84 µm) and a 15 s window 2.5 % (0.46 µm). Measured values are 4.16 µm and
1.01 µm — **above** the linear expectation, which is the predicted upward bias
of a trend fit over few blocks, and confirms the short-window readings are
noise-dominated rather than signal. Against the 15 s expectation of 0.46 µm,
the control's noise floor of 1.57 µm is **3.4x larger than the signal being
looked for**.

The "22.6 % of full drift is inside the used 60 s" figure in the summary CSV is
therefore an **upper bound**, not an estimate.

### Per-unit, full 600 s

Sorted by excursion. The static control column is the **same unit** measured
the same way on the no-drift recording, so it is that unit's own noise floor.

| unit | rate | trend net | excursion | pitch | static control excursion |
|---|---:|---:|---:|---:|---:|
| u51 | 8.87 Hz | +88.84 µm | 66.97 µm | **2.68** | 23.10 µm |
| u5 | 1.99 Hz | −27.22 µm | 58.69 µm | **2.35** | 27.26 µm |
| u11 | 3.66 Hz | −19.14 µm | 44.44 µm | **1.78** | 4.66 µm |
| u21 | 7.84 Hz | +28.08 µm | 33.42 µm | **1.34** | 4.28 µm |
| u62 | 1.11 Hz | −18.39 µm | 21.03 µm | 0.84 | 5.45 µm |
| u63 | 1.72 Hz | −19.02 µm | 20.55 µm | 0.82 | 3.55 µm |
| u25 | 3.10 Hz | −14.79 µm | 15.34 µm | 0.61 | 2.72 µm |
| u43 | 3.68 Hz | −11.96 µm | 13.58 µm | 0.54 | 5.92 µm |
| u68 | 9.18 Hz | −10.34 µm | 12.74 µm | 0.51 | 0.99 µm |
| u20 | 5.34 Hz | −10.26 µm | 11.38 µm | 0.46 | 1.76 µm |
| u4 | 9.55 Hz | +10.69 µm | 9.09 µm | 0.36 | — |

u51 and u5 have **noisy controls** (23.1 and 27.3 µm): their COM readout is
intrinsically unstable, so their large drift-scene excursions are inflated and
should not be quoted alone. u11 and u21 are the clean large movers — 44.4 and
33.4 µm against controls of 4.7 and 4.3 µm.

### Illustration of the noise floor

Static `u5` reports **19.71 µm** of trend "motion" inside a 15 s window. That
unit does not move at all, and the value is 20x the entire real drift measured
in the same window of the drift scene (1.01 µm). Per-unit short-window
estimates are not usable; only the population statistic against the control is.

---

## 4. W2 — the deployed shift grid is narrower than the motion

`SHIFT_FRACTIONAL = (-0.5, -0.25, 0.0, 0.25, 0.5)` and `estimate_best_shift`
returns an **absolute** offset against a static template, not an integrated
one, so the representable range is \(\pm0.5\) pitch with hard saturation.

Excursion is the statistic that answers this, since it is exactly the span a
fixed reference has to cover. A unit whose excursion exceeds **1.0 pitch**
cannot be covered by *any* choice of reference point; one whose excursion
exceeds **0.5 pitch** is not covered by a reference at one end of its travel.

| Condition | Units out of 11 | Which |
|---|---:|---|
| excursion > 1.0 pitch — uncoverable by any reference | **4** | u51, u5, u11, u21 |
| excursion > 0.5 pitch | **9** | + u62, u63, u25, u43, u68 |
| restricting to units whose static control is clean (<6 µm) | 2 and 7 | u11, u21 uncoverable |

The population median excursion is **0.822 pitch**, already outside
\(\pm0.5\). The largest clean mover, u11, travels 1.78 pitch — 3.6x the grid
half-width — against a control of 0.19 pitch.

This is a **range** limitation, not a resolution one, and it is invisible at
60 s because no unit leaves the grid inside that window (median excursion
there is 0.194 pitch). It is independent of the quantization and
gain-profiling defects reported in
[`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md): fixing
those does not widen the grid.

The non-monotone shape compounds this. `estimate_best_shift` reports an
absolute offset against a static template rather than integrating increments,
so a unit that leaves the grid and returns is not merely mis-estimated while
away — it saturates at the boundary and gives no indication that it has done
so.

Units also move in opposing directions at the same time in the trajectory plot.
That is **not** offered as evidence about the drift field's structure: with an
oscillatory profile the sign of a trend depends on which phase a window catches,
and the COM readout is biased by probe-edge footprint truncation in a
direction that depends on where a unit sits on the array. Q5 needs its own
measurement — see §5.

---

## 5. W3 — the channel selection is frozen, and it does go stale

### The mechanism

Every template is a vector indexed by **slot**, and the slot-to-channel map is
chosen once from a single integer per unit:

```
event_peak_channel(vmax, vmin)      per-event argmax channel
majority_home(homes, labels, units) per-unit home = mode over the fit split
geometry_k_table(geom, K)           table[home] = the K channel ids for that home
gather_unit_posneg(..., homes)      x is always gathered on table[fit_home]
```

`CausalTracker.unit_homes` is assigned once in `__init__` and is only ever
read afterwards. **No arm writes to it, including EMA** — EMA rewrites the
values of `mu_pos` / `mu_neg`, never the channel set they are indexed on.

So \(\delta\) is continuous and estimated per block, while the home is an
integer produced by an `argmax` that is not part of any tracked state.

The shift bank cannot compensate for this. `precompute_shift_mats` returns
matrices of shape \((n_\delta, K, K)\): the shift is a **closed map from the K
selected slots onto the same K slots**. No channel outside `table[fit_home]`
can enter, at any \(\delta\).

### The geometry makes the grid narrower than a single row

| Quantity | Value |
|---|---:|
| Electrode row spacing along the primary (drift) axis | **20.0 µm** |
| `pitch_um` carried by the pipeline (`median_nearest_neighbor_spacing`) | 25.01 µm |
| \(\pm0.5\) pitch, in micrometres | ±12.5 µm |
| \(\pm0.5\) pitch, **in rows** | **±0.63** |
| \(\delta\) needed to reach the adjacent row | 0.80 |

The probe is a staggered two-column 16-channel siprobe
(\(x\in\{11,27,43,57\}\), \(y\in\{80,100,\dots,220\}\)). Its nearest neighbour
is the **diagonal** site in the other column, so
`median_nearest_neighbor_spacing` returns 25.01 µm while the step a unit must
travel to change rows is 20 µm. `probe_meta` carries the former as `pitch_um`,
and every \(\delta\) in the repository is in those units.

Two consequences, both structural:

1. **One row of drift is already outside the grid.** The maximum representable
   displacement is 0.63 rows.
2. **The grid never lands on the channel lattice.** In row units the grid is
   \(\{-0.63,-0.31,0,0.31,0.63\}\); reaching a neighbouring row would need
   \(\delta=0.8\), which is not a grid point. Every non-zero \(\delta\) is
   therefore an off-lattice IDW interpolation, and off-lattice IDW is a
   contraction that attenuates the template — the same mechanism behind the
   amplitude bias in
   [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md).

This does **not** invalidate §4: the excursions and the grid are both quoted in
the pipeline's own `pitch_um`, so that comparison is internally consistent. In
row units the median excursion is 20.55/20.0 = **1.03 rows**, which is the same
statement in more physical units.

### Measured: 8 of 11 units outlive their home channel

`fit_home` reproduces `majority_home` on the pipeline's fit split (mode of the
per-event peak channel over the first 30 s). A block is counted when **its own
majority home** differs — a stray minority of events does not count.

| unit | fit home | blocks with a different majority home | distinct homes visited | static control |
|---|---:|---:|---:|---:|
| u63 | 7 | **48 / 120** | 3 | **0 / 120** |
| u68 | 9 | 47 / 120 | 3 | 11 / 120 |
| u20 | 12 | **41 / 120** | 2 | **0 / 120** |
| u43 | 7 | **34 / 120** | 2 | **0 / 120** |
| u25 | 4 | **24 / 120** | 2 | **0 / 120** |
| u62 | 6 | 24 / 110 | 2 | 1 / 110 |
| u5 | 14 | **17 / 120** | 2 | **0 / 120** |
| u4 | 7 | **10 / 120** | 2 | — |
| u11, u21, u51 | 3, 14, 15 | 0 / 120 | 1 | 0 / 120 |

Eight units migrate. Broken down by what their own static control says:

| Static control | Units | Reading |
|---|---|---|
| exactly 0 / 120 | u5, u20, u25, u43, u63 | **drift, unambiguously** |
| 1 / 110 | u62 | drift |
| 11 / 120 | u68 | straddles two channels intrinsically; only the 4x increase is drift |
| no control | u4 | falls below the 50-spike floor on the static recording |

So **six of the eight migrate against a control that is zero or one block**,
which is what rules out readout instability as the explanation.

### The three units that do not migrate are the edge units

u51, u21, and u11 sit at homes 15, 14, and 3 — the top and bottom ends of the
array (\(y=220\) and \(y=100\)). They are also the units with the **largest COM
excursions** in §3 (2.68, 1.34, and 1.78 pitch).

That is not a contradiction, it is the boundary: a unit at the array end cannot
move its peak channel outward, so **the home pins while the footprint keeps
moving**. Home migration therefore *undercounts* drift for edge units, and COM
excursion *overcounts* it for them through footprint truncation. The two
readouts are complementary, and between them every one of the 11 units shows
one symptom or the other.

### Why widening the grid is not sufficient

The three defects sit at different levels and a wider grid only reaches the
second one:

| Level | Defect | Fixed by a wider grid? |
|---|---|---|
| Slot alignment | \(\mu\) is indexed on `table[fit_home]`. A one-row migration misaligns it by 20 µm, which is **1.6x the largest representable \(\delta\)** | **No** |
| Grid range | \(\pm0.63\) rows against 1.03 rows of median motion | Yes |
| Window closure | `mats` is \((n_\delta,K,K)\); outside channels cannot enter | No — needs a larger \(K\) or a different frame |

There is also no failure signal. `interpolation_matrix` normalises IDW weights
over the live source slots, so a query point pushed past the array still
returns a convex combination of the edge channels. An out-of-range shift
produces an **attenuated, smeared template rather than an error**, which is why
none of this was visible in the tracker metrics.

Expressing the drift that is actually in this recording requires the channel
selection to become part of the tracked state — re-gathering \(x\) on a home
that follows \(\hat\delta\), a larger \(K\), or dropping the home-plus-fixed-slot
parameterisation for one indexed by physical coordinates. That is a different
algorithm, not a parameter change.

On this 16-channel probe \(K=12\) already spans 100 µm of a 140 µm array, so
the window-closure level is mostly latent here. It binds on a dense probe with
a small \(K\) (Yger 252-channel, Neuropixels), where the K-slot window is a
small fraction of the array.

---

## 6. Consequences

### Q6 — "Frozen ties the shared-shift trackers on natural HJ"

The natural-HJ arm of that verdict was decided on a window where the
drift-to-noise ratio is 0.64. **No tracker of any design could have won there,
and Frozen could not have lost.** The result must be read as **untested**, not
as a tie.

This supersedes the paragraph in
[`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md) §3 which
concluded that Q6 "most likely stands" because "the drift is small and the
estimator resolution is comparable to it". The correct statement for the test
window is stronger and simpler: **there is no drift to track in it.**

Two parts of the Q6/Q8 evidence are **not** affected:

- **Trackers stay quiet on static recordings.** This is a safety property and
  does not require drift to be present.
- **Controlled-ramp responses.** These remain valid as estimator-consistency
  checks, subject to their own separate inverse-crime caveat.

### EMA\* — what the HJ-drift gain is attributed to

[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md) reports
\(+1.07\) pp for EMA\* on the HJ drift test split and \(-0.14\) pp on HJ
static. **The number stands; the attribution does not.** There is no ongoing
motion in that test window for an EMA to track.

The available explanation is a **stale-template** effect rather than tracking:
\(\mu\) is fitted on the first 30 s, which does contain motion, so the fitted
mean is a time-average over a moving target and is offset from the test
window's true footprint. An EMA removes a constant offset of that kind. This
would also explain the drift/static asymmetry, since the static scene has no
such offset to remove.

**This is a hypothesis consistent with the data, not a measurement.** It is
separable — a Frozen arm refitted on the last block of the fit split would
capture the offset without any online update — and that separation has not
been run.

### Q5 — drift field structure

Q5 was already reopened by the amplitude-bias diagnostic. This record adds an
independent and more basic reason: the coherence \(C(d)=0.153\) was measured on
1 s blocks inside the same 60 s window. At the recording's rates that is
~5 events per unit-block, and 1 event for u62 (1.11 Hz), against a 0.25-pitch
grid step — on data whose drift-to-noise ratio has now been measured. **A
correlation computed there is a correlation of noise with noise.**

The full recording does contain a large, structured signal (§3, §4), so Q5
becomes answerable for the first time at 600 s.

---

## 7. What this does not show

- It does not measure any tracker. No accuracy, margin, or assignment number is
  produced or changed here.
- It does not show a tracker would win at 600 s. It shows the existing
  experiment could not have detected it either way.
- It does not establish the drift field's spatial structure, and the opposing
  directions noted in §4 are not evidence about it: the COM readout includes
  probe-edge footprint truncation, which is not corrected for and biases units
  near the array ends.
- It does not separate real motion from readout instability for **u51 and u5**,
  whose static controls are 23.1 and 27.3 µm. Their drift-scene excursions are
  quoted but are not load-bearing; u11 and u21 carry the same conclusion with
  clean controls.
- The 25 s smoothing window for excursion was chosen to sit below the ~100 s
  timescale visible in the trajectories and above the block noise. It was not
  swept, and excursion is undefined on the 15 s window as a result.
- It covers **one drift scene and one static control** (`*16c_600s_11`). Those
  are exactly the recordings the amended results were measured on, so no
  extrapolation is needed for the amendments — but the motion budget of the
  other five HJ drift scenes is unmeasured.
- It does not touch MEArec, which has no drift scene in the locked set.
- **W3 does not measure the cost of home staleness.** It shows the frozen home
  is wrong for up to 40 % of the recording; it does not measure how many
  assignments that costs. That requires a tracker run, which is deliberately
  not done here.
- W3's block-majority criterion is a **lower bound** on channel-selection
  error: a block whose majority home still matches can contain a large minority
  of events peaking elsewhere (u20 on the static control has 15 % of events off
  its home with zero migrating blocks).
- The row-step / `pitch_um` distinction is specific to a staggered layout. It
  is reported because this probe is staggered, not as a general claim.
- No test-split number of any prior experiment was consulted for any decision
  here.

---

## 8. Next

1. **Unlock the duration.** The data is already 600 s; the full-length load
   costs ~12 s. The 60 s lock is a convention, not a constraint.
2. **Make the channel selection part of the tracked state.** This is the
   binding constraint from §5 and it is not a parameter change. The cheapest
   version is to re-gather \(x\) on a home that follows \(\hat\delta\) — one
   integer per unit per block, with the same shift bank — which removes the
   slot-misalignment level without touching the identity feature. Before
   spending anything on it, measure what home staleness actually costs by
   running a Frozen arm against an oracle arm that re-gathers on each block's
   true majority home. If the gap is small, W3 is a latent defect on this probe
   and the priority drops.
3. **Widen the shift grid** — necessary but not sufficient. The largest
   measured excursion is 2.68 pitch, so a centred reference needs at least
   \(\pm1.4\) pitch and \(\pm2\) with margin, against \(\pm0.5\) today. Pick
   grid points that land on the channel lattice (multiples of \(\delta=0.8\) on
   this probe), so \(\delta=0\) stops being the only non-attenuated candidate.
4. **Rerun Q6** at 600 s with a gain-profiled, sub-grid estimator, and report a
   paired significance test rather than a point difference.
5. **Rerun Q5** at 600 s with a block length set so the per-unit-block standard
   error is below the drift amplitude under test, precision-weighted by event
   count.
6. **Separate the EMA\* attribution** with a refit-Frozen arm before describing
   EMA as drift tracking anywhere.

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.run_drift_window_budget --pilot --quiet
```

Outputs and digests:
[`drift_window_budget_provenance.md`](drift_window_budget_provenance.md).
