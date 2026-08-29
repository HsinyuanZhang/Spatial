# Two drift regimes: COM is the state, and the home channel must be tracked

Status: **architectural reading of the 600 s window-budget measurement.**
Nothing is selected. No tracker is promoted. Date: 2026-08-28.

Evidence: [`drift_window_budget.md`](drift_window_budget.md) (W1–W3),
plot `output/drift_window_budget/plots/com_drift_full_vs_window.png`.
Identity freeze from the 60 s knives remains in force
([`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md),
[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md)).
This note does **not** reopen those identity conclusions. It changes **where
they may be applied as drift claims**.

---

## 1. The 60 s cut and the 600 s recording are different problems

Every tracker number in this repository (Frozen vs D3/D4, EMA*, Oracle-EMA,
\(C(d)=0.153\), \(L_{\mathrm{flex}}\)) was decided on the **first 60 s** of
`drift16c_600s_11`, with causal evaluation on the last 25 % of that cut —
**15 s**. The recording itself is 600 s.

Instrument: amplitude-weighted **COM of the P2P footprint** along the primary
axis, GT spike times, median per 5 s block. The static scene
`static16c_600s_11` is the noise floor through the same path. COM is used
because it is **invariant to a global gain** (unlike `estimate_best_shift`)
and **unbounded** (unlike the \(\pm0.5\) pitch grid).

| Window | Drift median \(\lvert\mathrm{net}\rvert\) | Static control | Drift / noise |
|---|---:|---:|---:|
| full **600 s** | **18.39 µm (0.735 pitch)** | 0.33 µm | **55.9** |
| used 60 s | 4.16 µm (0.166 pitch) | 1.46 µm | 2.85 |
| **test 15 s** | **1.01 µm (0.040 pitch)** | **1.57 µm** | **0.64** |

In the window that decided Q6, the drift scene is **quieter than the no-drift
control**. There is nothing to track. On the full recording the same units
move a median **0.82 pitch excursion** (out and back, not a ramp): away over
~100–250 s, return near t≈300 s, then repeat. The COM plot makes this
unmissable — the orange/red 60 s band sits in a nearly flat prefix, while
the rest of the 600 s is tens of micrometres of structured motion.

So:

- **60 s conclusions about identity, L2, POSNEG vs P2P, free-shift
  \(L_{\mathrm{flex}}\), EMA collapse on MEArec, and “trackers stay quiet on
  static” still stand.** Those do not require drift to be present.
- **60 s conclusions about natural-HJ tracking do not.** Frozen tying D3/D4
  is *untested*, not a tie. EMA\*’s +1.07 pp on HJ drift is not tracking; the
  available reading is a stale-template offset from fitting \(\mu\) on a
  moving first 30 s.

A 600 s rerun is a **new experiment**, not a longer version of the old one.

---

## 2. COM change is the state observable

On 600 s the thing that actually moves is the **spatial centre of the
footprint**, not a latent \(\delta\) inside a \(K\times K\) interpolation.

| Why COM, not `estimate_best_shift` | Consequence |
|---|---|
| Invariant to global amplitude | Does not convert under-amplitude events into false motion ([`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md) E1) |
| Unbounded | Can report 2.7 pitch of travel; the deployed grid cannot |
| Model-free | Not the same operator the tracker later applies (no inverse crime with \(S(\delta)\)) |
| Complementary to home | Edge units (u11, u21, u51) **pin their peak channel** while COM still travels — home undercounts, COM overcounts via truncation |

Tracked state should be written against a **physical coordinate** (COM along
the probe axis, in µm or in electrode rows of 20 µm), not against a
closed interpolation of the fit-home slots.

The 600 s motion is **not monotone**. A trend net cancels on the out-and-back
trip (u5: trend 27 µm, excursion 59 µm). Any 600 s tracker must be an
**integrator** (or an excursion-capable readout), not an absolute offset
against a static template. Units also move in **opposing directions** at the
same time in the COM plot — that is not yet a Q5 field measurement (edge
truncation biases sign), but it already forbids “one global \(\delta(t)\) is
enough” as a default.

---

## 3. Channel selection has to be tracked state

This is the binding 600 s defect. It is **not** a wider \(\delta\) grid.

Every current template is a vector indexed by slot, and the slot-to-channel
map is chosen **once**:

```
event_peak_channel → majority_home on the fit split → table[fit_home]
gather always uses table[fit_home]
CausalTracker.unit_homes is written in __init__ and never updated
  (Frozen, EMA, global shift, tile shift — none of them write the home)
```

The shift bank is a closed map \((n_\delta, K, K)\): \(S(\delta)\) remixes
the **same** \(K\) channels. No electrode outside `table[fit_home]` can
enter, at any \(\delta\).

Geometry makes this worse than it looks in “pitch” units:

| Quantity | Value |
|---|---:|
| Row step along the drift axis | **20.0 µm** |
| Pipeline `pitch_um` (staggered diagonal) | 25.01 µm |
| Deployed grid \(\pm0.5\) pitch | **±0.63 rows** |
| \(\delta\) to the adjacent row | 0.80 (not a grid point) |
| Median 600 s excursion | 20.55 µm = **1.03 rows** |

One row of travel is already **outside** the representable \(\delta\). Every
nonzero grid point is off-lattice IDW, which **attenuates** the template
(the same contraction that turns under-amplitude events into false \(\delta\)).

### Measured home staleness (600 s, 5 s blocks)

`fit_home` = mode of the per-event peak channel on the first 30 s (the
pipeline’s own fit). A block “migrates” when **that block’s majority home**
differs.

| unit | fit home | blocks off that home | distinct homes | static control |
|---|---:|---:|---:|---|
| u63 | 7 | **48 / 120** | 3 | **0 / 120** |
| u68 | 9 | 47 / 120 | 3 | 11 / 120 |
| u20 | 12 | **41 / 120** | 2 | **0 / 120** |
| u43 | 7 | 34 / 120 | 2 | **0 / 120** |
| u25 | 4 | 24 / 120 | 2 | **0 / 120** |
| u62 | 6 | 24 / 110 | 2 | 1 / 110 |
| u5 | 14 | 17 / 120 | 2 | **0 / 120** |
| u4 | 7 | 10 / 120 | 2 | — |
| u11, u21, u51 | 3, 14, 15 | 0 / 120 | 1 | 0 / 120 |

**8 of 11 units leave the fit home.** Six of those do it against a static
control of zero or one block, so this is drift, not readout chatter. The
three that never migrate sit at the **array ends**: the peak channel cannot
move outward, while COM still shows 1.3–2.7 pitch of travel. Home migration
and COM excursion together cover every unit.

EMA cannot fix this. EMA rewrites the **values** of \(\mu\) on the frozen
channel set; it never changes which channels those values belong to. After
a one-row migration the template is a 20 µm-misaligned vector, which is
**1.6× the largest representable \(\delta\)**.

Widening the grid only fixes range. It does not realign slots, and it does
not let a new channel enter the \(K\)-window.

---

## 4. What the next identity/state split actually is

Keep the 60 s identity object:

```
absolute POSNEG, candidate-home physical offsets, L2 / biased-dot, K=12
```

Change the **frame** from a frozen fit-home to a **tracked physical frame**:

```
state   θ = (COM along the probe, integer home / row)
        historical, predicted, then frozen for the current event

query   gather x on table[home(t)], not table[fit_home]
        optional δ inside that row as a residual, on-lattice

identity μ_u  stays put in the *current* physical frame
        EMA of μ is not a substitute for moving the frame
```

Rules carried forward:

- Predict \(\theta\) from history → freeze → classify → update only if
  confident. No per-candidate free shift/gain on the current event
  (\(L_{\mathrm{flex}}\) still applies).
- No population coherence → do not invent a unit-free \(\delta_u\). COM
  trajectories already show opposing signs; **tile or unit-home integer**
  is the candidate, not one global \(\delta(t)\), and not a free per-query
  search.
- Static recordings must stay quiet: home must not chatter
  (static control is already 0/120 for the clean migrators).
- \(\eta^*=0.005\) EMA remains the **value-update** baseline, not the
  frame-update baseline. A tracked-home method has to beat Frozen **and**
  EMA* on **600 s**, not on the empty 15 s window.

---

## 5. Smallest next measurement (before building a tracker)

W3 shows the frozen home is wrong for up to 40 % of blocks. It does **not**
show how many assignments that costs.

COM tracking is isolated first
([`com_tracking_range.md`](com_tracking_range.md)): on 600 s, a **5 s hold of
the last GT COM** stays inside one electrode row (20 µm) for 95–100 % of
blocks; frozen fit COM does not, for the movers. That licenses COM as a
trackable state **before** identity re-gather. It does not yet test
self-assigned COM.

Cheapest **identity-frame** arm after that, 600 s, GT events, candidate-home
POSNEG L2, \(K=12\):

| Arm | Frame | Role |
|---|---|---|
| Frozen fit-home | `table[majority_home(fit)]` for the whole recording | baseline |
| Oracle block-home | re-gather on each 5 s block’s **true majority home** | DIAGNOSTIC upper bound on W3 |
| Optional: COM-quantized home | home = nearest electrode to a causal COM predictor | first deployable state |

If Frozen ≈ oracle-home on this 16-channel probe (\(K=12\) already spans
~100 µm of 140 µm), W3 is mostly latent here and the 600 s knife is
**on-lattice \(\delta\) + integrator**, not home tracking. If oracle-home
opens a real same-home / worst-unit gap, **channel selection is the next
state variable**, and widening \(\delta\) is secondary.

Do not rerun Q6 at 600 s with the old closed \(K\to K\) bank. That experiment
cannot express the motion it would be asked to track.

---

## 6. What this does not license

- It does not say a tracker wins at 600 s. It says the old experiment could
  not have seen a win or a loss.
- It does not reopen cosine, free shift, QDA, or \(K\) sweeps.
- It does not treat COM WTA as a unique-ID classifier (that already failed on
  dense MEA). COM here is **state**, not identity.
- It does not skip the static-quiet and MEArec-collapse gates. Those still
  kill EMA-style identity corruption.
