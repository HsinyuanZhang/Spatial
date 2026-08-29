# Drift window budget — provenance

Snapshot for [`drift_window_budget.md`](drift_window_budget.md).
Date: 2026-08-28.

## Repository state

```text
repository revision (HEAD)   c729bffa029724a0740c5573b92f43f3dcaff55f
branch                       shape-orthogonal-descriptor
```

The runner and its test are new untracked working-tree files at this revision
and are identified by digest below.

## Environment

```text
conda env   spint
python      3.10.15
numpy       2.2.6
scipy       1.15.3
platform    linux 6.8.0-124-generic, CPU only
```

## Command

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting
python -m Spatial.experiments.run_drift_window_budget --pilot --quiet
```

Wall time ~12 s per scene pair, peak RSS ~6 GB (the loader materializes the
full 600 s x 16 ch array as float64 before filtering).

`--scenes` defaults to `drift16c_600s_11 static16c_600s_11`. Unlike the other
drift runners this one is **not** duration-locked: the whole point of the
measurement is the full 600 s.

## Source digests (SHA-256)

```text
025b5a3f2957c3e6481638b9f6e04436e23679daacee578de82d0db0bca0c80b  Spatial/experiments/run_drift_window_budget.py
f661150ddfe8748fac0b4eee2d93f01b85599812a7667475f05fe49ac3a0f149  Spatial/algorithms/spatial_footprint.py
bc184f69352e5d0fec1f3bc25a77215805b31f29bb7cdab99410dcd7d067e05b  Spatial/data/loader.py
c15525f10c76dee06c6219b3ce8c7494694a50d94ba683bdd174f6ba9e405401  Spatial/algorithms/deformable_spatial_template.py
```

`event_peak_channel` is imported from `deformable_spatial_template.py` at the
digest above, so W3's per-event home is produced by **the deployed function**,
not a reimplementation.

### Claims read from source rather than executed

`algorithms/drift_tracking.py` and `algorithms/spatial_shift_matching.py` are
**not imported** by this runner. Three statements in §4–§5 of the result
document are read from those files and are recorded here so an auditor can
check them against the digests:

| Claim | Source | Digest |
|---|---|---|
| The shift grid is \((-0.5,-0.25,0,0.25,0.5)\) pitch | `spatial_shift_matching.SHIFT_FRACTIONAL` | `fa670da9…` |
| The shift bank is a closed \(K\to K\) map (`precompute_shift_mats` returns \((n_\delta,K,K)\)) | `spatial_shift_matching.precompute_shift_mats` | `fa670da9…` |
| `unit_homes` is assigned once in `CausalTracker.__init__` (line 319) and only read afterwards (396, 434, 442, 484, 494, 529, 549, 572, 626) — **no arm, including EMA, writes it** | `drift_tracking.py` | `e7415fc3…` |

Both digests are the same ones recorded by the POSNEG/MVM, EMA\*, and
estimator-diagnostics records, so these statements describe the code those
results were produced with.

`interpolation_matrix` normalising IDW weights over live sources — the reason
an out-of-range shift degrades silently rather than failing — is at digest
`c15525f1…` above, lines 144–166.

## Output digests (SHA-256)

```text
75a39e18f923e0505613ec2bd1a95857d4f14a750996973e405cbe6cfee6a18e  Spatial/output/drift_window_budget/drift_window_budget_per_unit.csv
c106c8888f81052fcb58e12fa9fedbee740d8e91d204fb26ffef2f7aed079f0f  Spatial/output/drift_window_budget/drift_window_budget_summary.csv
4301cbee5eb2bb556af7f60b191a57bd503af07bab61dbb64e85247cebd57a4f  Spatial/output/drift_window_budget/drift_window_budget_summary.json
```

The per-unit CSV carries the W3 columns (`home_fit_home`,
`home_n_blocks_home_differs`, `home_n_distinct_block_homes`,
`home_frac_events_off_fit_home`, `row_step_um`); the summary CSV carries the
per-scene rollup including `shift_grid_halfwidth_rows`.

A companion manifest is written by the runner to
`Spatial/output/drift_window_budget/outputs.sha256`. A PNG of the per-unit COM
trajectories over the full 600 s, with the used 60 s and the 15 s test window
shaded, is written to `plots/com_drift_full_vs_window.png` and is not hashed.

### Order of construction

The runner grew in three passes, recorded because each pass changed what the
previous one concluded:

1. **W1/W2, trend statistic only.** Reported 0.735 pitch of median motion.
2. **Excursion added.** The trajectory PNG from pass 1 showed the HJ drift
   profile to be an out-and-back excursion rather than a ramp — a shape on
   which a linear trend cancels and understates the motion. Both statistics are
   reported; neither replaced the other, and the headline test-window
   conclusion is unchanged between them.
3. **W3 added.** Asked whether the defect is a parameter range at all. It is
   not: the channel selection is frozen and goes stale, which a wider grid does
   not fix.

The digests above are from a run that includes all three.

### Determinism

The runner was executed twice after W3 was added. The per-unit CSV reproduced
**bit-identically** (`75a39e18…`). No seeds are consumed: block edges are
fixed, the per-block statistics are a median and a mode, the trend is a
least-squares line fit, and the excursion is a rolling median.

## Contract tests

`Spatial/tests/test_drift_window_budget.py`, 12 tests, all passing:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_drift_window_budget.py -q
```

Two are load-bearing:

- `test_com_is_invariant_to_global_amplitude_scale` — the whole W1 argument
  depends on the readout not inheriting the amplitude-to-motion bias documented
  in [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md); if it
  did, the static control would not be a valid noise floor.
- `test_row_step_is_not_the_nearest_neighbour_pitch_on_a_staggered_probe` —
  the "\(\pm0.5\) pitch is only 0.63 rows" claim in §5 rests entirely on this
  distinction, so it is pinned on the actual HJ geometry rather than asserted.

`test_home_migration_ignores_a_minority_of_stray_events` pins the block-majority
criterion, which is what makes W3 a lower bound rather than a stray-event count.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required in this environment because an
unrelated `dandi` pytest plugin fails to import against the installed `click`.

## Inputs

| scene | source | duration | valid units | GT spikes | channels | pitch |
|---|---|---:|---:|---:|---:|---:|
| `drift16c_600s_11` | HJ `drift_siprobe_16c_600s_11` | 600 s | 11 | 33 617 | 16 | 25.01 µm |
| `static16c_600s_11` | HJ `static_siprobe_16c_600s_11` | 600 s | 10 | — | 16 | 25.01 µm |

Both are the **full-length versions of the locked development recordings**
already used by the POSNEG/MVM, EMA\*, and estimator-diagnostics records, which
consume their first 60 s. Dataset locations follow
[`REPRODUCING.md`](REPRODUCING.md).

The static scene contributes 10 units rather than 11 because one GT unit falls
below the 50-spike inclusion floor of this runner.

## Evidence boundary

| Contract item | This record |
|---|---|
| Truth boundary | GT event times, GT unit rows. **No assignment, no classification, no template fit** — this measures the input recording |
| Split | none for W1/W2. W3 reads the first 30 s to reproduce `majority_home` on the pipeline's fit split, then evaluates the full 600 s; the 60 s and 15 s windows are named sub-intervals, not train/test roles |
| Fit boundary | nothing is fitted. The only estimates are a per-block median, a per-segment line fit, and a per-block mode |
| Signal path | full-recording zero-phase Butterworth 300–6000 Hz order 3. **Not causal** |
| Control | `static16c_600s_11` through the identical path. All drift numbers are quoted against it |
| Metrics | trend net displacement (µm and pitch); excursion (25 s-smoothed max−min); p90–p10 block spread; drift-to-control ratio for each; W3 block-majority home migration counts and the row-step / `pitch_um` ratio |
| Aggregation | median of per-unit \(\lvert\text{net}\rvert\) within a scene; scenes never pooled |
| Latency | not applicable; no online claim |
| Hardware boundary | none; no bit, area, or energy quantity is produced |
| Decision | **no configuration selected.** No accuracy number anywhere in the repository is changed |
| Confirmation | not authorized and not opened |
| Negative evidence | retained: the noise-inflation of short-window trend fits is measured and reported (§3), including the static `u5` 19.71 µm false excursion; W3's three non-migrating units are reported and explained as edge pinning rather than dropped |

## Amendments recorded against other documents

| Document | Section | Amendment |
|---|---|---|
| [`drift_tracking_results.md`](drift_tracking_results.md) | Verdict Q6 | The natural-HJ arm was decided on a 15 s window with drift-to-noise 0.64. Reclassified from "tested, tied" to **untested**. The static-quiet and controlled-ramp arms are unaffected |
| [`drift_tracking_results.md`](drift_tracking_results.md) | Verdict Q7 | "State can move while identity stays put" holds for \(\delta\) and \(\mu\), but the **channel selection is neither state nor tracked**: `unit_homes` is never updated, and 8 of 11 units outlive their fit home over 600 s |
| [`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md) | §8.1 Q5, §8.2 Q6 | Same Q6 reclassification. Q5's \(C(d)=0.153\) additionally rests on 1 s blocks (~5 events per unit-block, 1 for u62) inside a window now measured to carry no drift signal |
| [`ema_star_identity_convergence.md`](ema_star_identity_convergence.md) | HJ drift test result | The \(+1.07\) pp stands as a number; its attribution to drift tracking does not. Stale-template offset is the available explanation and is untested |
| [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md) | §3, "Consequence for Q5" | Its "Q6 is not overturned … the drift is small and the estimator resolution is comparable to it" is superseded: in the test window there is no drift at all |

No number in those documents is rewritten. The amendments are scope and
attribution changes, recorded here and in
[`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md).

## Not covered

- Five of the six HJ drift scenes are unmeasured. The amendments above apply to
  scene 11 only, which is the scene the amended results were measured on.
- No corrected \(C(d)\) and no 600 s tracker comparison were run.
- The opposing per-unit directions noted in §4 of the result document are not
  corrected for probe-edge footprint truncation and are not a Q5 finding.
- `u51` and `u5` have static-control excursions of 23.1 and 27.3 µm, so their
  drift-scene excursions are not separated from readout instability. They are
  reported but are not load-bearing.
- The 25 s excursion smoothing width was chosen once, not swept.
- **The cost of home staleness is not measured.** W3 shows the frozen home is
  wrong for up to 40 % of the recording; how many assignments that costs needs
  a tracker run, which is out of scope here.
- W3's block-majority criterion is a lower bound on channel-selection error;
  blocks whose majority still matches can hold a large minority of off-home
  events.
- The row-step / `pitch_um` distinction is specific to a staggered layout and
  is not claimed to generalise.
- Peak memory was not instrumented; the ~6 GB figure is a coarse observation.
