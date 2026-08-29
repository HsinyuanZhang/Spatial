# Shift-estimator / EMA-dimension diagnostics — provenance

Snapshot for [`drift_estimator_diagnostics.md`](drift_estimator_diagnostics.md).
Date: 2026-08-28.

## Repository state

```text
repository revision (HEAD)   c729bffa029724a0740c5573b92f43f3dcaff55f
branch                       shape-orthogonal-descriptor
```

`algorithms/drift_tracking.py` and `algorithms/spatial_shift_matching.py` are
untracked working-tree files at this revision, as they were for the POSNEG/MVM
and EMA\* knives. They are identified by digest below, not by revision.

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
python -m Spatial.experiments.run_drift_estimator_diagnostics --pilot --quiet
```

`--duration` is locked to 60 s by the runner. `--only {a,b,both}` selects a
diagnostic; the manifest below is from a single `both` run.

## Source digests (SHA-256)

```text
e7415fc35d1468a369397d347df1563efa47441fa125803762655a0ea348fdea  Spatial/algorithms/drift_tracking.py
fa670da90c5d31db49c35dcaee3583d53be6d6bc8aed8354652ee4de26c9623a  Spatial/algorithms/spatial_shift_matching.py
28869b5b80faf38c94f2c4cea612125448269dee57c6cd4a1b7e08820f29855f  Spatial/experiments/run_drift_estimator_diagnostics.py
```

The `drift_tracking.py` digest is **identical** to the one recorded in
[`ema_star_identity_convergence.md`](ema_star_identity_convergence.md). This
diagnostic measured the deployed estimator and updater without modifying them.

## Output digests (SHA-256)

```text
b40afe52f721c28eb57753dee2c4e6cead572dfad00c45f2d1c0a371be2b3f18  Spatial/output/drift_estimator_diag/drift_estimator_diag_summary.json
82fdc3a506421e64e29fd636b4a9d7e99f96b7653d1c6ac77dc23c56b0a188db  Spatial/output/drift_estimator_diag/ema_collapse_threshold.csv
fd49b2ad4ae1fa990f67744655060989f0249623159f33c74f92ba09c76ac576  Spatial/output/drift_estimator_diag/ema_rate_vs_k.csv
75ff0d53109c56a9fbe340951225caaea917910a32f9b6bbb7bac5cc499a6c85  Spatial/output/drift_estimator_diag/estimator_amplitude_bias.csv
7bb7c80345948f24a8bac4719e37acf69974e57f4b2597b89617bdea942ebd90  Spatial/output/drift_estimator_diag/estimator_amplitude_bins.csv
9ca4550e6778c8d43b51581c271b935199454b23253117a36066d815482fdd5b  Spatial/output/drift_estimator_diag/estimator_amplitude_matched_gap.csv
```

A companion manifest is written by the runner to
`Spatial/output/drift_estimator_diag/outputs.sha256`. The summary JSON digest
is not reproducible across runs because it embeds the source manifest; the
five CSVs are.

### Determinism

Diagnostic B was run twice (once as `--only b`, once inside `--only both`).
`ema_rate_vs_k.csv` and `ema_collapse_threshold.csv` reproduced
**bit-identically** (`fd49b2ad…`, `82fdc3a5…`). No seeds are consumed by
either diagnostic: the split is chronological, \(\mu\) is a mean, and both
estimators are deterministic grid minimizations.

## Inputs

Same four locked development recordings as the POSNEG/MVM and EMA\* knives,
resolved through `run_shape_descriptor_pilot.LOCKED_DEVELOPMENT_RECORDINGS`:

| rec_key | source | events at 60 s | channels |
|---|---|---:|---:|
| `hj_drift` | HJ `drift16c_600s_11` | 3371 | 16 |
| `hj_static` | HJ `static16c_600s_11` | 2788 | 16 |
| `mearec_10u` | `rec_v1_units10_snr5_seed202601` | 4611 | 32 |
| `mearec_20u` | `rec_v1_units20_snr5_seed202601` | 9224 | 32 |

Dataset locations follow [`REPRODUCING.md`](REPRODUCING.md). MEArec files that
share one spike-train seed across SNR renderings are not independent temporal
or biological replicates; only one SNR rendering per unit count is used here.

## Evidence boundary

| Contract item | This record |
|---|---|
| Truth boundary | GT event times, GT unit rows. E1 scores each event against its **own GT unit**, so it is an estimator measurement, not an assignment result |
| Split | chronological 50/25/25, equal timestamps unsplit. E1 uses cal+test; E2 uses cal only |
| Fit boundary | \(\mu\), homes, and the E2 confidence threshold come from the fit and calibration splits. Nothing is fitted on test |
| Signal path | full-recording zero-phase Butterworth 300–6000 Hz order 3. **Not causal** |
| Metrics | \(\lvert\hat\delta\rvert\), fraction of non-zero \(\hat\delta\), Spearman \(\rho\) against \(\|x\|/\|\mu\|\), amplitude-matched means; cal accuracy / same-home / worst-unit for E2 |
| Aggregation | event-weighted within a recording; recordings are never pooled |
| Latency | not applicable; no online claim |
| Hardware boundary | none; no bit, area, or energy quantity is produced |
| Decision | **no configuration selected.** `gain` / `gain_sub` are diagnostic arms. \(\eta^*=0.005\) at \(K=12\) is unchanged |
| Confirmation | not authorized and not opened |
| Negative evidence | retained: the E1 residual dependence on HJ drift (\(\rho=-0.43\) after correction) and the high-ratio residual bin are reported, not hidden |

## Amendments recorded against other documents

| Document | Section | Amendment |
|---|---|---|
| [`posneg_mvm_shift_drift.md`](posneg_mvm_shift_drift.md) | §8.1, Q5 | The static \(\lvert\hat\delta\rvert\approx0.05\) pitch floor is about half amplitude bias, not noise. \(C(d)=0.153\) is inside the correctable error budget; Q5 and the D5 / shared-gain skips are reopened |
| [`drift_tracking_results.md`](drift_tracking_results.md) | Verdict Q5 | Same amendment |
| [`ema_star_identity_convergence.md`](ema_star_identity_convergence.md) | §C, §G | \(\eta^*=0.005\) stands but is \(K=12\)-scoped. Collapse boundary is \(\eta_{\mathrm{crit}}\approx1/(3K)\); the scope rule for other probes is \(\eta\le1/(5K)\) |

No number in those documents is rewritten. The amendments are scope and
interpretation changes, recorded here and in
[`RESULTS_OVERVIEW.md`](RESULTS_OVERVIEW.md).

## Not covered

- No corrected \(C(d)\) was measured. The \(C\approx0.35\) figure in the result
  document is an extrapolation under stated assumptions.
- No contract test accompanies the runner yet.
- Controlled-drift runs are untouched; their separate self-consistency problem
  (perturbation and tracker share the interpolation operator, applied to
  already-extracted extrema) is noted but not measured here.
