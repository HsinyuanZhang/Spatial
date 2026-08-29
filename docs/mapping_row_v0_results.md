# Mapping-row v0 results

Status: **development gate FAIL. Stop this mapping relation.**
Date: 2026-08-25.
Isolation: GT events, GT rows, zero-phase bandpass, chronological 50/25/25.
Not online, not CAM/RTL, not a selected architecture.

Command:

```bash
python -m Spatial.experiments.run_mapping_row_static --pilot --duration 60 --quiet
```

CSV / JSON: `Spatial/output/mapping_row_v0/`.
Provenance: [`mapping_row_v0_provenance.md`](mapping_row_v0_provenance.md).

## Joint gate (must pass both families)

| Criterion | J0 | J1 | J2 | J3 | J4 | J5 |
|---|---|---|---|---|---|---|
| HJ recall ≥ 0.99 | **0.9916** | 0.9877 | **0.9903** | 0.9877 | 0.9844 | 0.9883 |
| MEArec recall ≥ 0.99 | **0.9942** | **0.9957** | **0.9939** | **0.9948** | **0.9948** | **0.9939** |
| Waveform reads ≥ 4× (HJ / MEA) | 1.48 / 1.25 | 1.49 / 1.24 | 1.78 / 1.17 | 1.53 / 1.29 | 1.56 / 1.28 | 1.52 / 1.29 |
| Not via C=1 early exit | yes (C>1 ≈ 1.0) | yes | yes | yes | yes | yes |
| True/interferer gap vs J0 | — | mixed | **narrower** | **narrower** | **narrower** | **narrower** |

No arm jointly hits 0.99 recall **and** 4× traffic. Interval arms do not
open the true-vs-nearest-interferer gap; they shrink it.

J0 recall 0.9916 / 0.9942 matches the frozen D9/B5 p99.9 source already in
the DAC macros (`\HJSourceRecall`, `\MEASourceRecall`).

## Family aggregates (event-weighted recall)

| Arm | HJ R | MEA R | HJ mean C | MEA mean C | HJ gap mean (p50) | MEA gap mean (p50) | HJ × | MEA × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 mean L1 P2P | 0.9916 | 0.9942 | 7.11 | 11.76 | 9.31 (9.5) | 8.84 (12) | 1.48 | 1.25 |
| J1 mean L1 POSNEG | 0.9877 | 0.9957 | 7.06 | 11.91 | 9.84 (9.5) | 13.70 (17.5) | 1.49 | 1.24 |
| J2 interval, raw POSNEG | 0.9903 | 0.9939 | 5.90 | 12.61 | 3.12 (3) | 3.27 (4) | 1.78 | 1.17 |
| J3 interval, g-shape | 0.9877 | 0.9948 | 6.88 | 11.57 | 2.75 (2.5) | 1.36 (2) | 1.53 | 1.29 |
| J4 no precision shift | 0.9844 | 0.9948 | 6.71 | 11.68 | 3.39 (3.5) | 2.29 (2.5) | 1.56 | 1.28 |
| J5 + censoring | 0.9883 | 0.9939 | 6.88 | 11.59 | 2.75 (2.5) | 1.38 (2) | 1.52 | 1.29 |

`frac_C>1` stays ≥ 0.995 on almost every recording. Unique-hit early exit is
not how any arm gets its traffic number.

## Per spec interpretation

- **J1 vs J0.** POSNEG as a drop-in L1 input does not beat P2P on HJ recall
  (0.9877 < 0.9916). MEArec gap widens (13.7 vs 8.8) but traffic does not
  move. Not a joint win.
- **J3 vs J2.** g-normalization does not rescue the interval relation (HJ
  recall slightly worse; gap still ~2–3 vs J0 ~9).
- **J4 ≈ J3.** Precision shift is not the failure mode.
- **J5 ≈ J3.** Loud-template/quiet-query censoring does not reopen the gap.
- **J3 at high recall still ~1.3× reads.** Stop this mapping relation. Do
  **not** sweep 5/95, mask counts, or `P`.

The collapsed gap is the diagnostic the spec asked for. Interval interiors
score 0 for every unit whose 5/95 box still covers the query, including
nearby interferers. That is “do not score distance to the midpoint” working
as specified — and it **removes** the L1-to-mean margin that J0 uses. On
these 9–16-D low-bit descriptors the map is a re-skinned average template,
not a better separator.

Worst-unit recall on HJ drift11 falls to 0.877 on J1/J4 (J0 0.913). No
family-wide collapse, but the mapping does not protect the weakest unit.

## What this does not show

- Dense-probe COM codebook was not run (no development pass).
- Confirmation HJ/MEArec files were not touched.
- 85–93% spatial unique-ID in the Overleaf abstract is a different metric
  (argmin assignment), not this candidate gate.
- Indexed SRAM vs CAM is not compared.

## Next

Do **not** confirm. Do **not** reopen percentile / mask / prototype grids.

The queued time-representation decomposition and the LEVEL1_ROADMAP bulk/tail
policy are now both recorded:

- [`temporal_representation_decomp_results.md`](temporal_representation_decomp_results.md)
  — float dyadic is dictionary-limited; signed-5 raw prefix is the least-bad
  compact code; compact Level-2 still misses ≤1 pp vs teacher on MEArec.
- [`bulk_tail_escalation_results.md`](bulk_tail_escalation_results.md)
  — empty-inner escalation is killed on D9 and `com_posneg`.
