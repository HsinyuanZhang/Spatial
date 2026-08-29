# Bulk / tail escalation (not certified early-exit)

Status: **kill. Empty-inner → outer p99.9 does not jointly hit 0.99 recall
and ≥2× traffic on either descriptor.**
Date: 2026-08-25.
Isolation: GT events, GT rows, zero-phase bandpass, chronological 50/25/25.
Locked four development recordings. Unique inner hits stay candidates, not
labels. Traffic is 64×5 bits only when `|C|>1`.

This is LEVEL1_ROADMAP direction 1: a tight bulk radius plus an explicit
second path for empty inner lists. It is **not** the Wilson-certified
singleton exit already closed by `run_certified_dual_range_pipeline.py`.

## Commands

```bash
python -m Spatial.experiments.run_bulk_tail_escalation --pilot --duration 60
python -m Spatial.experiments.run_bulk_tail_escalation --pilot --duration 60 --arm com_posneg
```

## Kill criterion (pre-registered)

If total traffic including escalation does not beat 2× **at 0.99
end-to-end recall** on both families, the escalation architecture is not
the answer.

## Family aggregates (event-weighted recall)

| Arm | Family | Recall | Outer p99.9 recall | mean C | esc. rate | traffic × | Outer × |
|---|---|---:|---:|---:|---:|---:|---:|
| D9 COM+P2P | HJ | 0.8942 | 0.9916 | 2.20 | 0.0695 | 6.40 | ~1.48 |
| D9 COM+P2P | MEArec | 0.9167 | 0.9942 | 4.42 | 0.0095 | 4.30 | ~1.31 |
| `com_posneg` | HJ | 0.9279 | 0.9928 | 1.81 | 0.0632 | 8.81 | ~1.50 |
| `com_posneg` | MEArec | 0.9315 | 0.9962 | 3.95 | 0.0269 | 7.93 | ~1.25 |

Per-recording D9:

| Recording | Recall | Outer R | mean C | esc | × |
|---|---:|---:|---:|---:|---:|
| HJ drift11 | 0.8470 | 0.9905 | 2.57 | 0.083 | 4.70 |
| HJ static11 | 0.9512 | 0.9928 | 1.75 | 0.056 | 8.10 |
| MEArec 10u | 0.9254 | 0.9957 | 2.27 | 0.017 | 4.93 |
| MEArec 20u | 0.9124 | 0.9935 | 5.49 | 0.002 | 3.67 |

Per-recording `com_posneg`:

| Recording | Recall | Outer R | mean C | esc | × |
|---|---:|---:|---:|---:|---:|
| HJ drift11 | 0.9098 | 0.9941 | 2.00 | 0.075 | 7.32 |
| HJ static11 | 0.9498 | 0.9914 | 1.59 | 0.052 | 10.30 |
| MEArec 10u | 0.9523 | 0.9957 | 1.57 | 0.050 | 11.94 |
| MEArec 20u | 0.9211 | 0.9965 | 5.14 | 0.003 | 3.92 |

Traffic including escalation **does** beat 2× (and usually 4×). Recall
does **not** stay at 0.99. The joint kill fires.

## Why recall falls

The outer p99.9 list still has 0.99 recall at ~1.3–1.5×, matching J0 / the
frozen D9 source. Escalation only runs when the inner p90 list is
**empty**. Events whose inner list is nonempty but missing the true unit
never escalate, and they dominate the miss budget:

- D9 inner-only recall is 0.77 / 0.90 / 0.91 / 0.91 on the four files.
- Escalation rate is 8.3% / 5.6% / 1.7% / 0.2%.
- Family recall lands at 0.89–0.92 (D9) or 0.93 (`com_posneg`).

A label-free controller cannot know that a nonempty inner list is wrong.
Escalating unique inner hits is the certified-exit branch already killed.
Escalating every nonempty list rebuilds the outer radius.

## `com_posneg` vs D9

The roadmap predicted a larger bulk/tail gap on `com_posneg` (2.9× tighter
clouds). The tighter descriptor does raise inner quality and traffic (HJ
0.894 → 0.928 recall, 6.4× → 8.8×). It does not restore 0.99. The tail
that sets the 0.99 radius is still not an empty-list event.

Do **not** sweep inner percentiles. The failure mode is structural.

## What this does not show

- Confirmation was not run.
- Dense-probe codebook escalation was not this experiment.
- Logical 64×5 traffic is not PPA.

CSV: `Spatial/output/bulk_tail_escalation/` and
`Spatial/output/bulk_tail_escalation_com_posneg/`.
Provenance: [`bulk_tail_escalation_provenance.md`](bulk_tail_escalation_provenance.md).
