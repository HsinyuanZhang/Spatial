# Temporal representation decomposition behind the frozen D9 p99.9 source

Status: **completed diagnostic. Shared dyadic dictionary is information-limited.
Least-bad compact code is a signed-5 raw prefix, especially on HJ. Compact
Level-2 still misses the ≤1 pp vs teacher gate.**
Date: 2026-08-25.
Isolation: GT events, GT rows, zero-phase bandpass, chronological 50/25/25,
frozen uniform D9/B5 p99.9 candidate source (`C>1` almost always).
Not online, not a selected architecture, not confirmation.

Command:

```bash
python -m Spatial.experiments.run_temporal_representation_decomp --pilot --quiet
```

CSV / JSON: `Spatial/output/temporal_representation_decomp/`.
This is the experiment queued by
[`causal_temporal_sketch_results.md`](causal_temporal_sketch_results.md)
after the signed-5 dyadic sketch failed.

## Question

On identical candidates and horizons, is the sketch failure

1. the dyadic dictionary (float dyadic already loses the teacher),
2. 5-bit quantization of an otherwise sufficient representation, or
3. an insufficient waveform horizon?

## Event-weighted family means (ambiguous ≈ overall; `C>1` ≈ 1)

Headline rows. `delay=16` is a 32-sample prefix; `delay=8, M=16` is the
least-bad dyadic sketch point already reported.

| Representation | Delay | Features | HJ | MEArec | All | vs teacher (all) |
|---|---:|---:|---:|---:|---:|---|
| full float teacher | 48 | 64 | 0.7760 | 0.7687 | 0.7710 | — |
| signed full 64×5 | 48 | 64 | 0.7532 | 0.7051 | 0.7199 | −5.10 pp |
| prefix float | 16 | 32 | 0.7727 | 0.7754 | 0.7746 | **+0.36 pp** |
| prefix signed-5 | 16 | 32 | **0.7649** | 0.7126 | 0.7287 | −4.22 pp |
| dyadic float | 8 | 16 | 0.7208 | 0.7187 | 0.7193 | −5.16 pp |
| dyadic signed-5 | 8 | 16 | 0.7149 | 0.6479 | 0.6685 | −10.24 pp |
| Fisher-M signed-5 on prefix | 8 | 16 | 0.7435 | 0.6684 | 0.6915 | −7.94 pp |

Prefix-float at delay 16 already matches the 64-sample teacher, so the
horizon is not the bottleneck. Float dyadic at the same delay/width as the
failed sketch loses **5.2 pp** to the teacher: the fixed dictionary is
information-limited before quantization. Signed-5 dyadic then loses another
~5 pp, concentrated on MEArec.

Raw signed-5 prefix at delay 16 is the least-bad compact code:

- HJ 0.7649 vs teacher 0.7760 (**−1.1 pp**). Nearly free on this family.
- MEArec 0.7126 vs teacher 0.7687 (**−5.6 pp**). This is the remaining
  5-bit hit.
- Full 64×5 SAD (0.7199 all) is no better than the 32-sample signed-5
  prefix (0.7287 all). Extra late samples do not buy identity after 5-bit
  quantization.

Fisher taps on the same signed-5 prefix do not beat the raw prefix. Do not
return to a learned tap bank as the next compact object.

## Interpretation

1. **Stop extending the dyadic Fisher grid.** Float features with the same
   IDs already fail the teacher. A wider sweep over the same dictionary
   cannot close a 5 pp information gap.
2. **Do not treat signed-5 dyadic as a quantization problem to be solved
   with per-feature scale.** The dictionary is first.
3. **A banked sample SRAM plus selected-tap SAD is not justified** by this
   table: raw prefix signed-5 is strictly better than Fisher-M signed-5 at
   matched `M`.
4. The compact Level-2 accuracy gate (≤1 pp vs teacher on both families)
   is still closed. HJ almost clears it with a 32-sample signed-5 prefix;
   MEArec does not.

This does not reopen pair-conditioned taps, morphology-12, or shared
dyadic as a main line. It identifies the next compact object, if Level-2
is kept at all: **signed-5 raw prefix around delay 16**, with the MEArec
5-bit gap as the remaining representation question.

## What this does not show

- Confirmation recordings were not touched (the parent sketch gate failed).
- Waveform acquisition latency (`t+48` vs `t+16`) is not a hardware
  measurement.
- The 0.77 teacher accuracy is candidate-internal WTA behind a p99.9
  spatial source, not end-to-end sorting.

## Reproduction

```bash
MPLCONFIGDIR=/tmp python -m Spatial.experiments.run_temporal_representation_decomp --pilot --quiet
```

Provenance: [`temporal_representation_decomp_provenance.md`](temporal_representation_decomp_provenance.md).
