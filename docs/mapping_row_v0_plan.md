# Mapping-row v0 plan

Frozen static contrast. Spec:
[`STAR-Mem_MappingRow_v0_执行说明.md`](STAR-Mem_MappingRow_v0_执行说明.md).

This plan copies the locked fields. It is not a result.

## Question

On a normalized spatial shape, does a **per-unit interval / weighted-violation
map** separate true vs nearest interferer better than **mean P2P + integer L1
radius**, and does it reach **candidate recall ≥ 0.99** with **≥ 4×** waveform
reads on both HJ and MEArec?

Offline GT-event / GT-row isolation. No online update, no CAM/RTL.

## Locked protocol

| Field | Value |
|---|---|
| Recordings | `drift16c_600s_11`, `static16c_600s_11`, `rec_v1_units10_snr5_seed202601`, `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s |
| Split | chronological 50 / 25 / 25, equal timestamps unsplit |
| Neighborhood | K=7 KNN including self (same as existing D9 L1) |
| Filter | full-recording zero-phase 300–6000 Hz |
| J0 | 5-bit COM+P2P, mean integer centroid, calibration p99.9 L1 radius |
| J1 | 5-bit COM + 4-bit g-normalized POSNEG, same mean+L1 relation |
| J2 | raw POSNEG (fit-column scale) + COM, 5/95 interval + width→`p_i` |
| J3 | g-normalized POSNEG + COM, `g` only `scale_alert` |
| J4 | J3 with all `p_i=0` |
| J5 | J3 + loud-template/quiet-query censoring |
| Box percentiles | fit 5 / 95 only |
| `p_i` | `clamp(ceil(log2(width))-1, 0, 3)` |
| `g` range | fit 1 / 99, alert only |
| `tau_candidate` | ceil(unit calibration 99.9 of `S_u`) |
| Directory | all rows (no tile prune) so arms share one coarse set |
| Traffic | `64×5` bits × `\|C\|` only when `\|C\|>1`; no unique-hit early exit |

## Stop rules

- J3 at 0.99 recall still ≈ 1.2× reads → stop this mapping relation.
- J1 > J0 and J3 fails → POSNEG helped, interval relation did not; do not sweep percentiles.
- Joint pass on both families required before `--confirm`.
