# Shift-Min + CiM Multi-Copy (P1)

Status: **pilot complete** — see [`shift_cim_results.md`](shift_cim_results.md).

## Goal

On P0 relative-offset footprints, add 1D shift-min matching, compare L1 vs L2,
and account for CiM Option A (``2S+1`` pre-shifted rows per unit).

## Locked defaults

- Footprint: `relative`, K=7
- S ∈ {0, 1, 2}
- Metrics: L1 and L2
- Codes: float and 4-bit per-spike-max
- Data: 12 HJ + 4 MEArec SNR5, 60s, chrono 50/50

## Success bar

1. HJ drift: S≥1 L1 lowers median true-unit P2P distance vs S=0
2. Best-S L1 4-bit within 1 pp argmin of float L1 (family mean)
3. Publish CiM area ×(2S+1) accounting table

## Reproduce

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_shift_match.py -q
python -m Spatial.experiments.run_shift_cim_pilot --duration 60 --quiet
```
