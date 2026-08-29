# Relative-Offset Footprint (P0)

Status: **pilot complete** — see [`relative_offset_footprint_results.md`](relative_offset_footprint_results.md).

## Goal

Replace distance-ranked KNN P2P slots with a fixed primary-axis relative-offset
patch so column *j* always means the same physical offset from home. This is
the CiM-friendly Stage-1b representation; shift-min is deferred to P1.

## Locked defaults

- Layout: 1D along `primary_axis(geom)` (higher-variance axis)
- `half_width=3` → `K=7` (matches existing descriptor width)
- OOB slots: channel id `-1`, P2P `0`
- KNN path remains the default elsewhere; this pilot is A/B only

## Protocol

- Duration 60s, chronological 50/50
- Families: all 12 HJ + MEArec SNR5 (4 files) by default
- Methods: `knn` vs `relative`
- Metrics: P2P-only argmin accuracy; COM∧P2P gate recall @ train 95th;
  mean `|C|`; median true-unit P2P L1

## Success bar

Relative family-mean P2P argmin ≥ KNN, and HJ median true-unit P2P L1 ≤ KNN.

## Reproduce

```bash
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_spatial_footprint.py -q
python -m Spatial.experiments.run_relative_footprint_pilot --duration 60 --quiet
```
