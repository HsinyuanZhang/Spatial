# Deformable spatial template Phase-1 provenance

Date: 2026-08-27.
Isolation: GT events, GT rows, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25.

## Inputs

Same four locked development recordings as mapping-row v0.
See [`mapping_row_v0_provenance.md`](mapping_row_v0_provenance.md).

## Sources

```text
c15525f10c76dee06c6219b3ce8c7494694a50d94ba683bdd174f6ba9e405401  Spatial/algorithms/deformable_spatial_template.py
56f98fbf3f528ad728ee9e4f852eb8644bb8bbcd8b00503a27326d00fc3375fa  Spatial/experiments/run_deformable_spatial_static.py
9df882c38b83b263f73f471735b298c888d21aa5d07a5480cd4b09adf99c9cbe  Spatial/tests/test_deformable_spatial_template.py
37f7ba58dc9f46df5d14211d4eca825214d9018e2cf6258b471ba97009bd8cca  Spatial/docs/deformable_spatial_template_plan.md
```

## Outputs

```text
40c2adf9e98755475033947448d3daeba1429d76cb775dbe6775d2585962ba3e  Spatial/output/deformable_spatial_static/deformable_spatial_static_pilot.csv
889e11e0adee8ab870347e8bd61c37dad89d76784aef7220d674d044bbcdbf73  Spatial/output/deformable_spatial_static/deformable_spatial_static_summary.json
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_deformable_spatial_template.py -q
```

10 passed.
