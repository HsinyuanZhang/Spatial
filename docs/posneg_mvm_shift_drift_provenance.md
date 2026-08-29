# POSNEG MVM / shift / causal-tracking provenance

Date: 2026-08-27.
Isolation: GT events, GT units, zero-phase 300–6000 Hz, 60 s,
chronological 50/25/25, equal timestamps unsplit.
Recordings: HJ `drift16c_600s_11`, HJ `static16c_600s_11`,
MEArec `rec_v1_units10_snr5_seed202601`, MEArec `rec_v1_units20_snr5_seed202601`.

Killed Mapping Row / deformable-template runners and their result files were
not modified.

## Commands

```bash
conda activate spint
cd /home/xinyuan/SNN_SpikeSorting
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_posneg_mvm_shift_tracking.py -q
python -m Spatial.experiments.run_posneg_mvm_ablation --pilot --duration 60
python -m Spatial.experiments.run_spatial_shift_ablation --pilot --duration 60
python -m Spatial.experiments.run_drift_coherence --pilot --duration 60
python -m Spatial.experiments.run_drift_tracking_compare --pilot --duration 60
python -m Spatial.experiments.run_controlled_drift --pilot --duration 60
```

Two pytest invocations: 8 passed / 8 passed.
Primary matching CSV rerun hash match:
`1a2564e5e9d949347e386dacabef5cf70b8fa36cad8ccfa649ef4f46b1497430`.

## Sources

```text
16319096721cc16064c6ea8709780f5733806b411593e9b8f83775c80927c35e  Spatial/algorithms/posneg_mvm_matching.py
fa670da90c5d31db49c35dcaee3583d53be6d6bc8aed8354652ee4de26c9623a  Spatial/algorithms/spatial_shift_matching.py
8f662417b64753ad6689070be193144dd64684b7e2b16212ed90f8b0786c7304  Spatial/algorithms/drift_tracking.py
f82cab5ef1ceede8f5b737556d37c144a36224a594104df1ba2e9175a35f4bc8  Spatial/experiments/posneg_mvm_common.py
36e2dcf3528ff9e5ded9b635ef4ef01fe9329e9fe73652a9356f168f2c13cc17  Spatial/experiments/run_posneg_mvm_ablation.py
4d5471aff6c465053a497d68be0241aa784b55d70f5cb949a36ed43b9873d689  Spatial/experiments/run_spatial_shift_ablation.py
6695af3588ba5e168a4fcf90f9ca408b08b76e0ec62491e35ca23082d4a4213e  Spatial/experiments/run_drift_coherence.py
bfcb71b25ce539e55b0b5062daec368143cfa2b70a7465bb6d0d69475585e0e9  Spatial/experiments/run_drift_tracking_compare.py
e0726ce49dd93835f10e0bd601f9a315e674f97d52098fde27b73f7f5b3e936e  Spatial/experiments/run_controlled_drift.py
5ca86182ec434d54f055c13b796d4a033409b4dffcfbd30a4085851f00cfbf4b  Spatial/tests/test_posneg_mvm_shift_tracking.py
c0cf1c2bf95542d2ccc413fb8ae44699ff007c459a3507d1c440d91c8e596e1f  Spatial/docs/posneg_mvm_shift_drift_plan.md
```

## Outputs

```text
1a2564e5e9d949347e386dacabef5cf70b8fa36cad8ccfa649ef4f46b1497430  Spatial/output/posneg_mvm_ablation/posneg_mvm_ablation_pilot.csv
a215865c30081f40c73456756d2c2560072f1023c9a1e828b2c39e84afd046f3  Spatial/output/posneg_mvm_ablation/posneg_mvm_ablation_summary.json
955e3903fbbbd12f29fcce55dec62bb52857f2fcb2d19ae8d8418f2e53699137  Spatial/output/posneg_mvm_ablation/posneg_mvm_confused_pairs.csv
bbc3564958eb313ea4bbc640f013c8a33e8216cc8bb6684af3353192e8391586  Spatial/output/spatial_shift_ablation/spatial_shift_ablation_pilot.csv
cb8de1d0e2c875aee8af58992cb4ae9e39d5288c579b3feb8311a566fb7c1a84  Spatial/output/spatial_shift_ablation/spatial_shift_ablation_summary.json
3ed163c1c2b208328a2380c2fe479c71f3cccb4ab862928694afb6d52bb6e58e  Spatial/output/drift_coherence/drift_coherence_pilot.csv
2fabcc1b62a3d7fa141e0553bd4e94fcfda4ee073491aee7204d425d184f2186  Spatial/output/drift_coherence/drift_coherence_pairs.csv
d0283c0c8ac7ab2034668e4c61c03ffe6e92add451da456c5029918884285fab  Spatial/output/drift_tracking_compare/drift_tracking_compare_pilot.csv
8553114dda5ffc53ced9eb2b67eef1323046e552ae0d347d4247c808c1e38746  Spatial/output/controlled_drift/controlled_drift_pilot.csv
```

Recompute `sha256sum Spatial/algorithms/spatial_shift_matching.py` after the cosine \(I_{\mathrm{true}}\) inversion fix; the hash above is post-fix, matching the shift CSV in this list.

## Conditional arms

| Arm | Outcome |
|---|---|
| A5 angular+radial | ran; \(\lambda_{\mathrm{amp}}=0.1\) from calibration |
| C1 \(\{\pm 2\}\) | skipped; C1 cal \(L_{\mathrm{flex}}<0\) failed |
| D5 smooth field | skipped; nearest \(C(d)=0.15<0.3\) |
| D6 slow \(\mu\) | ran (HJ-drift cal D3 ≥ Frozen); test same-home tie, \(\mu\) moves |
| Shared gain | skipped; gain pair corr 0.05 |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Spatial/tests/test_posneg_mvm_shift_tracking.py -q
```

8 passed (twice).
