# POSNEG MVM matching, polarity ablation, shift leakage, and causal tracking — frozen plan

This is a new independent knife. It does not revive Mapping Row, interval QDA,
or free per-event deformable templates. Reconstruction / best-alignment
distance is not a success metric. Success is **better sorting**, especially
same-home assignment and same-home margin.

Protocol frozen before any test-set look. Grids below are constants. Test-set
outcomes do not retune them.

## Questions

| ID | Question |
|---|---|
| Q1 | Can biased-dot \(S=2x^\top\mu-\|\mu\|^2\) replace squared L2 without changing assignment? Is \(S=Mx+b\) with \(b_u=-\|\mu_u\|^2\) the MVM form of L2 when \(x\) is shared? |
| Q2 | What amplitude information do pure-dot and cosine delete, and is that deletion helpful on HJ drift vs harmful on HJ static / MEArec 20u? |
| Q3 | Where does POSNEG beat P2P: positive peak, negative trough, polarity mass ratio \(r_{pn}\), spatial displacement \(\Delta c\), or just higher dimension? |
| Q4 | Is moving-template / shift-min robustness, or impostor flexibility? Answer with \(L_{\mathrm{flex}}\) and same-home margin. |
| Q5 | Is inferred drift global, local/tile, a smooth field, or unit-specific noise? Static recordings must stay quiet. |
| Q6 | Which **causal** tracker improves classification vs Frozen: EMA, global shared shift, local/tile shared shift, (gated) smooth field, (gated) slow \(\mu\) update? |
| Q7 | Can state move while identity \(\mu_u\) stays put? |
| Q8 | Do static / no-drift trackers stay quiet? |

## Locked fields

| Field | Frozen value |
|---|---|
| Recordings | HJ `drift16c_600s_11`, HJ `static16c_600s_11`, MEArec `rec_v1_units10_snr5_seed202601`, MEArec `rec_v1_units20_snr5_seed202601` |
| Duration | 60 s |
| Split | chronological 50% fit / 25% calibration / 25% test; equal timestamps unsplit (`stable_chronological_three_way_split`) |
| Isolation | GT events, GT unit labels (component isolation; labeled as GT in every table) |
| Filter | zero-phase Butterworth 300–6000 Hz, order 3 |
| Observation | analog POSNEG extrema in \(\pm 15\) samples; P2P is \(p_i=q_i^++q_i^-\) from the same extrema |
| \(K\) | \(\{7,12,16\}\) **geometry** nearest electrodes around a reference home, ordered by physical offset, never strongest-\(K\) |
| Default frame | **candidate-home**: unit \(u\) is fit and queried on physical-offset slots of its fit majority home \(h_u\) |
| Ablation frame | event-argmax recentering, A1 L2 POSNEG at \(K=12\) only, labeled `A1_event_argmax` |
| Scoring for B/C/D | squared L2; biased-dot is reported as the MVM twin, not a second classifier to mix with features |
| Confirmation corpus | closed |
| Seed | 0 |
| Hardware energy | not a selection criterion; MAC / abs / norm counts are recorded only |

MEArec 10-unit historically has no co-home rivals. Same-home fields there are
NaN, never 0, and are never averaged with MEArec 20u.

## Frozen grids (no test-set search)

| Symbol | Value | Role |
|---|---|---|
| Scoring A0–A4 | L1; squared L2; biased-dot; pure-dot; cosine | always run |
| \(\lambda_{\mathrm{amp}}\) candidates | \(\{0.1, 0.5, 1.0, 2.0\}\) | A5 only; one global value from **calibration** (accuracy, then home-margin P10) |
| Polarity ratios B4 | \(1:1\), \(1:2\), \(2:1\) | global, never per-unit |
| Discrete shifts C1 | \(s\in\{-1,0,+1\}\) pitch | FREE-SHIFT DIAGNOSTIC |
| Extra discrete | \(\{-2,-1,0,+1,+2\}\) | **not run** unless C1 calibration \(L_{\mathrm{flex}}<0\) **and** same-home margin P10 improves vs C0 |
| Fractional C2 | \(\delta\in\{-0.5,-0.25,0,+0.25,+0.5\}\) pitch | FREE-SHIFT DIAGNOSTIC |
| \(\eta_{\mathrm{EMA}}\) | 0.05 | D1 |
| \(\eta_{\mathrm{state}}\) | 0.20 | D3/D4 shared-shift EMA |
| \(\eta_\mu\) | 0.005 | D6 only; requires \(\eta_\mu\ll\eta_{\mathrm{state}}\) |
| Block length | 1.0 s | GT coherence / gain-coherence |
| Tile width | 3 pitches along the primary axis | D4 |
| Confidence | update iff \(M_{\mathrm{all}}>\max(0, \text{cal Frozen }M_{\mathrm{all}}\text{ P10})\) | causal trackers |
| \(K_{\mathrm{identity}},K_{\mathrm{state}}\) | Frozen at every \(K\in\{7,12,16\}\); trackers at \((7,12)\) and \((12,16)\) | identity vs state support |
| Tracking stream | test split only, chronological, \(\mu\) from fit | causal |
| Controlled perturbations | gain ramp \(1\to 1.5\); gain step \(1\to 1.5\) at test midpoint; spatial ramp \(0\to 0.5\) pitch; two-tile opposite \(\pm 0.5\) pitch | static recordings only |

### Conditional arms (must skip, not silently omit, unless the trigger fires)

| Arm | Trigger (calibration or GT-coherence only) |
|---|---|
| A5 angular+radial | Cosine cal same-home accuracy **strictly exceeds** L2 on HJ drift **and** MEArec 20u cosine same-home is not more than 2 pp below L2 |
| C1 extra \(\{\pm 2\}\) | C1 cal \(L_{\mathrm{flex}}<0\) and cal home-margin P10 \(>\) C0 |
| D5 smooth field \(\delta(r,t)=c_0+c_1 r\) | D4 local coherence: nearest-bin \(C(d)>0.3\) on HJ drift, \(C(d)\) falls with distance, and static mean \(\lvert\hat\delta\rvert<0.15\) pitch |
| D6 slow \(\mu\) | Best of D3/D4 is **not worse** than Frozen same-home (or overall if same-home NaN) on HJ drift **and** does not drop HJ static / MEArec 20u same-home by \(>2\) pp |
| Shared/tile gain | Mean pairwise \(\mathrm{corr}(g_u,g_v)>0.3\) on HJ drift GT blocks **and** static mean \(\lvert\log g_u\rvert<0.05\) |

If a trigger fails, the summary JSON records `skipped` plus the failed trigger.

## Part A — scoring (representation fixed at raw POSNEG concat)

On fit candidate-home mean \(\mu_u\):

| Arm | Formula | Higher better? | MVM? |
|---|---|---|---|
| A0 | \(D=\sum_i\lvert x_i-\mu_{u,i}\rvert\) | no | no |
| A1 | \(D=\|x-\mu_u\|^2\) | no | equivalent to A2 when \(x\) is shared |
| A2 | \(S=2x^\top\mu_u-\|\mu_u\|^2\) | yes | yes, \(S=Mx+b\), \(b_u=-\|\mu_u\|^2\), \(M=2\mu\) |
| A3 | \(S=x^\top\mu_u\) | yes | yes, no amplitude penalty |
| A4 | \(S=(x^\top\mu_u)/(\|x\|\|\mu_u\|)\) | yes | no (needs norms) |
| A5 | \(S=\cos(x,\mu_u)-\lambda_{\mathrm{amp}}(\log\|x\|-\log\|\mu_u\|)^2\) | yes | gated |

Sanity (unit test + experiment): for a **shared** \(x\), \(\mathrm{argmin}\) A1 \(=\) \(\mathrm{argmax}\) A2 within float tolerance, and \(S=-D+\|x\|^2\).

Candidate-home gathering makes \(x_u\) depend on \(h_u\). Same-home units share \(x\), so same-home assignment must match. Overall assignment may differ by \(\|x_u\|^2\); that disagreement rate is reported, not “fixed”.

## Part B — representation (scoring fixed at A1 L2 / A2 biased-dot)

| Arm | Feature |
|---|---|
| B0 | P2P \(p=q^++q^-\) |
| B1 | raw POSNEG \([q^+;q^-]\) |
| B2 | POS-only \(q^+\) |
| B3 | NEG-only \(q^-\) |
| B4 | weighted POSNEG L2 with global \(\lambda_+\!:\!\lambda_-\) |
| B5 | B1 \(+ r_{pn}=\log[(\sum q^++\varepsilon)/(\sum q^-+\varepsilon)]\) |
| B6 | B1 \(+\Delta c=c_+-c_-\) (two components on 2-D geometry) |
| B7 | B1 \(+ r_{pn}+\Delta c\) |

Same-home confused pairs: list pairs that P2P places close (template L2) and POSNEG stretches, with POS-only / NEG-only distances.

If B5/B6/B7 do not beat raw POSNEG, the derived features are already in the concat and are not stored.

## Part C — moving convolution / shift (DIAGNOSTIC only)

C0 fixed candidate-home template. C1 discrete shift-bank. C2 fractional interpolation.

Score paths: biased-dot moving correlation **and** cosine moving correlation. The biased-dot path **must not** divide by \(\|x\|\|\mu\|\).

Required extras: \(I_{\mathrm{true}}\), nearest-wrong-same-home \(I_{\mathrm{wrong}}\) (nearest under **fixed** C0, then that unit’s own free-shift improvement), \(L_{\mathrm{flex}}=I_{\mathrm{wrong}}-I_{\mathrm{true}}\), chosen-shift histogram, drift vs static \(\lvert s\rvert\), static false nonzero shifts, same-home margin.

A reconstruction drop without a same-home-margin gain is not success. \(L_{\mathrm{flex}}\ge 0\) stops this branch (no retune, no \(\{\pm 2\}\) expansion).

## Part D — causal tracking (not free shift)

Order: **predict state → freeze → classify → update only if confident**.

| Arm | Role |
|---|---|
| D0 Frozen | deployable baseline; \(\mu_u\) fixed, \(\delta=0\) |
| D1 EMA | \(\mu_u\leftarrow(1-\eta)\mu_u+\eta x\) after high-confidence assignment |
| D2 free per-event shift | Part C best bank; **ORACLE / DIAGNOSTIC**, not a tracker |
| D3 global shared \(\delta(t)\) | every unit uses the same predicted \(\delta\); current event cannot pick a private shift |
| D4 local/tile shared \(\delta_g(t)\) | 3-pitch primary-axis tiles |
| D5 linear field | gated |
| D6 shared shift + slow canonical \(\mu\) | gated; never used to rescue a failed tracker |

Absolute amplitude stays in \(\mu_u\). No per-candidate gain. Shared/tile gain only after the gain-coherence trigger.

Before causal tracking: GT-only 1 s block state-ownership on HJ drift, HJ static, MEArec 10u, MEArec 20u. Estimate \(\delta_{u,b}\) and \(g_{u,b}\) per unit/block by matching that unit’s events to \(\mu_u\). Plot \(C(d)=\mathrm{corr}(\delta_u,\delta_v)\) vs \(\|r_u-r_v\|\). Strong chaotic “drift” on static data means the estimator is untrusted.

Controlled synthetic drift is applied to the **test** portion of static/no-drift recordings only. Fit stays unperturbed.

## Metrics (every arm, per recording)

1. overall assignment accuracy
2. same-home assignment accuracy (NaN if no co-home rivals)
3. \(M_{\mathrm{home}}=D_{\mathrm{nearest\ wrong\ samehome}}-D_{\mathrm{true}}\): P10, median, P90, \(P(M>0)\)
4. per-unit recall mean; worst-unit recall
5. paired \(\Delta\) vs L2 (Part A) or vs B1/C0/D0 as appropriate
6. operation form / MVM flag / MAC / abs / norm counts
7. shift: \(I_{\mathrm{true}}\), \(I_{\mathrm{wrong}}\), \(L_{\mathrm{flex}}\), chosen-shift stats
8. tracking: \(\Delta\mathrm{Acc}_{\mathrm{home}}\), \(\Delta M_{\mathrm{home}}\) vs Frozen; \(\|\mu(t)-\mu(0)\|\) over nearest same-home template distance; false-motion \(\lvert\hat\delta\rvert\) on static data
9. controlled: \(\lvert\hat\delta-\delta_{\mathrm{GT}}\rvert\), settling time, false motion on unperturbed prefix, identity-template movement

Every table is per recording (HJ drift, HJ static, MEArec 10u, MEArec 20u). Family averages, if printed, are supplementary and not the decision numbers.

## Stop conditions (no retune)

1. Reconstruction / best alignment improves, same-home margin falls → kill that arm.
2. Cosine helps HJ drift but collapses dense/static same-home → skip A5, cosine is not deployable.
3. \(L_{\mathrm{flex}}\ge 0\) → free shift is impostor flexibility; do not expand the bank.
4. Shared/local tracker does not beat Frozen on HJ drift, or hurts static classification → skip D5/D6.
5. State estimator produces lots of false motion on no-drift data → estimator untrusted; do not deploy D3/D4.
6. Slow \(\mu\) update is not used to rescue a failed state tracker.
7. Larger \(K\) that only lifts overall accuracy while hardest same-home margin stays down is not selected.

## Out of scope

Mapping Row, free per-event gain/offset/polarity/\(h\), deformable Phase-2 runners, confirmation corpus, 0.99-recall / 4× traffic as this knife’s gate, neural nets, unit-specific \(B_u\), CiM energy as a selection criterion, per-unit polarity weights, large-range shift scans, test-set hyperparameter search.

## Runners (new files only)

```text
python -m Spatial.experiments.run_posneg_mvm_ablation --pilot --duration 60
python -m Spatial.experiments.run_spatial_shift_ablation --pilot --duration 60
python -m Spatial.experiments.run_drift_coherence --pilot --duration 60
python -m Spatial.experiments.run_drift_tracking_compare --pilot --duration 60
python -m Spatial.experiments.run_controlled_drift --pilot --duration 60
```

Do not launch `run_mapping_row_static`, `run_deformable_spatial_static`,
`run_deformable_spatial_tracking`, or `run_deformable_spatial_controlled_drift`.
