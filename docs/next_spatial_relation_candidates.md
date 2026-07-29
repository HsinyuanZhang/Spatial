# Next 5-bit Spatial-Relation Candidates

Status: **divergence and convergence completed after the masked-L1 pilot;
power-of-two weighted L1 selected for the next bounded test**

## Starting point

The project is not searching for another generic spike-sorting feature. The
specific unresolved tension is:

> HJ benefits strongly when each unit suppresses its unstable 5-bit spatial
> coordinates, but binary suppression creates don't-care directions and
> increases MEArec collisions.

The next relation must therefore retain unit-specific feature relevance
without reopening an unbounded learned model or abandoning a bit-exact
near-memory mapping.

## Divergence

Failure analysis, abstraction, composition/decomposition, and the simplicity
test produced the following raw candidates. Rejected items remain recorded so
that later negative results do not trigger an undocumented repeat.

| ID | Candidate | Low-bit memory relation | Main risk |
|---:|---|---|---|
| 1 | Ranked power-of-two weighted L1 | 2-bit shift code/dimension plus centroid/radius | all nine lanes still toggle |
| 2 | Quantized inverse-dispersion L1 | per-unit reciprocal scale approximated by shifts | scale normalization/tuning leakage |
| 3 | Ternary weighted L1 | weights `{0,1,2,4}` | reintroduces hard don't-care collisions |
| 4 | Two complementary masks | union of two masked rows/unit | repeats multi-row storage/union broadening |
| 5 | Unit-specific 45-bit allocation | more bits on selected coordinates, fewer on others | variable decoder and cross-row comparison |
| 6 | Residual-sign CAM signature | learned positive/negative centroid residual bits | Hamming relation may lose magnitude |
| 7 | Thermometer residual code | per-dimension distance threshold bits | storage expands rapidly |
| 8 | Compressed per-feature boxes | shared centroid plus quantized asymmetric widths | close to the rejected 90-bit Range-CAM box |
| 9 | Rank-only feature ordering | compare ordinal nearest coordinates | weak metric semantics and sorting cost |
| 10 | Learned Boolean spatial hash | short unit code tested by Hamming/CAM | training complexity and explainability |
| 11 | Product-quantized descriptor | local subspace codebooks | codebook/WTA cost and nonuniform macros |
| 12 | Temporal descriptor delta | present 5-bit footprint minus prior event/unit state | online state/confidence dependency |
| 13 | Short temporal sketch | 5-bit waveform/spatial slope or energy bits before Level 2 | causal window and acquisition latency |
| 14 | Channel-handoff state bit | unit row conditioned on neighboring central channel | state multiplication and directory coupling |
| 15 | Tile-conditioned descriptor basis | different fixed feature ordering per physical tile | needs realistic high-density 2D data |
| 16 | Power-of-two diagonal Mahalanobis score | squared residuals with shift weights | multiplier/square cost defeats simple CiM |
| 17 | Cross-multiplied normalized top-L | compare `d_u/(r_u+1)` without division | ranking/WTA does not directly fix broad candidates |
| 18 | Pairwise unit contrast sketch | a few fit-selected feature differences per confusion pair | O(U²) state repeats pair-tap weakness |

## Convergence filters

Candidates were filtered by five constraints:

1. **Failure fidelity:** directly address hard don't-care collisions rather
   than adding another row or radius.
2. **Simplicity:** fixed integer operations, no learned neural network,
   floating-point scale, or test-selected bit width.
3. **Hardware honesty:** row payload and arithmetic can be stated before
   results; no implicit multiplier, divider, ADC, or analog precision claim.
4. **Falsifiability:** the same four-recording development gate can reject the
   idea cheaply before a 32-recording sweep.
5. **Path to DAC evidence:** if it passes, it can produce bit-exact RTL vectors
   and a clear comparison with 54-bit unweighted L1 and 63-bit masked L1.

The strongest five are:

| Rank | Candidate | Why retained | Why not first/why first |
|---:|---|---|---|
| 1 | Ranked power-of-two weighted L1 | preserves every coordinate, directly interpolates between all-dims and binary mask, shift-add mapping | **selected** |
| 2 | Short causal temporal sketch | changes the missing information, not just the range | larger experiment; event timing/filter boundary first |
| 3 | Tile-conditioned descriptor basis | likely relevant to the final 1024-channel architecture | blocked on realistic high-density local-unit data |
| 4 | Residual-sign CAM signature | compact native CAM relation | needs a separate collision/bit-length design |
| 5 | Unit-specific bit allocation | directly targets a fixed row-bit budget | variable precision complicates fair arithmetic and macro layout |

## Selected idea

### Two-sentence pitch

Binary masks help HJ but create unlimited distance tolerance along disabled
coordinates, which increases dense-unit MEArec collisions. Store a 2-bit
power-of-two weight code for each of the nine 5-bit coordinates, keep every
weight nonzero, and implement the unit-specific weighted L1 distance with
shifts and additions instead of multipliers.

### Core tension

The selected relation spends 20 additional logical bits over a 54-bit
unweighted row and keeps all nine absolute-difference lanes. It is justified
only if preserving weak evidence from unstable dimensions retains the HJ
recall movement while preventing MEArec candidate broadening and materially
reducing 320-bit waveform-template traffic.

### Abstraction level

- Up: unit-specific diagonal metrics under a strict digital memory contract.
- Current: ranked shift-weighted L1 over a nine-dimensional 5-bit descriptor.
- Down: two fixed rank profiles, two fit-only score modes, three calibrated
  radii, and four development recordings.
- Sideways: operand-strength scheduling resembles quantized attention, but the
  implementation is a static row word rather than a neural attention module.

## Three validation experiments

1. **No-I/O semantics:** prove integer boundaries, nonzero weights,
   deterministic ranks/ties, calibration isolation, maximum accumulator
   width, active-row behavior, and uniform-weight equivalence.
2. **Representative all-row pilot:** run the same two HJ and two MEArec
   development recordings used to falsify binary masks, with a formal
   per-record and cross-family gate.
3. **Disjoint confirmation:** only if one weighted configuration passes,
   freeze it and run the remaining ten HJ and eighteen MEArec recordings
   without directory tuning; require 0.99 recall and 4× traffic separately in
   both confirmation families.

## Two-week feasibility pilot

The bounded schedule is:

```text
days 1-2    freeze bit contract, ranks, profiles, split, fallback, gate
days 3-5    implement core and exact no-I/O tests
days 6-8    implement runner and hardware/access counters
days 9-10   run four-recording pilot and audit paired deltas
days 11-12  if pass, run one frozen disjoint confirmation configuration
days 13-14  document result, provenance, and RTL/macro decision
```

The repository automation should complete the development portion faster, but
the decision boundary remains identical.

## Strongest objection

**Objection:** nonzero weights eliminate masked-L1 lane savings, add shift
control and a wider accumulator, and may merely rescale the same broad
candidate frontier.

**Response:** that is the central kill condition, not an implementation detail.
The pilot separately counts row comparisons, all nine absolute-difference
lanes, shifted lanes, additions, configuration bits, and waveform traffic.
The branch stops if it does not move both HJ and MEArec in the same direction;
no lane-energy advantage is assumed.

## Deferred order after this test

If weighted L1 fails, do not add more score temperatures or weight levels.
The next scientifically distinct branch is a short causal temporal sketch,
followed by tile-conditioned evaluation once realistic high-density 2D data
exists. RTL/PPA work should proceed on stable primitives and baselines, but a
new weighted macro is justified only after the representation gate.
