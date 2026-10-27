# Independent uncertainty export validation

Validated on 27 October 2026 for uncertainty configuration schema 2.

## Behaviour and integration

**1,234 unique tests and 49 subtests passed** across physical/PC xPass, top-pass,
versions, aerial/uncertainty, cache/resume, all movement margins, visualization,
inference startup policy and auxiliary-input reuse. This comprises the 1,233-test
regression suite plus the focused missing-uncertainty-export test. Version resume
checks were also repeated after making the root raw top-pass definition explicit.

Coverage includes:

- Independent Gaussian reference calculations, one-scale support, circular
  angles, zero scales, retained failure weights and trajectory-specific endpoints.
- Bit-exact raw max, raw endpoint top-N, raw top-pass, default aliases and lane
  table equality when uncertainty is enabled; separate uncertainty winner details.
- xT expectation and ranking; ground-only uncertainty height/time diagnostics;
  prefixed elevation and geometry through all saved visualization adapters.
- Coexisting exports, uncertainty-only generation, PC-only selectors, schema-1
  rejection, missing-field repair, no alias fallback for a missing uncertainty
  export, and required-inference startup failure reporting.
- Production CLI generation/resume, transactional Sportec/SkillCorner caches,
  all movement margins and serial versus spawned Windows workers.

## Comparison with the previous implementation

Loaded `physical_pass_model.py` from commit
`5882108be7411978cec398a70f54988e10fbf5b7` and compared benchmark modification 1, states 1
and 2, with ground/aerial and xT on/off: eight cases. All previous raw outputs
are bit-exact both with uncertainty disabled and with the new separate export
enabled. There are 504 ground-only or 576 aerial columns per comparison.
Disabled metadata and lane fingerprints also match exactly.

Details: `out/pc_xpass_uncertainty_independent/raw_equivalence.json`.

## Runtime and memory

Both variants export raw max and raw top-pass 1/5/10. The enabled variant adds
only the separate uncertainty metric and its diagnostics. Settings: two saved
states, one thread, ground speeds 3-25 m/s at 2 m/s spacing, default aerial grid,
3-degree azimuths, 1 m radial spacing, acceleration 7 m/s squared, maximum player
speed 8 m/s, zero reaction time, ground quadratic drag, no position discount and
xT ranking. Initialization is separate; off/on order alternates between repeats.

| Movement mode | Repeats | Off: seconds per two states | On | Added runtime |
|---|---:|---:|---:|---:|
| TTA, `accel7_v8_radius1_refined` | 5 | 2.010 | 2.167 | 7.8% |
| Distance, `distance_accel7_v8_v1` | 3 | 8.022 | 8.049 | 0.3% |

Sampled absolute process RSS was approximately 1.126 GB for TTA and 1.124 GB
for distance in both modes. These are shared-process measurements rather than
incremental-memory estimates. Both modes retain 12 ground plus 15 aerial profiles
per azimuth; uncertainty generates no additional flight profiles. The diagnostic
also verifies raw-score equality and leaves prediction caches untouched.

Measurements:
`out/pc_xpass_uncertainty_independent/benchmark.json` and
`out/pc_xpass_uncertainty_independent_distance/benchmark.json`.

These are small-sample timings with variation between repetitions, not season-wide
throughput or probability-calibration guarantees. Earlier timings for the former
coupled uncertainty mode remain historical and do not describe this export design.
