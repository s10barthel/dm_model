# PC xPass uncertainty validation

Implementation validated on 27 October 2026.

## Numerical and integration checks

Across the physical xPass, PC top-pass/version, aerial, uncertainty, cache/resume,
movement and visualization suites, **1,031 unique tests and 49 subtests passed**.
This combines the 930-test broad regression run, the top-pass/version checks,
and the final 34-test uncertainty module after adding six integration cases.

The uncertainty tests independently enumerate the retained Gaussian outcomes
and compare expected probabilities, xT ranking and top-pass grouping against
the production scorer. They also cover the requested elevation neighbours,
cutoff boundaries, circular angles, zero scales, failures retaining their weight,
and a case where expectation changes the winning intention. Integration checks
include all three movement margins, position discount, serial versus spawned
Windows workers, transactional Sportec/SkillCorner caches, missing-field repair,
and production CLI generation plus cache-only resume.

Disabled mode was compared against `physical_pass_model.py` at commit
`a88d0ed38d02dcbfb9283b09bba8497b584d3681` on benchmark modification 1, states 1
and 2. Ground/aerial and xT on/off give eight comparisons. Every output value
is bit-exact (504 columns ground-only, 576 aerial), and metadata plus lane
fingerprints match exactly. Details: `out/pc_xpass_uncertainty/disabled_equivalence.json`.

## Runtime

The diagnostic uses two saved benchmark states, one worker thread, initial
ground speeds 3–25 m/s at 2 m/s spacing, 3-degree horizontal spacing, 1 m radial
spacing, default aerial candidates (15 profiles), acceleration 7 m/s², maximum
player speed 8 m/s, zero reaction time and ground quadratic drag. Position discount
is off, xT ranking is on, and max plus top-pass 1/5/10 are exported. It alternates
off/on order between warmed repetitions. Prediction caches are not written.

| Movement mode | Repeats | Uncertainty off, seconds per two states | On | Added runtime |
|---|---:|---:|---:|---:|
| TTA, `accel7_v8_radius1_refined` | 5 | 1.909 | 1.953 | 2.3% |
| Distance, `distance_accel7_v8_v1` | 3 | 7.856 | 8.099 | 3.1% |

In the TTA run, sampled absolute process RSS was about 1.125 GB in both modes.
This is a shared-process measurement, not an incremental-memory estimate.
The flight-profile cache contained exactly 15 profiles before and after all
uncertainty computations: averaging generated no additional flight profiles.
Raw measurements: `out/pc_xpass_uncertainty/benchmark.json`.
The distance run used about 1.124 GB absolute RSS in both modes and likewise
retained exactly 15 flight profiles. Its measurements are in
`out/pc_xpass_uncertainty_distance/benchmark.json`.

These small samples verify implementation overhead; they do not establish
whole-season throughput or predictive calibration of the uncertainty scales.
