# Aerial PC xPass validation and local measurements

## Explicit speed-list revision

27 October 2026. New runs use aerial configuration schema 3, with default
speeds 21/23/25/27/29 m/s and elevations 20/25/30 degrees: 15 profiles per
azimuth, independent of the ground speed step. Current usage and compatibility
are documented in [aerial PC xPass](pc_xpass_high_pass.md).

The relevant physical/control/cache/visualization regression suite passed
**989 tests and 49 subtests**. This includes production CLI generation/resume
and actual Windows worker consistency. Ground-only scores, metadata and
fingerprints matched the pre-list Git HEAD exactly across all five endpoint
normalization modes, with and without xT (10 comparisons).

The frozen reference fixture was extended using the original pinned upstream
function, executed separately from the adapted simulator. All 24 historical
arrays were reproduced and preserved exactly; 15 new default trajectories were
added, giving 39 references. Provenance, source/function hashes, coefficients
and generation parameters are in `tests/fixtures/high_pass_upstream.json`.

Both historical range formats retain their recorded grids and ranking semantics
for read-only consumers. Tests cover historical defaults and inherited speed
steps, stored-score/visualization reads, and rejection of numerical work without
changing historical metadata. New-run tests cover canonical list ordering,
custom irregular/single-value lists, obsolete-option rejection, speed-step
independence and shared ground/aerial top-pass groups.

## Separate cutoffs and speed/azimuth grouping revision

This preceding revision used the former 24-profile range grid.

27 October 2026. The relevant physical/control/cache/visualization regression
suite passed **966 tests and 49 subtests** after introducing separate 2 m lane
and 0.5 m control cutoffs and grouping top-pass across elevations. Two additional
cases, with and without xT ranking, verified aggregation when N exceeds the
available speed/azimuth pairs, bringing the checked total to **968 tests**.

Coverage includes exact cutoff boundaries on ascent and descent, independent
lane/control masks with shared underlying formulas, all five endpoint
normalizations, exact speed grouping, stable ties, and xT representative
selection. CLI alias resolution, new-cache generation/resume, legacy stored-score
and visualization reads, and rejection of legacy numerical work are checked.
Windows two-worker output matches serial output with nondefault independent
cutoffs. Production CLI generation/resume and existing lane-sidecar schemas pass.

Ground-only behavior was compared directly against the pre-revision Git HEAD:
all scores, metadata and lane-survival fingerprints matched exactly across the
five endpoint normalization modes, both with and without xT (10 comparisons).

## Initial implementation record

The initial measurements below describe the first implementation, before the
separate cutoffs and speed/azimuth grouping revision. See the
[current behavior and cache compatibility](pc_xpass_high_pass.md).

27 October 2026. The initial implementation used the user-approved configuration: aerial
elevations 15/20/25/30 degrees, total release speeds 20..30 m/s, inherited speed
step, a 2 m inclusive contact cutoff, ground-level launches, zero spin and the
authors' 25 Hz flight/bounce/rolling updates. Ground candidates retain their
existing motion law. The feature defaults off.

## Correctness coverage

The combined physical/control/cache/visualization regression run passed **935
tests and 49 subtests**. A further full production-CLI generation/resume test
passed using temporary benchmark inputs and a temporary cache root, bringing
the checked test count to **936**. That test verifies root/dataset metadata,
schema-2 exports, completed status, inherited settings and reuse without calling
the trajectory simulator. A real-data benchmark CLI dry run also completed.

Disabled-mode behavior was also compared directly with the pre-change Git HEAD
implementation on both saved states, with and without xT ranking. All 504 output
columns per state, metadata and lane-survival fingerprints matched exactly.
The comparison record is `out/pc_xpass_high_pass/legacy_equivalence.json`.

The focused suite covers the entire default grid against frozen output of the
unmodified pinned upstream simulator, including impacts and rolling. Adapted
positions rounded to the source's float32 output match exactly. The tail bound
is checked against a longer source trajectory. Additional checks cover:

- CLI validation, canonical angle lists and speed-step inheritance after resume.
- Contact immediately below/at/above the cutoff and renewed eligibility on descent.
- Combined top-N and distinct top-pass selection under all endpoint normalizations,
  both with and without xT ranking; selected geometry and opponent-lane reconstruction.
- Accelerated TTA, exact-time distance, and empirical reachability.
- Normal/location resume, old-cache compatibility, missing-field repair and mismatched physics.
- Sportec/SkillCorner transactional cache publication and reload, generic cache-only
  visualization, all external visualization adapters, and elevation labels.
- Actual Windows two-worker generation matching single-worker main/lane tables.

The frozen reference fixture is `tests/fixtures/high_pass_upstream.npz`, generated
from revision `cc6af7b00769fc780d890eca49f530c2dce97e4b` with the original
`_simulate_ball_with_spin` function and zero spin. The adapted source and permission
provenance are recorded in `_vendor/montecarlo_pass_search`.

## Saved-state runtime comparison

Two saved benchmark states (modification 1, game states 1 and 2), one computation
worker, 3-degree azimuth grid, 1 m radial grid, ground speeds 3..25 by 2 m/s and
the default 24 aerial profiles. Both use acceleration 7 m/s^2, speed caps 8 m/s,
zero reaction delay, ground quadratic drag, xT ranking, top-N 10/25 and top-pass
5/10. TTA uses a 1 m blocking radius; distance mode uses its standard spatial
allowance and no blocking radius.

| Measurement, both states combined | Ground only | Ground + aerial | Ratio |
|---|---:|---:|---:|
| Warm TTA computation, median of 3 | 0.931 s | 2.565 s | 2.75x |
| Fresh TTA caches incl. lane exports, median of 3 | 1.216 s | 2.916 s | 2.40x |
| Warm distance computation, median of 2 | 3.844 s | 8.311 s | 2.16x |
| Fresh distance caches incl. lane exports, one measurement | 4.216 s | 9.792 s | 2.32x |

The first numerical TTA state took 0.844 s off and 1.470 s on; the first distance
state took 2.140 s off and 4.377 s on. These include numerical initialization and
flight-profile construction but exclude process startup, imports, graph loading
and movement-table preparation. Graph loading took approximately 0.1 s.

The enabled run built 24 profiles once and reused them thereafter. Absolute
process working set sampled every 10 ms peaked around 919 MB off / 1,206 MB on
for TTA, and 891 MB off / 1,199 MB on for distance. The same process measures both
variants, so these are sampled absolute memory values, not isolated incremental
allocations or exact allocation peaks.

Stored-score reuse took approximately 0.45..0.60 s for both states. These raw
benchmark fixtures lack some finite observed-pass auxiliary fields: reuse reports
two copied stored-score rows and zero numerical misses, rather than two complete
cache hits. Reuse timings include that auxiliary handling and should not be
described as pure cache-read latency. Focused tests separately verify complete
cache hits and that cache-only reads never invoke the flight simulator.

Reports and receiver selections:

- `out/pc_xpass_high_pass/benchmark.json` and `predictions.csv` (TTA).
- `out/pc_xpass_high_pass_distance/benchmark.json` and `predictions.csv` (distance).

Reproduce with `scripts/diagnostics/benchmark_pc_xpass_high_pass.py`; see
[usage](pc_xpass_high_pass.md). Temporary prediction caches are removed after
measurement. Normal prediction caches and latest pointers are not modified.
These small measurements do not establish whole-season throughput, parallel
scaling, or predictive accuracy.
