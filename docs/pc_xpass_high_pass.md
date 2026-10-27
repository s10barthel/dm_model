# Aerial candidates for PC xPass

Enable aerial alternatives during generation with `--high-pass on`. The default
is `off`. Ground and aerial options are ranked together for each receiver.

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_v2 `
  --high-pass on --high-pass-angle 15 20 25 30 `
  --high-pass-min-speed 20 --high-pass-max-speed 30 `
  --speed-step 2 --high-pass-lane-cutoff 2 `
  --high-pass-control-cutoff 0.5 --top-pass 5 10
```

Add the usual dataset, frame and movement settings. The same aerial flags
are accepted by `scripts/run_hawkeye_loc.py`. Saved-cache consumers inherit the
selected cache's configuration and require no aerial-generation flags.

| Option | Default | Definition |
|---|---|---|
| `--high-pass {on,off}` | `off` | Add aerial alternatives. |
| `--high-pass-angle FLOAT [FLOAT ...]` | `15 20 25 30` | Elevations above the pitch, degrees. |
| `--high-pass-min-speed FLOAT` | `20` | Minimum total 3D release speed, m/s. |
| `--high-pass-max-speed FLOAT` | `30` | Maximum total 3D release speed, m/s. |
| `--high-pass-speed-step FLOAT` | Resolved `--speed-step` | Aerial speed spacing, m/s. |
| `--high-pass-lane-cutoff FLOAT` | `2` | Maximum lane interception height, metres, inclusive. |
| `--high-pass-control-cutoff FLOAT` | `0.5` | Maximum endpoint control height, metres, inclusive. |
| `--high-pass-cutoff FLOAT` | Unset | Deprecated alias explicitly setting both cutoffs. |

Angles are sorted/deduplicated and must be strictly between 0 and 90 degrees.
Speeds, spacing and both cutoffs must be finite and positive. Minimum speed may equal
maximum speed. A speed grid includes `min + k*step <= max`; step 3 yields
20, 23, 26, 29. Explicit aerial tuning requires `--high-pass on` and all flags
require PC xPass. With step 2 the default grid adds 24 trajectories per azimuth.

The deprecated alias emits a warning and cannot be combined with either new
cutoff option. For example, `--high-pass-cutoff 2` explicitly sets both thresholds
to 2 m, overriding the new 0.5 m control default. New generation settings store
the two resolved values. The Python configuration API follows the same alias
and conflict rules.

## Flight and contact

Aerial passes launch at ground level from the existing hypothetical XY origin,
with zero spin. The attributed Kang simulator uses gravity, fixed per-step air
damping, bouncing and rolling at an internal 25 Hz. Profiles are immutable and
reused within each worker. Dataset recording frequency does not alter that step.
The source, permission provenance, exact revision and adaptations are recorded
in [`_vendor/montecarlo_pass_search`](../_vendor/montecarlo_pass_search/README.md).

Ground passes retain their existing `--ball-dec` law. This option applies only
to ground candidates; aerial passes continue the Kang law through bouncing and
rolling. Aerial profiles stop at 150 m horizontal range or when a conservative
bound on remaining travel is at most 1 mm. Unreached grid distances are masked.

Ball time and height are interpolated at horizontal radial distances. The
existing player movement model uses that time. Every player's raw lane
interception is zero above the lane cutoff, before opponent prefix maxima.
Independently, every player's raw endpoint control is zero above the control
cutoff, before endpoint competition. Normalized receiver control remains zero
there under every normalization mode. Equality is eligible. A ball between
0.5 and 2 m can therefore be intercepted along the lane but cannot provide an
eligible controlled reception at that endpoint. Both cutoffs are unrelated to
learned pass-height labels or blend thresholds.

## Selection, exports and caches

Max and top-N selection operate over all candidate cells. Top-pass retains the
best elevation and endpoint per `(initial total speed, azimuth)` pair, then
selects across those distinct pairs. Ground and aerial profiles with the same
speed compete within one pair. Speeds are compared exactly, without binning:
the default ground grid (3, 5, ..., 25) and aerial grid (20, 22, ..., 30) do not
overlap. Ties retain canonical ground-first, elevation and distance order. xT ranking still
selects by xPass times endpoint xT, so raw selected xPass need not increase.

Enabled wide tables add `<receiver>__elevation`, `__target_z`, and `__flight_time`.
The existing `__angle` remains horizontal azimuth; `__speed` is total release
speed. Ground selections report zero elevation/height. These details describe
the best ranked pass, including when the displayed probability is a top-N mean.
Physical xPass panels label its elevation as `best N°`.

Enabled lane-control sidecars use schema 2 and add `elevation`, `target_z` and
`flight_time`. The product of `1 - lane_control` across opponents still gives
the selected pass's lane survival.

Enabled configurations (schema 2, with separate `lane_cutoff` and
`control_cutoff`), physics coefficients/source checksum, interpolation,
termination and ranking definitions enter cache identities. The aerial top-pass
definition is `best_endpoint_per_speed_azimuth_over_elevation_v3`.
Use a fresh cache ID for this revision or to change either cutoff. Normal
resumes reject incompatible overrides; location
resumes retain their cached-setting precedence. Missing aerial metadata means
off, preserving old ground-only cache identities and output schemas.

Historical aerial caches with a single `cutoff` remain readable for stored-score
inference and visualization. Their shared height cutoff and
`best_endpoint_per_speed_azimuth_elevation_v2` ranking are retained as recorded;
loading does not recalculate or reinterpret their scores. Generation, repair,
resume and numerical reuse of those caches are rejected. The aerial export
columns and lane-sidecar schema 2 are unchanged by this revision.

## Validation and runtime measurement

See the [validation report and local measurements](pc_xpass_high_pass_validation.md)
for test coverage and measured overhead on two saved states.

`tests/test_pc_xpass_high_pass.py` compares the complete default grid against
frozen outputs of the pinned upstream simulator and covers contact eligibility,
ranking, movement modes, cache resume/repair, sidecars and spawned workers.

Run a small comparison without modifying normal prediction caches:

```powershell
python scripts/diagnostics/benchmark_pc_xpass_high_pass.py `
  --states 2 --repeats 3 --cache-repeats 3 `
  --accel-table-id accel7_v8_radius1_refined
```

For exact-time distance control use `--margin distance` and its compatible
`--accel-table-id`. Reports separate source loading, initialization, warmed
computation, fresh cache writes, cache reuse and sampled process memory. The
default output is `out/pc_xpass_high_pass`. Sample measurements are not
whole-season throughput guarantees or predictive-accuracy validation.
