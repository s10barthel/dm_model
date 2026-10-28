# Aerial candidates for PC xPass

Enable aerial alternatives during generation with `--high-pass on`. The default
is `off`. Ground and aerial options are ranked together for each receiver.
Optional [launch uncertainty](pc_xpass_uncertainty.md) adds a separate expected
success metric using existing trajectories; raw max/top-pass exports stay independent.

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_v3 `
  --high-pass on --high-pass-angle 20 25 30 `
  --high-pass-speed 21 23 25 27 29 --high-pass-lane-cutoff 2 `
  --high-pass-control-cutoff 0.5 --top-pass 5 10
```

Add the usual dataset, frame and movement settings. The same aerial flags
are accepted by `scripts/run_hawkeye_loc.py`. Saved-cache consumers inherit the
selected cache's configuration and require no aerial-generation flags.

| Option | Default | Definition |
|---|---|---|
| `--high-pass {on,off}` | `off` | Add aerial alternatives. |
| `--high-pass-bounce {on,off}` | `on` | Continue after first ground contact; `off` ends the trajectory at that contact, inclusive. |
| `--high-pass-angle FLOAT [FLOAT ...]` | `20 25 30` | Elevations above the pitch, degrees. |
| `--high-pass-speed FLOAT [FLOAT ...]` | `21 23 25 27 29` | Total 3D release speeds, m/s. |
| `--high-pass-lane-cutoff FLOAT` | `2` | Maximum lane interception height, metres, inclusive. |
| `--high-pass-control-cutoff FLOAT` | `0.5` | Maximum endpoint control height, metres, inclusive. |
| `--high-pass-cutoff FLOAT` | Unset | Deprecated alias explicitly setting both cutoffs. |

Angle and speed lists are nonempty, sorted and deduplicated. Angles must be
finite and strictly between 0 and 90 degrees. Speeds and both cutoffs must be
finite and positive. Speeds may be irregularly spaced or contain a single value,
for example `--high-pass-speed 19.5 24 28`. Ground `--speed-step` does not affect
aerial speeds. Explicit aerial tuning requires `--high-pass on` and all flags
require PC xPass. The defaults add 15 trajectories per azimuth.

The former `--high-pass-min-speed`, `--high-pass-max-speed` and
`--high-pass-speed-step` options have been removed and are rejected. The Python
configuration API accepts `high_pass_speed` and no longer accepts range or step
parameters; arguments after the enable/angle parameters are keyword-only.
It also accepts `high_pass_bounce="on"` or `"off"`, defaulting to `"on"`.

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
to ground candidates; by default aerial passes continue the Kang law through
bouncing and rolling. With `--high-pass-bounce off`, each aerial profile ends at
the first post-launch sample whose height is zero, including that sample. The
initial ground-level launch sample does not count as contact. Contact uses the
existing 25 Hz simulator, without estimating a sub-step impact time. The radial
grid is unchanged: first contact is eligible when it coincides with an evaluated
grid distance; all greater distances are unreachable for that aerial profile.
This boundary applies to receptions, lane interceptions, ranking and uncertainty.

For example, add `--high-pass-bounce off` to a new aerial generation run:

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_first_contact `
  --high-pass on --high-pass-bounce off --top-pass 5 10
```

In both modes, aerial profiles stop at 150 m horizontal range or when a conservative
bound on remaining travel is at most 1 mm. Unreached grid distances are masked.
The existing maximum-step protection also applies.

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
the default ground grid (3, 5, ..., 25) shares speeds 21, 23 and 25 with aerial
profiles. Ties retain canonical ground-first, elevation and distance order. xT ranking still
selects by xPass times endpoint xT, so raw selected xPass need not increase.

Enabled wide tables add `<receiver>__elevation`, `__target_z`, and `__flight_time`.
The existing `__angle` remains horizontal azimuth; `__speed` is total release
speed. Ground selections report zero elevation/height. These details describe
the best ranked pass, including when the displayed probability is a top-N mean.
Physical xPass panels label its elevation as `best N°`.

Enabled lane-control sidecars use schema 2 and add `elevation`, `target_z` and
`flight_time`. The product of `1 - lane_control` across opponents still gives
the selected pass's lane survival.

Enabled configurations (schema 3, with explicit `angles`/`speeds` lists and
separate `lane_cutoff`/`control_cutoff`), physics coefficients/source checksum, interpolation,
termination and ranking definitions enter cache identities. The aerial top-pass
definition is `best_endpoint_per_speed_azimuth_over_elevation_v3`.
Schema 3 also supports an optional `bounce` field. Missing means `on`; canonical
bounce-on configurations and generation settings omit the default, preserving
existing bounce-on cache identities and resumes. Bounce-off configurations store
`bounce: "off"` and physics termination `first_ground_contact_inclusive_v1`;
generation settings store `high_pass_bounce: "off"`.
Use a fresh cache ID when changing bounce mode, speed, angle or cutoff. Normal
resumes reject incompatible overrides; location
resumes retain their cached-setting precedence. Missing aerial metadata means
off, preserving old ground-only cache identities and output schemas.

Both historical range-based aerial formats remain readable for stored-score
inference and visualization. Schema 2 retains its separate cutoffs and current
speed/azimuth ranking. The original single-`cutoff` format retains its shared
height threshold and `best_endpoint_per_speed_azimuth_elevation_v2` ranking.
Historical grids are validated using their recorded settings and original
defaults; loading does not recalculate, reinterpret or rewrite scores or
metadata. Generation, repair, resume and numerical reuse of either historical
format are rejected. Use a fresh cache ID for list-based generation. The aerial
export columns and lane-sidecar schema 2 are unchanged by this revision.

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
