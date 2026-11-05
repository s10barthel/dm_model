# Logarithmic progression through xT versions

`--target-family xt` supports both fitted xT and analytical logarithmic
progression surfaces. Surface selection belongs to the immutable xT version;
absolute/gain representation and negative-gain clipping belong to the feature
run. Training reads the stored targets without transforming them again.

## Surface generation

```powershell
.\.venv\Scripts\python.exe scripts/generate_xt.py `
  --train-count 765 --version log_progression_k2 `
  --log-progression --k 2 --min-goal-distance 1
```

Use the source feature run's split selector when creating a version for refresh.
Defaults are `k=2` and a minimum goal distance of one metre. Positive finite `k`
values are accepted; values above one increase the lateral-distance penalty.
The minimum distance must be positive, finite and below the maximum distance.

In attacking coordinates, the goal centre is `(105, 34)`:

```
d = sqrt((105 - x)^2 + k * (y - 34)^2)
D = sqrt(105^2 + k * 34^2)
V = log(D / clip(d, d_min, D)) / log(D / d_min)
```

The surface is zero at the farthest pitch corners and one within the minimum
distance of goal. Each halving of weighted distance adds the same value until
the minimum distance is reached. Shots use the surface alone; xG does not
override logarithmic progression values. Fitted xT keeps its existing shot rule.

Logarithmic mode does not fit a model or need training-match events. It retains
split provenance for compatibility with feature runs. It rejects GLM/source-grid
fitting options and writes no artificial source-grid or fit-sample artifacts.
Versions include grids, continuous surface exports, a plot, metadata and match
annotations. Limited exports and `--fit-only` builds remain partial and cannot
be selected as completed versions.

## Target generation and refresh

For a fresh feature run, add these settings to the normal generation command:

```text
--xt-version log_progression_k2
--return_type next_3
--xt-target-representation gain
--xt-clip-negative-gains
```

For an existing compatible feature run:

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <source-feature-run-id> `
  --refresh-target-family xt --xt-version log_progression_k2 `
  --xt-target-representation gain --xt-clip-negative-gains
```

This creates a derived feature run sharing unchanged graph artifacts. The xT
refresh stages target replacements and does not load graphs or receiver
models. Acquisition locations are cached; initial preparation reads only
ball-coordinate tracking columns. It covers canonical, carry, intent-training, augmented and
success-intent labels. Carries use their own starting coordinates and retain
their saved future-return anchors and team perspective.

For version-2 `next_N` variants, let `Bc` be the opponent acquisition-start
surface value (see [turnover returns](turnover_returns.md)).

Let `Ys`, `Yc` denote the selected future teammate/opponent returns, and `B` the
surface value at the labeled action's start:

| Settings | Scoring | Conceding |
| --- | --- | --- |
| `absolute` | `Ys` | `Yc` |
| `gain` | `Ys - B` | `Yc - Bc` |
| `gain` with clipping | `max(0, Ys - B)` | `max(0, Yc - Bc)` |

No opponent candidate means zero conceding. Baselines use the positional
surface directly, including for shots. The return is computed before subtracting
the baseline. Clipping applies to both next-window gains. Historical version-1
gains and non-next returns retain a zero conceding baseline until their applicable
definition changes; xT refresh upgrades next-window gains to version 2.

The default representation is `absolute`, with clipping disabled. Extensions
inherit settings unless explicitly changed. Changing settings requires an xT
refresh. Use `--no-xt-clip-negative-gains` to disable inherited clipping, or
`--xt-target-representation absolute` to return to absolute targets.

These target options also work with learned xT versions. Gain targets require a
pinned version with an evaluable surface; unversioned historical sidecars alone
do not define a baseline.

Logarithmic progression supplies xT targets for `next_N`, `next_N_skip1`,
`next_N+`, `next_N+_skip1`, `extended_possession[_N]`, and `in_N`. Discounted returns remain in the run for other target families; their
`scores_xt` and `concedes_xt` labels are `NaN`. Feature metadata and the refresh
report list them in `xt_unavailable_return_types`. A mixed-return refresh must
create a derived run, preserving the source run's discounted xT labels. A later
refresh to learned xT recalculates every return and clears the unavailable list.

## Training and provenance

Train against the completed refreshed run using the normal options:

```text
--feature-run-id <refreshed-feature-run-id>
--target-family xt --return_type next_3
```

Existing outcome training requires negative-gain clipping for gain targets.
Unclipped gains may be generated for inspection, but signed-model training is
outside this change. No position-based loss weighting is applied.

Feature metadata records `xt_targets` and the pinned surface version/hash.
Model records and bundles preserve that definition, reject mixed xT definitions,
and check runtime feature compatibility. Resuming training cannot change a saved
xT definition. Old feature metadata defaults to absolute targets. Prepared-cache
source signatures already include label files and feature metadata, so refreshes
invalidate affected entries without manual cache deletion.

## Isolated real-data pilot

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/pilot_log_progression.py `
  --source data/features/runs/<completed-feature-run-id> `
  --match-id <match-id> --match-id <another-match-id>
```

The pilot keeps all source returns and requires a supported primary return. It
constructs a private surface and annotations, links a small artifact
subset, and verifies refresh against normal label generation with real tracking.
It checks source signatures and non-xT labels, graph hard-link preservation, and
zero replacements on a repeated refresh. Its report includes pitch-region target
means and zero-target fractions. It never registers a production version or
updates latest pointers. The output must be on the source volume for hard links.

Fresh generation stores base-stop labels by default. Set-piece variants are
opt-in via `--stop-condition set-pieces`; extensions must list their return types.
See [turnover returns](turnover_returns.md) for migration and training commands.
