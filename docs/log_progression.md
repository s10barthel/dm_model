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
refresh stages target replacements and does not load graphs, tracking, or
receiver models. It covers canonical, carry, intent-training, augmented and
success-intent labels. Carries use their own starting coordinates and retain
their saved future-return anchors and team perspective.

Let `Ys`, `Yc` denote the selected future teammate/opponent returns, and `B` the
surface value at the labeled action's start:

| Settings | Scoring | Conceding |
| --- | --- | --- |
| `absolute` | `Ys` | `Yc` |
| `gain` | `Ys - B` | `Yc` |
| `gain` with clipping | `max(0, Ys - B)` | `Yc` |

The opponent's baseline is zero because it initially has no possession. Starting
baselines use the positional surface directly, including for shots. No future
candidate means a zero future value. The return is computed before subtracting
the baseline. Negative scoring gains are then optionally clipped.

The default representation is `absolute`, with clipping disabled. Extensions
inherit settings unless explicitly changed. Changing settings requires an xT
refresh. Use `--no-xt-clip-negative-gains` to disable inherited clipping, or
`--xt-target-representation absolute` to return to absolute targets.

These target options also work with learned xT versions. Gain targets require a
pinned version with an evaluable surface; unversioned historical sidecars alone
do not define a baseline.

**Initially, logarithmic progression supports only `next_N`, `next_N_skip1`, and
`in_N`.** A refresh checks every recorded return type before modifying artifacts.
A source containing discounted returns cannot be refreshed to logarithmic
progression in this implementation. Generate a feature run containing supported
returns, or continue using its learned-xT version. Learned xT retains its existing
return types.

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

The pilot selects supported returns from the source, preserving its primary
return. It constructs a private surface and annotations, links a small artifact
subset, and verifies refresh against normal label generation with real tracking.
It checks source signatures and non-xT labels, graph hard-link preservation, and
zero replacements on a repeated refresh. Its report includes pitch-region target
means and zero-target fractions. It never registers a production version or
updates latest pointers. The output must be on the source volume for hard links.
