# pc-xPass deceleration and cache versions

[Aerial candidate generation](pc_xpass_high_pass.md) is optional and defaults
off. Enabled runs retain `--ball-dec` for ground candidates and fingerprint a
separate Kang aerial-physics block, explicit speed/angle lists, 2 m lane cutoff, 0.5 m control cutoff
and ranking definition. Use a fresh cache ID for this revision or changed aerial
settings. Both historical range-based aerial formats remain readable but cannot be
resumed, repaired or numerically reused. Missing aerial metadata means off.
Saved-cache inference and visualization read these results without loading
the flight simulator; normal/location resume precedence follows existing rules.

Optional player acceleration and separate movement-table preparation are described
in [pc_xpass_acceleration.md](pc_xpass_acceleration.md). Acceleration-based
[distance raw control](pc_xpass_distance.md) is available with `--margin distance`.
TTA raw control also supports an optional
[blocking radius](pc_xpass_acceleration.md#tta-with-a-blocking-radius), default 0.

pc-xPass models each speed-grid value as the initial release speed. Select its
motion with `--ball-dec <number|drag>`; the default remains `0.45`.

- A non-negative finite number specifies constant deceleration in m/s^2. `0`
  gives constant-speed passes. The ball never reverses: sampled targets beyond
  its stopping distance are excluded from scoring and ranking.
- `drag` uses straight-line quadratic air drag, with arrival time
  `expm1(k * distance) / (k * initial_speed)`. The fixed paper constants are air
  density 1.22 kg/m^3, drag coefficient 0.25, ball cross-sectional area 0.038 m^2,
  and mass 0.42 kg. Thus `k = 0.013797619047619048 m^-1`. All finite pitch
  distances are reachable for positive initial speeds; zero distance takes zero
  time. This model has no finite stopping distance or travel-time cutoff and
  does not include rolling resistance.

Both `generate_physical_xpass.py` and `run_hawkeye_loc.py` accept this selector.
Accessible Space is unchanged. Drag is an alternative model, not a claim of
improved predictive accuracy; player/control parameters are not retuned.

## Normal caches

```powershell
# Create a fresh version with an automatically generated ID.
python scripts/generate_physical_xpass.py --pc-xpass --ball-dec 0.45

# Create a fresh quadratic-drag version with top-pass 10.
python scripts/generate_physical_xpass.py --pc-xpass --ball-dec drag --top-pass 10

# Create a named version, or resume it with its recorded settings.
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id dec045

# Explicitly select the version for inference.
python scripts/run_hawkeye.py --pc-xpass --pc-xpass-id dec045
```

Each version lives in `data/pc_xpass/<pc_xpass_id>/`, with root metadata and separate
`sportec`, `skillcorner`, `benchmark`, and `hawkeye` subdirectories. The command
prints its generated ID.

Generation without an ID always creates a new version. Resuming an ID inherits
omitted generation settings and rejects explicitly conflicting settings. The
metadata records settings, physics/schema versions, invocation history, coverage,
and completion status. Compatible missing rows can be added to a version.
Numeric caches retain physics version 2 and their existing metadata/fingerprints.
Drag caches use physics version 3, record `ball_dec: "drag"`, and include the
fixed constants and derived coefficient in `ball_drag`. The drag definition is
validated on reuse and included in computation dependencies and lane-survival
fingerprints. Changing between numeric deceleration and drag requires a fresh
cache ID; existing caches are not migrated or regenerated automatically.

Normal inference and training consumers default to the ID in `data/pc_xpass/latest.json`. Successful
generation updates this global pointer; failed, incomplete, and dry runs do not.
A deliberately limited successful run can become latest. If it lacks your dataset,
select an appropriate ID explicitly; consumers do not search older versions.

`--pc-xpass-id` is also available for training with lane-survival features,
evaluation, EPV generation, and visualization. It selects a cache without enabling
additional model features. Outputs record the resolved ID for reproducibility.

## Hawkeye location and freeze caches

```powershell
# Always create a fresh location cache, independently of normal latest.
python scripts/run_hawkeye_loc.py --ball-dec 0.45

# Reuse an existing location cache; compute missing rows using its settings.
python scripts/run_hawkeye_loc.py --pc-xpass-id <location-cache-id>
```

These caches live in `data/pc_xpass/hawkeye_loc/<pc_xpass_id>/`. An explicit ID must
already exist. Cached generation settings override conflicting CLI flags with a
warning, including deceleration and metric-generation settings. Input selections,
geometry, model selection, and execution controls still come from the invocation.
Location runs never advance the normal latest pointer.

Saved-component visualization inherits the cache and metric recorded in the selected
component run, as described below.

## Visualizing saved component runs

All five visualization commands accept `--only-physical-xpass`. It selects only the cached physical view without loading checkpoints or running inference. Other `--only-*` selections are additive and load their own dependencies; existing `--show-*` additions remain available. Combining `--show-physical-xpass` with the new selector produces one physical output. Existing output formats, filenames, and frame suffixes are retained.

For standard Hawkeye, Benchmark, and SkillCorner, no component run is needed when physical xPass is the only selected output. Scenes come from tracking/input data and the existing situation/modification/possession selectors. Supply both an explicit cache ID/directory and an explicit metric; there is no latest-cache or default-metric fallback. If a component run is explicitly supplied, its scene selection and cache/metric inheritance remain in effect. Hawkeye `loc`/`freeze` still requires a matching component export for its row identities and geometry. Sportec still requires feature/resolved-action artifacts and uses the selected pc-xPass cache's recorded feature run when available, unless overridden with `--feature-run-id`. Inference commands keep their existing cache defaults.

```powershell
python scripts/visualize_hawkeye.py --situation-id SITUATION --only-physical-xpass --pc-xpass-id CACHE_ID --xpass-version top25
python scripts/visualize_benchmark.py --modification 1 --only-physical-xpass --pc-xpass-id CACHE_ID --top-pass 10
python scripts/visualize_skillcorner.py --match-id MATCH --index 1 --only-physical-xpass --physical-cache-dir CACHE_DIR --xpass-version max
```

Physical-only output requires an existing cache and rejects `--no-physical-cache`. Hawkeye `--only-tracking` conflicts with it. Saved-run comparisons support physical-only selection; checkpoint/mask comparisons need at least one model-backed output as well.

`visualize_hawkeye.py` (standard, location, and freeze), `visualize_benchmark.py`,
and `visualize_skillcorner.py` support only pc-xPass for `--show-physical-xpass`.
They inherit the selected component run's cache ID or recorded dataset cache
directory and metric selector. They never default to latest or `top10`.

```powershell
# Display the same cached pc-xPass metric used by the saved component run.
python scripts/visualize_hawkeye.py --component-run-id COMPONENT_RUN --show-physical-xpass

# Change only the metric; retain the component's cache.
python scripts/visualize_hawkeye.py --component-run-id COMPONENT_RUN --show-physical-xpass --top-pass 10

# Explicitly select a cache and metric, including for a run that did not use pc-xPass.
python scripts/visualize_hawkeye.py --component-run-id COMPONENT_RUN --show-physical-xpass --pc-xpass-id CACHE_ID --xpass-version top25

# Select a legacy pc-xPass dataset directory explicitly.
python scripts/visualize_hawkeye.py --component-run-id COMPONENT_RUN --show-physical-xpass --physical-cache-dir data/pc_xpass/hawkeye --xpass-version max
```

An explicit cache ID or directory replaces the inherited cache selection. A
cache override retains the component's metric unless `--xpass-version` or
`--top-pass` is also supplied. Explicit ID and directory overrides cannot be
combined. `--pc-xpass` remains accepted for compatibility and is optional;
`--top-pass` works without it. Supported metrics are `max`, `top<N>`, and
`top-pass<N>`; ordinary physical xPass and `noise-kernel` are rejected.

If the component did not use pc-xPass, used ordinary physical xPass, or has
insufficient historical metadata to establish usage, provide both an explicit
cache ID/directory and an explicit metric. Merely recording cache defaults or
using a cache for lane-survival features does not establish pc-xPass usage.
For known pc-xPass runs with incomplete settings, supply the missing settings.
Structured historical settings and legacy recorded directories are supported;
command strings and external model checkpoints are not used to guess settings.

Recorded relative directories use the saved invocation working directory when
available, otherwise the project root. Explicit relative directories use the
current working directory. Missing caches, unsupported formats or metrics, and
missing rows fail without selecting another cache or generating data.

Saved-run comparisons resolve each variant's settings independently. CLI
overrides apply to every variant. Console output and visualization metadata
record the selected cache, metric, inherited fields, and explicit overrides.
Overrides affect only the physical xPass panel; saved component predictions
are not recomputed. Outside cache comparison mode, these scripts do not access the
physical-panel cache without `--show-physical-xpass`, `--only-physical-xpass`,
`--compare-pass-success`, or `--only-compare-pass-success`.
Scripts that run inference retain their own selection behavior.

To vary caches instead of component/checkpoint runs, all five visualization commands
accept `--compare --pc-xpass-id CACHE_A --pc-xpass-id CACHE_B` (repeat the singular
selector). At least two distinct IDs are required, in display order. `--pc-xpass`
is implied; one metric and all supplied runtime settings apply to every cache.
Use `--show-physical-xpass` or `--only-physical-xpass` for raw panels. In inference
commands, `--use-physical-xpass` also recomputes blended pass success and derived
pass score. The selected cache supplies all lane-survival consumers, including
auxiliaries; existing compatibility checks remain in force. Saved visualizers load
one component source once, inherit its metric unless overridden, and leave stored
predictions unchanged. Physical-only scene discovery follows standalone behavior,
including the saved geometry requirement for Hawkeye location/freeze modes.

Cache comparisons are PNG-only and cannot vary models/components simultaneously,
use directory overrides or `--no-physical-cache`, or combine with numbered masks,
tracking-only output, `--compare-pass-success`, or `--only-compare-pass-success`.
One shared unnumbered mask is supported by inference commands. Every cache is
preflighted for the dataset and metric; missing rows or required auxiliary columns
fail with the offending cache ID and scene instead of dropping frames. No fallback
or generation occurs. Comparison metadata records `comparison_mode: pc_xpass`,
fixed models/source, shared CLI settings, and each cache's resolved provenance.
Evaluation commands are unchanged.

The two inference visualizers, `run_and_visualize_hawkeye.py` and
`visualize_action_components.py`, also compare versions or blend weights within
one explicit `--pc-xpass-id`. Exactly one selector may vary: cache ID, version,
or weight. Repeat the singular flags for two or more distinct values; their
command-line order is the panel order. `--pc-xpass` is implied. Unvaried version
and weight defaults remain `top10` and `v3`.

```powershell
# Compare three cached metrics without loading learned models.
python scripts/run_and_visualize_hawkeye.py --compare --pc-xpass-id CACHE_ID `
  --xpass-version max --top-pass 10 --xpass-version uncertainty `
  --situation-id SITUATION --only-physical-xpass

# Compare blended success with fixed cache, metric, and auxiliary checkpoints.
python scripts/run_and_visualize_hawkeye.py --compare --pc-xpass-id CACHE_ID `
  --xpass-version max --xpass-weight v5 --xpass-weight v4 `
  --use-physical-xpass --pass-success-model-id pass_success/MODEL `
  --xpass-pass-height-model-id pass_height/HEIGHT `
  --xpass-pass-intent-model-id pass_intent/INTENT `
  --situation-id SITUATION --only-pass-success --show-physical-xpass
```

`--top-pass N` / `--top_pass N` contributes `top-passN` to the same ordered
version list as `--xpass-version`. Counts may repeat with different values.
Select uncertainty with `--xpass-version uncertainty`; `--uncertainty true/false`
remains a generation option. Every selected metric must already exist in the
cache with compatible metadata. Normalized duplicate selections, repeated
selectors without `--compare`, and combinations varying multiple settings are
rejected. For example, two cache IDs plus two versions are invalid, as are
`--top-pass 10 --xpass-version top-pass10` in comparison mode.

Weight comparisons require explicit `--use-physical-xpass` and a selected final
pass-success or pass-score output. Raw physical panels alone cannot compare
weights: their values are identical across weight variants. v4/v5 require one
explicit height auxiliary; v5 also requires an explicit intent auxiliary. Each
auxiliary loads once and is used only by variants needing it. Selectors unused
by every variant are rejected. v4 tuning flags apply only to v4 variants; v5
flags apply only to v5 variants, and both sets may be supplied when comparing
those weights. Other comparison restrictions above also apply.

Inference comparison metadata retains `comparison_mode: pc_xpass` and records
`comparison_dimension` as `cache_id`, `version`, or `weight`. New variant IDs are
`version:<normalized-version>` and `weight:<weight>`; cache comparison IDs are
unchanged. Each variant records its cache, version, metric, weight, and effective
model/auxiliary settings. Errors identify those settings and the failing scene.
Saved-component visualizers retain their existing cache-only comparison behavior.

All five visualization commands support `--compare-pass-success`, which stacks
the existing model pass-success, pc-xPass, and final pass-success images into one
PNG. `--only-compare-pass-success` selects only that combined output. Both modes
load pc-xPass automatically without `--show-physical-xpass`, using an explicit
cache ID/directory or the saved component run's provenance. No latest-cache
fallback is allowed. Saved-run metric inheritance and overrides work as above;
inference commands retain their existing metric defaults. Blending settings and
saved final predictions are unchanged. These modes reject `--compare`, numbered
inference masks, and GIF/MP4 output.

```powershell
python scripts/visualize_hawkeye.py --component-run-id COMPONENT_RUN --only-compare-pass-success
python scripts/visualize_benchmark.py --component-run-id COMPONENT_RUN --pc-xpass-id CACHE_ID --xpass-version top25 --compare-pass-success
```

## Existing caches and directory overrides

Existing unversioned caches remain in place and are not registered as latest.
Select them explicitly, for example:

```powershell
python scripts/run_hawkeye.py --pc-xpass --physical-cache-dir data/pc_xpass/hawkeye
python scripts/run_hawkeye_loc.py --pc-xpass-cache-dir data/pc_xpass/legacy_location
```

Do not combine an ID with a cache-directory override. Legacy metadata without
deceleration is interpreted as constant speed; location generation through that
explicit directory retains zero deceleration. No automatic migration occurs.

## Consistent physical resumes

Resume with `--pc-xpass --pc-xpass-id <id>`. Omitted calculation settings and
`--export-lane-control` inherit recorded values; explicit conflicts require a new
version. Physics and ranking dependencies are validated. Sportec pins the resolved
feature run and source/reconstruction provenance; Hawkeye pins `--freeze-ballreceipt`
(initial default: true). Contracts and reconstruction sources are saved before
computation. Paths may move if source identities and reconstructed hashes match.

Data selection remains flexible: season, match/situation IDs, dataset switches,
limits, `--scope`, `--frames`, and Hawkeye times are not frozen or automatically
restored. Workers, device, and batch/window sizes may change. Repeat the original
selection to finish the same requested coverage, or change it to extend the cache.
Partition/source coverage accumulates, and sampling reports are saved per invocation.

Older Sportec cache formats require a fresh version. Saved states in the current
format are reconstructed and checked before new partitions are added; unknown
or incompatible sources cause an error.
Ctrl+C marks the invocation incomplete where possible. Saved rows remain reusable
if abrupt shutdown leaves status marked running.

PC-xPass generation is independent of learned models. It does not load height checkpoints, write height predictions, expand metrics from model requirements, or refresh height fields. Height options are rejected during generation.

New generation and resumes use `physical_input_contract = pc_xpass_physical_inputs_v1`. Sportec uses `sportec_possession_states_v3` / `match_physical_states_v3`: state identity retains match, period, frame, actual possessor, and source/reconstruction fingerprints, without learned graph options. A fresh Sportec cache is required; existing caches are left intact and no migration is provided. Other compatible physical cache formats remain readable. New generation resumes require the physical-input contract.

## Height and intent auxiliaries

For active v4 blending and height-weighted evaluation, supply `--xpass-pass-height-model-id pass_height/<id>`. V5 additionally requires `--xpass-pass-intent-model-id pass_intent/<id>`, even with `--v5-discount false`. V1-v3 reject unused auxiliary selectors. Ordinary component selectors and bundles do not select these auxiliary roles.

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id states_v3 --sportec-feature-run-id FEATURE_RUN --top-pass 10 --scope actions --no-skillcorner --no-benchmark --no-hawkeye
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id FEATURE_RUN --pc-xpass --use-physical-xpass --pc-xpass-id states_v3 --xpass-version top-pass10 --xpass-weight v5 --xpass-pass-height-model-id pass_height/HEIGHT --xpass-pass-intent-model-id pass_intent/INTENT
```

Auxiliaries use separate original-state graph copies with their own features, goal policies, masks, and lane-survival requirements. Alignment uses physical state and candidate player, including canonical action associations. Height uses independent sigmoid probabilities. Hidden components still supply required predictions. Required numerical fields must already exist: for example `top_pass_10` requires generation with `--top-pass 10`.

`--pass-height-threshold` asserts the selected auxiliary checkpoint's cutoff for blending; it cannot reinterpret probabilities. Unknown historical cutoffs remain unknown and cannot satisfy an explicit assertion. Missing, ambiguous, or nonfinite required auxiliary predictions raise errors, with no cached-height fallback or silent sample removal. Unavailable physical-state behavior remains unchanged. Original training labels, carry eligibility, and evaluation observed-target conventions are preserved.

Outputs record auxiliary checkpoints, effective configurations/masks, height definitions, and input provenance. Predictions may be reused within a run for identical dependencies; there is no persistent auxiliary prediction cache. Obsolete height fields and pending height-refresh metadata do not affect physical reads or completion.

Runtime auxiliaries apply the checkpoint's standard feature preparation. The old generation enrichment helper did not apply all of that preparation to raw graphs, so runtime height probabilities can differ from obsolete cached height columns. With identical prepared inputs, height predictions and blending formulas are unchanged; numerical physical results are unaffected.

## Sportec and SkillCorner match checkpoints

Both `--scope actions` and `--scope frames` write one parquet per actual match:

```text
<dataset>/matches/<match_id>.parquet
<dataset>/lane_control/<match_id>.parquet  # with --export-lane-control
<dataset>/completion/<match_id>.json
```

Sportec declares `cache_format: "match_physical_states_v3"` and state contract
`sportec_possession_states_v3`. SkillCorner retains `match_possession_frames_v1`
and `possession_states_v1`. All older Sportec formats require regeneration into
a new cache version, without conversion or fallback. Benchmark and Hawkeye retain
their existing formats.

SkillCorner rows contain actual `match_id`, integer `possession_id`, `possessor_id`,
`state_frame_id`, and `possession_provenance_id`. Their key is match, possession,
and frame; lane rows additionally identify receiver and opponent.

Sportec rows additionally contain `state_id`, `period_id`, and `state_partition_id`.
The physical identity is match, period, frame, actual possessor, and reconstruction
provenance. Workers group states by period and actual possessor and keep frame IDs
as action indexes. The internal `possession_id` column aliases the worker partition;
it is **not** a semantic spell ID. Semantic spell IDs, owners, nullable endpoint
references, successors, and canonical action associations live in the per-match
manifest, embedded in both parquet metadata and the completion record. One physical
state can serve multiple roles, including the end of a lost carry and the opponent's
next acquisition.

Sportec source fingerprints cover canonical events, synchronized controls, canonical
resolved action fields, tracking, lineups, graph options, and the reconstruction
version. Training-only carry columns do not change physical state identity.
`--scope actions` includes exact spell endpoints and canonical pre/post observations;
`--scope frames` adds strided interior states only where control is supported.
Invalid canonical endpoints stay missing; nearby frames are not replacements.

Results are buffered in memory and published once per completed match. Interrupted
or failed matches lose unsaved work; compatible completed rows remain reusable.
Changing scope or stride adds coverage without removing existing physical rows.
Worker counts, numerical calculations, and batching policy are unchanged.

Completion records contain requested coverage, row counts, output checksums, and
reconstruction metadata. Publication marks a match pending, replaces its outputs,
and marks it completed. Readers reject pending or inconsistent commits. Resume the
original request to recompute an unfinished match. Height refresh preserves the
published spell manifest and lane-control data.

Inference validates provenance, selects its physical partition, and looks up the
frame. Sportec training and evaluation obtain explicit canonical action views through
the manifest, mapping event action indexes to physical rows. Their existing labels
and carry handling are unchanged. Sportec visualization resolves the exact canonical
pre-action observation, including partial spells, and fails when that state is
invalid. It does not guess by frame proximity.

Sportec v3 readers can reuse an existing physical partition across feature runs
when the runtime descriptor matches the cached descriptor in every field except
`feature_run_id`. This requires complete source/reconstruction fingerprints and
matching physical-input/state contracts, match, partition, period, and possessor.
Extension lineage alone does not establish compatibility. For example, an xT-only
target refresh can reuse physical results when the state-defining content is
unchanged. Ambiguous matches and changed or incomplete fingerprints are rejected.
The shared lookup covers inference, visualization, and lane-control reads while
retaining completion, checksum, and coverage checks. Returned rows keep their
cached identities; `pc_xpass_cache_resolution` frame attributes record exact or
equivalent resolution, both feature-run IDs, and requested/resolved provenance
keys. Existing files, runtime state IDs, and manifests are not rewritten. Generation
and resume publication remain strict: cross-run equivalence does not authorize
appending to or changing an existing cache version.

Component inference exports `possession_manifest.json`. Ranking resolves spell
start/end/next roles through this manifest, never through a shift of filtered pass
rows. Unknown successors stay missing; restart outcomes are retained and goals/period
boundaries terminate links. Opponent values are converted to the original actor's
perspective. See [the Sportec validation notes](sportec_possession_states.md).


## Pass-height separation validation (2026-10-12)

The focused regression set passed 1,037 tests and 51 subtests in about one minute.
It covers physical generation/resume, atomic checkpoints, Sportec reconstruction,
canonical lookups, lane coverage, auxiliary selectors, independent graph preparation,
candidate alignment, blending/evaluation, visual components, and training graphs.

A bounded check used two saved Sportec graphs from `DFL-MAT-J03YDU` and checkpoint
`pass_height/pass_height_20260703T120403_205360_c72e946d`. All 22 height outputs
matched the old helper on identically prepared inputs within `3e-11`. Physical
results were exactly equal with the original and minimal edge schemas. Cold
creation wrote two rows without height fields; resume reused both rows without
physical recomputation. The existing goal-node nearest-opponent sidecars were
refilled on resume; scheduling and that behavior were left unchanged.

On raw, unprepared graphs the old helper differed by up to `0.236` in this small
fixture. Runtime predictions now follow the checkpoint's standard preparation.
The smoke computation took under two seconds after imports. Existing caches were
not modified. SkillCorner, Hawkeye, Benchmark, and relocated Hawkeye were covered
by automated fixtures, without real-data runs. No full rebuild or benchmark ran.
