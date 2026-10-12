# pc-xPass deceleration and cache versions

pc-xPass models each speed-grid value as the initial release speed. The ball slows
at `--ball-dec` metres per second squared (default `0.45`). Use `--ball-dec 0` for
constant-speed passes. The ball never reverses: sampled targets beyond its stopping
distance are excluded from scoring and ranking. Accessible Space is unchanged.

## Normal caches

```powershell
# Create a fresh version with an automatically generated ID.
python scripts/generate_physical_xpass.py --pc-xpass --ball-dec 0.45

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

Normal consumers default to the ID in `data/pc_xpass/latest.json`. Successful
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

Location visualization inherits the cache ID recorded in the selected component
run. An explicit `--pc-xpass-id` or cache-directory override takes precedence.

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
