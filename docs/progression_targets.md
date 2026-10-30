# Goal-distance and progression target refresh

`--target-family progression` trains scoring on the fraction of starting distance
eliminated by the selected future position. Conceding reuses
`concedes_goal_distance` without a relative transformation.

For starting distance `d0 > 0`, scoring is
`clip((d0 - d_future) / d0, 0, 1)`. Distance is Euclidean distance to the centre
of the opponent goal. No future teammate candidate, no advancement, or a start
within one micrometre of goal gives zero. Required missing/nonfinite starting
coordinates fail with the match/file/action context.

Supported progression return types are `next_N`, `next_N_skip1`, and `in_N`.
They retain existing goal-distance candidate selection, team perspective, early
shot behavior and boundaries. Carries are labeled training examples; they do not
count as future eligible passes/crosses/shots. Carries retain their return-event
anchor and use their own starting position for progression. Discounted returns
remain valid goal-distance targets, with unavailable (`NaN`) progression values.

## Repair a run after adding a season

Run annotation generation first. Existing aggregate outputs cause the generator
to skip work unless `--overwrite` is supplied:

```powershell
.\.venv\Scripts\python.exe scripts/generate_goal_distance.py --overwrite
```

Then refresh goal distance and progression together:

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id feature_20261029T201116_375036_6a227f77 `
  --refresh-target-family goal_distance `
  --refresh-target-family progression `
  --artifact-copy-mode hardlink
```

Flag order is irrelevant. The command checks annotation and event-input coverage
for every recorded match before cloning. Missing annotations list the affected
match IDs and fail; zero-valued annotations are valid. Generating annotations
alone does not repair labels already stored in a feature run.

The default creates a derived run and prints its ID. It validates every recorded
match and writes only changed artifacts. The failed run
`feature_20261030T170936_969556_c3405c51` is not a source for this recovery.

## Target sources and compatibility

The refresh reads raw goal-distance closeness from per-match sidecars and
calculates each recorded return using the same dispatcher as normal generation.
Sidecar default `next_5` return columns never substitute for run-specific targets.
Saved `pass_endpoints` event chronology is used when available; otherwise the
synchronized event CSV is used. Identities and carry anchors must resolve exactly.

All recorded receiver modes, canonical/carry labels, intent-training, augmented,
and success-intent artifacts are covered. Duplicate augmented action IDs are
supported. Label order, dtype and existing column indices are preserved.

Progression uses the selected label tensor's goal-distance scoring target,
matched to resolved-action starting coordinates through `action_index`.
Missing return values in carry action tables therefore do not prevent progression
when the label tensors are valid. Action-table progression reflects primary-return
training labels; rows with no primary training label remain unavailable.

Standalone commands are supported:

- `--refresh-target-family goal_distance` repairs scoring/conceding and updates
  any progression columns already present. It does not add progression to old
  label tensors.
- `--refresh-target-family progression` verifies the saved goal-distance label
  tensors against the annotations and current return definitions before adding
  progression. Stale/nonfinite values fail with an instruction to use both flags.
  This command never silently repairs goal-distance labels.

New progression columns are appended. Older feature runs remain usable for other
target families; training/evaluation reject unavailable progression targets.
Progression extension requires the label schema containing pass-height columns;
earlier schemas must be upgraded first.

## Storage, reports and failures

Unchanged `.pt`/`.parquet` files are hard-linked. No graph payloads, tracking or
receiver models are loaded. Changed labels/action tables are staged before any
replacement, and each changed file is written once even for a combined refresh.
Input SHA-256 fingerprints and artifact signatures are rechecked before
replacement; annotation/event fingerprints are also checked afterward.

Run target-only operations separately from receiver changes, new return types,
other target families, graph extensions and carry generation. Existing carries
are automatic: do not add `--use-carries` to a refresh command. Existing unrelated
refresh combinations retain their previous behavior.

Add `--in-place` to update the source run explicitly. Avoid concurrent readers
during an in-place refresh. Hard links require the same physical volume and a
supporting filesystem; `--artifact-copy-mode copy` explicitly permits full copies.
There is no automatic copy fallback.

Inspect `metadata.json` for `status: completed`, `goal_distance_targets` and/or
`progression_targets`. `goal_distance_refresh_report.json` contains input hashes,
changed rows, target statistics, rewritten bytes/files and timings. The report's
`source_run_id` identifies the actual refresh source. Repeating a completed
refresh rewrites no data artifacts when targets match.

Failures record match, file, receiver mode, return type, carry variant and example
bad action IDs. `clone_complete` and `target_replacements` distinguish a completed
clone from actual target changes. Replacement is atomic per file, not across the
whole run. Failed or partial derived runs are not usable for training. Staging
failures replace no target files and clean up temporary files; source artifacts
remain protected by atomic replacement of hard-linked outputs.

Prepared dataset caches incorporate source signatures and preparation-code
hashes; no manual cache deletion is needed. Use the completed run ID with normal
training options, for example:

```powershell
.\.venv\Scripts\python.exe scripts/train_relevant_models.py `
  --feature-run-id <completed-feature-run-id> `
  --intended-receiver-mode original `
  --target-family progression --return_type next_3
```

## Isolated pilot

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/pilot_goal_distance_refresh.py
```

The pilot operates on `DFL-MAT-J03WEL` (the added season) and `DFL-MAT-J03YI3`
(already annotated). Its directory is under
`data/features/goal_distance_refresh_pilots`, outside published feature runs and
on the source volume. Missing annotations are generated only inside that isolated
directory. No production annotations, feature runs or latest pointer are changed.

It verifies missing-input rejection, repaired labels, unchanged existing
healthy goal-distance labels, preserved unrelated columns and source files,
retained graph hard links, and no changes on repeated/standalone refreshes.
Graph/tracking loads and receiver inference are prohibited during refresh.
See the generated `pilot_report.json` for the results.
