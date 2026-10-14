# Targeted xT refresh

`--refresh-target-family xt` refreshes existing xT targets, including carries,
without loading graphs or tracking data or invoking receiver inference. All
recorded receiver modes and return types are included.

## Storage and safety

Derived extensions, including carry extensions, default to hard-linking unchanged
`.pt` and `.parquet` files. Metadata, logs and diagnostics remain independent.
Hard links require a supporting filesystem on the same physical volume. The
implementation probes the actual directories, including through Windows
junctions. There is no automatic copy fallback. `--artifact-copy-mode copy`
explicitly opts into a full copy.

Repository feature writers save to a neighboring temporary file, close it, then
replace the destination. This detaches changed files from their other hard links,
protecting both source and derived runs. Deleting one run also leaves the other
run's files available. Manual edits and external programs that overwrite shared
files in place bypass this protection; use the atomic save helpers instead.

The refresh validates the artifact inventory, then stages all changed targets
before replacing any data artifact. It rechecks event/annotation fingerprints and
source artifact signatures before replacement, and input fingerprints afterward.
Temporary space is needed for changed labels and action tables, rather than
another copy of hundreds of GiB of graphs. Unchanged files retain their links.

Missing, empty, duplicate, incomplete and invalid annotations fail validation;
valid zeros are accepted. Checks cover event identities, carry anchors, team
perspective and label action IDs. Unrecorded variants and inconsistent metadata
also fail. Failed outputs remain unusable. The latest-run pointer changes only
after successful publication. Explicit `--overwrite-feature-run` is supported,
but replacement is atomic per file, not transactional across the run.

## Updated artifacts

- Label tensors: only `scores_xt` and `concedes_xt`, across canonical/carry,
  intent-training, augmented and success-intent variants. Shapes, dtypes, row
  order and every other column are preserved.
- Resolved actions: primary-return `scores_xt`/`concedes_xt`; existing raw
  `xT`/`scores_xT`/`concedes_xT` columns are updated for canonical event rows.
  Synthetic carry rows have no independent event annotation. Receiver choices,
  indices and carry anchors are retained.
- Metadata retains the source split, graph schema, receiver model, return order
  and carry variant; clone statistics and refresh diagnostics are added.

Directories without a return suffix and resolved-action targets use the first
recorded return type. Carry returns use saved `return_event_index`, swapping
score/concede when the anchor event belongs to the other team.

Run xT refresh separately from other target refreshes, additional return types,
receiver replacement/addition, graph extensions and carry generation. Existing
carries are automatic: do not add `--use-carries` to the refresh command.

## Production commands (not executed)

Run from `C:\dm_model`. All 918 production xT sidecars were present during this
implementation; annotation generation may already have been completed separately.
The refresh validates their contents. If regeneration is needed:

```powershell
.\.venv\Scripts\python.exe scripts/generate_xt.py `
  --train-count 765 --use-interaction `
  --use-nonlinear x --use-nonlinear y --overwrite
```

Wait for annotation generation to finish. Create a new derived feature run:

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id feature_20260910T110759_053479_8b508137 `
  --refresh-target-family xt `
  --artifact-copy-mode hardlink
```

The source split and next-action setting are inherited unless explicitly supplied;
explicit mismatches are rejected. The new run ID is printed. Its `metadata.json`
must say `completed`. Inspect `xt_refresh_report.json` for input SHA-256 hashes,
changed rows, target means/nonzero counts, rewritten bytes and timings.

Refresh all 918 matches after refitting xT because existing seasons can change.
Receiver retraining is unnecessary for this repair.

For subsequent training, pin the new feature-run ID and use `--dataset-loading
disk` or the existing automatic setting. Cache identity includes resolved feature
and label paths, so new caches are built automatically; old caches need not be
deleted. Direct `train.py` calls must also update explicit `--feature_dir` and
`--label_dir` paths and use a new training run ID. Production refresh, cache
rebuilding and training are left for subsequent execution.

## Isolated pilot

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/pilot_xt_refresh.py
```

The pilot creates its own directory under `data/features/xt_refresh_pilots`, outside
published runs and on the source volume. Matches: `DFL-MAT-J03WEL` (the historically
affected 2022/23 season), `DFL-MAT-J03YDU` and `DFL-MAT-J04034`. It rejects a missing
annotation, then re-exports that match into an isolated fixture using persisted
xT coefficients. It does not fit a model or write production annotations.

Graph/tracking loads are prohibited during refresh. Normal `Match.construct_labels`
with original tracking supplies the reference targets. Checks cover non-xT
columns, source data hashes, graph signatures/sharing, and a second refresh that
must rewrite no artifacts. Reports include linked/rewritten bytes, timing and
sampled process RSS during refresh.
