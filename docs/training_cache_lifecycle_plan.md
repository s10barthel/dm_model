# Retain training caches until the next model starts

Status: implemented and verified, 15 November 2026.

## Agreed behavior

The cleanup boundary is each individual model invocation, including each
validation stage and final refit. A multi-model wrapper must not protect all
previous models' caches until the wrapper finishes.

For each model:

1. Resolve and validate the effective model settings, inputs, and split lists.
2. Determine all prepared entries needed by this model's training and validation
   datasets, without preparing graphs or requiring space for a replacement cache.
3. Claim that complete set and remove obsolete, unused managed entries.
4. Reuse valid compatible entries and build missing or invalid entries.
5. Retain the resulting cache after success, failure, or cancellation. Release
   the model's live claims when its readers stop. The next model decides what
   to reuse or remove.

Deletion happens after model startup validation but before dataset construction
and its free-space checks. This frees space before writing replacements. A
configuration needed by a later model is allowed to be deleted now and rebuilt
later; limiting disk usage takes precedence over reuse across the whole bundle.

Example: model A finishes and leaves its cache. Model B starts with compatible
preparation and reuses it. Model C starts with different preparation, deletes
A/B's unused entries, and builds its own cache. C's cache remains after C exits.

## Compatibility and scope

Use the existing preparation identity and per-match source signatures. Do not
introduce a comparison of arbitrary model CLI arguments. Learning rate, epochs,
optimizer, and other training-only settings continue to allow cache reuse.
Changes to preparation options, preprocessing code, or relevant source inputs
continue to invalidate entries as they do today. Preserve existing compatible
views, including outcome scoring/conceding and deferred height relabeling.

The desired set is the union of this model's actual train and validation entry
paths, potentially spanning more than one preparation directory. Prune obsolete
matches and source versions inside an otherwise compatible directory as well
as obsolete configuration directories. A final refit declares its refit split;
a memory-loading model declares an empty prepared set for its configured root.

Retain the existing explicit exceptions:

- `--keep-cache` persistently protects the entries it uses from future pruning.
  Default retention until the next model does not set this persistent flag.
- Untracked/pre-existing files and caches created through the library's existing
  explicit retention behavior remain protected.
- Entries being built or read by another live process remain protected.
- Source features, labels, checkpoints, physical caches, and the separately
  managed IPW probability cache are outside this cleanup.

These exceptions can make total disk usage exceed one model's cache size. Report
their sizes and reasons when they prevent space reclamation.

## 1. Separate cache planning from dataset preparation

Change `prepared_dataset.py` to expose a read-only preparation plan and consume
the same plan during dataset construction.

- Extract the current option normalization, preparation identity, source
  signatures, and per-match path calculation into shared helpers. Planning and
  writing must not implement these calculations separately.
- Include resolved paths, normalized options, cache ID, match IDs, source
  signatures, and expected `.pt`/manifest paths in the plan.
- Preserve current defaults, special-case normalization, and loaded-code
  stability checks. Planning must not construct `ActionDataset`, deserialize
  prepared graphs, write cache payloads, or perform the replacement-space check.
- Verify source stability when consuming a plan. Keep checksum/payload validation
  and atomic publication in the existing per-entry preparation path. Rebuild a
  corrupt requested entry without evicting other entries required by this model.
- Keep ordinary library callers working without requiring model-start cleanup.

The complete train/validation union must be known before either dataset begins
preparation. Cleanup inside every dataset constructor would risk deleting the
other split or previously prepared parts of the same model.

## 2. Give each model its own live cache claims

Refactor `cache_lifetime.py` to distinguish wrapper reporting from model usage.

- Keep the wrapper descriptor for workflow attribution and aggregate reports.
  Create a separate model-use identifier for each `train.py` invocation, linked
  to that descriptor. A standalone invocation uses the same model-use protocol.
- Propagate model-use identity to preparation workers. Register the entire
  desired set before pruning, including entries that have not been built yet.
- Keep those claims through preparation, training, validation, and any remaining
  reads. Release them at model exit after its readers are done. An idle wrapper
  must not keep the previous model's entries actively claimed.
- Make a subsequent stage/refit create fresh live claims, even when it belongs
  to the same wrapper. A resumed model also declares its current requirements
  again; it must not require the old cache to still exist.
- Reap claims belonging to dead processes using PID and process creation time.
  Continue to treat uncertain process access conservatively and protect live
  child readers even if their parent has died.
- Remove historical failed/abandoned status as a reason to block training-start
  pruning. Status remains diagnostic metadata, not permanent ownership.

Persist the model-use claims and retention policy in an additive, versioned
SQLite migration. Do not reset the ownership database or reinterpret historical
`keep`/`retained` flags as ordinary default retention.

## 3. Prune obsolete entries before allocating replacement space

Add one explicit model-start operation in `cache_lifetime.py`:

1. Register the model's complete desired set under the ownership lock.
2. Reconcile dead process records and enumerate managed entries outside that set.
3. For each candidate, recheck current claims and persistent protection under
   the lock before deleting its known `.pt` and `.json` pair.
4. Remove successfully deleted entries and their obsolete references from the
   ownership records; remove empty preparation directories where possible.
5. Return a report, then allow normal preparation and its disk reserve checks.

Reuse the existing resolved-path validation and root confinement. Preserve
unrecognized files. Do not recursively delete arbitrary directories. Existing
lock files must not be removed while concurrent processes could use them.

Maintain the existing entry-lock-then-root-lock order for deletion as well as
preparation: do not hold the root lock while waiting on an entry lock. Recheck
eligibility after acquiring the locks so a concurrent model's newly registered
claims are honored. Current-model reservations protect validation entries while
training entries are being prepared.

Report permission/locking errors without falsely claiming space was recovered.
If protected entries or deletion failures leave insufficient space, retain the
existing explicit disk-space error before preparation; do not silently switch
loading mode. Keep cleanup idempotent across interruption and partial deletion.

## 4. Integrate each training entry path

In `train.py`, call model-start cleanup after resume/refit settings, source paths,
and split membership have been resolved and validated, immediately before the
first dataset constructor. Build both dataset plans at that point. Run the same
startup cleanup with an empty desired set for memory mode.

In `scripts/train_relevant_models.py`, retain sequential execution and command
ordering. Ensure every child training command, validation stage, source refit,
and final refit uses the per-model lifecycle. Do not move pruning into wrapper
startup or wait for wrapper completion to release model claims.

Update `training_context()` and workflow finalization so success/failure records
and artifact publication remain intact, but finishing a training model or its
wrapper no longer deletes its default-retained cache. Do not change training,
validation, checkpoint selection, or IPW calculation semantics.

Adapt `scripts/verify_dataset_loading.py` to declare the union of its required
entries once before its representative and full-split checks, rather than
triggering cleanup independently for each verification dataset.

## 5. Handle evaluation and existing ownership records

Evaluation uses the same cache root and ownership machinery. Keep evaluation's
current lifecycle in this change, but make the lifetime policy explicit so its
completion cleanup cannot sweep entries retained under the new training policy.
An active evaluator must also protect its entries from training-start pruning.
Outside such protection or explicit retention, managed obsolete entries in the
shared root are eligible for training-start pruning.

Existing tracked training entries, including the two configuration directories
currently retained by failed wrapper sessions, enter the new startup policy.
Old failed/completed session references alone must not protect them. Preserve
conservative protection for genuinely live legacy sessions until they exit.
Do not delete anything merely by importing code, opening the database, or
running the schema migration; pruning requires a validated model start.

Editing `prepared_dataset.py` changes the current preprocessing fingerprint.
Expect a possible one-time rebuild when this refactor is introduced. Do not
relabel old identities or weaken fingerprint validation to avoid that rebuild.
Obsolete tracked entries can be pruned before their replacements are built.

## 6. Explain retention and reclamation in reports

Extend model and wrapper `cache_lifetime` metadata and startup output to show:

- The cleanup trigger (`model_start`), model-use ID, and preparation identities.
- Reclaimed entry count/bytes and compatible entries available for reuse.
- Retained entry count/bytes by reason: current model, active other model,
  explicit keep, or pre-existing/unmanaged files.
- Cleanup errors and the fact that model-exit retention awaits the next model.

Keep existing useful fields compatible where possible. Store per-model startup
reports and aggregate them at wrapper level without implying that exit performed
the deletion. Update `--keep-cache` help in `dataset_loading.py`, the README,
`docs/intent_dataset_loading.md`, and the shared-lifecycle explanation in
`docs/evaluation_dataset_cache.md`.

## 7. Verification and acceptance criteria

Use temporary fixture roots for all deletion tests. Do not test cleanup against
`data/cache/intent_datasets` or remove existing real cache files while implementing.

Update `tests/test_cache_lifetime.py`, `tests/test_prepared_dataset.py`, and the
relevant evaluation, training/refit, and verification-script tests to cover:

1. A successful model leaves its cache; the next compatible model reuses it.
2. Training-only setting changes reuse preparation; changed preparation removes
   obsolete entries before the first new payload write and free-space check.
3. Train and validation plans with different directories both survive startup;
   obsolete matches and old source versions within a kept directory are removed.
4. A three-model wrapper frees each previous incompatible model's cache even
   while the parent wrapper remains alive. Later stages may rebuild it.
5. Failed, cancelled, or killed models leave reusable completed entries, but
   their historical records cannot block the next incompatible model's cleanup.
6. Resume/refit rebuilds missing entries without requiring the failed original
   session to complete. Final refits and memory mode use the intended boundary.
7. Concurrent models, workers, and evaluators protect all reserved/read entries;
   model exit releases claims while wrapper exit preserves retained training data.
8. Explicit keep, library retention, untracked files, path validation, and PID
   reuse protection still work. Evaluation completion preserves retained training
   entries while retaining its own documented behavior for evaluation-only data.
9. Partial deletion, cleanup errors, source changes after planning, corruption,
   migration, and repeated cleanup remain recoverable and correctly reported.

Run focused lifetime/prepared/evaluation tests, followed by affected training,
refit, and verification-script checks. Use a tiny sequential-model fixture to
assert peak cache occupancy: the previous incompatible unprotected cache is gone
before the next is written. Cache sizes may still exceed that model's needs when
other live readers, explicit retention, or unmanaged files prevent reclamation.

## Implementation and validation

Implemented the read-only `PreparedDatasetPlan`, per-model live claims,
model-start pruning, additive ownership migration, wrapper report aggregation,
training/verification integration, and evaluation retention protection described
above. Default caches survive model exit; historical failures no longer block
their reclamation by the next model.

Validation covered 282 passing test cases across `test_cache_lifetime.py`,
`test_prepared_dataset.py`, `test_evaluation_cache.py`,
`test_all_task_training.py`, `test_training_runtime.py`,
`test_training_controls.py`, and `test_training_refit.py`. One additional
file-symlink check was skipped because this Windows account cannot create file
symlinks. Checks include real forward/backward training, resume equivalence,
refits, concurrent readers, process death, and pruning before replacement writes
and free-space checks. `git diff --check` passed.

Verification used isolated temporary caches. No cleanup was invoked against the
existing repository cache. The new policy applies at future validated model
starts; the changed preprocessing fingerprint can require an initial rebuild.
