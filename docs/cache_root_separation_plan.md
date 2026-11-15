# Separate training and evaluation cache roots

Status: implementation plan, 15 November 2026. Runtime code and caches are unchanged.

## Proposed layout and behavior

Use the suggested paths, anchored to the repository rather than the process's
working directory:

| Purpose | Default directory |
| --- | --- |
| Prepared training and validation datasets, including refits | `C:/dm_model/cache/training` |
| Prepared evaluation datasets | `C:/dm_model/cache/evaluation` |

Each root has its own `.ownership.sqlite`, lock files, session reports, and hashed
preparation directories. Training startup scans and prunes only its resolved
training root. Evaluation cleanup scans only its resolved evaluation root.
`--keep-cache` protects entries within the root where they are used; it does not
protect or retain entries in the other root.

Preserve the current lifetime policies: training retains entries after each
model and prunes obsolete unused entries at the next model's startup; evaluation
cleans eligible entries after successful evaluation-workflow completion. Failed
evaluations retain completed entries. Explicit retention and live-reader
protection continue to apply within each root.

Separate roots isolate cleanup, ownership, and retention. They still consume the
same drive's free space: a large retained evaluation cache can leave insufficient
space for training. Training will no longer reclaim that evaluation data. The
existing reserve check remains applicable to both roots.

## Decisions to settle before runtime implementation

The recommendations below are proposals, not assumed user answers. The plan can
be reviewed now; implementation of the affected behavior waits for explicit
answers if these decisions remain open.

| Decision | Options | Recommendation |
| --- | --- | --- |
| Existing shared cache | Start with new roots and leave `data/cache/intent_datasets` untouched; or build a separate, controlled migration tool. | Start with new roots. This avoids moving active files or misclassifying opaque cache identities and ownership records. Budget for rebuilding; old data continues occupying disk until separately handled. |
| Old checkpoint/refit paths | Redirect missing or saved legacy-default paths to the new training default; or preserve every saved path until explicitly overridden. | Redirect the known legacy default. Otherwise old resumes silently keep using shared storage. Preserve saved custom paths and let an explicit runtime override win. |
| Evaluation lifetime | Keep workflow-completion cleanup; or introduce training-style pruning at each evaluation model's startup. | Keep the current evaluation policy for this change. It supports reuse across compared checkpoints. A different evaluation retention policy can be considered separately. |

The proposed directory names follow the user's suggestion literally: repository
`cache/training` and `cache/evaluation`, rather than `data/cache/...`.

## 1. Define defaults and resolve roots centrally

In `dataset_loading.py`:

- Introduce explicit training and evaluation default constants, plus the old
  shared path as a legacy-path constant used only for compatibility decisions.
- Give `add_dataset_loading_arguments()` and any flag-generation fallback an
  explicit purpose so evaluation cannot accidentally inherit a training default.
- Resolve cache location centrally with precedence: explicit command-line or
  operational override, applicable saved custom location, purpose-specific
  default. Apply the selected legacy-path decision between saved settings and
  fallback. Track whether a path was explicitly requested; argparse defaults
  must not masquerade as explicit overrides.
- Normalize paths before comparing saved legacy defaults, including Windows
  path spelling and case. An explicitly supplied legacy directory is still an
  explicit override and must not be silently redirected.
- Keep `--dataset-cache-dir PATH` as an exact root override on direct training
  and evaluation commands. Do not append a purpose subdirectory to that argument.
- Retain a temporary `DEFAULT_CACHE_DIR` alias to the training default if needed
  by library callers, while moving internal call sites to named defaults.

The defaults are independent. A user who explicitly chooses the same custom
root for both purposes opts back into the existing shared ownership/cleanup
behavior; document this rather than silently changing explicit paths.

## 2. Route every entry point to its purpose

Training defaults apply to `train.py`, `scripts/train_relevant_models.py`, and
the training portion of `scripts/main.py`. Evaluation defaults apply to
`test.py` and `scripts/evaluate_relevant_models.py`. Audit child command builders
so they forward the resolved root explicitly and consistently.

`scripts/main.py` currently forwards `--dataset-cache-dir` to training only.
Preserve that meaning and add `--training-dataset-cache-dir` as a clearer alias.
Add `--evaluation-dataset-cache-dir` for its evaluation stage. Forward each root
only to its matching stage. Avoid changing the existing scope of other pipeline
loading and retention flags as part of a cache-location change.

Keep `scripts/verify_dataset_loading.py` isolated under its explicit
`--output-dir/cache`, as today. Update `scripts/verify_intent_loader.py` to use
the named training default when no explicit `--cache-dir` is given. Library
evaluation call sites must pass an evaluation root explicitly; the general
prepared-dataset library default remains the training default for compatibility.

## 3. Handle resume and refit before opening cache ownership

The current training resume path restores `dataset_cache_dir` from checkpoint
arguments, while refits copy source settings. Merely changing parser defaults
would not affect these paths.

If the recommended legacy-path decision is accepted:

- Normalize missing and saved legacy-default roots to `cache/training` for
  resumes, continuation, source refits, and final refits.
- Preserve saved custom roots, with explicit operational overrides taking
  precedence. Add `--dataset-cache-dir` to direct resume runtime options and
  wrapper resume allowlists/forwarding, so relocating an old custom root remains
  possible without changing the model recipe.
- Resolve the effective root before `workflow_context()` or `training_context()`
  opens a database. Ensure wrapper reporting and child dataset preparation use
  the same resolution rules; `saved.get('dataset_cache_dir')` must not override
  an explicitly selected new root.
- Create new live cache claims in the resolved root. Keep old descriptors only
  for safe detection of an original process that is still running and provenance;
  do not reconnect ownership in the legacy root as part of relocation.
- Record the effective root and its purpose in new run metadata without editing
  old checkpoints or rewriting source-run artifacts.

Evaluation chooses its own evaluation root from its invocation. It must not
adopt the training cache location stored inside the checkpoint being evaluated.
Cache location remains operational metadata; relocation must not alter model
settings, split selection, sample order, RNG restoration, or propensity inputs.

## 4. Keep ownership scoped to purpose and root

In `cache_lifetime.py`, replace the shared default fallbacks in training,
evaluation, and wrapper contexts with the appropriate named defaults.

Record cache purpose in descriptors/reports and preserve parent attribution
within a purpose. An evaluator must not attach to a training wrapper's ownership
merely because `DM_MODEL_DATASET_CACHE_SESSION` is present; give it its own
evaluation owner. Evaluation children still attach to their evaluation wrapper.
Handle legacy descriptors using their existing kind and explicit root without
silently creating cross-purpose ownership.

Collectors continue to operate on one resolved root. Do not add parent-directory
scans, cross-root deletion, automatic legacy fallback, or copying of ownership
databases between roots. Preserve existing shared-root protections for explicit
custom configurations and for legacy runs.

No preparation/payload schema change is needed just to separate locations.
Preserve current identity and integrity checks. Editing fingerprinted loader
code can nevertheless invalidate existing identities; do not weaken validation
or promise reuse across a code change to avoid a rebuild.

## 5. Rollout and documentation

- Add repository `/cache/` to `.gitignore` before new defaults are exercised.
- Update CLI help, README examples, `docs/intent_dataset_loading.md`, and
  `docs/evaluation_dataset_cache.md` with both roots and their policies.
- Explain pipeline override precedence, saved-path handling, explicit shared-root
  opt-in, and how `--keep-cache` can increase disk use independently in each root.
- Print and record the resolved absolute root and purpose, so unexpected path
  inheritance is visible in startup logs and reports.
- If the recommended fresh-root rollout is accepted, do not move, copy, prune,
  or delete the old shared cache. There is no silent fallback to it on a miss.
  Reclaiming its disk space or migrating its entries requires a separate explicit
  action once no process uses it.
- Keep physical-xPass caches, IPW probability caches, source artifacts, model
  checkpoints, and evaluation results at their existing locations.

## 6. Verification and acceptance criteria

Use temporary repository roots/cache paths in all tests. Do not exercise cleanup
against the existing cache or create production cache directories during testing.

Extend the existing loading, cache-lifetime, pipeline, evaluation, training,
resume, and refit tests to verify:

1. Fresh direct and wrapper invocations select the correct root from any working
   directory; each root gets separate ownership, lock, and report files.
2. Training startup removes obsolete training entries while preserving evaluation
   entries and its database. Successful evaluation cleanup preserves training
   entries and its database.
3. `--keep-cache` persists protection within its root. Pins, failed workflows,
   and active readers in one default root do not affect the other root's policy.
4. Pipeline defaults and stage-specific overrides reach only the intended child
   commands; existing training-only override behavior remains compatible.
5. Direct CLI overrides remain exact paths. Deliberate shared overrides retain
   the existing shared-root safety rules.
6. Resume/refit resolution covers missing paths, legacy-default paths, saved
   custom paths, and explicit overrides according to the chosen policy. Actual
   resumed training remains numerically equivalent.
7. Evaluation ignores a checkpoint's training-cache location; inherited training
   ownership cannot couple it back to the training root.
8. Legacy cache files and ownership records remain unchanged throughout fresh
   starts and path resolution. Read-only checks of an original session must not
   migrate or modify its old database.
9. Verification tools retain their intended isolation; source/physical/IPW caches
   and model artifacts are unaffected.

Run focused root-selection/isolation tests, then the affected existing training,
evaluation, resume, refit, and pipeline checks. No real cache movement or cleanup
is part of implementing or testing this plan.
