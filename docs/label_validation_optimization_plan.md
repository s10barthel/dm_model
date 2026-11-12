# Final label validation optimization plan

Status: implemented and verified on 13 November 2026. See
[measured results](feature_runtime_efficiency.md#validation-on-13-november-2026).

## Goal and observed cause

Reduce the time spent validating added return-type labels while retaining the
existing coverage, alignment, and publication checks.

`audit_return_type_extension()` in `scripts/generate_relevant_features.py`
constructs reference and candidate label directories inside the match loop.
Each call to `get_action_label_dir()` or `get_intent_train_label_dir()` invokes
`project_config._label_root()`, which reads and parses `metadata.json` to select
the label layout. The layout and directories remain constant during this audit.

For `feature_20261112T200025_087890_f4a8e4f1`, the audit has:

- Three receiver modes.
- 918 matches for ordinary labels, 765 for intent-training labels, and 918 for
  carry labels: 7,803 match/variant checks across the three modes.
- One added return type, `time_15`, with base and set-piece conditions.
- Three path-builder calls per check: one reference and two candidate calls.

That produces approximately 23,409 metadata reads in the main loop. At
68,470,516 bytes per metadata file, this is about 1.60 TB of logical reads.
This is not a measurement of physical disk traffic: Windows can cache reads,
but Python still decodes and parses the JSON repeatedly.

## Recommended scope

Implement local directory precomputation in the return-type audit. Keep the
existing path helpers and their layout-detection behavior. With this run's
configuration, the main audit needs approximately 18 directory resolutions:
nine receiver/action groups, each with one reference and one candidate directory.
The two conditions share the candidate directory resolution.

`refresh_inventory()` also invokes the path helpers, but only per directory,
not per match. Its existing calls remain in this first change. Consequently,
the whole audit will still parse metadata dozens of times; this proposal does
not claim to reduce the entire phase to one metadata read. The important
guarantee is that the count no longer grows with match count.

## Implementation sequence

### 1. Establish a reproducible baseline

- Capture the pre-change audit source for a comparison benchmark.
- Instrument reads of the target run's `metadata.json`, filtering out unrelated
  metadata reads and excluding fixture setup.
- Record audit wall time and metadata parse count on isolated fixtures with
  both label layouts, all receiver modes, carries, intent-training labels, and
  base/set-piece conditions.
- Keep the baseline small enough to avoid repeating hours of unnecessary work.
  Do not run the unoptimized full audit against the active production run.

### 2. Precompute directories and condition lists

Change `audit_return_type_extension()` in
`scripts/generate_relevant_features.py`:

1. Retain the current inventory and split preflight.
2. Compute the ordered, deduplicated conditions for each selected return type
   once, using the same base/advertised/requested conditions as today.
3. For each existing action variant and receiver mode, resolve its reference
   directory and each selected return's base directory before the match loop.
   Deduplicate resolutions if the reference and candidate return are identical.
4. Inside the loop, append `<match_id>.pt` to these saved directories. Continue
   using `stop_conditions.label_path()` for non-base condition paths so the
   existing condition-layout rules remain centralized.
5. Keep the current iteration order, file-loading behavior, and validation
   predicates. Keep caches and tensors local to this audit call; do not retain
   all matches' payloads in memory.

Preserve checks for split membership, intent-training coverage, reference
shape, finite integral action IDs, saved-action membership, uniqueness where
required, candidate dtype/columns, and exact action-ID alignment. Intent-training
labels must continue to permit their intentional repeated IDs.

The audit remains read-only. Successful validation still precedes completion
metadata publication, and failures still prevent publication through the
existing exception path. Training eligibility rules remain in force.

### 3. Add focused regression coverage

Extend `tests/test_carry_return_extension.py`, reusing its existing audit
fixtures and corruption cases:

- **Scaling:** compare small and larger match sets with identical directory
  configurations. Metadata-read counts must remain equal as matches increase.
  Calls made during `refresh_inventory()` are included in this count.
- **Condition reuse:** adding a set-piece condition must not trigger another
  path-helper metadata read for each match or duplicate the candidate directory
  resolution.
- **Layouts and variants:** cover legacy top-level labels and `labels/base`,
  all receiver modes, carries present/absent, multiple selected return types,
  and intent-training labels.
- **Freshness between invocations:** ensure the next audit resolves directories
  anew after a valid layout transition; no cache survives the previous audit.
- **Validation equivalence:** retain missing-file, reordered/wrong-ID,
  wrong-split, and missing/misaligned set-piece rejection. Add focused cases for
  malformed tensors and duplicate ordinary action IDs if existing coverage
  does not exercise those branches.
- **Publication and immutability:** retain tests proving failed audits cannot
  publish completion and that auditing leaves source and target artifacts
  byte-for-byte unchanged.

Use deterministic read-count assertions as the performance regression test.
Avoid wall-time thresholds in automated tests.

Run the focused suites first:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_carry_return_extension.py tests/test_feature_run_extension.py tests/test_extension_failures.py
```

If implementation stays within the audit function, these are the primary
regression suites. Run adjacent layout/refresh suites if a failure or an
implementation change implicates their shared behavior.

### 4. Measure and document the result

- Benchmark the captured baseline and optimized audit against identical,
  isolated fixtures. Keep file count, tensor contents, metadata size, and
  storage conditions comparable; distinguish first and repeated executions.
- Include realistic large metadata in a bounded benchmark, plus a smaller
  metadata case to distinguish parsing savings from label/parquet I/O.
- Record elapsed time, target metadata reads, and approximate logical metadata
  bytes read. Verify identical pass/fail outcomes and unchanged artifacts.
- For the current run's configuration, confirm the main audit's path-helper
  calls fall from approximately 23,409 to approximately 18. Account separately
  for the directory-level inventory calls.
- Update `docs/feature_runtime_efficiency.md` with measured results and the
  implementation's scope. Do not promise a particular full-run duration before
  measuring the remaining tensor and parquet work.

## Design choices and recommendations

| Choice | Options | Recommendation |
|---|---|---|
| Where to fix repeated reads | Hoist directory construction locally; add a shared metadata-snapshot API; add a process-wide cache | Start with local precomputation. It addresses the match-count multiplier with the smallest behavior change. |
| Remaining inventory reads | Keep directory-level reads; extend shared helpers to accept a resolved layout | Keep them initially and measure. Consider an explicit per-operation layout snapshot only if their remaining cost is material. |
| Parallel validation | Keep the current serial audit; dispatch matches to workers | Keep serial execution for this change. Reassess after removing repeated JSON parsing; worker startup, memory, and storage contention need their own measurements. |
| Progress reporting | Include it in the performance fix; make it a separate follow-up | Follow up separately with elapsed time and completed/total match/variant checks. Prefer throttled console output; repeatedly publishing the large metadata file would add overhead. |
| Current running process | Let it continue; stop and rerun validation through a recovery workflow | Let it continue. Editing the source does not replace code already loaded by the process, and restart/recovery is separate work. |

The recommendations above define the implemented scope. There are no
unresolved functional requirements blocking that scope.

## Completion criteria

- Metadata parsing during the audit is independent of the number of matches.
- Legacy and canonical layouts and every selected variant retain their existing
  acceptance/rejection behavior.
- The focused regression suites pass, with any unrelated baseline failures
  explicitly identified.
- A bounded comparison records the reduction in metadata reads and elapsed
  time, without relying on a speculative speedup.
- Auditing preserves artifact contents and publishes completion only after all
  checks succeed.

Metadata schema changes, provenance compaction, training before validation,
validation bypass flags, and recovery of the currently running process are
outside this optimization's scope.

## Implementation result

- Directory construction and condition deduplication now run outside the match
  loop, with reference/candidate directory reuse confined to each action
  variant and receiver mode.
- The focused suites pass 198 tests and 40 subtests; the original baseline
  passed 183 tests and 40 subtests.
- In both layouts, increasing the bounded fixture from 2 to 7 matches left
  optimized metadata reads at 40, while the baseline increased from 67 to 202.
- With copied 68 MB production metadata and two synthetic matches, the bounded
  audit was approximately 40% faster. Inventory checking still contributes
  directory-level metadata reads, as planned.
- Baseline/optimized comparisons preserved artifact hashes and rejection
  behavior. The production process and its run artifacts were not modified.
