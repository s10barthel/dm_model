# Evaluation dataset reuse

`test.py` and `scripts/evaluate_relevant_models.py` accept the shared dataset
loading options described in [intent_dataset_loading.md](intent_dataset_loading.md).
Fresh evaluation defaults to `--dataset-loading auto`, which uses the prepared
disk cache for supported tasks, including outcome scoring and conceding.
`--dataset-loading memory` selects the original in-memory path.
Evaluations with nested xPass auxiliary graphs use memory in `auto` mode because
the packed format does not support nested graphs. Explicit `disk` mode reports
that restriction before preparation; use `memory` or `auto` for those evaluations.

The evaluator prepares graphs and complete label tensors per match. Compatible
checkpoints reuse those entries, including scoring/conceding pairs and different
outcome learning or severity modes. Checkpoint IDs and output directories do not
separate otherwise identical preparations. Inference and metrics still run for
each checkpoint; cache hits still verify integrity and reconstruct CPU graphs.

Cache identity includes preparation options, resolved feature/label/diagnostic
paths and preprocessing code. Per-match entries track source file paths, sizes
and modification times. Filters and feature transformations must agree to reuse
entries. Source changes invalidate the affected matches. Match selection and
order are applied independently to the reusable per-match entries.

Comparison exports retain semantic node identities. Missing legacy IDs are
recovered during per-match preparation; tracking and resolved action sources
participate in cache validation. Prepared selection counts, validation coverage,
skipped-row reasons and split provenance remain available on cache hits.

## Ownership and retention

The evaluation wrapper owns one cache workflow for all its model subprocesses.
Children release their sessions without deleting entries needed by later
checkpoints. The wrapper keeps model evaluation sequential. Direct `test.py`
invocations own their own workflow.

By default, successful workflow completion cleans up eligible managed entries.
Use `--keep-cache` to retain entries between separate scoring and conceding
commands. Failed evaluation workflows preserve completed entries, and
existing retention/active-reader rules apply. Training instead retains its cache
until the next model starts and prunes obsolete unused entries at that point.
Evaluation completion cannot sweep those retained training entries; active
evaluation readers also protect their entries from training-start pruning.
Retained entries consume disk space;
the cache retains the existing disk reserve check. `--dataset-cache-dir PATH`
selects another cache location.

Evaluation streams one prepared match at a time with `num_workers=0`. Preparation
also defaults to one worker; `--dataset-preparation-workers` controls independent
preparation workers, each of which requires memory for a match.

Console output reports built/reused entries and preparation size/time. Each model's
`metadata.json` records `dataset_loading`, including the cache ID, preparation
statistics and load/reconstruction timing. Wrapper manifests record dataset
loading for evaluated entries and the final cache lifetime report.

## Four outcome pairs

Run these sequentially from the repository root in PowerShell. Retention allows
the second command to reuse the first command's prepared data.

```powershell
python scripts/evaluate_relevant_models.py `
  --compare `
  --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261113T090248_952123_83e14e48 `
  --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261113T191803_593648_d489e101 `
  --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261114T082436_305823_d9efffdf `
  --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261114T191504_765855_c98b9ead `
  --season 24_25 `
  --season-half second `
  --filter pass `
  --keep-cache

python scripts/evaluate_relevant_models.py `
  --compare `
  --outcome-conceding-model-id outcome_conceding/outcome_conceding_20261113T090248_952123_35508bec `
  --outcome-conceding-model-id outcome_conceding/outcome_conceding_20261113T191803_593648_f595ee23 `
  --outcome-conceding-model-id outcome_conceding/outcome_conceding_20261114T082436_305823_7f466f71 `
  --outcome-conceding-model-id outcome_conceding/outcome_conceding_20261114T191504_765855_4aecc967 `
  --season 24_25 `
  --season-half second `
  --filter pass `
  --keep-cache
```
