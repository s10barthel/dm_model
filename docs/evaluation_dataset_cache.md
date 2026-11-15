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

Evaluation uses repository-root `cache/evaluation`; training uses `cache/training`.
Each has its own ownership database, locks, and session reports. Direct `test.py`
invocations and every model/configuration subprocess in the evaluation wrapper
use the same lifecycle as training: validate inputs, plan and claim required
entries, prune obsolete unused managed entries, then reuse or build the dataset.
Pruning happens before replacement disk-space checks and writes. Memory-mode
evaluation declares an empty prepared set and can reclaim obsolete entries.

After success, failure, or cancellation, completed cache entries remain until
the next evaluation model starts. Compatible checkpoints reuse them, while an
incompatible next model removes obsolete unprotected entries even if the wrapper
is still running. A later comparison stage may rebuild a pruned configuration.
The wrapper aggregates reports; it does not keep earlier model claims alive.

`--keep-cache` persistently protects used entries from later startup pruning.
Omitting it on a later command does not remove an earlier protection. Existing
untracked files, library retention, and live-reader claims remain protected.
Training cleanup does not scan the default evaluation root, and evaluation
cleanup does not scan the default training root. Both still consume free space
on the same drive and enforce the existing reserve check.

`--dataset-cache-dir PATH` overrides the evaluation root exactly. Evaluation
ignores the training-cache path saved in the model checkpoint. Choosing the same
custom path for both purposes deliberately restores shared-root behavior and
its reader/retention protection. The combined pipeline exposes independent
`--training-dataset-cache-dir` (alias `--dataset-cache-dir`) and
`--evaluation-dataset-cache-dir` controls; its existing `--keep-cache` flag still
applies to training. Pass `--keep-cache` to the evaluation command to pin its data.

The legacy `data/cache/intent_datasets` directory is not moved, copied, or cleaned
by the switch to new defaults, and there is no automatic fallback to it. Existing
legacy data continues occupying disk space until separately handled.

Evaluation streams one prepared match at a time with `num_workers=0`. Preparation
also defaults to one worker; `--dataset-preparation-workers` controls independent
preparation workers, each of which requires memory for a match.

Console output reports built/reused entries and preparation size/time. Each model's
`metadata.json` records `dataset_loading`, including the cache ID, preparation
statistics and load/reconstruction timing. Wrapper manifests record dataset
loading for evaluated entries and the final cache lifetime report.

## Four outcome pairs

Run these sequentially from the repository root in PowerShell. Compatible data
is retained for reuse by default. The explicit `--keep-cache` flags below also
protect those entries from future incompatible model startups.

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
