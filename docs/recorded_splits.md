# Reusing historical training and test splits

Training and evaluation on existing feature runs load their saved manifests from
`data/splits/manifests/<split_manifest_id>.json`. The manifests contain the exact
development and test match IDs and the original universe fingerprint. Loading
them is read-only and does not use the current global match universe.

New feature generation can create a manifest for newly added data. Existing
feature runs and checkpoints continue to use their recorded manifest IDs.
Keep the manifest files alongside archived artifacts: missing or inconsistent
manifests cause an error rather than a new split calculation.

## Resolution rules

- Evaluation and resumed training use checkpoint provenance, checked against
  its original feature run. Without a checkpoint manifest, they use the feature
  run's provenance.
- New training uses the feature run's manifest. Direct training and the training
  wrapper infer omitted selectors; explicit `--train-split` or `--train-count`
  values assert the saved selector and cannot replace it.
- Only two legacy feature runs are explicitly mapped to
  `train_50pct_910f288a826f`:
  `feature_20260628T024614_842494_f7e03958` and
  `feature_20260827T004621_803230_f03ded3f`.
  They use all 306 matches from 2023/24 for development and all 306 matches from
  2024/25 for testing. Other legacy runs need verified provenance before use.
- `feature_20260910T110759_053479_8b508137` retains its recorded
  `train_count_765_8018228599b2` manifest: 765 development and 153 test matches.
- Validation is derived within the saved development pool. Filtering missing
  files does not move either the test boundary or validation boundaries.
  Final refits use the full saved development pool. New holdout and expanding
  runs reserve one shared latest validation cohort; expanding training sizes are
  configurable. Historical successive-block splits retain their recorded meaning.
  See [configurable validation and refitting](configurable_validation.md).

Evaluation metadata records `split_provenance`, including manifest ID, fingerprint,
resolution source, requested/loaded/contributing match IDs and counts, and skips.
Training records `split_manifest`, `split_resolution_source`, and `split_datasets`
with the equivalent information. Loaded matches can have no contributing rows
after action-level filtering; these counts are intentionally distinct.

## Visualization and independently trained components

Visualizers and their Hawkeye, SkillCorner, and benchmark component producers
allow checkpoints with different training selectors and manifests. They retain
each model's provenance rather than choosing one component's split for the rest.
Jobs that use a shared split to select their input data retain strict agreement.

Visualization console output and the versioned `model_membership` metadata block
report recorded training, validation, or development membership, verified holdout,
or unknown status for each participating checkpoint, including learned PC-xPass
auxiliaries. Allocation and contributing-match membership are reported separately;
allocation alone does not establish that a match supplied usable training rows.
Missing or conflicting evidence prevents a verified-holdout claim but does not
prevent visualization. Saved reports can be reused without the original weights.
Native Hawkeye, SkillCorner, and synthetic benchmark IDs are not compared to
Sportec training IDs without a verified mapping, so their membership is unknown.

Evaluation still checks its selected cohort against every participating model's
recorded development membership. Training sets may differ; the evaluation cohort
must be held out for every component. Paired comparisons also require identical
evaluated action populations.

## Earlier 459-match evaluations

The expanded-universe fallback previously assigned 153 matches from the original
development pool to the test set alongside the original 306 held-out matches.
Those 459-match evaluations are not independent historical test assessments.
Rerun evaluation with the corrected resolver to obtain the original 306-match
cohort and its bootstrap intervals. Existing results and checkpoint metadata are
not migrated or overwritten by this correction.
