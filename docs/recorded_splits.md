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

## Evaluation data sources

`--feature-run-id` selects evaluation graphs, primary labels, and default diagnostic
labels without replacing the checkpoint's historical training manifest. An input run
may record a different split; it is not used to infer which matches the checkpoint
saw, or to replace its default test cohort. Explicit season selection takes precedence
for evaluation populations. Unknown historical membership retains the interactive
confirmation below, and a missing default cohort requires explicit selection.

The evaluation wrapper preflights every primary model and auxiliary against the
requested cohort before launching any evaluator. Missing required graph/label files
stop the job; no intersection or regeneration is performed. Tensor-level validation
and paired action alignment remain enforced during evaluation. Both entry points
share path resolution, label selection, and graph compatibility checks.

Diagnostic labels default to an explicitly selected primary run. The optional
`--diagnostic-feature-run-id` changes diagnostics only; pass-height diagnostic
ancestry is checked against the effective input run. Without either flag, existing
checkpoint-based defaults are retained. Source selection does not change prediction
semantics or target definitions. Portable `evaluation_data` metadata records the
requested/resolved input and diagnostic sources, actual paths, label variants, and
original training feature run; result destinations and pc-xPass caches are independent.

## Visualization and independently trained components

Visualizers and their Hawkeye, SkillCorner, and benchmark component producers
allow checkpoints with different training selectors and manifests. They retain
each model's provenance rather than choosing one component's split for the rest.
Jobs that use a shared split to select their input data retain strict agreement.

For Sportec identities, visualization console output and the version-2
`model_membership` metadata block report recorded training, validation, or
development membership, verified holdout, or unknown status for each participating
checkpoint, including learned PC-xPass auxiliaries. Allocation and
contributing-match membership are reported separately;
allocation alone does not establish that a match supplied usable training rows.
Missing or conflicting evidence prevents a verified-holdout claim but does not
prevent visualization. Saved reports can be reused without the original weights.
Native Hawkeye, SkillCorner, synthetic benchmark, and unknown-provider identities
are not compared to Sportec training IDs. Their reports store
`check_status: "not_checked"`, reason `no_verified_sportec_match_id`, the
participating model IDs/roles, and an empty `entries` list. Membership-specific
checkpoint and manifest lookups are skipped. A single informational notice is
printed per run; tracking-only runs with no participating models remain silent.
No cross-provider identity mappings are inferred, and unchecked identities are
never described as held out.

Sportec reports with known identities have `check_status: "checked"`; absent
checkpoint evidence still produces an entry with `status: "unknown"`. Missing
Sportec identities have explicit `not_checked` entries. A report containing both
known and missing identities has `check_status: "partially_checked"`.
Missing-membership/checkpoint diagnostics print once per model per run, while
metadata retains details for every identity and comparison variant. Comparisons
share reporter caches and console deduplication across incremental saves and
final publication.

Saved version-1 and version-2 Sportec reports remain reusable after checkpoints
are archived. Old non-Sportec reports are normalized to the unchecked form when
producing a new report. Existing exports are not rewritten.

Evaluation still checks its selected cohort against every participating model's
recorded development membership. Training sets may differ; the evaluation cohort
must be held out for every component to claim verified evaluation membership.
Paired comparisons also require identical evaluated action populations.

For historical checkpoints with absent membership evidence, the evaluation wrapper
and direct `test.py` allow an exploratory evaluation only after an interactive
`yes`/`y`. The warning lists all affected primary and auxiliary models. The wrapper
asks once and records cohort-scoped approval before launching child evaluators;
children independently check membership and validate the saved approval. No public
flag permits unattended acceptance, and `--accept-comparison-warnings` is independent.
Empty or invalid answers reprompt; refusal, EOF, interruption, and noninteractive
input cancel with exit status 1. A requested output directory retains cancellation
metadata without producing metrics (standalone runs use `membership_cancellation.json`).

Known overlap remains an error even when other membership is missing. Malformed or
conflicting records and missing/corrupt explicitly referenced manifests also remain
errors. This exception does not relax feature, cache, or example-alignment checks.
When no historical split identifies a default test cohort, an explicit `--season`
selection is required; no current split or default training percentage is substituted.
Historical split fields remain null when unavailable. Learning curves still require
recorded training sizes and split provenance.

Successful outputs include `evaluation_membership` with verified/unverified status,
per-model checks and requested cohorts, affected model IDs, and the confirmation
decision and method. Unknown exclusion is not reported as zero verified overlap.
Comparison metric and difference CSVs carry `evaluation_membership_status`; a pair
is unverified if either side depends on an unverified primary or auxiliary model.
Comparison plots explicitly label these results as "Evaluation membership unverified".

## Earlier 459-match evaluations

The expanded-universe fallback previously assigned 153 matches from the original
development pool to the test set alongside the original 306 held-out matches.
Those 459-match evaluations are not independent historical test assessments.
Rerun evaluation with the corrected resolver to obtain the original 306-match
cohort and its bootstrap intervals. Existing results and checkpoint metadata are
not migrated or overwritten by this correction.
