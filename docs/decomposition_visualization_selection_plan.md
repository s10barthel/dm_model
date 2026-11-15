# Selective decomposition labels in mixed visualizations

Status: implemented, 2026-11-15. Live and saved standard Hawkeye visualization,
Sportec live visualization, checkpoint comparisons, and physical-cache comparisons
use selective annotations. The recommended policy defaults below are applied.

## Requested behavior

Keep `--decomposition` as an opt-in annotation request. It must coexist with
the normal component selection, including pass success, pass score, combined
outcomes, physical xPass, and other available plots. It must not remove plots
or change which checkpoint pairs are compared.

Show occurrence x conditional severity below player dots only when all three
conditions hold:

1. `--decomposition` is enabled.
2. The displayed component is one of `outcome_scoring_success`,
   `outcome_scoring_failure`, `outcome_conceding_success`, or
   `outcome_conceding_failure`.
3. That component's own source checkpoint has `learning_mode="decomposition"`.

The usual expected value remains above each dot. Both scalar severity and
distributional severity qualify. Ordinary scalar and ordinary distributional
learning modes do not qualify. Determine eligibility from validated checkpoint
settings using `models.outcome_decomposition.is_decomposition`, never from model
ID dates, numerical predictions, or the severity mode alone.

Eligibility is per task and per comparison variant. For example, a pair with
decomposition scoring and scalar conceding annotates only scoring. Other plots
render normally, including derived outcomes whose inputs use decomposition.

## Verified example

The supplied checkpoints have these saved settings for both outcome tasks:

| Run timestamp | Learning mode | Severity mode | Individual outcome labels with the flag |
| --- | --- | --- | --- |
| `20261113T191803` | scalar | N/A | Expected value only |
| `20261114T191504` | distributional | N/A | Expected value only |
| `20261114T082436` | decomposition | scalar | Expected value and occurrence x severity |
| `20261113T090248` | decomposition | distributional | Expected value and occurrence x severity |

After implementation, the original command should retain all its currently
available plots and annotate only the last two pairs' individual outcome plots.
Existing unrelated model-omission and comparison confirmations still apply.

## Implementation sequence

1. Separate component selection from annotation eligibility.

   In `scripts/outcome_decomposition_visualization.py`, replace the requirement
   that every selected component belong to `COMPONENTS` with a resolver for the
   eligible subset. Accept mixed selections and selections with no eligible
   components. Preserve ordinary selection and dependency validation in
   `scripts/visualization_availability.py`; the flag adds labels and must not
   select, exclude, or require extra plots or models.

   Change `validate_models` so it considers only selected individual outcome
   components, resolves their own task's checkpoint, and enables annotation only
   for decomposition checkpoints. Treat a non-decomposition checkpoint as normal
   ineligibility. Missing required models and invalid decomposition definitions
   continue through the existing hard-error paths. Return a structured result
   listing each selected outcome source, its mode, eligible components, and any
   reason labels do not apply.

2. Integrate the result into live visualization and comparisons.

   Update `scripts/run_and_visualize_hawkeye.py`,
   `scripts/visualize_action_components.py`, and
   `scripts/inference_visualization_comparison.py` to resolve eligibility after
   final availability filtering and before inference. Resolve it independently
   for every comparison variant; avoid using the first pair's settings as a
   global decision or mutating `args.decomposition` while visiting variants.

   Explicitly set or clear `visualize_outcome_decomposition` on applicable model
   instances so reused models cannot retain annotation state from a prior call.
   Preserve the existing prediction-cache distinction for this attribute and
   verify shared-checkpoint reuse across masks and physical-cache variants.

3. Enforce annotation eligibility at rendering boundaries.

   Gate calls to `annotate` by the displayed component as well as its resolved
   source eligibility. Audit `derive_components`, `compute_outcome`, and
   `compute_pass_score` and their callers for pandas attribute propagation:
   derived tables must not inherit active decomposition annotation metadata.
   A derived plot must never show one input model's heads as its own labels.

   Retain three-decimal labels, dot placement, pitch rotation, forced/missing
   handling, and ordinary expected-value decoding. Enabling the flag must not
   change prediction values, pass-score arithmetic, exported value tables, or
   selected comparison pairs.

4. Apply the same policy to supported saved Hawkeye visualizations.

   Update `scripts/visualization_availability.py::select_saved_components`,
   `scripts/outcome_decomposition_visualization.py::attach_saved`,
   `scripts/saved_visualization_comparison.py`, and the Hawkeye table adapter in
   `datatools/hawkeye.py`. Compute eligible components per source run after
   availability filtering. Load and attach saved heads only for that subset.
   Cover both saved-run comparisons and the fixed-source physical-cache
   comparison path, which currently follows separate setup code.

   Use recorded per-model learning settings/definitions as the source of truth;
   fall back to the exact referenced checkpoint's validated record when needed.
   If eligibility remains unknown, leave labels off with a specific diagnostic
   rather than inferring a learning mode from values or timestamps.

   Runs with no eligible selected components need no sidecar. For a known eligible
   component, a missing sidecar, incomplete finite-prediction coverage, duplicate
   semantic keys, or inconsistent expected values remains an actionable error.
   Retain parent-model, mask, scene, frame, branch, and candidate identity checks;
   never borrow another source run's heads. Continue requiring regeneration of
   eligible exports that lack saved heads.

5. Record the effective behavior and update usage documentation.

   Preserve the requested `decomposition` flag in run metadata and additionally
   record the effective components per source checkpoint and comparison variant.
   Include learning mode and reasons for ineligibility/unknown eligibility.
   Distinguish an empty effective subset from a failed attempt to load required
   heads. Update CLI help and `docs/decomposition_outcome_learning.md`, replacing
   the current all-components restriction with the selective annotation rule.

6. Verify with focused regression tests and a small rendering smoke run.

   Extend `tests/test_outcome_decomposition.py` and existing visualization
   selection, comparison, versioning, and combined-outcome coverage. Test:

   - Flag off; all scalar; ordinary distributional; both decomposition severity
     modes; and scoring/conceding pairs with different learning modes.
   - Mixed plots with the flag, including pass score, combined success/failure,
     pass success, and physical xPass. Only eligible individual components get
     below-dot labels; component selections and numeric values remain identical.
   - Four paired variants matching the supplied modes, reversed variant order,
     repeated checkpoint reuse, flag on/off reuse, and mask/cache comparisons.
   - Saved sources with eligible and ineligible tasks; no eligible components
     without a sidecar; eligible components with missing, partial, duplicate,
     incorrect-model/mask, or numerically inconsistent saved heads; and unknown
     legacy provenance.
   - Existing forced/missing labels, away-team rotation, option conflicts, and
     graph/cache validation behavior.

   Use small fixtures for regression coverage. Run one small Hawkeye rendering
   smoke check with the supplied checkpoint modes where local inputs/cache
   coverage permit it; inspect labels and compare ordinary values with the flag
   disabled. No retraining or checkpoint edits are needed.

## Policy choices and recommendations

| Choice | Options | Recommendation |
| --- | --- | --- |
| Activation | Keep the flag; always annotate eligible models | Keep `--decomposition` opt-in and automatically restrict its effects to eligible sources. |
| No eligible selected components | Fail; continue without added labels | Continue with one informational notice per run and record the empty subset. No new confirmation prompt. |
| Eligible saved source lacks valid heads | Skip labels; fail with recovery instructions | Fail with source/component details and the export regeneration instruction. |
| Unknown legacy saved provenance | Guess; leave labels off; fail the entire run | Leave labels off with a diagnostic. Resolve exact checkpoint metadata first when available. |
| Coach ratings, selections, and saved loc/freeze modes | Extend support; retain existing restrictions | Retain existing restrictions for this change; combining those annotations or adding head support to those modes is separate work. |

The requested eligibility rule and mixed-plot behavior are settled. The remaining
policy choices above have recommended defaults and do not block writing or
reviewing the implementation.

## Implementation verification

The final focused regression run passed 522 tests and eight subtests. Thirteen
pre-existing failures were excluded after reproducing them with the original HEAD
modules executed in memory: five combined-outcome fixture failures and eight
visualization-versioning fixture failures. The new selective-annotation test file
contributes 33 passing cases, including mixed live modes, saved sources, source
identity checks, and fixed-source physical-cache comparisons.

A CPU smoke run used all four supplied scoring/conceding checkpoint pairs and the
specified uncertainty cache for situation
`41558c01-d87a-4efd-b0e9-ddfcf8470a77` at `--time-norm 0`. The unavailable pass-height
plot was explicitly excluded to keep this verification noninteractive. Both flag
settings produced all nine available plot types, with identical aligned prediction
tables. With the flag enabled, the per-pair effective component counts were
`[0, 0, 4, 4]`, producing eight annotated panels. The rendered comparison was also
visually inspected.

Local artifacts are `out/selective_decomposition_regression.xml` and
`out/selective_decomposition_smoke_8edac869/verification.json`; the smoke directory
contains both flag-on and flag-off images and prediction tables.
