# Progression targets

`--target-family progression` trains scoring on the fraction of starting distance
to goal eliminated by the selected future position. Conceding uses the existing
`concedes_goal_distance` label unchanged.

For starting distance `d0 > 0`, the scoring label is
`clip((d0 - d_future) / d0, 0, 1)`. Distance is Euclidean distance to the centre of
the opponent goal. No future teammate candidate, no advancement, or a start within
one micrometre of goal gives zero. Missing/nonfinite starting coordinates are
reported as errors, not treated as zero progression.

Supported return types are `next_N`, `next_N_skip1`, and `in_N`. They retain the
goal-distance selection rules, eligible actions, team perspective, and boundaries.
`in_N` uses its selected state (including its existing early-shot rule), not a
window maximum. Discounted returns are unsupported. Mixed feature runs retain
their discounted artifacts with unavailable (`NaN`) progression targets.

Fresh label generation appends `scores_progression` to the existing schema.
Training/evaluation reject unavailable progression labels with an extension
instruction. Existing target families remain usable with older feature runs.

## Extend an existing run

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <source-feature-run-id> `
  --refresh-target-family progression `
  --artifact-copy-mode hardlink
```

This creates a derived run and prints its ID. Unchanged graphs are hard-linked;
changed labels and resolved-action tables are staged and replaced atomically.
The operation reads saved goal-distance returns and resolved action positions;
it does not load graphs, tracking, or receiver models, or recompute return windows.
Carries use their own starting positions and their already-resolved return labels.
All recorded receiver modes, carry variants, intent-training, augmented and
success-intent labels are covered. Repeating the refresh makes no artifact changes
when the targets already match.

Run this operation separately from other extensions. Add `--in-place` to update
the source run instead; avoid concurrent readers during an in-place refresh.
Hard links require a compatible filesystem on the same volume; use
`--artifact-copy-mode copy` explicitly if needed.

Check `metadata.json` for `status: completed` and `progression_targets`, and
`progression_refresh_report.json` for rewritten file/byte counts. Source signatures
are checked before replacement; replacement is atomic per file, not across the
entire run. Failed/partial extensions follow the existing feature-run failure
handling. The extension requires labels containing the existing pass-height
columns; older schemas must be upgraded first.

Use the resulting run with your normal training options:

```powershell
.\.venv\Scripts\python.exe scripts/train_relevant_models.py `
  --feature-run-id <extended-feature-run-id> `
  --intended-receiver-mode original `
  --target-family progression --return_type next_3
```

Prepared dataset caches incorporate source signatures and preparation-code hashes;
they are rebuilt as necessary without manually deleting existing caches.
