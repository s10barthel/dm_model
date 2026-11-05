# Empty future-window targets

New feature runs exclude actions with no eligible future action from xT,
goal-distance, EPV and progression outcome training and evaluation. This also
applies to xT gain targets, before or after negative-gain clipping. Both scoring
and conceding targets are unavailable for an empty window. Goal and xG labels
retain their existing initial-action rules.

The policy applies under both `base` and `set-pieces` stops. It covers `next_N`
and its variants, `time_<seconds>`, `extended_possession[_N]`, and discounted
future-window returns. `in_N` is a specific-state selection and retains its
existing definition. An action removed by `_skip1` does not make a contribution;
if that leaves the selected window empty, the spatial targets are unavailable.

Membership depends on action type, team, saved anchors and the selected horizon,
never on target values. A nonempty window with a zero return, or actions from
only one team, remains valid. Missing target annotations and unknown timing
retain their separate validation and availability rules.

Graphs and action identities are retained. A trailing `empty_window_reason`
label column records availability (0), a set-piece stop (1), a goal (2), period
end (3), match end (4), a time limit (5), or `_skip1` (6). Affected target cells
are NaN. Dataset selection uses the explicit reason for the selected family;
arbitrary nonfinite spatial targets are not silently accepted.

The standard `outcome_scoring` and `outcome_conceding` tasks already exclude
labelled shots. Future shots still contribute to earlier actions' returns.
Refresh reports count all saved actions; dataset exclusion counts are after
the normal task eligibility filters, including that shot exclusion.

## Migrate a saved run

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <source> --refresh-empty-window-policy `
  --num-workers 8 --worker-thread-limit 1
```

This dedicated operation creates a derived run. It inherits the source split,
preserves its recorded surfaces and target representations, and migrates every
existing return/stop/receiver/carry variant. It does not add set-piece variants.
`--in-place` and combinations with other refreshes are rejected. A legacy run
must first complete `--refresh-stop-conditions`.

The refresh only needs saved synchronized events (preferring frozen pass
endpoints), resolved actions, existing labels and existing EPV certification
records. Extended `next_N+_skip1` windows also reuse frozen acquisition locations
when available. It does not recompute returns or read target annotation files, graphs
or tracking. It uses the shared bounded process-worker runtime, honors
`--num-workers <N|auto>` and `--worker-thread-limit`, and prepares each match
once while caching timelines and membership across variants.

Workers stage their output; the parent verifies input fingerprints and artifact
signatures before replacing files. Replacements preserve hard-linked source
artifacts. A repeated migration is idempotent. `empty_window_refresh_report.json`
records progress, workers, exclusions by reason and variant, and rewritten files.

The `empty_window_policy` definition is saved in feature metadata and checkpoint
stop-condition definitions. Existing runs and models retain their old policy
until an explicit migration. Generation and later target refreshes preserve the
run's recorded policy.
