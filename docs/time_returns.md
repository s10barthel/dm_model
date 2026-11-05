# Time-window returns

Use `time_<seconds>` wherever a return type is accepted, for example
`--return-type time_10` or `--return-type time_2.5`. Durations must be finite
and positive. Equivalent names such as `time_10.0` normalize to `time_10`.
`_skip1`, `+`, and `_spstop` suffixes are unsupported; use the existing
`--stop-condition base|set-pieces` option for boundaries.

The window starts at the labelled action's synchronized start frame. The
pipeline clock is `frame_id / 25` seconds; recorded `seconds`, `utc_timestamp`,
and reception/end times do not replace missing start frames. This matches the
feature pipeline's 25 fps clock and also works for synthetic carries, whose
clock starts at the carry segment's actual start rather than its terminal
`return_event_index`.

An action starting exactly at the cutoff is included. Its existing target value
contributes even if the action finishes after the cutoff. The clock measures
elapsed time, including stoppages, until an existing stop boundary ends the
window. Goals are included and then stop the window; periods and subsequent
goal-kicks stop it. `set-pieces` also stops before the configured restarts.
A labelled restart can look into its own play.

The initial action contributes to goal and xG returns. xT, goal-distance, and
EPV returns use future eligible passes, crosses, and shots. xG combines shot
probabilities as `1 - product(1 - xG)`; the other families retain their existing
goal/max aggregation and team perspectives. Progression derives from the
time-window goal-distance return. xT gain targets retain action-start scoring
baselines and, for definition version 2, opponent-acquisition conceding
baselines. Logarithmic-progression xT surfaces support time returns.

Canonical event identities and stop segments are preserved. Future canonical
events are selected by elapsed time even if their frame IDs are out of order;
events earlier than the labelled action in canonical order or synchronized
time cannot contribute. Equal-time future events retain canonical order.
Carries select candidates in the play containing their actual start.

Missing, negative, noninteger, or nonfinite start frames produce `NaN` returns.
An untimed future candidate also makes the affected return unknown: because its
position relative to the cutoff cannot be established, preceding windows in
the same stop segment receive `NaN`. Windows after that candidate and in other
segments can still be labelled. No timestamp estimation or alternate clock is
used. Empty, fully timed windows have zero raw return.

Generation prints missing-timing and missing-return action counts. Refresh
reports save `time_return_coverage` entries by match, return type, stop condition,
receiver mode, and carry variant, plus aggregate `time_return_totals`. Counts
refer to actions, not target cells; variants are reported separately. The
versioned definition is saved as `time_returns` in run metadata.

Timed-outcome training and evaluation exclude rows lacking the selected target
pair and report `excluded_by_missing_time_return` in dataset selection counts.
Labels remain `NaN` on disk. Other tasks can still use the same actions.

To add these labels to a migrated feature run without rebuilding its graphs:

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <source> --return-type time_10
```

Use the source run's split selector (`--train-count` or `--train-split`) when
required by the extension workflow. Derived runs preserve source artifacts.
