# Failed-pass success-intent observations

Use `--refresh-failed-pass-trajectories --extend-feature-run-id <source-run>`
with the usual `--num-workers` and `--worker-thread-limit` options. This creates a
derived run, preserves the source, and excludes stale model-mode artifacts.
The source must already contain current canonical pinned endpoints; otherwise run
the existing endpoint/trajectory refresh first.
Then use the existing `--extend-feature-run-id <refreshed-run>
--intended-receiver-model-id <checkpoint>` workflow to regenerate model mode.
Retraining success-intent is not required. The existing
`--replace-intended-receiver-model` still replaces an already active model mode.
For fresh generation, enable `--failed-pass-contact-detection`; extensions inherit
the selected run's policy. Legacy runs keep their recorded legacy policy.

```powershell
python scripts\generate_relevant_features.py `
  --refresh-failed-pass-trajectories `
  --extend-feature-run-id <source-feature-run> `
  --num-workers 6 `
  --worker-thread-limit 1

python scripts\generate_relevant_features.py `
  --extend-feature-run-id <refreshed-feature-run> `
  --intended-receiver-model-id <success-intent-checkpoint> `
  --num-workers 6 `
  --worker-thread-limit 1
```

Only unsuccessful-pass model observations change. Official endpoints, duration,
height, heuristic modes, carry timing and successful-pass inputs stay unchanged.
Detection uses XY; model features retain speed and XYZ direction and the one-second
cap. Each same-nearest-defender encounter within 1 m is gated using incoming and
outgoing velocities over 0.08 s, with a 0.12 s launch guard and 10 degrees OR 5 m/s
absolute speed-change thresholds. Accepted encounters receive local scores;
the maximum in the earliest qualifying cluster determines impact. The preceding
trustworthy frame is the clean cutoff. Up to 0.08 s confirmation lookahead is
allowed; confirmation samples never enter the shortened trajectory.

Checks require finite, contiguous tracking and at least 3 m/s incoming speed.
An isolated position is flagged when its adjacent-step reversal is at least
120 degrees, its midpoint residual exceeds 1 m, and an adjacent step exceeds
60 m/s. These are versioned engineering thresholds, not independently validated
physical limits. No broad-peak rejection or additional smoothing is used.

Per-pass decisions, cutoffs, scores and quality flags are saved in
`failed_pass_trajectories/<match>.parquet`. Refresh records include original and
updated feature vectors. Match summaries and unavailable reasons are in
`failed_pass_trajectory_refresh/`. Saved and on-demand inference consume the same
records. Missing coverage and stale graphs are rejected.

Model mode never falls back to angle-only. Unscorable actions retain unknown model
receivers and diagnostic reasons. Events remain for chronology; downstream datasets
requiring intended receivers skip unavailable targets.

Resume failed or abruptly interrupted processing using the dedicated refresh option with
`--extend-feature-run-id <failed-derived-run> --in-place`. Interrupted cloning must
restart from the original source. Reprocessing uses the original official cutoff,
so repeated refreshes do not cumulatively shorten trajectories. All-unavailable
runs write diagnostics before failing. Full production regeneration and training
remain separate from isolated implementation validation.
