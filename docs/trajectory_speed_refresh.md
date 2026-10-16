# Success-intent trajectory speed refresh

Success-intent inputs retain four values: `[speed, direction_x, direction_y,
direction_z]`. Direction is normalized net ball displacement over the existing
up-to-one-second interval, truncated at reception (or the established out-of-play
endpoint). Speed is now the magnitude of that same 3D displacement divided by
elapsed time between its endpoints. Tracking timestamps are used when available;
the action-endpoint fallback uses frame difference / frame rate. This is **early-pass
net displacement speed**, not instantaneous launch speed or distance travelled
along a curved path. Smoothed ball velocities no longer contribute to this summary.

## Recommended workflow

```powershell
# 1. Create a derived run. Unchanged .pt/.parquet files are hardlinked by default.
python scripts/generate_relevant_features.py --extend-feature-run-id OLD_RUN --refresh-trajectory-speed --run-id NEW_RUN

# 2. Train a new success-intent checkpoint using NEW_RUN and your training settings.
python scripts/train_relevant_models.py --feature-run-id NEW_RUN --only-success-intent

# 3. Replace model-backed receiver annotations in NEW_RUN using the printed model ID.
python scripts/generate_relevant_features.py --extend-feature-run-id NEW_RUN --in-place --replace-intended-receiver-model --intended-receiver-model-id success_intent/NEW_MODEL
```

Use the new checkpoint's actual ID; do not reuse the old checkpoint. Step 3 uses
the existing model-variant addition/replacement mechanism and refreshes its
dependent artifacts. Speed refresh alone leaves labels, model-backed annotations,
their checkpoint provenance, eligibility and minimum pass duration unchanged.

## Refresh options and safety

- `--refresh-trajectory-speed` requires `--extend-feature-run-id` and cannot be
  combined with other generation/refresh operations. Split and filtering settings
  are inherited from the source run.
- Omit `--in-place` to preserve the original experiment. Hardlinks require a
  supporting filesystem on the same volume; `--artifact-copy-mode copy` explicitly
  opts into full copying instead.
- `--in-place` updates the selected run without cloning. `--overwrite-feature-run`
  remains an equivalent compatibility alias; neither accepts `--run-id`.
- `--num-workers` (including `auto`) and `--worker-thread-limit` control match
  staging. The command rewrites only the success-intent speed column and graph
  definition tag; it checks saved directions against source data before publication.
- All graph replacements are complete-file renames, so changes never propagate
  through hardlinks to the original run. Staging needs space for the changed graphs.
- A completed v2 run is an explicit no-op. If interrupted during publication,
  consumers reject the incomplete run. Rerun with `--refresh-trajectory-speed
  --in-place --extend-feature-run-id TARGET_RUN` to repair the target; already
  replaced graphs are handled safely. Inspect diagnostics first if source data changed.
  If interruption occurred while cloning the run itself, create a new derived run
  from the original instead; an incomplete clone is never published as complete.

## Compatibility

`trajectory_speed_definition` is `net_displacement_3d_speed_v2` for new/refreshed
graphs. Missing tags mean `smoothed_mean_3d_speed_v1`. Training records the definition
in checkpoint arguments, metadata and feature signatures. Resume/refit and inference
reject mismatches despite identical tensor dimensions. An old checkpoint is never
upgraded by looking at its mutable source run.

Legacy checkpoints can still use matching stored legacy graphs. On-demand graph
construction uses only v2 and requires a v2 checkpoint. Refresh and retrain rather
than attempting legacy on-demand receiver annotation. Prepared dataset identities
include the definition; changed source signatures invalidate prepared entries.
