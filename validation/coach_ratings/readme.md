# Coach Ratings Visualization

`add_pass_scores.py` joins coach ratings with one Hawkeye component run and writes the enriched
coach-rating CSV. `visualize_coach_ratings.py` renders Hawkeye pass-score clips for coach-rated
situations and overlays:

- model pass-score values above the attacking player dots
- coach `Scores` values below the matched player dots

The script writes one animation and one BallReceipt snapshot per selected situation id.

## CLI flags

- `--component-id <hawkeye_component_run_id>`: required. Selects a Hawkeye component run under `dm_model/data/component_runs/hawkeye/<component-id>`. The scripts read `hawkeye_data.parquet` from that run.
- `--situation-id <id>`: optional, may be passed multiple times. Restricts output to one or more coach-rated Hawkeye situation ids. Default: all ids from `coach_ratings/coach_ratings.csv` that still have at least one non-empty `Scores` value.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `dm_model/hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `dm_model/hawkeye_data/ball_data_selected.csv`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
- `--output-dir <path>`: visualization output directory. Default: `coach_ratings/visualizations`.

`--situation-id`, `--tracking-csv`, `--ball-csv`, `--show-trajectories`, `--gif`, and `--output-dir` apply to `visualize_coach_ratings.py`.

## Outputs

For each selected `id`, the script writes:

- `<id>.mp4` or `<id>.gif`: 1-second clip from `BallReceipt` to `BallReceipt + 1`
- `<id>.png`: snapshot at `abs_time = BallReceipt`

All files are written directly into the output directory. No per-id subfolders are created.

## Example

```powershell
python validation/coach_ratings/code/add_pass_scores.py --component-id hawkeye_component_20260430T093826_929736_42dec1e2
```

```powershell
python validation/coach_ratings/code/visualize_coach_ratings.py --component-id hawkeye_component_20260430T093826_929736_42dec1e2 --situation-id 01edd46e-9542-4f81-aab1-a157c1556ba2
```
