# Coach Ratings Visualization

`visualize_coach_ratings.py` renders Hawkeye pass-score clips for coach-rated situations and overlays:

- model pass-score values above the attacking player dots
- coach `Scores` values below the matched player dots

The script writes one animation and one BallReceipt snapshot per selected situation id.

## CLI flags

- `--situation-id <id>`: optional, may be passed multiple times. Restricts output to one or more coach-rated Hawkeye situation ids. Default: all ids from `coach_ratings/coach_ratings.csv` that still have at least one non-empty `Scores` value.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `dm_model/hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `dm_model/hawkeye_data/ball_data_selected.csv`.
- `--component-dir <path>`: Hawkeye component directory containing `hawkeye_data.parquet` and `metadata.json`. Default: `dm_model/data/component_runs/hawkeye`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
- `--output-dir <path>`: visualization output directory. Default: `coach_ratings/visualizations`.

## Outputs

For each selected `id`, the script writes:

- `<id>.mp4` or `<id>.gif`: 1-second clip from `BallReceipt` to `BallReceipt + 1`
- `<id>.png`: snapshot at `abs_time = BallReceipt`

All files are written directly into the output directory. No per-id subfolders are created.

## Example

```powershell
python coach_ratings/visualize_coach_ratings.py --situation-id 01edd46e-9542-4f81-aab1-a157c1556ba2
```
