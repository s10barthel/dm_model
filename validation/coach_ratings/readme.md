# Coach Ratings Visualization

## Data preparation

### `preprocess_coach_ratings.py`

Builds the common coach-ratings dataset by combining Patrick's and Jelle's questionnaire exports.
It normalizes the `uefa_player_id` and `id` keys, resolves duplicate key rows, and adds the
question/player numbers, scores, individual Jelle coach ratings, and the source label (`mrp`).

- **Inputs:** `questionnaire_data/xy_data_patrick.csv`, `xy_data_jelle.csv`,
  `coach_ratings_patrick.xlsx`, `coach_ratings_metadata_jelle.xlsx`, and
  `coach_ratings_jelle.csv`.
- **Output:** `output/preprocessed_coach_ratings.csv`.
- **Dataset role:** complete, auditable source dataset; it retains rows even when coach `Scores`
  are missing.
- **CLI options:** none currently; all input and output paths are defined in the script.

Run it with:

```powershell
python validation/coach_ratings/code/preprocess_coach_ratings.py
```

### `add_pass_scores.py`

Joins scored coach-rating rows to one Hawkeye component run via `id` and `uefa_player_id`. It
calculates maximum, mean, median, and BallReceipt pass scores, keeps the features of the best pass,
and ranks coach and model scores within each situation.

- **Inputs:** `output/preprocessed_coach_ratings.csv` and
  `data/component_runs/hawkeye/<component-id>/hawkeye_data.parquet`.
- **Output:** `output/<component-id>_coach_ratings.csv`, including pass-score features and
  coach/model ranking columns. It excludes rows without a non-empty coach `Scores` value, making
  it ready for coach-rating validation analyses.
- **CLI options:** `--component-id <hawkeye_component_run_id>` (**required**) selects the Hawkeye
  component-run directory.

Run it with:

```powershell
python validation/coach_ratings/code/add_pass_scores.py --component-id <hawkeye_component_run_id>
```

### `coach_ratings_analysis.R`

The R analysis has moved to the sibling `data_analysis` workspace:
`data_analysis/code/coach_ratings_analysis.R`. It consumes a versioned
`output/<component-id>_coach_ratings.csv` export; its CLI options, analysis description, and
versioned results are documented in `data_analysis/readme.md`.

```powershell
Rscript ../data_analysis/code/coach_ratings_analysis.R --component-id <hawkeye_component_run_id>
```

## Visualization

`visualize_coach_ratings.py` renders Hawkeye pass-score clips for coach-rated situations and
loads the component-independent `output/preprocessed_coach_ratings.csv` for coach annotations.
It overlays:

- model pass-score values above the attacking player dots
- coach `Scores` values below the matched player dots

The script writes one animation and one BallReceipt snapshot per selected situation id.

## CLI flags

- `--component-id <hawkeye_component_run_id>`: required. Selects a Hawkeye component run under `dm_model/data/component_runs/hawkeye/<component-id>`. The scripts read `hawkeye_data.parquet` from that run.
- `--situation-id <id>`: optional, may be passed multiple times. Restricts output to one or more coach-rated Hawkeye situation ids. Default: all ids from `output/preprocessed_coach_ratings.csv` that still have at least one non-empty `Scores` value.
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
