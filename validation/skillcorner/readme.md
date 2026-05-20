# SkillCorner Validation

This folder contains the SkillCorner validation scripts and the CSV outputs they produce.

## Folder Structure

```text
validation/skillcorner/
├── code/
│   ├── skillcorner_postprocessing.py
│   └── skillcorner_filter.py
├── output/
│   ├── skillcorner_summary.csv
│   ├── skillcorner_actions_raw.csv
│   ├── skillcorner_actions.csv
│   ├── skillcorner_matches.csv
│   └── skillcorner_players.csv
└── readme.md
```

- `code/`: executable SkillCorner validation scripts
- `output/`: generated CSV outputs from the SkillCorner validation flow

## Scripts

### `code/skillcorner_postprocessing.py`

Reads SkillCorner component parquet outputs from `data/component_runs/skillcorner/<component_run_id>/<match_id>/`, reshapes them into long-form model data, calculates pass, risk, reward, game-state, EPV, normalized score, and targeted-pass rank metrics, joins those scores back onto filtered SkillCorner event rows, and writes:

- `output/skillcorner_summary.csv`

The summary CSV is the row-level input for the follow-up filtering and aggregation step.

Select the component run with one of:

```powershell
python validation\skillcorner\code\skillcorner_postprocessing.py --component-run-id <component_run_id>
python validation\skillcorner\code\skillcorner_postprocessing.py --component-run-root data\component_runs\skillcorner\<component_run_id>
```

If neither option is provided, the script uses the latest SkillCorner run registered in `data/component_runs/skillcorner/latest.json`. That file is updated by `scripts/run_skillcorner.py` when it creates a run under the default output directory. To choose the run id at creation time, use:

```powershell
python scripts\run_skillcorner.py --run-id <component_run_id>
```

`scripts/run_skillcorner.py` also runs SkillCorner postprocessing and filtering automatically after a successful component run, saving the derived CSVs next to that run's `metadata.json`.

### `code/skillcorner_filter.py`

Reads:

- `output/skillcorner_summary.csv`
- the external SkillCorner id mapping CSV (`skillcorner_id.csv`)

Then it:

1. filters the id table to rows with a non-empty `participant`
2. filters SkillCorner action rows to players that exist in that id table
3. writes a raw filtered action export
4. adds `participant`, keeps the requested scoring and event-context columns, removes rows only when all scoring/metric columns are empty
5. aggregates scores per participant-match
6. aggregates scores per participant

It writes:

- `output/skillcorner_actions_raw.csv`
- `output/skillcorner_actions.csv`
- `output/skillcorner_matches.csv`
- `output/skillcorner_players.csv`

Use `--playing-time {minutes_played,minutes_tip,minutes_otip}` to choose the denominator for per-90 metrics. The default is `minutes_played`.

## Outputs

### `output/skillcorner_summary.csv`

Row-level SkillCorner event export produced by `skillcorner_postprocessing.py`. It retains the filtered event data and appends:

- `pass_score`
- `risk`
- `reward`
- `game_state_value_start`
- `game_state_value_end`
- `game_state_value_next`
- `action_epv`
- `dm_score`
- `pass_dm_score`
- `carry_epv`
- `pass_epv`
- `z_dm_score`
- `z_pass_dm_score`
- `rank`

Metric definitions:

- `pass_score`: targeted receiver pass value selected at the event end frame
- `risk`: expected conceding value for the targeted receiver pass
- `reward`: expected scoring value for the targeted receiver pass
- `game_state_value_start`: start-frame value calculated as `sum(pass_intent * pass_score)` over receivers
- `game_state_value_end`: end-frame value calculated the same way
- `game_state_value_next`: next row's `game_state_value_start`, signed from the current row's team perspective
- `action_epv`: `game_state_value_next - game_state_value_start`
- `dm_score`: default `pass_score - game_state_value_start`; for `foul_suffered`, `game_state_value_end - game_state_value_start`; for `possession_loss`, `action_epv`
- `pass_dm_score`: `pass_score - game_state_value_end`
- `carry_epv`: `game_state_value_end - game_state_value_start`
- `pass_epv`: `game_state_value_next - game_state_value_end`
- `z_dm_score`: `(pass_score - game_state_value_start) / sqrt(pass_score_std_start^2 + r^2)`
- `z_pass_dm_score`: `(pass_score - game_state_value_end) / sqrt(pass_score_std_end^2 + r^2)`
- `rank`: dense descending rank of the targeted receiver's end-frame `pass_score`; best available pass is `1`

For the normalized metrics, `pass_score_std_start` and `pass_score_std_end` are computed from all receiver pass scores at the selected start/end component frame. The stabilizer `r` is the 1st percentile of non-empty start-frame pass-score standard deviations.

### `output/skillcorner_actions_raw.csv`

The full `skillcorner_summary.csv` table filtered down to rows whose `player_id` exists in the participant id mapping. All remaining original columns are preserved.

### `output/skillcorner_actions.csv`

A reduced row-level action table with only:

- `participant`
- `pass_score`
- `risk`
- `reward`
- `game_state_value_start`
- `game_state_value_end`
- `game_state_value_next`
- `action_epv`
- `dm_score`
- `pass_dm_score`
- `carry_epv`
- `pass_epv`
- `z_dm_score`
- `z_pass_dm_score`
- `rank`
- `minute_start`
- `duration`
- `period`
- `player_position`
- `game_state`
- `team_score`
- `opponent_team_score`
- `team_in_possession_phase_type`
- `team_out_of_possession_phase_type`
- `distance_covered`
- `speed_avg`
- `speed_avg_band`
- `separation_start`
- `separation_end`
- `separation_gain`
- `one_touch`
- `quick_pass`
- `carry`
- `pass_outcome`
- `high_pass`
- `player_targeted_xpass_completion`
- `player_targeted_xthreat`
- `end_type`
- `match_id`
- `minutes_tip`
- `minutes_otip`
- `minutes_played`

Rows are removed only when all scoring/metric columns are empty.

### `output/skillcorner_matches.csv`

Participant-match aggregates derived from `skillcorner_actions.csv`, including:

- `participant`, `match_id`
- `player_position`: most frequent position in that participant-match
- `minutes_tip`, `minutes_otip`, `minutes_played`
- `[metric]_sum` and `[metric]_per90` for every action metric listed in `skillcorner_actions.csv`

Per-90 metrics use the playing-time column selected by `--playing-time` and are empty when the selected playing time is empty or zero.

### `output/skillcorner_players.csv`

Participant-level aggregates derived from `skillcorner_actions.csv` and `skillcorner_matches.csv`, including:

- `actions`
- `minutes_tip`, `minutes_otip`, `minutes_played`
- `[metric]_sum`: sum of match-level metric sums
- `[metric]_per90`: average of match-level per-90 values
- existing action-level averages and medians such as `pass_score_avg`, `pass_score_median`, `rank_avg`, and `rank_median`
