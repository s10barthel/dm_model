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
│   └── skillcorner_players.csv
└── readme.md
```

- `code/`: executable SkillCorner validation scripts
- `output/`: generated CSV outputs from the SkillCorner validation flow

## Scripts

### `code/skillcorner_postprocessing.py`

Reads SkillCorner component parquet outputs from `data/component_runs/skillcorner/<component_run_id>/<match_id>/`, reshapes them into long-form model data, calculates `pass_score`, `risk`, `reward`, and `game_state_value`, joins those scores back onto filtered SkillCorner event rows, and writes:

- `output/skillcorner_summary.csv`

The summary CSV is the row-level input for the follow-up filtering and aggregation step.

### `code/skillcorner_filter.py`

Reads:

- `output/skillcorner_summary.csv`
- the external SkillCorner id mapping CSV (`skillcorner_id.csv`)

Then it:

1. filters the id table to rows with a non-empty `participant`
2. filters SkillCorner action rows to players that exist in that id table
3. writes a raw filtered action export
4. adds `participant`, keeps only the requested scoring columns, removes rows without `dm_score`
5. aggregates scores per participant

It writes:

- `output/skillcorner_actions_raw.csv`
- `output/skillcorner_actions.csv`
- `output/skillcorner_players.csv`

## Outputs

### `output/skillcorner_summary.csv`

Row-level SkillCorner event export produced by `skillcorner_postprocessing.py`. It retains the filtered event data and appends:

- `pass_score`
- `risk`
- `reward`
- `game_state_value`
- `dm_score`

### `output/skillcorner_actions_raw.csv`

The full `skillcorner_summary.csv` table filtered down to rows whose `player_id` exists in the participant id mapping. All remaining original columns are preserved.

### `output/skillcorner_actions.csv`

A reduced row-level action table with only:

- `participant`
- `pass_score`
- `risk`
- `reward`
- `game_state_value`
- `dm_score`
- `match_id`

Rows with empty `dm_score` are removed.

### `output/skillcorner_players.csv`

Participant-level aggregates derived from `skillcorner_actions.csv`, including:

- `actions`
- `pass_score_sum`, `pass_score_avg`, `pass_score_median`
- `risk_sum`, `risk_avg`, `risk_median`
- `reward_sum`, `reward_avg`, `reward_median`
- `game_state_value_sum`, `game_state_value_avg`, `game_state_value_median`
- `dm_score_sum`, `dm_score_avg`, `dm_score_median`
