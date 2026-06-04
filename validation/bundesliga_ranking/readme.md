# Bundesliga Ranking Validation

This folder contains the scripts and intermediate data for validating Bundesliga player rankings against FC25 ratings and model-based action scores.

## Folder structure

```text
validation/bundesliga_ranking/
├── code/
│   ├── bundesliga_ranking.py
│   └── scrape_FIFA_ratings.py
├── output/
│   ├── fc25_ratings.csv
│   ├── fc25_ratings_ids.csv
│   ├── bundesliga_actions.csv
│   ├── bundesliga_matches.csv
│   ├── bundesliga_players.csv
│   └── bundesliga_positions.csv
├── minutes_played/
├── soccerdata_cache/
├── test/
└── tmp/
```

- `code/`: executable scripts
- `output/`: generated CSV outputs
- `minutes_played/`: reusable per-match cache derived from raw Bundesliga XML
- `soccerdata_cache/`: cached SoFIFA HTML pages used by `soccerdata`
- `test/`: tests for validation logic
- `tmp/`: temporary files created during SoFIFA scraping

## Scripts

### `code/scrape_FIFA_ratings.py`

Scrapes FC25 player ratings for the Bundesliga from SoFIFA and writes them to `output/fc25_ratings.csv`.

What it does:
- uses `soccerdata` plus SeleniumBase to fetch the Bundesliga FC25 roster
- reads/caches SoFIFA player pages under `soccerdata_cache/`
- parses rating pages locally, with support for German and English SoFIFA labels
- rejects Cloudflare/block pages instead of caching bad HTML
- validates that all expected rating columns are populated before export

Default output:
- `validation/bundesliga_ranking/output/fc25_ratings.csv`

Relevant settings in the script:
- `LEAGUE = "GER-Bundesliga"`
- `VERSION = 250044`
- `CACHE_ROOT = validation/bundesliga_ranking/soccerdata_cache`
- `OUTPUT_CSV = validation/bundesliga_ranking/output/fc25_ratings.csv`
- `BROWSER_PROFILE_ROOT = dm_model/data/chrome_profiles/sofifa`

How to run:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\scrape_FIFA_ratings.py
```

Notes:
- This script is not fully standalone. It reuses the local SoFIFA cache and may try to open Chrome if cache entries are missing.
- If Chrome/SeleniumBase cannot start, scraping will fail before export.

### `code/bundesliga_ranking.py`

Builds the Bundesliga validation outputs by combining:
- lineup and event data from `dm_model/data`
- component run outputs from `dm_model/data/component_runs`
- FC25 ratings from `output/fc25_ratings.csv`

What it does:
- loads Bundesliga player ids from the lineup parquet
- loads FC25 ratings and matches them to Bundesliga players
- resolves the component run to use
- reads model component outputs for each Bundesliga match
- joins model scores back to synced events
- derives and caches `minutes_played` from raw Bundesliga lineup/event XML
- exports row-level, match-level, player-level, and player-position-level summaries

Default outputs:
- `validation/bundesliga_ranking/output/fc25_ratings_ids.csv`
- `validation/bundesliga_ranking/output/bundesliga_actions.csv`
- `validation/bundesliga_ranking/output/bundesliga_matches.csv`
- `validation/bundesliga_ranking/output/bundesliga_players.csv`
- `validation/bundesliga_ranking/output/bundesliga_positions.csv`

The ranking outputs use the same DM metric set as the SkillCorner validation:
`pass_score`, `risk`, `reward`, `game_state_value_start`, `game_state_value_end`,
`game_state_value_next`, `action_epv`, `dm_score`, `pass_dm_score`, `carry_epv`,
`pass_epv`, `z_dm_score`, `z_pass_dm_score`, and `rank`.

`bundesliga_matches.csv` contains one row per `player_id` and `match_id`, with
`minutes_played`, dominant `advanced_position`, `[metric]_sum`, and `[metric]_per90`.
`bundesliga_players.csv` keeps player totals, averages, medians, and the same
match-derived `[metric]_sum` and `[metric]_per90` columns.

How to run:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\bundesliga_ranking.py
```

Example with an explicit season prefix:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\bundesliga_ranking.py --season-prefix DFL-MAT-J03
```

## Relevant settings

### `bundesliga_ranking.py` command line arguments

- `--season-prefix`
  - default: `DFL-MAT-J04`
  - filters lineup rows and match directories to one DFL season prefix
- `--component-run-id`
  - selects a specific component run id from `data/component_runs`
- `--component-run-root`
  - directly points to a specific component run folder
- `--component-runs-dir`
  - overrides the default `dm_model/data/component_runs`
- `--event-synced-dir`
  - overrides the default `dm_model/data/event_synced`
- `--lineup-path`
  - overrides the default `dm_model/data/lineup/line_up.parquet`
- `--fc25-ratings-path`
  - overrides the default `validation/bundesliga_ranking/output/fc25_ratings.csv`
- `--output-dir`
  - overrides the default `validation/bundesliga_ranking/output`
- `--bundesliga-data-dir`
  - raw Bundesliga season directory used for `minutes_played` derivation
  - can be passed more than once
  - defaults to `Bundesliga_season_23_24` and `Bundesliga_season_24_25`
- `--minutes-played-cache-dir`
  - overrides the default `validation/bundesliga_ranking/minutes_played`
- `--refresh-minutes-played-cache`
  - rebuilds cached `minutes_played` files from raw XML instead of reusing them

Run help:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\bundesliga_ranking.py --help
```

## Recommended workflow

1. Generate or refresh FC25 ratings:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\scrape_FIFA_ratings.py
```

2. Build the Bundesliga validation outputs:

```powershell
.\.venv\Scripts\python.exe validation\bundesliga_ranking\code\bundesliga_ranking.py --season-prefix DFL-MAT-J03
```

Use the season prefix that exists in your current lineup and component-run data.

## Current dependencies and assumptions

- `scrape_FIFA_ratings.py` expects `soccerdata`, `seleniumbase`, `lxml`, and `pandas`
- `bundesliga_ranking.py` expects access to:
  - `dm_model/data/lineup/line_up.parquet`
  - `dm_model/data/event_synced/`
  - `dm_model/data/component_runs/`
  - `Bundesliga_season_23_24/` and/or `Bundesliga_season_24_25/` for `minutes_played`
- `fc25_ratings.csv` must exist before running `bundesliga_ranking.py`
- the default season prefix in the script may need to be overridden if your current data uses a different DFL prefix
