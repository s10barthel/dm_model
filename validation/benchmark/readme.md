# Benchmark Postprocessing

`benchmark_postprocessing.py` reads one benchmark component export, filters rows without `pass_intent`, computes a per-row `pass_score`, aggregates one `game_state_value` per benchmark state, and writes one summary row per modification.

The script compares `game_state` 1 and 2 for each modification and exports:

- `game_state_value_1`
- `game_state_value_2`
- `higher_state_id`
- `model_rating`
- `agreement`

The summary CSV is written to `dm_model/validation/benchmark/benchmark_summary.csv`.

## CLI Flags

- `--component-run-id <component_run_id>`: optional benchmark component run id. If omitted, the script uses the latest benchmark component run registered in `dm_model/data/component_runs/benchmark/latest.json`.

## Input

The script reads:

- `dm_model/data/component_runs/benchmark/<component_run_id>/benchmark_data.csv`

## Output

The script writes:

- `dm_model/validation/benchmark/benchmark_summary.csv`

## Example

```powershell
python dm_model/validation/benchmark/benchmark_postprocessing.py --component-run-id benchmark_component_20260420T104926_654573_56f8e2e7
```

You can also run it without a run id to use the latest benchmark export:

```powershell
python dm_model/validation/benchmark/benchmark_postprocessing.py
```
