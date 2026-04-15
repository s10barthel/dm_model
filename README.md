# dm_model

This repository is a scoped adaptation of the upstream DEFCON implementation for Sportec Bundesliga data, currently centered on season-based training/testing across 2023/24 and 2024/25.
Upstream DEFCON source code: https://github.com/hyunsungkim-ds/defcon

The model structure is copied from DEFCON, the upstream source code for the paper "Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks". This repository keeps the downstream DEFCON runtime close to upstream, narrows the retained scope, and adds the dataset- and workflow-specific adaptations needed for Sportec, HawkEye, and SkillCorner data.

## Table Of Contents

- [Changes From Upstream DEFCON](#changes-from-upstream-defcon)
- [Upstream Baseline](#upstream-baseline)
- [Expected Raw Data Layout](#expected-raw-data-layout)
- [Produced Data Layout](#produced-data-layout)
- [Run-Id Workflow](#run-id-workflow)
- [Split Definition](#split-definition)
- [Environment Setup](#environment-setup)
- [Main Pipeline Runner](#main-pipeline-runner)
- [End-to-End Workflow](#end-to-end-workflow)
  - [1. Preprocess Sportec data](#1-preprocess-sportec-data)
  - [2. Generate soft-target artifacts when xT or goal_distance targets are needed](#2-generate-soft-target-artifacts-when-xt-or-goal_distance-targets-are-needed)
  - [3. Generate graph features and labels](#3-generate-graph-features-and-labels)
  - [4. Train the retained models](#4-train-the-retained-models)
  - [5. Evaluate the retained models on the test set](#5-evaluate-the-retained-models-on-the-test-set)
  - [6. Export per-match component predictions](#6-export-per-match-component-predictions)
  - [7. Visualize one action at a time](#7-visualize-one-action-at-a-time)
  - [8. Run frame-level inference on HawkEye data](#8-run-frame-level-inference-on-hawkeye-data)
  - [9. Run frame-level inference on SkillCorner data](#9-run-frame-level-inference-on-skillcorner-data)
- [Outcome Target Selection](#outcome-target-selection)
  - [`--return_type` is xG-only](#return_type-is-xg-only)
  - [Where to switch targets](#where-to-switch-targets)
- [Notes](#notes)
- [CLI Reference](#cli-reference)

## Changes From Upstream DEFCON

The main adaptations in this repository are:

- Sportec-specific preprocessing and synchronization to turn Sportec XML tracking/event data into DEFCON-compatible inputs
- xT as an alternative soft target variable for `outcome_scoring` and `outcome_conceding`
- temporal `t0 / t-12 / t-25` augmentation for `action_intent` and `pass_intent` training data
- narrowed modeled pass family: only `pass` and `cross` are treated as pass-like actions, so set pieces are excluded from the modeled pass category
- a HawkEye inference/visualization pipeline for frame-level external data
- a SkillCorner inference/visualization pipeline for synchronized external tracking+event data
- a HawkEye BallReceipt freeze option that can freeze the possessor and ball state after the receipt frame

These changes are implementation adaptations to upstream DEFCON behavior, not separate model families or new research claims.

The retained model scope here is:

- `(a1)` action selection probability
- `(b1)` pass success probability
- `(c1)` outcome-conditioned goal-scoring probability
- `(c2)` outcome-conditioned goal-conceding probability

The following upstream parts are intentionally out of scope here:

- `(b2)` shot-blocking probability
- `(c3)` UxG
- `(d1)` defender responsibility / player defensive scores

## Upstream Baseline

- DEFCON snapshot: `_vendor/defcon` at commit `f8a3a2b987b376221afe30453b0f145b965edcae`
- ELASTIC snapshot: `_vendor/elastic` at commit `bc41bcdf43451ae639c6ae7b299c1ccd3712d00e`

The copied/adapted runtime files live at the project root. `_vendor` is kept as a reference copy of the upstream baseline.

## Expected Raw Data Layout

The raw Sportec XML files are expected under:

- `Bundesliga_season_23_24/tracking_data`
- `Bundesliga_season_23_24/event_data`
- `Bundesliga_season_23_24/match_information`
- `Bundesliga_season_24_25/tracking_data`
- `Bundesliga_season_24_25/event_data`
- `Bundesliga_season_24_25/match_information/master`
- `Bundesliga_season_24_25/match_information/starting_players`
- `Bundesliga_season_24_25/KPI_Merged`

`scripts/preprocess_sportec.py` now handles the observed raw-format differences between the two seasons and still writes the same canonical processed outputs under `data/ajax`.

## Produced Data Layout

After preprocessing, the project writes DEFCON-style files under `data/ajax`:

- `tracking/*.parquet`
- `tracking_processed/*.parquet`
- `event/event.parquet`
- `event_synced/*.csv`
- `xT/xT.csv`
- `goal_distance/goal_distance.csv`
- `xT/xT_grid.csv`
- `xT/fit_metadata.json`
- `xT/matches/*.csv`
- `lineup/line_up.parquet`
- `features/runs/<feature_run_id>/...`
- `features/runs/latest.json`
- `component_runs/<component_run_id>/...`
- `component_runs/latest.json`
- `component_runs/hawkeye/<component_run_id>/...`
- `component_runs/skillcorner/<component_run_id>/...`
- `splits/match_splits.json`

`event_synced` is stored as CSV because the upstream DEFCON code reads CSV, even though the README mentions Parquet.

The main output directories are:

- `data/ajax/event_synced` for canonical synced event tables used by the DEFCON-style pipeline
- `data/ajax/xT` for xT exports and xT fitting artifacts
- `data/ajax/features/runs/<feature_run_id>` for versioned graph tensors and label tensors used by training/evaluation
- `data/ajax/component_runs/<component_run_id>` for versioned per-match component prediction exports from `scripts/run_relevant_models.py`
- `data/ajax/component_runs/hawkeye/<component_run_id>` for versioned HawkEye exports
- `data/ajax/component_runs/skillcorner/<component_run_id>` for versioned SkillCorner exports
- `saved/<task>/<model_run_id>` for trained checkpoint runs
- `saved/bundles/<bundle_id>` for machine-readable training bundle manifests

## Run-Id Workflow

There are now three different versioned artifact families:

- `feature_run_id` for generated Sportec graph/label artifacts under `data/ajax/features/runs/...`
- `model_id = <task>/<model_run_id>` for trained checkpoints under `saved/<task>/...`
- `component_run_id` for exported predictions under `data/ajax/component_runs/...`

Feature and component runs use explicit `latest.json` pointers:

- `data/ajax/features/runs/latest.json`
- `data/ajax/component_runs/latest.json`
- `data/ajax/component_runs/hawkeye/latest.json`
- `data/ajax/component_runs/skillcorner/latest.json`

Checkpoint runs are resolved from checkpoint metadata instead of one global pointer. The wrappers scan `saved/<task>/*/metadata.json` and choose the latest compatible checkpoint for the requested task/context.

The default behavior is `latest + pin`:

- if you do not pass a run id or model id, downstream scripts resolve the latest compatible artifact automatically
- if reproducibility matters, pass explicit run ids and explicit model ids
- if multiple compatible checkpoints exist but differ in feature signature, the wrapper fails and requires explicit model ids instead of guessing

The feature run root mirrors the old flat layout, but inside one dedicated run folder:

- `data/ajax/features/runs/<feature_run_id>/action_graphs/*.pt`
- `data/ajax/features/runs/<feature_run_id>/post_action_graphs/*.pt`
- `data/ajax/features/runs/<feature_run_id>/action_graphs_intent_train/*.pt`
- `data/ajax/features/runs/<feature_run_id>/action_graphs_success_intent/*.pt`
- `data/ajax/features/runs/<feature_run_id>/action_labels_disc_0.9*.pt`
- `data/ajax/features/runs/<feature_run_id>/action_labels_intent_train_disc_0.9*.pt`
- `data/ajax/features/runs/<feature_run_id>/resolved_actions*.parquet`
- `data/ajax/features/runs/<feature_run_id>/metadata.json`

The component run root contains one folder per processed match:

- `data/ajax/component_runs/<component_run_id>/<match_id>/action_intent.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/pass_intent.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/pass_success.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
- `data/ajax/component_runs/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
- `data/ajax/component_runs/<component_run_id>/metadata.json`

The external-data adapters follow the same pattern:

- `data/ajax/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
- `data/ajax/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
- `data/ajax/component_runs/hawkeye/<component_run_id>/metadata.json`
- `data/ajax/component_runs/skillcorner/<component_run_id>/<match_id>/*.parquet`
- `data/ajax/component_runs/skillcorner/<component_run_id>/metadata.json`

Checkpoint runs also write metadata:

- `saved/<task>/<model_run_id>/args.json`
- `saved/<task>/<model_run_id>/metadata.json`
- `saved/<task>/<model_run_id>/best_weights.pt`
- `saved/bundles/<bundle_id>/metadata.json`

The run metadata records the relevant toggles used for that invocation. For component runs this includes the per-model feature signatures, so settings such as `poss_vel_aware`, `ball_z_aware`, `extend_features`, and `add_v_edge_features` are visible in the saved metadata.

## Split Definition

The dataset split is season-based:

1. all `Bundesliga_season_23_24` matches become the train pool
2. all `Bundesliga_season_24_25` matches become the test set

To keep upstream `train.py` behavior, the `23_24` train pool is deterministically split again into:

- `80%` model-training matches
- `20%` validation matches

The internal train/validation split is deterministic: sort the `23_24` train-pool matches by actual `KickoffTime`, break ties with `MatchId`, then take the first `80%` for training and the remaining `20%` for validation.

## Environment Setup

Create and activate a virtual environment first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then install PyTorch and PyTorch Geometric for the machine that will actually run the models inside that virtual environment.

1. Install `torch`, `torchvision`, and `torchaudio` using the official PyTorch command for your CPU/CUDA setup.
2. Install `torch_geometric` against that PyTorch build.

## Main Pipeline Runner

Use `scripts/main.py` when you want the scoped end-to-end runner described in this README without any visualization steps.

```powershell
python scripts/main.py
python scripts/main.py --use_xt
python scripts/main.py --use_goal_distance
```

The runner executes these stages in order:

1. `scripts/preprocess_sportec.py`
2. `scripts/generate_xt.py` only when `--use_xt` is enabled, or `scripts/generate_goal_distance.py` only when `--use_goal_distance` is enabled
3. `scripts/generate_relevant_features.py`
4. `scripts/train_relevant_models.py`
5. `scripts/evaluate_relevant_models.py`
6. `scripts/run_relevant_models.py`
7. `scripts/run_hawkeye.py`
8. `scripts/run_skillcorner.py`

Useful options:

- `--skip-preprocess`, `--skip-xt`, `--skip-features`, `--skip-train`, `--skip-evaluate`, `--skip-run-relevant`, `--skip-hawkeye`, `--skip-skillcorner`
- `--add_v_edge_features` to append the optional velocity-angle edge features during feature generation
- `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, and `--extend-features` / `--no-extend-features` to override the training feature profile passed into `scripts/train_relevant_models.py`
- `--overwrite` to rebuild supported preprocessing and xT outputs
- `--relevant-split train|test|all` to control `scripts/run_relevant_models.py`
- `--device cuda:0|cpu` for evaluation and inference scripts
- `--dry-run` to print the resolved commands without running them

When `scripts/main.py` triggers feature generation itself, it creates fresh feature run ids and passes them through to downstream train/evaluate/inference steps automatically.
Checkpoint selection inside the downstream training/evaluation/inference scripts still follows the metadata-based latest-compatible resolution described above.

## End-to-End Workflow

The sections below describe each stage individually. `scripts/main.py` wraps steps 1-6, 8, and 9, and intentionally excludes the visualization scripts.

### 1. Preprocess Sportec data

```powershell
python scripts/preprocess_sportec.py
```

Useful options:

- `--match-id DFL-MAT-...` to process only selected matches
- `--limit N` to smoke-test on the first `N` matches
- `--overwrite` to rebuild existing outputs
- `--skip-sync` to stop before ELASTIC synchronization

This step does all custom work that DEFCON does not provide:

- Sportec XML discovery
- lineup reconstruction
- derivation of `mins_played`, `start_time`, and `end_time` from substitutions, red cards, and final whistles
- Kloppy event/tracking conversion
- SPADL action generation with `socceraction`
- ELASTIC synchronization
- split-manifest creation

### 2. Generate soft-target artifacts when xT or goal_distance targets are needed

```powershell
python scripts/generate_xt.py
python scripts/generate_goal_distance.py
```

Useful options:

- `--match-id DFL-MAT-...` to export only selected matches into `data/ajax/xT`
- `--limit N` to restrict xT export generation to the first `N` available synced matches
- `--overwrite` to rebuild existing xT outputs

This is a separate post-preprocessing step. It:

- reads canonical synced event CSVs from `data/ajax/event_synced`
- fits the xT surface on the train split only
- uses only `pass`, `cross`, and `shot` actions for xT fitting/export
- writes:
  - `data/ajax/xT/xT.csv`
  - `data/ajax/xT/xT_grid.csv`
  - `data/ajax/xT/fit_metadata.json`
  - per-match sidecar files under `data/ajax/xT/matches/`
  - `data/ajax/goal_distance/goal_distance.csv`
  - `data/ajax/goal_distance/metadata.json`
  - per-match sidecar files under `data/ajax/goal_distance/matches/`

The aggregate `xT.csv` contains only `pass`, `cross`, and `shot` rows, with both `xG` and xT-derived target columns.
The aggregate `goal_distance.csv` contains only `pass`, `cross`, and `shot` rows, with `goal_distance`, `scores_goal_distance`, and `concedes_goal_distance`.

### 3. Generate graph features and labels

```powershell
python scripts/generate_relevant_features.py
python scripts/generate_relevant_features.py --run-id feature_20260414T123456_abcdef12
python scripts/generate_relevant_features.py --add_v_edge_features
```

Run this again after `scripts/generate_xt.py` or `scripts/generate_goal_distance.py` whenever you want to train outcome models with `--use_xt` or `--use_goal_distance`. The soft-target exports alone are not enough; the label tensors under `data/ajax/features` must be regenerated so they include the corresponding target columns.

Each invocation creates a new feature-artifact run under `data/ajax/features/runs/<feature_run_id>/` and updates `data/ajax/features/runs/latest.json`.

Useful options:

- `--run-id <feature_run_id>` to pin the created run id instead of auto-generating one
- `--add_v_edge_features` to append the optional velocity-angle edge features
- `--use-original-intended-receiver` and `--use-intended-receiver-model` to choose intended-receiver resolution mode

This writes, inside the run root:

- `action_graphs/*.pt`
- `post_action_graphs/*.pt`
- `action_labels_disc_0.9*.pt`
- `action_graphs_intent_train/*.pt`
- `action_labels_intent_train_disc_0.9*.pt`
- `action_graphs_success_intent/*.pt`
- `resolved_actions*.parquet`
- `metadata.json`

### 4. Train the retained models

```powershell
python scripts/train_relevant_models.py
python scripts/train_relevant_models.py --feature-run-id <feature_run_id>
python scripts/train_relevant_models.py --bundle-id model_bundle_20260414T123456_abcdef12
```

This trains the four retained models and the auxiliary upstream-compatible models needed by the retained wrapper. New checkpoints get auto-generated model ids of the form:

- `pass_intent/<model_run_id>`
- `action_intent/<model_run_id>`
- `pass_success/<model_run_id>`
- `outcome_scoring/<model_run_id>`
- `outcome_conceding/<model_run_id>`
- `failure_receiver/<model_run_id>`

The wrapper also writes one bundle manifest under `saved/bundles/<bundle_id>/metadata.json` so later steps can reuse the exact produced model ids without relying on ambient "latest" behavior.

Wrapper behavior:

- `python scripts/train_relevant_models.py` keeps the current retained default, which trains the outcome models with `--use_xg`
- `python scripts/train_relevant_models.py --use_xt` switches only `outcome_scoring` and `outcome_conceding` to xT targets while keeping separate xT checkpoints
- `python scripts/train_relevant_models.py --use_goal_distance` switches only `outcome_scoring` and `outcome_conceding` to goal-distance targets
- `action_intent`, `pass_intent`, and `pass_success` are unchanged by the xT toggle
- unless you override them explicitly, all wrapper-trained models now use the same feature defaults: `possessor_aware`, `keeper_aware`, `ball_z_aware`, and `poss_vel_aware` on; `extend_features` and `xy_only` off
- the wrapper exposes `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, and `--extend-features` / `--no-extend-features`
- `--extend-features` requires possessor-aware features; `scripts/train_relevant_models.py` and `scripts/main.py` fail early if you request `--no-possessor-aware --extend-features`
- if `--feature-run-id` is omitted, the wrapper resolves the latest successful feature run automatically
- if you pass `--bundle-id`, that manifest id is pinned instead of auto-generated
- legacy numeric checkpoint ids still load, but newly trained checkpoints now default to auto-generated model run ids

Model checkpoints are saved under `saved/<task>/<model_run_id>/`, with `args.json`, `metadata.json`, and the saved weights in each run folder.

### 5. Evaluate the retained models on the test set

```powershell
python scripts/evaluate_relevant_models.py
python scripts/evaluate_relevant_models.py --outcome-scoring-model-id outcome_scoring/<model_run_id> --outcome-conceding-model-id outcome_conceding/<model_run_id>
python scripts/evaluate_relevant_models.py --feature-run-id <feature_run_id>
```

This runs the original `test.py` flow for the retained model family:

- `action_intent/<model_run_id>`
- `pass_success/<model_run_id>`
- `outcome_scoring/<model_run_id>`
- `outcome_conceding/<model_run_id>`

`test.py` uses the target configuration saved inside the checkpoint. There is no separate `--use_xg` / `--use_xt` / `--use_goal_distance` switch at evaluation time for an already trained model.

If `--feature-run-id` is omitted, evaluation resolves the latest successful feature run automatically.

If the model ids are omitted, the wrapper resolves the latest compatible checkpoints automatically. If multiple compatible checkpoints exist with different feature signatures, it fails and asks for explicit model ids instead of guessing.

### 6. Export per-match component predictions

```powershell
python scripts/run_relevant_models.py --split test
python scripts/run_relevant_models.py --split test --feature-run-id <feature_run_id>
python scripts/run_relevant_models.py --split test --feature-run-id <feature_run_id> --run-id component_20260414T123456_abcdef12
```

Each invocation creates a new component run under `data/ajax/component_runs/<component_run_id>/` and updates `data/ajax/component_runs/latest.json`.

Useful options:

- `--feature-run-id <feature_run_id>` to pin the feature artifacts used for inference
- `--run-id <component_run_id>` to pin the created component run id instead of auto-generating one
- `--match-id DFL-MAT-...` to restrict inference to one or more matches

For each processed match, this writes:

- `action_intent.parquet`
- `pass_intent.parquet`
- `pass_success.parquet`
- `outcome_scoring_success.parquet`
- `outcome_scoring_failure.parquet`
- `outcome_conceding_success.parquet`
- `outcome_conceding_failure.parquet`

under `data/ajax/component_runs/<component_run_id>/<match_id>/`, together with `data/ajax/component_runs/<component_run_id>/metadata.json`.

If the model ids are omitted, the wrapper resolves the latest compatible checkpoints automatically. The component-run metadata records the resolved model ids and their feature signatures.

### 7. Visualize one action at a time

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --feature-run-id <feature_run_id>
```

`--action-id` refers to the `action_id` column in `data/ajax/event_synced/<match_id>.csv`.

The script always writes 8 PNGs under `data/ajax/visualizations/<match_id>/<action_id>/`:

- `action_intent.png`
- `pass_intent.png`
- `pass_success.png`
- `outcome_scoring_success.png`
- `outcome_scoring_failure.png`
- `outcome_conceding_success.png`
- `outcome_conceding_failure.png`
- `pass_score.png`

Useful option:

- `--show-trajectories` to render dashed recent player trajectories
- `--feature-run-id <feature_run_id>` to use a specific versioned feature run; otherwise the latest successful feature run is used
- `--row-index <index>` to use the legacy internal modeled-action row index instead of the CSV `action_id`
- `--original-event-id <sportec_event_id>` to look up the action by the raw Sportec event id

### 8. Run frame-level inference on HawkEye data

```powershell
python scripts/run_hawkeye.py
python scripts/run_hawkeye.py --run-id hawkeye_component_20260414T123456_abcdef12
```

This processes `hawkeye_data/centroid_data_team.csv` and `hawkeye_data/ball_data_selected.csv` and writes versioned consolidated component outputs to:

- `data/ajax/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
- `data/ajax/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
- `data/ajax/component_runs/hawkeye/<component_run_id>/metadata.json`

Useful options:

- `--situation-id ...` to restrict inference to selected situations
- `--limit N` to smoke-test on the first `N` situations
- `--run-id <component_run_id>` to pin the created HawkEye export run id instead of auto-generating one
- `--no-freeze-ballreceipt` to disable the default BallReceipt freeze for the possessor and the ball

If the model ids are omitted, the script resolves the latest compatible checkpoints automatically. The saved metadata records the resolved model ids and their feature signatures.

To visualize one HawkEye situation as MP4s:

```powershell
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id>
```

This writes 8 MP4s under `data/ajax/visualizations/hawkeye/<situation_id>/`.

Useful option:

- `--component-run-id <component_run_id>` to read a specific versioned Hawkeye component export; otherwise the latest successful Hawkeye component run is used
- `--component-dir <path>` to point directly at a Hawkeye component-run root
- `--gif` to write GIFs instead of the default MP4 animations

`scripts/visualize_hawkeye.py` now reads the probabilities from `scripts/run_hawkeye.py` outputs and rebuilds only the raw HawkEye geometry for rendering. If you want the old direct-inference behavior, use:

```powershell
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id>
```

### 9. Run frame-level inference on SkillCorner data

```powershell
python scripts/run_skillcorner.py
python scripts/run_skillcorner.py --run-id skillcorner_component_20260414T123456_abcdef12
```

This reads synchronized SkillCorner files from `skillcorner_data/` and writes versioned per-match parquet outputs under:

- `data/ajax/component_runs/skillcorner/<component_run_id>/<match_id>/`
- `data/ajax/component_runs/skillcorner/<component_run_id>/metadata.json`

The SkillCorner adapter processes `player_possession` events frame by frame and exports the same retained DEFCON components as the Sportec and HawkEye adapters, including `pass_intent`.

Useful options:

- `--match-id ...` to restrict inference to selected matches
- `--limit N` to smoke-test on the first `N` selected matches
- `--run-id <component_run_id>` to pin the created SkillCorner export run id instead of auto-generating one

If the model ids are omitted, the script resolves the latest compatible checkpoints automatically. The saved metadata records the resolved model ids and their feature signatures.

To visualize one SkillCorner possession as MP4s:

```powershell
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index>
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id>
```

This writes 8 MP4s under `data/ajax/visualizations/skillcorner/<match_id>/<index>/`.

Useful option:

- `--gif` to write GIFs instead of the default MP4 animations
- `--component-run-id <component_run_id>` to read a specific versioned SkillCorner component export; otherwise the latest successful SkillCorner component run is used

## Outcome Target Selection

Outcome target selection affects `outcome_scoring` and `outcome_conceding`.

- Binary goals:
  - use neither `--use_xg`, `--use_xt`, nor `--use_goal_distance`
  - this is available through the low-level `train.py` entrypoint
- xG:
  - pass `--use_xg`
  - `--return_type` controls which xG target family is used
- xT:
  - pass `--use_xt`
  - `--return_type` is ignored for xT semantics
- goal_distance:
  - pass `--use_goal_distance`
  - `--return_type` is ignored for goal-distance semantics

`--use_xg`, `--use_xt`, and `--use_goal_distance` are mutually exclusive.

### `--return_type` is xG-only

`--return_type` only changes how xG-based soft targets are constructed:

- `disc_<gamma>` uses discounted xG returns
- `next_<N>` uses non-discounted xG returns over the next `N` actions

Example:

```powershell
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123456_abcdef12 --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123500_bcdef123 --model gat --use_xg --return_type next_10 ...
```

For xT, the target is fixed by the xT module:

- event-level `xT` is the zone value at `start_x,start_y`
- `scores_xT` / `concedes_xT` are the maximum future teammate/opponent xT values over the next 5 eligible `pass`/`cross`/`shot` actions

For goal_distance, the target is fixed by the goal-distance module:

- event-level `goal_distance` is the inverted normalized distance-to-goal score at `start_x,start_y`, scaled to `[0, 100]`
- `scores_goal_distance` / `concedes_goal_distance` are the maximum future teammate/opponent goal-distance values over the next 5 eligible `pass`/`cross`/`shot` actions

So `--return_type` does not change xT or goal-distance behavior.

### Where to switch targets

Use `scripts/train_relevant_models.py` when you want the retained default setup:

```powershell
python scripts/train_relevant_models.py
python scripts/train_relevant_models.py --use_xt
python scripts/train_relevant_models.py --use_goal_distance
```

Use `train.py` directly when you need explicit low-level control, especially for binary-goal outcome training:

```powershell
python train.py --task outcome_scoring --model gat ...
python train.py --task outcome_scoring --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --model gat --use_xt ...
python train.py --task outcome_scoring --model gat --use_goal_distance ...
python train.py --task outcome_scoring --model gat --feature_run_id <feature_run_id> ...
```

The existing low-level feature toggles on `train.py` are:

- `--xy_only`
- `--possessor_aware`
- `--keeper_aware`
- `--ball_z_aware`
- `--poss_vel_aware`
- `--extend_features`

These same controls are now exposed in the wrappers as hyphenated flags on `scripts/train_relevant_models.py` and `scripts/main.py`. The wrappers keep the shared default profile described above, while `train.py` stays the low-level source of truth. Evaluation and inference wrappers do not repeat these flags because the trained checkpoint already stores them in `args.json`, and inference replays them from checkpoint metadata.

If you need to preserve the old numeric naming convention for a one-off run, `train.py` still accepts `--trial <n>` and writes `saved/<task>/<nn>/` for backward compatibility.

Note: the low-level entrypoints do not use the same spelling here:

- `train.py` uses `--feature_run_id`
- `test.py` uses `--feature-run-id`
- the wrapper scripts use `--feature-run-id`

If you generate or rebuild xT or goal-distance artifacts, rerun `scripts/generate_relevant_features.py` before any `--use_xt` or `--use_goal_distance` training run so the feature label tensors are refreshed.

## Notes

- `scripts/main.py` is the scoped pipeline runner for this repository.
- root-level `main.py` still reflects the full upstream defensive-score pipeline and is not part of this scoped reproduction.
- `datatools/tabular_feature.py` and the UxG/defensive-score path were left largely upstream because they are out of scope here.
- The core GNN training and evaluation flow stays close to upstream DEFCON; most project-specific changes are in preprocessing, target construction, and extra-data adapters.
- This repository is intended to track code, docs, and vendor snapshots only. Raw data, processed data, intermediate features, model outputs, and local environments are intentionally excluded from git.

## CLI Reference

This appendix covers every current `scripts/*.py` CLI entrypoint, including `scripts/main.py`. The legacy repo-root `main.py` is intentionally not included here because it is part of the upstream defensive-score path, not the scoped workflow described above.

### `scripts/main.py`

- `--use_xt`: use xT-backed outcome models instead of xG-backed outcome models. Default: off.
- `--use_goal_distance`: use goal-distance-backed outcome models instead of xG-backed outcome models. Default: off.
- `--use-original-intended-receiver`: use the original intended-receiver labels. Default: off.
- `--use-intended-receiver-model`: use the learned intended-receiver model workflow. Default: off.
- `--skip-preprocess`: skip `scripts/preprocess_sportec.py`. Default: off.
- `--skip-xt`: skip `scripts/generate_xt.py`. Default: off.
- `--skip-goal-distance`: skip `scripts/generate_goal_distance.py`. Default: off.
- `--skip-features`: skip `scripts/generate_relevant_features.py`. Default: off.
- `--skip-train`: skip `scripts/train_relevant_models.py`. Default: off.
- `--skip-evaluate`: skip `scripts/evaluate_relevant_models.py`. Default: off.
- `--skip-run-relevant`: skip `scripts/run_relevant_models.py`. Default: off.
- `--skip-hawkeye`: skip `scripts/run_hawkeye.py`. Default: off.
- `--skip-skillcorner`: skip `scripts/run_skillcorner.py`. Default: off.
- `--add_v_edge_features`: add the optional velocity-angle edge features during feature generation. Default: off.
- `--xy-only` / `--no-xy-only`: override the training wrapper's xy-only setting for this pipeline run. Default: no override, so `scripts/train_relevant_models.py` keeps `xy_only` off.
- `--possessor-aware` / `--no-possessor-aware`: override the training wrapper's possessor-awareness setting for this pipeline run. Default: no override, so possessor-aware stays on.
- `--keeper-aware` / `--no-keeper-aware`: override the training wrapper's keeper-awareness setting for this pipeline run. Default: no override, so keeper-aware stays on.
- `--ball-z-aware` / `--no-ball-z-aware`: override the training wrapper's ball-height setting for this pipeline run. Default: no override, so ball-height stays on.
- `--poss-vel-aware` / `--no-poss-vel-aware`: override the training wrapper's possessor-velocity setting for this pipeline run. Default: no override, so possessor-velocity awareness stays on.
- `--extend-features` / `--no-extend-features`: override the training wrapper's extended handcrafted node-feature setting for this pipeline run. Default: no override, so extended features stay off.
- `--overwrite`: allow supported preprocessing and xT outputs to be rebuilt. Default: off.
- `--relevant-split {train,test,all}`: split passed through to `scripts/run_relevant_models.py`. Default: `test`.
- `--device <device>`: device passed to evaluation and inference stages. Default: `cuda:0`.
- `--dry-run`: print the resolved commands without executing them. Default: off.

### `scripts/preprocess_sportec.py`

- `--match-id <id>`: process only one or more specific Sportec match ids. Default: all discovered matches.
- `--limit <N>`: process only the first `N` discovered matches. Default: no limit.
- `--overwrite`: rebuild existing outputs. Default: off.
- `--skip-sync`: stop before event-tracking synchronization. Default: off.
- `--sync-source {sportec_kpi,elastic}`: synchronization source for canonical event outputs. Default: `sportec_kpi`.

### `scripts/generate_xt.py`

- `--match-id <id>`: export xT sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--overwrite`: overwrite existing xT outputs. Default: off.

### `scripts/generate_goal_distance.py`

- `--match-id <id>`: export goal-distance sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--overwrite`: overwrite existing goal-distance outputs. Default: off.

### `scripts/generate_relevant_features.py`

- `--use-original-intended-receiver`: build features/labels with the original intended-receiver labels. Default: off.
- `--use-intended-receiver-model`: build features/labels for the learned intended-receiver workflow. Default: off.
- `--intended-receiver-model-id <model_id>`: success-intent checkpoint used by the learned intended-receiver workflow. Default: `success_intent/00`.
- `--add_v_edge_features`: append cosine/sine velocity-angle edge features. Default: off.
- `--run-id <feature_run_id>`: pin the feature run id instead of auto-generating one. Default: auto-generate a new feature run id.

### `scripts/train_relevant_models.py`

- `--use_xt`: train the outcome models against xT targets instead of xG targets. Default: off.
- `--use_goal_distance`: train the outcome models against goal-distance targets instead of xG targets. Default: off.
- `--feature-run-id <feature_run_id>`: pin the feature run used for training. Default: latest successful feature run.
- `--use-original-intended-receiver`: train with original intended-receiver labels. Default: off.
- `--use-intended-receiver-model`: train in learned intended-receiver mode. Default: off.
- `--success-intent-only`: train only the `success_intent` model. Default: off.
- `--xy-only` / `--no-xy-only`: override the wrapper default for `xy_only`. Default wrapper behavior: `xy_only` off.
- `--possessor-aware` / `--no-possessor-aware`: override the wrapper default for possessor-awareness features. Default wrapper behavior: on.
- `--keeper-aware` / `--no-keeper-aware`: override the wrapper default for keeper-awareness features. Default wrapper behavior: on.
- `--ball-z-aware` / `--no-ball-z-aware`: override the wrapper default for ball-height features. Default wrapper behavior: on.
- `--poss-vel-aware` / `--no-poss-vel-aware`: override the wrapper default for possessor-velocity relation features. Default wrapper behavior: on.
- `--extend-features` / `--no-extend-features`: override the wrapper default for extended handcrafted node features. Default wrapper behavior: off.
- `--outcome-scoring-trial <n>`: override the auto-generated run id for `outcome_scoring` with a legacy numeric id. Default: none.
- `--outcome-conceding-trial <n>`: override the auto-generated run id for `outcome_conceding` with a legacy numeric id. Default: none.
- `--bundle-id <bundle_id>`: pin the training bundle manifest id. Default: auto-generate a new bundle id.

### `scripts/evaluate_relevant_models.py`

- `--use_xt`: resolve xT-compatible outcome checkpoints by default. Default: off.
- `--use_goal_distance`: resolve goal-distance-compatible outcome checkpoints by default. Default: off.
- `--feature-run-id <feature_run_id>`: pin the feature run used for evaluation. Default: latest successful feature run.
- `--use-original-intended-receiver`: resolve checkpoints in original intended-receiver mode. Default: off.
- `--use-intended-receiver-model`: resolve checkpoints in learned intended-receiver mode. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--success-intent-model-id <model_id>`: explicit `success_intent` checkpoint id. Default: auto-resolve the latest compatible `success_intent` checkpoint.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--device <device>`: device passed to `test.py`. Default: `cuda:0`.

### `scripts/run_relevant_models.py`

- `--split {train,test,all}`: choose which Sportec split to export. Default: `test`.
- `--match-id <id>`: restrict export to one or more specific matches. Default: all matches in the selected split.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--use_xt`: resolve xT-compatible outcome checkpoints by default. Default: off.
- `--use_goal_distance`: resolve goal-distance-compatible outcome checkpoints by default. Default: off.
- `--feature-run-id <feature_run_id>`: pin the Sportec feature run used for inference. Default: latest successful feature run.
- `--run-id <component_run_id>`: pin the created component export run id. Default: auto-generate a new component run id.
- `--use-original-intended-receiver`: resolve checkpoints in original intended-receiver mode. Default: off.
- `--use-intended-receiver-model`: resolve checkpoints in learned intended-receiver mode. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--output-dir <path>`: parent directory for the created component run folder. Default: `data/ajax/component_runs`.

### `scripts/run_hawkeye.py`

- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--situation-id <id>`: restrict inference to one or more specific Hawkeye situation ids. Default: all valid situations.
- `--limit <N>`: process only the first `N` selected situations. Default: no limit.
- `--freeze-ballreceipt`: freeze possessor and ball state after `BallReceipt`. Default: on.
- `--no-freeze-ballreceipt`: disable BallReceipt freezing. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--use_xt`: resolve xT-compatible outcome checkpoints by default. Default: off.
- `--use_goal_distance`: resolve goal-distance-compatible outcome checkpoints by default. Default: off.
- `--use-original-intended-receiver`: resolve checkpoints in original intended-receiver mode. Default: off.
- `--use-intended-receiver-model`: resolve checkpoints in learned intended-receiver mode. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--run-id <component_run_id>`: pin the created HawkEye component run id. Default: auto-generate a new HawkEye component run id.
- `--output-dir <path>`: parent directory for the created Hawkeye run folder. Default: `data/ajax/component_runs/hawkeye`.

### `scripts/run_skillcorner.py`

- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--match-id <id>`: restrict inference to one or more specific SkillCorner match ids. Default: all discoverable valid matches.
- `--limit <N>`: process only the first `N` selected matches. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--use_xt`: resolve xT-compatible outcome checkpoints by default. Default: off.
- `--use_goal_distance`: resolve goal-distance-compatible outcome checkpoints by default. Default: off.
- `--use-original-intended-receiver`: resolve checkpoints in original intended-receiver mode. Default: off.
- `--use-intended-receiver-model`: resolve checkpoints in learned intended-receiver mode. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--run-id <component_run_id>`: pin the created SkillCorner component run id. Default: auto-generate a new SkillCorner component run id.
- `--output-dir <path>`: parent directory for the created SkillCorner run folder. Default: `data/ajax/component_runs/skillcorner`.

### `scripts/visualize_action_components.py`

- `--match-id <id>`: Sportec match id to visualize. Default: required.
- `--action-id <action_id>`: CSV `action_id` from `data/ajax/event_synced/<match_id>.csv`. Default: one of the identifier options is required.
- `--row-index <index>`: legacy modeled-action row index. Default: off.
- `--original-event-id <id>`: raw Sportec event id lookup. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--feature-run-id <feature_run_id>`: feature run used for visualization-time inference. Default: latest successful feature run.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--use_xt`: switch the default outcome-checkpoint resolution to xT-compatible checkpoints. Default: off.
- `--use_goal_distance`: switch the default outcome-checkpoint resolution to goal-distance-compatible checkpoints. Default: off.
- `--use-original-intended-receiver`: switch the default legacy checkpoint family to original intended-receiver mode. Default: off.
- `--use-intended-receiver-model`: switch the default legacy checkpoint family to learned intended-receiver mode. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: auto-resolve latest compatible checkpoint.
- `--output-dir <path>`: visualization root directory. Default: `data/ajax/visualizations`.

### `scripts/visualize_hawkeye.py`

- `--situation-id <id>`: restrict visualization to one or more Hawkeye situation ids from the selected component run. Default: all situations in the selected component run.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--component-run-id <component_run_id>`: versioned Hawkeye component run to visualize. Default: latest successful Hawkeye component run.
- `--component-dir <path>`: explicit Hawkeye component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
- `--output-dir <path>`: visualization root directory. Default: `data/ajax/visualizations/hawkeye`.

### `scripts/run_and_visualize_hawkeye.py`

- `--situation-id <id>`: Hawkeye situation id to visualize. Default: one of `--situation-id` or `--action-id` is required at runtime.
- `--action-id <id>`: alias for `--situation-id`. Default: off.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--freeze-ballreceipt`: freeze possessor and ball state after `BallReceipt`. Default: on.
- `--no-freeze-ballreceipt`: disable BallReceipt freezing. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id. Default: `action_intent/00`.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id. Default: `pass_intent/20`.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id. Default: `pass_success/20`.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id. Default: `outcome_scoring/20`.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id. Default: `outcome_conceding/20`.
- `--output-dir <path>`: visualization root directory. Default: `data/ajax/visualizations/hawkeye`.

### `scripts/visualize_skillcorner.py`

- `--match-id <id>`: SkillCorner match id to visualize. Default: required.
- `--index <player_possession_index>`: SkillCorner `player_possession` event index. Default: required.
- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--component-run-id <component_run_id>`: versioned SkillCorner component run to visualize. Default: latest successful SkillCorner component run.
- `--component-dir <path>`: explicit component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--output-dir <path>`: visualization root directory. Default: `data/ajax/visualizations/skillcorner`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
