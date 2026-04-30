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
  - [2. Generate soft-target artifacts when xT, goal_distance, or EPV targets are needed](#2-generate-soft-target-artifacts-when-xt-goal_distance-or-epv-targets-are-needed)
  - [3. Generate graph features and labels](#3-generate-graph-features-and-labels)
  - [4. Train the retained models](#4-train-the-retained-models)
  - [5. Evaluate the retained models on the test set](#5-evaluate-the-retained-models-on-the-test-set)
  - [6. Export per-match component predictions](#6-export-per-match-component-predictions)
  - [7. Visualize one action at a time](#7-visualize-one-action-at-a-time)
  - [8. Run frame-level inference on HawkEye data](#8-run-frame-level-inference-on-hawkeye-data)
  - [9. Run benchmark inference on local benchmark data](#9-run-benchmark-inference-on-local-benchmark-data)
  - [10. Run frame-level inference on SkillCorner data](#10-run-frame-level-inference-on-skillcorner-data)
- [Outcome Target Selection](#outcome-target-selection)
  - [`--return_type` Applies To All Outcome Target Families](#return_type-applies-to-all-outcome-target-families)
  - [Where to switch targets](#where-to-switch-targets)
- [Intended-Receiver Workflow](#intended-receiver-workflow)
- [EPV Workflow](#epv-workflow)
- [Notes](#notes)
- [CLI Reference](#cli-reference)
- [Script I/O Reference](#script-io-reference)

## Changes From Upstream DEFCON

The main adaptations in this repository are:

- Sportec-specific preprocessing and synchronization to turn Sportec XML tracking/event data into DEFCON-compatible inputs
- xT, goal-distance, and EPV as alternative soft target variables for `outcome_scoring` and `outcome_conceding`
- temporal `t0 / t-12 / t-25` augmentation for `action_intent` and `pass_intent` training data
- narrowed modeled pass family: only `pass` and `cross` are treated as pass-like actions, so set pieces are excluded from the modeled pass category
- a HawkEye inference/visualization pipeline for frame-level external data
- a local benchmark inference/visualization pipeline for paired one-frame game-state comparisons
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

`scripts/preprocess_sportec.py` now handles the observed raw-format differences between the two seasons and still writes the same canonical processed outputs under `data`.

## Produced Data Layout

After preprocessing, the project writes DEFCON-style files under `data`:

- `tracking/*.parquet`
- `tracking_processed/*.parquet`
- `event/event.parquet`
- `event_synced/*.csv`
- `xT/xT.csv`
- `goal_distance/goal_distance.csv`
- `epv/epv.csv`
- `xT/xT_grid.csv`
- `xT/fit_metadata.json`
- `xT/matches/*.csv`
- `goal_distance/matches/*.csv`
- `epv/metadata.json`
- `epv/matches/*.csv`
- `lineup/line_up.parquet`
- `features/runs/<feature_run_id>/...`
- `features/runs/latest.json`
- `component_runs/<component_run_id>/...`
- `component_runs/latest.json`
- `component_runs/hawkeye/<component_run_id>/...`
- `component_runs/benchmark/<component_run_id>/...`
- `component_runs/skillcorner/<component_run_id>/...`
- `splits/match_splits.json`

`event_synced` is stored as CSV because the upstream DEFCON code reads CSV, even though the README mentions Parquet.

The main output directories are:

- `data/event_synced` for canonical synced event tables used by the DEFCON-style pipeline
- `data/xT` for xT exports and xT fitting artifacts
- `data/goal_distance` for goal-distance exports
- `data/epv` for model-derived EPV exports
- `data/features/runs/<feature_run_id>` for versioned graph tensors and label tensors used by training/evaluation
- `data/component_runs/<component_run_id>` for versioned per-match component prediction exports from `scripts/run_relevant_models.py`
- `data/component_runs/hawkeye/<component_run_id>` for versioned HawkEye exports
- `data/component_runs/benchmark/<component_run_id>` for versioned benchmark exports
- `data/component_runs/skillcorner/<component_run_id>` for versioned SkillCorner exports
- `saved/<task>/<model_run_id>` for trained checkpoint runs
- `saved/bundles/<bundle_id>` for machine-readable training bundle manifests

## Run-Id Workflow

There are now three different versioned artifact families:

- `feature_run_id` for generated Sportec graph/label artifacts under `data/features/runs/...`
- `model_id = <task>/<model_run_id>` for trained checkpoints under `saved/<task>/...`
- `component_run_id` for exported predictions under `data/component_runs/...`

Feature and component runs use explicit `latest.json` pointers:

- `data/features/runs/latest.json`
- `data/component_runs/latest.json`
- `data/component_runs/hawkeye/latest.json`
- `data/component_runs/benchmark/latest.json`
- `data/component_runs/skillcorner/latest.json`

Checkpoint runs are referenced explicitly as `model_id = <task>/<model_run_id>`. The retained wrappers now prefer one of these two explicit handoff styles:

- `feature_run_id` for training
- `bundle_id` or explicit per-task model ids for evaluation and inference

`data/features/runs/latest.json` is still written for convenience and manual exploration, but the retained wrappers no longer depend on wrapper-level latest-compatible checkpoint resolution.

The feature run root mirrors the old flat layout, but inside one dedicated run folder:

- `data/features/runs/<feature_run_id>/action_graphs/*.pt`
- `data/features/runs/<feature_run_id>/post_action_graphs/*.pt`
- `data/features/runs/<feature_run_id>/action_graphs_intent_train/*.pt`
- `data/features/runs/<feature_run_id>/action_graphs_success_intent/*.pt`
- `data/features/runs/<feature_run_id>/action_labels_<return_type>*.pt`
- `data/features/runs/<feature_run_id>/action_labels_intent_train_<return_type>*.pt`
- `data/features/runs/<feature_run_id>/resolved_actions*.parquet`
- `data/features/runs/<feature_run_id>/metadata.json`

The component run root contains one folder per processed match:

- `data/component_runs/<component_run_id>/<match_id>/action_intent.parquet`
- `data/component_runs/<component_run_id>/<match_id>/pass_intent.parquet`
- optionally `data/component_runs/<component_run_id>/<match_id>/success_intent.parquet` when a `success_intent` checkpoint is provided
- `data/component_runs/<component_run_id>/<match_id>/pass_success.parquet`
- `data/component_runs/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
- `data/component_runs/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
- `data/component_runs/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
- `data/component_runs/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
- `data/component_runs/<component_run_id>/metadata.json`

The external-data adapters follow the same pattern:

- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
- `data/component_runs/hawkeye/<component_run_id>/metadata.json`
- `data/component_runs/benchmark/<component_run_id>/benchmark_data.parquet`
- `data/component_runs/benchmark/<component_run_id>/benchmark_data.csv`
- `data/component_runs/benchmark/<component_run_id>/metadata.json`
- `data/component_runs/skillcorner/<component_run_id>/<match_id>/*.parquet`
- `data/component_runs/skillcorner/<component_run_id>/metadata.json`

Checkpoint runs also write metadata:

- `saved/<task>/<model_run_id>/args.json`
- `saved/<task>/<model_run_id>/metadata.json`
- `saved/<task>/<model_run_id>/best_weights.pt`
- `saved/bundles/<bundle_id>/metadata.json`

The run metadata records the relevant toggles used for that invocation. For component runs this includes the per-model feature signatures and graph schema, so settings such as `poss_vel_aware`, `ball_z_aware`, `extend_features`, `v_edge_feature_mode`, `edge_in_dim`, and `add_v_edge_features` are visible in the saved metadata.

## Current Artifact Contract

The current pipeline now follows an explicit-artifact contract:

- `scripts/generate_relevant_features.py` can include multiple `--return_type` values in one feature run and always writes the full velocity-angle edge-feature schema.
- Each generated feature run always contains `original` and `angle_only` intended-receiver variants, and it additionally contains `model` only when `--intended-receiver-model-id <success_intent/model_run_id>` is supplied.
- `scripts/train_relevant_models.py` now requires an explicit `--feature-run-id` and supports per-model toggles. `--target-family` and `--return_type` are required only when an outcome model is enabled, and `--intended-receiver-mode` is required only when a mode-dependent model is enabled.
- Training decides whether to use the stored velocity-angle edge features via `--v-edge-features`, `--v-edge-features-no-poss`, or `--no-v-edge-features`; feature generation no longer has an `--add_v_edge_features` toggle.
- `scripts/evaluate_relevant_models.py`, `scripts/run_relevant_models.py`, `scripts/run_hawkeye.py`, `scripts/run_benchmark.py`, `scripts/run_skillcorner.py`, and `scripts/visualize_action_components.py` now prefer `--bundle-id` or explicit model ids. Sportec runtime scripts can combine compatible checkpoints from different feature runs; they automatically choose the newest compatible source feature run unless `--feature-run-id` is supplied.
- `scripts/main.py` now threads explicit `feature_run_id` and `bundle_id` values between stages. It does not auto-bootstrap learned intended-receiver mode anymore; for `--intended-receiver-mode model`, provide or first create a `success_intent` checkpoint and pass it as `--intended-receiver-model-id`.


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

Use `scripts/main.py` when you want the scoped end-to-end runner without any visualization steps.

```powershell
python scripts/main.py --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only
python scripts/main.py --target-family xt --return_type next_5 --intended-receiver-mode angle_only
python scripts/main.py --target-family goal_distance --return_type next_3 --intended-receiver-mode original
python scripts/main.py --target-family epv --return_type next_5 --intended-receiver-mode angle_only --bundle-id <source_bundle_id>
```

The runner executes these stages in order:

1. `scripts/preprocess_sportec.py`
2. `scripts/generate_xt.py` only when `--target-family xt` is selected, `scripts/generate_goal_distance.py` only when `--target-family goal_distance` is selected, or `scripts/generate_epv.py` only when `--target-family epv` is selected
3. `scripts/generate_relevant_features.py`
4. `scripts/train_relevant_models.py`
5. `scripts/evaluate_relevant_models.py`
6. `scripts/run_relevant_models.py`
7. `scripts/run_hawkeye.py`
8. `scripts/run_benchmark.py`
9. `scripts/run_skillcorner.py`

Useful options:

- `--skip-preprocess`, `--skip-xt`, `--skip-goal-distance`, `--skip-epv`, `--skip-features`, `--skip-train`, `--skip-evaluate`, `--skip-run-relevant`, `--skip-hawkeye`, `--skip-benchmark`, `--skip-skillcorner`
- `--feature-run-id <feature_run_id>` to pin or reuse a feature run id
- `--bundle-id <bundle_id>` to pin or reuse a model bundle id; required when `scripts/main.py` generates EPV artifacts
- `--intended-receiver-model-id <success_intent/model_run_id>` when feature generation should also include the `model` intended-receiver variant
- `--v-edge-features` / `--v-edge-features-no-poss` / `--no-v-edge-features` to control whether training uses all stored velocity-angle edge features, masks possessor-incident velocity edge columns, or drops velocity edge columns entirely; default: on
- `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, and `--extend-features` / `--no-extend-features` to override the training feature profile passed into `scripts/train_relevant_models.py`
- `--benchmark-input-dir <path>` to point `scripts/run_benchmark.py` at a local benchmark checkout
- `--overwrite` to rebuild supported preprocessing and target-artifact outputs
- `--relevant-split train|test|all` to control `scripts/run_relevant_models.py`
- `--device cuda:0|cpu` for evaluation and inference scripts
- `--dry-run` to print the resolved commands without running them

When `scripts/main.py` runs feature generation and training itself, it creates fresh `feature_run_id` and `bundle_id` values and passes them forward automatically. If you skip training but keep downstream evaluation or inference stages enabled, you must supply `--bundle-id`. If you skip feature generation but keep training enabled, you must supply `--feature-run-id`.

## End-to-End Workflow

The sections below describe each stage individually. `scripts/main.py` wraps steps 1-6, 8, 9, and 10, and intentionally excludes the visualization scripts.

### 1. Preprocess Sportec data

```powershell
python scripts/preprocess_sportec.py
```

Inputs:

- raw Sportec season folders under `Bundesliga_season_23_24/...` and `Bundesliga_season_24_25/...`
- for `24_25`, the split metadata layout under `match_information/master`, `match_information/starting_players`, and `KPI_Merged`

Outputs:

- `data/tracking/*.parquet`
- `data/tracking_processed/*.parquet`
- `data/event/event.parquet`
- `data/event_synced/*.csv`
- `data/lineup/line_up.parquet`
- `data/splits/match_splits.json`

Useful options:

- `--match-id DFL-MAT-...` to process only selected matches
- `--limit N` to smoke-test on the first `N` matches
- `--overwrite` to rebuild existing outputs
- `--skip-sync` to stop before ELASTIC synchronization

Subset-safe aggregate behavior:

- when you run a subset with `--match-id` or `--limit`, the per-match files for the selected matches are still processed as usual
- `data/lineup/line_up.parquet` and `data/splits/match_splits.json` are rebuilt from the currently processed match universe, so subset runs no longer shrink those global files to only the selected matches
- `data/event/event.parquet` is merged incrementally in subset mode, so reprocessed matches are refreshed without dropping untouched matches
- full runs without subset filters still rebuild the aggregate files from the full successful run

This step does the Sportec-specific work that DEFCON does not provide:

- Sportec XML discovery
- lineup reconstruction
- derivation of `mins_played`, `start_time`, and `end_time` from substitutions, red cards, and final whistles
- Kloppy event/tracking conversion
- SPADL action generation with `socceraction`
- ELASTIC synchronization
- split-manifest creation

### 2. Generate soft-target artifacts when xT, goal_distance, or EPV targets are needed

```powershell
python scripts/generate_xt.py
python scripts/generate_goal_distance.py
python scripts/generate_epv.py --bundle-id <source_bundle_id>
```

Inputs:

- `data/event_synced/*.csv`
- `data/splits/match_splits.json`
- for EPV, a model bundle or explicit source model ids for `pass_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding`; mixed-source checkpoints use the newest compatible runtime feature run unless `--feature-run-id` is supplied

Outputs:

- for xT:
  - `data/xT/xT.csv`
  - `data/xT/xT_grid.csv`
  - `data/xT/fit_metadata.json`
  - `data/xT/matches/*.csv`
- for goal-distance:
  - `data/goal_distance/goal_distance.csv`
  - `data/goal_distance/metadata.json`
  - `data/goal_distance/matches/*.csv`
- for EPV:
  - `data/epv/epv.csv`
  - `data/epv/metadata.json`
  - `data/epv/matches/*.csv`

Useful options:

- `--match-id DFL-MAT-...` to export only selected matches
- `--limit N` to restrict export generation to the first `N` available synced matches
- `--overwrite` to rebuild existing outputs

This is a separate post-preprocessing step. It reads canonical synced event CSVs, fits or exports the sidecar targets, and writes per-match artifacts used later during feature generation.

EPV computes `sum(pass_intent * pass_score)` across players, where `pass_score` combines pass success with the scoring and conceding outcome models. Shot actions use `max(epv, xG)`, matching the xT shot-value convention.

### 3. Generate graph features and labels

```powershell
python scripts/generate_relevant_features.py --return_type disc_0.9
python scripts/generate_relevant_features.py --return_type disc_0.9 --return_type next_5 --return_type next_3
python scripts/generate_relevant_features.py --return_type next_5 --return_type next_5_skip1 --return_type disc_0.9_skip1
python scripts/generate_relevant_features.py --run-id feature_20260414T123456_abcdef12 --return_type disc_0.9 --intended-receiver-model-id success_intent/<model_run_id>
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --intended-receiver-model-id success_intent/<model_run_id>
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --intended-receiver-model-id success_intent/<new_model_run_id> --replace-intended-receiver-model
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --refresh-target-family epv
```

Inputs:

- `data/tracking_processed/*.parquet`
- `data/event_synced/*.csv`
- `data/lineup/line_up.parquet`
- `data/splits/match_splits.json`
- optional sidecars from `data/xT/matches/*.csv`, `data/goal_distance/matches/*.csv`, and `data/epv/matches/*.csv`
- optional learned intended-receiver checkpoint referenced by `--intended-receiver-model-id`

Each invocation creates a new feature-artifact run under `data/features/runs/<feature_run_id>/` and updates `data/features/runs/latest.json` after completion. `--extend-feature-run-id` creates a new derived run by copying a completed base run and generating only newly requested return types, refreshed target labels, and/or the model intended-receiver variant. If the base run already contains `model` artifacts, a different `--intended-receiver-model-id` is rejected unless `--replace-intended-receiver-model` is supplied; replacement still creates a new derived run and regenerates only copied model-mode artifacts.

Behavior:

- every feature run always includes the `original` and `angle_only` intended-receiver variants
- `model` is included only when `--intended-receiver-model-id` is supplied
- graphs are written once per run and always include the full velocity-angle edge-feature schema
- labels are written for every requested `--return_type`

Useful options:

- `--run-id <feature_run_id>` to pin the created run id instead of auto-generating one
- `--extend-feature-run-id <existing_feature_run_id>` to create a new derived run from an existing completed run without rebuilding shared graph tensors
- `--refresh-target-family <xt|goal_distance|epv>` with `--extend-feature-run-id` to rebuild copied label tensors from current target sidecars without rebuilding graph tensors
- repeat `--return_type <disc_gamma|disc_gamma_skip1|next_N|next_N_skip1|in_N>` to include multiple resolved return semantics in one feature run
- `--intended-receiver-model-id <model_id>` to additionally include the `model` intended-receiver variant
- `--replace-intended-receiver-model` with `--extend-feature-run-id` and `--intended-receiver-model-id` to regenerate copied model-mode artifacts with a different `success_intent` checkpoint in the derived run

Terminal output progression:

A normal (full) feature run executes 5 sequential steps. Before each subprocess starts, it prints a top-level progress line such as `Feature generation step 1/5: train split with post_action + augment_blocks`, then prints these messages for each match (e.g., `[1/306] ... [306/306]`):

1. **train split with post_action + augment_blocks:** prints `"Successfully saved {N} augmented events for mode={mode}."` then `"Successfully saved for {N} events."`
2. **test split with post_action:** prints `"Successfully saved for {N} events."`
3. **train split with intent_train_augmented:** prints `"Successfully saved {N} augmented intent samples."`
4. **train split with success_intent:** prints `"Successfully saved success-intent graphs for {N} events."`
5. **test split with success_intent:** prints `"Successfully saved success-intent graphs for {N} events."`

An `--extend-feature-run-id` run with new `--return_type` values only executes 3 steps, printing `Feature generation step X/3: ...` before each step:

1. **train split (labels-only):** prints `"Successfully saved labels-only action labels."`
2. **test split (labels-only):** prints `"Successfully saved labels-only action labels."`
3. **train split with intent_train_augmented (labels-only):** prints `"Successfully saved labels-only intent-training labels."`

An `--extend-feature-run-id` run with both new `--return_type` values and `--intended-receiver-model-id` executes 6 steps, printing `Feature generation step X/6: ...` before each step:

1. **train split (labels-only):** prints `"Successfully saved labels-only action labels."`
2. **test split (labels-only):** prints `"Successfully saved labels-only action labels."`
3. **train split with intent_train_augmented (labels-only):** prints `"Successfully saved labels-only intent-training labels."`
4. **train split with model mode (labels-only):** prints `"Successfully saved labels-only action labels."`
5. **test split with model mode (labels-only):** prints `"Successfully saved labels-only action labels."`
6. **train split with model mode intent_train_augmented (labels-only):** prints `"Successfully saved labels-only intent-training labels."`

An `--extend-feature-run-id` run with `--refresh-target-family` executes the same labels-only shape for the copied run's existing return types and intended-receiver modes, but passes `--overwrite-labels` so copied label tensors are rebuilt from the current xT, goal-distance, and EPV sidecars.

This writes, inside the run root:

- `action_graphs/*.pt`
- `post_action_graphs/*.pt`
- `action_graphs_intent_train/*.pt`
- `action_graphs_success_intent/*.pt`
- `action_labels_<return_type>*.pt` for each requested `return_type` and intended-receiver mode
- `success_intent_labels/*.pt`
- `action_labels_intent_train_<return_type>*.pt` for each requested `return_type` and intended-receiver mode
- `resolved_actions*.parquet` for each intended-receiver mode
- `metadata.json`

Every generated action-label tensor also carries canonical outcome diagnostic columns:

- `scores_goal_next10`
- `concedes_goal_next10`

These are binary goal labels over `next_10`, generated independently of the selected `--return_type`.

### 4. Train the retained models

```powershell
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --success-intent-only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xt --return_type in_3 --intended-receiver-mode original --no-action-intent --no-pass-intent --no-success-intent --no-pass-success --no-failure-receiver --bundle-id model_bundle_20260414T123456_abcdef12
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type next_3 --intended-receiver-mode model --no-success-intent --no-v-edge-features
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family epv --return_type next_5 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --return_type disc_0.9 --intended-receiver-mode model --no-action-intent --no-pass-intent --no-success-intent --no-outcome-scoring --no-outcome-conceding --pass-intent-model-id pass_intent/<model_run_id>
```

Inputs:

- `data/features/runs/<feature_run_id>/...`

Outputs:

- `saved/pass_intent/<model_run_id>/...`
- `saved/action_intent/<model_run_id>/...`
- `saved/pass_success/<model_run_id>/...`
- `saved/outcome_scoring/<model_run_id>/...`
- `saved/outcome_conceding/<model_run_id>/...`
- `saved/success_intent/<model_run_id>/...` when `success_intent` is enabled
- `saved/failure_receiver/<model_run_id>/...`
- `saved/bundles/<bundle_id>/metadata.json`

Behavior:

- the default wrapper run enables `action_intent`, `pass_intent`, `success_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding`; `failure_receiver` is off by default and can be enabled explicitly
- `--action-intent` / `--no-action-intent`, `--pass-intent` / `--no-pass-intent`, `--success-intent` / `--no-success-intent`, `--pass-success` / `--no-pass-success`, `--outcome-scoring` / `--no-outcome-scoring`, `--outcome-conceding` / `--no-outcome-conceding`, and `--failure-receiver` / `--no-failure-receiver` let you rerun only the subset you need
- `--target-family` and `--return_type` are required only when `outcome_scoring` or `outcome_conceding` is enabled
- `--intended-receiver-mode` is required only when a mode-dependent model is enabled: `action_intent`, `pass_intent`, `pass_success`, `outcome_scoring`, `outcome_conceding`, or `failure_receiver`
- `--success-intent-only` trains `success_intent` from the observed synced `receiver_id` on successful pass actions only
- `--success-intent-only` is mode-independent, does not accept `--intended-receiver-mode`, and cannot be combined with the per-model toggles
- `pass_success` uses a `pass_intent` checkpoint as its IPW model; train `pass_intent` in the same wrapper run or use `--no-pass-intent --pass-intent-model-id pass_intent/<model_run_id>` to reuse a compatible existing checkpoint
- an external `--pass-intent-model-id` must match the selected feature run, intended-receiver mode, graph schema, velocity edge-feature mode, and feature flags; its return type and target family are ignored because `pass_intent` only supplies IPW propensities for `pass_success`
- outcome model loss uses the selected `--target-family` and `--return_type`; outcome F1/ROC AUC/Brier diagnostics use canonical `goal next_10` labels for comparability across target families and return types
- older feature runs without embedded `goal next_10` diagnostic columns can still be used if the selected run exposes `action_labels_next_10<intended_receiver_suffix>` or if you pass `--diagnostic-feature-run-id <feature_run_id>` pointing to compatible `next_10` labels
- reusing `--bundle-id` updates the existing bundle manifest by replacing only the retrained task ids and preserving untouched task ids
- training chooses whether to use the stored velocity-angle edge features via `--v-edge-features`, `--v-edge-features-no-poss`, or `--no-v-edge-features`; default: on
- wrapper batch-size defaults are `256` for `action_intent`, `pass_intent`, `success_intent`, and `failure_receiver`, and `512` for `pass_success`, `outcome_scoring`, and `outcome_conceding`; `--batch-size` overrides all defaults, and per-model `--<model>-batch-size` flags take highest precedence
- unless you override them explicitly, wrapper-trained models use the shared defaults `possessor_aware`, `keeper_aware`, `ball_z_aware`, and `poss_vel_aware` on, with `extend_features` and `xy_only` off

In the intended-receiver workflow, `success_intent` is the learned intended-receiver checkpoint. It is trained independently of the `original` / `angle_only` / `model` intended-receiver modes. `failure_receiver` is a separate auxiliary model used for failed-pass / opponent-receiver handling.

### 5. Evaluate the retained models on the test set

```powershell
python scripts/evaluate_relevant_models.py --bundle-id <bundle_id>
python scripts/evaluate_relevant_models.py --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
python scripts/evaluate_relevant_models.py --action-intent-model-id action_intent/<model_run_id> --pass-intent-model-id pass_intent/<model_run_id> --pass-success-model-id pass_success/<model_run_id> --outcome-scoring-model-id outcome_scoring/<model_run_id> --outcome-conceding-model-id outcome_conceding/<model_run_id>
```

Inputs:

- feature artifacts referenced by the selected checkpoint metadata
- checkpoint runs under `saved/<task>/<model_run_id>/...`

Outputs:

- no dedicated output files
- metrics are printed to stdout by `test.py`

`test.py` uses the target configuration, graph schema, and diagnostic-label metadata saved inside each checkpoint. Pass `--diagnostic-feature-run-id <feature_run_id>` when evaluating older checkpoints whose selected feature run lacks canonical `goal next_10` diagnostics. The wrapper now prefers `--bundle-id` for the main retained-model set. `success_intent` is optional and can be supplied explicitly if you want it evaluated too.

### 6. Export per-match component predictions

```powershell
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id>
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --run-id component_20260414T123456_abcdef12
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
python scripts/run_relevant_models.py --split test --action-intent-model-id action_intent/<model_run_id> --pass-intent-model-id pass_intent/<model_run_id> --pass-success-model-id pass_success/<model_run_id> --outcome-scoring-model-id outcome_scoring/<model_run_id> --outcome-conceding-model-id outcome_conceding/<model_run_id>
```

Inputs:

- feature artifacts from the selected runtime feature run; compatible mixed-source checkpoints are allowed, and the runtime feature run is chosen automatically from the selected source runs unless `--feature-run-id` is supplied
- checkpoint runs under `saved/<task>/<model_run_id>/...`
- `data/event_synced/<match_id>.csv` indirectly via the generated feature artifacts and resolved-action tables

Each invocation creates a new component run under `data/component_runs/<component_run_id>/` and updates `data/component_runs/latest.json`.
When a `success_intent` checkpoint is supplied explicitly or present in the selected bundle, each processed match also gets `success_intent.parquet`.

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--feature-run-id <feature_run_id>` to override the automatically selected runtime feature run
- `--run-id <component_run_id>` to pin the created component run id instead of auto-generating one
- `--match-id DFL-MAT-...` to restrict inference to one or more matches
- `--success-intent-model-id success_intent/<model_run_id>` to additionally export `success_intent.parquet` for each processed match

### 7. Visualize one action at a time

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
```

`--action-id` refers to the `action_id` column in `data/event_synced/<match_id>.csv`.

Inputs:

- feature artifacts from the selected runtime feature run; compatible mixed-source checkpoints are allowed, and the runtime feature run is chosen automatically from the selected source runs unless `--feature-run-id` is supplied
- checkpoint runs under `saved/<task>/<model_run_id>/...`
- `data/event_synced/<match_id>.csv`

Outputs:

- `data/visualizations/<match_id>/<action_id>/action_intent.png`
- `data/visualizations/<match_id>/<action_id>/pass_intent.png`
- `data/visualizations/<match_id>/<action_id>/pass_success.png`
- `data/visualizations/<match_id>/<action_id>/outcome_scoring_success.png`
- `data/visualizations/<match_id>/<action_id>/outcome_scoring_failure.png`
- `data/visualizations/<match_id>/<action_id>/outcome_conceding_success.png`
- `data/visualizations/<match_id>/<action_id>/outcome_conceding_failure.png`
- `data/visualizations/<match_id>/<action_id>/pass_score.png`
- optionally `intended_recipient.png` when a `success_intent` checkpoint is supplied

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--feature-run-id <feature_run_id>` to override the automatically selected runtime feature run
- `--show-trajectories` to render dashed recent player trajectories
- `--row-index <index>` to use the legacy internal modeled-action row index instead of the CSV `action_id`
- `--original-event-id <sportec_event_id>` to look up the action by the raw Sportec event id

### 8. Run frame-level inference on HawkEye data

```powershell
python scripts/run_hawkeye.py --bundle-id <bundle_id>
python scripts/run_hawkeye.py --bundle-id <bundle_id> --run-id hawkeye_component_20260414T123456_abcdef12
```

Inputs:

- `hawkeye_data/centroid_data_team.csv`
- `hawkeye_data/ball_data_selected.csv`
- checkpoint runs under `saved/<task>/<model_run_id>/...`

This writes versioned consolidated component outputs to:

- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
- `data/component_runs/hawkeye/<component_run_id>/metadata.json`

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--situation-id ...` to restrict inference to selected situations
- `--limit N` to smoke-test on the first `N` situations
- `--run-id <component_run_id>` to pin the created HawkEye export run id instead of auto-generating one
- `--no-freeze-ballreceipt` to disable the default BallReceipt freeze for the possessor and the ball

To visualize one HawkEye situation as MP4s:

```powershell
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id>
```

`scripts/visualize_hawkeye.py` reads the probabilities from `scripts/run_hawkeye.py` outputs and rebuilds only the raw HawkEye geometry for rendering. If you want the old direct-inference behavior, use:

```powershell
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id>
```

### 9. Run benchmark inference on local benchmark data

```powershell
python scripts/run_benchmark.py --bundle-id <bundle_id>
python scripts/run_benchmark.py --bundle-id <bundle_id> --modification 1
python scripts/run_benchmark.py --bundle-id <bundle_id> --run-id benchmark_component_20260420T123456_abcdef12
```

Inputs:

- `benchmark/modification_<n>/game_state_1.csv`
- `benchmark/modification_<n>/game_state_2.csv`
- `benchmark/modification_<n>/modification.csv`
- checkpoint runs under `saved/<task>/<model_run_id>/...`

This writes one consolidated benchmark export run to:

- `data/component_runs/benchmark/<component_run_id>/benchmark_data.parquet`
- `data/component_runs/benchmark/<component_run_id>/benchmark_data.csv`
- `data/component_runs/benchmark/<component_run_id>/metadata.json`

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--input-dir <path>` to point at a different local benchmark checkout
- `--modification ...` to restrict inference to selected benchmark modifications
- `--limit N` to smoke-test on the first `N` selected modifications
- `--run-id <component_run_id>` to pin the created benchmark export run id instead of auto-generating one

To visualize one benchmark state as PNGs:

```powershell
python scripts/visualize_benchmark.py --modification 1 --game-state 1
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id>
```

### 10. Run frame-level inference on SkillCorner data

```powershell
python scripts/run_skillcorner.py --bundle-id <bundle_id>
python scripts/run_skillcorner.py --bundle-id <bundle_id> --run-id skillcorner_component_20260414T123456_abcdef12
```

Inputs:

- `skillcorner_data/<match_id>_tracking.jsonl`
- `skillcorner_data/<match_id>_match.json`
- `skillcorner_data/<match_id>_dynamic_events.csv`
- checkpoint runs under `saved/<task>/<model_run_id>/...`

This writes versioned per-match parquet outputs under:

- `data/component_runs/skillcorner/<component_run_id>/<match_id>/`
- `data/component_runs/skillcorner/<component_run_id>/metadata.json`

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--match-id ...` to restrict inference to selected matches
- `--limit N` to smoke-test on the first `N` selected matches
- `--run-id <component_run_id>` to pin the created SkillCorner export run id instead of auto-generating one

To visualize one SkillCorner possession as MP4s:

```powershell
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index>
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id>
```

## Outcome Target Selection

Outcome target selection affects `outcome_scoring` and `outcome_conceding`.

- In the low-level `train.py`, select the outcome family with `--use_xg`, `--use_xt`, `--use_goal_distance`, or `--use_epv`, or omit all four for binary goals.
- In `scripts/train_relevant_models.py` and `scripts/main.py`, select the outcome family with `--target-family {goal,xg,xt,goal_distance,epv}`.
- `--return_type` controls the return semantics for all outcome families and for the shared action-label directories in the selected feature run.

Training loss and diagnostics intentionally use different targets:

- loss target: the selected `--target-family` and `--return_type`
- comparable event diagnostics: canonical binary `goal next_10` labels in `scores_goal_next10` / `concedes_goal_next10`

The logged metric names remain `f1`, `roc_auc`, and `brier`. For non-goal target families, treat them as goal-event proxy diagnostics rather than true probability-calibration metrics for the trained target.

### `--return_type` Applies To All Outcome Target Families

`--return_type` accepts five resolved forms overall:

- `disc_<gamma>` uses discounted returns
- `disc_<gamma>_skip1` uses discounted returns but skips the first rated future non-shot action
- `next_<N>` uses non-discounted lookahead returns
- `next_<N>_skip1` uses non-discounted lookahead returns but skips the first rated future non-shot action
- `in_<N>` uses the state at the Nth future eligible action and is supported only for `xt`, `goal_distance`, and `epv`

Example:

```powershell
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123450_abcdef12 --model gat --return_type next_10 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123456_abcdef12 --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123510_cdef1234 --model gat --use_xt --return_type disc_0.9 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123520_def12345 --model gat --use_goal_distance --return_type next_7 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123530_ef123456 --model gat --use_epv --return_type next_5 ...
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xg --return_type disc_0.9 --intended-receiver-mode angle_only
```

- Binary goals:
  - `next_<N>` uses the current `scores` / `concedes` logic over the next `N` events, including the current event
  - `next_<N>_skip1` uses the same event window, but suppresses the first future event contribution when that first future event is not a shot
  - `disc_<gamma>` writes discounted goal occurrence into `scores` / `concedes`
  - `disc_<gamma>_skip1` uses the same discounted goal scan, but suppresses the first future event contribution when that first future event is not a shot
- xG:
  - `next_<N>` uses non-discounted xG returns over the next `N` events
  - `next_<N>_skip1` uses the same xG event window, but suppresses the first future event contribution when that first future event is not a shot
  - `disc_<gamma>` keeps the existing discounted xG-probability logic
  - `disc_<gamma>_skip1` uses the same discounted xG scan, but suppresses the first future event contribution when that first future event is not a shot
- xT:
  - `next_<N>` uses the maximum future teammate/opponent xT over the next `N` eligible `pass` / `cross` / `shot` actions
  - `next_<N>_skip1` uses the same eligible-action window, but skips the first rated future action when that action is not a shot
  - `in_<N>` uses the xT value at the Nth future eligible `pass` / `cross` / `shot` action, unless an earlier eligible `shot` occurs first; only one of `scores_xT` / `concedes_xT` is non-zero
  - `disc_<gamma>` uses `max(gamma^k * xT)` over future eligible actions until the stop condition
  - `disc_<gamma>_skip1` skips the first rated future non-shot action and then applies weights `1, gamma, gamma^2, ...` to the remaining eligible actions
- goal_distance:
  - `next_<N>` uses the maximum future teammate/opponent goal-distance value over the next `N` eligible `pass` / `cross` / `shot` actions
  - `next_<N>_skip1` uses the same eligible-action window, but skips the first rated future action when that action is not a shot
  - `in_<N>` uses the goal-distance value at the Nth future eligible `pass` / `cross` / `shot` action, unless an earlier eligible `shot` occurs first; only one of `scores_goal_distance` / `concedes_goal_distance` is non-zero
  - `disc_<gamma>` uses `max(gamma^k * goal_distance)` over future eligible actions until the stop condition
  - `disc_<gamma>_skip1` skips the first rated future non-shot action and then applies weights `1, gamma, gamma^2, ...` to the remaining eligible actions
- EPV:
  - `next_<N>` uses the maximum future teammate/opponent EPV over the next `N` eligible `pass` / `cross` / `shot` actions
  - `next_<N>_skip1` uses the same eligible-action window, but skips the first rated future action when that action is not a shot
  - `in_<N>` uses the EPV value at the Nth future eligible `pass` / `cross` / `shot` action, unless an earlier eligible `shot` occurs first; only one of `scores_epv` / `concedes_epv` is non-zero
  - `disc_<gamma>` uses `max(gamma^k * epv)` over future eligible actions until the stop condition
  - `disc_<gamma>_skip1` skips the first rated future non-shot action and then applies weights `1, gamma, gamma^2, ...` to the remaining eligible actions

### Where to switch targets

Use `scripts/train_relevant_models.py` when you want the retained default setup:

```powershell
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xg --return_type disc_0.9 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xt --return_type next_5 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type next_3 --intended-receiver-mode original
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family epv --return_type next_5 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xt --return_type in_3 --intended-receiver-mode angle_only --no-action-intent --no-pass-intent --no-success-intent --no-pass-success --no-failure-receiver
```

The direct wrapper also supports partial reruns:

- retrain only `success_intent`: `python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --success-intent-only`
- rerun only the two outcome models with a different target configuration: `python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xt --return_type in_3 --intended-receiver-mode angle_only --no-action-intent --no-pass-intent --no-success-intent --no-pass-success --no-failure-receiver`
- rerun everything except `success_intent` after you already trained it once: `python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only --no-success-intent`

Use `train.py` directly when you need explicit low-level control:

```powershell
python train.py --task outcome_scoring --model gat ...
python train.py --task outcome_scoring --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --model gat --use_xt --return_type next_5 ...
python train.py --task outcome_scoring --model gat --use_goal_distance --return_type disc_0.9 ...
python train.py --task outcome_scoring --model gat --use_epv --return_type next_5 ...
python train.py --task outcome_scoring --model gat --feature_run_id <feature_run_id> ...
```

The existing low-level feature toggles on `train.py` are:

- `--xy_only`
- `--possessor_aware`
- `--keeper_aware`
- `--ball_z_aware`
- `--poss_vel_aware`
- `--extend_features`

These same controls are exposed in the wrappers as hyphenated flags on `scripts/train_relevant_models.py` and `scripts/main.py`. The wrappers keep the shared default profile described above, while `train.py` stays the low-level source of truth.

If you need to preserve the old numeric naming convention for a one-off run, `train.py` still accepts `--trial <n>` and writes `saved/<task>/<nn>/` for backward compatibility.

Note: the low-level entrypoints do not use the same spelling here:

- `train.py` uses `--feature_run_id`
- `test.py` uses `--feature-run-id`
- the wrapper scripts use `--feature-run-id`

If you generate or rebuild xT, goal-distance, or EPV artifacts after a feature run already exists, create a derived labels-only feature run before any matching target-family training run, for example `scripts/generate_relevant_features.py --extend-feature-run-id <feature_run_id> --refresh-target-family epv`. A full feature rebuild also works, but is no longer required just to refresh target labels.

## Intended-Receiver Workflow

There are three intended-receiver modes in the codebase:

- `original`: use the original labels from the synced event data
- `angle_only`: use the heuristic relabeling flow without a learned receiver model
- `model`: use a learned intended-receiver checkpoint

Every feature run now includes `original` and `angle_only` automatically. The `model` variant is included only when feature generation is given a pinned `--intended-receiver-model-id`.

The learned workflow is now explicit:

1. generate a feature run without `--intended-receiver-model-id`
2. train `success_intent` with `scripts/train_relevant_models.py --success-intent-only --feature-run-id <feature_run_id>`
3. generate a derived model-mode feature run with `scripts/generate_relevant_features.py --extend-feature-run-id <feature_run_id> --intended-receiver-model-id success_intent/<model_run_id>`
4. train the retained models on that new feature run with `--intended-receiver-mode model`

If you also need additional return semantics, repeat `--return_type` on the extension command. The derived run preserves the base run's existing return types first, then appends any newly requested return types.

`success_intent` is the teammate-selection intended-receiver model. It is trained from observed successful-pass receivers (`receiver_id`) and does not belong to any intended-receiver mode. `failure_receiver` is a separate auxiliary model used for failed-pass / opponent-receiver handling; it is not the intended-teammate model itself.

## EPV Workflow

EPV is a bootstrapped target family: first train the source component models on an existing feature run, then use those checkpoints to generate EPV sidecars, refresh labels in a derived feature run, and finally train the outcome models on `--target-family epv`.

The workflow is explicit:

1. generate a feature run with the return semantics you want available later, for example `scripts/generate_relevant_features.py --return_type next_5`
2. train the source component models with a non-EPV target family, for example `scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type next_5 --intended-receiver-mode angle_only`
3. generate EPV sidecars with `scripts/generate_epv.py --bundle-id <source_bundle_id>`
4. generate a derived labels-only feature run with `scripts/generate_relevant_features.py --extend-feature-run-id <feature_run_id> --refresh-target-family epv`
5. train the retained models on the derived feature run with `scripts/train_relevant_models.py --feature-run-id <epv_feature_run_id> --target-family epv --return_type next_5 --intended-receiver-mode angle_only`

If the source bundle is incomplete, pass explicit overrides to `scripts/generate_epv.py` with `--pass-intent-model-id`, `--pass-success-model-id`, `--outcome-scoring-model-id`, and `--outcome-conceding-model-id`.

## Notes

- `scripts/main.py` is the scoped pipeline runner for this repository.
- root-level `main.py` still reflects the full upstream defensive-score pipeline and is not part of this scoped reproduction.
- `datatools/tabular_feature.py` and the UxG/defensive-score path were left largely upstream because they are out of scope here.
- The core GNN training and evaluation flow stays close to upstream DEFCON; most project-specific changes are in preprocessing, target construction, and extra-data adapters.
- This repository is intended to track code, docs, and vendor snapshots only. Raw data, processed data, intermediate features, model outputs, and local environments are intentionally excluded from git.

## CLI Reference

This appendix covers every current `scripts/*.py` CLI entrypoint, including `scripts/main.py`. The legacy repo-root `main.py` is intentionally not included here because it is part of the upstream defensive-score path, not the scoped workflow described above.

### `scripts/main.py`

- `--target-family {goal,xg,xt,goal_distance,epv}`: retained outcome family passed to training. Required unless `--skip-train` is set.
- `--return_type <disc_gamma|disc_gamma_skip1|next_N|next_N_skip1|in_N>`: resolved return semantics passed to feature generation and training. `in_N` is valid only for `xt`, `goal_distance`, and `epv`. Required when feature generation or training is enabled.
- `--intended-receiver-mode {original,angle_only,model}`: retained-model training mode. Required unless `--skip-train` is set.
- `--intended-receiver-model-id <model_id>`: optional `success_intent` checkpoint used to add the `model` intended-receiver variant during feature generation.
- `--feature-run-id <feature_run_id>`: explicit feature run id to reuse or assign.
- `--bundle-id <bundle_id>`: explicit model bundle id to reuse or assign.
- `--success-intent-model-id <model_id>`: optional `success_intent` checkpoint forwarded to evaluation.
- `--skip-preprocess`, `--skip-xt`, `--skip-goal-distance`, `--skip-epv`, `--skip-features`, `--skip-train`, `--skip-evaluate`, `--skip-run-relevant`, `--skip-hawkeye`, `--skip-benchmark`, `--skip-skillcorner`: skip individual stages.
- `--benchmark-input-dir <path>`: local benchmark data root passed to `scripts/run_benchmark.py`.
- `--v-edge-features` / `--v-edge-features-no-poss` / `--no-v-edge-features`: control whether training uses all stored velocity-angle edge features, masks possessor-incident velocity edge columns, or drops velocity edge columns entirely. Default: on.
- `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, `--extend-features` / `--no-extend-features`: override the training feature profile.
- `--overwrite`: allow supported preprocessing and target-artifact outputs to be rebuilt.
- `--relevant-split {train,test,all}`: split passed through to `scripts/run_relevant_models.py`.
- `--device <device>`: device passed to evaluation and inference stages.
- `--dry-run`: print the resolved commands without executing them.

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

### `scripts/generate_epv.py`

- `--bundle-id <bundle_id>`: source model bundle containing `pass_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding`.
- `--pass-intent-model-id`, `--pass-success-model-id`, `--outcome-scoring-model-id`, `--outcome-conceding-model-id`: explicit source checkpoint overrides.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: newest compatible source feature run from the selected models or bundle.
- `--match-id <id>`: export EPV sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--overwrite`: overwrite existing EPV outputs. Default: off.

### `scripts/generate_relevant_features.py`

- repeat `--return_type <disc_gamma|disc_gamma_skip1|next_N|next_N_skip1|in_N>`: write labels for one or more return semantics in the same feature run. `in_N` is valid only for `xt`, `goal_distance`, and `epv`.
- `--intended-receiver-model-id <model_id>`: optional `success_intent` checkpoint used to additionally include the `model` intended-receiver variant.
- `--run-id <feature_run_id>`: pin the feature run id instead of auto-generating one.
- `--extend-feature-run-id <feature_run_id>`: create a new derived feature run from an existing completed run, copying existing artifacts and generating only newly requested return types, refreshed target labels, or the model intended-receiver variant.
- `--refresh-target-family {xt,goal_distance,epv}`: with `--extend-feature-run-id`, overwrite copied label tensors in the derived run from current target sidecars without rebuilding graph tensors. Repeat to record multiple refreshed target families.
- `--replace-intended-receiver-model`: with `--extend-feature-run-id`, allow a different `--intended-receiver-model-id` when the base run already contains `model` artifacts; only model-mode artifacts are regenerated in the new derived run.

### `scripts/train_relevant_models.py`

- `--target-family {goal,xg,xt,goal_distance,epv}`: retained outcome family. Required when `outcome_scoring` or `outcome_conceding` is enabled.
- `--return_type <disc_gamma|disc_gamma_skip1|next_N|next_N_skip1|in_N>`: resolved return semantics for the selected label directory. `in_N` is valid only for `xt`, `goal_distance`, and `epv`. Required when an outcome model is enabled; otherwise the wrapper falls back to the first available return type in the selected feature run.
- `--feature-run-id <feature_run_id>`: pin the feature run used for training. Required.
- `--diagnostic-feature-run-id <feature_run_id>`: optional feature run containing compatible `action_labels_next_10*` labels for canonical goal-event diagnostics when the selected run lacks embedded `goal next_10` diagnostic columns.
- `--intended-receiver-mode {original,angle_only,model}`: intended-receiver mode used for retained-model training. Required when any of `action_intent`, `pass_intent`, `pass_success`, `outcome_scoring`, `outcome_conceding`, or `failure_receiver` is enabled.
- `--success-intent-only`: train only the mode-independent `success_intent` model from successful pass receivers. This flag does not accept `--intended-receiver-mode`.
- `--action-intent` / `--no-action-intent`, `--pass-intent` / `--no-pass-intent`, `--success-intent` / `--no-success-intent`, `--pass-success` / `--no-pass-success`, `--outcome-scoring` / `--no-outcome-scoring`, `--outcome-conceding` / `--no-outcome-conceding`, `--failure-receiver` / `--no-failure-receiver`: enable or disable individual wrapper-managed checkpoints. Default: on for all except `failure_receiver`.
- `--pass-intent-model-id <pass_intent/model_run_id>`: existing compatible `pass_intent` checkpoint to use as the `pass_success` IPW model when `--no-pass-intent` is set.
- `--batch-size <n>` / `--batch_size <n>`: override the wrapper batch size for every low-level model training command.
- `--action-intent-batch-size <n>`, `--pass-intent-batch-size <n>`, `--success-intent-batch-size <n>`, `--pass-success-batch-size <n>`, `--outcome-scoring-batch-size <n>`, `--outcome-conceding-batch-size <n>`, `--failure-receiver-batch-size <n>`: override one model's batch size. Model-specific flags override `--batch-size`.
- `--bundle-id <bundle_id>`: pin the training bundle manifest id.
- `--v-edge-features` / `--v-edge-features-no-poss` / `--no-v-edge-features`: control whether training uses all stored velocity-angle edge features, masks possessor-incident velocity edge columns, or drops velocity edge columns entirely. Default: on.
- `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, `--extend-features` / `--no-extend-features`: override the wrapper training defaults.
- `--outcome-scoring-trial <n>` and `--outcome-conceding-trial <n>`: override the auto-generated run ids for those tasks with legacy numeric ids.

### `scripts/evaluate_relevant_models.py`

- `--bundle-id <bundle_id>`: preferred explicit model bundle to evaluate.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--diagnostic-feature-run-id <feature_run_id>`: optional diagnostic feature run passed to `test.py` for outcome models.
- `--device <device>`: device passed to `test.py`. Default: `cuda:0`.

### `scripts/run_relevant_models.py`

- `--split {train,test,all}`: choose which Sportec split to export. Default: `test`.
- `--match-id <id>`: restrict export to one or more specific matches. Default: all matches in the selected split.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: newest compatible source feature run from the selected models or bundle.
- `--run-id <component_run_id>`: pin the created component export run id. Default: auto-generate a new component run id.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id used to export `success_intent.parquet`.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--output-dir <path>`: parent directory for the created component run folder. Default: `data/component_runs`.

### `scripts/run_hawkeye.py`

- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--situation-id <id>`: restrict inference to one or more specific Hawkeye situation ids. Default: all valid situations.
- `--limit <N>`: process only the first `N` selected situations. Default: no limit.
- `--freeze-ballreceipt`: freeze possessor and ball state after `BallReceipt`. Default: on.
- `--no-freeze-ballreceipt`: disable BallReceipt freezing. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created HawkEye component run id. Default: auto-generate a new HawkEye component run id.
- `--output-dir <path>`: parent directory for the created Hawkeye run folder. Default: `data/component_runs/hawkeye`.

### `scripts/run_benchmark.py`

- `--input-dir <path>`: local benchmark data root. Default: `benchmark`.
- `--modification <id>`: restrict inference to one or more specific benchmark modifications. Default: all valid modifications.
- `--limit <N>`: process only the first `N` selected modifications. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created benchmark component run id. Default: auto-generate a new benchmark component run id.
- `--output-dir <path>`: parent directory for the created benchmark run folder. Default: `data/component_runs/benchmark`.

### `scripts/run_skillcorner.py`

- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--match-id <id>`: restrict inference to one or more specific SkillCorner match ids. Default: all discoverable valid matches.
- `--limit <N>`: process only the first `N` selected matches. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created SkillCorner component run id. Default: auto-generate a new SkillCorner component run id.
- `--output-dir <path>`: parent directory for the created SkillCorner run folder. Default: `data/component_runs/skillcorner`.

### `scripts/visualize_action_components.py`

- `--match-id <id>`: Sportec match id to visualize. Default: required.
- `--action-id <action_id>`: CSV `action_id` from `data/event_synced/<match_id>.csv`. Default: one of the identifier options is required.
- `--row-index <index>`: legacy modeled-action row index. Default: off.
- `--original-event-id <id>`: raw Sportec event id lookup. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: newest compatible source feature run from the selected models or bundle.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id used for intended-recipient overlays.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--output-dir <path>`: visualization root directory. Default: `data/visualizations`.

### `scripts/visualize_hawkeye.py`

- `--situation-id <id>`: restrict visualization to one or more Hawkeye situation ids from the selected component run. Default: all situations in the selected component run.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--component-run-id <component_run_id>`: versioned Hawkeye component run to visualize. Default: latest successful Hawkeye component run.
- `--component-dir <path>`: explicit Hawkeye component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.
- `--output-dir <path>`: visualization root directory. Default: `data/visualizations/hawkeye`.

### `scripts/visualize_benchmark.py`

- `--input-dir <path>`: local benchmark data root. Default: `benchmark`.
- `--modification <id>`: restrict visualization to one or more benchmark modifications from the selected component run. Default: all available modifications in the selected run.
- `--game-state {1,2}`: restrict visualization to one or more game states. Default: both game states present in the selected run.
- `--component-run-id <component_run_id>`: versioned benchmark component run to visualize. Default: latest successful benchmark component run.
- `--component-dir <path>`: explicit benchmark component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--output-dir <path>`: visualization root directory. Default: `data/visualizations/benchmark`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.

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
- `--output-dir <path>`: visualization root directory. Default: `data/visualizations/hawkeye`.

### `scripts/visualize_skillcorner.py`

- `--match-id <id>`: SkillCorner match id to visualize. Default: required.
- `--index <player_possession_index>`: SkillCorner `player_possession` event index. Default: required.
- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--component-run-id <component_run_id>`: versioned SkillCorner component run to visualize. Default: latest successful SkillCorner component run.
- `--component-dir <path>`: explicit component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--output-dir <path>`: visualization root directory. Default: `data/visualizations/skillcorner`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--gif`: write GIFs instead of MP4s. Default: off, so MP4s are written.

## Script I/O Reference

This appendix summarizes the primary input and output files for each `scripts/*.py` entrypoint. Paths below are code-level defaults and conventions, not a snapshot of whatever files happen to be present on a given device.

### `scripts/main.py`

- Inputs: the inputs of the enabled downstream stages, typically raw Sportec season folders plus optional HawkEye and SkillCorner raw directories, plus explicit `feature_run_id` or `bundle_id` values when prerequisite stages are skipped.
- Outputs: no unique files of its own; it orchestrates the outputs of the enabled downstream scripts.

### `scripts/preprocess_sportec.py`

- Inputs:
  - `Bundesliga_season_23_24/tracking_data/*`
  - `Bundesliga_season_23_24/event_data/*`
  - `Bundesliga_season_23_24/match_information/*`
  - `Bundesliga_season_24_25/tracking_data/*`
  - `Bundesliga_season_24_25/event_data/*`
  - `Bundesliga_season_24_25/match_information/master/*`
  - `Bundesliga_season_24_25/match_information/starting_players/*`
  - `Bundesliga_season_24_25/KPI_Merged/*`
- Outputs:
  - `data/tracking/*.parquet`
  - `data/tracking_processed/*.parquet`
  - `data/event/event.parquet`
  - `data/event_synced/*.csv`
  - `data/lineup/line_up.parquet`
  - `data/splits/match_splits.json`

### `scripts/generate_xt.py`

- Inputs:
  - `data/event_synced/*.csv`
  - `data/splits/match_splits.json`
- Outputs:
  - `data/xT/xT.csv`
  - `data/xT/xT_grid.csv`
  - `data/xT/fit_metadata.json`
  - `data/xT/matches/*.csv`

### `scripts/generate_goal_distance.py`

- Inputs:
  - `data/event_synced/*.csv`
- Outputs:
  - `data/goal_distance/goal_distance.csv`
  - `data/goal_distance/metadata.json`
  - `data/goal_distance/matches/*.csv`

### `scripts/generate_epv.py`

- Inputs:
  - `data/event_synced/*.csv`
  - `data/tracking_processed/*.parquet`
  - `data/lineup/line_up.parquet`
  - a compatible runtime feature run for Sportec graphs and resolved actions
  - source checkpoints from `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/epv/epv.csv`
  - `data/epv/metadata.json`
  - `data/epv/matches/*.csv`

### `scripts/generate_relevant_features.py`

- Inputs:
  - `data/tracking_processed/*.parquet`
  - `data/event_synced/*.csv`
  - `data/lineup/line_up.parquet`
  - `data/splits/match_splits.json`
  - optional target sidecars under `data/xT/matches/*.csv`, `data/goal_distance/matches/*.csv`, and `data/epv/matches/*.csv`
  - optional learned intended-receiver checkpoint `saved/success_intent/<model_run_id>/...`
    That checkpoint also determines the edge schema of the transient failed-pass relabeling graphs used during feature generation.
- Outputs:
  - `data/features/runs/<feature_run_id>/action_graphs/*.pt`
  - `data/features/runs/<feature_run_id>/post_action_graphs/*.pt`
  - `data/features/runs/<feature_run_id>/action_graphs_intent_train/*.pt`
  - `data/features/runs/<feature_run_id>/action_graphs_success_intent/*.pt`
  - `data/features/runs/<feature_run_id>/action_labels_<return_type>*.pt` for each requested `return_type` and intended-receiver mode
  - `data/features/runs/<feature_run_id>/success_intent_labels/*.pt`
  - `data/features/runs/<feature_run_id>/action_labels_intent_train_<return_type>*.pt` for each requested `return_type` and intended-receiver mode
  - `data/features/runs/<feature_run_id>/resolved_actions*.parquet` for each intended-receiver mode
  - `data/features/runs/<feature_run_id>/metadata.json`

### `scripts/train_relevant_models.py`

- Inputs:
  - `data/features/runs/<feature_run_id>/...`
- Outputs:
  - `saved/pass_intent/<model_run_id>/...`
  - `saved/action_intent/<model_run_id>/...`
  - `saved/pass_success/<model_run_id>/...`
  - `saved/outcome_scoring/<model_run_id>/...`
  - `saved/outcome_conceding/<model_run_id>/...`
  - `saved/success_intent/<model_run_id>/...` when `success_intent` is enabled
  - `saved/failure_receiver/<model_run_id>/...`
  - `saved/bundles/<bundle_id>/metadata.json`

### `scripts/evaluate_relevant_models.py`

- Inputs:
  - feature artifacts resolved from the selected checkpoint metadata
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - no dedicated files; metrics are printed to stdout by `test.py`

### `scripts/run_relevant_models.py`

- Inputs:
  - feature artifacts from the selected runtime feature run
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/component_runs/<component_run_id>/<match_id>/action_intent.parquet`
  - `data/component_runs/<component_run_id>/<match_id>/pass_intent.parquet`
  - optionally `data/component_runs/<component_run_id>/<match_id>/success_intent.parquet` when a `success_intent` checkpoint is selected
  - `data/component_runs/<component_run_id>/<match_id>/pass_success.parquet`
  - `data/component_runs/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
  - `data/component_runs/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
  - `data/component_runs/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
  - `data/component_runs/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
  - `data/component_runs/<component_run_id>/metadata.json`

### `scripts/visualize_action_components.py`

- Inputs:
  - feature artifacts from the selected runtime feature run
  - `data/event_synced/<match_id>.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/visualizations/<match_id>/<action_id>/action_intent.png`
  - `data/visualizations/<match_id>/<action_id>/pass_intent.png`
  - `data/visualizations/<match_id>/<action_id>/pass_success.png`
  - `data/visualizations/<match_id>/<action_id>/outcome_scoring_success.png`
  - `data/visualizations/<match_id>/<action_id>/outcome_scoring_failure.png`
  - `data/visualizations/<match_id>/<action_id>/outcome_conceding_success.png`
  - `data/visualizations/<match_id>/<action_id>/outcome_conceding_failure.png`
  - `data/visualizations/<match_id>/<action_id>/pass_score.png`
  - optionally `data/visualizations/<match_id>/<action_id>/intended_recipient.png`

### `scripts/run_hawkeye.py`

- Inputs:
  - `hawkeye_data/centroid_data_team.csv`
  - `hawkeye_data/ball_data_selected.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
  - `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
  - `data/component_runs/hawkeye/<component_run_id>/metadata.json`

### `scripts/visualize_hawkeye.py`

- Inputs:
  - `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
  - `data/component_runs/hawkeye/<component_run_id>/metadata.json`
  - `hawkeye_data/centroid_data_team.csv`
  - `hawkeye_data/ball_data_selected.csv`
- Outputs:
  - `data/visualizations/hawkeye/<situation_id>/*.mp4`
  - or `data/visualizations/hawkeye/<situation_id>/*.gif`

### `scripts/run_benchmark.py`

- Inputs:
  - `benchmark/modification_<n>/game_state_1.csv`
  - `benchmark/modification_<n>/game_state_2.csv`
  - `benchmark/modification_<n>/modification.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/component_runs/benchmark/<component_run_id>/benchmark_data.parquet`
  - `data/component_runs/benchmark/<component_run_id>/benchmark_data.csv`
  - `data/component_runs/benchmark/<component_run_id>/metadata.json`

### `scripts/visualize_benchmark.py`

- Inputs:
  - `data/component_runs/benchmark/<component_run_id>/benchmark_data.parquet`
  - `data/component_runs/benchmark/<component_run_id>/metadata.json`
  - `benchmark/modification_<n>/game_state_1.csv`
  - `benchmark/modification_<n>/game_state_2.csv`
  - `benchmark/modification_<n>/modification.csv`
- Outputs:
  - `data/visualizations/benchmark/modification_<n>/game_state_<m>/*.png`

### `scripts/run_and_visualize_hawkeye.py`

- Inputs:
  - `hawkeye_data/centroid_data_team.csv`
  - `hawkeye_data/ball_data_selected.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/visualizations/hawkeye/<situation_id>/*.mp4`
  - or `data/visualizations/hawkeye/<situation_id>/*.gif`

### `scripts/run_skillcorner.py`

- Inputs:
  - `skillcorner_data/<match_id>_tracking.jsonl`
  - `skillcorner_data/<match_id>_match.json`
  - `skillcorner_data/<match_id>_dynamic_events.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/action_intent.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/pass_intent.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/pass_success.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
  - `data/component_runs/skillcorner/<component_run_id>/metadata.json`

### `scripts/visualize_skillcorner.py`

- Inputs:
  - `data/component_runs/skillcorner/<component_run_id>/<match_id>/*.parquet`
  - `skillcorner_data/<match_id>_tracking.jsonl`
  - `skillcorner_data/<match_id>_match.json`
  - `skillcorner_data/<match_id>_dynamic_events.csv`
- Outputs:
  - `data/visualizations/skillcorner/<match_id>/<index>/*.mp4`
  - or `data/visualizations/skillcorner/<match_id>/<index>/*.gif`
