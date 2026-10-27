# dm_model

This repository is a scoped adaptation of the upstream DEFCON implementation for Sportec Bundesliga data, currently centered on Bundesliga data from 2022/23 through 2024/25.
Upstream DEFCON source code: https://github.com/hyunsungkim-ds/defcon

The model structure is copied from DEFCON, the upstream source code for the paper "Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks". This repository keeps the downstream DEFCON runtime close to upstream, narrows the retained scope, and adds the dataset- and workflow-specific adaptations needed for Sportec, HawkEye, and SkillCorner data.

## Table Of Contents

- [Changes From Upstream DEFCON](#changes-from-upstream-defcon)
- [Upstream Baseline](#upstream-baseline)
- [Expected Raw Data Layout](#expected-raw-data-layout)
- [Produced Data Layout](#produced-data-layout)
- [Run-Id Workflow](#run-id-workflow)
- [Split Definition](#split-definition)
- [Prepared dataset cache and bounded training memory](docs/intent_dataset_loading.md)
- [Training controls, checkpoints and monitoring](docs/training_runtime.md)
- [Endpoint repair, duration selection, and rollout](docs/endpoint_policy.md)
- [Empirical reachability circles for pc-xPass](docs/reachability.md)
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
- [physical_xpass_workflow](#physical_xpass_workflow)
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

- `Bundesliga_season_22_23/tracking_data`
- `Bundesliga_season_22_23/event_data`
- `Bundesliga_season_22_23/starting_players`
- `Bundesliga_season_22_23/master`
- `Bundesliga_season_22_23/events_advanced`
- `Bundesliga_season_23_24/tracking_data`
- `Bundesliga_season_23_24/event_data`
- `Bundesliga_season_23_24/match_information`
- `Bundesliga_season_24_25/tracking_data`
- `Bundesliga_season_24_25/event_data`
- `Bundesliga_season_24_25/match_information/master`
- `Bundesliga_season_24_25/match_information/starting_players`
- `Bundesliga_season_24_25/KPI_Merged`

`scripts/preprocess_sportec.py` handles the observed raw-format differences between the three seasons and still writes the same canonical processed outputs under `data`. The 2023/24 synchronization feed is a semicolon-delimited CSV; the 2022/23 and 2024/25 feeds use the extensionless AdvancedEvents XML format.

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
- `xT/xT_source_grid.csv`
- `xT/xT_xy_surface.csv`
- `xT/xT_glm_fit_sample.csv`
- `xT/xT_xy_surface_3d.png`
- `xT/fit_metadata.json`
- `xT/matches/*.csv`
- `goal_distance/matches/*.csv`
- `epv/metadata.json`
- `epv/matches/*.csv`
- `lineup/line_up.parquet`
- `features/runs/<feature_run_id>/...`
- `features/runs/latest.json`
- `component_runs/sportec/<component_run_id>/...`
- `component_runs/sportec/latest.json`
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
- `data/component_runs/sportec/<component_run_id>` for versioned per-match component prediction exports from `scripts/run_relevant_models.py`
- `data/component_runs/hawkeye/<component_run_id>` for versioned HawkEye exports
- `data/component_runs/benchmark/<component_run_id>` for versioned benchmark exports
- `data/component_runs/skillcorner/<component_run_id>` for versioned SkillCorner exports
- `data/runtime_physical_xpass/<dataset>` for runtime physical xPass caches used by inference and visualization
- `data/pc_xpass/<dataset>` for pitch-control-style xPass caches used by inference and visualization when `--pc-xpass` is selected
- `data/visualizations/sportec/<visualization_run_id>` and visualization subfamilies for versioned rendered outputs and metadata
- `saved/<task>/<model_run_id>` for trained checkpoint runs
- `saved/bundles/<bundle_id>` for machine-readable training bundle manifests

## Run-Id Workflow

There are now four different versioned artifact families:

- `feature_run_id` for generated Sportec graph/label artifacts under `data/features/runs/...`
- `model_id = <task>/<model_run_id>` for trained checkpoints under `saved/<task>/...`
- `component_run_id` for exported Sportec predictions under `data/component_runs/sportec/...`
- `visualization_run_id` for rendered Sportec plots under `data/visualizations/sportec/...`

Feature and component runs use explicit `latest.json` pointers:

- `data/features/runs/latest.json`
- `data/component_runs/sportec/latest.json`
- `data/component_runs/hawkeye/latest.json`
- `data/component_runs/benchmark/latest.json`
- `data/component_runs/skillcorner/latest.json`

Visualization runs do not use `latest.json`; the source `component_run_id` or model ids are recorded in each visualization run's `metadata.json`.

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
- `data/features/runs/<feature_run_id>/physical_xpass/matches/*.parquet` when physical xPass sidecars are precomputed
- `data/features/runs/<feature_run_id>/metadata.json`

The component run root contains one folder per processed match:

- `data/component_runs/sportec/<component_run_id>/<match_id>/action_intent.parquet`
- `data/component_runs/sportec/<component_run_id>/<match_id>/pass_intent.parquet`
- optionally `data/component_runs/sportec/<component_run_id>/<match_id>/success_intent.parquet` when a `success_intent` checkpoint is provided
- `data/component_runs/sportec/<component_run_id>/<match_id>/pass_success.parquet`
- `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
- `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
- `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
- `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
- `data/component_runs/sportec/<component_run_id>/metadata.json`

The external-data adapters follow the same pattern:

- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.parquet`
- `data/component_runs/hawkeye/<component_run_id>/hawkeye_data.csv`
- `data/component_runs/hawkeye/<component_run_id>/metadata.json`
- `data/component_runs/benchmark/<component_run_id>/benchmark_data.parquet`
- `data/component_runs/benchmark/<component_run_id>/benchmark_data.csv`
- `data/component_runs/benchmark/<component_run_id>/metadata.json`
- `data/component_runs/skillcorner/<component_run_id>/<match_id>/*.parquet`
- `data/component_runs/skillcorner/<component_run_id>/metadata.json`

Visualization run roots contain rendered media plus `metadata.json`:

- `data/visualizations/sportec/<visualization_run_id>/metadata.json`
- `data/visualizations/hawkeye/<visualization_run_id>/metadata.json`
- `data/visualizations/benchmark/<visualization_run_id>/metadata.json`
- `data/visualizations/skillcorner/<visualization_run_id>/metadata.json`

Checkpoint runs also write metadata:

- `saved/<task>/<model_run_id>/args.json`
- `saved/<task>/<model_run_id>/metadata.json`
- `saved/<task>/<model_run_id>/best_weights.pt`
- `saved/bundles/<bundle_id>/metadata.json`

The run metadata records the settings used for that invocation. Checkpoints with explicit selectors save `feature_selections` and `feature_mask_version`; feature signatures include resolved per-feature role permissions under `feature_masks`. Component runs record each model's signature and graph schema, including edge modes and dimensions. Inference-mask metadata records the original request, canonical selections, resolved operations, effectiveness, and configuration identity. Historical feature fields remain readable for older checkpoints.

## Current Artifact Contract

The current pipeline now follows an explicit-artifact contract:

- `scripts/generate_relevant_features.py` can include multiple `--return_type` values in one feature run and always writes the full velocity-angle edge-feature schema.
- Each generated feature run always contains `original` and `angle_only` intended-receiver variants, and it additionally contains `model` only when `--intended-receiver-model-id <success_intent/model_run_id>` is supplied.
- `scripts/train_relevant_models.py` now requires an explicit `--feature-run-id` and supports per-model toggles. `--target-family` and `--return_type` are required only when an outcome model is enabled, and `--intended-receiver-mode` is required only when a mode-dependent model is enabled.
- Training selects stored velocity-angle edge inputs with `--velocity-angle-edge-features {all,no-poss,none}`; the default is `none`. Feature generation no longer has an `--add_v_edge_features` toggle. The older training spellings remain compatibility aliases.
- `scripts/evaluate_relevant_models.py`, `scripts/run_relevant_models.py`, `scripts/run_hawkeye.py`, `scripts/run_benchmark.py`, `scripts/run_skillcorner.py`, and `scripts/visualize_action_components.py` now prefer `--bundle-id` or explicit model ids. Sportec runtime scripts can combine compatible checkpoints from different feature runs; they automatically choose the newest compatible source feature run unless `--feature-run-id` is supplied. Component exports with physical xPass and caching enabled instead select the cache's feature run and fail if it is unavailable or incompatible.
- Visualization scripts write versioned run folders with `metadata.json`; component-based visualizers record the selected `component_run_id` or explicit component directory, and direct model-based visualizers record the model ids used.
- `scripts/main.py` now threads explicit `feature_run_id` and `bundle_id` values between stages. It does not auto-bootstrap learned intended-receiver mode anymore; for `--intended-receiver-mode model`, provide or first create a `success_intent` checkpoint and pass it as `--intended-receiver-model-id`.


## Split Definition

Preprocessing records every successfully preprocessed Sportec match in a canonical sequence sorted by `MatchId`. The user-facing `--train-split <percentage>` option assigns the first percentage to model development and retains the chronologically later remainder as the independent test set. The percentage is an integer from 1 through 99 and uses floor rounding. Alternatively, `--train-count <int>` selects an exact development match count, including validation matches, before filtering for available features. The two options are mutually exclusive; the count must be at least 1 and less than the total match count.

Resolved match assignments are stored as immutable, fingerprinted manifests under `data/splits/manifests`. Feature runs, model runs, bundles, and evaluations record that manifest identity so incompatible artifacts cannot be mixed. Feature and xT generation default to `--train-split 50`, which gives 459 development and 459 test matches for the current 918-match universe. Training and evaluation load the saved manifest independently of the current universe; explicit flags assert its selector. Two verified legacy feature runs have explicit historical mappings; unmapped legacy runs are rejected. See [recorded split provenance](docs/recorded_splits.md). Physical xPass generation has no split selector. For the first two and a half seasons as development data, use `--train-count 765`: this gives 765 development matches (2022/23, 2023/24, and 2024/25 matchdays 1-17) and 153 test matches (2024/25 matchdays 18-34). Count and percentage selections have distinct manifest identities even if their assignments coincide. Evaluation infers the selector from model provenance; an explicit selector must match its mode and value.

`--validation-mode holdout` is the default, reserving the last 20% of development matches for validation. `holdout_80_20` remains an alias requiring exactly that split. `--validation-count` or `--validation-fraction` can change the reservation. `--validation-mode expanding` requires a list of `--training-counts` or `--training-fractions` and trains fresh models on nested prefixes against the same reserved validation cohort. Training fractions refer to the pool remaining after validation is reserved. The independent test set never changes.

Refitting is optional with `--final-refit`. It starts fresh on all development games, copying the largest validation stage's best epoch and per-epoch learning rates, without replaying rollbacks or applying a minimum duration. Stages are saved as `holdout`, `fold_1` through `fold_N`, and optionally `final_refit`; the latest delivered stage is published at the model root. Separate source-driven refit batches accept repeatable `--refit-source-model-id TASK/RUN[/STAGE]` without any feature or model configuration flags. See [configurable validation and refitting](docs/configurable_validation.md) for examples, rounding, provenance, source requirements, and batch behavior.

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
- `scripts/main.py` still forwards the legacy velocity-edge switches `--v-edge-features`, `--v-edge-features-no-poss`, and `--no-v-edge-features` (default: off), plus the older boolean feature-profile switches listed in its [CLI reference](#scriptsmainpy).
- For explicit selections such as `--vel-node-features none --vx defending --vy defending`, call `scripts/train_relevant_models.py` directly. These selectors are shared with `train.py` and are not forwarded by `scripts/main.py`. Use `--skip-train --bundle-id <bundle_id>` when continuing the pipeline with a separately trained bundle. See [training input controls](#training-input-controls-and-automatic-exclusions).
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

- raw Sportec season folders under `Bundesliga_season_22_23/...`, `Bundesliga_season_23_24/...`, and `Bundesliga_season_24_25/...`
- for `22_23`, the root-level split metadata layout under `master`, `starting_players`, and `events_advanced`
- for `24_25`, the split metadata layout under `match_information/master`, `match_information/starting_players`, and `KPI_Merged`

Outputs:

- `data/tracking/*.parquet`
- `data/tracking_processed/*.parquet`
- `data/event/event.parquet`
- `data/event_synced/*.csv`
- `data/control_events_synced/*.parquet`
- `data/carry_segments/*.parquet`
- `data/carry_segments/audits/*.parquet`
- `data/lineup/line_up.parquet`
- `data/splits/match_splits.json`

Useful options:

- `--match-id DFL-MAT-...` to process only selected matches
- `--season {22_23,23_24,24_25}` to process selected seasons; repeat the option to select more than one
- `--limit N` to smoke-test on the first `N` matches
- `--overwrite` to rebuild existing outputs
- `--skip-sync` to stop before ELASTIC synchronization
- `--skip-carry-artifacts` to build canonical and synchronized outputs without carry sidecars
- `--carry-artifacts-only` to rebuild carry sidecars from existing canonical outputs without preprocessing them again

To add 2022/23 first and then refresh carry artifacts consistently for all seasons:

```powershell
python scripts/preprocess_sportec.py --season 22_23 --skip-carry-artifacts
python scripts/preprocess_sportec.py --carry-artifacts-only
```

`--carry-artifacts-only` and `--skip-carry-artifacts` are mutually exclusive. An unfiltered carry-only run processes every discovered season and overwrites the per-match control-event, carry-segment, and carry-audit sidecars. A normal unfiltered preprocessing run revisits every discovered match; without `--overwrite`, only existing raw and processed tracking Parquets are reused automatically.

Subset-safe aggregate behavior:

- when you run a subset with `--season`, `--match-id`, or `--limit`, the per-match files for the selected matches are still processed as usual
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

The canonical train/test split remains percentage-based over the MatchId-ordered processed universe. Adding the third complete season expands that universe from 612 to 918 matches, changes its fingerprint, and regenerates percentage manifests without assigning entire seasons to either split.

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
  - `data/xT/xT_source_grid.csv`
  - `data/xT/xT_xy_surface.csv`
  - `data/xT/xT_glm_fit_sample.csv`
  - `data/xT/xT_xy_surface_3d.png`
  - `data/xT/fit_metadata.json`
  - `data/xT/matches/*.csv`
- for goal-distance:
  - `data/goal_distance/goal_distance.csv`
  - `data/goal_distance/metadata.json`
  - `data/goal_distance/matches/*.csv`
  - raw `goal_distance` values are bounded to `[0.0, 1.0]`
- for EPV:
  - `data/epv/epv.csv`
  - `data/epv/metadata.json`
  - `data/epv/matches/*.csv`

Useful options:

- `--match-id DFL-MAT-...` to export only selected matches
- `--limit N` to restrict export generation to the first `N` available synced matches
- `--source-grid-l N` and `--source-grid-w N` to fit the socceraction source xT grid at custom dimensions; defaults are `12` and `8`
- `--reuse-source-grid` to load `data/xT/xT_source_grid.csv` and skip socceraction source-grid fitting
- `--use-interaction` to add the x-depth by centrality interaction to the x/y logit GLM
- `--use-nonlinear x` and `--use-nonlinear y` to add constrained squared and cubic GLM terms; repeat the flag to use both axes
- `--fit-only` to write only model artifacts for testing, without updating `data/xT/xT.csv` or `data/xT/matches/*.csv`
- `--overwrite` to rebuild existing outputs

This is a separate post-preprocessing step. It reads canonical synced event CSVs, fits or exports the sidecar targets, and writes per-match artifacts used later during feature generation.
xT generation fits an upstream socceraction grid on training actions, writes it to `xT_source_grid.csv`, then distills that source grid into a smooth x/y logit GLM surface. `normalized_x` is `start_x / 105`; `normalized_centrality` is the smooth symmetric feature `sin(pi * start_y / 68)`. Optional interaction and constrained squared/cubic terms can make this surface more flexible. Monotonicity toward goal and toward the pitch center is enforced during fitting and checked again before outputs are written; if the final guardrail detects a tiny violation, generation records the warning in metadata and still writes the outputs for inspection. Event-level `xT` values come from the GLM surface with shot values floored by `xG`; `xT_grid.csv` is retained as a compatibility projection of that surface onto the legacy 12x8 cell centers. Use `--fit-only` when comparing model variants without changing event-level sidecars. For one-time migration from an old xT run, rename the old socceraction `data/xT/xT_grid.csv` to `data/xT/xT_source_grid.csv`, then run `python scripts/generate_xt.py --reuse-source-grid --overwrite`.

EPV computes `sum(pass_intent * pass_score)` across players, where `pass_score` combines pass success with the scoring and conceding outcome models. Shot actions use `max(epv, xG)`, matching the xT shot-value convention.

### 3. Generate graph features and labels

```powershell
python scripts/generate_relevant_features.py --return_type disc_0.9
python scripts/generate_relevant_features.py --return_type disc_0.9 --return_type next_5 --return_type next_3
python scripts/generate_relevant_features.py --return_type next_5 --return_type next_5_skip1 --return_type disc_0.9_skip1
python scripts/generate_relevant_features.py --run-id feature_20260414T123456_abcdef12 --return_type disc_0.9 --intended-receiver-model-id success_intent/<model_run_id>
python scripts/generate_relevant_features.py --return_type disc_0.9 --next-action-conditions-off
python scripts/generate_relevant_features.py --return_type disc_0.9 --num-workers auto --worker-thread-limit 1
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --intended-receiver-model-id success_intent/<model_run_id>
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --return_type disc_max_0.9 --in-place
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --intended-receiver-model-id success_intent/<new_model_run_id> --replace-intended-receiver-model
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --refresh-target-family epv
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --refresh-target-family epv --in-place
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --pass-height
python scripts/generate_relevant_features.py --return_type disc_0.9 --pass-height --pass-height-threshold 1.8
python scripts/generate_relevant_features.py --extend-feature-run-id <base_feature_run_id> --pass-height --pass-height-threshold 1.8
```

Inputs:

- `data/tracking_processed/*.parquet`
- `data/event_synced/*.csv`
- `data/lineup/line_up.parquet`
- `data/splits/match_splits.json`
- optional sidecars from `data/xT/matches/*.csv`, `data/goal_distance/matches/*.csv`, and `data/epv/matches/*.csv`
- optional learned intended-receiver checkpoint referenced by `--intended-receiver-model-id`

By default, each invocation creates a new feature-artifact run under `data/features/runs/<feature_run_id>/` and updates `data/features/runs/latest.json` after completion. `--extend-feature-run-id` creates a derived run by default, hard-linking unchanged artifacts from a completed base run and generating only the requested additions or refreshes. `--artifact-copy-mode copy` explicitly opts into a full copy. `--in-place` applies the requested extension or refresh to the existing run; `--overwrite-feature-run` is a compatibility alias with identical behavior. Both spellings may be supplied together. In-place operations do not accept `--run-id`, retain the base run ID, and leave `latest.json` unchanged. Operation flags still authorize specific changes: a different receiver model requires `--replace-intended-receiver-model`, and target refresh requires `--refresh-target-family`. A destination flag alone does not regenerate artifacts. Extensions must preserve the base run's split and next-action selection setting. xT-only refresh inherits those settings unless explicitly supplied.

In-place failure handling preserves the original completed state when data artifacts are known to be unchanged, while recording the failed attempt. Partial or uncertain changes mark the run failed. Artifact replacement remains atomic per file, not transactional across the run. Use a derived run when other processes must continue reading the affected artifacts.

Behavior:

- every feature run always includes the `original` and `angle_only` intended-receiver variants
- `model` is included only when `--intended-receiver-model-id` is supplied
- graphs are written once per run and always include the full velocity-angle edge-feature schema
- labels are written for every requested `--return_type`
- pass/cross filtering keeps the current next-action consistency conditions by default; `--next-action-conditions-off` keeps valid synced pass/cross actions even when `receiver_id` does not match `next_player_id`, the receiver is not `out`, and the next action is not `foul` or `freekick_short`

Useful options:

- `--run-id <feature_run_id>` to pin the created run id instead of auto-generating one
- `--extend-feature-run-id <existing_feature_run_id>` to create a new derived run from an existing completed run without rebuilding shared graph tensors
- `--in-place` with `--extend-feature-run-id` to apply the requested additions or refreshes to the existing run; incompatible with `--run-id`
- `--overwrite-feature-run` as a compatibility alias for `--in-place`, with identical behavior and no runtime deprecation warning
- `--refresh-target-family <xt|goal_distance|epv>` with `--extend-feature-run-id` to rebuild copied label tensors from current target sidecars without rebuilding graph tensors
- `--pass-height` to enable pass-height label configuration during a new run, or with `--extend-feature-run-id` to rebuild copied label tensors so `pass_height` training labels are present without rebuilding graph tensors
- `--pass-height-threshold <meters>` with `--pass-height` to classify a pass as high when its maximum `ball_z` is at least the supplied value; defaults to `2.0` metres and is recorded in run metadata
- repeat `--return_type <disc_gamma|disc_gamma_skip1|disc_max_gamma|disc_max_gamma_skip1|disc_poly_max_b_z|disc_poly_max_b_z_spstop|next_N|next_N_skip1|in_N>` to include multiple resolved return semantics in one feature run
- `--intended-receiver-model-id <model_id>` to additionally include the `model` intended-receiver variant
- `--next-action-conditions-on` / `--next-action-conditions-off` to keep or disable the pass/cross next-action consistency filter; default: on
- `--num-workers <N|auto>` to parallelize matches inside each `datatools/graph_feature.py` subprocess; default: `1`
- `--worker-thread-limit <N>` to cap BLAS/OpenMP-style thread pools per worker; default: `1`
- `--replace-intended-receiver-model` with `--extend-feature-run-id` and `--intended-receiver-model-id` to regenerate copied model-mode artifacts with a different `success_intent` checkpoint in the derived run

Terminal output progression:

A normal (full) feature run executes 5 sequential steps. Before each subprocess starts, it prints a top-level progress line such as `Feature generation step 1/5: train split with post_action + augment_blocks`.

With `--num-workers 1`, each subprocess keeps the old sequential per-match flow and prints match headers such as `[1/306] ... [306/306]`, followed by these messages:

1. **train split with post_action + augment_blocks:** prints `"Successfully saved {N} augmented events for mode={mode}."` then `"Successfully saved for {N} events."`
2. **test split with post_action:** prints `"Successfully saved for {N} events."`
3. **train split with intent_train_augmented:** prints `"Successfully saved {N} augmented intent samples."`
4. **train split with success_intent:** prints `"Successfully saved success-intent graphs for {N} events."`
5. **test split with success_intent:** prints `"Successfully saved success-intent graphs for {N} events."`

With `--num-workers > 1`, each subprocess still runs the same 5 top-level steps sequentially, but it parallelizes matches inside that step. In that mode:

- the inner per-match progress bars are disabled
- the subprocess shows one outer progress bar: `compute matches`
- completed matches are reported as concise lines such as `DONE <match_id>: ...`
- skipped matches are reported as `SKIP <match_id>: ...`

`--num-workers auto` resolves to `max(1, min(6, cpu_count - 2))`, which is `6` on a 16-logical-core machine. `--worker-thread-limit 1` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` per worker to avoid oversubscription.

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

For `goal_distance` and `epv`, an `--extend-feature-run-id` run with `--refresh-target-family` uses labels-only generation for the existing return types and receiver modes with `--overwrite-labels`. The dedicated xT-only refresh updates xT columns across all existing variants, including carries, without loading tracking or graphs; run it separately from other operations. `--pass-height` uses labels-only generation to backfill `pass_max_ball_z` and `pass_high`; combine it with `--pass-height-threshold <meters>` for a per-run cutoff. The default destination is a derived run sharing unchanged artifacts. With `--in-place` (compatibility alias `--overwrite-feature-run`), the requested changes are applied to the existing run.

When adding or replacing model-mode receivers in a carry-enabled run, two carry labels-only stages follow the ordinary label stages. A receiver-only extension therefore has five stages; adding new return types at the same time has eight. The complete ordered workload is printed before generation, with continuous stage numbering and per-stage/total elapsed times. Match progress and ETA refer to the current stage; no overall ETA is predicted.

Carry label extensions retain the existing original/angle-only carry membership, action order, and pre-action/post-action graphs, including unavailable graphs. They do not derive carry segments or rebuild graph features. With graph validation enabled, retained carry graphs are checked once per match without rewriting or output readback. Missing or misaligned carry artifacts fail the extension. Full generation and adding carries for the first time still generate graphs.

Failed-pass predictions can be reused from ordinary model-mode resolved actions produced earlier in the same extension attempt. Reuse requires matching checkpoint/input provenance and stable event, launch, player and observation identities. Unavailable predictions retain their diagnostic reasons; unmatched inputs are inferred normally. This is not a general cache across runs. Reuse/inference/unavailability counts and separate carry graph-validation reports are recorded. Completion is published only after expected model-label coverage and action alignment have been audited.


Extension failures and manual repair:

- Match failures in `generation_reports/<split>_<variant>.json` retain the match ID and error summary, and include the original exception type and full worker traceback. Input-reader failures also identify the absolute `input_path` and `input_format`; JSON decoding errors include `json_error.line`, `column`, and character `position`, including when JSON is embedded in Parquet metadata. These details are captured before crossing the worker-process boundary. Console output remains a concise `SKIP` line with the input path when known. JSON parsing remains strict, and incomplete receiver extensions still fail their completion checks.
- Extension metadata includes `extension_attempt`: the operation, last phase, intended target settings, current command or graph file, known replacement count, original exception chain, secondary failures, and leftover temporary paths. Original feature settings stay advertised until final metadata publication succeeds.
- `artifact_state=unchanged` means no published artifact was modified. A failed in-place attempt in that state retains the original run's `completed` status, and the failure is recorded in its extension history. This can occur when an edge upgrade fails during validation or temporary writing, or its first replacement is blocked.
- `partial` means some replacements or earlier steps completed; `unknown` means a subprocess, destructive step, or interrupted replacement may have changed files. These failures mark the run `failed`. A process terminated abruptly may leave `in_progress`; its recorded file and count are only a snapshot, not proof of where it stopped or how many files actually changed. Ordinary run resolution, dataset loading, and prepared-cache consumption reject incomplete runs, including historical runs explicitly marked failed or running. Raw metadata loading remains available for inspection; legacy metadata without status remains supported.
- `complete` means the selected artifact operations finished. If metadata publication failed, inspect the files and intended settings before repairing the metadata. A summary-index refresh failure is reported separately and does not invalidate successful artifact publication.
- Repair is manual: stop the failed writer before removing reported temporary files. For an unchanged in-place run, resolve the disk/lock/input problem and retry after cleanup. For a partial or uncertain run, inspect all affected families and complete or regenerate the interrupted transformation, checking schemas, original feature columns, graph/label alignment, and expected match coverage. Changing `status` alone does not repair artifacts. For a failed derived copy, use a verified completed source to create a fresh derived run; there is no automatic resume, rollback, staged-file reuse, or automatic promotion of failed runs.
- Disk requirements are unchanged: numeric edge upgrades stage all updated graph files before replacing any originals, and derived extensions still copy the base run. Failure cleanup attempts every known temporary file; any files that cannot be removed are reported without hiding the original write error. Metadata JSON is published through temporary writing and replacement.

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
Action labels also carry pass-height columns. `pass_high` is `1` for passes whose maximum `ball_z` between the pass frame and receive frame is at least the run's `pass_height_threshold_meters` metadata value (default `2.0` metres), and `0` otherwise; `pass_max_ball_z` stores that maximum height.

Training and evaluation can select a different cutoff with `--pass-height-threshold` using the stored continuous height, without copying the feature run. See [pass-height target cutoffs](docs/pass_height_thresholds.md) for defaults, evaluation overrides, shared caches, and legacy compatibility.

### 4. Train the retained models

```powershell
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --only-success-intent
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family xt --return_type in_3 --intended-receiver-mode original --no-action-intent --no-pass-intent --no-success-intent --no-pass-success --no-failure-receiver --bundle-id model_bundle_20260414T123456_abcdef12
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type next_3 --intended-receiver-mode model --no-success-intent --velocity-angle-edge-features none
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family epv --return_type next_5 --intended-receiver-mode angle_only
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --return_type disc_0.9 --intended-receiver-mode model --no-action-intent --no-pass-intent --no-success-intent --no-outcome-scoring --no-outcome-conceding --pass-intent-model-id pass_intent/<model_run_id>
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --intended-receiver-mode angle_only --only-pass-height
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode angle_only --only-outcome-scoring --only-outcome-conceding
```

Inputs:

- `data/features/runs/<feature_run_id>/...`

Outputs:

- `saved/pass_intent/<model_run_id>/...`
- `saved/action_intent/<model_run_id>/...`
- `saved/pass_success/<model_run_id>/...`
- `saved/pass_height/<model_run_id>/...` when `pass_height` is enabled
- `saved/outcome_scoring/<model_run_id>/...`
- `saved/outcome_conceding/<model_run_id>/...`
- `saved/success_intent/<model_run_id>/...` when `success_intent` is enabled
- `saved/failure_receiver/<model_run_id>/...`
- `saved/bundles/<bundle_id>/metadata.json`

Behavior:

- the default wrapper run enables `action_intent`, `pass_intent`, `success_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding`; `pass_height` and `failure_receiver` are off by default and can be enabled explicitly
- `--action-intent` / `--no-action-intent`, `--pass-intent` / `--no-pass-intent`, `--success-intent` / `--no-success-intent`, `--pass-success` / `--no-pass-success`, `--pass-height` / `--no-pass-height`, `--outcome-scoring` / `--no-outcome-scoring`, `--outcome-conceding` / `--no-outcome-conceding`, and `--failure-receiver` / `--no-failure-receiver` let you rerun only the subset you need
- `--target-family` and `--return_type` are required only when `outcome_scoring` or `outcome_conceding` is enabled
- `--intended-receiver-mode` is required only when a mode-dependent model is enabled: `action_intent`, `pass_intent`, `pass_success`, `pass_height`, `outcome_scoring`, `outcome_conceding`, or `failure_receiver`
- `--only-success-intent` trains `success_intent` from the observed synced `receiver_id` on successful pass actions only
- `--only-success-intent` is mode-independent, does not accept `--intended-receiver-mode`, and cannot be combined with the per-model toggles
- `--only-pass-height` trains only the `pass_height` checkpoint. Its target is derived as `pass_max_ball_z >= T`, where `T` is the explicit `--pass-height-threshold` or the selected feature run's recorded cutoff.
- `pass_success` uses inverse propensity weighting by default via `--pass-success-ipw`; it uses a `pass_intent` checkpoint as its IPW model, either from the same wrapper run or from `--no-pass-intent --pass-intent-model-id pass_intent/<model_run_id>`
- `--no-pass-success-ipw` trains `pass_success` without inverse propensity weighting; use `--no-pass-success-ipw --no-pass-intent` when you want pass-success only and do not want to train or supply a propensity model
- an external `--pass-intent-model-id` applies only when pass-success IPW is enabled and must match the selected feature run, intended-receiver mode, graph schema, velocity edge-feature mode, and feature flags; its return type and target family are ignored because `pass_intent` only supplies IPW propensities for `pass_success`
- `--pass-height-ipw` enables `pass_intent` inverse-propensity weighting for `pass_height`; when `pass_intent` is not trained in the same wrapper call, supply `--pass-height-ipw-model-id pass_intent/<model_run_id>`
- outcome model loss uses the selected `--target-family` and `--return_type`; outcome F1/ROC AUC/Brier diagnostics use canonical `goal next_10` labels for comparability across target families and return types
- older feature runs without embedded `goal next_10` diagnostic columns can still be used if the selected run exposes `action_labels_next_10<intended_receiver_suffix>` or if you pass `--diagnostic-feature-run-id <feature_run_id>` pointing to compatible `next_10` labels
- reusing `--bundle-id` updates the existing bundle manifest by replacing only the retrained task ids and preserving untouched task ids
- training selects stored velocity-angle edges with `--velocity-angle-edge-features {all,no-poss,none}`; default: `none`
- wrapper batch-size defaults are `256` for `action_intent`, `pass_intent`, `success_intent`, and `failure_receiver`, and `512` for `pass_success`, `outcome_scoring`, and `outcome_conceding`; `--batch-size` overrides all defaults, and per-model `--<model>-batch-size` flags take highest precedence
- omitted selectors preserve the wrapper's existing profile: node motion, positions, keeper/possessor identity, ball height, player possessor geometry, goal geometry, offside, and goal-node context are enabled; possessor-relative velocity angles, goal-node possessor geometry, extended/pass-lane features, lane survival, and the `xy-only` preset are disabled. Direct `train.py` defaults differ; see [training input controls](#training-input-controls-and-automatic-exclusions).

In the intended-receiver workflow, `success_intent` is the learned intended-receiver checkpoint. It is trained independently of the `original` / `angle_only` / `model` intended-receiver modes. `failure_receiver` is a separate auxiliary model used for failed-pass / opponent-receiver handling.

### 5. Evaluate the retained models on the test set

```powershell
python scripts/evaluate_relevant_models.py --bundle-id <bundle_id>
python scripts/evaluate_relevant_models.py --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
python scripts/evaluate_relevant_models.py --action-intent-model-id action_intent/<model_run_id>
python scripts/evaluate_relevant_models.py --action-intent-model-id action_intent/<model_run_id> --pass-intent-model-id pass_intent/<model_run_id> --pass-success-model-id pass_success/<model_run_id> --outcome-scoring-model-id outcome_scoring/<model_run_id> --outcome-conceding-model-id outcome_conceding/<model_run_id>
```

Inputs:

- feature artifacts referenced by the selected checkpoint metadata
- checkpoint runs under `saved/<task>/<model_run_id>/...`

Outputs:

- metrics printed to stdout by `test.py`
- per-model JSON/CSV evaluation artifacts and the cross-run metrics summary
- `pass_success_height_metrics.csv` for pass success, containing counts, success prevalence, ROC-AUC, and Brier score for observed-high and observed-non-high passes

`test.py` uses the target configuration, graph schema, and diagnostic-label metadata saved inside each checkpoint. Pass `--diagnostic-feature-run-id <feature_run_id>` when evaluating older checkpoints whose selected feature run lacks canonical `goal next_10` diagnostics. Without a bundle, supply at least one model ID and the wrapper evaluates exactly the explicitly selected subset. With a bundle, it evaluates every supported task present in the bundle; explicit IDs add missing tasks or override matching bundle entries. This includes independently selected `success_intent` and `pass_height` checkpoints.

Test dataset construction is checkpoint-driven and uses the same shared feature configuration as training, including node-feature ablations, velocity and relative-speed edge modes, physical-xPass inputs, and pc-xPass lane-survival inputs. Evaluation validates the reconstructed node/edge widths against the checkpoint. Physical-xPass cache metadata and lane-survival mode/fingerprint must also match; missing or incompatible caches stop evaluation with an error instead of silently changing the model inputs.

Evaluation loads each checkpoint's `best_weights.pt` (selected during training by validation loss) and evaluates it on the held-out 2024/25 test matches. It uses the same evaluation routine as validation, but the test metrics are not training-set or validation-set metrics. Use validation loss for checkpoint selection and early stopping; use this held-out test evaluation for final performance reporting. Avoid repeatedly changing the model based on test results, since that would turn the test set into another validation set.

Pass-success evaluation is stratified by the observed binary `pass_high` label by default. Use `--no-observed-pass-height-stratification` to disable this analysis. Feature runs without genuine pass-height label columns must be backfilled before stratification; they are rejected rather than interpreting padded labels as non-high passes.

Use `--evaluate-xpass --top-pass N` (or `--evaluate-xpass --xpass-version <max|topN>`) to compare the learning predictor with receiver-specific probabilities already stored in the read-only pc-xPass cache. Use `--evaluate-combined-success` with an explicit `--xpass-weight`; v4 additionally requires explicit `--discount`, `--v4-power`, and `--v4-zero`. Missing match, action, receiver, or blend-input cache data aborts evaluation and is never generated or skipped. Comparable pooled and observed-height results are written to `pass_success_predictor_metrics.csv`.

Outcome artifacts also include pooled 95% match-bootstrap confidence intervals by default (2,000 resamples, seed 42). Both evaluation CLIs accept `--outcome-bootstrap-resamples` and `--outcome-bootstrap-seed`; set resamples to `0` to disable. See [outcome bootstrap documentation](docs/outcome_bootstrap.md) for metrics, metadata, and interpretation.

For `outcome_scoring` and `outcome_conceding`, the reported quantities have two target definitions:

- `ce_loss` is computed against the selected outcome target family and `--return_type`: binary goal, xG, xT, goal distance, or EPV.
- `roc_auc`, `brier`, and `log_loss` are computed against the canonical binary `next_10` goal diagnostics: `scores_goal_next10` for scoring and `concedes_goal_next10` for conceding. A non-zero diagnostic label is positive. Precision, recall, and F1 are omitted by default; pass `--f1-outcome-threshold <0..1>` to report them at an explicitly prespecified operating point. Classification uses `prediction > threshold`.

Consequently, do not compare outcome-model `ce_loss` across target families. On the same held-out matches and canonical diagnostic labels, `roc_auc` is the clearest cross-family comparison because it measures ranking of actual next-10-action goals/concessions without a threshold. Threshold-dependent metrics are reported only when `--f1-outcome-threshold` is supplied because output scales can differ by target family. `brier` uses the same binary event ground truth, but its probability-calibration interpretation is strongest for models whose output is itself a goal probability; for xT, goal-distance, and EPV models it is a goal-event proxy diagnostic.

Dataset-feature parity does not imply loss-weighting parity. Test evaluation currently does not reconstruct training/validation class weights or inverse-propensity weights, so a weighted training/validation `ce_loss` is not directly comparable with the unweighted test `ce_loss`.

### 6. Export per-match component predictions

```powershell
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id>
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --use-physical-xpass
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --run-id component_20260414T123456_abcdef12
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --pass-height-model-id pass_height/<model_run_id>
python scripts/run_relevant_models.py --split test --action-intent-model-id action_intent/<model_run_id> --pass-intent-model-id pass_intent/<model_run_id> --pass-success-model-id pass_success/<model_run_id> --outcome-scoring-model-id outcome_scoring/<model_run_id> --outcome-conceding-model-id outcome_conceding/<model_run_id>
```

Inputs:

- feature artifacts from the selected runtime feature run; compatible mixed-source checkpoints are allowed. With physical xPass and caching enabled, the default is the selected cache's feature run; otherwise selection uses compatible model/bundle source runs. `--feature-run-id` overrides automatic selection.
- checkpoint runs under `saved/<task>/<model_run_id>/...`
- `data/event_synced/<match_id>.csv` indirectly via the generated feature artifacts and resolved-action tables

Each invocation creates a new component run under `data/component_runs/sportec/<component_run_id>/` and updates `data/component_runs/sportec/latest.json`.
When a `success_intent` checkpoint is supplied explicitly or present in the selected bundle, each processed match also gets `success_intent.parquet`.
When a `pass_height` checkpoint is supplied explicitly or present in the selected bundle, each processed match also gets `pass_height.parquet`.

Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--feature-run-id <feature_run_id>` to override the automatically selected runtime feature run
- `--run-id <component_run_id>` to pin the created component run id instead of auto-generating one
- `--match-id DFL-MAT-...` to restrict inference to one or more matches
- `--success-intent-model-id success_intent/<model_run_id>` to additionally export `success_intent.parquet` for each processed match
- `--pass-height-model-id pass_height/<model_run_id>` to additionally export `pass_height.parquet` for each processed match
- `--use-physical-xpass` to blend `pass_success` with cached runtime physical xPass; generate `data/runtime_physical_xpass/sportec` first
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` to select the cached xPass metric for the blend; default: `top10`
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` to select the xPass/model blend weight; default: `v3`
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.

### 7. Visualize action components

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --use-physical-xpass --show-physical-xpass
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --show-pass-height
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --action-id 456 --bundle-id <bundle_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --run-id visualization_20260414T123456_abcdef12
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --only-outcome-scoring
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --success-intent-model-id success_intent/<model_run_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --only-success-intent --success-intent-model-id success_intent/<model_run_id> --compare-intended-receiver --pass-endpoint true
python scripts/visualize_action_components.py --match-id DFL-MAT-... --player-id DFL-OBJ-... --spadl-type pass --success false --bundle-id <bundle_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --spadl-type pass --start-x-gt 50 --end-x-lt 105 --bundle-id <bundle_id>
python scripts/visualize_action_components.py --match-id DFL-MAT-... --spadl-type pass --first 100 --bundle-id <bundle_id>
```

`--action-id` refers to the `action_id` column in `data/event_synced/<match_id>.csv`.
You can provide one or more explicit `--action-id`, `--row-index`, or `--original-event-id` values, or omit them and select rows only with CSV filters.

Inputs:

- feature artifacts from the selected runtime feature run; compatible mixed-source checkpoints are allowed, and the runtime feature run is chosen automatically from the selected source runs unless `--feature-run-id` is supplied
- checkpoint runs under `saved/<task>/<model_run_id>/...`
- `data/event_synced/<match_id>.csv`

Outputs:

- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/action_intent.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_intent.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_success.png`
- optionally `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_height.png` when `--show-pass-height` or `--only-pass-height` is selected and a model is available
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_scoring_success.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_scoring_failure.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_conceding_success.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_conceding_failure.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_score.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_failure.png`
- `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_success.png`
- optionally `success_intent.png` when a `success_intent` checkpoint is supplied
- `data/visualizations/sportec/<visualization_run_id>/metadata.json`

Pass images mark the actual endpoint with a black x by default. Use `--pass-endpoint false` to hide it. The endpoint uses a finite action `end_x/end_y` pair, falling back to tracked ball coordinates at `receive_frame_id`.

Add `--compare-intended-receiver` to include the success-intent image with three intended-receiver rings: teal for `original` (inner), orange for `angle_only` (middle), and purple for `model` (outer). Fixed ring sizes keep all three visible when they agree. The existing probability annotations remain visible. Heuristic choices are recomputed for selected failed passes; the model ring uses the current displayed prediction, independently of saved receiver labels. Successful passes use the actual receiver for all three modes. `--intended-receiver-mode` continues to select runtime labels.

Missing required action data is logged in `skipped_actions` in run metadata, and processing continues. If every action is skipped, metadata is still written before the command reports failure.


Useful options:

- `--bundle-id <bundle_id>` to use the model set recorded by one training wrapper run
- `--feature-run-id <feature_run_id>` to override the automatically selected runtime feature run
- `--run-id <visualization_run_id>` to pin the created visualization run id instead of auto-generating one
- `--success-intent` adds the success-intent image alongside selected component groups; `--only-success-intent` selects it alone and `--no-success-intent` suppresses it. These replace the old intended-recipient options, which are no longer accepted. A success-intent checkpoint must be selected explicitly or through the bundle.
- `--compare-intended-receiver` adds receiver rings only to the success-intent image and automatically includes that image, even alongside another `--only-*` selection. It conflicts with `--no-success-intent` and also works in checkpoint/mask comparison panels.
- `--pass-endpoint true/false` controls the actual pass-end marker on every pass image; default: `true`.
- `--only-action-intent`, `--only-pass-intent`, `--only-pass-success`, `--only-outcome-scoring`, `--only-outcome-conceding`, `--only-pass-score`, and `--only-success-intent` to render only selected component groups; repeated `--only-*` flags are additive
- `--show-pass-height` to include the optional pass-height component when a `pass_height` checkpoint is selected
- `--only-action-intent`, `--only-pass-intent`, `--only-pass-success`, `--only-pass-height`, `--only-outcome-scoring`, `--only-outcome-conceding`, `--only-pass-score`, and `--only-success-intent` to render only selected component groups; repeated `--only-*` flags are additive
- `--no-action-intent`, `--no-pass-intent`, `--no-pass-success`, `--no-pass-height`, `--no-outcome-scoring`, `--no-outcome-conceding`, `--no-pass-score`, and `--no-success-intent` to suppress selected component groups; `--no-*` takes precedence over `--only-*`
- `outcome_failure` and `outcome_success` are rendered by default across Sportec, Benchmark, SkillCorner, Hawkeye, and the combined Hawkeye inference/visualization command, in each command's supported output formats. They subtract conceding from scoring for the corresponding failure/success case, using three-decimal numerical annotations (including negatives and zero), without value-based colour coding.
- `--no-outcome-failure` and `--no-outcome-success` remove the new plots; `--only-outcome-failure` and `--only-outcome-success` select them alone or together. There are no `--show-outcome-*` flags. Existing `--only-*` selections do not implicitly add the new plots.
- `--only-pass-score` automatically loads its required pass-success, scoring, and conceding inputs without displaying their plots. Combined outcome plots likewise load their inputs independently; `--no-*` suppresses plots, not dependency computation.
- Derived values align by player/action/frame identifiers. Missing operands remain missing; an unavailable source component produces a clear error. Derived plots use the existing output directories and metadata, with component names `outcome_failure` and `outcome_success`; no additional model or saved component table is required.
- `--show-trajectories` to render dashed recent player trajectories
- `--use-physical-xpass` to blend the pass-success inference output with cached runtime physical xPass
- `--show-physical-xpass` to render the cached physical xPass metric itself
- `--only-physical-xpass` to render cached physical xPass alone, without model IDs, checkpoint loading, or inference. Combine it with other `--only-*` selections to add outputs and their dependencies; `--show-*` additions still work.
- `--show-model-pass-success` to render **Model Pass Success** separately: the selected checkpoint's probability after normal offside handling and before inference-time physical xPass blending. Physical inputs used inside a checkpoint remain part of its prediction. This flag works independently of `--show-physical-xpass`, with or without blending, and adds its plot even when `--only-*` or `--no-pass-success` hides ordinary pass success.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` to select the cached xPass metric for both blending and physical xPass rendering; default: `top10`
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` to select the xPass/model blend weight; default: `v3`
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--action-id <id>`, `--row-index <index>`, and `--original-event-id <sportec_event_id>` can each be repeated to visualize multiple selected actions from the same match
- `--player-id`, `--object-id`, `--advanced-position`, `--team-id`, `--spadl-type`, `--success`, `--offside`, and `--next-type` filter rows from `data/event_synced/<match_id>.csv`; repeated values for one column are OR alternatives
- `--start-x-lt`, `--start-x-gt`, `--start-y-lt`, `--start-y-gt`, `--end-x-lt`, `--end-x-gt`, `--end-y-lt`, and `--end-y-gt` filter coordinate columns with strict lower-than or greater-than comparisons
- all CSV filters are off by default, enabled filters are combined with AND logic, and unprocessed/problematic rows are skipped with warnings instead of stopping the whole batch

All five visualization commands support `--only-physical-xpass` in their existing output formats. Standard Hawkeye, Benchmark, and SkillCorner can render it without a component run: provide an explicit `--pc-xpass-id` or `--physical-cache-dir` and an explicit `--xpass-version` or `--top-pass`. These saved-data commands use pc-xPass only and never select the latest cache implicitly. Explicit component runs retain their scene selection and cache/metric inheritance. Hawkeye `loc`/`freeze` still requires a matching component export for row selection and scene geometry. Sportec retains its feature/resolved-action requirements, preferring the selected pc-xPass cache's feature run when no `--feature-run-id` is given. Inference commands retain their existing cache and metric defaults. Caches must already exist; `--no-physical-cache` is rejected for physical-only output. Hawkeye `--only-tracking` conflicts with this selector. Checkpoint/mask comparisons require an additional model-backed selected output; saved-run comparisons can select physical xPass alone.

```powershell
# Render cached physical xPass without checkpoint IDs.
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --only-physical-xpass --pc-xpass --pc-xpass-id <cache_id>
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --only-physical-xpass --physical-cache-dir <cache_dir>

# Render directly from tracking/scene data, without a saved component run.
python scripts/visualize_hawkeye.py --situation-id <hawkeye_id> --only-physical-xpass --pc-xpass-id <cache_id> --xpass-version top25
python scripts/visualize_benchmark.py --modification 1 --only-physical-xpass --pc-xpass-id <cache_id> --top-pass 10
python scripts/visualize_skillcorner.py --match-id <match_id> --index 1 --only-physical-xpass --pc-xpass-id <cache_id> --xpass-version top25
```

The separate model view is available in all five visualization commands, including checkpoint/mask comparisons, Hawkeye `loc`/`freeze`, and supported PNG/GIF/MP4 exports. Its files use the name `model_pass_success` with the usual frame suffixes. For example:

```powershell
# Display blended pass success, cached physical xPass, and checkpoint pass success together.
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --use-physical-xpass --show-physical-xpass --show-model-pass-success

# Display only checkpoint pass success, with ordinary pass success hidden.
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --pass-success-model-id pass_success/<model_id> --only-pass-success --no-pass-success --show-model-pass-success

# Render the separate checkpoint result from a newly generated saved component run.
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --show-model-pass-success
```

Component producers now save checkpoint probabilities whenever they run pass-success inference, including without blending. Older saved component runs must be regenerated before requesting this view; a blended value alone cannot recover the checkpoint probability. Missing predictions remain unavailable, and ordinary pass score continues to use the final pass-success result. A checkpoint that needs physical inputs internally still requires those inputs for its separate model view. Metadata records the definition, structural availability, and generated output paths.

All five visualization commands also support `--compare-pass-success` to add one PNG containing the existing **Model Pass Success**, **Physical xPass**, and final **Pass Success** images, stacked vertically in that order. The images retain their titles, annotations, and resolution. Files use `compare_pass_success` with the command's usual frame suffixes. Benchmark stacks its three paired-state images, so the result contains six pitch panels.

Use `--only-compare-pass-success` to select just this combined output. As with other `--only-*` options, additional explicit component selections can add outputs. Neither `--show-model-pass-success` nor `--show-physical-xpass` is required: the inputs are rendered automatically, and standalone files are saved only when selected through the existing options. `--no-pass-success` hides its standalone output without removing it from the combined image.

This mode requires an existing pc-xPass cache, selected explicitly with `--pc-xpass-id` or `--physical-cache-dir`, or inherited from a saved component run together with its metric. It never falls back to the latest cache. Existing metric selection rules apply. Saved component runs must contain `model_pass_success`. The mode preserves the existing final prediction and blending settings; when blending is disabled, the first and third views can show the same probabilities. It supports PNG only and rejects `--compare`, numbered inference masks, and Hawkeye `--only-tracking`; ordinary unnumbered inference masks remain supported.

```powershell
# Add the combined image while retaining normally selected component outputs.
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --bundle-id <bundle_id> --pc-xpass-id <cache_id> --pc-xpass --use-physical-xpass --compare-pass-success

# Save only the combined image, inheriting the saved run's pc-xPass cache and metric.
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --only-compare-pass-success

# Select a pc-xPass cache and metric explicitly for a saved run.
python scripts/visualize_skillcorner.py --match-id <match_id> --index 1 --component-run-id <component_run_id> --pc-xpass-id <cache_id> --xpass-version top25 --only-compare-pass-success
```

### 8. Run frame-level inference on HawkEye data

```powershell
python scripts/run_hawkeye.py --bundle-id <bundle_id>
python scripts/run_hawkeye.py --bundle-id <bundle_id> --time-norm 0 --time-norm 1
python scripts/run_hawkeye.py --bundle-id <bundle_id> --use-physical-xpass
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
- `--use-physical-xpass` to blend `pass_success` with cached runtime physical xPass; generate `data/runtime_physical_xpass/hawkeye` first
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` to select the cached xPass metric for the blend; default: `top10`
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` to select the xPass/model blend weight; default: `v3`
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.

To visualize one HawkEye situation as MP4s:

```powershell
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id>
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --show-physical-xpass
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --coach-ratings --selections
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --selections --selection-format percentages
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id_1> --situation-id <hawkeye_id_2>
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --run-id hawkeye_visualization_20260414T123456_abcdef12
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --only-pass-score
```

`scripts/visualize_hawkeye.py` reads the probabilities from `scripts/run_hawkeye.py` outputs and rebuilds only the raw HawkEye geometry for rendering. Each invocation writes to `data/visualizations/hawkeye/<visualization_run_id>/` and records the source component run in `metadata.json`. If you want the old direct-inference behavior, use:

```powershell
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id>
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --use-physical-xpass --show-physical-xpass
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --coach-ratings --selections
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --selections --selection-format percentages
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id_1> --situation-id <hawkeye_id_2> --bundle-id <bundle_id>
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --run-id hawkeye_visualization_20260414T123456_abcdef12
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --only-outcome-scoring
```

#### HawkEye visualization modes

`scripts/visualize_hawkeye.py --mode` determines which component-run type is rendered:

- `standard` (default) visualizes a regular `scripts/run_hawkeye.py` component run at the original tracking positions. It supports PNG, GIF, and MP4 output. For PNG, frames are selected relative to `BallReceipt` (default `--time-norm 0`); the component-run metadata determines whether BallReceipt was frozen.
- `loc` visualizes a location-component run produced with `scripts/run_hawkeye_loc.py --mode loc`. For each selected location row, it moves the possessor/tracking geometry by `PositionX / 100` in x and `-PositionY / 100` in y (the stored coordinates are centimetres), reproducing the relocated hypothetical position used during inference. It exports one PNG per component at that row's `resolved_time_norm`.
- `freeze` visualizes a location-component run produced with `scripts/run_hawkeye_loc.py --mode freeze`. It uses the same frozen-BallReceipt construction but applies no `PositionX`/`PositionY` displacement, preserving the original player location. It likewise exports PNG only at the row's `resolved_time_norm`.

`loc` and `freeze` automatically use pc-xPass caches and require a matching location-component run: its `metadata.json` must have `inference_mode` equal to the requested visualization mode.

### 9. Run benchmark inference on local benchmark data

```powershell
python scripts/run_benchmark.py --bundle-id <bundle_id>
python scripts/run_benchmark.py --bundle-id <bundle_id> --use-physical-xpass
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
- `--use-physical-xpass` to blend `pass_success` with cached runtime physical xPass; generate `data/runtime_physical_xpass/benchmark` first
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` to select the cached xPass metric for the blend; default: `top10`
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` to select the xPass/model blend weight; default: `v3`
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.

To visualize one benchmark state as PNGs:

```powershell
python scripts/visualize_benchmark.py --modification 1 --game-state 1
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id> --show-physical-xpass
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id>
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id> --run-id benchmark_visualization_20260420T123456_abcdef12
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id> --no-pass-intent --no-pass-score
```

Benchmark visualizations are written under `data/visualizations/benchmark/<visualization_run_id>/` with `metadata.json` recording the source benchmark component run.

### 10. Run frame-level inference on SkillCorner data

```powershell
python scripts/run_skillcorner.py --bundle-id <bundle_id>
python scripts/run_skillcorner.py --bundle-id <bundle_id> --use-physical-xpass
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
- `--use-physical-xpass` to blend `pass_success` with cached runtime physical xPass; generate `data/runtime_physical_xpass/skillcorner` first
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` to select the cached xPass metric for the blend; default: `top10`
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` to select the xPass/model blend weight; default: `v3`
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.

To visualize one SkillCorner possession as MP4s:

```powershell
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index>
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id> --show-physical-xpass
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index_1> --index <player_possession_index_2>
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id>
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id> --run-id skillcorner_visualization_20260414T123456_abcdef12
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id> --only-pass-intent
```

SkillCorner visualizations are written under `data/visualizations/skillcorner/<visualization_run_id>/` with `metadata.json` recording the source SkillCorner component run.

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

`--return_type` accepts nine resolved forms overall:

- `disc_<gamma>` uses discounted returns
- `disc_<gamma>_skip1` uses discounted returns but skips the first rated future non-shot action
- `disc_max_<gamma>` uses the maximum discounted future eligible value and is supported only for `xt`, `goal_distance`, and `epv`
- `disc_max_<gamma>_skip1` uses the same discounted max scan, but skips the first rated future non-shot action
- `disc_poly_max_<b>_<z>` uses the maximum polynomially weighted future eligible value and is supported only for `xt` and `goal_distance`
- `disc_poly_max_<b>_<z>_spstop` uses the same polynomial maximum and additionally stops at every set-piece restart
- `next_<N>` uses non-discounted lookahead returns
- `next_<N>_skip1` uses non-discounted lookahead returns but skips the first rated future non-shot action
- `in_<N>` uses the state at the Nth future eligible action and is supported only for `xt`, `goal_distance`, and `epv`

Example:

```powershell
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123450_abcdef12 --model gat --return_type next_10 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123456_abcdef12 --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123510_cdef1234 --model gat --use_xt --return_type disc_0.9 ...
python train.py --task outcome_scoring --run-id outcome_scoring_20260414T123515_cdef1235 --model gat --use_xt --return_type disc_poly_max_0.05_2_spstop ...
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
  - `disc_<gamma>` combines future eligible xT values with xG-style probability-complement aggregation; the first considered post-action/reception row has exponent `0`, and later rows use raw event offset from that anchor
  - `disc_<gamma>_skip1` skips the first considered future non-shot action and shifts the discount anchor to the next contributing row
  - `disc_max_<gamma>` uses the maximum `gamma^k * xT` over future eligible actions separately for teammate and opponent actions, where the first contributing eligible action has `k=0`
  - `disc_max_<gamma>_skip1` skips the first future eligible non-shot action without consuming discount rank
  - `disc_poly_max_<b>_<z>` uses the maximum `max(0, 1 - b*k^z) * xT`; the first future eligible action has `k=0`, only eligible actions advance the shared chronological rank, and `_skip1` is not supported
  - `disc_poly_max_<b>_<z>_spstop` additionally stops before throw-ins, corners, free kicks, penalties, and goal kicks; the restart and later actions cannot contribute, while ordinary eligible actions before it remain candidates; a scan whose analyzed event is itself a set piece starts inside the new situation and continues normally
  - discounted probability scans clip contributing values to `[0.0, 1.0]`
- goal_distance:
  - raw `goal_distance` is a bounded proximity-to-goal score in `[0.0, 1.0]`, using `1.0 * (1 - raw_distance / sqrt(105^2 + 34^2))`
  - `next_<N>` uses the maximum future teammate/opponent goal-distance value over the next `N` eligible `pass` / `cross` / `shot` actions
  - `next_<N>_skip1` uses the same eligible-action window, but skips the first rated future action when that action is not a shot
  - `in_<N>` uses the goal-distance value at the Nth future eligible `pass` / `cross` / `shot` action, unless an earlier eligible `shot` occurs first; only one of `scores_goal_distance` / `concedes_goal_distance` is non-zero
  - `disc_<gamma>` combines future eligible goal-distance values with xG-style probability-complement aggregation; the first considered post-action/reception row has exponent `0`, and later rows use raw event offset from that anchor
  - `disc_<gamma>_skip1` skips the first considered future non-shot action and shifts the discount anchor to the next contributing row
  - `disc_max_<gamma>` uses the maximum `gamma^k * goal_distance` over future eligible actions separately for teammate and opponent actions, where the first contributing eligible action has `k=0`
  - `disc_max_<gamma>_skip1` skips the first future eligible non-shot action without consuming discount rank
  - `disc_poly_max_<b>_<z>` and its `_spstop` form use the same eligible-action ranking, polynomial weighting, clamping, and set-piece boundaries described for xT
  - discounted probability scans clip contributing values to `[0.0, 1.0]`
- EPV:
  - `next_<N>` uses the maximum future teammate/opponent EPV over the next `N` eligible `pass` / `cross` / `shot` actions
  - `next_<N>_skip1` uses the same eligible-action window, but skips the first rated future action when that action is not a shot
  - `in_<N>` uses the EPV value at the Nth future eligible `pass` / `cross` / `shot` action, unless an earlier eligible `shot` occurs first; only one of `scores_epv` / `concedes_epv` is non-zero
  - `disc_<gamma>` combines future eligible EPV values with xG-style probability-complement aggregation; the first considered post-action/reception row has exponent `0`, and later rows use raw event offset from that anchor
  - `disc_<gamma>_skip1` skips the first considered future non-shot action and shifts the discount anchor to the next contributing row
  - `disc_max_<gamma>` uses the maximum `gamma^k * EPV` over future eligible actions separately for teammate and opponent actions, where the first contributing eligible action has `k=0`
  - `disc_max_<gamma>_skip1` skips the first future eligible non-shot action without consuming discount rank
  - discounted probability scans clip contributing values to `[0.0, 1.0]`

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

- retrain only `success_intent`: `python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --only-success-intent`
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

Both training entry points accept the same explicit retained-input selectors:

- Node features/groups: `--vx`, `--vy`, `--speed`, `--accel`, `--pos-node-features`, `--vel-node-features`, `--goal-features`, `--poss-geometry`, `--poss-velocity-angles`, `--offside`, `--extended-node-features`, `--pass-lane-features`, and `--lane-survival`, each followed by `all`, `attacking`, `defending`, `no-poss`, or `none`.
- Boolean inputs: `--possessor-id`, `--keeper-id`, `--ball-z`, and `--goal-poss-geometry`, each followed by `true` or `false`.
- Edge groups: `--velocity-angle-edge-features`, `--relative-speed-edge-features`, `--position-direction-edge-features`, `--relative-velocity-edge-features`, and `--separation-rate-edge-features`, each followed by `all`, `no-poss`, or `none`.
- Graph/edge policy: `--goal-nodes true/false` and `--goal-edge-movement true/false`.

Omitting selectors keeps the existing defaults for the chosen entry point. The
wrapper and direct training profiles differ; they are listed in
[training input controls](#training-input-controls-and-automatic-exclusions).
Existing flags remain compatibility aliases. `scripts/main.py` retains its older
boolean profile interface; use the training wrapper directly for explicit
selectors, then resume the pipeline with `--skip-train --bundle-id <bundle_id>`.
See [feature selections](docs/feature_selections.md) for the complete contract.

### Scaling and directed interaction edges

Fresh feature generation and fresh model training enable positional direction,
relative velocity, and separation rate. Velocity-angle and relative-speed edge
inputs remain disabled by default in training. Each training group accepts one
retention value:

| Group | Training flag | Values | Fresh training default |
| --- | --- | --- | --- |
| Velocity angle | `--velocity-angle-edge-features` | `all`, `no-poss`, `none` | `none` |
| Relative speed | `--relative-speed-edge-features` | `all`, `no-poss`, `none` | `none` |
| Positional direction | `--position-direction-edge-features` | `all`, `no-poss`, `none` | `all` |
| Relative velocity | `--relative-velocity-edge-features` | `all`, `no-poss`, `none` | `all` |
| Separation rate | `--separation-rate-edge-features` | `all`, `no-poss`, `none` | `all` |

These selectors are available on `train.py` and `scripts/train_relevant_models.py`.
`all` uses every stored value in the group; `no-poss` zeros both incoming and
outgoing possessor-incident values. `none` excludes the group from the selected
training edge schema; runtime inference masks instead preserve checkpoint width
and zero the corresponding inputs. Relative-speed inputs require velocity-angle
inputs. Edge selectors do not accept `attacking` or `defending`.

Feature generation keeps its existing enable/disable switches on
`datatools/graph_feature.py` and `scripts/generate_relevant_features.py`, such as
`--position-direction-edge-features` / `--no-position-direction-edge-features`.
Training exclusions operate on learned inputs, allowing stored artifacts to be
reused. Generation switches do not take the training retention values.

For a directed edge from source i to destination j, define `r = position_j - position_i`, `d = ||r||`, and `u = velocity_j - velocity_i`. Positional direction stores `(r_x/d, r_y/d)`, relative velocity stores `(u_x, u_y)` in m/s, and separation rate stores `dot(r/d, u)` in m/s. Negative separation rate means approaching; positive means moving apart. Reversing the edge negates direction and relative velocity while preserving separation rate. Calculations use the existing graph coordinate frame. Goals are stationary points. At distances no greater than `1e-6` m, direction and separation rate are zero; relative velocity is retained. All five new self-edge attributes are zero.

`--goal-edge-movement false` zeros relative velocity, separation rate, relative speed, and both velocity-angle attributes on **every edge involving a goal**, including player-to-goal, goal-to-player, and goal self-edges. It preserves edges, distance, positional direction, and the teammate indicator. The default is `true`. This is independent of `--goal-poss-geometry`, which controls goal-node inputs. Goal and possessor exclusions combine. Node selectors affect only their named columns; the runtime `motion` mask also covers velocity-derived edge inputs.

`--scaling` enables the saved `physical_v1` input profile; it defaults off. Scaling divides copies of learned inputs without centering or clipping:

| Quantity | Divisor |
|---|---:|
| x coordinates and x displacements | Pitch length (105 m) |
| y coordinates and y displacements | Pitch width (68 m) |
| Scalar distances, including polar radius | 100 m |
| Speed, velocity components, relative speed/velocity, separation rate | 10 m/s |
| Acceleration | 10 m/s^2 |
| Ball height | 10 m |
| Opponent/interceptor/blocker counts | 11 |
| Binary indicators, probabilities, unit directions, sine/cosine, angles in radians | 1 |

Scaling applies at encoder, decoder skip-input, and destination-input boundaries, including grid decoding. Already normalized destination coordinates are not divided again. Zero masks remain zero, signs are preserved, and values may exceed one. Stored graphs, prepared tensors, graph-distance thresholds, spatial processing, physical xPass calculations, physical logits, and labels retain their original units or values. Scaling alone requires no graph regeneration.

Models save feature modes, goal-edge movement, scaling profile, and an explicit ordered edge schema. Evaluation and inference use those settings automatically; auxiliary/IPW models use their own settings. Historical checkpoints missing the new settings keep their original feature layout and unscaled behavior. Resume, continuation, and refit inherit source settings and retain existing compatibility and fingerprint checks. Legacy velocity-angle calculations, including their stationary-direction and unsigned-angle conventions, are unchanged.

To add the new groups to a completed legacy feature run using its preserved raw coordinates and velocities:

```powershell
python scripts/generate_relevant_features.py --extend-feature-run-id <source_run_id> --run-id <upgraded_run_id> --extend-interaction-edge-features
```

Use the source run's split selector when it differs from the default. This creates a derived run by default, preserving labels, node identities, graph structure, existing edge attributes, and null graph placeholders. Use `--in-place` (compatibility alias `--overwrite-feature-run`) to apply the upgrade to the existing run. Upgrade covers action, post-action, intent-training, success-intent, augmented, and available carry graph families. Required raw coordinates and velocities must be present and finite; otherwise regenerate the feature run. Training reports missing requested attributes instead of substituting zeros. Prepared caches invalidate through the normal schema/options/fingerprint mechanism.

### Training input controls and automatic exclusions

Training flags describe **where inputs remain available**. Both
`train.py` and `scripts/train_relevant_models.py` accept these node selections:

| Value | Retain the selected columns on |
| --- | --- |
| `all` | Every node |
| `attacking` | Players on the possessing team, including the possessor |
| `defending` | Players on the opposing team |
| `no-poss` | Every node except the possessor |
| `none` | No nodes |

Both goalkeepers count as players. Team selections zero the selected columns on
synthetic goal nodes. `true` and `false` are accepted aliases for `all` and `none`;
boolean-only controls use `true`/`false`. Node masks preserve tensor width, source
artifacts, and the original coordinates/identities needed by graph processing,
physical inputs, and destination features. Independently controlled spatial and
motion-derived inputs remain available unless selected separately.

Groups provide defaults; narrower groups and individual features override them,
independent of argument order. For example, `--vel-node-features none --vx defending
--vy defending` retains only defenders' node `vx`/`vy`, with scalar speed and
acceleration zero everywhere. `--vel-node-features defending` instead retains all
four motion columns for defenders. `--vel-node-features no-poss --accel none`
retains motion for everyone except the possessor, with acceleration disabled
throughout. Contradictory explicit assignments to the same flag are rejected.
`pass-lane-features` is a narrower group within `extended-node-features`.

Omitting selectors preserves each entry point's existing profile. The following
defaults describe the effective inherited inputs; they are not automatically
supplied selector arguments. Saved checkpoint settings determine evaluation,
inference, resume, continuation, and source refit. Each auxiliary/IPW model uses
its own saved settings.

| Control | Named learned inputs | Direct default | Training wrapper default |
| --- | --- | --- | --- |
| `--pos-node-features SELECTION` | `x`, `y` | `all` | `all` |
| `--vel-node-features SELECTION` | `vx`, `vy`, `speed`, `accel` | `all`* | `all` |
| `--vx SELECTION`, `--vy SELECTION` | Individual velocity component | `all`* | `all` |
| `--speed SELECTION`, `--accel SELECTION` | Scalar speed or acceleration | `all`* | `all` |
| `--possessor-id true/false` | Only `is_possessor` | `false`* | `true` |
| `--keeper-id true/false` | Only `is_keeper` | `false` | `true` |
| `--ball-z true/false` | Ball height | `false` | `true` |
| `--poss-geometry SELECTION` | Possessor distance and normalized direction | `none`* | `all` on players |
| `--poss-velocity-angles SELECTION` | Cosine/sine of velocity angle relative to possessor | `none` | `none` |
| `--goal-features SELECTION` | Goal distance and normalized direction | `all` | `all` |
| `--goal-poss-geometry true/false` | Allow possessor distance/direction on goal nodes | `false` | `false` |
| `--offside SELECTION` | Appended offside indicator, when present | `all` | `all` |
| `--extended-node-features SELECTION` | Six handcrafted features | `none`** | `none`** |
| `--pass-lane-features SELECTION` | Nearest-opponent-to-pass and potential-interceptor inputs | `none` | `none` |
| `--lane-survival SELECTION` | Cached lane-survival input on supported node tasks | `none` | `none` |

\* Direct training retains the legacy possessor-awareness profile: with no
selectors, identity and possessor-relative geometry are disabled. Its legacy
possessor-only motion switch defaults off but applies only when legacy possessor
awareness is enabled. Explicit `--possessor-id` affects only identity; explicit
feature/group selectors override the selected columns of the inherited profile.

\*\* `success_intent` uses a separate pass-trajectory block, which retains its
historical behavior and is not the six-feature extended group. `--speed none`
does not disable its separate pass-trajectory speed input.

`--goal-poss-geometry true` allows `poss_dist`, `poss_dx_norm`, and
`poss_dy_norm` on goal nodes only where the possessor-geometry selection also
retains them. Its fresh-run default is `false`; older checkpoints missing the
setting preserve their historical enabled behavior. Node-velocity selections do
not disable possessor-relative velocity angles or velocity-derived edge inputs.

For example:

```powershell
# Keep defenders' vx/vy, with speed and acceleration zero for every node.
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --only-success-intent --vel-node-features none --vx defending --vy defending

# Remove scalar speed only; preserve the other default inputs.
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --only-success-intent --speed none
```

Lane-survival selection and computation are separate. Use
`--lane-survival defending --lane-survival-mode top_pass_10` to keep that cached
input for defenders. Computation defaults to `max` when enabled; `top_N` and
`top_pass_N` require the corresponding metric in the selected pc-xPass cache.

#### Legacy feature-flag compatibility

Existing model IDs, checkpoint weights, saved outputs, and older command lines
remain usable without retraining or rewriting. Legacy switches preserve their
original defaults, cumulative masks, and coupled effects. Prefer explicit
selectors for new configurations:

| Legacy training spelling | Preferred spelling for new configurations |
| --- | --- |
| `--no-pos-node-features` | `--pos-node-features none` |
| `--no-vel-node-features` | `--vel-node-features none` |
| `--no-speed`, `--no-accel`, `--no-offside` | `--speed none`, `--accel none`, `--offside none` |
| Wrapper `--poss-vel-aware` / `--no-poss-vel-aware`; direct `--poss_vel_aware` | `--vel-node-features all` to retain possessor motion, or `no-poss` to exclude it |
| `--poss-rel-vel-aware`; direct `--poss_rel_vel_aware` | `--poss-velocity-angles all` or `none` |
| `--keeper-aware`, `--ball-z-aware`; direct underscored spellings | `--keeper-id true/false`, `--ball-z true/false` |
| `--no-poss-geometry`, `--no-goal-features` | `--poss-geometry none`, `--goal-features none` |
| `--goal-poss-dist` | `--goal-poss-geometry true` |
| `--extend-features`; direct `--extend_features` | `--extended-node-features all` or `none` |
| `--v-edge-features`, `--v-edge-features-no-poss`, `--no-v-edge-features` | `--velocity-angle-edge-features all`, `no-poss`, or `none` |
| Edge switches ending in `-no-poss` or starting with `--no-` | The corresponding edge flag followed by `no-poss` or `none` |
| `--lane-survival [max/top_N/top_pass_N]`, `--no-lane-survival` | `--lane-survival all/none` plus `--lane-survival-mode MODE` |
| `--goal-nodes-aware`, `--no-goal-nodes` | `--goal-nodes true/false` |

Legacy `--possessor-aware` / direct `--possessor_aware` controls the entire
historical block `13:19`, not just identity. The new `--possessor-id` is
independent of `--poss-geometry` and `--poss-velocity-angles`. Legacy
`--xy-only` / direct `--xy_only` remains a preset that zeros columns `7:12`
and `13:19`; it is not literally x/y only. Its original interactions, including
the rejection of the legacy `--no-pos-node-features` combination, remain intact.

#### Graph structure and prediction candidates

These controls affect structure or output eligibility rather than masking a
learned feature column:

| Control | Effect | Fresh default |
| --- | --- | --- |
| `--goal-nodes true/false` | Retain/remove goal nodes and incident edges | `true` for the eight retained tasks |
| Wrapper `--action-intent-include-goals true/false`; direct `--include-goals true/false` | Include/exclude action-intent shot candidates and shot examples | `true` for action intent; excluded for other retained tasks |
| `--sparsify {none,delaunay}` (wrapper); `{none,distance,delaunay}` (direct) | Filter edges | No filtering |
| `--max_edge_dist <metres>` (direct) | Distance-sparsification threshold; does not enable filtering itself | 10 m, inactive |
| `--filter_blockers` (direct) | Remove nodes behind the possessor by the x-position/team rule | Off |
| Task suffix `_oppo_agn` | Remove opposing-team nodes automatically | Only for those direct-training tasks |

Input goal nodes are independent of explicit goal geometry and prediction
eligibility. Enabling action-intent goal predictions requires input goal nodes.
Legacy task-dependent goal policies remain checkpoint-driven.

The training wrapper accepts `--sparsify {none,delaunay}` and defaults to `none`.
For example, train all selected components with Delaunay connectivity using:

```powershell
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal --return_type disc_0.9 --intended-receiver-mode original --sparsify delaunay
```

Omit `--sparsify` or pass `--sparsify none` for all pairwise edges after other
node filtering. Existing graph artifacts remain reusable. Saved arguments,
feature signatures, and prepared-cache identities record the mode. Resume and
source refit inherit saved settings and reject explicit `--sparsify` overrides;
automatic final refits inherit their validation source's selection. This option
is available on the training wrapper and direct `train.py`, not `scripts/main.py`.
Distance filtering remains a direct-training option.

Metadata summary CSVs include a `sparsify` column for saved models, bundles,
component runs, and visualizations, using each row's saved model setting.
Missing settings remain blank. Rebuild existing summaries with
`python scripts/backfill_metadata_summaries.py`; retraining is unnecessary.

Distance sparsification retains an edge when its raw distance is **at most**
`max_edge_dist`, **or either endpoint is the possessor**, regardless of distance.
This applies to goal edges too, preserves both possessor-edge directions, and
requires possessor awareness. Delaunay sparsification instead retains directed
edges between triangulation neighbours using original coordinates; it has no
special possessor exemption and is applied only when more than three nodes
remain. Neither option removes nodes. Fresh `gat`, `transformer`, `relationship`,
and `relationship-attention` models restore one explicit self-edge per node after
sparsification; legacy checkpoints retain
their saved self-loop behavior.

Despite its name, `filter_blockers` is not a geometric shot-blocking test. It
retains a node only if it is the possessor or belongs to the opposing team,
**and** `node_x > possessor_x - 5 m`. It removes other teammates and nodes farther
behind the possessor, with their incident edges, and requires possessor awareness.
Coordinates follow the source graph's orientation.

Automatic example selection also affects the training population:

- `TASK_CONFIG` determines whether passes, carries, and shots are eligible for
  each task. `success_*` tasks retain successful actions; `failure_*` tasks retain
  failed actions. Shot success defaults to `--shot_success unblocked`, meaning
  unblocked versus blocked rather than scored versus not scored; `goal` selects
  the alternative interpretation.
- `--min_pass_dur` defaults to **0.5 seconds** in both training entry points and
  removes shorter passes. It does not remove carries or shots on duration grounds.
- Tasks requiring intended receivers exclude missing intended targets. Receiver
  tasks use in-play-only observations when `--include_out` is absent (the direct
  default; the wrapper does not enable it), excluding missing/out-of-play receiver
  observations. Version-2 failure-receiver handling additionally validates that
  observed receivers are opponents rather than goal nodes or teammates.
- Disabling action-intent goal candidates excludes shot-labelled examples, as
  described under goal input context below.
- Missing source graphs and graphs without exactly one identifiable possessor
  are skipped. Task-specific invalid receiver/target and pass-height observations
  are also skipped or rejected by validation; datasets record `skipped_rows` and
  match-level loading exclusions. Cached physical/auxiliary inputs have additional
  availability/compatibility requirements described in their sections.

The core implementation references are `models/dataset_config.py`,
`dataset.py`, `datatools/utils.py`, `models/goal_context.py`, and
`models/gnn.py`; wrapper defaults are in `scripts/train_relevant_models.py`.
Run metadata and per-model feature signatures record resolved input settings.
Consult the selected feature run for preprocessing and augmentation provenance;
this table describes training input preparation rather than every upstream
tracking-quality or label-generation rule.

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
2. train `success_intent` with `scripts/train_relevant_models.py --only-success-intent --feature-run-id <feature_run_id>`
3. generate a derived model-mode feature run with `scripts/generate_relevant_features.py --extend-feature-run-id <feature_run_id> --intended-receiver-model-id success_intent/<model_run_id>`
4. train the retained models on that new feature run with `--intended-receiver-mode model`

If you also need additional return semantics, repeat `--return_type` on the extension command. The derived run preserves the base run's existing return types first, then appends any newly requested return types.

`success_intent` is the teammate-selection intended-receiver model. It is trained from observed successful-pass receivers (`receiver_id`) and does not belong to any intended-receiver mode. `failure_receiver` is a separate auxiliary model used for failed-pass / opponent-receiver handling; it is not the intended-teammate model itself.

## physical_xpass_workflow

PC-xPass generation can add aerial candidates with `--high-pass on`. The default
elevations are `--high-pass-angle 20 25 30`, and total release speeds are
`--high-pass-speed 21 23 25 27 29`, independently of ground `--speed-step`.
These lists add 15 aerial trajectories per horizontal angle. Lane interception uses
`--high-pass-lane-cutoff 2` metres and endpoint control uses
`--high-pass-control-cutoff 0.5` metres. Top-pass retains one representative per
speed and horizontal angle, selecting its best elevation and endpoint. Use a
fresh cache ID for this revision. See [aerial PC-xPass](docs/pc_xpass_high_pass.md)
for physics, combined ranking, cache compatibility and measurement commands.

Physical xPass is now a precompute-first runtime signal from Jonas Bischofberger's `accessible-space` package. Generate the runtime caches once, then inference and visualization read those caches without running the expensive physical simulation.

Runtime mode is the default:

```powershell
python scripts/generate_physical_xpass.py
python scripts/generate_physical_xpass.py --no-skillcorner
python scripts/generate_physical_xpass.py --no-sportec --benchmark-modification <id> --hawkeye-situation-id <id>
python scripts/generate_physical_xpass.py --pass-height-model-id pass_height/<model_run_id>
python scripts/generate_physical_xpass.py --pc-xpass --top-n 10 --top-pass 10
python scripts/generate_physical_xpass.py --pc-xpass --top-n 10 --top-pass 5 10
python scripts/generate_physical_xpass.py --pc-xpass --export-lane-control --pc-xpass-id <new_run_id>
```

For **pc-xPass**, top metrics are opt-in: omitting both flags generates maximum output only. `--top-n N` retains the mean over the best N speed–angle–endpoint combinations; `--top-n-values 5 10` explicitly requests multiple counts. `--top-pass 5 10` (also `--top_pass`) instead selects the best endpoint for each speed–angle pair, ranks those distinct pairs, and exports a mean for each requested count. Endpoint and pair ranking both use xPass × xT with `--top-xt`, or xPass otherwise; the exported means remain xPass. Each family has separate lane-survival and receiver-control diagnostics. This describes good execution options, not an explicit execution-noise distribution.

Downstream inference and visualization select exactly one count directly with `--pc-xpass --top-pass 5`; no `--xpass-version` flag is needed. This is equivalent to `--pc-xpass --xpass-version top-pass5`. The evaluation commands already use pc-xPass and accept `--top-pass 5` directly. Generating the metric does not change downstream defaults: existing commands defaulting to `top10` still need `--top-n 10` during generation or an explicit alternative selection. Conflicting selectors and unavailable cache metrics raise errors. Original physical xPass defaults are unchanged.

Optional **per-opponent lane-control export**: add `--export-lane-control` when creating a fresh pc-xPass version. The main cache remains wide (one row per state with receiver-specific columns); a separate long-format Parquet sidecar stores one row per state, receiver, and opponent at `data/pc_xpass/<run_id>/<dataset>/lane_control/<match_id>.parquet`. It contains `match_id`, `action_index`, `physical_state_hash`, nullable `frame_scope` and `state_frame_id`, `receiver_id`, `opponent_id`, `lane_control`, and the selected pass's `speed`, `angle`, `distance`, `target_x`, and `target_y`.

Each contribution is the opponent's maximum interception control over lane samples **strictly before the endpoint**; `product(1 - lane_control)` reconstructs the selected pass's lane survival. Selection follows the main cache, including xPass × xT ranking with `--top-xt`. The export is off by default and does not change scoring or existing cache columns. **No backfill:** it cannot be enabled on an existing version created without it. Export-enabled versions can resume unfinished computation, but missing sidecars for existing cache hits cause an incomplete-coverage error rather than recomputation. Coverage manifests include valid empty exports. See [lane-control export documentation](docs/lane_control_export.md).

Default runtime output locations:

- `data/runtime_physical_xpass/sportec`
- `data/runtime_physical_xpass/skillcorner`
- `data/runtime_physical_xpass/benchmark`
- `data/runtime_physical_xpass/hawkeye`

The cached row contains `pass_distance`, per-player nearest-opponent distances for v2 blending, optional `ball_z` gating, and the selected physical xPass metrics:

- `<player_id>`: default `noise_kernel_xpass`
- `<player_id>__max_xpass`: original max-over-grid xPass
- `<player_id>__topmean_xpass`: mean of the top `N` finite grid values; `N` is controlled by `scripts/generate_physical_xpass.py --top-n` and defaults to `10`
- `<player_id>__distance_to_nearest_opponent`: distance in meters from that player to the nearest finite opponent node; this is filled from graph geometry and is not an xPass probability column
- `<player_id>__pass_height`: legacy physical-xPass enrichment only; new PC-xPass generation omits it and PC-xPass consumers ignore obsolete values
- `ball_z`: row-level cached ball height. Existing xPass metric rows are reused and updated in place when only `ball_z` is missing.

Current runtime defaults are `--consider-teammates`, `--speed-aggregation package_max`, speeds `3..22 m/s` in `1 m/s` steps, coarse angle search plus adaptive `2.5` degree local refinement, `--top-n 10`, `--num-workers auto`, `--physical-batch-size 16`, and `--worker-thread-limit 1`. Runtime caches are updated in place and compatible existing rows are reused; old max-only runtime caches are incompatible and should be deleted or moved before regeneration.

PC-xPass generation loads no learned models and stores no pass-height predictions. Select height and intent auxiliaries explicitly during inference/evaluation. Generation rejects `--pass-height-model-id`, `--pass-height-device`, and `--pass-height-threshold` in PC-xPass mode. Ordinary legacy physical-xPass behavior is outside this workflow.

Add `--pc-xpass` to generate the pitch-control-style cache family under `data/pc_xpass/<run_id>/<dataset>` instead. pc-xPass uses the same dataset selectors and runtime workflow, but stores top-N columns as `<player_id>__top<N>_xpass`; top-N generation requires an explicit `--top-n` or `--top-n-values`. To export several pc top-N columns in one cache, use `--top-n-values 5 10 25`; `--top-n` still controls the unsuffixed default player columns. Supply `--pc-xpass-id <run_id>` to resume an existing version; omitting the ID creates a new version. See [resuming physical generation](#resuming-a-version).

pc-xPass motion and control assumptions are generation-time settings. `--reaction-time` accepts a fixed seconds value or `dist_pass`, where each player uses `clip(distance_to_passer / --dist-pass-div, --dist-pass-min, --dist-pass-max)` with defaults `50`, `0.2`, and `0.7`. Player movement speed defaults to `--max-player-speed 5.0`; `--max-player-speed-off` and `--max-player-speed-def` can override attacking and defending players separately. Lane survival and endpoint control use separate sigmoid settings: `--lane-power`, `--lane-inflection-point`, `--control-power`, and `--control-inflection-point`, all defaulting to `15` and `0.3` for their respective power/inflection values.

Endpoint receiver control defaults to `--endpoint-normalization normal`, which keeps raw control when total endpoint control is at most `1` and otherwise normalizes by total raw control. Alternative pc-xPass modes are `normal-one` (receiver plus strongest challenger), `subtract` (receiver raw minus all challenger raw controls), and `subtract-one` (receiver raw minus strongest challenger). `--boost-def-endpoint-control <factor>` multiplies defending-player endpoint raw controls before this competition step; default is `1.0`.

pc-xPass also applies a target-position discount by default for pass end locations that move the receiver farther from the opponent goal than the receiver's current position. Disable it with `--use-position-discount false`. The default discount uses `--position-discount-power 2.0` and `--position-discount-distance 20.0`: no backward goal-distance change gives discount `1`, a `10m` backward goal-distance change gives `0.5`, and `20m` or more gives `0`.

Use `--pc-xpass --top-xt` to rank max and top-N target locations by `xPass × xT`, using the bilinearly interpolated canonical surface at `data/xT/xT_xy_surface.csv`. This changes only location selection: `__max_xpass`, `__top<N>_xpass`, and the unsuffixed top-N column still contain xPass values, not xPass×xT. The selected detail columns describe the highest xT-ranked location. xT-ranked and normal pc-xPass caches are incompatible, so rerun generation when switching this flag.

### Inference-time blending

Use physical xPass at inference by adding `--use-physical-xpass` to the inference script. The pass-success model still runs normally. With v1-v5, each receiver probability is blended with cached physical xPass:

```text
pass_success = (1 - w) * physical_xpass + w * pass_success_model
```

Select the blend weight with `--xpass-weight {v0,v1,v2,v3,v4,v5}`. The default is `v3`.

`v0` copies the selected raw physical xPass probability directly into `pass_success`, so pass-score and EPV calculations use physical values without blending. Cached probabilities, including `0` and `1`, are preserved without model-input epsilon or floor clipping. Existing offside masking and learned carry predictions still apply; the checkpoint also remains available as `model_pass_success`. No distance, nearest-opponent, height, or intent values are needed for v0 weighting. An active `--ball-z-limit` is rejected with v0 because it would switch back to learned predictions; omit it or use `--ball-z-limit none`. Missing physical inputs retain the existing failure or missing-value policy, with no learned fallback. Combined-v0 evaluation reports a learned-model weight of zero.

```bash
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id FEATURE_RUN --pc-xpass --use-physical-xpass --pc-xpass-id CACHE --top-pass 10 --xpass-weight v0
```

`v1` is the original distance-only weight:

```text
w = min(pass_distance / 100, 1.0)
```

`v2` uses the receiver's nearest-opponent distance:

```text
x = clip(pass_distance / 100, 0, 1)
y = distance_to_nearest_opponent / 100
w = clip(0.5 * sin((pi / 0.8) * x)^3 * (1 + y * 2), 0, 1)
pass_success = (1 - w) * physical_xpass + w * pass_success_model
```

`v3` is the default symmetric distance weight:

```text
x = clip(pass_distance / 100, 0, 1)
w = clip(0.75 / (1.0 + (x * (1.0 - x) / 0.25)^(-8)), 0, 1)
pass_success = (1 - w) * physical_xpass + w * pass_success_model
```

`v4` uses auxiliary per-player pass-height probability and discounts the model weight for very long passes:

```text
x = pass_distance / 100
w = pass_height * cos((pi / 2) * (x / z)^n)  if x <= z, otherwise 0
pass_success = (1 - w) * physical_xpass + w * pass_success_model
```

The defaults are `n=2` and `z=0.8`; override them with `--v4-power <float>` and `--v4-zero <float>` when using `--xpass-weight v4`. Add `--discount false` to disable the cosine distance discount and use `w = pass_height` directly.

`v5` uses the same auxiliary per-player pass-height probability as v4, but replaces the distance discount with a linear discount based on the runtime pass-intent probability:

```text
intent_factor = clip(pass_intent / t, 0, 1)
w = pass_height * intent_factor
pass_success = (1 - w) * physical_xpass + w * pass_success_model
```

The default threshold is `t=0.01`; override it with `--v5-intent-threshold <float>`. Thus intent probabilities `0`, `0.005`, `0.009`, and at least `0.01` apply discounts of 100%, 50%, 10%, and 0% to the learning-model weight. Use `--v5-discount false` to disable the intent discount and use `w = pass_height`. V5 requires explicit `--xpass-pass-intent-model-id pass_intent/<id>` and `--xpass-pass-height-model-id pass_height/<id>` selectors, including when `--v5-discount false` is used. Both auxiliary predictions are computed at runtime and aligned by physical state and candidate player; neither is stored in the PC-xPass cache.

`--xpass-weight v2` requires the cached `<player_id>__distance_to_nearest_opponent` columns. If they are missing or non-finite for a blended player, inference fails clearly instead of falling back to v1 or pure model predictions. Rerun `scripts/generate_physical_xpass.py` to backfill these columns; existing xPass metric values are reused and are not recomputed when only nearest-opponent distances are missing.

PC-xPass `--xpass-weight v4` requires `--xpass-pass-height-model-id pass_height/<id>`. V5 additionally requires `--xpass-pass-intent-model-id pass_intent/<id>`. Height-weighted evaluation also requires the height auxiliary. Component `--pass-height-model-id`, component `--pass-intent-model-id`, and bundles select displayed/exported models independently and cannot satisfy these auxiliary requirements. Different component and auxiliary checkpoints are allowed. Unused auxiliary selectors are rejected.

Each auxiliary uses its own features, goal policy, masks, and lane-survival inputs from the original state graph, even when its component is hidden. Height probabilities are independent sigmoids, not normalized across candidates. Missing or nonfinite required predictions raise an error; obsolete cache height columns are ignored. Existing behavior for unavailable physical states is preserved. `--pass-height-threshold` asserts the auxiliary checkpoint's height definition for blending and cannot change prediction meaning. Required lane-survival metrics must already exist; for example, generate `--top-pass 10` for a checkpoint using `top_pass_10`.

```powershell
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id FEATURE_RUN --use-physical-xpass --pc-xpass --pc-xpass-id CACHE --xpass-weight v4 --xpass-pass-height-model-id pass_height/HEIGHT
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id FEATURE_RUN --use-physical-xpass --pc-xpass --pc-xpass-id CACHE --xpass-weight v5 --xpass-pass-height-model-id pass_height/HEIGHT --xpass-pass-intent-model-id pass_intent/INTENT
python scripts/evaluate_relevant_models.py --pass-success-model-id pass_success/SUCCESS --evaluate-combined-success --pc-xpass-id CACHE --xpass-version top10 --xpass-weight v5 --xpass-pass-height-model-id pass_height/HEIGHT --xpass-pass-intent-model-id pass_intent/INTENT
```

Add `--ball-z-limit <float>` to ignore physical xPass for high-ball passes. When cached `ball_z` is greater than the limit, the blend weight is forced to `1.0`, so the final pass-success value is the pass-success model output. The default is `--ball-z-limit none`, which ignores `ball_z` and keeps existing behavior. Cached `ball_z` is required only when a numeric limit is set; rerun `scripts/generate_physical_xpass.py` to backfill it without recomputing existing xPass metrics.

Examples:

```powershell
python scripts/run_relevant_models.py --split test --bundle-id <bundle_id> --use-physical-xpass
python scripts/run_hawkeye.py --bundle-id <bundle_id> --use-physical-xpass
python scripts/run_benchmark.py --bundle-id <bundle_id> --use-physical-xpass
python scripts/run_skillcorner.py --bundle-id <bundle_id> --use-physical-xpass
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --use-physical-xpass --xpass-version top10 --xpass-weight v2
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --use-physical-xpass --pc-xpass --xpass-version top25 --xpass-weight v3
```

Metric selection applies only to inference-time physical xPass blending:

- `--xpass-version top10`: default. Original physical xPass reads `<player_id>__topmean_xpass` and requires cache metadata `top_n=10`; pc-xPass reads `<player_id>__top10_xpass`.
- `--top-pass N` / `--top_pass N`: directly select the mean over the best N distinct passes, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed; `--xpass-version top-pass5` is an equivalent alternative.
- `--xpass-version top<N>`: dynamic top-N selection, for example `top25` or `top50`. The requested version must be present in the cache.
- `--xpass-version max`: use `<player_id>__max_xpass`.
- `--xpass-version uncertainty`: use the separately exported `<player_id>__uncertainty_xpass` (pc-xPass only).
- `--xpass-version noise-kernel`: use unsuffixed original physical xPass columns. This is invalid for pc-xPass.

PC-xPass generation supports `--uncertainty true` (default `false`) with fixed
`--speed-uncertainty 2`, `--angle-uncertainty 3`, `--elevation-uncertainty 6`.
It adds `<player>__uncertainty_xpass`, selected with `--xpass-version uncertainty`,
by averaging existing trajectory outcomes before selecting the best intention.
Raw max, `top<N>` and `top-pass<N>` exports remain independent and can coexist. See
[uncertainty behaviour, outputs and validation](docs/pc_xpass_uncertainty.md).

Inference reads physical xPass caches only. It does not compute or prewarm missing rows. If cache rows, selected metric columns, required `pass_distance` values, or v2 nearest-opponent distances are missing/unusable, the affected pass rows are skipped or failed according to the selected weighting mode.

### Visualization

Use `--show-physical-xpass` to render the cached physical xPass surface/option values next to the model visualizations. Visualization scripts also read runtime caches only and fail clearly if the requested cache or metric is missing.

```powershell
python scripts/visualize_action_components.py --match-id <match_id> --action-id <action_id> --bundle-id <bundle_id> --show-physical-xpass
python scripts/visualize_action_components.py --match-id <match_id> --action-id <action_id> --bundle-id <bundle_id> --use-physical-xpass --show-physical-xpass
python scripts/visualize_hawkeye.py --component-run-id <component_run_id> --situation-id <hawkeye_id> --show-physical-xpass
python scripts/run_and_visualize_hawkeye.py --situation-id <hawkeye_id> --bundle-id <bundle_id> --use-physical-xpass --show-physical-xpass
python scripts/visualize_benchmark.py --modification 1 --game-state 1 --component-run-id <component_run_id> --show-physical-xpass
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index> --component-run-id <component_run_id> --show-physical-xpass
```

For scripts that both run inference and visualize (`visualize_action_components.py` and `run_and_visualize_hawkeye.py`), either `--top-pass N` or `--xpass-version` selects the same cached xPass metric for both inference blending and the physical xPass visualization.

### Legacy training sidecars

If `--feature-run-id` is specified, the script falls back to the original Sportec feature-run sidecar mode:

```powershell
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id>
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --overwrite --speed-aggregation package_max --num-workers auto --physical-batch-size 16
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --overwrite --max-speed 20
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --match-id <match_id> --overwrite --num-workers 1
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --limit 100 --num-workers 1
python scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --reuse-cache-dir data/features/runs/<old_feature_run_id>/physical_xpass --overwrite
```

This legacy mode is for training-time physical-xPass checkpoints and remains independent from inference-time blending. Training-time sidecars are stored under:

- `data/features/runs/<feature_run_id>/physical_xpass/metadata.json`
- `data/features/runs/<feature_run_id>/physical_xpass/matches/<match_id>.parquet`

Train pass success with training-time physical xPass only if you explicitly want the legacy physical-xPass model architecture:

```powershell
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass
```

Only the `pass_success` low-level training command receives physical xPass flags. Other models trained in the same wrapper run ignore them. You can still combine such a checkpoint with inference-time `--use-physical-xpass`, but that double-counts physical xPass and is mainly useful for compatibility checks.

### Inference script xPass flags

- `--use-physical-xpass`: enable inference-time pass-success blending from the runtime cache.
- `--pc-xpass` / `--pc_xpass`: read `data/pc_xpass/<dataset>` instead of `data/runtime_physical_xpass/<dataset>`.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select blend weighting. Default: `v3`; `v2` requires cached `<player_id>__distance_to_nearest_opponent` values. PC-xPass v4/v5 require explicit runtime auxiliaries below.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--ball-z-limit <float|none>`: with a numeric limit, use 100% pass-success model weight when cached `ball_z` exceeds the limit. Default: `none`.
- `--physical-cache-dir <path>`: override the runtime physical xPass cache directory; default is `data/runtime_physical_xpass/<dataset>`.
- `--no-physical-cache`: disable the runtime cache override. Do not use this with inference-time blending unless you intentionally want cache lookup disabled.
- `--refresh-physical-cache`: deprecated and ignored for inference; run `scripts/generate_physical_xpass.py` to refresh/fill caches.
- `--physical-num-workers`, `--physical-worker-thread-limit`, and `--physical-batch-size`: retained for CLI compatibility/metadata, but inference no longer computes xPass rows.

The main model variant is a prior-like logit offset:

```text
logit_final = beta0 + beta1 * logit(physical_xpass) + delta_gat
```

`beta0` starts at `0.0`, `beta1` starts at `1.0`, and `delta_gat` is the residual correction learned by the GAT. This is not fully Bayesian updating, but it gives the physical model privileged baseline status: the GAT learns when observed data systematically correct that physical baseline.

Training-time physical flags:

- `--use_physical_xpass`: enable legacy feature-run sidecar loading and training-time physical pass-success integration.
- `--model-variant {gat_baseline,gat_plus_phys_feature,gat_phys_logit_offset,gat_phys_logit_offset_regularized}`: choose the architecture. `gat_baseline` is unchanged; `gat_plus_phys_feature` concatenates `logit(physical_xpass)` as an ablation; `gat_phys_logit_offset` is the recommended offset model; `gat_phys_logit_offset_regularized` is the offset model with optional residual safeguards.
- `--physical-cache-dir`: override the physical sidecar directory. By default it uses `<feature_run_root>/physical_xpass`.
- `--physical-eps 1e-4`: clamp probabilities to `[eps, 1-eps]` before taking logits.
- `--physical-xpass-floor <prob>`: optional lower floor applied before taking logits, e.g. `0.01 -> 0.2` when set to `0.2`. Default: unset.
- `--freeze-beta1`: freeze `beta1` at `1.0`; default is learned.
- `--freeze-beta0`: freeze `beta0` at `0.0`; default is learned.
- `--residual-regularization-lambda`: optional L2 penalty on the observed-target residual `delta_gat`.
- `--residual-clip-value`: optional tanh bound, `delta_gat = c * tanh(raw_delta / c)`.
- `--residual-distance-threshold 30.0`: distance cutoff for distance-specific residual controls; short passes are `NODE_FEATURE_POSS_DIST <= threshold`, long passes are `> threshold`.
- `--short-residual-regularization-lambda` / `--long-residual-regularization-lambda`: optional short/long overrides for residual L2. Unset values fall back to `--residual-regularization-lambda`.
- `--short-residual-clip-value` / `--long-residual-clip-value`: optional short/long overrides for residual clipping. Unset values fall back to `--residual-clip-value`.

The physical simulation uses AS/DAS-default normalization (`normalize=True` here), then the loader clamps before `logit`. `accessible-space` notes that probabilities/possibilities are not always perfectly normalized because normalization is numerically difficult across players and the ball trajectory. Requesting normalization reduces that issue; clipping then prevents infinite logits from exact 0 or 1 values.

Using both `--freeze-beta0` and `--freeze-beta1` gives `logit_final = logit(physical_xpass) + delta_gat`; residual controls still apply. To use physical xPass exactly as-is, both betas must be frozen and the GAT residual must also be removed or forced to zero.

Compare these runs:

```powershell
# Baseline GAT
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --model-variant gat_baseline

# Physical feature ablation
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_plus_phys_feature

# Recommended physical logit offset
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset

# Offset + residual L2
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset_regularized --residual-regularization-lambda 0.01

# Offset + residual clipping
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset_regularized --residual-clip-value 2.0

# Offset + stronger short-pass L2 than long-pass L2
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset_regularized --short-residual-regularization-lambda 0.02 --long-residual-regularization-lambda 0.002

# Offset + tighter short-pass clipping than long-pass clipping
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset_regularized --short-residual-clip-value 1.0 --long-residual-clip-value 3.0

# Offset + physical xPass floor for very low physical probabilities
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --target-family goal_distance --return_type disc_0.5_skip1 --intended-receiver-mode angle_only --use_physical_xpass --model-variant gat_phys_logit_offset_regularized --physical-xpass-floor 0.2 --short-residual-clip-value 0.25 --long-residual-clip-value 1.0
```

Tuning L2/clipping as ablations means training otherwise comparable runs where only the residual constraint changes. Start with no L2 and no clipping (`residual_regularization_lambda=0.0`, `residual_clip_value=None`) to see the clean offset effect, then test whether constraining the residual improves calibration or hypothetical-pass maps. Distance-specific settings are useful when you want the model to stay close to physical xPass on short passes but allow larger GAT corrections on long passes.

`include_out=True` is not supported initially because `player_cum_prob` is player-indexed and has no natural out-of-play node. IPW remains active when configured: the final observed-target BCE is still weighted by `batch_ipw`.

For visualization, use the runtime-cache `--show-physical-xpass` workflow described above. Visualization does not read legacy feature-run training sidecars by default.

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

This appendix covers every current `scripts/*.py` CLI entrypoint, including `scripts/main.py`, plus the low-level `train.py` and `test.py` entrypoints used by the training and evaluation wrappers. The legacy repo-root `main.py` is intentionally not included here because it is part of the upstream defensive-score path, not the scoped workflow described above.

#### Shared pass-height target option

`--pass-height-threshold <meters>` accepts a finite, positive observed maximum-ball-height cutoff, with the inclusive definition `pass_max_ball_z >= cutoff`. It is distinct from `--classification-threshold`, which thresholds predicted probabilities. New `pass_height` training uses the explicit cutoff or, when omitted, the selected feature run's recorded cutoff; it fails if neither is known. Evaluation defaults to the checkpoint's saved target definition, while an explicit value relabels only the observed evaluation targets and diagnostics—not the meaning of its predictions. Legacy checkpoints keep their stored labels unless explicitly overridden. The option is available on `scripts/main.py`, `scripts/train_relevant_models.py`, `train.py`, `scripts/evaluate_relevant_models.py`, `test.py`, and the height-aware physical/xPass generation and inference commands below. In xPass consumers it asserts the selected auxiliary checkpoint height definition where one is required; it cannot turn a model trained at one cutoff into a predictor for another.

### `scripts/main.py`

- `--train-split <percentage>`: assign the first integer percentage (1-99, floor rounding) of canonical `MatchId` order to development data and reserve the remainder for independent testing. Default: `50`; see [Split Definition](#split-definition).
- `--train-count <int>`: exact number of development matches in canonical `MatchId` order, including validation. Mutually exclusive with `--train-split`; for the 918-match dataset use `--train-count 765`. In evaluation, this optionally checks the count recorded by the selected model.
- `--validation-mode {holdout,holdout_80_20,expanding}`: use one chronological holdout or explicitly sized nested training stages with a shared latest validation set. Default: `holdout`; the old alias requires 20% validation.
- `--validation-count N` / `--validation-fraction F`: mutually exclusive validation sizes within the saved development pool. Default fraction: `0.2`.
- `--training-counts N ...` / `--training-fractions F ...`: mutually exclusive training-size lists, excluding validation. Fractions use the remaining pool. Expanding requires a list; holdout accepts one size or defaults to the entire remaining pool.
- `--final-refit`: optionally refit all development games using the largest validation stage's best epoch and recorded per-epoch learning rates. No validation, early stopping, rollback, or minimum epoch requirement. See [validation and refitting](docs/configurable_validation.md).
- `--target-family {goal,xg,xt,goal_distance,epv}`: retained outcome family passed to training. Required when an outcome model is selected.
- `--return_type <disc_gamma|disc_gamma_skip1|disc_max_gamma|disc_max_gamma_skip1|disc_poly_max_b_z|disc_poly_max_b_z_spstop|next_N|next_N_skip1|in_N>`: resolved return semantics passed to feature generation and training. `disc_max_gamma` and `in_N` are valid only for `xt`, `goal_distance`, and `epv`; polynomial max is valid only for `xt` and `goal_distance`. Required when feature generation or an outcome model is enabled.
- `--intended-receiver-mode {original,angle_only,model}`: retained-model training mode. Required when a mode-dependent model is selected.
- `--pass-height-threshold <meters>`: forward one observed pass-height cutoff to training, evaluation, and height-aware downstream stages. See [Shared pass-height target option](#shared-pass-height-target-option).
- `--only-action-intent`, `--only-pass-intent`, `--only-success-intent`, `--only-pass-success`, `--only-pass-height`, `--only-outcome-scoring`, `--only-outcome-conceding`, `--only-failure-receiver`: train exactly the selected models. Repeated flags are additive and cannot be mixed with ordinary per-model toggles.
- `--intended-receiver-model-id <model_id>`: optional `success_intent` checkpoint used to add the `model` intended-receiver variant during feature generation.
- `--feature-run-id <feature_run_id>`: explicit feature run id to reuse or assign.
- `--bundle-id <bundle_id>`: explicit model bundle id to reuse or assign.
- `--success-intent-model-id <model_id>`: optional `success_intent` checkpoint forwarded to evaluation.
- `--skip-preprocess`, `--skip-xt`, `--skip-goal-distance`, `--skip-epv`, `--skip-features`, `--skip-train`, `--skip-evaluate`, `--skip-run-relevant`, `--skip-hawkeye`, `--skip-benchmark`, `--skip-skillcorner`: skip individual stages.
- `--benchmark-input-dir <path>`: local benchmark data root passed to `scripts/run_benchmark.py`.
- `--v-edge-features` / `--v-edge-features-no-poss` / `--no-v-edge-features`: legacy pipeline switches forwarded to training for velocity-angle edge inputs. Default: off. The training entry points additionally accept `--velocity-angle-edge-features {all,no-poss,none}`.
- `--xy-only` / `--no-xy-only`, `--possessor-aware` / `--no-possessor-aware`, `--keeper-aware` / `--no-keeper-aware`, `--ball-z-aware` / `--no-ball-z-aware`, `--poss-vel-aware` / `--no-poss-vel-aware`, `--poss-rel-vel-aware` / `--no-poss-rel-vel-aware`, `--offside` / `--no-offside`, `--extend-features` / `--no-extend-features`, `--pass-lane-features` / `--no-pass-lane-features`: legacy boolean switches forwarded to the training profile. These pipeline switches do not take retention values.
- Explicit selectors such as `--vx defending`, `--vel-node-features none`, `--poss-geometry none`, `--goal-features none`, and `--goal-nodes false` are available on `scripts/train_relevant_models.py` and `train.py`, not on `scripts/main.py`. Train separately and use `--skip-train --bundle-id <bundle_id>` for later pipeline stages.
- `--overwrite`: allow supported preprocessing and target-artifact outputs to be rebuilt.
- `--relevant-split {train,test,all}`: split passed through to `scripts/run_relevant_models.py`.
- `--device <device>`: device passed to evaluation and inference stages.
- `--dry-run`: print the resolved commands without executing them.

### `scripts/preprocess_sportec.py`

- `--carry-artifacts-only`: build or refresh control-event, carry-segment, and carry-audit sidecars from existing canonical outputs without rerunning preprocessing. Default: off.
- `--skip-carry-artifacts`: skip carry sidecar generation during normal preprocessing. Default: off; mutually exclusive with `--carry-artifacts-only`.
- `--match-id <id>`: process only one or more specific Sportec match ids. Default: all discovered matches.
- `--limit <N>`: process only the first `N` discovered matches. Default: no limit.
- `--overwrite`: rebuild existing outputs. Default: off.
- `--skip-sync`: stop before event-tracking synchronization. Default: off.
- `--sync-source {sportec_kpi,elastic}`: synchronization source for canonical event outputs. Default: `sportec_kpi`.

### `scripts/generate_xt.py`

- `--train-split <percentage>`: development percentage (integer 1-99) used to fit the xT surface. Default: `50`.
- `--train-count <int>`: exact number of development matches in canonical `MatchId` order, including validation. Mutually exclusive with `--train-split`; for the 918-match dataset use `--train-count 765`. In evaluation, this optionally checks the count recorded by the selected model.
- `--match-id <id>`: export xT sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--source-grid-l <N>`: socceraction source-grid length bins. Default: `12`.
- `--source-grid-w <N>`: socceraction source-grid width bins. Default: `8`.
- `--reuse-source-grid`: load `data/xT/xT_source_grid.csv` and skip socceraction source-grid fitting. Default: off.
- `--use-interaction`: include `normalized_x * normalized_centrality` in the x/y logit GLM. Default: off.
- `--use-nonlinear {x,y}`: include constrained squared and cubic GLM terms for the selected axis. Repeat to include both axes. Default: off.
- `--fit-only`: write only model-level xT artifacts and skip `xT.csv` plus per-match sidecars. `--match-id` and `--limit` are ignored in this mode. Default: off.
- `--overwrite`: overwrite existing xT outputs. Default: off.

### `scripts/generate_goal_distance.py`

- `--match-id <id>`: export goal-distance sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--overwrite`: overwrite existing goal-distance outputs. Default: off.

### `scripts/generate_epv.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--bundle-id <bundle_id>`: source model bundle containing `pass_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding`.
- `--pass-intent-model-id`, `--pass-success-model-id`, `--outcome-scoring-model-id`, `--outcome-conceding-model-id`: explicit source checkpoint overrides.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: newest compatible source feature run from the selected models or bundle.
- `--match-id <id>`: export EPV sidecars only for one or more specific match ids. Default: all available synced matches.
- `--limit <N>`: process only the first `N` available matches. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--overwrite`: overwrite existing EPV outputs. Default: off.
- `--use-physical-xpass` / `--use_physical_xpass`: blend only the `pass_success` inference used inside EPV calculation with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches from `data/pc_xpass/sportec` instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric for the pass-success blend. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--ball-z-limit <float|none>`: with a numeric limit, use the pass-success model only when cached `ball_z` exceeds the limit. Default: `none`.
- `--physical-cache-dir <path>`: Sportec runtime physical xPass cache override. Default: `data/runtime_physical_xpass/sportec`.

### `scripts/generate_physical_xpass.py`

#### Resuming a version

Create a fresh physical-only Sportec cache, then resume it with the same ID:

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id states_v3 --sportec-feature-run-id FEATURE_RUN --top-pass 5 10 --top-n-values 10 25 --scope actions --no-skillcorner --no-benchmark --no-hawkeye
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id states_v3 --scope actions --no-skillcorner --no-benchmark --no-hawkeye
```

Replace IDs and retain your intended numerical settings. Omitted physical settings and the resolved Sportec feature run inherit on resume; explicit conflicts require a new version. Match checkpoints publish physical rows, manifests, and lane-control coverage together. Interrupted work is recomputed while completed work is reused. There is no height enrichment or refresh phase.

Selection options, including matches, seasons, dataset switches, `--scope`, and `--frames`, are not restored automatically. Worker/batch/window settings may change. Physical source and reconstruction fingerprints remain validated.

New generation records `physical_input_contract = pc_xpass_physical_inputs_v1`; resume requires that contract. Sportec requires a fresh cache using `sportec_possession_states_v3` / `match_physical_states_v3`. Old caches remain intact, with no migration or identity mapping. Compatible physical caches for other datasets remain readable; their obsolete height fields are ignored. Model graph options, masks, checkpoints, and height cutoffs do not participate in the new physical identities.

See [PC-xPass cache documentation](docs/pc_xpass_caches.md) for details.

#### Matching inference and pc-xPass frame coverage

For receipt 102, release 119, and `--scope frames --frames 5`, both paths select
102, 107, 112, 117, and 119. Sportec keeps invalid exact endpoints missing;
interior samples do not replace their event values or shift the sampling grid.
At a supported loss endpoint, the graph uses the winning player, matching the
existing training carry's post-state. The spell owner remains the original player.
SkillCorner retains its existing inward endpoint fallback.

For example (replace the example feature run, bundle, and cache IDs):

```bash
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id frames_demo --sportec-feature-run-id FEATURE_RUN --scope frames --frames 5 --no-skillcorner --no-benchmark --no-hawkeye
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id FEATURE_RUN --scope frames --frames 5 --use-physical-xpass --pc-xpass --pc-xpass-id frames_demo
```

Use the same `--scope`/`--frames` for SkillCorner inference and generation too;
the generator takes `--skillcorner-input-dir` / `--skillcorner-match-id` while
the inference runner takes `--input-dir` / `--match-id`.

Sportec uses `sportec_possession_states_v3` and cache format
`match_physical_states_v3`: physical states identify match, period, frame, actual
possessor, and provenance. A spell/association manifest separates ownership and
canonical event roles from physical state identity. SkillCorner retains
`possession_states_v1` and `match_possession_frames_v1`. Both datasets write one
`matches/<match_id>.parquet` per actual match, plus optional lane-control output.
Older Sportec caches require regeneration into a **new version**; no migration
or fallback reader is provided. See [the cache format documentation](docs/pc_xpass_caches.md#sportec-and-skillcorner-match-checkpoints).

Results are buffered across computation batches and saved after each complete
match. An interrupted match loses unsaved work; compatible completed rows are
reused without rewriting an unchanged parquet. Scope/stride are outside state
identity, so expanding coverage preserves overlapping states. Sportec groups
worker inputs by period and actual possessor; different players at one frame stay
distinct, and the same physical state can serve several spell or event roles.
Sportec provenance includes the feature run and source fingerprints;
full reconstruction descriptors and the spell manifest are embedded in each parquet. Inference remains
cache-read-only and rejects pending or inconsistent match commits.
For Sportec possession-state inference, physical blending requires pc-xPass;
legacy ordinary physical-xPass rows use event indexes and cannot safely supply
these frame-indexed states. Inference without physical blending is unaffected.

The old SkillCorner `--frames-first-and-last` / `--frames-all` switches and their
generator-prefixed equivalents have been removed. Use `--scope actions` or
`--scope frames --frames 1` instead. Ordinary physical-xPass generation, training
features, carry generation, and Benchmark/Hawkeye frame selection are unchanged.

Sportec generation processes the canonical match universe in order, or the explicit `--match-id` selection, skipping matches with missing required artifacts. It does not resolve a train/test split. `--split`, `--train-split`, and `--train-count` are no longer accepted.
- default mode: generate runtime physical xPass caches for Sportec, SkillCorner, Benchmark, and Hawkeye under `data/runtime_physical_xpass/<dataset>`.
- `--feature-run-id <feature_run_id>`: enable legacy Sportec feature-run sidecar mode under `data/features/runs/<feature_run_id>/physical_xpass`.
- `--no-sportec`, `--no-skillcorner`, `--no-benchmark`, `--no-hawkeye`: skip selected runtime datasets.
- `--match-id <id>`: restrict Sportec matches. Default: all matches in the canonical preprocessed match universe.
- `--season {22_23,23_24,24_25}`: restrict Sportec to selected seasons; repeat to select multiple seasons. Intersects with `--match-id`. Requires accessible raw match-information files and unambiguous membership for every candidate match; missing membership, conflicting seasons, or an empty selection fail. Other datasets remain enabled. Cannot be combined with `--no-sportec`. Default: no season filter.
- `--skillcorner-input-dir`, `--skillcorner-match-id`, `--skillcorner-limit`: SkillCorner runtime selectors.
- `--scope {actions,frames}` and `--frames N`: pc-xPass-only frame selection for Sportec and SkillCorner. Sportec `actions` selects exact spell endpoints and preserved canonical observations; `frames` adds every Nth interior frame in supported control intervals. SkillCorner retains its endpoint/stride selection. N defaults to 1. These options do not affect Benchmark/Hawkeye and are rejected without `--pc-xpass` or in legacy feature-sidecar mode.
- `--sportec-feature-run-id <feature_run_id>`: input feature run for Sportec pc-xPass. Default for a new version: latest; on resume: the recorded resolved run. Explicitly selecting a different run is rejected. Select the **same feature run as inference**. Both paths reconstruct inference spells from canonical actions and `data/control_events_synced/<match_id>.parquet`; training carry audits do not control inference eligibility.
- `--benchmark-input-dir`, `--benchmark-modification`, `--benchmark-limit`: benchmark runtime selectors.
- `--hawkeye-tracking-csv`, `--hawkeye-ball-csv`, `--hawkeye-situation-id`, `--hawkeye-limit`: Hawkeye runtime selectors.
- `--freeze-ballreceipt` / `--no-freeze-ballreceipt`: Hawkeye state construction. Initial default: enabled. A versioned pc-xPass resume inherits the recorded value and rejects an explicit conflict.
- `--limit <N>`: in Sportec/legacy mode, process only the first `N` pass actions across selected matches. Default: no limit.
- `--overwrite`: legacy feature-run mode only. Runtime caches are updated in place.
- `--reuse-cache-dir <path>`: reuse compatible Sportec rows from another `physical_xpass` directory and compute only misses. Reuse requires matching source, teammate policy, AS-default parameters, and `physical_eps`.
- `--return-type <return_type>` and `--intended-receiver-mode <mode>`: optional reference label directory selectors. Defaults are inferred from the feature run.
- `--physical-eps <eps>`: clamp stored physical xPass metric probabilities. Default: `1e-4`.
- `--ignore-teammates` / `--consider-teammates`: choose reduced target-plus-defenders simulation or all-player simulation. Default: `--consider-teammates`. With `--pc-xpass`, `--ignore-teammates` is shorthand for ignoring teammates in both lane survival and endpoint control.
- `--ignore-teammates-lane-survival`: pc-xPass only; ignore attacking teammates along the pass lane while still considering them for endpoint control unless `--ignore-teammates-control` is also set.
- `--ignore-teammates-control`: pc-xPass only; ignore attacking teammates at the pass end location while still considering them for lane survival unless `--ignore-teammates-lane-survival` is also set.
- `--max-speed <m/s>` / `--max_speed <m/s>`: upper speed-grid value. Default: `22`.
- `--speed-step <m/s>`: speed-grid step from `3` to `--max-speed`. Default: `1`.
- `--coarse-n-angles <N>`, `--refine-top-k-angles <N>`, `--refine-angle-radius <deg>`, `--angle-step <deg>`: adaptive angle search controls. Defaults: `36`, `2`, `10`, and `2.5`.
- `--top-n <N>`: number of highest finite xPass grid values averaged for the default top-N metric. Original physical xPass stores this as `__topmean_xpass` with metadata `top_n=N`; pc-xPass stores it as `__top<N>_xpass` and uses it for unsuffixed default player columns. Default: `10` for original physical xPass; opt-in for pc-xPass.
- `--top-n-values <N...>`: pc-xPass only; export additional top-N columns in one run, for example `--top-n-values 5 10 25` writes `__top5_xpass`, `__top10_xpass`, and `__top25_xpass`. An explicitly supplied `--top-n` value is also included.
- `--top-pass <N...>` / `--top_pass <N...>`: pc-xPass only; export `__top_pass<N>_xpass` and matching diagnostics for the best N distinct speed-angle pairs after endpoint optimization. Opt-in; accepts multiple positive integers.
- `--pc-xpass` / `--pc_xpass`: generate versioned pc-xPass caches under `data/pc_xpass/<run_id>/<dataset>` instead of runtime physical xPass caches.
- `--ball-dec <number|drag>`: pc-xPass constant deceleration in m/s^2, or fixed quadratic air drag. Default: `0.45`; `0` disables slowing. Drag uses `k = 0.013797619047619048 m^-1` and requires a fresh cache ID when switching from a numeric mode; see [ball motion and cache versions](docs/pc_xpass_caches.md).
- `--pc-xpass-id <run_id>`: create a named pc-xPass version or resume an existing one, reusing saved rows and restoring omitted calculation/input contracts. Data selections and execution settings remain flexible; repeat the original selections to finish the same coverage. Default without an ID: create a new version. See [resuming physical generation](#resuming-a-version).
- `--export-lane-control`: pc-xPass only; export per-opponent interception controls for each receiver's selected best pass to `lane_control/<match_id>.parquet` within the dataset cache. Default: off. Uses long-format state/receiver/opponent rows and includes state identifiers and selected-pass geometry; does not change the main cache or scoring. Enable only when creating a fresh version; no backfill of existing cache rows is performed. Resume inherits the recorded option, and missing sidecars for cache hits raise an incomplete-coverage error. Valid with `--no-max` when another xPass metric remains enabled; incompatible with `--reuse-cache-dir` and legacy `--feature-run-id` mode. See [export schema and semantics](docs/lane_control_export.md).
- `--top-xt`: pc-xPass only; rank max, top-N, and top-pass options by xPass × interpolated xT while exporting xPass probabilities. Best-pass details and optional opponent lane controls follow the same selected option. Default: off.
- `--margin {tta,reachability}`: pc-xPass only; choose the player-movement margin model. `tta` retains the reaction-time/max-speed model and is the default. `reachability` uses a fitted empirical reachability artifact and requires `--reachability-model-id`; see [Empirical reachability circles for pc-xPass](docs/reachability.md).
- `--reaction-time <seconds|dist_pass>`: pc-xPass only; fixed player reaction time or distance-to-passer mode. With `dist_pass`, each player uses `clip(distance_to_passer / --dist-pass-div, --dist-pass-min, --dist-pass-max)`. Default: `0.25`.
- `--dist-pass-div <float>`: pc-xPass only; divisor for `--reaction-time dist_pass`. Default: `50`.
- `--dist-pass-min <seconds>`: pc-xPass only; minimum reaction time for `--reaction-time dist_pass`. Default: `0.2`.
- `--dist-pass-max <seconds>`: pc-xPass only; maximum reaction time for `--reaction-time dist_pass`. Default: `0.7`.
- `--max-player-speed <m/s>`: pc-xPass only; overall player movement speed after reaction time. Default: `5.0`.
- `--max-player-speed-off <m/s>`: pc-xPass only; override movement speed for attacking players, including the possessor and teammates. Default: use `--max-player-speed`.
- `--max-player-speed-def <m/s>`: pc-xPass only; override movement speed for defending players. Default: use `--max-player-speed`.
- `--lane-power <float>` and `--lane-inflection-point <float>`: pc-xPass only; sigmoid parameters for lane-survival interception control. Defaults: `15` and `0.3`.
- `--control-power <float>` and `--control-inflection-point <float>`: pc-xPass only; sigmoid parameters for endpoint receiver control. Defaults: `15` and `0.3`.
- `--endpoint-normalization {share,normal,normal-one,subtract,subtract-one}` / `--endpoint_normalization {...}`: pc-xPass only; endpoint receiver-control competition mode. Default: `share`.
  - `--endpoint-normalization share`: multiply the receiver's raw endpoint control by its share of the total eligible endpoint control: `receiver_control = p_receiver^2 / S` for `S > 0`, otherwise `0`. This is not simple division by the sum: the resulting controls need not sum to one.
  - The eligible set contains the receiver, defenders, and other attacking teammates, excluding the passer. With `--ignore-teammates-control` (or `--ignore-teammates`), it contains only the receiver and defenders. For `share`, `S` sums finite raw controls clipped to `[0, 1]`; defender controls are multiplied by `--boost-def-endpoint-control` and clipped to `[0, 1]` before summation. Non-finite competitor controls contribute zero; a non-finite receiver control remains `NaN`.
  - `normal`: retain the receiver's raw control when the eligible total is at most one; otherwise divide it by that total (`receiver_control = p_receiver / max(1, S)`). Unlike `share`, boosted defender controls are not clipped before this total is computed.
- `--boost-def-endpoint-control <float>`: pc-xPass only; multiply defending-player endpoint raw controls before endpoint normalization. Default: `1.0`.
- `--use-position-discount <true|false>`: pc-xPass only; apply the goal-distance target-position discount before max/top-N aggregation. Default: `true`.
- `--position-discount-power <float>`: pc-xPass only; power for the target-position discount. Default: `2.0`.
- `--position-discount-distance <m>`: pc-xPass only; backward goal-distance delta where the target-position discount reaches zero. Default: `20.0`.
- `--pass-height-model-id`, `--pass-height-device`, `--pass-height-threshold`: rejected in PC-xPass generation; select height during inference with `--xpass-pass-height-model-id`.


- `--no-noise-kernel`, `--no-max`, `--no-topmean`: skip selected physical xPass output metrics. At least one metric must remain enabled. For top-only inference caches, use `--no-noise-kernel --no-max` and pass the matching `--xpass-version top<N>` during inference/visualization.
- `--num-workers <N|auto>`, `--max-auto-workers <N>`, `--worker-thread-limit <N>`, and `--physical-batch-size <N>`: runtime cache generation parallelism controls.
- `--sportec-runtime-prefetch`: PC-xPass only; prepare one Sportec input window ahead while the existing workers compute the current window. Default: off. Cache work and match publication stay on the main thread. Dry runs and single-worker runs remain sequential. Complete logical groups may exceed the row-window target; this bounds preparation windows, not total memory or match results. Uses spawn workers; other datasets retain their existing scheduling.
- `--sportec-runtime-prefetch`: PC-xPass only; prepare one Sportec input window ahead while the existing workers compute the current window. Default: off. Cache work and match publication stay on the main thread. Dry runs and single-worker runs remain sequential. Complete logical groups may exceed the row-window target; this bounds preparation windows, not total memory or match results. Uses spawn workers; other datasets retain their existing scheduling.
- `--sportec-runtime-row-window <N>` / `--hawkeye-runtime-row-window <N>`: PC-xPass only; buffer eligible input states and cap outstanding computation rows. Default: `physical_batch_size * resolved_workers * 2` (256 with eight workers and 16-state tasks). Existing `--skillcorner-runtime-row-window` and `--benchmark-runtime-row-window` use the same automatic selection. Cache hits count toward input buffering. Sportec and SkillCorner flush within each match; Hawkeye can combine situations; Benchmark combines modifications.
- `--sportec-runtime-match-window <N>`: legacy physical-xPass only; matches per compute window (default 4). Explicit use with `--pc-xpass` is rejected; use the Sportec row-window option instead.

PC-xPass generation lazily creates one worker pool per dataset and reuses it across windows and matches. A cold single-task window runs serially; subsequent small tails reuse an existing pool. Tasks retain at most `--physical-batch-size` states, except a smaller explicit row budget requires smaller tasks. Oversized input groups use bounded submissions while preserving complete match/situation publication. The row budget bounds queued computation, not prepared graphs or accumulated match results. These execution settings do not change numerical settings, state identities, manifests, or cache compatibility; existing compatible versions can be resumed without rebuilding.

Dataset/run statistics include a separate `scheduling` record: preparation between windows, cache scan, dispatch/compute, per-task durations, task occupancy, parent handling, writes/publication, and pool lifecycle counts. Sportec also records producer work, consumer waiting, backpressure, and observed preparation/compute overlap under `scheduling.preparation`. With prefetch, the existing between-window gap measures consumer delay rather than total preparation work; overlapping durations must not be summed as sequential phases. Task occupancy estimates occupied worker slots; it is not CPU utilization. Compare 256/512 with identical selected states, actual group buffering, fresh physical caches, and all workers fully warmed; retain 256 when results are inconclusive. See [runtime measurements and recommendations](docs/pc_xpass_runtime.md).

- `--dry-run`: inspect physical cache compatibility/hits without computing or changing cache rows/metadata.
- `--no-normalize`: deprecated compatibility flag; ignored because the AS-default max source uses `normalize=True`.

### `scripts/generate_relevant_features.py`

- `--train-split <percentage>`: assign the first integer percentage (1-99, floor rounding) of canonical `MatchId` order to development data and reserve the remainder for independent testing. Default: `50`; see [Split Definition](#split-definition).
- `--train-count <int>`: exact number of development matches in canonical `MatchId` order, including validation. Mutually exclusive with `--train-split`; for the 918-match dataset use `--train-count 765`. In evaluation, this optionally checks the count recorded by the selected model.
- `--use-carries`: generate additive Sportec carry-augmented base graphs and labels for both splits. Requires preprocessed carry sidecars. Default: off. With `--extend-feature-run-id`, create a derived run or use `--in-place` to add carries to a completed run. `--overwrite-feature-run` is also accepted as an alias for `--in-place`. Canonical artifacts are preserved.
- repeat `--return_type <disc_gamma|disc_gamma_skip1|disc_max_gamma|disc_max_gamma_skip1|disc_poly_max_b_z|disc_poly_max_b_z_spstop|next_N|next_N_skip1|in_N>`: write labels for one or more return semantics in the same feature run. `disc_max_gamma` and `in_N` are valid only for `xt`, `goal_distance`, and `epv`; polynomial max is valid only for `xt` and `goal_distance`.
- `--intended-receiver-model-id <model_id>`: optional `success_intent` checkpoint used to additionally include the `model` intended-receiver variant.
- `--run-id <feature_run_id>`: pin the feature run id instead of auto-generating one.
- `--extend-feature-run-id <feature_run_id>`: extend a completed run; by default create a derived run that shares unchanged artifacts through hard links. Use `--in-place` to update the existing run.
- `--in-place`: with `--extend-feature-run-id`, apply explicitly requested additions, refreshes, or replacements to the existing feature run. Incompatible with `--run-id`; the run ID and latest pointer are unchanged.
- `--overwrite-feature-run`: compatibility alias for `--in-place`. Both spellings normalize to the same argument and metadata mode; existing commands remain valid.
- `--num-workers <N|auto>`: parallelize match processing inside each `datatools/graph_feature.py` subprocess. Default: `1`.
- `--worker-thread-limit <N>`: set per-worker `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. Default: `1`.
- `--refresh-target-family {xt,goal_distance,epv}`: with `--extend-feature-run-id`, refresh targets from current sidecars without rebuilding graphs. Creates a derived run by default; `--in-place` selects the existing run. xT refresh must run separately from other operations; other target families may be repeated.
- `--pass-height`: with `--extend-feature-run-id`, overwrite copied label tensors so `pass_max_ball_z` and `pass_high` are available for `pass_height` training. The stored compatibility label uses the feature run's `--pass-height-threshold` (default `2.0`); training can derive a different label from `pass_max_ball_z` without copying this run.
- `--pass-height-threshold <meters>`: with `--pass-height`, set the feature run's stored compatibility-label cutoff. Default: `2.0` metres.
- `--replace-intended-receiver-model`: with `--extend-feature-run-id`, allow a different `--intended-receiver-model-id` when the base run already contains `model` artifacts; only model-mode artifacts are regenerated in the new derived run.
- `--next-action-conditions-on` / `--next-action-conditions-off`: keep or disable the pass/cross next-action consistency filter. Default: on. Derived runs must match the base run's setting.

### `scripts/train_relevant_models.py`

- `--sparsify {none,delaunay}`: connectivity for every selected training task. Default: `none` (all pairwise edges after node filtering). Resume and source refit use saved settings and reject this override.

When split flags are omitted, training infers the selector from feature-run metadata, recovering it from the recorded immutable manifest if needed. Explicit selectors are assertions and must match, including count versus percentage mode. Inconsistent modern provenance or a changed match-universe identity fails. Only runs with no split provenance assume the historical 50% split. The resolved selector, match counts, and source are printed and forwarded to training commands.

- `--min_pass_dur <seconds>`: minimum pass duration applied to every selected component. Default: `0.5` seconds.
- `--dataset-loading {auto,memory,disk}`: fresh runs default to prepared disk loading for all eight wrapper tasks, including IPW configurations. `memory` explicitly loads the full dataset into RAM. Training buffers four matches by default; validation loads one.
- `--batch-loader {auto,pyg}`: defaults to direct packed batching for prepared disk datasets, with PyG fallback for unsupported schemas. Full monitoring selects PyG; memory datasets use PyG.
- `--monitoring {off,light,on}`: defaults to `off`. `light` retains resource sampling and logs every 50 batches; `on` enables full telemetry and collation diagnostics.
- `--keep-cache`: retain prepared graph/label caches after success. Otherwise newly managed entries are removed after the complete workflow succeeds and all consumers finish. Failed/interrupted workflows, pre-existing entries and explicit retention requests protect caches. See [prepared dataset loading](docs/intent_dataset_loading.md).
- `--resume-id <task/run_id[/stage]>`: resume one unfinished model run with its saved training settings. A unique bare run id is also accepted. Accepts `--keep-cache`, `--batch-loader`, `--monitoring`, `--crash-dump`, `--procdump-path`, and matching pass-height or goal settings; incompatible overrides are rejected. Bundle ids are not accepted.
- `--use-carries`: train `action_intent`, `pass_success`, `outcome_scoring`, and `outcome_conceding` on the feature run's carry-augmented artifacts. Pass-only tasks retain canonical artifacts. Requires a feature run generated or extended with `--use-carries`. Default: off.
- `--train-split <percentage>`: optional assertion of the feature run's development percentage (1-99).
- `--train-count <int>`: optional assertion of the feature run's exact development match count, including validation. Mutually exclusive with `--train-split`.
- `--validation-mode {holdout,holdout_80_20,expanding}`: use one chronological holdout or explicitly sized nested training stages with a shared latest validation set. Default: `holdout`; the old alias requires 20% validation.
- `--validation-count N` / `--validation-fraction F`: mutually exclusive validation sizes within the saved development pool. Default fraction: `0.2`.
- `--training-counts N ...` / `--training-fractions F ...`: mutually exclusive training-size lists, excluding validation. Fractions use the remaining pool. Expanding requires a list; holdout accepts one size or defaults to the entire remaining pool.
- `--refit-source-model-id TASK/RUN[/STAGE]`: repeatable independent refit sources; implies refitting and skips validation. Saved source settings determine each job. Only operational overrides are accepted; no other CLI options are required. Outputs and job status are listed in a separate refit batch manifest.
- `--final-refit`: optionally refit all development games using the largest validation stage's best epoch and recorded per-epoch learning rates. No validation, early stopping, rollback, or minimum epoch requirement. See [validation and refitting](docs/configurable_validation.md).
- `--target-family {goal,xg,xt,goal_distance,epv}`: retained outcome family. Required when `outcome_scoring` or `outcome_conceding` is enabled.
- `--return_type <disc_gamma|disc_gamma_skip1|disc_max_gamma|disc_max_gamma_skip1|disc_poly_max_b_z|disc_poly_max_b_z_spstop|next_N|next_N_skip1|in_N>`: resolved return semantics for the selected label directory. `disc_max_gamma` and `in_N` are valid only for `xt`, `goal_distance`, and `epv`; polynomial max is valid only for `xt` and `goal_distance`. Required when an outcome model is enabled; otherwise the wrapper falls back to the first available return type in the selected feature run.
- `--feature-run-id <feature_run_id>`: pin the feature run used for validation training. Required unless sources are supplied for an independent refit batch.
- `--diagnostic-feature-run-id <feature_run_id>`: optional feature run containing compatible `action_labels_next_10*` labels for canonical goal-event diagnostics when the selected run lacks embedded `goal next_10` diagnostic columns.
- `--intended-receiver-mode {original,angle_only,model}`: intended-receiver mode used for retained-model training. Required when any of `action_intent`, `pass_intent`, `pass_success`, `pass_height`, `outcome_scoring`, `outcome_conceding`, or `failure_receiver` is enabled.
- `--only-action-intent`, `--only-pass-intent`, `--only-success-intent`, `--only-pass-success`, `--only-pass-height`, `--only-outcome-scoring`, `--only-outcome-conceding`, `--only-failure-receiver`: train exactly the selected models. Repeated flags are additive. They cannot be mixed with ordinary per-model toggles, and prerequisites such as an IPW `pass_intent` model are not enabled implicitly. The legacy `--success-intent-only` spelling remains accepted but is hidden from help.
- `--action-intent` / `--no-action-intent`, `--pass-intent` / `--no-pass-intent`, `--success-intent` / `--no-success-intent`, `--pass-success` / `--no-pass-success`, `--pass-height` / `--no-pass-height`, `--outcome-scoring` / `--no-outcome-scoring`, `--outcome-conceding` / `--no-outcome-conceding`, `--failure-receiver` / `--no-failure-receiver`: enable or disable individual wrapper-managed checkpoints. Default: on for all except `pass_height` and `failure_receiver`.
- `--pass-success-ipw` / `--no-pass-success-ipw`: enable or disable inverse propensity weighting for `pass_success` only. Default: enabled.
- `--pass-intent-model-id <pass_intent/model_run_id>`: existing compatible `pass_intent` checkpoint to use as the `pass_success` IPW model when `--pass-success-ipw --no-pass-intent` is set.
- `--pass-height-ipw` / `--no-pass-height-ipw`: enable or disable inverse propensity weighting for `pass_height`. Default: disabled.
- `--pass-height-threshold <meters>`: target cutoff for a new `pass_height` training run. Omit it to inherit the selected feature run's recorded cutoff. A resumed run inherits its checkpoint cutoff and rejects a conflicting explicit value.
- `--pass-height-ipw-model-id <pass_intent/model_run_id>`: existing compatible `pass_intent` checkpoint to use as the `pass_height` IPW model when `--pass-height-ipw --no-pass-intent` is set.
- `--ipw-batch-size <N>`: positive IPW inference batch size, independent of training batch size. Default: `256`.
- `--ipw-probability-cache {on,off}`: reuse per-match propensity probabilities during IPW preparation. Default: `on`.
- `--ipw-probability-cache-dir <path>`: location for the IPW probability cache. Default: `data/cache/ipw_probabilities` under the repository.
- `--batch-size <n>` / `--batch_size <n>`: override the wrapper batch size for every low-level model training command.
- `--action-intent-batch-size <n>`, `--pass-intent-batch-size <n>`, `--success-intent-batch-size <n>`, `--pass-success-batch-size <n>`, `--pass-height-batch-size <n>`, `--outcome-scoring-batch-size <n>`, `--outcome-conceding-batch-size <n>`, `--failure-receiver-batch-size <n>`: override one model's batch size. Model-specific flags override `--batch-size`.
- `--start_lr <float>` / `--min_lr <float>`: override starting/minimum learning rates for every selected model. Both must be finite and positive, with minimum no greater than start. Defaults: start `0.002` for intent, pass-success, pass-height, and failure-receiver models; start `0.0002` for outcome models; minimum `0.00001` for all models.
- `--lr-reduction-patience <int>`: positive number of consecutive validation-loss misses before reducing LR at the start of the next epoch. Default: `3`.
- `--lr-reduction-factor <float>`: multiply LR by this factor on reduction, clamping to `--min_lr`. Must be finite and strictly between `0` and `1`. Default: `0.5`.
- `--lr-rollback` / `--no-lr-rollback`: restore the best validation model and matching Adam state on LR reduction, or continue from the current model and optimizer state. Default: rollback enabled. The best validation checkpoint is saved under either policy.
- `--early-stopping-patience <int>`: stop after this many consecutive validation-loss misses once the minimum duration and post-reduction protection are satisfied. Positive integer; default: `10`.
- `--early-stopping-min-epochs <int>`: minimum epoch before early stopping can end a validation training stage. Positive integer; default: `30`.
- `--early-stopping-min-delta <float>`: minimum absolute validation-loss improvement required to reset the stopping and LR-reduction counters. Default: `0.00001`.
- `--early-stopping-min-epochs-after-lr-reduction <int>`: completed epochs required after the latest actual LR reduction before early stopping. Nonnegative integer; `0` disables protection. Default: `5`. The reduction epoch counts as the first; further reductions remain possible and renew protection. Being at the minimum rate does not renew it.
- `--no-early-stopping`: disable validation-loss early stopping; adaptive LR reductions remain active.
- `--max-epochs <int>`: hard epoch cap for each validation training stage, overriding minimum-duration and post-reduction protection requirements. Positive integer; default: `100`. Final refits use their source best epoch and recorded rates instead of this cap.
- `--seed <int>`: seed Python, NumPy, and PyTorch for every selected model, including final refits. Range: `0` through `2**32 - 1`; default: `100`.
- `--bundle-id <bundle_id>`: pin the training bundle manifest id.

Mask selectors below are shared with direct `train.py`. Node `SELECTION` values
are `all`, `attacking`, `defending`, `no-poss`, and `none`; edge selections are
`all`, `no-poss`, and `none`. Boolean controls accept `true`/`false`. Omitted
selectors preserve the current profile; [the defaults table](#training-input-controls-and-automatic-exclusions)
compares wrapper and direct training. Groups provide defaults and individual
features override them, independent of argument order.

- `--vx SELECTION`, `--vy SELECTION`: retain an individual node-velocity component on the selected players. Default: `all` in the wrapper, including the possessor.
- `--vel-node-features SELECTION`: group selection for `vx`, `vy`, `speed`, and `accel`. Default: `all`. For defenders' components only, use `--vel-node-features none --vx defending --vy defending`.
- `--speed SELECTION`, `--accel SELECTION`: individual scalar motion inputs. Default: `all`. These do not control `success_intent` pass-trajectory speed or stored edge movement features.
- `--pos-node-features SELECTION`: select learned `x`/`y` inputs. Default: `all`; original geometry remains available to graph processing, physical calculations, and destination features.
- `--possessor-id true/false`: retain/zero only the possessor indicator. Default: `true` in the wrapper; independent of possessor geometry and relative velocity angles.
- `--keeper-id true/false`, `--ball-z true/false`: retain/zero keeper identity or ball height. Both default to `true` in the wrapper.
- `--poss-geometry SELECTION`: possessor-relative distance/direction, preserving the separate identity setting. Default: retained on players; goal values additionally require `--goal-poss-geometry true`.
- `--poss-velocity-angles SELECTION`: cosine/sine of player velocity angle relative to the possessor. Default: `none`.
- `--goal-features SELECTION`: goal-relative distance/direction; does not remove goal nodes or suppress ball height. Default: `all`.
- `--goal-poss-geometry true/false`: allow/suppress possessor distance/direction on goal nodes. Default: `false` for new training; older checkpoints without the setting retain their original enabled behavior.
- `--offside SELECTION`: select the appended offside input when available. Default: retained when present.
- `--extended-node-features SELECTION`: select the six handcrafted features. Default: `none`; the separate `success_intent` trajectory block keeps its historical behavior.
- `--pass-lane-features SELECTION`: select `nearest_opponent_to_pass` and `potential_interceptors`. Default: `none`. This narrower group overrides the corresponding members of `extended-node-features`.
- `--velocity-angle-edge-features {all,no-poss,none}`: select stored velocity-angle edge inputs. Default: `none`.
- `--relative-speed-edge-features {all,no-poss,none}`: select raw relative-speed edge input. Default: `none`; requires velocity-angle inputs.
- `--position-direction-edge-features {all,no-poss,none}`, `--relative-velocity-edge-features {all,no-poss,none}`, `--separation-rate-edge-features {all,no-poss,none}`: select directed interaction groups. All default to `all`. `no-poss` zeros both incoming and outgoing possessor edges.
- `--goal-edge-movement true/false`: allow/suppress movement inputs on all goal-incident edges while retaining distance and positional direction. Default: `true`.
- `--lane-survival SELECTION`: enable/select the cached lane-survival input on supported node tasks. Default: `none`.
- `--lane-survival-mode {max,top_N,top_pass_N}`: choose lane-survival computation, for example `top_25` or `top_pass_10`. Defaults to `max` when enabled. The selected metric must exist in the cache. Older `--lane-survival [MODE]` and `--no-lane-survival` forms remain accepted.
- `--pc-xpass-id <run_id>`: pin the cache used for lane survival, for example `--lane-survival defending --lane-survival-mode top_pass_10 --pc-xpass-id <pc_xpass_run_id>`. Pass the version name without `data/pc_xpass/`; Sportec reads `data/pc_xpass/<run_id>/sportec`. Defaults to the latest pc-xPass version when lane survival is enabled. Cannot be combined with `--physical-cache-dir`. Sportec requires `match_physical_states_v3` and a canonical action manifest; older caches require regeneration into a new version.
- `--goal-nodes true/false`: retain/remove goal nodes and incident edges as GNN context. Default: `true` for all eight retained tasks. Independent of `--goal-features` and prediction candidates; legacy checkpoints keep their historical policy.
- `--action-intent-include-goals true/false`: include/exclude the action-intent shooting candidate. Default: `true` for new runs. `false` excludes shot-labelled training/evaluation examples and goal predictions; `true` requires input goal nodes. Other retained tasks always exclude goal predictions. Resume rejects conflicting changes.
- `--xy-only` / `--no-xy-only`: legacy preset, off by default. It zeros `7:12` and `13:19`, rather than retaining only x/y. Older negative/awareness flags remain accepted with their historical effects; see [legacy compatibility](#legacy-feature-flag-compatibility).
- `--use_physical_xpass` / `--use-physical-xpass`: enable physical xPass for `pass_success` only.
- `--model-variant {gat_baseline,gat_plus_phys_feature,gat_phys_logit_offset,gat_phys_logit_offset_regularized}`: choose the pass-success physical xPass architecture. Default: `gat_phys_logit_offset`.
- `--physical-cache-dir <path>`: physical xPass sidecar directory override. Default: `<feature_run_root>/physical_xpass`.
- `--physical-eps <eps>`: physical probability clamp epsilon. Default: `1e-4`.
- `--physical-xpass-floor <prob>`: optional lower physical xPass floor before logit conversion. Default: unset.
- `--freeze-beta1`: freeze `beta1` at `1.0`. Default: learned.
- `--freeze-beta0`: freeze `beta0` at `0.0`. Default: learned.
- `--residual-regularization-lambda <value>`: optional observed-target residual L2. Default: `0.0`.
- `--residual-clip-value <value>`: optional residual clipping bound. Default: unset.
- `--residual-distance-threshold <meters>`: short/long residual cutoff. Default: `30.0`; short is `<= threshold`, long is `> threshold`.
- `--short-residual-regularization-lambda <value>` / `--long-residual-regularization-lambda <value>`: optional short/long L2 overrides.
- `--short-residual-clip-value <value>` / `--long-residual-clip-value <value>`: optional short/long clipping overrides.
- `--outcome-scoring-trial <n>` and `--outcome-conceding-trial <n>`: override the auto-generated run ids for those tasks with legacy numeric ids.

Training controls apply to every selected model and are saved in arguments and metadata. Resume restores saved controls and the latest reduction-start epoch; training-control overrides are rejected with `--resume-id`. Unfinished checkpoints from different training code remain subject to the existing fingerprint check. See [Training controls, checkpoints and monitoring](docs/training_runtime.md) for details.

Example with explicit training controls:

```bash
python scripts/train_relevant_models.py --feature-run-id <feature_run_id> --only-success-intent --lr-reduction-patience 3 --lr-reduction-factor 0.5 --early-stopping-min-epochs-after-lr-reduction 5 --no-lr-rollback --max-epochs 60 --seed 42
```

### `train.py`

`train.py` is the low-level training entrypoint normally launched by `scripts/train_relevant_models.py`. Use the wrapper for new and resumed runs because it restores the complete saved configuration.

- Feature/group selectors use the [same names and values as the training wrapper](#training-input-controls-and-automatic-exclusions), including `--vx`, `--vy`, `--vel-node-features`, the five edge-group flags, and `--lane-survival` / `--lane-survival-mode`. Direct training preserves its own default profile: keeper/ball-height/possessor awareness are off, unlike the wrapper. Omitting selectors does not replace that profile.
- `--possessor-id true/false`, `--keeper-id true/false`, `--ball-z true/false`, `--goal-poss-geometry true/false`, `--goal-nodes true/false`, and `--goal-edge-movement true/false`: the same independent input/policy controls as the wrapper; see the defaults table.
- `--include-goals true/false`: direct action-intent candidate control, corresponding to wrapper `--action-intent-include-goals true/false`. Default: `true` for new action-intent runs; goals cannot be candidates for other retained tasks. Input goal nodes must be enabled when goal predictions are enabled.
- Legacy underscored awareness flags, negative masks, and `--xy_only` remain supported; see [legacy compatibility](#legacy-feature-flag-compatibility).

- `--max-epochs <int>` / `--n_epochs <int>`: mutually exclusive aliases for the hard epoch cap. Default: `200` for direct launches; the wrapper supplies its own cap or selected refit duration.
- `--seed <int>`: random seed from `0` through `2**32 - 1`. Default: `128` for direct launches.
- `--lr-reduction-patience`, `--lr-reduction-factor`, `--lr-rollback` / `--no-lr-rollback`, and `--early-stopping-min-epochs-after-lr-reduction`: same meanings and defaults as the wrapper above (`3`, `0.5`, rollback enabled, and `5`).
- `--early-stopping-patience`, `--early-stopping-min-epochs`, `--early-stopping-min-delta`, and `--no-early-stopping`: same meanings and defaults as the wrapper above (`10`, `30`, `0.00001`, and stopping enabled).
- `--start_lr <float>` / `--min_lr <float>`: starting/minimum rates. Both default to `0.0001` for direct launches; the wrapper supplies its model-specific defaults or overrides.

- `--resume-run-id <run_id>`: legacy resume selector. It is retained for CLI compatibility but is intentionally rejected; use `scripts/train_relevant_models.py --resume-id <task/run_id>` instead.
- `--pass-height-threshold <meters>`: target cutoff for low-level new training. It is normally supplied by the training wrapper; see [Shared pass-height target option](#shared-pass-height-target-option).

### `scripts/evaluate_relevant_models.py`

Evaluation-only overrides leave checkpoint training provenance unchanged:

- `--feature-run-id <id>` selects evaluation graphs, the primary label variants required by each checkpoint, and default diagnostic labels. It is supported by ordinary evaluations, model comparisons, masks, learning curves, and direct `test.py`. Without it, each checkpoint retains its recorded input run. This does not change weights, native target definitions, or historical training membership.
- `--diagnostic-feature-run-id <id>` remains an optional diagnostic-label override for supported tasks. It never selects input graphs. When both flags are supplied, primary data comes from `--feature-run-id` and diagnostic labels come from the diagnostic override. With only an explicit primary run, diagnostics must come from that run; archived diagnostic paths are not used as fallback. With neither flag, existing checkpoint defaults are retained.

Before any evaluator starts, all requested matches are checked for the required graph and label files, and model/auxiliary graph requirements are validated. Missing artifacts stop the complete job with a consolidated error; the selected cohort is never reduced automatically. The selected data run may have a different split from the checkpoint: historical membership and held-out checks remain based on the checkpoint's records. Saved `evaluation_data` metadata records requested/resolved sources, actual paths, label definitions, and the original training feature-run ID. `--evaluation-output-dir` still selects result destinations, and pc-xPass caches remain independently selected and validated.

For a historical model whose original input run is unavailable, add an available compatible run:

```powershell
python scripts/evaluate_relevant_models.py `
  --compare `
  --pass-intent-model-id pass_intent/pass_intent_20260617T003107_673410_04a90875 `
  --pass-intent-model-id pass_intent/pass_intent_20260812T005214_787091_6fe83a18 `
  --feature-run-id <compatible_run> `
  --pc-xpass-cache-dir 'C:\dm_model\data\pc_xpass_\sportec' `
  --season 24_25 --season-half second --filter pass --min-pass-duration 0.5
```

The selected run must provide both required label variants (`disc_0.7` and `next_10` in this example), compatible input graphs, and the requested matches. Selecting a run does not regenerate missing labels.

- `--intended-receiver-mode original|angle_only|model` selects a common receiver reference; omission retains each checkpoint's resolution.
- `--diagnostic-return-type next_3` changes the common binary goal diagnostic (default `next_10`). Supported definitions are `next_<N>` and undiscounted `disc_1` equivalents, optionally `_skip1`; continuous diagnostics are rejected. Native continuous-target metrics remain available.
- `--filter pass` evaluates only rows with `is_pass == 1`, including when a checkpoint uses carry-inclusive artifacts. Omission retains the checkpoint's action population. The filter applies before inference to all metrics, diagnostics, exports, and bootstrap intervals; it does not alter graphs or targets. Filtered-out samples are counted separately from unusable rows.
- `--min-pass-duration SECONDS` applies a common finite, nonnegative pass-duration filter, retaining passes with duration greater than or equal to the threshold. Zero removes the minimum; omission retains checkpoint settings. Combine it with `--filter pass` to compare passes under a shared duration threshold. Requested and effective thresholds, the filter, and selection counts are recorded in evaluation metadata and each experiment-manifest entry.
- `--season 24_25 --season-half second` explicitly selects matchdays 18?34. Omit the half for the full season; `first` selects 1?17. Seasons can be repeated. Requested matches are never silently intersected with existing test lists. Training/validation overlap and unavailable required artifacts fail, including for participating learned auxiliaries. Missing historical exclusion evidence requires the separate interactive confirmation described below.

Statistical and visualization comparisons explain material limitations and ask for explicit `yes` or `no`; empty answers do not accept warnings. `--accept-comparison-warnings` accepts them for unattended execution. Warnings and acceptance are saved in run metadata. This option never bypasses statistical action-alignment or held-out exclusion errors. Season, receiver-reference, diagnostic-horizon, pass-filter, and duration overrides cannot be combined with learning-curve mode. The artifact source selector `--feature-run-id` is supported; recorded training sizes, splits, and series-consistency requirements remain enforced.

- Task-specific model selectors are repeatable. Without `--compare` or masks, every specified checkpoint is evaluated independently, including several checkpoints of one task and different tasks sharing a run identifier. Explicit selections replace the corresponding task's bundle entry.
- `--compare`: compare at least two completed checkpoints of one task. Use repeatable task-specific selectors; there is no generic `--model-id`. Feature configurations, architectures, seeds, and training sizes may differ. Comparison requires identical evaluated action identities. Different targets, receiver resolution, and intent candidate membership produce warnings requiring explicit acceptance; target differences are qualified per metric. Training and validation overlap remains an error.

New prediction exports align by `(match_id, action_index)` even when artifact row positions differ. The original `source_index` remains available for tracing each artifact. Comparisons of older exports without event IDs retain source-row alignment; mixing old and new identity formats requires re-exporting predictions. Event sets must match exactly; no intersection is performed. Native continuous targets are compared with `rtol=1e-6, atol=1e-8`; binary goal labels and execution branches are compared exactly.

For the July and October outcome models, use a shared pass population:

```powershell
python scripts/evaluate_relevant_models.py --compare --filter pass --min-pass-duration 0.1 --outcome-scoring-model-id outcome_scoring/outcome_scoring_20260731T153011_091203_ec216b1d --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261016T100710_860045_d961f19d --diagnostic-feature-run-id feature_20261015T011549_082721_d5ab6c0a --season 24_25 --season-half second
```

- `--inference-mask [TASK:|all:]FEATURE[=SELECTION] ...`: compare each selected checkpoint without additional masks against one combined inference configuration. Bare canonical names mean `=none`; explicit selectors such as `vx=defending` retain only the selected subset of the checkpoint's normal inputs. Repeating the flag adds selections to that configuration. See [shared inference masks](#shared-inference-masks) for scope, precedence, and legacy aliases.
- `--inference-mask-1 [TASK:|all:]FEATURE[=SELECTION] ...`, `--inference-mask-2 ...`: define independent configurations using the same selectors, compared with normal checkpoint inference separately for each selected checkpoint. Numbered identifiers must be positive and unique; do not mix numbered and unnumbered forms. Multiple tasks/checkpoints are allowed, but masked evaluation never compares different checkpoint runs. Masks take precedence over an additional `--compare` flag.
- `--list-inference-masks`: list supported groups, individual features, expansions, selectors, and bare-name defaults without loading a checkpoint.
- `--comparison-bootstrap-resamples <N>` and `--comparison-bootstrap-seed <N>`: comparison whole-match bootstrap settings. Defaults: 2000 and 42. Zero resamples disables intervals; otherwise at least two contributing matches are required. Reports include all within-group pairwise differences and pointwise 95% intervals.
- `--compare` and inference masks cannot be combined with learning-curve mode. Combined-v5 evaluation requires exactly one explicit pass-intent checkpoint as its shared auxiliary dependency.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to evaluate.
- `--learning-curve`: compare completed subset checkpoints on the same recorded test set. With a bundle or any existing `--<task>-model-id`, use its explicit bundle stage list, or discover completed holdout/expanding stages and the promoted final refit. Tasks with fewer than two sizes are reported as skipped. The default single-model evaluation is unchanged.
- `--learning-curve-model-id <task/run_id[/fold_N]>`: repeat to select checkpoint series directly for a task; these entries replace automatic discovery for that task. At least two distinct nested training sizes are required. Results include aligned predictions, match-bootstrap intervals and paired differences under `data/evaluation_runs/learning_curve_<timestamp>/` by default.
- `--learning-curve-bootstrap-resamples <N>` and `--learning-curve-bootstrap-seed <N>`: paired whole-match bootstrap settings for learning curves. Defaults: 2000 resamples and seed 42.
  Example: `python scripts/evaluate_relevant_models.py --learning-curve --outcome-scoring-model-id outcome_scoring/<run_id> --outcome-conceding-model-id outcome_conceding/<run_id>`.
- At least one explicit model ID is required when `--bundle-id` is omitted. Bundle evaluation includes every supported task present in the bundle, while explicit IDs supplement or override its entries.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id.
- `--pass-height-threshold <meters>`: explicitly relabel observed pass-height targets and height-based diagnostics for every selected checkpoint. By default each checkpoint is evaluated against its saved cutoff; predictions always retain their saved meaning.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--diagnostic-feature-run-id <feature_run_id>`: optional diagnostic feature run passed to `test.py` for outcome models.
- `--device <device>`: device passed to `test.py`. Default: `cuda:0`.
- `--weighted-pass-success-metrics`: additionally report high-pass-weighted ROC-AUC and Brier score for the
  learning-based `pass_success` checkpoint. Requires a compatible physical PC-xPass cache for distances and an
  explicit `--xpass-pass-height-model-id` auxiliary. Component height models are evaluated independently.
- `--evaluate-xpass`: evaluate cached receiver-specific physical pc-xPass probabilities with the same pooled and
  observed-pass-height-stratified pass-success metrics. Requires a selected `pass_success` model and either `--top-pass N` or `--xpass-version`.
- `--evaluate-combined-success`: evaluate the deployed blend of the learning-based pass-success prediction and cached
  physical pc-xPass. Requires a selected `pass_success` model, either `--top-pass N` or `--xpass-version`, and `--xpass-weight`.
- `--top-pass N` / `--top_pass N`: directly select one cached pc-xPass top-pass count to evaluate, for example `--evaluate-xpass --top-pass 5`. No `--xpass-version` flag is needed.
- `--xpass-version <max|topN|top-passN>`: alternative cached pc-xPass metric selector, for example `top25`. Supply this or `--top-pass N` for
  `--evaluate-xpass` and `--evaluate-combined-success`; the requested metric must be present in the read-only cache.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}`: blend weighting rule for `--evaluate-combined-success`. `v4` also requires explicit
  `--discount`, `--v4-power`, and `--v4-zero`; those arguments are rejected for other combined-weight versions.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for combined-v5 evaluation. Default: `0.01`; requires the explicit height and intent auxiliary selectors.
- `--v5-discount <true|false>`: apply the pass-intent discount during combined-v5 evaluation. Default: `true`; `false` uses `w = pass_height`. Both v5 options are valid only with `--evaluate-combined-success --xpass-weight v5`.
- `--pc-xpass-cache-dir <path>`: pc-xPass dataset directory for lane-survival inputs, weighted pass-success evaluation,
  physical-xPass evaluation, or combined pass-success evaluation. Use the Sportec directory containing `metadata.json`,
  for example `C:\dm_model\data\pc_xpass_\sportec`. The wrapper forwards this selection to every checkpoint evaluator,
  including model comparisons, mask evaluations, and learning curves. Lane-survival caches must match the checkpoint's
  recorded fingerprint.
- `--pc-xpass-id <id>`: select a versioned pc-xPass cache instead of a directory override; the two selectors cannot be
  combined. When a cache is required and neither selector is supplied, the wrapper resolves the latest Sportec version
  once and passes that ID to each evaluator.
- Historical checkpoints missing training/validation membership can be evaluated after an interactive `yes` confirmation.
  When no recorded default test cohort exists, select one with `--season` (and optionally `--season-half`). The wrapper
  checks all primary and auxiliary models, asks once, and passes the recorded approval to its evaluators. Direct `test.py`
  evaluations offer the same confirmation. Known overlap, invalid/conflicting provenance, broken recorded manifests,
  missing feature artifacts, and incompatible caches remain errors. `--accept-comparison-warnings` does not authorize
  unverified membership, and no public unattended-acceptance flag is provided.
  Accepted results retain `evaluation_membership` metadata; comparison CSVs include `evaluation_membership_status`,
  and plots identify unverified evaluation membership. These results are exploratory assessments, not verified held-out
  measurements. Refusal, EOF, interrupted input, or noninteractive execution cancels with exit status 1; standalone
  cancellations are recorded in `membership_cancellation.json` when an output directory was requested.
- `--discount {true,false}`, `--v4-power <float>`, and `--v4-zero <float>`: v4 production-weight settings for
  weighted pass-success evaluation (defaults: `true`, `4`, and `0.7`) or required explicit settings for combined-v4
  evaluation.

### `test.py`

`test.py` is the low-level single-checkpoint evaluator normally launched by `scripts/evaluate_relevant_models.py`.

- `--filter pass` and `--min-pass-duration SECONDS` have the same evaluation-only behavior as the wrapper options above.
- `--pass-height-threshold <meters>`: evaluate with relabelled observed height targets and cutoff-relative diagnostics. Omit it to use the checkpoint definition; this never changes the event predicted by the checkpoint.

### Shared inference masks

Evaluation, component inference, Hawkeye, action-component visualization, and EPV
generation accept `[TASK:|all:]FEATURE[=SELECTION]` after `--inference-mask`.
`scripts/main.py` forwards these inference selections to learned evaluation and
inference stages, including EPV generation; it does not change training or feature
generation. Bare canonical names default to `=none`. Omitting inference masks
entirely preserves normal checkpoint inference, including its saved training masks.

Node selectors are `all`, `attacking`, `defending`, `no-poss`, and `none`, with the
[same retained-player meanings as training](#training-input-controls-and-automatic-exclusions).
Edge selectors are `all`, `no-poss`, and `none`; boolean-only input and goal controls
accept `all`/`none` or `true`/`false`. `all` retains whatever inputs the checkpoint
normally receives; it does not enable features excluded during training.

```powershell
# Compare no additional masks against two independent motion configurations.
python scripts/evaluate_relevant_models.py --pass-success-model-id pass_success/run_a --inference-mask-1 all:vel-node-features --inference-mask-2 all:vel-node-features=defending

# Keep only defenders' node vx/vy; remove scalar speed/acceleration everywhere.
python scripts/run_and_visualize_hawkeye.py --bundle-id BUNDLE --situation-id SITUATION --inference-mask all:vel-node-features all:vx=defending all:vy=defending

# Restrict only the pass-success component; leave other participating tasks normal.
python scripts/run_and_visualize_hawkeye.py --bundle-id BUNDLE --situation-id SITUATION --inference-mask pass_success:vel-node-features=no-poss
```

Use `TASK:` to select a component or `all:` for every participating model task,
including auxiliaries. Unscoped names are allowed only with one participating
task. A configuration scoped away from a checkpoint and its dependencies is
skipped in evaluation and recorded in the manifest. Outside evaluation, an
unnumbered configuration runs one masked inference pipeline. The two inference
visualizers also accept numbered configurations for PNG comparison; other inference
scripts accept only the combined unnumbered form.

Explicit selectors replace bare shorthand for the same feature/group, independent
of order: `vx vx=defending` is the same as `vx=defending`. Broad groups provide
defaults and narrower groups/individual features override them. For example,
`vel-node-features vx=defending vy=defending` suppresses all motion columns except
defenders' `vx`/`vy`. Conflicting explicit values for the same feature/group are
rejected; identical assignments are deduplicated. Numbered configurations remain
independent; do not mix numbered and unnumbered forms.

`--list-inference-masks` lists groups, semantic columns, expansions, accepted
selectors, and bare-name defaults without loading checkpoints. Canonical names
match the training interface:

| Name | Learned inputs selected |
| --- | --- |
| `pos-node-features` | Node x/y |
| `vel-node-features` | Node vx, vy, speed, and acceleration |
| `vx`, `vy`, `speed`, `accel` | Individual motion input |
| `goal-features` | Goal distance and normalized goal direction |
| `poss-geometry` | Possessor distance and normalized possessor direction |
| `poss-velocity-angles` | Cosine/sine of velocity angle relative to possessor |
| `possessor-id`, `keeper-id`, `ball-z` | Only the named identity or ball-height input; boolean selections |
| `offside` | Appended offside indicator |
| `extended-node-features`, `pass-lane-features`, `lane-survival` | Corresponding stored handcrafted or sidecar inputs |
| `velocity-angle-edge-features`, `relative-speed-edge-features` | Stored velocity-angle or relative-speed edge inputs |
| `position-direction-edge-features`, `relative-velocity-edge-features`, `separation-rate-edge-features` | Directed interaction edge inputs |
| `goal-poss-geometry` | Possessor geometry on goal nodes only; boolean selections |
| `goal-edge-movement` | Movement inputs on goal-incident edges only; boolean selections |
| `motion` | Node motion, possessor-relative velocity angles, velocity-angle/relative-speed/relative-velocity/separation-rate edges; `all`, `no-poss`, or `none` |

Individual schema-qualified names such as `node.vx`, `node.goal_dist`, and
`edge.relative_speed` remain supported, including explicit selectors. Unknown
names, absent required features, and ambiguous schemas fail validation. Inputs
already excluded during training or unused by the architecture are recorded as
having no additional effect. Structural options such as removing goal nodes are
not inference feature masks.

Inference selections **only further restrict saved training inputs**. For example,
`vx=defending` on an attacking-only checkpoint leaves zero `vx` everywhere. New
selectors also work with older checkpoints: the resolver reconstructs their legacy
settings and schema, with no retraining or rewriting required.

Legacy inference aliases retain their historical effects:

| Legacy name | Canonical selection or effect |
| --- | --- |
| `v-edge-features` | `velocity-angle-edge-features=none` |
| `poss-rel-vel-aware` | `poss-velocity-angles=none` |
| `ball-z-aware`, `keeper-aware`, `extend-features` | `ball-z=none`, `keeper-id=none`, `extended-node-features=none` |
| `poss-vel-aware` | `vel-node-features=no-poss` |
| `v-edge-features-no-poss`, `mask-possessor-v-edge-features` | `velocity-angle-edge-features=no-poss` |
| `relative-speed-edge-features-no-poss`, `mask-possessor-relative-speed-edge-features` | `relative-speed-edge-features=no-poss` |
| `possessor-aware` | Remove all six legacy possessor-awareness inputs (`13:19`), not just identity |

The legacy subset aliases above retain their exclusions when bare; the `=none`
shorthand rule applies to canonical names. Old saved visualization metadata remains
readable. Equivalent shorthand and explicit canonical forms share a configuration
identity; metadata records original requests, canonical selections, operations,
and effectiveness.

Masking occurs after feature derivation at encoder and decoder learned-input
boundaries, including skip connections and grid decoding. Original geometry and
identities still determine graph processing, candidates, rendering, and physical
or cache inputs. Graph topology and tensor widths remain unchanged. Masking x/y
measures reliance on those columns while other spatial inputs remain independently
controlled; likewise, node-velocity selection leaves velocity-derived edges unless
those are selected separately.

Physical xPass values remain unchanged. Masked learned predictions can change the
blended result, and masked pass-intent predictions may change v5 blending weights.
Evaluation exports and summaries distinguish inference configurations. Ordinary
Hawkeye inference retains its layout; PNG comparison layouts are described below.
For the complete shared contract, see [feature selections](docs/feature_selections.md).

### Carry availability in inference visualizations

`scripts/run_and_visualize_hawkeye.py` and `scripts/visualize_action_components.py`
allow a pass-success checkpoint with `use_carries=False` alongside carry-trained
action-intent and outcome checkpoints. In each frame, the ball carrier's candidate
represents the carry option, including synthetic Hawkeye pass frames. Its
`pass_success`, optional `model_pass_success`, and derived `pass_score` values are
unavailable when the success checkpoint was trained without carries. Teammate
pass predictions and carry-trained outcome values remain available. This masking
also applies when all selected checkpoints have `use_carries=False`.

Ordinary visualizations emit a warning; comparisons retain explicit warning
acceptance through an interactive answer or `--accept-comparison-warnings`.
Metadata records warnings and per-checkpoint `source_use_carries`. Sportec uses
carry-inclusive runtime artifacts when any selected carry-sensitive checkpoint
requires them. Comparison exports retain unsupported values as null/`N/A`.
Other carry-setting mismatches and default model consistency validation remain
strict; raw inference predictions are unchanged.

### PNG prediction comparison visualizations

All five visualization commands support pc-xPass cache comparisons with `--compare`
and repeated **singular** `--pc-xpass-id` selectors. Select at least two distinct IDs;
IDs retain their selection order. Every cache uses the same metric, blending settings,
auxiliary checkpoints, and (for inference commands) common unnumbered mask. `--pc-xpass`
is implied. Cache comparisons cannot also vary checkpoints or component runs, and
reject cache-directory overrides, `--no-physical-cache`, numbered masks,
`--compare-pass-success`, `--only-compare-pass-success`, and tracking-only output.
They support PNG only and export the usual aligned prediction tables and per-variant
cache provenance (`comparison_mode: pc_xpass`). Missing cache inputs fail with the
cache ID and scene; no fallback cache is selected and no cache is generated.

The inference visualizers `run_and_visualize_hawkeye.py` and
`visualize_action_components.py` additionally accept repeated `--xpass-version`
(including `uncertainty`) or repeated `--xpass-weight` with one explicit
`--pc-xpass-id`. Exactly one of cache ID, version, or weight may vary.
`--top-pass N` / `--top_pass N` adds `top-passN` to the ordered version list and
can be mixed with `--xpass-version`. Duplicate normalized values and repeated
selectors without `--compare` are rejected. Unvaried defaults remain `top10` and
`v3`; `--pc-xpass` is implied for all pc-xPass comparisons.

Weight comparisons require explicit `--use-physical-xpass` and a final
pass-success or pass-score output. v4/v5 require one explicit height auxiliary;
v5 also requires intent. Auxiliaries load once and bind only to variants needing
them; v4/v5 tuning flags apply to their respective variants. Raw physical panels
stay identical when only weights vary. Metadata adds `comparison_dimension`
(`cache_id`, `version`, or `weight`) and each variant's effective settings.
All existing comparison restrictions still apply. See
[pc-xPass comparison details](docs/pc_xpass_caches.md) for examples and validation.
Saved-component visualizers retain cache-only comparisons; `--uncertainty`
remains a generation option, while inference uses `--xpass-version uncertainty`.

```powershell
# Compare raw cached probabilities without loading models or a component source.
python scripts/visualize_hawkeye.py --compare `
  --pc-xpass-id cache_a --pc-xpass-id cache_b `
  --situation-id SITUATION --only-physical-xpass --top-pass 10

# Recompute blended success and pass score with one fixed model bundle.
python scripts/run_and_visualize_hawkeye.py --compare `
  --pc-xpass-id cache_a --pc-xpass-id cache_b --bundle-id BUNDLE `
  --situation-id SITUATION --use-physical-xpass --xpass-version top25 --xpass-weight v3 `
  --only-pass-success --only-pass-score --show-physical-xpass

# Compare max, top-pass10, and uncertainty within one cache.
python scripts/run_and_visualize_hawkeye.py --compare --pc-xpass-id CACHE_ID `
  --xpass-version max --top-pass 10 --xpass-version uncertainty `
  --situation-id SITUATION --only-physical-xpass

# Compare v5 and v4 with the same metric and fixed auxiliary checkpoints.
python scripts/run_and_visualize_hawkeye.py --compare --pc-xpass-id CACHE_ID `
  --xpass-version max --xpass-weight v5 --xpass-weight v4 `
  --use-physical-xpass --pass-success-model-id pass_success/MODEL `
  --xpass-pass-height-model-id pass_height/HEIGHT --xpass-pass-intent-model-id pass_intent/INTENT `
  --situation-id SITUATION --only-pass-success
```

Inference commands recompute blended pass success and dependent pass score when
blending is enabled. Each cache also supplies lane-survival inputs to participating
models and auxiliaries, so those learned predictions may change with a fixed checkpoint;
existing lane-survival compatibility checks still apply. Raw physical panels use
`--show-physical-xpass` or `--only-physical-xpass` and are unaffected by blend weights.
Saved visualizers resolve one fixed component source using their usual source-selection
rules, inherit its metric unless overridden, and change only the physical panel;
saved pass success and pass score are not recomputed. Physical-only output can run
without component predictions; Hawkeye location/freeze modes still require their
saved geometry source and use the `hawkeye_loc` cache namespace. Evaluation commands
retain their existing single-cache behavior.

Saved-component visualization supports `--compare` with repeated `--component-run-id` in
`scripts/visualize_hawkeye.py`, `scripts/visualize_skillcorner.py`, and `scripts/visualize_benchmark.py`.
Select at least two distinct completed runs. These scripts display stored predictions and reject all
inference-mask arguments. Repeated component IDs require `--compare`; comparison cannot use
`--component-dir` or tracking-only output.

```powershell
python scripts/visualize_hawkeye.py --compare `
  --component-run-id hawkeye_run_a --component-run-id hawkeye_run_b `
  --situation-id SITUATION --only-pass-intent --only-pass-score
```

The inference visualizers `scripts/visualize_action_components.py` and
`scripts/run_and_visualize_hawkeye.py` support two separate modes:

| Options | Behavior |
|---|---|
| Unnumbered `--inference-mask` alone | One masked prediction pipeline; no comparison |
| `--compare` and repeated same-task model IDs | Compare checkpoint predictions |
| `--compare` and equally repeated scoring/conceding model IDs | Compare ordered outcome checkpoint pairs |
| `--compare`, repeated same-task model IDs, and unnumbered masks | Compare masked predictions; each checkpoint runs once, with no unmasked variant |
| One or more numbered masks | One unmasked pipeline, then each independent numbered configuration |

```powershell
# Compare two checkpoints, applying the same mask to both.
python scripts/run_and_visualize_hawkeye.py --compare `
  --pass-success-model-id pass_success/run_a --pass-success-model-id pass_success/run_b `
  --inference-mask all:vel-node-features=defending --situation-id SITUATION --only-pass-success

# Compare two complete outcome configurations. First scoring pairs with first conceding.
python scripts/run_and_visualize_hawkeye.py --compare `
  --outcome-scoring-model-id outcome_scoring/run_a --outcome-conceding-model-id outcome_conceding/run_a `
  --outcome-scoring-model-id outcome_scoring/run_b --outcome-conceding-model-id outcome_conceding/run_b `
  --situation-id SITUATION --only-outcome-success --only-outcome-failure

# Compare each numbered configuration against normal checkpoint inference.
python scripts/visualize_action_components.py --match-id MATCH --action-id 10 `
  --pass-success-model-id pass_success/run_a --only-pass-success `
  --inference-mask-1 all:vel-node-features --inference-mask-2 all:vel-node-features=defending
```

Alternative checkpoints must vary one model task or the scoring/conceding outcome pair. Paired
comparison requires equal-length lists with at least two entries each, paired by selection order;
no combinations are generated automatically. A checkpoint may appear in several distinct pairs,
but duplicate complete pairs are rejected. To hold conceding fixed, repeat its ID explicitly for
each scoring checkpoint. Both outcome tasks must contribute to the selected outputs.
When varying an outcome model, derived `outcome_success`, `outcome_failure`, and `pass_score`
require explicit pairs; a single counterpart or bundle default cannot supply the other compared
checkpoint. Use `--only-outcome-scoring` or `--only-outcome-conceding` to compare one model's raw
outputs. Comparing raw scoring and conceding outputs together also requires explicit pairs.
Other model tasks remain fixed during paired comparisons. Fixed supporting models can be selected
individually or through a bundle; explicit selections override that task's bundle entry. Comparing
another task, such as pass success, can retain a fixed outcome pair. Numbered
masks require one checkpoint per participating task and reject `--compare`, repeated alternative
checkpoints, and mixed numbered/unnumbered masks. Numbers must be positive and unique, and are
processed numerically. Unscoped masks require one participating task, including dependencies;
otherwise use `task:name` or `all:name`. Existing evaluation CLI behavior is unchanged.

Comparison output is PNG only. Hawkeye, SkillCorner, and action-component comparisons use two
columns in selection order, wrapping onto new rows; odd counts leave the final slot blank. Benchmark
keeps game states 1 and 2 as rows, with all compared runs arranged horizontally as columns.
The full-image title contains the scene identifier, time where available, and component. Panel
titles identify component-run IDs, qualified model IDs, or `Unmasked` / `mask-<n>: <inputs>`.
Common unnumbered masks appear in the panel labels. Long labels wrap without changing pitch sizes.
Outcome-pair labels show both full checkpoint IDs; variant IDs encode both IDs independently of
selection position. Known target-family or return-horizon differences within a pair used for derived
outputs require the existing warning acceptance, as do differences between complete variants.
Each variant renders with its own runtime context and geometry, as in standalone visualization.
Different receiver modes, target definitions, schemas, geometry, and scene membership produce
warnings rather than compatibility errors. Different image sizes are padded without distortion.
The union of independently selected scenes is displayed, with unavailable panels where needed.
Candidate ordering is aligned by semantic identity when possible, and missing values remain `N/A`.
Ambiguous numerical exports do not prevent images: native predictions are retained instead.
Standalone input/rendering failures still fail the affected variant.

Comparison PNGs are saved progressively, after the required results for the current action,
situation, or possession are available. Benchmark images wait for both game states of the current
modification. The terminal prints the output directory at startup and progress after each group.
New comparison warnings require acceptance before the affected images are saved; accepting an
issue also covers later occurrences of the same issue between the same variants and component.
Frame IDs and changing roster counts alone do not trigger another confirmation. Detailed occurrences
remain in the metadata. Earlier images remain available if a later warning is declined, the run
fails, or it is interrupted. Metadata is updated during the run; prediction tables are finalized
when it completes.

Each comparison run retains scene/component PNG paths and, where semantic alignment is possible, exports `aligned_predictions.csv`,
separate `variants/variant_<hash>/predictions.parquet` files, and `metadata.json`. CSV columns use
`prediction:<variant-id>`; tables retain full precision and candidates not displayed on the pitch.
Metadata records source runs/checkpoints, fixed dependencies, requested/resolved masks, effectiveness,
selected frames, geometry context, warnings, and artifact paths. No difference columns or statistical
comparison metrics are produced. Failed comparisons record a failed status; declined warnings record a cancelled status.

Inference masks measure reliance on learned inputs, not performance after retraining without those
features, and may create unfamiliar inputs. Physical inputs remain unchanged; masked pass-intent
predictions may still change v5 blending weights and downstream pass scores.

### `scripts/run_relevant_models.py`

Sportec reconstructs inference spells from canonical actions and synchronized
control events. Opponent receptions and supported claims/tackles can start spells;
identified passing restarts are single-frame spells. Partial or ambiguous spells
retain their known observations and leave unknown boundaries missing. This does
not change training carry definitions, segments, eligibility, or labels. Feature
runs remain required.

An explicit `--feature-run-id` takes precedence. Otherwise, when the pass-success
checkpoint requires physical xPass or `--use-physical-xpass` is supplied, and
physical caching is enabled, component inference uses `source_inputs.feature_run_id`
from the selected Sportec cache's `metadata.json`. Use a cache directory override
or a pc-xPass cache ID (which defaults to latest); combining an ID and directory
override remains invalid. The cache's
feature run must pass the selected models' graph and resolved-action validation;
missing metadata, unavailable features, or incompatibility stop inference without
falling back to another feature run. Select a compatible cache or supply an
explicit feature run to resolve selection failures. Strict provenance and frame
coverage checks still apply to every selected cache, including with an explicit
feature run. Other carry, split, and optional-model checks also remain in effect.
Without physical xPass, or with `--no-physical-cache`, automatic selection retains
the newest compatible model/bundle source run, searching other runs only if none
of those qualify. The chosen ID and selection source are printed before inference;
cache-based selection is recorded as `runtime_feature_run_selection: "physical_xpass_cache"`.

- `--scope {actions,frames}`: select exact spell endpoints and preserved canonical observations (default: `actions`), or add strided states in supported intervals.
- `--frames N`: positive integer frame-ID stride, valid only with `--scope frames`; default 1.

Both scopes evaluate all applicable state-based components, including action
intent at possession start. Event-specific success intent remains event-based.
Outputs identify each physical state and its roles. Each match also exports
`possession_manifest.json` with ordered spells and canonical event associations.
Ranking obtains start/end values directly from a pass's spell and its next value
from the next genuine spell's start, in the original actor's team perspective.
It does not skip unknown successors or cross goals/period boundaries. Intermediate
predictions are not averaged into action scores. Carry value is `V(end)-V(start)`;
pass value is `V(next)-V(end)`; possession value is `V(next)-V(start)`.

See [Sportec possession validation and regeneration](docs/sportec_possession_states.md)
for the coverage audit and commands to create the required fresh cache.

- `--pass-height-threshold <meters>`: select observed-height diagnostic labels. For PC-xPass blending, assert the explicitly selected auxiliary checkpoint's recorded cutoff without changing prediction meaning.
- `--train-split <percentage>`: optional check that the requested development percentage matches the selected bundle/model split. Default: infer from model provenance (`50` for legacy artifacts); mismatches fail.
- `--train-count <int>`: exact number of development matches in canonical `MatchId` order, including validation. Mutually exclusive with `--train-split`; for the 918-match dataset use `--train-count 765`. In evaluation, this optionally checks the count recorded by the selected model.
- Carry-augmented inference is selected automatically from checkpoint metadata and requires a feature run with carry artifacts; there is no runtime `--use-carries` flag.
- `--split {train,test,all}`: choose which Sportec split to export. Default: `all`.
- `--match-id <id>`: restrict export to one or more specific matches. Default: all matches in the selected split.
- `--season {22_23,23_24,24_25}`: restrict Sportec export to selected seasons; repeat to select multiple seasons. Intersects with `--match-id` and `--split`; use `--season 24_25` for the full eligible season (the default split is `all`). Requires accessible raw match-information files and unambiguous membership for every candidate match; missing membership, conflicting seasons, or an empty selection fail. Default: no season filter.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: the selected physical xPass cache's feature run when physical xPass and caching are enabled; missing or incompatible cache feature runs fail. Otherwise, use the newest compatible source feature run from the selected models or bundle. Explicit selection retains strict cache checks.
- `--run-id <component_run_id>`: pin the created component export run id. Default: auto-generate a new component run id.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id used to export `success_intent.parquet`.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id used to export `pass_height.parquet`.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--output-dir <path>`: parent directory for the created component run folder. Default: `data/component_runs/sportec`.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/sportec`.
- `--no-physical-cache`: compatibility flag that disables the runtime cache override; not recommended with `--use-physical-xpass`.
- `--refresh-physical-cache`: deprecated and ignored for inference; run `scripts/generate_physical_xpass.py` to refresh/fill caches.
- `--physical-num-workers <N|auto>` / `--num-workers <N|auto>`, `--physical-worker-thread-limit <N>` / `--worker-thread-limit <N>`, and `--physical-batch-size <N>`: retained for CLI compatibility/metadata; inference does not compute xPass rows.

### `scripts/fit_reachability.py`

Fits reusable empirical player-reachability circles for pc-xPass. Artifacts are stored under `data/pc_xpass/reachability/<model-id>`; run the commands in order. See [Empirical reachability circles for pc-xPass](docs/reachability.md) for data requirements and example commands.

- `prepare --model-id <id> (--train-split <percentage> | --train-count <N>)`: resolve and save the training-pool manifest, reserve deterministic movement-holdout matches, and write resumable aligned-trajectory shards. `--frame-stride <N>` defaults to `1`; quality thresholds `--max-speed <m/s>`, `--max-acceleration <m/s2>`, and `--jump-tolerance <m>` default to `15`, `30`, and `0.5`.
- `fit --model-id <id>`: fit circle tables from prepared shards. `--source-model-id <id>` reuses preparation shards from an existing model; `--envelopes 0.99 0.995 0.999 max` selects frontier variants (all by default); `--min-samples <N>` controls neighbouring speed-bin pooling and defaults to `10000`.
- `diagnose --model-id <id>`: produce held-out movement diagnostics, tables, and figures. `--bootstrap <N>` controls match-bootstrap resamples (default `20`).

### `scripts/run_hawkeye.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--situation-id <id>`: restrict inference to one or more specific Hawkeye situation ids. Default: all valid situations.
- `--limit <N>`: process only the first `N` selected situations. Default: no limit.
- `--freeze-ballreceipt`: freeze possessor and ball state after `BallReceipt`. Default: on.
- `--no-freeze-ballreceipt`: disable BallReceipt freezing. Default: off.
- `--time-norm <seconds>` / `--time_norm <seconds>`: restrict HawkEye inference to the frame nearest to `abs_time - BallReceipt`; repeat to select multiple individual frames. Cannot be combined with `--time-norm-start` or `--time-norm-end`. Default: process all frames.
- `--time-norm-start <seconds>` / `--time_norm_start <seconds>` and `--time-norm-end <seconds>` / `--time_norm_end <seconds>`: restrict HawkEye inference to the inclusive frame range whose boundaries are nearest to `abs_time - BallReceipt`. Default: process all frames.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created HawkEye component run id. Default: auto-generate a new HawkEye component run id.
- `--output-dir <path>`: parent directory for the created Hawkeye run folder. Default: `data/component_runs/hawkeye`.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/hawkeye`.
- `--no-physical-cache`: compatibility flag that disables the runtime cache override; not recommended with `--use-physical-xpass`.
- `--refresh-physical-cache`: deprecated and ignored for inference; run `scripts/generate_physical_xpass.py` to refresh/fill caches.
- `--physical-num-workers <N|auto>` / `--num-workers <N|auto>`, `--physical-worker-thread-limit <N>` / `--worker-thread-limit <N>`, and `--physical-batch-size <N>`: retained for CLI compatibility/metadata; inference does not compute xPass rows.

### `scripts/run_benchmark.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--input-dir <path>`: local benchmark data root. Default: `benchmark`.
- `--modification <id>`: restrict inference to one or more specific benchmark modifications. Default: all valid modifications.
- `--limit <N>`: process only the first `N` selected modifications. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created benchmark component run id. Default: auto-generate a new benchmark component run id.
- `--output-dir <path>`: parent directory for the created benchmark run folder. Default: `data/component_runs/benchmark`.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/benchmark`.
- `--no-physical-cache`: compatibility flag that disables the runtime cache override; not recommended with `--use-physical-xpass`.
- `--refresh-physical-cache`: deprecated and ignored for inference; run `scripts/generate_physical_xpass.py` to refresh/fill caches.
- `--physical-num-workers <N|auto>` / `--num-workers <N|auto>`, `--physical-worker-thread-limit <N>` / `--worker-thread-limit <N>`, and `--physical-batch-size <N>`: retained for CLI compatibility/metadata; inference does not compute xPass rows.

### `scripts/run_skillcorner.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--match-id <id>`: restrict inference to one or more specific SkillCorner match ids. Default: all discoverable valid matches.
- `--scope {actions,frames}`: first/last valid possession states (default: `actions`), or strided states.
- `--frames N`: process every Nth frame ID relative to the original possession start, plus both endpoints. Default 1 in frames scope; invalid with actions scope. This is a stride, **not a frame count per action**.
- `--limit <N>`: process only the first `N` selected matches. Default: no limit.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <component_run_id>`: pin the created SkillCorner component run id. Default: auto-generate a new SkillCorner component run id.
- `--output-dir <path>`: parent directory for the created SkillCorner run folder. Default: `data/component_runs/skillcorner`.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/skillcorner`.
- `--no-physical-cache`: compatibility flag that disables the runtime cache override; not recommended with `--use-physical-xpass`.
- `--refresh-physical-cache`: deprecated and ignored for inference; run `scripts/generate_physical_xpass.py` to refresh/fill caches.
- `--physical-num-workers <N|auto>` / `--num-workers <N|auto>`, `--physical-worker-thread-limit <N>` / `--worker-thread-limit <N>`, and `--physical-batch-size <N>`: retained for CLI compatibility/metadata; inference does not compute xPass rows.

### `scripts/visualize_action_components.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--match-id <id>`: Sportec match id to visualize. Default: required.
- `--action-id <action_id>`: CSV `action_id` from `data/event_synced/<match_id>.csv`. Default: one of the identifier options is required.
- `--row-index <index>`: legacy modeled-action row index. Default: off.
- `--original-event-id <id>`: raw Sportec event id lookup. Default: off.
- `--first <N>`: visualize only the first `N` eligible modeled events after filters. Only valid when selecting by filters; incompatible with `--action-id`, `--row-index`, and `--original-event-id`.
- `--player-id <id>`, `--object-id <id>`, `--advanced-position <value>`, `--team-id <id>`, `--spadl-type <value>`, `--success <true|false>`, `--offside <true|false>`, and `--next-type <value>`: filter rows from `data/event_synced/<match_id>.csv`; repeat a flag to OR values within that column. Example: `--spadl-type pass --success false` visualizes unsuccessful passes.
- `--start-x-lt <n>`, `--start-x-gt <n>`, `--start-y-lt <n>`, `--start-y-gt <n>`, `--end-x-lt <n>`, `--end-x-gt <n>`, `--end-y-lt <n>`, and `--end-y-gt <n>`: strict coordinate filters. Different filter types are combined with AND logic.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--bundle-id <bundle_id>`: preferred explicit model bundle to run.
- `--feature-run-id <feature_run_id>`: optional runtime feature run used to load Sportec graphs and resolved actions. Default: newest compatible source feature run from the selected models or bundle.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--show-pass-height`: include the optional pass-height component when a `pass_height` checkpoint is selected. Default: off.
- `--show-physical-xpass`: render cached runtime physical xPass. Default: off.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` inference with cached runtime physical xPass. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric for both blending and rendering. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting; rendering still shows the selected raw cached xPass metric. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override for inference/visualization. Default: `data/runtime_physical_xpass/sportec`.
- `--no-physical-cache`, `--refresh-physical-cache`, `--physical-num-workers`, `--physical-worker-thread-limit`, and `--physical-batch-size`: compatibility flags shared with inference; visualization does not compute xPass rows.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id.
- `--success-intent-model-id <model_id>`: optional explicit `success_intent` checkpoint id used for success-intent visualizations.
- `--success-intent`: add success-intent rendering alongside selected component groups.
- `--compare-intended-receiver`: include success-intent rendering with `original`, `angle_only`, and current `model` receiver rings. Incompatible with `--no-success-intent`.
- `--pass-endpoint true/false`: show the actual endpoint on every pass image. Default: `true`.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id.
- `--run-id <visualization_run_id>`: pin the created Sportec visualization run id. Default: auto-generate one.
- `--output-dir <path>`: parent directory for the created visualization run folder. Default: `data/visualizations/sportec`.
- `--only-*` / `--no-*` component group flags: select or suppress `action-intent`, `pass-intent`, `pass-success`, `pass-height`, `outcome-scoring`, `outcome-conceding`, `pass-score`, and `success-intent`. Repeated `--only-*` flags are additive; `--no-*` takes precedence.

### `scripts/visualize_hawkeye.py`

- `--mode {standard,loc,freeze}`: visualization source/geometry mode. `standard` (default) reads a regular Hawkeye component run and supports PNG, MP4, or GIF. `loc` reads a matching `hawkeye_loc_component` run and applies its `PositionX`/`PositionY` possessor offset; `freeze` reads a matching freeze location run and applies no offset. `loc` and `freeze` support PNG only, use pc-xPass, and require component metadata whose `inference_mode` matches the selected mode.
- `--situation-id <id>`: restrict visualization to one or more Hawkeye situation ids from the selected component run. Default: all situations in the selected component run.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--component-run-id <component_run_id>`: versioned Hawkeye component run to visualize. Default: latest successful Hawkeye component run.
- `--component-dir <path>`: explicit Hawkeye component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--coach-ratings`: restrict rendering to situations with scored coach ratings in `validation/coach_ratings/output/coach_ratings.csv`, then add those ratings below player annotations in `pass_score` outputs. Default: off.
- `--selections`: add `CAVE | HMD` selection labels from `validation/selections/per_action_option_counts.csv` below player annotations in `pass_intent` and `pass_score` outputs. Default: off. The `pass_score` title shows the per-setting denominators as `Selections (CAVE n=<count> | HMD n=<count>)`.
- `--selection-format {counts,percentages}`: format `--selections` labels as raw counts such as `10 | 14` (default), or rounded percentages such as `45% | 64%`.
- `--output {png,mp4,gif}`: visualization format. Default: `png`.
- `--time-norm <seconds>` / `--time_norm <seconds>`: in PNG mode, export the frame nearest to `abs_time - BallReceipt`; repeat to export multiple stills. Default: `0`.
- `--time-norm-start <seconds>` / `--time_norm_start <seconds>` and `--time-norm-end <seconds>` / `--time_norm_end <seconds>`: in `mp4`/`gif` mode, render the inclusive frame range whose boundaries are nearest to `abs_time - BallReceipt`. Default: render all frames.
- `--run-id <visualization_run_id>`: pin the created HawkEye visualization run id. Default: auto-generate one.
- `--output-dir <path>`: parent directory for the created visualization run folder. Default: `data/visualizations/hawkeye`.
- `--show-physical-xpass`: render cached runtime physical xPass. Default: off.
- `--show-pass-height`: include the optional pass-height component when present in the component run. Default: off.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/hawkeye`.
- `--pc-xpass` / `--pc_xpass`: render pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric to render. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--only-*` / `--no-*` component group flags: select or suppress `action-intent`, `pass-intent`, `pass-success`, `pass-height`, `outcome-scoring`, `outcome-conceding`, and `pass-score`. Repeated `--only-*` flags are additive; `--no-*` takes precedence.

### `scripts/visualize_benchmark.py`

- `--input-dir <path>`: local benchmark data root. Default: `benchmark`.
- `--modification <id>`: restrict visualization to one or more benchmark modifications from the selected component run. Default: all available modifications in the selected run.
- `--game-state {1,2}`: restrict visualization to one or more game states. Default: both game states present in the selected run.
- `--component-run-id <component_run_id>`: versioned benchmark component run to visualize. Default: latest successful benchmark component run.
- `--component-dir <path>`: explicit benchmark component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--run-id <visualization_run_id>`: pin the created benchmark visualization run id. Default: auto-generate one.
- `--output-dir <path>`: parent directory for the created visualization run folder. Default: `data/visualizations/benchmark`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--show-physical-xpass`: render cached runtime physical xPass. Default: off.
- `--show-pass-height`: include the optional pass-height component when present in the component run. Default: off.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/benchmark`.
- `--pc-xpass` / `--pc_xpass`: render pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric to render. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--only-*` / `--no-*` component group flags: select or suppress `action-intent`, `pass-intent`, `pass-success`, `pass-height`, `outcome-scoring`, `outcome-conceding`, and `pass-score`.

### `scripts/run_and_visualize_hawkeye.py`

- `--pass-height-threshold <meters>`: select the observed-height definition used by height-aware pass-success diagnostics and physical/xPass lookups; see [Shared pass-height target option](#shared-pass-height-target-option).
- `--situation-id <id>`: Hawkeye situation id to visualize. Default: one of `--situation-id` or `--action-id` is required at runtime.
- `--action-id <id>`: alias for `--situation-id`. Default: off.
- `--tracking-csv <path>`: Hawkeye player-tracking CSV. Default: `hawkeye_data/centroid_data_team.csv`.
- `--ball-csv <path>`: Hawkeye ball-tracking CSV. Default: `hawkeye_data/ball_data_selected.csv`.
- `--freeze-ballreceipt`: freeze possessor and ball state after `BallReceipt`. Default: on.
- `--no-freeze-ballreceipt`: disable BallReceipt freezing. Default: off.
- `--device <device>`: inference device. Default: `cuda:0`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--coach-ratings`: restrict rendering to situations with scored coach ratings in `validation/coach_ratings/output/coach_ratings.csv`, then add those ratings below player annotations in `pass_score` outputs. Default: off.
- `--selections`: add `CAVE | HMD` selection labels from `validation/selections/per_action_option_counts.csv` below player annotations in `pass_intent` and `pass_score` outputs. Default: off. The `pass_score` title shows the per-setting denominators as `Selections (CAVE n=<count> | HMD n=<count>)`.
- `--selection-format {counts,percentages}`: format `--selections` labels as raw counts such as `10 | 14` (default), or rounded percentages such as `45% | 64%`.
- `--output {png,mp4,gif}`: visualization format. Default: `png`.
- `--time-norm <seconds>` / `--time_norm <seconds>`: in PNG mode, export the frame nearest to `abs_time - BallReceipt`; repeat to export multiple stills. Default: `0`.
- `--time-norm-start <seconds>` / `--time_norm_start <seconds>` and `--time-norm-end <seconds>` / `--time_norm_end <seconds>`: in `mp4`/`gif` mode, render the inclusive frame range whose boundaries are nearest to `abs_time - BallReceipt`. Default: render all frames.
- `--bundle-id <bundle_id>`: preferred model bundle containing the enabled component checkpoints. Default: none.
- `--action-intent-model-id <model_id>`: explicit `action_intent` checkpoint id override. Default: none.
- `--pass-intent-model-id <model_id>`: explicit `pass_intent` checkpoint id override. Default: none.
- `--pass-success-model-id <model_id>`: explicit `pass_success` checkpoint id override. Default: none.
- `--pass-height-model-id <model_id>`: optional explicit `pass_height` checkpoint id override. Default: none.
- `--outcome-scoring-model-id <model_id>`: explicit `outcome_scoring` checkpoint id override. Default: none.
- `--outcome-conceding-model-id <model_id>`: explicit `outcome_conceding` checkpoint id override. Default: none.
- `--run-id <visualization_run_id>`: pin the created HawkEye visualization run id. Default: auto-generate one.
- `--output-dir <path>`: parent directory for the created visualization run folder. Default: `data/visualizations/hawkeye`.
- `--use-physical-xpass` / `--use_physical_xpass`: blend `pass_success` inference with cached runtime physical xPass. Default: off.
- `--show-physical-xpass`: render cached runtime physical xPass. Default: off.
- `--show-pass-height`: include the optional pass-height component when a `pass_height` checkpoint is selected. Default: off.
- `--pc-xpass` / `--pc_xpass`: read pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric for both blending and rendering. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--xpass-weight {v0,v1,v2,v3,v4,v5}` / `--xpass_weight {v0,v1,v2,v3,v4,v5}`: select xPass/model blend weighting; rendering still shows the selected raw cached xPass metric. Default: `v3`.
- `--xpass-pass-height-model-id pass_height/<id>`: required for active PC-xPass v4/v5 blending or height-weighted evaluation; independent of component models/bundles.
- `--xpass-pass-intent-model-id pass_intent/<id>`: required for active PC-xPass v5, including with its discount disabled. Unused auxiliary selectors are rejected.
- `--v5-intent-threshold <float>`: positive finite pass-intent threshold for `v5`, with model weight `w = pass_height * clip(pass_intent / threshold, 0, 1)`. Default: `0.01`; only valid with `--xpass-weight v5`. Requires explicit height and intent auxiliaries, computed at inference.
- `--v5-discount <true|false>`: enable the `v5` pass-intent discount. Default: `true`; `false` uses `w = pass_height`. Only valid with `--xpass-weight v5`.
- `--v4-power <float>`: power for the `v4` pass-height distance discount. Default: `2.0`; only valid with `--xpass-weight v4`.
- `--v4-zero <float>`: zero point for the `v4` pass-height distance discount in `x = pass_distance / 100`. Default: `0.8`; only valid with `--xpass-weight v4`.
- `--discount <true|false>`: with `--xpass-weight v4`, enable or disable the cosine distance discount. Default: `true`; `false` uses `w = pass_height`.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/hawkeye`.
- `--no-physical-cache`, `--refresh-physical-cache`, `--physical-num-workers`, `--physical-worker-thread-limit`, and `--physical-batch-size`: compatibility flags shared with inference; this script does not compute xPass rows.
- `--only-*` / `--no-*` component group flags: select or suppress `action-intent`, `pass-intent`, `pass-success`, `pass-height`, `outcome-scoring`, `outcome-conceding`, and `pass-score`.

### `scripts/visualize_skillcorner.py`

- `--match-id <id>`: SkillCorner match id to visualize. Default: required.
- `--index <player_possession_index>`: SkillCorner `player_possession` event index. Default: required.
- `--input-dir <path>`: SkillCorner data root. Default: `skillcorner_data`.
- `--component-run-id <component_run_id>`: versioned SkillCorner component run to visualize. Default: latest successful SkillCorner component run.
- `--component-dir <path>`: explicit component-run root override. Default: none; when set it overrides `--component-run-id`.
- `--run-id <visualization_run_id>`: pin the created SkillCorner visualization run id. Default: auto-generate one.
- `--output-dir <path>`: parent directory for the created visualization run folder. Default: `data/visualizations/skillcorner`.
- `--show-trajectories`: draw dashed recent player trajectories. Default: off.
- `--output {png,mp4,gif}`: visualization format. Default: `png`.
- `--only-first`: in PNG mode, render only the first addressable possession frame. Default: off.
- `--only-last`: in PNG mode, render only the last addressable possession frame. Default: off.
- `--show-physical-xpass`: render cached runtime physical xPass. Default: off.
- `--show-pass-height`: include the optional pass-height component when present in the component run. Default: off.
- `--physical-cache-dir <path>`: runtime physical xPass cache override. Default: `data/runtime_physical_xpass/skillcorner`.
- `--pc-xpass` / `--pc_xpass`: render pc-xPass caches instead of runtime physical xPass caches. Default: off.
- `--xpass-version <max|noise-kernel|topN|top-passN|uncertainty>` / `--x_pass_version <...>`: select the cached xPass metric to render. Default: `top10`.
- `--top-pass N` / `--top_pass N`: select one cached top-pass count directly, for example `--pc-xpass --top-pass 5`. No `--xpass-version` flag is needed. Requires `--pc-xpass`; equivalent to `--xpass-version top-passN`.
- `--only-*` / `--no-*` component group flags: select or suppress `action-intent`, `pass-intent`, `pass-success`, `pass-height`, `outcome-scoring`, `outcome-conceding`, and `pass-score`.

## Script I/O Reference

This appendix summarizes the primary input and output files for each `scripts/*.py` entrypoint. Paths below are code-level defaults and conventions, not a snapshot of whatever files happen to be present on a given device.

### `scripts/main.py`

- Inputs: the inputs of the enabled downstream stages, typically raw Sportec season folders plus optional HawkEye and SkillCorner raw directories, plus explicit `feature_run_id` or `bundle_id` values when prerequisite stages are skipped.
- Outputs: no unique files of its own; it orchestrates the outputs of the enabled downstream scripts.

### `scripts/preprocess_sportec.py`

- Inputs:
  - `Bundesliga_season_22_23/tracking_data/*`
  - `Bundesliga_season_22_23/event_data/*`
  - `Bundesliga_season_22_23/starting_players/*`
  - `Bundesliga_season_22_23/master/*`
  - `Bundesliga_season_22_23/events_advanced/*`
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
  - `data/control_events_synced/*.parquet`
  - `data/carry_segments/*.parquet`
  - `data/carry_segments/audits/*.parquet`
  - `data/lineup/line_up.parquet`
  - `data/splits/match_splits.json`

### `scripts/generate_xt.py`

- Inputs:
  - `data/event_synced/*.csv`
  - `data/splits/match_splits.json`
- Outputs:
  - `data/xT/xT.csv`
  - `data/xT/xT_grid.csv`
  - `data/xT/xT_source_grid.csv`
  - `data/xT/xT_xy_surface.csv`
  - `data/xT/xT_glm_fit_sample.csv`
  - `data/xT/xT_xy_surface_3d.png`
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

### `scripts/generate_physical_xpass.py`

- Inputs:
  - Sportec runtime mode: compatible feature-run graphs/resolved actions plus `data/event_synced/*.csv`
  - HawkEye runtime mode: `hawkeye_data/centroid_data_team.csv` and `hawkeye_data/ball_data_selected.csv`
  - benchmark runtime mode: `benchmark/modification_<n>/...`
  - SkillCorner runtime mode: `skillcorner_data/<match_id>_tracking.jsonl`, `<match_id>_match.json`, and `<match_id>_dynamic_events.csv`
  - optional `saved/pass_height/<model_run_id>/...` when `--pass-height-model-id pass_height/<model_run_id>` is supplied
- Outputs:
  - default runtime cache: `data/runtime_physical_xpass/<dataset>/metadata.json`
  - default runtime rows: `data/runtime_physical_xpass/<dataset>/matches/*.parquet`
  - pc-xPass version metadata with `--pc-xpass`: `data/pc_xpass/<run_id>/metadata.json`
  - pc-xPass dataset metadata: `data/pc_xpass/<run_id>/<dataset>/metadata.json`
  - pc-xPass rows: `data/pc_xpass/<run_id>/<dataset>/matches/*.parquet`
  - optional opponent lane controls with `--export-lane-control`: `data/pc_xpass/<run_id>/<dataset>/lane_control/<match_id>.parquet` (one row per state/receiver/opponent)
  - optional lane-control coverage manifests: `data/pc_xpass/<run_id>/<dataset>/lane_control/coverage/<match_id>.json` (including completed states with no export rows)
  - legacy feature-run sidecars with `--feature-run-id`: `data/features/runs/<feature_run_id>/physical_xpass/...`
  - obsolete per-player `<player_id>__pass_height` columns may exist in older caches; PC-xPass consumers ignore them

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
  - `saved/pass_height/<model_run_id>/...` when `pass_height` is enabled
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
  - `data/component_runs/sportec/<component_run_id>/<match_id>/action_intent.parquet`
  - `data/component_runs/sportec/<component_run_id>/<match_id>/pass_intent.parquet`
  - optionally `data/component_runs/sportec/<component_run_id>/<match_id>/success_intent.parquet` when a `success_intent` checkpoint is selected
  - `data/component_runs/sportec/<component_run_id>/<match_id>/pass_success.parquet`
  - optionally `data/component_runs/sportec/<component_run_id>/<match_id>/pass_height.parquet` when a `pass_height` checkpoint is selected
  - `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_scoring_success.parquet`
  - `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_scoring_failure.parquet`
  - `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_conceding_success.parquet`
  - `data/component_runs/sportec/<component_run_id>/<match_id>/outcome_conceding_failure.parquet`
  - `data/component_runs/sportec/<component_run_id>/metadata.json`

### `scripts/visualize_action_components.py`

- Inputs:
  - feature artifacts from the selected runtime feature run
  - `data/event_synced/<match_id>.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/action_intent.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_intent.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_success.png`
  - optionally `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_height.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_scoring_success.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_scoring_failure.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_conceding_success.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_conceding_failure.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/pass_score.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_failure.png`
  - `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/outcome_success.png`
  - optionally `data/visualizations/sportec/<visualization_run_id>/<match_id>/<action_id>/success_intent.png`
  - `data/visualizations/sportec/<visualization_run_id>/metadata.json`

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
  - `data/visualizations/hawkeye/<visualization_run_id>/<situation_id>/*.{png,mp4,gif}`
  - `data/visualizations/hawkeye/<visualization_run_id>/metadata.json`

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
  - `data/visualizations/benchmark/<visualization_run_id>/modification_<n>/game_state_<m>/*.png`
  - `data/visualizations/benchmark/<visualization_run_id>/metadata.json`

### `scripts/run_and_visualize_hawkeye.py`

- Inputs:
  - `hawkeye_data/centroid_data_team.csv`
  - `hawkeye_data/ball_data_selected.csv`
  - `saved/<task>/<model_run_id>/...`
- Outputs:
  - `data/visualizations/hawkeye/<visualization_run_id>/<situation_id>/*.{png,mp4,gif}`
  - `data/visualizations/hawkeye/<visualization_run_id>/metadata.json`

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
  - optionally `data/component_runs/skillcorner/<component_run_id>/<match_id>/pass_height.parquet` when a `pass_height` checkpoint is selected
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
  - `data/visualizations/skillcorner/<visualization_run_id>/<match_id>/<index>/*.{png,mp4,gif}`
  - `data/visualizations/skillcorner/<visualization_run_id>/metadata.json`

## Sportec season selection

`generate_physical_xpass.py` and `run_relevant_models.py` accept repeatable
`--season` values (`22_23`, `23_24`, `24_25`). For example:

```bash
python scripts/generate_physical_xpass.py --season 24_25
python scripts/run_relevant_models.py --split all --season 24_25
```

Season filters intersect with `--match-id` and the existing eligible match set.
Inference still defaults to `--split test`; use `--split all` for the full eligible
season. Repeat `--season` to select multiple seasons. Cache generation filters
Sportec only: other datasets remain enabled unless explicitly disabled.
`--season` cannot be combined with `--no-sportec`. Existing limits and cache
reuse/overwrite behavior are unchanged.

Membership is discovered from `DFL-MAT-*` match-information filenames under
`RAW_SEASON_ROOTS`; raw events and tracking are not required. Requested season
metadata directories must be accessible, and every candidate match must have
unambiguous membership. Missing membership, conflicting seasons, and empty
selections fail explicitly. No preprocessing or metadata migration is required,
and commands without `--season` do not inspect raw season directories.


### Goal input context and prediction candidates

New training runs for `action_intent`, `pass_intent`, `success_intent`, `pass_success`,
`pass_height`, `failure_receiver`, `outcome_scoring`, and `outcome_conceding` save
`goal_context_version: 2`.
`goal_nodes_aware` controls goal nodes and their incident edges in the GNN input.
Use `--goal-nodes true` (the new-run default) or `--goal-nodes false` in direct
training or the training wrapper. `--goal-features none` independently zeros
explicit goal-relative geometry; it does not remove nodes.

`include_goals` controls prediction candidates only. Direct action-intent training
accepts `--include-goals true/false` (default: `true`). The multi-model wrapper
accepts `--action-intent-include-goals true/false` (default: `true`), applied
only to action intent. The other seven tasks require `include_goals=False`: goals
can supply context but never receive predictions or enter their softmax.

Action intent with goal predictions disabled excludes shot-labelled training and
evaluation examples and records `shot_candidate_disabled` counts. Enabling goal
predictions while disabling input goal nodes is an error. For example, train
pass/carry-only action intent without goals using `--goal-nodes false` and
`--include-goals false` in direct training, or `--goal-nodes false` and
`--action-intent-include-goals false` in the wrapper. Legacy positive/negative
spellings remain accepted.

Checkpoints without a version marker retain version-1 task-dependent node removal
and prediction behavior. Resume restores the saved settings and rejects conflicts.
To use contextual goals with a model previously trained without them, start a new
training run. Model arguments, metadata, feature signatures, and prepared cache
identities distinguish the policies; existing saved checkpoints are not rewritten.
Goal candidate indices are mapped explicitly to graph nodes in loss, inference,
IPW, and combined xPass evaluation. Auxiliary models use their own input policy.

The Hawkeye goal-position/annotation correction is separate from this change;
correct Hawkeye goal geometry before using newly goal-aware models on those scenes.

### GNN architecture selection

Both `train.py` and `scripts/train_relevant_models.py` accept
`--gnn-architecture {gat,transformer,relationship,relationship-attention}`.
Fresh runs default to `gat`; the wrapper forwards the selection to every task.
`--model` remains a compatibility alias, including existing GCN/GIN commands.
Supplying conflicting architecture selections is an error.

| Choice | Message construction and aggregation |
| --- | --- |
| `gat` | Existing GAT; edge attributes influence attention weights. |
| `transformer` | PyG TransformerConv; edges contribute to attention and message content. |
| `relationship` | Joint sender/receiver/edge MLP, separate incoming/outgoing means, then a node-update MLP. |
| `relationship-attention` | Same relationship blocks with separate multi-head incoming/outgoing attention. |

The relationship options are Stöckl-inspired adaptations, not exact reproductions.
Both blocks have three linear layers with ELU between them; widths and head count
follow the existing GNN settings. Each graph layer recomputes relationships using
updated player representations and original edge inputs. TransformerConv uses
concatenated heads, a root contribution, no beta gate, and zero attention dropout.
For these four options, embedding width must be divisible by the positive head
count, and the number of graph layers must be positive.

Fresh models record architecture version 1 and self-loop policy `explicit_v1`.
All four receive exactly one explicit self-edge per node, including after
sparsification. Original self-edge values are preserved in preprocessing metadata;
missing values follow the existing feature rules. Training and inference masks
also apply to restored loops. GAT no longer replaces these values with incoming
edge averages. Optional velocity-angle conventions and feature defaults are
unchanged. Existing graph artifacts remain usable; prepared caches rebuild through
normal code-fingerprint invalidation and remain shared across architectures.

Checkpoints without the new settings resolve their architecture from `model` and
retain `legacy` self-loop behaviour, including historical GAT loop averaging.
Resume and source refit inherit the saved architecture and policy; existing code
compatibility checks still apply. Auxiliary/IPW models use their own saved settings.
Use a newly trained GAT baseline to compare against the new architectures under
the corrected self-loop policy. Architecture identity is recorded separately from
input-feature signatures in model and bundle metadata.


Prepared training caches use lossless packed CPU tensors with lazy graph
reconstruction for all eight wrapper tasks. Version 1 caches rebuild once;
`--keep-cache` and workflow cleanup retain their existing behavior. See
[dataset loading and cache lifetime](docs/intent_dataset_loading.md).

For model-only early-contact cutoffs, see [Failed-pass trajectory refresh](docs/failed_pass_trajectories.md).

Graph connectivity validation and verified output publication default to on. Use
`--graph-validation off` to disable the added checks for a generation, refresh,
training or evaluation invocation. See [graph validation](docs/graph_validation.md)
for scope, cache behavior and computation costs.
