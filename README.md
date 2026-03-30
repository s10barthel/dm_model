# dm_model

This repository is a scoped adaptation of the upstream DEFCON implementation for Sportec Bundesliga 2023/24 data.
Upstream DEFCON source code: https://github.com/hyunsungkim-ds/defcon

The model structure is copied from DEFCON, the upstream source code for the paper "Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks". This repository keeps the downstream DEFCON runtime close to upstream, narrows the retained scope, and adds the dataset- and workflow-specific adaptations needed for Sportec, HawkEye, and SkillCorner data.

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

## Produced Data Layout

After preprocessing, the project writes DEFCON-style files under `data/ajax`:

- `tracking/*.parquet`
- `tracking_processed/*.parquet`
- `event/event.parquet`
- `event_synced/*.csv`
- `xT/xT.csv`
- `xT/xT_grid.csv`
- `xT/fit_metadata.json`
- `xT/matches/*.csv`
- `lineup/line_up.parquet`
- `features/...`
- `defcon_components/...`
- `splits/match_splits.json`

`event_synced` is stored as CSV because the upstream DEFCON code reads CSV, even though the README mentions Parquet.

The main output directories are:

- `data/ajax/event_synced` for canonical synced event tables used by the DEFCON-style pipeline
- `data/ajax/xT` for xT exports and xT fitting artifacts
- `data/ajax/features` for graph tensors and label tensors used by training/evaluation

## Split Definition

The dataset split is deterministic:

1. sort matches by actual `KickoffTime`
2. break ties with `MatchId`
3. first `245` matches become the train pool
4. remaining `61` matches become the test set

To keep upstream `train.py` behavior, the `245`-match train pool is deterministically split again into:

- `200` model-training matches
- `45` validation matches

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

## End-to-End Workflow

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

### 2. Generate xT artifacts when xT targets are needed

```powershell
python scripts/generate_xt.py
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

The aggregate `xT.csv` contains only `pass`, `cross`, and `shot` rows, with both `xG` and xT-derived target columns.

### 3. Generate graph features and labels

```powershell
python scripts/generate_relevant_features.py
```

Run this again after `scripts/generate_xt.py` whenever you want to train outcome models with `--use_xt`. xT exports alone are not enough; the label tensors under `data/ajax/features` must be regenerated so they include the xT target columns.

This writes:

- `data/ajax/features/action_graphs/*.pt`
- `data/ajax/features/action_labels_disc_0.9/*.pt`
- `data/ajax/features/action_graphs_intent_train/*.pt`
- `data/ajax/features/action_labels_intent_train_disc_0.9/*.pt`

### 4. Train the retained models

```powershell
python scripts/train_relevant_models.py
```

This trains the four retained models and one auxiliary upstream-compatible model:

- auxiliary: `pass_intent/20`
- retained: `action_intent/00`
- retained: `pass_success/20`
- retained: `outcome_scoring/20`
- retained: `outcome_conceding/20`

The auxiliary `pass_intent/20` run is kept because the upstream `pass_success` training uses it for inverse-propensity weighting.

Wrapper behavior:

- `python scripts/train_relevant_models.py` keeps the current retained default, which trains the outcome models with `--use_xg`
- `python scripts/train_relevant_models.py --use_xt` switches only `outcome_scoring` and `outcome_conceding` to xT targets
- `action_intent`, `pass_intent`, and `pass_success` are unchanged by the xT toggle

Model checkpoints are saved under `saved/<task>/<trial>/`.

### 5. Evaluate the retained models on the test set

```powershell
python scripts/evaluate_relevant_models.py
```

This runs the original `test.py` flow for:

- `action_intent/00`
- `pass_success/20`
- `outcome_scoring/20`
- `outcome_conceding/20`

`test.py` uses the target configuration saved inside the checkpoint. There is no separate `--use_xg` / `--use_xt` switch at evaluation time for an already trained model.

### 6. Export per-match component predictions

```powershell
python scripts/run_relevant_models.py --split test
```

For each processed match, this writes:

- `action_intent.parquet`
- `pass_success.parquet`
- `outcome_scoring_success.parquet`
- `outcome_scoring_failure.parquet`
- `outcome_conceding_success.parquet`
- `outcome_conceding_failure.parquet`

under `data/ajax/defcon_components/<match_id>/`.

### 7. Visualize one action at a time

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123
```

The script always writes 6 PNGs under `data/ajax/visualizations/<match_id>/<action_id>/`:

- `action_intent.png`
- `pass_success.png`
- `outcome_scoring_success.png`
- `outcome_scoring_failure.png`
- `outcome_conceding_success.png`
- `outcome_conceding_failure.png`

Useful option:

- `--show-trajectories` to render dashed recent player trajectories

### 8. Run frame-level inference on HawkEye data

```powershell
python scripts/run_hawkeye.py
```

This processes `hawkeye_data/centroid_data_team.csv` and `hawkeye_data/ball_data_selected.csv` and writes consolidated component outputs to:

- `data/ajax/defcon_components/hawkeye_data.parquet`
- `data/ajax/defcon_components/hawkeye_data.csv`

Useful options:

- `--situation-id ...` to restrict inference to selected situations
- `--limit N` to smoke-test on the first `N` situations
- `--no-freeze-ballreceipt` to disable the default BallReceipt freeze for the possessor and the ball

To visualize one HawkEye situation as GIFs:

```powershell
python scripts/visualize_hawkeye.py --situation-id <hawkeye_id>
```

This writes 6 GIFs under `data/ajax/visualizations/hawkeye/<situation_id>/`.

### 9. Run frame-level inference on SkillCorner data

```powershell
python scripts/run_skillcorner.py
```

This reads synchronized SkillCorner files from `skillcorner_data/` and writes per-match parquet outputs under:

- `data/ajax/defcon_components/skillcorner/<match_id>/`

The SkillCorner adapter processes `player_possession` events frame by frame and exports the same 6 retained DEFCON components.

To visualize one SkillCorner possession as GIFs:

```powershell
python scripts/visualize_skillcorner.py --match-id <match_id> --index <player_possession_index>
```

This writes 6 GIFs under `data/ajax/visualizations/skillcorner/<match_id>/<index>/`.

## Outcome Target Selection

Outcome target selection affects `outcome_scoring` and `outcome_conceding`.

- Binary goals:
  - use neither `--use_xg` nor `--use_xt`
  - this is available through the low-level `train.py` entrypoint
- xG:
  - pass `--use_xg`
  - `--return_type` controls which xG target family is used
- xT:
  - pass `--use_xt`
  - `--return_type` is ignored for xT semantics

`--use_xg` and `--use_xt` are mutually exclusive.

### `--return_type` is xG-only

`--return_type` only changes how xG-based soft targets are constructed:

- `disc_<gamma>` uses discounted xG returns
- `next_<N>` uses non-discounted xG returns over the next `N` actions

Example:

```powershell
python train.py --task outcome_scoring --trial 20 --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --trial 21 --model gat --use_xg --return_type next_10 ...
```

For xT, the target is fixed by the xT module:

- event-level `xT` is the zone value at `start_x,start_y`
- `scores_xT` / `concedes_xT` are the maximum future teammate/opponent xT values over the next 5 eligible `pass`/`cross`/`shot` actions

So `--return_type` does not change xT behavior.

### Where to switch targets

Use `scripts/train_relevant_models.py` when you want the retained default setup:

```powershell
python scripts/train_relevant_models.py
python scripts/train_relevant_models.py --use_xt
```

Use `train.py` directly when you need explicit low-level control, especially for binary-goal outcome training:

```powershell
python train.py --task outcome_scoring --trial 20 --model gat ...
python train.py --task outcome_scoring --trial 20 --model gat --use_xg --return_type disc_0.9 ...
python train.py --task outcome_scoring --trial 20 --model gat --use_xt ...
```

If you generate or rebuild xT artifacts, rerun `scripts/generate_relevant_features.py` before any `--use_xt` training run so the feature label tensors are refreshed.

## Notes

- `main.py` still reflects the full upstream defensive-score pipeline and is not part of this scoped reproduction.
- `datatools/tabular_feature.py` and the UxG/defensive-score path were left largely upstream because they are out of scope here.
- The core GNN training and evaluation flow stays close to upstream DEFCON; most project-specific changes are in preprocessing, target construction, and extra-data adapters.
- This repository is intended to track code, docs, and vendor snapshots only. Raw data, processed data, intermediate features, model outputs, and local environments are intentionally excluded from git.
