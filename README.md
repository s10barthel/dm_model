# dm_model

Scoped reproduction of DEFCON for Sportec Bundesliga 2023/24 data.

This project keeps the downstream DEFCON code close to upstream and adds the missing Sportec-specific preprocessing needed to produce DEFCON-compatible inputs. The retained model scope is:

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

The copied/adapted runtime files live at the project root. `_vendor` is kept only as a reference copy.

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
- `lineup/line_up.parquet`
- `features/...`
- `defcon_components/...`
- `splits/match_splits.json`

`event_synced` is stored as CSV because the upstream DEFCON code reads CSV, even though the README mentions Parquet.

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

Create the Conda environment first:

```powershell
conda env create -f environment.yml
conda activate dm_model
```

Then install PyTorch and PyTorch Geometric for the machine that will actually run the models.

1. Install `torch`, `torchvision`, and `torchaudio` using the official PyTorch command for your CPU/CUDA setup.
2. Install `torch_geometric` against that PyTorch build.

Everything else needed by this project is covered by `environment.yml` and `requirements.txt`.

If you prefer `pip` inside an existing environment:

```powershell
pip install -r requirements.txt
```

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

### 2. Generate graph features and labels

```powershell
python scripts/generate_relevant_features.py
```

This writes:

- `data/ajax/features/action_graphs/*.pt`
- `data/ajax/features/action_labels_disc_0.9/*.pt`

### 3. Train the retained models

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

Model checkpoints are saved under `saved/<task>/<trial>/`.

### 4. Evaluate the retained models on the test set

```powershell
python scripts/evaluate_relevant_models.py
```

This runs the original `test.py` flow for:

- `action_intent/00`
- `pass_success/20`
- `outcome_scoring/20`
- `outcome_conceding/20`

### 5. Export per-match component predictions

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

### 6. Visualize one action at a time

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123
```

By default the visualization uses the `success` branch for the outcome-conditioned models. To plot the failure-conditioned version instead:

```powershell
python scripts/visualize_action_components.py --match-id DFL-MAT-... --action-id 123 --outcome-case failure
```

The script writes one PNG per component under `data/ajax/visualizations/<match_id>/<action_id>/`.

## Notes

- `main.py` still reflects the full upstream defensive-score pipeline and is not part of this scoped reproduction.
- `datatools/tabular_feature.py` and the UxG/defensive-score path were left largely upstream because they are out of scope here.
- The preprocessing script is the main project-specific adaptation; downstream GNN training and evaluation stay close to upstream DEFCON.
- This repository is intended to track code, docs, and vendor snapshots only. Raw data, processed data, intermediate features, model outputs, and local environments are intentionally excluded from git.
