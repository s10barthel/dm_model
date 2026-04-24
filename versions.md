# training

# (0) 
python scripts/generate_relevant_features.py return_type next_5

feature_20260422T095614_782759_0a968ef6
helper for success_intent training

# (1a) 
python scripts/train_relevant_models.py --success-intent-only --feature-run-id feature_20260422T095614_782759_0a968ef6 --no-poss-vel-aware

Model bundle id: model_bundle_20260424T023636_811868_5ac8690a
Model bundle manifest: C:\dm_model\saved\bundles\model_bundle_20260424T023636_811868_5ac8690a\metadata.json
success_intent: success_intent/success_intent_20260424T023636_811868_4dfe0d54

# (1b) 
python scripts/train_relevant_models.py --success-intent-only --feature-run-id feature_20260422T095614_782759_0a968ef6

# (2) 
python scripts/generate_relevant_features.py --extend-feature-run-id feature_20260422T095614_782759_0a968ef6 --indended-receiver-model-id <success-intent 1b> --return_type disc_0.9 --return_type next_5 --return_type next_3 --return_type next_5_skip1 --return_type next_3_skip1 --return_type in_3 --return_type disc_0.5_skip1

# (3a) 
python scripts/train_relevant_models.py --feature-run-id <feature-run-id 2> --target-family xT --return_type disc_0.5_skip1 --intended-receiver-mode model --no-success-intent --no-poss-vel-aware

# (3b) 
python scripts/train_relevant_models.py --feature-run-id <feature-run-id 2> --target-family xT --return_type disc_0.5_skip1 --intended-receiver-mode model --no-success-intent

# (3c) 
python scripts/train_relevant_models.py --feature-run-id <feature-run-id 2> --target-family <> --return_type <> --intended-receiver-mode model --no-success-intent --no-action-intent --no-pass-intent --no-pass-success --no-failure-receiver

# 
(..)


## runs

# (4) 
python scripts/evaluate_relevant_models.py --bundle-id <> --success-intent-model-id <1b>

# (5) 
python scripts/run_relevant_models.py --split test --bundle-id <3b> --success-intent-model-id <1b>

# (6) 
python scripts/run_hawkeye.py --bundle-id <3a>

# (7) 
python scripts/run_benchmark.py --bundle-id <3b>

# (8) 
python scripts/run_skillcorner.py ----bundle-id <3b>


## visualizations

# (09) 
python scripts/visualize_action_components.py --match-id <> --action-id <> --bundle-id <3b> --success-intent-model-id <1b>

# (10) 
python scripts/visualize_hawkeye.py --situation-id <>

# (11) 
python scripts/visualize_benchmark.py --modification <>

# (12) 
python scripts/visualize_skillcorner.py --match-id <> --index <>