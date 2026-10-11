import json
from unittest.mock import patch

import pytest

import project_config as pc
from scripts import learning_curve
from scripts import train_relevant_models as wrapper
from scripts.evaluate_relevant_models import learning_curve_selection, parse_args as evaluation_args


def test_shared_latest_counts_and_outer_boundary(tmp_path):
    ids = [f"m{i:04}" for i in range(918)]
    manifest = {"train": ids[:765], "test": ids[765:]}
    stages = pc.plan_validation_stages(manifest["train"], validation_mode="expanding",
                                      validation_count=153, training_counts=[306, 459, 612])
    for stage, count in zip(stages, [306, 459, 612]):
        assert stage["train_match_ids"] == ids[:count]
        assert stage["validation_match_ids"] == ids[612:765]
        train, valid, test = pc.load_model_splits(manifest=manifest, validation_mode="expanding",
            validation_protocol=pc.VALIDATION_PROTOCOL_SHARED, training_count=count, validation_count=153)
        assert list(train) == ids[:count] and list(valid) == ids[612:765] and list(test) == ids[765:]
    features = tmp_path / "features"
    features.mkdir()
    for index in (305, 611, 612, 764, 765):
        (features / f"{ids[index]}.pt").touch()
    train, valid, test = pc.load_model_splits(features, manifest=manifest, validation_mode="expanding",
        validation_protocol=pc.VALIDATION_PROTOCOL_SHARED, training_count=306, validation_count=153)
    assert list(train) == [ids[305]] and list(valid) == [ids[612], ids[764]] and list(test) == [ids[765]]


@pytest.mark.parametrize("size", [5, 306, 612, 765])
def test_holdout_alias_and_default_rounding(size):
    ids = [str(i) for i in range(size)]
    expected_train, expected_valid = pc.derive_model_train_valid(ids)
    for mode in ("holdout", "holdout_80_20"):
        stage = pc.plan_validation_stages(ids, validation_mode=mode)[0]
        assert stage["train_match_ids"] == expected_train.tolist()
        assert stage["validation_match_ids"] == expected_valid.tolist()
    with pytest.raises(ValueError, match="20%"):
        pc.plan_validation_stages(ids, validation_mode="holdout_80_20", validation_fraction=.3)


@pytest.mark.parametrize("fractions", [[1], [.5, .75, 1], [.25, .5, .75, 1]])
def test_fraction_denominator_and_arbitrary_stages(fractions):
    stages = pc.plan_validation_stages(list(range(765)), validation_mode="expanding",
                                      training_fractions=fractions)
    assert [s["training_count"] for s in stages] == [int(612 * f) for f in fractions]
    assert len(stages) == len(fractions)
    assert stages[-1]["validation_match_ids"] == list(range(612, 765))


@pytest.mark.parametrize("options", [
    {"validation_mode": "expanding"}, {"validation_count": 0}, {"validation_count": 10},
    {"validation_fraction": float("nan")}, {"validation_fraction": 1},
    {"validation_fraction": 0}, {"validation_count": 2, "validation_fraction": .2},
    {"training_counts": [0]}, {"training_counts": [9]}, {"training_counts": [1, 2]},
    {"validation_mode": "expanding", "training_counts": [3, 3]},
    {"validation_mode": "expanding", "training_counts": [4, 3]},
    {"validation_mode": "expanding", "training_fractions": [.2, .21]},
    {"training_counts": [2], "training_fractions": [.5]},
    {"training_fractions": [float("inf")]}, {"training_counts": []},
])
def test_invalid_stage_plans(options):
    with pytest.raises(ValueError):
        pc.plan_validation_stages(list(range(10)), **options)


def test_holdout_single_subset_and_legacy_expanding():
    stage = pc.plan_validation_stages(list(range(765)), training_counts=[306])[0]
    assert stage["training_count"] == 306 and stage["validation_count"] == 153
    train, valid, _ = pc.load_model_splits(manifest={"train": list(range(765)), "test": []},
                                         validation_mode="expanding", validation_fold=3)
    assert (len(train), len(valid)) == (637, 128)


def test_stage_discovery_and_explicit_bundle_list(tmp_path, monkeypatch):
    for stage in ("fold_1", "fold_2", "fold_3", "fold_4", "final_refit"):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps({"status": "completed"}))
        (directory / "best_weights.pt").touch()
    (tmp_path / "metadata.json").write_text('{"status":"completed","final_refit":true}')
    monkeypatch.setattr(learning_curve, "get_model_path", lambda _: tmp_path)
    ids = learning_curve.discover("pass_height/run")
    assert ids == [f"pass_height/run/fold_{n}" for n in range(1, 5)] + ["pass_height/run"]
    listed = [f"pass_height/run/fold_{n}" for n in range(1, 5)] + ["pass_height/run/final_refit"]
    with patch("scripts.evaluate_relevant_models.load_bundle_record", return_value={
            "model_ids": {"pass_height": "pass_height/run"}, "stage_model_ids": {"pass_height": listed}}):
        selected, skipped = learning_curve_selection(evaluation_args(["--bundle-id", "bundle", "--learning-curve"]))
    assert selected["pass_height"] == listed and not skipped


def test_shared_validation_identity_mismatch_is_rejected(monkeypatch):
    base = {"task": "pass_height", "seed": 1, "test_match_ids": ["test"],
            "validation_protocol": pc.VALIDATION_PROTOCOL_SHARED, "validation_match_ids": ["valid"],
            "validation_provenance": {"contributing_match_ids": ["valid"]}}
    records = [{**base, "model_id": "pass_height/a", "train_matches": 1, "train_match_ids": ["a"],
                "validation_example_identity": {"sha256": "one"}},
               {**base, "model_id": "pass_height/b", "train_matches": 2, "train_match_ids": ["a", "b"],
                "validation_example_identity": {"sha256": "two"}}]
    monkeypatch.setattr(learning_curve, "checkpoint", lambda model: records[model.endswith("b")])
    with pytest.raises(ValueError, match="Usable validation"):
        learning_curve.validate_series(["pass_height/a", "pass_height/b"], expected_task="pass_height")


@pytest.mark.parametrize("mode,counts,refit", [
    ("holdout", None, False), ("holdout", [20], True),
    ("expanding", [10, 20, 30, 40], False), ("expanding", [10, 20, 40], True),
    ("expanding", [40], True)])
def test_wrapper_stage_order_dependencies_and_refit_source(tmp_path, monkeypatch, mode, counts, refit):
    args = wrapper.parse_args(["--refit-source-model-id", "pass_intent/source"])
    args.refit_source_model_id = []
    args.final_refit = refit
    args.validation_mode = mode
    args.training_counts = counts
    args.validation_count = 10
    args.train_count = 50
    args.train_split = None
    args.feature_run_id = "feature"
    args.available_intended_receiver_modes = []
    args.available_return_types = []
    args.split_manifest = {"manifest_id": "fixture", "train": list(range(50)), "test": [50], "metadata": {}}
    tasks = {"pass_intent": "pass_intent/run", "pass_success": "pass_success/run"}
    commands = [["--task", "pass_intent", "--run-id", "run"],
                ["--task", "pass_success", "--run-id", "run", "--ipw_model_id", "pass_intent/run"]]
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "get_model_bundle_root", lambda _: tmp_path / "bundle")
    monkeypatch.setattr(wrapper, "load_model_bundle_metadata", lambda *a, **k: {})
    monkeypatch.setattr(wrapper, "build_training_commands", lambda _: (commands, tasks, "original", "feature", {}))
    monkeypatch.setattr(wrapper, "diagnostic_metadata", lambda *a, **k: {})
    monkeypatch.setattr(wrapper, "derive_bundle_shared_context", lambda *a: {})
    stage_names = [s["artifact_stage"] for s in pc.plan_validation_stages(list(range(50)),
        validation_mode=mode, validation_count=10, training_counts=counts)]
    summaries = {task: [{"fold": i, "train_matches": i * 10, "validation_matches": 10,
                         "best_epoch": i, "metrics": {"ce_loss": .5}} for i in range(1, len(stage_names)+1)] for task in tasks}
    monkeypatch.setattr(wrapper, "_read_fold_summaries", lambda *a: summaries)
    prepared = []
    def prepare(source, run_id):
        prepared.append(source)
        return {"args": {"task": source.split('/')[0]}, "recipe": {"source_model_id": source}}
    monkeypatch.setattr(wrapper, "prepare_refit", prepare)
    launched = []
    monkeypatch.setattr(wrapper, "run_training", lambda command, **kw: launched.append(command))
    wrapper.main()
    assert len(launched) == 2 * (len(stage_names) + refit)
    for i, stage in enumerate(stage_names):
        first, second = launched[2*i:2*i+2]
        assert wrapper.get_cli_value(first, "--task") == "pass_intent"
        assert wrapper.get_cli_value(second, "--ipw_model_id") == f"pass_intent/run/{stage}"
        assert wrapper.get_cli_value(first, "--artifact-stage") == stage
        assert wrapper.get_cli_value(first, "--training-step-index") == str(2*i+1)
        assert wrapper.get_cli_value(second, "--training-step-index") == str(2*i+2)
        assert wrapper.get_cli_value(first, "--training-step-total") == str(len(launched))
    assert prepared == ([f"{model}/{stage_names[-1]}" for model in tasks.values()] if refit else [])
    metadata = json.loads((tmp_path / "bundle/metadata.json").read_text())
    assert metadata["model_ids"] == tasks and metadata["aggregate_validation_metrics"] == {}
    assert metadata["stage_model_ids"]["pass_intent"] == [f"pass_intent/run/{s}" for s in stage_names] + (["pass_intent/run/final_refit"] if refit else [])


@pytest.mark.parametrize("options", [
    ["--validation-mode", "expanding", "--validation-count", "153", "--training-counts", "306", "459", "612", "--final-refit"],
    ["--validation-mode", "holdout", "--validation-fraction", ".25", "--training-fractions", ".8"],
])
def test_pipeline_forwards_configurable_validation(monkeypatch, options):
    import sys
    from scripts import main
    monkeypatch.setattr(sys, "argv", ["main.py", "--train-count", "765", "--target-family", "xt",
        "--return_type", "disc_0.9", "--intended-receiver-mode", "original", *options])
    commands = main.build_commands(main.parse_args())
    command = next(c for c in commands if c[1] == "scripts/train_relevant_models.py")
    for i, token in enumerate(options):
        if token.startswith("--"):
            assert token in command
            if token != "--final-refit":
                actual, expected = command[command.index(token)+1], options[i+1]
                if "fraction" in token:
                    assert float(actual) == float(expected)
                else:
                    assert actual == expected



def test_validation_identity_is_independent_of_batch_size():
    import torch
    from torch_geometric.data import Batch, Data
    from training_refit import ValidationExampleIdentity
    graphs = [Data(x=torch.zeros(1, 1), evaluation_match_id="match",
                   evaluation_source_row=torch.tensor([i])) for i in range(4)]
    labels = torch.arange(8).reshape(4, 2)
    combined, separate = ValidationExampleIdentity(), ValidationExampleIdentity()
    combined.update(Batch.from_data_list(graphs), labels)
    for graph, label in zip(graphs, labels):
        separate.update(Batch.from_data_list([graph]), label.unsqueeze(0))
    assert combined.result() == separate.result()
    assert combined.result()["count"] == 4
