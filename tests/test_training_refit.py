import copy
import faulthandler
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import training_refit as refit
from scripts import train_relevant_models as wrapper
from training_state import CHECKPOINT_NAME, load_checkpoint, resolve_resume_checkpoint
from test_training_controls import synthetic_training


@pytest.fixture
def sources(tmp_path, monkeypatch):
    manifest = {"manifest_id": "fixture", "train_count": 10, "train_split_percent": None,
                "train": [f"m{i}" for i in range(10)], "test": ["test"], "metadata": {}}
    monkeypatch.setattr(refit, "load_feature_run_metadata", lambda *a, **k: {"split_manifest_id": "fixture"})
    monkeypatch.setattr(refit, "resolve_artifact_split", lambda *a, **k: (manifest, "fixture"))
    monkeypatch.setattr(refit, "SAVED_DIR", tmp_path / "saved")
    monkeypatch.setattr(wrapper, "SAVED_DIR", tmp_path / "saved")
    def create(model_id, **changes):
        directory = refit.source_path(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        settings = {"task": model_id.split('/')[0], "model": "gat", "feature_run_id": "feature",
                    "seed": 42, "batch_size": 17, "accumulation_steps": 2, "start_lr": .002,
                    "min_lr": .00001, "ipw_model_id": "none", "final_refit": False,
                    "artifact_stage": model_id.split('/')[2] if len(model_id.split('/')) == 3 else None}
        metadata = {"status": "completed", "task": settings["task"], "best_epoch": 3, "last_epoch": 4,
                    "validation_match_ids": ["m8", "m9"], "train_match_ids": [f"m{i}" for i in range(8)],
                    "learning_rate_history_version": 1,
                    "learning_rate_history": [{"epoch": i, "lr": .002 if i < 3 else .001} for i in range(1, 5)]}
        metadata.update(changes)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        (directory / "args.json").write_text(json.dumps(settings))
        (directory / "best_weights.pt").write_bytes(b"not loaded by refit preparation")
        return directory, settings, metadata
    return create, manifest


def test_source_reconstruction_is_fresh_and_truncates_history(sources, tmp_path):
    create, manifest = sources
    directory, original, _ = create("outcome_scoring/source")
    payload = refit.prepare_refit("outcome_scoring/source", "new", operational={"device": "cpu"})
    args, recipe = payload["args"], payload["recipe"]
    assert args["seed"] == original["seed"] and args["batch_size"] == 17
    assert args["accumulation_steps"] == 2 and args["device"] == "cpu"
    assert args["n_epochs"] == 3 and args["final_refit"] and not args["lr_rollback"] and not args["early_stopping"]
    assert not args["cont"] and args["resume_run_id"] is None
    assert args["split_manifest"] == manifest and args["training_count"] is None
    assert recipe["learning_rate_history"] == [{"epoch": 1, "lr": .002}, {"epoch": 2, "lr": .002}, {"epoch": 3, "lr": .001}]
    assert recipe["rollback_replayed"] is False
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    assert refit.load_refit_config(config_path) == args
    assert json.loads((directory / "args.json").read_text()) == original


@pytest.mark.parametrize("saved", ["missing", None, False, True])
def test_refit_inherits_speed_setting(sources, saved):
    create, _ = sources
    directory, settings, _ = create("pass_intent/source")
    if saved != "missing":
        settings["speed_aware"] = saved
        (directory / "args.json").write_text(json.dumps(settings))
    payload = refit.prepare_refit("pass_intent/source", "new")
    assert payload["args"]["speed_aware"] == (saved is not False)
    assert json.loads((directory / "args.json").read_text()) == settings


@pytest.mark.parametrize("changes", [
    {"status": "running"}, {"final_refit": True}, {"validation_match_ids": []},
    {"learning_rate_history_version": None}, {"best_epoch": None}, {"best_epoch": 6},
    {"last_epoch": 3}, {"train_match_ids": ["test"]},
    {"train_match_ids": [f"m{i}" for i in reversed(range(8))]}, {"validation_match_ids": ["m0"]},
    {"learning_rate_history": [{"epoch": 2, "lr": .002}] * 4},
    {"learning_rate_history": [{"epoch": i, "lr": 0} for i in range(1, 5)]},
])
def test_invalid_source_recipes_are_rejected(sources, changes):
    create, _ = sources
    create("pass_intent/source", **changes)
    with pytest.raises(ValueError):
        refit.prepare_refit("pass_intent/source", "new")


@pytest.mark.parametrize("flags", [
    ["--validation-mode", "holdout"], ["--training-counts", "10"], ["--train-count", "50"],
    ["--seed", "12"], ["--start_lr", ".001"], ["--only-outcome-scoring"],
    ["--batch-size", "64"], ["--feature-run-id", "feature"],
    ["--refit-source-model-id", "outcome_scoring/source"],
])
def test_source_cli_rejects_configuration_conflicts(flags):
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--refit-source-model-id", "outcome_scoring/source", *flags])


def test_source_cli_needs_no_other_flags_and_accepts_operational_overrides():
    args = wrapper.parse_args(["--refit-source-model-id", "outcome_scoring/a",
                              "--refit-source-model-id", "outcome_scoring/b", "--device", "cpu"])
    assert args.final_refit and args.refit_operational == {"device": "cpu"}
    assert len(args.refit_source_model_id) == 2


@pytest.mark.parametrize("fail", [False, True])
def test_batch_independent_same_task_sources_and_failure_manifest(sources, monkeypatch, fail):
    create, _ = sources
    for name in ("outcome_scoring/a", "outcome_scoring/b", "pass_intent/c"):
        create(name)
    args = wrapper.parse_args(["--refit-source-model-id", "outcome_scoring/a",
                              "--refit-source-model-id", "outcome_scoring/b",
                              "--refit-source-model-id", "pass_intent/c"])
    launched = []
    def launch(command, **kw):
        launched.append(command)
        if fail and len(launched) == 2:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(wrapper, "run_training", launch)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            wrapper.run_refit_batch(args)
    else:
        wrapper.run_refit_batch(args)
    batch = next((wrapper.SAVED_DIR / "refit_batches").glob("*/metadata.json"))
    metadata = json.loads(batch.read_text())
    assert [j["status"] for j in metadata["jobs"]] == (["completed", "failed", "unstarted"] if fail else ["completed"] * 3)
    assert len({j["output_model_id"] for j in metadata["jobs"]}) == 3
    assert metadata["status"] == ("failed" if fail else "completed")
    assert not (wrapper.SAVED_DIR / "bundles").exists()
    assert all("--refit-config" in cmd and "--validation-mode" not in cmd for cmd in launched)


def test_batch_preflights_every_source_before_launch(sources, monkeypatch):
    create, _ = sources
    create("pass_intent/a")
    create("pass_intent/b", learning_rate_history_version=None)
    args = wrapper.parse_args(["--refit-source-model-id", "pass_intent/a", "--refit-source-model-id", "pass_intent/b"])
    launched = []
    monkeypatch.setattr(wrapper, "run_training", lambda *a, **k: launched.append(a))
    with pytest.raises(ValueError, match="structured"):
        wrapper.run_refit_batch(args)
    assert not launched and not (wrapper.SAVED_DIR / "refit_batches").exists()


def test_history_validation_and_replay_preserve_adam_state():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    recipe = {"learning_rate_history": [{"epoch": 1, "lr": .002}, {"epoch": 2, "lr": .001}]}
    assert refit.apply_refit_rate(optimizer, recipe, 2) == .001
    assert optimizer.param_groups[0]["lr"] == .001
    for key in state["state"]:
        for field in state["state"][key]:
            torch.testing.assert_close(state["state"][key][field], optimizer.state_dict()["state"][key][field])
    metadata = {"learning_rate_history_version": 1}
    refit.append_rate_history(metadata, 1, .002)
    refit.append_rate_history(metadata, 2, .001)
    with pytest.raises(ValueError):
        refit.append_rate_history(metadata, 2, .001)
    legacy = {}
    refit.append_rate_history(legacy, 8, .001)
    assert not legacy.get("learning_rate_history_version")


def test_actual_refit_rates_fresh_state_full_pool_and_resume(synthetic_training, monkeypatch):
    run, observations, epoch, utils, root = synthetic_training
    # Delaunay needs a non-collinear, fully connected source graph.
    from datatools import config
    from torch_geometric.data import Data
    from scripts.generate_relevant_features import append_interactions_to_graph
    for path in (root / "features").glob("*.pt"):
        source_graph = torch.load(path, weights_only=False)[0]
        x = source_graph.x.clone()
        x[:, config.NODE_FEATURE_Y] = torch.tensor([34., 35., 33., 36., 32.])
        edges = torch.cartesian_prod(torch.arange(len(x)), torch.arange(len(x))).t()
        xy = x[:, config.NODE_FEATURE_X:config.NODE_FEATURE_Y + 1]
        distances = torch.cdist(xy, xy)[edges[0], edges[1]]
        teams = x[:, config.NODE_FEATURE_IS_TEAMMATE]
        teammates = (teams[edges[0]] == teams[edges[1]]).float()
        graph = Data(x=x, edge_index=edges, edge_attr=torch.stack([distances, teammates], dim=1))
        torch.save([append_interactions_to_graph(graph)], path)
    initial_weights = []
    def validation(args, model, *a, **kw):
        if kw["train"] and args._epoch == 1:
            initial_weights.append(copy.deepcopy(model.state_dict()))
        result = epoch(args, model, *a, **kw)
        if not kw["train"]:
            result = {"ce_loss": .5 if args._epoch >= 6 else 1.0}
        return result
    monkeypatch.setattr(utils, "run_epoch", validation)
    source = run("source", ["--sparsify", "delaunay", "--max-epochs", "8", "--no-early-stopping", "--early-stopping-min-epochs", "30"])
    assert source["metadata"]["best_epoch"] == 6
    source_dir = root / "saved/pass_intent/source"
    settings = json.loads((source_dir / "args.json").read_text())
    settings["feature_run_id"] = "fixture_feature"
    (source_dir / "args.json").write_text(json.dumps(settings))
    monkeypatch.setattr(refit, "load_feature_run_metadata", lambda *a, **k: {})
    monkeypatch.setattr(refit, "resolve_artifact_split", lambda *a, **k: (settings["split_manifest"], "fixture"))
    payload = refit.prepare_refit("pass_intent/source", "refit", saved_root=root / "saved")
    assert payload["args"]["sparsify"] == "delaunay"
    config_path = root / "recipe.json"
    config_path.write_text(json.dumps(payload))
    entry = Path(__file__).resolve().parents[1] / "train.py"
    def execute(argv):
        monkeypatch.setattr(sys, "argv", ["train.py", *argv])
        try:
            runpy.run_path(str(entry), run_name="__main__")
        finally:
            faulthandler.disable()
    observations.clear()
    execute(["--refit-config", str(config_path)])
    completed = load_checkpoint(root / "saved/pass_intent/refit/final_refit" / CHECKPOINT_NAME)
    assert completed["args"]["sparsify"] == "delaunay"
    assert observations == [(i, .002 if i < 5 else .001) for i in range(1, 7)]
    assert completed["metadata"]["train_match_count"] == 5 and completed["metadata"]["validation_match_count"] == 0
    assert completed["metadata"]["last_epoch"] == 6 and not completed["metadata"]["stopped_early"]
    assert all(int(s["step"]) == 6 for s in completed["optimizer"]["state"].values())
    for name in initial_weights[0]:
        torch.testing.assert_close(initial_weights[0][name], initial_weights[1][name], atol=0, rtol=0)
    assert completed["metadata"]["learning_rate_history"] == payload["recipe"]["learning_rate_history"]
    # Interrupt a second refit; its checkpoint owns the recipe even after the JSON and source disappear.
    payload["args"].update(run_id="interrupted", model_id="pass_intent/interrupted")
    config_path.write_text(json.dumps(payload))
    def interrupt(args, *a, **kw):
        if args._epoch == 3:
            raise RuntimeError("interruption")
        return validation(args, *a, **kw)
    monkeypatch.setattr(utils, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        execute(["--refit-config", str(config_path)])
    checkpoint_path = root / "saved/pass_intent/interrupted/final_refit" / CHECKPOINT_NAME
    assert load_checkpoint(checkpoint_path)["metadata"]["last_epoch"] == 2
    config_path.unlink()
    (source_dir / "args.json").unlink()
    monkeypatch.setattr(utils, "run_epoch", validation)
    execute(["--resume-checkpoint", str(checkpoint_path)])
    resumed = load_checkpoint(checkpoint_path)
    assert resumed["args"]["sparsify"] == "delaunay"
    assert resumed["metadata"]["learning_rate_history"] == completed["metadata"]["learning_rate_history"]
    for name in completed["model"]:
        torch.testing.assert_close(completed["model"][name], resumed["model"][name], atol=0, rtol=0)
    assert resolve_resume_checkpoint("pass_intent/interrupted/final_refit", root / "saved") == checkpoint_path.resolve()


@pytest.mark.parametrize("invalid", [False, True])
def test_dependency_fingerprint_and_immutable_stage(sources, invalid):
    import hashlib
    create, _ = sources
    dependency, _, _ = create("pass_intent/dependency", split_manifest_id="fixture", artifact_stage="holdout")
    create("pass_intent/dependency/holdout", split_manifest_id="fixture")
    fingerprint = hashlib.sha256((dependency / "best_weights.pt").read_bytes()).hexdigest()
    source, args, _ = create("pass_success/source", ipw_dependency_weights_sha256=fingerprint)
    args["ipw_model_id"] = "pass_intent/dependency"
    (source / "args.json").write_text(json.dumps(args))
    if invalid:
        (dependency / "holdout/best_weights.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="Immutable propensity"):
            refit.prepare_refit("pass_success/source", "new")
    else:
        payload = refit.prepare_refit("pass_success/source", "new")
        assert payload["args"]["ipw_model_id"] == "pass_intent/dependency/holdout"
        assert payload["recipe"]["ipw_dependency_weights_sha256"] == fingerprint


def test_source_root_uses_immutable_configuration(sources):
    create, _ = sources
    create("pass_intent/source/holdout")
    directory, args, _ = create("pass_intent/source")
    args["artifact_stage"] = "holdout"
    args["seed"] = 1234
    (directory / "args.json").write_text(json.dumps(args))
    payload = refit.prepare_refit("pass_intent/source", "new")
    assert payload["recipe"]["source_model_id"] == "pass_intent/source/holdout"
    assert payload["args"]["seed"] == 42


@pytest.mark.parametrize("overrides", [{"dataset_loading": "memory"}, {"dataset_buffer_matches": 8}])
def test_operational_overrides_preserve_sampling_semantics(sources, overrides):
    create, _ = sources
    create("pass_intent/source")  # auto resolves to disk
    with pytest.raises(ValueError, match="sampling semantics|sample order"):
        refit.prepare_refit("pass_intent/source", "new", operational=overrides)
    payload = refit.prepare_refit("pass_intent/source", "new", operational={"dataset_cache_dir": "new_cache"})
    assert payload["args"]["dataset_cache_dir"] == "new_cache"


def test_completed_stage_is_immutable_and_cannot_replace_later_delivery(tmp_path):
    from training_state import publish_checkpoint_artifacts, optimizer_snapshot
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    root = tmp_path / "run"
    stages = {}
    for stage in ("fold_1", "final_refit"):
        directory = root / stage
        directory.mkdir(parents=True)
        state = optimizer_snapshot(model, optimizer)
        checkpoint = {**state, "args": {"artifact_stage": stage},
                      "metadata": {"artifact_stage": stage, "status": "completed", "last_epoch": 2},
                      "best": state, "best_acc_weights": state["model"] if stage == "fold_1" else None,
                      "finished": True}
        publish_checkpoint_artifacts(checkpoint, directory)
        stages[stage] = checkpoint
    assert not (root / "best_acc_weights.pt").exists()
    fold = root / "fold_1"
    original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in fold.iterdir()}
    publish_checkpoint_artifacts(stages["fold_1"], fold)
    assert json.loads((root / "metadata.json").read_text())["artifact_stage"] == "final_refit"
    assert original == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in fold.iterdir()}
    (fold / "best_weights.pt").unlink()  # repair interrupted publication
    publish_checkpoint_artifacts(stages["fold_1"], fold)
    assert (fold / "best_weights.pt").exists()
    assert json.loads((root / "metadata.json").read_text())["artifact_stage"] == "final_refit"


def test_crash_capture_resolves_refit_stage(sources, tmp_path, monkeypatch):
    import crash_capture
    import project_config
    create, _ = sources
    create("outcome_scoring/source")
    payload = refit.prepare_refit("outcome_scoring/source", "new")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    monkeypatch.setattr(project_config, "get_model_run_root", lambda task, run_id: tmp_path / task / run_id)
    assert crash_capture._run_directory(["python", "train.py", "--refit-config", str(config)], tmp_path) == tmp_path / "outcome_scoring/new/final_refit"


@pytest.mark.parametrize("split", ["training", "validation"])
def test_empty_usable_split_fails_before_any_epoch(synthetic_training, split):
    run, observations, _, _, root = synthetic_training
    requested = "m0" if split == "training" else "m4"
    (root / "features" / f"{requested}.pt").unlink()
    flags = ["--training-count", "1"] if split == "training" else []
    with pytest.raises(ValueError, match=f"No usable {split} samples"):
        run("empty", flags)
    assert not observations
    metadata = json.loads((root / "saved/pass_intent/empty/metadata.json").read_text())
    provenance = metadata["split_datasets"]["train" if split == "training" else split]
    assert provenance["requested_match_ids"] == [requested]
    assert provenance["loaded_match_count"] == 0
    assert provenance["contributing_match_count"] == 0


@pytest.mark.parametrize("task", ["pass_height", "failure_receiver"])
@pytest.mark.parametrize("version", [None, 1, 2])
def test_refit_reconstruction_preserves_goal_policy(sources, tmp_path, task, version):
    from models.goal_context import goal_policy, resolve_training_goal_settings
    create, _ = sources
    directory, args, _ = create(f"{task}/source")
    args["goal_nodes_aware"] = True
    if version is not None:
        args["goal_context_version"] = version
    if version == 2:
        args["include_goals"] = False
    (directory / "args.json").write_text(json.dumps(args))
    expected = goal_policy(args)
    payload = refit.prepare_refit(f"{task}/source", "new")
    config = tmp_path / "refit.json"
    config.write_text(json.dumps(payload))
    restored = SimpleNamespace(**refit.load_refit_config(config))
    assert resolve_training_goal_settings(restored, vars(restored).copy()) == expected


@pytest.mark.parametrize("version", [None, 1, 2])
def test_actual_refit_entrypoint_preserves_goal_policy(synthetic_training, monkeypatch, version):
    run, _, _, _, root = synthetic_training
    run("goal_source", ["--max-epochs", "1"])
    source_dir = root / "saved/pass_intent/goal_source"
    args = json.loads((source_dir / "args.json").read_text())
    args["feature_run_id"] = "fixture_feature"
    if version is None:
        args.pop("goal_context_version", None)
    else:
        args["goal_context_version"] = version
    (source_dir / "args.json").write_text(json.dumps(args))
    monkeypatch.setattr(refit, "load_feature_run_metadata", lambda *a, **k: {})
    monkeypatch.setattr(refit, "resolve_artifact_split", lambda *a, **k: (args["split_manifest"], "fixture"))
    payload = refit.prepare_refit("pass_intent/goal_source", "goal_refit", saved_root=root / "saved")
    path = root / "goal_recipe.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", ["train.py", "--refit-config", str(path)])
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        faulthandler.disable()
    checkpoint = load_checkpoint(root / "saved/pass_intent/goal_refit/final_refit" / CHECKPOINT_NAME)
    assert checkpoint["args"]["goal_context_version"] == (version or 1)


def test_refit_preserves_saved_resolved_and_historical_auto(sources):
    create, _ = sources
    directory, settings, _ = create("outcome_scoring/source")
    historical = refit.prepare_refit("outcome_scoring/source", "historical")
    assert historical["args"]["dataset_loading"] == "memory"
    settings.update(dataset_loading="auto", resolved_dataset_loading="disk", dataset_buffer_matches=3)
    (directory / "args.json").write_text(json.dumps(settings))
    prepared = refit.prepare_refit("outcome_scoring/source", "prepared", operational={"keep_cache": True})
    assert prepared["args"]["dataset_loading"] == "disk" and prepared["args"]["dataset_buffer_matches"] == 3
    assert prepared["args"]["keep_cache"]
    with pytest.raises(ValueError, match="sampling semantics"):
        refit.prepare_refit("outcome_scoring/source", "wrong", operational={"dataset_loading": "memory"})
    assert wrapper.parse_args(["--resume-id", "outcome_scoring/source", "--keep-cache"]).keep_cache
    assert wrapper.parse_args(["--refit-source-model-id", "outcome_scoring/source", "--keep-cache"]).refit_operational == {"keep_cache": True}


@pytest.mark.parametrize("mode", [True, False])
def test_refit_graph_validation_is_operational_and_defaults_on(sources, mode):
    create, _ = sources
    directory, args, _ = create("pass_intent/source")
    args["graph_validation"] = False
    (directory / "args.json").write_text(json.dumps(args))
    assert refit.prepare_refit("pass_intent/source", "new")["args"]["graph_validation"] is True
    payload = refit.prepare_refit("pass_intent/source", "new", operational={"graph_validation": mode})
    assert payload["args"]["graph_validation"] is mode


@pytest.mark.parametrize("mode", [None, "off", "on"])
def test_actual_refit_validation_uses_invocation_policy(synthetic_training, monkeypatch, mode):
    run, _, _, _, root = synthetic_training
    run("policy_source", ["--max-epochs", "2", "--no-early-stopping", "--graph-validation", "off"])
    source = root / "saved/pass_intent/policy_source"
    settings = json.loads((source / "args.json").read_text())
    settings["feature_run_id"] = "fixture_feature"
    (source / "args.json").write_text(json.dumps(settings))
    monkeypatch.setattr(refit, "load_feature_run_metadata", lambda *a, **kw: {})
    monkeypatch.setattr(refit, "resolve_artifact_split", lambda *a, **kw: (settings["split_manifest"], "fixture"))
    payload = refit.prepare_refit("pass_intent/policy_source", "policy_refit", saved_root=root / "saved",
                                 operational={"graph_validation": False})
    config = root / "policy_recipe.json"
    config.write_text(json.dumps(payload))
    flags = [] if mode is None else ["--graph-validation", mode]
    monkeypatch.setattr(sys, "argv", ["train.py", "--refit-config", str(config), *flags])
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        faulthandler.disable()
    result = load_checkpoint(root / "saved/pass_intent/policy_refit/final_refit" / CHECKPOINT_NAME)
    assert result["args"]["graph_validation"] is (mode != "off")
    assert result["metadata"]["dataset_loading"]["train"]["graph_validation"]["enabled"] is (mode != "off")
