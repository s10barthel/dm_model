import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import pc_xpass_versions as versions
import project_config as config
from models import utils as model_utils
from scripts import run_relevant_models as script


SCHEMA = {"node_in_dim": 26, "edge_in_dim": 5, "add_v_edge_features": True,
          "add_relative_speed_edge_features": True}
SOURCE_CONTEXT = {"source_feature_run_ids": {"pass_success": "feature_source"}}


@pytest.fixture
def feature_runs(tmp_path, monkeypatch):
    root = tmp_path / "features"
    monkeypatch.setattr(config, "FEATURE_RUNS_DIR", root)
    monkeypatch.setattr(model_utils, "FEATURE_RUNS_DIR", root)
    for name, created_at in (("feature_source", "2026-06-28"), ("feature_cache", "2026-09-10")):
        path = root / name
        config.get_resolved_action_dir("model", root=path).mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({
            "created_at": created_at, "intended_receiver_modes": ["model"],
        }), encoding="utf-8")
    monkeypatch.setattr(model_utils, "infer_feature_graph_schema", lambda path: dict(SCHEMA))
    return root


@pytest.fixture
def cache_dir(tmp_path, feature_runs):
    path = tmp_path / "pc_xpass" / "cache_version" / "sportec"
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({
        "cache_format": "match_physical_states_v3",
        "source_inputs": {"feature_run_id": "feature_cache"},
    }), encoding="utf-8")
    (path.parent / "metadata.json").write_text("{}", encoding="utf-8")
    (path.parent.parent / "latest.json").write_text('{"run_id": "cache_version"}', encoding="utf-8")
    return path


def select(cache_dir, model_args=None, **overrides):
    args = SimpleNamespace(use_physical_xpass=True, **overrides)
    return script.resolve_component_feature_run_context(
        args, SOURCE_CONTEXT, None, SCHEMA, str(cache_dir), model_args or {},
    )


def test_cache_feature_run_outside_model_candidates_is_selected(cache_dir):
    selected = select(cache_dir)
    assert selected["feature_run_id"] == "feature_cache"
    assert selected["selection"] == "physical_xpass_cache"
    assert selected["candidate_feature_run_ids"] == ["feature_source"]


@pytest.mark.parametrize("model_args", [
    {"task": "pass_success", "inference_use_physical_xpass": True},
    {"task": "pass_success", "use_physical_xpass": True, "model_variant": "gat_plus_phys_feature"},
])
def test_checkpoint_required_physical_xpass_prefers_cache(cache_dir, model_args):
    args = SimpleNamespace(use_physical_xpass=False)
    assert script.model_uses_physical_xpass(model_args) or script.inference_uses_physical_xpass(model_args)
    selected = script.resolve_component_feature_run_context(
        args, SOURCE_CONTEXT, None, SCHEMA, str(cache_dir), model_args,
    )
    assert selected["selection"] == "physical_xpass_cache"


def test_explicit_feature_run_does_not_read_cache_metadata(cache_dir):
    (cache_dir / "metadata.json").unlink()
    selected = select(cache_dir, feature_run_id="feature_source")
    assert selected["feature_run_id"] == "feature_source"
    assert selected["selection"] == "explicit"


@pytest.mark.parametrize("disabled", [False, True])
def test_unused_or_disabled_physical_cache_keeps_existing_selection(cache_dir, disabled):
    (cache_dir / "metadata.json").unlink()
    args = SimpleNamespace(use_physical_xpass=disabled, no_physical_cache=disabled)
    selected = script.resolve_component_feature_run_context(
        args, SOURCE_CONTEXT, None, SCHEMA, str(cache_dir), {},
    )
    assert selected["feature_run_id"] == "feature_source"
    assert selected["selection"] == "newest_compatible"


@pytest.mark.parametrize("metadata", [
    None, "{invalid json", "[]", "{}", '{"source_inputs": []}',
    '{"source_inputs": {"feature_run_id": null}}',
    '{"source_inputs": {"feature_run_id": 123}}',
    '{"source_inputs": {"feature_run_id": "../feature_source"}}',
])
def test_missing_or_invalid_cache_metadata_fails(cache_dir, metadata):
    path = cache_dir / "metadata.json"
    if metadata is None:
        path.unlink()
    else:
        path.write_text(metadata, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        select(cache_dir)
    message = str(error.value)
    assert str(cache_dir) in message
    assert "--feature-run-id" in message
    assert "strict cache provenance checks" in message


@pytest.fixture
def main_invocation(tmp_path, monkeypatch, cache_dir):
    """Exercise main's selection and exports without loading real matches/models."""
    args = script.parse_args([])
    args.device = "cpu"
    args.output_dir = str(tmp_path / "components")
    args.run_id = "component_test"
    args.use_physical_xpass = True
    args.pc_xpass = True
    monkeypatch.setattr(config, "PC_XPASS_DIR", cache_dir.parent.parent)
    tasks = ("action_intent", "pass_intent", "pass_success", "outcome_scoring", "outcome_conceding")
    model_ids = {task: task for task in tasks}
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script, "resolve_model_selection", lambda **kwargs: (model_ids, SOURCE_CONTEXT, None))
    monkeypatch.setattr(script, "load_model", lambda task, device: SimpleNamespace(args={"task": task}))
    monkeypatch.setattr(script, "configure_auxiliary_models", lambda *args: None)
    monkeypatch.setattr(script, "validate_model_graph_schemas", lambda models: dict(SCHEMA))
    monkeypatch.setattr(script, "configure_inference_masks", lambda *args: {})
    monkeypatch.setattr(script, "get_model_provenance", lambda model: {"feature_signature": model})
    monkeypatch.setattr(script, "checked_split_selector", lambda *args: {})
    monkeypatch.setattr(script, "split_metadata", lambda context: {})
    monkeypatch.setattr(script, "physical_xpass_inference_lookup_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(script, "resolve_match_ids", lambda *args, **kwargs: ["match"])
    match = SimpleNamespace(possession_report={}, sportec_state_plan=SimpleNamespace(manifest=lambda: {}))
    load_match = Mock(return_value=match)
    monkeypatch.setattr(script, "load_match", load_match)
    state = SimpleNamespace(event_index=1)
    monkeypatch.setattr(script, "build_sportec_possessions", lambda *args, **kwargs: [(state, {})])
    frame = pd.DataFrame({"home_1": [.5]})
    monkeypatch.setattr(script, "infer_skillcorner_components", lambda *args, **kwargs: {
        "action_intent": frame, "pass_success": frame,
    })
    monkeypatch.setattr(script, "sportec_component_table", lambda prediction, state: prediction)
    return args, load_match


@pytest.mark.parametrize("selector", ["latest", "id", "directory", "runtime_directory"])
def test_main_selects_cache_feature_run_and_exports_metadata(main_invocation, cache_dir, selector, capsys, monkeypatch):
    args, load_match = main_invocation
    if selector == "id":
        args.pc_xpass_id = "cache_version"
    elif selector in ("directory", "runtime_directory"):
        args.physical_cache_dir = str(cache_dir)
        args.pc_xpass = selector == "directory"
    resolve_cache = Mock(wraps=versions.cache_dir)
    monkeypatch.setattr(versions, "cache_dir", resolve_cache)
    script.main()
    root = Path(args.output_dir) / args.run_id
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["runtime_feature_run_id"] == "feature_cache"
    assert metadata["runtime_feature_run_selection"] == "physical_xpass_cache"
    assert metadata["physical_cache_dir"] == str(cache_dir)
    assert metadata["source_feature_run_ids"] == SOURCE_CONTEXT["source_feature_run_ids"]
    assert (root / "match" / "pass_success.parquet").is_file()
    assert load_match.call_args.kwargs["feature_root"].name == "feature_cache"
    assert resolve_cache.call_count == (1 if selector in ("latest", "id") else 0)
    assert "Runtime feature run: feature_cache (selection: physical_xpass_cache)" in capsys.readouterr().out


@pytest.mark.parametrize("metadata", [None, "{invalid json"])
def test_main_unreadable_versioned_cache_stops_with_selection_context(main_invocation, cache_dir, metadata):
    args, load_match = main_invocation
    path = cache_dir / "metadata.json"
    if metadata is None:
        path.unlink()
    else:
        path.write_text(metadata, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        script.main()
    message = str(error.value)
    assert str(cache_dir.parent.parent) in message
    assert f"ID: {args.pc_xpass_id or 'latest'}" in message
    assert "--feature-run-id" in message
    load_match.assert_not_called()


@pytest.mark.parametrize("failure, expected", [
    ("missing_run", "does not exist"),
    ("missing_actions", "Resolved actions"),
    ("schema", "incompatible"),
    ("mode", "intended_receiver_mode='original'"),
    ("metadata", "metadata not found"),
])
def test_main_invalid_cache_feature_run_stops_before_match_inference(
    main_invocation, cache_dir, feature_runs, monkeypatch, failure, expected,
):
    args, load_match = main_invocation
    path = cache_dir / "metadata.json"
    if failure == "missing_run":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["source_inputs"]["feature_run_id"] = "feature_missing"
        path.write_text(json.dumps(metadata), encoding="utf-8")
    elif failure == "missing_actions":
        config.get_resolved_action_dir("model", root=feature_runs / "feature_cache").rmdir()
    elif failure == "schema":
        monkeypatch.setattr(model_utils, "infer_feature_graph_schema", lambda path:
                            {"node_in_dim": 1, "edge_in_dim": 2} if path.parent.name == "feature_cache" else dict(SCHEMA))
    elif failure == "mode":
        args.intended_receiver_mode = "original"
    else:
        # Keep cache selection valid, but fail when reading its source feature metadata.
        monkeypatch.setattr(versions, "cache_dir", lambda *args: cache_dir)
        path.unlink()
    with pytest.raises(ValueError) as error:
        script.main()
    assert expected in str(error.value)
    assert str(cache_dir) in str(error.value)
    if failure != "metadata":
        assert ("feature_missing" if failure == "missing_run" else "feature_cache") in str(error.value)
    load_match.assert_not_called()
    assert not (Path(args.output_dir) / args.run_id).exists()
