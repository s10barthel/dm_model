from argparse import Namespace
import json
from unittest.mock import Mock

import pandas as pd
import pytest

import pc_xpass_match_cache as match_cache
import physical_pass_model as physics
import project_config
from scripts import saved_visualization_xpass as panel
from scripts import visualize_benchmark as bv
from scripts import visualize_hawkeye as hv
from scripts import visualize_skillcorner as sv


def write_cache(path, **changes):
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "pc_xpass", "cache_format": match_cache.FORMAT,
        "available_metrics": ["max_xpass", "top10_xpass", "top25_xpass", "top_pass10_xpass"],
        "top_pass_definition": physics.PC_XPASS_TOP_PASS_DEFINITION,
    }
    metadata.update(changes)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def component(**changes):
    metadata = {
        "run_id": "component", "physical_xpass_runtime_source": "pc_xpass",
        "physical_xpass_cache_summary": {"physical_xpass_required": True},
        "x_pass_version": "top25",
    }
    metadata.update(changes)
    return metadata


def args(**changes):
    options = dict(show_physical_xpass=True, physical_cache_dir=None,
                   pc_xpass_id=None, x_pass_version=None, pc_xpass=False)
    options.update(changes)
    return Namespace(**options)


@pytest.fixture
def version_root(tmp_path, monkeypatch):
    root = tmp_path / "pc_xpass"
    monkeypatch.setattr(project_config, "PC_XPASS_DIR", root)
    return root


@pytest.mark.parametrize("dataset", ["hawkeye", "benchmark", "skillcorner", "hawkeye_loc"])
def test_inherits_pinned_id_and_metric_without_mutating_cli(version_root, dataset):
    root = version_root / "saved" if dataset != "hawkeye_loc" else version_root / "hawkeye_loc" / "saved"
    write_cache(root)
    expected = root if dataset == "hawkeye_loc" else write_cache(root / dataset)
    # Neither a conflicting saved path nor a latest pointer may change the pinned ID.
    (version_root / "latest.json").write_text('{"run_id": "different"}')
    metadata = component(pc_xpass_id="saved", physical_cache_dir="wrong", x_pass_version="top-pass10")
    cli, original = args(), args()
    settings = panel.resolve_saved_xpass(cli, metadata, dataset)
    assert settings.cache_dir == expected
    assert settings.metric == "top_pass10_xpass"
    assert settings.pc_xpass_id == "saved"
    assert settings.namespace == ("hawkeye_loc" if dataset == "hawkeye_loc" else "normal")
    assert settings.metadata()["physical_xpass_resolution"]["inherited_fields"] == ["cache", "metric"]
    assert vars(cli) == vars(original)


@pytest.mark.parametrize("mode", ["loc", "freeze"])
def test_legacy_location_inherits_directory_and_structured_invocation(tmp_path, mode):
    cache = write_cache(tmp_path / "legacy")
    metadata = {
        "inference_mode": mode, "pc_xpass_cache_dir": str(cache),
        "invocation": {"effective_cli_options": {"x_pass_version": "top-pass10"}},
    }
    settings = panel.resolve_saved_xpass(args(), metadata, "hawkeye_loc")
    assert settings.cache_dir == cache and settings.x_pass_version == "top-pass10"
    assert settings.component_usage == "pc_xpass"


def test_dedicated_settings_precede_invocation_and_incidental_metric(tmp_path):
    cache = write_cache(tmp_path / "cache")
    metadata = component(x_pass_version=None, physical_xpass_metric="topmean_xpass",
                         pc_xpass_settings={"x_pass_version": "top25"}, physical_cache_dir=str(cache),
                         invocation={"effective_cli_options": {"x_pass_version": "top10"}})
    assert panel.resolve_saved_xpass(args(), metadata, "hawkeye").x_pass_version == "top25"


@pytest.mark.parametrize("column,version", [("max_xpass", "max"), ("top25_xpass", "top25"),
                                            ("top_pass10_xpass", "top-pass10")])
def test_metric_column_supplies_unambiguous_selector(tmp_path, column, version):
    cache = write_cache(tmp_path / "cache")
    metadata = component(x_pass_version=None, physical_xpass_metric=column, physical_cache_dir=str(cache))
    assert panel.resolve_saved_xpass(args(), metadata, "hawkeye").x_pass_version == version


@pytest.mark.parametrize("override", ["metric", "directory", "id", "complete"])
def test_overrides_replace_only_requested_fields(version_root, override):
    root = write_cache(version_root / "saved")
    original = write_cache(root / "hawkeye")
    other = write_cache(version_root / "other")
    selected = write_cache(other / "hawkeye")
    metadata = component(pc_xpass_id="saved")
    options = {}
    if override in {"metric", "complete"}:
        options["x_pass_version"] = "max"
    if override in {"directory", "complete"}:
        options["physical_cache_dir"] = str(selected)
    if override == "id":
        options["pc_xpass_id"] = "other"
    settings = panel.resolve_saved_xpass(args(**options), metadata, "hawkeye")
    assert settings.cache_dir == (original if override == "metric" else selected)
    assert settings.x_pass_version == ("max" if override in {"metric", "complete"} else "top25")
    assert settings.overrides == options


@pytest.mark.parametrize("usage", ["disabled", "unknown", "ordinary_physical"])
@pytest.mark.parametrize("explicit", ["none", "flag", "metric", "cache", "complete"])
def test_non_pc_runs_require_explicit_cache_and_metric(tmp_path, usage, explicit):
    cache = write_cache(tmp_path / "cache")
    metadata = component(physical_cache_dir=str(cache))
    if usage == "disabled":
        metadata["physical_xpass_cache_summary"]["physical_xpass_required"] = False
    elif usage == "unknown":
        metadata.pop("physical_xpass_cache_summary")
    else:
        metadata["physical_xpass_runtime_source"] = physics.PHYSICAL_XPASS_SOURCE
    options = {}
    if explicit == "flag":
        options["pc_xpass"] = True
    if explicit in {"metric", "complete"}:
        options["x_pass_version"] = "max"
    if explicit in {"cache", "complete"}:
        options["physical_cache_dir"] = str(cache)
    if explicit == "complete":
        result = panel.resolve_saved_xpass(args(**options), metadata, "hawkeye")
        assert result.component_usage == usage
        assert result.x_pass_version == "max"
        assert result.metadata()["physical_xpass_resolution"]["inherited_fields"] == []
    else:
        with pytest.raises(ValueError, match="Component run 'component'.*requires both an explicit"):
            panel.resolve_saved_xpass(args(**options), metadata, "hawkeye")


@pytest.mark.parametrize("record", ["summary", "invocation"])
def test_disabled_usage_overrules_location_metadata(tmp_path, record):
    metadata = component(inference_mode="loc", pc_xpass_settings={"x_pass_version": "top10"},
                         physical_cache_dir=str(write_cache(tmp_path / "cache")))
    if record == "summary":
        metadata["physical_xpass_cache_summary"]["physical_xpass_required"] = False
    else:
        metadata.pop("physical_xpass_cache_summary")
        metadata["invocation"] = {"effective_cli_options": {"use_physical_xpass": False}}
    with pytest.raises(ValueError, match="usage is disabled"):
        panel.resolve_saved_xpass(args(), metadata, "hawkeye_loc")


@pytest.mark.parametrize("missing", ["cache", "metric"])
def test_incomplete_pc_settings_require_only_missing_override(tmp_path, missing):
    cache = write_cache(tmp_path / "cache")
    metadata = component(physical_cache_dir=str(cache))
    metadata.pop("physical_cache_dir" if missing == "cache" else "x_pass_version")
    with pytest.raises(ValueError, match="No recorded pc-xPass"):
        panel.resolve_saved_xpass(args(), metadata, "hawkeye")
    options = {"physical_cache_dir": str(cache)} if missing == "cache" else {"x_pass_version": "max"}
    assert panel.resolve_saved_xpass(args(**options), metadata, "hawkeye") is not None


@pytest.mark.parametrize("invocation_base", [True, False])
def test_recorded_relative_paths_use_saved_base(tmp_path, monkeypatch, invocation_base):
    base = tmp_path / "saved_working_directory"
    cache = write_cache(base / "cache")
    monkeypatch.setattr(panel, "PROJECT_ROOT", base)
    monkeypatch.chdir(tmp_path)
    metadata = component(physical_cache_dir="cache")
    if invocation_base:
        monkeypatch.setattr(panel, "PROJECT_ROOT", tmp_path)
        metadata["invocation"] = {"working_directory": str(base)}
    assert panel.resolve_saved_xpass(args(), metadata, "hawkeye").cache_dir == cache
    explicit = write_cache(tmp_path / "cache")
    assert panel.resolve_saved_xpass(args(physical_cache_dir="cache"), metadata, "hawkeye").cache_dir == explicit


@pytest.mark.parametrize("problem", ["missing", "ordinary", "unknown_source", "unavailable_metric", "wrong_format", "conflict"])
def test_invalid_caches_fail_before_loading_rows(tmp_path, problem):
    cache = write_cache(tmp_path / "cache")
    dataset = "hawkeye"
    options = dict(physical_cache_dir=str(cache), x_pass_version="top25")
    error, message = ValueError, "pc_xpass"
    if problem == "missing":
        options["physical_cache_dir"] = str(cache / "missing")
        error, message = FileNotFoundError, "metadata not found"
    elif problem == "ordinary":
        write_cache(cache, source=physics.PHYSICAL_XPASS_SOURCE)
    elif problem == "unknown_source":
        write_cache(cache, source=None)
    elif problem == "unavailable_metric":
        write_cache(cache, available_metrics=["max_xpass"])
        message = "no top25_xpass"
    elif problem == "wrong_format":
        write_cache(cache, cache_format="old")
        dataset, message = "skillcorner", "Unsupported"
    else:
        options["pc_xpass_id"] = "also_selected"
        message = "cannot be combined"
    with pytest.raises(error, match=message):
        panel.resolve_saved_xpass(args(**options), {}, dataset)


def test_pinned_missing_id_never_falls_back_to_recorded_path_or_latest(version_root):
    cache = write_cache(version_root / "available")
    (version_root / "latest.json").write_text('{"run_id":"available"}')
    with pytest.raises(FileNotFoundError, match="missing"):
        panel.resolve_saved_xpass(args(), component(pc_xpass_id="missing", physical_cache_dir=str(cache)), "hawkeye")


def test_disabled_panel_never_accesses_cache(monkeypatch):
    read = Mock(side_effect=AssertionError("unexpected cache access"))
    monkeypatch.setattr(panel.versions, "read_metadata", read)
    assert panel.resolve_saved_xpass(args(show_physical_xpass=False, pc_xpass_id="missing"), {}, "hawkeye") is None
    read.assert_not_called()


@pytest.mark.parametrize("script,required", [(hv, []), (bv, []), (sv, ["--match-id", "m", "--index", "1"])])
def test_saved_cli_preserves_omissions_and_accepts_pc_metrics_without_flag(script, required):
    assert script.parse_args(required).x_pass_version is None
    for selector in ["--top-pass", "--top_pass"]:
        assert script.parse_args([*required, selector, "10"]).x_pass_version == "top-pass10"
    for selector in ["--xpass-version", "--x-pass-version", "--x_pass_version"]:
        assert script.parse_args([*required, selector, "top25"]).x_pass_version == "top25"
    with pytest.raises(SystemExit):
        script.parse_args([*required, "--xpass-version", "noise-kernel"])
    with pytest.raises(SystemExit):
        script.parse_args([*required, "--top-pass", "10", "--xpass-version", "top25"])
    with pytest.raises(ValueError, match="cannot be combined"):
        script.parse_args([*required, "--pc-xpass-id", "saved", "--physical-cache-dir", "cache"])


def test_resolved_panel_loads_cached_values_and_rejects_missing_rows(tmp_path):
    cache = write_cache(tmp_path / "cache")
    (cache / "matches").mkdir()
    pd.DataFrame([{"match_id": "s", "action_index": 3, "home_2__top25_xpass": .42}]).to_parquet(
        cache / "matches" / "s.parquet", index=False)
    # Hawkeye uses the event/frame cache layout rather than SkillCorner possessions.
    write_cache(cache, cache_format=None)
    settings = panel.resolve_saved_xpass(args(), component(physical_cache_dir=str(cache)), "hawkeye")
    table = physics.load_runtime_physical_xpass_visualization_table(
        settings.cache_dir, "s", [3], metric=settings.metric, x_pass_version=settings.x_pass_version)
    assert table.loc[3, "home_2"] == .42
    with pytest.raises(KeyError, match="action_index=4"):
        physics.load_runtime_physical_xpass_visualization_table(
            settings.cache_dir, "s", [4], metric=settings.metric, x_pass_version=settings.x_pass_version)
