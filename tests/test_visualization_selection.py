import argparse

import numpy as np
import pandas as pd
import pytest

from scripts.visualization_selection import (
    GROUP_TO_COMPONENTS, add_component_selection_args, available_model_components,
    filter_component_selection, resolve_component_selection,
)
from scripts.visualization_availability import saved_component_inputs, select_inference_components


def select(*flags):
    parser = argparse.ArgumentParser()
    add_component_selection_args(parser, include_success_intent=True)
    return resolve_component_selection(parser.parse_args(flags), include_success_intent=True)


def test_defaults_include_all_standalone_groups():
    selection = select()
    assert set(selection.requested_component_groups) == set(GROUP_TO_COMPONENTS) - {"compare_pass_success"}
    assert selection.explicit_component_groups == []


@pytest.mark.parametrize("group", [g for g in GROUP_TO_COMPONENTS if g != "compare_pass_success"])
def test_each_group_can_be_selected_or_excluded(group):
    flag = group.replace("_", "-")
    assert select(f"--only-{flag}").rendered_components == list(GROUP_TO_COMPONENTS[group])
    assert not set(GROUP_TO_COMPONENTS[group]) & set(select(f"--no-{flag}").rendered_components)
    with pytest.raises(SystemExit):
        select(f"--only-{flag}", f"--no-{flag}")


def test_programmatic_same_component_conflict():
    args = argparse.Namespace(only_model_pass_success=True, no_model_pass_success=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_component_selection(args)


def test_other_components_can_be_excluded_and_dependencies_remain_required():
    selection = select("--only-pass-score", "--no-pass-success", "--no-outcome-scoring")
    assert selection.rendered_components == ["pass_score"]
    assert set(selection.required_component_groups) == {"pass_success", "outcome_scoring", "outcome_conceding", "pass_score"}
    assert selection == select("--no-outcome-scoring", "--no-pass-success", "--only-pass-score")


def test_partial_models_filter_defaults_but_explicit_missing_inputs_fail():
    available = available_model_components({"pass_success": "pass_success/a"})
    selection = filter_component_selection(select(), available)
    assert selection.rendered_components == ["pass_success", "model_pass_success"]
    assert "pass_height" in selection.skipped_components
    with pytest.raises(ValueError, match="pass_score.*missing"):
        filter_component_selection(select("--only-pass-score"), available)
    with pytest.raises(ValueError, match="no visualization outputs"):
        filter_component_selection(select(), set())


def test_legacy_additions_are_explicit_and_warn():
    with pytest.warns(FutureWarning):
        selection = select("--only-physical-xpass", "--show-model-pass-success")
    assert selection.rendered_components == ["model_pass_success", "physical_xpass"]
    with pytest.raises(ValueError, match="model_pass_success"):
        filter_component_selection(selection, {"physical_xpass"})


def test_export_placeholders_are_not_available_but_missing_model_values_are():
    raw = pd.DataFrame({"pass_success": [.8], "pass_height": [np.nan], "model_pass_success": [np.nan]})
    assert saved_component_inputs(raw) == {"pass_success", "model_pass_success"}
    assert "pass_height" in saved_component_inputs(raw, {"models": {"pass_height": {}}})
    assert saved_component_inputs({"pass_height": pd.DataFrame(), "pass_success": pd.DataFrame({"a": [.8]})}) == {"pass_success"}


def test_inference_selection_has_no_implicit_checkpoint_discovery():
    parser = argparse.ArgumentParser()
    add_component_selection_args(parser)
    args = parser.parse_args([])
    selection = select_inference_components(args, {"pass_intent": "pass_intent/a"})
    assert selection.rendered_components == ["pass_intent", "physical_xpass"]
    assert selection.required_component_groups == ["pass_intent"]


@pytest.mark.parametrize("module,required", [
    ("visualize_action_components", ["--match-id", "m"]), ("run_and_visualize_hawkeye", []),
    ("visualize_hawkeye", []), ("visualize_benchmark", []),
    ("visualize_skillcorner", ["--match-id", "m", "--index", "1"]),
])
def test_all_parsers_validate_same_component_only(module, required):
    from importlib import import_module
    command = import_module(f"scripts.{module}")
    with pytest.raises(SystemExit):
        command.parse_args([*required, "--only-model-pass-success", "--no-model-pass-success"])
    args = command.parse_args([*required, "--only-pass-score", "--no-pass-success"])
    assert resolve_component_selection(args).rendered_components == ["pass_score"]


def test_skillcorner_partial_loading_is_opt_in(tmp_path):
    from datatools.skillcorner import load_skillcorner_component_tables
    folder = tmp_path / "m"
    folder.mkdir()
    pd.DataFrame({"1": [.5]}, index=[10]).to_parquet(folder / "pass_intent.parquet")
    assert set(load_skillcorner_component_tables(tmp_path, "m", allow_partial=True)) == {"pass_intent"}
    with pytest.raises(FileNotFoundError):
        load_skillcorner_component_tables(tmp_path, "m")


def test_optional_cache_probe_does_not_hide_invalid_cache_errors():
    from types import SimpleNamespace
    from physical_pass_model import PhysicalXpassMetricUnavailable
    from scripts.visualization_availability import probe_saved_xpass
    args = SimpleNamespace()
    def missing(*a, **kw):
        raise PhysicalXpassMetricUnavailable("missing metric")
    assert probe_saved_xpass(args, {}, "hawkeye", missing) == (None, "missing metric")
    with pytest.raises(PhysicalXpassMetricUnavailable):
        probe_saved_xpass(args, {}, "hawkeye", missing, required=True)
    def malformed(*a, **kw):
        raise ValueError("invalid metadata")
    with pytest.raises(ValueError, match="invalid metadata"):
        probe_saved_xpass(args, {}, "hawkeye", malformed)
