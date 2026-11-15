"""Mixed plots and checkpoints retain values while annotating eligible sources only."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from datatools.hawkeye import build_hawkeye_component_tables
from datatools.viz_helpers import compute_outcome, compute_pass_score
from scripts import inference_visualization_comparison as live
from scripts import saved_visualization_comparison as saved
from scripts import run_and_visualize_hawkeye as direct
from scripts import visualize_action_components as sportec
from scripts import visualize_hawkeye as hawkeye
from scripts.outcome_decomposition_visualization import (
    COMPONENTS, annotate, attach_saved, prepare_saved, report_selection, validate_models,
)
from scripts.visualization_comparison import cached_prediction
from test_outcome_decomposition import configuration
from test_pc_xpass_visualization_comparison import caches
from test_visualization_comparison import (
    _raw_hawkeye, inference_fixture, outcome_pipeline, saved_fixture,
)


PLOTS = ["pass_success", *sorted(COMPONENTS), "outcome_success", "outcome_failure", "pass_score", "physical_xpass"]
ONLY_FLAGS = ["--only-" + name for name in (
    "pass-success", "outcome-scoring", "outcome-conceding", "outcome-success", "outcome-failure", "pass-score")]


def settings(task, mode="decomposition", severity="scalar"):
    args = configuration(severity, task) if mode == "decomposition" else dict(learning_mode=mode, task=task)
    return dict(args, model_id=task + "/" + mode + "_" + severity)


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
@pytest.mark.parametrize("other_mode", ["scalar", "distributional"])
def test_mixed_models_and_reused_state(severity, other_mode):
    scoring = SimpleNamespace(args=settings("outcome_scoring", severity=severity))
    conceding = SimpleNamespace(args=settings("outcome_conceding", mode=other_mode))
    models = dict(outcome_scoring=scoring, outcome_conceding=conceding)
    original = copy.deepcopy([model.args for model in models.values()])
    result = validate_models(SimpleNamespace(decomposition=True), PLOTS, models)
    assert set(result["components"]) == {"outcome_scoring_success", "outcome_scoring_failure"}
    assert scoring.visualize_outcome_decomposition
    assert not conceding.visualize_outcome_decomposition
    validate_models(SimpleNamespace(decomposition=False), PLOTS, models)
    assert not scoring.visualize_outcome_decomposition
    validate_models(SimpleNamespace(decomposition=True), ["pass_score"], models)
    assert not scoring.visualize_outcome_decomposition
    assert original == [model.args for model in models.values()]


def test_eligible_model_definition_stays_strict():
    model = SimpleNamespace(args=dict(task="outcome_scoring", learning_mode="decomposition"))
    with pytest.raises(ValueError, match="decomposition definition"):
        validate_models(SimpleNamespace(decomposition=True), PLOTS, dict(outcome_scoring=model))


def test_no_eligible_notice_is_once_and_needs_no_confirmation(capsys):
    args = SimpleNamespace(decomposition=True)
    result = validate_models(args, ["pass_score", "physical_xpass"], {})
    report_selection(args, result)
    report_selection(args, result)
    assert capsys.readouterr().out.count("no eligible") == 1
    assert args._decomposition_selection["components"] == []


def test_derived_values_strip_inherited_heads():
    frame = pd.DataFrame({"home_1": [.1]}, index=[7])
    frame.attrs.update(show_outcome_decomposition=True, outcome_decomposition={7: {"home_1": [.5, .2]}})
    for source in (frame, frame.iloc[0]):
        combined = compute_outcome(source, source)
        score = compute_pass_score(source, source, source, source, source)
        assert not combined.attrs.get("show_outcome_decomposition")
        assert "outcome_decomposition" not in score.attrs
        assert (combined == 0).all().all()
        assert (score == 0).all().all()
        assert source.attrs["show_outcome_decomposition"]


def test_render_boundary_and_rotation(tmp_path):
    row = pd.Series({"away_1": .1}, name=7)
    row.attrs.update(show_outcome_decomposition=True, outcome_decomposition={7: {"away_1": [.5, .2]}})
    snapshot = pd.DataFrame({"away_1_x": [30.], "away_1_y": [20.]})
    fig, ax = plt.subplots()
    for name in ("pass_score", "outcome_success", "outcome_failure", "physical_xpass", "pass_success"):
        annotate(ax, snapshot, "away", row, component_name=name)
    assert not ax.texts
    annotate(ax, snapshot, "away", row, component_name="outcome_scoring_success", rotate_pitch=True)
    assert [text.get_text() for text in ax.texts] == ["0.500 x 0.200"]
    fig.savefig(tmp_path / "selective_labels.png")
    plt.close(fig)


def saved_source(mode="decomposition", severity="scalar"):
    raw = _raw_hawkeye()
    raw["pass_success"] = [.6, .6]
    for component in COMPONENTS:
        raw[component] = [.1, .1]
    scoring = settings("outcome_scoring", mode, severity)
    conceding = settings("outcome_conceding", "scalar")
    info = dict(run_id="fixture", status="completed", processed_situation_ids=["s"], freeze_ballreceipt=True,
                model_records=dict(outcome_scoring=scoring, outcome_conceding=conceding),
                models={"outcome_scoring": scoring["model_id"], "outcome_conceding": conceding["model_id"]})
    rows = []
    if mode == "decomposition":
        rows = [dict(scene_id="s", frame_id=10, candidate_id=candidate,
                     task="outcome_scoring", execution_branch=branch, model_id=scoring["model_id"],
                     inference_mask_id="unmasked", definition_id=scoring["outcome_decomposition_definition"]["definition_sha256"],
                     prediction=.1, occurrence_probability=.5, conditional_severity=.2)
                for candidate in ("home_1", "home_2") for branch in (0, 1)]
    return raw, info, rows


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
def test_saved_mixed_source_attaches_only_eligible_heads(severity, saved_fixture):
    subject, _ = saved_fixture
    raw, info, rows = saved_source(severity=severity)
    raw.attrs["outcome_decomposition_rows"] = rows
    result = prepare_saved(SimpleNamespace(decomposition=True), raw, info, PLOTS)
    assert set(result["components"]) == {"outcome_scoring_success", "outcome_scoring_failure"}
    tables = build_hawkeye_component_tables(raw, subject)
    for name in COMPONENTS:
        assert bool(tables[name].attrs.get("show_outcome_decomposition")) == name.startswith("outcome_scoring")
        assert (tables[name] == .1).all().all()
    prepare_saved(SimpleNamespace(decomposition=False), raw, info, PLOTS)
    tables = build_hawkeye_component_tables(raw, subject)
    assert not any(table.attrs.get("show_outcome_decomposition") for table in tables.values())


@pytest.mark.parametrize("mode", ["scalar", "distributional", "unknown"])
def test_ineligible_saved_sources_do_not_load_sidecars(mode, monkeypatch):
    raw, info, _ = saved_source(mode=mode)
    if mode == "unknown":
        info = {}
    raw.attrs["outcome_decomposition_path"] = "must-not-be-read.parquet"
    read = Mock(side_effect=AssertionError("unexpected sidecar read"))
    monkeypatch.setattr(pd, "read_parquet", read)
    result = prepare_saved(SimpleNamespace(decomposition=True), raw, info, PLOTS)
    assert not result["components"]
    read.assert_not_called()


def test_saved_exact_checkpoint_fallback_and_unknown_diagnostic(monkeypatch, capsys):
    raw, info, rows = saved_source()
    record = info.pop("model_records")["outcome_scoring"]
    info["models"].pop("outcome_conceding")
    lookup = Mock(return_value=dict(record, args=record))
    monkeypatch.setattr("models.utils.get_model_record", lookup)
    raw.attrs["outcome_decomposition_rows"] = rows
    args = SimpleNamespace(decomposition=True)
    result = prepare_saved(args, raw, info, PLOTS)
    lookup.assert_called_once_with(record["model_id"])
    assert len(result["components"]) == 2
    report_selection(args, result)
    assert "outcome_conceding: learning mode unknown" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "partial", "duplicate", "model_id", "inference_mask_id", "definition_id", "prediction", "product"])
def test_saved_eligible_heads_remain_strict(problem, saved_fixture):
    subject, _ = saved_fixture
    raw, info, rows = saved_source()
    if problem == "partial":
        rows = rows[1:]
    elif problem == "duplicate":
        rows += [dict(rows[0])]
    elif problem in ("model_id", "inference_mask_id", "definition_id"):
        rows[0][problem] = "wrong-source"
    elif problem == "prediction":
        rows[0]["prediction"] = .2
    elif problem == "product":
        rows[0]["conditional_severity"] = .9
    if problem != "missing":
        raw.attrs["outcome_decomposition_rows"] = rows
    with pytest.raises(ValueError):
        prepare_saved(SimpleNamespace(decomposition=True), raw, info, PLOTS)
        build_hawkeye_component_tables(raw, subject)


def test_saved_selected_subset_and_forced_missing_values(saved_fixture):
    subject, _ = saved_fixture
    raw, info, rows = saved_source()
    # Only success is displayed, so failure heads are not required.
    rows = [row for row in rows if row["execution_branch"] == 1]
    for row in rows:
        row.update(prediction=0., occurrence_probability=0., conditional_severity=float("nan"), forced=True)
    raw["outcome_scoring_success"] = 0.
    raw.attrs["outcome_decomposition_rows"] = rows
    prepare_saved(SimpleNamespace(decomposition=True), raw, info, ["outcome_scoring_success", "pass_score"])
    tables = build_hawkeye_component_tables(raw, subject)
    row = tables["outcome_scoring_success"].iloc[0]
    ax = Mock()
    annotate(ax, subject.tracking.iloc[[0]], "home", row, component_name="outcome_scoring_success")
    assert [call.args[0] for call in ax.annotate.call_args_list] == ["- x -", "- x -"]


def test_saved_task_specific_mask_identity(saved_fixture):
    subject, _ = saved_fixture
    raw, info, rows = saved_source()
    info["inference_masks"] = dict(requested=["outcome_scoring:accel"], models={
        "outcome_scoring": dict(configuration_id="mask_scoring")}, configuration_id="combined_mask")
    for row in rows:
        row["inference_mask_id"] = "mask_scoring"
    raw.attrs["outcome_decomposition_rows"] = rows
    prepare_saved(SimpleNamespace(decomposition=True), raw, info, PLOTS)
    assert build_hawkeye_component_tables(raw, subject)["outcome_scoring_success"].attrs["show_outcome_decomposition"]
    info["inference_masks"]["models"] = {}
    with pytest.raises(ValueError, match="establish saved inference mask"):
        prepare_saved(SimpleNamespace(decomposition=True), raw, info, PLOTS)


@pytest.mark.parametrize("reverse", [False, True])
def test_live_four_modes_keep_all_plots_and_values(tmp_path, monkeypatch, outcome_pipeline, reverse):
    script, required, _, _, _, render = outcome_pipeline
    loader = live.load_model
    modes = [("scalar", "scalar"), ("distributional", "scalar"),
             ("decomposition", "scalar"), ("decomposition", "distributional")]
    def load(identifier, device):
        model = loader(identifier, device)
        if identifier.startswith("outcome_"):
            mode, severity = modes[int(identifier.rsplit("/", 1)[1])]
            model.args.update(settings(model.args["task"], mode, severity), model_id=identifier)
        return model
    monkeypatch.setattr(live, "load_model", load)
    def predict(subject, model, **kwargs):
        frame = pd.DataFrame({"home_1": [.1], "home_2": [.1], "away_3": [.1]}, index=[10])
        if model.args["task"].startswith("outcome_"):
            if model.visualize_outcome_decomposition:
                frame.attrs.update(show_outcome_decomposition=True,
                                   outcome_decomposition={10: {candidate: [.5, .2] for candidate in frame.columns}})
            return frame, frame.copy()
        return frame, None
    monkeypatch.setattr(script, "inference_gnn", predict)
    if script is direct:
        monkeypatch.setattr(script, "load_runtime_physical_xpass_visualization_table",
                            lambda *a, **kw: pd.DataFrame({"home_1": [.3], "home_2": [.3], "away_3": [.3]}, index=[10]))
    else:
        monkeypatch.setattr(script, "load_runtime_physical_xpass_visualization_component",
                            lambda *a, **kw: pd.Series({"home_1": .3, "home_2": .3, "away_3": .3}))
    order = list(reversed(range(4))) if reverse else list(range(4))
    flags = []
    for i in order:
        for task in ("outcome_scoring", "outcome_conceding"):
            flags += ["--" + task.replace("_", "-") + "-model-id", f"{task}/{i}"]
    base = [*required, "--compare", *flags, "--pass-success-model-id", "pass_success/fixed",
            *ONLY_FLAGS, "--only-physical-xpass", "--accept-comparison-warnings", "--output-dir", str(tmp_path)]
    if reverse:
        base += ["--inference-mask", "all:accel"]
    results = []
    parse = script.parse_args
    for enabled in (False, True):
        render.reset_mock()
        args = parse([*base, "--run-id", str(enabled), *(["--decomposition"] if enabled else [])])
        monkeypatch.setattr(script, "parse_args", lambda: args)
        script.main()
        root = tmp_path / str(enabled)
        meta = json.loads((root / "metadata.json").read_text())
        assert set(meta["rendered_components"]) == set(PLOTS)
        for variant, i in zip(meta["variants"], order):
            effective = variant["provenance"]["decomposition_selection"]["components"]
            assert set(effective) == (COMPONENTS if enabled and i >= 2 else set())
        for invocation in render.call_args_list:
            component, row = invocation.args[2:4] if script is direct else invocation.args[3:5]
            if component not in COMPONENTS:
                assert not row.attrs.get("show_outcome_decomposition")
        results.append(pd.read_csv(root / "aligned_predictions.csv"))
    pd.testing.assert_frame_equal(*results)


@pytest.mark.parametrize("reverse", [False, True])
def test_saved_comparison_mixed_sources(tmp_path, monkeypatch, saved_fixture, reverse):
    def load(path):
        raw, info, rows = saved_source(mode="scalar" if str(path) == "a" else "decomposition")
        if rows:
            raw.attrs["outcome_decomposition_rows"] = rows
        return raw, info
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", load)
    render = Mock(side_effect=lambda *a, **kw: Image.new("RGB", (80, 60)))
    monkeypatch.setattr(hawkeye, "render_frame_image", render)
    ids = ["b", "a"] if reverse else ["a", "b"]
    args = hawkeye.parse_args(["--compare", "--component-run-id", ids[0], "--component-run-id", ids[1],
                              "--decomposition", *ONLY_FLAGS, "--accept-comparison-warnings",
                              "--output-dir", str(tmp_path), "--run-id", "saved"])
    saved.run_comparison(args, hawkeye)
    metadata = json.loads((tmp_path / "saved/metadata.json").read_text())
    for variant in metadata["variants"]:
        effective = variant["provenance"]["decomposition_selection"]["components"]
        assert set(effective) == ({"outcome_scoring_success", "outcome_scoring_failure"} if variant["id"] == "b" else set())
    assert render.call_count == 16


def test_prediction_cache_separates_annotation_state():
    model = SimpleNamespace(args=dict(task="outcome_scoring"))
    predict = Mock(return_value=(pd.DataFrame({"home_1": [.1]}, index=[7]), None))
    cache, scene = {}, object()
    for enabled in (False, False, True, True, False):
        model.visualize_outcome_decomposition = enabled
        cached_prediction(cache, predict, scene, model)
    assert predict.call_count == 2


@pytest.mark.parametrize("mode", ["scalar", "decomposition"])
def test_saved_cache_comparison_prepares_fixed_source_heads(tmp_path, monkeypatch, saved_fixture, caches, mode):
    monkeypatch.setattr(hawkeye, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    raw, info, rows = saved_source(mode=mode)
    if rows:
        raw.attrs["outcome_decomposition_rows"] = rows
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", lambda path: (raw, info))
    render = Mock(side_effect=lambda *a, **kw: Image.new("RGB", (80, 60)))
    monkeypatch.setattr(hawkeye, "render_frame_image", render)
    args = hawkeye.parse_args(["--compare", "--component-run-id", "fixed", "--pc-xpass-id", "a", "--pc-xpass-id", "b",
                              "--xpass-version", "top10",
                              "--decomposition", *ONLY_FLAGS, "--only-physical-xpass", "--accept-comparison-warnings",
                              "--output-dir", str(tmp_path), "--run-id", "fixed_source"])
    saved.run_comparison(args, hawkeye)
    metadata = json.loads((tmp_path / "fixed_source/metadata.json").read_text())
    assert metadata["comparison_mode"] == "pc_xpass"
    assert render.call_count == 18
    for variant in metadata["variants"]:
        assert bool(variant["provenance"]["decomposition_selection"]["components"]) == (mode == "decomposition")
    assert sum(bool(call.args[3].attrs.get("show_outcome_decomposition")) for call in render.call_args_list) == (
        4 if mode == "decomposition" else 0)
