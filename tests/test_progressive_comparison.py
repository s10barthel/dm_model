"""Publication order, warning reuse, and interrupted comparison exports."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
from unittest.mock import Mock

from models import membership, utils
from test_model_membership import record


from scripts.comparison_warnings import ComparisonCancelled
from scripts.visualization_comparison import (
    ComparisonExport, ComparisonGroup, Variant, render_comparison_groups,
    render_independent_groups, save_comparison_png,
)


def scene(identifier, frame=1, candidates=("one", "two"), geometry=None, render=None):
    return {"scene": identifier, "frame": frame, "folder": identifier, "suffix": "",
            "title": identifier, "geometry": geometry or {},
            "tables": [{"pass_success": pd.DataFrame([dict.fromkeys(candidates, .3)], index=[frame])}],
            "render": render or (lambda *a: Image.new("RGB", (80, 60), "green"))}


def report_for(root, accept=True):
    return ComparisonExport(root, [Variant("a", "a"), Variant("b", "b")],
                            SimpleNamespace(accept_comparison_warnings=accept), "checkpoints")


def test_reordered_plans_match_semantic_keys_and_export_union(tmp_path):
    report = report_for(tmp_path)
    loaded = []
    def group(variant, identifier):
        def load():
            if identifier == "second":
                assert (tmp_path / "first/pass_success.png").exists()
            loaded.append((variant, identifier))
            yield scene(identifier)
        return ComparisonGroup(identifier, load)
    report.start()
    render_comparison_groups(report, [[group("a", "first"), group("a", "second")],
                                     [group("b", "second"), group("b", "first"), group("b", "extra")]],
                             ["pass_success"])
    report.finish()
    assert loaded == [("a", "first"), ("b", "first"), ("a", "second"), ("b", "second"), ("b", "extra")]
    assert len(report.outputs) == 3
    values = pd.read_csv(tmp_path / "aligned_predictions.csv")
    assert values.loc[values.scene_id.eq("extra"), "prediction:a"].isna().all()


def test_each_component_saved_before_rendering_next_component(tmp_path):
    report = report_for(tmp_path)
    def render(component, row):
        if component == "pass_intent":
            assert (tmp_path / "first/pass_success.png").exists()
        return Image.new("RGB", (80, 60))
    current = scene("first", render=render)
    current["tables"][0]["pass_intent"] = current["tables"][0]["pass_success"].copy()
    render_independent_groups(report, [[current], [current]], ["pass_success", "pass_intent"])
    assert len(report.outputs) == 2


def test_warning_acceptance_reused_across_frames_and_counts(tmp_path, monkeypatch):
    report = report_for(tmp_path, accept=False)
    prompts = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    def answer(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "yes"
        # The new geometry warning waits while earlier images stay visible.
        assert (tmp_path / "first/pass_success.png").exists()
        assert (tmp_path / "second/pass_success.png").exists()
        assert not (tmp_path / "third/pass_success.png").exists()
        return "no"
    monkeypatch.setattr("builtins.input", answer)
    a = [scene("first"), scene("second", 2, ("three", "four", "five")), scene("third", 3)]
    b = [scene("first", candidates=("one",)), scene("second", 2, ("three", "four")),
         scene("third", 3, candidates=("one",), geometry={"offset": 2})]
    groups = [[ComparisonGroup(item["scene"], lambda item=item: [item]) for item in variant]
              for variant in (a, b)]
    with pytest.warns(UserWarning, match="Candidate membership"):
        with pytest.raises(ComparisonCancelled) as failure:
            render_comparison_groups(report, groups, ["pass_success"])
    report.failed(failure.value)
    assert len(prompts) == 2
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "cancelled"
    assert len(metadata["output_paths"]) == 2
    occurrences = [message for message in metadata["warnings"] if message.startswith("Candidate membership")]
    assert len(occurrences) == 3  # Every occurrence is retained, including accepted repeats.
    assert len({metadata["warning_keys"][message] for message in occurrences}) == 1


def test_different_membership_pattern_requires_new_acceptance(tmp_path, monkeypatch):
    report = report_for(tmp_path, accept=False)
    prompts = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "yes")
    report.record("first", 1, "pass_success", [pd.Series({"one": .3}), pd.Series({"two": .2})])
    report.policy.confirm()
    report.record("second", 2, "pass_success", [pd.Series({"three": .3}), pd.Series(dtype=float)])
    report.policy.confirm()
    assert len(prompts) == 2


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_incomplete_png_never_published_and_temporary_file_cleaned(tmp_path, error):
    path = tmp_path / "plot.png"
    with Image.new("RGB", (30, 20)) as image:
        save_comparison_png(image, path)
    original = path.read_bytes()
    def partial_save(temporary, **kwargs):
        assert kwargs["format"] == "PNG"
        temporary.write_bytes(b"incomplete")
        raise error("interrupted write")
    with pytest.raises(error, match="interrupted write"):
        save_comparison_png(SimpleNamespace(save=partial_save), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["sportec_known", "sportec_missing", "hawkeye"])
def test_membership_reporter_is_shared_across_checkpoints_and_publication(tmp_path, monkeypatch, capsys, kind):
    lookup = Mock(return_value=record())
    if kind == "sportec_missing":
        lookup.side_effect = FileNotFoundError("archived")
    if kind == "hawkeye":
        lookup.side_effect = AssertionError("Native IDs must not load membership evidence")
    monkeypatch.setattr(utils, "get_model_record", lookup)
    variants = [Variant(name, name, {"metadata": {"models": {"pass_success": "pass_success/a"}}})
                for name in ("a", "b")]
    report = ComparisonExport(tmp_path, variants, SimpleNamespace(accept_comparison_warnings=True), "checkpoints")
    report.metadata["script"] = "run_and_visualize_hawkeye.py" if kind == "hawkeye" else "visualize_action_components.py"
    report.metadata["rendered_components"] = ["pass_success"]
    report.start()
    for number in range(3):
        report.scenes.extend({"scene_id": str(number), "match_id": "test", "variant_id": name} for name in ("a", "b"))
        report.checkpoint()
    report.finish()
    output = capsys.readouterr().out
    saved = json.loads((tmp_path / "metadata.json").read_text())
    result = saved["model_membership"]
    assert saved["status"] == "completed"
    assert result["version"] == 2
    assert {item["variant_id"] for item in result["models"]} == {"a", "b"}
    if kind == "hawkeye":
        lookup.assert_not_called()
        assert output.count(membership.SKIPPED_MESSAGE) == 1
        assert "Model membership |" not in output
        assert result["check_status"] == "not_checked" and result["entries"] == []
    else:
        lookup.assert_called_once_with("pass_success/a")
        expected = "held_out" if kind == "sportec_known" else "unknown"
        assert len(result["entries"]) == 6
        assert {item["status"] for item in result["entries"]} == {expected}
        assert {item["variant_id"] for item in result["entries"]} == {"a", "b"}
        assert output.count("Model membership |") == (2 if kind == "sportec_known" else 1)
