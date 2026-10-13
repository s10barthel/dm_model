import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch_geometric.data import Data

from datatools import config, sportec_seasons
from dataset import ActionDataset
from models.evaluation_context import binary_goal_return, check_heldout, evaluation_selection
from models.outcome_bootstrap import outcome_bootstrap
from scripts.comparison_warnings import ComparisonWarnings, ComparisonCancelled
from scripts.visualization_comparison import ComparisonExport, Variant, render_independent_groups
from scripts import evaluate_relevant_models as wrapper, model_comparison


def test_confirmation_requires_explicit_answer_and_only_new_warnings(monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(["", "perhaps", "YES", "no"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    policy = ComparisonWarnings()
    policy.add("receiver differs")
    policy.confirm()
    policy.add("receiver differs")
    policy.confirm()
    assert len(policy.acceptances) == 1
    policy.add("new limitation")
    with pytest.raises(ComparisonCancelled):
        policy.confirm()
    assert policy.cancelled


def test_confirmation_eof_noninteractive_and_explicit_flag(monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    policy = ComparisonWarnings()
    policy.add("different reference")
    with pytest.raises(ComparisonCancelled, match="accept-comparison-warnings"):
        policy.confirm()
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(ComparisonCancelled, match="no explicit answer"):
        policy.confirm()
    policy = ComparisonWarnings(True)
    policy.add("different reference")
    policy.confirm()
    assert policy.acceptances[0]["method"] == "command_line"


@pytest.mark.parametrize("value", ["next_3", "next_10_skip1", "disc_1", "disc_1.0_skip1"])
def test_binary_goal_definitions(value):
    assert binary_goal_return(value) == value


@pytest.mark.parametrize("value", ["disc_0.9", "in_3", "disc_max_1", "next_0"])
def test_continuous_or_invalid_diagnostic_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        binary_goal_return(value)


def test_cli_forwards_evaluation_overrides():
    args = wrapper.parse_args(["--intended-receiver-mode", "original", "--diagnostic-return-type", "next_3",
                               "--min-pass-duration", "0", "--season", "24_25", "--season-half", "second"])
    command = wrapper.add_task_evaluation_options([], args, "outcome_scoring")
    assert command[:8] == ["--intended-receiver-mode", "original", "--diagnostic-return-type", "next_3",
                           "--min-pass-duration", "0.0", "--season-half", "second"]
    assert command[command.index("--season")+1] == "24_25"
    assert wrapper.parse_args([]).min_pass_duration is None
    for value in ["-1", "nan", "inf"]:
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--min-pass-duration", value])
    with pytest.raises(ValueError, match="requires --season"):
        evaluation_selection(wrapper.parse_args(["--season-half", "first"]))


def test_season_half_uses_matchday_and_matchplan_fallback(tmp_path, monkeypatch):
    root = tmp_path / "season"
    directory = root / "match_information/starting_players"
    directory.mkdir(parents=True)
    (directory / "DFL-MAT-A.xml").write_text('<Root><General MatchId="DFL-MAT-A" MatchDay="17" KickoffTime="2025-03-01"/></Root>')
    (directory / "DFL-MAT-B.xml").write_text('<Root><General MatchId="DFL-MAT-B" MatchDay="18" KickoffTime="2024-08-01"/></Root>')
    (directory / "DFL-MAT-C.xml").write_text('<Root/>')
    matchplan = root / "match_information/master"
    matchplan.mkdir(parents=True)
    (matchplan / "matchplan").write_text('<Root><Fixture MatchId="DFL-MAT-C" MatchDay="34"/></Root>')
    monkeypatch.setitem(sportec_seasons.RAW_SEASON_ROOTS, "24_25", root)
    assert sportec_seasons.heldout_season_match_ids(["24_25"], "first") == ["DFL-MAT-A"]
    assert sportec_seasons.heldout_season_match_ids(["24_25"], "second") == ["DFL-MAT-B", "DFL-MAT-C"]
    assert len(sportec_seasons.heldout_season_match_ids(["24_25"])) == 3
    (directory / "DFL-MAT-D.xml").write_text('<Root/>')
    with pytest.raises(ValueError, match="MatchDay"):
        sportec_seasons.heldout_season_match_ids(["24_25"], "second")


def test_training_and_validation_are_hard_errors():
    record = {"model_id": "pass_success/test", "metadata": {
        "train_match_ids": ["train"], "validation_match_ids": ["validation"]}}
    assert check_heldout(record, ["test"])["overlap_count"] == 0
    for match, category in [("train", "training"), ("validation", "validation")]:
        with pytest.raises(ValueError, match=category):
            check_heldout(record, [match])
    with pytest.raises(ValueError, match="missing recorded membership"):
        check_heldout({"model_id": "unknown"}, ["test"])


def test_discussed_checkpoints_second_half_is_heldout():
    from models.utils import get_model_record
    for run in ("pass_success_20260928T150259_573568_9ea1b93e", "pass_success_20261001T200208_942925_343ea82e"):
        if not (Path("saved/pass_success") / run / "metadata.json").exists():
            pytest.skip("Local checkpoints are not installed")
        record = get_model_record("pass_success/" + run)
        assert check_heldout(record, sportec_seasons.heldout_season_match_ids(["24_25"], "second"))["requested_match_count"] == 153
        with pytest.raises(ValueError, match="validation"):
            check_heldout(record, sportec_seasons.heldout_season_match_ids(["24_25"], "first"))


def test_diagnostic_sidecar_overrides_embedded_next10_without_rewriting(tmp_path):
    features, labels, diagnostic = [tmp_path / name for name in ("features", "labels", "diagnostic")]
    for folder in (features, labels, diagnostic):
        folder.mkdir()
    x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM))
    x[:, 0] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones((2, 2)), node_ids=["home_1", "home_2"])
    rows = torch.zeros((2, len(config.LABEL_COLUMNS)))
    for name, value in [("is_pass", 1), ("intent_index", 1), ("scores_goal_next10", 1)]:
        rows[:, config.LABEL_INDEX[name]] = value
    rows[:, 7] = torch.tensor([.5, 1.5])
    rows[:, 0] = torch.arange(2)
    torch.save([graph, graph.clone()], features / "match.pt")
    torch.save(rows, labels / "match.pt")
    common = rows.clone()
    common[:, config.LABEL_INDEX["scores"]] = torch.tensor([0., 1.])
    torch.save(common, diagnostic / "match.pt")
    before = (labels / "match.pt").read_bytes()
    dataset = ActionDataset(["match"], feature_dir=features, label_dir=labels, task="outcome_scoring",
                            min_pass_dur=1., evaluation_diagnostic_label_dir=diagnostic,
                            require_goal_next10_diagnostics=False, train=False)
    assert len(dataset) == 1
    assert dataset.labels[0, config.LABEL_INDEX["scores_goal_next10"]] == 1
    assert dataset.features[0].evaluation_goal_diagnostic.tolist() == [1., 0.]
    assert (labels / "match.pt").read_bytes() == before
    common[0, config.LABEL_INDEX["scores"]] = .5
    torch.save(common, diagnostic / "match.pt")
    with pytest.raises(ValueError, match="binary"):
        ActionDataset(["match"], feature_dir=features, label_dir=labels, task="outcome_scoring",
                      evaluation_diagnostic_label_dir=diagnostic, require_goal_next10_diagnostics=False)


def test_nondefault_diagnostic_bootstrap_names():
    evaluation = {"prediction": np.array([.1, .9, .2, .8]), "target": np.array([.2, .7, .3, .6]),
                  "diagnostic": np.array([0, 1, 0, 1]), "execution_branch": np.ones(4),
                  "match_id": np.array(["a", "a", "b", "b"]), "diagnostic_return_type": "next_3"}
    table, metadata = outcome_bootstrap(evaluation, model_id="outcome_scoring/test", task="outcome_scoring", resamples=2)
    assert "goal_next_3_diagnostic" in set(table.evaluation_target)
    assert metadata["diagnostic_target"] == "goal_next_3_diagnostic"


def test_independent_images_union_geometry_and_duplicate_export(tmp_path):
    variants = [Variant("a", "a"), Variant("b", "b")]
    report = ComparisonExport(tmp_path, variants, SimpleNamespace(accept_comparison_warnings=True), "components")
    def scene(identifier, frame, size, duplicate=False):
        table = pd.DataFrame([[.2, .3]], index=[frame], columns=["player", "player" if duplicate else "other"])
        return {"scene": identifier, "frame": frame, "folder": identifier, "suffix": "", "title": identifier,
                "geometry": {"offset": size}, "tables": [{"pass_success": table}],
                "render": lambda component, row: Image.new("RGB", size, "red")}
    render_independent_groups(report, [[scene("same", 1, (100, 50), True), scene("a_only", 2, (100, 60))],
                                       [scene("same", 1, (80, 60)), scene("b_only", 3, (80, 60))]], ["pass_success"])
    report.finish()
    assert len(list(tmp_path.rglob("*.png"))) == 3
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "completed" and not metadata["aligned_export_available"]
    assert any("Scene/frame membership" in warning for warning in metadata["warnings"])
    assert all(Path(variant["predictions"]).exists() for variant in metadata["variants"])


def test_statistical_different_targets_warn_but_different_actions_fail(tmp_path):
    records = []
    for index in range(2):
        folder = tmp_path / str(index)
        folder.mkdir()
        pd.DataFrame({"match_id": ["test", "test"], "source_index": [0, 1],
                      "target": [index, 1], "prediction": [.2, .8]}).to_csv(folder / "learning_curve_predictions.csv", index=False)
        records.append({"model_id": f"pass_success/{index}", "task": "pass_success", "output_dir": str(folder)})
    frames, messages = model_comparison.load_aligned_predictions(records)
    assert len(frames) == 2 and messages[0]["column"] == "target"
    policy = ComparisonWarnings(True)
    report = model_comparison.write_report(records, tmp_path / "report", resamples=0, seed=1, warning_policy=policy)
    assert report["reference_differences"] and report["warning_acceptances"]
    frame = pd.read_csv(tmp_path / "1/learning_curve_predictions.csv").iloc[:1]
    frame.to_csv(tmp_path / "1/learning_curve_predictions.csv", index=False)
    with pytest.raises(ValueError, match="action populations"):
        model_comparison.load_aligned_predictions(records)


def test_known_limitation_does_not_prompt_again_with_row_counts(monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    calls = []
    monkeypatch.setattr("builtins.input", lambda prompt: calls.append(prompt) or "yes")
    policy = ComparisonWarnings()
    policy.add("Receiver modes differ", key="receiver")
    policy.confirm()
    policy.add("Receivers differ on 183 passes", key="receiver")
    policy.confirm()
    assert len(calls) == 1 and len(policy.messages) == 2


def test_native_target_qualification_leaves_common_goal_metrics_comparable():
    records = [{"task": "outcome_scoring", "model_id": f"outcome_scoring/{family}",
                "target_definitions": {"target_family": family, "return_type": horizon}}
               for family, horizon in [("xt", "next_3"), ("goal", "next_10")]]
    frames = [pd.DataFrame({"target": [0, 1], "soft_target": soft, "receiver_id": ["a", "b"],
                            "diagnostic_return_type": ["next_10", "next_10"]}) for soft in ([.2, .7], [0, 1])]
    assert model_comparison.reference_qualification("outcome_scoring", "roc_auc", records, frames) == "common_references"
    assert model_comparison.reference_qualification("outcome_scoring", "xt_rmse", records, frames) == "different_references"


def test_observed_receiver_exports_follow_common_reference():
    import argparse
    import copy
    from torch_geometric.loader import DataLoader
    from models.gnn import GNN
    from models.utils import run_epoch
    from test_masked_evaluation_smoke import model_args, samples
    torch.manual_seed(17)
    arguments = model_args()
    model = GNN(dict(arguments)).eval()
    native = samples()
    for graph, _, _ in native:
        graph.x[2, 0] = 1
        graph.node_ids[2] = "home_3"
    common = copy.deepcopy(native)
    for _, labels, _ in common:
        labels[config.LABEL_INDEX["intent_index"]] = 2
    rows = []
    for population in (native, common):
        _, predictions, _ = run_epoch(argparse.Namespace(**arguments), model, DataLoader(population, batch_size=2),
                                      device="cpu", return_learning_curve_rows=True, return_semantic_prediction_rows=True)
        rows.append(predictions)
    assert {row["receiver_id"] for row in rows[0]} == {"home_2"}
    assert {row["receiver_id"] for row in rows[1]} == {"home_3"}
    assert [(row["match_id"], row["source_index"]) for row in rows[0]] == [(row["match_id"], row["source_index"]) for row in rows[1]]


def test_cancelled_statistical_run_saves_acceptance_state(tmp_path, monkeypatch):
    entries = [{"task": "outcome_scoring", "model_id": f"outcome_scoring/{family}", "test_match_ids": ["test"],
                "target_definitions": {"target_family": family},
                "checkpoint_record": {"args": {}, "metadata": {"train_match_ids": [], "validation_match_ids": []}}}
               for family in ("xt", "goal")]
    args = wrapper.parse_args(["--compare", "--outcome-scoring-model-id", "outcome_scoring/xt",
                               "--outcome-scoring-model-id", "outcome_scoring/goal", "--evaluation-output-dir", str(tmp_path)])
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "resolve_evaluation_entries", lambda arguments: (entries, {}, None))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **kw: pytest.fail("Inference started before acceptance"))
    with pytest.raises(ComparisonCancelled):
        wrapper.main()
    metadata = json.loads(next(tmp_path.rglob("experiment_manifest.json")).read_text())
    assert metadata["status"] == "cancelled" and metadata["comparison_cancelled"]
    assert metadata["warnings"] and not metadata["warning_acceptances"]


def test_feature_manifest_provenance_can_supply_exclusion_boundary(monkeypatch):
    import models.evaluation_context as context
    monkeypatch.setattr(context, "load_feature_run_metadata", lambda *a, **kw: {"split_manifest_id": "feature_manifest"})
    monkeypatch.setattr(context, "load_recorded_split_manifest", lambda identifier: {"train": ["development"], "test": ["test"]})
    record = {"model_id": "legacy/model", "feature_run_id": "feature", "metadata": {}}
    assert context.recorded_manifest_id(record) == "feature_manifest"
    assert check_heldout(record, ["test"])["overlap_count"] == 0
    with pytest.raises(ValueError, match="development"):
        check_heldout(record, ["development"])
