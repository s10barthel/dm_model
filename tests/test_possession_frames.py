from argparse import ArgumentParser
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from datatools.possession_frames import (
    add_frame_selection_arguments, resolve_frame_selection_arguments,
    select_possession_frames, possession_cache_identity,
)


def sample(start=102, end=119, stride=5, missing=(), scope="frames"):
    return select_possession_frames(range(start, end + 1), start, end,
                                    lambda f: None if f in missing else f,
                                    scope=scope, frames=stride)


@pytest.mark.parametrize("start,end,stride,expected", [
    (102, 119, 5, [102, 107, 112, 117, 119]),
    (102, 104, 5, [102, 104]), (102, 102, 5, [102]),
    (102, 104, 1, [102, 103, 104]), (102, 119, 99, [102, 119]),
])
def test_sampling(start, end, stride, expected):
    states, _ = sample(start, end, stride)
    assert [frame for frame, _, _ in states] == expected


def test_invalid_endpoints_do_not_shift_grid():
    states, report = sample(missing={102, 103, 107, 119})
    assert [f for f, _, _ in states] == [104, 112, 117, 118]
    assert report["endpoint_substitutions"] == 2


def test_missing_tracking_rows_do_not_shift_grid():
    states, _ = select_possession_frames([102, 103, 108, 112, 119], 102, 119, lambda f: f,
                                        scope="frames", frames=5)
    assert [f for f, _, _ in states] == [102, 112, 119]


def test_all_invalid_and_single_valid():
    assert sample(missing=range(102, 120))[0] == []
    states, _ = sample(missing=set(range(102, 120)) - {110})
    assert states == [(110, 110, "start_end")]


@pytest.mark.parametrize("argv", [["--frames", "5"], ["--scope", "frames", "--frames", "0"],
                                  ["--scope", "frames", "--frames", "-1"],
                                  ["--scope", "frames", "--frames", "1.5"]])
def test_invalid_cli(argv):
    parser = ArgumentParser()
    add_frame_selection_arguments(parser)
    with pytest.raises(SystemExit):
        resolve_frame_selection_arguments(parser, parser.parse_args(argv))


def test_cli_defaults():
    parser = ArgumentParser()
    add_frame_selection_arguments(parser)
    args = parser.parse_args([])
    resolve_frame_selection_arguments(parser, args)
    assert (args.scope, args.frames) == ("actions", 1)


def spell_fixture(tmp_path, monkeypatch):
    from datatools import sportec_possessions as sp
    root = tmp_path / "carry_segments"
    (root / "audits").mkdir(parents=True)
    rows = [dict(carry_id=1, period_id=1, carrier_id="home_1", start_frame=102, terminal_frame=119,
                 success=True, reason="short_or_missing_return", retained=False, return_event_index=None,
                 start_event_id="receipt:a", terminal_event_id="b"),
            dict(carry_id=2, period_id=1, carrier_id="home_1", start_frame=120, terminal_frame=125,
                 success=None, reason="ambiguous_tackle", retained=False),
            dict(carry_id=3, period_id=1, carrier_id="away_1", start_frame=119, terminal_frame=122,
                 success=False, reason="short_or_missing_return", retained=False)]
    pd.DataFrame(rows).to_parquet(root / "audits" / "match.parquet", index=False)
    monkeypatch.setattr(sp, "CARRY_SEGMENTS_DIR", root)
    monkeypatch.setattr(sp, "load_run_metadata", lambda *a, **k: {})
    tracking = pd.DataFrame({"period_id": 1, "ball_x": 2., "ball_y": 3., "ball_accel": 0.,
                             "home_1_x": 2., "home_1_y": 3., "away_1_x": 5., "away_1_y": 6.},
                            index=range(102, 126))
    monkeypatch.setattr(sp, "construct_graph_for_frame", lambda *a, **k: SimpleNamespace(frame=a[1], carrier=a[2]))
    monkeypatch.setattr(sp.utils, "find_active_players", lambda tracking, frame, team, **k:
                        ([f"{team}_1"], ["away_1" if team == "home" else "home_1"]))
    return sp, SimpleNamespace(tracking=tracking, fps=25), tmp_path / "feature_run"


def test_short_spells_without_training_labels_and_loss_possessor(tmp_path, monkeypatch):
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    states = list(sp.build_sportec_possessions(match, "match", feature_root, scope="frames", frames=5))
    assert len(states) == 2
    first, report = states[0]
    assert first.actions.index.tolist() == [102, 107, 112, 117, 119]
    assert first.actions.object_id.unique().tolist() == ["home_1"]
    assert report["excluded_spells"] == 1
    second = states[1][0]
    assert second.actions.object_id.unique().tolist() == ["away_1"]
    assert first.pc_cache_match_id != second.pc_cache_match_id
    assert first.actions.loc[119, "object_id"] == "home_1"
    assert second.actions.loc[119, "object_id"] == "away_1"


def test_generation_and_inference_state_identity_and_components(tmp_path, monkeypatch):
    from scripts.generate_physical_xpass import runtime_cache_items_from_graphs
    from datatools import skillcorner
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    sparse = list(sp.build_sportec_possessions(match, "match", feature_root, scope="frames", frames=5))[0][0]
    dense = list(sp.build_sportec_possessions(match, "match", feature_root, scope="frames", frames=1))[0][0]
    assert sparse.pc_cache_match_id == dense.pc_cache_match_id
    items = runtime_cache_items_from_graphs(sparse.pc_cache_match_id, sparse.graph_features_0, sparse.labels,
                                           frame_scope="frame_id", state_frame_ids=sparse.actions.index.tolist())
    assert items[0]["state_frame_id"] == sparse.actions.index.tolist()
    assert items[0]["labels"][:, 0].tolist() == sparse.actions.index.tolist()
    calls = []

    def predict(state, model, **kwargs):
        assert kwargs["graph_override"] is state.graph_features_0
        assert kwargs["label_override"] is state.labels
        calls.append(model)
        result = pd.DataFrame({"home_1": .5}, index=state.actions.index)
        return result, result

    monkeypatch.setattr(skillcorner, "inference_gnn", predict)
    models = {name: name for name in ["action_intent", "pass_intent", "pass_success", "pass_height",
                                    "outcome_scoring", "outcome_conceding"]}
    components = skillcorner.infer_skillcorner_components(sparse, models)
    assert set(calls) == set(models)
    assert components["action_intent"].index.tolist() == [102, 107, 112, 117, 119]
    exported = sp.sportec_component_table(components["action_intent"], sparse)
    assert exported.state_frame_id.tolist() == [102, 107, 112, 117, 119]
    assert exported.frame_role.tolist() == ["start", "interior", "interior", "interior", "end"]
    endpoint_state = list(sp.build_sportec_possessions(match, "match", feature_root))[0][0]
    assert skillcorner.infer_skillcorner_components(endpoint_state, models)["action_intent"].index.tolist() == [102, 119]


def test_artifact_provenance_changes_cache_partition(tmp_path, monkeypatch):
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    first = list(sp.build_sportec_possessions(match, "match", feature_root))[0][0]
    changed_run = list(sp.build_sportec_possessions(match, "match", tmp_path / "different_run"))[0][0]
    assert first.pc_cache_match_id != changed_run.pc_cache_match_id
    path = sp.CARRY_SEGMENTS_DIR / "audits" / "match.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "terminal_frame"] = 118
    table.to_parquet(path, index=False)
    changed_spells = list(sp.build_sportec_possessions(match, "match", feature_root))[0][0]
    assert first.pc_cache_match_id != changed_spells.pc_cache_match_id


def test_missing_artifacts_are_not_reconstructed(tmp_path, monkeypatch):
    sp, _, feature_root = spell_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="whole-spell artifact"):
        sp.load_control_spells("missing", feature_root)


def test_incompatible_artifact_definition_and_duplicate_spells(tmp_path, monkeypatch):
    sp, _, feature_root = spell_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sp, "load_run_metadata", lambda *a, **k: {"carry_variant": {"definition": "obsolete"}})
    with pytest.raises(ValueError, match="carry definition"):
        sp.load_control_spells("match", feature_root)
    monkeypatch.setattr(sp, "load_run_metadata", lambda *a, **k: {})
    path = sp.CARRY_SEGMENTS_DIR / "audits" / "match.parquet"
    table = pd.read_parquet(path)
    table.loc[1, "carry_id"] = table.loc[0, "carry_id"]
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate control-spell"):
        sp.load_control_spells("match", feature_root)


def test_inference_uses_possession_partition_not_legacy_match_cache(tmp_path, monkeypatch):
    from inference import resolve_physical_state_match_id, resolve_pc_state_match_id
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    state = list(sp.build_sportec_possessions(match, "match", feature_root))[0][0]
    assert resolve_pc_state_match_id(state) == state.pc_cache_match_id
    assert resolve_physical_state_match_id(state, SimpleNamespace(args={"pc_xpass": True})) == state.pc_cache_match_id
    with pytest.raises(ValueError, match="event indexes"):
        resolve_physical_state_match_id(state, SimpleNamespace(args={"pc_xpass": False}))


def test_pc_cache_multiple_frames_round_trip(tmp_path, monkeypatch):
    import physical_pass_model as physical
    monkeypatch.setattr(physical, "validate_runtime_physical_xpass_visualization_cache", lambda *a: {})
    partition = possession_cache_identity("sportec", "match", 1, "home_1", {"feature_run_id": "run"})
    def rows(frames):
        return pd.DataFrame({"action_index": frames, "state_frame_id": frames, "frame_scope": "frame_id"})
    physical._write_runtime_physical_xpass_rows(tmp_path, partition, rows([102, 119]))
    physical._write_runtime_physical_xpass_rows(tmp_path, partition, rows([102, 107, 112, 117, 119]))
    loaded = physical.load_physical_xpass_match(tmp_path, partition, frame_scope="frame_id")
    assert loaded.index.tolist() == [102, 107, 112, 117, 119]
    other = possession_cache_identity("sportec", "match", 2, "away_1", {"feature_run_id": "run"})
    physical._write_runtime_physical_xpass_rows(tmp_path, other, rows([119]))
    assert len(physical.load_physical_xpass_match(tmp_path, partition, frame_scope="frame_id")) == 5


@pytest.mark.parametrize("scope,stride,expected", [("actions", 1, [102, 119]), ("frames", 5, [102, 107, 112, 117, 119])])
def test_sportec_generator_uses_shared_artifacts_and_sampling(tmp_path, monkeypatch, scope, stride, expected):
    from scripts import generate_physical_xpass as gen, run_relevant_models
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    argv = ["--pc-xpass", "--sportec-feature-run-id", "feature_run", "--scope", scope]
    if scope == "frames":
        argv += ["--frames", str(stride)]
    args = gen.parse_args(argv)
    monkeypatch.setattr(gen, "resolve_feature_run_id", lambda value, **k: value)
    monkeypatch.setattr(gen, "resolve_feature_root", lambda value: feature_root)
    monkeypatch.setattr(gen, "resolve_reference_label_context", lambda *a: (None, "disc_0.7", "model"))
    monkeypatch.setattr(gen, "resolve_match_ids", lambda *a: ["match"])
    monkeypatch.setattr(gen.pc_versions, "cache_dir", lambda *a: tmp_path / "cache")
    monkeypatch.setattr(run_relevant_models, "load_match", lambda *a, **k: match)
    monkeypatch.setattr(gen, "resolve_runtime_row_window", lambda *a: 100)
    captured, metadata = [], {}

    def prewarm(items, **kwargs):
        captured.extend(items)
        return {"cache_written": sum(len(item["graphs"]) for item in items)}

    monkeypatch.setattr(gen, "prewarm_runtime_items", prewarm)
    monkeypatch.setattr(gen, "write_runtime_dataset_metadata", lambda *a, **k: metadata.update(k))
    result = gen.run_runtime_sportec(args)
    assert not result["skipped"]
    assert captured[0]["state_frame_id"] == expected
    inference_state = list(sp.build_sportec_possessions(match, "match", feature_root, scope=scope, frames=stride))[0][0]
    assert captured[0]["match_id"] == inference_state.pc_cache_match_id
    assert captured[0]["labels"].equal(inference_state.labels)
    assert metadata["source_inputs"]["feature_run_id"] == "feature_run"
    assert len(metadata["source_inputs"]["state_partitions"]) == 2


def test_pc_only_cli_and_explicit_runtime_feature_run():
    from scripts import generate_physical_xpass as gen
    for argv in (["--scope", "frames"], ["--frames", "5"],
                 ["--sportec-feature-run-id", "run"],
                 ["--pc-xpass", "--feature-run-id", "legacy", "--scope", "frames"]):
        with pytest.raises(SystemExit):
            gen.parse_args(argv)
    args = gen.parse_args(["--pc-xpass", "--scope", "frames", "--frames", "5", "--sportec-feature-run-id", "run"])
    assert (args.scope, args.frames, args.sportec_feature_run_id) == ("frames", 5, "run")


def test_event_summary_uses_only_linked_endpoints_not_interiors(tmp_path, monkeypatch):
    sp, match, feature_root = spell_fixture(tmp_path, monkeypatch)
    match.events = pd.DataFrame([
        dict(action_id=4, original_event_id="incoming", period_id=1, frame_id=90, receive_frame_id=102,
             object_id="home_2", receiver_id="home_1", spadl_type="pass"),
        dict(action_id=5, original_event_id="release", period_id=1, frame_id=119, receive_frame_id=130,
             object_id="home_1", receiver_id="home_2", spadl_type="pass"),
    ])
    state = list(sp.build_sportec_possessions(match, "match", feature_root, scope="frames", frames=5))[0][0]
    predictions = pd.DataFrame({"home_2": [.1, .2, .3, .4, .5]}, index=state.actions.index)
    table = sp.sportec_component_table(predictions, state)
    view = sp.sportec_event_endpoint_view(table)
    assert view.action_id.tolist() == [4, 5]
    assert view.original_event_id.tolist() == ["incoming", "release"]
    assert view.frame_scope.tolist() == ["receive_frame_id", "frame_id"]
    assert view.home_2.tolist() == [.1, .5]


def test_skillcorner_missing_receipt_does_not_reanchor_stride(monkeypatch):
    from test_skillcorner_postprocessing import make_skillcorner_possession, patch_skillcorner_graph_builder
    from datatools import skillcorner
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession(list(range(103, 120)))
    possession.original_start_frame = 102
    possession.original_end_frame = 119
    actions, _, _, _ = skillcorner._build_actions_and_labels(possession, scope="frames", frames=5)
    assert actions.index.tolist() == [103, 107, 112, 117, 119]
    assert possession.frame_selection["endpoint_substitutions"] == 1


def test_real_graph_construction_and_action_intent_on_sampled_states(tmp_path, monkeypatch):
    from datatools import graph_feature, utils
    from inference import inference_gnn
    from test_graph_feature_regressions import make_offside_match_and_snapshot, DummyNodeModel

    find_active_players = utils.find_active_players
    sp, _, feature_root = spell_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sp, "construct_graph_for_frame", graph_feature.construct_graph_for_frame)
    monkeypatch.setattr(utils, "find_active_players", find_active_players)
    match, snapshot = make_offside_match_and_snapshot()
    match.max_players, match.fps = 24, 25
    match.tracking = pd.concat([snapshot] * 24, ignore_index=True)
    match.tracking.index = range(102, 126)
    match.tracking["period_id"] = 1
    match.tracking["ball_accel"] = 0.
    match.tracking["home_2_x"] = [65. + i / 10 for i in range(24)]
    state = list(sp.build_sportec_possessions(match, "match", feature_root, scope="frames", frames=5))[0][0]
    assert state.actions.index.tolist() == [102, 107, 112, 117, 119]
    assert not torch.equal(state.graph_features_0[0].x, state.graph_features_0[1].x)
    assert state.graph_features_0[0].node_ids == state.graph_features_0[1].node_ids
    model = DummyNodeModel("action_intent", [0.] * 1000, node_in_dim=graph_feature.infer_node_feature_dim(extend=True))
    model.args["extend_features"] = True
    predictions, _ = inference_gnn(state, model, device="cpu", graph_override=state.graph_features_0,
                                   label_override=state.labels)
    assert predictions.index.tolist() == [102, 107, 112, 117, 119]
    assert predictions.notna().any(axis=1).all()
