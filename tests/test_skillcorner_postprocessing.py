from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pandas as pd
import torch
from torch_geometric.data import Data

import scripts.run_skillcorner as run_skillcorner
from datatools import config
from datatools.graph_feature import load_frame_snapshot
from datatools import skillcorner
from datatools.skillcorner import SkillcornerPossession, load_skillcorner_events
from validation.skillcorner.code import skillcorner_postprocessing as post


class RecordingTqdm:
    calls: list["RecordingTqdm"] = []

    def __init__(self, iterable, *args, **kwargs) -> None:
        self.iterable = list(iterable)
        self.args = args
        self.kwargs = kwargs
        self.postfixes: list[dict[str, object]] = []
        self.writes: list[str] = []
        RecordingTqdm.calls.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs) -> None:
        self.postfixes.append(dict(kwargs))

    def write(self, message: str) -> None:
        self.writes.append(message)


def test_load_frame_snapshot_keeps_current_phase_only() -> None:
    tracking = pd.DataFrame(
        [
            {"frame_id": 9, "phase_id": 1, "home_1_x": 1.0, "home_1_y": 2.0},
            {"frame_id": 10, "phase_id": 2, "home_1_x": 1.5, "home_1_y": 2.5},
        ]
    ).set_index("frame_id", drop=False)

    snapshot = load_frame_snapshot(tracking, tracking, 10)

    assert snapshot.index.tolist() == [10]
    assert snapshot["phase_id"].tolist() == [2]


def make_skillcorner_possession(
    frame_ids: list[int],
    *,
    has_ball: dict[int, bool] | None = None,
    missing_possessor_frames: set[int] | None = None,
) -> SkillcornerPossession:
    has_ball = has_ball or {}
    missing_possessor_frames = missing_possessor_frames or set()
    tracking_rows = []
    frame_meta_rows = []
    for frame_id in frame_ids:
        possessor_x = pd.NA if frame_id in missing_possessor_frames else 30.0 + frame_id
        possessor_y = pd.NA if frame_id in missing_possessor_frames else 20.0
        ball_present = has_ball.get(frame_id, True)
        tracking_rows.append(
            {
                "frame_id": frame_id,
                "period_id": 1,
                "ball_x": 40.0 if ball_present else pd.NA,
                "ball_y": 30.0 if ball_present else pd.NA,
                "ball_owning_home_away": "home",
                "home_1_x": possessor_x,
                "home_1_y": possessor_y,
                "away_2_x": 60.0,
                "away_2_y": 30.0,
                "home_goal_x": 105.0,
                "home_goal_y": 34.0,
                "away_goal_x": 0.0,
                "away_goal_y": 34.0,
            }
        )
        frame_meta_rows.append(
            {
                "frame_id": frame_id,
                "match_id": "match-1",
                "index": 7,
                "period": 1,
                "player_id": 1,
                "attacking_side": "left_to_right",
                "possessor_object_id": "home_1",
                "has_ball": ball_present,
            }
        )

    tracking = pd.DataFrame(tracking_rows).set_index("frame_id", drop=False)
    frame_meta = pd.DataFrame(frame_meta_rows).set_index("frame_id", drop=True)
    return SkillcornerPossession(
        match_id="match-1",
        event_index=7,
        fps=25.0,
        tracking=tracking,
        phases=pd.DataFrame(),
        frame_meta=frame_meta,
        actions=pd.DataFrame(),
        labels=torch.empty((0, len(config.LABEL_COLUMNS))),
        graph_features_0=[],
    )


def patch_skillcorner_graph_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_graph_frames: set[int] | None = None,
) -> list[int]:
    missing_graph_frames = missing_graph_frames or set()
    calls: list[int] = []

    def fake_construct_graph_for_frame(possession, frame_id, *args, **kwargs):
        calls.append(int(frame_id))
        if int(frame_id) in missing_graph_frames:
            return None
        return Data()

    monkeypatch.setattr(skillcorner, "construct_graph_for_frame", fake_construct_graph_for_frame)
    return calls


def test_skillcorner_default_frame_mode_selects_only_boundary_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12, 13, 14])

    actions, labels, graphs, stats = skillcorner._build_actions_and_labels(possession)

    assert actions.index.tolist() == [10, 14]
    assert labels[:, 0].tolist() == [10, 14]
    assert len(graphs) == 2
    assert calls == [10, 14]
    assert stats["total_frames"] == 5
    assert stats["evaluated_frames"] == 2
    assert stats["selected_frames"] == 2
    assert stats["valid_frames"] == 2


def test_skillcorner_all_frame_mode_preserves_all_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12, 13, 14])

    actions, labels, graphs, stats = skillcorner._build_actions_and_labels(possession, frames_mode="all")

    assert actions.index.tolist() == [10, 11, 12, 13, 14]
    assert labels[:, 0].tolist() == [10, 11, 12, 13, 14]
    assert len(graphs) == 5
    assert stats["evaluated_frames"] == 5
    assert stats["selected_frames"] == 5
    assert stats["valid_frames"] == 5


def test_skillcorner_first_last_frame_mode_dedupes_single_valid_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12], has_ball={10: False, 12: False})

    actions, _labels, graphs, stats = skillcorner._build_actions_and_labels(possession, frames_mode="first_and_last")

    assert actions.index.tolist() == [11]
    assert len(graphs) == 1
    assert stats["evaluated_frames"] == 3
    assert stats["selected_frames"] == 1
    assert stats["valid_frames"] == 1
    assert stats["skipped_missing_ball"] == 2


def test_skillcorner_first_last_frame_mode_falls_back_from_invalid_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch, missing_graph_frames={11})
    possession = make_skillcorner_possession(
        [10, 11, 12, 13, 14, 15],
        has_ball={10: False},
        missing_possessor_frames={15},
    )

    actions, _labels, _graphs, stats = skillcorner._build_actions_and_labels(
        possession,
        frames_mode="first_and_last",
    )

    assert actions.index.tolist() == [12, 14]
    assert stats["evaluated_frames"] == 5
    assert stats["selected_frames"] == 2
    assert stats["valid_frames"] == 2
    assert stats["skipped_missing_ball"] == 1
    assert stats["skipped_missing_graph"] == 1
    assert stats["skipped_missing_possessor"] == 1


def test_run_skillcorner_frame_mode_cli_defaults_and_flags() -> None:
    assert run_skillcorner.parse_args([]).frames_mode == "first_and_last"
    assert run_skillcorner.parse_args(["--frames-first-and-last"]).frames_mode == "first_and_last"
    assert run_skillcorner.parse_args(["--frames-all"]).frames_mode == "all"

    with pytest.raises(SystemExit):
        run_skillcorner.parse_args(["--frames-first-and-last", "--frames-all"])


def test_run_skillcorner_progress_formatting_helpers() -> None:
    assert run_skillcorner.format_match_progress(3, 20, "117670") == "[3/20] match_id=117670 | 17 games left"
    assert run_skillcorner.format_match_progress(19, 20, "117670") == "[19/20] match_id=117670 | 1 game left"
    assert run_skillcorner.format_possession_progress("117670", 12, 48) == "  match 117670 possession 12/48"
    assert (
        run_skillcorner.format_skillcorner_inference_progress("117670", 12, 48, 345)
        == "match 117670 inference 12/48 | event_index=345 | 36 possessions left"
    )
    assert (
        run_skillcorner.format_skillcorner_inference_progress("117670", 47, 48, 987)
        == "match 117670 inference 47/48 | event_index=987 | 1 possession left"
    )
    assert (
        run_skillcorner.format_possession_skip("117670", 12, 48, 345, "ValueError: example")
        == "  SKIP match 117670 possession 12/48 event_index=345: ValueError: example"
    )
    assert (
        run_skillcorner.format_match_completion(
            "117670",
            {
                "processed_possessions": 45,
                "possessions": 48,
                "skipped_possessions": 3,
                "selected_frames": 90,
                "evaluated_frames": 94,
                "valid_frames": 90,
                "total_frames": 250,
            },
        )
        == "  DONE match 117670: 45/48 possessions, 3 skipped, 90 selected frames, "
        "94 evaluated frames, 90/250 valid frames"
    )


def test_run_skillcorner_main_prints_match_centered_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    args = argparse.Namespace(
        input_dir="skillcorner_data",
        match_id=None,
        limit=None,
        device="cpu",
        bundle_id=None,
        action_intent_model_id=None,
        pass_intent_model_id=None,
        pass_success_model_id=None,
        outcome_scoring_model_id=None,
        outcome_conceding_model_id=None,
        run_id="test_run",
        output_dir=str(tmp_path),
        frames_mode="first_and_last",
        physical_cache_dir=None,
        no_physical_cache=False,
        refresh_physical_cache=False,
        physical_num_workers="auto",
        physical_worker_thread_limit=1,
        physical_batch_size=16,
    )
    shared_context = {
        "intended_receiver_mode": "mode",
        "return_type": "return",
        "target_family": "target",
    }
    model_ids = {
        "action_intent": "action_intent/model",
        "pass_intent": "pass_intent/model",
        "pass_success": "pass_success/model",
        "outcome_scoring": "outcome_scoring/model",
        "outcome_conceding": "outcome_conceding/model",
    }
    contexts = {
        "m1": {
            "events": pd.DataFrame({"index": [10, 20]}),
            "player_meta": pd.DataFrame({"player_id": [1]}),
        },
        "m2": {
            "events": pd.DataFrame({"index": [30]}),
            "player_meta": pd.DataFrame({"player_id": [1]}),
        },
    }
    possession_stats = {
        ("m1", 10): {
            "total_frames": 2,
            "valid_frames": 0,
            "evaluated_frames": 2,
            "selected_frames": 0,
            "skipped_missing_ball": 2,
            "skipped_missing_possessor": 0,
            "skipped_missing_graph": 0,
        },
        ("m1", 20): {
            "total_frames": 4,
            "valid_frames": 2,
            "evaluated_frames": 2,
            "selected_frames": 2,
            "skipped_missing_ball": 0,
            "skipped_missing_possessor": 0,
            "skipped_missing_graph": 0,
        },
        ("m2", 30): {
            "total_frames": 4,
            "valid_frames": 2,
            "evaluated_frames": 2,
            "selected_frames": 2,
            "skipped_missing_ball": 0,
            "skipped_missing_possessor": 0,
            "skipped_missing_graph": 0,
        },
    }

    monkeypatch.setattr(run_skillcorner, "parse_args", lambda: args)
    monkeypatch.setattr(run_skillcorner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run_skillcorner, "resolve_model_selection", lambda **kwargs: (model_ids, shared_context, None))
    monkeypatch.setattr(run_skillcorner, "discover_skillcorner_matches", lambda *args, **kwargs: (["m1", "m2"], {}))
    monkeypatch.setattr(run_skillcorner, "load_skillcorner_models", lambda **kwargs: {"models": object()})
    monkeypatch.setattr(run_skillcorner, "validate_model_graph_schemas", lambda model_specs: {"add_v_edge_features": False})
    monkeypatch.setattr(
        run_skillcorner,
        "get_model_provenance",
        lambda model_id: {"feature_signature": {"model_id": model_id}},
    )
    monkeypatch.setattr(run_skillcorner, "build_skillcorner_match_context", lambda match_id, input_dir: contexts[match_id])

    def fake_build_skillcorner_possession(context, event_index, **kwargs):
        match_id = "m1" if event_index in {10, 20} else "m2"
        possession = SimpleNamespace(physical_xpass_runtime_stats=None)
        return possession, possession_stats[(match_id, int(event_index))]

    monkeypatch.setattr(run_skillcorner, "build_skillcorner_possession", fake_build_skillcorner_possession)
    monkeypatch.setattr(run_skillcorner, "infer_skillcorner_components", lambda possession, model_specs, device: {})
    monkeypatch.setattr(run_skillcorner, "build_skillcorner_component_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_skillcorner, "_save_component_table", lambda *args, **kwargs: None)
    written_metadata = {}
    monkeypatch.setattr(run_skillcorner, "write_run_metadata", lambda _path, metadata: written_metadata.update(metadata))
    monkeypatch.setattr(run_skillcorner, "write_latest_run", lambda *args, **kwargs: None)
    postprocessing_calls = []
    filter_calls = []

    def fake_run_skillcorner_postprocessing(**kwargs):
        postprocessing_calls.append(kwargs)
        summary_path = Path(kwargs["output_file"])
        return pd.DataFrame(), {"events_with_dm_score": 2, "event_rows": 3}, summary_path

    def fake_run_skillcorner_filter(**kwargs):
        filter_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        paths = {
            "actions_raw_path": output_dir / "skillcorner_actions_raw.csv",
            "actions_path": output_dir / "skillcorner_actions.csv",
            "players_path": output_dir / "skillcorner_players.csv",
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"skillcorner_actions_rows": 1}, paths

    monkeypatch.setattr(run_skillcorner, "run_skillcorner_postprocessing", fake_run_skillcorner_postprocessing)
    monkeypatch.setattr(run_skillcorner, "run_skillcorner_filter", fake_run_skillcorner_filter)
    RecordingTqdm.calls = []
    monkeypatch.setattr(run_skillcorner, "tqdm", RecordingTqdm)

    run_skillcorner.main()

    assert len(RecordingTqdm.calls) == 2
    assert RecordingTqdm.calls[0].kwargs["desc"] == "match m1 possessions"
    assert RecordingTqdm.calls[0].kwargs["total"] == 1
    assert RecordingTqdm.calls[0].postfixes == [{"event_index": 20}]
    assert RecordingTqdm.calls[0].writes == ["match m1 inference 1/1 | event_index=20 | 0 possessions left"]
    assert RecordingTqdm.calls[1].kwargs["desc"] == "match m2 possessions"
    assert RecordingTqdm.calls[1].kwargs["total"] == 1
    assert RecordingTqdm.calls[1].postfixes == [{"event_index": 30}]
    assert RecordingTqdm.calls[1].writes == ["match m2 inference 1/1 | event_index=30 | 0 possessions left"]
    output = capsys.readouterr().out
    assert "[1/2] match_id=m1 | 1 game left" in output
    assert "[2/2] match_id=m2 | 0 games left" in output
    assert "  match m1: 2 eligible possessions" in output
    assert "  match m1 possession 1/2" in output
    assert (
        "  SKIP match m1 possession 1/2 event_index=10: "
        "ValueError: no valid frames were available after SkillCorner graph construction."
    ) in output
    assert "  DONE match m1: 1/2 possessions, 1 skipped, 2 selected frames, 4 evaluated frames, 2/6 valid frames" in output
    assert "  DONE match m2: 1/1 possessions, 0 skipped, 2 selected frames, 2 evaluated frames, 2/4 valid frames" in output
    assert postprocessing_calls == [
        {
            "component_run_root": tmp_path / "test_run",
            "event_data_dir": Path("skillcorner_data"),
            "output_file": tmp_path / "test_run" / "skillcorner_summary.csv",
        }
    ]
    assert filter_calls == [
        {
            "skillcorner_data_path": tmp_path / "test_run" / "skillcorner_summary.csv",
            "output_dir": tmp_path / "test_run",
        }
    ]
    assert "Saved SkillCorner summary to" in output
    assert "filtered actions rows: 1" in output
    assert "Physical xPass cache: not used; pass-success model does not require physical xPass." in output
    assert written_metadata["physical_xpass_cache_summary"]["reason"] == "physical_xpass_not_required"


def test_filter_event_rows_and_drop_empty_columns() -> None:
    event_data = pd.DataFrame(
        [
            {"match_id": 117670, "index": 1, "event_type_id": 8, "start_type_id": 1, "frame_start": 100, "frame_end": 105, "player_targeted_id": 10, "always_empty": pd.NA},
            {"match_id": 117670, "index": 2, "event_type_id": 8, "start_type_id": 2, "frame_start": 110, "frame_end": 115, "player_targeted_id": 11, "always_empty": pd.NA},
            {"match_id": 117670, "index": 3, "event_type_id": 7, "start_type_id": 1, "frame_start": 120, "frame_end": 125, "player_targeted_id": 12, "always_empty": pd.NA},
        ]
    )
    event_data = post.normalize_event_identifiers(event_data)

    filtered = post.filter_event_rows(event_data)
    filtered = post.drop_all_empty_columns(filtered)

    assert filtered["index"].tolist() == [1]
    assert "always_empty" not in filtered.columns


def test_load_skillcorner_events_filters_start_type_id() -> None:
    raw_events = pd.DataFrame(
        [
            {
                "match_id": 117670,
                "index": 1,
                "event_type": "player_possession",
                "frame_start": 100,
                "frame_end": 105,
                "period": 1,
                "start_type_id": 1,
                "attacking_side": "left_to_right",
                "player_id": 10,
            },
            {
                "match_id": 117670,
                "index": 2,
                "event_type": "player_possession",
                "frame_start": 110,
                "frame_end": 115,
                "period": 1,
                "start_type_id": 2,
                "attacking_side": "left_to_right",
                "player_id": 11,
            },
        ]
    )
    with patch("datatools.skillcorner.pd.read_csv", return_value=raw_events):
        filtered = load_skillcorner_events("117670", ".")

    assert filtered["index"].tolist() == [1]


def test_select_nearest_frame_match_prefers_earlier_or_later_on_ties() -> None:
    candidate_rows = pd.DataFrame(
        [
            {"event_row_id": 0, "frame_start": 12, "frame": 10, "game_state_value": 1.0},
            {"event_row_id": 0, "frame_start": 12, "frame": 14, "game_state_value": 2.0},
            {"event_row_id": 1, "frame_end": 12, "frame": 10, "pass_score": 3.0},
            {"event_row_id": 1, "frame_end": 12, "frame": 14, "pass_score": 4.0},
        ]
    )

    earlier = post.select_nearest_frame_match(
        candidate_rows.loc[candidate_rows["event_row_id"].eq(0)].copy(),
        event_id_column="event_row_id",
        event_frame_column="frame_start",
        candidate_frame_column="frame",
        tie_break="earlier",
        value_columns=["game_state_value"],
    )
    later = post.select_nearest_frame_match(
        candidate_rows.loc[candidate_rows["event_row_id"].eq(1)].copy(),
        event_id_column="event_row_id",
        event_frame_column="frame_end",
        candidate_frame_column="frame",
        tie_break="later",
        value_columns=["pass_score"],
    )

    assert earlier["game_state_value"].tolist() == [1.0]
    assert later["pass_score"].tolist() == [4.0]


def test_prune_component_frames_keeps_first_and_last_only() -> None:
    component = pd.DataFrame(
        [
            {"match_id": "117670", "frame": 100, "index": 1, "player_id": 9, "111": 0.1},
            {"match_id": "117670", "frame": 101, "index": 1, "player_id": 9, "111": 0.2},
            {"match_id": "117670", "frame": 105, "index": 1, "player_id": 9, "111": 0.3},
            {"match_id": "117670", "frame": 200, "index": 2, "player_id": 10, "111": 0.4},
        ]
    )

    pruned = post.prune_component_frames(component)

    assert pruned[["index", "frame"]].to_dict("records") == [
        {"index": 1, "frame": 100},
        {"index": 1, "frame": 105},
        {"index": 2, "frame": 200},
    ]


def test_melt_component_frame_preserves_possessor_and_renames_receiver() -> None:
    component = pd.DataFrame(
        [
            {
                "match_id": 117670,
                "frame": 337,
                "index": 0,
                "period": 1,
                "player_id": 63637,
                "attacking_side": "right_to_left",
                "63801": 0.4,
                "69889": 0.6,
                "shot": 0.01,
            }
        ]
    )

    melted = post.melt_component_frame(component, "action_intent", "test_component")

    assert list(melted.columns) == ["match_id", "frame", "index", "player_id", "receiver_id", "action_intent"]
    assert melted["match_id"].tolist() == ["117670", "117670"]
    assert melted["player_id"].tolist() == [63637, 63637]
    assert melted["receiver_id"].tolist() == [63801, 69889]
    assert melted["action_intent"].tolist() == [0.4, 0.6]


def test_compute_model_scores_calculates_pass_score_risk_reward_and_game_state_value() -> None:
    model_data = pd.DataFrame(
        [
            {
                "match_id": "117670",
                "frame": 337,
                "index": 0,
                "player_id": 63637,
                "receiver_id": 63801,
                "action_intent": 0.2,
                "pass_intent": 0.7,
                "pass_success": 0.8,
                "outcome_scoring_success": 0.5,
                "outcome_scoring_failure": 0.2,
                "outcome_conceding_success": 0.1,
                "outcome_conceding_failure": 0.3,
            },
            {
                "match_id": "117670",
                "frame": 337,
                "index": 0,
                "player_id": 63637,
                "receiver_id": 69889,
                "action_intent": 0.8,
                "pass_intent": 0.3,
                "pass_success": 0.2,
                "outcome_scoring_success": 0.6,
                "outcome_scoring_failure": 0.1,
                "outcome_conceding_success": 0.1,
                "outcome_conceding_failure": 0.05,
            },
        ]
    )

    scored = post.compute_model_scores(model_data)

    pass_scores = scored.sort_values("receiver_id")["pass_score"].tolist()
    risks = scored.sort_values("receiver_id")["risk"].tolist()
    rewards = scored.sort_values("receiver_id")["reward"].tolist()
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(pass_scores, [0.3, 0.14]))
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(risks, [0.14, 0.06]))
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(rewards, [0.44, 0.2]))
    assert all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        for left, right in zip(scored["game_state_value"].tolist(), [0.252, 0.252])
    )


def test_add_scores_to_event_data_adds_start_end_next_values_and_special_dm_scores() -> None:
    model_data = pd.DataFrame(
        [
            {"match_id": "117670", "index": 0, "frame": 337, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.20, "risk": 0.02, "reward": 0.22, "game_state_value": 0.25},
            {"match_id": "117670", "index": 0, "frame": 337, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.30, "risk": 0.03, "reward": 0.33, "game_state_value": 0.25},
            {"match_id": "117670", "index": 0, "frame": 346, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.40, "risk": 0.04, "reward": 0.44, "game_state_value": 0.35},
            {"match_id": "117670", "index": 0, "frame": 346, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.50, "risk": 0.05, "reward": 0.55, "game_state_value": 0.35},
            {"match_id": "117670", "index": 1, "frame": 400, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.10, "risk": 0.01, "reward": 0.11, "game_state_value": 0.10},
            {"match_id": "117670", "index": 1, "frame": 400, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.20, "risk": 0.02, "reward": 0.22, "game_state_value": 0.10},
            {"match_id": "117670", "index": 1, "frame": 410, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.18, "risk": 0.01, "reward": 0.19, "game_state_value": 0.18},
            {"match_id": "117670", "index": 1, "frame": 410, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.28, "risk": 0.02, "reward": 0.30, "game_state_value": 0.18},
            {"match_id": "117670", "index": 2, "frame": 500, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.30, "risk": 0.03, "reward": 0.33, "game_state_value": 0.30},
            {"match_id": "117670", "index": 2, "frame": 500, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.40, "risk": 0.04, "reward": 0.44, "game_state_value": 0.30},
            {"match_id": "117670", "index": 2, "frame": 510, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.31, "risk": 0.03, "reward": 0.34, "game_state_value": 0.31},
            {"match_id": "117670", "index": 2, "frame": 510, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.41, "risk": 0.04, "reward": 0.45, "game_state_value": 0.31},
            {"match_id": "117670", "index": 3, "frame": 600, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.45, "risk": 0.04, "reward": 0.49, "game_state_value": 0.45},
            {"match_id": "117670", "index": 3, "frame": 600, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.55, "risk": 0.05, "reward": 0.60, "game_state_value": 0.45},
            {"match_id": "117670", "index": 3, "frame": 610, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.60, "risk": 0.05, "reward": 0.65, "game_state_value": 0.46},
            {"match_id": "117670", "index": 3, "frame": 610, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.50, "risk": 0.05, "reward": 0.55, "game_state_value": 0.46},
            {"match_id": "117670", "index": 4, "frame": 700, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.20, "risk": 0.02, "reward": 0.22, "game_state_value": 0.20},
            {"match_id": "117670", "index": 4, "frame": 700, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.30, "risk": 0.03, "reward": 0.33, "game_state_value": 0.20},
            {"match_id": "117670", "index": 4, "frame": 710, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.21, "risk": 0.02, "reward": 0.23, "game_state_value": 0.21},
            {"match_id": "117670", "index": 4, "frame": 710, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.31, "risk": 0.03, "reward": 0.34, "game_state_value": 0.21},
            {"match_id": "117670", "index": 5, "frame": 800, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.50, "risk": 0.05, "reward": 0.55, "game_state_value": 0.50},
            {"match_id": "117670", "index": 5, "frame": 800, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.60, "risk": 0.06, "reward": 0.66, "game_state_value": 0.50},
            {"match_id": "117670", "index": 5, "frame": 810, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.65, "risk": 0.06, "reward": 0.71, "game_state_value": 0.51},
            {"match_id": "117670", "index": 5, "frame": 810, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.60, "risk": 0.06, "reward": 0.66, "game_state_value": 0.51},
            {"match_id": "999", "index": 6, "frame": 900, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.70, "risk": 0.07, "reward": 0.77, "game_state_value": 0.70},
            {"match_id": "999", "index": 6, "frame": 900, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.80, "risk": 0.08, "reward": 0.88, "game_state_value": 0.70},
            {"match_id": "999", "index": 6, "frame": 910, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.71, "risk": 0.07, "reward": 0.78, "game_state_value": 0.72},
            {"match_id": "999", "index": 6, "frame": 910, "player_id": 63637, "receiver_id": 70000, "pass_score": 0.81, "risk": 0.08, "reward": 0.89, "game_state_value": 0.72},
        ]
    )
    event_data = pd.DataFrame(
        [
            {"match_id": "117670", "index": 0, "team_id": 1, "end_type": "pass", "frame_start": 335, "frame_end": 347, "player_id": 63637, "player_targeted_id": 69889},
            {"match_id": "117670", "index": 1, "team_id": 1, "end_type": "foul_suffered", "frame_start": 400, "frame_end": 410, "player_id": 63637, "player_targeted_id": pd.NA},
            {"match_id": "117670", "index": 2, "team_id": 1, "end_type": "possession_loss", "frame_start": 500, "frame_end": 510, "player_id": 63637, "player_targeted_id": pd.NA},
            {"match_id": "117670", "index": 3, "team_id": 1, "end_type": "pass", "frame_start": 600, "frame_end": 610, "player_id": 63637, "player_targeted_id": 69889},
            {"match_id": "117670", "index": 4, "team_id": 1, "end_type": "possession_loss", "frame_start": 700, "frame_end": 710, "player_id": 63637, "player_targeted_id": pd.NA},
            {"match_id": "117670", "index": 5, "team_id": 2, "end_type": "pass", "frame_start": 800, "frame_end": 810, "player_id": 63637, "player_targeted_id": 69889},
            {"match_id": "999", "index": 6, "team_id": 1, "end_type": "possession_loss", "frame_start": 900, "frame_end": 910, "player_id": 63637, "player_targeted_id": pd.NA},
        ]
    )
    event_data = post.normalize_event_identifiers(event_data)

    scored = post.add_scores_to_event_data(model_data, event_data)

    assert scored.columns.tolist()[-14:] == [
        "pass_score",
        "risk",
        "reward",
        "game_state_value_start",
        "game_state_value_end",
        "game_state_value_next",
        "action_epv",
        "dm_score",
        "pass_dm_score",
        "carry_epv",
        "pass_epv",
        "z_dm_score",
        "z_pass_dm_score",
        "rank",
    ]
    assert math.isclose(scored.loc[0, "pass_score"], 0.40, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "risk"], 0.04, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "reward"], 0.44, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "game_state_value_start"], 0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "game_state_value_end"], 0.35, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "game_state_value_next"], 0.10, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "action_epv"], -0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "dm_score"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "pass_dm_score"], 0.05, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "carry_epv"], 0.10, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "pass_epv"], -0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "z_dm_score"], 1.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "z_pass_dm_score"], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "rank"], 2.0, rel_tol=1e-9, abs_tol=1e-9)

    assert pd.isna(scored.loc[1, "pass_score"])
    assert pd.isna(scored.loc[1, "risk"])
    assert pd.isna(scored.loc[1, "reward"])
    assert math.isclose(scored.loc[1, "game_state_value_start"], 0.10, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[1, "game_state_value_end"], 0.18, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[1, "action_epv"], 0.20, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[1, "dm_score"], 0.08, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[1, "pass_dm_score"])
    assert math.isclose(scored.loc[1, "carry_epv"], 0.08, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[1, "pass_epv"], 0.12, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[1, "z_dm_score"])
    assert pd.isna(scored.loc[1, "z_pass_dm_score"])
    assert pd.isna(scored.loc[1, "rank"])

    assert math.isclose(scored.loc[2, "game_state_value_next"], 0.45, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "action_epv"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "dm_score"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "carry_epv"], 0.01, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "pass_epv"], 0.14, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "pass_score"], 0.60, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "action_epv"], -0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "dm_score"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "pass_dm_score"], 0.14, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "pass_epv"], -0.26, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "z_dm_score"], 1.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "z_pass_dm_score"], 1.4, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[3, "rank"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[4, "game_state_value_next"], -0.50, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[4, "action_epv"], -0.70, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[4, "dm_score"], -0.70, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[4, "carry_epv"], 0.01, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[4, "pass_epv"], -0.71, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[5, "game_state_value_next"])
    assert pd.isna(scored.loc[5, "action_epv"])
    assert pd.isna(scored.loc[5, "pass_epv"])
    assert math.isclose(scored.loc[5, "rank"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[6, "game_state_value_next"])
    assert pd.isna(scored.loc[6, "action_epv"])
    assert pd.isna(scored.loc[6, "pass_epv"])
    assert pd.isna(scored.loc[6, "dm_score"])
