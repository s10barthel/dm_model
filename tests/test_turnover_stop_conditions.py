import numpy as np
import pandas as pd
import pytest

from datatools.return_windows import TurnoverWindows, boundary_segments
from datatools.stop_conditions import action_targets, definition
from datatools.xt_targets import LogProgressionSurface, transform_action_targets
from project_config import parse_return_type, validate_return_type_for_target_family
from tests.test_xt_refresh import sample


def events(teams, types=None):
    n = len(teams)
    frame = pd.DataFrame(dict(object_id=[t + "_1" for t in teams],
        spadl_type=types or ["pass"] * n, period_id=[1] * n,
        action_id=range(n), original_event_id=range(n), start_x=np.linspace(10, 90, n),
        start_y=[34.] * n, success=[False] * n, expected_goal=[np.nan] * n,
        xT=np.linspace(.1, .9, n), goal_distance=np.linspace(.1, .9, n), epv=np.linspace(.1, .9, n)))
    return frame


def test_restart_excludes_eligible_acquisition_from_budget_and_restarts_once():
    e = events(["home", "home", "away", "away", "home", "home", "away"])
    w = TurnoverWindows(e, "base")
    positions, acquisition = w.select(0, "home", 3, True)
    assert acquisition == 2
    assert positions.tolist() == [1, 2, 3, 4, 5]
    assert w.select(0, "home", 3)[0].tolist() == [1, 2, 3]


@pytest.mark.parametrize("kind", ["interception", "ball_recovery", "tackle", "keeper_claim", "keeper_pick_up",
    "keeper_save", "keeper_sweeper", "shot_block", "keeper_punch", "clearance"])
def test_simple_acquisition_without_success_or_control_checks(kind):
    e = events(["home", "home", "away", "away", "away", "home", "away"],
               ["pass", "pass", kind, "pass", "pass", "pass", "pass"])
    positions, acquisition = TurnoverWindows(e, "base").select(0, "home", 3, True)
    assert acquisition == 2
    assert positions.tolist() == [1, 3, 4, 5]


def test_bad_touch_excluded_and_latest_original_action_bounds_search():
    e = events(["home", "away", "home", "away", "away", "home"],
               ["pass", "interception", "pass", "bad_touch", "pass", "pass"])
    assert TurnoverWindows(e, "base").select(0, "home", 3, True)[1] == 4


def test_earliest_acquisition_and_skip_does_not_hide_turnover():
    e = events(["home", "away", "away", "away", "away"],
               ["pass", "interception", "keeper_save", "pass", "pass"])
    p, a = TurnoverWindows(e, "base").select(0, "home", 2, True, True)
    assert a == 1
    assert p.tolist() == [4]


def test_boundaries_goals_restarts_periods():
    e = events(["home"] * 7, ["pass", "throw_in", "pass", "goalkick", "shot", "pass", "pass"])
    e.loc[4, ["expected_goal", "success"]] = [.4, True]
    e.loc[6, "period_id"] = 2
    assert boundary_segments(e, "base") == [(0, 3), (3, 5), (5, 6), (6, 7)]
    assert boundary_segments(e, "set-pieces") == [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7)]


def test_gain_uses_acquisition_surface_rotates_and_clips():
    e = events(["home", "away", "away", "away"], ["pass", "interception", "pass", "pass"])
    surface = LogProgressionSurface()
    e["xT"] = surface.predict_xy(np.where(e.object_id.str.startswith("away"), 105 - e.start_x, e.start_x), e.start_y)
    pairs = pd.DataFrame([[0., 0.]], columns=["scores_xt", "concedes_xt"])
    settings = dict(version=2, representation="gain", clip_negative_gains=True)
    result = transform_action_targets(e.iloc[:1], pairs, settings, surface, events=e, return_type="next_3", stop_condition="base")
    assert result.iloc[0, 1] == 0  # Opponent only moves backward after acquisition.
    settings["clip_negative_gains"] = False
    assert transform_action_targets(e.iloc[:1], pairs, settings, surface, events=e, return_type="next_3").iloc[0, 1] < 0


def test_next_plus_validation():
    assert parse_return_type("next_3+_skip1") == ("next_plus", 3, True)
    assert validate_return_type_for_target_family("next_3+", "xt") == "next_3+"
    with pytest.raises(ValueError):
        validate_return_type_for_target_family("next_3+", "goal")


def test_all_family_stops_and_in_state_shortening():
    e = events(["home"] * 4, ["pass", "pass", "throw_in", "pass"])
    meta = dict(surface_type="log_progression", k=2, min_goal_distance=1)
    base = action_targets(e.iloc[:1], e, "next_3", "base", {}, None, meta)
    pieces = action_targets(e.iloc[:1], e, "next_3", "set-pieces", {}, None, meta)
    assert base.scores_xt.iloc[0] == .9
    assert pieces.scores_xt.iloc[0] == pytest.approx(e.xT.iloc[1])
    shortened = action_targets(e.iloc[:1], e, "in_2", "set-pieces", {}, None, meta)
    assert shortened.scores_xt.iloc[0] == 0


def test_availability_per_return_and_legacy():
    metadata = dict(stop_conditions=dict(version=1, available={"next_3": ["base"], "next_5": ["base", "set-pieces"]}))
    assert definition(metadata, "next_5", "set-pieces")["mode"] == "set-pieces"
    with pytest.raises(ValueError):
        definition(metadata, "next_3", "set-pieces")
    with pytest.raises(ValueError):
        definition({}, "next_3")
    assert definition({}, "next_3", legacy=True)["version"] == 0


def test_staged_migration_optional_variants_and_xt_only_refresh(sample, tmp_path, monkeypatch):
    import json
    import torch
    import project_config
    from datatools.stop_conditions import refresh_labels
    from datatools.xt_refresh import refresh_inventory, fingerprint
    root, event_dir, sidecars, metadata, source, _, _ = sample
    source = source.assign(start_x=10., start_y=20.)
    source.drop(columns="xT").to_csv(event_dir / "m.csv", index=False)
    for name in ("goal_distance", "epv"):
        folder = tmp_path / name
        folder.mkdir()
        source[["action_id"]].assign(**{name: [.1, .4, .8]}).to_csv(folder / "m.csv", index=False)
        monkeypatch.setattr(project_config, name.upper() + "_MATCH_DIR", folder)
    graphs = {p: fingerprint(p) for p in root.rglob("*.pt") if "graphs" in p.parent.name}
    requests = {r.removesuffix("_spstop"): ["base"] for r in metadata["return_types"]}
    refresh_labels(root, metadata, requests, event_dir=event_dir, xt_match_dir=sidecars)
    assert not (root / "labels" / "set-pieces").exists()
    (root / "metadata.json").write_text(json.dumps(metadata))
    refresh_inventory(root, metadata)
    repeated = refresh_labels(root, metadata, requests, event_dir=event_dir, xt_match_dir=sidecars)
    assert repeated["rewritten_files"] == 0
    refresh_labels(root, metadata, {"next_3": ["set-pieces"]}, event_dir=event_dir, xt_match_dir=sidecars)
    assert metadata["stop_conditions"]["available"]["next_3"] == ["base", "set-pieces"]
    assert metadata["stop_conditions"]["available"]["next_3_skip1"] == ["base"]
    selected = root / "labels" / "set-pieces" / "action_labels_next_3" / "m.pt"
    piece_labels = torch.load(selected, weights_only=True)
    from datatools import config
    piece_labels[:, config.LABEL_INDEX["scores_xg"]] += .2
    torch.save(piece_labels, selected)
    refresh_labels(root, metadata, {"next_3": ["base", "set-pieces"]},
        target_families=["goal_distance", "progression"], event_dir=event_dir, xt_match_dir=sidecars)
    torch.testing.assert_close(torch.load(selected, weights_only=True)[:, config.LABEL_INDEX["scores_xg"]],
                               piece_labels[:, config.LABEL_INDEX["scores_xg"]])
    before = {p: torch.load(p, weights_only=True) for p in (root / "labels").rglob("*.pt")}
    metadata["xt_targets"] = dict(version=2, representation="gain", clip_negative_gains=True)
    refresh_labels(root, metadata, metadata["stop_conditions"]["available"], xt_only=True,
        event_dir=event_dir, xt_match_dir=sidecars,
        surface_metadata=dict(surface_type="log_progression", k=2, min_goal_distance=1))
    keep = [i for i in range(len(config.LABEL_COLUMNS)) if i not in (config.LABEL_INDEX["scores_xt"], config.LABEL_INDEX["concedes_xt"])]
    for path, tensor in before.items():
        torch.testing.assert_close(torch.load(path, weights_only=True)[:, keep], tensor[:, keep], equal_nan=True)
    assert all(fingerprint(p) == digest for p, digest in graphs.items())


def test_cli_set_piece_requires_explicit_returns_and_migration_is_separate(monkeypatch):
    import sys
    from scripts.generate_relevant_features import parse_args
    monkeypatch.setattr(sys, "argv", ["generate", "--extend-feature-run-id", "base", "--stop-condition", "set-pieces"])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, "argv", ["generate", "--extend-feature-run-id", "base", "--refresh-stop-conditions", "--return-type", "next_3"])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, "argv", ["generate", "--return-type", "next_3+", "--stop-condition", "set-pieces"])
    args = parse_args()
    assert args.requested_return_types == ["next_3+"]
    assert args.stop_condition == "set-pieces"


@pytest.mark.parametrize("selector", [
    dict(train_count=765, train_split_percent=None),
    dict(train_count=None, train_split_percent=60),
])
def test_stop_migration_cli_inherits_saved_selection(monkeypatch, selector):
    import sys
    from scripts import generate_relevant_features as generator
    metadata = dict(selector, next_action_conditions_enabled=False)
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(sys, "argv", ["generate", "--extend-feature-run-id", "base", "--refresh-stop-conditions"])
    args = generator.parse_args()
    assert args.train_count == selector["train_count"]
    assert args.train_split == selector["train_split_percent"]
    assert args.next_action_conditions_enabled is False


def test_stop_migration_cli_preserves_explicit_split_for_validation(monkeypatch):
    import sys
    from scripts import generate_relevant_features as generator
    metadata = dict(train_count=765, train_split_percent=None, next_action_conditions_enabled=False)
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(sys, "argv", ["generate", "--extend-feature-run-id", "base", "--refresh-stop-conditions",
                                      "--train-split", "60"])
    args = generator.parse_args()
    assert args.train_count is None and args.train_split == 60
    assert args.next_action_conditions_enabled is False


def test_legacy_additions_require_migration_and_existing_return_can_add_set_pieces():
    from tests.test_feature_run_extension import FeatureRunExtensionPlanTests, make_args, make_metadata
    helper = FeatureRunExtensionPlanTests()
    metadata = make_metadata()
    metadata.pop("stop_conditions")
    with pytest.raises(ValueError, match="refresh-stop-conditions"):
        helper.build_plan(make_args(["next_3+"]), metadata)
    metadata = make_metadata(["next_3"])
    args = make_args(["next_3"])
    args.stop_condition = "set-pieces"
    plan = helper.build_plan(args, metadata)
    assert not plan.added_return_types
    assert not plan.command_steps


def test_fresh_base_match_and_refresh_calculation_agree():
    from datatools.match import Match
    from datatools.xt_refresh import action_return_pairs
    e = events(["home", "away", "away", "home", "away"], ["pass", "interception", "pass", "goalkick", "pass"])
    meta = dict(surface_type="log_progression", k=2, min_goal_distance=1)
    settings = dict(version=2, representation="gain", clip_negative_gains=True)
    surface = LogProgressionSurface()
    match = Match.__new__(Match)
    match.events, match.stop_condition, match.xt_surface_metadata = e, "base", meta
    for ret in ("next_3", "next_3+", "next_3+_skip1"):
        kind, n, skip = parse_return_type(ret)
        raw = match._get_events_for_return_type(ret, kind, n, skip)
        fresh = transform_action_targets(e, action_return_pairs(e, raw), settings, surface,
                                         events=e, return_type=ret, stop_condition="base")
        refreshed = action_targets(e, e, ret, "base", settings, surface, meta)
        np.testing.assert_allclose(fresh, refreshed[["scores_xt", "concedes_xt"]])


@pytest.mark.parametrize("return_type", ["next_3+", "next_3+_skip1"])
def test_full_label_generation_with_carry_perspective(sample, return_type):
    import torch
    from datatools.match import Match
    from datatools import config
    _, _, _, _, source, _, actions = sample
    source = source.assign(start_x=10., start_y=20.)
    surface = LogProgressionSurface()
    settings = dict(version=2, representation="gain", clip_negative_gains=True)
    metadata = dict(surface_type="log_progression", k=2, min_goal_distance=1)
    match = Match.__new__(Match)
    match.events, match.stop_condition, match.xt_surface_metadata = source, "base", metadata
    match.xt_targets, match.xt_surface = settings, surface
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.tracking = pd.DataFrame({"home_1_x": [1.], "home_1_y": [2.], "away_1_x": [3.], "away_1_y": [4.]})
    generated = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=return_type)
    refreshed = action_targets(actions, source, return_type, "base", settings, surface, metadata)
    xt_columns = [config.LABEL_INDEX[name] for name in ("scores_xt", "concedes_xt")]
    torch.testing.assert_close(generated[:, xt_columns], torch.tensor(refreshed[["scores_xt", "concedes_xt"]].to_numpy(), dtype=generated.dtype))
    assert generated[:, config.LABEL_INDEX["scores"]].isnan().all()


def test_migration_failure_does_not_replace_any_artifact(sample):
    from datatools.stop_conditions import refresh_labels
    from datatools.xt_refresh import fingerprint
    root, event_dir, sidecars, metadata, source, _, _ = sample
    source = source.assign(start_x=10., start_y=20.)
    source.drop(columns="xT").to_csv(event_dir / "m.csv", index=False)
    original = {path: fingerprint(path) for path in root.rglob("*") if path.is_file()}
    bad = pd.read_csv(sidecars / "m.csv")
    bad.loc[0, "xT"] = np.nan
    bad.to_csv(sidecars / "m.csv", index=False)
    with pytest.raises(ValueError, match="Invalid xT"):
        refresh_labels(root, metadata, {"next_3": ["base"]}, event_dir=event_dir, xt_match_dir=sidecars)
    assert all(fingerprint(path) == digest for path, digest in original.items())
    assert "stop_conditions" not in metadata


def test_models_reject_mixed_stop_conditions():
    from models.utils import validate_model_record_consistency
    record = dict(feature_run_id="f", intended_receiver_mode="original", return_type="next_3",
                  target_family="goal", graph_schema=dict(node_in_dim=26, edge_in_dim=2),
                  stop_condition_definition=dict(version=1, mode="base"))
    records = dict(outcome_scoring=record, outcome_conceding={**record})
    assert validate_model_record_consistency(records)["stop_condition_definition"] == record["stop_condition_definition"]
    records["outcome_conceding"]["stop_condition_definition"] = dict(version=1, mode="set-pieces")
    with pytest.raises(ValueError, match="stop conditions"):
        validate_model_record_consistency(records)


def test_pipeline_forwards_stop_selection():
    from types import SimpleNamespace
    from scripts.main import append_return_type_flag
    command = append_return_type_flag(["python", "scripts/train_relevant_models.py"],
        SimpleNamespace(return_type="next_3+", stop_condition="set-pieces"))
    assert command[-4:] == ["--return_type", "next_3+", "--stop-condition", "set-pieces"]


def test_evaluation_resolves_recorded_variant_and_rejects_legacy_mix(tmp_path, monkeypatch):
    import json
    from models import evaluation_data
    metadata = dict(label_layout_version=1, stop_conditions=dict(version=1, available={"next_3": ["base", "set-pieces"]}))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(evaluation_data, "resolve_feature_root", lambda _: tmp_path)
    model = dict(task="outcome_scoring", feature_run_id="f", return_type="next_3", intended_receiver_mode="original",
                 stop_condition_definition=dict(version=1, mode="set-pieces"))
    resolved = evaluation_data.resolve_evaluation_data({}, model)
    assert resolved["label_dir"] == str(tmp_path / "labels" / "set-pieces" / "action_labels_next_3")
    with pytest.raises(ValueError):
        evaluation_data.resolve_evaluation_data({}, {**model, "stop_condition_definition": None})


def test_raw_control_locations_change_baseline_and_budget_and_are_cached(tmp_path, monkeypatch):
    import project_config
    from datatools.xt_acquisitions import acquisition_context, CONTEXT_KEY
    canonical = events(["home", "away", "away", "home"])
    canonical["original_event_id"] = ["a", "b", "c", "d"]
    canonical["frame_id"] = [10, 30, 40, 50]
    folder = tmp_path / "controls"
    folder.mkdir()
    monkeypatch.setattr(project_config, "CONTROL_EVENTS_SYNCED_DIR", folder)
    controls = pd.DataFrame(dict(event_id=["a", "win", "block", "b", "c", "d"],
        event_kind=["pass", "tackle", "claim", "pass", "pass", "pass"],
        frame_id=[10, 15, 20, 30, 40, 50], period_id=[1] * 6,
        object_id=["home_1", None, "away_1", "away_1", "away_1", "home_1"],
        winner_id=[None, "away_1", None, None, None, None],
        claim_type=[None, None, "Block", None, None, None],
        possession_change=[False] * 6, winner_result=[None, "fouled", None, None, None, None]))
    controls.to_parquet(folder / "m.parquet")
    tracking = pd.DataFrame(dict(ball_x=[45., 40.], ball_y=[34., 34.]), index=[15, 20])
    prepared, cache, provenance = acquisition_context(canonical, "m", tmp_path, tracking=tracking)
    assert len(prepared.attrs[CONTEXT_KEY]) == 2
    assert provenance["ball_files_loaded"] == 0  # Reuse generation's already loaded tracking.
    window = TurnoverWindows(prepared, "base")
    positions, acquisition = window.select(window.events.index.get_indexer([0])[0], "home", 2, True)
    assert window.events.index[positions].tolist() == [1, 2]
    assert window.events.iloc[acquisition].start_x == 45.
    assert window.events.iloc[acquisition].spadl_type == "tackle"
    surface = LogProgressionSurface()
    from datatools.return_windows import action_xt_windows
    _, baseline = action_xt_windows(canonical.iloc[:1], prepared, "next_2+", surface, "base")
    assert baseline[0] == pytest.approx(surface.predict_xy(60., 34.))
    cache[0].parent.mkdir()
    cache[1].to_parquet(cache[0])
    original_read = pd.read_parquet
    def cached_only(path, *args, **kwargs):
        assert path == cache[0]
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", cached_only)
    cached, new_cache, cached_inputs = acquisition_context(canonical, "m", tmp_path)
    assert new_cache is None and cached_inputs["ball_files_loaded"] == 0
    assert cached.attrs[CONTEXT_KEY] == prepared.attrs[CONTEXT_KEY]
    with pytest.raises(ValueError, match="canonical events"):
        acquisition_context(canonical.assign(frame_id=[11, 30, 40, 50]), "m", tmp_path)
