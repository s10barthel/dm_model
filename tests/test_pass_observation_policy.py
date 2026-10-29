"""Regression coverage for independent receipts, clean observations and eligibility."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import config
from datatools.pass_trajectory import PassEndpointResolver, PASS_DURATION, PASS_DURATION_KEY
from datatools.failed_pass_trajectory import detect, KEY, VERSION, prepare_new_match
from datatools.pass_eligibility import policy_metadata
from datatools.graph_feature import summarize_ball_trajectory
from datatools.match import Match
from datatools.carry_policy import reconstruct_spells
from dataset import ActionDataset


def evidence(*, success=False, receipt=None, contact=None):
    tracking = pd.DataFrame(dict(period_id=1, ball_state="alive", timestamp=np.arange(61)/25,
        ball_x=np.arange(61)*.4, ball_y=0., ball_z=0., home_1_x=0., home_1_y=0.,
        home_2_x=np.arange(61)*.4, home_2_y=0., away_1_x=100., away_1_y=100.), index=range(61))
    action = pd.Series(dict(frame_id=0, receive_frame_id=receipt, original_event_id="pass", period_id=1,
        action_type="pass", spadl_type="pass", object_id="home_1", receiver_id="home_2" if success else "out",
        success=success, source_event_success=success, endpoint_source="kpi_csv" if receipt is not None else "unresolved"))
    events = [dict(event_id="pass", raw_tag="Play", frame_id=0, period_id=1, reception_frame_id=receipt)]
    if contact is not None:
        events.append(dict(event_id="flag", raw_tag="TacklingGame", frame_id=contact, period_id=1))
    return action, tracking, pd.DataFrame(events)


@pytest.mark.parametrize("contact,cutoff", [(0, None), (1, None), (2, None), (3, None), (4, 3), (16, 15), (28, 25)])
def test_official_contact_excludes_contaminated_frame(contact, cutoff):
    action, tracking, events = evidence(contact=contact)
    endpoint = PassEndpointResolver(tracking, events=events).resolve(action)
    assert endpoint.trajectory == cutoff
    if cutoff is None:
        assert endpoint.observation_reason == "insufficient_clean_intervals"
    else:
        assert detect(action, tracking, cutoff)["status"] != "unavailable"


@pytest.mark.parametrize("cutoff,usable", [(2, False), (3, True)])
def test_three_intervals_are_four_samples(cutoff, usable):
    action, tracking, _ = evidence()
    result = detect(action, tracking, cutoff)
    assert (result["status"] != "unavailable") == usable
    assert result["clean_intervals"] == cutoff


def test_missing_full_endpoint_does_not_prevent_receiver_input(tmp_path):
    action, tracking, events = evidence()
    resolver = PassEndpointResolver(tracking, events=events)
    actions = resolver.annotate(pd.DataFrame([action], index=[7]))
    assert pd.isna(actions.at[7, "pass_duration"])
    assert actions.at[7, "trajectory_end_frame_id"] == 25
    match = SimpleNamespace(actions=actions, tracking=tracking, fps=25, match_id="m", runtime_feature_root=tmp_path,
                            failed_pass_trajectory_policy=VERSION)
    prepare_new_match(match)
    assert np.isfinite(summarize_ball_trajectory(match, 7)).all()
    record = pd.read_parquet(tmp_path / "failed_pass_trajectories/m.parquet").iloc[0]
    assert record.status == "unchanged"
    assert record.clean_intervals == 25


@pytest.mark.parametrize("unknown_first", [True, False])
def test_only_untimed_contacts_that_can_affect_prefix_reject(unknown_first):
    action, tracking, events = evidence()
    unknown = dict(event_id="untimed", raw_tag="TacklingGame", frame_id=np.nan, period_id=1)
    later = dict(event_id="later", raw_tag="Play", frame_id=40, period_id=1)
    events = pd.concat([events, pd.DataFrame([unknown, later] if unknown_first else [later, unknown])], ignore_index=True)
    endpoint = PassEndpointResolver(tracking, events=events).resolve(action)
    assert endpoint.trajectory == (None if unknown_first else 25)
    if unknown_first:
        assert endpoint.observation_reason == "uncertain_early_contact"


def test_success_uses_official_receipt_despite_failed_pass_guards():
    action, tracking, events = evidence(success=True, receipt=30, contact=2)
    tracking.loc[0, "ball_state"] = "dead"
    events = pd.concat([events.iloc[:1], pd.DataFrame([dict(event_id="untimed", raw_tag="Foul", frame_id=np.nan,
        period_id=1)]), events.iloc[1:]], ignore_index=True)
    output = PassEndpointResolver(tracking, events=events).annotate(pd.DataFrame([action]))
    row = output.iloc[0]
    assert row.receive_frame_id == row.official_receipt_frame_id == 30
    assert row.pass_duration == 1.2
    assert pd.isna(row.trajectory_end_frame_id)
    assert row.observation_reason == "unusable_launch_state"
    assert row.receipt_status == row.duration_status == "valid"


def test_new_pass_filter_preserves_official_success_and_unknown_duration():
    action, tracking, events = evidence(success=True)
    actions = PassEndpointResolver(tracking, events=events).annotate(pd.DataFrame([action]))
    actions["training_sample_eligible"] = False
    actions["next_player_id"], actions["next_type"], actions["offside"] = "away_1", "tackle", False
    match = Match.__new__(Match)
    match.events, match.tracking, match.next_action_conditions_enabled = actions, tracking, True
    result = match.filter_passes()
    assert len(result) == 1 and result.iloc[0].success
    assert pd.isna(result.iloc[0].pass_duration)
    pd.testing.assert_frame_equal(match.events, actions)


def dataset_artifacts(root, *, augmented=False):
    features, labels_dir = root / "features", root / "labels"
    features.mkdir(parents=True)
    labels_dir.mkdir()
    (root / "metadata.json").write_text(json.dumps({PASS_DURATION_KEY: PASS_DURATION, **policy_metadata()}))
    x = torch.zeros((3, 24 if augmented else config.NODE_FEATURE_CORE_DIM))
    x[:2, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.cartesian_prod(torch.arange(3), torch.arange(3)).T.contiguous(),
                 edge_attr=torch.zeros((9, 2)))
    from datatools.trajectory import TRAJECTORY_SPEED
    graph.trajectory_speed_definition = TRAJECTORY_SPEED
    rows = torch.zeros((7, len(config.LABEL_COLUMNS)))
    rows[:, 0] = torch.arange(7)
    for name, value in dict(is_pass=1, n_players=3, intent_index=1, receiver_index=2, is_real=1).items():
        rows[:, config.LABEL_INDEX[name]] = value
    rows[:, config.LABEL_INDEX["success"]] = torch.tensor([1, 1, 0, 0, 0, 1, 1])
    rows[:, config.LABEL_INDEX["duration"]] = torch.tensor([float("nan"), .04, float("nan"), 1., 1., 1., 1.])
    rows[4, config.LABEL_INDEX["intent_index"]] = -1
    torch.save([graph.clone() for _ in range(6)] + [None], features / "m.pt")
    torch.save(rows, labels_dir / "m.pt")
    sidecar = root / "failed_pass_trajectories"
    sidecar.mkdir()
    pd.DataFrame([dict(event_key=f"pass-{i}", action_index=i, policy=VERSION,
        status="unavailable" if i == 3 else "unchanged", reason="uncertain_early_contact" if i == 3 else None,
        original_cutoff=25, cutoff=25, clean_intervals=25) for i in [2, 3, 4]]).to_parquet(sidecar / "m.parquet")
    return dict(feature_dir=features, label_dir=labels_dir, edge_in_dim=2, min_pass_dur=0., extend_features=False)


@pytest.mark.parametrize("task", ["action_intent", "pass_intent", "pass_success", "outcome_scoring", "outcome_conceding"])
def test_task_eligibility_ignores_duration_and_requires_clean_failed_receiver(tmp_path, task):
    options = dataset_artifacts(tmp_path)
    dataset = ActionDataset(["m"], task=task, **options)
    assert dataset.labels[:, 0].tolist() == [0, 1, 2, 5]
    assert dataset.selection_counts["excluded_by_observation"] == 1
    assert dataset.skipped_rows["graph_none"] == 1
    assert dataset.skipped_rows["failed_observation:uncertain_early_contact"] == 1
    with pytest.raises(ValueError, match="min_pass_dur must be 0"):
        ActionDataset(["m"], task=task, **dict(options, min_pass_dur=.1))


def test_prepared_and_ordinary_loaders_share_gate_and_sidecar_invalidation(tmp_path):
    import prepared_dataset as prepared
    options = dict(dataset_artifacts(tmp_path), task="pass_intent")
    ordinary = ActionDataset(["m"], **options)
    disk = prepared.PreparedActionDataset(["m"], cache_dir=tmp_path / "cache", progress=False, **options)
    assert disk.sample_count == len(ordinary)
    before = disk._source_signatures("m")
    path = tmp_path / "failed_pass_trajectories/m.parquet"
    table = pd.read_parquet(path)
    table.loc[table.action_index.eq(2), ["status", "reason"]] = ["unavailable", "tracking_jump"]
    table.to_parquet(path)
    assert before != disk._source_signatures("m")
    updated = prepared.PreparedActionDataset(["m"], cache_dir=tmp_path / "cache", progress=False, **options)
    assert updated.sample_count == len(ordinary) - 1


@pytest.mark.parametrize("missing_launch", [False, True])
def test_successful_receipt_starts_carry_independently_of_training(missing_launch):
    action, tracking, _ = evidence(success=True, receipt=10)
    action["official_receipt_frame_id"] = 10
    action["pass_endpoint_source"] = "unavailable"
    action["receive_frame_id"] = np.nan
    action["training_sample_eligible"] = False
    if missing_launch:
        action["frame_id"] = np.nan
    controls = pd.DataFrame([dict(event_id="release", event_kind="pass", frame_id=50, period_id=1, object_id="home_2")])
    result = reconstruct_spells(controls, pd.DataFrame([action]), tracking)
    assert result.start_kind.tolist() == ["receipt"]
    assert result.iloc[0].start_frame == 10
    assert result.iloc[0].valid
    tracking.loc[10, "home_2_x"] += 20
    invalid = reconstruct_spells(controls, pd.DataFrame([action]), tracking)
    assert invalid.start_kind.tolist() == ["receipt"]
    assert not invalid.iloc[0].valid
    assert invalid.iloc[0].reason == "acquisition_distance"



def test_success_intent_has_its_own_trajectory_graph_requirement(tmp_path):
    from datatools.trajectory import TRAJECTORY_SPEED
    options = dataset_artifacts(tmp_path, augmented=True)
    dataset = ActionDataset(["m"], task="success_intent", trajectory_speed_definition=TRAJECTORY_SPEED, **options)
    assert dataset.labels[:, 0].tolist() == [0, 1, 5]
    assert dataset.skipped_rows["graph_none"] == 1


def test_training_snapshots_policy_and_rejects_changed_resume():
    from datatools.pass_eligibility import resolve_training_policy, training_policy
    metadata = {PASS_DURATION_KEY: PASS_DURATION, **policy_metadata()}
    args = SimpleNamespace(min_pass_dur=0.)
    resolve_training_policy(args, metadata)
    assert args.pass_input_policy == training_policy(metadata)
    saved = {"pass_input_policy": args.pass_input_policy}
    resolve_training_policy(args, metadata, saved)
    changed = dict(metadata, failed_pass_trajectory_policy="different")
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_training_policy(args, changed, saved)
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_training_policy(args, metadata, {})
    # Legacy checkpoints can still resume against their recorded legacy inputs.
    resolve_training_policy(args, {PASS_DURATION_KEY: "canonical_official_pass_end_v3"}, {})


def test_invalid_quality_records_cannot_bypass_gate(tmp_path):
    options = dataset_artifacts(tmp_path)
    path = tmp_path / "failed_pass_trajectories/m.parquet"
    table = pd.read_parquet(path)
    table.loc[table.action_index.eq(2), "clean_intervals"] = np.nan
    table.to_parquet(path)
    with pytest.raises(ValueError, match="Invalid failed-pass quality"):
        ActionDataset(["m"], task="pass_intent", **options)



def test_stored_legacy_graph_replays_its_original_short_observation(tmp_path, monkeypatch):
    from datatools.failed_pass_trajectory import LEGACY_VERSION, validate_graph_coverage
    action, tracking, _ = evidence()
    root = tmp_path
    (root / "metadata.json").write_text(json.dumps({KEY: LEGACY_VERSION}))
    folder = root / "failed_pass_trajectories"
    folder.mkdir()
    pd.DataFrame([dict(event_key="pass", status="unchanged", cutoff=1, original_cutoff=1,
        policy=LEGACY_VERSION)]).to_parquet(folder / "m.parquet")
    match = SimpleNamespace(actions=pd.DataFrame([action], index=[7]), tracking=tracking,
        runtime_feature_root=root, match_id="m", fps=25)
    resolver = SimpleNamespace(resolve=lambda a: SimpleNamespace(trajectory=25))
    monkeypatch.setattr("datatools.pass_trajectory.ensure_pass_endpoints", lambda *a, **k: resolver)
    graph = Data(x=torch.ones((2, 24)))
    graph.trajectory_end_frame_id = 1
    setattr(graph, KEY, LEGACY_VERSION)
    validate_graph_coverage(match, [graph], [[7]])
