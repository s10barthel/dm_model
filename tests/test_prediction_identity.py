import json

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
from datatools.utils import drop_nodes, filter_features_and_labels
from models.node_selection import selection_layout, selection_loss_metrics
from models.prediction_identity import resolve_legacy_node_ids, selection_prediction_identities


def graph(ids=("home_1", "home_2", "away_1")):
    x = torch.zeros((3, 20))
    x[:2, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[:, 3:5] = torch.tensor([[10., 20.], [30., 40.], [50., 60.]])
    edges = torch.tensor([[0, 1], [1, 0]])
    return Data(x=x, edge_index=edges, edge_attr=torch.ones(2, 2), node_ids=list(ids))


def labels():
    value = torch.zeros((1, len(config.LABEL_COLUMNS)))
    value[0, config.LABEL_INDEX["action_index"]] = 7
    value[0, config.LABEL_INDEX["intent_index"]] = 1
    value[0, 1] = 1
    value[0, 4] = 3
    return value


def test_intent_export_tracks_semantic_identity_through_candidate_order():
    first = graph()
    batch = Batch.from_data_list([first])
    args = {"task": "pass_intent", "goal_context_version": 2, "include_goals": False}
    layout = selection_layout(batch, labels(), "pass_intent", False, args)
    _, predictions, targets, _, _ = selection_loss_metrics(torch.tensor([0., 2., 1.]), layout)
    original = selection_prediction_identities(batch, layout, targets, predictions)[0]
    assert original["target_id"] == original["prediction_id"] == "home_2"
    assert json.loads(original["candidate_ids"]) == ["home_1", "home_2"]
    reordered = graph(("home_2", "home_1", "away_1"))
    reordered.x = first.x[torch.tensor([1, 0, 2])]
    value = labels()
    value[0, 5] = 0
    batch = Batch.from_data_list([reordered])
    layout = selection_layout(batch, value, "pass_intent", False, args)
    _, predictions, targets, _, _ = selection_loss_metrics(torch.tensor([2., 0., 1.]), layout)
    changed = selection_prediction_identities(batch, layout, targets, predictions)[0]
    assert changed["target_id"] == original["target_id"]
    assert changed["prediction_id"] == original["prediction_id"]
    assert json.loads(changed["candidate_ids"]) == ["home_2", "home_1"]


def test_missing_or_duplicate_semantic_identities_fail():
    value = graph()
    del value.node_ids
    batch = Batch.from_data_list([value])
    with pytest.raises(ValueError, match="node IDs"):
        selection_prediction_identities(batch, [], torch.tensor([0]), torch.tensor([0]))
    batch = Batch.from_data_list([graph(("home_1", "home_1", "away_1"))])
    layout = selection_layout(batch, labels(), "pass_intent", False)
    with pytest.raises(ValueError, match="Invalid semantic"):
        selection_prediction_identities(batch, layout, torch.tensor([1]), torch.tensor([1]))


@pytest.mark.parametrize("away", [False, True])
def test_legacy_identity_recovery_validates_geometry_after_node_removal(away):
    value = graph()
    tracking = pd.DataFrame({"home_1_x": [10.], "home_1_y": [20.],
                             "home_2_x": [30.], "home_2_y": [40.],
                             "away_1_x": [50.], "away_1_y": [60.]}, index=[100])
    actions = pd.DataFrame({"frame_id": [100], "object_id": ["away_1" if away else "home_1"]}, index=[7])
    if away:
        value.x[:, 3:5] = torch.tensor(config.FIELD_SIZE) - value.x[:, 3:5]
        value.x[:, 0] = 1 - value.x[:, 0]
    value.x = value.x[[0, 2]]
    assert resolve_legacy_node_ids(value, labels()[0], tracking, actions) == ["home_1", "away_1"]
    value.x[0, 3] = 11
    with pytest.raises(ValueError, match="unique node identity"):
        resolve_legacy_node_ids(value, labels()[0], tracking, actions)


def test_original_possessor_sidecar_survives_feature_and_node_masks():
    source = graph()
    original = source.x.clone()
    options = dict(task="pass_intent", edge_in_dim=2, possessor_aware=False, xy_only=False,
                   extend_features=False, pass_lane_features=False, keeper_aware=True,
                   ball_z_aware=True, poss_vel_aware=True, poss_rel_vel_aware=False,
                   sparsify="none", include_goals=False, goal_context_version=2, goal_nodes_aware=False)
    prepared, prepared_labels = filter_features_and_labels([source], labels(), options)
    assert torch.equal(source.x, original)
    assert not prepared[0].x[:, 13].any()
    assert prepared[0].inference_possessor_mask.tolist() == [True, False, False]
    reduced, _ = drop_nodes(prepared[0], prepared_labels[0], torch.tensor([True, False, True]))
    assert reduced.inference_possessor_mask.tolist() == [True, False]
