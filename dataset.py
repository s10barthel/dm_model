from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from datatools.config import LABEL_COLUMNS, LABEL_INDEX, TASK_CONFIG
from datatools.utils import (
    drop_goal_nodes,
    drop_non_blocker_nodes,
    drop_opponent_nodes,
    sparsify_edges,
)


class ActionDataset(Dataset):
    def __init__(
        self,
        match_ids,
        feature_dir="data/ajax/features/action_graphs",
        label_dir="data/ajax/features/action_labels_disc_0.9",
        task=None,
        inplay_only=False,
        min_pass_dur=0.0,
        shot_success_type="unblocked",
        xy_only=False,
        possessor_aware=True,
        keeper_aware=True,
        ball_z_aware=True,
        poss_vel_aware=True,
        extend_features=True,
        drop_non_blockers=False,
        sparsify="none",
        max_edge_dist=10.0,
        train=True,
    ):
        feature_root = Path(feature_dir)
        label_root = Path(label_dir)
        self.requested_match_ids = [str(match_id) for match_id in match_ids]
        self.loaded_match_ids: list[str] = []
        self.skipped_matches: dict[str, str] = {}

        features = []
        label_tensors: list[torch.Tensor] = []

        for match_id in tqdm(self.requested_match_ids):
            feature_path = feature_root / f"{match_id}.pt"
            label_path = label_root / f"{match_id}.pt"
            if not feature_path.exists() or not label_path.exists():
                missing_parts = []
                if not feature_path.exists():
                    missing_parts.append("feature")
                if not label_path.exists():
                    missing_parts.append("label")
                self.skipped_matches[match_id] = f"missing_{'_and_'.join(missing_parts)}"
                continue

            try:
                match_features = torch.load(feature_path, weights_only=False)
                match_labels = torch.load(label_path, weights_only=False)
            except Exception as exc:
                self.skipped_matches[match_id] = f"{type(exc).__name__}: {exc}"
                continue

            if not isinstance(match_features, list):
                self.skipped_matches[match_id] = "feature_tensor_is_not_a_list"
                continue
            if not isinstance(match_labels, torch.Tensor) or match_labels.ndim != 2:
                self.skipped_matches[match_id] = "label_tensor_has_invalid_shape"
                continue
            if len(match_features) != int(match_labels.shape[0]):
                self.skipped_matches[match_id] = (
                    f"feature_label_length_mismatch:{len(match_features)}!={int(match_labels.shape[0])}"
                )
                continue

            features.extend(match_features)
            label_tensors.append(match_labels)
            self.loaded_match_ids.append(match_id)

        if label_tensors:
            labels = torch.cat(label_tensors)
        else:
            labels = torch.empty((0, len(LABEL_COLUMNS)), dtype=torch.float32)
            self.features = []
            self.labels = labels
            self.ip_weights = None
            return

        condition: torch.Tensor = torch.ones(labels.shape[0]).bool()

        if not TASK_CONFIG.at[task, "pass"]:
            condition &= labels[:, 1] == 0
        if not TASK_CONFIG.at[task, "dribble"]:
            condition &= labels[:, 2] == 0
        if not TASK_CONFIG.at[task, "shot"]:
            condition &= labels[:, 3] == 0

        if not train and task == "shot_blocking":  # Only evaluate shot instances when testing
            condition &= labels[:, 3] == 1

        if task.startswith("success"):  # Only include successful actions
            pass_success = (labels[:, 1] == 1) & (labels[:, LABEL_INDEX["success"]] == 1)
            dribble_success = (labels[:, 2] == 1) & (labels[:, LABEL_INDEX["success"]] == 1)

            if shot_success_type == "goal":
                shot_success = (labels[:, 3] == 1) & (labels[:, LABEL_INDEX["success"]] == 1)
            elif shot_success_type == "unblocked":
                shot_success = (labels[:, 3] == 1) & (labels[:, LABEL_INDEX["blocked"]] == 0)
            else:
                shot_success = labels[:, 3] == 1

            condition &= pass_success | dribble_success | shot_success

        if task.startswith("failure"):  # Only include failed actions
            pass_failure = (labels[:, 1] == 1) & (labels[:, LABEL_INDEX["success"]] == 0)
            dribble_failure = (labels[:, 2] == 1) & (labels[:, LABEL_INDEX["success"]] == 0)

            if shot_success_type == "goal":
                oppo_received = []
                for i, graph in enumerate(features):
                    receiver_index = int(labels[i, 6].item())
                    if graph is None or receiver_index < 0 or receiver_index >= graph.x.shape[0]:
                        oppo_received.append(False)
                    else:
                        oppo_received.append(bool((graph.x[receiver_index, 0] == 0).item()))
                shot_failure = (labels[:, 3] == 1) & (labels[:, LABEL_INDEX["success"]] == 0) & torch.tensor(oppo_received)
            elif shot_success_type == "unblocked":
                shot_failure = (labels[:, 3] == 1) & (labels[:, LABEL_INDEX["blocked"]] == 1)
            else:
                shot_failure = labels[:, 3] == 1

            condition &= pass_failure | dribble_failure | shot_failure

        if TASK_CONFIG.at[task, "intended"]:
            # Only include actions with valid intended receivers
            condition &= labels[:, 5] != -1

        if inplay_only:
            # Only include actions with valid receivers (excluding out-of-play)
            condition &= labels[:, 6] != -1

        if min_pass_dur > 0:
            # Remove passes with not enough durations
            condition &= (labels[:, 1] == 0) | (labels[:, 7] >= min_pass_dur)

        self.features = []
        self.labels = []

        for i in tqdm(condition.nonzero()[:, 0].numpy()):
            graph: Data = features[i]
            graph_labels: torch.Tensor = labels[i]

            if graph is None:
                continue

            try:
                possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
            except RuntimeError:
                continue

            if task == "failure_receiver" and inplay_only:
                n_teammates = int((graph.x[:, 0] == 1).sum().item())
                n_opponents = int((graph.x[:, 0] == 0).sum().item())
                receiver_index = int(graph_labels[6].item()) - n_teammates

                # Skip mislabeled failures where the recorded receiver is not an opponent
                if receiver_index < 0 or receiver_index >= n_opponents:
                    continue

            if xy_only:  # Do not refer to handcrafted features
                graph.x[7:12] = 0
                graph.x[13:19] = 0

            if not possessor_aware:  # Do not refer to possessor-related features
                assert not extend_features
                graph.x[:, 13:] = 0

            if not poss_vel_aware:  # Ignore the features related to the ball possessor's velocity
                if possessor_aware:
                    graph.x[graph.x[:, 13] == 1, 5:9] = 0
                graph.x[:, 17:19] = 0

            if not keeper_aware:  # Do not distinguish between goalkeepers and outfield players
                graph.x[:, 1] = 0

            if not ball_z_aware:  # Set the ball height for every action as 0
                graph.x[:, 12] = 0

            if not extend_features:
                graph.x[:, 19:] = 0

            if not TASK_CONFIG.at[task, "include_goals"]:
                graph, graph_labels = drop_goal_nodes(graph, graph_labels)

            if task.endswith("oppo_agn"):
                graph, graph_labels = drop_opponent_nodes(graph, graph_labels)

            if drop_non_blockers:
                assert possessor_aware
                possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
                graph, graph_labels = drop_non_blocker_nodes(graph, graph_labels, 13)

            if sparsify == "distance":
                assert possessor_aware
                possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
                graph = sparsify_edges(graph, "distance", possessor_index, max_edge_dist)
            elif sparsify == "delaunay" and graph.x.shape[0] > 3:
                graph = sparsify_edges(graph, "delaunay")

            if task == "failure_receiver":
                intent_onehot = torch.zeros(graph.x.shape[0])
                intent_onehot[labels[i, 5].long()] = 1
                graph.x = torch.cat([graph.x, intent_onehot.unsqueeze(1)], -1)

            self.features.append(graph)
            self.labels.append(graph_labels)

        if self.labels:
            self.labels = torch.stack(self.labels, axis=0)
        else:
            self.labels = torch.empty((0, labels.shape[1]), dtype=labels.dtype)
        self.ip_weights = None

    def set_inverse_propensity_weights(self, ip_weights: torch.Tensor):
        assert len(ip_weights) == len(self)
        self.ip_weights = ip_weights

    def balance_real_and_augmented(self):
        real_indices = torch.nonzero(self.labels[:, LABEL_INDEX["is_real"]] == 1).flatten()
        augmented_indices = torch.nonzero(self.labels[:, LABEL_INDEX["is_real"]] == 0).flatten()

        if len(real_indices) < len(augmented_indices):
            sampled_indices = torch.randperm(len(augmented_indices))[: len(real_indices)]
            augmented_indices = augmented_indices[sampled_indices]

        indices = torch.cat([real_indices, augmented_indices])
        self.features = [self.features[i] for i in indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        if self.ip_weights is None:
            return self.features[i], self.labels[i], torch.tensor(1.0, dtype=torch.float32)
        else:
            return self.features[i], self.labels[i], self.ip_weights[i]
