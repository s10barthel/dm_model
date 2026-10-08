"""Versioned goal context and prediction eligibility, shared by all consumers."""
from dataclasses import dataclass
from collections.abc import Mapping

import torch

from datatools import config

GOAL_CONTEXT_TASKS = frozenset({"action_intent", "pass_intent", "success_intent",
                               "pass_success", "outcome_scoring", "outcome_conceding"})


def _get(args, key, default=None):
    return args.get(key, default) if isinstance(args, Mapping) else getattr(args, key, default)


@dataclass(frozen=True)
class GoalPolicy:
    version: int
    goal_nodes_aware: bool
    include_goals: bool
    input_goals: bool

    def settings(self):
        return {"goal_context_version": self.version,
                "goal_nodes_aware": self.goal_nodes_aware, "include_goals": self.include_goals}


def goal_policy(args):
    task = _get(args, "task")
    default_candidates = bool(config.TASK_CONFIG.at[task, "include_goals"]) if task in config.TASK_CONFIG.index else False
    version = _get(args, "goal_context_version", 1)
    version = 1 if version is None else int(version)
    if version not in (1, 2):
        raise ValueError(f"Unsupported goal_context_version={version}.")
    nodes = _get(args, "goal_nodes_aware", True)
    nodes = True if nodes is None else bool(nodes)
    if version == 1 or task not in GOAL_CONTEXT_TASKS:
        return GoalPolicy(1, nodes, default_candidates, nodes and default_candidates)
    candidates = _get(args, "include_goals")
    candidates = default_candidates if candidates is None else bool(candidates)
    if candidates and task != "action_intent":
        raise ValueError(f"{task} requires include_goals=False; goals may only provide input context.")
    if candidates and not nodes:
        raise ValueError("include_goals=True requires goal_nodes_aware=True.")
    return GoalPolicy(2, nodes, candidates, nodes)


def resolve_training_goal_settings(args, resume=None):
    requested = {k: _get(args, k) for k in ("goal_nodes_aware", "include_goals")}
    if resume is not None:
        policy = goal_policy(resume)
        for key, value in requested.items():
            if value is not None and bool(value) != getattr(policy, key):
                raise ValueError(f"Cannot change {key} when resuming; start a new training run.")
    else:
        if _get(args, "task") not in GOAL_CONTEXT_TASKS and requested["include_goals"] is not None:
            raise ValueError("Explicit include_goals is supported only for the six goal-context tasks.")
        policy = goal_policy({"task": _get(args, "task"), "goal_context_version": 2, **requested})
    for key, value in policy.settings().items():
        setattr(args, key, value)
    return policy


def validate_source_goals(graph, policy, input_goals=None):
    expected = policy.input_goals if input_goals is None else input_goals
    if (policy.version != 2 and input_goals is None) or not expected:
        return
    goals = graph.x[:, config.NODE_FEATURE_IS_GOAL] == 1
    teams = graph.x[:, config.NODE_FEATURE_IS_TEAMMATE]
    if int((goals & (teams == 1)).sum()) != 1 or int((goals & (teams == 0)).sum()) != 1:
        raise ValueError("Goal context requires one attacking and one defending goal node in the source graph; rebuild compatible features.")


def candidate_mask(graph, args, exclude_possessor=False):
    """Mask node outputs; retaining a node as context does not make it a target."""
    task = _get(args, "task")
    policy = goal_policy(args)
    teammate = graph.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
    output_filter = config.TASK_CONFIG.at[task, "out_filter"] if task is not None else "teammates"
    mask = teammate if output_filter == "teammates" else (~teammate if output_filter == "opponents" else torch.ones_like(teammate))
    if policy.version == 2 and not policy.include_goals:
        mask = mask & (graph.x[:, config.NODE_FEATURE_IS_GOAL] != 1)
    if exclude_possessor:
        mask = mask & (graph.x[:, config.NODE_FEATURE_IS_POSSESSOR] != 1)
    return mask


def target_candidate_position(indices, target):
    positions = torch.where(indices == int(target))[0]
    if positions.numel() != 1:
        raise ValueError(f"Observed target node {int(target)} is not an eligible prediction candidate.")
    return int(positions.item())


def validate_observed_goal_target(graph, label, args):
    if goal_policy(args).version != 2:
        return
    target = label[config.LABEL_INDEX["intent_index"]]
    if not torch.isfinite(target) or target != target.trunc():
        raise ValueError("Observed target must be a finite integer node index.")
    target_candidate_position(torch.where(candidate_mask(graph, args))[0], int(target))
