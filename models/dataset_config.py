from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataset import requires_goal_next10_diagnostics
from models.utils import (
    mask_possessor_relative_speed_edge_features_for_mode,
    mask_possessor_v_edge_features_for_mode,
    normalize_relative_speed_edge_feature_mode,
    normalize_v_edge_feature_mode,
    validate_relative_speed_edge_feature_mode,
)
from physical_pass_model import (
    PC_XPASS_SOURCE,
    model_uses_physical_xpass,
    normalize_pc_xpass_lane_survival_mode,
    pc_xpass_lane_survival_metadata_fingerprint,
    validate_pc_xpass_lane_survival_mode_cache_metadata,
    validate_physical_xpass_cache_metadata,
)


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _bool_arg(args: Any, name: str, default: bool) -> bool:
    value = _get_arg(args, name, default)
    return bool(default) if value is None else bool(value)


def build_action_dataset_kwargs(
    args: Any,
    *,
    train: bool,
    diagnostic_label_dir: str | None,
    pass_height_diagnostic_label_dir: str | None = None,
    require_goal_next10_diagnostics: bool | None = None,
    physical_cache_dir: str | None = None,
    lane_survival_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Build the feature-sensitive ActionDataset options shared by train and test."""
    task = str(_get_arg(args, "task"))
    v_edge_feature_mode = normalize_v_edge_feature_mode(
        _get_arg(args, "v_edge_feature_mode", None),
        use_v_edge_features=_get_arg(args, "use_v_edge_features", None),
        mask_possessor_v_edge_features=_get_arg(args, "mask_possessor_v_edge_features", None),
        add_v_edge_features=_get_arg(args, "add_v_edge_features", None),
        edge_in_dim=_get_arg(args, "edge_in_dim", None),
    )
    relative_speed_edge_feature_mode = normalize_relative_speed_edge_feature_mode(
        _get_arg(args, "relative_speed_edge_feature_mode", None),
        use_relative_speed_edge_features=_get_arg(args, "use_relative_speed_edge_features", None),
        mask_possessor_relative_speed_edge_features=_get_arg(
            args, "mask_possessor_relative_speed_edge_features", None
        ),
        add_relative_speed_edge_features=_get_arg(args, "add_relative_speed_edge_features", None),
        edge_in_dim=_get_arg(args, "edge_in_dim", None),
    )
    validate_relative_speed_edge_feature_mode(v_edge_feature_mode, relative_speed_edge_feature_mode)
    lane_survival = bool(_get_arg(args, "lane_survival", False))
    lane_survival_mode = (
        normalize_pc_xpass_lane_survival_mode(_get_arg(args, "lane_survival_mode", None))
        if lane_survival
        else None
    )
    require_diagnostics = (
        requires_goal_next10_diagnostics(task)
        if require_goal_next10_diagnostics is None
        else bool(require_goal_next10_diagnostics)
    )

    return {
        "task": task,
        "inplay_only": task.split("_")[1] == "receiver" and not bool(_get_arg(args, "include_out", False)),
        "min_pass_dur": float(_get_arg(args, "min_pass_dur", 0.0)),
        "shot_success_type": str(_get_arg(args, "shot_success", "unblocked")),
        "xy_only": _bool_arg(args, "xy_only", False),
        "possessor_aware": _bool_arg(args, "possessor_aware", False),
        "keeper_aware": _bool_arg(args, "keeper_aware", False),
        "ball_z_aware": _bool_arg(args, "ball_z_aware", False),
        "poss_vel_aware": _bool_arg(args, "poss_vel_aware", False),
        "poss_rel_vel_aware": _bool_arg(args, "poss_rel_vel_aware", False),
        "poss_geometry_aware": _bool_arg(args, "poss_geometry_aware", True),
        "goal_features_aware": _bool_arg(args, "goal_features_aware", True),
        "goal_nodes_aware": _bool_arg(args, "goal_nodes_aware", True),
        "accel_aware": _bool_arg(args, "accel_aware", True),
        "offside_aware": _bool_arg(args, "offside_aware", True),
        "extend_features": _bool_arg(args, "extend_features", False),
        "drop_non_blockers": _bool_arg(args, "filter_blockers", False),
        "sparsify": _get_arg(args, "sparsify", "none"),
        "max_edge_dist": float(_get_arg(args, "max_edge_dist", 10.0)),
        "edge_in_dim": int(_get_arg(args, "edge_in_dim", 2)),
        "v_edge_feature_mode": v_edge_feature_mode,
        "relative_speed_edge_feature_mode": relative_speed_edge_feature_mode,
        "mask_possessor_v_edge_features": mask_possessor_v_edge_features_for_mode(v_edge_feature_mode),
        "mask_possessor_relative_speed_edge_features": mask_possessor_relative_speed_edge_features_for_mode(
            relative_speed_edge_feature_mode
        ),
        "train": bool(train),
        "diagnostic_label_dir": diagnostic_label_dir,
        "pass_height_diagnostic_label_dir": pass_height_diagnostic_label_dir,
        "require_goal_next10_diagnostics": require_diagnostics,
        "use_physical_xpass": model_uses_physical_xpass(args),
        "physical_cache_dir": physical_cache_dir,
        "physical_eps": float(_get_arg(args, "physical_eps", 1e-4)),
        "physical_xpass_floor": _get_arg(args, "physical_xpass_floor", None),
        "require_observed_pass_height": _bool_arg(args, "require_observed_pass_height", False),
        "require_pass_height_labels": _bool_arg(args, "require_pass_height_labels", False),
        "pass_height_cache_dir": None,
        "evaluation_xpass_cache_dir": None,
        "evaluation_xpass_metric": None,
        "evaluation_xpass_require_nearest": False,
        "evaluation_xpass_require_height": False,
        "lane_survival": lane_survival,
        "lane_survival_mode": lane_survival_mode,
        "lane_survival_cache_dir": lane_survival_cache_dir,
    }


_IPW_CHECKPOINT_FEATURE_KEYS = (
    "xy_only",
    "possessor_aware",
    "keeper_aware",
    "ball_z_aware",
    "poss_vel_aware",
    "poss_rel_vel_aware",
    "poss_geometry_aware",
    "goal_features_aware",
    "goal_nodes_aware",
    "accel_aware",
    "offside_aware",
    "extend_features",
    "drop_non_blockers",
    "sparsify",
    "max_edge_dist",
    "edge_in_dim",
    "v_edge_feature_mode",
    "relative_speed_edge_feature_mode",
    "mask_possessor_v_edge_features",
    "mask_possessor_relative_speed_edge_features",
    "lane_survival",
    "lane_survival_mode",
    "lane_survival_cache_dir",
)


def build_ipw_dataset_kwargs(
    target_dataset_kwargs: dict[str, Any],
    ipw_model_args: dict[str, Any],
    ipw_model_metadata: dict[str, Any] | None,
    *,
    diagnostic_label_dir: str | None,
    require_goal_next10_diagnostics: bool,
) -> dict[str, Any]:
    """Configure an IPW dataset for its checkpoint without changing target eligibility.

    The target model determines the task, labels, and row eligibility.  Feature
    transforms are reconstructed from the IPW checkpoint because it may require
    lane-survival even when the target model does not.
    """
    checkpoint_args = dict(ipw_model_args)
    checkpoint_args["task"] = target_dataset_kwargs["task"]
    lane_metadata = dict((ipw_model_metadata or {}).get("lane_survival") or {})
    checkpoint_args["lane_survival"] = bool(
        lane_metadata.get("enabled", checkpoint_args.get("lane_survival", False))
    )
    if checkpoint_args["lane_survival"]:
        checkpoint_args["lane_survival_mode"] = (
            lane_metadata.get("mode") or checkpoint_args.get("lane_survival_mode")
        )
        cache_dir = lane_metadata.get("cache_dir") or checkpoint_args.get("lane_survival_cache_dir")
        if not cache_dir:
            raise ValueError("IPW checkpoint requires lane_survival but does not record a pc-xPass cache directory.")
        cache_metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
        checkpoint_args["lane_survival_mode"] = validate_pc_xpass_lane_survival_mode_cache_metadata(
            cache_metadata,
            checkpoint_args["lane_survival_mode"],
        )
        expected_fingerprint = lane_metadata.get("cache_fingerprint") or checkpoint_args.get(
            "lane_survival_cache_fingerprint"
        )
        if expected_fingerprint and str(expected_fingerprint) != pc_xpass_lane_survival_metadata_fingerprint(cache_metadata):
            raise ValueError("IPW checkpoint lane-survival pc-xPass cache fingerprint does not match its training cache.")
    else:
        cache_dir = None

    checkpoint_kwargs = build_action_dataset_kwargs(
        checkpoint_args,
        train=bool(target_dataset_kwargs["train"]),
        diagnostic_label_dir=diagnostic_label_dir,
        require_goal_next10_diagnostics=require_goal_next10_diagnostics,
        physical_cache_dir=None,
        lane_survival_cache_dir=cache_dir,
    )
    result = dict(target_dataset_kwargs)
    for key in _IPW_CHECKPOINT_FEATURE_KEYS:
        result[key] = checkpoint_kwargs[key]
    result["use_physical_xpass"] = False
    result["physical_cache_dir"] = None
    return result
