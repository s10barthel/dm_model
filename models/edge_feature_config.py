"""Canonical configuration for velocity-based graph edge features.

This module is intentionally small because its behavior contributes to the
prepared intent-dataset cache identity. Training-loop changes must not invalidate
prepared graphs.
"""
from __future__ import annotations

from typing import Any


V_EDGE_FEATURE_MODE_ALL = "all"
V_EDGE_FEATURE_MODE_NONE = "none"
V_EDGE_FEATURE_MODE_NO_POSS = "no_poss"
V_EDGE_FEATURE_MODES = (
    V_EDGE_FEATURE_MODE_ALL,
    V_EDGE_FEATURE_MODE_NONE,
    V_EDGE_FEATURE_MODE_NO_POSS,
)
RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL = "all"
RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE = "none"
RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS = "no_poss"
RELATIVE_SPEED_EDGE_FEATURE_MODES = (
    RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL,
    RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE,
    RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS,
)


def normalize_v_edge_feature_mode(
    v_edge_feature_mode: str | None = None,
    *,
    use_v_edge_features: bool | None = None,
    mask_possessor_v_edge_features: bool | None = None,
    add_v_edge_features: bool | None = None,
    edge_in_dim: int | None = None,
) -> str:
    if v_edge_feature_mode is not None:
        mode = str(v_edge_feature_mode).strip().replace("-", "_")
        if mode in V_EDGE_FEATURE_MODES:
            return mode
        raise ValueError(
            f"Invalid v_edge_feature_mode={v_edge_feature_mode!r}. "
            f"Expected one of: {', '.join(V_EDGE_FEATURE_MODES)}."
        )
    if bool(mask_possessor_v_edge_features):
        return V_EDGE_FEATURE_MODE_NO_POSS
    if use_v_edge_features is not None:
        return V_EDGE_FEATURE_MODE_ALL if bool(use_v_edge_features) else V_EDGE_FEATURE_MODE_NONE
    if add_v_edge_features is not None:
        return V_EDGE_FEATURE_MODE_ALL if bool(add_v_edge_features) else V_EDGE_FEATURE_MODE_NONE
    if edge_in_dim is not None:
        return V_EDGE_FEATURE_MODE_ALL if int(edge_in_dim) > 2 else V_EDGE_FEATURE_MODE_NONE
    return V_EDGE_FEATURE_MODE_NONE


def normalize_relative_speed_edge_feature_mode(
    relative_speed_edge_feature_mode: str | None = None,
    *,
    use_relative_speed_edge_features: bool | None = None,
    mask_possessor_relative_speed_edge_features: bool | None = None,
    add_relative_speed_edge_features: bool | None = None,
    edge_in_dim: int | None = None,
) -> str:
    if relative_speed_edge_feature_mode is not None:
        mode = str(relative_speed_edge_feature_mode).strip().replace("-", "_")
        if mode in RELATIVE_SPEED_EDGE_FEATURE_MODES:
            return mode
        raise ValueError(
            f"Invalid relative_speed_edge_feature_mode={relative_speed_edge_feature_mode!r}. "
            f"Expected one of: {', '.join(RELATIVE_SPEED_EDGE_FEATURE_MODES)}."
        )
    if bool(mask_possessor_relative_speed_edge_features):
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS
    if use_relative_speed_edge_features is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if bool(use_relative_speed_edge_features) else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    if add_relative_speed_edge_features is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if bool(add_relative_speed_edge_features) else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    if edge_in_dim is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if int(edge_in_dim) > 4 else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    return RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE


def use_v_edge_features_for_mode(v_edge_feature_mode: str | None) -> bool:
    return normalize_v_edge_feature_mode(v_edge_feature_mode) != V_EDGE_FEATURE_MODE_NONE


def mask_possessor_v_edge_features_for_mode(v_edge_feature_mode: str | None) -> bool:
    return normalize_v_edge_feature_mode(v_edge_feature_mode) == V_EDGE_FEATURE_MODE_NO_POSS


def use_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode: str | None) -> bool:
    return normalize_relative_speed_edge_feature_mode(relative_speed_edge_feature_mode) != RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE


def mask_possessor_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode: str | None) -> bool:
    return normalize_relative_speed_edge_feature_mode(relative_speed_edge_feature_mode) == RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS


def validate_relative_speed_edge_feature_mode(
    v_edge_feature_mode: str | None,
    relative_speed_edge_feature_mode: str | None,
) -> None:
    if use_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode) and not use_v_edge_features_for_mode(v_edge_feature_mode):
        raise ValueError("Relative-speed edge features require velocity-angle edge features.")


def normalize_v_edge_feature_args(args: dict[str, Any]) -> dict[str, Any]:
    mode = normalize_v_edge_feature_mode(
        args.get("v_edge_feature_mode"),
        use_v_edge_features=args.get("use_v_edge_features"),
        mask_possessor_v_edge_features=args.get("mask_possessor_v_edge_features"),
        add_v_edge_features=args.get("add_v_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    relative_speed_mode = normalize_relative_speed_edge_feature_mode(
        args.get("relative_speed_edge_feature_mode"),
        use_relative_speed_edge_features=args.get("use_relative_speed_edge_features"),
        mask_possessor_relative_speed_edge_features=args.get("mask_possessor_relative_speed_edge_features"),
        add_relative_speed_edge_features=args.get("add_relative_speed_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    validate_relative_speed_edge_feature_mode(mode, relative_speed_mode)
    args["v_edge_feature_mode"] = mode
    args["use_v_edge_features"] = use_v_edge_features_for_mode(mode)
    args["mask_possessor_v_edge_features"] = mask_possessor_v_edge_features_for_mode(mode)
    args["relative_speed_edge_feature_mode"] = relative_speed_mode
    args["use_relative_speed_edge_features"] = use_relative_speed_edge_features_for_mode(relative_speed_mode)
    args["mask_possessor_relative_speed_edge_features"] = mask_possessor_relative_speed_edge_features_for_mode(relative_speed_mode)
    return args
