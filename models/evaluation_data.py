"""Resolve and preflight evaluation artifacts independently of training provenance."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

from datatools import config
from dataset import requires_goal_next10_diagnostics
from models.edge_schema import edge_names, schema_for_names, INTERACTION_GROUPS
from models.utils import (infer_feature_graph_schema, infer_training_edge_schema, validate_feature_graph_schema,
                          get_model_feature_graph_schema)
from models.utils import get_model_record
from project_config import (DEFAULT_INTENDED_RECEIVER_MODE, get_action_label_dir, get_success_intent_label_dir,
                            load_feature_run_metadata, resolve_feature_root, resolve_feature_run_id)


def validate_checkpoint_edge_schema(feature_schema, model_args):
    settings = vars(model_args) if not isinstance(model_args, dict) else model_args
    model_schema = schema_for_names(edge_names(settings), explicit=settings.get("edge_feature_names") is not None)
    model_schema["node_in_dim"] = get_model_feature_graph_schema(SimpleNamespace(args=settings))["node_in_dim"]
    reconstructed = infer_training_edge_schema(
        feature_schema,
        v_edge_feature_mode=settings.get("v_edge_feature_mode"),
        relative_speed_edge_feature_mode=settings.get("relative_speed_edge_feature_mode"),
        **{f"{group}_edge_feature_mode": settings.get(f"{group}_edge_feature_mode", "none")
           for group in INTERACTION_GROUPS},
    )
    if edge_names(reconstructed) != edge_names(model_schema):
        raise ValueError(
            "Checkpoint edge-feature settings do not reconstruct its recorded graph schema: "
            f"reconstructed={reconstructed}, checkpoint={model_schema}."
        )
    validate_feature_graph_schema(feature_schema, model_schema, context="Selected feature artifacts")
    return model_schema


def resolve_goal_next10_diagnostic_context(
    cli_args: argparse.Namespace,
    model_args: argparse.Namespace,
    feature_root: Path,
) -> tuple[str | None, str | None]:
    if not requires_goal_next10_diagnostics(getattr(model_args, "task", None)):
        return None, None
    if getattr(cli_args, "diagnostic_return_type", None) is not None:
        return (cli_args.diagnostic_feature_run_id or getattr(cli_args, "feature_run_id", None)
                or getattr(model_args, "feature_run_id", None)), None

    mode = getattr(model_args, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE
    if mode == "unknown":
        mode = DEFAULT_INTENDED_RECEIVER_MODE

    if cli_args.diagnostic_feature_run_id or getattr(cli_args, "feature_run_id", None):
        diagnostic_feature_run_id = resolve_feature_run_id(
            cli_args.diagnostic_feature_run_id or cli_args.feature_run_id,
            required=True,
            allow_latest=False,
        )
        diagnostic_root = resolve_feature_root(diagnostic_feature_run_id)
        diagnostic_label_dir = get_action_label_dir(
            config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
            intended_receiver_mode=mode,
            root=diagnostic_root,
            use_carries=bool(getattr(model_args, "use_carries", False)),
        )
        if not diagnostic_label_dir.exists():
            raise FileNotFoundError(f"Canonical goal-next10 diagnostic labels not found at {diagnostic_label_dir}.")
        return diagnostic_feature_run_id, str(diagnostic_label_dir)

    model_diagnostic_label_dir = getattr(model_args, "diagnostic_label_dir", None)
    if model_diagnostic_label_dir and Path(model_diagnostic_label_dir).exists():
        return getattr(model_args, "diagnostic_feature_run_id", None), str(model_diagnostic_label_dir)

    selected_label_dir = get_action_label_dir(
        config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
        intended_receiver_mode=mode,
        root=feature_root,
    )
    if selected_label_dir.exists():
        return getattr(model_args, "feature_run_id", None), str(selected_label_dir)
    return getattr(model_args, "feature_run_id", None), None


def validate_diagnostic_feature_run_ancestry(
    diagnostic_feature_run_id: str,
    selected_feature_run_id: str,
) -> list[str]:
    """Return the diagnostic lineage after proving it reaches the selected feature run."""
    current = str(diagnostic_feature_run_id)
    expected = str(selected_feature_run_id)
    lineage: list[str] = []
    seen: set[str] = set()
    while current:
        if current in seen:
            raise ValueError(f"Feature-run ancestry contains a cycle at {current!r}.")
        seen.add(current)
        lineage.append(current)
        if current == expected:
            return lineage
        metadata = load_feature_run_metadata(current, required=True) or {}
        current = str(metadata.get("derived_from_feature_run_id") or "")
    raise ValueError(
        f"Diagnostic feature run {diagnostic_feature_run_id!r} is not equal to or derived from "
        f"selected input feature run {selected_feature_run_id!r}."
    )


def resolve_pass_height_diagnostic_context(
    cli_args: argparse.Namespace,
    model_args: argparse.Namespace,
    selected_feature_run_id: str | None,
    *,
    required: bool,
) -> tuple[str | None, str | None, float | None, list[str] | None]:
    task = str(getattr(model_args, "task", ""))
    diagnostic_id = cli_args.diagnostic_feature_run_id or (getattr(cli_args, "feature_run_id", None) if required else None)
    if not diagnostic_id or task not in {"pass_success", "pass_height"}:
        return None, None, None, None
    if not required:
        raise ValueError(
            "--diagnostic-feature-run-id is unused for pass_success when no pass-height evaluation is enabled."
        )
    if not selected_feature_run_id:
        raise ValueError("Pass-height diagnostic evaluation requires a checkpoint feature_run_id.")

    diagnostic_run_id = resolve_feature_run_id(
        diagnostic_id,
        required=True,
        allow_latest=False,
    )
    lineage = validate_diagnostic_feature_run_ancestry(diagnostic_run_id, str(selected_feature_run_id))
    metadata = load_feature_run_metadata(diagnostic_run_id, required=True) or {}
    threshold = metadata.get("pass_height_threshold_meters")
    if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError(
            f"Diagnostic feature run {diagnostic_run_id!r} must record a positive finite "
            "pass_height_threshold_meters value."
        )

    mode = getattr(model_args, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE
    if mode == "unknown":
        mode = DEFAULT_INTENDED_RECEIVER_MODE
    return_type = getattr(model_args, "return_type", None)
    if not return_type:
        raise ValueError(f"{task} checkpoint does not record return_type for diagnostic-label resolution.")
    diagnostic_root = resolve_feature_root(diagnostic_run_id)
    label_dir = get_action_label_dir(str(return_type), intended_receiver_mode=mode, root=diagnostic_root,
                                   use_carries=bool(getattr(model_args, "use_carries", False)))
    if not label_dir.exists():
        raise FileNotFoundError(f"Pass-height diagnostic labels not found at {label_dir}.")
    return diagnostic_run_id, str(label_dir), float(threshold), lineage


def _settings(source):
    return dict(source) if isinstance(source, dict) else vars(source).copy()


def resolve_evaluation_data(cli_args, model_args):
    """Resolve artifact locations without changing the checkpoint's prediction definition."""
    settings = _settings(model_args)
    model = SimpleNamespace(**settings)
    cli = SimpleNamespace(**{
        "feature_run_id": None, "feature_dir": None, "diagnostic_feature_run_id": None,
        "diagnostic_return_type": None, "intended_receiver_mode": None,
        "observed_pass_height_stratification": False, "evaluate_xpass": False,
        "evaluate_combined_success": False, "weighted_pass_success_metrics": False,
        **_settings(cli_args),
    })
    if cli.intended_receiver_mode is not None:
        model.intended_receiver_mode = cli.intended_receiver_mode
    training_id = getattr(model, "feature_run_id", None)
    selected_id = cli.feature_run_id or training_id
    if selected_id:
        root = resolve_feature_root(selected_id)
        graph_name = Path(getattr(model, "feature_dir", "data/features/action_graphs")).name
        feature_dir = cli.feature_dir or str(root / graph_name)
        label_name = Path(getattr(model, "label_dir", f"data/features/action_labels_{getattr(model, 'return_type', 'disc_0.9')}")).name
        label_dir = root / label_name
    else:
        feature_dir = cli.feature_dir or getattr(model, "feature_dir", "data/features/action_graphs")
        root = Path(feature_dir).parent
        label_dir = Path(getattr(model, "label_dir", f"data/features/action_labels_{getattr(model, 'return_type', 'disc_0.9')}"))
    mode = getattr(model, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE
    mode = DEFAULT_INTENDED_RECEIVER_MODE if mode == "unknown" else mode
    carries = bool(getattr(model, "use_carries", False))
    task = str(getattr(model, "task", ""))
    if cli.feature_run_id or cli.intended_receiver_mode is not None:
        if task == "success_intent":
            label_dir = get_success_intent_label_dir(root)
        else:
            if not getattr(model, "return_type", None):
                raise ValueError(f"{task} checkpoint does not record return_type for evaluation-label resolution.")
            label_dir = get_action_label_dir(model.return_type, intended_receiver_mode=mode, root=root, use_carries=carries)
    goal_id, goal_dir = resolve_goal_next10_diagnostic_context(cli, model, root)
    height_required = bool(
        task == "pass_height" and (cli.feature_run_id or cli.diagnostic_feature_run_id)
        or task == "pass_success" and (cli.observed_pass_height_stratification or cli.weighted_pass_success_metrics
                                       or cli.evaluate_xpass or cli.evaluate_combined_success)
    )
    height_id, height_dir, threshold, lineage = resolve_pass_height_diagnostic_context(
        cli, model, selected_id, required=height_required)
    evaluation_goal_dir = None
    if task in {"outcome_scoring", "outcome_conceding"} and cli.diagnostic_return_type is not None:
        diagnostic_root = resolve_feature_root(cli.diagnostic_feature_run_id) if cli.diagnostic_feature_run_id else root
        evaluation_goal_dir = str(get_action_label_dir(cli.diagnostic_return_type, intended_receiver_mode=mode,
                                                       root=diagnostic_root, use_carries=carries))
    return {
        "requested_feature_run_id": cli.feature_run_id, "training_feature_run_id": training_id,
        "feature_run_id": selected_id, "feature_root": str(root), "feature_dir": str(feature_dir),
        "label_dir": str(label_dir), "diagnostic_feature_run_id": goal_id,
        "diagnostic_label_dir": goal_dir, "evaluation_diagnostic_label_dir": evaluation_goal_dir,
        "pass_height_diagnostic_feature_run_id": height_id, "pass_height_diagnostic_label_dir": height_dir,
        "observed_pass_height_threshold_meters": threshold, "pass_height_diagnostic_lineage": lineage,
        "label_definitions": {"task": task, "return_type": getattr(model, "return_type", None),
                              "intended_receiver_mode": mode, "use_carries": carries,
                              "diagnostic_return_type": cli.diagnostic_return_type or "next_10"},
    }


class EvaluationDataError(ValueError):
    """Consolidated artifact failures discovered before any checkpoint is evaluated."""


def preflight_evaluation_data(requests):
    """Check each primary cohort and its auxiliary input requirements without inference."""
    results, errors, schemas = {}, [], {}
    for request in requests:
        identifier = request["model_id"]
        try:
            settings = request.get("model_args")
            if settings is None:
                settings = get_model_record(identifier)["args"]
            context = resolve_evaluation_data(request["cli_args"], settings)
        except (OSError, ValueError) as exc:
            errors.append(f"{identifier}: {exc}")
            continue
        matches = list(dict.fromkeys(map(str, request["match_ids"])))
        for key in ("feature_dir", "label_dir", "diagnostic_label_dir", "evaluation_diagnostic_label_dir",
                    "pass_height_diagnostic_label_dir"):
            if context[key] is None:
                continue
            directory = Path(context[key])
            missing = [match for match in matches if not (directory / f"{match}.pt").is_file()]
            if missing:
                shown = ", ".join(missing[:10]) + (f" (and {len(missing) - 10} more)" if len(missing) > 10 else "")
                errors.append(f"{identifier}: missing {key} files under {directory}; matches: {shown}")
            elif not directory.is_dir():
                errors.append(f"{identifier}: missing {key} directory {directory}")
        try:
            graph_dir = context["feature_dir"]
            if graph_dir not in schemas:
                schemas[graph_dir] = infer_feature_graph_schema(graph_dir)
            schema = schemas[graph_dir]
            context["feature_schema"] = schema
            context["model_schema"] = validate_checkpoint_edge_schema(schema, settings)
            for auxiliary in request.get("auxiliaries", []):
                validate_checkpoint_edge_schema(schema, auxiliary["args"])
        except (OSError, ValueError) as exc:
            errors.append(f"{identifier}: {exc}")
        context["requested_match_ids"] = matches
        results[identifier] = context
    if errors:
        raise EvaluationDataError("Evaluation data preflight failed before starting any model:\n- " + "\n- ".join(errors))
    return results
