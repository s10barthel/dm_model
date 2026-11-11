"""Cross-checkpoint xT limitations, independent of CLI confirmation policy."""
from __future__ import annotations

import json


def warning_key(category, details):
    return json.dumps([category, details], sort_keys=True, separators=(",", ":"))


def _leaves(value, prefix=""):
    if isinstance(value, dict) and value:
        return {key: item for name, child in value.items()
                for key, item in _leaves(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def differences(values):
    flattened = {name: _leaves(value) for name, value in values.items()}
    keys = sorted(set().union(*(value.keys() for value in flattened.values())))
    return {key: {name: value.get(key, "<missing>") for name, value in flattened.items()}
            for key in keys if len({json.dumps(value.get(key, "<missing>"), sort_keys=True)
                                    for value in flattened.values()}) > 1}


def xt_consistency(records, warning_collector=None):
    definitions = {task: record.get("xt_target_definition") for task, record in records.items()}
    contracts = {task: record.get("outcome_learning_contract") for task, record in records.items()}
    identities = {task: record.get("model_id", task) for task, record in records.items()}
    target_diff, contract_diff = differences(definitions), differences(contracts)
    if contract_diff and any(value is None for value in contracts.values()):
        contract_diff = {"": contracts}
    if warning_collector is None:
        if target_diff:
            raise ValueError("Selected xT outcome checkpoints do not agree on surface and target definition.")
        if contract_diff:
            raise ValueError("Selected xT outcome checkpoints do not agree on outcome learning contract.")
    elif target_diff or contract_diff:
        fields = {**target_diff, **{("outcome_learning_contract" + (f".{key}" if key else "")): value
                                  for key, value in contract_diff.items()}}
        details = "; ".join(f"{identities[task]}: " + json.dumps({
            key: ("legacy bounded output contract" if key == "outcome_learning_contract" and values[task] is None
                  else values[task]) for key, values in fields.items()}, sort_keys=True)
            for task in records)
        same_surface = all(isinstance(value, dict) and value.get("xt_version") is not None
                           and value.get("xt_fit_metadata_sha256") is not None for value in definitions.values())
        same_surface = same_surface and not any(key in target_diff for key in ("xt_version", "xt_fit_metadata_sha256"))
        surface = " The xT surface and fit fingerprint agree." if same_surface else ""
        message = (f"Selected xT outcome checkpoints have different definitions: {details}.{surface} "
                   "Each checkpoint retains its own output interpretation; derived outcome and pass scores "
                   "combine these definitions without automatic conversion or calibration.")
        warning_collector.add(message, key=warning_key("mixed_xt_outcomes", {
            "model_ids": identities, "definitions": definitions, "contracts": contracts}))
    return dict(
        xt_target_definition=next(iter(definitions.values())) if definitions and not target_diff else None,
        outcome_learning_contract=next(iter(contracts.values())) if contracts and not contract_diff else None,
        source_xt_target_definitions=definitions,
        source_outcome_learning_contracts=contracts,
        source_outcome_model_ids=identities,
    )


def runtime_reference(shared, metadata):
    from datatools.xt_targets import training_target_definition
    definitions = shared.get("source_xt_target_definitions") or {}
    if not definitions and shared.get("xt_target_definition") is not None:
        definitions = {"outcome": shared["xt_target_definition"]}
    if not definitions or all(value is None for value in definitions.values()):
        return None
    actual = training_target_definition(metadata)
    return dict(xt_target_definition=actual,
                matching_tasks=[task for task, value in definitions.items() if value == actual],
                mismatching_tasks=[task for task, value in definitions.items() if value != actual],
                native_target_metrics_require_matching_reference=True)


def record_runtime_reference(shared, resolved, warning_collector=None):
    reference = resolved.get("xt_target_reference")
    if reference is None:
        return resolved
    shared["runtime_xt_reference"] = reference
    # A runtime definition already described by the selected checkpoints adds
    # no new limitation; retain per-task label matches without another prompt.
    if reference["mismatching_tasks"] and not reference["matching_tasks"]:
        if warning_collector is None:
            raise ValueError("Runtime feature run uses a different xT surface or target definition than the outcome checkpoints.")
        details = dict(feature_run_id=resolved["feature_run_id"],
                       model_ids=shared.get("source_outcome_model_ids", {}),
                       source_definitions=shared.get("source_xt_target_definitions", {}),
                       runtime_definition=reference["xt_target_definition"])
        warning_collector.add(
            f"Runtime feature run {resolved['feature_run_id']!r} uses xT labels different from every "
            f"selected outcome checkpoint: {json.dumps(details, sort_keys=True)}. "
            "Predictions retain each checkpoint's definition. These runtime labels cannot be used as "
            "either checkpoint's native-target evaluation reference.",
            key=warning_key("runtime_xt_reference", details))
    return resolved
