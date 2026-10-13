"""Explicit learned-model dependencies of PC-xPass consumers, never of physics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _prediction_settings(model):
    from models.inference_masks import inference_mask_metadata
    return json.dumps([model.args, inference_mask_metadata(model)], sort_keys=True, default=str)


@dataclass(frozen=True)
class AuxiliaryPredictionRecord:
    """A successful component prediction on shared, immutable collector inputs."""

    task: str
    model_id: str
    probabilities: Any
    match: Any
    graphs: Any
    labels: Any
    feature_action_indices: tuple[int, ...] | None
    post_action: bool
    device: str
    settings: str


def record_component_prediction(records, match, model, probabilities, device, *,
                                post_action=False, graph_override=None, label_override=None,
                                feature_action_indices=None):
    """Record validated output; collectors keep these source inputs immutable until reuse."""
    if records is None or model.args.get("task") not in {"pass_intent", "pass_height"}:
        return
    model_id = model.args.get("model_id")
    if not model_id:
        return
    if graph_override is None:
        from inference import resolve_match_graphs
        graphs, feature_action_indices = resolve_match_graphs(match, model, post_action)
        labels = match.labels
    else:
        graphs, labels = graph_override, label_override
    task = model.args["task"]
    records[task] = AuxiliaryPredictionRecord(
        task, str(model_id), probabilities, match, graphs, labels,
        None if feature_action_indices is None else tuple(int(i) for i in feature_action_indices),
        bool(post_action), str(device), _prediction_settings(model),
    )


def _reused_prediction(record, match, auxiliary, device, *, graphs, labels,
                       feature_action_indices, event_indices, post_action):
    if not isinstance(record, AuxiliaryPredictionRecord):
        return None
    indices = None if feature_action_indices is None else tuple(int(i) for i in feature_action_indices)
    if (record.task != auxiliary.args.get("task")
            or record.model_id != auxiliary.args.get("model_id")
            or record.match is not match or record.graphs is not graphs or record.labels is not labels
            or record.feature_action_indices != indices or record.post_action != bool(post_action)
            or record.device != str(device) or record.settings != _prediction_settings(auxiliary)):
        return None
    probabilities = record.probabilities
    requested = [int(i) for i in (labels[:, 0].tolist() if event_indices is None else event_indices)]
    if (not probabilities.index.is_unique or not probabilities.columns.is_unique
            or not all(i in probabilities.index for i in requested)):
        return None
    return probabilities.loc[requested].copy()


def add_auxiliary_arguments(parser):
    parser.add_argument("--xpass-pass-height-model-id", help="Explicit pass_height/<id> auxiliary for PC-xPass v4/v5.")
    parser.add_argument("--xpass-pass-intent-model-id", help="Explicit pass_intent/<id> auxiliary for PC-xPass v5.")


def validate_auxiliary_selectors(args, *, active=None, version=None, weighted=None):
    if weighted is None:
        weighted = bool(getattr(args, "weighted_pass_success_metrics", False))
    if active is None:
        active = bool((getattr(args, "pc_xpass", False) and getattr(args, "use_physical_xpass", False))
                      or getattr(args, "evaluate_combined_success", False))
    version = version or getattr(args, "xpass_weight", "v3")
    required = {"pass_height": bool(weighted or (active and version in {"v4", "v5"})),
                "pass_intent": bool(active and version == "v5")}
    selected = {}
    for task, needed in required.items():
        name = f"xpass_{task}_model_id"
        value = getattr(args, name, None)
        flag = "--" + name.replace("_", "-")
        if needed and not value:
            raise ValueError(f"{flag} is required explicitly for this PC-xPass operation; component models/bundles do not select auxiliaries.")
        if value and not needed:
            raise ValueError(f"{flag} is unused for this operation.")
        if value:
            if not isinstance(value, str) or not value.startswith(task + "/") or not value.split("/", 1)[1]:
                raise ValueError(f"{flag} must reference a {task}/<id> checkpoint.")
            selected[task] = value
    return selected


def configure_auxiliary_models(args, models, device, *, active=None):
    """Add separately loaded roles to the schema/mask dependency set."""
    from models.utils import load_model, get_model_provenance
    from models.pass_height import check_height_probability, model_definition
    if active is None and "pass_success" not in models:
        active = False
    selected = validate_auxiliary_selectors(args, active=active)
    auxiliaries, records = {}, {}
    for task, model_id in selected.items():
        model = load_model(model_id, device)
        if model is None:
            raise FileNotFoundError(f"Missing PC-xPass {task} auxiliary checkpoint: {model_id}")
        if model.args.get("task") != task:
            raise ValueError(f"PC-xPass {task} auxiliary has task {model.args.get('task')!r}: {model_id}")
        auxiliaries[task] = model
        models[f"xpass_{task}"] = model
        records[task] = {"role": f"xpass_{task}", "model_id": model_id,
                         "model_record": get_model_provenance(model_id),
                         "input_alignment": "physical_state_and_candidate_player",
                         "input_graph": "original_unmasked_state"}
        if task == "pass_height":
            height_definition = model_definition(model.args)
            check_height_probability(height_definition, getattr(args, "pass_height_threshold", None))
            records[task]["pass_height_definition"] = height_definition
    primary = models.get("pass_success")
    if primary is not None and auxiliaries:
        primary._xpass_auxiliary_models = auxiliaries
        primary.args["xpass_auxiliary_models"] = records
        if "pass_height" in records:
            primary.args["xpass_pass_height_definition"] = records["pass_height"]["pass_height_definition"]
    args._xpass_auxiliary_records = records
    return auxiliaries


def auxiliary_metadata(models):
    from models.inference_masks import inference_mask_metadata
    primary = models.get("pass_success")
    records = getattr(primary, "args", {}).get("xpass_auxiliary_models", {})
    return {task: {**record, "effective_model_args": dict(models[f"xpass_{task}"].args),
                   "inference_masks": inference_mask_metadata(models[f"xpass_{task}"])}
            for task, record in records.items()}



def original_input_fingerprint(graphs, labels):
    """Hash original graph inputs, including edges and candidate identities."""
    digest = hashlib.sha256()
    for graph in graphs:
        if graph is None:
            digest.update(b"missing")
            continue
        for name, value in sorted(graph.to_dict().items()):
            digest.update(name.encode())
            if hasattr(value, "detach"):
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_auxiliaries(match, model, device, *, graphs, labels, event_indices=None, post_action=False,
                        feature_action_indices=None, precomputed_auxiliaries=None):
    """Reuse only identical original inputs and effective model preparations within a match."""
    from inference import inference_gnn
    from models.inference_masks import inference_mask_metadata
    predictions = {}
    cache = match.__dict__.setdefault("_xpass_auxiliary_predictions", {})
    for task, auxiliary in getattr(model, "_xpass_auxiliary_models", {}).items():
        reused = _reused_prediction(
            (precomputed_auxiliaries or {}).get(task), match, auxiliary, device,
            graphs=graphs, labels=labels, feature_action_indices=feature_action_indices,
            event_indices=event_indices, post_action=post_action,
        )
        if reused is not None:
            predictions[task] = reused
            continue
        digest = hashlib.sha256()
        digest.update(json.dumps([auxiliary.args, inference_mask_metadata(auxiliary),
                                 list(event_indices) if event_indices is not None else None,
                                 post_action, feature_action_indices], sort_keys=True, default=str).encode())
        digest.update(original_input_fingerprint(graphs, labels).encode())
        key = (task, id(auxiliary), digest.hexdigest())
        if key not in cache:
            if feature_action_indices is not None:
                lookup = {int(index): graph for index, graph in zip(feature_action_indices, graphs)}
                source = [lookup.get(int(label[0])) for label in labels]
                source_labels = labels
            else:
                source, source_labels = graphs, labels
            cache[key] = inference_gnn(match, auxiliary, device=device, post_action=post_action,
                                       event_indices=event_indices, graph_override=source,
                                       label_override=source_labels, auxiliary_prediction=True)[0]
        predictions[task] = cache[key]
    return predictions


def observed_auxiliary_predictions(batch, labels, models, device):
    """Align evaluation targets through immutable original node positions."""
    import torch
    from torch_geometric.data import Batch
    from datatools import config
    from models.goal_context import candidate_mask
    from models.utils import adapt_batch_graphs_for_model
    results = {}
    for role, model in models.items():
        source = getattr(batch, f"xpass_{role}_graph", None)
        if source is None:
            raise ValueError(f"PC-xPass auxiliary {role} requires separately prepared evaluation inputs.")
        auxiliary = adapt_batch_graphs_for_model(Batch.from_data_list(source).to(device), model.args,
                                                context=f"PC-xPass auxiliary {role}")
        model.eval()
        with torch.no_grad():
            logits = model(auxiliary).reshape(-1)
        values = []
        for i in range(batch.num_graphs):
            if not bool(labels[i, config.LABEL_INDEX["is_pass"]]):
                continue
            target = int(labels[i, config.LABEL_INDEX["intent_index"]])
            primary_position = int(batch.ptr[i]) + target
            original_position = batch.source_node_positions[primary_position]
            graph_mask = auxiliary.batch == i
            eligible = candidate_mask(auxiliary, model.args)[graph_mask]
            if role == "pass_intent":
                possessor = getattr(auxiliary, "inference_possessor_mask",
                                    auxiliary.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1)
                eligible = eligible & ~possessor[graph_mask].bool()
            positions = auxiliary.source_node_positions[graph_mask]
            matched = torch.where(eligible & (positions == original_position))[0]
            if matched.numel() != 1:
                raise ValueError(f"PC-xPass auxiliary {role} cannot align target in match {batch.evaluation_match_id[i]}, action {int(labels[i, 0])}.")
            graph_logits = logits[graph_mask]
            if role == "pass_height":
                probability = torch.sigmoid(graph_logits[matched.item()])
            else:
                candidate_probs = torch.softmax(graph_logits[eligible], dim=0)
                target_position = torch.where(torch.where(eligible)[0] == matched.item())[0].item()
                probability = candidate_probs[target_position]
            if not bool(torch.isfinite(probability)):
                raise ValueError(f"Nonfinite PC-xPass auxiliary {role} prediction in match {batch.evaluation_match_id[i]}, action {int(labels[i, 0])}.")
            values.append(probability)
        results[role] = torch.stack(values) if values else logits.new_empty((0,))
    return results
