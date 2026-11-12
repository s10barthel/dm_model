"""Independent component predictions on prepared dataset states."""
import pandas as pd

from inference import PhysicalXPassNoUsableRowsError, inference_xpass_v0


def load_component_models(model_ids, device):
    from models.utils import load_model
    models = {task: load_model(identifier, device) for task, identifier in model_ids.items() if identifier}
    missing = [task for task, model in models.items() if model is None]
    if missing:
        raise FileNotFoundError("Missing model checkpoints for: " + ", ".join(missing))
    return models


def infer_components(state, models, device, *, predictor, preflight, xpass_runtime=None):
    if state.labels.numel() == 0 or not state.graph_features_0:
        return {}
    runtime = models.get("pass_success", xpass_runtime)
    inputs = {"graph_override": state.graph_features_0, "label_override": state.labels}
    preflight(state, runtime, **inputs)
    components, auxiliaries = {}, {}
    for task in ("action_intent", "pass_intent", "pass_height"):
        components[task] = pd.DataFrame()
        if task in models:
            kwargs = {"precomputed_auxiliaries": auxiliaries} if task != "action_intent" else {}
            components[task], _ = predictor(state, models[task], device=device, post_action=False, **inputs, **kwargs)
    components["pass_success"], components["model_pass_success"] = pd.DataFrame(), pd.DataFrame()
    try:
        if "pass_success" in models:
            components["pass_success"], components["model_pass_success"] = predictor(
                state, models["pass_success"], device=device, post_action=False, **inputs,
                return_model_pass_success=True, pass_intent_probs=components["pass_intent"],
                precomputed_auxiliaries=auxiliaries)
        elif runtime is not None:
            components["pass_success"] = inference_xpass_v0(state, runtime,
                graphs=state.graph_features_0, labels=state.labels)
    except PhysicalXPassNoUsableRowsError:
        pass
    for task in ("outcome_scoring", "outcome_conceding"):
        failure, success = pd.DataFrame(), pd.DataFrame()
        if task in models:
            failure, success = predictor(state, models[task], device=device, post_action=False, **inputs)
        components[task + "_failure"], components[task + "_success"] = failure, success
    return components


def blank_state_predictions(state):
    """Retain state/candidate identities when a component is intentionally absent."""
    from datatools.config import LABEL_INDEX
    players = list(dict.fromkeys(str(player) for graph in state.graph_features_0 for player in graph.node_ids))
    indices = state.labels[:, LABEL_INDEX["action_index"]].int().tolist()
    return pd.DataFrame(float("nan"), index=indices, columns=players)
