"""Optional occurrence x severity labels in existing outcome pitch plots."""
import math

from datatools import config

COMPONENTS = {f"outcome_{task}_{branch}" for task in ("scoring", "conceding")
              for branch in ("success", "failure")}


def validate_options(args):
    if not getattr(args, "decomposition", False):
        return
    if getattr(args, "coach_ratings", False) or getattr(args, "selections", False):
        raise ValueError("--decomposition cannot be combined with --coach-ratings or --selections.")
    if getattr(args, "only_tracking", False):
        raise ValueError("--decomposition requires individual outcome components.")
    if getattr(args, "mode", "standard") in {"loc", "freeze"}:
        raise ValueError("--decomposition currently requires standard frames with saved or live head predictions.")


def validate_components(args, names):
    validate_options(args)
    return [name for name in names if name in COMPONENTS]


def validate_models(args, names, models):
    from models.outcome_decomposition import validate_definition
    selected = validate_components(args, names)
    requested = bool(getattr(args, "decomposition", False))
    result = dict(requested=requested, components=[], sources={})
    for task in ("outcome_scoring", "outcome_conceding"):
        model = models.get(task)
        components = [name for name in selected if name.rsplit("_", 1)[0] == task]
        if model is not None:
            model.visualize_outcome_decomposition = False
        if not requested or not components:
            continue
        if model is None:
            if requested:
                raise ValueError(f"Missing required checkpoint for {task}.")
            continue
        settings = dict(model.args if isinstance(model.args, dict) else vars(model.args))
        settings.setdefault("learning_mode", "scalar")
        source = _source(settings, settings.get("model_id"), components, requested)
        if source["components"]:
            validate_definition(settings.get("outcome_decomposition_definition"))
            model.visualize_outcome_decomposition = True
        result["sources"][task] = source
        result["components"].extend(source["components"])
    return result


def _source(settings, model_id, components, requested):
    from models.outcome_decomposition import is_decomposition
    mode = settings.get("learning_mode")
    eligible = requested and is_decomposition(settings)
    reason = None
    if not requested:
        reason = "flag disabled"
    elif mode is None:
        reason = "learning mode unknown"
    elif not eligible:
        reason = f"learning_mode={mode}"
    return dict(model_id=model_id, learning_mode=mode,
                components=list(components) if eligible else [], reason=reason)


def report_selection(args, result):
    """Record the effective policy and issue at most one informational notice per run."""
    args._decomposition_selection = result
    if not getattr(args, "decomposition", False):
        return
    selections = list(result.get("variants", {}).values()) if "variants" in result else [result]
    unknown = [f"{source.get('model_id') or task}: learning mode unknown"
               for selection in selections for task, source in selection["sources"].items()
               if source["learning_mode"] is None]
    messages = []
    if not any(selection["components"] for selection in selections):
        messages.append("no eligible individual outcome components; rendering without decomposition labels")
    if unknown:
        messages.append("labels unavailable for " + "; ".join(dict.fromkeys(unknown)))
    if messages and not getattr(args, "_decomposition_notice_reported", False):
        print("Decomposition: " + ". ".join(messages) + ".")
        args._decomposition_notice_reported = True


def _saved_settings(metadata, task):
    """Prefer recorded learning settings; consult only the exact referenced checkpoint."""
    from models.utils import get_model_record
    record = metadata.get("model_records", {}).get(task)
    entry = metadata.get("models", {}).get(task)
    if not isinstance(record, dict):
        record = entry if isinstance(entry, dict) else {}
    model_id = record.get("model_id") or (entry if isinstance(entry, str) else None)
    model_id = model_id or metadata.get("model_ids", {}).get(task)
    model_id = model_id or metadata.get("model_selection", {}).get("selected_models", {}).get(task)
    settings = {**(record.get("args") or {}), **record}
    if settings.get("learning_mode") is None and settings.get("outcome_decomposition_definition"):
        settings["learning_mode"] = "decomposition"
    needs_definition = (settings.get("learning_mode") == "decomposition"
                        and settings.get("outcome_decomposition_definition") is None)
    if model_id and (settings.get("learning_mode") is None or needs_definition):
        try:
            resolved = get_model_record(model_id)
        except FileNotFoundError:
            pass
        else:
            resolved = {**resolved["args"], **resolved}
            for key in ("learning_mode", "outcome_decomposition_definition"):
                if settings.get(key) is None:
                    settings[key] = resolved.get(key)
    return settings, model_id


def prepare_saved(args, raw, metadata, names):
    """Resolve per-source eligibility before requiring or attaching any saved heads."""
    from models.outcome_decomposition import validate_definition
    selected = validate_components(args, names)
    requested = bool(getattr(args, "decomposition", False))
    result = dict(requested=requested, components=[], sources={})
    attrs = getattr(raw, "attrs", {})
    attrs["show_outcome_decomposition"] = False
    attrs["decomposition_components"] = []
    if requested:
        for task in ("outcome_scoring", "outcome_conceding"):
            components = [name for name in selected if name.rsplit("_", 1)[0] == task]
            if not components:
                continue
            settings, model_id = _saved_settings(metadata, task)
            source = _source(settings, model_id, components, requested)
            if source["components"]:
                definition = validate_definition(settings.get("outcome_decomposition_definition"))
                source["definition_id"] = definition["definition_sha256"]
                masks = metadata.get("inference_masks") or {}
                mask = masks.get("models", {}).get(task, {})
                if masks.get("requested") and not mask.get("configuration_id"):
                    raise ValueError(f"Cannot establish saved inference mask for {model_id or task}.")
                source["inference_mask_id"] = mask.get("configuration_id", "unmasked")
                result["components"].extend(components)
            result["sources"][task] = source
    if result["components"]:
        path = attrs.get("outcome_decomposition_path")
        if path and not attrs.get("outcome_decomposition_rows"):
            import pandas as pd
            attrs["outcome_decomposition_rows"] = pd.read_parquet(path).to_dict("records")
        if not attrs.get("outcome_decomposition_rows"):
            raise ValueError(f"Component run {metadata.get('run_id', '<unidentified>')}: "
                             f"missing saved occurrence/severity predictions for {result['components']}; "
                             "regenerate exports with --export-outcome-decomposition.")
        attrs["show_outcome_decomposition"] = True
        attrs["decomposition_components"] = result["components"]
    attrs["decomposition_selection"] = result
    return result


def annotate(ax, snapshot, attacking_prefix, probs, *, rotate_pitch=False, component_name=None):
    if component_name is not None and component_name not in COMPONENTS:
        return
    if probs is None or not probs.attrs.get("show_outcome_decomposition"):
        return
    mapping = probs.attrs.get("outcome_decomposition", {})
    # Hawkeye uses frame IDs as action indices. Sportec series retain the model
    # action index even when a comparison table subsequently receives a frame index.
    index = probs.attrs.get("decomposition_action_index", probs.name)
    pairs = mapping.get(index, mapping.get(str(index), {}))
    if not pairs:
        raise ValueError(f"Missing decomposition predictions for displayed frame/action {index}.")
    for candidate in probs.index:
        if not isinstance(candidate, str) or not candidate.startswith(attacking_prefix):
            continue
        if candidate not in pairs:
            continue
        q, m = pairs[candidate]
        xcol, ycol = candidate + "_x", candidate + "_y"
        if xcol not in snapshot or ycol not in snapshot:
            continue
        x, y = snapshot[xcol].iloc[-1], snapshot[ycol].iloc[-1]
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if rotate_pitch:
            x, y = config.FIELD_SIZE[0] - x, config.FIELD_SIZE[1] - y
        text = f"{q:.3f} x {m:.3f}" if math.isfinite(q) and math.isfinite(m) else "- x -"
        ax.annotate(text, xy=(x, y), xytext=(0, -3.5), textcoords="offset points",
                    ha="center", va="top", color="black", fontsize=10, zorder=7)


def attach_saved(tables, rows, scene_id, components=None, sources=None):
    """Attach saved heads using semantic scene, frame, branch and candidate keys."""
    selected = COMPONENTS & tables.keys()
    if components is not None:
        selected &= set(components)
    for table in tables.values():
        for key in ("show_outcome_decomposition", "outcome_decomposition", "decomposition_action_index"):
            table.attrs.pop(key, None)
    for component in selected:
        table = tables[component]
        if table.empty:
            continue
        task, branch = component.rsplit("_", 1)
        mapping = {}
        for row in rows:
            if str(row["scene_id"]) != str(scene_id) or row["task"] != task or int(row["execution_branch"]) != int(branch == "success"):
                continue
            frame, candidate = int(row["frame_id"]), row["candidate_id"]
            if frame not in table.index or candidate not in table.columns:
                continue
            if sources is not None:
                source = sources[task]
                for key in ("model_id", "definition_id", "inference_mask_id"):
                    if source.get(key) is not None and row.get(key) != source[key]:
                        raise ValueError(f"Saved decomposition {key} disagrees with source {source.get('model_id') or task} for {component}.")
            pairs = mapping.setdefault(frame, {})
            if candidate in pairs:
                raise ValueError("Ambiguous saved decomposition predictions; model/mask keys must identify one source.")
            actual = table.at[frame, candidate]
            prediction = row["prediction"]
            if math.isfinite(actual) and not math.isclose(actual, prediction, abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("Saved decomposition predictions disagree with outcome component values.")
            q, m = float(row["occurrence_probability"]), float(row["conditional_severity"])
            if not row.get("forced", False) and math.isfinite(prediction) and not math.isclose(q*m, prediction, abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("Saved occurrence x severity does not match its expected value.")
            pairs[candidate] = [q, m]
        for frame in table.index:
            for candidate, value in table.loc[frame].items():
                if math.isfinite(value) and candidate not in mapping.get(frame, {}):
                    raise ValueError(f"Missing saved decomposition for {component}, frame {frame}, candidate {candidate}.")
        table.attrs.update(outcome_decomposition=mapping, show_outcome_decomposition=True)
