"""Checkpoint and runtime-mask comparison adapters for inference visualizers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

from models.inference_masks import configure_inference_masks, flatten_mask_values, resolve_inference_mask_configurations
from pc_xpass_auxiliary import configure_auxiliary_models, auxiliary_metadata
from models.edge_schema import generation_kwargs
from models.utils import (
    get_model_record, load_bundle_record, load_model,
    validate_model_record_consistency, validate_model_graph_schemas,
)
from scripts.visualization_comparison import (
    MODEL_TASKS, ComparisonExport, Variant, selection_list,
    validate_metadata_definitions,
    render_independent_groups,
)


def resolve_variants(args, mod, sportec):
    bundle = load_bundle_record(args.bundle_id) if args.bundle_id else None
    bundle_ids = dict((bundle or {}).get("model_ids") or {})
    explicit = {task: selection_list(getattr(args, f"{task}_model_id", None)) for task in MODEL_TASKS}
    selection = mod.resolve_component_selection(args, **({"include_intended_recipient": bool(
        explicit["success_intent"] or bundle_ids.get("success_intent") or getattr(args, "only_intended_recipient", False)
    )} if sportec else {}))
    required = ["success_intent" if task == "intended_recipient" else task
                for task in selection.required_component_groups
                if task in MODEL_TASKS or task == "intended_recipient"]
    base = {}
    for task in required:
        identifiers = explicit[task] or selection_list(bundle_ids.get(task))
        if not identifiers:
            raise ValueError(f"Missing --{task.replace('_', '-')}-model-id or bundle checkpoint for selected components.")
        base[task] = identifiers[0]
    varied = next((task for task, ids in explicit.items() if len(ids) > 1), None)
    if varied and varied not in required:
        raise ValueError(f"Compared task {varied} does not contribute to the selected visualization components.")
    common_masks = flatten_mask_values(args.inference_mask)
    if args.visualization_comparison == "checkpoints":
        specifications = [(identifier, identifier, {**base, varied: identifier}, common_masks)
                          for identifier in explicit[varied]]
    else:
        specifications = [("unmasked", "Unmasked", base, [])]
        for entry in resolve_inference_mask_configurations(args):
            number = entry["id"].split("_")[-1]
            specifications.append((entry["id"], f"mask-{number}: " + ", ".join(entry["masks"]), base, entry["masks"]))
    models, records, contexts, variants = {}, {}, [], []
    for identifier, label, ids, masks in specifications:
        pipeline_records = {}
        pipeline_models = {}
        for task, model_id in ids.items():
            if model_id.split("/", 1)[0] != task:
                raise ValueError(f"Checkpoint {model_id!r} does not match task {task!r}.")
            if model_id not in records:
                record = get_model_record(model_id)
                if record.get("task") != task:
                    raise ValueError(f"Checkpoint {model_id!r} has a different task.")
                if task == "pass_height" and not record.get("pass_height_definition"):
                    from models.pass_height import model_definition
                    record["pass_height_definition"] = model_definition(record.get("args", {}), record.get("metadata"))
                records[model_id] = record
                model = load_model(model_id, device="cpu" if not torch.cuda.is_available() else args.device)
                if model is None:
                    raise FileNotFoundError(f"Missing model checkpoint: {model_id}")
                if model.args.get("task") != task:
                    raise ValueError(f"Loaded checkpoint {model_id!r} has an incompatible model task.")
                model.eval()
                models[model_id] = model
            pipeline_records[task] = records[model_id]
            pipeline_models["intended_recipient" if task == "success_intent" else task] = models[model_id]
        context = validate_model_record_consistency(pipeline_records, require_feature_run_id=False,
            require_intended_receiver_mode=False, require_return_type=False, require_target_family=False)
        contexts.append(context)
        auxiliaries = configure_auxiliary_models(args, pipeline_models, "cpu" if not torch.cuda.is_available() else args.device)
        for role, auxiliary in auxiliaries.items():
            models[f"auxiliary:{identifier}:{role}"] = auxiliary
        variant_schema = validate_model_graph_schemas(pipeline_models)
        variants.append(Variant(identifier, label, {"model_ids": ids, "models": pipeline_records,
            "source_context": context, "graph_schema": variant_schema}, masks, pipeline_models))
    shared = {key: value if all(item.get(key) == value for item in contexts) else None
              for key, value in contexts[0].items()}
    shared["source_feature_run_ids"] = {identifier: record.get("feature_run_id") for identifier, record in records.items()}
    if bundle:
        shared["bundle_feature_run_id"] = bundle.get("feature_run_id")
        for key in ("feature_run_id", "intended_receiver_mode", "return_type", "target_family"):
            shared[key] = shared.get(key) or bundle.get(key)
    schema = variants[0].provenance["graph_schema"]
    return variants, selection, shared, bundle, schema


def configure_physical(args, mod, variants, dataset):
    cache = args.physical_cache_dir or str(mod.pc_versions.cache_dir(dataset, args) if args.pc_xpass
                                         else mod.get_runtime_physical_xpass_dir(dataset))
    args.physical_cache_dir = cache
    seen = set()
    for variant in variants:
        lane_cache = mod.pc_versions.lane_cache_dir(dataset, args, variant.models)
        mod.configure_lane_survival_runtime_cache(variant.models, lane_cache)
        model = variant.models.get("pass_success")
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        if args.use_physical_xpass:
            model.args.update(inference_use_physical_xpass=True, pc_xpass=bool(args.pc_xpass),
                              x_pass_version=args.x_pass_version, xpass_weight=args.xpass_weight,
                              v4_discount=args.v4_discount, v5_intent_threshold=args.v5_intent_threshold,
                              v5_discount=args.v5_discount, ball_z_limit=args.ball_z_limit)
            for key in ("v4_power", "v4_zero", "pass_height_threshold"):
                if getattr(args, key, None) is not None:
                    model.args[key] = getattr(args, key)
        if mod.model_uses_physical_xpass(model.args) or mod.inference_uses_physical_xpass(model.args):
            model.args.update(physical_runtime_cache_disabled=args.no_physical_cache,
                physical_runtime_cache_refresh=False, physical_runtime_cache_read_only=True,
                physical_num_workers=args.physical_num_workers,
                physical_worker_thread_limit=args.physical_worker_thread_limit,
                physical_batch_size=args.physical_batch_size)
            if not args.no_physical_cache:
                model.args["physical_cache_dir"] = cache


def _configure_variant(variant):
    primary = variant.models.get("pass_success")
    if primary is not None:
        primary._xpass_auxiliary_models = {role: variant.models[f"xpass_{role}"]
                                         for role in ("pass_height", "pass_intent") if f"xpass_{role}" in variant.models}
    metadata = configure_inference_masks(variant.models, variant.masks)
    variant.provenance["inference_masks"] = metadata
    variant.provenance["xpass_auxiliaries"] = auxiliary_metadata(variant.models)


def hawkeye_groups(args, mod, variants, selection, schema, device):
    tracking = mod.clean_hawkeye_tracking(mod.load_hawkeye_tracking(args.tracking_csv))
    ball = mod.clean_hawkeye_ball(mod.load_hawkeye_ball(args.ball_csv))
    ids = mod.resolve_hawkeye_situation_ids(tracking,
        requested_ids=[str(item) for item in (args.situation_id or []) + (args.action_id or [])])
    overlays = mod.load_overlay_data(include_coach_ratings=args.coach_ratings, include_selections=args.selections)
    if args.coach_ratings:
        ids, _ = mod.filter_coach_rated_situation_ids(ids, overlays.coach_ratings)
    if not ids:
        raise ValueError("No selected Hawkeye comparison situations remain.")
    for identifier in ids:
        scene_tracking = tracking.loc[tracking.id.eq(str(identifier))].copy()
        subject, _, _ = mod.build_hawkeye_situation(scene_tracking, ball,
            freeze_ballreceipt=args.freeze_ballreceipt, add_v_edge_features=bool(schema["add_v_edge_features"]),
            add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)), **generation_kwargs(schema))
        selected_frames = mod.resolve_hawkeye_png_frames(subject, mod.resolve_ballreceipt(scene_tracking), mod.requested_time_norms(args))
        frames = list(dict.fromkeys(int(item["frame_id"]) for item in selected_frames))
        tables, cache = [], {}
        physical = None
        if args.show_physical_xpass:
            physical = mod.load_runtime_physical_xpass_visualization_table(args.physical_cache_dir, str(subject.match_id),
                frames, metric=mod.physical_xpass_metric(args), x_pass_version=args.x_pass_version)
        for variant in variants:
            _configure_variant(variant)
            values = mod.collect_situation_components(subject, variant.models, device, frames,
                selection.rendered_components, prediction_cache=cache)
            if physical is not None:
                values["physical_xpass"] = physical.copy(deep=True)
            tables.append(values)
        scores, labels, stats = mod.build_situation_overlays(overlays, str(identifier), scene_tracking, subject,
                                                           selection_format=args.selection_format)
        overlay_kwargs = dict(coach_scores=scores, selection_labels=labels, selections_enabled=args.selections,
                              selection_denominators=stats["selection_denominators"])
        for selected in selected_frames:
            frame = int(selected["frame_id"])
            yield {"scene": str(identifier), "frame": frame, "tables": tables, "folder": str(identifier),
                   "suffix": "_" + selected["label"], "title": f"{identifier} | {subject.frame_meta.at[frame, 'abs_time']:.3f}",
                   "selection_denominators": stats["selection_denominators"] if args.selections else None,
                   "geometry": {"selected_frame": selected, "freeze_ballreceipt": args.freeze_ballreceipt},
                   "render": lambda component, row, subject=subject, frame=frame, overlay_kwargs=overlay_kwargs:
                       mod.render_frame_image(subject, frame, component, row, show_trajectories=args.show_trajectories,
                                              comparison=True, **overlay_kwargs)}


def sportec_groups(args, mod, variants, selection, shared, bundle, schema, device):
    runtime = mod.resolve_runtime_feature_run_context(args.feature_run_id, shared, bundle, args.intended_receiver_mode,
                                                     schema, context="Selected comparison visualization feature artifacts")
    feature_root = Path(runtime["feature_root"])
    return_type = mod.resolve_runtime_return_type(shared, args.return_type)
    subject = mod.load_match(args.match_id, intended_receiver_mode=runtime["intended_receiver_mode"],
        return_type=return_type, feature_root=feature_root, add_v_edge_features=bool(schema["add_v_edge_features"]),
        add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)),
        **generation_kwargs(schema),
        use_carries=bool(shared.get("use_carries", (bundle or {}).get("use_carries", False))))
    actions = mod.resolve_action_indices(subject, args)
    if not actions:
        raise ValueError("No matching modeled actions selected for comparison.")
    for variant in variants:
        variant.provenance["runtime_feature_context"] = {key: str(value) if isinstance(value, Path) else value for key, value in runtime.items()}
    for action_index, display_id in actions:
        frame = int(subject.actions.at[action_index, "frame_id"])
        tables, cache = [], {}
        for variant in variants:
            _configure_variant(variant)
            values = mod.collect_action_components(subject, variant.models, feature_root, device, action_index,
                display_id, Path(args.output_dir), show_trajectories=args.show_trajectories,
                show_physical_xpass=args.show_physical_xpass, physical_cache_dir=args.physical_cache_dir,
                physical_xpass_metric_name=mod.physical_xpass_metric(args), physical_xpass_version_name=args.x_pass_version,
                rendered_components=selection.rendered_components, prediction_cache=cache)
            tables.append({component: pd.DataFrame([row], index=[frame]) for component, row in values.items()})
        action = subject.actions.loc[action_index]
        yield {"scene": f"{args.match_id}/action_{display_id}", "frame": frame, "tables": tables,
               "folder": str(Path(str(args.match_id)) / str(display_id)), "suffix": "",
               "title": f"{action['spadl_type']} {display_id}",
               "geometry": {"match_id": str(args.match_id), "action_id": str(display_id), "action_index": int(action_index)},
               "render": lambda component, row, action_index=action_index, display_id=display_id:
                   mod.render_component(subject, action_index, display_id, component, row, Path(args.output_dir),
                                        show_trajectories=args.show_trajectories, comparison=True)}


def run_comparison(args, mod):
    sportec = Path(mod.__file__).stem == "visualize_action_components"
    dataset = "sportec" if sportec else "hawkeye"
    device = args.device if torch.cuda.is_available() else "cpu"
    variants, selection, shared, bundle, schema = resolve_variants(args, mod, sportec)
    configure_physical(args, mod, variants, dataset)
    # Validate all scopes/schemas before any expensive inference or artifact output.
    for variant in variants:
        _configure_variant(variant)
    names = list(selection.rendered_components)
    if args.show_physical_xpass:
        names.append("physical_xpass")
    root = Path(args.output_dir) / (args.run_id or mod.generate_run_id(dataset + "_visualization_comparison"))
    report = ComparisonExport(root, variants, args, args.visualization_comparison)
    report.metadata.update(run_id=root.name, script=Path(mod.__file__).name, graph_schema=schema,
                           source_context=shared, bundle_id=args.bundle_id)
    try:
        validate_metadata_definitions([variant.provenance for variant in variants], report.policy, tasks=required_tasks(selection))
        report.policy.confirm()
        same_schema = all(variant.provenance["graph_schema"] == schema for variant in variants)
        same_context = all(variant.provenance["source_context"] == variants[0].provenance["source_context"] for variant in variants)
        if same_schema and (not sportec or same_context):
            groups = list(sportec_groups(args, mod, variants, selection, variants[0].provenance["source_context"], bundle, schema, device)
                          if sportec else hawkeye_groups(args, mod, variants, selection, schema, device))
            variant_groups = [[{**scene, "tables": [scene["tables"][index]]} for scene in groups] for index in range(len(variants))]
        else:
            variant_groups = [sportec_groups(args, mod, [variant], selection, variant.provenance["source_context"], bundle,
                              variant.provenance["graph_schema"], device) if sportec else hawkeye_groups(
                              args, mod, [variant], selection, variant.provenance["graph_schema"], device) for variant in variants]
        render_independent_groups(report, variant_groups, names)
        report.finish()
    except Exception as exc:
        report.failed(exc)
        raise
    finally:
        # Shared model instances must not retain the last numbered intervention.
        for variant in variants:
            configure_inference_masks(variant.models, [])
    print(f"Saved {dataset} comparison visualizations and prediction tables to {root}")


def required_tasks(selection):
    return ["success_intent" if task == "intended_recipient" else task for task in selection.required_component_groups]
