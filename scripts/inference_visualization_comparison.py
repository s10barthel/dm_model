"""Checkpoint, pc-xPass cache, and runtime-mask comparisons for inference visualizers."""
from __future__ import annotations

from pc_xpass_inference_policy import (PCXPassBlendingError, start_inference_attempt, validate_pc_xpass_startup, inference_attempt, cli_blending_guard)

import json
from contextlib import contextmanager
from copy import copy, deepcopy
from pathlib import Path
import pandas as pd
import torch

from models.inference_masks import configure_inference_masks, flatten_mask_values, resolve_inference_mask_configurations
from pc_xpass_auxiliary import configure_auxiliary_models, auxiliary_metadata, validate_auxiliary_selectors
from models.edge_schema import generation_kwargs
from models.membership import explicit_split_provenance
from physical_pass_model import (
    PC_XPASS_SOURCE, physical_xpass_metric_for_version,
    validate_physical_xpass_cache_metadata, validate_x_pass_version_available,
    validate_pc_xpass_lane_survival_mode_cache_metadata,
)
from models.utils import (
    get_model_record, load_bundle_record, load_model,
    validate_model_record_consistency, validate_model_graph_schemas,
)
from scripts.visualization_comparison import (
    MODEL_TASKS, OUTCOME_TASKS, ComparisonExport, ComparisonGroup, Variant, selection_list,
    validate_metadata_definitions,
    render_comparison_groups,
)
from scripts.comparison_warnings import metadata_differences
from scripts.visualization_carries import carry_visualization_warnings


DERIVED_OUTCOMES = {"outcome_success", "outcome_failure", "pass_score"}


def pc_xpass_specifications(args):
    """Expand only the varied selector, retaining scalar settings for every pipeline."""
    dimension = getattr(args, "comparison_dimension", None) or "cache_id"
    selector = {"cache_id": "pc_xpass_id", "version": "x_pass_version", "weight": "xpass_weight"}[dimension]
    base = {key: selection_list(getattr(args, key))[0]
            for key in ("pc_xpass_id", "x_pass_version", "xpass_weight")}
    specifications = {}
    for value in selection_list(getattr(args, selector)):
        settings = {**base, selector: value}
        identifier = value if dimension == "cache_id" else f"{dimension}:{value}"
        label = f"{value}\n{settings['pc_xpass_id']} | {settings['x_pass_version']} | {settings['xpass_weight']}"
        specifications[identifier] = (label, settings)
    return specifications


def variant_context(variant, scene=None):
    settings = {**variant.pc_xpass_settings, **variant.provenance}
    location = f"{scene}, " if scene else ""
    return (f"pc-xPass cache {settings.get('pc_xpass_id', variant.identifier)!r}, {location}"
            f"version {settings.get('x_pass_version')!r}, weight {settings.get('xpass_weight')!r}")


def resolve_variants(args, mod, sportec):
    bundle = load_bundle_record(args.bundle_id) if args.bundle_id else None
    bundle_ids = dict((bundle or {}).get("model_ids") or {})
    explicit = {task: selection_list(getattr(args, f"{task}_model_id", None)) for task in MODEL_TASKS}
    selection = mod.resolve_component_selection(args, **({"include_success_intent": bool(
        explicit["success_intent"] or bundle_ids.get("success_intent")
        or getattr(args, "only_success_intent", False) or getattr(args, "success_intent", False)
        or getattr(args, "compare_intended_receiver", False)
    )} if sportec else {}))
    required = [task for task in selection.required_component_groups if task in MODEL_TASKS]
    varied = [task for task, ids in explicit.items() if len(ids) > 1]
    paired = set(varied) == set(OUTCOME_TASKS)
    for task in varied:
        if task not in required:
            raise ValueError(f"Compared task {task} does not contribute to the selected visualization components.")
    if (args.visualization_comparison == "checkpoints" and set(varied) & set(OUTCOME_TASKS)
            and set(OUTCOME_TASKS).issubset(required) and not paired):
        raise ValueError("Selected outcome outputs require explicit scoring/conceding checkpoint pairs. "
                         "Repeat --outcome-scoring-model-id and --outcome-conceding-model-id equally, "
                         "including repeating a fixed counterpart explicitly for each pair; "
                         "bundle defaults do not supply compared counterparts. "
                         "For a single outcome model, select only its raw outputs with --only-outcome-scoring "
                         "or --only-outcome-conceding.")
    base = {}
    for task in required:
        identifiers = explicit[task] or selection_list(bundle_ids.get(task))
        if not identifiers:
            raise ValueError(f"Missing --{task.replace('_', '-')}-model-id or bundle checkpoint for selected components.")
        base[task] = identifiers[0]
    common_masks = flatten_mask_values(args.inference_mask)
    physical_specs = {}
    if args.visualization_comparison == "pc_xpass":
        physical_specs = pc_xpass_specifications(args)
        specifications = [(identifier, label, base, common_masks)
                          for identifier, (label, _) in physical_specs.items()]
    elif args.visualization_comparison == "checkpoints":
        if paired:
            specifications = [("outcome_pair:" + json.dumps(pair, separators=(",", ":")),
                               "\n".join(pair), {**base, **dict(zip(OUTCOME_TASKS, pair))}, common_masks)
                              for pair in zip(*(explicit[task] for task in OUTCOME_TASKS))]
        else:
            task = varied[0]
            specifications = [(identifier, identifier, {**base, task: identifier}, common_masks)
                              for identifier in explicit[task]]
    else:
        specifications = [("unmasked", "Unmasked", base, [])]
        for entry in resolve_inference_mask_configurations(args):
            number = entry["id"].split("_")[-1]
            specifications.append((entry["id"], f"mask-{number}: " + ", ".join(entry["masks"]), base, entry["masks"]))
    models, records, contexts, variants = {}, {}, [], []
    shared_auxiliaries = None
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
            pipeline_models[task] = models[model_id]
        context = (validate_model_record_consistency(pipeline_records, require_feature_run_id=False,
            require_intended_receiver_mode=False, require_return_type=False, require_target_family=False,
            allow_pass_only_success=True) if pipeline_records else {})
        contexts.append(context)
        if args.visualization_comparison == "pc_xpass":
            if shared_auxiliaries is None:
                auxiliary_options = copy(args)
                weights = selection_list(args.xpass_weight)
                auxiliary_options.xpass_weight = "v5" if "v5" in weights else "v4" if "v4" in weights else weights[0]
                auxiliary_models = {}
                if not pipeline_models:
                    validate_auxiliary_selectors(auxiliary_options, active=False)
                shared_auxiliaries = (configure_auxiliary_models(auxiliary_options, auxiliary_models,
                    "cpu" if not torch.cuda.is_available() else args.device,
                    active=bool(args.use_physical_xpass and "pass_success" in pipeline_models)) if pipeline_models else {})
                args._xpass_auxiliary_records = getattr(auxiliary_options, "_xpass_auxiliary_records", {})
            weight = physical_specs[identifier][1]["xpass_weight"]
            roles = {"pass_height", "pass_intent"} if weight == "v5" else {"pass_height"} if weight == "v4" else set()
            auxiliaries = {role: model for role, model in shared_auxiliaries.items() if role in roles}
            pipeline_models.update({f"xpass_{role}": model for role, model in auxiliaries.items()})
        else:
            auxiliaries = (configure_auxiliary_models(args, pipeline_models, "cpu" if not torch.cuda.is_available() else args.device)
                           if pipeline_models else {})
        for role, auxiliary in auxiliaries.items():
            models[f"auxiliary:{identifier}:{role}"] = auxiliary
        variant_schema = validate_model_graph_schemas(pipeline_models)
        variants.append(Variant(identifier, label, {"model_ids": ids, "models": pipeline_records,
            "source_splits": {task: explicit_split_provenance(record) for task, record in pipeline_records.items()},
            "source_context": context, "graph_schema": variant_schema}, masks, pipeline_models,
            pc_xpass_settings=physical_specs[identifier][1] if physical_specs else {}))
    shared = {key: value if all(item.get(key) == value for item in contexts) else None
              for key, value in contexts[0].items()}
    shared["source_feature_run_ids"] = {identifier: record.get("feature_run_id") for identifier, record in records.items()}
    if bundle:
        shared["bundle_feature_run_id"] = bundle.get("feature_run_id")
        for key in ("feature_run_id", "intended_receiver_mode", "return_type", "target_family"):
            shared[key] = shared.get(key) or bundle.get(key)
    schema = validate_model_graph_schemas(models) if physical_specs else variants[0].provenance["graph_schema"]
    if physical_specs:
        for variant in variants:
            variant.provenance["graph_schema"] = schema
    return variants, selection, shared, bundle, schema


def configure_physical(args, mod, variants, dataset):
    if getattr(args, "visualization_comparison", None) == "pc_xpass":
        configure_cache_variants(args, mod, variants, dataset)
        return
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


def configure_cache_variants(args, mod, variants, dataset):
    """Resolve caches independently and capture settings without mutating shared models."""
    for variant in variants:
        options = copy(args)
        settings = variant.pc_xpass_settings or dict(pc_xpass_id=variant.identifier,
            x_pass_version=args.x_pass_version, xpass_weight=args.xpass_weight)
        vars(options).update(settings)
        variant.pc_xpass_settings = settings
        options.visualization_comparison = None
        options.physical_cache_dir = None
        try:
            metric = physical_xpass_metric_for_version(options.x_pass_version, pc_xpass=True)
            cache = mod.pc_versions.cache_dir(dataset, options)
            metadata = validate_physical_xpass_cache_metadata(cache, expected_source=PC_XPASS_SOURCE)
            validate_x_pass_version_available(cache, source=PC_XPASS_SOURCE,
                                              x_pass_version=options.x_pass_version, metric=metric)
            originals = {key: model.args for key, model in variant.models.items()}
            try:
                for model in variant.models.values():
                    model.args = deepcopy(model.args)
                configure_physical(options, mod, [variant], dataset)
                primary = variant.models.get("pass_success")
                if primary is not None and hasattr(args, "_xpass_auxiliary_records"):
                    records = {role: deepcopy(record) for role, record in args._xpass_auxiliary_records.items()
                               if f"xpass_{role}" in variant.models}
                    primary.args["xpass_auxiliary_models"] = records
                    primary.args.pop("xpass_pass_height_definition", None)
                    if "pass_height" in records:
                        primary.args["xpass_pass_height_definition"] = records["pass_height"]["pass_height_definition"]
                if primary is not None and args.use_physical_xpass:
                    irrelevant = ("v4_power", "v4_zero", "v4_discount") if options.xpass_weight != "v4" else ()
                    irrelevant += ("v5_intent_threshold", "v5_discount") if options.xpass_weight != "v5" else ()
                    for key in irrelevant:
                        primary.args.pop(key, None)
                for key, model in variant.models.items():
                    if model.args.get("lane_survival"):
                        validate_pc_xpass_lane_survival_mode_cache_metadata(metadata, model.args.get("lane_survival_mode"))
                    if key == "pass_success":
                        model.args["physical_xpass_require_complete"] = True
                    if key == "pass_success" and (mod.model_uses_physical_xpass(model.args)
                                                  or mod.inference_uses_physical_xpass(model.args)):
                        model.args["pc_xpass"] = True
                    if model.args.get("lane_survival") or key == "pass_success":
                        model.args["pc_xpass_id"] = options.pc_xpass_id
                        model.args["pc_xpass_namespace"] = options.pc_xpass_namespace
                variant.runtime_model_args = {key: deepcopy(model.args) for key, model in variant.models.items()}
            finally:
                for key, model in variant.models.items():
                    model.args = originals[key]
            variant.provenance.update(**settings, physical_cache_dir=str(cache),
                pc_xpass_namespace=options.pc_xpass_namespace, physical_xpass_runtime_source=PC_XPASS_SOURCE,
                physical_xpass_metric=metric,
                effective_model_args=variant.runtime_model_args)
        except (ValueError, FileNotFoundError) as exc:
            raise type(exc)(f"{variant_context(variant)}: {exc}") from exc


@contextmanager
def active_variant(variant, scene):
    """Scope cache configuration, masks and auxiliary bindings to one prediction pipeline."""
    if "pc_xpass_id" not in variant.provenance:
        _configure_variant(variant)
        yield
        return
    originals, masks = [], []
    missing = object()
    primary = variant.models.get("pass_success")
    auxiliary_binding = getattr(primary, "_xpass_auxiliary_models", missing)
    try:
        for key, model in variant.models.items():
            originals.append((model, model.args))
            model.args = deepcopy(variant.runtime_model_args[key])
            for component in (model, getattr(model, "encoder", None), getattr(model, "decoder", None)):
                if component is not None:
                    masks.append((component, getattr(component, "inference_masks", missing)))
        _configure_variant(variant)
        yield
    except PCXPassBlendingError as exc:
        raise PCXPassBlendingError(f"{variant_context(variant, scene)}: {exc}",
                                  stage=exc.stage, context={**exc.context, "cache_id": variant.provenance["pc_xpass_id"],
                                                            "version": variant.provenance["x_pass_version"],
                                                            "weight": variant.provenance["xpass_weight"],
                                                            "scene": scene}) from exc
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise type(exc)(f"{variant_context(variant, scene)}: {exc}") from exc
    finally:
        for model, original in originals:
            model.args = original
        for component, original in masks:
            if original is missing:
                if hasattr(component, "inference_masks"):
                    del component.inference_masks
            else:
                component.inference_masks = original
        if primary is not None:
            if auxiliary_binding is missing:
                if hasattr(primary, "_xpass_auxiliary_models"):
                    del primary._xpass_auxiliary_models
            else:
                primary._xpass_auxiliary_models = auxiliary_binding


def _configure_variant(variant):
    primary = variant.models.get("pass_success")
    if primary is not None:
        primary._xpass_auxiliary_models = {role: variant.models[f"xpass_{role}"]
                                         for role in ("pass_height", "pass_intent") if f"xpass_{role}" in variant.models}
    metadata = configure_inference_masks(variant.models, variant.masks)
    variant.provenance["inference_masks"] = metadata
    variant.provenance["xpass_auxiliaries"] = auxiliary_metadata(variant.models)


def hawkeye_groups(args, mod, variants, selection, schema, device, *, deferred=False):
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
        def load_group(identifier=identifier):
            scene_tracking = tracking.loc[tracking.id.eq(str(identifier))].copy()
            subject, _, _ = mod.build_hawkeye_situation(scene_tracking, ball,
                freeze_ballreceipt=args.freeze_ballreceipt, **({"build_graphs": False} if not variants[0].models else {}),
                add_v_edge_features=bool(schema["add_v_edge_features"]),
                add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)), **generation_kwargs(schema))
            selected_frames = mod.resolve_hawkeye_png_frames(subject, mod.resolve_ballreceipt(scene_tracking), mod.requested_time_norms(args))
            frames = list(dict.fromkeys(int(item["frame_id"]) for item in selected_frames))
            tables, cache, physical_tables = [], {}, {}
            for variant in variants:
                with active_variant(variant, f"situation {identifier}, frames {frames}"):
                    values = (mod.collect_situation_components(subject, variant.models, device, frames,
                        selection.rendered_components, prediction_cache=cache) if variant.models else {})
                    if "physical_xpass" in selection.rendered_components:
                        directory = variant.provenance.get("physical_cache_dir", args.physical_cache_dir)
                        version = variant.provenance.get("x_pass_version", args.x_pass_version)
                        metric = variant.provenance.get("physical_xpass_metric") or mod.physical_xpass_metric(args)
                        physical_key = (directory, metric, version)
                        if physical_key not in physical_tables:
                            physical_tables[physical_key] = mod.load_runtime_physical_xpass_visualization_table(
                                directory, str(subject.match_id), frames, metric=metric, x_pass_version=version)
                        values["physical_xpass"] = physical_tables[physical_key].copy(deep=True)
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

        group = ComparisonGroup(('hawkeye', str(identifier)), load_group)
        if deferred:
            yield group
        else:
            yield from group.load()


def sportec_groups(args, mod, variants, selection, shared, bundle, schema, device, *, deferred=False):
    feature_run_id = args.feature_run_id
    if not feature_run_id and not selection.required_component_groups and variants[0].provenance.get("physical_cache_dir"):
        info = mod.pc_versions.read_metadata(variants[0].provenance["physical_cache_dir"])
        feature_run_id = (info.get("source_inputs") or {}).get("feature_run_id")
    runtime = mod.resolve_runtime_feature_run_context(feature_run_id, shared, bundle, args.intended_receiver_mode,
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
    possession_states = None
    if args.pc_xpass or any(model.args.get("pc_xpass") or model.args.get("lane_survival")
                            for variant in variants for model in variant.models.values()):
        from datatools.sportec_possessions import build_sportec_possessions
        # This subject is a fixed snapshot for the group; select independent frame
        # views below instead of rebuilding every physical graph for every action.
        possession_states = list(build_sportec_possessions(subject, args.match_id, feature_root,
            scope="actions", **getattr(subject, "runtime_graph_options", {})))
    for variant in variants:
        variant.provenance["runtime_feature_context"] = {key: str(value) if isinstance(value, Path) else value for key, value in runtime.items()}
    for action_index, display_id in actions:
        def load_group(action_index=action_index, display_id=display_id):
            frame = int(subject.actions.at[action_index, "frame_id"])
            tables, cache = [], {}
            for variant in variants:
                with active_variant(variant, f"match {args.match_id}, action {display_id}"):
                    values = mod.collect_action_components(subject, variant.models, feature_root, device, action_index,
                        display_id, Path(args.output_dir), show_trajectories=args.show_trajectories,
                        show_physical_xpass=args.show_physical_xpass, physical_cache_dir=variant.provenance.get("physical_cache_dir", args.physical_cache_dir),
                        physical_xpass_metric_name=variant.provenance.get("physical_xpass_metric") or mod.physical_xpass_metric(args),
                        physical_xpass_version_name=variant.provenance.get("x_pass_version", args.x_pass_version),
                        rendered_components=selection.rendered_components, prediction_cache=cache,
                        possession_states=possession_states)
                mod.prepare_action_overlays(
                    subject, action_index, values.get("success_intent"),
                    pass_endpoint=getattr(args, "pass_endpoint", True),
                    compare_intended_receiver=getattr(args, "compare_intended_receiver", False),
                )
                panel_tables = {}
                for component, row in values.items():
                    table = pd.DataFrame([row], index=[frame])
                    table.attrs.update(row.attrs)
                    panel_tables[component] = table
                tables.append(panel_tables)
            action = subject.actions.loc[action_index]
            yield {"scene": f"{args.match_id}/action_{display_id}", "frame": frame, "tables": tables,
                   "folder": str(Path(str(args.match_id)) / str(display_id)), "suffix": "",
                   "title": f"{action['spadl_type']} {display_id}",
                   "geometry": {"match_id": str(args.match_id), "action_id": str(display_id), "action_index": int(action_index)},
                   "render": lambda component, row, action_index=action_index, display_id=display_id:
                       mod.render_component(subject, action_index, display_id, component, row, Path(args.output_dir),
                                            show_trajectories=args.show_trajectories, comparison=True,
                                            pass_endpoint=getattr(args, "pass_endpoint", True),
                                            compare_intended_receiver=getattr(args, "compare_intended_receiver", False))}

        group = ComparisonGroup(('sportec', str(args.match_id), str(display_id)), load_group)
        if deferred:
            yield group
        else:
            yield from group.load()


@inference_attempt
def run_comparison(args, mod):
    root = Path(args.output_dir) / (args.run_id or mod.generate_run_id("visualization_comparison"))
    start_inference_attempt(root, args)
    sportec = Path(mod.__file__).stem == "visualize_action_components"
    dataset = "sportec" if sportec else "hawkeye"
    device = args.device if torch.cuda.is_available() else "cpu"
    variants, selection, shared, bundle, schema = resolve_variants(args, mod, sportec)
    with cli_blending_guard(args):
        configure_physical(args, mod, variants, dataset)
    # Validate all scopes/schemas before any expensive inference or artifact output.
    for variant in variants:
        with active_variant(variant, "preflight"):
            validate_pc_xpass_startup(variant.models.get("pass_success"))
    names = list(selection.rendered_components)
    report = ComparisonExport(root, variants, args, args.visualization_comparison)
    report.metadata.update(run_id=root.name, script=Path(mod.__file__).name, graph_schema=schema,
                           source_context=shared, bundle_id=args.bundle_id, rendered_components=names,
                           only_physical_xpass=bool(getattr(args, "only_physical_xpass", False)))
    if args.visualization_comparison == "pc_xpass":
        report.metadata["comparison_dimension"] = getattr(args, "comparison_dimension", None) or "cache_id"
    if "physical_xpass" in names and args.visualization_comparison != "pc_xpass":
        report.metadata.update(physical_cache_dir=args.physical_cache_dir,
            physical_xpass_runtime_source=mod.physical_xpass_panel_source(args, args.physical_cache_dir),
            physical_xpass_metric=mod.physical_xpass_metric(args), x_pass_version=args.x_pass_version)
    if sportec:
        report.metadata.update(pass_endpoint=getattr(args, "pass_endpoint", True),
                               compare_intended_receiver=getattr(args, "compare_intended_receiver", False),
                               skipped_actions=[])
    try:
        validate_metadata_definitions([variant.provenance for variant in variants], report.policy, tasks=required_tasks(selection))
        for variant in variants:
            for message in carry_visualization_warnings(variant.provenance["source_context"].get("source_use_carries", {})):
                report.policy.add(f"Within variant {variant.identifier}: {message}")
        if DERIVED_OUTCOMES.intersection(names):
            for variant in variants:
                records = [variant.provenance["models"][task] for task in OUTCOME_TASKS]
                for message in metadata_differences(records, ("target_family", "return_type"),
                        prefix=f"Within outcome pair {variant.identifier}: "):
                    report.policy.add(message)
        report.policy.confirm()
        report.start()
        same_schema = all(variant.provenance["graph_schema"] == schema for variant in variants)
        same_context = all(variant.provenance["source_context"] == variants[0].provenance["source_context"] for variant in variants)
        if same_schema and (not sportec or same_context):
            groups = (sportec_groups(args, mod, variants, selection, variants[0].provenance["source_context"], bundle, schema, device, deferred=True)
                      if sportec else hawkeye_groups(args, mod, variants, selection, schema, device, deferred=True))
            variant_groups = [groups]
        else:
            variant_groups = [sportec_groups(args, mod, [variant], selection, variant.provenance["source_context"], bundle,
                              variant.provenance["graph_schema"], device, deferred=True) if sportec else hawkeye_groups(
                              args, mod, [variant], selection, variant.provenance["graph_schema"], device, deferred=True) for variant in variants]
        render_comparison_groups(report, variant_groups, names, shared=same_schema and (not sportec or same_context),
                                 skip_failed_groups=sportec and args.visualization_comparison != "pc_xpass")
        if sportec and not report.outputs:
            raise RuntimeError("All selected actions were skipped or failed during rendering.")
        report.finish()
    except (Exception, KeyboardInterrupt) as exc:
        report.failed(exc)
        raise
    finally:
        # Shared model instances must not retain the last numbered intervention.
        if args.visualization_comparison != "pc_xpass":
            for variant in variants:
                configure_inference_masks(variant.models, [])
    print(f"Saved {dataset} comparison visualizations and prediction tables to {root}")


def required_tasks(selection):
    return list(selection.required_component_groups)
