"""Dataset adapters for comparisons of already-exported component runs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from models.inference_masks import flatten_mask_values
from scripts.visualization_comparison import (
    ComparisonExport, Variant, complete_component, compose_panels, derive_components,
    require_row, require_same_selection, scene_title, selection_list, source_components,
    validate_metadata_definitions,
)


def strict_long_tables(raw, subject, components, kind, frames):
    """Resolve original semantic keys before any legacy pivot can discard duplicates."""
    if kind == "hawkeye":
        from datatools.hawkeye import _resolve_component_object_id
        lookup = subject.frame_meta.reset_index()[["frame_id", "abs_time", "possession_team"]]
        lookup = lookup.loc[lookup.frame_id.isin(frames)]
        if lookup.duplicated(["abs_time", "possession_team"]).any():
            raise ValueError("Ambiguous Hawkeye frame identities.")
        raw = raw.merge(lookup, on=["abs_time", "possession_team"], how="inner", validate="many_to_one")
    else:
        from datatools.benchmark import _resolve_component_object_id
        raw = raw.copy()
        raw["frame_id"] = int(subject.frame_meta.index.min())
    result = {}
    for component in components:
        if component not in raw:
            raise ValueError(f"Missing exported component {component}.")
        records = []
        for _, row in raw.loc[raw[component].notna()].iterrows():
            candidate = _resolve_component_object_id(row, subject, component)
            if candidate is None:
                raise ValueError(f"Cannot establish semantic candidate identity for {component}.")
            records.append({"frame_id": int(row.frame_id), "candidate": candidate, "value": float(row[component])})
        values = pd.DataFrame(records, columns=["frame_id", "candidate", "value"])
        if values.duplicated(["frame_id", "candidate"]).any():
            raise ValueError(f"Duplicate prediction keys for {component}.")
        result[component] = values.pivot(index="frame_id", columns="candidate", values="value")
        for frame in frames:
            require_row(result[component], frame, component)
    return result


def _physical(args, mod, subject, frames, *, dataset, cache_match=None):
    cache = args.physical_cache_dir or str(mod.pc_versions.cache_dir(dataset, args)
        if args.pc_xpass or dataset == "hawkeye_loc" else mod.get_runtime_physical_xpass_dir(dataset))
    kwargs = {}
    if dataset == "skillcorner" and args.pc_xpass:
        kwargs["cache_selection"] = mod.match_cache.selection_for(subject)
    return mod.load_runtime_physical_xpass_visualization_table(
        cache, str(cache_match or subject.match_id), frames, metric=mod.physical_xpass_metric(args),
        x_pass_version=args.x_pass_version, **kwargs,
    )


def _overlays(args, mod, tracking, subject, overlay_data):
    scores, labels, stats = mod.build_situation_overlays(
        overlay_data, str(subject.situation_id), tracking, subject, selection_format=args.selection_format,
    )
    return dict(coach_scores=scores, selection_labels=labels, selections_enabled=bool(args.selections),
                selection_denominators=stats["selection_denominators"])


def location_key(row, mode):
    from scripts.visualize_hawkeye import visualization_position_offsets
    x, y = visualization_position_offsets(row, mode)
    values = (float(row["resolved_time_norm"]), x, y)
    if not np.isfinite(values).all():
        raise ValueError("Hawkeye location comparison requires finite resolved times and geometry.")
    return (str(row["_action_id"]), *values)


def validate_hawkeye_geometry(raw_frames):
    """Compare available original geometry on common semantic player/frame keys."""
    keys = ["id", "abs_time", "team", "uefa_player_id"]
    columns = [column for column in ("centroid_x", "centroid_y", "smooth_x_speed", "smooth_y_speed", "role", "PlayerID")
               if all(column in frame for frame in raw_frames)]
    if not columns:
        return
    reference = None
    for raw in raw_frames:
        rows = raw.loc[raw.team.notna(), keys + columns].drop_duplicates()
        if rows.duplicated(keys).any():
            raise ValueError("Ambiguous exported Hawkeye scene geometry.")
        geometry = rows.set_index(keys).sort_index()
        if reference is None:
            reference = geometry
            continue
        common = reference.index.intersection(geometry.index)
        try:
            pd.testing.assert_frame_equal(reference.loc[common], geometry.loc[common], check_dtype=False,
                                          check_exact=False, rtol=1e-10, atol=1e-10)
        except AssertionError as exc:
            raise ValueError("Hawkeye component runs have different original scene geometry.") from exc


def hawkeye_groups(args, mod, exports, metadata, names):
    tracking = mod.clean_hawkeye_tracking(mod.load_hawkeye_tracking(args.tracking_csv))
    ball = mod.clean_hawkeye_ball(mod.load_hawkeye_ball(args.ball_csv))
    overlay_data = mod.load_overlay_data(include_coach_ratings=args.coach_ratings, include_selections=args.selections)
    mode = args.mode
    if mode in {"loc", "freeze"}:
        if any(item.get("inference_mode") != mode for item in metadata):
            raise ValueError(f"Every selected component run must use inference_mode={mode}.")
        selections = [mod.select_location_component_rows(raw, situation_ids=args.situation_id, limit=args.limit)
                      for raw in exports]
        indexed = []
        for rows in selections:
            mapping = {}
            for _, row in rows.iterrows():
                key = location_key(row, mode)
                if key in mapping:
                    raise ValueError("Ambiguous duplicate Hawkeye location scene identities.")
                mapping[key] = row
            indexed.append(mapping)
        require_same_selection([list(items) for items in indexed], "Hawkeye location scenes and geometry")
        entries = [(key, [items[key] for items in indexed]) for key in indexed[0]]
    else:
        selections = [mod.resolve_hawkeye_component_situation_ids(raw, metadata=item, requested_ids=args.situation_id)
                      for raw, item in zip(exports, metadata)]
        require_same_selection(selections, "Hawkeye situations")
        entries = [(identifier, None) for identifier in selections[0]]
    if args.coach_ratings:
        ids = [str(key[0] if rows is not None else key) for key, rows in entries]
        eligible, _ = mod.filter_coach_rated_situation_ids(ids, overlay_data.coach_ratings)
        entries = [(key, rows) for key, rows in entries if str(key[0] if rows is not None else key) in set(eligible)]
    if not entries:
        raise ValueError("No selected Hawkeye comparison scenes remain.")
    for key, location_rows in entries:
        identifier = str(key[0] if location_rows is not None else key)
        scene_tracking = tracking.loc[tracking.id.eq(identifier)].copy()
        if scene_tracking.empty:
            raise ValueError(f"Missing tracking for Hawkeye situation {identifier}.")
        kwargs = {}
        if location_rows is not None:
            scene_tracking, _ = mod.apply_hawkeye_possessor_offset(scene_tracking, offset_x=key[2], offset_y=key[3])
            kwargs["align_frozen_ball_to_possessor"] = True
        subject, _, _ = mod.build_hawkeye_situation(
            scene_tracking, ball, freeze_ballreceipt=True if location_rows is not None else bool(metadata[0].get("freeze_ballreceipt", True)),
            build_graphs=False, **kwargs,
        )
        times = [key[1]] if location_rows is not None else mod.requested_time_norms(args)
        selected_frames = mod.resolve_hawkeye_png_frames(subject, mod.resolve_ballreceipt(scene_tracking), times)
        frames = list(dict.fromkeys(int(item["frame_id"]) for item in selected_frames))
        tables, selected_raw = [], []
        for index, raw in enumerate(exports):
            rows = raw.loc[raw.id.eq(identifier)].copy()
            if location_rows is not None:
                row_id = int(location_rows[index]["selection_row_id"])
                rows = rows.loc[pd.to_numeric(rows.selection_row_id).eq(row_id)]
                # Verify all candidate rows agree on the location metadata, before selecting their values.
                for column in ("original_action_id", "PositionX", "PositionY", "resolved_time_norm"):
                    if rows[column].nunique(dropna=False) != 1:
                        raise ValueError(f"Inconsistent Hawkeye location geometry in {column}.")
            table = strict_long_tables(rows, subject, source_components(names), "hawkeye", frames)
            selected_raw.append(rows.loc[rows.abs_time.isin(subject.frame_meta.loc[frames, "abs_time"])])
            if "physical_xpass" in names:
                if location_rows is not None and not args.physical_cache_dir and not getattr(args, "pc_xpass_id", None):
                    args.pc_xpass_id = metadata[index].get("pc_xpass_id")
                table["physical_xpass"] = _physical(args, mod, subject, frames,
                    dataset="hawkeye_loc" if location_rows is not None else "hawkeye",
                    cache_match=str(location_rows[index]["loc_situation_id"]) if location_rows is not None else None)
            tables.append(derive_components(table, names))
        validate_hawkeye_geometry(selected_raw)
        overlay_kwargs = _overlays(args, mod, scene_tracking, subject, overlay_data)
        scene_id = identifier if location_rows is None else f"{identifier}@{key[1]:.17g},{key[2]:.17g},{key[3]:.17g}"
        folder = identifier if location_rows is None else f"{identifier}_{int(location_rows[0]['selection_row_id'])}"
        for selected in selected_frames:
            frame = int(selected["frame_id"])
            time = float(subject.frame_meta.loc[frame, "abs_time"])
            yield [{"scene": scene_id, "frame": frame, "tables": tables,
                    "folder": folder, "suffix": "" if location_rows is not None else "_" + selected["label"],
                    "selection_denominators": overlay_kwargs["selection_denominators"] if args.selections else None,
                    "title": f"{identifier} | {time:.3f}", "geometry": {"location_key": key, "selected_frame": selected},
                    "render": lambda component, row, subject=subject, frame=frame, overlays=overlay_kwargs:
                        mod.render_frame_image(subject, frame, component, row, show_trajectories=args.show_trajectories,
                                               comparison=True, **overlays)}]


def benchmark_groups(args, mod, exports, metadata, names):
    selections = [mod.resolve_benchmark_component_states(raw, metadata=item,
                    requested_modifications=args.modification, requested_game_states=args.game_state)
                  for raw, item in zip(exports, metadata)]
    require_same_selection(selections, "benchmark modification/state pairs")
    modifications = sorted({pair[0] for pair in selections[0]})
    if any(not {(item, 1), (item, 2)}.issubset(selections[0]) for item in modifications):
        raise ValueError("Benchmark comparison requires both game states 1 and 2 for each selected modification.")
    for modification in modifications:
        data = mod.load_benchmark_modification_data(modification, args.input_dir)
        group = []
        for state_id in (1, 2):
            subject, _, _ = mod.build_benchmark_state(data[f"game_state_{state_id}"], modification_id=modification,
                game_state_id=state_id, higher_state_id=int(data["higher_state_id"]), build_graphs=False)
            frame = int(subject.frame_meta.index.min())
            tables = []
            for raw in exports:
                rows = raw.loc[raw.modification.eq(modification) & raw.game_state.eq(state_id)]
                # The exported original state must agree with the common scene, not only its numeric ID.
                state_data = data[f"game_state_{state_id}"]
                geometry_columns = [column for column in ("team", "player", "pos_x", "pos_y", "pos_z",
                                    "smooth_x_speed", "smooth_y_speed", "event_player") if column in rows and column in state_data]
                if geometry_columns:
                    original = state_data[geometry_columns].drop_duplicates().sort_values(geometry_columns).reset_index(drop=True)
                    exported = rows.loc[rows.team.notna(), geometry_columns].drop_duplicates().sort_values(geometry_columns).reset_index(drop=True)
                    try:
                        pd.testing.assert_frame_equal(original, exported, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
                    except AssertionError as exc:
                        raise ValueError("Benchmark component run geometry differs from the selected state.") from exc
                table = strict_long_tables(rows, subject, source_components(names), "benchmark", [frame])
                if "physical_xpass" in names:
                    table["physical_xpass"] = _physical(args, mod, subject, [frame], dataset="benchmark")
                tables.append(derive_components(table, names))
            group.append({"scene": f"modification_{modification}/game_state_{state_id}", "frame": frame,
                "tables": tables, "folder": f"modification_{modification}", "suffix": "",
                "title": f"Modification {modification}", "geometry": {"game_state": state_id},
                "render": lambda component, row, subject=subject: mod.render_state_image(
                    subject, component, row, show_trajectories=args.show_trajectories, comparison=True)})
        yield group


def skillcorner_groups(args, mod, exports, metadata, names):
    context = mod.build_skillcorner_match_context(args.match_id, args.input_dir)
    for index in mod.resolve_indices(args):
        subject, _ = mod.build_skillcorner_possession(context, index)
        selected_frames = mod.resolve_skillcorner_png_frames(list(subject.frame_meta.index), args)
        if not selected_frames:
            raise ValueError(f"No frames for SkillCorner possession {index}.")
        frames = [int(item["frame_id"]) for item in selected_frames]
        tables, scope_reference = [], {}
        for raw_tables in exports:
            table = {}
            for component in source_components(names):
                raw = raw_tables.get(component)
                if raw is None or raw.empty:
                    raise ValueError(f"Missing SkillCorner component {component}.")
                rows = raw.loc[raw["index"].eq(index) & raw.frame.isin(frames)].copy()
                if rows.duplicated(["index", "frame"]).any():
                    raise ValueError(f"Duplicate SkillCorner prediction keys for {component}.")
                rows = rows.set_index("frame")
                if component not in scope_reference:
                    scope_reference[component] = rows
                else:
                    reference = scope_reference[component]
                    columns = [column for column in ("frame_role", "original_start_frame", "original_end_frame",
                               "period", "player_id", "attacking_side") if column in rows and column in reference]
                    if columns:
                        try:
                            pd.testing.assert_frame_equal(reference.reindex(frames)[columns], rows.reindex(frames)[columns],
                                                          check_dtype=False)
                        except AssertionError as exc:
                            raise ValueError(f"Incompatible SkillCorner frame scope for {component}.") from exc
                predictions = {}
                for frame in frames:
                    if frame not in rows.index:
                        raise ValueError(f"Missing selected SkillCorner frame {frame} for {component}.")
                    predictions[frame] = mod.build_visualization_probs(rows.loc[frame])
                    if component == "action_intent" and pd.notna(rows.loc[frame].get("shot")):
                        predictions[frame]["home_goal"] = float(rows.loc[frame, "shot"])
                table[component] = pd.DataFrame.from_dict(predictions, orient="index")
            if "physical_xpass" in names:
                table["physical_xpass"] = _physical(args, mod, subject, frames, dataset="skillcorner")
            tables.append(derive_components(table, names))
        for selected in selected_frames:
            frame = int(selected["frame_id"])
            yield [{"scene": f"{args.match_id}/possession_{index}", "frame": frame, "tables": tables,
                "folder": str(Path(str(args.match_id)) / str(index)), "suffix": "_" + selected["label"],
                "title": f"{args.match_id} | possession {index} | frame {frame}", "geometry": {"selected_frame": selected},
                "render": lambda component, row, subject=subject, frame=frame: mod.render_frame_image(
                    subject, frame, component, row.drop(index="home_goal", errors="ignore"),
                    show_trajectories=args.show_trajectories, comparison=True)}]


def run_comparison(args, mod):
    kind = Path(mod.__file__).stem.removeprefix("visualize_")
    identifiers = selection_list(args.component_run_id)
    variants, exports, metadata = [], [], []
    for identifier in identifiers:
        if kind == "hawkeye":
            getter = mod.get_hawkeye_loc_component_run_root if args.mode in {"loc", "freeze"} else mod.get_hawkeye_component_run_root
            folder = getter(identifier)
            raw, info = mod.load_hawkeye_component_run(folder)
        elif kind == "benchmark":
            folder = mod.get_benchmark_component_run_root(identifier)
            raw, info = mod.load_benchmark_component_run(folder)
        else:
            folder = mod.get_skillcorner_component_run_root(identifier)
            info = mod.load_run_metadata(folder, required=True)
            raw = mod.load_skillcorner_component_tables(folder, args.match_id)
        complete_component(info, identifier)
        recorded_masks = info.get("inference_masks") or {}
        requested_masks = flatten_mask_values(recorded_masks.get("requested", [])) if isinstance(recorded_masks, dict) else []
        variants.append(Variant(identifier, identifier,
            {"component_run_id": identifier, "component_dir": str(folder), "metadata": info,
             "inference_masks": recorded_masks}, requested_masks))
        exports.append(raw)
        metadata.append(info)
    validate_metadata_definitions([
        {**item, "freeze_ballreceipt": bool(item.get("freeze_ballreceipt", True))} if kind == "hawkeye" else item
        for item in metadata])
    names = mod.resolve_component_selection(args).rendered_components
    if args.show_physical_xpass:
        names = [*names, "physical_xpass"]
    root = Path(args.output_dir) / (args.run_id or mod.generate_run_id(kind + "_visualization_comparison"))
    report = ComparisonExport(root, variants, args, "components")
    report.metadata.update(run_id=root.name, script=Path(mod.__file__).name)
    adapters = {"hawkeye": hawkeye_groups, "benchmark": benchmark_groups, "skillcorner": skillcorner_groups}
    try:
        for group in adapters[kind](args, mod, exports, metadata, names):
            report.scenes.extend({"scene_id": scene["scene"], "frame_id": scene["frame"], **scene["geometry"]} for scene in group)
            for component in list(dict.fromkeys([*names, *source_components(names)])):
                images, labels = [], []
                for scene in group:
                    rows = [require_row(table.get(component), scene["frame"], component) for table in scene["tables"]]
                    rows = report.record(scene["scene"], scene["frame"], component, rows)
                    if component in names:
                        images.extend(scene["render"](component, row) for row in rows)
                        labels.extend(entry.label for entry in variants)
                if component not in names:
                    continue
                image = compose_panels(images, labels, scene_title(group[0], component),
                    columns=len(variants) if kind == "benchmark" else 2,
                    row_labels=["Game state 1", "Game state 2"] if kind == "benchmark" else None)
                path = root / group[0]["folder"] / f"{component}{group[0]['suffix']}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                report.outputs.append(str(path.resolve()))
        report.finish()
    except Exception as exc:
        report.failed(exc)
        raise
    print(f"Saved {kind} comparison visualizations and prediction tables to {root}")
