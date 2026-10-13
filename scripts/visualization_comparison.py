"""Shared selection, alignment, PNG composition, and exports for visual comparisons."""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from models.inference_masks import flatten_mask_values, resolve_inference_mask_configurations
from project_config import write_run_metadata
from scripts.comparison_warnings import ComparisonWarnings, ComparisonCancelled, add_warning_argument, metadata_differences

MODEL_TASKS = ("action_intent", "pass_intent", "success_intent", "pass_success", "pass_height",
               "outcome_scoring", "outcome_conceding")


class RepeatableSelection(argparse.Action):
    """Preserve the scalar value of existing single-selection invocations."""
    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, self.dest, None)
        setattr(namespace, self.dest, values if previous is None else selection_list(previous) + [values])


def selection_list(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def add_compare_argument(parser):
    add_warning_argument(parser)
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple selected IDs in PNG panels and export aligned prediction tables.")
    parser.epilog = "Comparisons export PNG panels and aligned prediction tables; existing single-result output is preserved."


def validate_comparison_args(parser, args, *, inference=False):
    try:
        mode = None
        if inference:
            configurations = resolve_inference_mask_configurations(args)
            numbered = any(entry["id"] != "mask" for entry in configurations)
            selections = {task: selection_list(getattr(args, f"{task}_model_id", None)) for task in MODEL_TASKS}
            for task, ids in selections.items():
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Duplicate {task} checkpoint selection.")
                for identifier in ids:
                    if identifier.split("/", 1)[0] != task or "/" not in identifier:
                        raise ValueError(f"Checkpoint {identifier!r} does not match selector task {task!r}.")
            repeated = [task for task, ids in selections.items() if len(ids) > 1]
            if numbered:
                if args.compare:
                    raise ValueError("Numbered inference masks already enable mask comparison; omit --compare.")
                if repeated:
                    raise ValueError("Numbered masks require one checkpoint per task, including fixed supporting models.")
                mode = "masks"
            elif args.compare:
                if len(repeated) != 1:
                    raise ValueError("--compare requires at least two checkpoints of exactly one varied model task.")
                mode = "checkpoints"
            elif repeated:
                raise ValueError("Repeated model selectors require --compare.")
        else:
            ids = selection_list(args.component_run_id)
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate component-run selection.")
            if args.compare:
                if len(ids) < 2:
                    raise ValueError("--compare requires at least two --component-run-id selections.")
                if args.component_dir:
                    raise ValueError("--compare requires component-run IDs, without --component-dir.")
                if getattr(args, "only_tracking", False):
                    raise ValueError("--compare cannot be combined with --only-tracking.")
                mode = "components"
            elif len(ids) > 1:
                raise ValueError("Repeated component-run IDs require --compare.")
        if mode and getattr(args, "output", "png") != "png":
            raise ValueError("Visualization comparisons support PNG output only.")
        args.visualization_comparison = mode
    except ValueError as exc:
        parser.error(str(exc))
    return args


@dataclass
class Variant:
    identifier: str
    label: str
    provenance: dict = field(default_factory=dict)
    masks: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)


def validate_metadata_definitions(metadata, policy=None, tasks=None):
    """Describe differences; independent visualization never requires equality."""
    keys = ("target_family", "return_type", "intended_receiver_mode", "use_carries", "freeze_ballreceipt",
            "inference_mode", "pass_height_definition", "label_source", "soft_target", "graph_schema")
    messages = metadata_differences(metadata, keys, prefix="Visualization ")
    models = [item.get("model_records") or item.get("models", {}) for item in metadata]
    for task in tasks or MODEL_TASKS:
        records = [item[task] for item in models if isinstance(item, dict) and isinstance(item.get(task), dict)]
        if records:
            messages.extend(validate_metadata_definitions(records))
    if policy is not None:
        for message in messages:
            policy.add(message)
    return messages


def require_same_selection(selections, description):
    if not selections or not selections[0]:
        raise ValueError(f"No {description} selected for comparison.")
    reference = set(selections[0])
    if any(set(items) != reference for items in selections[1:]):
        raise ValueError(f"Selected {description} differ across runs; specify matching selectors. No intersection is performed.")


def require_row(table, frame_id, component):
    if table is None or table.empty or frame_id not in table.index:
        raise ValueError(f"Missing selected frame {frame_id} for component {component}.")
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"Duplicate prediction keys for component {component}.")
    row = pd.to_numeric(table.loc[frame_id], errors="raise").astype(float).dropna()
    if row.empty:
        raise ValueError(f"No predictions at selected frame {frame_id} for component {component}.")
    return row


def align_rows(rows, labels, messages, context):
    """Align the union of semantic candidates; never equate missing with zero."""
    for row in rows:
        if row is None or row.empty:
            raise ValueError(f"Missing predictions for {context}.")
        if not row.index.is_unique or row.index.hasnans:
            raise ValueError(f"Duplicate or missing semantic candidate identities for {context}.")
        if not np.isfinite(pd.to_numeric(row.dropna(), errors="raise")).all():
            raise ValueError(f"Non-finite predictions for {context}.")
    memberships = [set(row.dropna().index.astype(str)) for row in rows]
    candidates = sorted(set().union(*memberships))
    if any(items != memberships[0] for items in memberships[1:]):
        message = f"Candidate membership differs for {context}: " + ", ".join(
            f"{label}={len(items)}" for label, items in zip(labels, memberships))
        messages.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return [row.rename(index=str).reindex(candidates) for row in rows]


def _font(size):
    # Matplotlib is already required and provides a portable, Unicode-capable font.
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _wrap(text, width, font):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for character in paragraph:
            trial = line + character
            if line and font.getlength(trial) > width:
                lines.append(line)
                line = character
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def compose_panels(images, labels, title, *, columns=2, row_labels=None, subtitle=None):
    """Fixed-size tiles, common header, and equal wrapped-title space."""
    if not images or len(images) != len(labels):
        raise ValueError("Expected one image and label per comparison panel.")
    width, height = max(item.width for item in images), max(item.height for item in images)
    columns = min(columns, len(images))
    rows = math.ceil(len(images) / columns)
    font = _font(max(14, width // 65))
    title_font = _font(max(18, width // 52))
    line_height = font.size + 8
    wrapped = [_wrap(label, width - 40, font) for label in labels]
    label_height = max(label.count("\n") + 1 for label in wrapped) * line_height + 20
    gutter = max(20, width // 50)
    row_gutter = int(max(font.getlength(label) for label in row_labels)) + 40 if row_labels else 0
    canvas_width = row_gutter + columns * width + (columns - 1) * gutter
    header = _wrap(title + ("\n" + subtitle if subtitle else ""), canvas_width - 40, title_font)
    header_height = (header.count("\n") + 1) * (title_font.size + 10) + 30
    canvas = Image.new("RGB", (canvas_width, header_height + rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((canvas_width / 2, 15), header, font=title_font, fill="black", anchor="ma", align="center", spacing=10)
    for index, (tile, label) in enumerate(zip(images, wrapped)):
        row, column = divmod(index, columns)
        x = row_gutter + column * (width + gutter)
        y = header_height + row * (height + label_height)
        draw.multiline_text((x + width / 2, y + 5), label, font=font, fill="black", anchor="ma", align="center", spacing=8)
        canvas.paste(tile.convert("RGB"), (x + (width-tile.width)//2, y + label_height + (height-tile.height)//2))
    if row_labels:
        for row, label in enumerate(row_labels):
            draw.multiline_text((10, header_height + row * (height + label_height) + label_height + height / 2),
                                _wrap(label, row_gutter - 20, font), font=font, fill="black", anchor="lm")
    return canvas


def scene_title(scene, component):
    title = scene["title"] + " | " + component.replace("_", " ").title()
    denominators = scene.get("selection_denominators")
    if component == "pass_score" and denominators is not None:
        title += f" | Selections (CAVE n={int(denominators.get('CAVE', 0))} | HMD n={int(denominators.get('HMD', 0))})"
    return title


class ComparisonExport:
    def __init__(self, root, variants, args, mode):
        self.root = Path(root)
        self.variants = variants
        self.rows = [[] for _ in variants]
        self.warnings = []
        self.policy = ComparisonWarnings(getattr(args, "accept_comparison_warnings", False))
        self.export_aligned = True
        self.scenes = []
        self.outputs = []
        self.recorded = set()
        identities = [variant.identifier for variant in variants]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate comparison variant identities.")
        self.metadata = {"status": "running", "created_at": datetime.now().isoformat(timespec="seconds"),
                         "comparison_mode": mode, "arguments": vars(args), "variants": [
                             {"id": variant.identifier, "label": variant.label, "requested_masks": variant.masks,
                              "provenance": variant.provenance} for variant in variants], "warnings": self.warnings,
                         "scenes": self.scenes, "output_paths": self.outputs}

    def record(self, scene, frame_id, component, rows):
        if len(rows) != len(self.variants):
            raise ValueError("Every comparison variant must contribute a prediction row.")
        native = [row if row is not None else pd.Series(dtype=float) for row in rows]
        try:
            available = [row for row in native if not row.empty]
            candidates = sorted({str(candidate) for row in available for candidate in row.index})
            if any(not row.index.is_unique or row.index.hasnans or
                   not row.index.astype(str).is_unique for row in available):
                raise ValueError("ambiguous semantic candidate keys")
            if any(not np.isfinite(pd.to_numeric(row.dropna(), errors="raise")).all() for row in available):
                raise ValueError("non-finite prediction values")
            aligned = [row.rename(index=str).reindex(candidates) for row in native]
            memberships = [set(row.dropna().index.astype(str)) for row in native]
            if memberships and any(items != memberships[0] for items in memberships[1:]):
                message = f"Candidate membership differs for {scene}, frame {frame_id}, {component}: " + ", ".join(
                    f"{variant.label}={len(items)}" for variant, items in zip(self.variants, memberships))
                if message not in self.warnings:
                    self.warnings.append(message)
                    warnings.warn(message, UserWarning, stacklevel=2)
                    self.policy.add(message)
        except (TypeError, ValueError) as exc:
            self.export_aligned = False
            self.policy.add(f"Aligned prediction export unavailable for {scene}/{component}: {exc}. Native predictions and images are retained.")
            aligned = native
        key = (str(scene), int(frame_id), component)
        if key in self.recorded:
            return aligned
        self.recorded.add(key)
        for index, row in enumerate(aligned):
            for candidate, value in row.items():
                self.rows[index].append({"scene_id": str(scene), "frame_id": int(frame_id), "component": component,
                                         "candidate_id": str(candidate), "value": value, "available": bool(pd.notna(value))})
        return aligned

    def finish(self):
        self.root.mkdir(parents=True, exist_ok=True)
        aligned = None
        keys = ["scene_id", "frame_id", "component", "candidate_id"]
        for index, (variant, rows) in enumerate(zip(self.variants, self.rows)):
            frame = pd.DataFrame(rows)
            if frame.empty:
                frame = pd.DataFrame(columns=[*keys, "value", "available"])
            if frame.duplicated(keys).any():
                self.export_aligned = False
                self.policy.add(f"Aligned export unavailable: duplicate prediction identities for {variant.label}. Native rows are retained.")
            folder = "variant_" + hashlib.sha256(variant.identifier.encode()).hexdigest()[:12]
            path = self.root / "variants" / folder / "predictions.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                frame.to_parquet(path, index=False)
            except (ValueError, TypeError, OSError) as exc:
                path = path.with_suffix(".csv")
                frame.to_csv(path, index=False)
                self.policy.add(f"Parquet export unavailable for {variant.label}: {exc}; native CSV retained.")
            if self.export_aligned:
                values = frame.set_index(keys)["value"].rename("prediction:" + variant.identifier)
                aligned = values.to_frame() if aligned is None else aligned.join(values, how="outer", validate="one_to_one")
            self.metadata["variants"][index]["predictions"] = str(path.resolve())
        path = self.root / "aligned_predictions.csv"
        if self.export_aligned and aligned is not None:
            aligned.reset_index().to_csv(path, index=False, float_format="%.17g", na_rep="N/A")
            self.metadata["aligned_predictions"] = str(path.resolve())
        self.policy.confirm()
        self.metadata.update(self.policy.metadata(), status="completed", aligned_export_available=self.export_aligned)
        write_run_metadata(self.root, self.metadata)

    def failed(self, exc):
        self.metadata.update(self.policy.metadata(), status="cancelled" if isinstance(exc, ComparisonCancelled) else "failed", error=f"{type(exc).__name__}: {exc}")
        write_run_metadata(self.root, self.metadata)


def complete_component(metadata, identifier):
    if metadata.get("status") != "completed":
        raise ValueError(f"Component run {identifier!r} is not completed.")


def variant_description(variant):
    records = variant.provenance.get("models") or {}
    context = variant.provenance.get("source_context") or variant.provenance.get("metadata") or {}
    descriptions = []
    modes = list(dict.fromkeys(str(record["intended_receiver_mode"]) for record in records.values()
                              if record.get("intended_receiver_mode")))
    if not modes and context.get("intended_receiver_mode"):
        modes = [str(context["intended_receiver_mode"])]
    if modes:
        descriptions.append("Receiver mode: " + "/".join(modes))
    for task, record in records.items():
        if task.startswith("outcome_"):
            descriptions.append(f"{task}: {record.get('target_family', 'unknown')} / {record.get('return_type', 'unknown')}")
        if task == "pass_height" and record.get("pass_height_definition"):
            descriptions.append(f"Height target: >= {record['pass_height_definition'].get('threshold_meters')} m")
    runtime_mode = (variant.provenance.get("runtime_feature_context") or {}).get("intended_receiver_mode")
    if runtime_mode:
        descriptions.append("Runtime receiver: " + str(runtime_mode))
    if context.get("use_carries") is not None:
        descriptions.append("Carries: " + ("included" if context["use_carries"] else "excluded"))
    if context.get("inference_mode"):
        descriptions.append("Scene mode: " + str(context["inference_mode"]))
    if variant.masks:
        descriptions.append("Masked: " + ", ".join(variant.masks))
    return variant.label + ("\n" + " | ".join(descriptions) if descriptions else "")


def geometry_description(scene):
    geometry = scene.get("geometry", {})
    parts = []
    if "freeze_ballreceipt" in geometry:
        parts.append("Freeze at receipt: " + ("yes" if geometry["freeze_ballreceipt"] else "no"))
    key = geometry.get("location_key")
    if isinstance(key, (tuple, list)) and len(key) >= 4:
        parts.append(f"Time norm: {key[1]:g} | Offset: ({key[2]:g}, {key[3]:g})")
    for name in ("frame_role", "original_start_frame", "original_end_frame", "attacking_side"):
        if name in geometry:
            parts.append(f"{name}: {geometry[name]}")
    return " | ".join(parts)


def render_independent_groups(report, variant_groups, names, *, benchmark=False):
    """Compose standalone scenes; missing correspondence never vetoes images."""
    import matplotlib
    matplotlib.use("Agg")
    maps = []
    buckets = {}
    key_buckets = {}
    for index, groups in enumerate(variant_groups):
        scenes = {}
        for group in groups:
            for scene in group if isinstance(group, list) else [group]:
                key = (str(scene["scene"]), int(scene["frame"]))
                if key in scenes:
                    report.policy.add(f"Ambiguous scene identity {key} for {report.variants[index].label}; displayed separately.")
                    key = (key[0] + f"/variant_{index}_scene_{len(scenes)}", key[1])
                    report.export_aligned = False
                scenes[key] = scene
                report.scenes.append({"variant_id": report.variants[index].identifier,
                                      "scene_id": key[0], "frame_id": key[1], **scene.get("geometry", {})})
                bucket = key_buckets.setdefault(key, (scene["folder"], scene["suffix"]))
                if key not in buckets.setdefault(bucket, []):
                    buckets[bucket].append(key)
        maps.append(scenes)
    if any(set(items) != set(maps[0]) for items in maps[1:]):
        report.policy.add("Scene/frame membership differs. The union is displayed with unavailable panels; no correspondence is invented.")
    prepared = []
    for (folder, suffix), keys in buckets.items():
        for component in dict.fromkeys([*names, *(name for key in keys for scenes in maps
                                                if key in scenes for name in scenes[key]["tables"][0])]):
            images, labels = [], []
            reference_scene = None
            for key in keys:
                rows = []
                scene_geometries = []
                for index, scenes in enumerate(maps):
                    scene = scenes.get(key)
                    reference_scene = reference_scene or scene
                    if scene is None or component not in scene["tables"][0]:
                        rows.append(None)
                        if component in names:
                            images.append(None)
                            labels.append(variant_description(report.variants[index]) + "\nUnavailable for this scene/frame")
                        continue
                    table = scene["tables"][0][component]
                    row = table.loc[scene["frame"]]
                    rows.append(row)
                    scene_geometries.append(repr(scene.get("geometry", {})))
                    if component in names:
                        images.append(scene["render"](component, row))
                        labels.append(variant_description(report.variants[index]) + "\n" + scene["title"] +
                                      "\n" + geometry_description(scene))
                if len(set(scene_geometries)) > 1:
                    report.policy.add(f"Scene geometry/settings differ for {key}; each panel uses its own scene.")
                report.record(key[0], key[1], component, rows)
            if component not in names:
                continue
            size = next((item.size for item in images if item is not None), (800, 500))
            for index, item in enumerate(images):
                if item is None:
                    item = Image.new("RGB", size, "#eeeeee")
                    ImageDraw.Draw(item).text((30, 30), "Unavailable for this scene/frame", fill="black", font=_font(20))
                    images[index] = item
            prepared.append((folder, suffix, component, images, labels, reference_scene, len(keys)))
    report.policy.confirm()
    for folder, suffix, component, images, labels, scene, count in prepared:
        row_labels = ["Game state 1", "Game state 2"] if benchmark and count == 2 else None
        image = compose_panels(images, labels, scene_title(scene, component),
                               columns=len(report.variants) if benchmark else 2, row_labels=row_labels)
        path = report.root / folder / f"{component}{suffix}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        report.outputs.append(str(path.resolve()))


def source_components(names):
    from datatools.viz_helpers import DERIVED_COMPONENT_INPUTS
    return sorted({source for name in names if name != "physical_xpass"
                   for source in DERIVED_COMPONENT_INPUTS.get(name, (name,))})


def derive_components(components, names):
    from datatools.viz_helpers import compute_outcome, compute_pass_score, validate_derived_inputs
    validate_derived_inputs(names, components)
    result = dict(components)
    for name in names:
        if name in {"outcome_success", "outcome_failure"}:
            case = name.removeprefix("outcome_")
            result[name] = compute_outcome(result[f"outcome_scoring_{case}"], result[f"outcome_conceding_{case}"])
        elif name == "pass_score":
            result[name] = compute_pass_score(**{key: result[key] for key in source_components([name])})
    return result


def cached_prediction(cache, predict, scene, model, **kwargs):
    """Cache only learned predictions with identical checkpoints, masks, and dependencies."""
    if cache is None:
        return predict(scene, model, **kwargs)
    from models.inference_masks import inference_mask_metadata
    intent = kwargs.get("pass_intent_probs")
    intent_signature = None
    if intent is not None and model.args.get("xpass_weight") == "v5":
        intent_signature = (tuple(intent.columns), pd.util.hash_pandas_object(intent).values.tobytes())
    height = kwargs.get("pass_height_probs")
    height_signature = None if height is None else (tuple(height.columns), pd.util.hash_pandas_object(height).values.tobytes())
    auxiliary_signature = tuple((role, id(auxiliary), inference_mask_metadata(auxiliary)["configuration_id"],
                                 repr(sorted(auxiliary.args.items())))
                                for role, auxiliary in sorted(getattr(model, "_xpass_auxiliary_models", {}).items()))
    from pc_xpass_auxiliary import original_input_fingerprint
    source_graphs = kwargs.get("graph_override")
    source_labels = kwargs.get("label_override")
    if source_graphs is None:
        source_graphs = getattr(scene, "graph_features_1" if kwargs.get("post_action") else "graph_features_0", None)
        source_labels = getattr(scene, "labels", None)
    source_signature = (original_input_fingerprint(source_graphs, source_labels)
                        if source_graphs is not None and source_labels is not None else None)
    key = (id(model), inference_mask_metadata(model)["configuration_id"],
           id(scene), tuple(kwargs.get("event_indices") or ()), intent_signature,
           height_signature, auxiliary_signature, repr(sorted(model.args.items())),
           bool(kwargs.get("post_action")), source_signature)
    if key not in cache:
        cache[key] = predict(scene, model, **kwargs)
    elif kwargs.get("precomputed_auxiliaries") is not None:
        from pc_xpass_auxiliary import record_component_prediction
        record_component_prediction(
            kwargs["precomputed_auxiliaries"], scene, model, cache[key][0], kwargs.get("device", "cuda"),
            post_action=kwargs.get("post_action", False), graph_override=kwargs.get("graph_override"),
            label_override=kwargs.get("label_override"),
        )
    return tuple(value.copy(deep=True) if isinstance(value, pd.DataFrame) else value for value in cache[key])
