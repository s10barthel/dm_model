"""Compose existing pass-success views without changing their predictions."""
import json
from pathlib import Path

from datatools.viz_helpers import stack_images_vertically


COMPARISON_COMPONENT = "compare_pass_success"
COMPARISON_INPUTS = ("model_pass_success", "physical_xpass", "pass_success")


def compare_pass_success_requested(args):
    return bool(getattr(args, "compare_pass_success", False) or
                getattr(args, "only_compare_pass_success", False))


def needs_physical_xpass(args):
    selection = getattr(args, "_component_selection", None)
    if selection is not None:
        return bool({"physical_xpass", COMPARISON_COMPONENT}.intersection(selection.rendered_components))
    return bool(getattr(args, "show_physical_xpass", False) or
                getattr(args, "only_physical_xpass", False) or compare_pass_success_requested(args))


def needs_model_pass_success(args):
    selection = getattr(args, "_component_selection", None)
    if selection is not None:
        return bool({"model_pass_success", COMPARISON_COMPONENT}.intersection(selection.rendered_components))
    return bool(getattr(args, "show_model_pass_success", False) or compare_pass_success_requested(args))


def physical_xpass_panel_source(args, cache_dir):
    """Describe a rendered cache independently of pass-success model settings."""
    from physical_pass_model import PC_XPASS_SOURCE, PHYSICAL_XPASS_SOURCE
    metadata_path = Path(cache_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.is_file() else {}
    return metadata.get("source") or (PC_XPASS_SOURCE if getattr(args, "pc_xpass", False) else PHYSICAL_XPASS_SOURCE)


def input_components(components):
    """Expand a composite output into its inputs, preserving standalone selections."""
    names = [name for name in components if name != COMPARISON_COMPONENT]
    if COMPARISON_COMPONENT in components:
        names.extend(name for name in COMPARISON_INPUTS if name not in names)
    return names


def save_component_pngs(components, render, output_dir, suffix=""):
    """Render each input once and save only selected standalone/composite PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if COMPARISON_COMPONENT not in components:
        paths = []
        for name in components:
            path = output_dir / f"{name}{suffix}.png"
            render(name).save(path)
            paths.append(str(path.resolve()))
        return paths
    images = {}
    paths = []

    def get_image(name):
        if name not in images:
            images[name] = render(name)
        return images[name]

    try:
        for name in components:
            path = output_dir / f"{name}{suffix}.png"
            if name == COMPARISON_COMPONENT:
                with stack_images_vertically([get_image(source) for source in COMPARISON_INPUTS]) as image:
                    image.save(path)
            else:
                get_image(name).save(path)
            paths.append(str(path.resolve()))
    finally:
        for image in images.values():
            image.close()
    return paths


def comparison_metadata(args, output_root):
    return {
        "compare_pass_success": compare_pass_success_requested(args),
        "only_compare_pass_success": bool(getattr(args, "only_compare_pass_success", False)),
        "compare_pass_success_components": list(COMPARISON_INPUTS) if compare_pass_success_requested(args) else [],
        "compare_pass_success_output_paths": [
            str(path.resolve()) for path in sorted(Path(output_root).rglob("compare_pass_success*.png"))
        ],
    }
