from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import generate_epv, main as pipeline_main, run_and_visualize_hawkeye
from scripts import run_benchmark, run_hawkeye, run_relevant_models, run_skillcorner, visualize_action_components


@pytest.mark.parametrize(
    "script,required",
    [
        (run_and_visualize_hawkeye, []),
        (run_relevant_models, []),
        (visualize_action_components, ["--match-id", "match"]),
        (generate_epv, []),
        (run_skillcorner, []),
    ],
)
def test_inference_consumers_accept_one_combined_mask(script, required):
    args = script.parse_args([*required, "--inference-mask", "all:accel", "all:goal-features"])
    assert args.inference_mask == [["all:accel", "all:goal-features"]]
    with pytest.raises(SystemExit):
        script.parse_args([*required, "--inference-mask-1", "accel"])


@pytest.mark.parametrize("script", [run_hawkeye, run_benchmark])
def test_legacy_inference_cli_accepts_combined_masks(script):
    with patch("sys.argv", [script.__name__, "--inference-mask", "all:accel"]):
        args = script.parse_args()
    assert args.inference_mask == [["all:accel"]]


def test_mask_listing_bypasses_required_match_selection():
    output = io.StringIO()
    with contextlib.redirect_stdout(output), pytest.raises(SystemExit) as exc:
        visualize_action_components.parse_args(["--list-inference-masks"])
    assert exc.value.code == 0
    assert "pos-node-features" in output.getvalue()


def test_pipeline_only_forwards_masks_to_learned_consumers():
    argv = [
        "scripts/main.py", "--target-family", "xt", "--return_type", "next_5",
        "--intended-receiver-mode", "angle_only", "--feature-run-id", "features",
        "--bundle-id", "bundle", "--inference-mask", "all:accel", "all:goal-features",
    ]
    with patch("sys.argv", argv):
        commands = pipeline_main.build_commands(pipeline_main.parse_args())
    consumers = {
        "scripts/evaluate_relevant_models.py", "scripts/run_relevant_models.py",
        "scripts/run_hawkeye.py", "scripts/run_benchmark.py", "scripts/run_skillcorner.py",
    }
    for command in commands:
        if command[1] in consumers:
            index = command.index("--inference-mask")
            assert command[index + 1:] == ["all:accel", "all:goal-features"]
        else:
            assert "--inference-mask" not in command


def test_pipeline_forwards_masks_to_epv_generation():
    argv = [
        "scripts/main.py", "--target-family", "epv", "--bundle-id", "bundle",
        "--skip-train", "--skip-features", "--inference-mask", "all:accel",
    ]
    with patch("sys.argv", argv):
        commands = pipeline_main.build_commands(pipeline_main.parse_args())
    epv_command = next(command for command in commands if command[1] == "scripts/generate_epv.py")
    assert epv_command[-2:] == ["--inference-mask", "all:accel"]


def test_direct_hawkeye_configures_masks_before_rendering_and_records_metadata(tmp_path):
    args = run_and_visualize_hawkeye.parse_args([
        "--only-pass-success", "--pass-success-model-id", "pass_success/run",
        "--situation-id", "situation", "--output-dir", str(tmp_path), "--run-id", "masked",
        "--inference-mask", "pos-node-features",
    ])
    model = SimpleNamespace(args={})
    mask_metadata = {"requested": ["pos-node-features"], "models": {"pass_success": {"status": "applied"}}}
    calls = []

    def configure(models, masks):
        assert models == {"pass_success": model}
        assert masks == [["pos-node-features"]]
        calls.append("configured")
        return mask_metadata

    def render(**kwargs):
        assert calls == ["configured"]
        assert kwargs["model_specs"] == {"pass_success": model}
        output_dir = kwargs["output_root"] / kwargs["situation_id"]
        output_dir.mkdir()
        output_path = output_dir / "pass_success.png"
        output_path.write_text("image", encoding="utf-8")
        return output_dir, {}, {"frame_ids": [0], "selected_frames": [], "output_paths": [str(output_path)]}

    with (
        patch.object(run_and_visualize_hawkeye, "parse_args", return_value=args),
        patch.object(run_and_visualize_hawkeye.torch.cuda, "is_available", return_value=False),
        patch.object(run_and_visualize_hawkeye, "load_hawkeye_tracking", return_value=pd.DataFrame()),
        patch.object(run_and_visualize_hawkeye, "clean_hawkeye_tracking", return_value=pd.DataFrame({"id": ["situation"]})),
        patch.object(run_and_visualize_hawkeye, "load_hawkeye_ball", return_value=pd.DataFrame()),
        patch.object(run_and_visualize_hawkeye, "clean_hawkeye_ball", return_value=pd.DataFrame()),
        patch.object(run_and_visualize_hawkeye, "resolve_model_selection", return_value=({"pass_success": "pass_success/run"}, {}, None)),
        patch.object(run_and_visualize_hawkeye, "load_model", return_value=model),
        patch.object(run_and_visualize_hawkeye, "configure_inference_masks", side_effect=configure),
        patch.object(run_and_visualize_hawkeye, "validate_model_graph_schemas", return_value={"add_v_edge_features": False}),
        patch.object(run_and_visualize_hawkeye, "render_situation", side_effect=render),
    ):
        run_and_visualize_hawkeye.main()
    metadata = json.loads((tmp_path / "masked" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["inference_masks"] == mask_metadata


def test_epv_does_not_silently_reuse_unmasked_artifacts(tmp_path):
    (tmp_path / "epv.csv").write_text("action_id,epv\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(overwrite=False, inference_mask=[["all:accel"]])
    with (
        patch.object(generate_epv, "parse_args", return_value=args),
        patch.object(generate_epv, "ensure_project_dirs"),
        patch.object(generate_epv, "EPV_DIR", tmp_path),
        pytest.raises(ValueError, match="--overwrite"),
    ):
        generate_epv.main()


def test_epv_uses_returned_pass_intent_predictions_for_blending():
    match = SimpleNamespace(actions=pd.DataFrame({"action_id": [11]}, index=[0]))
    models = {task: object() for task in generate_epv.REQUIRED_EPV_MODEL_TASKS}
    masked_intent = pd.DataFrame({"home_2": [0.25], "home_3": [0.75]}, index=[0])
    pass_success = pd.DataFrame({"home_2": [0.8], "home_3": [0.4]}, index=[0])
    zero = pd.DataFrame({"home_2": [0.0], "home_3": [0.0]}, index=[0])

    def infer(received_match, model, **kwargs):
        assert received_match is match
        if model is models["pass_intent"]:
            return masked_intent, None
        if model is models["pass_success"]:
            assert kwargs["pass_intent_probs"] is masked_intent
            return pass_success, None
        if model is models["outcome_scoring"]:
            return zero, zero + 1.0
        return zero, zero

    with patch.object(generate_epv, "inference_gnn", side_effect=infer):
        results = generate_epv.infer_match_epv(match, models, "cpu")
    assert results["action_id"].tolist() == [11]
    assert float(results.loc[0, "epv"]) == pytest.approx(0.5)
