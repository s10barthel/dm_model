from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import generate_xt
from scripts.migrate_xt_legacy import migrate_legacy_xt
from xt_versions import REQUIRED_FILES, resolve_xt_version, validate_xt_version_id


def make_xt_version(root: Path, version: str, *, status: str = "complete", matches=("m",)) -> Path:
    version_root = root / version
    (version_root / "matches").mkdir(parents=True)
    for name in REQUIRED_FILES:
        if name != "fit_metadata.json":
            (version_root / name).write_text(name, encoding="utf-8")
    for match_id in matches:
        (version_root / "matches" / f"{match_id}.csv").write_text("action_id,xT\n1,0.2\n", encoding="utf-8")
    (version_root / "fit_metadata.json").write_text(json.dumps({
        "version": version, "status": status, "split_manifest_id": "split",
    }), encoding="utf-8")
    return version_root


def test_selection_resolves_latest_once_and_rejects_incomplete_or_missing_matches(tmp_path):
    root = tmp_path / "xT"
    make_xt_version(root, "first")
    make_xt_version(root, "partial", status="partial")
    (root / "latest.json").write_text('{"version":"first"}', encoding="utf-8")
    selected = resolve_xt_version("latest", xt_dir=root, required_match_ids=["m"], split_manifest_id="split")
    assert selected.version == "first"
    assert selected.match_dir == root / "first" / "matches"
    assert selected.metadata_sha256 == hashlib.sha256((selected.root / "fit_metadata.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not a completed"):
        resolve_xt_version("partial", xt_dir=root)
    with pytest.raises(FileNotFoundError, match="missing required outputs"):
        resolve_xt_version("first", xt_dir=root, required_match_ids=["other"])
    with pytest.raises(ValueError, match="uses split"):
        resolve_xt_version("first", xt_dir=root, split_manifest_id="other")
    for invalid in ("../outside", "a/b", ".staging", "latest.json"):
        with pytest.raises(ValueError):
            validate_xt_version_id(invalid)


def test_generator_publishes_unique_versions_and_keeps_partial_builds_unselectable(tmp_path, monkeypatch):
    root = tmp_path / "xT"
    events = tmp_path / "event_synced"
    events.mkdir()
    (events / "m.csv").write_text("event", encoding="utf-8")
    args = SimpleNamespace(version="v1", fit_only=False, match_id=None, limit=None,
        reuse_source_grid=False, source_xt_version=None, source_grid_l=12, source_grid_w=8,
        use_interaction=False, use_nonlinear=None)
    monkeypatch.setattr(generate_xt, "ROOT", tmp_path)
    monkeypatch.setattr(generate_xt, "XT_DIR", root)
    monkeypatch.setattr(generate_xt, "EVENT_SYNCED_DIR", events)
    monkeypatch.setattr(generate_xt, "parse_args", lambda: args)
    monkeypatch.setattr(generate_xt, "ensure_project_dirs", lambda: root.mkdir(exist_ok=True))
    monkeypatch.setattr(generate_xt, "load_split_manifest", lambda **kw: {
        "manifest_id": "split", "metadata": {}, "train": ["m"]})
    monkeypatch.setattr(generate_xt, "split_selector", lambda value: {})
    monkeypatch.setattr(generate_xt, "split_metadata", lambda value: {})
    monkeypatch.setattr(generate_xt, "fit_actions_for_train_split", lambda ids: (pd.DataFrame({"x": [1]}), ["m"], []))
    monkeypatch.setattr(generate_xt, "collect_export_matches", lambda ids: ([pd.DataFrame({"m": [1]})], ["m"], []))
    monkeypatch.setattr(generate_xt, "fit_xt_surface", lambda *a, **kw: "grid")
    fit = SimpleNamespace(source_grid=SimpleNamespace(shape=(8, 12)), projection_grid=None,
        model=SimpleNamespace(terms=("x",), to_metadata=lambda: {}),
        fit_sample=pd.DataFrame({"x": [1]}), diagnostics={})
    monkeypatch.setattr(generate_xt, "fit_xt_xy_glm", lambda *a, **kw: fit)

    def save_model(_fit, output):
        output.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            if name not in {"fit_metadata.json", "xT.csv"}:
                (output / name).write_text(name, encoding="utf-8")

    def save_exports(_events, _fit, output, sidecars):
        sidecars.mkdir(parents=True)
        (sidecars / "m.csv").write_text("action_id,xT\n1,0.2\n", encoding="utf-8")
        (output / "xT.csv").write_text("xT\n0.2\n", encoding="utf-8")
        return ["m"], [], []

    monkeypatch.setattr(generate_xt, "save_model_outputs", save_model)
    monkeypatch.setattr(generate_xt, "save_export_outputs", save_exports)
    generate_xt.main()
    assert resolve_xt_version("v1", xt_dir=root).metadata["status"] == "complete"
    assert json.loads((root / "latest.json").read_text())["version"] == "v1"
    with pytest.raises(FileExistsError):
        generate_xt.main()
    assert (root / "v1" / "xT.csv").read_text() == "xT\n0.2\n"
    args.version, args.fit_only = "fit_only", True
    generate_xt.main()
    assert json.loads((root / "fit_only" / "fit_metadata.json").read_text())["status"] == "partial"
    with pytest.raises(ValueError, match="not a completed"):
        resolve_xt_version("fit_only", xt_dir=root)
    assert json.loads((root / "latest.json").read_text())["version"] == "v1"


def test_generator_failure_does_not_publish_a_version(tmp_path, monkeypatch):
    root = tmp_path / "xT"
    root.mkdir()
    args = SimpleNamespace(version="broken")
    monkeypatch.setattr(generate_xt, "parse_args", lambda: args)
    monkeypatch.setattr(generate_xt, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr(generate_xt, "load_split_manifest", lambda **kw: {})
    monkeypatch.setattr(generate_xt, "split_selector", lambda value: {})
    monkeypatch.setattr(generate_xt, "XT_DIR", root)
    monkeypatch.setattr(generate_xt, "_build_version", lambda *a: (_ for _ in ()).throw(RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        generate_xt.main()
    assert not (root / "broken").exists()
    assert not (root / ".staging" / "broken").exists()


def test_legacy_migration_moves_artifacts_and_rewrites_metadata(tmp_path):
    root = tmp_path / "xT"
    root.mkdir()
    for name in REQUIRED_FILES:
        if name != "fit_metadata.json":
            (root / name).write_text(name, encoding="utf-8")
    (root / "fit_metadata.json").write_text(json.dumps({
        "fit_only": False, "split_manifest_id": "split",
        "source_grid_path": "data/xT/xT_source_grid.csv",
        "output_files": {"xT": "data/xT/xT.csv"},
    }), encoding="utf-8")
    (root / "matches").mkdir()
    (root / "matches" / "m.csv").write_text("action_id,xT\n1,0.2\n", encoding="utf-8")
    events = tmp_path / "events"
    events.mkdir()
    (events / "m.csv").write_text("events", encoding="utf-8")
    destination = migrate_legacy_xt(xt_dir=root, event_dir=events)
    assert destination == root / "legacy"
    assert not (root / "matches").exists()
    selected = resolve_xt_version("legacy", xt_dir=root, required_match_ids=["m"])
    assert selected.metadata["source_grid_path"] == "data/xT/legacy/xT_source_grid.csv"
    assert selected.metadata["output_files"]["xT"] == "data/xT/legacy/xT.csv"
    with pytest.raises(FileExistsError):
        migrate_legacy_xt(xt_dir=root, event_dir=events)


def test_feature_worker_commands_pin_the_selected_xt_version(monkeypatch):
    from scripts import generate_relevant_features as features

    monkeypatch.setattr(sys, "argv", ["generate_relevant_features.py", "--xt-version", "legacy"])
    args = features.parse_args()
    args.run_id = "fixture_feature"
    commands = [features.with_mode_flags(step.command, args)
                for step in features.full_generation_commands(sys.executable)]
    assert commands
    assert all(command[command.index("--xt-version") + 1] == "legacy" for command in commands)


def test_pc_xpass_surface_comes_from_selected_version(tmp_path, monkeypatch):
    import physical_pass_model as physics

    root = tmp_path / "selected"
    root.mkdir()
    surface = root / "xT_xy_surface.csv"
    pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1],
                  "xT": [0.1, 0.2, 0.3, 0.4]}).to_csv(surface, index=False)
    monkeypatch.setattr(physics, "resolve_xt_version", lambda version: SimpleNamespace(root=root))
    metadata = physics.pc_xpass_metadata(
        physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, top_xt=True, xt_version="selected")
    assert metadata["xt_version"] == "selected"
    assert metadata["xt_surface"]["path"] == str(surface)
    assert metadata["xt_surface"]["sha256"] == hashlib.sha256(surface.read_bytes()).hexdigest()


def test_pc_xpass_top_xt_inherits_default_sportec_feature_version(monkeypatch):
    from scripts import generate_physical_xpass as cli

    monkeypatch.setattr(cli, "resolve_feature_run_id", lambda *args, **kwargs: "feature_selected")
    monkeypatch.setattr(cli, "load_feature_run_metadata", lambda *args, **kwargs: {"xt_version": "selected"})
    monkeypatch.setattr(cli, "resolve_xt_version", lambda version: SimpleNamespace(version=version))
    args = cli.parse_args(["--pc-xpass", "--top-xt"])
    assert args.xt_version == "selected"


def test_pinned_sidecars_cannot_silently_fall_back_to_event_columns(tmp_path):
    from datatools.xt import merge_xt_annotations

    events = pd.DataFrame({"action_id": [1, 2], "xT": [0.9, 0.9]})
    with pytest.raises(FileNotFoundError, match="Pinned xT"):
        merge_xt_annotations(events, "m", tmp_path, strict=True)
    sidecar = tmp_path / "m.csv"
    pd.DataFrame({"action_id": [1], "xT": [0.1], "scores_xT": [0.2],
                  "concedes_xT": [0.0]}).to_csv(sidecar, index=False)
    with pytest.raises(ValueError, match="does not cover"):
        merge_xt_annotations(events, "m", tmp_path, strict=True)
    pd.DataFrame({"action_id": [1, 2], "xT": [0.1, 0.2], "scores_xT": [0.2, 0.3],
                  "concedes_xT": [0.0, 0.0]}).to_csv(sidecar, index=False)
    assert merge_xt_annotations(events, "m", tmp_path, strict=True)["xT"].tolist() == [0.1, 0.2]


def test_xt_cli_requires_source_version_for_reuse_and_has_no_overwrite(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_xt.py", "--reuse-source-grid"])
    with pytest.raises(SystemExit):
        generate_xt.parse_args()
    monkeypatch.setattr(sys, "argv", ["generate_xt.py", "--reuse-source-grid",
                                      "--source-xt-version", "legacy"])
    assert generate_xt.parse_args().source_xt_version == "legacy"
    monkeypatch.setattr(sys, "argv", ["generate_xt.py", "--overwrite"])
    with pytest.raises(SystemExit):
        generate_xt.parse_args()


def test_pipeline_passes_one_new_xt_version_to_generation_and_features(monkeypatch):
    from scripts import main as pipeline

    monkeypatch.setattr(sys, "argv", ["main.py", "--target-family", "xt", "--return_type", "next_3",
        "--skip-preprocess", "--skip-train", "--skip-evaluate", "--skip-run-relevant",
        "--skip-hawkeye", "--skip-benchmark", "--skip-skillcorner"])
    args = pipeline.parse_args()
    commands = pipeline.build_commands(args)
    xt_command = next(command for command in commands if command[1] == "scripts/generate_xt.py")
    feature_command = next(command for command in commands if command[1] == "scripts/generate_relevant_features.py")
    version = xt_command[xt_command.index("--version") + 1]
    assert version.startswith("xt_")
    assert feature_command[feature_command.index("--xt-version") + 1] == version
    assert "--overwrite" not in xt_command


def test_pc_xpass_cli_inherits_sportec_feature_run_xt_version(monkeypatch):
    from scripts import generate_physical_xpass as pc

    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: "feature_run")
    monkeypatch.setattr(pc, "load_feature_run_metadata", lambda *a, **kw: {"xt_version": "legacy"})
    args = pc.parse_args(["--pc-xpass", "--top-xt", "--sportec-feature-run-id", "feature_run"])
    assert args.xt_version == "legacy"
