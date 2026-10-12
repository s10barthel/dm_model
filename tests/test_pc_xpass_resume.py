from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import pc_xpass_resume as resume
import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate
from test_physical_xpass import make_graph, make_label


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "pc_xpass")
    args = generate.parse_args([
        "--pc-xpass", "--pc-xpass-id", "resume", "--no-sportec", "--no-skillcorner",
        "--no-benchmark", "--min-speed", "3", "--max-speed", "3", "--angle-step", "90",
        "--radial-gridsize", "5", "--num-workers", "1",
    ])
    resume.prepare_contracts(args, ["hawkeye"])
    versions.start_generation(args)
    resume.persist_contracts(args)
    items = [{"match_id": "one", "graphs": [make_graph()], "labels": torch.stack([make_label()])}]
    for dataset in ("hawkeye", "benchmark"):
        generate.prewarm_runtime_items(items, cache_dir=versions.cache_dir(dataset, args), args=args)
    calls = []
    monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", lambda *a, **k: pytest.fail("Physical resume predicted height"))
    return args, items, calls
















def test_hawkeye_contract_inherits_and_rejects_conflict(setup_cache):
    args, _, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata["dataset_contracts"]["hawkeye"]["freeze_ballreceipt"] = False
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    omitted = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume"])
    resume.prepare_contracts(omitted, ["hawkeye"])
    assert omitted.freeze_ballreceipt is False
    explicit = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume", "--freeze-ballreceipt"])
    with pytest.raises(ValueError, match="freeze_ballreceipt"):
        resume.prepare_contracts(explicit, ["hawkeye"])


def test_sportec_unknown_source_requires_explicit_input(setup_cache, monkeypatch):
    args, _, _ = setup_cache
    path = resume.root(args) / "sportec" / "matches" / "unknown.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"action_index": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="Regenerate"):
        resume.prepare_contracts(args, ["sportec"])
    args.sportec_feature_run_id = "features"
    args._pc_explicit.add("sportec_feature_run_id")
    monkeypatch.setattr("project_config.resolve_feature_run_id", lambda *a, **kw: "features")
    with pytest.raises(ValueError, match="Regenerate"):
        resume.prepare_contracts(args, ["sportec"])




def test_contracts_for_automatically_named_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    args = generate.parse_args(["--pc-xpass", "--no-sportec"])
    resume.prepare_contracts(args, ["hawkeye"])
    versions.start_generation(args)
    resume.persist_contracts(args)
    assert versions.read_metadata(resume.root(args))["dataset_contracts"]["hawkeye"]["freeze_ballreceipt"] is True


def test_sportec_control_provenance_cannot_change_partition_identity(setup_cache):
    args, _, _ = setup_cache
    first = {"match_id": "match", "feature_run_id": "features", "control_sha256": "a"}
    resume.record_source(args, "sportec", "partition1", first)
    with pytest.raises(ValueError, match="source provenance"):
        resume.record_source(args, "sportec", "partition2", {**first, "control_sha256": "b"})


def test_lane_export_restored_and_selections_remain_flexible(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "lane", "--export-lane-control"])
    versions.start_generation(args)
    selected = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "lane", "--scope", "frames", "--frames", "2",
                                    "--no-skillcorner", "--num-workers", "3"])
    assert selected.export_lane_control
    assert selected.frames == 2 and selected.no_skillcorner and selected.num_workers == "3"






def test_changed_physical_input_cannot_overwrite_versioned_row(setup_cache):
    args, items, _ = setup_cache
    items[0]["graphs"][0].x[0, 0] += 1
    path = resume.root(args) / "hawkeye" / "matches" / "one.parquet"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Changed input state"):
        generate.prewarm_runtime_items(items, cache_dir=resume.root(args) / "hawkeye", args=args)
    assert path.read_bytes() == original






def test_changed_ranking_dependency_rejected_without_writes(setup_cache):
    args, _, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata["computation_dependencies"]["top_pass_definition"] = "other"
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    original = (resume.root(args) / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="dependencies changed"):
        resume.prepare_contracts(args, ["hawkeye"])
    assert (resume.root(args) / "metadata.json").read_bytes() == original


def test_main_marks_keyboard_interrupt_incomplete(setup_cache, monkeypatch):
    args, _, _ = setup_cache
    monkeypatch.setattr(generate, "parse_args", lambda: args)
    monkeypatch.setattr(generate, "run_runtime_mode", lambda a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        generate.main()
    assert versions.read_metadata(resume.root(args))["status"] == "incomplete"


def test_skillcorner_boundary_change_cannot_create_mixed_partition(setup_cache):
    args, _, _ = setup_cache
    first = {"match_id": "match", "possession_id": 1, "possessor_id": "home_1",
             "original_start_frame": 10, "original_end_frame": 20}
    resume.record_source(args, "skillcorner", "partition1", first)
    with pytest.raises(ValueError, match="possession definition"):
        resume.record_source(args, "skillcorner", "partition2", {**first, "original_end_frame": 21})


def test_state_contract_version_cannot_change(setup_cache):
    args, _, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata["dataset_contracts"]["skillcorner"] = {"state_contract": "other"}
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    with pytest.raises(ValueError, match="state contract"):
        resume.prepare_contracts(args, ["skillcorner"])




def test_resume_ignores_obsolete_height_metadata_and_never_inherits_model(setup_cache):
    args, items, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata["pass_height_enrichment"] = {"model_id": "pass_height/obsolete", "pending": {"model_id": "pass_height/obsolete"}}
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    resumed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume"])
    assert resumed.pass_height_model_id is None
    stats = generate.prewarm_runtime_items(items, cache_dir=resume.root(args) / "hawkeye", args=resumed)
    assert stats["cache_hits"] == 1
    versions.finish_generation(args)
    assert versions.read_metadata(resume.root(args))["status"] == "completed"


def test_dry_contract_validation_does_not_write(setup_cache):
    args, _, _ = setup_cache
    args.dry_run = True
    before = {path: path.read_bytes() for path in resume.root(args).rglob("*") if path.is_file()}
    resume.prepare_contracts(args, ["hawkeye"])
    resume.persist_contracts(args)
    assert before == {path: path.read_bytes() for path in resume.root(args).rglob("*") if path.is_file()}


def test_resume_requires_new_physical_input_contract(setup_cache):
    args, _, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata.pop("physical_input_contract")
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    with pytest.raises(ValueError, match="physical input contract"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume"])
