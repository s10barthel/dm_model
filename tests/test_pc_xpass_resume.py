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
    args.pass_height_model_id = "pass_height/new"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "best_weights.pt").write_bytes(b"weights")
    args._pass_height_model_record = {"model_path": str(model_path), "pass_height_definition": {
        "threshold_meters": 1.5, "operator": ">=", "target_source": "pass_max_ball_z"}}
    args._pass_height_model = Namespace(args={"task": "pass_height"})
    args.pass_height_device = "cpu"
    calls = []

    def predict(graphs, model, **kwargs):
        calls.append(kwargs["match_id"])
        return [{player: 0.3 for player in physics._pass_height_output_player_ids(graph)} for graph in graphs]

    monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", predict)
    return args, items, calls


def runners_for(items):
    def run(args):
        return generate.prewarm_runtime_items(items, cache_dir=resume.root(args) / "hawkeye", args=args)
    return {dataset: run for dataset in resume.DATASETS}


def test_full_refresh_includes_excluded_dataset_and_preserves_physics(setup_cache):
    args, items, calls = setup_cache
    before = resume.inventory(resume.root(args))
    resume.refresh_all(args, runners_for(items))
    assert len(calls) == 2
    after = resume.inventory(resume.root(args))
    for dataset in ("hawkeye", "benchmark"):
        pd.testing.assert_frame_equal(before[dataset]["one"], after[dataset]["one"][before[dataset]["one"].columns])
        assert after[dataset]["one"][resume.MODEL_COLUMN].notna().all()
    assert "pending" not in versions.read_metadata(resume.root(args))["pass_height_enrichment"]
    resume.refresh_all(args, runners_for(items))
    assert len(calls) == 2


def test_interrupted_refresh_resumes_only_unfinished_rows(setup_cache):
    args, items, calls = setup_cache
    runners = runners_for(items)

    def interrupt(replay):
        raise KeyboardInterrupt

    runners["hawkeye"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        resume.refresh_all(args, runners)
    assert len(calls) == 1  # benchmark precedes Hawkeye in refresh inventory order
    assert versions.read_metadata(resume.root(args))["pass_height_enrichment"]["pending"]
    versions.finish_generation(args)
    assert versions.read_metadata(resume.root(args))["status"] == "incomplete"
    assert not (versions.config.PC_XPASS_DIR / "latest.json").exists()
    resume.refresh_all(args, runners_for(items))
    assert len(calls) == 2


def test_pending_refresh_blocks_height_but_allows_physical_reads(setup_cache):
    args, items, _ = setup_cache
    metadata = versions.read_metadata(resume.root(args))
    metadata["pass_height_enrichment"] = {"pending": {"model_id": "pass_height/new"}}
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    frame = physics.load_physical_xpass_match(resume.root(args) / "hawkeye", "one")
    physics.attach_physical_xpass_to_graph(make_graph(), make_label(), frame, match_id="one")
    with pytest.raises(ValueError, match="pending"):
        physics.attach_pass_height_to_graph(make_graph(), make_label(), frame, match_id="one")
    with pytest.raises(ValueError, match="pending"):
        physics.attach_evaluation_xpass_to_graph(make_graph(), make_label(), frame, match_id="one",
                                                metric="max", require_pass_height=True)


@pytest.mark.parametrize("failure", ["missing", "changed", "incomplete"])
def test_failed_reconstruction_or_prediction_keeps_pending(setup_cache, monkeypatch, failure):
    args, items, _ = setup_cache
    if failure == "changed":
        items[0]["graphs"][0].x[0, 0] += 1
    elif failure == "missing":
        items = []
    else:
        monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", lambda *a, **kw: [{}])
    with pytest.raises(ValueError):
        resume.refresh_all(args, runners_for(items))
    assert versions.read_metadata(resume.root(args))["pass_height_enrichment"]["pending"]


def test_dry_refresh_and_contract_checks_do_not_write(setup_cache):
    args, items, _ = setup_cache
    args.dry_run = True
    before = {path: path.read_bytes() for path in resume.root(args).rglob("*") if path.is_file()}
    resume.prepare_contracts(args, ["hawkeye"])
    resume.persist_contracts(args)
    resume.refresh_all(args, runners_for(items))
    after = {path: path.read_bytes() for path in resume.root(args).rglob("*") if path.is_file()}
    assert before == after


def test_model_switch_and_omitted_model_inheritance(setup_cache):
    args, items, _ = setup_cache
    resume.refresh_all(args, runners_for(items))
    omitted = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume"])
    assert omitted.pass_height_model_id == "pass_height/new"
    changed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume", "--pass-height-model-id", "pass_height/next"])
    assert changed.pass_height_model_id == "pass_height/next"
    metadata = versions.read_metadata(resume.root(args))
    metadata["pass_height_enrichment"]["pending"] = resume.model_identity(args)
    versions.atomic_json(resume.root(args) / "metadata.json", metadata)
    with pytest.raises(ValueError, match="pending"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "resume", "--pass-height-model-id", "pass_height/next"])


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
    with pytest.raises(ValueError, match="provide --sportec-feature-run-id"):
        resume.prepare_contracts(args, ["sportec"])
    args.sportec_feature_run_id = "features"
    args._pc_explicit.add("sportec_feature_run_id")
    monkeypatch.setattr("project_config.resolve_feature_run_id", lambda *a, **kw: "features")
    resume.prepare_contracts(args, ["sportec"])
    assert args._pc_dataset_contracts["sportec"]["feature_run_id"] == "features"


def test_legacy_recovery_does_not_commit_unverified_partitions(setup_cache):
    args, items, _ = setup_cache
    args.pass_height_model_id = None
    frames = resume.inventory(resume.root(args))["hawkeye"]
    replay = resume.replay_args(args, "hawkeye", versions.read_metadata(resume.root(args)))
    replay._pc_refresh = resume.HeightRefresh(replay, "hawkeye", frames, None)
    replay._pc_recovered_sources = {}
    resume.record_source(replay, "hawkeye", "one", {"situation_id": "one"})
    assert "reconstruction" not in versions.read_metadata(resume.root(args))
    replay._pc_refresh.consume(items)
    replay._pc_refresh.verify()
    assert replay._pc_recovered_sources == {"one": {"situation_id": "one"}}


def test_contracts_for_automatically_named_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    args = generate.parse_args(["--pc-xpass", "--no-sportec"])
    resume.prepare_contracts(args, ["hawkeye"])
    versions.start_generation(args)
    resume.persist_contracts(args)
    assert versions.read_metadata(resume.root(args))["dataset_contracts"]["hawkeye"]["freeze_ballreceipt"] is True


def test_sportec_spell_artifact_cannot_change_partition_identity(setup_cache):
    args, _, _ = setup_cache
    first = {"match_id": "match", "feature_run_id": "features", "carry_definition": "v1", "spell_artifact_sha256": "a"}
    resume.record_source(args, "sportec", "partition1", first)
    with pytest.raises(ValueError, match="spell artifact"):
        resume.record_source(args, "sportec", "partition2", {**first, "spell_artifact_sha256": "b"})


def test_lane_export_restored_and_selections_remain_flexible(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "lane", "--export-lane-control"])
    versions.start_generation(args)
    selected = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "lane", "--scope", "frames", "--frames", "2",
                                    "--no-skillcorner", "--num-workers", "3"])
    assert selected.export_lane_control
    assert selected.frames == 2 and selected.no_skillcorner and selected.num_workers == "3"


def test_refreshed_rows_are_cache_hits_for_normal_generation(setup_cache):
    args, items, calls = setup_cache
    resume.refresh_all(args, runners_for(items))
    stats = generate.prewarm_runtime_items(items, cache_dir=resume.root(args) / "hawkeye", args=args)
    assert stats["cache_hits"] == 1
    assert len(calls) == 2


def test_new_rows_get_atomic_height_provenance(setup_cache):
    args, items, calls = setup_cache
    resume.refresh_all(args, runners_for(items))
    new_items = [{**items[0], "labels": torch.stack([make_label(action_index=8)])}]
    generate.prewarm_runtime_items(new_items, cache_dir=resume.root(args) / "hawkeye", args=args)
    frame = pd.read_parquet(resume.root(args) / "hawkeye" / "matches" / "one.parquet")
    assert frame[resume.MODEL_COLUMN].eq(resume.fingerprint(resume.model_identity(args))).all()
    stats = generate.prewarm_runtime_items(new_items, cache_dir=resume.root(args) / "hawkeye", args=args)
    assert stats["cache_hits"] == 1
    assert len(calls) == 3


def test_changed_physical_input_cannot_overwrite_versioned_row(setup_cache):
    args, items, _ = setup_cache
    items[0]["graphs"][0].x[0, 0] += 1
    path = resume.root(args) / "hawkeye" / "matches" / "one.parquet"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Changed input state"):
        generate.prewarm_runtime_items(items, cache_dir=resume.root(args) / "hawkeye", args=args)
    assert path.read_bytes() == original


def test_interrupt_during_atomic_write_is_recoverable(setup_cache, monkeypatch):
    args, items, calls = setup_cache
    write = physics._write_runtime_physical_xpass_rows
    path = resume.root(args) / "benchmark" / "matches" / "one.parquet"
    original = path.read_bytes()
    monkeypatch.setattr(physics, "_write_runtime_physical_xpass_rows", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        resume.refresh_all(args, runners_for(items))
    assert path.read_bytes() == original
    monkeypatch.setattr(physics, "_write_runtime_physical_xpass_rows", write)
    resume.refresh_all(args, runners_for(items))
    assert versions.read_metadata(resume.root(args))["pass_height_enrichment"]["status"] == "completed"


def test_failed_write_cannot_verify_refresh_coverage(setup_cache, monkeypatch):
    args, items, _ = setup_cache
    frames = resume.inventory(resume.root(args))["hawkeye"]
    refresh = resume.HeightRefresh(args, "hawkeye", frames, resume.model_identity(args))
    def fail(*a, **kw):
        raise PermissionError("locked")
    monkeypatch.setattr(physics, "_write_runtime_physical_xpass_rows", fail)
    with pytest.raises(PermissionError):
        refresh.consume(items)
    with pytest.raises(ValueError, match="Unresolved"):
        refresh.verify()


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


def test_refresh_only_invocation_completes_and_publishes(setup_cache, monkeypatch):
    args, items, _ = setup_cache
    args.no_sportec = args.no_hawkeye = args.no_skillcorner = args.no_benchmark = True
    resume.record_source(args, "hawkeye", "one", {"situation_id": "one"})
    resume.record_source(args, "benchmark", "one", {"modification_id": 1})
    runner = runners_for(items)["hawkeye"]
    monkeypatch.setattr(generate, "run_runtime_hawkeye", runner)
    monkeypatch.setattr(generate, "run_runtime_benchmark", runner)
    monkeypatch.setattr(generate, "prepare_pass_height_context", lambda a: None)
    monkeypatch.setattr(generate, "prepare_runtime_graph_schema", lambda a: None)
    generate.run_runtime_mode(args)
    assert versions.read_metadata(resume.root(args))["status"] == "completed"
    assert json.loads((versions.config.PC_XPASS_DIR / "latest.json").read_text())["run_id"] == "resume"
