from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

import pc_xpass_match_cache as cache
import physical_pass_model as physics
from test_physical_xpass import make_graph, make_label


def descriptor(pid=1):
    return dict(match_id="match", possession_id=pid, possessor_id=f"home_{pid}",
                feature_run_id="features", spell_artifact_sha256="artifact", carry_definition="definition")


def rows(key="one", frames=(10, 20), value=0.5):
    return pd.DataFrame(dict(match_id=key, action_index=frames, state_frame_id=frames,
                             physical_state_hash=physics.physical_state_hash(make_graph()), frame_scope="frame_id", home_1=value))


def save(root, *, lane=False):
    cache.initialize(root)
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.write("one", rows())
        transaction.expected = {"one": {10, 20}}
        if lane:
            transaction.write_lane("one", [], rows().to_dict("records"))
        transaction.commit(export_lane=lane)


@pytest.mark.parametrize("dataset", ["sportec", "skillcorner"])
def test_explicit_identity_and_shared_boundary(tmp_path, dataset):
    root = tmp_path / dataset
    cache.initialize(root)
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.register("two", descriptor(2))
        transaction.write("one", rows())
        transaction.write("two", rows("two", (20, 30), 0.8))
        transaction.commit()
    assert [p.name for p in (root / "matches").glob("*.parquet")] == ["match.parquet"]
    frame, metadata = cache._read_parquet(root / "matches/match.parquet")
    assert set(cache.REQUIRED).issubset(frame)
    assert set(frame.match_id) == {"match"}
    assert metadata["descriptors"]["two"] == descriptor(2)
    assert cache.load_possession(root, "match", cache.Selection(1, "one")).home_1.tolist() == [0.5, 0.5]
    assert cache.load_possession(root, "match", cache.Selection(2, "two")).home_1.tolist() == [0.8, 0.8]
    with pytest.raises(ValueError, match="provenance"):
        cache.load_possession(root, "match", cache.Selection(1, "wrong"))


def test_batches_write_once_and_unchanged_resume_never_writes_parquet(tmp_path, monkeypatch):
    cache.initialize(tmp_path)
    original = cache.pq.write_table
    writes = []
    def record(*args, **kwargs):
        writes.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(cache.pq, "write_table", record)
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.write("one", rows(frames=(10,)))
        transaction.write("one", rows(frames=(20,)))
        assert not writes
        transaction.commit()
    assert len(writes) == 1
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.commit()
    assert len(writes) == 1
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.write("one", rows(frames=(15,)))
        transaction.commit()
    assert len(writes) == 2
    assert cache.load_possession(tmp_path, "match", cache.Selection(1, "one")).state_frame_id.tolist() == [10, 15, 20]


def test_failed_match_keeps_previous_output(tmp_path):
    save(tmp_path)
    original = (tmp_path / "matches/match.parquet").read_bytes()
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.write("one", rows(frames=(15,)))
        transaction.failed = True
        with pytest.raises(ValueError, match="failed"):
            transaction.commit()
    assert (tmp_path / "matches/match.parquet").read_bytes() == original


def test_interrupted_publication_rejected_and_recomputed(tmp_path, monkeypatch):
    save(tmp_path)
    replace = Path.replace
    def fail(source, target):
        if source.name.endswith(".pending.parquet"):
            raise OSError("interrupted")
        return replace(source, target)
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.write("one", rows(frames=(15,)))
        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", fail)
            with pytest.raises(OSError, match="interrupted"):
                transaction.commit()
    with pytest.raises(ValueError, match="pending"):
        cache.load_match(tmp_path, "match")
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        assert transaction.parts == {}
        transaction.register("one", descriptor())
        transaction.write("one", rows())
        transaction.commit()
    assert len(cache.load_match(tmp_path, "match")[0]["matches"]) == 2


def test_empty_match_and_missing_coverage(tmp_path):
    cache.initialize(tmp_path)
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.commit()
    assert cache.load_match(tmp_path, "match")[0]["matches"].empty
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.expected = {"one": {10}}
        with pytest.raises(ValueError, match="missing requested"):
            transaction.commit()
        transaction.write("one", rows(frames=(10,)))
        with pytest.raises(ValueError, match="lane-control"):
            transaction.commit(export_lane=True)


def test_duplicate_and_changed_provenance_rejected(tmp_path):
    save(tmp_path)
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        with pytest.raises(ValueError, match="Duplicate"):
            transaction.write("one", rows(frames=(10, 10)))
        with pytest.raises(ValueError, match="provenance"):
            transaction.register("new", descriptor())
        changed = dict(descriptor(2), spell_artifact_sha256="changed")
        with pytest.raises(ValueError, match="provenance"):
            transaction.register("two", changed)


def test_main_update_preserves_lane_output_values(tmp_path):
    save(tmp_path, lane=True)
    original = pd.read_parquet(tmp_path / "lane_control/match.parquet")
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        enriched = transaction.read("one")
        enriched["home_1__pass_height"] = 0.3
        transaction.write("one", enriched)
        transaction.commit()
    pd.testing.assert_frame_equal(original, pd.read_parquet(tmp_path / "lane_control/match.parquet"))


@pytest.mark.parametrize("old_name", ["match", "possession_states_v1_sportec_hash"])
def test_old_formats_rejected(tmp_path, old_name):
    (tmp_path / "matches").mkdir()
    rows().to_parquet(tmp_path / "matches" / f"{old_name}.parquet")
    with pytest.raises(ValueError, match="Regenerate"):
        cache.initialize(tmp_path)
    with pytest.raises(ValueError, match="Regenerate"):
        cache.load_possession(tmp_path, "match", cache.Selection(1, "one"))


@pytest.mark.parametrize("dataset", ["sportec", "skillcorner"])
def test_numerical_batch_round_trip_reuse_extension_and_visualization(tmp_path, monkeypatch, dataset):
    root = tmp_path / dataset
    cache.initialize(root)
    graph = make_graph()
    def item(frame):
        label = make_label().clone()
        label[physics.LABEL_INDEX["action_index"]] = frame
        return dict(match_id="one", graphs=[graph], labels=torch.stack([label]),
                    frame_scope="frame_id", state_frame_id=[frame])
    def warm(transaction, frame):
        items = [item(frame)]
        transaction.expect(items)
        return physics.prewarm_physical_xpass_runtime_cache(items, cache_dir=root, source="pc_xpass",
            max_speed=3, min_speed=3, speed_step=2, angle_step=90, radial_gridsize=5,
            num_workers=1, export_lane_control=True)
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("one", descriptor())
        assert warm(transaction, 10)["cache_misses"] == 1
        assert warm(transaction, 20)["cache_misses"] == 1
        transaction.commit(export_lane=True)
    selection = cache.Selection(1, "one", {})
    frame = physics.load_physical_xpass_match(root, "match", cache_selection=selection)
    expected = physics.compute_graph_pc_xpass_metrics(graph, max_speed=3, min_speed=3, speed_step=2,
                                                     angle_step=90, radial_gridsize=5)
    # The serialized row must equal a direct computation, independent of file grouping.
    for column, value in expected.items():
        if column.endswith("__max_xpass"):
            assert frame.loc[10, column] == pytest.approx(value, nan_ok=True)
    with cache.MatchAccumulator(root, "match") as transaction:
        assert warm(transaction, 10)["cache_misses"] == 0
        assert warm(transaction, 15)["cache_misses"] == 1
        transaction.commit(export_lane=True)
    table = physics.load_runtime_physical_xpass_visualization_table(root, "match", [10, 15, 20],
             metric="max_xpass", cache_selection=cache.Selection(1, "one", {}))
    assert table.index.tolist() == [10, 15, 20]
    with pytest.raises(ValueError, match="Possession selection"):
        physics.load_physical_xpass_match(root, "match")


@pytest.mark.parametrize("dataset", ["sportec", "skillcorner"])
@pytest.mark.parametrize("scope", ["actions", "frames"])
def test_dataset_runner_publishes_actual_match_once(tmp_path, monkeypatch, dataset, scope):
    from scripts import generate_physical_xpass as generate, run_relevant_models
    import pc_xpass_versions as versions
    from types import SimpleNamespace

    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "versions")
    args = generate.parse_args(["--pc-xpass", "--scope", scope, "--no-sportec", "--no-skillcorner",
        "--no-benchmark", "--min-speed", "3", "--max-speed", "3", "--angle-step", "90",
        "--radial-gridsize", "5", "--num-workers", "1", "--export-lane-control"])
    versions.start_generation(args)
    args.scope = scope
    frames = [10, 20] if scope == "actions" else [10, 15, 20]
    report = dict(feature_run_id="feature", carry_definition="definition", spell_artifact_sha256="hash",
                  possessions={"1": {"selected_frames": len(frames)}, "2": {"selected_frames": len(frames)}})
    def state(pid):
        labels = []
        for frame in frames:
            label = make_label().clone()
            label[physics.LABEL_INDEX["action_index"]] = frame
            labels.append(label)
        return SimpleNamespace(pc_cache_match_id=str(pid), match_id="match", event_index=pid,
            actions=pd.DataFrame({"object_id": f"home_{pid}", "player_id": pid, "period_id": 1}, index=frames),
            labels=torch.stack(labels), graph_features_0=[make_graph() for _ in frames],
            original_start_frame=10, original_end_frame=20, frame_selection={})
    monkeypatch.setattr(generate, "resolve_runtime_row_window", lambda *a: 1)
    if dataset == "sportec":
        args.sportec_feature_run_id = "feature"
        monkeypatch.setattr(generate, "resolve_feature_run_id", lambda *a, **k: "feature")
        monkeypatch.setattr(generate, "resolve_feature_root", lambda *a: tmp_path / "feature")
        monkeypatch.setattr(generate, "resolve_reference_label_context", lambda *a: (None, "disc_0.7", "model"))
        monkeypatch.setattr(generate, "resolve_match_ids", lambda *a: ["match"])
        monkeypatch.setattr(run_relevant_models, "load_match", lambda *a, **k: SimpleNamespace(possession_report=report))
        monkeypatch.setattr(generate, "build_sportec_possessions", lambda *a, **k: iter([(state(1), report), (state(2), report)]))
    else:
        monkeypatch.setattr(generate, "discover_skillcorner_matches", lambda *a, **k: (["match"], {}))
        monkeypatch.setattr(generate, "build_skillcorner_match_context", lambda *a: {"events": pd.DataFrame({"index": [1, 2]})})
        monkeypatch.setattr(generate, "build_skillcorner_possession", lambda context, pid, **k: (state(pid), {}))
    runner = generate.run_runtime_sportec if dataset == "sportec" else generate.run_runtime_skillcorner
    result = runner(args)
    assert not result["skipped"]
    root = Path(args._pc_version_root) / dataset
    frame = pd.read_parquet(root / "matches/match.parquet")
    assert len(frame) == 2 * len(frames)
    assert set(frame.match_id) == {"match"}
    assert len(list((root / "matches").glob("*.parquet"))) == 1
    previous = (root / "matches/match.parquet").stat().st_mtime_ns
    again = runner(args)
    assert not again["skipped"]
    assert again["stats"]["cache_misses"] == 0
    assert (root / "matches/match.parquet").stat().st_mtime_ns == previous


def test_height_refresh_shared_frames_commits_match_and_preserves_lanes(tmp_path, monkeypatch):
    import pc_xpass_resume as resume
    root = tmp_path / "sportec"
    save(root, lane=True)
    args = Namespace(_pc_version_root=str(tmp_path), _pass_height_model=Namespace(args={}), pass_height_device="cpu")
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("two", descriptor(2))
        transaction.write("two", rows("two"))
        transaction.write_lane("two", [], rows("two").to_dict("records"))
        transaction.commit(export_lane=True)
    before_lane = (root / "lane_control/match.parquet").read_bytes()
    source = pd.read_parquet(root / "matches/match.parquet")
    refresh = resume.HeightRefresh(args, "sportec", {"match": source}, {"model_id": "new"})
    monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", lambda graphs, model, **k:
        [{player: 0.3 for player in physics._pass_height_output_player_ids(graph)} for graph in graphs])
    with cache.MatchAccumulator(root, "match") as transaction:
        for key in ("one", "two"):
            for frame in (10, 20):
                label = make_label().clone()
                label[physics.LABEL_INDEX["action_index"]] = frame
                refresh.consume([dict(match_id=key, graphs=[make_graph()], labels=torch.stack([label]), frame_scope="frame_id")])
        refresh.verify()
        transaction.commit()
    after = pd.read_parquet(root / "matches/match.parquet")
    assert after[resume.MODEL_COLUMN].notna().all()
    assert (root / "lane_control/match.parquet").read_bytes() == before_lane


def test_match_height_digest_ignores_columns_from_other_possessions():
    import pc_xpass_resume as resume
    graph, label = make_graph(), make_label()
    row = rows().iloc[0].to_dict()
    with_other_roster = {**row, "home_99__max_xpass": float("nan")}
    assert resume.input_digest(graph, label, row, omit_missing=True) == resume.input_digest(
        graph, label, with_other_roster, omit_missing=True)


def test_match_inventory_is_lazy_and_excludes_staging_files(tmp_path, monkeypatch):
    import pc_xpass_resume as resume
    root = tmp_path / "sportec"
    save(root)
    (root / "matches/.match.pending.parquet").write_bytes(b"unfinished")
    reads = []
    original = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda path, **k: (reads.append(path), original(path, **k))[1])
    inventory = resume.inventory(tmp_path, datasets=["sportec"], lazy_possessions=True)
    assert not reads
    assert list(inventory["sportec"]) == ["match"]
    assert len(inventory["sportec"]["match"]) == 2
    assert len(reads) == 1


def test_corrupt_commit_and_missing_lane_coverage_are_rejected(tmp_path):
    save(tmp_path, lane=True)
    path = tmp_path / "completion/match.json"
    record = cache.read_json(path)
    record["lane_states"] = []
    cache.atomic_json(path, record)
    with pytest.raises(ValueError, match="lane-control coverage"):
        cache.load_possession(tmp_path, "match", cache.Selection(1, "one"), lane=True)
    with (tmp_path / "matches/match.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Inconsistent"):
        cache.load_match(tmp_path, "match")


def test_full_height_refresh_streams_matches_and_recovers_pending_commit(tmp_path, monkeypatch):
    import pc_xpass_resume as resume
    import pc_xpass_versions as versions
    from scripts import generate_physical_xpass as generate
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "versions")
    args = generate.parse_args(["--pc-xpass", "--no-sportec", "--no-skillcorner", "--no-benchmark",
        "--min-speed", "3", "--max-speed", "3", "--angle-step", "90", "--radial-gridsize", "5", "--num-workers", "1"])
    versions.start_generation(args)
    root = Path(args._pc_version_root)
    for dataset in ("sportec", "skillcorner"):
        save(root / dataset, lane=True)
    # An interrupted publication must be recomputed, not treated as refreshed.
    path = root / "skillcorner/completion/match.json"
    pending = cache.read_json(path)
    pending["status"] = "pending"
    cache.atomic_json(path, pending)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "best_weights.pt").write_bytes(b"test model")
    args.pass_height_model_id = "height/new"
    args._pass_height_model = Namespace(args={"task": "pass_height"})
    args._pass_height_model_record = {"model_path": str(model_path)}
    args.export_lane_control = True
    monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", lambda graphs, model, **k:
        [{player: 0.3 for player in physics._pass_height_output_player_ids(graph)} for graph in graphs])
    seen = []
    def runner(dataset):
        def run(replay):
            assert getattr(replay, "match_id" if dataset == "sportec" else "skillcorner_match_id") == ["match"]
            seen.append(dataset)
            transaction = cache.begin(replay, root / dataset, "match")
            try:
                transaction.register("one", descriptor())
                for frame in (10, 20):
                    item = dict(match_id="one", graphs=[make_graph()], labels=torch.stack([make_label(action_index=frame)]),
                                frame_scope="frame_id", state_frame_id=[frame])
                    generate.prewarm_runtime_items([item], cache_dir=root / dataset, args=replay)
                cache.finish(transaction, replay)
            finally:
                cache.end(transaction)
            return {"skipped": {}}
        return run
    runners = {dataset: runner(dataset) for dataset in ("sportec", "skillcorner")}
    resume.refresh_all(args, runners)
    assert seen == ["sportec", "skillcorner"]
    for dataset in seen:
        frame = cache.load_match(root / dataset, "match")[0]["matches"]
        assert frame[resume.MODEL_COLUMN].notna().all()
    assert "pending" not in versions.read_metadata(root)["pass_height_enrichment"]
    resume.refresh_all(args, runners)
    assert seen == ["sportec", "skillcorner"]

