from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def equivalent_cache(tmp_path):
    from datatools.sportec_possessions import STATE_CONTRACT
    from pc_xpass_contract import PHYSICAL_INPUT_CONTRACT

    root = tmp_path / "sportec"
    source = dict(match_id="match", possession_id=1, state_partition_id=1, period_id=1,
        possessor_id="home_1", feature_run_id="september", state_contract=STATE_CONTRACT,
        physical_input_contract=PHYSICAL_INPUT_CONTRACT, canonical_sha256="canonical",
        control_sha256="control", resolved_actions_sha256="actions", tracking_sha256="tracking",
        lineup_sha256="lineup")
    cache.initialize(root)
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("cached", source)
        values = rows("cached")
        values["home_1__max_xpass"] = values.home_1
        transaction.write("cached", values)
        records = [dict(row, receiver_id="home_2", opponent_id="away_3")
                   for row in values.to_dict("records")]
        transaction.write_lane("cached", records, values.to_dict("records"))
        transaction.commit(export_lane=True)
    runtime = dict(source, feature_run_id="october")
    return root, cache.Selection(1, "runtime", {}, runtime)


def test_equivalent_run_preserves_cache_rows_and_files(equivalent_cache):
    root, selection = equivalent_cache
    before = {p: (cache.checksum(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    source = cache.load_possession(root, "match", cache.Selection(1, "cached"))
    result = physics.load_physical_xpass_match(root, "match", cache_selection=selection)
    assert result.index.tolist() == [10, 20]
    assert result.home_1.tolist() == [0.5, 0.5]
    assert result.state_id.tolist() == source.state_id.tolist()
    assert result.possession_provenance_id.tolist() == ["cached", "cached"]
    assert result.attrs["pc_xpass_cache_resolution"] == dict(mode="equivalent",
        runtime_feature_run_id="october", cache_feature_run_id="september",
        requested_provenance_id="runtime", resolved_provenance_id="cached")
    assert source.attrs["pc_xpass_cache_resolution"]["mode"] == "exact"
    assert before == {p: (cache.checksum(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_selection_copies_runtime_descriptor(equivalent_cache):
    _, selection = equivalent_cache
    descriptor = dict(selection.descriptor, additional={"version": 1})
    state = SimpleNamespace(pc_cache_match_id="runtime", event_index=1, pc_state_descriptor=descriptor)
    selected = cache.selection_for(state)
    descriptor["tracking_sha256"] = "changed"
    descriptor["additional"]["version"] = 2
    assert selected.descriptor["tracking_sha256"] == "tracking"
    assert selected.descriptor["additional"] == {"version": 1}
    assert selected.memo is state._pc_match_cache


def test_equivalent_run_resolves_lane_coverage_and_visualization_frames(equivalent_cache):
    root, selection = equivalent_cache
    lane = cache.load_possession(root, "match", selection, lane=True, requested_frames=[20])
    assert lane.receiver_id.tolist() == ["home_2", "home_2"]
    assert lane.possession_provenance_id.tolist() == ["cached", "cached"]
    table = physics.load_runtime_physical_xpass_visualization_table(root, "match", [20],
        metric="max_xpass", cache_selection=selection)
    assert table.index.tolist() == [20]
    assert table.loc[20, "home_1"] == pytest.approx(0.5)


@pytest.mark.parametrize("field", ["canonical_sha256", "control_sha256", "resolved_actions_sha256",
    "tracking_sha256", "lineup_sha256", "state_contract", "physical_input_contract", "possessor_id",
    "period_id", "possession_id", "state_partition_id", "match_id", "additional_rule"])
def test_equivalent_run_rejects_changed_content(equivalent_cache, field):
    root, selection = equivalent_cache
    selection.descriptor[field] = "changed"
    with pytest.raises(ValueError, match=field):
        cache.load_possession(root, "match", selection)


@pytest.mark.parametrize("field", ["canonical_sha256", "control_sha256", "resolved_actions_sha256",
    "tracking_sha256", "lineup_sha256", "feature_run_id"])
@pytest.mark.parametrize("side", ["runtime", "cached"])
def test_equivalent_run_requires_complete_descriptors(equivalent_cache, field, side):
    root, selection = equivalent_cache
    _, _, record = cache.load_match(root, "match")
    descriptors = {key: dict(value) for key, value in record["descriptors"].items()}
    target = selection.descriptor if side == "runtime" else descriptors["cached"]
    del target[field]
    with pytest.raises(ValueError, match=field):
        cache._resolve_possession(root, "match", selection, descriptors)


def test_equivalent_run_rejects_ambiguous_descriptors(equivalent_cache):
    root, selection = equivalent_cache
    _, _, record = cache.load_match(root, "match")
    descriptors = {**record["descriptors"], "duplicate": dict(record["descriptors"]["cached"])}
    with pytest.raises(ValueError, match="ambiguous"):
        cache._resolve_possession(root, "match", selection, descriptors)


@pytest.mark.parametrize("lane", [False, True])
def test_equivalent_run_requires_requested_frames(equivalent_cache, lane):
    root, selection = equivalent_cache
    with pytest.raises(ValueError, match="missing requested possession frames"):
        cache.load_possession(root, "match", selection, lane=lane, requested_frames=[30])


def test_equivalent_run_keeps_integrity_and_lane_coverage_checks(equivalent_cache):
    root, selection = equivalent_cache
    path = root / "completion/match.json"
    record = cache.read_json(path)
    record["lane_states"] = []
    cache.atomic_json(path, record)
    with pytest.raises(ValueError, match="lane-control coverage"):
        cache.load_possession(root, "match", selection, lane=True)
    with (root / "matches/match.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Inconsistent"):
        cache.load_possession(root, "match", selection)


def test_equivalent_run_does_not_relax_generation_or_other_datasets(equivalent_cache):
    root, selection = equivalent_cache
    with cache.MatchAccumulator(root, "match") as transaction:
        with pytest.raises(ValueError, match="provenance"):
            transaction.register(selection.provenance_id, selection.descriptor)
        with pytest.raises(ValueError, match="provenance"):
            transaction.register("cached", selection.descriptor)
    _, _, record = cache.load_match(root, "match")
    with pytest.raises(ValueError, match="provenance"):
        cache._resolve_possession(root.parent / "skillcorner", "match", selection, record["descriptors"])
    with pytest.raises(ValueError, match="provenance"):
        cache.load_possession(root, "match", cache.Selection(1, "runtime", {}))


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
    with pytest.raises(ValueError, match="canonical action manifest" if dataset == "sportec" else "Possession selection"):
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
            pc_state_descriptor=descriptor(pid),
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
        monkeypatch.setattr(run_relevant_models, "load_match", lambda *a, **k: SimpleNamespace(possession_report=report,
            sportec_state_plan=SimpleNamespace(manifest=lambda: {"provenance": report, "spells": [], "actions": []})))
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


def test_comparison_adapter_reads_equivalent_cache_for_both_checkpoints(equivalent_cache, monkeypatch):
    from scripts import inference_visualization_comparison as comparison, visualize_action_components as visualizer
    from scripts.visualization_comparison import Variant
    from test_visualization_comparison import TinyModel

    root, selection = equivalent_cache
    subject = SimpleNamespace(actions=pd.DataFrame(dict(frame_id=[20], object_id=["home_1"], spadl_type=["pass"])))
    # This cache adapter fixture has no tracking/overlay inputs.
    monkeypatch.setattr(visualizer, "prepare_action_overlays", lambda *a, **k: None)
    args = visualizer.parse_args(["--match-id", "match", "--feature-run-id", "october",
        "--show-physical-xpass", "--physical-cache-dir", str(root)])
    calls = []
    monkeypatch.setattr(visualizer, "resolve_runtime_feature_run_context", lambda explicit, *a, **k: {
        "feature_run_id": explicit, "feature_root": root.parent / explicit, "intended_receiver_mode": "model"})
    monkeypatch.setattr(visualizer, "load_match", lambda *a, **k: subject)
    monkeypatch.setattr(visualizer, "resolve_action_indices", lambda *a: [(0, "1")])

    def collect(subject, models, feature_root, *a, **kwargs):
        assert feature_root.name == "october"
        values = physics.load_physical_xpass_match(root, "match", cache_selection=selection)
        calls.append((models["pass_success"].identifier, values.attrs["pc_xpass_cache_resolution"]["mode"]))
        return {"pass_success": pd.Series({"home_1": values.loc[20, "home_1"]})}

    monkeypatch.setattr(visualizer, "collect_action_components", collect)
    variants = [Variant(name, name, {}, [], {"pass_success": TinyModel("pass_success/" + name)})
                for name in ("older", "newer")]
    schema = {"node_in_dim": 26, "edge_in_dim": 5, "add_v_edge_features": True,
              "add_relative_speed_edge_features": True}
    groups = list(comparison.sportec_groups(args, visualizer, variants,
        SimpleNamespace(rendered_components=["pass_success"]), {}, None, schema, "cpu"))
    assert calls == [("pass_success/older", "equivalent"), ("pass_success/newer", "equivalent")]
    assert [table["pass_success"].loc[20, "home_1"] for table in groups[0]["tables"]] == [0.5, 0.5]



def test_version_validation_memo_rechecks_changed_metadata(equivalent_cache, monkeypatch):
    root, _ = equivalent_cache
    path = root / "metadata.json"
    metadata = dict(cache.read_json(path), source="pc_xpass", available_x_pass_versions=["max"],
                    available_metrics=["max_xpass"])
    cache.atomic_json(path, metadata)
    original = cache.json.loads
    parses = []
    def parse(value, **kwargs):
        parses.append(True)
        return original(value, **kwargs)
    monkeypatch.setattr(cache.json, "loads", parse)
    memo = {}
    for _ in range(2):
        physics.validate_x_pass_version_available(root, source="pc_xpass", x_pass_version="max",
                                                 metric="max_xpass", memo=memo)
    assert len(parses) == 1
    cache.atomic_json(path, dict(metadata, source="physical_xpass"))
    with pytest.raises(ValueError, match="has source"):
        physics.validate_x_pass_version_available(root, source="pc_xpass", x_pass_version="max",
                                                 metric="max_xpass", memo=memo)
    assert len(parses) == 2
