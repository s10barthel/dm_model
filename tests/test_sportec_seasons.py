import argparse
import ast
from pathlib import Path

import numpy as np
import pytest

from datatools import sportec_seasons as seasons


@pytest.fixture
def metadata(monkeypatch, tmp_path):
    roots = {season: tmp_path / season for season in ("22_23", "23_24", "24_25")}
    monkeypatch.setattr(seasons, "RAW_SEASON_ROOTS", roots)
    for season, root in roots.items():
        relative, _ = seasons._METADATA_LAYOUTS[season]
        (root / relative).mkdir(parents=True)
    def add(season, match_id, extension=""):
        relative, _ = seasons._METADATA_LAYOUTS[season]
        (roots[season] / relative / (match_id + extension)).write_text("not XML")
    add("22_23", "DFL-MAT-a")
    add("23_24", "DFL-MAT-b", ".xml")
    add("24_25", "DFL-MAT-c")
    return roots, add


@pytest.fixture
def half_metadata(monkeypatch, tmp_path):
    roots = {season: tmp_path / season for season in ("22_23", "23_24", "24_25")}
    monkeypatch.setattr(seasons, "RAW_SEASON_ROOTS", roots)
    for season, root in roots.items():
        relative, _ = seasons._METADATA_LAYOUTS[season]
        directory = root / relative
        directory.mkdir(parents=True)
        for matchday in (1, 17, 18, 34):
            match_id = f"DFL-MAT-{season}-{matchday:02}"
            (directory / f"{match_id}.xml").write_text(
                f'<Root><General MatchId="{match_id}" MatchDay="{matchday}"/></Root>'
            )
    return roots


def test_layouts_union_order_and_duplicates(metadata):
    candidates = ["DFL-MAT-c", "DFL-MAT-a", "DFL-MAT-b", "DFL-MAT-c"]
    assert seasons.select_season_match_ids(candidates, ["24_25", "22_23", "24_25"]) == ["DFL-MAT-c", "DFL-MAT-a"]
    assert seasons.select_season_match_ids(candidates, ["23_24"]) == ["DFL-MAT-b"]


def test_no_season_does_not_discover(monkeypatch):
    monkeypatch.setattr(seasons, "RAW_SEASON_ROOTS", None)
    assert seasons.select_season_match_ids(["unknown", "unknown"], None) == ["unknown", "unknown"]


@pytest.mark.parametrize("season_half", [None, "first"])
def test_unresolved_reports_ids_and_locations(metadata, season_half):
    roots, _ = metadata
    with pytest.raises(ValueError, match="Unresolved.*DFL-MAT-missing") as exc:
        seasons.select_season_match_ids(["DFL-MAT-missing"], ["24_25"], season_half)
    assert str(roots["22_23"]) in str(exc.value)


@pytest.mark.parametrize("season_half", [None, "first"])
def test_conflicting_membership(metadata, season_half):
    _, add = metadata
    add("24_25", "DFL-MAT-a", ".xml")
    with pytest.raises(ValueError, match="Conflicting.*DFL-MAT-a"):
        seasons.select_season_match_ids(["DFL-MAT-c"], ["24_25"], season_half)


@pytest.mark.parametrize("season_half", [None, "first"])
def test_missing_requested_directory(metadata, season_half):
    roots, _ = metadata
    directory = roots["24_25"] / "match_information/starting_players"
    (directory / "DFL-MAT-c").unlink()
    directory.rmdir()
    with pytest.raises(FileNotFoundError, match="24_25.*unavailable"):
        seasons.select_season_match_ids(["DFL-MAT-a"], ["24_25"], season_half)


def test_empty_and_invalid(metadata):
    with pytest.raises(ValueError, match="No Sportec matches"):
        seasons.select_season_match_ids(["DFL-MAT-a"], ["24_25"])
    with pytest.raises(ValueError, match="Unknown Sportec seasons"):
        seasons.select_season_match_ids([], ["99_00"])


@pytest.mark.parametrize("season_half, matchdays", [
    (None, [34, 18, 17, 1]), ("first", [17, 1]), ("second", [34, 18]),
])
def test_season_half_boundaries_and_candidate_order(half_metadata, season_half, matchdays):
    candidates = [f"DFL-MAT-24_25-{day:02}" for day in (34, 18, 17, 1, 18)]
    assert seasons.select_season_match_ids(candidates, ["24_25"], season_half) == [
        f"DFL-MAT-24_25-{day:02}" for day in matchdays
    ]


def test_season_half_repeated_seasons_and_empty_intersection(half_metadata):
    candidates = ["DFL-MAT-24_25-34", "DFL-MAT-22_23-18", "DFL-MAT-23_24-18", "DFL-MAT-24_25-17"]
    assert seasons.select_season_match_ids(candidates, ["24_25", "22_23", "24_25"], "second") == candidates[:2]
    with pytest.raises(ValueError, match="No Sportec matches.*season-half"):
        seasons.select_season_match_ids(["DFL-MAT-24_25-17"], ["24_25"], "second")


@pytest.mark.parametrize("matchplan_path", ["master/matchplan", "match_information/master/matchplan"])
def test_season_half_matchplan_fallback(half_metadata, matchplan_path):
    root = half_metadata["24_25"]
    directory = root / seasons._METADATA_LAYOUTS["24_25"][0]
    (directory / "DFL-MAT-24_25-18.xml").write_text("<Root/>")
    matchplan = root / matchplan_path
    matchplan.parent.mkdir(parents=True, exist_ok=True)
    matchplan.write_text('<Root><Fixture MatchId="DFL-MAT-24_25-18" MatchDay="18"/></Root>')
    assert seasons.select_season_match_ids(
        ["DFL-MAT-24_25-17", "DFL-MAT-24_25-18"], ["24_25"], "second"
    ) == ["DFL-MAT-24_25-18"]


@pytest.mark.parametrize("matchday", [None, "invalid", "0", "35"])
def test_season_half_invalid_matchday(half_metadata, matchday):
    directory = half_metadata["24_25"] / seasons._METADATA_LAYOUTS["24_25"][0]
    attributes = f'MatchDay="{matchday}"' if matchday is not None else ""
    (directory / "DFL-MAT-24_25-17.xml").write_text(f"<Root><General {attributes}/></Root>")
    with pytest.raises(ValueError, match="MatchDay"):
        seasons.select_season_match_ids(["DFL-MAT-24_25-18"], ["24_25"], "second")


def test_season_half_requires_season_and_valid_half():
    with pytest.raises(ValueError, match="requires --season"):
        seasons.select_season_match_ids([], None, "first")
    with pytest.raises(ValueError, match="Unknown Sportec season half"):
        seasons.select_season_match_ids([], ["24_25"], "third")


def test_cli():
    parser = argparse.ArgumentParser()
    seasons.add_season_argument(parser)
    assert parser.parse_args(["--season", "24_25", "--season", "22_23"]).season == ["24_25", "22_23"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--season", "2024/25"])


def test_physical_selection_and_conflict(metadata, monkeypatch, tmp_path):
    from scripts import generate_physical_xpass as physical
    monkeypatch.setattr(physical, "load_match_universe", lambda: {"match_ids": ["DFL-MAT-a", "DFL-MAT-b", "DFL-MAT-c"]})
    args = physical.parse_args(["--season", "24_25"])
    assert physical.resolve_match_ids(args, tmp_path) == ["DFL-MAT-c"]
    assert not args.no_skillcorner and not args.no_benchmark and not args.no_hawkeye
    args = physical.parse_args(["--season", "24_25", "--match-id", "DFL-MAT-c", "--match-id", "DFL-MAT-a"])
    assert physical.resolve_match_ids(args, tmp_path) == ["DFL-MAT-c"]
    with pytest.raises(SystemExit):
        physical.parse_args(["--season", "24_25", "--no-sportec"])


@pytest.mark.parametrize("mode", [[], ["--pc-xpass"], ["--feature-run-id", "feature_run"]])
def test_physical_half_selection(half_metadata, monkeypatch, tmp_path, capsys, mode):
    from scripts import generate_physical_xpass as physical
    candidates = ["DFL-MAT-24_25-18", "DFL-MAT-22_23-18", "DFL-MAT-24_25-17", "DFL-MAT-24_25-34"]
    monkeypatch.setattr(physical, "load_match_universe", lambda: {"match_ids": candidates})
    flags = [*mode, "--season", "24_25", "--season-half", "second"]
    args = physical.parse_args(flags)
    assert physical.resolve_match_ids(args, tmp_path) == [candidates[0], candidates[3]]
    assert args.selected_sportec_match_ids == [candidates[0], candidates[3]]
    assert not args.no_skillcorner and not args.no_benchmark and not args.no_hawkeye
    assert "season half: second" in capsys.readouterr().out
    args = physical.parse_args([*flags, "--match-id", candidates[3], "--match-id", candidates[2]])
    assert physical.resolve_match_ids(args, tmp_path) == [candidates[3]]


@pytest.mark.parametrize("flags, message", [
    (["--season-half", "first"], "requires --season"),
    (["--season", "24_25", "--season-half", "third"], "invalid choice"),
    (["--season", "24_25", "--season-half", "second", "--no-sportec"], "cannot be combined"),
])
def test_physical_half_cli_errors(flags, message, capsys):
    from scripts import generate_physical_xpass as physical
    assert physical.parse_args([]).season_half is None
    with pytest.raises(SystemExit):
        physical.parse_args(flags)
    assert message in capsys.readouterr().err


def test_inference_split_intersection(metadata, monkeypatch, tmp_path):
    from scripts import run_relevant_models as inference
    monkeypatch.setattr(inference, "load_base_splits", lambda *a, **kw: (np.array(["DFL-MAT-a", "DFL-MAT-c"]), np.array(["DFL-MAT-b"])))
    assert inference.parse_args(["--season", "24_25"]).split == "all"
    assert inference.parse_args(["--season", "24_25", "--split", "test"]).split == "test"
    assert inference.resolve_match_ids("all", None, tmp_path, requested_seasons=["24_25"]) == ["DFL-MAT-c"]
    with pytest.raises(ValueError, match="No Sportec matches"):
        inference.resolve_match_ids("test", None, tmp_path, requested_seasons=["24_25"])
    assert inference.resolve_match_ids("all", ["DFL-MAT-b"], tmp_path, requested_seasons=["23_24"]) == ["DFL-MAT-b"]


def test_all_cache_modes_use_shared_resolver():
    source = Path(__file__).resolve().parents[1] / "scripts/generate_physical_xpass.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for name in ("run_legacy_feature_mode", "run_runtime_sportec", "run_runtime_sportec_possessions"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_match_ids" for node in ast.walk(function))


def test_runtime_preflight_fails_before_dataset_processing(metadata, monkeypatch):
    from scripts import generate_physical_xpass as physical
    monkeypatch.setattr(physical, "load_match_universe", lambda: {"match_ids": ["DFL-MAT-unknown"]})
    def unexpected(*args, **kwargs):
        pytest.fail("Generation started before season validation")
    monkeypatch.setattr(physical.pc_versions, "start_generation", unexpected)
    args = physical.parse_args(["--season", "24_25"])
    with pytest.raises(ValueError, match="Unresolved"):
        physical.run_runtime_mode(args)


@pytest.mark.parametrize("mode", [[], ["--pc-xpass"]])
@pytest.mark.parametrize("failure, message", [("empty", "No Sportec matches"), ("matchday", "MatchDay")])
def test_runtime_half_preflight_fails_before_generation(half_metadata, monkeypatch, mode, failure, message):
    from scripts import generate_physical_xpass as physical
    if failure == "matchday":
        directory = half_metadata["24_25"] / seasons._METADATA_LAYOUTS["24_25"][0]
        (directory / "DFL-MAT-24_25-17.xml").write_text("<Root/>")
    monkeypatch.setattr(physical.pc_resume, "prepare_contracts", lambda *args: None)
    def unexpected(*args, **kwargs):
        pytest.fail("Generation started before season-half validation")
    monkeypatch.setattr(physical.pc_versions, "start_generation", unexpected)
    args = physical.parse_args([
        *mode, "--season", "24_25", "--season-half", "second", "--match-id", "DFL-MAT-24_25-17",
    ])
    with pytest.raises(ValueError, match=message):
        physical.run_runtime_mode(args)


@pytest.mark.parametrize("mode", [[], ["--pc-xpass"]])
@pytest.mark.parametrize("season_half", [None, "second"])
def test_runtime_metadata_records_season_half(half_metadata, monkeypatch, tmp_path, mode, season_half):
    from scripts import generate_physical_xpass as physical
    flags = [*mode, "--season", "24_25", "--match-id", "DFL-MAT-24_25-18"]
    if season_half:
        flags.extend(["--season-half", season_half])
    args = physical.parse_args(flags)
    selected = physical.resolve_match_ids(args, tmp_path)
    metadata = {}
    monkeypatch.setattr(physical, "write_run_metadata", lambda path, value: metadata.update(value))
    physical.write_runtime_dataset_metadata("sportec", tmp_path / "cache", args, stats={}, source_inputs={}, skipped={})
    assert metadata["source_inputs"]["requested_seasons"] == ["24_25"]
    assert metadata["source_inputs"]["requested_season_half"] == season_half
    assert metadata["source_inputs"]["selected_match_ids"] == selected


def test_legacy_metadata_records_season_half(half_metadata, monkeypatch, tmp_path):
    from scripts import generate_physical_xpass as physical
    monkeypatch.setattr(physical, "resolve_feature_run_id", lambda *args, **kwargs: "feature_run")
    monkeypatch.setattr(physical, "resolve_feature_root", lambda *args: tmp_path / "features")
    monkeypatch.setattr(physical, "resolve_reference_label_dir", lambda *args: tmp_path / "labels")
    monkeypatch.setattr(physical, "get_physical_xpass_dir", lambda *args: tmp_path / "cache")
    metadata = {}
    monkeypatch.setattr(physical, "write_run_metadata", lambda path, value: metadata.update(value))
    args = physical.parse_args([
        "--feature-run-id", "feature_run", "--season", "24_25", "--season-half", "second",
        "--match-id", "DFL-MAT-24_25-18", "--num-workers", "1",
    ])
    physical.run_legacy_feature_mode(args)
    assert metadata["requested_seasons"] == ["24_25"]
    assert metadata["requested_season_half"] == "second"
    assert metadata["selected_match_ids"] == ["DFL-MAT-24_25-18"]
    assert metadata["skipped_match_ids"] == {"DFL-MAT-24_25-18": "missing_graph_or_label"}


def test_pc_cache_can_resume_with_other_season_half(half_metadata, monkeypatch, tmp_path):
    from scripts import generate_physical_xpass as physical
    versions = physical.pc_versions
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "pc_xpass")
    monkeypatch.setattr(physical, "load_match_universe", lambda: {
        "match_ids": [f"DFL-MAT-24_25-{day:02}" for day in (34, 18, 17, 1)],
    })
    monkeypatch.setattr(physical, "write_run_metadata", lambda path, value: versions.atomic_json(path / "metadata.json", value))
    for half in ("first", "second"):
        args = physical.parse_args([
            "--pc-xpass", "--pc-xpass-id", "season_halves", "--season", "24_25", "--season-half", half,
        ])
        physical.resolve_match_ids(args, tmp_path)
        versions.start_generation(args)
        cache_dir = versions.cache_dir("sportec", args)
        physical.write_runtime_dataset_metadata("sportec", cache_dir, args, stats={}, source_inputs={}, skipped={})
        versions.finish_generation(args)
    metadata = versions.read_metadata(cache_dir)
    assert metadata["source_inputs"]["requested_season_half"] == "second"
    assert metadata["source_inputs"]["selected_match_ids"] == [
        f"DFL-MAT-24_25-{day:02}" for day in (17, 1, 34, 18)
    ]
    assert metadata["selection_history"][-1]["requested_season_half"] == "second"
    version_metadata = versions.read_metadata(Path(args._pc_version_root))
    assert [entry["arguments"]["season_half"] for entry in version_metadata["invocations"]] == ["first", "second"]
