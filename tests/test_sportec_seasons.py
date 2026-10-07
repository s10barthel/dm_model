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


def test_layouts_union_order_and_duplicates(metadata):
    candidates = ["DFL-MAT-c", "DFL-MAT-a", "DFL-MAT-b", "DFL-MAT-c"]
    assert seasons.select_season_match_ids(candidates, ["24_25", "22_23", "24_25"]) == ["DFL-MAT-c", "DFL-MAT-a"]
    assert seasons.select_season_match_ids(candidates, ["23_24"]) == ["DFL-MAT-b"]


def test_no_season_does_not_discover(monkeypatch):
    monkeypatch.setattr(seasons, "RAW_SEASON_ROOTS", None)
    assert seasons.select_season_match_ids(["unknown", "unknown"], None) == ["unknown", "unknown"]


def test_unresolved_reports_ids_and_locations(metadata):
    roots, _ = metadata
    with pytest.raises(ValueError, match="Unresolved.*DFL-MAT-missing") as exc:
        seasons.select_season_match_ids(["DFL-MAT-missing"], ["24_25"])
    assert str(roots["22_23"]) in str(exc.value)


def test_conflicting_membership(metadata):
    _, add = metadata
    add("24_25", "DFL-MAT-a", ".xml")
    with pytest.raises(ValueError, match="Conflicting.*DFL-MAT-a"):
        seasons.select_season_match_ids(["DFL-MAT-c"], ["24_25"])


def test_missing_requested_directory(metadata):
    roots, _ = metadata
    directory = roots["24_25"] / "match_information/starting_players"
    (directory / "DFL-MAT-c").unlink()
    directory.rmdir()
    with pytest.raises(FileNotFoundError, match="24_25.*unavailable"):
        seasons.select_season_match_ids(["DFL-MAT-a"], ["24_25"])


def test_empty_and_invalid(metadata):
    with pytest.raises(ValueError, match="No Sportec matches"):
        seasons.select_season_match_ids(["DFL-MAT-a"], ["24_25"])
    with pytest.raises(ValueError, match="Unknown Sportec seasons"):
        seasons.select_season_match_ids([], ["99_00"])


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


def test_inference_split_intersection(metadata, monkeypatch, tmp_path):
    from scripts import run_relevant_models as inference
    monkeypatch.setattr(inference, "load_base_splits", lambda *a, **kw: (np.array(["DFL-MAT-a", "DFL-MAT-c"]), np.array(["DFL-MAT-b"])))
    assert inference.parse_args(["--season", "24_25"]).split == "test"
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
