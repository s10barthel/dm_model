from argparse import ArgumentParser


from types import SimpleNamespace


import pandas as pd


import pytest


import torch


from datatools.possession_frames import (
    add_frame_selection_arguments, resolve_frame_selection_arguments,
    select_possession_frames, possession_cache_identity,
)


def sample(start=102, end=119, stride=5, missing=(), scope="frames"):
    return select_possession_frames(range(start, end + 1), start, end,
                                    lambda f: None if f in missing else f,
                                    scope=scope, frames=stride)


@pytest.mark.parametrize("start,end,stride,expected", [
    (102, 119, 5, [102, 107, 112, 117, 119]),
    (102, 104, 5, [102, 104]), (102, 102, 5, [102]),
    (102, 104, 1, [102, 103, 104]), (102, 119, 99, [102, 119]),
])
def test_sampling(start, end, stride, expected):
    states, _ = sample(start, end, stride)
    assert [frame for frame, _, _ in states] == expected


def test_invalid_endpoints_do_not_shift_grid():
    states, report = sample(missing={102, 103, 107, 119})
    assert [f for f, _, _ in states] == [104, 112, 117, 118]
    assert report["endpoint_substitutions"] == 2


def test_missing_tracking_rows_do_not_shift_grid():
    states, _ = select_possession_frames([102, 103, 108, 112, 119], 102, 119, lambda f: f,
                                        scope="frames", frames=5)
    assert [f for f, _, _ in states] == [102, 112, 119]


def test_all_invalid_and_single_valid():
    assert sample(missing=range(102, 120))[0] == []
    states, _ = sample(missing=set(range(102, 120)) - {110})
    assert states == [(110, 110, "start_end")]


@pytest.mark.parametrize("argv", [["--frames", "5"], ["--scope", "frames", "--frames", "0"],
                                  ["--scope", "frames", "--frames", "-1"],
                                  ["--scope", "frames", "--frames", "1.5"]])
def test_invalid_cli(argv):
    parser = ArgumentParser()
    add_frame_selection_arguments(parser)
    with pytest.raises(SystemExit):
        resolve_frame_selection_arguments(parser, parser.parse_args(argv))


def test_cli_defaults():
    parser = ArgumentParser()
    add_frame_selection_arguments(parser)
    args = parser.parse_args([])
    resolve_frame_selection_arguments(parser, args)
    assert (args.scope, args.frames) == ("actions", 1)


def test_pc_cache_multiple_frames_round_trip(tmp_path, monkeypatch):
    import physical_pass_model as physical
    import pc_xpass_match_cache as cache
    monkeypatch.setattr(physical, "validate_runtime_physical_xpass_visualization_cache", lambda *a: {})
    cache.initialize(tmp_path)
    def rows(frames, key):
        return pd.DataFrame({"match_id": key, "physical_state_hash": "hash", "action_index": frames,
                             "state_frame_id": frames, "frame_scope": "frame_id"})
    with cache.MatchAccumulator(tmp_path, "match") as transaction:
        for pid, frames in [(1, [102, 119]), (2, [119])]:
            key = str(pid)
            transaction.register(key, dict(match_id="match", possession_id=pid, possessor_id="home_1"))
            physical._write_runtime_physical_xpass_rows(tmp_path, key, rows(frames, key))
        physical._write_runtime_physical_xpass_rows(tmp_path, "1", rows([102, 107, 112, 117, 119], "1"))
        transaction.commit()
    loaded = physical.load_physical_xpass_match(tmp_path, "match", cache_selection=cache.Selection(1, "1"))
    assert loaded.index.tolist() == [102, 107, 112, 117, 119]
    assert len(physical.load_physical_xpass_match(tmp_path, "match", cache_selection=cache.Selection(2, "2"))) == 1


def test_pc_only_cli_and_explicit_runtime_feature_run():
    from scripts import generate_physical_xpass as gen
    for argv in (["--scope", "frames"], ["--frames", "5"],
                 ["--sportec-feature-run-id", "run"],
                 ["--pc-xpass", "--feature-run-id", "legacy", "--scope", "frames"]):
        with pytest.raises(SystemExit):
            gen.parse_args(argv)
    args = gen.parse_args(["--pc-xpass", "--scope", "frames", "--frames", "5", "--sportec-feature-run-id", "run"])
    assert (args.scope, args.frames, args.sportec_feature_run_id) == ("frames", 5, "run")


def test_skillcorner_missing_receipt_does_not_reanchor_stride(monkeypatch):
    from test_skillcorner_postprocessing import make_skillcorner_possession, patch_skillcorner_graph_builder
    from datatools import skillcorner
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession(list(range(103, 120)))
    possession.original_start_frame = 102
    possession.original_end_frame = 119
    actions, _, _, _ = skillcorner._build_actions_and_labels(possession, scope="frames", frames=5)
    assert actions.index.tolist() == [103, 107, 112, 117, 119]
    assert possession.frame_selection["endpoint_substitutions"] == 1
