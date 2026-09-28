import argparse
import gc
import json
import random
from pathlib import Path
from types import SimpleNamespace
import weakref
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from datatools import config
from models import utils
import ipw_preparation as ipw
from ipw_options import add_ipw_arguments, ipw_flags, restore_ipw_defaults
from training_state import capture_rng, restore_rng
from training_monitor import TrainingMonitor


class Predictor(torch.nn.Module):
    args = {"edge_in_dim": 2}

    def forward(self, graphs):
        assert not self.training and not torch.is_grad_enabled()
        assert graphs.edge_attr.shape[1] == 2
        return graphs.x[:, config.NODE_FEATURE_X] / 10


@pytest.fixture
def setup(tmp_path, monkeypatch):
    features, labels, model = (tmp_path / name for name in ("features", "labels", "model"))
    for directory in (features, labels, model):
        directory.mkdir()
    (model / "args.json").write_text('{}')
    (model / "best_weights.pt").write_bytes(b'checkpoint')
    monkeypatch.setattr(ipw, "get_model_path", lambda _: model)
    monkeypatch.setattr(ipw, "load_model", lambda *_: Predictor())
    for match in range(3):
        graphs = []
        rows = torch.zeros(4, len(config.LABEL_COLUMNS))
        for row in range(4):
            n = 3 + row
            x = torch.zeros(n, config.NODE_FEATURE_CORE_DIM)
            x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
            x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
            x[:, config.NODE_FEATURE_X] = torch.arange(n) * (match + 1)
            graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 5))
            graphs.append(graph)
            for key, value in {"action_index": row // 2, "is_pass": row != 2, "is_dribble": row == 2,
                               "intent_index": 0 if row == 2 else 1, "duration": 1,
                               "pass_high": row % 2, "pass_max_ball_z": 2}.items():
                rows[row, config.LABEL_INDEX[key]] = value
        rows[3, config.LABEL_INDEX["duration"]] = 0.01  # filtered row
        torch.save(graphs, features / f"m{match}.pt")
        torch.save(rows, labels / f"m{match}.pt")
    options = {"task": "pass_success", "min_pass_dur": 0.1, "extend_features": False}
    def create(**overrides):
        kwargs = dict(model_id="pass_intent/fixture", model_args=Predictor.args, feature_dir=features,
                      label_dir=labels, options=options, batch_size=2, cache_dir=tmp_path / "cache")
        kwargs.update(overrides)
        return ipw.IPWPreparer(**kwargs)
    main = ActionDataset(["m0", "m1"], feature_dir=features, label_dir=labels, **options)
    return create, main, features, labels, options, model


def test_eager_equivalence_reuse_rng_and_lifetime(setup, monkeypatch):
    create, main, features, labels, options, _ = setup
    with patch.object(utils, "load_model", return_value=Predictor().eval()):
        eager = utils.estimate_propensity(main, device="cpu", pin_memory=False)
    expected = ipw.normalized_ipw(eager, main.labels)
    refs = []
    original = ipw.ActionDataset
    def bounded(*args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in refs)
        result = original(*args, **kwargs)
        refs.append(weakref.ref(result))
        # Exercise isolation even if future preparation consumes randomness.
        random.random(); np.random.rand(); torch.rand(1)
        return result
    # Keep the actual constructor signature for bound default resolution.
    bounded.__signature__ = __import__('inspect').signature(original)
    preparer = create()
    monkeypatch.setattr(ipw, "ActionDataset", bounded)
    initial = capture_rng()
    weights, stats = preparer.prepare(main, ["m0", "m1"])
    after = capture_rng()
    assert torch.equal(initial["torch"], after["torch"])
    assert initial["python"] == after["python"]
    assert np.array_equal(initial["numpy"][1], after["numpy"][1])
    assert stats["misses"] == 2 and all(r() is None for r in refs)
    torch.testing.assert_close(weights, expected)
    assert weights.device.type == "cpu" and not weights.requires_grad
    monkeypatch.setattr(ipw, "ActionDataset", original)
    reused = create()
    with patch.object(ipw, "load_model", side_effect=AssertionError("loaded model")), \
         patch.object(ipw, "ActionDataset", side_effect=AssertionError("built graphs")):
        cached, stats = reused.prepare(main, ["m0", "m1"])
    assert stats["hits"] == 2 and reused.model is None
    torch.testing.assert_close(cached, weights, rtol=0, atol=0)
    off = create(cache="off", cache_dir=features / "unused")
    streamed, _ = off.prepare(main, ["m0", "m1"])
    torch.testing.assert_close(streamed, weights)
    assert not off.cache_root.exists()
    assert torch.equal(torch.get_rng_state(), initial["torch"])
    expected_order = [g.evaluation_source_row.tolist() for g, _, _ in DataLoader(main, batch_size=2, shuffle=True)]
    restore_rng(initial)
    actual_order = [g.evaluation_source_row.tolist() for g, _, _ in DataLoader(main, batch_size=2, shuffle=True)]
    assert actual_order == expected_order


def test_split_normalization_alignment_and_pass_height(setup):
    create, main, features, labels, options, _ = setup
    preparer = create()
    first, _ = preparer.prepare(main, ["m0", "m1"])
    valid = ActionDataset(["m2"], feature_dir=features, label_dir=labels, **options)
    second, _ = preparer.prepare(valid, ["m2"], split="validation")
    for data, weights in ((main, first), (valid, second)):
        carries = data.labels[:, config.LABEL_INDEX["is_dribble"]] == 1
        torch.testing.assert_close(weights[~carries].mean(), torch.tensor(1.))
        assert (weights[carries] == 1).all()
    main.features.reverse()
    main.labels = main.labels.flip(0)
    reordered, _ = preparer.prepare(main, ["m0", "m1"])
    torch.testing.assert_close(reordered, first.flip(0))
    main.labels[0, config.LABEL_INDEX["intent_index"]] = 99
    with pytest.raises(ValueError, match="alignment"):
        preparer.prepare(main, ["m0", "m1"])
    height_options = dict(options, task="pass_height")
    height = ActionDataset(["m0"], feature_dir=features, label_dir=labels, **height_options)
    height_weights, _ = create(options=height_options).prepare(height, ["m0"])
    assert len(height_weights) == len(height) > 0


def test_cache_invalidation_corruption_and_interruption(setup, monkeypatch):
    create, main, features, labels, options, model = setup
    preparer = create()
    original = ipw.predict_probabilities
    def fail_second(dataset, *args, **kwargs):
        if dataset.features[0].evaluation_match_id == "m1":
            raise RuntimeError("interrupted")
        return original(dataset, *args, **kwargs)
    with patch.object(ipw, "predict_probabilities", side_effect=fail_second):
        with pytest.raises(RuntimeError, match="interrupted"):
            preparer.prepare(main, ["m0", "m1"])
    _, stats = create().prepare(main, ["m0", "m1"])
    assert stats["hits"] == stats["misses"] == 1
    entry = next(preparer.cache_root.glob('*.json'))
    entry.write_text('corrupt')
    _, stats = create().prepare(main, ["m0", "m1"])
    assert stats["misses"] == 1
    source = features / "m0.pt"
    source.touch()
    _, stats = create().prepare(main, ["m0", "m1"])
    assert stats["misses"] == 1
    assert create(batch_size=8, pin_memory=True).cache_id == preparer.cache_id
    assert create(options=dict(options, min_pass_dur=.2)).cache_id != preparer.cache_id
    (model / "best_weights.pt").write_bytes(b'new checkpoint')
    assert create().cache_id != preparer.cache_id
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        preparer.prepare(main, ["m0", "m1"])
    changed = create()
    monkeypatch.setattr(ipw, "inference_fingerprint", lambda: "changed")
    with pytest.raises(RuntimeError, match="code changed"):
        changed.prepare(main, ["m0", "m1"])


def test_empty_missing_and_duplicate_samples(setup):
    create, main, features, labels, options, _ = setup
    empty = ActionDataset(["missing"], feature_dir=features, label_dir=labels, **options)
    preparer = create()
    weights, _ = preparer.prepare(empty, ["missing"])
    assert weights.numel() == 0 and preparer.model is None
    with pytest.raises(ValueError, match="outside"):
        preparer.prepare(main, ["m0"])
    with pytest.raises(ValueError, match="Duplicate match"):
        preparer.prepare(main, ["m0", "m0"])
    main.features.append(main.features[0].clone())
    main.labels = torch.cat([main.labels, main.labels[:1]])
    records = ipw.sample_records(main)
    assert records[-1][:2] == records[0][:2] and records[-1][2] == 1
    with pytest.raises(ValueError, match="alignment"):
        preparer.prepare(main, ["m0", "m1"])


def test_options_forwarding_defaults_and_monitoring_off(setup, tmp_path):
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    parser = argparse.ArgumentParser()
    add_ipw_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.ipw_batch_size == 256 and defaults.ipw_probability_cache == 'on'
    for value in ('0', '-1'):
        with pytest.raises(SystemExit):
            parser.parse_args(['--ipw-batch-size', value])
    old = SimpleNamespace()
    restore_ipw_defaults(old)
    assert vars(old) == vars(defaults)
    old.ipw_batch_size = 128
    restore_ipw_defaults(old)
    assert old.ipw_batch_size == 128
    args = make_training_args('feature_run', pass_success_ipw=False, ipw_batch_size=128,
                              ipw_probability_cache='off', ipw_probability_cache_dir='somewhere')
    with patch.object(wrapper, 'resolve_feature_run_id', return_value='feature_run'), \
         patch.object(wrapper, 'resolve_feature_root', return_value=Path('fixture_features')):
        for command in wrapper.build_training_commands(args)[0]:
            for flag, value in zip(ipw_flags(args)[::2], ipw_flags(args)[1::2]):
                assert wrapper.get_cli_value(command, flag) == value
    create, main, *_ = setup
    monitor = TrainingMonitor(tmp_path / 'monitor', enabled=False)
    create(monitor=monitor).prepare(main, ['m0', 'm1'])
    assert not (tmp_path / 'monitor' / 'training_monitor.jsonl').exists()


def test_inference_fingerprint_excludes_training_helpers(monkeypatch):
    original = Path.read_text
    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.as_posix().endswith('models/utils.py'):
            text += '\ndef unrelated_training_metric():\n    return 123\n'
        return text
    before = ipw.inference_fingerprint()
    with patch.object(Path, 'read_text', changed):
        assert ipw.inference_fingerprint() == before


def test_sidecar_signatures_and_source_mutation_guard(setup):
    create, main, features, labels, options, _ = setup
    preparer = create(options=dict(options, diagnostic_label_dir=str(labels)))
    before = preparer._source_signatures('m0')
    assert str((labels / 'm0.pt').resolve()) in [s[0] for s in before]
    preparer = create()
    original = ipw.predict_probabilities
    def change_source(*args, **kwargs):
        result = original(*args, **kwargs)
        (features / 'm0.pt').touch()
        return result
    with patch.object(ipw, 'predict_probabilities', side_effect=change_source):
        with pytest.raises(RuntimeError, match='sources changed'):
            preparer.prepare(main, ['m0', 'm1'])
    assert not list(preparer.cache_root.glob('*.json'))


def test_duplicate_occurrences_and_invalid_cached_probabilities(setup):
    create, main, *_ = setup
    main.features.append(main.features[0].clone())
    main.labels = torch.cat([main.labels, main.labels[:1]])
    preparer = create()
    original = ipw.ActionDataset
    def duplicate(*args, **kwargs):
        data = original(*args, **kwargs)
        if args[0] == ['m0']:
            data.features.append(data.features[0].clone())
            data.labels = torch.cat([data.labels, data.labels[:1]])
        return data
    with patch.object(ipw, 'ActionDataset', side_effect=duplicate):
        weights, _ = preparer.prepare(main, ['m0', 'm1'])
        torch.testing.assert_close(weights[0], weights[-1])
        path = preparer.cache_root / (ipw._digest('m0') + '.json')
        entry = json.loads(path.read_text())
        entry['payload']['probabilities'][0] = 2.0
        entry['sha256'] = ipw._digest(entry['payload'])
        path.write_text(json.dumps(entry))
        _, stats = preparer.prepare(main, ['m0', 'm1'])
        assert stats['misses'] == stats['hits'] == 1
