"""Cache roots, saved-path resolution, and independent model lifetimes."""
import argparse
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import dataset_loading as loading
from cache_lifetime import (evaluation_context, training_context,
                            workflow_context, process_identity, workflow_running)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    paths = {purpose: tmp_path / 'cache' / purpose for purpose in ('training', 'evaluation')}
    monkeypatch.setattr(loading, 'DEFAULT_TRAINING_CACHE_DIR', str(paths['training']))
    monkeypatch.setattr(loading, 'DEFAULT_EVALUATION_CACHE_DIR', str(paths['evaluation']))
    monkeypatch.setattr(loading, 'LEGACY_CACHE_DIR', str(tmp_path / 'legacy'))
    return paths


def write_entry(session, name='b'):
    path = session.root / ('a' * 32) / (name * 32 + '.pt')
    with session.entry(path):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'graphs')
        session.recorded(path, built=True, size=6)
    return path


def test_defaults_are_repository_anchored_and_have_separate_parsers(tmp_path, monkeypatch):
    repo = Path(loading.__file__).resolve().parent
    monkeypatch.chdir(tmp_path)
    assert Path(loading.DEFAULT_TRAINING_CACHE_DIR) == repo / 'cache' / 'training'
    assert Path(loading.DEFAULT_EVALUATION_CACHE_DIR) == repo / 'cache' / 'evaluation'
    for purpose in ('training', 'evaluation'):
        parser = argparse.ArgumentParser()
        loading.add_dataset_loading_arguments(parser, purpose=purpose)
        args = parser.parse_args([])
        assert args.dataset_cache_dir == loading.default_cache_dir(purpose)
        assert not args._dataset_cache_dir_explicit
        custom = parser.parse_args(['--dataset-cache-dir', 'exact root'])
        assert custom._dataset_cache_dir_explicit
        assert loading.configure_dataset_cache(custom, purpose) == str(tmp_path / 'exact root')


@pytest.mark.parametrize('purpose', ['training', 'evaluation'])
def test_saved_legacy_redirects_but_explicit_and_custom_paths_win(roots, tmp_path, purpose):
    legacy = loading.LEGACY_CACHE_DIR
    assert loading.resolve_dataset_cache_dir(purpose, saved=legacy) == str(roots[purpose])
    assert loading.resolve_dataset_cache_dir(purpose) == str(roots[purpose])
    assert loading.resolve_dataset_cache_dir(purpose, saved=tmp_path / 'custom') == str(tmp_path / 'custom')
    assert loading.resolve_dataset_cache_dir(purpose, requested=legacy) == legacy
    parser = argparse.ArgumentParser()
    loading.add_dataset_loading_arguments(parser, purpose=purpose)
    args = parser.parse_args([])
    assert loading.configure_dataset_cache(args, purpose, saved={'dataset_cache_dir': tmp_path / 'custom'}) == str(tmp_path / 'custom')
    args = parser.parse_args(['--dataset-cache-dir', legacy])
    assert loading.configure_dataset_cache(args, purpose, saved={}) == legacy


def test_model_start_pruning_is_independent_in_both_directions(roots):
    paths = {}
    for purpose, context in [('training', training_context), ('evaluation', evaluation_context)]:
        with context(SimpleNamespace(keep_cache=False)) as session:
            path = session.root / ('a' * 32) / ('b' * 32 + '.pt')
            session.start_model([path])
            paths[purpose] = write_entry(session)
    evaluation_db = (roots['evaluation'] / '.ownership.sqlite').read_bytes()
    with training_context(SimpleNamespace(keep_cache=False)) as session:
        desired = session.root / ('a' * 32) / ('c' * 32 + '.pt')
        report = session.start_model([desired])
        assert report['deleted_entries'] == 1
        paths['training'] = write_entry(session, 'c')
    assert paths['evaluation'].exists()
    assert (roots['evaluation'] / '.ownership.sqlite').read_bytes() == evaluation_db
    training_db = (roots['training'] / '.ownership.sqlite').read_bytes()
    with evaluation_context(SimpleNamespace(keep_cache=False)) as session:
        report = session.start_model([])
        assert report['deleted_entries'] == 1
    assert paths['training'].exists() and not paths['evaluation'].exists()
    assert (roots['training'] / '.ownership.sqlite').read_bytes() == training_db


def test_evaluation_does_not_inherit_training_owner_or_keep(roots):
    with workflow_context(SimpleNamespace(keep_cache=True)) as training:
        with evaluation_context(SimpleNamespace(keep_cache=False)) as evaluator:
            assert evaluator.root == roots['evaluation']
            assert evaluator.id != training.descriptor['id']
            assert not evaluator.descriptor.get('parent')
            assert not evaluator.descriptor['keep']
            assert evaluator.descriptor['purpose'] == 'evaluation'
            evaluator.start_model([])
    assert training.report['model_starts'] == []
    assert [Path(r['root']) for r in training.report['roots']] == [roots['training']]


@pytest.mark.parametrize('purpose,context', [('training', training_context), ('evaluation', evaluation_context)])
def test_explicit_keep_protects_obsolete_entries_in_each_root(roots, purpose, context):
    with context(SimpleNamespace(keep_cache=True)) as session:
        path = write_entry(session)
    with context(SimpleNamespace(keep_cache=False)) as session:
        report = session.start_model([])
        assert report['retention_reasons']['keep_cache'] == 1
        assert path.exists()
    other = 'evaluation' if purpose == 'training' else 'training'
    assert not roots[other].exists()


def test_legacy_liveness_and_resume_leave_old_database_unchanged(roots):
    legacy = Path(loading.LEGACY_CACHE_DIR)
    legacy.mkdir()
    old_db = legacy / '.ownership.sqlite'
    with sqlite3.connect(old_db) as db:
        db.execute('CREATE TABLE sessions(id TEXT, status TEXT)')
        db.execute("INSERT INTO sessions VALUES ('old', 'completed')")
    pid, created = process_identity()
    descriptor = dict(id='old', root=str(legacy), pid=pid, created=created, kind='direct')
    original = old_db.read_bytes()
    assert not workflow_running(descriptor)
    args = SimpleNamespace(dataset_cache_dir=str(legacy), resume_checkpoint='old.pt',
                           cache_session=descriptor, keep_cache=False)
    with training_context(args) as session:
        assert session.root == roots['training']
        session.start_model([])
    assert old_db.read_bytes() == original
    assert {p.name for p in legacy.iterdir()} == {'.ownership.sqlite'}


@pytest.mark.parametrize('train_flag', ['--dataset-cache-dir', '--training-dataset-cache-dir'])
def test_pipeline_forwards_separate_root_overrides(tmp_path, monkeypatch, train_flag):
    import sys
    from scripts import main
    train, evaluation = str(tmp_path / 'train'), str(tmp_path / 'evaluate')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--train-count', '765', '--target-family', 'xt',
        '--return_type', 'disc_0.9', '--intended-receiver-mode', 'original',
        train_flag, train, '--evaluation-dataset-cache-dir', evaluation])
    commands = main.build_commands(main.parse_args())
    for script, expected in [('train_relevant_models.py', train), ('evaluate_relevant_models.py', evaluation)]:
        command = next(c for c in commands if c[1] == 'scripts/' + script)
        assert command[command.index('--dataset-cache-dir') + 1] == expected


def test_training_wrapper_accepts_resume_cache_override(tmp_path):
    from scripts import train_relevant_models as wrapper
    args = wrapper.parse_args(['--resume-id', 'pass_intent/source', '--dataset-cache-dir', str(tmp_path)])
    assert args._dataset_cache_dir_explicit and args.dataset_cache_dir == str(tmp_path)
