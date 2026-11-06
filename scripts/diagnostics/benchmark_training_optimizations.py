"""Compare runtime optimizations with a Git baseline on trusted real matches.

Creates isolated disposable caches under --output; never changes source runs.
Run while training is idle. Metric replay covers a full split even when the
GPU trial uses a bounded subset. Profiling and throughput runs are separate.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
from torch_geometric.data import Batch

from cache_lifetime import CacheWorkflow
from dataset import ActionDataset
from datatools import config
from models.dataset_config import build_action_dataset_kwargs
from models.gnn import GNN
import models.node_selection as selection
import models.utils as utils
import packed_batch_loader as batching
import ipw_preparation as ipw
from prepared_dataset import PreparedActionDataset, _signature
from training_state import environment_versions
from scripts.diagnostics.benchmark_training_pipeline import assert_value_equal


def legacy_function(module, filename, name, revision):
    source = subprocess.check_output(['git', 'show', f'{revision}:{filename}'], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


def measure(function, repeats, cuda=False):
    values = []
    function()  # warm-up is excluded
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if cuda:
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(values), seconds=values)


class ProcessMemory:
    def __enter__(self):
        self.stop = threading.Event()
        self.rss = self.private = 0
        def sample():
            parent = psutil.Process()
            while not self.stop.is_set():
                rss = private = 0
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        info = process.memory_info()
                        rss += info.rss
                        private += getattr(info, 'private', info.rss)
                    except psutil.Error:
                        pass
                self.rss, self.private = max(self.rss, rss), max(self.private, private)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class LimitedLoader:
    def __init__(self, loader, steps):
        self.loader, self.steps, self.collate_fn = loader, min(steps, len(loader)), loader.collate_fn
    def __len__(self):
        return self.steps
    def __iter__(self):
        iterator = iter(self.loader)
        try:
            for _ in range(self.steps):
                yield next(iterator)
        finally:
            iterator.close()


class NodeScores(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    def forward(self, graphs):
        return graphs.x[:, config.NODE_FEATURE_X] / 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-revision', default='HEAD')
    parser.add_argument('--matches', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--existing-cache', action='store_true', help='Read the source run cache without modifying ownership or entries.')
    parser.add_argument('--skip-warmup', action='store_true', help='For full-epoch confirmation after short warmed trials.')
    parser.add_argument('--phases', choices=('both', 'train', 'validation'), default='both')
    parser.add_argument('--record-gpu-drift', action='store_true', help='Record long-run CUDA drift; exact target/order checks still apply.')
    parser.add_argument('--workers', default='1,2,4')
    parser.add_argument('--suite', choices=('prepare', 'components', 'training'), default='components')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    options = parser.parse_args()
    if min(options.matches, options.repeats, options.steps) < 1:
        parser.error('matches, repeats and steps must be positive')
    for process in psutil.process_iter(['cmdline']):
        try:
            if any(Path(arg).name == 'train.py' for arg in process.info['cmdline'] or []):
                raise RuntimeError(f'Training {process.pid} is active')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    options.output.mkdir(parents=True, exist_ok=False)
    options.threads = options.threads or (8 if options.suite == 'training' else 1)
    torch.set_num_threads(options.threads)
    metadata = json.loads(options.source_metadata.read_text())
    args = metadata['training_args']
    dirs = metadata.get('resolved_dirs', {})
    feature_dir = dirs.get('train_feature_dir') or args.get('train_feature_dir') or args['feature_dir']
    label_dir = dirs.get('train_label_dir') or args.get('train_label_dir') or args['label_dir']
    all_ids = metadata['train_match_ids']
    ids = [all_ids[int(i)] for i in np.linspace(0, len(all_ids) - 1, min(options.matches, len(all_ids)))]
    kwargs = build_action_dataset_kwargs(args, train=True, diagnostic_label_dir=args.get('diagnostic_label_dir'),
        physical_cache_dir=args.get('physical_cache_dir'), lane_survival_cache_dir=args.get('lane_survival_cache_dir'),
        require_goal_next10_diagnostics=args.get('require_goal_next10_diagnostics', False))
    report = dict(environment=environment_versions(), baseline_revision=subprocess.check_output(
        ['git', 'rev-parse', options.baseline_revision], text=True).strip(), selected_matches=ids,
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}, results={})
    def save():
        (options.output / 'results.json').write_text(json.dumps(report, indent=2))
    def build(root, workers=1, workflow=None):
        return PreparedActionDataset(ids, feature_dir=feature_dir, label_dir=label_dir, cache_dir=root,
            progress=False, shuffle=True, seed=args['seed'], preparation_workers=workers,
            cache_session=None if workflow is None else workflow.session_for(root), **kwargs)
    save()
    if options.suite == 'prepare':
        trials = [(int(w), r) for w in options.workers.split(',') for r in range(options.repeats)]
        random.Random(100).shuffle(trials)
        report['trials'] = []
        for workers, repeat in trials:
            root = options.output / f'cache_{workers}_{repeat}'
            workflow = CacheWorkflow(root)
            success = False
            try:
                with ProcessMemory() as memory, redirect_stdout(io.StringIO()):
                    ds = build(root, workers, workflow)
                report['trials'].append(dict(workers=workers, repeat=repeat, preparation=ds.preparation,
                    peak_rss_bytes=memory.rss, peak_private_bytes=memory.private, samples=len(ds)))
                success = True
            finally:
                workflow.finish(success)
            save()
            print(f'Preparation workers={workers}: {ds.preparation["seconds"]:.2f}s', flush=True)
        return

    root = options.output / 'cache'
    workflow = None if options.existing_cache else CacheWorkflow(root)
    success = False
    try:
        if options.existing_cache:
            if options.suite != 'training':
                raise ValueError('--existing-cache requires --suite training')
            saved = metadata['dataset_loading']['train']
            manifests = {}
            for path in Path(saved['cache_root']).glob('*.json'):
                entry = json.loads(path.read_text())
                manifests[entry['match_id']] = (path, entry)
            ds = PreparedActionDataset.__new__(PreparedActionDataset)
            ds.feature_dir, ds.label_dir = Path(feature_dir), Path(label_dir)
            ds.options = kwargs
            ds.cache_id = saved['cache_id']
            ds.pass_height_threshold = saved.get('pass_height_threshold_meters')
            ds.seed, ds.epoch, ds.shuffle, ds.buffer_matches = args['seed'], 0, True, 4
            ds.ip_weights, ds.entries, ds.sample_count = None, [], 0
            ds._reconstruction_timing = {'seconds': 0.0}
            ds.last_load_seconds = ds.last_iteration_seconds = 0.0
            ds._source_signatures = lambda match: [_signature(s[0]) for s in manifests[match][1]['sources']]
            for match in ids:
                path, entry = manifests[match]
                if Path(entry['sources'][0][0]).resolve() != (Path(feature_dir) / (match + '.pt')).resolve():
                    raise ValueError('Saved cache feature path does not match source run')
                if Path(entry['sources'][1][0]).resolve() != (Path(label_dir) / (match + '.pt')).resolve():
                    raise ValueError('Saved cache label path does not match source run')
                verified = ds._read_entry(path.with_suffix('.pt'), match, ds._source_signatures(match))
                if verified is None:
                    raise ValueError(f'Existing cache failed validation: {match}')
                verified['offset'] = ds.sample_count
                ds.entries.append(verified)
                ds.sample_count += verified['count']
            report['existing_cache_read_only'] = True
        else:
            with redirect_stdout(io.StringIO()):
                ds = build(root, workflow=workflow)
        if not options.existing_cache and args['task'] in {'outcome_scoring', 'outcome_conceding'}:
            other = 'outcome_conceding' if args['task'] == 'outcome_scoring' else 'outcome_scoring'
            for entry in ds.entries[:2]:
                with redirect_stdout(io.StringIO()):
                    eager = ActionDataset([entry['match_id']], feature_dir=feature_dir,
                                         label_dir=label_dir, **dict(kwargs, task=other))
                payload = ds._load_entry(entry)
                assert len(eager) == len(payload['graphs'])
                assert_value_equal(eager.labels, payload['labels'])
                for i, graph in enumerate(payload['graphs']):
                    assert_value_equal(eager.features[i].to_dict(), graph.to_dict())
                del eager, payload
            report['outcome_pair_real_matches_equal'] = min(2, len(ds.entries))
        if options.suite == 'components':
            payloads = [ds._load_entry(entry) for entry in ds.entries[:4]]
            rows = [(m, r) for m, payload in enumerate(payloads) for r in range(len(payload['graphs']))]
            random.Random(100).shuffle(rows)
            rows = rows[:512]
            old_batch = legacy_function(batching, 'packed_batch_loader.py', 'direct_packed_batch', options.baseline_revision)
            new_values, old_values = batching.direct_packed_batch(payloads, rows), old_batch(payloads, rows)
            assert_value_equal(new_values[0].to_dict(), old_values[0].to_dict())
            assert_value_equal(new_values[0]._slice_dict, old_values[0]._slice_dict)
            assert_value_equal(new_values[0]._inc_dict, old_values[0]._inc_dict)
            assert_value_equal(new_values[1:], old_values[1:])
            for label, function in [('baseline', old_batch), ('optimized', batching.direct_packed_batch)]:
                report['results']['batch_' + label] = measure(lambda: function(payloads, rows), options.repeats)
            graphs, labels, _ = new_values
            intent_args = dict(args, task='pass_intent', gnn_task='node_selection', include_goals=False)
            old_layout = legacy_function(selection, 'models/node_selection.py', 'selection_layout', options.baseline_revision)
            assert_value_equal(old_layout(graphs, labels, 'pass_intent', False, intent_args),
                               selection.selection_layout(graphs, labels, 'pass_intent', False, intent_args))
            for label, function in [('baseline', old_layout), ('optimized', selection.selection_layout)]:
                report['results']['selection_' + label] = measure(
                    lambda: function(graphs, labels, 'pass_intent', False, intent_args), options.repeats)
            samples = [(payloads[m]['graphs'][r], payloads[m]['labels'][r], torch.tensor(1.)) for m, r in rows]
            model = NodeScores(intent_args).to(options.device)
            old_ipw = legacy_function(ipw, 'ipw_preparation.py', 'predict_probabilities', options.baseline_revision)
            def predict(function):
                return function(samples, model, device=options.device, batch_size=256, pin_memory=False)
            torch.testing.assert_close(predict(old_ipw), predict(ipw.predict_probabilities), rtol=1e-6, atol=1e-7)
            for label, function in [('baseline', old_ipw), ('optimized', ipw.predict_probabilities)]:
                report['results']['ipw_' + label] = measure(lambda: predict(function), options.repeats,
                                                           options.device == 'cuda')
            rng = np.random.default_rng(100)
            count = metadata['dataset_loading']['train']['sample_count']
            targets, predictions = rng.binomial(1, .025, count), rng.uniform(.001, .999, count)
            def metrics(full):
                if full:
                    for end in range(512, count + 1, 512 * 50):
                        utils.calc_binary_metrics(targets[:end], predictions[:end], include_calibration=False)
                return utils.calc_binary_metrics(targets, predictions)
            for label, full in [('baseline', True), ('optimized', False)]:
                report['results']['full_split_metrics_' + label] = measure(lambda: metrics(full), options.repeats)
            report['exact_batches_and_selection'] = True
        else:
            initial = torch.load(options.source_metadata.parent / 'best_weights.pt', map_location='cpu', weights_only=False)
            old_batch = legacy_function(batching, 'packed_batch_loader.py', 'direct_packed_batch', options.baseline_revision)
            old_layout = legacy_function(selection, 'models/node_selection.py', 'selection_layout', options.baseline_revision)
            old_epoch = legacy_function(utils, 'models/utils.py', 'run_epoch', options.baseline_revision)
            report['trials'] = []
            references = {}
            phases = ('train', 'validation') if options.phases == 'both' else (options.phases,)
            def trial(phase, label):
                random.seed(100); np.random.seed(100); torch.manual_seed(100)
                model = GNN(copy.deepcopy(args)).to(options.device)
                model.load_state_dict(initial)
                optimizer = torch.optim.Adam(model.parameters(), lr=args['start_lr'])
                settings = SimpleNamespace(**dict(args, interim_metrics='loss', monitoring='off'))
                ds.shuffle = phase == 'train'
                ds.buffer_matches = 4 if ds.shuffle else 1
                actual_loader = batching.make_batch_loader(ds, batch_size=args['batch_size'])
                loader = LimitedLoader(actual_loader, options.steps)
                with ExitStack() as stack, redirect_stdout(io.StringIO()):
                    if label == 'baseline':
                        stack.enter_context(patch.object(batching, 'direct_packed_batch', old_batch))
                        stack.enter_context(patch.object(selection, 'selection_layout', old_layout))
                    function = old_epoch if label == 'baseline' else utils.run_epoch
                    if options.device == 'cuda': torch.cuda.synchronize()
                    started = time.perf_counter()
                    metrics, evaluation = function(settings, model, loader, optimizer, options.device,
                        train=phase == 'train', return_outcome_evaluation=True)
                    if options.device == 'cuda': torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                return elapsed, metrics, evaluation, state, actual_loader.last_batching_seconds
            if not options.skip_warmup:
                for phase in phases:
                    for label in ('baseline', 'optimized'):
                        trial(phase, label)
            trials = [(phase, label, repeat) for phase in phases
                      for label in ('baseline', 'optimized') for repeat in range(options.repeats)]
            random.Random(100).shuffle(trials)
            for phase, label, repeat in trials:
                elapsed, metrics, evaluation, state, batching_seconds = trial(phase, label)
                prior_evaluation, prior_state = references.setdefault(phase, (evaluation, state))
                for field in ('target', 'diagnostic', 'execution_branch'):
                    np.testing.assert_array_equal(prior_evaluation[field], evaluation[field])
                if not options.record_gpu_drift:
                    np.testing.assert_allclose(prior_evaluation['prediction'], evaluation['prediction'], rtol=1e-5, atol=1e-6)
                difference = max((state[k] - prior_state[k]).abs().max().item() for k in state)
                report['trials'].append(dict(phase=phase, variant=label, repeat=repeat,
                    seconds=elapsed, metrics=metrics, iteration_seconds=ds.last_iteration_seconds,
                    batching_seconds=batching_seconds, load_seconds=ds.last_load_seconds,
                    max_parameter_difference=difference,
                    max_prediction_difference=float(np.max(np.abs(prior_evaluation['prediction'] - evaluation['prediction']))),
                    samples=len(evaluation['prediction'])))
                save()
                print(f'{phase} {label}: {elapsed:.2f}s', flush=True)
        success = True
    finally:
        report['cache_cleanup'] = workflow.finish(success) if workflow is not None else 'source cache unmodified'
        save()
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()
