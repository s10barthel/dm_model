"""Short production-loader comparison on existing trusted cache matches.

Checks source signatures/checksums; never prepares caches or writes checkpoints.
Validation uses training examples for throughput, not holdout evaluation.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import gc
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from models.gnn import GNN
from models.utils import run_epoch
from packed_batch_loader import make_batch_loader, batch_loader_metadata
from prepared_dataset import PreparedActionDataset
from training_monitor import TrainingMonitor
from training_state import environment_versions
from scripts.diagnostics.benchmark_training_pipeline import assert_value_equal, write_json


def cache_view(args, metadata, shuffle, matches=4):
    cache = Path(metadata['dataset_loading']['train']['cache_root'])
    manifests = {json.loads(p.read_text())['match_id']: p for p in cache.glob('*.json')}
    ids = metadata['train_match_ids']
    selected = [ids[i*(len(ids)-1)//(matches-1)] for i in range(matches)]
    if matches > len(ids):
        raise ValueError('Requested more matches than the training split contains')
    ds = PreparedActionDataset.__new__(PreparedActionDataset)
    ds.feature_dir, ds.label_dir = Path(args['feature_dir']).resolve(), Path(args['label_dir']).resolve()
    ds.options = {'diagnostic_label_dir': args.get('diagnostic_label_dir')}
    ds.cache_id = metadata['dataset_loading']['train']['cache_id']
    ds.pass_height_threshold = None
    ds.buffer_matches = args.get('dataset_buffer_matches', 4) if shuffle else 1
    ds.seed, ds.epoch, ds.shuffle = args['seed'], 2, shuffle
    ds.ip_weights, ds.entries, ds.sample_count = None, [], 0
    ds._reconstruction_timing = {'seconds': 0.0}
    ds.last_load_seconds = ds.last_iteration_seconds = 0.0
    for match in selected:
        entry = ds._read_entry(manifests[match].with_suffix('.pt'), match, ds._source_signatures(match))
        if entry is None:
            raise RuntimeError(f'Existing cache failed integrity checks: {match}')
        entry['offset'] = ds.sample_count
        ds.entries.append(entry)
        ds.sample_count += entry['count']
    return ds, selected


class TimedLimit:
    def __init__(self, loader, steps):
        self.loader, self.steps, self.seconds = loader, min(steps, len(loader)), 0.0
        self.collate_fn = loader.collate_fn
    def __len__(self):
        return self.steps
    def __iter__(self):
        iterator = iter(self.loader)
        try:
            for _ in range(self.steps):
                start = time.perf_counter()
                batch = next(iterator)
                self.seconds += time.perf_counter() - start
                yield batch
        finally:
            if hasattr(iterator, 'close'):
                iterator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--matches', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--contention', choices=['isolated', 'pc-xpass-active'], required=True)
    options = parser.parse_args()
    if options.steps < 1 or options.repeats < 1 or options.matches < 2:
        parser.error('steps and repeats must be positive; matches must be at least two')
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in (p.info['name'] or '').lower() and any(Path(a).name == 'train.py' for a in p.cmdline()):
                raise RuntimeError(f'Training {p.pid} is active')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    options.output.mkdir(parents=True, exist_ok=True)
    args = json.loads((options.run/'args.json').read_text())
    metadata = json.loads((options.run/'metadata.json').read_text())
    torch.set_num_threads(8)
    datasets = {phase: cache_view(args, metadata, phase == 'training', options.matches)[0] for phase in ('training','validation')}
    for phase, ds in datasets.items():
        loaders = [TimedLimit(make_batch_loader(ds, batch_size=args['batch_size'], batch_loader=m), options.steps) for m in ('pyg','auto')]
        for left, right in zip(*loaders):
            for a, b in zip(left, right):
                assert_value_equal(a.to_dict() if hasattr(a,'to_dict') else a, b.to_dict() if hasattr(b,'to_dict') else b)
            assert_value_equal(left[0]._slice_dict, right[0]._slice_dict)
            assert_value_equal(left[0]._inc_dict, right[0]._inc_dict)
        del loaders, left, right
    report = dict(environment=environment_versions(),contention=options.contention,run=str(options.run),
                  selected_matches=[e['match_id'] for e in datasets['training'].entries],steps=options.steps,
                  repeats=options.repeats,threads=8,batch_size=args['batch_size'],exact_batches=True,
                  timing_scope='Production loaders/run_epoch; warm file reads, mandatory validation, interim/final metrics included. Initialization and comparisons excluded.',trials=[])
    model = GNN(copy.deepcopy(args)).cuda()
    initial = torch.load(options.run/'best_weights.pt',map_location='cpu',weights_only=False)
    references = {}
    def trial(phase, mode, monitoring, repeat, warmup=False):
        random.seed(100);np.random.seed(100);torch.manual_seed(100);torch.cuda.manual_seed_all(100)
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(),lr=args['start_lr'])
        settings = SimpleNamespace(**copy.deepcopy(args))
        settings.monitoring, settings._epoch = monitoring, 2
        directory = options.output/f'{phase}_{mode}_{monitoring}_{repeat}'
        directory.mkdir(exist_ok=True)
        monitor = TrainingMonitor(directory,mode=monitoring)
        settings._monitor = monitor
        # Actual training starts the sampler before preparation/IPW. Exclude its
        # initial asynchronous startup/query from the short epoch measurement.
        if monitor.process is not None:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                paths = [directory/name for name in ('resource_monitor.jsonl','gpu_monitor.jsonl')]
                if all(p.exists() and p.stat().st_size for p in paths):
                    break
                time.sleep(.02)
        loader = make_batch_loader(datasets[phase],batch_size=args['batch_size'],batch_loader=mode,monitoring=monitoring)
        limited = TimedLimit(loader,options.steps)
        process, rss, stopped = psutil.Process(), [], threading.Event()
        def sample_rss():
            while not stopped.is_set():
                rss.append(process.memory_info().rss)
                stopped.wait(.1)
        sampler = threading.Thread(target=sample_rss,daemon=True)
        sampler.start()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            with (directory/'console.txt').open('w') as stream,contextlib.redirect_stdout(stream):
                metrics,evaluation = run_epoch(settings,model,limited,optimizer,'cuda',train=phase=='training',return_outcome_evaluation=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        finally:
            stopped.set();sampler.join();monitor.close()
        result = dict(phase=phase,loader=mode,monitoring=monitoring,repeat=repeat,seconds=elapsed,
                      ms_per_batch=1000*elapsed/len(limited),samples_per_second=len(evaluation['prediction'])/elapsed,
                      loader_seconds=limited.seconds,load_seconds=datasets[phase].last_load_seconds,
                      peak_rss_mib=max(rss)/1024**2,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
                      loader_metadata=batch_loader_metadata(loader))
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if not warmup:
            if phase not in references:
                references[phase] = state,evaluation
            else:
                prior,predictions = references[phase]
                result['max_parameter_difference'] = max((state[k]-prior[k]).abs().max().item() for k in state)
                result['max_prediction_difference'] = float(np.max(np.abs(evaluation['prediction']-predictions['prediction'])))
                for key in ('target','diagnostic','execution_branch'):
                    np.testing.assert_array_equal(evaluation[key],predictions[key])
                np.testing.assert_allclose(evaluation['prediction'],predictions['prediction'],rtol=1e-5,atol=1e-6)
            report['trials'].append(result)
            write_json(options.output/'results.json',report)
            print('TRIAL',json.dumps(result),flush=True)
        del optimizer,state
        gc.collect()
    for phase in datasets:
        trial(phase,'pyg','off',-1,warmup=True)
    cases = [(phase,mode,monitor) for phase in datasets for mode,monitor in [('pyg','off'),('auto','off'),('auto','light')]]
    for repeat in range(options.repeats):
        order = list(cases);random.Random(300+repeat).shuffle(order)
        for phase,mode,monitor in order:
            trial(phase,mode,monitor,repeat)
    report['summary'] = {}
    for phase,mode,monitor in cases:
        records = [r for r in report['trials'] if (r['phase'],r['loader'],r['monitoring']) == (phase,mode,monitor)]
        report['summary'][f'{phase}_{mode}_{monitor}'] = {key:statistics.median(r[key] for r in records) for key in
            ('seconds','ms_per_batch','samples_per_second','loader_seconds','load_seconds','peak_rss_mib','peak_gpu_mib')}
    write_json(options.output/'results.json',report)
    print('SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    main()
