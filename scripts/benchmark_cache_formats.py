"""Compare the graph-object baseline with the production packed-cache loader.

Each phase runs in a fresh process under a 24 GiB process-tree memory guard.
Packed trials exercise production preparation, validation, and iteration.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def dataset_kwargs(config, split):
    from models.dataset_config import build_action_dataset_kwargs
    args = config['args']
    return build_action_dataset_kwargs(args, train=split == 'training',
        diagnostic_label_dir=args.get('diagnostic_label_dir'),
        require_goal_next10_diagnostics=args.get('require_goal_next10_diagnostics'),
        physical_cache_dir=args.get('physical_cache_dir'), lane_survival_cache_dir=args.get('lane_survival_cache_dir'))


def atomic_save(payload, path):
    import torch
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    write_json(path.with_suffix('.json'), dict(bytes=path.stat().st_size, sha256=digest.hexdigest()))


def configuration(output):
    return json.loads((Path(output) / 'config.json').read_text())


def cache_path(output, mode, match_id):
    if mode == 'packed':
        return Path(configuration(output)['packed_paths'][match_id])
    return Path(output) / 'baseline' / (match_id + '.pt')


def build_cache(options, config):
    import torch
    from dataset import ActionDataset
    from prepared_dataset import PreparedActionDataset
    from cache_lifetime import CacheWorkflow
    session = CacheWorkflow(Path(options.output_dir) / 'production_cache', keep=True, kind='benchmark').session_for(
        Path(options.output_dir) / 'production_cache') if options.mode == 'packed' else None
    result = dict(preprocess_seconds=0., packing_seconds=0., save_and_hash_seconds=0., bytes=0, samples=0, matches=[])
    for split in ('training', 'validation'):
        dirs = config['dirs']
        prefix = 'train' if split == 'training' else 'valid'
        kwargs = dataset_kwargs(config, split)
        for match_id in config[split + '_ids']:
            if options.mode == 'packed':
                old = config.get('packed_paths', {}).get(match_id)
                if old:
                    # Isolated benchmark entries only; rebuild each creation trial.
                    target = Path(old).resolve()
                    target.relative_to((Path(options.output_dir) / 'production_cache').resolve())
                    target.unlink(missing_ok=True)
                    target.with_suffix('.json').unlink(missing_ok=True)
                started = time.perf_counter()
                dataset = PreparedActionDataset([match_id], feature_dir=dirs[prefix + '_feature_dir'],
                    label_dir=dirs[prefix + '_label_dir'], cache_dir=Path(options.output_dir) / 'production_cache',
                    cache_session=session, progress=False, **kwargs)
                elapsed = time.perf_counter() - started
                entry = dataset.entries[0]
                config.setdefault('packed_paths', {})[match_id] = entry['path']
                write_json(Path(options.output_dir) / 'config.json', config)
                row = dict(match_id=match_id, samples=len(dataset), bytes=entry['bytes'],
                    preprocess_seconds=dataset.preparation['transform_seconds'],
                    packing_seconds=dataset.preparation['packing_seconds'],
                    save_and_hash_seconds=dataset.preparation['save_and_hash_seconds'],
                    creation_seconds=elapsed)
                result['matches'].append(row)
                for name in ('preprocess_seconds', 'packing_seconds', 'save_and_hash_seconds', 'bytes', 'samples'):
                    result[name] += row[name]
                del dataset
                gc.collect()
                continue
            path = cache_path(options.output_dir, options.mode, match_id)
            path.parent.mkdir(exist_ok=True)
            started = time.perf_counter()
            with contextlib.redirect_stderr(io.StringIO()):
                dataset = ActionDataset([match_id], feature_dir=dirs[prefix + '_feature_dir'],
                                        label_dir=dirs[prefix + '_label_dir'], **kwargs)
            preprocess = time.perf_counter() - started
            started = time.perf_counter()
            if options.mode == 'baseline':
                payload = dict(graphs=dataset.features, labels=dataset.labels)
            packing = time.perf_counter() - started
            started = time.perf_counter()
            atomic_save(payload, path)
            saving = time.perf_counter() - started
            row = dict(match_id=match_id, samples=len(dataset), bytes=path.stat().st_size,
                       preprocess_seconds=preprocess, packing_seconds=packing, save_and_hash_seconds=saving)
            result['matches'].append(row)
            for name in ('preprocess_seconds', 'packing_seconds', 'save_and_hash_seconds', 'bytes', 'samples'):
                result[name] += row[name]
            del payload, dataset
            gc.collect()
    result['creation_seconds'] = sum(row.get('creation_seconds', row['preprocess_seconds'] + row['packing_seconds'] + row['save_and_hash_seconds']) for row in result['matches'])
    return result


class CacheDataset:
    """A bounded iterator matching production match-group shuffle and batching."""
    def __new__(cls, *args, **kwargs):
        # Delay torch imports until inside workers.
        from torch.utils.data import IterableDataset
        if cls is CacheDataset:
            child = type('CacheIterableDataset', (CacheDataset, IterableDataset), {})
            return object.__new__(child)
        return object.__new__(cls)

    def __init__(self, output, mode, ids, *, shuffle, seed, buffer_matches=4):
        self.output, self.mode, self.ids = output, mode, list(ids)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.buffer_matches = buffer_matches if shuffle else 1
        self.load_seconds, self.reconstruct_seconds = 0., 0.
        # Small manifests avoid opening the full cache during length queries.
        config = configuration(output)
        counts = config.get('sample_counts', {})
        self.count = sum(counts[m] for m in ids)
        # Mirror the old prepared reader's startup manifest/checksum validation.
        for match_id in self.ids:
            path = cache_path(output, 'baseline', match_id)
            manifest = json.loads(path.with_suffix('.json').read_text())
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b''):
                    digest.update(block)
            if path.stat().st_size != manifest['bytes'] or digest.hexdigest() != manifest['sha256']:
                raise ValueError(f'Invalid baseline cache: {match_id}')

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import torch
        rng = random.Random(f'{self.seed}:{self.epoch}')
        ids = list(self.ids)
        if self.shuffle:
            rng.shuffle(ids)
        self.load_seconds, self.reconstruct_seconds = 0., 0.
        group = []
        try:
            for offset in range(0, len(ids), self.buffer_matches):
                for match_id in ids[offset:offset + self.buffer_matches]:
                    started = time.perf_counter()
                    group.append(torch.load(cache_path(self.output, self.mode, match_id), weights_only=False,
                                            map_location='cpu', mmap=False))
                    self.load_seconds += time.perf_counter() - started
                rows = [(m, r) for m, p in enumerate(group) for r in range(len(p['labels']))]
                if self.shuffle:
                    rng.shuffle(rows)
                for m, row in rows:
                    graph = group[m]['graphs'][row]
                    yield graph, group[m]['labels'][row], torch.tensor(1.)
                group.clear()
        finally:
            group.clear()


class TimedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.waits = []
    def __len__(self):
        return len(self.loader)
    def __getattr__(self, key):
        return getattr(self.loader, key)
    def __iter__(self):
        self.waits = []
        iterator = iter(self.loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.waits.append(time.perf_counter() - started)
            yield batch


def make_dataset(options, config, split):
    if options.mode == 'baseline':
        return CacheDataset(options.output_dir, options.mode, config[split + '_ids'],
            shuffle=split == 'training', seed=config['args']['seed'],
            buffer_matches=config['args'].get('dataset_buffer_matches', 4))
    from prepared_dataset import PreparedActionDataset
    prefix = 'train' if split == 'training' else 'valid'
    return PreparedActionDataset(config[split + '_ids'], feature_dir=config['dirs'][prefix + '_feature_dir'],
        label_dir=config['dirs'][prefix + '_label_dir'], cache_dir=Path(options.output_dir) / 'production_cache',
        progress=False, shuffle=split == 'training', seed=config['args']['seed'],
        buffer_matches=config['args'].get('dataset_buffer_matches', 4), **dataset_kwargs(config, split))


def loaders(options, config):
    from torch_geometric.loader import DataLoader
    train = make_dataset(options, config, 'training')
    valid = make_dataset(options, config, 'validation')
    batch_size = config['args']['batch_size']
    return [(dataset, TimedLoader(DataLoader(dataset, batch_size=batch_size, num_workers=0,
                                            pin_memory=config['args'].get('pin_memory', False))))
            for dataset in (train, valid)]


def loader_statistics(dataset, loader):
    waits = loader.waits
    return dict(load_seconds=getattr(dataset, 'last_load_seconds', getattr(dataset, 'load_seconds', 0.)),
                reconstruct_seconds=getattr(dataset, 'last_reconstruct_seconds', getattr(dataset, 'reconstruct_seconds', 0.)),
                batch_delivery_seconds=sum(waits), first_batch_seconds=waits[0] if waits else 0,
                median_batch_seconds=statistics.median(waits) if waits else 0,
                max_batch_seconds=max(waits) if waits else 0, batches=len(waits))


def read_cache(options, config):
    started = time.perf_counter()
    pairs = loaders(options, config)
    result = dict(splits={}, dataset_setup_seconds=time.perf_counter() - started)
    for split, (dataset, loader) in zip(('training', 'validation'), pairs):
        dataset.set_epoch(1)
        started = time.perf_counter()
        graphs_seen = 0
        checksum = 0.
        for graphs, labels, weights in loader:
            graphs_seen += graphs.num_graphs
            # Collation touches every tensor; sums also consume the result.
            checksum += float(graphs.x.sum()) + float(graphs.edge_attr.sum()) + float(labels.sum())
        result['splits'][split] = dict(loader_statistics(dataset, loader), samples=graphs_seen,
                                       checksum=checksum, wall_seconds=time.perf_counter() - started)
    return result


def equal_samples(first, second):
    import torch
    ga, la, wa = first
    gb, lb, wb = second
    if set(ga.keys()) != set(gb.keys()):
        raise AssertionError('Graph keys differ')
    for key in ga.keys():
        a, b = ga[key], gb[key]
        if isinstance(a, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape:
                raise AssertionError(f'Tensor schema differs: {key}')
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        elif type(a) != type(b) or a != b:
            raise AssertionError(f'Graph attribute differs: {key}')
    torch.testing.assert_close(la, lb, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(wa, wb, rtol=0, atol=0)


def equivalence(options, config):
    import torch
    from torch_geometric.loader import DataLoader
    checked = 0
    for match_id in config['training_ids'] + config['validation_ids']:
        baseline = torch.load(cache_path(options.output_dir, 'baseline', match_id), weights_only=False)
        for mode in ('packed',):
            from packed_graphs import decode_payload
            packed = decode_payload(torch.load(cache_path(options.output_dir, mode, match_id), weights_only=False))
            assert len(baseline['graphs']) == len(packed['graphs'])
            for row in range(len(packed['graphs'])):
                equal_samples((baseline['graphs'][row], baseline['labels'][row], torch.tensor(1.)),
                              (packed['graphs'][row], packed['labels'][row], torch.tensor(1.)))
                checked += 1
            del packed
        del baseline
    batches = 0
    for epoch in (1, 2):
        datasets = [make_dataset(SimpleNamespace(output_dir=options.output_dir, mode=mode), config, 'training')
                    for mode in ('baseline', 'packed')]
        for dataset in datasets:
            dataset.set_epoch(epoch)
        iters = [iter(DataLoader(d, batch_size=config['args']['batch_size'], num_workers=0)) for d in datasets]
        for samples in zip(*iters, strict=True):
            equal_samples(samples[0], samples[1])
            batches += 1
    return dict(equivalent=True, sample_comparisons=checked, shuffled_batches=batches)


def train_cache(options, config):
    import numpy as np
    import torch
    from models.gnn import GNN
    from models.utils import run_epoch
    args = SimpleNamespace(**config['args'])
    args.monitoring = 'off'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    started = time.perf_counter()
    model = GNN(dict(vars(args))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    (train, train_loader), (valid, valid_loader) = loaders(options, config)
    dataset_setup_seconds = time.perf_counter() - started
    epochs = []
    for epoch in range(1, options.epochs + 1):
        args._epoch = epoch
        train.set_epoch(epoch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = run_epoch(args, model, train_loader, optimizer, device, 1., train=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        train_stats = loader_statistics(train, train_loader)
        started = time.perf_counter()
        validation = run_epoch(args, model, valid_loader, optimizer, device, 1., train=False)
        torch.cuda.synchronize()
        valid_seconds = time.perf_counter() - started
        epochs.append(dict(epoch=epoch, training_seconds=train_seconds, validation_seconds=valid_seconds,
                           training_loader=train_stats, validation_loader=loader_statistics(valid, valid_loader),
                           train_metrics=metrics, validation_metrics=validation))
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(state, Path(options.output_dir) / (options.result_name + '_weights.pt'))
    return dict(setup_seconds=setup_seconds, dataset_setup_seconds=dataset_setup_seconds, epochs=epochs,
                training_and_validation_seconds=sum(e['training_seconds'] + e['validation_seconds'] for e in epochs),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu=torch.cuda.get_device_name())


def worker(options):
    import torch
    # Both formats pay the same production module imports before phase timing.
    import prepared_dataset
    torch.set_num_threads(options.threads)
    config = configuration(options.output_dir)
    functions = dict(build=build_cache, read=read_cache, equivalence=equivalence, train=train_cache)
    result = functions[options.phase](options, config)
    write_json(Path(options.output_dir) / (options.result_name + '.json'), result)


def supervise_phase(output, phase, mode, name, *, threads, epochs):
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--phase', phase, '--mode', mode,
               '--output-dir', str(output), '--result-name', name, '--threads', str(threads), '--epochs', str(epochs)]
    peak = dict(rss_bytes=0, private_bytes=0)
    started = time.perf_counter()
    with (output / (name + '.log')).open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        watched = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    memories = [p.memory_info() for p in [watched, *watched.children(recursive=True)] if p.is_running()]
                except psutil.Error:
                    memories = []
                peak['rss_bytes'] = max(peak['rss_bytes'], sum(m.rss for m in memories))
                peak['private_bytes'] = max(peak['private_bytes'], sum(getattr(m, 'private', m.vms) for m in memories))
                if max(peak.values()) > 24 * 1024**3:
                    child.terminate()
                    raise RuntimeError('Comparison exceeded the 24 GiB memory guard')
                time.sleep(.05)
            if child.returncode:
                raise RuntimeError(f'{name} failed with exit {child.returncode}; see {output / (name + ".log")}')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
    result = json.loads((output / (name + '.json')).read_text())
    result.update(phase=phase, mode=mode, peak_process_tree=peak, process_elapsed_seconds=time.perf_counter() - started)
    write_json(output / (name + '.json'), result)
    print(json.dumps(dict(trial=name, mode=mode, phase=phase, peak_private_GiB=peak['private_bytes']/1024**3)), flush=True)
    return result


def orchestrate(options):
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(Path(options.source_metadata).read_text())
    ids = sorted(metadata['train_match_ids'], key=lambda m: (Path(metadata['resolved_dirs']['train_feature_dir']) / (m + '.pt')).stat().st_size)
    train = list(dict.fromkeys(ids[int((len(ids)-1)*fraction)] for fraction in (.2,.5,.8)))
    valid = metadata['validation_match_ids']
    valid = [sorted(valid, key=lambda m: (Path(metadata['resolved_dirs']['valid_feature_dir']) / (m + '.pt')).stat().st_size)[len(valid)//2]]
    config = dict(source_metadata=str(Path(options.source_metadata).resolve()), args=metadata['training_args'],
                  dirs=metadata['resolved_dirs'], training_ids=train, validation_ids=valid,
                  threads=options.threads, epochs_per_trial=options.epochs,
                  note='Fresh processes; OS file cache is not flushed. Read/training trials follow preparation and equality checks.')
    write_json(output/'config.json', config)
    report = dict(status='running', config=config, builds=[], reads=[], training=[])
    write_json(output/'results.json', report)
    try:
        for repetition in range(2):
            for mode in (('baseline','packed') if repetition==0 else ('packed','baseline')):
                result = supervise_phase(output,'build',mode,f'build_{repetition}_{mode}',threads=options.threads,epochs=options.epochs)
                report['builds'].append(result)
                config = configuration(output)
                report['config'] = config
                if mode=='baseline':
                    config['sample_counts'] = {r['match_id']:r['samples'] for r in result['matches']}
                    write_json(output/'config.json',config)
                write_json(output/'results.json',report)
        report['equivalence'] = supervise_phase(output,'equivalence','baseline','equivalence',threads=options.threads,epochs=options.epochs)
        for repetition in range(4):
            modes = ['baseline','packed']
            rotation = repetition % len(modes)
            modes = modes[rotation:] + modes[:rotation]
            for mode in modes:
                report['reads'].append(supervise_phase(output,'read',mode,f'read_{repetition}_{mode}',threads=options.threads,epochs=options.epochs))
                write_json(output/'results.json',report)
        for repetition in range(2):
            modes = ['baseline','packed'] if repetition==0 else ['packed','baseline']
            for mode in modes:
                report['training'].append(supervise_phase(output,'train',mode,f'train_{repetition}_{mode}',threads=options.threads,epochs=options.epochs))
                write_json(output/'results.json',report)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        write_json(output/'results.json',report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--source-metadata')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--phase',choices=('build','read','equivalence','train'))
    parser.add_argument('--mode',choices=('baseline','packed'))
    parser.add_argument('--result-name')
    options=parser.parse_args()
    worker(options) if options.worker else orchestrate(options)


if __name__=='__main__':
    main()
