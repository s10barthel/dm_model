"""Short outcome-training comparisons. Production code and run artifacts stay untouched.

All variants use the real model, run_epoch, loss, optimizer, diagnostics, and
metrics. Trial weights, optimizer, RNG, sample order, and batch size are reset.
CUDA synchronizes only at the timing boundaries. Packed batching and deferred
metrics are benchmark-only prototypes, not general production replacements.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from models.gnn import GNN
from models.graph_diagnostics import install_collation_diagnostics
from models import utils
from packed_graphs import PackedGraphs
from training_monitor import TrainingMonitor
from training_state import environment_versions


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


class Samples(IterableDataset):
    def __init__(self, packed, rows):
        self.packed, self.rows = packed, rows
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        for match, row in self.rows:
            payload = self.packed[match]
            yield payload['graphs'][row], payload['labels'][row], torch.tensor(1.)


def direct_batch(packed, rows):
    fields = packed[rows[0][0]]['raw']['fields']
    node_counts = [packed[m]['raw']['fields']['x']['offsets'][r+1]
                   - packed[m]['raw']['fields']['x']['offsets'][r] for m, r in rows]
    ptr = torch.tensor([0, *np.cumsum(node_counts).tolist()], dtype=torch.long)
    batch = Batch()
    slices, increments = {}, {}
    for key, descriptor in fields.items():
        if descriptor.get('rows') is not None:
            raise ValueError('Prototype does not support optional fields')
        values, lengths = [], []
        for match, row in rows:
            field = packed[match]['raw']['fields'][key]
            if field['kind'] == 'tensor':
                start, end = field['offsets'][row:row+2]
                values.append(field['value'].narrow(field['dim'], start, end-start))
                lengths.append(end-start)
            else:
                values.append(field['value'] if field['kind']=='constant' else field['value'][row])
        if descriptor['kind'] == 'tensor':
            batch[key] = torch.cat(values, dim=descriptor['dim'])
            if key == 'edge_index':
                batch[key] += torch.repeat_interleave(ptr[:-1], torch.tensor(lengths)).unsqueeze(0)
            elif 'index' in key or key == 'face':
                raise ValueError('Unsupported index increment rule: '+key)
            slices[key] = torch.tensor([0, *np.cumsum(lengths).tolist()])
            increments[key] = ptr[:-1] if key=='edge_index' else torch.zeros(len(rows), dtype=torch.long)
        elif key == 'num_nodes':
            batch.num_nodes = sum(values)
        else:
            batch[key] = torch.tensor(values) if isinstance(values[0], (int, float)) else copy.deepcopy(values)
            slices[key] = torch.arange(len(rows)+1)
            increments[key] = torch.zeros(len(rows), dtype=torch.long) if isinstance(values[0], (int,float)) else None
    batch.ptr = ptr
    batch.batch = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor(node_counts))
    batch._num_graphs = len(rows)
    batch._slice_dict, batch._inc_dict = slices, increments
    labels = torch.stack([packed[m]['labels'][r] for m,r in rows])
    return batch, labels, torch.ones(len(rows))


class DirectLoader:
    collate_fn = None
    def __init__(self, packed, rows, batch_size):
        self.packed, self.rows, self.batch_size = packed, rows, batch_size
    def __len__(self):
        return (len(self.rows)+self.batch_size-1)//self.batch_size
    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            yield direct_batch(self.packed, self.rows[start:start+self.batch_size])


class DiskLoader:
    collate_fn = None
    def __init__(self, paths, rows, batch_size, direct=False):
        self.paths, self.rows, self.batch_size, self.direct = paths, rows, batch_size, direct
    def __len__(self):
        return (len(self.rows)+self.batch_size-1)//self.batch_size
    def __iter__(self):
        packed=[]
        for path in self.paths:
            raw=torch.load(path,map_location='cpu',weights_only=False)
            packed.append(dict(raw=raw,graphs=PackedGraphs(raw),labels=raw['labels']))
        loader=(DirectLoader(packed,self.rows,self.batch_size) if self.direct else
                DataLoader(Samples(packed,self.rows),batch_size=self.batch_size,num_workers=0))
        yield from loader


class ReadyLoader:
    collate_fn = None
    def __init__(self, batches):
        self.batches = batches
    def __len__(self):
        return len(self.batches)
    def __iter__(self):
        for graphs, labels, weights in self.batches:
            yield graphs.clone(), labels.clone(), weights.clone()


class PrefetchLoader:
    """One prepared CPU batch ahead; preserve order and propagate exceptions."""
    def __init__(self, loader, pin=False):
        self.loader, self.pin = loader, pin
        self.collate_fn = loader.collate_fn
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        iterator, sentinel = iter(self.loader), object()
        def get_next():
            value = next(iterator, sentinel)
            if value is not sentinel and self.pin:
                value = tuple(v.pin_memory() for v in value)
            return value
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(get_next)
            while True:
                value = future.result()
                if value is sentinel:
                    return
                future = pool.submit(get_next)
                yield value


def epoch_prototype(deferred=False, non_blocking=False):
    source = inspect.getsource(utils.run_epoch)
    # Omit interim console reporting equally in baseline and prototypes.
    source = source.replace('if train and batch_index % args.print_freq == 0:', 'if False:')
    if non_blocking:
        source = source.replace('.to(device)', '.to(device, non_blocking=True)')
    if deferred:
        start = source.index('            elif args.task in ["outcome_scoring", "outcome_conceding"]:')
        end = source.index('            elif args.task in ["intent_return", "intent_return_oppo_agn"]:', start)
        block = source[start:end]
        for old,new in [
            ('torch.sigmoid(pred).cpu().detach().numpy()', 'torch.sigmoid(pred).detach()'),
            ('y = y.cpu().detach().numpy()', 'y = y.detach()'),
            ('np.asarray(y_hat, dtype=float)', 'y_hat'),
            ('np.asarray(y, dtype=float)', 'y'),
            ('target.cpu().detach().numpy().astype(float)', 'target.detach()'),
            ('outcome.cpu().detach().numpy().astype(int)', 'outcome.detach()')]:
            block = block.replace(old,new)
        source = source[:start]+block+source[end:]
        source = source.replace('metrics["ce_loss"] += pred_loss.item()', 'metrics["ce_loss"] += pred_loss.detach().double()')
        source = source.replace('metrics["l1_loss"] += l1_loss.item()', 'metrics["l1_loss"] += l1_loss.detach().double()')
        marker = '    if accumulator is not None:\n        accumulator.flush()'
        flush = '''    for values in (binary_predictions, binary_targets, outcome_predictions,
                   outcome_targets, outcome_diagnostics, outcome_execution_branches):
        if values:
            values[:] = [torch.cat(values).cpu().numpy()]
    for key in ("ce_loss", "l1_loss"):
        if isinstance(metrics.get(key), torch.Tensor):
            metrics[key] = metrics[key].item()
'''
        assert source.count(marker)==1
        source = source.replace(marker,flush+marker)
    namespace = dict(utils.__dict__)
    exec(compile(source,'<benchmark_epoch_prototype>','exec'),namespace)
    return namespace['run_epoch']


def assert_value_equal(left,right,path='batch'):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0,equal_nan=True,msg=path)
    elif isinstance(left,dict):
        assert set(left)==set(right),path
        for key in left:
            assert_value_equal(left[key],right[key],path+'.'+key)
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)):
            assert_value_equal(a,b,f'{path}[{i}]')
    else:
        assert left==right,(path,left,right)


def cpu_correctness(packed, rows, args_dict, weights_path):
    """Deterministic CPU check of the actual loss and two Adam updates."""
    torch.set_num_threads(1)
    initial=torch.load(weights_path,map_location='cpu',weights_only=False)
    results=[]
    states=[]
    model=GNN(copy.deepcopy(args_dict)).cpu()
    for optimized in (False,True):
        model.load_state_dict(initial)
        torch.manual_seed(100)
        optimizer=torch.optim.Adam(model.parameters(),lr=args_dict['start_lr'])
        settings=SimpleNamespace(**copy.deepcopy(args_dict))
        settings.monitoring='off'
        settings._monitor=None
        settings._epoch=1
        loader=(DirectLoader(packed,rows[:16],8) if optimized else
                DataLoader(Samples(packed,rows[:16]),batch_size=8,num_workers=0))
        function=epoch_prototype(deferred=optimized,non_blocking=optimized)
        results.append(function(settings,model,loader,optimizer,'cpu',train=True,
                                return_outcome_evaluation=True))
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    for key in states[0]:
        torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0,msg=key)
    for key in ('prediction','target','diagnostic','execution_branch'):
        np.testing.assert_array_equal(results[0][1][key],results[1][1][key])
    assert set(results[0][0])==set(results[1][0])
    for key in results[0][0]:
        np.testing.assert_allclose(results[0][0][key],results[1][0][key],rtol=0,atol=1e-12,equal_nan=True,err_msg=key)
    return dict(samples=16,steps=2,batch_size=8,device='cpu',threads=1,
                parameter_updates_exact=True,prediction_values_exact=True,
                target_values_exact=True,metrics_absolute_tolerance=1e-12)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--correctness-only',action='store_true')
    parser.add_argument('--suite',choices=['full','followup'],default='full')
    parser.add_argument('--contention',choices=['isolated','pc-xpass-active'],required=True)
    options=parser.parse_args()
    options.output.mkdir(parents=True,exist_ok=True)
    args_dict=json.loads((options.run/'args.json').read_text())
    metadata=json.loads((options.run/'metadata.json').read_text())
    cache=Path(metadata['dataset_loading']['train']['cache_root'])
    manifests={json.loads(p.read_text())['match_id']:p for p in cache.glob('*.json')}
    ids=metadata['train_match_ids']
    selected=[ids[i] for i in (0,len(ids)//3,2*len(ids)//3,len(ids)-1)]
    packed,load_times=[],[]
    torch.set_num_threads(1)
    for match in selected:
        path=manifests[match].with_suffix('.pt')
        started=time.perf_counter()
        raw=torch.load(path,map_location='cpu',weights_only=False)
        load_times.append(time.perf_counter()-started)
        packed.append(dict(raw=raw,graphs=PackedGraphs(raw),labels=raw['labels']))
    rows=[(m,r) for m,p in enumerate(packed) for r in range(len(p['graphs']))]
    random.Random('100:2').shuffle(rows)
    required=args_dict['batch_size']*options.steps
    if len(rows)<required:
        raise ValueError(f'Need {required} distinct samples, got {len(rows)}')
    rows=rows[:required]
    batch_size=args_dict['batch_size']
    batches=[]
    collater=Collater(None)
    for start in range(0,len(rows),batch_size):
        group=rows[start:start+batch_size]
        reference=collater([(packed[m]['graphs'][r],packed[m]['labels'][r],torch.tensor(1.)) for m,r in group])
        direct=direct_batch(packed,group)
        assert_value_equal(reference[0].to_dict(),direct[0].to_dict())
        assert_value_equal(reference[0]._slice_dict,direct[0]._slice_dict,'slice_dict')
        assert_value_equal(reference[0]._inc_dict,direct[0]._inc_dict,'inc_dict')
        assert_value_equal(reference[1],direct[1],'labels')
        assert_value_equal(reference[2],direct[2],'weights')
        batches.append(tuple(reference))
    report=dict(environment=environment_versions(),contention=options.contention,
                timing_scope=('Includes warm cache file reads, reconstruction, batching, training or validation, CPU validation, and final metrics; excludes initialization' if options.suite=='followup' else 'RAM-resident packed tensors; includes reconstruction, batching, training, CPU validation, and final metrics; excludes disk reads and initialization'),
                interim_console_metrics=False, suite=options.suite,
                run=str(options.run),
                selected_matches=selected,samples=len(rows),batch_size=batch_size,
                steps=options.steps,repeats=options.repeats,cache_load_seconds=load_times,
                cache_bytes=sum(manifests[m].with_suffix('.pt').stat().st_size for m in selected),
                direct_batch_exact_equivalence=True,trials=[],cpu=[])
    if options.correctness_only:
        report['cpu_correctness']=cpu_correctness(packed,rows,args_dict,options.run/'best_weights.pt')
        write_json(options.output/'results.json',report)
        print('CORRECTNESS',json.dumps(report['cpu_correctness']),flush=True)
        return
    for threads in ((1,2,4,8) if options.suite=='full' else ()):
        torch.set_num_threads(threads)
        for method in ('standard','direct'):
            times=[]
            for repeat in range(3):
                loader=(DataLoader(Samples(packed,rows),batch_size=batch_size,num_workers=0)
                        if method=='standard' else DirectLoader(packed,rows,batch_size))
                started=time.perf_counter()
                for _ in loader:
                    pass
                times.append(time.perf_counter()-started)
            report['cpu'].append(dict(threads=threads,method=method,seconds=times,
                                    median_ms_per_batch=1000*statistics.median(times)/options.steps))
    write_json(options.output/'results.json',report)
    print('PREPARED',json.dumps({k:report[k] for k in ('samples','cache_load_seconds','direct_batch_exact_equivalence','cpu')}),flush=True)
    if options.prepare_only:
        return
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid == psutil.Process().pid or 'python' not in process.info['name'].lower():
            continue
        try:
            command = process.cmdline()
            if any(Path(arg).name == 'train.py' for arg in command):
                raise RuntimeError(f'Training process {process.pid} is active; refusing GPU benchmark')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    model=GNN(copy.deepcopy(args_dict)).cuda()
    initial=torch.load(options.run/'best_weights.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(initial)
    functions={(d,n):epoch_prototype(d,n) for d in (False,True) for n in (False,True)}
    reference_state,reference_eval=None,None

    def trial(case,repeat):
        nonlocal reference_state,reference_eval
        torch.set_num_threads(case['threads'])
        random.seed(100);np.random.seed(100);torch.manual_seed(100);torch.cuda.manual_seed_all(100)
        model.load_state_dict(initial)
        optimizer=torch.optim.Adam(model.parameters(),lr=args_dict['start_lr'])
        settings=SimpleNamespace(**copy.deepcopy(args_dict))
        settings.monitoring='on' if case.get('monitor') else 'off'
        settings._epoch=1
        directory=options.output/f"{case['name']}_{repeat}"
        directory.mkdir(exist_ok=True)
        monitor=TrainingMonitor(directory,enabled=case.get('monitor',False))
        if case.get('sampled_monitor'):
            event=monitor.event
            def sampled_event(stage,device=None,**details):
                if details.get('batch',0)%50==0:
                    event(stage,device,**details)
            monitor.event=sampled_event
        settings._monitor=monitor
        if case.get('disk'):
            loader=DiskLoader([manifests[m].with_suffix('.pt') for m in selected],rows,batch_size,case.get('direct',False))
        elif case.get('ready'):
            loader=ReadyLoader(batches)
        elif case.get('direct'):
            loader=DirectLoader(packed,rows,batch_size)
        else:
            loader=DataLoader(Samples(packed,rows),batch_size=batch_size,num_workers=0,
                              pin_memory=case.get('pin',False))
            install_collation_diagnostics(loader,enabled=case.get('diagnostics',False),directory=directory)
        if case.get('prefetch'):
            loader=PrefetchLoader(loader,pin=case.get('pin',False))
        function=functions[(case.get('deferred',False),case.get('non_blocking',False))]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        try:
            with (directory/'console.txt').open('w') as stream,contextlib.redirect_stdout(stream):
                metrics,evaluation=function(settings,model,loader,optimizer,'cuda',train=not case.get('validation',False),
                                            return_outcome_evaluation=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
        finally:
            monitor.close()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if reference_state is None and not case.get('warmup') and not case.get('validation'):
            reference_state,reference_eval=state,evaluation
        difference,prediction_difference=None,None
        if reference_state is not None and not case.get('warmup') and not case.get('validation'):
            difference=max((state[k]-reference_state[k]).abs().max().item() for k in state)
            prediction_difference=float(np.max(np.abs(evaluation['prediction']-reference_eval['prediction'])))
            for key in ('target','diagnostic','execution_branch'):
                np.testing.assert_array_equal(evaluation[key],reference_eval[key])
        result=dict(case=case['name'],repeat=repeat,seconds=elapsed,
                    ms_per_batch=1000*elapsed/options.steps,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    parameter_max_abs_difference=difference,prediction_max_abs_difference=prediction_difference,
                    ce_loss=metrics['ce_loss'],settings=case,
                    process_rss_mib=psutil.Process().memory_info().rss/1024**2)
        print('TRIAL',json.dumps(result),flush=True)
        if not case.get('warmup'):
            report['trials'].append(result)
            write_json(options.output/'results.json',report)
        del optimizer,state
        return result

    trial(dict(name='warmup',threads=8,warmup=True),0)
    sweep=[dict(name='monitoring_on_t8',threads=8,monitor=True,diagnostics=True)]
    sweep += [dict(name=f'baseline_t{t}',threads=t) for t in (8,4,2,1)]
    for repeat in range(options.repeats if options.suite=='full' else 0):
        order=list(sweep)
        random.Random(200+repeat).shuffle(order)
        for case in order:
            trial(case,repeat)
    baseline_times=({t:statistics.median(r['seconds'] for r in report['trials'] if r['case']==f'baseline_t{t}')
                    for t in (1,2,4,8)} if options.suite=='full' else {8:0})
    best=min(baseline_times,key=baseline_times.get)
    cases=[dict(name='best_baseline',threads=best),
           dict(name='monitoring_on_best',threads=best,monitor=True,diagnostics=True),
           dict(name='diagnostics_only',threads=best,diagnostics=True),
           dict(name='telemetry_only',threads=best,monitor=True),
           dict(name='deferred_metrics',threads=best,deferred=True),
           dict(name='pinned_blocking',threads=best,pin=True),
           dict(name='pinned_nonblocking',threads=best,pin=True,non_blocking=True),
           dict(name='prefetch_standard',threads=best,prefetch=True),
           dict(name='direct_packed_batch',threads=best,direct=True),
           dict(name='combined',threads=best,direct=True,prefetch=True,deferred=True,pin=True,non_blocking=True),
           dict(name='ready_batches',threads=best,ready=True,deferred=True)]
    if options.suite=='followup':
        common=dict(threads=8,disk=True)
        cases=[dict(common,name='stream_baseline'),
               dict(common,name='stream_direct',direct=True),
               dict(common,name='stream_prefetch',prefetch=True),
               dict(common,name='stream_direct_prefetch',direct=True,prefetch=True),
               dict(common,name='stream_direct_prefetch_deferred',direct=True,prefetch=True,deferred=True),
               dict(common,name='stream_direct_prefetch_pinned_nb',direct=True,prefetch=True,pin=True,non_blocking=True),
               dict(common,name='stream_direct_prefetch_nb',direct=True,prefetch=True,non_blocking=True),
               dict(common,name='stream_sampled_monitor',monitor=True,sampled_monitor=True),
               dict(common,name='validation_baseline',validation=True),
               dict(common,name='validation_direct_prefetch',validation=True,direct=True,prefetch=True)]
    for repeat in range(options.repeats):
        order=list(cases)
        random.Random(300+repeat).shuffle(order)
        for case in order:
            trial(case,repeat)
    report['best_threads']=best
    report['summary']={}
    for name in dict.fromkeys(r['case'] for r in report['trials']):
        records=[r for r in report['trials'] if r['case']==name]
        report['summary'][name]=dict(median_seconds=statistics.median(r['seconds'] for r in records),
            min_seconds=min(r['seconds'] for r in records),max_seconds=max(r['seconds'] for r in records),
            max_parameter_difference=max((r['parameter_max_abs_difference'] for r in records if r['parameter_max_abs_difference'] is not None),default=None),
            max_prediction_difference=max((r['prediction_max_abs_difference'] for r in records if r['prediction_max_abs_difference'] is not None),default=None))
    write_json(options.output/'results.json',report)
    print('SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    main()
