"""Compare a fixed holdout sample across TTA, envelopes and optional parameter sweeps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import psutil

import reachability as reach
from pc_xpass_versions import atomic_json
from physical_pass_model import compute_graph_pc_xpass_metrics, _candidate_target_indices
from project_config import get_action_graph_dir, resolve_feature_root, resolve_feature_run_id


def variants(meta, model_id, sweeps):
    result = [('tta', dict(margin='tta'))]
    for e in ['0.999'] + [e for e in meta['envelopes'] if e != '0.999']:
        if e in meta['envelopes']:
            result.append(('reach_' + e, dict(margin='reachability', reachability_config=reach.configuration(
                'reachability', dict(model_id=model_id, envelope=e)))))
    for i, sweep in enumerate(sweeps):
        if set(sweep) - set(reach.DEFAULTS):
            raise ValueError('Unknown sweep settings: ' + str(set(sweep) - set(reach.DEFAULTS)))
        result.append((f'sweep_{i}', dict(margin='reachability', reachability_config=reach.configuration(
            'reachability', dict(model_id=model_id, **sweep)))))
    return result


def select_states(graph_dir, matches, count, seed):
    rng = np.random.default_rng(seed)
    chosen, seen, fingerprints = [], 0, {}
    missing = [m for m in matches if not (graph_dir / (m + '.pt')).is_file()]
    if missing:
        raise FileNotFoundError('Missing holdout action graphs: ' + ', '.join(missing))
    for mid in sorted(matches):
        path = graph_dir / (mid + '.pt')
        fingerprints[mid] = reach.digest(path)
        for index, graph in enumerate(torch.load(path, weights_only=False, map_location='cpu')):
            if graph is None or not _candidate_target_indices(graph):
                continue
            seen += 1
            item = dict(match=mid, index=index)
            if len(chosen) < count:
                chosen.append(item)
            else:
                j = int(rng.integers(seen))
                if j < count:
                    chosen[j] = item
    if len(chosen) < count:
        raise ValueError(f'Only {len(chosen)} eligible states; explicitly lower --count')
    return dict(states=sorted(chosen, key=lambda x:(x['match'], x['index'])), sources=fingerprints,
                count=count, seed=seed, eligible=seen, graph_dir=str(graph_dir.resolve()))


def load_states(selection):
    graphs = []
    by_match = {}
    for item in selection['states']:
        mid = item['match']
        if mid not in by_match:
            path = Path(selection['graph_dir']) / (mid + '.pt')
            if reach.digest(path) != selection['sources'][mid]:
                raise ValueError('Selected graph source changed: ' + mid)
            by_match[mid] = torch.load(path, weights_only=False, map_location='cpu')
        graphs.append(by_match[mid][item['index']])
    return graphs


def measure(graphs, settings, repeats):
    torch.set_num_threads(1)
    compute_graph_pc_xpass_metrics(graphs[0], **settings)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.002):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    elapsed = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            results = [compute_graph_pc_xpass_metrics(g, **settings) for g in graphs]
            elapsed.append(time.perf_counter() - started)
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set(); thread.join()
    return results, dict(median_seconds=float(np.median(elapsed)), repeats=elapsed,
                         baseline_rss=baseline, peak_rss=peak[0], incremental_peak_rss=peak[0]-baseline)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--feature-run-id', required=True, help='Read-only graph input selector for this diagnostic.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100);parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sweeps-json', type=Path, help='JSON list of reachability configuration overrides.')
    parser.add_argument('--grid-json', type=Path, help='JSON compute settings, shared across variants; excludes movement settings.')
    args=parser.parse_args()
    if args.count <= 0 or args.repeats <= 0:
        parser.error('count and repeats must be positive')
    meta,_=reach.load_model(args.model_id)
    feature_id=resolve_feature_run_id(args.feature_run_id,required=True,allow_latest=False)
    graph_dir=get_action_graph_dir(resolve_feature_root(feature_id))
    args.output.mkdir(parents=True,exist_ok=True)
    selection_path=args.output/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        if selection['count']!=args.count or selection['seed']!=args.seed or selection['graph_dir']!=str(graph_dir.resolve()):
            raise ValueError('Subset settings differ; use a new output directory')
        if not {s['match'] for s in selection['states']} <= set(meta['holdout_matches']):
            raise ValueError('Saved subset is outside movement holdout')
    else:
        selection=select_states(graph_dir,meta['holdout_matches'],args.count,args.seed)
        atomic_json(selection_path,selection)
    graphs=load_states(selection)
    shared=dict(min_speed=5.,max_speed=25.,speed_step=2.,angle_step=2.5,radial_gridsize=3.,
        lane_power=15.,lane_inflection_point=.2,control_power=15.,control_inflection_point=.2,
        reaction_time_mode='dist_pass',dist_pass_div=50.,dist_pass_min=.2,dist_pass_max=.7,
        top_n=10,top_n_values=[10,25],use_position_discount=False)
    if args.grid_json:
        overrides=json.loads(args.grid_json.read_text())
        if set(overrides)&{'margin','reachability_config','diagnostic_details'}:
            raise ValueError('grid-json cannot replace variant configuration')
        shared.update(overrides)
    sweeps=json.loads(args.sweeps_json.read_text()) if args.sweeps_json else []
    timings, rows, raw_rows, comparisons = {}, [], [], []
    baseline = None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, options in variants(meta,args.model_id,sweeps):
        settings={**shared,**options}
        # Separate worker per variant gives comparable warmed RSS baselines.
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            result,timing=pool.submit(measure,graphs,settings,args.repeats).result()
        timings[name]=timing
        variant_rows=[]
        for i,(graph,series) in enumerate(zip(graphs,result)):
            details={}
            compute_graph_pc_xpass_metrics(graph,diagnostic_details=details,**settings)
            if not details:
                continue
            for pi, player in enumerate(details['player_ids']):
                raw_rows.append(dict(variant=name,state=i,player=player,
                    initial_speed=float(np.linalg.norm(details['player_positions'][pi,2:4])),
                    raw_lane_mean=float(np.nanmean(details['raw_lane'][pi])),
                    raw_control_mean=float(np.nanmean(details['raw_control'][pi])),
                    lane_survival_mean=float(np.nanmean(details['lane_survival']))))
            for node in graph.node_ids:
                if node not in details['receiver_control'] or not np.isfinite(series.get(node,np.nan)):
                    continue
                ix=list(graph.node_ids).index(node)
                row=dict(variant=name,state=i,match=selection['states'][i]['match'],receiver=node,
                    xpass=float(series[node]), initial_speed=float(torch.linalg.vector_norm(graph.x[ix,5:7])),
                    pass_distance=float(series.get(node+'__distance',np.nan)),ball_speed=float(series.get(node+'__speed',np.nan)),
                    lane_survival=float(series.get(node+'__top10_lane_survival',series.get(node+'__lane_survival',np.nan))),
                    endpoint_control=float(series.get(node+'__top10_control_prob',series.get(node+'__control_prob',np.nan))),
                    max_xpass=float(series.get(node+'__max_xpass',np.nan)),
                    max_lane_survival=float(series.get(node+'__lane_survival',np.nan)),
                    max_endpoint_control=float(series.get(node+'__control_prob',np.nan)),
                    endpoint_control_grid_mean=float(np.nanmean(details['receiver_control'][node])))
                variant_rows.append(row)
            np.savez_compressed(args.output/f'{name}_state_{i}_components.npz',lane=details['lane_survival'],
                endpoint=np.stack(list(details['receiver_control'].values())),
                receivers=np.array(list(details['receiver_control'])),target_x=details['target_x'],target_y=details['target_y'])
        frame=pd.DataFrame(variant_rows)
        frame['receiver_rank']=frame.groupby('state').xpass.rank(ascending=False,method='min')
        rows.extend(frame.to_dict('records'))
        if baseline is None:
            baseline=frame
        else:
            joined=frame.merge(baseline,on=['state','receiver'],suffixes=('','_tta'))
            for field in ['xpass','lane_survival','endpoint_control','receiver_rank']:
                joined[field+'_delta']=joined[field]-joined[field+'_tta']
            comparisons.append(joined)
            for state in joined.groupby('state').xpass_delta.apply(lambda x:x.abs().max()).nlargest(5).index:
                subset=joined[joined.state==state]
                fig,ax=plt.subplots()
                x=np.arange(len(subset));ax.bar(x-.2,subset.xpass_tta,.4,label='TTA');ax.bar(x+.2,subset.xpass,.4,label=name)
                ax.set_xticks(x,subset.receiver,rotation=45);ax.legend();fig.tight_layout()
                fig.savefig(args.output/f'{name}_difference_{state}.png');plt.close(fig)
        print(name,timing,flush=True)
    pd.DataFrame(rows).to_csv(args.output/'receiver_results.csv',index=False)
    pd.DataFrame(raw_rows).to_csv(args.output/'raw_components.csv',index=False)
    raw_frame=pd.DataFrame(raw_rows)
    raw_base=raw_frame[raw_frame.variant=='tta']
    raw_delta=raw_frame[raw_frame.variant!='tta'].merge(raw_base,on=['state','player'],suffixes=('','_tta'))
    for field in ['raw_lane_mean','raw_control_mean']:
        raw_delta[field+'_delta']=raw_delta[field]-raw_delta[field+'_tta']
    raw_delta['speed_group']=pd.cut(raw_delta.initial_speed,[0,1,3,5,8,np.inf],include_lowest=True).astype(str)
    raw_delta.groupby(['variant','speed_group'])[['raw_lane_mean_delta','raw_control_mean_delta']].agg(['count','mean','std']).to_csv(args.output/'raw_by_player_speed.csv')
    if comparisons:
        delta=pd.concat(comparisons,ignore_index=True)
        delta.to_csv(args.output/'differences.csv',index=False)
        for field,bins in [('initial_speed',[0,1,3,5,8,np.inf]),('pass_distance',[0,10,20,40,80,np.inf]),('ball_speed',[0,5,10,15,20,np.inf])]:
            delta['group']=pd.cut(delta[field+'_tta'],bins,include_lowest=True).astype(str)
            delta.groupby(['variant','group'])[['xpass_delta','lane_survival_delta','endpoint_control_delta','receiver_rank_delta']].agg(['count','mean','std']).to_csv(args.output/f'by_{field}.csv')
    for name,timing in timings.items():
        timing['runtime_ratio']=timing['median_seconds']/timings['tta']['median_seconds']
        timing['peak_rss_ratio']=timing['peak_rss']/timings['tta']['peak_rss']
    atomic_json(args.output/'metadata.json',dict(fingerprint=meta['fingerprint'],feature_run_id=feature_id,
        shared_settings=shared,variants=variants(meta,args.model_id,sweeps),timings=timings,
        memory_measurement='2ms sampled worker RSS; warmed baseline and incremental peak also reported'))


if __name__=='__main__':
    main()
