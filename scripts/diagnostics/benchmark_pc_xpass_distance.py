"""Validate spatial lookup and compare pc-xPass compute/fresh-cache costs."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
import torch
import physical_pass_model as physics
import pc_xpass_acceleration as movement
import pc_xpass_distance as spatial
from pc_xpass_versions import atomic_json
from scripts.diagnostics.benchmark_pc_xpass_acceleration import SETTINGS
from scripts.diagnostics.benchmark_pc_xpass_optimization import STATE_PAIRS
from datatools.benchmark import build_benchmark_state,load_benchmark_modification_data
from scripts import generate_physical_xpass as generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table-id',required=True)
    p.add_argument('--tta-table-id',default='accel7_v8')
    p.add_argument('--states',type=int,default=12)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--cache-states',type=int,default=2)
    p.add_argument('--cache-repeats',type=int,default=3)
    p.add_argument('--validation-points',type=int,default=96)
    p.add_argument('--hawkeye-graphs',type=Path,help='Optional previously reconstructed graph dictionary, keyed by situation ID')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out/pc_xpass_distance')
    args=p.parse_args()
    if min(args.states,args.repeats,args.cache_states,args.cache_repeats,args.validation_points)<1:p.error('Counts must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    items=[];graphs=[]
    for modification,state_id in STATE_PAIRS[:args.states]:
        data=load_benchmark_modification_data(modification,ROOT/'benchmark')
        state,_,_=build_benchmark_state(data[f'game_state_{state_id}'],modification,state_id,int(data['higher_state_id']))
        items.append(dict(match_id=state.match_id,graphs=state.graph_features_0[:1],labels=state.labels[:1]))
        graphs.append((f'benchmark_{modification}_{state_id}',state.graph_features_0[0]))
    if args.hawkeye_graphs:
        for sid,value in torch.load(args.hawkeye_graphs,weights_only=False).items():graphs.append((sid,value['graph']))
    spatial_config=spatial.configuration(7,args.table_id,max_speeds=(8,))
    tta_config=movement.configuration(7,args.tta_table_id,max_speeds=(8,))
    meta,_=spatial.load_table(args.table_id)
    settings={**SETTINGS,'reaction_time_mode':'fixed','reaction_time':0.,'ball_dec':'drag',
              'max_player_speed':8.,'max_player_speed_off':8.,'max_player_speed_def':8.}
    report=dict(states=len(graphs),settings=settings,distance_control=dict(power=3.,allowance=1.),distance_table=spatial_config,
                table_preparation_seconds=meta['cumulative_preparation_seconds'],
                table_bytes=(movement.table_path(args.table_id)/'table.npz').stat().st_size,
                table_independent_holdouts=meta['independent_holdout_points'],variants={})
    scores={};comparisons=[]
    for name,config in [('tta',tta_config),('distance',spatial_config)]:
        options={**settings,'acceleration_config':config,'margin':name}
        physics.compute_graph_pc_xpass_metrics(graphs[0][1],**options)
        elapsed=[];before=movement.COUNTERS.copy()
        for repeat in range(args.repeats):
            started=time.perf_counter();rows=[]
            for label,graph in graphs:
                stamp=time.perf_counter();rows.append(physics.compute_graph_pc_xpass_metrics(graph,**options))
                print(name,repeat,label,round(time.perf_counter()-stamp,3),flush=True)
            elapsed.append(time.perf_counter()-started)
        scores[name]=rows
        report['variants'][name]=dict(seconds=elapsed,median_seconds=float(np.median(elapsed)),
            counters={k:movement.COUNTERS[k]-before[k] for k in before},
            measurement='numerical computation; graph loading/cache writes/table preparation excluded')
        atomic_json(args.output_dir/'benchmark.json',report)
    for (label,_),a,b in zip(graphs,scores['tta'],scores['distance']):
        for key in a.index:
            if key.endswith('__top_pass10_xpass') and np.isfinite(a[key]):comparisons.append(dict(state=label,receiver=key.split('__')[0],tta=float(a[key]),distance=float(b[key])))
    pd.DataFrame(comparisons).to_csv(args.output_dir/'predictions.csv',index=False)
    report['numerical_runtime_ratio']=report['variants']['distance']['median_seconds']/report['variants']['tta']['median_seconds']
    print('Validating signed distances independently...',flush=True)
    rng=np.random.default_rng(20261026);checks=[]
    for index in range(args.validation_points):
        if index<args.validation_points//2:
            beta=float(rng.uniform(.02,1));theta=float(rng.uniform(0,math.pi));duration=float(np.exp(rng.uniform(math.log(.05),math.log(15))))
            support=spatial.solve_support(beta,theta,duration*7/8,.00005)
            target=support.point*64/7+rng.uniform(-3,3)*support.normal
            position=np.zeros(2);velocity=np.array([beta*8,0.])
        else:
            _,graph=graphs[index%len(graphs)]
            pos,_,_,origin,_,_=physics._full_as_default_simulation_inputs(graph,candidate_indices=physics._candidate_target_indices(graph))
            player=pos[int(rng.integers(len(pos)))];position=player[:2];velocity=player[2:]
            target=rng.uniform([-45,-28],[45,28])
            tb,_=physics.pc_xpass_ball_arrival_times(np.array([np.linalg.norm(target-origin)]),np.array([float(rng.choice(np.arange(3,26,2)))]),'drag')
            duration=float(tb.item())
        reference=spatial.solve_distance(position,velocity,target,duration,7,8)
        value=float(spatial.signed_distances(np.array([[*position,*velocity]]),target[:1],target[1:],np.array([[[duration]]]),[8.],spatial_config).item())
        error=max(reference.lower-value,value-reference.upper,0.)
        checks.append(dict(index=index,position=position.tolist(),velocity=velocity.tolist(),target=target.tolist(),
                           duration=duration,lookup=value,lower=reference.lower,upper=reference.upper,error_outside_bracket=error,
                           worst_error_against_bracket=max(abs(value-reference.lower),abs(value-reference.upper))))
        if index%12==0:print('validation',index,'error',error,flush=True)
    validation=dict(table=spatial_config,points=len(checks),max_error_outside_bracket=max(c['error_outside_bracket'] for c in checks),
                    max_reference_width=max(c['upper']-c['lower'] for c in checks),
                    max_worst_error_against_bracket=max(c['worst_error_against_bracket'] for c in checks),checks=checks,
                    interpretation='independent sampled signed-distance validation, not an exhaustive error proof')
    atomic_json(args.output_dir/'validation.json',validation)
    assert validation['max_worst_error_against_bracket']<=.01
    common=['--pc-xpass','--accel','7','--max-player-speed','8','--max-player-speed-off','8','--max-player-speed-def','8',
            '--reaction-time','0','--ball-dec','drag','--top-pass','10','--top-xt','--use-position-discount','false','--num-workers','1',
            '--max-speed','25','--min-speed','3','--speed-step','2','--angle-step','3','--radial-gridsize','1']
    report['fresh_caches']=dict(states=min(args.cache_states,len(items)),repeats=args.cache_repeats,variants={})
    for name,table in [('tta',args.tta_table_id),('distance',args.table_id)]:
        flags=generate.parse_args(common+['--margin',name,'--accel-table-id',table])
        elapsed=[];warm=[]
        for repeat in range(args.cache_repeats):
            with tempfile.TemporaryDirectory(prefix='pc_distance_cache_') as temporary:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    started=time.perf_counter();stats=generate.prewarm_runtime_items(items[:args.cache_states],cache_dir=Path(temporary),args=flags);elapsed.append(time.perf_counter()-started)
                    started=time.perf_counter();hits=generate.prewarm_runtime_items(items[:args.cache_states],cache_dir=Path(temporary),args=flags);warm.append(time.perf_counter()-started)
                assert stats['cache_misses']==min(args.cache_states,len(items)) and hits['cache_misses']==0
            print('cache',name,repeat,elapsed[-1],flush=True)
        report['fresh_caches']['variants'][name]=dict(seconds=elapsed,median_seconds=float(np.median(elapsed)),warm_cache_seconds=warm)
        atomic_json(args.output_dir/'benchmark.json',report)
    report['fresh_cache_runtime_ratio']=report['fresh_caches']['variants']['distance']['median_seconds']/report['fresh_caches']['variants']['tta']['median_seconds']
    memory=psutil.Process().memory_info()
    report['process_peak_memory_bytes_including_loaded_graphs']=getattr(memory,'peak_wset',memory.rss)
    atomic_json(args.output_dir/'benchmark.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings','variants')},indent=2),flush=True)


if __name__=='__main__':main()
