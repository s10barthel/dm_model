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
import pc_xpass_distance_control as distance_control
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
    p.add_argument('--distance-lane-power',type=float,default=2.,help='Lane power for the unequal-curves variant (default 2)')
    p.add_argument('--distance-control-power',type=float,default=4.,help='Endpoint power for the unequal-curves variant (default 4)')
    p.add_argument('--distance-lane-allowance',type=float,default=.5,help='Lane allowance for the unequal-curves variant (default .5 m)')
    p.add_argument('--distance-control-allowance',type=float,default=1.,help='Endpoint allowance for the unequal-curves variant (default 1 m)')
    p.add_argument('--include-numpy',action='store_true',help='Also time the original NumPy distance lookup and check prediction equivalence')
    p.add_argument('--hawkeye-graphs',type=Path,help='Optional previously reconstructed graph dictionary, keyed by situation ID')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out/pc_xpass_distance')
    args=p.parse_args()
    if min(args.states,args.repeats,args.cache_states,args.cache_repeats,args.validation_points)<1:p.error('Counts must be positive')
    split_options={key:getattr(args,key) for key in distance_control.OVERRIDE_KEYS}
    try:
        split_settings=distance_control.configuration(**split_options)
    except ValueError as exc:
        p.error(str(exc))
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
    report=dict(states=len(graphs),settings=settings,distance_control=distance_control.configuration(),
                distance_split_control=split_settings,distance_table=spatial_config,
                table_preparation_seconds=meta['cumulative_preparation_seconds'],
                table_bytes=(movement.table_path(args.table_id)/'table.npz').stat().st_size,
                table_independent_holdouts=meta['independent_holdout_points'],variants={})
    scores={};comparisons=[]
    compiled_lookup=spatial._lookup_distances
    variants=[('tta',tta_config),('distance',spatial_config),('distance_split',spatial_config)]
    if args.include_numpy:variants.append(('distance_numpy',spatial_config))
    report['timing_order']='rotating variant order per repetition after warmup'
    for name,config in variants:
        spatial._lookup_distances=spatial._lookup_distances_numpy if name=='distance_numpy' else compiled_lookup
        options={**settings,'acceleration_config':config,'margin':'tta' if name=='tta' else 'distance'}
        if name=='distance_split':options.update(split_options)
        started=time.perf_counter()
        physics.compute_graph_pc_xpass_metrics(graphs[0][1],**options)
        report['variants'][name]=dict(seconds=[],warmup_including_one_full_state_seconds=time.perf_counter()-started,
            counters={key:0 for key in movement.COUNTERS},
            measurement='numerical computation; graph loading/cache writes/table preparation excluded')
    for repeat in range(args.repeats):
        offset=repeat%len(variants)
        for name,config in variants[offset:]+variants[:offset]:
            spatial._lookup_distances=spatial._lookup_distances_numpy if name=='distance_numpy' else compiled_lookup
            options={**settings,'acceleration_config':config,'margin':'tta' if name=='tta' else 'distance'}
            if name=='distance_split':options.update(split_options)
            before=movement.COUNTERS.copy();started=time.perf_counter();rows=[]
            for label,graph in graphs:
                stamp=time.perf_counter();rows.append(physics.compute_graph_pc_xpass_metrics(graph,**options))
                print(name,repeat,label,round(time.perf_counter()-stamp,3),flush=True)
            elapsed=time.perf_counter()-started
            scores[name]=rows
            record=report['variants'][name]
            record['seconds'].append(elapsed)
            record['median_seconds']=float(np.median(record['seconds']))
            for key in before:record['counters'][key]+=movement.COUNTERS[key]-before[key]
            atomic_json(args.output_dir/'benchmark.json',report)
    spatial._lookup_distances=compiled_lookup
    if args.include_numpy:
        for compiled,original in zip(scores['distance'],scores['distance_numpy']):
            pd.testing.assert_series_equal(compiled,original,check_exact=False,rtol=1e-12,atol=1e-12)
        report['compiled_numpy_predictions_match']=True
        report['distance_speedup_over_numpy']=report['variants']['distance_numpy']['median_seconds']/report['variants']['distance']['median_seconds']
    for (label,_),a,b in zip(graphs,scores['tta'],scores['distance']):
        for key in a.index:
            if key.endswith('__top_pass10_xpass') and np.isfinite(a[key]):comparisons.append(dict(state=label,receiver=key.split('__')[0],tta=float(a[key]),distance=float(b[key])))
    pd.DataFrame(comparisons).to_csv(args.output_dir/'predictions.csv',index=False)
    report['numerical_runtime_ratio']=report['variants']['distance']['median_seconds']/report['variants']['tta']['median_seconds']
    report['split_control_numerical_runtime_ratio']=report['variants']['distance_split']['median_seconds']/report['variants']['distance']['median_seconds']
    assert report['variants']['distance_split']['counters']['queries']==report['variants']['distance']['counters']['queries']
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
    cache_variants=[('tta',args.tta_table_id),('distance',args.table_id),('distance_split',args.table_id)]
    if args.include_numpy:cache_variants.append(('distance_numpy',args.table_id))
    report['fresh_caches']['variants']={name:dict(seconds=[],warm_cache_seconds=[]) for name,_ in cache_variants}
    for repeat in range(args.cache_repeats):
        offset=repeat%len(cache_variants)
        for name,table in cache_variants[offset:]+cache_variants[:offset]:
            spatial._lookup_distances=spatial._lookup_distances_numpy if name=='distance_numpy' else compiled_lookup
            extra=[]
            if name=='distance_split':
                for key,value in split_options.items():extra.extend(['--'+key.replace('_','-'),str(value)])
            flags=generate.parse_args(common+['--margin','tta' if name=='tta' else 'distance','--accel-table-id',table]+extra)
            with tempfile.TemporaryDirectory(prefix='pc_distance_cache_') as temporary:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    started=time.perf_counter();stats=generate.prewarm_runtime_items(items[:args.cache_states],cache_dir=Path(temporary),args=flags);elapsed=time.perf_counter()-started
                    started=time.perf_counter();hits=generate.prewarm_runtime_items(items[:args.cache_states],cache_dir=Path(temporary),args=flags);warm=time.perf_counter()-started
                assert stats['cache_misses']==min(args.cache_states,len(items)) and hits['cache_misses']==0
            record=report['fresh_caches']['variants'][name]
            record['seconds'].append(elapsed)
            record['warm_cache_seconds'].append(warm)
            record['median_seconds']=float(np.median(record['seconds']))
            print('cache',name,repeat,elapsed,flush=True)
            atomic_json(args.output_dir/'benchmark.json',report)
    spatial._lookup_distances=compiled_lookup
    report['fresh_cache_runtime_ratio']=report['fresh_caches']['variants']['distance']['median_seconds']/report['fresh_caches']['variants']['tta']['median_seconds']
    report['split_control_fresh_cache_runtime_ratio']=report['fresh_caches']['variants']['distance_split']['median_seconds']/report['fresh_caches']['variants']['distance']['median_seconds']
    if args.include_numpy:
        report['fresh_cache_speedup_over_numpy']=report['fresh_caches']['variants']['distance_numpy']['median_seconds']/report['fresh_caches']['variants']['distance']['median_seconds']
    memory=psutil.Process().memory_info()
    report['process_peak_memory_bytes_including_loaded_graphs']=getattr(memory,'peak_wset',memory.rss)
    atomic_json(args.output_dir/'benchmark.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings','variants')},indent=2),flush=True)


if __name__=='__main__':main()
