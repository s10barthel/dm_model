"""Benchmark acceleration separately from movement-table preparation and cache I/O."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import pc_xpass_acceleration as motion
from pc_xpass_versions import atomic_json
from scripts.diagnostics.benchmark_pc_xpass_optimization import load_graphs
from scripts.diagnostics.compare_reachability_xpass import measure

SETTINGS=dict(eps=.0001,top_n=None,top_pass_values=[10],top_xt=True,
    ignore_teammates_lane_survival=True,ignore_teammates_control=False,
    reaction_time_mode="dist_pass",dist_pass_div=50.,dist_pass_min=0.,dist_pass_max=.7,
    max_player_speed=5.,max_player_speed_off=5.,max_player_speed_def=5.,
    lane_power=15.,lane_inflection_point=.2,control_power=15.,control_inflection_point=.2,
    endpoint_normalization="share",boost_def_endpoint_control=1.,use_position_discount=False,
    max_speed=25.,min_speed=3.,speed_step=2.,angle_step=3.,radial_gridsize=1.)

def worker(graphs,settings,repeats):
    before=dict(motion.COUNTERS)
    rows,stats=measure(graphs,settings,repeats)
    stats["acceleration_stats_including_warmup"]={k:motion.COUNTERS[k]-before[k] for k in before}
    return rows,stats

def load_hawkeye(situation_id):
    from datatools.hawkeye import build_hawkeye_situation,clean_hawkeye_tracking,clean_hawkeye_ball,load_hawkeye_ball
    from scripts.visualize_hawkeye import resolve_ballreceipt,resolve_hawkeye_png_frames
    chunks=pd.read_csv(ROOT/"hawkeye_data/centroid_data_team.csv",chunksize=100000,low_memory=False)
    selected=[c.loc[c.id==situation_id] for c in chunks]
    selected=[c for c in selected if len(c)]
    if not selected:
        raise ValueError("Hawkeye situation not found: "+situation_id)
    tracking=clean_hawkeye_tracking(pd.concat(selected))
    ball=clean_hawkeye_ball(load_hawkeye_ball(ROOT/"hawkeye_data/ball_data_selected.csv"))
    situation,_,_=build_hawkeye_situation(tracking,ball,freeze_ballreceipt=True)
    selection=resolve_hawkeye_png_frames(situation,resolve_ballreceipt(tracking),[0.])[0]
    return situation.graph_features_0[list(situation.actions.index).index(selection["frame_id"])]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--accel-table-id",required=True)
    p.add_argument("--accel",type=float,default=7.)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--states",type=int,default=12)
    p.add_argument("--hawkeye-id")
    p.add_argument("--output",type=Path,default=ROOT/"out/pc_xpass_acceleration_benchmark.json")
    args=p.parse_args()
    if args.repeats<1 or args.states<1:
        p.error("repeats and states must be positive")
    torch.set_num_threads(1)
    print("Loading benchmark states...",flush=True)
    graphs=[g for _,_,g in load_graphs()][:args.states]
    if args.hawkeye_id:
        print("Loading Hawkeye scene "+args.hawkeye_id+"...",flush=True)
        graphs.append(load_hawkeye(args.hawkeye_id))
    config=motion.configuration(args.accel,args.accel_table_id,max_speeds=(5.,))
    meta,_=motion.load_table(args.accel_table_id)
    preparation_seconds=meta.get("cumulative_preparation_seconds",meta.get("preparation_seconds"))
    source_id=meta.get("definition",{}).get("source_table_id")
    if source_id and "cumulative_preparation_seconds" not in meta:
        source_meta,_=motion.load_table(source_id)
        preparation_seconds+=source_meta.get("cumulative_preparation_seconds",source_meta.get("preparation_seconds",0.))
    variants={"existing":SETTINGS,
        "zero_reaction":{**SETTINGS,"reaction_time_mode":"fixed","reaction_time":0.},
        "acceleration":{**SETTINGS,"reaction_time_mode":"fixed","reaction_time":0.,"acceleration_config":config}}
    report=dict(states=len(graphs),settings=SETTINGS,acceleration=config,
        table_preparation_seconds=preparation_seconds,
        table_incremental_preparation_seconds=meta.get("preparation_seconds"),
        table_bytes=(motion.table_path(args.accel_table_id)/"table.npz").stat().st_size,
        measurement="numerical computation only; excludes graph loading and prediction-cache I/O",variants={})
    rows={}
    for name,settings in variants.items():
        started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn")) as pool:
            rows[name],stats=pool.submit(worker,graphs,settings,args.repeats).result()
        stats["process_wall_seconds"]=time.perf_counter()-started
        report["variants"][name]=stats
        atomic_json(args.output,report)
        print(name,json.dumps(stats),flush=True)
    report["runtime_ratio_to_existing"]=report["variants"]["acceleration"]["median_seconds"]/report["variants"]["existing"]["median_seconds"]
    report["runtime_ratio_to_zero_reaction"]=report["variants"]["acceleration"]["median_seconds"]/report["variants"]["zero_reaction"]["median_seconds"]
    comparison=[]
    for index,(before,after) in enumerate(zip(rows["existing"],rows["acceleration"])):
        for key in before.index:
            if key.endswith("_xpass") and np.isfinite(before[key]) and np.isfinite(after[key]):
                comparison.append(dict(state=index,metric=key,existing=before[key],acceleration=after[key],change=after[key]-before[key]))
    pd.DataFrame(comparison).to_csv(args.output.with_suffix(".csv"),index=False)
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()

