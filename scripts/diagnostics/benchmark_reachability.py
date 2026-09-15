"""Measure inference overhead on real benchmark graphs using synthetic circles."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import reachability as reach
from pc_xpass_versions import atomic_json
from scripts.fit_reachability import atomic_npz
from scripts.diagnostics.benchmark_pc_xpass_optimization import load_graphs
from scripts.diagnostics.compare_reachability_xpass import measure


def worker(graphs, options, repeats, artifact_root):
    reach.ROOT = Path(artifact_root)
    return measure(graphs, options, repeats)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'out/reachability_benchmark.json')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    graphs=[g for _,_,g in load_graphs()]
    options=dict(min_speed=5.,max_speed=25.,speed_step=2.,angle_step=2.5,radial_gridsize=3.,
        top_n=10,top_n_values=[10,25],lane_power=15.,lane_inflection_point=.2,
        control_power=15.,control_inflection_point=.2,reaction_time_mode='dist_pass',
        dist_pass_div=50.,dist_pass_min=.2,dist_pass_max=.7,use_position_discount=False)
    results={}
    with tempfile.TemporaryDirectory(prefix='reachability_benchmark_') as temporary:
        path=Path(temporary)/'synthetic';path.mkdir()
        times=np.r_[0.,np.arange(1,26)/25.,np.arange(30,151,5)/25.]
        speeds=np.arange(0,12,.5/3.6)
        c=speeds[:,None]*(1-np.exp(-times))[None,:]
        c[0]=0
        r=np.broadcast_to(8*(times-1+np.exp(-times)),c.shape).copy()
        atomic_npz(path/'circles.npz',speeds=speeds,times=times,**{'c_0.999':c,'r_0.999':r})
        atomic_json(path/'metadata.json',dict(status='complete',envelopes=['0.999'],fingerprint=reach.digest(path/'circles.npz')))
        for mode in ['tta','reachability']:
            setting={**options,'margin':mode}
            if mode=='reachability':
                setting['reachability_config']={'model_id':'synthetic'}
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                rows,stats=pool.submit(worker,graphs,setting,args.repeats,temporary).result()
            results[mode]=stats
            print(mode,stats,flush=True)
        # Exercise identical computations in two independent processes.
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(worker,graphs[:2],setting,1,temporary) for _ in range(2)]
            left,right=[f.result()[0] for f in futures]
            for a,b in zip(left,right):
                pd.testing.assert_series_equal(a,b)
    results['runtime_ratio']=results['reachability']['median_seconds']/results['tta']['median_seconds']
    results['peak_rss_ratio']=results['reachability']['peak_rss']/results['tta']['peak_rss']
    results.update(graphs=len(graphs),circles='synthetic; not fitted Bundesliga circles',settings=options,
                   two_worker_equal=True,memory_measurement='2 ms sampled RSS after warmup')
    atomic_json(args.output,results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
