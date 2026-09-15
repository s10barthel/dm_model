from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import reachability as reach
from scripts import fit_reachability as fitting
from pc_xpass_versions import atomic_json


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(reach, 'ROOT', tmp_path)
    root = tmp_path / 'test'
    root.mkdir()
    times = np.array([0., .2, 1., 5., 6.])
    arrays = dict(speeds=np.array([0., 5.]), times=times)
    for e in fitting.ENVELOPES:
        arrays['c_' + e] = np.zeros((2, len(times)))
        arrays['r_' + e] = np.tile(times * 5, (2, 1))
    fitting.atomic_npz(root / 'circles.npz', **arrays)
    atomic_json(root / 'metadata.json', dict(status='complete', envelopes=fitting.ENVELOPES,
                fingerprint=reach.digest(root / 'circles.npz')))
    return root


def test_shifted_circle_and_extreme():
    angles = np.linspace(0, 2*np.pi, 7200, endpoint=False)
    points = np.column_stack([2 + 3*np.cos(angles), 3*np.sin(angles)])
    c, r, _ = fitting.frontier_circle(points, 'max')
    np.testing.assert_allclose([c,r], [2,3], atol=1e-5)
    extreme = np.vstack([points, [100,0]])
    assert fitting.frontier_circle(extreme, 'max')[1] > 10
    assert fitting.frontier_circle(extreme, '.999')[1] < 4
    assert fitting.frontier_circle(points, '.999', True)[0] == 0


def test_pool_and_split():
    points = pd.DataFrame({'bin': [0,1,1,2], 'x': [0]*4, 'y':[0]*4})
    selected, width = fitting.pooled(points, 1, 4)
    assert len(selected) == 4 and width == 1
    fit, holdout = fitting.split_pool([str(i) for i in range(30)])
    assert len(holdout) == 3 and not set(fit) & set(holdout)
    assert (fit, holdout) == fitting.split_pool([str(i) for i in range(30)])


def test_roles():
    lineup = pd.DataFrame({'object_id':['home_1','home_2'], 'advanced_position':['goal_keeper','center_back']})
    players, excluded = fitting.eligible_players(lineup, ['home_1_x','home_2_x','away_1_x'])
    assert players == ['home_2']
    assert excluded == {'goalkeeper':1, 'unresolved_role':1}


def tracking():
    n=200
    frame=pd.DataFrame(dict(period_id=np.ones(n), episode_id=np.ones(n), ball_state=['Alive']*n,
        home_2_x=np.arange(n)/25, home_2_y=np.zeros(n), home_2_vx=np.ones(n),
        home_2_vy=np.zeros(n), home_2_accel=np.zeros(n)), index=np.arange(n))
    return frame


@pytest.mark.parametrize('problem', ['jump','missing','episode','period','dead','speed','acceleration'])
def test_quality_and_boundaries(problem):
    frame=tracking(); raw=frame.copy()
    args=Namespace(max_speed=15., max_acceleration=30., jump_tolerance=.5, frame_stride=1)
    if problem == 'jump': raw.loc[80,'home_2_x'] += 20
    if problem == 'missing':
        frame=frame.drop(80);raw=raw.drop(80)
    if problem == 'episode': frame.loc[80:,'episode_id']=2
    if problem == 'period': frame.loc[80:,'period_id']=2
    if problem == 'dead': frame.loc[80,'ball_state']='Dead'
    if problem == 'speed': frame.loc[80,'home_2_vx']=20
    if problem == 'acceleration': frame.loc[80,'home_2_accel']=40
    rows, report=fitting.extract_player(raw,frame,'home_2',args)
    # No uninterrupted 6-second trajectory can survive a break near the middle.
    assert all(not (r.horizon == 150).any() for r in rows)


def test_interpolation_extrapolation_rotation(artifact):
    cfg=reach.configuration('reachability',dict(model_id='test'))
    _, arrays=reach.load_model('test')
    c,r=reach.circle(arrays,100,np.array([0,.5,5,5.5]),'0.999',5,8)
    np.testing.assert_allclose(r,[0,2.5,25,29])
    np.testing.assert_allclose(c,0)
    p=np.array([[1.,2.,3.,4.]])
    x=np.array([[2.,3.]]); y=np.array([[4.,5.]]); t=np.array([[[.5,1.]]])
    first=reach.spatial_margins(p,x,y,t,cfg)
    rotated=p[:,[1,0,3,2]]*np.array([-1,1,-1,1])
    np.testing.assert_allclose(first,reach.spatial_margins(rotated,-y,x,t,cfg))


def test_sigmoid_equivalence(artifact):
    from physical_pass_model import _pc_xpass_arrival_margins, pc_xpass_raw_control_with_params
    cfg=reach.configuration('reachability',dict(model_id='test'))
    players=np.array([[0,0,0,0.]])
    x=np.array([[1.,2.,4.]]); y=np.zeros_like(x); t=np.array([[[.3,.6,1.]]])
    old=_pc_xpass_arrival_margins(players,x,y,t,reaction_time=0,max_player_speed=5)
    new=reach.spatial_margins(players,x,y,t,cfg)
    np.testing.assert_allclose(pc_xpass_raw_control_with_params(old,power=15,inflection_point=.2),
                               pc_xpass_raw_control_with_params(new,power=3,inflection_point=1))


def test_config_fingerprint_and_cli(artifact):
    from scripts.generate_physical_xpass import parse_args
    from physical_pass_model import pc_xpass_metadata, pc_xpass_lane_survival_metadata_fingerprint
    args=parse_args(['--pc-xpass','--margin','reachability','--reachability-model-id','test'])
    assert args.reachability_config['lane_power']==3
    assert parse_args(['--pc-xpass']).margin=='tta'
    old=pc_xpass_metadata('consider_teammates')
    new=pc_xpass_metadata('consider_teammates',margin='reachability',reachability_config=args.reachability_config)
    assert pc_xpass_lane_survival_metadata_fingerprint(old)!=pc_xpass_lane_survival_metadata_fingerprint(new)
    with pytest.raises(ValueError,match='fingerprint'):
        reach.configuration('reachability',dict(model_id='test',fingerprint='wrong'))
    for extra in [['--reachability-lane-power','0'],['--reachability-extrapolation-start','7']]:
        with pytest.raises(SystemExit):
            parse_args(['--pc-xpass','--margin','reachability','--reachability-model-id','test',*extra])


def test_shared_graph_and_worker_configuration(artifact):
    from tests.test_physical_xpass import make_graph
    from physical_pass_model import compute_graph_pc_xpass_metrics, compute_graphs_pc_xpass_metrics
    cfg=reach.configuration('reachability',dict(model_id='test'))
    graph=make_graph()
    options=dict(margin='reachability',reachability_config=cfg,min_speed=5,max_speed=7,angle_step=45)
    one=compute_graph_pc_xpass_metrics(graph,**options)
    many=compute_graphs_pc_xpass_metrics([graph,graph],**options)
    pd.testing.assert_series_equal(one,many[0]);pd.testing.assert_series_equal(one,many[1])
    assert np.isfinite(one.to_numpy()).any()


def test_tta_does_not_load_artifact():
    from tests.test_physical_xpass import make_graph
    from physical_pass_model import compute_graph_pc_xpass_metrics
    with patch.object(reach,'load_model',side_effect=AssertionError('TTA must not load circles')):
        a=compute_graph_pc_xpass_metrics(make_graph(),max_speed=3,angle_step=90)
        b=compute_graph_pc_xpass_metrics(make_graph(),max_speed=3,angle_step=90,margin='tta')
    pd.testing.assert_series_equal(a,b)


def test_version_roundtrip_and_incompatible_cache(artifact, tmp_path, monkeypatch):
    import pc_xpass_versions as versions
    from scripts.generate_physical_xpass import parse_args
    from physical_pass_model import _ensure_runtime_physical_xpass_cache, PC_XPASS_SOURCE
    monkeypatch.setattr(versions.config, 'PC_XPASS_DIR', tmp_path/'versions')
    args=parse_args(['--pc-xpass','--pc-xpass-id','v1','--margin','reachability','--reachability-model-id','test'])
    versions.start_generation(args)
    resumed=parse_args(['--pc-xpass','--pc-xpass-id','v1'])
    assert resumed.margin=='reachability'
    assert resumed.reachability_config==args.reachability_config
    with pytest.raises(ValueError):
        parse_args(['--pc-xpass','--pc-xpass-id','v1','--margin','tta'])
    cache=tmp_path/'cache'
    common=dict(source=PC_XPASS_SOURCE,teammate_policy='consider_teammates',speed_aggregation='exact_separate_speed')
    _ensure_runtime_physical_xpass_cache(cache,**common,margin='reachability',reachability_config=args.reachability_config)
    with pytest.raises(ValueError,match='margin'):
        _ensure_runtime_physical_xpass_cache(cache,**common)
    with pytest.raises(ValueError,match='reachability'):
        _ensure_runtime_physical_xpass_cache(cache,**common,margin='reachability',
            reachability_config={**args.reachability_config,'lane_power':4})


def test_pc_only_and_reserved():
    from scripts.generate_physical_xpass import parse_args
    from pc_xpass_versions import validate_id
    with pytest.raises(SystemExit): parse_args(['--margin','reachability'])
    with pytest.raises(ValueError): validate_id('reachability')


def test_prepare_fit_diagnose_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(fitting, 'ROOT', tmp_path)
    monkeypatch.setattr(reach, 'ROOT', tmp_path/'artifacts')
    monkeypatch.setattr(fitting, 'HORIZONS', np.array([1,25,125,150]))
    monkeypatch.setattr(fitting, 'resolve_split_manifest', lambda *a,**k: dict(train=['m1','m2'],test=['untouched']))
    for folder in ['tracking','tracking_processed','lineup']:
        (tmp_path/'data'/folder).mkdir(parents=True)
    for mid in ['m1','m2']:
        frame=tracking();frame.index.name='frame_id'
        frame.to_parquet(tmp_path/'data/tracking'/f'{mid}.parquet')
        frame.to_parquet(tmp_path/'data/tracking_processed'/f'{mid}.parquet')
    pd.DataFrame(dict(stats_perform_match_id=['m1','m2'],object_id=['home_2']*2,
                      advanced_position=['center_back']*2)).to_parquet(tmp_path/'data/lineup/line_up.parquet')
    args=fitting.parse_args(['prepare','--model-id','tiny','--train-count','2'])
    fitting.prepare(args)
    shard=reach.model_path('tiny')/'shards/m1.parquet'
    stamp=shard.stat().st_mtime_ns
    fitting.prepare(args)
    assert shard.stat().st_mtime_ns==stamp
    fitting.fit(fitting.parse_args(['fit','--model-id','tiny','--min-samples','2']))
    meta,arrays=reach.load_model('tiny')
    assert arrays['times'][-1]==6 and meta['status']=='complete'
    assert set(meta['fit_matches']).isdisjoint(meta['holdout_matches'])
    fitting.diagnose(fitting.parse_args(['diagnose','--model-id','tiny','--bootstrap','1']))
    assert (reach.model_path('tiny')/'diagnostics/overall_coverage.csv').is_file()
    with pytest.raises(ValueError,match='immutable'):
        fitting.fit(fitting.parse_args(['fit','--model-id','tiny']))


def test_worker_dispatch(artifact, tmp_path):
    from tests.test_physical_xpass import make_graph
    import physical_pass_model as physical
    cfg=reach.configuration('reachability',dict(model_id='test'))
    task=dict(misses=[dict(graph=make_graph())],source=physical.PC_XPASS_SOURCE,eps=1e-4,
              teammate_policy='consider_teammates',margin='reachability',reachability_config=cfg)
    # Capture the worker's call before unrelated cache-row assembly.
    with patch.object(physical,'compute_graphs_pc_xpass_metrics',side_effect=RuntimeError('captured')) as compute:
        with pytest.raises(RuntimeError,match='captured'):
            physical._compute_runtime_physical_xpass_chunk(task)
    assert compute.call_args.kwargs['margin']=='reachability'
    assert compute.call_args.kwargs['reachability_config']==cfg


@pytest.mark.parametrize('dataset',['sportec','skillcorner','benchmark','hawkeye'])
def test_runtime_dataset_cache_smoke(artifact,tmp_path,dataset):
    import torch
    import physical_pass_model as p
    from tests.test_physical_xpass import make_graph, make_label
    cfg=reach.configuration('reachability',dict(model_id='test'))
    items=[dict(match_id=dataset,graphs=[make_graph()],labels=torch.stack([make_label()]))]
    options=dict(cache_dir=tmp_path/dataset,source=p.PC_XPASS_SOURCE,margin='reachability',
        reachability_config=cfg,num_workers=1,min_speed=5,max_speed=5,angle_step=90,radial_gridsize=10)
    first=p.prewarm_physical_xpass_runtime_cache(items,**options)
    assert first['cache_written']==1
    with patch.object(p,'compute_graphs_pc_xpass_metrics',side_effect=AssertionError('must reuse')):
        second=p.prewarm_physical_xpass_runtime_cache(items,**options)
    assert second['cache_hits']==1
    if dataset=='sportec':
        with pytest.raises(ValueError,match='reuse cache'):
            p.prewarm_physical_xpass_runtime_cache(items,**{**options,'cache_dir':tmp_path/'other',
                'reuse_cache_dir':options['cache_dir'],'reachability_config':{**cfg,'lane_power':4}})


def test_comparison_command(artifact,tmp_path,monkeypatch):
    import json
    import sys
    import torch
    from concurrent.futures import Future
    from scripts.diagnostics import compare_reachability_xpass as compare
    from tests.test_physical_xpass import make_graph
    meta=json.loads((artifact/'metadata.json').read_text())
    meta['holdout_matches']=['match']
    atomic_json(artifact/'metadata.json',meta)
    graphs=tmp_path/'graphs';graphs.mkdir()
    torch.save([make_graph(),make_graph()],graphs/'match.pt')
    monkeypatch.setattr(compare,'resolve_feature_run_id',lambda *a,**k:'features')
    monkeypatch.setattr(compare,'resolve_feature_root',lambda *a:tmp_path)
    monkeypatch.setattr(compare,'get_action_graph_dir',lambda *a:graphs)
    class InlinePool:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def submit(self,fn,*args):
            future=Future();future.set_result(fn(*args));return future
    monkeypatch.setattr(compare,'ProcessPoolExecutor',InlinePool)
    grid=tmp_path/'grid.json'
    atomic_json(grid,dict(min_speed=5,max_speed=5,angle_step=90,radial_gridsize=10))
    out=tmp_path/'comparison'
    monkeypatch.setattr(sys,'argv',['compare','--model-id','test','--feature-run-id','features',
        '--output',str(out),'--count','2','--repeats','1','--grid-json',str(grid)])
    compare.main()
    report=json.loads((out/'metadata.json').read_text())
    assert 'reach_0.999' in report['timings']
    assert (out/'raw_by_player_speed.csv').is_file()
    assert (out/'by_pass_distance.csv').is_file()
    assert set(pd.read_csv(out/'receiver_results.csv').variant)=={'tta','reach_0.99','reach_0.995','reach_0.999','reach_max'}
