"""Signed exact-time distance, support certificates, and pc-xPass integration."""
import copy
import json
import math
import numpy as np
import pandas as pd
import pytest
import torch
import pc_xpass_acceleration as motion
import pc_xpass_distance as spatial
import pc_xpass_distance_control as distance_control
import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate
from scripts import prepare_pc_xpass_acceleration as prepare
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph,make_label


@pytest.fixture
def table(tmp_path,monkeypatch):
    """Synthetic support interpolant for plumbing; physical cases tested below."""
    monkeypatch.setattr(motion,'ROOT',tmp_path/'tables')
    root=motion.table_path('spatial');root.mkdir(parents=True)
    np.savez_compressed(root/'table.npz',low=[[0,0,0]],high=[[1,math.pi,math.log1p(500)]],
                        values=np.zeros((1,8)),children=[[-1,-1]],axis=[-1],fallback=[False])
    meta=dict(status='complete',movement_version=motion.VERSION,spatial_version=spatial.VERSION,
              fingerprint=motion._digest(root/'table.npz'),distance_error_normalized=.0001,max_distance=150.,max_time=500.)
    (root/'metadata.json').write_text(json.dumps(meta))
    return root


@pytest.mark.parametrize('duration',[0.,.1,.5,2.,40.])
def test_standing_disk_signed_inside_outside(duration):
    cap=8.;a=7.;r=float(spatial.standing_radius(duration*a/cap))*cap**2/a
    for target in [[0.,0.],[r,0.],[r+1.,0.],[0.,r+3.]]:
        value=spatial.solve_distance([0,0],[0,0],target,duration,a,cap)
        assert value.lower==pytest.approx(np.linalg.norm(target)-r)
        assert value.upper==value.lower


def test_initial_momentum_exact_time_not_earliest_arrival():
    assert motion.solve_arrival([0,0],[4,0],[1,0],7,8).time<.22
    assert spatial.solve_distance([0,0],[4,0],[1,0],.5,7,8).distance==pytest.approx(.125)
    assert spatial.solve_distance([0,0],[4,0],[1,0],.25,7,8).distance<0


@pytest.mark.parametrize('speed,angle,duration',[(.5,.7,1.2),(1.,1.8,.7),(.7,2.4,4.5),(.5,1.1,150.)])
def test_support_continuous_bounds_and_projection(speed,angle,duration):
    result=spatial.solve_support(speed,angle,duration,.0005)
    assert 0<=result.upper-result.lower<=.0005
    assert result.lower<=result.normal@result.point<=result.upper
    assert np.linalg.norm(result.point)<=duration+1e-6
    # Projection distance lower bound cannot exceed the independent fixed-time primal.
    if duration<5:
        q=result.point+result.normal*.3
        residual,lower,_=motion._fixed_time(np.array([speed,0.]),q,duration,256)
        assert result.normal@q-result.upper<=residual+1e-5
        assert lower<=.3+.001


def test_reference_invariances_and_nontrivial_signed_interior():
    v=np.array([3.,2.]);q=np.array([2.,1.]);off=np.array([10.,20.]);rot=np.array([[0.,-1.],[-1.,0.]])
    r=spatial.solve_distance([0,0],v,q,1.4,7,8)
    assert r.upper-r.lower<=.005 and r.upper<0
    reflected=spatial.solve_distance(off,rot@v,off+rot@q,1.4,7,8)
    assert reflected.distance==pytest.approx(r.distance,abs=.005)
    scaled=spatial.solve_distance([0,0],2*v,2*q,1.4,14,16)
    assert scaled.distance==pytest.approx(2*r.distance,abs=.01)
    clipped=spatial.solve_distance([0,0],v*10,q,1.4,7,8)
    expected=spatial.solve_distance([0,0],v*8/np.linalg.norm(v),q,1.4,7,8)
    assert clipped.distance==pytest.approx(expected.distance,abs=.005)


def test_unresolved_never_accepted(monkeypatch):
    with pytest.raises(motion.UnresolvedArrival):
        spatial.solve_distance([0,0],[3,2],[2,1],1.4,7,8,max_evaluations=8)
    with pytest.raises(motion.UnresolvedArrival):
        spatial.solve_support(.7,1.,2.,1e-12,max_intervals=32)


def test_analytic_angular_global_maximum():
    rng=np.random.default_rng(17)
    for _ in range(40):
        lo,hi=sorted(rng.uniform(0,math.pi,2));c0,c1=rng.normal(size=2)
        q=rng.normal(size=2);q[1]=abs(q[1])
        exact=spatial._interval_max(*q,lo,hi,c0,c1)
        theta=np.linspace(lo,hi,20001)
        sampled=q[0]*np.cos(theta)+q[1]*np.sin(theta)-c0-(c1-c0)*(theta-lo)/(hi-lo)
        assert exact==pytest.approx(sampled.max(),abs=1e-7)


def test_cli_cache_identity_inheritance_and_location(table,tmp_path,monkeypatch):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'caches')
    flags=['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial','--pc-xpass-id','distance_run']
    args=generate.parse_args(flags)
    assert args.distance_power==3 and args.distance_allowance==1 and args.reaction_time==0
    versions.start_generation(args)
    loaded=generate.parse_args(['--pc-xpass','--pc-xpass-id','distance_run'])
    assert loaded.margin=='distance' and loaded.distance_power==3
    for extra in [['--distance-power','4'],['--distance-allowance','2'],['--margin','tta']]:
        with pytest.raises(ValueError):generate.parse_args(['--pc-xpass','--pc-xpass-id','distance_run',*extra])
    loc=location.parse_args(['--margin','distance','--accel','7','--accel-table-id','spatial'])
    assert loc.acceleration_config['reachability_time']=='exact'
    loc.pc_xpass_id='location_distance';versions.start_generation(loc,location=True)
    inherited=location.parse_args(['--pc-xpass-id','location_distance'])
    assert inherited.margin=='distance' and inherited.distance_allowance==1
    with pytest.warns(UserWarning):
        inherited=location.parse_args(['--pc-xpass-id','location_distance','--distance-allowance','2'])
    assert inherited.distance_allowance==1
    with pytest.raises(ValueError,match='Distance acceleration table'):
        motion.configuration(7,'spatial',max_speeds=(8,))
    spatial.configuration(14,'spatial',max_speeds=(8,))
    with pytest.raises(ValueError,match='coarse'):spatial.configuration(1,'spatial',max_speeds=(100,))


@pytest.mark.parametrize('extra',[
    ['--accel','off'],['--distance-power','0'],['--distance-power','nan'],
    ['--distance-allowance','-1'],['--distance-allowance','inf'],
    ['--lane-power','15'],['--control-inflection-point','0'],['--reaction-time','.1']])
def test_distance_invalid_cli(table,extra):
    with pytest.raises(SystemExit):generate.parse_args(['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial',*extra])


def test_distance_flags_rejected_outside_mode():
    with pytest.raises(SystemExit):generate.parse_args(['--pc-xpass','--distance-power','3'])
    with pytest.raises(SystemExit):generate.parse_args(['--distance-allowance','1'])
    args=generate.parse_args(['--pc-xpass'])
    assert 'distance_power' not in versions.settings(args)


@pytest.mark.parametrize('entrypoint', [generate, location])
@pytest.mark.parametrize('overrides', [
    {}, {'distance_lane_power': 2.}, {'distance_control_power': 6.},
    {'distance_lane_allowance': 0.}, {'distance_control_allowance': .2},
    dict(distance_lane_power=2., distance_control_power=6., distance_lane_allowance=0., distance_control_allowance=.2),
])
def test_distance_overrides_cli_inheritance_and_resume(table,tmp_path,monkeypatch,entrypoint,overrides):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'caches')
    flags=['--margin','distance','--accel','7','--accel-table-id','spatial','--distance-power','4','--distance-allowance','.7']
    if entrypoint is generate:
        flags.insert(0,'--pc-xpass')
    for key,value in overrides.items():
        flags.extend(['--'+key.replace('_','-'),str(value)])
    args=entrypoint.parse_args(flags)
    expected=dict(distance_lane_power=4.,distance_control_power=4.,distance_lane_allowance=.7,distance_control_allowance=.7)
    expected.update(overrides)
    assert {key:getattr(args,key) for key in expected}==expected
    args.pc_xpass_id='split'
    versions.start_generation(args,location=entrypoint is location)
    inherited_flags=['--pc-xpass-id','split']+(['--pc-xpass'] if entrypoint is generate else [])
    loaded=entrypoint.parse_args(inherited_flags)
    versions.start_generation(loaded,location=entrypoint is location)
    assert {key:getattr(loaded,key) for key in expected}==expected
    saved=versions.read_metadata(versions.version_root('split',location=entrypoint is location))['generation_settings']
    assert saved['distance_control_version']==distance_control.VERSION
    assert {key:saved[key] for key in expected}==expected
    for key,value in expected.items():
        matching=entrypoint.parse_args(inherited_flags+['--'+key.replace('_','-'),str(value)])
        assert getattr(matching,key)==value
        conflicting=inherited_flags+['--'+key.replace('_','-'),str(value+1)]
        if entrypoint is generate:
            with pytest.raises(ValueError,match='cached value'):
                entrypoint.parse_args(conflicting)
        else:
            with pytest.warns(UserWarning,match='cached value'):
                loaded=entrypoint.parse_args(conflicting)
            assert getattr(loaded,key)==value


@pytest.mark.parametrize('key,bad',[
    (key,bad) for key in distance_control.OVERRIDE_KEYS for bad in ('nan','inf','-1','0')
    if bad!='0' or key.endswith('power')])
def test_distance_override_validation(table,key,bad):
    flags=['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial',
           '--'+key.replace('_','-'),bad]
    with pytest.raises(SystemExit):
        generate.parse_args(flags)
    opts={key:float(bad)}
    config=spatial.configuration(7,'spatial',max_speeds=(5,))
    with pytest.raises(ValueError,match=key.replace('_','-')):
        physics.compute_graph_pc_xpass_metrics(make_graph(),margin='distance',reaction_time=0,acceleration_config=config,**opts)
    with pytest.raises(ValueError,match=key.replace('_','-')):
        physics.pc_xpass_metadata('consider_teammates',margin='distance',acceleration_config=config,**opts)


@pytest.mark.parametrize('key',distance_control.OVERRIDE_KEYS)
def test_distance_overrides_require_distance_mode(key):
    flag='--'+key.replace('_','-')
    for prefix in ([],['--pc-xpass'],['--pc-xpass','--margin','reachability']):
        with pytest.raises(SystemExit):
            generate.parse_args([*prefix,flag,'1'])
    with pytest.raises(ValueError,match='require --margin distance'):
        physics.compute_graph_pc_xpass_metrics(make_graph(),**{key:1.})
    with pytest.raises(ValueError,match='require --margin distance'):
        physics.pc_xpass_metadata('consider_teammates',**{key:1.})
    with pytest.raises(ValueError,match='require pc-xPass'):
        physics.prewarm_physical_xpass_runtime_cache([],cache_dir='unused',source=physics.PHYSICAL_XPASS_SOURCE,
                                                    teammate_policy='consider_teammates',**{key:1.})


@pytest.mark.parametrize('key,value',[(key,2. if key.endswith('power') else 0.) for key in distance_control.OVERRIDE_KEYS])
def test_distance_curve_independence_and_shared_array(table,monkeypatch,key,value):
    calls=[]
    def signed_distance(players,x,y,times,*args,**kwargs):
        calls.append(True)
        shape=(len(players),*np.broadcast_shapes(x.shape,times.shape))
        return np.broadcast_to(np.linspace(-1,2,len(players))[:,None,None,None],shape)
    monkeypatch.setattr(spatial,'signed_distances',signed_distance)
    opts=dict(margin='distance',acceleration_config=spatial.configuration(7,'spatial',max_speeds=(5,)),
              reaction_time=0,angle_step=90,radial_gridsize=40,distance_power=4.,distance_allowance=.7)
    base={};split={};equal={}
    baseline=physics.compute_graph_pc_xpass_metrics(make_graph(),diagnostic_details=base,**opts)
    physics.compute_graph_pc_xpass_metrics(make_graph(),diagnostic_details=split,**opts,**{key:value})
    equal_settings=dict(distance_lane_power=4.,distance_control_power=4.,distance_lane_allowance=.7,distance_control_allowance=.7)
    equal_row=physics.compute_graph_pc_xpass_metrics(make_graph(),diagnostic_details=equal,**opts,**equal_settings)
    assert len(calls)==3  # Exactly one distance calculation per graph, including unequal curves.
    pd.testing.assert_series_equal(baseline,equal_row,check_exact=True)
    assert base['raw_lane'] is base['raw_control']
    assert equal['raw_lane'] is equal['raw_control']
    changed='raw_lane' if '_lane_' in key else 'raw_control'
    unchanged='raw_control' if changed=='raw_lane' else 'raw_lane'
    np.testing.assert_array_equal(base[unchanged],split[unchanged])
    assert not np.allclose(base[changed],split[changed],equal_nan=True)
    if changed=='raw_control':
        np.testing.assert_array_equal(base['lane_survival'],split['lane_survival'])


@pytest.mark.parametrize('missing', ['distance_control_version',*distance_control.PARAMETER_KEYS,'unsupported_version'])
def test_old_distance_contract_rejected_everywhere(table,tmp_path,monkeypatch,missing):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'caches')
    flags=['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial',
           '--pc-xpass-id','old','--angle-step','90','--radial-gridsize','40']
    args=generate.parse_args(flags)
    versions.start_generation(args)
    root=versions.version_root('old')
    root_metadata=versions.read_metadata(root)
    metadata=physics.pc_xpass_metadata('consider_teammates',margin='distance',acceleration_config=args.acceleration_config,reaction_time=0)
    for record in (root_metadata['generation_settings'],metadata):
        if missing=='unsupported_version':
            record['distance_control_version']='unsupported'
        else:
            record.pop(missing)
    versions.atomic_json(root/'metadata.json',root_metadata)
    directory=tmp_path/'old_predictions';directory.mkdir()
    versions.atomic_json(directory/'metadata.json',metadata)
    locroot=versions.version_root('old',location=True);locroot.mkdir(parents=True)
    versions.atomic_json(locroot/'metadata.json',root_metadata)
    item=dict(match_id='fixture',graphs=[make_graph()],labels=torch.stack([make_label()]))
    consumers=[
        lambda:generate.parse_args(['--pc-xpass','--pc-xpass-id','old']),
        lambda:location.parse_args(['--pc-xpass-id','old']),
        lambda:location.parse_args(['--pc-xpass-cache-dir',str(directory)]),
        lambda:versions.read_metadata(directory),
        lambda:physics.validate_physical_xpass_cache_metadata(directory,expected_source='pc_xpass'),
        lambda:physics.validate_runtime_physical_xpass_visualization_cache(directory),
        lambda:physics.load_physical_xpass_match(directory,'fixture'),
        lambda:physics.physical_xpass_inference_lookup_config({'pc_xpass':True},cache_dir=directory),
        lambda:physics.pc_xpass_lane_survival_metadata_fingerprint(metadata),
        lambda:generate.prewarm_runtime_items([item],cache_dir=directory,args=args),
        lambda:generate.prewarm_runtime_items([item],cache_dir=tmp_path/'new',reuse_cache_dir=directory,args=args),
    ]
    before=(directory/'metadata.json').read_bytes()
    for consumer in consumers:
        with pytest.raises(ValueError,match='regenerate predictions under a new'):
            consumer()
    assert (directory/'metadata.json').read_bytes()==before


def test_distance_fingerprints_and_other_modes(table):
    opts=dict(margin='distance',acceleration_config=spatial.configuration(7,'spatial',max_speeds=(5,)))
    metadata=physics.pc_xpass_metadata('consider_teammates',**opts)
    fingerprint=physics.pc_xpass_lane_survival_metadata_fingerprint(metadata)
    for key in distance_control.OVERRIDE_KEYS:
        changed={**metadata,key:metadata[key]+1}
        assert fingerprint!=physics.pc_xpass_lane_survival_metadata_fingerprint(changed)
    # Distance-only fields must never enter the fingerprints of other modes.
    for margin in ('tta','reachability'):
        other=physics.pc_xpass_metadata('consider_teammates')
        if margin=='reachability':
            other.update(margin=margin,reachability_config={'fixture':True})
        assert not set(distance_control.SETTING_KEYS)&other.keys()
        altered={**other,**{key:'unused' for key in distance_control.SETTING_KEYS}}
        assert physics.pc_xpass_lane_survival_metadata_fingerprint(other)==physics.pc_xpass_lane_survival_metadata_fingerprint(altered)


def test_distance_version_settings_checked_before_metadata_augmentation(table,tmp_path,monkeypatch):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'caches')
    args=generate.parse_args(['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial',
                             '--pc-xpass-id','version','--angle-step','90','--radial-gridsize','40'])
    versions.start_generation(args)
    root=versions.version_root('version')
    before=(root/'metadata.json').read_bytes()
    changed=copy.copy(args);changed.distance_lane_power=8.
    item=dict(match_id='fixture',graphs=[make_graph()],labels=torch.stack([make_label()]))
    with pytest.raises(ValueError,match='Incompatible distance control settings'):
        generate.prewarm_runtime_items([item],cache_dir=root,args=changed)
    assert (root/'metadata.json').read_bytes()==before


def test_table_corruption_and_version(table):
    config=spatial.configuration(7,'spatial',max_speeds=(8,))
    with pytest.raises(ValueError,match='fingerprint'):spatial.configuration(7,'spatial',max_speeds=(8,),fingerprint='changed')
    (table/'table.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupted'):spatial.load_table('spatial')


def test_vectorized_signed_lookup_and_far_fallback(table,monkeypatch):
    config=spatial.configuration(7,'spatial',max_speeds=(8,))
    pos=np.array([[0.,0.,0.,0.],[0.,0.,4.,0.]])
    # First case analytic; second exercises the signed support interpolant.
    result=spatial.signed_distances(pos,np.array([[0.,10.]]),np.zeros((1,2)),np.array([[[2.,2.]]]),[8.,8.],config)
    radius=float(spatial.standing_radius(2*7/8))*64/7
    np.testing.assert_allclose(result,np.broadcast_to(np.array([-radius,10-radius]),result.shape))
    called=[]
    def fallback(*args,**kwargs):called.append(args);return spatial.SpatialResult(1,1)
    monkeypatch.setattr(spatial,'solve_distance',fallback)
    x=spatial.signed_distances(pos[1:],np.array([200.]),np.array([0.]),np.array([[[2.]]]),[8.],config)
    assert x.item()==1 and len(called)==1
    meta,arrays=spatial.load_table('spatial');arrays['fallback'][:]=True
    with pytest.raises(AssertionError):
        monkeypatch.setattr(spatial,'solve_distance',lambda *a,**kw:(_ for _ in ()).throw(AssertionError()))
        spatial.signed_distances(pos[1:],np.array([1.]),np.array([0.]),np.array([[[2.]]]),[8.],config)


def test_graph_raw_control_aggregation_worker_and_cache(table,tmp_path,monkeypatch):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'caches')
    args=generate.parse_args(['--pc-xpass','--margin','distance','--accel','7','--accel-table-id','spatial',
          '--pc-xpass-id','prediction','--angle-step','90','--radial-gridsize','40','--distance-power','4','--distance-allowance','.7',
          '--distance-lane-power','2','--distance-control-power','5','--distance-lane-allowance','0',
          '--distance-control-allowance','.4','--num-workers','1','--top-n','10'])
    versions.start_generation(args)
    g=make_graph();details={}
    opts=dict(margin='distance',acceleration_config=args.acceleration_config,reaction_time=0,angle_step=90,
              radial_gridsize=40,distance_power=4.,distance_allowance=.7,use_position_discount=args.use_position_discount,
              distance_lane_power=2.,distance_control_power=5.,distance_lane_allowance=0.,distance_control_allowance=.4)
    row=physics.compute_graph_pc_xpass_metrics(g,diagnostic_details=details,**opts)
    pd.testing.assert_series_equal(row,physics.compute_graphs_pc_xpass_metrics([g],**opts)[0])
    pos,teams,players,origin,lookup,possessor=physics._full_as_default_simulation_inputs(g,candidate_indices=physics._candidate_target_indices(g))
    tb=details['ball_times']
    distance=spatial.signed_distances(pos,details['target_x'],details['target_y'],tb,np.full(len(pos),5.),args.acceleration_config)
    expected=1/(1+np.exp(-np.clip(5*(.4-distance),-60,60)))
    valid=np.isfinite(details['raw_control'])
    np.testing.assert_allclose(details['raw_control'][valid],expected[valid])
    expected_lane=1/(1+np.exp(-np.clip(2*(-distance),-60,60)))
    np.testing.assert_allclose(details['raw_lane'][valid],expected_lane[valid])
    raw=details['raw_lane'].copy();raw[teams=='attack']=0
    mask=np.isfinite(details['lane_survival'])
    np.testing.assert_allclose(details['lane_survival'][mask],physics.pc_xpass_lane_survival_from_raw(raw)[mask])
    item=dict(match_id='fixture',graphs=[g],labels=torch.stack([make_label()]))
    directory=versions.cache_dir('hawkeye',args)
    first=generate.prewarm_runtime_items([item],cache_dir=directory,args=args)
    second=generate.prewarm_runtime_items([item],cache_dir=directory,args=args)
    assert first['cache_misses']==1 and second['cache_hits']==1
    saved=physics.load_physical_xpass_match(directory,'fixture').iloc[0]
    for key,value in row.items():
        assert saved[key]==pytest.approx(value,nan_ok=True),key
    stored=versions.read_metadata(directory)
    for key in distance_control.SETTING_KEYS:
        assert stored[key]==getattr(args,key)
    changed=copy.copy(args);changed.distance_allowance=1
    with pytest.raises(ValueError,match='distance'):generate.prewarm_runtime_items([item],cache_dir=directory,args=changed)
    metadata=physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,**opts)
    another={**metadata,'distance_power':5}
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(metadata)!=physics.pc_xpass_lane_survival_metadata_fingerprint(another)
    monkeypatch.setattr(spatial,'load_table',lambda *a:(_ for _ in ()).throw(AssertionError('no table for cache-only')))
    lookup=physics.physical_xpass_inference_lookup_config({'pc_xpass':True},cache_dir=directory)
    assert lookup['margin']=='distance' and lookup['distance_power']==4
    assert lookup['distance_control_version']==distance_control.VERSION
    for key in distance_control.OVERRIDE_KEYS:
        assert lookup[key]==opts[key]


def test_preparation_distance_publication_resume_and_time_domain(tmp_path,monkeypatch):
    monkeypatch.setattr(motion,'ROOT',tmp_path/'tables')
    class InlinePool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def map(self,fn,tasks,**kw):return map(fn,tasks)
    monkeypatch.setattr(prepare,'ProcessPoolExecutor',InlinePool)
    args=prepare.parse_args(['--margin','distance','--table-id','built','--accel','7','--max-player-speed','8',
                             '--max-time','.001','--max-cells','127','--workers','1'])
    metadata=prepare.build(args)
    assert metadata['reachability_time']=='exact' and metadata['independent_holdout_failed_leaves']==0
    assert prepare.build(args)['fingerprint']==metadata['fingerprint']
    with pytest.raises(ValueError):motion.load_table('built')
    args.max_time=.002
    with pytest.raises(ValueError,match='different preparation'):prepare.build(args)
    defaults=prepare.parse_args(['--margin','distance','--table-id','auto','--accel','7'])
    assert defaults.max_time>150/3


def test_support_slice_interpolation_global_max_and_conservative_fallback():
    # Two angular intervals with independently varying speed/time. Check the
    # vectorized implementation against dense angular evaluation, including
    # maxima in interval interiors rather than only at sampled normals.
    arrays=dict(low=np.array([[0,0,0],[0,0,0],[0,math.pi/2,0]],float),
                high=np.array([[1,math.pi,2],[1,math.pi/2,2],[1,math.pi,2]],float),
                children=np.array([[1,2],[-1,-1],[-1,-1]]),axis=np.array([1,-1,-1]),
                fallback=np.array([False,False,False]),values=np.zeros((3,8)))
    for i in [1,2]:
        pts=arrays['low'][i]+(arrays['high'][i]-arrays['low'][i])*motion._CORNERS
        arrays['values'][i]=.2*pts[:,0]+.1*pts[:,2]+.1*np.abs(pts[:,1]-math.pi/2)
    rng=np.random.default_rng(91)
    queries=np.column_stack([rng.uniform(0,1,60),np.zeros(60),rng.uniform(0,2,60)])
    targets=rng.normal(size=(60,2));targets[:,1]=abs(targets[:,1])
    value,bad=spatial._lookup_distances(queries,targets,arrays)
    assert not bad.any()
    theta=np.linspace(0,math.pi,100001)
    for i in range(len(queries)):
        h=.2*queries[i,0]+.1*queries[i,2]+.1*np.abs(theta-math.pi/2)
        expected=np.max(targets[i,0]*np.cos(theta)+targets[i,1]*np.sin(theta)-h)
        assert value[i]==pytest.approx(expected,abs=1e-8)
    arrays['fallback'][2]=True
    _,bad=spatial._lookup_distances(np.array([[.1,0,1.]]),np.array([[100.,0.]]),arrays,.0001)
    assert not bad.item()  # Missing backward normals provably cannot win.


def _distance_process(graph,config,root):
    from pathlib import Path
    motion.ROOT=Path(root)
    opts=dict(margin='distance',reaction_time=0,
        acceleration_config=config,angle_step=90,radial_gridsize=40,distance_power=4,distance_allowance=.5,
        distance_lane_power=2,distance_control_power=5,distance_lane_allowance=0,distance_control_allowance=.4)
    direct=physics.compute_graph_pc_xpass_metrics(graph,**opts)
    task=dict(**opts,source='pc_xpass',eps=1e-4,teammate_policy='consider_teammates',
              enabled_metrics=['max_xpass','top10_xpass'],
              misses=[dict(graph=graph,match_id='fixture',action_index=7,physical_state_hash='fixture')])
    computed=pd.Series(physics._compute_runtime_physical_xpass_chunk(task)['rows'][0])
    pd.testing.assert_series_equal(direct,computed.reindex(direct.index),check_dtype=False)
    return computed


def test_distance_spawn_matches_serial(table):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    config=spatial.configuration(7,'spatial',max_speeds=(5,))
    graph=make_graph();expected=_distance_process(graph,config,table.parent)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        actual=pool.submit(_distance_process,graph,config,table.parent).result()
    pd.testing.assert_series_equal(expected,actual)


def test_one_sided_angular_limits_at_adaptive_faces():
    arrays=dict(low=np.array([[0,0,0],[0,0,0],[0,math.pi/2,0]],float),
                high=np.array([[1,math.pi,2],[1,math.pi/2,2],[1,math.pi,2]],float),
                children=np.array([[1,2],[-1,-1],[-1,-1]]),axis=np.array([1,-1,-1]),
                fallback=np.array([False,False,False]),values=np.array([[0]*8,[.2]*8,[.8]*8]))
    value,bad=spatial._lookup_distances(np.array([[.3,0,1.]]),np.array([[0.,2.]]),arrays)
    assert not bad.item() and value.item()==pytest.approx(1.8)


def _angular_table():
    arrays=dict(low=np.array([[0,0,0],[0,0,0],[0,math.pi/2,0]],float),
                high=np.array([[1,math.pi,2],[1,math.pi/2,2],[1,math.pi,2]],float),
                children=np.array([[1,2],[-1,-1],[-1,-1]]),axis=np.array([1,-1,-1]),
                # Internal nodes may be marked fallback in real artifacts.
                fallback=np.array([True,False,False]),values=np.zeros((3,8)))
    for i in [1,2]:
        pts=arrays['low'][i]+(arrays['high'][i]-arrays['low'][i])*motion._CORNERS
        arrays['values'][i]=.2*pts[:,0]+.1*pts[:,2]+.1*np.abs(pts[:,1]-math.pi/2)
    return arrays


@pytest.mark.parametrize('beta',[0.,.37,1.])
def test_compiled_lookup_matches_numpy_at_boundaries_and_reuses_kernels(beta,monkeypatch):
    arrays=_angular_table();rng=np.random.default_rng(123)
    times=np.r_[0.,2.,rng.uniform(0,2,98)]
    queries=np.column_stack([np.full(len(times),beta),np.zeros(len(times)),times])
    targets=rng.normal(size=(len(times),2));targets[:,1]=abs(targets[:,1])
    targets[:5]=[[0,0],[1,0],[-1,0],[0,2],[0,.01]]
    expected,bad=spatial._lookup_distances_numpy(queries,targets,arrays)
    monkeypatch.setattr(spatial,'_lookup_distances_numpy',lambda *a,**kw:pytest.fail('Expected compiled lookup'))
    actual,actual_bad=spatial._lookup_distances(queries,targets,arrays)
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=0)
    np.testing.assert_array_equal(actual_bad,bad)
    kernels=spatial._compiled_kernels()
    signatures=(tuple(kernels.support_profiles.signatures),tuple(kernels.angular_maxima.signatures))
    assert all(signatures)
    spatial._lookup_distances(queries,targets,arrays)
    assert spatial._compiled_kernels() is kernels
    assert signatures==(tuple(kernels.support_profiles.signatures),tuple(kernels.angular_maxima.signatures))


@pytest.mark.parametrize('partial_table',[False,True])
def test_general_lookup_retains_unknown_interval_bounds(partial_table,monkeypatch):
    arrays=_angular_table()
    queries=np.array([[.1,0,1.],[.1 if partial_table else .8,0,1.]])
    targets=np.array([[100.,0.],[-100.,0.]])
    if partial_table:arrays['fallback'][2]=True
    expected,bad=spatial._lookup_distances_numpy(queries,targets,arrays,.0001)
    monkeypatch.setattr(spatial,'_compiled_kernels',lambda:pytest.fail('Expected general lookup'))
    actual,actual_bad=spatial._lookup_distances(queries,targets,arrays,.0001)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual_bad,bad)
    if partial_table:np.testing.assert_array_equal(actual_bad,[False,True])


def _compiled_lookup_process():
    arrays=_angular_table();q=np.array([[.3,0,1.]])
    value,bad=spatial._lookup_distances(q,np.array([[0.,2.]]),arrays)
    kernels=spatial._compiled_kernels()
    return value,bad,bool(kernels.support_profiles.signatures and kernels.angular_maxima.signatures)


def test_spawn_worker_uses_compiled_lookup():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    expected,bad,compiled=_compiled_lookup_process()
    assert compiled
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        actual,actual_bad,worker_compiled=pool.submit(_compiled_lookup_process).result()
    assert worker_compiled
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual_bad,bad)


def test_empty_lookup_needs_no_compiler(monkeypatch):
    monkeypatch.setattr(spatial,'_compiled_kernels',lambda:pytest.fail('Empty query'))
    values,bad=spatial._lookup_distances(np.empty((0,3)),np.empty((0,2)),_angular_table())
    assert values.shape==bad.shape==(0,)
