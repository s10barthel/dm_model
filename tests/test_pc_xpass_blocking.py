"""Disk arrival: physical trajectories, raw-control semantics and cache identity."""
import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import pytest

import pc_xpass_acceleration as motion
import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate
from scripts import prepare_pc_xpass_acceleration as prepare
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph, make_label


def trajectory(result, velocity, target, radius, accel=7., cap=8.):
    assert 0 <= result.upper-result.lower <= .010000001
    assert result.position_error <= .01
    initial = np.asarray(velocity)*min(1., cap/max(np.linalg.norm(velocity), 1e-100))
    np.testing.assert_allclose(result.velocities[0], initial, atol=1e-7)
    assert np.max(np.linalg.norm(result.velocities, axis=1)) <= cap+1e-6
    dt = np.diff(result.times)
    if len(dt):
        assert np.max(np.linalg.norm(np.diff(result.velocities, axis=0)/dt[:, None], axis=1)) <= accel+1e-5
    end = np.sum(dt[:, None]*(result.velocities[:-1]+result.velocities[1:])/2, axis=0)
    assert np.linalg.norm(end-target) <= radius+.01000001


@pytest.mark.parametrize("velocity,target", [([0, 0], [20, 0]), ([-8, 0], [20, 0]),
    ([0, 8], [20, 0]), ([3, 4], [50, 20]), ([3, 4], [1.01, 0]), ([8, 0], [-1.01, 0]),
    ([0, 8], [.1, 0]), ([30, 40], [10, 3])])
def test_disk_trajectory(velocity, target):
    result = motion.solve_arrival([0, 0], velocity, target, 7, 8, blocking_radius=1.)
    trajectory(result, velocity, target, 1.)


def test_disk_exact_cases_and_zero_equivalence():
    z = motion.solve_arrival([0, 0], [8, 0], [-1, 0], 7, 8, blocking_radius=1)
    assert z.time == 0  # Already in the disk, even when moving away.
    for v in ([0, 0], [-8, 0], [0, 8]):
        point = motion.solve_arrival([0, 0], v, [20, 0], 7, 8)
        zero = motion.solve_arrival([0, 0], v, [20, 0], 7, 8, blocking_radius=0)
        assert zero.time == point.time
        np.testing.assert_array_equal(zero.velocities, point.velocities)
    assert motion.solve_arrival([0, 0], [0, 0], [20, 0], 7, 8, blocking_radius=1).time == pytest.approx(19/8+8/14)


def test_disk_monotonicity_scaling_and_nonradial_contact():
    v, q = np.array([0., 8.]), np.array([20., 0.])
    results = [motion.solve_arrival([0, 0], v, q, 7, 8, blocking_radius=r) for r in (0., .5, 1., 2.)]
    assert all(b.time <= a.time+.01 for a, b in zip(results, results[1:]))
    original = results[2]
    scaled = motion.solve_arrival([0, 0], 2*v, 2*q, 14, 16, blocking_radius=2.)
    assert scaled.time == pytest.approx(original.time, abs=.01)
    dt = np.diff(original.times)
    contact = np.sum(dt[:, None]*(original.velocities[:-1]+original.velocities[1:])/2, axis=0)
    assert abs(contact[1]) > .01  # Not the point on the initial radial line.
    rotated = motion.solve_arrival([10, 3], [-8, 0], [10, 23], 7, 8, blocking_radius=1.)
    assert rotated.time == pytest.approx(original.time, abs=.01)


def test_unresolved_disk_not_accepted():
    with pytest.raises(motion.UnresolvedArrival):
        motion.solve_arrival([0, 0], [0, 8], [20, 0], 7, 8, blocking_radius=1., max_evaluations=1)


def test_first_disk_contact_does_not_assume_monotonic_exact_time_reachability():
    velocity, target, radius = np.array([1., 0.]), np.array([.2, .01]), .01
    result = motion.solve_arrival([0, 0], velocity, target, 1., 1., blocking_radius=radius)
    trajectory(result, velocity, target, radius, accel=1., cap=1.)
    assert result.upper <= .2+.01
    residual, lower, _ = motion._fixed_time(velocity, target, .4, 32)
    assert residual > radius and lower > radius


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Fallback-only slices: genuine physical reference at every nontrivial query."""
    monkeypatch.setattr(motion, "ROOT", tmp_path/"tables")
    root = motion.table_path("disk");root.mkdir(parents=True)
    entries = []
    for i, cap in enumerate((8., 5.)):
        child = root/f"slice_{i}";child.mkdir()
        rho = 7/cap**2
        np.savez_compressed(child/"table.npz", low=np.array([[0., 0., 0.]]),
            high=np.array([[1., math.pi, 8.]]), values=np.zeros((1, 8)),
            children=np.array([[-1, -1]]), axis=np.array([-1]), fallback=np.array([True]))
        meta = dict(status="complete", movement_version=motion.VERSION, blocking_version=motion.BLOCKING_VERSION,
            blocking_radius_normalized=rho, fingerprint=motion._digest(child/"table.npz"),
            time_error_normalized=.00001, position_tolerance_normalized=.00001)
        versions.atomic_json(child/"metadata.json", meta)
        entries.append(dict(normalized_radius=rho, fingerprint=meta["fingerprint"]))
    np.savez_compressed(root/"table.npz", radii=np.array([e["normalized_radius"] for e in entries]),
                        fingerprints=np.array([e["fingerprint"] for e in entries]))
    versions.atomic_json(root/"metadata.json", dict(status="complete", movement_version=motion.VERSION,
        blocking_version=motion.BLOCKING_VERSION, blocking_slices=entries, fingerprint=motion._digest(root/"table.npz")))
    return root


def configuration():
    return motion.configuration(7, "disk", max_speeds=(8., 5.), blocking_radius=1.)


def test_bundle_compatibility_and_corruption(bundle):
    configuration()
    motion.configuration(14, "disk", max_speeds=(16.,), blocking_radius=2.)
    with pytest.raises(ValueError, match="normalized blocking"):
        motion.configuration(7, "disk", max_speeds=(8.,), blocking_radius=.7)
    with pytest.raises(ValueError, match="not interchangeable"):
        motion.configuration(7, "disk", max_speeds=(8.,))
    # Parent is cached; a changed child must still be checked.
    (bundle/"slice_0/table.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        configuration()


def test_disk_runtime_and_inside_shortcut(bundle):
    players = np.array([[0, 0, 0, 8], [0, 0, 0, 0]], float)
    cfg = configuration()
    got = motion.arrival_times(players, np.array([.5, 20.]), np.array([0., 0.]), np.array([8., 5.]), cfg)
    assert got[0, 0] == got[1, 0] == 0
    for i, cap in enumerate((8., 5.)):
        expected = motion.solve_arrival([0, 0], players[i, 2:], [20, 0], 7, cap, blocking_radius=1.)
        assert got[i, 1] == pytest.approx(expected.time)
    with pytest.raises(ValueError, match="Blocking radius"):
        motion.arrival_times(players, np.array([1.]), np.array([0.]), np.array([8., 5.]), {**cfg, "blocking_radius": 0.})


def test_repeated_disk_targets_are_solved_once_with_mask_preserved(bundle,monkeypatch):
    original=motion.solve_arrival;calls=[]
    def reference(*args,**kwargs):
        calls.append(args[2]);return original(*args,**kwargs)
    monkeypatch.setattr(motion,'solve_arrival',reference)
    targets=np.array([[20.,20.,2.],[20.,20.,2.]])
    valid=np.array([[True,True,False],[True,False,False]])
    got=motion.arrival_times(np.array([[0.,0.,0.,8.]]),targets,np.zeros_like(targets),[8.],configuration(),valid=valid)[0]
    assert len(calls)==1
    assert got[0,0]==got[0,1]==got[1,0]>0
    np.testing.assert_array_equal(got[~valid],0.)


def test_closed_disk_boundary_roundoff_is_not_an_extra_physical_radius(bundle):
    origin=np.array([-21.76,-24.7])
    theta=np.deg2rad(np.arange(0,360,3.))
    targets=origin+np.column_stack([np.cos(theta),np.sin(theta)])
    player=np.r_[origin,[8.,0.]]
    before=motion.COUNTERS['fallbacks']
    times=motion.arrival_times(player[None,:],targets[:,0],targets[:,1],[8.],configuration())
    np.testing.assert_array_equal(times,0.)
    assert motion.COUNTERS['fallbacks']==before
    for target in targets[::13]:
        assert motion.solve_arrival(origin,[8,0],target,7,8,blocking_radius=1.).time==0.
    # A genuinely outside target still requires a reversal, despite a small gap.
    outside=motion.solve_arrival([0,0],[8,0],[-1.-1e-8,0],7,8,blocking_radius=1.)
    assert outside.time>2.


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_radius(value):
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--blocking-radius", value])
    with pytest.raises(SystemExit):
        prepare.parse_args(["--table-id", "x", "--accel", "7", "--blocking-radius", value])


def test_cli_defaults_offsets_and_conflicts(bundle):
    a = generate.parse_args(["--pc-xpass", "--blocking-radius", "1"])
    assert a.lane_inflection_point == a.control_inflection_point == 0.
    a = generate.parse_args(["--pc-xpass", "--blocking-radius", "1", "--lane-inflection-point", ".15"])
    assert a.lane_inflection_point == .15 and a.control_inflection_point == 0.
    flags = ["--accel", "7", "--accel-table-id", "disk", "--blocking-radius", "1", "--max-player-speed", "8"]
    a = generate.parse_args(["--pc-xpass", *flags])
    assert a.acceleration_config["blocking_radius"] == 1.
    assert location.parse_args(flags).lane_inflection_point == 0.
    for flags in (["--blocking-radius", "0"], ["--blocking-radius", "1"]):
        with pytest.raises(SystemExit):generate.parse_args(flags)
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--margin", "distance", "--blocking-radius", "1"])


def test_no_acceleration_reaction_disk_contact():
    # Contact at t=.2 while the player is still on the fixed reaction trajectory.
    t = physics._pc_xpass_inertial_disk_arrival([0, 0, 5, 0], np.array([2.]), np.array([0.]), .7, 5., 1.)
    assert t[0] == pytest.approx(.2)
    t = physics._pc_xpass_inertial_disk_arrival([0, 0, -5, 0], np.array([2.]), np.array([0.]), .7, 5., 1.)
    assert t[0] == pytest.approx(.7+4.5/5)
    assert physics._pc_xpass_inertial_disk_arrival([0, 0, -5, 0], np.array([.5]), np.array([0.]), .7, 5., 1.)[0] == 0.


def test_raw_control_midpoint_and_aggregation_identity():
    # A stationary player needs one second to reach a 1 m disk around x=6 at 5 m/s.
    players = np.array([[0., 0., 0., 0.]])
    margins = physics._pc_xpass_arrival_margins(players, np.array([6.]), np.array([0.]), np.array([1.]),
                                               reaction_time=0., max_player_speed=5., blocking_radius=1.)
    assert physics.pc_xpass_raw_control_with_params(margins, power=15, inflection_point=0.)[0, 0] == .5
    graph=make_graph();d={}
    physics.compute_graph_pc_xpass_metrics(graph, blocking_radius=1., reaction_time=0.,
        lane_inflection_point=0., control_inflection_point=0., diagnostic_details=d, angle_step=90, radial_gridsize=10,
        endpoint_normalization='share')
    _,teams,_,_,_,_=physics._full_as_default_simulation_inputs(graph,candidate_indices=physics._candidate_target_indices(graph))
    raw=d['raw_lane'].copy();raw[teams=='attack']=0
    recomputed=physics.pc_xpass_lane_survival_from_raw(raw)
    valid=np.isfinite(d['lane_survival'])
    np.testing.assert_allclose(d['lane_survival'][valid],recomputed[valid])
    competing=np.nan_to_num(d['raw_control']).copy()
    node_ids=physics._node_ids(graph)
    _,_,_,_,_,pn=physics._full_as_default_simulation_inputs(graph,candidate_indices=physics._candidate_target_indices(graph))
    competing[d['player_ids'].index(node_ids[pn])]=0
    total=competing.sum(axis=0)
    for node,actual in d['receiver_control'].items():
        receiver=competing[d['player_ids'].index(node)]
        expected=np.divide(receiver**2,total,out=np.zeros_like(receiver),where=total>0)
        valid=np.isfinite(actual)
        np.testing.assert_allclose(actual[valid],expected[valid])


def test_metadata_zero_legacy_and_positive_identity():
    base = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
    zero = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,blocking_radius=0.)
    assert base == zero and 'blocking_radius' not in base
    disk = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,blocking_radius=1.)
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(base) != physics.pc_xpass_lane_survival_metadata_fingerprint(disk)


def test_prediction_version_inheritance(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'predictions')
    a=generate.parse_args(['--pc-xpass','--pc-xpass-id','radius','--blocking-radius','1'])
    versions.start_generation(a)
    b=generate.parse_args(['--pc-xpass','--pc-xpass-id','radius'])
    assert b.blocking_radius==1 and b.lane_inflection_point==0
    versions.start_generation(b)
    with pytest.raises(ValueError,match='blocking-radius'):
        generate.parse_args(['--pc-xpass','--pc-xpass-id','radius','--blocking-radius','0'])
    c=generate.parse_args(['--pc-xpass','--pc-xpass-id','old'])
    versions.start_generation(c)
    with pytest.raises(ValueError,match='blocking-radius'):
        generate.parse_args(['--pc-xpass','--pc-xpass-id','old','--blocking-radius','1'])


def test_location_radius_and_offsets_are_inherited(tmp_path,monkeypatch):
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'predictions')
    args=location.parse_args(['--blocking-radius','1'])
    args.pc_xpass_id='location_disk'
    versions.start_generation(args,location=True)
    with pytest.warns(UserWarning,match='blocking-radius'):
        loaded=location.parse_args(['--pc-xpass-id','location_disk','--blocking-radius','.5'])
    assert loaded.blocking_radius==1 and loaded.lane_inflection_point==0


def _disk_graph(graph,cfg,root):
    motion.ROOT=root
    return physics.compute_graph_pc_xpass_metrics(graph,reaction_time=0.,acceleration_config=cfg,blocking_radius=1.,
        max_player_speed=8.,angle_step=90.,radial_gridsize=30.,min_speed=15.,max_speed=15.,
        lane_inflection_point=0.,control_inflection_point=0.)


def test_spawn_and_cache_only(bundle,tmp_path,monkeypatch):
    cfg=configuration();graph=make_graph()
    expected=_disk_graph(graph,cfg,bundle.parent)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        actual=pool.submit(_disk_graph,graph,cfg,bundle.parent).result()
    pd.testing.assert_series_equal(expected,actual)
    meta=physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,reaction_time=0.,
                                 acceleration_config=cfg,blocking_radius=1.)
    cache=tmp_path/'cache';versions.atomic_json(cache/'metadata.json',meta)
    monkeypatch.setattr(motion,'load_table',lambda *a:pytest.fail('Cache-only must not read table'))
    assert physics.physical_xpass_inference_lookup_config({'pc_xpass':True},cache_dir=cache)['blocking_radius']==1.


def test_bundle_prepare_publication_and_refinement(tmp_path,monkeypatch):
    monkeypatch.setattr(motion,'ROOT',tmp_path/'tables')
    class InlinePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,tasks,**kwargs):
            for point,_,_ in tasks:
                t=math.expm1(point[2]);yield point,dict(lower=t,upper=t,residual=0.)
    monkeypatch.setattr(prepare,'ProcessPoolExecutor',InlinePool)
    flags=['--table-id','prepared','--accel','7','--max-player-speed-off','8','--max-player-speed-def','5','--blocking-radius','1','--workers','1']
    args=prepare.parse_args(flags);meta=prepare.build(args)
    assert len(meta['blocking_slices'])==2
    assert prepare.build(args)['fingerprint']==meta['fingerprint']
    motion.configuration(7,'prepared',max_speeds=(8.,5.),blocking_radius=1.)
    args.table_id='refined';args.source_table_id='prepared'
    assert prepare.build(args)['status']=='complete'
    args=prepare.parse_args(flags);args.blocking_radius=.7
    with pytest.raises(ValueError,match='different preparation settings'):prepare.build(args)


@pytest.mark.parametrize('accelerated',[False,True])
def test_prediction_cache_write_resume_and_radius_mismatch(bundle,tmp_path,monkeypatch,accelerated):
    import torch
    monkeypatch.setattr(versions.config,'PC_XPASS_DIR',tmp_path/'predictions')
    flags=['--pc-xpass','--pc-xpass-id','disk_predictions','--blocking-radius','1',
           '--reaction-time','0','--max-player-speed','8','--angle-step','90',
           '--radial-gridsize','40','--min-speed','15','--max-speed','15','--num-workers','1']
    if accelerated:flags+=['--accel','7','--accel-table-id','disk']
    args=generate.parse_args(flags);versions.start_generation(args)
    graph=make_graph();items=[dict(match_id='fixture',graphs=[graph],labels=torch.stack([make_label()]))]
    cache=versions.cache_dir('hawkeye',args)
    first=generate.prewarm_runtime_items(items,cache_dir=cache,args=args)
    second=generate.prewarm_runtime_items(items,cache_dir=cache,args=args)
    assert first['cache_misses']==1 and second['cache_hits']==1
    metadata=json.loads((cache/'metadata.json').read_text())
    assert metadata['blocking_radius']==1 and metadata['lane_inflection_point']==0
    loaded=generate.parse_args(['--pc-xpass','--pc-xpass-id','disk_predictions'])
    assert loaded.blocking_radius==1 and loaded.control_inflection_point==0
    bad=copy.copy(args);bad.blocking_radius=.7
    with pytest.raises(ValueError,match='[Bb]locking radius'):
        generate.prewarm_runtime_items(items,cache_dir=cache,args=bad)
    if not accelerated:
        assert physics.physical_xpass_inference_lookup_config({'pc_xpass':True},cache_dir=cache)['blocking_radius']==1
