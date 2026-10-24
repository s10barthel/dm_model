"""Diagnostic reconstruction of Cascioli 2025; see output metadata for conventions."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import physical_pass_model as ppm
from datatools.hawkeye import build_hawkeye_situation, clean_hawkeye_tracking, clean_hawkeye_ball, load_hawkeye_ball
from scripts.visualize_hawkeye import resolve_ballreceipt, resolve_hawkeye_png_frames


def paper_line(pos, teams, passer, origin, endpoint, speed=12., n=30):
    q = origin + np.linspace(0., 1., n)[:, None] * (endpoint-origin)
    delta = q[None,:,:] - (pos[:,None,:2] + .7*pos[:,None,2:])
    d = np.linalg.norm(delta, axis=-1)
    direction = np.divide(delta, d[...,None], out=np.zeros_like(delta), where=d[...,None]>0)
    v = np.clip(np.sum(pos[:,None,2:]*direction, axis=-1), 0., 12.)
    deff = np.maximum(d-np.where(teams=='defense', .7, 0.)[:,None], 0.)
    ta = (12.-v)/7.
    da = v*ta + 3.5*ta**2
    travel = np.where(deff<=da, (np.sqrt(v*v+14.*deff)-v)/7., ta+np.maximum(deff-da,0.)/12.)
    tti = .7+travel
    drag = 1.22*.25*.038/(2*.42)
    tb = np.expm1(drag*np.linalg.norm(q-origin,axis=-1))/(drag*speed)
    dt = np.diff(tb, prepend=0.)
    ready = 1./(1.+np.exp(-np.clip((tb[None,:]-tti)*np.pi/(np.sqrt(3.)*.2),-60,60)))
    ready[passer] = 0.
    exact = np.zeros(len(pos))
    euler = exact.copy()
    max_step = 0.
    for k in range(n):
        rates = 4.3*ready[:,k]
        rate = rates.sum()
        step = rate*dt[k]
        max_step = max(max_step,float(step))
        euler += (1.-euler.sum())*rates*dt[k]
        if rate>0:
            exact += (1.-exact.sum())*(-np.expm1(-step))*rates/rate
    defense = teams=='defense'
    return dict(block=float(exact[defense].sum()), receive=float(exact[~defense].sum()), free=float(1-exact.sum()),
                euler_block=float(euler[defense].sum()), euler_receive=float(euler[~defense].sum()),
                euler_free=float(1-euler.sum()), euler_valid=bool((euler>=-1e-10).all() and euler.sum()<=1+1e-10),
                max_euler_step=max_step)


def corridor(pos,teams,passer,origin,endpoint,speed=12.):
    diff = endpoint-origin
    side = np.array([-diff[1],diff[0]])*.1
    results = {label:paper_line(pos,teams,passer,origin,endpoint+offset,speed)
               for label,offset in [('center',np.zeros(2)),('left',side),('right',-side)]}
    out = {f'{label}_{key}':value for label,result in results.items() for key,value in result.items()}
    count = sum(r['block']>r['receive'] for r in results.values())
    out.update(blocked_lines=count,blocked_any=count>=1,blocked_majority=count>=2,blocked_all=count==3)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--situation-id',default='0291ae3b-9848-4f16-ab2e-2dc194ed8649')
    parser.add_argument('--time-norm',type=float,default=0.)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'out/cascioli_comparison')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    chunks=pd.read_csv(ROOT/'hawkeye_data/centroid_data_team.csv',chunksize=100000,low_memory=False)
    tracking=clean_hawkeye_tracking(pd.concat([c.loc[c.id==args.situation_id] for c in chunks]))
    ball=clean_hawkeye_ball(load_hawkeye_ball(ROOT/'hawkeye_data/ball_data_selected.csv'))
    situation,_,_=build_hawkeye_situation(tracking,ball,freeze_ballreceipt=True)
    selection=resolve_hawkeye_png_frames(situation,resolve_ballreceipt(tracking),[args.time_norm])[0]
    graph=situation.graph_features_0[list(situation.actions.index).index(selection['frame_id'])]
    details,lanes={},[]
    settings=dict(eps=.0001,top_n=None,top_pass_values=[10],top_xt=True,
        reaction_time_mode='dist_pass',dist_pass_div=50.,dist_pass_min=0.,dist_pass_max=.7,
        max_player_speed=5.,max_player_speed_off=5.,max_player_speed_def=5.,
        lane_power=15.,lane_inflection_point=.2,control_power=15.,control_inflection_point=.2,
        endpoint_normalization='share',boost_def_endpoint_control=1.,use_position_discount=False,
        max_speed=25.,min_speed=3.,speed_step=2.,angle_step=3.,radial_gridsize=1.,ball_dec=ppm.PC_XPASS_DEFAULT_BALL_DEC)
    scores=ppm.compute_graph_pc_xpass_metrics(graph,diagnostic_details=details,lane_control_records=lanes,**settings)
    scores.to_csv(args.output_dir/'pc_xpass_wide.csv',header=['value'])
    lane_df=pd.DataFrame(lanes)
    lane_df.to_csv(args.output_dir/'pc_selected_defender_controls.csv',index=False)
    candidates=ppm._candidate_target_indices(graph)
    pos,teams,players,origin,lookup,possessor_node=ppm._full_as_default_simulation_inputs(graph,candidate_indices=candidates)
    node_ids=ppm._node_ids(graph)
    passer=list(players).index(node_ids[possessor_node])
    pd.DataFrame(dict(player_id=players,team=teams,x=pos[:,0],y=pos[:,1],vx=pos[:,2],vy=pos[:,3],
        reaction_time=ppm._pc_xpass_dist_pass_reaction_times(pos,passer,dist_pass_div=50.,dist_pass_min=0.,dist_pass_max=.7))).to_csv(args.output_dir/'players.csv',index=False)
    rows,matched=[],[]
    xt=ppm._pc_xpass_xt_values(details['target_x'],details['target_y'])
    for node in candidates:
        receiver=str(node_ids[node]); prefix=receiver+'__'
        endpoint=np.array([scores[prefix+'target_x'],scores[prefix+'target_y']])
        row=dict(receiver_id=receiver,x=pos[lookup[node],0],y=pos[lookup[node],1],
            pc_top_pass10=scores[prefix+'top_pass10_xpass'],pc_selected_xpass=scores[prefix+'max_xpass'],
            pc_lane_survival=scores[prefix+'lane_survival'],pc_endpoint_control=scores[prefix+'control_prob'],
            pc_speed=scores[prefix+'speed'],pc_distance=scores[prefix+'distance'],pc_target_x=endpoint[0],pc_target_y=endpoint[1])
        for name,target,speed in [('current_12',pos[lookup[node],:2],12.),('selected_12',endpoint,12.),('selected_speed',endpoint,row['pc_speed'])]:
            row.update({name+'_'+k:v for k,v in corridor(pos,teams,passer,origin,target,speed).items()})
        control=details['receiver_control'][receiver]
        grid=details['lane_survival']*control
        indices=ppm._pc_xpass_ranked_pass_indices(grid,grid*xt)[:10]
        for rank,flat in enumerate(indices,1):
            si,ai,ri=np.unravel_index(flat,grid.shape)
            target=np.array([details['target_x'][ai,ri],details['target_y'][ai,ri]])
            result=paper_line(pos,teams,passer,origin,target,details['speeds'][si])
            matched.append(dict(receiver_id=receiver,rank=rank,speed=details['speeds'][si],target_x=target[0],target_y=target[1],
                pc_xpass=grid.ravel()[flat],pc_lane_survival=details['lane_survival'].ravel()[flat],pc_control=control.ravel()[flat],**result))
        rows.append(row)
    table=pd.DataFrame(rows); matched_df=pd.DataFrame(matched)
    matched_df.to_csv(args.output_dir/'matched_top10_paths.csv',index=False)
    means=matched_df.groupby('receiver_id')[['block','receive','free']].mean().add_prefix('matched_top10_')
    table=table.merge(means,on='receiver_id')
    table.to_csv(args.output_dir/'receiver_comparison.csv',index=False)
    metadata=dict(situation_id=args.situation_id,selection=selection,settings=settings,passer_id=str(players[passer]),
        ball_origin=origin.tolist(),team_map=situation.prefix_to_team,
        paper=dict(reaction_time=.7,max_speed=12.,acceleration=7.,block_radius=.7,ball_speed=12.,sigma=.2,control_rate=4.3,cone_full_width_factor=.2,samples=30),
        conventions=['BallReceipt frame; frozen BallReceipt pipeline default.',
            'All finite real players including keepers; passer excluded from interception.',
            '30 nodes include origin/end; preceding ball-time increment.',
            'Projected initial speed capped at 12 m/s; zero-distance direction is zero.',
            'block/receive/free use exact competing-hazard updates with node rates fixed in each interval.',
            'euler_* implement literal simultaneous paper recurrence without clipping.',
            'selected_speed/matched_top10 substitute pc launch speed but retain paper quadratic drag.',
            'Paper receive is any attacker, not specifically the named receiver.'])
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf8')
    for row in rows:
        subset=lane_df[lane_df.receiver_id==row['receiver_id']]
        assert np.isclose(np.prod(1.-subset.lane_control),row['pc_lane_survival'])
        assert np.isclose(row['pc_selected_xpass'],np.clip(row['pc_lane_survival']*row['pc_endpoint_control'],.0001,.9999))
        sub=matched_df[matched_df.receiver_id==row['receiver_id']]
        assert np.isclose(np.clip(sub.pc_xpass.mean(),.0001,.9999),row['pc_top_pass10'])
    assert np.allclose(matched_df[['block','receive','free']].sum(axis=1),1.)
    assert (matched_df[['block','receive','free']].values>=-1e-12).all()
    print(json.dumps(metadata,indent=2,default=str))
    print(table[['receiver_id','pc_top_pass10','pc_selected_xpass','pc_lane_survival','pc_endpoint_control','pc_speed',
        'current_12_center_block','current_12_center_receive','current_12_blocked_lines','selected_speed_center_block',
        'selected_speed_center_receive','matched_top10_block','matched_top10_receive']].to_string(index=False))
    print('All reconstruction and probability-conservation checks passed.')

if __name__=='__main__':
    main()

