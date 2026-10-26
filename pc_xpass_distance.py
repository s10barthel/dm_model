"""Exact-time acceleration reachability and signed spatial pc-xPass control.

The support table stores h(n,t)-R_standing(t). For fixed speed/time its
trilinear interpolant is piecewise linear in normal angle. Maximizing
n.q-h(n,t) over every interval gives signed Euclidean distance, including
negative distances inside a convex reachable region. Angular maximization of
the interpolant is analytic; interpolation accuracy is independently sampled.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import time
import numpy as np
import pc_xpass_acceleration as movement

VERSION = 'exact_time_support_v1'
REFERENCE_TOL = .001
LOOKUP_TOL = .005
SPATIAL_TOL = .01


def standing_radius(t):
    t=np.asarray(t, float)
    return np.where(t<=1, .5*t*t, t-.5)


@dataclass
class SpatialResult:
    lower: float
    upper: float
    evaluations: int = 0

    @property
    def distance(self):
        return .5*(self.lower+self.upper)


@dataclass
class SupportResult:
    lower: float
    upper: float
    point: np.ndarray
    normal: np.ndarray
    intervals: int = 0


def _support_mesh(speed, angle, duration, intervals):
    import clarabel
    from scipy import sparse
    p=np.array([math.cos(angle),math.sin(angle)])
    # Resolve the initial manoeuvre even for very long flight times. All
    # intervals, including the tail, remain in the continuous bound below.
    if duration>4:
        early=max(2,int(.875*intervals))
        knots=np.r_[np.linspace(0,4,early+1),np.linspace(4,duration,intervals-early+1)[1:]]
    else:
        knots=np.linspace(0,duration,intervals+1)
    widths=np.diff(knots)
    weights=np.r_[widths[0]/2,(widths[:-1]+widths[1:])/2,widths[-1]/2]
    rows=[];cols=[];values=[];rhs=[];cones=[]
    def block(entries,b,cone):
        offset=len(rhs)
        for row,col,value in entries:
            rows.append(offset+row);cols.append(col);values.append(value)
        rhs.extend(b);cones.append(cone)
    block([(0,0,1.),(1,1,1.)],[speed,0.],clarabel.ZeroConeT(2))
    for k in range(1,intervals+1):
        block([(1,2*k,-1.),(2,2*k+1,-1.)],[1.-1e-10,0.,0.],clarabel.SecondOrderConeT(3))
    start=len(rhs)
    for k,h in enumerate(widths):
        block([(j+1,2*k+j,1.) for j in range(2)]+[(j+1,2*(k+1)+j,-1.) for j in range(2)],
              [h*(1.-1e-10),0.,0.],clarabel.SecondOrderConeT(3))
    objective=-(weights[:,None]*p).ravel()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.max_iter=150
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-10
    matrix=sparse.csc_matrix((values,(rows,cols)),shape=(len(rhs),2*(intervals+1)))
    result=clarabel.DefaultSolver(sparse.csc_matrix((len(objective),len(objective))),objective,
                                  matrix,np.asarray(rhs),cones,settings).solve()
    if str(result.status) not in {'Solved','AlmostSolved'}:
        raise movement.UnresolvedArrival('Spatial support solver: '+str(result.status))
    velocity=np.asarray(result.x).reshape(-1,2)
    if (not np.isfinite(velocity).all() or np.linalg.norm(velocity[0]-[speed,0])>1e-7 or
            np.max(np.linalg.norm(velocity,axis=1))>1+1e-7 or
            np.max(np.linalg.norm(np.diff(velocity,axis=0),axis=1)/widths)>1+1e-7):
        raise movement.UnresolvedArrival('Spatial support trajectory violates movement constraints')
    point=weights@velocity
    costate=-np.asarray(result.z)[start:].reshape(-1,3)[:,1:]
    costate=np.vstack([costate[0]+.5*widths[0]*p,costate,np.zeros(2)])
    times=np.r_[0.,(knots[:-1]+knots[1:])/2,duration]
    dt=np.diff(times);slopes=np.diff(costate,axis=0)/dt[:,None]
    fractions=np.linspace(0,1,33)
    samples=costate[:-1,None,:]+fractions[None,:,None]*np.diff(costate,axis=0)[:,None,:]
    norms=np.linalg.norm(samples,axis=2)
    integral=dt@((.5*(norms[:,0]+norms[:,-1])+norms[:,1:-1].sum(axis=1))/32)
    upper=float(costate[0]@[speed,0]+dt@np.linalg.norm(p+slopes,axis=1)+integral)
    guard=2e-8*(1+duration)
    return SupportResult(float(p@point)-guard,upper+guard,point,p,intervals)


@lru_cache(maxsize=8192)
def solve_support(speed, angle, duration, tolerance, max_intervals=4096):
    if not (0<=speed<=1 and math.isfinite(angle) and math.isfinite(duration) and duration>=0 and tolerance>0):
        raise ValueError('Invalid normalized spatial support query')
    p=np.array([math.cos(angle),math.sin(angle)])
    if duration==0:
        return SupportResult(0.,0.,np.zeros(2),p)
    if speed==0:
        h=float(standing_radius(duration));return SupportResult(h,h,h*p,p)
    if speed+duration<=1:
        point=np.array([speed*duration,0.])+.5*duration**2*p
        h=float(p@point);return SupportResult(h,h,point,p)
    if abs(p[1])<1e-12:
        initial=speed*p[0];ramp=min(duration,1-initial)
        h=initial*ramp+.5*ramp*ramp+max(0,duration-ramp)
        return SupportResult(h,h,h*p,p)
    n=32
    while True:
        result=_support_mesh(speed,angle,duration,n)
        if result.upper-result.lower<=tolerance:
            return result
        if n>=max_intervals:
            raise movement.UnresolvedArrival('Spatial support bound exceeds tolerance at maximum mesh')
        n*=2


def _polygon_distance(points,target):
    from scipy.spatial import ConvexHull
    points=points[ConvexHull(points).vertices]
    edge=np.roll(points,-1,axis=0)-points
    normals=np.column_stack([edge[:,1],-edge[:,0]])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    signed=np.sum(normals*(target-points),axis=1)
    if signed.max()<=0:
        i=int(np.argmax(signed));return float(signed[i]),normals[i]
    weights=np.clip(np.sum((target-points)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
    delta=target-(points+weights[:,None]*edge)
    distance=np.linalg.norm(delta,axis=1);i=int(np.argmin(distance))
    return float(distance[i]),delta[i]/max(distance[i],1e-300)


def solve_distance(position,velocity,target,duration,accel,max_speed,*,tolerance=.005,max_evaluations=512):
    a=movement._positive(accel,'accel');cap=movement._positive(max_speed,'max_speed')
    tolerance=movement._positive(tolerance,'spatial tolerance')
    arrays=[np.asarray(v,float) for v in (position,velocity,target)]
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in arrays) or not math.isfinite(duration) or duration<0:
        raise ValueError('Spatial query needs finite 2-vectors and nonnegative finite time')
    origin,v,q=arrays;speed=np.linalg.norm(v);beta=min(speed/cap,1.)
    axis=v/speed if speed else np.array([1.,0.]);delta=q-origin
    length=cap*cap/a;t=duration*a/cap;eps=tolerance/length
    query=np.array([delta@axis,delta@np.array([-axis[1],axis[0]])])/length
    if beta==0 or beta+t<=1:
        center=np.array([beta*t,0.]);radius=float(standing_radius(t)) if beta==0 else .5*t*t
        value=(np.linalg.norm(query-center)-radius)*length
        return SpatialResult(value,value)
    # Coast is a feasible inner singleton; the acceleration-only disk is outer.
    coast=np.linalg.norm(query-[beta*t,0.])
    if .5*t*t<=eps:
        return SpatialResult((coast-.5*t*t)*length,coast*length)
    solved=[]
    for angle in np.linspace(0,2*math.pi,9)[:-1]:
        solved.append(solve_support(beta,float(angle),t,eps*.08))
    for _ in range(max_evaluations-len(solved)):
        points=np.array([s.point for s in solved])
        upper,normal=_polygon_distance(points,query)
        lower=max(float(s.normal@query-s.upper) for s in solved)
        if upper-lower<=eps:
            return SpatialResult(lower*length,upper*length,len(solved))
        solved.append(solve_support(beta,math.atan2(normal[1],normal[0]),t,eps*.08))
    raise movement.UnresolvedArrival('Signed spatial distance did not converge')


def reference_point(task):
    point,tolerance,_=task
    speed,angle,logtime=point
    try:
        r=solve_support(speed,angle,math.expm1(logtime),tolerance*.4)
        return point,dict(lower=r.lower,upper=r.upper,intervals=r.intervals)
    except movement.UnresolvedArrival as exc:
        return point,dict(unresolved=str(exc))


def _interval_max(qx,qy,lo,hi,c0,c1):
    slope=(c1-c0)/(hi-lo)
    best=np.maximum(qx*np.cos(lo)+qy*np.sin(lo)-c0,qx*np.cos(hi)+qy*np.sin(hi)-c1)
    radius=np.hypot(qx,qy)
    angle=np.arctan2(qy,qx)-np.arcsin(np.clip(slope/np.maximum(radius,1e-300),-1,1))
    active=(np.abs(slope)<radius)&(angle>=lo)&(angle<=hi)
    value=qx*np.cos(angle)+qy*np.sin(angle)-c0-slope*(angle-lo)
    return np.where(active,np.maximum(best,value),best)


@lru_cache(maxsize=8)
def _load(path,metadata_stamp,table_stamp):
    import json
    from pathlib import Path
    root=Path(path);meta=json.loads((root/'metadata.json').read_text())
    if (meta.get('status')!='complete' or meta.get('spatial_version')!=VERSION or
            meta.get('movement_version')!=movement.VERSION or movement._digest(root/'table.npz')!=meta.get('fingerprint')):
        raise ValueError('Incomplete, corrupted or incompatible distance acceleration table')
    with np.load(root/'table.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    n=len(arrays['low'])
    if (not n or arrays['low'].shape!=(n,3) or arrays['high'].shape!=(n,3) or
            arrays['values'].shape!=(n,8) or arrays['children'].shape!=(n,2) or
            arrays['axis'].shape!=(n,) or arrays['fallback'].shape!=(n,) or
            not np.isfinite(arrays['low']).all() or not np.isfinite(arrays['high']).all() or
            np.any(arrays['high']<=arrays['low']) or np.any(arrays['children']>=n)):
        raise ValueError('Invalid spatial table arrays')
    if (not np.allclose(arrays['low'][0], [0.,0.,0.], rtol=0, atol=1e-12) or
            not np.allclose(arrays['high'][0,:2], [1.,math.pi], rtol=0, atol=1e-12) or
            not math.isfinite(meta.get('distance_error_normalized', math.nan)) or meta['distance_error_normalized']<=0 or
            not math.isfinite(meta.get('max_distance', math.nan)) or meta['max_distance']<=0):
        raise ValueError('Invalid spatial table domain or accuracy')
    for i,child in enumerate(arrays['children']):
        if child[0]>=0 and (np.any(child<=i) or arrays['axis'][i] not in (0,1,2)):
            raise ValueError('Invalid spatial table tree')
        if child[0]>=0:
            axis=int(arrays['axis'][i]);middle=.5*(arrays['low'][i,axis]+arrays['high'][i,axis])
            left_hi=arrays['high'][i].copy();left_hi[axis]=middle
            right_lo=arrays['low'][i].copy();right_lo[axis]=middle
            if not (np.array_equal(arrays['low'][child[0]],arrays['low'][i]) and
                    np.array_equal(arrays['high'][child[0]],left_hi) and
                    np.array_equal(arrays['low'][child[1]],right_lo) and
                    np.array_equal(arrays['high'][child[1]],arrays['high'][i])):
                raise ValueError('Spatial table children do not partition their parent')
        if child[0]<0 and not arrays['fallback'][i] and not np.isfinite(arrays['values'][i]).all():
            raise ValueError('Invalid spatial interpolation leaf')
    return meta,arrays


def load_table(table_id):
    root=movement.table_path(table_id)
    try:return _load(str(root),(root/'metadata.json').stat().st_mtime_ns,(root/'table.npz').stat().st_mtime_ns)
    except FileNotFoundError as exc:raise ValueError('Missing distance acceleration table; prepare with --margin distance') from exc


def configuration(accel,table_id,*,max_speeds,fingerprint=None):
    a=movement.parse_accel(accel)
    if a is None or not table_id:raise ValueError('--margin distance requires --accel <number> and --accel-table-id')
    caps=[movement._positive(v,'max_player_speed') for v in max_speeds]
    meta,_=load_table(table_id)
    if fingerprint is not None and fingerprint!=meta['fingerprint']:raise ValueError('Distance table fingerprint differs from recorded cache')
    if meta['distance_error_normalized']*max(caps)**2/a>LOOKUP_TOL*(1+1e-10):
        raise ValueError('Distance table is too coarse for requested accel/speed')
    return dict(accel=a,table_id=table_id,fingerprint=meta['fingerprint'],movement_version=movement.VERSION,
                spatial_version=VERSION,initial_speed='clip_preserve_direction',reaction_time=0.,
                spatial_tolerance=SPATIAL_TOL,reachability_time='exact')


def _lookup_distances(queries,targets,arrays,table_error=0.):
    """Analytic global angular maximum, reusing each speed/time support slice.

    The union of table angular knots includes every piecewise-linear boundary,
    so subdividing onto this common grid changes no interpolated values.
    """
    if '_angles' not in arrays:
        arrays['_angles']=np.unique(np.r_[arrays['low'][:,1],arrays['high'][:,1]])
    angles=arrays['_angles'];lo,hi=angles[:-1],angles[1:];width=len(lo)
    unique,inverse=np.unique(queries[:,[0,2]],axis=0,return_inverse=True)
    points=np.column_stack([np.repeat(unique[:,0],width),np.tile(.5*(lo+hi),len(unique)),np.repeat(unique[:,1],width)])
    nodes=np.zeros(len(points),int)
    while True:
        active=arrays['children'][nodes,0]>=0
        if not active.any():break
        ix=np.flatnonzero(active);axis=arrays['axis'][nodes[ix]]
        mid=.5*(arrays['low'][nodes[ix],axis]+arrays['high'][nodes[ix],axis])
        nodes[ix]=arrays['children'][nodes[ix],(points[ix,axis]>=mid).astype(int)]
    weights=(points-arrays['low'][nodes])/(arrays['high'][nodes]-arrays['low'][nodes])
    # Evaluate both ends from the interval's own leaf. This also handles
    # one-sided limits at hanging faces in the adaptive interpolation tree.
    weights[:,1]=(np.tile(lo,len(unique))-arrays['low'][nodes,1])/(arrays['high'][nodes,1]-arrays['low'][nodes,1])
    profiles0=movement._interpolate(arrays['values'][nodes],weights).reshape(len(unique),width)
    weights[:,1]=(np.tile(hi,len(unique))-arrays['low'][nodes,1])/(arrays['high'][nodes,1]-arrays['low'][nodes,1])
    profiles1=movement._interpolate(arrays['values'][nodes],weights).reshape(len(unique),width)
    invalid=arrays['fallback'][nodes].reshape(len(unique),width)
    c0,c1=profiles0[inverse],profiles1[inverse];bad=invalid[inverse]
    projection0=targets[:,0,None]*np.cos(lo)+targets[:,1,None]*np.sin(lo)
    projection1=targets[:,0,None]*np.cos(hi)+targets[:,1,None]*np.sin(hi)
    result=np.max(np.where(~bad,np.maximum(projection0-c0,projection1-c1),-np.inf),axis=1)
    phi=np.arctan2(targets[:,1],targets[:,0]);radius=np.linalg.norm(targets,axis=1)
    upper=np.maximum(projection0,projection1)
    upper=np.where((phi[:,None]>=lo)&(phi[:,None]<=hi),radius[:,None],upper)
    upper-=np.minimum(c0,c1)
    ii,jj=np.nonzero(~bad&(upper>result[:,None]))
    if len(ii):
        slope=(profiles1[inverse[ii],jj]-profiles0[inverse[ii],jj])/(hi[jj]-lo[jj])
        angle=phi[ii]-np.arcsin(np.clip(slope/np.maximum(radius[ii],1e-300),-1,1))
        active=(np.abs(slope)<radius[ii])&(angle>=lo[jj])&(angle<=hi[jj])
        ii,jj,angle,slope=ii[active],jj[active],angle[active],slope[active]
        value=targets[ii,0]*np.cos(angle)+targets[ii,1]*np.sin(angle)-c0[ii,jj]-slope*(angle-lo[jj])
        np.maximum.at(result,ii,value)
    unknown=np.full(len(queries),-np.inf)
    ii,jj=np.nonzero(bad)
    if len(ii):
        # Feasible coast and brake-then-run sets prove that some unvalidated
        # normal intervals cannot affect this query. No player is excluded.
        tau=np.expm1(queries[ii,2]);beta=queries[ii,0];x,y=targets[ii].T
        bound=_interval_max(x-beta*tau,y,lo[jj],hi[jj],0.,0.)+standing_radius(tau)
        other=_interval_max(x-.5*beta**2,y,lo[jj],hi[jj],0.,0.)+standing_radius(tau)-standing_radius(np.maximum(tau-beta,0))
        bound=np.where(tau>=beta,np.minimum(bound,other),bound)
        np.maximum.at(unknown,ii,bound)
    return result,unknown>result-table_error


def signed_distances(players,target_x,target_y,t_ball,max_speeds,config,valid=None):
    meta,table=load_table(config['table_id'])
    if meta['fingerprint']!=config['fingerprint'] or config.get('spatial_version')!=VERSION:
        raise ValueError('Distance movement/table definition changed')
    a=movement._positive(config['accel'],'accel');caps=np.asarray(max_speeds,float)
    if caps.shape!=(len(players),) or not np.isfinite(caps).all() or np.any(caps<=0):raise ValueError('Invalid player speed caps')
    if meta['distance_error_normalized']*caps.max()**2/a>LOOKUP_TOL*(1+1e-10):raise ValueError('Distance table too coarse')
    x,y,t=np.broadcast_arrays(np.asarray(target_x,float),np.asarray(target_y,float),np.asarray(t_ball,float))
    selected=np.isfinite(t) if valid is None else np.broadcast_to(valid,t.shape)&np.isfinite(t)
    if np.any(t[selected]<0):raise ValueError('Negative ball time')
    if not np.isfinite(x[selected]).all() or not np.isfinite(y[selected]).all():raise ValueError('Nonfinite spatial target')
    output=np.zeros((len(players),x.size));allids=np.flatnonzero(selected.ravel())
    flat_x,flat_y,flat_t=x.ravel(),y.ravel(),t.ravel()
    for pi,(state,cap) in enumerate(zip(np.asarray(players,float),caps)):
        if not np.isfinite(state).all():raise ValueError('Nonfinite player input')
        speed=np.linalg.norm(state[2:]);beta=min(speed/cap,1.);axis=state[2:]/speed if speed else np.array([1.,0.])
        length=cap*cap/a
        for ids in np.array_split(allids,max(1,math.ceil(len(allids)/8192))):
            if not len(ids):continue
            delta=np.column_stack([flat_x[ids]-state[0],flat_y[ids]-state[1]])/length
            q=np.column_stack([delta@axis,np.abs(delta@np.array([-axis[1],axis[0]]))])
            tau=flat_t[ids]*a/cap
            values=np.zeros(len(ids));exact=(beta==0)|(beta+tau<=1)
            values[exact]=np.linalg.norm(q[exact]-np.column_stack([beta*tau[exact],np.zeros(exact.sum())]),axis=1)-(standing_radius(tau[exact]) if beta==0 else .5*tau[exact]**2)
            todo=np.flatnonzero(~exact)
            fallback=np.zeros(len(ids),bool)
            if len(todo):
                query=np.column_stack([np.full(len(todo),beta),np.zeros(len(todo)),np.log1p(tau[todo])])
                inside=(query[:,2]>=table['low'][0,2])&(query[:,2]<=table['high'][0,2])
                fallback[todo[~inside]]=True
                ix=todo[inside]
                if len(ix):
                    estimate,bad=_lookup_distances(query[inside],q[ix],table,meta['distance_error_normalized'])
                    values[ix]=estimate-standing_radius(tau[ix]);fallback[ix[bad]]=True
            # Distance coverage is physical separation, not an exclusion rule.
            fallback|=(np.linalg.norm(delta,axis=1)*length>meta['max_distance'])&~exact
            for j in np.flatnonzero(fallback):
                start=time.perf_counter()
                result=solve_distance(state[:2],state[2:],np.array([flat_x[ids[j]],flat_y[ids[j]]]),float(flat_t[ids[j]]),a,cap)
                values[j]=result.distance/length
                movement.COUNTERS['fallbacks']+=1;movement.COUNTERS['fallback_seconds']+=time.perf_counter()-start
            movement.COUNTERS['queries']+=len(ids)
            output[pi,ids]=values*length
    return output.reshape((len(players),*t.shape))
