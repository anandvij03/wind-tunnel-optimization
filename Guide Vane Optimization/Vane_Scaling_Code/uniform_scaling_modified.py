"""Single-slider corner vanes with automatic inward offsets above 10 degrees.
Run: python corner_vanes_clearance.py (Python standard library only).
Geometry checks are not CFD validation. Embedded drawing dimensions are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
from pathlib import Path

PAPER_RADIUS_MM = 2400.0
PAPER_CHORD_MM = PAPER_RADIUS_MM * math.sqrt(2)
PITCH_MM = 2070.0
NUMBER_OF_VANES = 5
CORNER_CENTER = (0.0, 0.0)       # actual centre of the INNER R2400 wall arc
DRAWING_LIMIT_MM = 2400.0 + 8844.4
GEOMETRY_GUARD_MM = 1.0         # numerical guard only, not a CFD flow criterion
MAX_CSV_DEVIATION_MM = 0.02
CONTOUR_TOLERANCE_MM = 0.3      # conservative straight-segment sampling error
# Explicit design policies, NOT literature-derived aerodynamic optima.
TE_FULL_THICKNESS_MM = 0.005 * PAPER_CHORD_MM  # 0.5% baseline chord
TE_BLEND_START = 0.80  # modify only last 20% of camber parameter
CAP_SEGMENTS = 8       # geometry samples, not a CFD mesh specification

# Exact 45-row Base_Data_Points.csv supplied by the user, in cm (x,y,z).
# Embedded so the generator runs from one Python file without a sidecar CSV.
BASELINE_CSV = """339.6631664,0.251911463,0
338.1821902,2.061038258,0
333.6558033,7.359685903,0
325.8568385,15.76081535,0
314.4805319,26.62091699,0
299.2430352,39.05566229,0
279.9953274,51.98229614,0
256.8308686,64.19749441,0
230.1641556,74.49058603,0
200.7605883,81.78143617,0
169.7056275,85.26287612,0
138.3122911,84.52164505,0
107.979326,79.61121121,0
90.23161671,74.6653966,0
80.02564072,71.05616113,0
55.53584402,59.78146196,0
35.25228608,46.97813554,0
19.53827385,33.93351668,0
8.417494873,21.86212295,0
1.673195941,11.76850024,0
-1.024607858,4.358721864,0
-1.10865277,3.163890561,0
0,0,0
3.144372983,-1.140307607,0
4.322379806,-1.092684718,0
11.62133474,1.024563526,0
21.84457844,5.914290012,0
34.99304494,12.97559018,0
51.0791886,21.47164564,0
70.07288853,30.56126084,0
91.83884607,39.34152165,0
100.7280612,42.39035631,0
116.0755888,46.9098409,0
142.2728217,52.44875895,0
169.7056275,55.32586891,0
197.4768088,55.18896782,0
224.6034396,52.03046607,0
250.1271546,46.20018836,0
273.21845,38.36042666,0
293.248,29.39411889,0
309.8106592,20.28818987,0
322.7035981,12.01559761,0
331.8721759,5.433377864,0
337.3425478,1.204998888,0
339.1593435,-0.251911463,0"""

Point = tuple[float, float]


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def mul(a: Point, k: float) -> Point:
    return a[0] * k, a[1] * k


def rotate_180(p: Point, center: Point) -> Point:
    """Half-turn around a vane's fixed chord midpoint."""
    return 2*center[0]-p[0], 2*center[1]-p[1]


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def turn(p: Point) -> Point:
    """Rotate the supplied chord-horizontal CSV +45 degrees into the duct."""
    q = 1 / math.sqrt(2)
    return q * (p[0] - p[1]), q * (p[0] + p[1])


def paper_half_thickness(t: float) -> float:
    """Paper NACA 0009 half thickness, using developed arc length in mm."""
    arc = math.pi * PAPER_RADIUS_MM / 2
    return 5 * 0.09 * arc * (
        0.2969 * math.sqrt(t) - 0.1260 * t - 0.3516 * t*t
        + 0.2843 * t*t*t - 0.1015 * t*t*t*t
    )


def paper_surface(t: float, sign: int) -> Point:
    """Analytic paper profile in the input CSV's chord-horizontal frame."""
    phi = math.pi * t / 2
    s, c = math.sin(phi), math.cos(phi)
    b = (PAPER_CHORD_MM / 2 * (1+s-c),
         PAPER_CHORD_MM / 2 * (s+c-1))
    n = ((s-c)/math.sqrt(2), (s+c)/math.sqrt(2))
    return add(b, mul(n, sign * paper_half_thickness(t)))


def nearest_station(p: Point, sign: int) -> tuple[float, float]:
    """Find the paper station underlying an input point, without scipy."""
    def error(t: float) -> float:
        q = paper_surface(t, sign)
        return dot(sub(p, q), sub(p, q))
    count = 400
    j = min(range(count+1), key=lambda i: error(i/count))
    low, high = max(0.0, (j-1)/count), min(1.0, (j+1)/count)
    for _ in range(75):
        a = low + (high-low)*0.381966011250105
        b = high - (high-low)*0.381966011250105
        if error(a) < error(b): high = b
        else: low = a
    t = (low+high)/2
    return t, math.sqrt(error(t))


def load_source() -> dict:
    rows = list(csv.reader(io.StringIO(BASELINE_CSV)))
    if len(rows) != 45 or any(len(row) != 3 for row in rows):
        raise ValueError('Expected exactly the supplied 45 rows of x,y,z')
    points = [tuple(map(float, row)) for row in rows]
    if any(not all(math.isfinite(v) for v in point) or abs(point[2]) > 1e-8
           for point in points):
        raise ValueError('Source contains nonfinite coordinates or nonzero z')
    points_mm = [(x*10, y*10) for x,y,_ in points]
    le_index = min(range(len(points_mm)),key=lambda j: math.hypot(*points_mm[j]))
    if le_index != 22 or math.hypot(*points_mm[le_index]) > 1e-6:
        raise ValueError('Expected upper TE -> LE (row 23) -> lower TE order')
    te_mid = mul(add(points_mm[0], points_mm[-1]), 0.5)
    input_chord = math.dist(te_mid, points_mm[le_index])
    if abs(input_chord-PAPER_CHORD_MM) > 0.05:
        raise ValueError(f'Source chord is {input_chord:.6f} mm; expected {PAPER_CHORD_MM:.6f} mm')
    # Keep the 45 original contour stations. The analytic formula reconstructs
    # the continuous geometry between points and avoids a spline-fit error.
    stations = []
    errors = []
    for j,p in enumerate(points_mm):
        sign = 1 if j <= le_index else -1
        t, error = nearest_station(p, sign)
        stations.append((t,sign))
        errors.append(error)
    if max(errors)>MAX_CSV_DEVIATION_MM:
        raise ValueError(f'CSV differs from paper NACA construction by {max(errors):.4f} mm')
    return dict(stations=stations, source_points=points_mm,
                chord_mm=input_chord, max_reconstruction_error_mm=max(errors),
                point_count=len(points_mm), path='embedded Base_Data_Points.csv')


def mesh_half_thickness(t: float) -> float:
    """Same mm thickness on every vane; C2 blend to a finite flat TE face."""
    u=max(0.0,min(1.0,(t-TE_BLEND_START)/(1-TE_BLEND_START)))
    blend=u**3*(10-15*u+6*u*u)
    delta=TE_FULL_THICKNESS_MM/2-paper_half_thickness(1.0)
    if delta < 0:
        raise ValueError('TE policy must not thin the original trailing edge')
    return paper_half_thickness(t)+delta*blend


def point_segment(p,a,b):
    v=sub(b,a);den=dot(v,v)
    t=max(0,min(1,dot(sub(p,a),v)/den)) if den else 0
    return math.dist(p,add(a,mul(v,t)))


def walls() -> list[list[Point]]:
    """Physical wall segments read from the dimensioned top-view drawing."""
    h=DRAWING_LIMIT_MM
    r=PAPER_RADIUS_MM
    # The black point in the supplied drawing is the CENTRE of this inner
    # fillet, not the virtual intersection of two walls tangent to a different
    # circle. It goes from (r,0) to (0,-r) about O=(0,0).
    inner=[(r,4000),(r,0)]
    inner += [(r*math.cos(math.pi*j/320),-r*math.sin(math.pi*j/320))
              for j in range(1,161)]
    inner += [(-4000,-r)]
    outer_arc=[(h-r+r*math.cos(math.radians(-90+90*j/160)),
                -h+r+r*math.sin(math.radians(-90+90*j/160))) for j in range(161)]
    return [inner,outer_arc,[(h-r,-h),(-4000,-h)],[(h,-h+r),(h,4000)],
            [(h,4000),(r,4000)], [(-4000,-h),(-4000,-r)]]


def segments(points: list[Point], closed: bool=False):
    return list(zip(points,points[1:]+([points[0]] if closed else [])))


def on_segment(p: Point,a: Point,b: Point,eps: float=1e-8) -> bool:
    return abs(cross(sub(b,a),sub(p,a)))<eps and dot(sub(p,a),sub(p,b))<=eps


def crosses(a: Point,b: Point,c: Point,d: Point) -> bool:
    ab1,ab2=cross(sub(b,a),sub(c,a)),cross(sub(b,a),sub(d,a))
    cd1,cd2=cross(sub(d,c),sub(a,c)),cross(sub(d,c),sub(b,c))
    eps=1e-8
    if ab1*ab2 < -eps and cd1*cd2 < -eps:
        return True
    return ((abs(ab1)<eps and on_segment(c,a,b)) or
            (abs(ab2)<eps and on_segment(d,a,b)) or
            (abs(cd1)<eps and on_segment(a,c,d)) or
            (abs(cd2)<eps and on_segment(b,c,d)))


def segment_gap(a: Point,b: Point,c: Point,d: Point) -> float:
    if crosses(a,b,c,d):return 0.0
    return min(point_segment(a,c,d),point_segment(b,c,d),
               point_segment(c,a,b),point_segment(d,a,b))


def poly_gap(p: list[Point], q: list[Point], closed_p: bool=True,
             closed_q: bool=True) -> float:
    result=float('inf')
    qq=[(a,b,(min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])))
        for a,b in segments(q,closed_q)]
    for a,b in segments(p,closed_p):
        ax0,ay0,ax1,ay1=min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])
        for c,d,(cx0,cy0,cx1,cy1) in qq:
            if ax0>cx1+result or cx0>ax1+result or ay0>cy1+result or cy0>ay1+result:
                continue
            result=min(result,segment_gap(a,b,c,d))
            if result<1e-7:return 0.0
    return result


def inside(p: Point, poly: list[Point]) -> bool:
    flag=False
    for a,b in segments(poly,True):
        if (a[1]>p[1])!=(b[1]>p[1]):
            x=a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if p[0]<x:flag=not flag
    return flag



import functools
import heapq
import time

ANCHOR_THETA = 10.0
MAX_THETA = 50.0
Q = math.sqrt(0.5)


def layout_parameters(theta):
    """Analytic solve, no iterative optimizer or artificial 23-degree cutoff.

    c_i = c_inner + 2*i*p*tan(theta/2), m_i = m_inner + i*p.
    LE_x and TE_y are fixed above 10 degrees by p*(1+tan(theta/2))=constant.
    The other endpoint coordinates move along the inlet/outlet flow direction.
    LE and TE rows remain straight and subtend precisely theta.
    """
    if not math.isfinite(theta) or not 0 <= theta <= MAX_THETA:
        raise ValueError('Theta must be between 0 and 50 degrees.')
    u=math.tan(math.radians(theta/2))
    anchor=math.tan(math.radians(min(theta,ANCHOR_THETA)/2))
    pitch=PITCH_MM*(1+anchor)/(1+u)
    c_inner=PAPER_CHORD_MM-4*PITCH_MM*anchor
    outer_r=(DRAWING_LIMIT_MM-PAPER_RADIUS_MM)*math.sqrt(2)+PAPER_RADIUS_MM
    m_inner=PAPER_RADIUS_MM+(outer_r-PAPER_RADIUS_MM-4*PITCH_MM)/2
    return pitch,c_inner,m_inner


def section_point(v,t,sign=0):
    phi=math.pi*t/2
    radius=v['radius_mm']+sign*mesh_half_thickness(t)*v['thickness_scale']
    return (v['circle_center'][0]+radius*math.cos(phi),
            v['circle_center'][1]-radius*math.sin(phi))


def sample_side(v,sign,tolerance):
    # sqrt(t) makes sampling well conditioned at the rounded leading edge.
    def at(u):return section_point(v,u*u,sign)
    def refine(a,b,pa,pb,depth=0):
        if max(point_segment(at(a+(b-a)*f),pa,pb) for f in (.25,.5,.75))<=tolerance:
            return [pa,pb]
        if depth==22:raise ValueError('Contour sampling failed to converge.')
        mid=(a+b)/2;pm=at(mid)
        return refine(a,mid,pa,pm,depth+1)[:-1]+refine(mid,b,pm,pb,depth+1)
    seeds=sorted({j/24 for j in range(25)}|{math.sqrt(TE_BLEND_START)})
    pts=[at(0)]
    for a,b in zip(seeds,seeds[1:]):pts+=refine(a,b,at(a),at(b))[1:]
    return pts


def geometry(theta,tolerance=.3):
    pitch,c_inner,m_inner=layout_parameters(theta)
    u=math.tan(math.radians(theta/2));vanes=[]
    for i in range(5):
        c=c_inner+2*i*pitch*u;r=c*Q;m=m_inner+i*pitch
        midpoint=(m*Q,-m*Q)
        v=dict(index=i+1,chord_mm=c,radius_mm=r,midpoint=midpoint,
               circle_center=(m*Q-r/2,-m*Q+r/2),
               thickness_scale=min(1,c/PAPER_CHORD_MM),
               inward_offset_mm=i*(PITCH_MM-pitch))
        v['le']=section_point(v,0);v['te']=section_point(v,1)
        v['upper']=list(reversed(sample_side(v,1,tolerance)))
        v['lower']=sample_side(v,-1,tolerance)
        v['trailing_edge']=[add(v['lower'][-1],mul(sub(v['upper'][0],v['lower'][-1]),j/CAP_SEGMENTS)) for j in range(CAP_SEGMENTS+1)]
        v['contour']=v['upper']+v['lower'][1:]+v['trailing_edge'][1:-1]
        v['te_thickness_mm']=TE_FULL_THICKNESS_MM*v['thickness_scale']
        v['camber_midpoint']=section_point(v,.5)
        vanes.append(v)
    le_row=sub(vanes[-1]['le'],vanes[0]['le']);te_row=sub(vanes[-1]['te'],vanes[0]['te'])
    measured=math.degrees(math.atan2(abs(cross(le_row,te_row)),dot(le_row,te_row)))
    if abs(measured-theta)>1e-8:raise ValueError('Endpoint row angle mismatch.')
    for v in vanes:
        for row,key in ((le_row,'le'),(te_row,'te')):
            if abs(cross(row,sub(v[key],vanes[0][key])))/math.hypot(*row)>1e-7:
                raise ValueError('An endpoint left its straight row.')
    return dict(theta_deg=theta,measured_theta_deg=measured,pitch_mm=pitch,
                sampling_tolerance_mm=tolerance,vanes=vanes)


def signed_wall_gaps(p):
    """Exact signed distances at a point to the two physical wall boundaries.
    Used in the original x>=0,y<=0 corner quadrant, also checked separately.
    """
    x,y=p;h=DRAWING_LIMIT_MM;r=PAPER_RADIUS_MM
    inner=math.hypot(x,y)-r
    outer=(r-math.hypot(x-(h-r),y-(-h+r))
           if x>h-r and y<-h+r else min(h-x,h+y))
    return inner,outer


def outer_normal_gap(v,t):
    """Distance from outer surface along its camber normal to the outer wall.
    A geometric passage diagnostic, not a flow-area or velocity prediction.
    """
    x,y=section_point(v,t,1);nx=math.cos(math.pi*t/2);ny=-math.sin(math.pi*t/2)
    h=DRAWING_LIMIT_MM;r=PAPER_RADIUS_MM;candidates=[]
    if nx>1e-12:
        d=(h-x)/nx
        if y+d*ny>=-h+r-1e-7:candidates.append(d)
    if ny<-1e-12:
        d=(-h-y)/ny
        if x+d*nx<=h-r+1e-7:candidates.append(d)
    ox=x-(h-r);oy=y-(-h+r);projection=ox*nx+oy*ny
    disc=projection**2+r*r-ox*ox-oy*oy
    if disc>=0:
        d=-projection+math.sqrt(disc)
        if x+d*nx>=h-r-1e-7 and y+d*ny<=-h+r+1e-7:candidates.append(d)
    if not candidates:raise ValueError('No outer wall intersection along camber normal.')
    return min(candidates)


def box(points):
    return min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)


def box_distance2(a,b):
    dx=max(a[0]-b[2],b[0]-a[2],0);dy=max(a[1]-b[3],b[1]-a[3],0)
    return dx*dx+dy*dy


def tree(ss):
    bounds=box([p for s in ss for p in s])
    if len(ss)<=6:return (bounds,ss,None)
    axis=0 if bounds[2]-bounds[0]>=bounds[3]-bounds[1] else 1
    ss=sorted(ss,key=lambda s:s[0][axis]+s[1][axis]);mid=len(ss)//2
    return (bounds,None,(tree(ss[:mid]),tree(ss[mid:])))


def fast_gap(a,b):
    """Exact polyline distance via bounding-box branch and bound."""
    best=math.dist(a[0],b[0]);queue=[];serial=0
    def push(x,y):
        nonlocal serial
        lower=box_distance2(x[0],y[0])
        if lower<best*best:
            serial+=1;heapq.heappush(queue,(lower,serial,x,y))
    push(tree(segments(a,True)),tree(segments(b,True)))
    while queue:
        lower,_,x,y=heapq.heappop(queue)
        if lower>=best*best:continue
        if x[1] is not None and y[1] is not None:
            for p,q in x[1]:
                for r,s in y[1]:
                    best=min(best,segment_gap(p,q,r,s))
                    if best<1e-9:return 0.
        elif y[2] is None or (x[2] is not None and (x[0][2]-x[0][0])+(x[0][3]-x[0][1])>(y[0][2]-y[0][0])+(y[0][3]-y[0][1])):
            for child in x[2]:push(child,y)
        else:
            for child in y[2]:push(x,child)
    return best


def self_crossing(poly):
    ss=segments(poly,True);items=sorted((box([a,b]),i,a,b) for i,(a,b) in enumerate(ss));active=[]
    for bounds,i,a,b in items:
        active=[v for v in active if v[0][2]>=bounds[0]]
        for other,j,c,d in active:
            if abs(i-j)==1 or {i,j}=={0,len(ss)-1}:continue
            if box_distance2(bounds,other)==0 and crosses(a,b,c,d):return True
        active.append((bounds,i,a,b))
    return False


@functools.lru_cache(maxsize=1)
def anchor_case():
    c=geometry(ANCHOR_THETA,.05)
    inner=min(signed_wall_gaps(p)[0] for p in c['vanes'][0]['contour'])
    outer=min(signed_wall_gaps(p)[1] for p in c['vanes'][-1]['contour'])
    widths=[outer_normal_gap(c['vanes'][-1],j/200) for j in range(201)]
    return c,inner,outer,widths


def check_geometry(case):
    reasons=[];vs=case['vanes'];tol=case['sampling_tolerance_mm']
    inner=float('inf');outer=float('inf');gaps=[]
    for v in vs:
        for p in v['contour']:
            a,b=signed_wall_gaps(p);inner=min(inner,a);outer=min(outer,b)
            if not (0<=p[0]<=DRAWING_LIMIT_MM and -DRAWING_LIMIT_MM<=p[1]<=0):
                reasons.append(f'Vane {v["index"]} leaves the corner quadrant.');break
        if self_crossing(v['contour']):reasons.append(f'Vane {v["index"]} self-intersects.')
    for i,v in enumerate(vs):
        for w in vs[i+1:]:
            gap=fast_gap(v['contour'],w['contour'])
            if gap<=GEOMETRY_GUARD_MM+2*tol:reasons.append('Vanes intersect or are too close.')
            if inside(v['contour'][0],w['contour']) or inside(w['contour'][0],v['contour']):reasons.append('One vane contains another.')
            if w['index']==v['index']+1:gaps.append(gap)
    if min(inner,outer)<GEOMETRY_GUARD_MM+2*tol:reasons.append('Insufficient wall clearance.')
    _,ref_inner,ref_outer,ref_widths=anchor_case()
    widths=[outer_normal_gap(vs[-1],j/200) for j in range(201)]
    # Compare against the accepted 10-degree case, allowing sampling error only.
    if case['theta_deg']>=ANCHOR_THETA:
        if inner+tol+.05<ref_inner:reasons.append('Inner wall gap is below the 10-degree reference.')
        if outer+tol+.05<ref_outer:reasons.append('Outer wall gap is below the 10-degree reference.')
        if min(a-b for a,b in zip(widths,ref_widths)) < -1e-6:reasons.append('Outer normal passage narrows below its 10-degree reference.')
    # Matched-normal projection, not actual flow cross-sectional area.
    # The actual surface-to-surface minimum above is also reported.
    p=case['pitch_mm'];u=math.tan(math.radians(case['theta_deg']/2))
    mid_widths=[p*(1+(math.sqrt(2)-1)*u)-mesh_half_thickness(.5)*(a['thickness_scale']+b['thickness_scale']) for a,b in zip(vs,vs[1:])]
    return dict(valid=not reasons,reasons=sorted(set(reasons)),inner_gap_mm=inner,outer_gap_mm=outer,
                vane_gaps_mm=gaps,min_vane_gap_mm=min(gaps),reference_inner_gap_mm=ref_inner,
                reference_outer_gap_mm=ref_outer,outer_normal_widths_mm=widths,
                reference_outer_normal_widths_mm=ref_widths,
                mid_bend_projected_gaps_mm=mid_widths,
                endpoint_transverse_pitch_mm=p*Q*(1+u))


@functools.lru_cache(maxsize=64)
def make_case(theta,export_quality=False):
    started=time.perf_counter()
    result=geometry(theta,.05 if export_quality else .3)
    result['validation']=check_geometry(result)
    result['walls']=walls()
    result['compute_ms']=(time.perf_counter()-started)*1000
    return result


def export_case(theta,destination):
    case=make_case(theta,True)
    if not case['validation']['valid']:raise ValueError('; '.join(case['validation']['reasons']))
    folder=destination/('theta_'+f'{theta:.2f}'+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True)
    for v in case['vanes']:
        for key in ('upper','lower','trailing_edge','contour'):
            points=v[key]+([v[key][0]] if key=='contour' else [])
            with (folder/f'vane_{v["index"]:02d}_{key}_xyz_cm.csv').open('w',newline='') as f:
                csv.writer(f).writerows((f'{x/10:.8f}',f'{y/10:.8f}','0') for x,y in points)
    report=dict(case,units='geometry in mm; XYZ CSV files in cm',
                model='Circular 90-degree camber with bent NACA0009 thickness; capped outer thickness and finite flat TE.',
                layout_rule='At <=10 deg: original chord-locked layout. Above 10 deg: unchanged inner vane; p=p0*(1+tan(5 deg))/(1+tan(theta/2)); c_i=c_inner(10)+2*i*p*tan(theta/2). Midpoints move inward. Middle vane and midpoint pitch are no longer fixed.',
                limits='Wall model is the existing dimensioned drawing, not newly imported STEP. Sampled geometry checks do not certify CFD behavior. Normal widths and projected widths are geometry diagnostics, not flow areas. No constant-area or pressure-loss claim.')
    (folder/'geometry_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (folder/'README.txt').write_text('XYZ CSV coordinates are centimetres. Separate upper/lower curves and a finite flat trailing edge are provided. Join the two trailing-edge endpoints with a straight line; do not smooth a periodic spline across this face. Closed contour repeats its first point. Geometry checked at 0.05 mm sampling tolerance. Verify CAD dimensions/units before CFD.\n',encoding='utf-8')
    return folder


PAGE=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corner vanes — automatic clearance</title><style>
*{box-sizing:border-box}body{margin:0;font:15px system-ui;color:#153449;background:#f2f6f8}main{max-width:1450px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 8px}p{line-height:1.5}.controls,.status,.plot{background:white;border:1px solid #d8e3e9;border-radius:12px;padding:16px;margin:12px 0}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}input{flex:1;min-width:180px;accent-color:#02799a}button{padding:10px 16px;border:0;border-radius:7px;background:#086b86;color:white;cursor:pointer}button:disabled{opacity:.45;cursor:default}.status{white-space:pre-line;font-size:14px}.bad{border-color:#c63737;background:#fff1ef}.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(270px,1fr);gap:16px}canvas{width:100%;display:block}#geometry{height:600px}#width{height:220px}.note{color:#526d7d;font-size:13px}.stats{font-size:14px;line-height:1.8}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:7px 4px;border-bottom:1px solid #e5ecf0}@media(max-width:850px){.layout{grid-template-columns:1fr}#geometry{height:480px}}</style></head><body><main>
<h1>Corner vanes — automatic inward offsets</h1><p>One theta slider. Above 10°, the inner vane stays unchanged and the other vanes shift inward as their chords grow. Both wall clearances are checked against the 10° reference.</p>
<div class="controls"><label for="theta">Theta</label><input id="theta" type="range" min="0" max="50" step="0.1" value="20"><strong id="angle">20.0°</strong><button id="export" disabled>Export points</button></div>
<div id="status" class="status">Building geometry…</div><div class="layout"><div class="plot"><canvas id="geometry"></canvas><p class="note">Blue: leading-edge row. Red: trailing-edge centre row. Dashed: current chord lines. All dimensions use the original drawing.</p></div><div class="plot"><h3>Wall and vane clearances</h3><div id="stats" class="stats"></div><h3>Outer-wall passage through the bend</h3><canvas id="width"></canvas><p class="note">Blue: current normal distance to the outer wall. Grey: 10° reference. This is a geometric distance, not flow area or a velocity prediction.</p><div id="table"></div><p class="note">The midpoint spacing and middle vane are allowed to change above 10°. Inlet/outlet transverse spacing stays fixed. The passages are not constant-area; CFD is still needed to evaluate acceleration, separation and pressure loss.</p></div></div></main><script>
const $=id=>document.getElementById(id);let last=null,sequence=0,timer=null,busy=false,pending=null;
function context(id,height){const canvas=$(id),w=canvas.clientWidth,d=devicePixelRatio||1;canvas.width=w*d;canvas.height=height*d;const c=canvas.getContext('2d');c.scale(d,d);return[c,w,height]}
function render(){if(!last)return;const [c,w,h]=context('geometry',innerWidth<850?480:600),scale=Math.min((w-45)/11600,(h-35)/11600),xy=p=>[22+p[0]*scale,12-p[1]*scale];
function path(ps,color,dash=[],fill=false){c.beginPath();ps.forEach((p,i)=>i?c.lineTo(...xy(p)):c.moveTo(...xy(p)));if(fill){c.closePath();c.fillStyle='#d9f1f6';c.fill()}c.strokeStyle=color;c.lineWidth=fill?1.5:1;c.setLineDash(dash);c.stroke();c.setLineDash([])}
last.walls.forEach(ps=>path(ps,'#536b78'));for(const v of last.vanes){path(v.contour,last.validation.valid?'#006c92':'#bf2828',[],true);path([v.le,v.te],'#8aa5b3',[4,4]);let p=xy(v.camber_midpoint);c.fillStyle='#183849';c.fillText(v.index,p[0]-9,p[1]-7)}path(last.vanes.map(v=>v.le),'#2463dd',[5,4]);path(last.vanes.map(v=>v.te),'#d74d4d',[5,4]);
const [g,gw,gh]=context('width',220),a=last.validation.outer_normal_widths_mm,b=last.validation.reference_outer_normal_widths_mm,max=Math.max(...a,...b)*1.15;g.font='12px system-ui';g.fillStyle='#526d7d';g.fillText('Gap (mm)',7,15);g.fillText('Inlet',40,210);g.fillText('Mid-bend',gw/2-25,210);g.fillText('Outlet',gw-46,210);
for(let k=0;k<4;k++){let y=30+k*50,value=max*(1-k/3);g.fillText(value.toFixed(0),1,y+4);g.beginPath();g.moveTo(42,y);g.lineTo(gw-10,y);g.strokeStyle='#e1e9ed';g.stroke()}
for(const [arr,color] of [[b,'#9caab3'],[a,'#02799a']]){g.beginPath();arr.forEach((v,i)=>{let x=42+i/(arr.length-1)*(gw-52),y=180-150*v/max;i?g.lineTo(x,y):g.moveTo(x,y)});g.strokeStyle=color;g.lineWidth=2;g.stroke()}}
async function request(theta,id){busy=true;try{const r=await fetch('/case?theta='+theta),data=await r.json();if(!r.ok)throw Error(data.error);if(id!==sequence)return;last=data;const v=data.validation;$('status').className='status'+(v.valid?'':' bad');$('status').textContent=v.valid?`Theta ${data.measured_theta_deg.toFixed(2)}° · geometry checks passed · ${Math.round(data.compute_ms)} ms calculation\n${theta>10?'Automatic inward offsets active; inner vane unchanged from 10°.':'Original layout up to 10°.'}`:v.reasons.join('\n');$('export').disabled=!v.valid;
$('stats').innerHTML=`Inner wall gap: <b>${v.inner_gap_mm.toFixed(1)} mm</b><br>Outer wall gap: <b>${v.outer_gap_mm.toFixed(1)} mm</b> (10°: ${v.reference_outer_gap_mm.toFixed(1)})<br>Smallest vane-to-vane gap: <b>${v.min_vane_gap_mm.toFixed(1)} mm</b><br>Midpoint pitch: ${data.pitch_mm.toFixed(1)} mm<br>Inlet/outlet transverse pitch: ${v.endpoint_transverse_pitch_mm.toFixed(1)} mm`;
$('table').innerHTML='<table><tr><th>Vane</th><th>Chord (cm)</th><th>Inward shift (mm)</th></tr>'+data.vanes.map(v=>`<tr><td>${v.index}</td><td>${(v.chord_mm/10).toFixed(1)}</td><td>${v.inward_offset_mm.toFixed(1)}</td></tr>`).join('')+'</table>';render();
}catch(e){if(id===sequence){last=null;$('status').textContent=e.message;$('status').className='status bad';$('export').disabled=true;for(const name of ['geometry','width']){const cv=$(name);cv.getContext('2d').clearRect(0,0,cv.width,cv.height)}$('stats').textContent='';$('table').textContent=''}}finally{busy=false;if(pending){const q=pending;pending=null;request(...q)}}}
function schedule(){const id=++sequence,theta=Number($('theta').value);$('angle').textContent=theta.toFixed(1)+'°';$('export').disabled=true;$('status').textContent='Updating geometry…';clearTimeout(timer);pending=null;timer=setTimeout(()=>{if(busy)pending=[theta,id];else request(theta,id)},70)}
$('theta').addEventListener('input',schedule);$('export').onclick=async()=>{if(!last)return;const theta=last.theta_deg,id=sequence;$('export').disabled=true;try{const r=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theta})}),d=await r.json();if(!r.ok)throw Error(d.error);if(id===sequence)$('status').textContent+='\nSaved: '+d.folder}catch(e){if(id===sequence)$('status').textContent+='\n'+e.message}finally{if(id===sequence)$('export').disabled=!last?.validation.valid}};window.addEventListener('resize',render);schedule();
</script></body></html>'''


def serve(open_browser=True):
    import http.server,urllib.parse,webbrowser
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data,status=200,html=False):
            body=(data if html else json.dumps(data,allow_nan=False)).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json');self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            url=urllib.parse.urlparse(self.path)
            if url.path=='/':return self.reply(PAGE,html=True)
            if url.path!='/case':return self.reply({'error':'Not found'},404)
            try:self.reply(make_case(float(urllib.parse.parse_qs(url.query)['theta'][0])))
            except (ValueError,KeyError,IndexError,OverflowError) as exc:self.reply({'error':str(exc)},400)
        def do_POST(self):
            if self.path!='/export':return self.reply({'error':'Not found'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<2048:raise ValueError('Invalid request size.')
                data=json.loads(self.rfile.read(size));theta=float(data['theta'])
                self.reply({'folder':str(export_case(theta,Path(__file__).resolve().parent/'clearance_exports'))})
            except (ValueError,TypeError,KeyError,OSError,OverflowError) as exc:self.reply({'error':str(exc)},400)
    with http.server.HTTPServer(('127.0.0.1',0),Handler) as server:
        url=f'http://127.0.0.1:{server.server_port}/';print('Open '+url,flush=True)
        if open_browser:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theta',type=float,help='Check a geometry without opening the browser.')
    parser.add_argument('--export',type=Path,help='Export directory; use with --theta.')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if args.theta is None:serve(not args.no_browser)
    else:
        c=make_case(args.theta);v=c['validation']
        print(json.dumps(dict(theta=c['measured_theta_deg'],valid=v['valid'],reasons=v['reasons'],pitch_mm=c['pitch_mm'],inner_gap_mm=v['inner_gap_mm'],outer_gap_mm=v['outer_gap_mm'],vane_gap_mm=v['min_vane_gap_mm'],compute_ms=c['compute_ms']),indent=2))
        if args.export:print(export_case(args.theta,args.export))
        if not v['valid']:raise SystemExit(2)
