"""Weighted spacing study for five circular-camber corner vanes.

Run without arguments for the browser. Requires NumPy and SciPy.
The 0-degree source layout is fixed. Above zero, fit first midpoint station
and common pitch, with chord c_i=c0+2*p*(i+b)*tan(theta/2).
Adjacent LE/TE steps are p*(n-k*u) and p*(n+k*u), with n perpendicular
to u and k=tan(theta/2), so their angle is exactly 2*atan(k)=theta.
Thus LE/TE rows remain straight, enclose exactly theta, and every chord
extends equally at LE and TE relative to its current midpoint. Normal
thickness remains at the source value (including the 50 mm TE gap).

J = .45*mean((A/mean(A)-1)^2)
  + .45*mean_s,j((g_j(s)/mean_j(g(s))-1)^2)
  + .10*mean_end,j((e_j/mean_j(e)-1)^2)
  + .005*((station_shift/1000)^2+(pitch/base_pitch-1)^2).
Areas are enclosed planform areas, not flow cross-sectional areas.
Local normal widths use ray/surface intersections at 15..85% camber
progress, averaging both directions when available between vanes.
End openings are measured separately. This is a geometric design study,
not a flow prediction; equal areas may compete with equal widths.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import heapq
import io
import json
import math
from pathlib import Path

BASELINE_CSV = """341.1790218963,1.7677669596,0
339.6831916784,3.5913521193,0
335.1053138117,8.9251495064,0
327.1756879034,17.3272605021,0
315.5134893506,28.0216913002,0
299.8332718842,40.0068822784,0
280.1678389523,52.3290531614,0
256.8356113068,64.2102270454,0
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
250.1224118932,46.1874557247,0
273.0459384467,38.0136696369,0
292.6577633153,28.4428989010,0
308.7777017501,18.8874155603,0
321.3847486963,10.4491524578,0
330.4226653884,3.8679142605,0
335.8415463217,-0.3253149734,0
337.6434880036,-1.7677669595,0"""
PITCH_MM = 2070.0
PAPER_RADIUS_MM = 2400.0
PAPER_CHORD_MM = PAPER_RADIUS_MM * math.sqrt(2)
FIRST_STATION_MM = 3693.83
NUMBER_OF_VANES = 5
MAX_THETA_DEG = 50.0
Q = 1 / math.sqrt(2)
CHORD_AXIS = (Q, Q)
MIDPOINT_AXIS = (Q, -Q)


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def mul(a: tuple[float, float], factor: float) -> tuple[float, float]:
    return a[0] * factor, a[1] * factor


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return math.degrees(math.atan2(abs(cross), dot))


def angle_from_down_vertical(vector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(vector[0], -vector[1]))


def source() -> tuple[list[tuple[float, float]], float, float]:
    rows = list(csv.reader(io.StringIO(BASELINE_CSV)))
    if len(rows) != 45 or any(len(row) != 3 for row in rows):
        raise ValueError("The embedded source must contain 45 XYZ rows")
    points = [(float(x) * 10, float(y) * 10) for x, y, z in rows]
    if any(float(row[2]) != 0 for row in rows):
        raise ValueError("Source z coordinates must be zero")
    if math.dist(points[22], (0, 0)) > 1e-7:
        raise ValueError("Source leading edge must be row 23 at the origin")
    te_mid = mul(add(points[0], points[-1]), 0.5)
    if abs(te_mid[1]) > 1e-6:
        raise ValueError("Source TE midpoint must be on the chord axis")
    return points, te_mid[0], math.dist(points[0], points[-1])


def paper_camber(t: float) -> tuple[float, float]:
    phi = math.pi * t / 2
    sine, cosine = math.sin(phi), math.cos(phi)
    return (PAPER_CHORD_MM / 2 * (1 + sine - cosine),
            PAPER_CHORD_MM / 2 * (sine + cosine - 1))


def paper_normal(t: float) -> tuple[float, float]:
    phi = math.pi * t / 2
    sine, cosine = math.sin(phi), math.cos(phi)
    return ((sine - cosine) * Q, (sine + cosine) * Q)


def original_half_thickness(t: float) -> float:
    arc = math.pi * PAPER_RADIUS_MM / 2
    return 5 * 0.09 * arc * (
        0.2969 * math.sqrt(t) - 0.1260 * t - 0.3516 * t * t
        + 0.2843 * t**3 - 0.1015 * t**4
    )


def half_thickness(t: float) -> float:
    if t <= 0.70:
        return original_half_thickness(t)
    fraction = min(1.0, max(0.0, (t - 0.70) / 0.30))
    weight = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
    return original_half_thickness(t) + (25 - original_half_thickness(1)) * weight


def paper_surface(t: float, sign: int) -> tuple[float, float]:
    return add(paper_camber(t), mul(paper_normal(t), sign * half_thickness(t)))


@functools.lru_cache(maxsize=1)
def source_stations() -> tuple[tuple[float, int], ...]:
    points, _, _ = source()
    stations = []
    for index, point in enumerate(points):
        sign = 1 if index <= 22 else -1
        def error(t: float) -> float:
            q = paper_surface(t, sign)
            return (point[0] - q[0])**2 + (point[1] - q[1])**2
        count = 400
        best = min(range(count + 1), key=lambda j: error(j / count))
        low, high = max(0, (best - 1) / count), min(1, (best + 1) / count)
        for _ in range(55):
            left = low + (high - low) * 0.381966011250105
            right = high - (high - low) * 0.381966011250105
            if error(left) < error(right):
                high = right
            else:
                low = left
        t = 0.0 if index == 22 else 1.0 if index in (0, 44) else (low + high) / 2
        if math.sqrt(error(t)) > 0.02:
            raise ValueError('Embedded source differs from the analytic profile')
        stations.append((t, sign))
    return tuple(stations)


@functools.lru_cache(maxsize=1)
def original_max_normal_thickness() -> float:
    return 2 * max(half_thickness(j / 2000) for j in range(2001))


def midpoint_pitch(theta_deg: float) -> float:
    """Apply the second code's pitch law from 0°, without its 10° hold."""
    k = math.tan(math.radians(theta_deg / 2))
    return PITCH_MM / (1 + k)


def profile_point(le: tuple[float, float], scale: float,
                  t: float, sign: int) -> tuple[float, float]:
    camber = mul(paper_camber(t), scale)
    normal = mul(paper_normal(t), sign * half_thickness(t))
    x, y = add(camber, normal)
    return le[0] + Q * (x + y), le[1] + Q * (x - y)


def signed_wall_gaps(p: tuple[float, float]) -> tuple[float, float]:
    x, y = p
    limit, radius = 11244.4, PAPER_RADIUS_MM
    outer_center = (limit - radius, -limit + radius)
    inner = math.hypot(x, y) - radius
    outer = (radius - math.hypot(x - outer_center[0], y - outer_center[1])
             if x > outer_center[0] and y < outer_center[1]
             else min(limit - x, limit + y))
    return inner, outer


def sampled_wall_clearance(vanes: list[dict]) -> tuple[float, bool, float, float]:
    """Dense sampled clearance against the dimensioned XY wall model."""
    limit = 11244.4
    minimum = math.inf
    first_inner = math.inf
    last_outer = math.inf
    inside = True
    for index, vane in enumerate(vanes):
        for j in range(501):
            t = j / 500
            for sign in (-1, 1):
                x, y = profile_point(vane['le'], vane['scale'], t, sign)
                inner, outer = signed_wall_gaps((x, y))
                minimum = min(minimum, inner, outer)
                if index == 0:
                    first_inner = min(first_inner, inner)
                if index == len(vanes) - 1:
                    last_outer = min(last_outer, outer)
                if not (0 <= x <= limit and -limit <= y <= 0):
                    inside = False
    return minimum, inside and minimum >= 0, first_inner, last_outer


def outer_normal_gap(vane: dict, t: float) -> float:
    """Distance from outer surface along the camber normal to the outer wall."""
    x, y = profile_point(vane['le'], vane['scale'], t, 1)
    phi = math.pi * t / 2
    nx, ny = math.sin(phi), -math.cos(phi)
    limit, radius = 11244.4, PAPER_RADIUS_MM
    tangent = limit - radius
    candidates = []
    if nx > 1e-12:
        distance = (limit - x) / nx
        if distance >= 0 and y + distance * ny >= -tangent - 1e-7:
            candidates.append(distance)
    if ny < -1e-12:
        distance = (-limit - y) / ny
        if distance >= 0 and x + distance * nx <= tangent + 1e-7:
            candidates.append(distance)
    ox, oy = x - tangent, y + tangent
    projection = ox * nx + oy * ny
    discriminant = projection**2 + radius**2 - ox**2 - oy**2
    if discriminant >= 0:
        for distance in (-projection - math.sqrt(discriminant),
                         -projection + math.sqrt(discriminant)):
            if distance >= 0 and x + distance * nx >= tangent - 1e-7 and y + distance * ny <= -tangent + 1e-7:
                candidates.append(distance)
    if not candidates:
        raise ValueError('No outer wall intersection along camber normal')
    return min(candidates)


def projected_mid_bend_gaps(vanes: list[dict]) -> list[float]:
    normal = MIDPOINT_AXIS
    gaps = []
    for inner, outer in zip(vanes, vanes[1:]):
        inner_surface = profile_point(inner['le'], inner['scale'], 0.5, 1)
        outer_surface = profile_point(outer['le'], outer['scale'], 0.5, -1)
        delta = sub(outer_surface, inner_surface)
        gaps.append(delta[0] * normal[0] + delta[1] * normal[1])
    return gaps


def geometry(theta_deg: float, common_growth: float = 1.0,
             first_station_mm: float = FIRST_STATION_MM,
             fitted_pitch_mm: float | None = None) -> dict:
    if not math.isfinite(theta_deg) or not 0 <= theta_deg <= MAX_THETA_DEG:
        raise ValueError("theta must be between 0 and 50 degrees")
    if not math.isfinite(common_growth) or common_growth <= 0:
        raise ValueError("common growth must be a positive number")
    if not math.isfinite(first_station_mm) or first_station_mm <= 0:
        raise ValueError("first station must be positive")
    points, chord0, te_gap0 = source()
    stations = source_stations()
    k = math.tan(math.radians(theta_deg / 2))
    pitch = midpoint_pitch(theta_deg) if fitted_pitch_mm is None else fitted_pitch_mm
    vanes = []
    for i in range(NUMBER_OF_VANES):
        midpoint = mul(MIDPOINT_AXIS, first_station_mm + i * pitch)
        chord = chord0 + 2 * pitch * (i + common_growth) * k
        scale = chord / chord0
        le = sub(midpoint, mul(CHORD_AXIS, chord / 2))
        te = add(midpoint, mul(CHORD_AXIS, chord / 2))
        # Scale the circular camber, but keep normal thickness at its source value.
        contour = [profile_point(le, scale, t, sign) for t, sign in stations]
        vanes.append(dict(index=i + 1, midpoint=midpoint, le=le, te=te,
                          chord_mm=chord, scale=scale, thickness_scale=1.0,
                          max_normal_thickness_mm=original_max_normal_thickness(),
                          te_gap_mm=te_gap0,
                          le_growth_mm=(chord - chord0) / 2,
                          te_growth_mm=(chord - chord0) / 2,
                          contour=contour))

    le_step = sub(vanes[1]["le"], vanes[0]["le"])
    te_step = sub(vanes[1]["te"], vanes[0]["te"])
    measured = angle_between(le_step, te_step)
    if abs(measured - theta_deg) > 1e-9:
        raise AssertionError("LE/TE row angle does not equal theta")
    for i, vane in enumerate(vanes):
        if math.dist(vane["contour"][22], vane["le"]) > 1e-6:
            raise AssertionError("Scaled source LE misses its target")
        actual_te_mid = mul(add(vane["contour"][0], vane["contour"][-1]), 0.5)
        if math.dist(actual_te_mid, vane["te"]) > 1e-6:
            raise AssertionError("Scaled source TE midpoint misses its target")
        if i:
            for key, step in (("le", le_step), ("te", te_step)):
                expected = add(vanes[0][key], mul(step, i))
                if math.dist(vane[key], expected) > 1e-6:
                    raise AssertionError(f"{key.upper()} points are not on one line")
    minimum_wall_gap, inside_walls, inner_wall_gap, outer_wall_gap = sampled_wall_clearance(vanes)
    outer_normal_widths = ([outer_normal_gap(vanes[-1], j / 200) for j in range(201)]
                           if inside_walls else None)
    return dict(theta_deg=theta_deg, measured_theta_deg=measured,
                common_growth=common_growth,
                first_station_mm=first_station_mm,
                le_line_angle_deg=angle_from_down_vertical(le_step),
                te_line_angle_deg=angle_from_down_vertical(te_step),
                midpoint_pitch_mm=pitch, source_chord_mm=chord0,
                source_te_gap_mm=te_gap0,
                sampled_min_wall_gap_mm=minimum_wall_gap,
                inner_wall_gap_mm=inner_wall_gap,
                outer_wall_gap_mm=outer_wall_gap,
                outer_normal_widths_mm=outer_normal_widths,
                mid_bend_projected_gaps_mm=projected_mid_bend_gaps(vanes),
                endpoint_transverse_pitch_mm=pitch * Q * (1 + k),
                inside_walls=inside_walls,
                vane_clearance_checked=False, vanes=vanes)


def export_csv(case: dict, path: Path) -> None:
    if case.get('spacing_fit') and not case['spacing_fit']['fit_applied']:
        raise ValueError('Export blocked: spacing proposal widens the outer-wall passage')
    if not case['inside_walls']:
        raise ValueError('Export blocked: sampled outline crosses a duct wall')
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for i, vane in enumerate(case["vanes"]):
            if i:
                writer.writerow([])
            for x, y in vane["contour"]:
                writer.writerow((f"{x / 10:.8f}", f"{y / 10:.8f}", "0"))


def point_segment_distance(point: tuple[float, float], a: tuple[float, float],
                           b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    fraction = (max(0.0, min(1.0, ((point[0] - a[0]) * dx +
                                    (point[1] - a[1]) * dy) / length2))
                if length2 else 0.0)
    return math.dist(point, (a[0] + fraction * dx, a[1] + fraction * dy))


def export_side(vane: dict, sign: int, tolerance: float = 0.05) -> list[tuple[float, float]]:
    """Adaptive profile sampling, matching the second code's export tolerance."""
    def at(u: float) -> tuple[float, float]:
        return profile_point(vane['le'], vane['scale'], u * u, sign)
    def refine(a: float, b: float, pa: tuple[float, float],
               pb: tuple[float, float], depth: int = 0) -> list[tuple[float, float]]:
        deviation = max(point_segment_distance(at(a + (b - a) * fraction), pa, pb)
                        for fraction in (0.25, 0.5, 0.75))
        if deviation <= tolerance:
            return [pa, pb]
        if depth >= 22:
            raise ValueError('Export contour sampling did not converge')
        mid = (a + b) / 2
        pm = at(mid)
        return refine(a, mid, pa, pm, depth + 1)[:-1] + refine(mid, b, pm, pb, depth + 1)
    seeds = sorted({j / 24 for j in range(25)} | {math.sqrt(0.70)})
    result = [at(seeds[0])]
    for a, b in zip(seeds, seeds[1:]):
        result.extend(refine(a, b, at(a), at(b))[1:])
    return result


def fine_contour(vane: dict, tolerance: float = 0.3) -> list[tuple[float, float]]:
    upper = list(reversed(export_side(vane, 1, tolerance)))
    lower = export_side(vane, -1, tolerance)
    return upper + lower[1:]


def cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def segments(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(points, points[1:] + points[:1]))


def on_segment(p: tuple[float, float], a: tuple[float, float],
               b: tuple[float, float]) -> bool:
    return abs(cross(sub(b, a), sub(p, a))) < 1e-8 and dot(sub(p, a), sub(p, b)) <= 1e-8


def crosses(a: tuple[float, float], b: tuple[float, float],
            c: tuple[float, float], d: tuple[float, float]) -> bool:
    ab1, ab2 = cross(sub(b, a), sub(c, a)), cross(sub(b, a), sub(d, a))
    cd1, cd2 = cross(sub(d, c), sub(a, c)), cross(sub(d, c), sub(b, c))
    if ab1 * ab2 < -1e-8 and cd1 * cd2 < -1e-8:
        return True
    return ((abs(ab1) < 1e-8 and on_segment(c, a, b)) or
            (abs(ab2) < 1e-8 and on_segment(d, a, b)) or
            (abs(cd1) < 1e-8 and on_segment(a, c, d)) or
            (abs(cd2) < 1e-8 and on_segment(b, c, d)))


def segment_gap(a: tuple[float, float], b: tuple[float, float],
                c: tuple[float, float], d: tuple[float, float]) -> float:
    if crosses(a, b, c, d):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def box(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def box_distance2(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return dx * dx + dy * dy


def tree(items: list[tuple[tuple[float, float], tuple[float, float]]]):
    bounds = box([point for segment in items for point in segment])
    if len(items) <= 6:
        return bounds, items, None
    axis = 0 if bounds[2] - bounds[0] >= bounds[3] - bounds[1] else 1
    items = sorted(items, key=lambda item: item[0][axis] + item[1][axis])
    middle = len(items) // 2
    return bounds, None, (tree(items[:middle]), tree(items[middle:]))


def fast_gap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Minimum full-outline segment distance, with bounding-box pruning."""
    best = math.dist(a[0], b[0])
    queue = []
    serial = 0
    def push(x, y) -> None:
        nonlocal serial
        lower = box_distance2(x[0], y[0])
        if lower < best * best:
            serial += 1
            heapq.heappush(queue, (lower, serial, x, y))
    push(tree(segments(a)), tree(segments(b)))
    while queue:
        lower, _, x, y = heapq.heappop(queue)
        if lower >= best * best:
            continue
        if x[1] is not None and y[1] is not None:
            for p, q in x[1]:
                for r, s in y[1]:
                    best = min(best, segment_gap(p, q, r, s))
                    if best < 1e-9:
                        return 0.0
        elif y[2] is None or (x[2] is not None and
              (x[0][2] - x[0][0]) + (x[0][3] - x[0][1]) >
              (y[0][2] - y[0][0]) + (y[0][3] - y[0][1])):
            for child in x[2]:
                push(child, y)
        else:
            for child in y[2]:
                push(x, child)
    return best


def inside(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    flag = False
    for a, b in segments(polygon):
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < x:
                flag = not flag
    return flag


def adjacent_surface_gaps(vanes: list[dict]) -> list[float]:
    outlines = [fine_contour(vane) for vane in vanes]
    gaps = []
    for a, b in zip(outlines, outlines[1:]):
        gap = fast_gap(a, b)
        if inside(a[0], b) or inside(b[0], a):
            gap = 0.0
        gaps.append(gap)
    return gaps


def midline_wall_gap(point: tuple[float, float], outer: bool) -> float:
    """Exact first wall hit from a point parallel to the corner bisector."""
    direction = MIDPOINT_AXIS if outer else mul(MIDPOINT_AXIS, -1)
    limit, radius = 11244.4, PAPER_RADIUS_MM
    candidates = []
    def straight_x(x: float, y0: float, y1: float) -> None:
        distance = (x - point[0]) / direction[0]
        y = point[1] + distance * direction[1]
        if distance >= 0 and y0 - 1e-7 <= y <= y1 + 1e-7:
            candidates.append(distance)
    def straight_y(y: float, x0: float, x1: float) -> None:
        distance = (y - point[1]) / direction[1]
        x = point[0] + distance * direction[0]
        if distance >= 0 and x0 - 1e-7 <= x <= x1 + 1e-7:
            candidates.append(distance)
    center = (limit - radius, -limit + radius) if outer else (0.0, 0.0)
    relative = sub(point, center)
    projection = dot(relative, direction)
    discriminant = projection**2 - dot(relative, relative) + radius**2
    if discriminant >= 0:
        root = math.sqrt(discriminant)
        for distance in (-projection - root, -projection + root):
            foot = add(point, mul(direction, distance))
            radial = sub(foot, center)
            if distance >= 0 and radial[0] >= -1e-7 and radial[1] <= 1e-7:
                candidates.append(distance)
    if outer:
        straight_x(limit, -limit + radius, 4000)
        straight_y(-limit, -4000, limit - radius)
    else:
        straight_x(radius, 0, 4000)
        straight_y(-radius, -4000, 0)
    if not candidates:
        raise ValueError('Midline ray does not meet the designated wall')
    return min(candidates)


# Dimensionless design weights; geometry alone does not predict flow uniformity.
FIT_WEIGHTS = dict(area=0.45, normal=0.45, opening=0.10)


def passage_metrics(theta, growth, station, pitch, samples=121):
    """Closed 2-D areas between extended LE/TE rows; true surface-ray widths.

    Width stations follow equal camber progress (15..85%); each ray follows
    the local camber normal and intersects the opposing surface. These are
    local channel sections, not assumed CFD streamlines. End openings are
    measured separately along the horizontal/vertical camber normals.
    """
    import numpy as np
    k = math.tan(math.radians(theta / 2))
    c0 = source()[1]
    t = np.linspace(0, 1, samples)
    phi = t * math.pi / 2
    normals = np.column_stack((np.sin(phi), -np.cos(phi)))
    h = np.array([half_thickness(float(v)) for v in t])
    sides = []
    les, tes = [], []
    for i in range(5):
        chord = c0 + 2 * pitch * (i + growth) * k
        mid = np.array(MIDPOINT_AXIS) * (station + i * pitch)
        le = mid - np.array(CHORD_AXIS) * chord / 2
        te = mid + np.array(CHORD_AXIS) * chord / 2
        # Circular camber in global coordinates, with unchanged thickness.
        radius = chord / math.sqrt(2)
        camber = le + np.column_stack((radius*np.sin(phi), radius*(1-np.cos(phi))))
        sides.append((camber - h[:, None]*normals, camber + h[:, None]*normals))
        les.append(le); tes.append(te)
    r, lim = PAPER_RADIUS_MM, 11244.4
    wall_phi = np.linspace(0, math.pi/2, 1001)
    inner = np.column_stack((r*np.sin(wall_phi), -r*np.cos(wall_phi)))
    outer_arc = np.column_stack((lim-r+r*np.sin(wall_phi), -lim+r-r*np.cos(wall_phi)))
    outer = np.vstack(([0., -lim], outer_arc, [lim, 0.]))

    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]

    def hit(origin, direction, curve):
        edges = np.diff(curve, axis=0)
        delta = curve[:-1]-origin
        den = cross(direction, edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = cross(delta, edges)/den
            fraction = cross(delta, direction)/den
        valid = (abs(den)>1e-10)&(distance>=-1e-7)&(fraction>=-1e-8)&(fraction<=1+1e-8)
        if not np.any(valid):
            raise ValueError('Normal or edge row does not intersect its passage boundary')
        idx = int(np.argmin(np.where(valid, distance, np.inf)))
        return float(distance[idx]), origin + distance[idx]*direction, idx

    def wall_section(curve, start, end, direction_start, direction_end):
        _, p0, i0 = hit(start, direction_start, curve)
        _, p1, i1 = hit(end, direction_end, curve)
        if i1 < i0:
            raise ValueError('Wall section is reversed')
        return np.vstack((p0, curve[i0+1:i1+1], p1))

    step_le = (les[1]-les[0]); step_le /= np.linalg.norm(step_le)
    step_te = (tes[1]-tes[0]); step_te /= np.linalg.norm(step_te)
    wall_inner = wall_section(inner, les[0], tes[0], -step_le, -step_te)
    wall_outer = wall_section(outer, les[-1], tes[-1], step_le, step_te)
    # LE has zero thickness. TE boundary includes its physical half thickness.
    boundaries = [(wall_inner, sides[0][0])]
    boundaries += [(sides[i][1], sides[i+1][0]) for i in range(4)]
    boundaries += [(sides[-1][1], wall_outer)]
    areas = []
    for low, high in boundaries:
        polygon = np.vstack((low, high[::-1]))
        areas.append(abs(float(np.sum(cross(polygon, np.roll(polygon, -1, axis=0)))))/2)
    indices = np.linspace(int(.15*(samples-1)), int(.85*(samples-1)), 19).astype(int)
    widths = []
    openings = []
    for j in range(6):
        if j == 0:
            origins, target, sign = sides[0][0], inner, -1
        elif j == 5:
            origins, target, sign = sides[-1][1], outer, 1
        else:
            origins, target, sign = sides[j-1][1], sides[j][0], 1
        values = []
        for idx in indices:
            distances = []
            try:
                distances.append(hit(origins[idx], sign*normals[idx], target)[0])
            except ValueError:
                pass
            if 0 < j < 5:
                try:
                    distances.append(hit(sides[j][0][idx], -normals[idx], sides[j-1][1])[0])
                except ValueError:
                    pass
            if not distances:
                raise ValueError(f'No passage normal at channel {j}, station {idx}')
            values.append(sum(distances)/len(distances))
        widths.append(values)
        # End planes use directional wall distances or adjacent edge projection.
        if j in (0, 5):
            openings.append([hit(origins[idx], sign*normals[idx], target)[0] for idx in (0, samples-1)])
        else:
            openings.append([float(np.dot(sides[j][0][idx]-origins[idx], normals[idx])) for idx in (0, samples-1)])
    widths, areas, openings = np.array(widths), np.array(areas), np.array(openings)
    if min(float(widths.min()), float(openings.min())) <= 1:
        raise ValueError('Passage closes')
    # Also reject a wall crossing anywhere on the sampled profile.
    clearance = min(min(signed_wall_gaps(tuple(point))) for side in sides for curve in side for point in curve)
    if clearance <= 1 or any(np.any(curve[:,0]<0) or np.any(curve[:,1]>0) for side in sides for curve in side):
        raise ValueError('Outline crosses wall or domain boundary')
    area_error = float(np.mean((areas/areas.mean()-1)**2))
    normal_error = float(np.mean((widths/widths.mean(axis=0)-1)**2))
    opening_error = float(np.mean((openings/openings.mean(axis=0)-1)**2))
    terms = dict(area=area_error, normal=normal_error, opening=opening_error)
    score = sum(FIT_WEIGHTS[key]*value for key, value in terms.items())
    return dict(areas_mm2=areas.tolist(), normal_widths_mm=widths.tolist(),
                openings_mm=openings.tolist(), terms=terms, score=score,
                min_sampled_wall_clearance_mm=float(clearance),
                station_progress=(indices/(samples-1)).tolist())


@functools.lru_cache(maxsize=128)
def spaced_case(theta_deg: float, common_growth: float = 1.0) -> dict:
    from scipy.optimize import minimize
    base = geometry(theta_deg, common_growth)
    p0 = base['midpoint_pitch_mm']
    baseline = passage_metrics(theta_deg, common_growth, FIRST_STATION_MM, p0, 401)
    # Zero-degree starting geometry is fixed; search freedom opens continuously.
    freedom = math.sin(math.radians(theta_deg * 90 / 50))
    bounds = [(-2*freedom, 2*freedom), (1-.35*freedom, 1+.35*freedom)]
    def evaluate(x):
        try:
            m = passage_metrics(theta_deg, common_growth, FIRST_STATION_MM+1000*x[0], p0*x[1])
            # Small preference for retaining the existing layout among similar fits.
            return m['score'] + .005*(x[0]**2+(x[1]-1)**2)
        except ValueError:
            return 1000 + float(x[0]**2+(x[1]-1)**2)
    best_x, best_score = [0., 1.], evaluate([0., 1.])
    if theta_deg > 0:
        for start in ([0., 1.], [.5*freedom, 1+.15*freedom], [-.5*freedom, 1+.2*freedom]):
            result = minimize(evaluate, start, method='Powell', bounds=bounds,
                              options=dict(xtol=1e-4, ftol=1e-6, maxiter=45))
            if result.fun < best_score:
                best_x, best_score = result.x, float(result.fun)
    case = geometry(theta_deg, common_growth, float(FIRST_STATION_MM+1000*best_x[0]), float(p0*best_x[1]))
    metrics = passage_metrics(theta_deg, common_growth, case['first_station_mm'], case['midpoint_pitch_mm'], 401)
    adjacent = adjacent_surface_gaps(case['vanes'])
    if not case['inside_walls'] or min(adjacent) <= 1:
        raise ValueError('Fitted geometry failed the refined clearance check')
    six = [case['inner_wall_gap_mm'], *adjacent, case['outer_wall_gap_mm']]
    base_adjacent = adjacent_surface_gaps(base['vanes'])
    case['comparison_vanes'] = base['vanes']
    case['spacing_fit'] = dict(
        fit_applied=True, weights=FIT_WEIGHTS, baseline_metrics=baseline, metrics=metrics,
        regularization_weight=.005, starting_first_station_mm=FIRST_STATION_MM,
        fitted_first_station_mm=case['first_station_mm'],
        station_shift_mm=case['first_station_mm']-FIRST_STATION_MM,
        starting_pitch_mm=p0, score=metrics['score'], starting_score=baseline['score'],
        full_six_gaps_mm=six, starting_full_six_gaps_mm=[base['inner_wall_gap_mm'], *base_adjacent, base['outer_wall_gap_mm']],
        six_gap_unevenness_mm=max(six)-min(six), spacing_equal_within_1mm=max(six)-min(six)<=1,
        method='45% normalized area variance + 45% local normal-width variance + 10% end-opening variance; first station and pitch fitted, theta preserved.',
        area_definition='Planform areas bounded by actual surfaces, duct walls and end connectors to extended LE/TE-row wall intersections. These are not flow cross-sectional areas.',
        normal_definition='Surface intersections along local camber normals at 19 progress stations from 15% to 85%; bidirectional means where available; endpoint openings measured separately.',
        search_bounds=bounds, search_coordinates='station shift in metres; pitch factor',
        original_outer_midbend_gap_mm=base['outer_normal_widths_mm'][100],
        proposed_outer_midbend_gap_mm=case['outer_normal_widths_mm'][100])
    case['vane_clearance_checked'] = True
    return case


def export_case(case: dict, destination: Path) -> Path:
    """Write the same per-vane CSV and report layout as the second code."""
    if not case['spacing_fit']['fit_applied']:
        raise ValueError('Export blocked: spacing proposal widens the outer-wall passage')
    if not case['inside_walls']:
        raise ValueError('Export blocked: sampled outline crosses a duct wall')
    theta = case['theta_deg']
    folder = destination / ('theta_' + f'{theta:.2f}' + '_'
                            + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True)
    report_vanes = []
    for vane in case['vanes']:
        upper = list(reversed(export_side(vane, 1)))
        lower = export_side(vane, -1)
        trailing_edge = [add(lower[-1], mul(sub(upper[0], lower[-1]), j / 8))
                         for j in range(9)]
        contour = upper + lower[1:] + trailing_edge[1:-1]
        curves = dict(upper=upper, lower=lower,
                      trailing_edge=trailing_edge, contour=contour)
        for key, points in curves.items():
            rows = points + ([points[0]] if key == 'contour' else [])
            with (folder / f'vane_{vane["index"]:02d}_{key}_xyz_cm.csv').open(
                    'w', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerows((f'{x / 10:.8f}', f'{y / 10:.8f}', '0')
                                             for x, y in rows)
        report_vanes.append({**vane, **curves})
    report = {**case, 'vanes': report_vanes,
              'units': 'geometry in mm; XYZ CSV files in cm',
              'model': 'Circular 90-degree camber with fixed original normal thickness and a 50 mm flat TE gap.',
              'layout_rule': 'First station and pitch fitted by normalized weighted area/width objective; c_i=c0+2*p*(i+b)*tan(theta/2).',
              'validation_limit': 'Full-outline inter-vane gaps use 0.3 mm contour sampling. Wall clearance uses 501 stations per side. Equal spacing is a fitted objective, not a guaranteed constraint.'}
    (folder / 'geometry_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (folder / 'README.txt').write_text(
        'XYZ CSV coordinates are centimetres. Each vane has separate upper, lower, '
        'trailing-edge and closed-contour files. Join trailing-edge endpoints with '
        'a straight line in CAD. The contour repeats its first point. Profile '
        'curves are sampled to 0.05 mm tolerance. Wall clearance and adjacent '
        'full-outline gaps are checked; exact six-gap equality is not guaranteed. Verify CAD units and '
        'dimensions before CFD.\n', encoding='utf-8')
    return folder


def svg_text(case: dict) -> str:
    limit = 11244.4
    radius = 2400.0
    tangent = limit - radius
    points = [point for vane in case['vanes'] for point in vane['contour']]
    margin = 300
    x_min = min(0, *(point[0] for point in points)) - margin
    x_max = max(limit, *(point[0] for point in points)) + margin
    y_min = min(0, *(-point[1] for point in points)) - margin
    y_max = max(limit, *(-point[1] for point in points)) + margin
    def pt(p: tuple[float, float]) -> str:
        return f"{p[0]:.2f},{-p[1]:.2f}"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min:.2f} {y_min:.2f} {x_max-x_min:.2f} {y_max-y_min:.2f}" role="img" aria-label="Five corner vanes and LE and TE rows">',
             f'<rect x="{x_min:.2f}" y="{y_min:.2f}" width="{x_max-x_min:.2f}" height="{y_max-y_min:.2f}" fill="white"/>',
             f'<path d="M {radius} 0 V 0 A {radius} {radius} 0 0 1 0 {radius}" fill="none" stroke="#64748b" stroke-width="35"/>',
             f'<path d="M {limit} 0 V {tangent} A {radius} {radius} 0 0 1 {tangent} {limit} H 0" fill="none" stroke="#64748b" stroke-width="35"/>']
    for vane in case["vanes"]:
        coords = " ".join(pt(p) for p in vane["contour"])
        parts.append(f'<polygon points="{coords}" fill="#93c5fd" fill-opacity=".35" stroke="#0f172a" stroke-width="18"/>')
        previous = case.get('comparison_vanes', case['vanes'])[vane['index']-1]
        starting_le, starting_te = previous['le'], previous['te']
        camber_points = [mul(add(profile_point(vane['le'], vane['scale'], j/80, 1),
                                profile_point(vane['le'], vane['scale'], j/80, -1)), .5) for j in range(81)]
        camber_coords = ' '.join(pt(point) for point in camber_points)
        parts.append(f'<polyline points="{camber_coords}" fill="none" stroke="#7c3aed" stroke-width="15" stroke-dasharray="40 35"/>')
        parts.append(f'<line data-role="starting-chord" x1="{starting_le[0]:.2f}" y1="{-starting_le[1]:.2f}" x2="{starting_te[0]:.2f}" y2="{-starting_te[1]:.2f}" stroke="#f59e0b" stroke-width="18" stroke-dasharray="35 55"/>')
        base_le = sub(vane['midpoint'], mul(CHORD_AXIS, case['source_chord_mm'] / 2))
        base_te = add(vane['midpoint'], mul(CHORD_AXIS, case['source_chord_mm'] / 2))
        parts.append(f'<line data-role="baseline-chord" x1="{base_le[0]:.2f}" y1="{-base_le[1]:.2f}" x2="{base_te[0]:.2f}" y2="{-base_te[1]:.2f}" stroke="#64748b" stroke-width="18" stroke-dasharray="70 45"/>')
        parts.append(f'<line data-role="current-chord" x1="{vane["le"][0]:.2f}" y1="{-vane["le"][1]:.2f}" x2="{vane["te"][0]:.2f}" y2="{-vane["te"][1]:.2f}" stroke="#15803d" stroke-width="20" stroke-dasharray="85 45"/>')
    midpoint_coords = " ".join(pt(v['midpoint']) for v in case['vanes'])
    parts.append(f'<polyline data-role="midpoint-row" points="{midpoint_coords}" fill="none" stroke="#0f172a" stroke-width="16" stroke-dasharray="35 45"/>')
    for key, color in (("le", "#2563eb"), ("te", "#dc2626")):
        coords = " ".join(pt(v[key]) for v in case["vanes"])
        parts.append(f'<polyline data-role="{key}-row" points="{coords}" fill="none" stroke="{color}" stroke-width="27"/>')
        for vane in case['vanes']:
            point = vane[key]
            parts.append(f'<circle cx="{point[0]:.2f}" cy="{-point[1]:.2f}" r="38" fill="{color}"/>')
    for vane in case['vanes']:
        parts.append(f'<circle cx="{vane["midpoint"][0]:.2f}" cy="{-vane["midpoint"][1]:.2f}" r="32" fill="#0f172a"/>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def export_svg(case: dict, path: Path) -> None:
    path.write_text(svg_text(case), encoding="utf-8")


def browser_gui(initial_theta: float, common_growth: float, open_browser: bool = True) -> None:
    import http.server
    import urllib.parse
    import webbrowser

    page = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corner vanes — weighted area and passage fit</title>
<style>*{box-sizing:border-box}body{margin:0;font:15px system-ui;color:#153449;background:#f2f6f8}
main{max-width:1450px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 8px}p{line-height:1.5}
.controls,.status,.plot{background:white;border:1px solid #d8e3e9;border-radius:12px;padding:16px;margin:12px 0}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.controls input{flex:1;min-width:180px;accent-color:#02799a}
button{padding:10px 16px;border:0;border-radius:7px;background:#086b86;color:white;cursor:pointer}
button:disabled{opacity:.45;cursor:default}.status{white-space:pre-line;font-size:14px}.bad{border-color:#c63737;background:#fff1ef}.warn{border-color:#d69e2e;background:#fff8e6}
.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(270px,1fr);gap:16px}
#drawing{width:100%;height:600px;object-fit:contain;display:block}#width{width:100%;height:220px;display:block}
.note{color:#526d7d;font-size:13px}.stats{font-size:14px;line-height:1.8;white-space:pre-line}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:7px 4px;border-bottom:1px solid #e5ecf0}
@media(max-width:850px){.layout{grid-template-columns:1fr}#drawing{height:480px}}</style></head><body><main>
<h1>Corner vanes — weighted area and passage fit</h1>
<p>First chord position and midpoint pitch are fitted together. Weights: 45% passage-area balance, 45% camber-normal width balance, 10% LE/TE opening balance, plus a small penalty on layout changes. Zero-degree placement is fixed. The requested LE/TE angle and original normal thickness are preserved.</p>
<div class="controls"><label for="theta">Theta</label><input id="theta" type="range" min="0" max="50" step="0.1" value="__THETA__"><strong id="thetaLabel"></strong>
<label for="growth">Common growth</label><input id="growth" type="range" min="0.05" max="2" step="0.05" value="__GROWTH__"><strong id="growthLabel"></strong>
<button id="export" disabled>Export points</button></div>
<div id="status" class="status">Building geometry…</div>
<div class="layout"><div class="plot"><img id="drawing" alt="Corner geometry and construction lines">
<p class="note">Blue: leading-edge row. Red: trailing-edge row. Green dashed: displayed chord lines. Orange dashed: original Anand chord lines. Purple dashed: mean camber lines. Gray dashed: original-length chords. Black dotted: midpoint row. Gray solid: duct walls.</p></div>
<div class="plot"><h3>Wall and passage diagnostics</h3><div id="stats" class="stats"></div>
<h3>Outer-wall passage through the bend</h3><canvas id="width"></canvas>
<p class="note">Blue: current normal distance to the outer wall. Gray: original Anand geometry at this angle. This is a geometric distance, not flow area or a velocity prediction.</p>
<div id="table"></div><p class="note">The six full-outline gaps are the inner wall, four adjacent vane surfaces, and outer wall. The fit balances enclosed area and local widths; exact equality is not guaranteed. Enclosed planform area is not flow cross-sectional area. Normal rays are sampled at equal camber progress, using both sides where available; they are not calculated streamlines.</p></div></div>
<p class="note">The existing dimensioned wall model is shown for context. Export uses the second code's separate-curve folder layout. The source outline here retains its 50 mm trailing-edge gap; the second code uses about 17 mm.</p>
<script>
const $=id=>document.getElementById(id);let last=null,serial=0,timer=null;
function chart(current,reference){let canvas=$('width'),w=canvas.clientWidth||350,h=220,d=devicePixelRatio||1;
 canvas.width=w*d;canvas.height=h*d;let c=canvas.getContext('2d');c.scale(d,d);c.clearRect(0,0,w,h);
 if(!current||!reference){c.fillStyle='#526d7d';c.fillText('Normal passage unavailable for this geometry',12,40);return}
 let max=Math.max(...current,...reference)*1.15;c.font='12px system-ui';c.fillStyle='#526d7d';
 c.fillText('Gap (mm)',7,15);c.fillText('LE',40,210);c.fillText('Mid-bend',w/2-25,210);c.fillText('TE',w-35,210);
 for(let k=0;k<4;k++){let y=30+k*50,value=max*(1-k/3);c.fillText(value.toFixed(0),1,y+4);c.beginPath();c.moveTo(42,y);c.lineTo(w-10,y);c.strokeStyle='#e1e9ed';c.stroke()}
 for(const [arr,color] of [[reference,'#9caab3'],[current,'#02799a']]){c.beginPath();arr.forEach((value,i)=>{let x=42+i/(arr.length-1)*(w-52),y=180-150*value/max;i?c.lineTo(x,y):c.moveTo(x,y)});c.strokeStyle=color;c.lineWidth=2;c.stroke()}}
async function refresh(){let id=++serial,t=Number($('theta').value),b=Number($('growth').value);
 $('thetaLabel').textContent=t.toFixed(1)+'°';$('growthLabel').textContent=b.toFixed(2);
 $('status').textContent='Updating geometry…';$('export').disabled=true;
 try{let response=await fetch('/case?theta='+t+'&growth='+b),data=await response.json();
  if(!response.ok)throw Error(data.error);if(id!==serial)return;last=data;
  $('drawing').src='/preview?theta='+t+'&growth='+b+'&v='+id;
   $('status').className='status'+(!data.inside_walls?' bad':'');
   const f=data.spacing_fit,m=f.metrics,o=f.baseline_metrics;
   $('status').textContent=`Theta ${data.measured_theta_deg.toFixed(2)}° · weighted geometric error ${f.starting_score.toFixed(4)} → ${f.score.toFixed(4)} · sampled clearances positive`;
   $('stats').textContent=`First chord station: ${f.starting_first_station_mm.toFixed(1)} → ${f.fitted_first_station_mm.toFixed(1)} mm\nMidpoint pitch: ${f.starting_pitch_mm.toFixed(1)} → ${data.midpoint_pitch_mm.toFixed(1)} mm\nOuter mid-bend passage: ${f.original_outer_midbend_gap_mm.toFixed(1)} → ${f.proposed_outer_midbend_gap_mm.toFixed(1)} mm\nMinimum wall clearance: ${data.sampled_min_wall_gap_mm.toFixed(1)} mm\nArea RMS imbalance: ${(100*Math.sqrt(o.terms.area)).toFixed(1)}% → ${(100*Math.sqrt(m.terms.area)).toFixed(1)}%\nNormal-width RMS imbalance: ${(100*Math.sqrt(o.terms.normal)).toFixed(1)}% → ${(100*Math.sqrt(m.terms.normal)).toFixed(1)}%\nEnd-opening RMS imbalance: ${(100*Math.sqrt(o.terms.opening)).toFixed(1)}% → ${(100*Math.sqrt(m.terms.opening)).toFixed(1)}%\nMaximum normal thickness: ${data.max_normal_thickness_mm.toFixed(1)} mm`;
   const labels=['Inner wall–1','1–2','2–3','3–4','4–5','5–outer wall'];
   $('table').innerHTML='<h3>Six passage areas and widths</h3><table><tr><th>Passage</th><th>Area before → after (m²)</th><th>Mid normal (mm)</th><th>LE / TE (mm)</th><th>Minimum gap (mm)</th></tr>'+labels.map((v,i)=>`<tr><td>${v}</td><td>${(o.areas_mm2[i]/1e6).toFixed(2)} → ${(m.areas_mm2[i]/1e6).toFixed(2)}</td><td>${m.normal_widths_mm[i][9].toFixed(0)}</td><td>${m.openings_mm[i].map(x=>x.toFixed(0)).join(' / ')}</td><td>${f.full_six_gaps_mm[i].toFixed(0)}</td></tr>`).join('')+'</table><h3>Chords</h3><table><tr><th>Vane</th><th>Chord (cm)</th><th>Inward shift (mm)</th></tr>'+data.vanes.map(v=>`<tr><td>${v.index}</td><td>${(v.chord_mm/10).toFixed(1)}</td><td>${v.inward_offset_mm.toFixed(1)}</td></tr>`).join('')+'</table>';
   chart(data.outer_normal_widths_mm,data.reference_outer_normal_widths_mm);
   $('export').disabled=!data.inside_walls||!data.spacing_fit.fit_applied;
 }catch(error){if(id===serial){last=null;$('status').className='status bad';$('status').textContent=error.message;$('stats').textContent='';$('table').textContent='';chart(null,null)}}}
function schedule(){clearTimeout(timer);timer=setTimeout(refresh,70)}
$('theta').addEventListener('input',schedule);$('growth').addEventListener('input',schedule);
$('export').addEventListener('click',async()=>{if(!last)return;let id=serial;$('export').disabled=true;
 try{let response=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({theta:Number($('theta').value),growth:Number($('growth').value)})});
 let data=await response.json();if(!response.ok)throw Error(data.error);
 if(id===serial)$('status').textContent+='\nSaved: '+data.folder;
 }catch(error){if(id===serial)$('status').textContent+='\nExport failed: '+error.message}
 finally{if(id===serial)$('export').disabled=!last?.inside_walls||!last?.spacing_fit.fit_applied}});
window.addEventListener('resize',()=>{if(last)chart(last.outer_normal_widths_mm,last.reference_outer_normal_widths_mm)});
refresh();</script></main></body></html>'''
    page = page.replace('__THETA__', str(initial_theta)).replace('__GROWTH__', str(common_growth))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: dict, status: int = 200) -> None:
            self.send_bytes(json.dumps(value).encode('utf-8'), 'application/json; charset=utf-8', status)

        def parameters(self) -> tuple[float, float]:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return float(query['theta'][0]), float(query['growth'][0])

        def do_GET(self) -> None:
            route = urllib.parse.urlparse(self.path).path
            if route == '/':
                self.send_bytes(page.encode('utf-8'), 'text/html; charset=utf-8')
                return
            try:
                theta, growth = self.parameters()
                case = spaced_case(theta, growth)
                if route == '/case':
                    reference = geometry(theta, growth)
                    self.send_json(dict(measured_theta_deg=case['measured_theta_deg'],
                                        le_line_angle_deg=case['le_line_angle_deg'],
                                        te_line_angle_deg=case['te_line_angle_deg'],
                                        midpoint_pitch_mm=case['midpoint_pitch_mm'],
                                        max_normal_thickness_mm=case['vanes'][0]['max_normal_thickness_mm'],
                                        sampled_min_wall_gap_mm=case['sampled_min_wall_gap_mm'],
                                        inner_wall_gap_mm=case['inner_wall_gap_mm'],
                                        outer_wall_gap_mm=case['outer_wall_gap_mm'],
                                        reference_outer_gap_mm=reference['outer_wall_gap_mm'],
                                        outer_normal_widths_mm=case['outer_normal_widths_mm'],
                                        reference_outer_normal_widths_mm=reference['outer_normal_widths_mm'],
                                        mid_bend_projected_gaps_mm=case['mid_bend_projected_gaps_mm'],
                                        endpoint_transverse_pitch_mm=case['endpoint_transverse_pitch_mm'],
                                        inside_walls=case['inside_walls'],
                                        spacing_fit=case['spacing_fit'],
                                        chords_mm=[v['chord_mm'] for v in case['vanes']],
                                        end_growth_mm=[v['le_growth_mm'] for v in case['vanes']],
                                        vanes=[dict(index=v['index'],chord_mm=v['chord_mm'],
                                                    inward_offset_mm=(v['index']-1)*(PITCH_MM-case['midpoint_pitch_mm'])-case['spacing_fit']['station_shift_mm'])
                                               for v in case['vanes']]))
                elif route == '/preview':
                    self.send_bytes(svg_text(case).encode('utf-8'), 'image/svg+xml; charset=utf-8')
                else:
                    self.send_json({'error': 'Unknown page'}, 404)
            except (KeyError, IndexError, ValueError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def do_POST(self) -> None:
            if self.path != '/export':
                self.send_json({'error': 'Unknown page'}, 404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                theta, growth = float(data['theta']), float(data['growth'])
                case = spaced_case(theta, growth)
                folder = export_case(case, Path(__file__).resolve().parent / 'clearance_exports')
                self.send_json({'folder': str(folder)})
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self.send_json({'error': str(exc)}, 400)

    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        url = f'http://127.0.0.1:{server.server_port}/'
        print('Open the vane browser interface at', url, flush=True)
        if open_browser and not webbrowser.open(url):
            print('Paste that address into your browser if it did not open automatically.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, help="0–50 degrees; starts the browser if omitted")
    parser.add_argument("--common-growth", type=float, default=1.0,
                        help="Positive shared chord-growth component; default 1")
    parser.add_argument("--csv", type=Path, help="Optional combined XYZ CSV in cm")
    parser.add_argument("--export", type=Path, help="Export per-vane CSV folder to this destination")
    parser.add_argument("--svg", type=Path, help="Optional geometry preview")
    parser.add_argument("--gui", action="store_true", help="Open the browser interface")
    parser.add_argument("--no-browser", action="store_true", help="Serve the GUI without opening a browser")
    args = parser.parse_args()
    if args.gui or args.theta is None:
        browser_gui(args.theta if args.theta is not None else 0.0,
                    args.common_growth, not args.no_browser)
        return
    case = spaced_case(args.theta, args.common_growth)
    report = {key: value for key, value in case.items() if key != "vanes"}
    report["vanes"] = [{key: value for key, value in vane.items() if key != "contour"}
                       for vane in case["vanes"]]
    print(json.dumps(report, indent=2))
    if args.csv:
        export_csv(case, args.csv)
    if args.export:
        print('Exported to', export_case(case, args.export))
    if args.svg:
        export_svg(case, args.svg)


if __name__ == "__main__":
    main()

