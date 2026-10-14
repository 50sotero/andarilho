"""2D polygon helpers on the horizontal plane. Points are ``(x, z)`` in meters."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mapbox_earcut
import numpy as np

Point = tuple[float, float]
EPS = 1e-9


def signed_area(poly: Sequence[Sequence[float]]) -> float:
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        total += x1 * z2 - x2 * z1
    return total / 2.0


def _orient(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> bool:
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def segments_intersect(
    a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]
) -> bool:
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True
    if abs(d1) <= EPS and _on_segment(c, d, a):
        return True
    if abs(d2) <= EPS and _on_segment(c, d, b):
        return True
    if abs(d3) <= EPS and _on_segment(a, b, c):
        return True
    return abs(d4) <= EPS and _on_segment(a, b, d)


def polygon_problems(poly: Sequence[Sequence[float]], *, min_area: float = 1e-4) -> list[str]:
    """Return human-readable problems: degenerate, repeated vertices, self-intersection."""
    problems: list[str] = []
    n = len(poly)
    if n < 3:
        return ["fewer than three vertices"]
    if any(not math.isfinite(v) for p in poly for v in p):
        return ["non-finite coordinate"]
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if math.dist(a, b) <= 1e-6:
            problems.append("repeated consecutive vertex")
            return problems
    if abs(signed_area(poly)) < min_area:
        problems.append("degenerate polygon (zero area)")
        return problems
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # adjacent edges share a vertex
            c, d = poly[j], poly[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                problems.append("self-intersecting polygon")
                return problems
    return problems


def point_in_polygon(p: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    """Even-odd rule; points exactly on an edge count as inside."""
    x, z = p
    inside = False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if abs(_orient(a, b, p)) <= 1e-12 and _on_segment(a, b, p):
            return True
        if (a[1] > z) != (b[1] > z):
            xi = a[0] + (z - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < xi:
                inside = not inside
    return inside


def polygons_intersect(p: Sequence[Sequence[float]], q: Sequence[Sequence[float]]) -> bool:
    for i in range(len(p)):
        for j in range(len(q)):
            if segments_intersect(p[i], p[(i + 1) % len(p)], q[j], q[(j + 1) % len(q)]):
                return True
    return False


def triangulate(
    outer: Sequence[Sequence[float]], holes: Sequence[Sequence[Sequence[float]]] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a polygon with holes. Returns (points Nx2, triangles Mx3)."""
    rings = [list(outer), *[list(h) for h in holes]]
    points = np.array([pt for ring in rings for pt in ring], dtype=np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    indices = mapbox_earcut.triangulate_float64(points, ends)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        raise ValueError("polygon could not be triangulated")
    return points, tris
