"""Capsule clearance queries against a triangle soup (world coordinates)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from andarilho_worker.geometry.glb import GlbNode

FLOOR_KINDS = frozenset({"floor"})


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    # Ericson, Real-Time Collision Detection, 5.1.5
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _segment_segment_distance(p1, q1, p2, q2) -> float:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0, 1)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _segment_hits_triangle(p, q, a, b, c) -> bool:
    d = q - p
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = e1 @ h
    if abs(det) < 1e-14:
        return False
    inv = 1.0 / det
    s = p - a
    u = inv * (s @ h)
    if u < 0 or u > 1:
        return False
    qv = np.cross(s, e1)
    v = inv * (d @ qv)
    if v < 0 or u + v > 1:
        return False
    t = inv * (e2 @ qv)
    return 0 <= t <= 1


def segment_triangle_distance(p, q, a, b, c) -> float:
    if _segment_hits_triangle(p, q, a, b, c):
        return 0.0
    best = min(
        float(np.linalg.norm(p - _closest_point_on_triangle(p, a, b, c))),
        float(np.linalg.norm(q - _closest_point_on_triangle(q, a, b, c))),
    )
    for u, v in ((a, b), (b, c), (c, a)):
        best = min(best, _segment_segment_distance(p, q, u, v))
    return best


def capsule_clearance(
    nodes: Iterable[GlbNode],
    feet: Sequence[float],
    *,
    radius: float = 0.25,
    height: float = 1.7,
    skip_kinds: frozenset[str] = FLOOR_KINDS,
) -> float:
    """Signed clearance (m) between a standing capsule and the non-floor geometry.

    The capsule's bottom sphere rests on ``feet``. Negative means penetration.
    """
    feet_v = np.asarray(feet, dtype=np.float64)
    p = feet_v + np.array([0.0, radius, 0.0])
    q = feet_v + np.array([0.0, height - radius, 0.0])
    lo = np.minimum(p, q) - radius - 1.0
    hi = np.maximum(p, q) + radius + 1.0
    best = float("inf")
    for node in nodes:
        if node.kind in skip_kinds:
            continue
        pos = node.positions
        tri_pos = pos[node.triangles]  # (M, 3, 3)
        tmin, tmax = tri_pos.min(axis=1), tri_pos.max(axis=1)
        near = np.all(tmax >= lo, axis=1) & np.all(tmin <= hi, axis=1)
        for tri in tri_pos[near]:
            best = min(best, segment_triangle_distance(p, q, tri[0], tri[1], tri[2]))
    return best - radius


def capsule_is_clear(nodes: Sequence[GlbNode], feet: Sequence[float], **kw) -> bool:
    return capsule_clearance(nodes, feet, **kw) >= -1e-6
