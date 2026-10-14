"""Deterministic synthetic Gaussian splats rendered from a layout.

Splats are emitted directly in the canonical world frame (meters, +Y up), so the
manifest's ``worldFromDecodedRender`` is identity for synthetic scenes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from andarilho_worker.geometry.layout import ensure_valid_layout, floor_supports
from andarilho_worker.geometry.ply import SH_C0, SplatCloud, splat_dtype
from andarilho_worker.geometry.polygon import point_in_polygon

COLORS = {
    "floor_a": (0.62, 0.48, 0.34),
    "floor_b": (0.52, 0.40, 0.29),
    "wall": (0.86, 0.85, 0.80),
    "wall_alt": (0.78, 0.82, 0.86),
    "door_frame": (0.36, 0.21, 0.11),
    "obstacle": (0.20, 0.36, 0.62),
}
OPACITY = 0.95
MAX_SPLATS = 4000


def _quat_y(theta: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) rotating by theta about +Y."""
    return (math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0)


def _record(pos, color, scale, quat) -> tuple:
    logit = math.log(OPACITY / (1 - OPACITY))
    dc = [(c - 0.5) / SH_C0 for c in color]
    return (
        *[float(v) for v in pos],
        *dc,
        logit,
        *[math.log(max(s, 1e-4)) for s in scale],
        *quat,
    )


def _steps(length: float, spacing: float) -> list[float]:
    n = max(1, int(round(length / spacing)))
    return [(i + 0.5) * length / n for i in range(n)]


def synthetic_splats(layout: Mapping[str, Any], *, spacing: float = 0.2) -> SplatCloud:
    ensure_valid_layout(layout)
    records: list[tuple] = []
    floors = {f["id"]: f for f in layout["floors"]}
    obstacles = layout.get("obstacles", [])

    for floor in layout["floors"]:
        y = float(floor["elevation"]) + 0.002
        xs = [p[0] for poly in floor["polygons"] for p in poly]
        zs = [p[1] for poly in floor["polygons"] for p in poly]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        for i, dx in enumerate(_steps(x1 - x0, spacing)):
            for j, dz in enumerate(_steps(z1 - z0, spacing)):
                x, z = x0 + dx, z0 + dz
                if not floor_supports(floor, (x, z)):
                    continue
                if any(point_in_polygon((x, z), o["footprint"]) for o in obstacles):
                    continue
                color = COLORS["floor_a"] if (i // 2 + j // 2) % 2 == 0 else COLORS["floor_b"]
                if x < 0:
                    color = tuple(c * 0.92 for c in color)
                s = spacing * 0.6
                records.append(_record((x, y, z), color, (s, 0.004, s), (1.0, 0.0, 0.0, 0.0)))

    openings_by_wall: dict[str, list[Mapping[str, Any]]] = {}
    for op in layout.get("openings", []):
        openings_by_wall.setdefault(op["wallId"], []).append(op)
    for wall in layout.get("walls", []):
        base = float(floors[wall["floorId"]]["elevation"])
        (sx, sz), (ex, ez) = wall["start"], wall["end"]
        length = math.dist(wall["start"], wall["end"])
        ux, uz = (ex - sx) / length, (ez - sz) / length
        quat = _quat_y(math.atan2(-uz, ux))
        ops = openings_by_wall.get(wall["id"], [])
        color = COLORS["wall_alt"] if ops else COLORS["wall"]
        for s in _steps(length, spacing):
            for h in _steps(wall["height"], spacing):
                hole = any(
                    o["offset"] <= s <= o["offset"] + o["width"]
                    and o.get("bottom", 0.0) <= h <= o["top"]
                    and o.get("state", "open") == "open"
                    for o in ops
                )
                if hole:
                    continue
                pos = (sx + ux * s, base + h, sz + uz * s)
                scale = (spacing * 0.6, spacing * 0.6, wall["thickness"] / 2)
                records.append(_record(pos, color, scale, quat))
        for o in ops:
            # Door frame: posts on both sides and a header across the top.
            a, b = o["offset"], o["offset"] + o["width"]
            bottom, top = o.get("bottom", 0.0), o["top"]
            for s in (a, b):
                for h in _steps(top - bottom, 0.1):
                    pos = (sx + ux * s, base + bottom + h, sz + uz * s)
                    records.append(
                        _record(
                            pos,
                            COLORS["door_frame"],
                            (0.03, 0.06, wall["thickness"] / 2 + 0.01),
                            quat,
                        )
                    )
            for s in _steps(b - a, 0.1):
                pos = (sx + ux * (a + s), base + top, sz + uz * (a + s))
                records.append(
                    _record(
                        pos, COLORS["door_frame"], (0.06, 0.03, wall["thickness"] / 2 + 0.01), quat
                    )
                )

    for obstacle in obstacles:
        base = float(floors[obstacle["floorId"]]["elevation"])
        top = base + obstacle["height"]
        fp = obstacle["footprint"]
        xs = [p[0] for p in fp]
        zs = [p[1] for p in fp]
        step = 0.1
        for dx in _steps(max(xs) - min(xs), step):
            for dz in _steps(max(zs) - min(zs), step):
                x, z = min(xs) + dx, min(zs) + dz
                if point_in_polygon((x, z), fp):
                    records.append(
                        _record(
                            (x, top, z),
                            COLORS["obstacle"],
                            (0.06, 0.004, 0.06),
                            (1.0, 0.0, 0.0, 0.0),
                        )
                    )
        for i in range(len(fp)):
            (ax, az), (bx, bz) = fp[i], fp[(i + 1) % len(fp)]
            edge = math.dist(fp[i], fp[(i + 1) % len(fp)])
            ux, uz = (bx - ax) / edge, (bz - az) / edge
            quat = _quat_y(math.atan2(-uz, ux))
            for s in _steps(edge, step):
                for h in _steps(obstacle["height"], step):
                    pos = (ax + ux * s, base + h, az + uz * s)
                    color = tuple(c * 0.8 for c in COLORS["obstacle"])
                    records.append(_record(pos, color, (0.06, 0.06, 0.004), quat))

    if len(records) > MAX_SPLATS:
        raise ValueError(f"synthetic splat budget exceeded ({len(records)} > {MAX_SPLATS})")
    return SplatCloud(np.array(records, dtype=splat_dtype()))
