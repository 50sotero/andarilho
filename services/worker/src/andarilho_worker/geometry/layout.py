"""Layout validation and collision mesh generation (README section 9).

All output geometry is in canonical world coordinates: meters, +Y up, layout 2D
points ``[x, z]`` map to world ``(x, elevation, z)``. Wall segments are generated
on either side of every opening plus a lintel above it; no solid wall is ever
extruded across an open doorway.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from andarilho_worker.contracts import layout_validator
from andarilho_worker.geometry.glb import MeshPart, write_glb
from andarilho_worker.geometry.polygon import (
    point_in_polygon,
    polygon_problems,
    polygons_intersect,
    signed_area,
    triangulate,
)

MAX_WALL_THICKNESS = 2.0
MAX_HEIGHT = 10.0
MAX_EXTENT = 500.0
SPAWN_FLOOR_TOLERANCE = 0.05


@dataclass(frozen=True)
class Finding:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class LayoutError(ValueError):
    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = list(findings)
        super().__init__("; ".join(f.message for f in self.findings[:5]))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_tree(v) for v in value)
    return True


def validate_layout(layout: Mapping[str, Any]) -> list[Finding]:
    """Schema + semantic checks. Returns findings (empty when valid)."""
    if not _finite_tree(layout):
        return [Finding("NON_FINITE", "layout contains a non-finite number")]
    schema_errors = sorted(layout_validator().iter_errors(layout), key=lambda e: list(e.path))
    if schema_errors:
        return [
            Finding("SCHEMA", f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
            for e in schema_errors[:20]
        ]
    findings: list[Finding] = []
    floors = {f["id"]: f for f in layout["floors"]}
    walls = {w["id"]: w for w in layout.get("walls", [])}
    if len(floors) != len(layout["floors"]):
        findings.append(Finding("DUPLICATE_ID", "floor ids must be unique"))
    if len(walls) != len(layout.get("walls", [])):
        findings.append(Finding("DUPLICATE_ID", "wall ids must be unique"))
    ids = [o["id"] for o in layout.get("openings", [])] + [
        o["id"] for o in layout.get("obstacles", [])
    ]
    if len(set(ids)) != len(ids):
        findings.append(Finding("DUPLICATE_ID", "opening/obstacle ids must be unique"))

    for floor in layout["floors"]:
        polys = floor["polygons"]
        for i, poly in enumerate(polys):
            for problem in polygon_problems(poly):
                findings.append(
                    Finding("INVALID_FLOOR", f"floor {floor['id']} polygon {i}: {problem}")
                )
            if any(abs(v) > MAX_EXTENT for p in poly for v in p):
                findings.append(Finding("OUT_OF_RANGE", f"floor {floor['id']} is too large"))
        for i, hole in enumerate(floor.get("holes", [])):
            problems = polygon_problems(hole)
            for problem in problems:
                findings.append(
                    Finding("INVALID_FLOOR", f"floor {floor['id']} hole {i}: {problem}")
                )
            if problems:
                continue
            owners = [p for p in polys if all(point_in_polygon(v, p) for v in hole)]
            if not owners or any(polygons_intersect(hole, p) for p in owners):
                findings.append(
                    Finding(
                        "INVALID_FLOOR", f"floor {floor['id']} hole {i} is not inside a polygon"
                    )
                )
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polygons_intersect(polys[i], polys[j]):
                    findings.append(
                        Finding(
                            "INVALID_FLOOR", f"floor {floor['id']} polygons {i} and {j} overlap"
                        )
                    )

    for wall in layout.get("walls", []):
        if wall["floorId"] not in floors:
            findings.append(Finding("UNKNOWN_FLOOR", f"wall {wall['id']} references unknown floor"))
        if math.dist(wall["start"], wall["end"]) < 1e-3:
            findings.append(Finding("INVALID_WALL", f"wall {wall['id']} is degenerate"))
        if not 0 < wall["thickness"] <= MAX_WALL_THICKNESS:
            findings.append(Finding("INVALID_WALL", f"wall {wall['id']} has impossible thickness"))
        if not 0 < wall["height"] <= MAX_HEIGHT:
            findings.append(Finding("INVALID_WALL", f"wall {wall['id']} has impossible height"))

    per_wall: dict[str, list[tuple[float, float]]] = {}
    for opening in layout.get("openings", []):
        wall = walls.get(opening["wallId"])
        if wall is None:
            findings.append(
                Finding("UNKNOWN_WALL", f"opening {opening['id']} references unknown wall")
            )
            continue
        length = math.dist(wall["start"], wall["end"])
        bottom = opening.get("bottom", 0.0)
        if opening["offset"] < 0 or opening["offset"] + opening["width"] > length + 1e-9:
            findings.append(
                Finding("INVALID_OPENING", f"opening {opening['id']} extends past its wall")
            )
        if opening["top"] <= bottom or opening["top"] > wall["height"] + 1e-9:
            findings.append(
                Finding("INVALID_OPENING", f"opening {opening['id']} has impossible heights")
            )
        span = (opening["offset"], opening["offset"] + opening["width"])
        for other in per_wall.get(wall["id"], []):
            if span[0] < other[1] - 1e-9 and other[0] < span[1] - 1e-9:
                findings.append(
                    Finding("INVALID_OPENING", f"opening {opening['id']} overlaps another")
                )
        per_wall.setdefault(wall["id"], []).append(span)

    for obstacle in layout.get("obstacles", []):
        if obstacle["floorId"] not in floors:
            findings.append(
                Finding("UNKNOWN_FLOOR", f"obstacle {obstacle['id']} references unknown floor")
            )
        for problem in polygon_problems(obstacle["footprint"]):
            findings.append(Finding("INVALID_OBSTACLE", f"obstacle {obstacle['id']}: {problem}"))
        if not 0 < obstacle["height"] <= MAX_HEIGHT:
            findings.append(
                Finding("INVALID_OBSTACLE", f"obstacle {obstacle['id']} has impossible height")
            )

    spawn = layout["spawn"]
    floor = floors.get(spawn["floorId"])
    if floor is None:
        findings.append(Finding("UNKNOWN_FLOOR", "spawn references unknown floor"))
    elif not findings:
        x, y, z = spawn["feet"]
        if abs(y - floor["elevation"]) > SPAWN_FLOOR_TOLERANCE:
            findings.append(Finding("SPAWN_NOT_ON_FLOOR", "spawn feet are not at floor elevation"))
        if not floor_supports(floor, (x, z)):
            findings.append(Finding("SPAWN_NOT_ON_FLOOR", "spawn is outside the approved floor"))
        for obstacle in layout.get("obstacles", []):
            if obstacle["floorId"] == floor["id"] and point_in_polygon(
                (x, z), obstacle["footprint"]
            ):
                findings.append(Finding("SPAWN_BLOCKED", "spawn is inside an obstacle"))
    return findings


def ensure_valid_layout(layout: Mapping[str, Any]) -> None:
    findings = validate_layout(layout)
    if findings:
        raise LayoutError(findings)


def floor_supports(floor: Mapping[str, Any], point: Sequence[float]) -> bool:
    inside = any(point_in_polygon(point, p) for p in floor["polygons"])
    return inside and not any(point_in_polygon(point, h) for h in floor.get("holes", []))


# --------------------------------------------------------------------------
# Mesh generation
# --------------------------------------------------------------------------


def _orient_outward(positions: np.ndarray, tris: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Flip triangles of a convex solid so normals point away from its center."""
    a, b, c = positions[tris[:, 0]], positions[tris[:, 1]], positions[tris[:, 2]]
    normals = np.cross(b - a, c - a)
    outward = ((a + b + c) / 3.0) - center
    flip = np.einsum("ij,ij->i", normals, outward) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


BOX_TRIS = np.array(
    [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int64,
)  # fmt: skip


def oriented_box(
    start: Sequence[float],
    end: Sequence[float],
    half_thickness: float,
    y0: float,
    y1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A box along the 2D segment start->end with the given half thickness."""
    sx, sz = start
    ex, ez = end
    dx, dz = ex - sx, ez - sz
    length = math.hypot(dx, dz)
    nx, nz = -dz / length * half_thickness, dx / length * half_thickness
    corners = [(sx + nx, sz + nz), (ex + nx, ez + nz), (ex - nx, ez - nz), (sx - nx, sz - nz)]
    pos = np.array(
        [(x, y0, z) for x, z in corners] + [(x, y1, z) for x, z in corners], dtype=np.float64
    )
    return pos, _orient_outward(pos, BOX_TRIS, pos.mean(axis=0))


def prism(
    footprint: Sequence[Sequence[float]], y0: float, y1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a simple polygon footprint between y0 and y1 (closed solid)."""
    ring = [tuple(p) for p in footprint]
    if signed_area(ring) < 0:
        ring.reverse()
    pts2, cap = triangulate(ring)
    n = len(ring)
    bottom = np.column_stack([pts2[:, 0], np.full(n, y0), pts2[:, 1]])
    top = np.column_stack([pts2[:, 0], np.full(n, y1), pts2[:, 1]])
    positions = np.vstack([bottom, top])
    tris: list[list[int]] = []
    tris += _facing(positions, cap, -1.0)
    tris += _facing(positions, cap + n, 1.0)
    for i in range(n):
        j = (i + 1) % n
        # ring is counter-clockwise in (x, z); outward side faces have this winding
        tris.append([i, n + j, j])
        tris.append([i, n + i, n + j])
    return positions, _orient_sides(positions, np.array(tris, dtype=np.int64), ring, y0, y1)


def _facing(positions: np.ndarray, tris: np.ndarray, sign: float) -> list[list[int]]:
    out = []
    for t in tris:
        a, b, c = positions[t[0]], positions[t[1]], positions[t[2]]
        ny = np.cross(b - a, c - a)[1]
        out.append(
            [int(t[0]), int(t[1]), int(t[2])]
            if ny * sign >= 0
            else [int(t[0]), int(t[2]), int(t[1])]
        )
    return out


def _orient_sides(
    positions: np.ndarray, tris: np.ndarray, ring: list[tuple[float, float]], y0: float, y1: float
) -> np.ndarray:
    """Make side faces of a (possibly concave) prism point outward."""
    out = tris.copy()
    for k, t in enumerate(out):
        a, b, c = positions[t[0]], positions[t[1]], positions[t[2]]
        normal = np.cross(b - a, c - a)
        if abs(normal[1]) > 1e-12 and abs(normal[0]) + abs(normal[2]) < 1e-12:
            continue  # caps already oriented
        centroid = (a + b + c) / 3.0
        horiz = np.array([normal[0], normal[2]])
        norm = np.linalg.norm(horiz)
        if norm == 0:
            continue
        probe = np.array([centroid[0], centroid[2]]) + horiz / norm * 1e-4
        if point_in_polygon(probe, ring):
            out[k] = [t[0], t[2], t[1]]
    return out


def _wall_pieces(wall: Mapping[str, Any], openings: list[Mapping[str, Any]], elevation: float):
    """Yield (name, kind, start2d, end2d, y0, y1) boxes for a wall with openings."""
    sx, sz = wall["start"]
    ex, ez = wall["end"]
    length = math.dist(wall["start"], wall["end"])
    ux, uz = (ex - sx) / length, (ez - sz) / length
    half_t = wall["thickness"] / 2.0
    height = wall["height"]

    def at(s: float) -> tuple[float, float]:
        return (sx + ux * s, sz + uz * s)

    ops = sorted(openings, key=lambda o: o["offset"])
    # Extend the wall by half its thickness at each end so corners close.
    cursor = -half_t
    seg = 0
    pieces = []
    for op in ops:
        a, b = op["offset"], op["offset"] + op["width"]
        if a - cursor > 1e-6:
            pieces.append((f"{wall['id']}-seg{seg}", "wall", at(cursor), at(a), 0.0, height))
            seg += 1
        bottom = op.get("bottom", 0.0)
        top = op["top"]
        if op.get("state", "open") == "closed":
            pieces.append(
                (f"{wall['id']}-closed-{op['id']}", "closed_opening", at(a), at(b), bottom, top)
            )
        if bottom > 1e-6:
            pieces.append((f"{wall['id']}-sill-{op['id']}", "sill", at(a), at(b), 0.0, bottom))
        if height - top > 1e-6:
            pieces.append((f"{wall['id']}-lintel-{op['id']}", "lintel", at(a), at(b), top, height))
        cursor = b
    end = length + half_t
    if end - cursor > 1e-6:
        pieces.append((f"{wall['id']}-seg{seg}", "wall", at(cursor), at(end), 0.0, height))
    for name, kind, p0, p1, y0, y1 in pieces:
        yield name, kind, p0, p1, half_t, elevation + y0, elevation + y1


def collision_parts(layout: Mapping[str, Any]) -> list[MeshPart]:
    """Build collision mesh parts in canonical world coordinates (identity nodes)."""
    ensure_valid_layout(layout)
    floors = {f["id"]: f for f in layout["floors"]}
    parts: list[MeshPart] = []
    for floor in layout["floors"]:
        elevation = float(floor["elevation"])
        for i, poly in enumerate(floor["polygons"]):
            outer = [tuple(p) for p in poly]
            if signed_area(outer) < 0:
                outer.reverse()
            holes = []
            for hole in floor.get("holes", []):
                if all(point_in_polygon(v, poly) for v in hole):
                    h = [tuple(p) for p in hole]
                    if signed_area(h) > 0:
                        h.reverse()
                    holes.append(h)
            pts2, tris = triangulate(outer, holes)
            pos = np.column_stack([pts2[:, 0], np.full(len(pts2), elevation), pts2[:, 1]])
            oriented = np.array(_facing(pos, tris, 1.0), dtype=np.int64)
            parts.append(
                MeshPart(f"{floor['id']}-poly{i}", "floor", pos, oriented, {"floorId": floor["id"]})
            )
    openings_by_wall: dict[str, list[Mapping[str, Any]]] = {}
    for op in layout.get("openings", []):
        openings_by_wall.setdefault(op["wallId"], []).append(op)
    for wall in layout.get("walls", []):
        elevation = float(floors[wall["floorId"]]["elevation"])
        for name, kind, p0, p1, half_t, y0, y1 in _wall_pieces(
            wall, openings_by_wall.get(wall["id"], []), elevation
        ):
            pos, tris = oriented_box(p0, p1, half_t, y0, y1)
            parts.append(MeshPart(name, kind, pos, tris, {"wallId": wall["id"]}))
    for obstacle in layout.get("obstacles", []):
        elevation = float(floors[obstacle["floorId"]]["elevation"])
        pos, tris = prism(obstacle["footprint"], elevation, elevation + obstacle["height"])
        parts.append(
            MeshPart(obstacle["id"], "obstacle", pos, tris, {"obstacleId": obstacle["id"]})
        )
    return parts


def collision_glb(layout: Mapping[str, Any]) -> bytes:
    return write_glb(collision_parts(layout))


def layout_bounds(layout: Mapping[str, Any], pad: float = 0.05) -> dict[str, list[float]]:
    """World bounds covering the generated collision geometry, padded and rounded outward."""
    parts = collision_parts(layout)
    allpos = np.vstack([p.positions for p in parts])
    lo = allpos.min(axis=0) - pad
    hi = allpos.max(axis=0) + pad
    return {
        "min": [math.floor(round(float(v) * 100, 4)) / 100 for v in lo],
        "max": [math.ceil(round(float(v) * 100, 4)) / 100 for v in hi],
    }


def rectangle_layout(
    min_xz: Sequence[float],
    max_xz: Sequence[float],
    *,
    elevation: float,
    height: float,
    spawn: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A single-room proposal: rectangle floor with four walls.

    The spawn is ``spawn`` (feet and yaw, such as where the capture walk started)
    when it lies on the floor, else the rectangle's centre facing -Z.
    """
    x0, z0 = (float(v) for v in min_xz)
    x1, z1 = (float(v) for v in max_xz)
    feet, yaw = [(x0 + x1) / 2, elevation, (z0 + z1) / 2], 0.0
    if spawn is not None:
        sx, sz = float(spawn["feet"][0]), float(spawn["feet"][2])
        if x0 < sx < x1 and z0 < sz < z1:
            feet, yaw = [sx, elevation, sz], float(spawn["yawRadians"])
    corners = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
    walls = []
    for i, name in enumerate(("north", "east", "south", "west")):
        walls.append(
            {
                "id": f"wall-{name}",
                "floorId": "floor-0",
                "start": corners[i],
                "end": corners[(i + 1) % 4],
                "thickness": 0.1,
                "height": height,
                "confidence": 0.3,
                "provenance": "proposal",
            }
        )
    return {
        "schemaVersion": "1.0",
        "floors": [{"id": "floor-0", "elevation": elevation, "polygons": [corners], "holes": []}],
        "walls": walls,
        "openings": [],
        "obstacles": [],
        "landmarks": [],
        "spawn": {"feet": feet, "yawRadians": yaw, "floorId": "floor-0"},
    }
