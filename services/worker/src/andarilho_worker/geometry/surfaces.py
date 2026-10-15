"""Layout surfaces: the visible faces of a layout as UV-mapped planar polygons (README section 9).

``layout_surfaces`` turns a validated layout into one :class:`Surface` per visible
face, built from the same boxes and polygons as ``layout.collision_parts`` so the
textured model follows the collision model. A surface is a plane with an origin
and two orthonormal axes: the point at surface coordinates ``(u, v)`` is
``origin + u * u_axis + v * v_axis`` in canonical world metres (+Y up).

``build_atlas`` shelf-packs every surface's bounding rectangle (a chart) into square
pages at one shared texel size. ``surfaces_to_parts`` lifts each surface's
triangulation into :class:`TexturedPart` records whose glTF UVs address those pages,
and ``chart_texels`` enumerates the texel centres of one chart for texturing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from andarilho_worker.geometry.layout import _wall_pieces, ensure_valid_layout
from andarilho_worker.geometry.polygon import point_in_polygon, signed_area, triangulate
from andarilho_worker.geometry.textured import TexturedPart

CEILING_HEIGHT = 2.5  # ceiling height above a floor that has no walls
ATLAS_RETRIES = 20
ATLAS_GROWTH = 1.25
CORNER_TOLERANCE = 1e-6
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Surface:
    name: str
    kind: str  # one of SURFACE_KINDS
    origin: np.ndarray  # (3,) world metres
    u_axis: np.ndarray  # (3,) unit
    v_axis: np.ndarray  # (3,) unit, u_axis x v_axis = normal
    normal: np.ndarray  # (3,) unit, towards the visible side
    outer: np.ndarray  # (K, 2) counter-clockwise polygon in (u, v) surface metres
    holes: list[np.ndarray] = field(default_factory=list)  # (H, 2) rings in (u, v)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChartPlacement:
    page: int
    x: int  # left edge of the chart rectangle, texels (padding included)
    y: int  # top edge, texels
    width: int  # texels, including padding on both sides
    height: int
    u_min: float  # surface metres of the chart's left edge (padding excluded)
    v_max: float  # surface metres of the chart's top edge (padding excluded)


@dataclass
class Atlas:
    texel_size: float  # metres per texel actually used
    page_size: int
    pages: int
    padding: int
    placements: list[ChartPlacement]  # index-aligned with the surfaces


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------


def _surface(
    name: str,
    kind: str,
    origin: Sequence[float],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    outer: Sequence[Sequence[float]],
    holes: Sequence[Sequence[Sequence[float]]],
    extras: Mapping[str, Any],
) -> Surface:
    """Build a surface, orienting its outer ring counter-clockwise and its holes clockwise."""
    outer_uv = np.asarray(outer, dtype=np.float64)
    if signed_area(outer_uv) < 0:
        outer_uv = outer_uv[::-1].copy()
    hole_uv = []
    for hole in holes:
        ring = np.asarray(hole, dtype=np.float64)
        if signed_area(ring) > 0:
            ring = ring[::-1].copy()
        hole_uv.append(ring)
    normal = np.cross(u_axis, v_axis)
    return Surface(
        name=name,
        kind=kind,
        origin=np.asarray(origin, dtype=np.float64),
        u_axis=np.asarray(u_axis, dtype=np.float64),
        v_axis=np.asarray(v_axis, dtype=np.float64),
        normal=normal / np.linalg.norm(normal),
        outer=outer_uv,
        holes=hole_uv,
        extras=dict(extras),
    )


def _horizontal(
    name: str,
    kind: str,
    y: float,
    outer_xz: Sequence[Sequence[float]],
    holes_xz: Sequence[Sequence[Sequence[float]]],
    *,
    up: bool,
    extras: Mapping[str, Any],
) -> Surface:
    """A horizontal face at height ``y``: (u, v) = (x, -z) facing up, (x, z) facing down."""
    sign = -1.0 if up else 1.0

    def to_uv(ring: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(ring, dtype=np.float64)
        return np.column_stack([arr[:, 0], sign * arr[:, 1]])

    return _surface(
        name,
        kind,
        (0.0, y, 0.0),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, sign]),
        to_uv(outer_xz),
        [to_uv(h) for h in holes_xz],
        extras,
    )


def _vertical(
    name: str,
    kind: str,
    a: Sequence[float],
    b: Sequence[float],
    normal2: Sequence[float],
    y0: float,
    y1: float,
    extras: Mapping[str, Any],
) -> Surface:
    """A vertical rectangle from the 2D (x, z) point ``a`` to ``b`` between heights y0 and y1.

    ``normal2`` is the horizontal unit normal of the visible side. Seen from that side
    with +Y up, ``u`` points to the right and the origin is the bottom-left corner.
    """
    normal = np.array([normal2[0], 0.0, normal2[1]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    u_axis = np.cross(UP, normal)
    p = np.array([a[0], y0, a[1]], dtype=np.float64)
    q = np.array([b[0], y0, b[1]], dtype=np.float64)
    if np.dot(q - p, u_axis) < 0:
        p, q = q, p
    length = math.dist(a, b)
    height = y1 - y0
    outer = [(0.0, 0.0), (length, 0.0), (length, height), (0.0, height)]
    return _surface(name, kind, p, u_axis, UP, outer, [], extras)


def _wall_surfaces(
    wall: Mapping[str, Any], openings: list[Mapping[str, Any]], elevation: float
) -> list[Surface]:
    start = np.array(wall["start"], dtype=np.float64)
    end = np.array(wall["end"], dtype=np.float64)
    length = math.dist(wall["start"], wall["end"])
    d = (end - start) / length  # along the wall
    left = np.array([-d[1], d[0]])  # horizontal normal on the left of the centreline
    extras = {"wallId": wall["id"]}
    out: list[Surface] = []
    for name, kind, p0_raw, p1_raw, half_t, y0, y1 in _wall_pieces(wall, openings, elevation):
        p0 = np.asarray(p0_raw, dtype=np.float64)
        p1 = np.asarray(p1_raw, dtype=np.float64)
        off = half_t * left
        out.append(_vertical(f"{name}:a", kind, p0 + off, p1 + off, left, y0, y1, extras))
        out.append(_vertical(f"{name}:b", kind, p0 - off, p1 - off, -left, y0, y1, extras))
        # A piece end at the wall's own extended ends is a corner; any other end is an
        # opening edge and gets a cap facing away from the piece.
        s0 = float(np.dot(p0 - start, d))
        s1 = float(np.dot(p1 - start, d))
        if abs(s0 + half_t) > CORNER_TOLERANCE:
            out.append(_vertical(f"{name}:cap0", kind, p0 - off, p0 + off, -d, y0, y1, extras))
        if abs(s1 - (length + half_t)) > CORNER_TOLERANCE:
            out.append(_vertical(f"{name}:cap1", kind, p1 - off, p1 + off, d, y0, y1, extras))
        if kind == "sill":
            ring = [p0 + off, p1 + off, p1 - off, p0 - off]
            out.append(_horizontal(f"{name}:top", kind, y1, ring, [], up=True, extras=extras))
        if kind == "lintel":
            ring = [p0 + off, p1 + off, p1 - off, p0 - off]
            out.append(_horizontal(f"{name}:bottom", kind, y0, ring, [], up=False, extras=extras))
    return out


def _obstacle_surfaces(obstacle: Mapping[str, Any], elevation: float) -> list[Surface]:
    oid = obstacle["id"]
    top = elevation + float(obstacle["height"])
    footprint = [(float(p[0]), float(p[1])) for p in obstacle["footprint"]]
    extras = {"obstacleId": oid}
    out = [_horizontal(f"{oid}:top", "obstacle", top, footprint, [], up=True, extras=extras)]
    # For a counter-clockwise footprint the outward normal of edge a->b is (dz, -dx).
    sign = 1.0 if signed_area(footprint) > 0 else -1.0
    n = len(footprint)
    for k in range(n):
        a, b = footprint[k], footprint[(k + 1) % n]
        dx, dz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dz)
        outward = (sign * dz / length, -sign * dx / length)
        out.append(_vertical(f"{oid}:side{k}", "obstacle", a, b, outward, elevation, top, extras))
    return out


def layout_surfaces(layout: Mapping[str, Any]) -> list[Surface]:
    """Every visible face of a valid layout, in floor, wall and obstacle order."""
    ensure_valid_layout(layout)
    floors = {f["id"]: f for f in layout["floors"]}
    walls_by_floor: dict[str, list[Mapping[str, Any]]] = {}
    for wall in layout.get("walls", []):
        walls_by_floor.setdefault(wall["floorId"], []).append(wall)
    openings_by_wall: dict[str, list[Mapping[str, Any]]] = {}
    for op in layout.get("openings", []):
        openings_by_wall.setdefault(op["wallId"], []).append(op)

    surfaces: list[Surface] = []
    for floor in layout["floors"]:
        elevation = float(floor["elevation"])
        walls = walls_by_floor.get(floor["id"], [])
        ceiling_y = elevation + max((float(w["height"]) for w in walls), default=CEILING_HEIGHT)
        extras = {"floorId": floor["id"]}
        for i, poly in enumerate(floor["polygons"]):
            outer = [tuple(p) for p in poly]
            holes = [h for h in floor.get("holes", []) if all(point_in_polygon(v, poly) for v in h)]
            name = f"{floor['id']}-poly{i}"
            surfaces.append(
                _horizontal(name, "floor", elevation, outer, holes, up=True, extras=extras)
            )
            name = f"{floor['id']}-ceiling{i}"
            surfaces.append(
                _horizontal(name, "ceiling", ceiling_y, outer, holes, up=False, extras=extras)
            )
    for wall in layout.get("walls", []):
        elevation = float(floors[wall["floorId"]]["elevation"])
        surfaces.extend(_wall_surfaces(wall, openings_by_wall.get(wall["id"], []), elevation))
    for obstacle in layout.get("obstacles", []):
        elevation = float(floors[obstacle["floorId"]]["elevation"])
        surfaces.extend(_obstacle_surfaces(obstacle, elevation))
    return surfaces


# --------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------


def _surface_bounds(surface: Surface) -> tuple[float, float, float, float]:
    u, v = surface.outer[:, 0], surface.outer[:, 1]
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def _shelf_pack(
    sizes: list[tuple[int, int]], names: list[str], page_size: int, max_pages: int
) -> tuple[list[tuple[int, int, int]], int] | None:
    """Shelf-pack charts (tallest first, ties by name) into pages; None when they do not fit."""
    slots: list[tuple[int, int, int]] = [(0, 0, 0)] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], names[i], i))
    page = x = y = shelf = 0
    for i in order:
        width, height = sizes[i]
        if width > page_size or height > page_size:
            return None
        if x + width > page_size:  # start a new shelf below the current one
            x, y, shelf = 0, y + shelf, 0
        if y + height > page_size:  # start a new page
            page, x, y, shelf = page + 1, 0, 0, 0
            if page >= max_pages:
                return None
        slots[i] = (page, x, y)
        x += width
        shelf = max(shelf, height)
    return slots, (page + 1 if sizes else 0)


def build_atlas(
    surfaces: Sequence[Surface],
    *,
    texel_size: float = 0.01,
    page_size: int = 2048,
    max_pages: int = 8,
    padding: int = 2,
) -> Atlas:
    """Pack each surface's chart into square pages, coarsening the texel size until it fits."""
    if texel_size <= 0 or page_size <= 0 or padding < 0:
        raise ValueError("atlas texel size and page size must be positive")
    bounds = [_surface_bounds(s) for s in surfaces]
    names = [s.name for s in surfaces]
    texel = texel_size
    for _ in range(ATLAS_RETRIES + 1):
        sizes = []
        for u_lo, v_lo, u_hi, v_hi in bounds:
            width = math.ceil((u_hi - u_lo) / texel - 1e-9) + 2 * padding
            height = math.ceil((v_hi - v_lo) / texel - 1e-9) + 2 * padding
            sizes.append((width, height))
        packed = _shelf_pack(sizes, names, page_size, max_pages)
        if packed is not None:
            slots, pages = packed
            placements = [
                ChartPlacement(
                    page=slots[i][0],
                    x=slots[i][1],
                    y=slots[i][2],
                    width=sizes[i][0],
                    height=sizes[i][1],
                    u_min=bounds[i][0],
                    v_max=bounds[i][3],
                )
                for i in range(len(surfaces))
            ]
            return Atlas(
                texel_size=texel,
                page_size=page_size,
                pages=pages,
                padding=padding,
                placements=placements,
            )
        texel *= ATLAS_GROWTH
    raise ValueError("layout surfaces do not fit the texture atlas")


def chart_grid(placement: ChartPlacement, atlas: Atlas) -> tuple[np.ndarray, np.ndarray]:
    """Surface (u, v) of every texel centre in a chart rectangle, each shaped (height, width)."""
    texel = atlas.texel_size
    cols = np.arange(placement.width) + 0.5 - atlas.padding
    rows = np.arange(placement.height) + 0.5 - atlas.padding
    u = placement.u_min + cols * texel
    v = placement.v_max - rows * texel
    return np.meshgrid(u, v)


def _in_ring(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd crossing test of (K, 2) points against one ring, vectorized over points."""
    px, pv = points[:, 0], points[:, 1]
    nxt = np.roll(ring, -1, axis=0)
    crossings = np.zeros(len(points), dtype=bool)
    for (ax, av), (bx, bv) in zip(ring, nxt, strict=True):
        straddle = (av > pv) != (bv > pv)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = ax + (pv - av) * (bx - ax) / (bv - av)
        crossings ^= straddle & (px < x_cross)
    return crossings


def chart_texels(
    surface: Surface, placement: ChartPlacement, atlas: Atlas
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texels of a chart rectangle (padding included), row by row.

    Returns ``(pixels, world, inside)``: ``pixels`` (K, 2) int ``[x, y]`` in the page,
    ``world`` (K, 3) texel centres in world metres, and ``inside`` (K,) which is True
    when the centre lies inside the outer ring and outside every hole.
    """
    u, v = chart_grid(placement, atlas)
    rows, cols = np.indices((placement.height, placement.width))
    pixels = np.column_stack([(placement.x + cols).ravel(), (placement.y + rows).ravel()]).astype(
        np.int64
    )
    uv = np.column_stack([u.ravel(), v.ravel()])
    inside = _in_ring(uv, surface.outer)
    for hole in surface.holes:
        inside ^= _in_ring(uv, hole)
    world = surface.origin + uv[:, :1] * surface.u_axis + uv[:, 1:] * surface.v_axis
    return pixels, world, inside


def surfaces_to_parts(surfaces: Sequence[Surface], atlas: Atlas) -> list[TexturedPart]:
    """Triangulate every surface, lift it to world space and give it atlas UVs."""
    parts: list[TexturedPart] = []
    for surface, placement in zip(surfaces, atlas.placements, strict=True):
        points, tris = triangulate(surface.outer, surface.holes)
        uv = np.asarray(points, dtype=np.float64)
        positions = surface.origin + uv[:, :1] * surface.u_axis + uv[:, 1:] * surface.v_axis
        tx = placement.x + atlas.padding + (uv[:, 0] - placement.u_min) / atlas.texel_size
        ty = placement.y + atlas.padding + (placement.v_max - uv[:, 1]) / atlas.texel_size
        uvs = np.column_stack([tx, ty]) / atlas.page_size
        # Counter-clockwise seen from the normal side: the geometric normal must agree.
        a, b, c = positions[tris[:, 0]], positions[tris[:, 1]], positions[tris[:, 2]]
        facing = np.einsum("ij,j->i", np.cross(b - a, c - a), surface.normal)
        flipped = tris.copy()
        flip = facing < 0
        flipped[flip] = flipped[flip][:, [0, 2, 1]]
        parts.append(
            TexturedPart(
                name=surface.name,
                kind=surface.kind,
                positions=positions,
                normals=np.tile(surface.normal, (len(positions), 1)),
                uvs=uvs,
                indices=flipped,
                page=placement.page,
                extras=dict(surface.extras),
            )
        )
    return parts
