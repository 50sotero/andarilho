"""Texture the layout's surfaces from registered video frames (README section 9).

The layout supplies the geometry (``surfaces.layout_surfaces``) and the atlas that
holds its colours. Each frame is a camera with known intrinsics and a pose in the
canonical world. For every surface texel the blend averages the frames that see it,
weighting each by how square and close the view is, and ignores frames where a nearer
surface hides the texel (checked against a depth map of the whole layout). The result
is one RGB image per atlas page, plus the fraction of each kind of surface that any
frame saw. Texels no frame sees keep ``unseen_rgb``, filled from their neighbours on
the same chart where possible. Furniture is the exception: a furniture texel that no
frame saw takes its part's colour in a fixed face shade, so an unseen piece never reads
grey.

``procedural_texture_layout`` paints the same atlas with deterministic patterns, for
synthetic layouts that have no frames.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from andarilho_worker.geometry.raster import depth_map
from andarilho_worker.geometry.surfaces import (
    Atlas,
    Surface,
    build_atlas,
    chart_grid,
    chart_texels,
    layout_surfaces,
    surfaces_to_parts,
)
from andarilho_worker.geometry.textured import SURFACE_KINDS, TexturedMesh, TexturedPart

LUMA = np.array([0.2126, 0.7152, 0.0722])
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BOARD_WIDTH = 0.18  # metres across one floor board
CHECKER_SIZE = 0.1  # metres per obstacle checker square
WOOD = (150.0, 102.0, 64.0)
PLASTER = (236.0, 228.0, 214.0)
CEILING_RGB = (246.0, 245.0, 242.0)
CHECKER_A = (196.0, 186.0, 170.0)
CHECKER_B = (150.0, 142.0, 130.0)
# Furniture faces are shaded by a fixed light from above and to the front, so an unseen
# face still reads as its part colour: 0.6 when it faces away from the light, 1.0 when it
# faces straight into it.
FURNITURE_LIGHT = np.array([0.35, 0.85, 0.4]) / np.linalg.norm([0.35, 0.85, 0.4])


@dataclass
class TextureFrame:
    name: str
    K: np.ndarray  # (3, 3) intrinsics in pixels
    R: np.ndarray  # (3, 3) world-to-camera rotation
    t: np.ndarray  # (3,) world-to-camera translation, canonical metres
    width: int
    height: int
    load: Callable[[], np.ndarray]  # (height, width, 3) uint8 RGB
    weight: float = 1.0


@dataclass(frozen=True)
class TextureParams:
    texel_size: float = 0.01
    page_size: int = 2048
    max_pages: int = 8
    padding: int = 2
    depth_scale: float = 0.25  # z-buffer resolution relative to the frame
    depth_tolerance_m: float = 0.03
    depth_tolerance_rel: float = 0.02
    edge_margin: float = 0.04  # fraction of the smaller image side for the weight ramp
    min_cos: float = 0.15
    weight_power: float = 4.0
    min_distance: float = 0.3
    near: float = 0.05
    hole_fill_texels: int = 6
    unseen_rgb: tuple[int, int, int] = (176, 176, 176)
    gain_clamp: tuple[float, float] = (0.6, 1.6)


# A frozen instance shared as the default (a call in a default argument would trip ruff B008).
DEFAULT_PARAMS = TextureParams()


def camera_in_world(
    R_colmap: np.ndarray,
    t_colmap: np.ndarray,
    world_from_colmap: Sequence[float],
    render_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A COLMAP world-to-camera pose expressed in canonical world metres.

    ``world_from_colmap`` is the 4x4 matrix (16 floats, column-major) that maps COLMAP
    world points ``X_c`` to canonical ``X = s * R_w @ X_c + t_w``, where ``s`` is the norm
    of the first column of its 3x3 block and ``R_w`` is that block divided by ``s``.

    The COLMAP camera gives ``x_c = R_c @ X_c + t_c``. Substituting ``s * X_c = R_w.T @
    (X - t_w)`` gives ``s * x_c = (R_c @ R_w.T) @ X + (s * t_c - R @ t_w)`` with
    ``R = R_c @ R_w.T``. So ``(R, t)`` maps a canonical point to the camera frame scaled by
    ``s``. A point projects to the same pixel either way because the projection divides
    by depth. A calibration ``render_scale`` c (world' = c * world about the origin) scales
    only the translation: ``t' = c * t``.
    """
    matrix = np.asarray(world_from_colmap, dtype=np.float64).reshape(4, 4, order="F")
    block = matrix[:3, :3]
    scale = float(np.linalg.norm(block[:, 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("world_from_colmap has no positive scale")
    r_world = block / scale
    t_world = matrix[:3, 3]
    R_c = np.asarray(R_colmap, dtype=np.float64).reshape(3, 3)
    t_c = np.asarray(t_colmap, dtype=np.float64).reshape(3)
    R = R_c @ r_world.T
    t = scale * t_c - R @ t_world
    return R, render_scale * t


# --------------------------------------------------------------------------
# Texel table
# --------------------------------------------------------------------------


@dataclass
class _ChartTable:
    """Every texel of every chart (padding included), in chart order."""

    surface: np.ndarray  # (M,) surface index
    page: np.ndarray  # (M,) atlas page
    pixel: np.ndarray  # (M, 2) [x, y] in the page
    inside: np.ndarray  # (M,) True for texels on the surface itself
    world: np.ndarray  # (M, 3) texel centres, metres


def _chart_table(surfaces, atlas: Atlas) -> _ChartTable:
    surface_ids, pages, pixels, inside, world = [], [], [], [], []
    for k, (surface, placement) in enumerate(zip(surfaces, atlas.placements, strict=True)):
        pix, pts, ins = chart_texels(surface, placement, atlas)
        surface_ids.append(np.full(len(pix), k, dtype=np.int64))
        pages.append(np.full(len(pix), placement.page, dtype=np.int64))
        pixels.append(pix)
        inside.append(ins)
        world.append(pts)
    if not surface_ids:
        return _ChartTable(
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            np.zeros((0, 2), np.int64),
            np.zeros(0, bool),
            np.zeros((0, 3)),
        )
    return _ChartTable(
        surface=np.concatenate(surface_ids),
        page=np.concatenate(pages),
        pixel=np.concatenate(pixels),
        inside=np.concatenate(inside),
        world=np.concatenate(world),
    )


def _world_triangles(parts: Sequence[TexturedPart]) -> np.ndarray:
    if not parts:
        return np.zeros((0, 3, 3))
    return np.concatenate([p.positions[p.indices] for p in parts])


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _pose(frame: TextureFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(frame.K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(frame.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(frame.t, dtype=np.float64).reshape(3)
    return K, R, t


def _depth_shape(frame: TextureFrame, params: TextureParams) -> tuple[int, int]:
    return (
        max(1, round(frame.height * params.depth_scale)),
        max(1, round(frame.width * params.depth_scale)),
    )


def _frame_depth(frame: TextureFrame, tris: np.ndarray, params: TextureParams) -> np.ndarray:
    dh, dw = _depth_shape(frame, params)
    K, R, t = _pose(frame)
    scaled_K = np.diag([dw / frame.width, dh / frame.height, 1.0]) @ K
    return depth_map(tris, R, t, scaled_K, dw, dh, near=params.near)


def _load(frame: TextureFrame) -> np.ndarray:
    image = np.asarray(frame.load())
    expected = (frame.height, frame.width, 3)
    if image.shape != expected:
        raise ValueError(f"frame {frame.name!r} image is {image.shape}, expected {expected}")
    # Colours are blended as 0..255 values; a float image in 0..1 would render near-black.
    if image.dtype != np.uint8:
        raise ValueError(f"frame {frame.name!r} image is {image.dtype}, expected uint8 RGB")
    return image


@dataclass
class _View:
    index: np.ndarray  # positions into the candidate texel arrays
    weight: np.ndarray  # (M,) float64, positive
    u: np.ndarray  # continuous pixel coordinates, COLMAP convention
    v: np.ndarray


def _view(
    world: np.ndarray,
    normal: np.ndarray,
    frame: TextureFrame,
    depth: np.ndarray,
    params: TextureParams,
) -> _View:
    """The texels a frame sees, with their weights and pixel positions."""
    K, R, t = _pose(frame)
    cam = world @ R.T + t
    z = cam[:, 2]
    ahead = np.flatnonzero(z > params.near)
    x, y, z = cam[ahead, 0], cam[ahead, 1], z[ahead]
    u = (K[0, 0] * x + K[0, 1] * y) / z + K[0, 2]
    v = (K[1, 0] * x + K[1, 1] * y) / z + K[1, 2]

    width, height = frame.width, frame.height
    edge = np.minimum(np.minimum(u, width - u), np.minimum(v, height - v))
    ramp = np.clip(edge / (params.edge_margin * min(width, height)), 0.0, 1.0)

    dh, dw = depth.shape
    col = np.clip(np.floor(u * (dw / width)).astype(np.intp), 0, dw - 1)
    row = np.clip(np.floor(v * (dh / height)).astype(np.intp), 0, dh - 1)
    zbuf = depth[row, col].astype(np.float64)  # inf where nothing is drawn: visible
    visible = z <= zbuf * (1.0 + params.depth_tolerance_rel) + params.depth_tolerance_m

    keep = np.flatnonzero(visible & (ramp > 0.0))
    idx = ahead[keep]
    centre = -R.T @ t
    d = world[idx] - centre
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    cos = -np.einsum("ij,ij->i", normal[idx], d) / dist
    weight = (
        (cos**2 / np.maximum(dist, params.min_distance) ** 2) * ramp[keep] * frame.weight
    ) ** params.weight_power
    usable = (cos >= params.min_cos) & (weight > 0.0)
    keep = keep[usable]
    return _View(
        index=ahead[keep],
        weight=weight[usable],
        u=u[keep],
        v=v[keep],
    )


def _sample(image: np.ndarray, uc: np.ndarray, vc: np.ndarray) -> np.ndarray:
    """Bilinear RGB samples (0..255) at continuous pixel-index coordinates."""
    height, width = image.shape[:2]
    c0_raw = np.floor(uc)
    r0_raw = np.floor(vc)
    fx = (uc - c0_raw)[:, None]
    fy = (vc - r0_raw)[:, None]
    c0_raw = c0_raw.astype(np.intp)
    r0_raw = r0_raw.astype(np.intp)
    c0 = np.clip(c0_raw, 0, width - 1)
    c1 = np.clip(c0_raw + 1, 0, width - 1)
    r0 = np.clip(r0_raw, 0, height - 1)
    r1 = np.clip(r0_raw + 1, 0, height - 1)
    top = image[r0, c0] * (1.0 - fx) + image[r0, c1] * fx
    bottom = image[r1, c0] * (1.0 - fx) + image[r1, c1] * fx
    return top * (1.0 - fy) + bottom * fy


def _gain(view: _View, rgb: np.ndarray, blend: np.ndarray, params: TextureParams) -> float:
    """Exposure of one frame relative to the pass-A blend, on its best-weighted texels."""
    frame_luma = rgb @ LUMA
    blend_luma = blend @ LUMA
    chosen = view.weight > np.median(view.weight)
    if not chosen.any():
        chosen = np.ones(view.weight.size, dtype=bool)
    ratio = blend_luma[chosen] / np.maximum(frame_luma[chosen], 1.0)
    return float(np.clip(np.median(ratio), *params.gain_clamp))


def _report(progress: Callable[[int, int], None] | None, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def _neighbour_mean(
    target: np.ndarray,
    source: np.ndarray,
    index_map: np.ndarray,
    ly: np.ndarray,
    lx: np.ndarray,
    chart: np.ndarray,
    colour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour of each target's 4-neighbours that are sources of the same chart."""
    height, width = index_map.shape
    rows, cols = ly[target], lx[target]
    total = np.zeros((target.size, 3))
    count = np.zeros(target.size)
    for dy, dx in DIRECTIONS:
        nr, nc = rows + dy, cols + dx
        in_page = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        nb = index_map[np.clip(nr, 0, height - 1), np.clip(nc, 0, width - 1)].astype(np.intp)
        ok = in_page & (nb >= 0)
        nb = np.where(ok, nb, 0)
        ok &= source[nb] & (chart[nb] == chart[target])
        total += np.where(ok[:, None], colour[nb], 0.0)
        count += ok
    return count > 0, total / np.maximum(count, 1.0)[:, None]


def _paint_pages(
    atlas: Atlas,
    table: _ChartTable,
    colour: np.ndarray,
    seen: np.ndarray,
    params: TextureParams,
) -> list[np.ndarray]:
    """Fill holes and padding page by page, then write the RGB pages."""
    size = atlas.page_size
    unseen = np.asarray(params.unseen_rgb, dtype=np.uint8)
    pages: list[np.ndarray] = []
    for page_index in range(atlas.pages):
        image = np.empty((size, size, 3), dtype=np.uint8)
        image[:] = unseen
        sel = np.flatnonzero(table.page == page_index)
        if sel.size:
            xs, ys = table.pixel[sel, 0], table.pixel[sel, 1]
            x0, y0 = int(xs.min()), int(ys.min())
            height = int(ys.max()) - y0 + 1
            width = int(xs.max()) - x0 + 1
            ly, lx = ys - y0, xs - x0
            index_map = np.full((height, width), -1, dtype=np.int32)
            index_map[ly, lx] = np.arange(sel.size, dtype=np.int32)
            chart = table.surface[sel]
            inside = table.inside[sel]
            paint = colour[sel].copy()
            filled = seen[sel].copy()
            # Holes on the surface: average the nearest filled neighbours, a few rounds.
            for _ in range(params.hole_fill_texels):
                target = np.flatnonzero(inside & ~filled)
                if target.size == 0:
                    break
                hit, mean = _neighbour_mean(target, filled, index_map, ly, lx, chart, paint)
                paint[target[hit]] = mean[hit]
                filled[target[hit]] = True
            # Padding and notches: dilate the chart's colours outwards so mipmaps stay on-chart.
            painted = inside.copy()
            for _ in range(atlas.padding):
                target = np.flatnonzero(~painted)
                if target.size == 0:
                    break
                hit, mean = _neighbour_mean(target, painted, index_map, ly, lx, chart, paint)
                paint[target[hit]] = mean[hit]
                painted[target[hit]] = True
            image[ys, xs] = np.clip(np.rint(paint), 0, 255).astype(np.uint8)
        pages.append(image)
    return pages


def _furniture_rgb(surface: Surface) -> np.ndarray:
    """A furniture face's part colour in its fixed shade, as 0..255 floats."""
    facing = max(0.0, float(surface.normal @ FURNITURE_LIGHT))
    shade = 0.6 + 0.4 * facing
    colour = np.asarray(surface.extras["color"], dtype=np.float64) * shade
    return np.clip(np.rint(colour), 0.0, 255.0)


def _coverage(surfaces, surface_of_inside: np.ndarray, seen: np.ndarray) -> dict[str, float]:
    count = len(surfaces)
    inside_per = np.bincount(surface_of_inside, minlength=count).astype(np.float64)
    seen_per = np.bincount(surface_of_inside, weights=seen.astype(np.float64), minlength=count)
    coverage: dict[str, float] = {}
    for kind in SURFACE_KINDS:
        mask = np.array([s.kind == kind for s in surfaces], dtype=bool)
        if not mask.any():
            continue
        total = inside_per[mask].sum()
        coverage[kind] = float(seen_per[mask].sum() / total) if total > 0 else 0.0
    # "all" is the architecture only, so the warning about unseen walls, floor and ceiling is
    # not diluted by furniture, which has its own entry.
    architecture = np.array([s.kind != "furniture" for s in surfaces], dtype=bool)
    total = inside_per[architecture].sum()
    seen_total = seen_per[architecture].sum()
    coverage["all"] = float(seen_total / total) if total > 0 else 0.0
    return coverage


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _prepare(
    layout: Mapping[str, Any], params: TextureParams
) -> tuple[list[Surface], Atlas, list[TexturedPart]]:
    """The layout's surfaces, their atlas and the UV-mapped parts (shared by both entry points)."""
    surfaces = layout_surfaces(layout)
    atlas = build_atlas(
        surfaces,
        texel_size=params.texel_size,
        page_size=params.page_size,
        max_pages=params.max_pages,
        padding=params.padding,
    )
    return surfaces, atlas, surfaces_to_parts(surfaces, atlas)


def texture_layout(
    layout: Mapping[str, Any],
    frames: Sequence[TextureFrame],
    params: TextureParams = DEFAULT_PARAMS,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> TexturedMesh:
    """Blend the frames onto the layout's atlas.

    Pass A visits each frame and accumulates weighted colours per texel. Pass B revisits
    each frame, scales it by an exposure gain measured against the pass-A blend, and
    accumulates the final blend. ``progress(done, total)`` is called once per frame in
    each pass, with ``total = 2 * len(frames)``.
    """
    surfaces, atlas, parts = _prepare(layout, params)
    table = _chart_table(surfaces, atlas)
    cand = np.flatnonzero(table.inside)  # the texels on the surfaces
    world = table.world[cand]
    normals = np.array([s.normal for s in surfaces], dtype=np.float64).reshape(-1, 3)
    normal = normals[table.surface[cand]]
    tris = _world_triangles(parts)

    total = 2 * len(frames)
    count = cand.size
    sum_w = np.zeros(count)
    sum_rgb = np.zeros((count, 3))
    # Pass B reuses these depth maps rather than rasterizing every frame twice. Each map holds
    # depth_scale**2 of the frame's pixels as float32 (about 0.5 MB at 1920 x 1080), so the
    # cache grows with the number of frames.
    depths: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        _report(progress, i, total)
        depth = _frame_depth(frame, tris, params)
        depths.append(depth)
        image = _load(frame)
        view = _view(world, normal, frame, depth, params)
        if view.index.size:
            rgb = _sample(image, view.u - 0.5, view.v - 0.5)
            sum_w[view.index] += view.weight
            sum_rgb[view.index] += view.weight[:, None] * rgb
        del image

    blend = sum_rgb / np.where(sum_w > 0, sum_w, 1.0)[:, None]
    final_w = np.zeros(count)
    final_rgb = np.zeros((count, 3))
    for i, frame in enumerate(frames):
        _report(progress, len(frames) + i, total)
        view = _view(world, normal, frame, depths[i], params)
        if view.index.size == 0:
            continue
        image = _load(frame)
        rgb = _sample(image, view.u - 0.5, view.v - 0.5)
        gain = _gain(view, rgb, blend[view.index], params)
        final_w[view.index] += view.weight
        final_rgb[view.index] += (view.weight * gain)[:, None] * rgb
        del image

    seen_c = final_w > 0
    colour = np.tile(np.asarray(params.unseen_rgb, dtype=np.float64), (len(table.inside), 1))
    seen = np.zeros(len(table.inside), dtype=bool)
    seen[cand] = seen_c
    colour[cand[seen_c]] = np.clip(final_rgb[seen_c] / final_w[seen_c, None], 0.0, 255.0)
    # Furniture no frame saw starts from its part colour in shade. _paint_pages overwrites
    # these texels with their neighbours' colours wherever a seen texel is close enough.
    furniture = np.array([s.kind == "furniture" for s in surfaces], dtype=bool)
    surface_rgb = np.zeros((len(surfaces), 3))
    for k in np.flatnonzero(furniture):
        surface_rgb[k] = _furniture_rgb(surfaces[k])
    unseen_furniture = table.inside & ~seen & furniture[table.surface]
    colour[unseen_furniture] = surface_rgb[table.surface[unseen_furniture]]

    coverage = _coverage(surfaces, table.surface[cand], seen_c)
    pages = _paint_pages(atlas, table, colour, seen, params)
    return TexturedMesh(
        parts=parts,
        pages=pages,
        coverage=coverage,
        texel_size=atlas.texel_size,
    )


def _procedural_rgb(
    kind: str, u: np.ndarray, v: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic RGB (0..255) for a chart grid of surface coordinates."""
    shape = u.shape
    if kind == "floor":
        board = np.floor(v / BOARD_WIDTH)
        _, index = np.unique(board, return_inverse=True)
        tones = rng.uniform(-0.07, 0.07, size=int(index.max()) + 1)
        across = v / BOARD_WIDTH - board
        seam = np.where((across < 0.03) | (across > 0.97), 0.8, 1.0)
        grain = 0.025 * np.sin(2 * np.pi * (u / 0.07 + 0.5 * np.sin(2 * np.pi * v / 0.9)))
        shade = (1.0 + tones[index.reshape(shape)] + grain) * seam
        return shade[..., None] * np.asarray(WOOD)
    if kind == "obstacle":
        parity = (np.floor(u / CHECKER_SIZE) + np.floor(v / CHECKER_SIZE)).astype(np.int64) % 2
        return np.where(parity[..., None] == 0, np.asarray(CHECKER_A), np.asarray(CHECKER_B))
    if kind == "ceiling":
        return np.asarray(CEILING_RGB) + rng.uniform(-1.5, 1.5, size=shape + (1,))
    # wall, sill, lintel and closed opening: warm off-white plaster
    return np.asarray(PLASTER) + rng.uniform(-4.0, 4.0, size=shape + (1,))


def procedural_texture_layout(
    layout: Mapping[str, Any], params: TextureParams = DEFAULT_PARAMS
) -> TexturedMesh:
    """The layout's atlas painted with deterministic per-kind patterns (coverage 1.0)."""
    surfaces, atlas, parts = _prepare(layout, params)
    pages = [
        np.empty((atlas.page_size, atlas.page_size, 3), dtype=np.uint8) for _ in range(atlas.pages)
    ]
    for page in pages:
        page[:] = np.asarray(params.unseen_rgb, dtype=np.uint8)
    for surface, placement in zip(surfaces, atlas.placements, strict=True):
        u, v = chart_grid(placement, atlas)
        h, w = u.shape
        if surface.kind == "furniture":
            rgb = np.broadcast_to(_furniture_rgb(surface), (h, w, 3))
        else:
            rng = np.random.default_rng(zlib.crc32(surface.name.encode("utf-8")))
            rgb = _procedural_rgb(surface.kind, u, v, rng)
        y0, x0 = placement.y, placement.x
        pages[placement.page][y0 : y0 + h, x0 : x0 + w] = np.clip(np.rint(rgb), 0, 255).astype(
            np.uint8
        )
    coverage = {s.kind: 1.0 for s in surfaces}
    coverage["all"] = 1.0
    return TexturedMesh(
        parts=parts,
        pages=pages,
        coverage=coverage,
        texel_size=atlas.texel_size,
    )
