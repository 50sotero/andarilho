"""Frame texturing (geometry.texture) against a test-only ray caster with known colours."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from andarilho_worker.geometry import texture as texture_module
from andarilho_worker.geometry.layout import floor_supports, rectangle_layout
from andarilho_worker.geometry.surfaces import (
    Atlas,
    ChartPlacement,
    Surface,
    build_atlas,
    chart_grid,
    chart_texels,
    layout_surfaces,
)
from andarilho_worker.geometry.texture import (
    TextureFrame,
    TextureParams,
    camera_in_world,
    procedural_texture_layout,
    texture_layout,
)
from andarilho_worker.geometry.textured import TexturedMesh

W, H = 160, 120
FX = FY = 110.0
CX, CY = 80.0, 60.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
UNSEEN = np.array([176, 176, 176])
PALETTE = np.array(
    [(200, 70, 60), (60, 160, 90), (70, 110, 210), (210, 160, 60), (150, 70, 190), (60, 180, 190)],
    dtype=np.float64,
)
EDGE = 3  # pixels at the image border that the truth masks leave out
MAE_BOUND = 6.0  # mean absolute colour error on truly seen texels (0..255)
# Camera pitches (radians, positive looks up): some look down so the floor is in view.
PITCHES = (-0.1, 0.25, -0.55, -0.1, 0.25, -0.5, -0.1, 0.25, -0.6, -0.1, 0.25, -0.45)


# --------------------------------------------------------------------------
# Test-only renderer: analytic plane hits against the surfaces, nearest wins
# --------------------------------------------------------------------------


CHECKER_PHASE = 0.07  # metres; keeps every face plane 0.02 m or more from a checker line


def truth(points: np.ndarray, surface_index: int) -> np.ndarray:
    """Ground-truth colour: a base colour per surface with a 0.2 m checker in world space.

    A point exactly on a checker line takes its parity from float rounding, so the checker is
    offset by CHECKER_PHASE to keep the room's walls, floor and ceiling off the lines.
    """
    cells = np.floor((points + CHECKER_PHASE) / 0.2).astype(np.int64).sum(axis=1) % 2
    shade = np.where(cells == 0, 1.0, 0.55)
    return shade[:, None] * PALETTE[surface_index % len(PALETTE)]


def _even_odd(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    px, pz = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    n = len(ring)
    for i in range(n):
        ax, az = ring[i]
        bx, bz = ring[(i + 1) % n]
        straddle = (az > pz) != (bz > pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = ax + (pz - az) * (bx - ax) / (bz - az)
        inside ^= straddle & (px < x_cross)
    return inside


def _cast(surfaces, origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_t = np.full(len(dirs), np.inf)
    best_id = np.full(len(dirs), -1)
    for k, s in enumerate(surfaces):
        denom = dirs @ s.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((s.origin - origin) @ s.normal) / denom
        candidates = np.flatnonzero((np.abs(denom) > 1e-12) & (t > 1e-6) & (t < best_t))
        if candidates.size == 0:
            continue
        points = origin + t[candidates, None] * dirs[candidates]
        rel = points - s.origin
        uv = np.column_stack([rel @ s.u_axis, rel @ s.v_axis])
        hit = _even_odd(uv, s.outer)
        for hole in s.holes:
            hit &= ~_even_odd(uv, hole)
        chosen = candidates[hit]
        best_t[chosen] = t[chosen]
        best_id[chosen] = k
    return best_t, best_id


def _look(eye, yaw: float, pitch: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (R, t) for a camera at ``eye``; yaw 0 looks down -Z, pitch up is positive."""
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    t = -R @ np.asarray(eye, dtype=np.float64)
    return R, t


def _trace(surfaces, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: the nearest surface's colour (H, W, 3) and its surface index (H, W)."""
    eye = -R.T @ t
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs_cam = np.stack([(cols - CX) / FX, (rows - CY) / FY, np.ones_like(cols)], axis=-1)
    dirs = dirs_cam.reshape(-1, 3) @ R  # R.T applied to each direction
    t_hit, ids = _cast(surfaces, eye, dirs)
    hit = np.flatnonzero(ids >= 0)
    points = np.full((H * W, 3), np.nan)
    points[hit] = eye + t_hit[hit, None] * dirs[hit]
    image = np.zeros((H * W, 3))
    for k in np.unique(ids[hit]):
        sel = hit[ids[hit] == k]
        image[sel] = truth(points[sel], int(k))
    rgb = np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(H, W, 3)
    return rgb, ids.reshape(H, W)


@dataclass
class _Shot:
    frame: TextureFrame
    ids: np.ndarray  # (H, W) surface each pixel's ray hits first, -1 for none


def _frame(surfaces, name: str, eye, yaw: float, pitch: float = 0.0, *, scale: float = 1.0):
    """A rendered camera; ``scale`` multiplies the exposure of the image it delivers."""
    R, t = _look(eye, yaw, pitch)
    rgb, ids = _trace(surfaces, R, t)
    if scale != 1.0:
        rgb = np.clip(np.rint(rgb * scale), 0, 255).astype(np.uint8)
    frame = TextureFrame(name=name, K=K, R=R, t=t, width=W, height=H, load=lambda: rgb)
    return _Shot(frame, ids)


def _room() -> dict[str, Any]:
    return rectangle_layout((-1.5, -1.25), (1.5, 1.25), elevation=0.0, height=2.6)


def _furnished_room() -> dict[str, Any]:
    """``_room`` with an open door and an open window, and a box standing in the room.

    The window is open because a closed pane fills the opening from the sill up, so no ray from
    inside the room reaches the sill's top.
    """
    layout = _room()
    layout["openings"] = [
        {
            "id": "door",
            "wallId": "wall-north",
            "offset": 0.9,
            "width": 0.8,
            "bottom": 0.0,
            "top": 2.1,
            "state": "open",
            "provenance": "user",
        },
        {
            "id": "window",
            "wallId": "wall-east",
            "offset": 0.6,
            "width": 0.9,
            "bottom": 0.9,
            "top": 1.8,
            "state": "open",
            "provenance": "user",
        },
    ]
    layout["obstacles"] = [
        {
            "id": "box",
            "floorId": "floor-0",
            "footprint": [[-0.9, 0.3], [-0.4, 0.3], [-0.4, 0.8], [-0.9, 0.8]],
            "height": 0.8,
            "provenance": "user",
        }
    ]
    return layout


def _big_room_frames(surfaces) -> list[TextureFrame]:
    """Eight cameras near the middle of a 12 x 10 m room, so every wall is 4.5 m or more away."""
    frames = []
    for k in range(8):
        yaw = np.deg2rad(45.0 * k)
        eye = [0.5 * np.cos(yaw + 0.7), 1.4, 0.5 * np.sin(yaw + 0.7)]
        frames.append(_frame(surfaces, f"hall-{k}", eye, yaw, pitch=-0.2).frame)
    return frames


def _aimed(surfaces, name: str, eye: Sequence[float], target: Sequence[float]) -> _Shot:
    """A rendered camera at ``eye`` whose optical axis points straight at ``target``."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    yaw = float(np.arctan2(d[0], -d[2]))
    pitch = float(np.arctan2(d[1], np.hypot(d[0], d[2])))
    return _frame(surfaces, name, eye, yaw, pitch)


def _room_shots(surfaces, count: int = 12, scales: Sequence[float] | None = None) -> list[_Shot]:
    shots = []
    for k in range(count):
        yaw = np.deg2rad(360.0 * k / count)
        offset = 0.25 * np.array([np.cos(yaw + 1.0), 0.0, np.sin(yaw + 1.0)])
        pitch = PITCHES[k % len(PITCHES)]
        scale = 1.0 if scales is None else scales[k]
        shots.append(
            _frame(surfaces, f"cam-{k}", [offset[0], 1.4, offset[2]], yaw, pitch, scale=scale)
        )
    return shots


def _room_frames(surfaces) -> list[TextureFrame]:
    return [shot.frame for shot in _room_shots(surfaces)]


# --------------------------------------------------------------------------
# Ground truth per chart texel, from the test renderer rather than from colours
# --------------------------------------------------------------------------


@dataclass
class _Chart:
    surface: int  # index into the surfaces
    kind: str
    page: int
    pixels: np.ndarray  # (M, 2) [x, y] in the page, chart order
    world: np.ndarray  # (M, 3) texel centres in metres
    inside: np.ndarray  # (M,) texel centre on the surface itself
    seen: np.ndarray  # (M,) some shot's first hit at the texel's pixel is this texel's surface


def _charts(surfaces, atlas, shots: Sequence[_Shot]) -> list[_Chart]:
    """Which chart texels each shot sees directly, with a margin from the image border."""
    charts = []
    for k, (surface, placement) in enumerate(zip(surfaces, atlas.placements, strict=True)):
        pixels, world, inside = chart_texels(surface, placement, atlas)
        seen = np.zeros(len(world), dtype=bool)
        cand = np.flatnonzero(inside)
        for shot in shots:
            cam = world[cand] @ shot.frame.R.T + shot.frame.t
            front = np.flatnonzero(cam[:, 2] > 0.05)
            z = cam[front, 2]
            u = FX * cam[front, 0] / z + CX  # COLMAP continuous pixel coordinates
            v = FY * cam[front, 1] / z + CY
            col = np.floor(u).astype(np.intp)
            row = np.floor(v).astype(np.intp)
            inner = (col >= EDGE) & (col < W - EDGE) & (row >= EDGE) & (row < H - EDGE)
            front, col, row = front[inner], col[inner], row[inner]
            # Planar surfaces: when a pixel's first hit is this surface, the texels projecting
            # into that pixel are unoccluded to within the pixel's footprint.
            seen[cand[front[shot.ids[row, col] == k]]] = True
        charts.append(_Chart(k, surface.kind, placement.page, pixels, world, inside, seen))
    return charts


def _colours(mesh: TexturedMesh, chart: _Chart, mask: np.ndarray) -> np.ndarray:
    px = chart.pixels[mask]
    return mesh.pages[chart.page][px[:, 1], px[:, 0]].astype(np.float64)


def _truth_coverage(charts: Sequence[_Chart]) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for chart in charts:
        tally = counts.setdefault(chart.kind, [0, 0])
        tally[0] += int(chart.inside.sum())
        tally[1] += int((chart.inside & chart.seen).sum())
    return {kind: seen / total for kind, (total, seen) in counts.items() if total}


@dataclass
class _Run:
    layout: dict[str, Any]
    surfaces: list
    mesh: TexturedMesh
    elapsed: float
    charts: list[_Chart]
    frames: list[TextureFrame]


def _texture_run(layout: dict[str, Any], shots: list[_Shot]) -> _Run:
    """Texture a layout from the given rendered cameras and chart the truth."""
    surfaces = layout_surfaces(layout)
    frames = [shot.frame for shot in shots]
    start = time.perf_counter()
    mesh = texture_layout(layout, frames)
    elapsed = time.perf_counter() - start
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)
    return _Run(layout, surfaces, mesh, elapsed, _charts(surfaces, atlas, shots), frames)


@pytest.fixture(scope="module")
def room_run() -> _Run:
    layout = _room()
    return _texture_run(layout, _room_shots(layout_surfaces(layout)))


@pytest.fixture(scope="module")
def furnished_run() -> _Run:
    layout = _furnished_room()
    surfaces = layout_surfaces(layout)
    # Close cameras at the window and the door: from the standard ring a 10 cm sill or lintel spans
    # only a few pixels, so its colour aliases. Each close camera looks straight at its face.
    close = [
        _aimed(surfaces, "window-sill", [0.85, 1.4, -0.2], [1.5, 0.9, -0.2]),
        _aimed(surfaces, "window-lintel", [0.85, 1.0, -0.2], [1.5, 1.8, -0.2]),
        _aimed(surfaces, "door-lintel", [-0.2, 1.4, -0.85], [-0.2, 2.1, -1.25]),
    ]
    return _texture_run(layout, _room_shots(surfaces) + close)


@pytest.fixture(scope="module")
def room_prefill(room_run: _Run) -> TexturedMesh:
    # The same frames blended without hole filling: only texels a frame saw are coloured.
    return texture_layout(room_run.layout, room_run.frames, TextureParams(hole_fill_texels=0))


def _face_errors(run: _Run, select: Callable[[Surface], bool]) -> np.ndarray:
    """Absolute colour errors on truly seen texels of the selected faces, shape (M, 3)."""
    parts = []
    for k, (surface, chart) in enumerate(zip(run.surfaces, run.charts, strict=True)):
        if not select(surface):
            continue
        mask = chart.inside & chart.seen
        colours = _colours(run.mesh, chart, mask)
        parts.append(np.abs(colours - truth(chart.world[mask], k)))
    return np.concatenate(parts)


def _kind_errors(run: _Run, kind: str) -> np.ndarray:
    """Absolute colour errors on truly seen texels of one kind, shape (M, 3)."""
    return _face_errors(run, lambda s: s.kind == kind)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_truly_seen_texels_reproduce_ground_truth(room_run):
    errors, painted = [], []
    for k, chart in enumerate(room_run.charts):
        mask = chart.inside & chart.seen
        colours = _colours(room_run.mesh, chart, mask)
        errors.append(np.abs(colours - truth(chart.world[mask], k)))
        painted.append(np.any(colours != UNSEEN, axis=1))
    errors = np.concatenate(errors)
    painted = np.concatenate(painted)
    assert errors.size > 200_000
    assert painted.mean() > 0.99
    assert errors.mean() < MAE_BOUND


def test_reported_coverage_is_not_below_direct_visibility(room_run):
    # Every texel the ray caster sees must count as seen. The blend may count a little more:
    # its depth buffer is 4x coarser than the frame, so texels within one depth cell of an
    # occluding edge can pass the visibility test (a documented trade-off of depth_scale).
    truth_cov = _truth_coverage(room_run.charts)
    for kind in ("wall", "floor", "ceiling"):
        assert room_run.mesh.coverage[kind] >= truth_cov[kind] - 0.01, kind
        assert room_run.mesh.coverage[kind] <= truth_cov[kind] + 0.15, kind


def _faces_into_room(layout, surface) -> bool:
    """True for the long wall face on the room side (the one a camera inside can see)."""
    if surface.kind != "wall" or not surface.name.endswith((":a", ":b")):
        return False
    centre_uv = surface.outer.mean(axis=0)
    centre = surface.origin + centre_uv[0] * surface.u_axis + centre_uv[1] * surface.v_axis
    probe = centre + 0.02 * surface.normal
    return floor_supports(layout["floors"][0], (probe[0], probe[2]))


def test_walls_the_cameras_faced_are_covered(room_run, room_prefill):
    # The walls the cameras faced are the meaningful measure: the kind-level "wall" figure also
    # holds caps and walls behind the cameras. They are counted before hole filling, so colours
    # copied in from neighbours do not count as seen.
    faced_seen = faced_total = truth_seen = 0
    for surface, chart in zip(room_run.surfaces, room_run.charts, strict=True):
        if not _faces_into_room(room_run.layout, surface):
            continue
        colours = _colours(room_prefill, chart, chart.inside)
        faced_total += len(colours)
        faced_seen += int(np.any(colours != UNSEEN, axis=1).sum())
        truth_seen += int((chart.inside & chart.seen).sum())
    assert faced_total > 100_000
    blend = faced_seen / faced_total
    assert blend > 0.6
    # The blend may see a little more than direct visibility (see the coverage test), but a
    # blend that drops texels a ray caster sees directly is broken.
    assert blend >= truth_seen / faced_total - 0.01
    assert set(room_run.mesh.coverage) >= {"wall", "floor", "ceiling", "all"}
    assert all(0.0 <= value <= 1.0 for value in room_run.mesh.coverage.values())


def _divided_hall() -> dict:
    """The 12 x 10 m room with a full-height divider 4.5 m east of its middle."""
    layout = rectangle_layout((-6.0, -5.0), (6.0, 5.0), elevation=0.0, height=2.5)
    divider = dict(layout["walls"][0], id="wall-divider", start=[4.5, -5.0], end=[4.5, 5.0])
    layout["walls"].append(divider)
    return layout


def test_divider_face_turned_away_from_every_camera_stays_unseen():
    # The divider has floor on both sides, so both long faces are textured, but every camera is
    # on its west side. The east face lies 0.1 m behind the west one, and from 3.9 m or more that
    # gap falls inside the depth tolerance (2% of the depth plus 3 cm), so the depth test alone
    # would let it through. The facing gate (cos >= min_cos) keeps it unseen: with the gate
    # open, the same frames paint it.
    layout = _divided_hall()
    surfaces = layout_surfaces(layout)
    frames = _big_room_frames(surfaces)
    params = TextureParams(texel_size=0.02)
    gated = texture_layout(layout, frames, params)
    open_gate = texture_layout(layout, frames, dataclasses.replace(params, min_cos=-1.0))
    atlas = build_atlas(surfaces, texel_size=gated.texel_size)

    far_total = far_painted = near_total = near_seen = 0
    for surface, placement in zip(surfaces, atlas.placements, strict=True):
        if surface.extras.get("wallId") != "wall-divider" or surface.kind != "wall":
            continue
        if not surface.name.endswith((":a", ":b")):
            continue
        pixels, _, inside = chart_texels(surface, placement, atlas)
        px = pixels[inside]
        gated_rgb = gated.pages[placement.page][px[:, 1], px[:, 0]]
        open_rgb = open_gate.pages[placement.page][px[:, 1], px[:, 0]]
        if surface.normal[0] > 0:  # faces east, away from every camera
            assert np.all(gated_rgb == UNSEEN), surface.name
            far_total += len(px)
            far_painted += int(np.any(open_rgb != UNSEEN, axis=1).sum())
        else:
            near_total += len(px)
            near_seen += int(np.any(gated_rgb != UNSEEN, axis=1).sum())
    # The west face must be seen, so the unseen east face is not an empty atlas; and the open
    # gate must paint the east face, so it is the gate that keeps it unseen.
    assert near_seen > 0.9 * near_total
    assert far_painted > 0.5 * far_total
    assert far_total > 50_000


def test_floor_is_in_view_and_reproduced(room_run):
    assert room_run.mesh.coverage["floor"] > 0.5
    errors = _kind_errors(room_run, "floor")
    assert errors.size > 50_000
    assert errors.mean() < MAE_BOUND


def test_sills_lintels_and_obstacle_faces_reproduce_ground_truth(furnished_run):
    # The sill top and the lintel undersides are thin strips of wall and the box's faces are
    # small, so each group is measured on its own rather than through the kind-level figures.
    groups = {
        "sill top": lambda s: s.kind == "sill" and s.name.endswith(":top"),
        "lintel underside": lambda s: s.kind == "lintel" and s.name.endswith(":bottom"),
        "obstacle": lambda s: s.kind == "obstacle",
    }
    for label, select in groups.items():
        errors = _face_errors(furnished_run, select)
        assert errors.size > 300, label
        assert errors.mean() < MAE_BOUND, label


def test_texture_layout_runs_quickly(room_run):
    assert room_run.elapsed < 30.0


def test_thirty_frames_over_the_room_finish_within_budget():
    # Spec budget: about 400k texels x 30 frames in well under 60 s; this room has ~0.7M.
    layout = _room()
    surfaces = layout_surfaces(layout)
    frames = [shot.frame for shot in _room_shots(surfaces, count=30)]
    start = time.perf_counter()
    mesh = texture_layout(layout, frames)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0
    assert mesh.coverage["all"] > 0.5


def test_texture_layout_reports_progress_once_per_pass():
    layout = _room()
    surfaces = layout_surfaces(layout)
    frames = _room_frames(surfaces)[:3]
    seen_calls: list[tuple[int, int]] = []
    texture_layout(layout, frames, progress=lambda done, total: seen_calls.append((done, total)))
    assert seen_calls == [(i, 6) for i in range(6)]


def test_occluded_far_side_stays_unseen():
    layout = rectangle_layout((0.0, 0.0), (6.0, 4.0), elevation=0.0, height=2.5)
    layout["walls"].append(
        {
            "id": "wall-split",
            "floorId": "floor-0",
            "start": [3.0, 0.0],
            "end": [3.0, 4.0],
            "thickness": 0.1,
            "height": 2.5,
            "provenance": "user",
        }
    )
    surfaces = layout_surfaces(layout)
    frames = []
    for i, (x, z) in enumerate([(1.0, 1.0), (1.5, 2.0), (2.0, 3.0), (1.2, 2.6)]):
        for j, yaw in enumerate((np.pi / 2 - 0.3, np.pi / 2, np.pi / 2 + 0.3)):
            frames.append(_frame(surfaces, f"near-{i}-{j}", [x, 1.4, z], yaw).frame)
    mesh = texture_layout(layout, frames)

    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)
    far_checked = near_seen = 0
    for surface, placement in zip(surfaces, atlas.placements, strict=True):
        pixels, world, inside = chart_texels(surface, placement, atlas)
        pixels, world = pixels[inside], world[inside]
        colours = mesh.pages[placement.page][pixels[:, 1], pixels[:, 0]]
        far = world[:, 0] > 3.25
        # Hole filling reaches at most a few texels across the seam, so the far side stays grey.
        assert np.all(colours[far] == UNSEEN), surface.name
        far_checked += int(far.sum())
        near = world[:, 0] < 2.9
        near_seen += int(np.any(colours[near] != UNSEEN, axis=1).sum())
    assert far_checked > 5_000
    assert near_seen > 5_000


def _touching(mask: np.ndarray) -> np.ndarray:
    """Cells of a 2D mask that share an edge with a True cell."""
    touch = np.zeros_like(mask)
    touch[1:, :] |= mask[:-1, :]
    touch[:-1, :] |= mask[1:, :]
    touch[:, 1:] |= mask[:, :-1]
    touch[:, :-1] |= mask[:, 1:]
    return touch


def test_floor_hole_stays_unseen_beyond_the_padding_reach():
    # A hole is not surface, so no frame colours it. Dilation reaches only the padding distance
    # into it: the hole's interior stays unseen while its edge takes the floor's colour.
    layout = rectangle_layout((-2.0, -1.5), (2.0, 1.5), elevation=0.0, height=2.5)
    layout["floors"][0]["holes"] = [[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]]
    layout["spawn"]["feet"] = [1.5, 0.0, 1.0]
    surfaces = layout_surfaces(layout)
    frames = []
    for k in range(8):
        angle = 2 * np.pi * k / 8
        eye = [np.cos(angle), 1.4, np.sin(angle)]
        yaw = np.arctan2(-np.cos(angle), np.sin(angle))  # facing the hole's centre
        frames.append(_frame(surfaces, f"hole-{k}", eye, yaw, pitch=-0.8 + 0.1 * (k % 2)).frame)
    mesh = texture_layout(layout, frames)
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)
    index = next(i for i, s in enumerate(surfaces) if s.kind == "floor")
    placement = atlas.placements[index]
    pixels, world, inside = chart_texels(surfaces[index], placement, atlas)
    colours = mesh.pages[placement.page][pixels[:, 1], pixels[:, 0]]
    # Signed distance from each texel centre to the hole's edge, positive inside the hole.
    x, z = world[:, 0], world[:, 2]
    depth = np.min(np.stack([x + 0.5, 0.5 - x, z + 0.5, 0.5 - z]), axis=0)
    hole = ~inside & (depth > 0)
    # The floor around the hole is seen, so the checks below are not vacuous.
    near_hole = inside & (np.maximum(np.abs(x), np.abs(z)) < 0.8)
    assert np.mean(np.any(colours[near_hole] != UNSEEN, axis=1)) > 0.9
    # Texels at least 0.05 m inside the hole are farther than the 2-texel dilation reaches.
    deep = hole & (depth >= 0.05)
    assert deep.sum() > 5_000
    assert np.all(colours[deep] == UNSEEN)
    # Hole texels that touch the floor are dilated from it, so mipmaps do not pull in grey.
    shape = (placement.height, placement.width)
    edge = _touching(inside.reshape(shape)) & hole.reshape(shape)
    assert edge.sum() > 300
    assert np.all(np.any(colours.reshape(shape + (3,))[edge] != UNSEEN, axis=1))


@pytest.mark.parametrize("padding", [0, 2])
def test_adjacent_charts_never_exchange_colours(padding):
    # Two 10 x 8 texel charts side by side on one page, their rectangles touching. Chart 0 is
    # seen only in its left seven columns, so its right columns are hole-filled from their
    # neighbours; chart 1 is seen everywhere. Each chart has its own flat colour, so a texel
    # holding the other colour is a leak across the chart boundary.
    block_w, block_h = 10, 8
    rect_w, rect_h = block_w + 2 * padding, block_h + 2 * padding
    rows, cols = np.indices((rect_h, rect_w))
    in_block = (
        (cols >= padding)
        & (cols < padding + block_w)
        & (rows >= padding)
        & (rows < padding + block_h)
    )
    colours = {0: np.array([200.0, 40.0, 40.0]), 1: np.array([40.0, 60.0, 220.0])}
    surface, pages, pixels, inside, seen, colour = [], [], [], [], [], []
    for k in (0, 1):
        seen_here = in_block & (cols - padding < 7) if k == 0 else in_block
        count = rect_h * rect_w
        surface.append(np.full(count, k, dtype=np.int64))
        pages.append(np.zeros(count, dtype=np.int64))
        pixels.append(np.column_stack([(k * rect_w + cols).ravel(), rows.ravel()]))
        inside.append(in_block.ravel())
        seen.append(seen_here.ravel())
        colour.append(np.where(seen_here.ravel()[:, None], colours[k], UNSEEN))
    table = texture_module._ChartTable(
        surface=np.concatenate(surface),
        page=np.concatenate(pages),
        pixel=np.concatenate(pixels),
        inside=np.concatenate(inside),
        world=np.zeros((2 * rect_w * rect_h, 3)),
    )
    atlas = Atlas(texel_size=0.01, page_size=64, pages=1, padding=padding, placements=[])
    image = texture_module._paint_pages(
        atlas, table, np.concatenate(colour), np.concatenate(seen), TextureParams()
    )[0]
    charts = {0: image[:rect_h, :rect_w], 1: image[:rect_h, rect_w : 2 * rect_w]}
    ring = _touching(in_block) & ~in_block  # padding texels that touch the chart
    assert ring.any() == (padding > 0)
    for k, mine in charts.items():
        assert not np.any(np.all(mine == colours[1 - k], axis=-1)), k
        assert np.all(mine[in_block] == colours[k]), k
        assert np.all(mine[ring] == colours[k]), k


def test_pixel_registration_is_exact_for_a_single_frame():
    # Red codes the column index and green the row index of every pixel, so a texel's
    # colour must equal its projected index coordinates (COLMAP continuous minus 0.5).
    layout = _room()
    surfaces = layout_surfaces(layout)
    shot = _room_shots(surfaces)[0]
    coded = np.zeros((H, W, 3), dtype=np.uint8)
    coded[..., 0] = np.arange(W)
    coded[..., 1] = np.arange(H)[:, None]
    coded[..., 2] = 200
    frame = dataclasses.replace(shot.frame, load=lambda: coded)
    mesh = texture_layout(layout, [frame])
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)

    worst = 0.0
    checked = painted = 0
    for chart in _charts(surfaces, atlas, [shot]):
        mask = chart.inside & chart.seen
        if not mask.any():
            continue
        cam = chart.world[mask] @ frame.R.T + frame.t
        expected = np.column_stack(
            [
                FX * cam[:, 0] / cam[:, 2] + CX - 0.5,
                FY * cam[:, 1] / cam[:, 2] + CY - 0.5,
            ]
        )
        colours = _colours(mesh, chart, mask)
        taken = colours[:, 2] == 200
        checked += mask.sum()
        painted += taken.sum()
        if taken.any():
            worst = max(worst, float(np.abs(colours[taken, :2] - expected[taken]).max()))
    assert checked > 20_000
    assert painted / checked > 0.98
    # A rounded sample lies within half a unit of its exact index; a half-pixel or one-pixel
    # offset in the sampling path pushes some texels past that bound.
    assert worst <= 0.5 + 1e-6


def test_bilinear_sampling_reads_index_coordinates():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 0] = np.arange(6)  # value equals the column index
    uc = np.array([0.0, 2.25, 4.5])
    samples = texture_module._sample(image, uc, np.ones(3))
    np.testing.assert_allclose(samples[:, 0], uc)


def test_exposure_gain_reads_bright_and_dim_frames(monkeypatch):
    layout = _room()
    surfaces = layout_surfaces(layout)
    scales = [1.2 if k % 3 == 0 else 0.8 if k % 3 == 1 else 1.0 for k in range(12)]
    frames = [shot.frame for shot in _room_shots(surfaces, scales=scales)]
    uniform = _room_frames(surfaces)

    gains: list[float] = []
    real_gain = texture_module._gain

    def recording(view, rgb, blend, params):
        gain = real_gain(view, rgb, blend, params)
        gains.append(gain)
        return gain

    monkeypatch.setattr(texture_module, "_gain", recording)
    mixed = texture_layout(layout, frames)
    assert len(gains) == len(frames)
    for k, gain in enumerate(gains):
        if scales[k] > 1.0:
            assert gain < 1.0, k  # a brighter frame is scaled down
        elif scales[k] < 1.0:
            assert gain > 1.0, k  # a dimmer frame is scaled up

    monkeypatch.setattr(texture_module, "_gain", lambda view, rgb, blend, params: 1.0)
    gain_free = texture_layout(layout, frames)
    monkeypatch.undo()
    reference = texture_layout(layout, uniform)

    with_gain, without, ref = [], [], []
    for a, b, c in zip(mixed.pages, gain_free.pages, reference.pages, strict=True):
        keep = np.any(c != UNSEEN, axis=-1) & np.any(a != UNSEEN, axis=-1)
        keep &= np.any(b != UNSEEN, axis=-1)
        with_gain.append(np.abs(a[keep].astype(float) - c[keep]))
        without.append(np.abs(b[keep].astype(float) - c[keep]))
        ref.append(keep)
    assert sum(r.sum() for r in ref) > 50_000
    assert np.concatenate(with_gain).mean() < np.concatenate(without).mean()


def test_split_atlas_reproduces_on_every_page():
    layout = _room()
    surfaces = layout_surfaces(layout)
    shots = _room_shots(surfaces)
    params = TextureParams(page_size=512)
    mesh = texture_layout(layout, [shot.frame for shot in shots], params)
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size, page_size=512)
    assert atlas.pages > 1
    assert len(mesh.pages) == atlas.pages
    charts = _charts(surfaces, atlas, shots)
    for page in range(atlas.pages):
        errors, painted = [], []
        for k, chart in enumerate(charts):
            if chart.page != page:
                continue
            mask = chart.inside & chart.seen
            colours = _colours(mesh, chart, mask)
            errors.append(np.abs(colours - truth(chart.world[mask], k)))
            painted.append(np.any(colours != UNSEEN, axis=1))
        errors = np.concatenate(errors)
        assert errors.size > 10_000, page
        assert np.concatenate(painted).mean() > 0.99, page
        assert errors.mean() < MAE_BOUND, page


def test_frames_must_load_as_uint8():
    layout = _room()
    surfaces = layout_surfaces(layout)
    shot = _room_shots(surfaces)[0]
    floats = dataclasses.replace(shot.frame, load=lambda: np.zeros((H, W, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="uint8"):
        texture_layout(layout, [floats])


def test_identical_inputs_give_identical_pages():
    layout = _room()
    surfaces = layout_surfaces(layout)
    frames = _room_frames(surfaces)[:4]
    first = texture_layout(layout, frames)
    second = texture_layout(layout, frames)
    assert len(first.pages) == len(second.pages) == 1
    for a, b in zip(first.pages, second.pages, strict=True):
        assert np.array_equal(a, b)
    assert first.coverage == second.coverage


def test_camera_in_world_matches_colmap_then_world_from_colmap():
    rng = np.random.default_rng(7)
    axis = np.array([0.3, 1.0, 0.2])
    axis /= np.linalg.norm(axis)
    angle = 0.4
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot_w = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    scale = 2.5
    translation = np.array([1.0, 2.0, 3.0])
    world_from_colmap = np.eye(4)
    world_from_colmap[:3, :3] = scale * rot_w
    world_from_colmap[:3, 3] = translation
    flat = world_from_colmap.reshape(-1, order="F")  # column-major, as glTF/three store it

    R_c, _ = _look([0.2, 0.1, -0.4], yaw=0.7, pitch=-0.2)
    t_c = np.array([0.1, -0.2, 3.0])
    R, t = camera_in_world(R_c, t_c, flat)

    x_colmap = rng.uniform(-2.0, 2.0, size=(25, 3))
    x_world = x_colmap @ (scale * rot_w).T + translation
    for xc, xw in zip(x_colmap, x_world, strict=True):
        cam_c = R_c @ xc + t_c
        cam_w = R @ xw + t
        np.testing.assert_allclose(cam_w, scale * cam_c, atol=1e-9)
        pixel_c = K @ (cam_c / cam_c[2])
        pixel_w = K @ (cam_w / cam_w[2])
        np.testing.assert_allclose(pixel_w, pixel_c, atol=1e-9)

    # A calibration scale about the origin multiplies the translation only.
    R2, t2 = camera_in_world(R_c, t_c, flat, render_scale=2.0)
    np.testing.assert_allclose(R2, R)
    np.testing.assert_allclose(t2, 2.0 * t)


def test_procedural_layout_is_deterministic_with_full_coverage():
    layout = _room()
    first = procedural_texture_layout(layout)
    second = procedural_texture_layout(layout)
    for a, b in zip(first.pages, second.pages, strict=True):
        assert np.array_equal(a, b)
    assert set(first.coverage) == {"floor", "wall", "ceiling", "all"}
    assert all(value == 1.0 for value in first.coverage.values())


def _chart_samples(
    mesh: TexturedMesh, surface: Surface, placement: ChartPlacement, atlas: Atlas
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface (u, v) and RGB of the texels on one chart's surface: (M,), (M,), (M, 3)."""
    pixels, _, inside = chart_texels(surface, placement, atlas)
    u, v = chart_grid(placement, atlas)
    rgb = mesh.pages[placement.page][pixels[:, 1], pixels[:, 0]].astype(np.float64)
    return u.ravel()[inside], v.ravel()[inside], rgb[inside]


def test_procedural_patterns_follow_their_kind():
    layout = _furnished_room()
    surfaces = layout_surfaces(layout)
    mesh = procedural_texture_layout(layout)
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)
    samples = [
        _chart_samples(mesh, surface, placement, atlas)
        for surface, placement in zip(surfaces, atlas.placements, strict=True)
    ]

    # Floor: 0.18 m boards along v. The darker seams are left out; within a board only the grain
    # varies, and neighbouring boards take their own tones.
    floor = next(i for i, s in enumerate(surfaces) if s.kind == "floor")
    _, floor_v, floor_rgb = samples[floor]
    scaled = floor_v / 0.18
    board = np.floor(scaled)
    across = scaled - board
    keep = (across > 0.03) & (across < 0.97)
    board, red = board[keep], floor_rgb[keep, 0]
    boards = np.unique(board)
    means = np.array([red[board == b].mean() for b in boards])
    spreads = [np.ptp(red[board == b]) for b in boards]
    assert boards.size >= 12  # the 2.5 m run along v holds about 14 boards
    assert max(spreads) < 10  # the grain alone spans 7.5 units of red
    assert np.ptp(means) > 5
    assert np.mean(np.abs(np.diff(means)) > 1) > 0.8

    # Obstacles: a 0.1 m checker in surface coordinates, exactly its two tones.
    for i, surface in enumerate(surfaces):
        if surface.kind != "obstacle":
            continue
        u, v, rgb = samples[i]
        parity = (np.floor(u / 0.1) + np.floor(v / 0.1)) % 2
        expected = np.where(
            parity[:, None] == 0, texture_module.CHECKER_A, texture_module.CHECKER_B
        )
        np.testing.assert_array_equal(rgb, expected, err_msg=surface.name)

    # The ceiling is whiter than the plaster walls.
    ceiling = next(i for i, s in enumerate(surfaces) if s.kind == "ceiling")
    wall_rgb = np.concatenate([samples[i][2] for i, s in enumerate(surfaces) if s.kind == "wall"])
    ceiling_luma = float((samples[ceiling][2] @ texture_module.LUMA).mean())
    assert ceiling_luma > float((wall_rgb @ texture_module.LUMA).mean()) + 5


def test_texture_params_round_trip_into_the_atlas():
    layout = _room()
    surfaces = layout_surfaces(layout)
    frames = _room_frames(surfaces)[:2]
    mesh = texture_layout(layout, frames, TextureParams(texel_size=0.02))
    assert mesh.texel_size == pytest.approx(0.02)
    assert all(page.shape == (2048, 2048, 3) and page.dtype == np.uint8 for page in mesh.pages)


# --------------------------------------------------------------------------
# Furniture
# --------------------------------------------------------------------------

SOFA_RGB = (42, 111, 143)  # the test piece's colour, #2a6f8f


def _sofa_room() -> dict[str, Any]:
    """``_room`` with a 0.5 x 0.8 x 1.0 m box standing 0.25 m off the west wall.

    The cameras circle within 0.25 m of the centre, so every camera is east of the box and the
    west wall behind it is hidden from all of them below the box's top.
    """
    layout = _room()
    layout["furniture"] = [
        {
            "id": "sofa",
            "floorId": "floor-0",
            "category": "sofa",
            "position": [-1.0, 0.0],
            "yawRadians": 0.0,
            "size": [0.5, 0.8, 1.0],
            "solid": True,
            "parts": [{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#2a6f8f"}],
        }
    ]
    return layout


def _shade(normal: np.ndarray) -> np.ndarray:
    """The part colour's fixed face shade, written out from the texturing rule."""
    light = np.array([0.35, 0.85, 0.4])
    light /= np.linalg.norm(light)
    return 0.6 + 0.4 * max(0.0, float(normal @ light))


@pytest.fixture(scope="module")
def sofa_run() -> _Run:
    layout = _sofa_room()
    return _texture_run(layout, _room_shots(layout_surfaces(layout)))


def test_furniture_the_frames_see_reproduces_ground_truth(sofa_run):
    # The top and the east face are in view. The north, south and west faces are not: the
    # cameras never get beyond them. So about 1.3 of the piece's 2.9 m2 is seen.
    truth = _truth_coverage(sofa_run.charts)["furniture"]
    assert truth == pytest.approx(1.3 / 2.9, abs=0.03)
    assert sofa_run.mesh.coverage["furniture"] >= truth - 0.01
    errors = _kind_errors(sofa_run, "furniture")
    assert errors.size > 5_000
    assert errors.mean() < MAE_BOUND


def test_the_wall_behind_a_piece_stays_unseen_where_the_piece_hides_it(sofa_run):
    # The west wall stands 0.25 m behind the piece. The band checked here is the wall behind the
    # piece's middle, below its top: away from the room corners, where the depth tolerance lets
    # grazing texels through, and away from the piece's side edges.
    index = next(
        k
        for k, s in enumerate(sofa_run.surfaces)
        if s.kind == "wall" and s.extras.get("wallId") == "wall-west" and s.normal[0] > 0
    )
    chart = sofa_run.charts[index]
    rows = np.unique(chart.pixels[:, 1]).size
    cols = np.unique(chart.pixels[:, 0]).size
    shape = (rows, cols)
    seen = chart.seen.reshape(shape)
    # The blend fills at most six texels from a seen one; seven steps keep clear of that.
    reach = seen.copy()
    for _ in range(7):
        reach |= _touching(reach)
    height = chart.world[:, 1].reshape(shape)
    along = chart.world[:, 2].reshape(shape)
    behind = (height > 0.1) & (height < 0.6) & (np.abs(along) < 0.3)
    deep = chart.inside.reshape(shape) & ~reach & behind
    assert deep.sum() > 1_000
    assert not np.any(seen[deep])
    colours = _colours(sofa_run.mesh, chart, np.ones(len(chart.pixels), dtype=bool))
    assert np.all(colours.reshape(shape + (3,))[deep] == UNSEEN)


def test_an_unseen_piece_face_takes_its_part_colour_in_shade(sofa_run):
    # The face on the west side looks at the west wall, away from every camera.
    index = next(
        k
        for k, s in enumerate(sofa_run.surfaces)
        if s.kind == "furniture" and np.allclose(s.normal, [-1, 0, 0])
    )
    chart = sofa_run.charts[index]
    assert not chart.seen.any()
    colours = _colours(sofa_run.mesh, chart, chart.inside)
    expected = np.clip(np.rint(np.array(SOFA_RGB) * _shade(np.array([-1.0, 0.0, 0.0]))), 0, 255)
    assert expected.tolist() == [25, 67, 86]
    assert np.all(colours == expected)


def test_coverage_counts_the_architecture_in_all_and_furniture_apart():
    def surface(kind: str) -> Surface:
        return Surface(
            name=kind,
            kind=kind,
            origin=np.zeros(3),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            outer=np.zeros((3, 2)),
        )

    surfaces = [surface("wall"), surface("furniture")]
    surface_of_inside = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    seen = np.array([True, True, True, False, False, False, False, False])
    coverage = texture_module._coverage(surfaces, surface_of_inside, seen)
    assert coverage["wall"] == pytest.approx(0.75)
    assert coverage["furniture"] == 0.0
    assert coverage["all"] == pytest.approx(0.75)  # furniture does not dilute the architecture


def test_procedural_furniture_takes_its_part_colour_in_shade():
    layout = _sofa_room()
    surfaces = layout_surfaces(layout)
    mesh = procedural_texture_layout(layout)
    atlas = build_atlas(surfaces, texel_size=mesh.texel_size)
    assert mesh.coverage["furniture"] == 1.0
    checked = 0
    for surface, placement in zip(surfaces, atlas.placements, strict=True):
        if surface.kind != "furniture":
            continue
        _, _, rgb = _chart_samples(mesh, surface, placement, atlas)
        expected = np.clip(np.rint(np.array(SOFA_RGB) * _shade(surface.normal)), 0, 255)
        assert np.all(rgb == expected), surface.name
        checked += len(rgb)
    assert checked > 0
