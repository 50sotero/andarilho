"""Layout surfaces, atlas packing and chart texels (geometry.surfaces)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from andarilho_worker.geometry.layout import floor_supports, rectangle_layout
from andarilho_worker.geometry.polygon import point_in_polygon, signed_area
from andarilho_worker.geometry.surfaces import (
    Surface,
    build_atlas,
    chart_texels,
    layout_surfaces,
    surfaces_to_parts,
)

FIXTURES = Path(__file__).resolve().parents[3] / "packages" / "scene-contract" / "fixtures"
WALL_KINDS = {"wall", "closed_opening", "sill", "lintel"}


def _room_with_door_and_window() -> dict:
    layout = rectangle_layout((0.0, 0.0), (4.0, 3.0), elevation=0.0, height=2.5)
    layout["openings"] = [
        {
            "id": "door",
            "wallId": "wall-north",
            "offset": 1.0,
            "width": 0.9,
            "bottom": 0.0,
            "top": 2.1,
            "state": "open",
            "provenance": "user",
        },
        {
            "id": "window",
            "wallId": "wall-east",
            "offset": 0.8,
            "width": 1.0,
            "bottom": 0.9,
            "top": 1.8,
            "state": "closed",
            "provenance": "user",
        },
    ]
    return layout


def _synthetic() -> dict:
    return json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())


@pytest.fixture
def room() -> dict:
    return _room_with_door_and_window()


@pytest.fixture
def surfaces(room):
    return layout_surfaces(room)


def test_door_and_window_yield_expected_kinds(surfaces):
    # Every wall of a lone room is exterior, so each piece keeps one long face (the room side)
    # beside its caps and its sill or lintel faces.
    counts = Counter(s.kind for s in surfaces)
    assert counts == {
        "wall": 10,
        "closed_opening": 3,
        "sill": 4,
        "lintel": 8,
        "floor": 1,
        "ceiling": 1,
    }


def test_synthetic_fixture_surfaces_cover_obstacle_and_divider():
    layout = _synthetic()
    surfaces = layout_surfaces(layout)
    counts = Counter(s.kind for s in surfaces)
    assert counts["floor"] == 1 and counts["ceiling"] == 1
    # The table's top plus four sides.
    assert counts["obstacle"] == 5
    # The divider's door: one lintel piece (five faces), no sill.
    assert counts["lintel"] == 5
    assert "sill" not in counts
    ceiling = next(s for s in surfaces if s.kind == "ceiling")
    assert ceiling.origin[1] == pytest.approx(2.6)


def test_normals_are_unit_and_axes_span_them(surfaces):
    for s in surfaces:
        assert np.linalg.norm(s.normal) == pytest.approx(1.0)
        assert np.linalg.norm(s.u_axis) == pytest.approx(1.0)
        assert np.linalg.norm(s.v_axis) == pytest.approx(1.0)
        assert np.dot(s.u_axis, s.v_axis) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(np.cross(s.u_axis, s.v_axis), s.normal, atol=1e-12)


def test_floor_and_ceiling_face_the_right_way(surfaces):
    floor = next(s for s in surfaces if s.kind == "floor")
    ceiling = next(s for s in surfaces if s.kind == "ceiling")
    np.testing.assert_allclose(floor.normal, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(ceiling.normal, [0, -1, 0], atol=1e-12)


def _surface_named(surfaces: list[Surface], name: str) -> Surface:
    return next(s for s in surfaces if s.name == name)


def _world_ring(s: Surface) -> np.ndarray:
    """World positions (K, 3) of a surface's outer ring."""
    return s.origin + s.outer[:, :1] * s.u_axis + s.outer[:, 1:] * s.v_axis


def test_sill_top_faces_up_at_the_sill_height(surfaces):
    # The sill is the wall below the window: its top sits at the opening's bottom edge, faces
    # up into the room, and spans the opening along the east wall (z from 0.8 to 1.8).
    top = _surface_named(surfaces, "wall-east-sill-window:top")
    np.testing.assert_allclose(top.normal, [0, 1, 0], atol=1e-12)
    ring = _world_ring(top)
    np.testing.assert_allclose(ring[:, 1], 0.9, atol=1e-12)
    assert ring[:, 2].min() == pytest.approx(0.8)
    assert ring[:, 2].max() == pytest.approx(1.8)
    # The door opens onto the floor, so its wall has no sill.
    assert not any(s.name.startswith("wall-north-sill") for s in surfaces)


def test_lintel_undersides_face_down_at_the_lintel_height(surfaces):
    # A lintel is the wall above an opening: its underside sits at the opening's top edge and
    # faces down into the opening. The door's lintel is at 2.1 m, the window's at 1.8 m.
    for name, height in (
        ("wall-north-lintel-door:bottom", 2.1),
        ("wall-east-lintel-window:bottom", 1.8),
    ):
        bottom = _surface_named(surfaces, name)
        np.testing.assert_allclose(bottom.normal, [0, -1, 0], atol=1e-12, err_msg=name)
        np.testing.assert_allclose(_world_ring(bottom)[:, 1], height, atol=1e-12, err_msg=name)


def _long_faces(surfaces: list[Surface]) -> list[Surface]:
    """The two long faces (``:a`` and ``:b``) of every wall piece."""
    return [s for s in surfaces if s.kind in WALL_KINDS and s.name.endswith((":a", ":b"))]


def _face_centre_xz(s: Surface) -> np.ndarray:
    centre_uv = s.outer.mean(axis=0)
    centre = s.origin + centre_uv[0] * s.u_axis + centre_uv[1] * s.v_axis
    return np.array([centre[0], centre[2]])


@pytest.mark.parametrize("name", ["room", "synthetic"])
def test_long_faces_point_away_from_their_wall_centreline(name):
    # A long face sits half a wall thickness off the centreline, and its normal must point away
    # from the centreline. The synthetic divider keeps both faces, so both sides are checked.
    layout = _room_with_door_and_window() if name == "room" else _synthetic()
    walls = {w["id"]: w for w in layout["walls"]}
    faces = _long_faces(layout_surfaces(layout))
    assert faces
    for s in faces:
        wall = walls[s.extras["wallId"]]
        start = np.array(wall["start"], dtype=np.float64)
        d = np.array(wall["end"], dtype=np.float64) - start
        d /= np.linalg.norm(d)
        centre = _face_centre_xz(s)
        # The centreline point at the same position along the wall as the face centre.
        along = float(np.dot(centre - start, d))
        offset = centre - (start + along * d)
        normal = np.array([s.normal[0], s.normal[2]])
        assert float(np.dot(offset, normal)) > 0, s.name
        assert np.linalg.norm(offset) == pytest.approx(wall["thickness"] / 2), s.name


def test_room_side_faces_point_into_the_room(room, surfaces):
    # Looking 0.2 m along its normal, a long face must end up in the room. The outer face of an
    # exterior wall is never seen from the floor, so each piece of a lone room keeps only one.
    floor = room["floors"][0]
    pieces: Counter[str] = Counter()
    for s in _long_faces(surfaces):
        centre = _face_centre_xz(s)
        normal = np.array([s.normal[0], s.normal[2]])
        assert floor_supports(floor, centre), s.name
        assert floor_supports(floor, centre + 0.2 * normal), s.name
        pieces[s.name[:-2]] += 1
    assert pieces and set(pieces.values()) == {1}


def test_interior_walls_keep_both_long_faces():
    # The divider has floor on both sides, so both of its long faces can be seen; an outer wall
    # keeps only its room side.
    pieces = Counter(s.name[:-2] for s in _long_faces(layout_surfaces(_synthetic())))
    divider = {piece for piece in pieces if piece.startswith("wall-divider")}
    assert len(divider) == 3  # the two segments either side of the door and its lintel
    for piece, count in pieces.items():
        assert count == (2 if piece in divider else 1), piece


def _holed_room() -> dict:
    layout = rectangle_layout((0.0, 0.0), (4.0, 3.0), elevation=0.0, height=2.5)
    layout["floors"][0]["holes"] = [[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]]
    layout["spawn"]["feet"] = [0.3, 0.0, 0.3]  # the rectangle's centre is under the hole
    return layout


def test_floor_and_ceiling_carry_the_hole():
    layout = _holed_room()
    surfaces = layout_surfaces(layout)
    atlas = build_atlas(surfaces, texel_size=0.02)
    parts = surfaces_to_parts(surfaces, atlas)
    for kind in ("floor", "ceiling"):
        index = next(i for i, s in enumerate(surfaces) if s.kind == kind)
        surface, part = surfaces[index], parts[index]
        assert len(surface.holes) == 1, kind
        assert abs(signed_area(surface.holes[0])) == pytest.approx(1.0), kind

        a, b, c = (part.positions[part.indices[:, k]] for k in range(3))
        area = float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)) / 2)
        assert area == pytest.approx(12.0 - 1.0), kind
        centroids = part.positions[part.indices].mean(axis=1)
        in_hole = (centroids[:, 0] > 1.0) & (centroids[:, 0] < 2.0)
        in_hole &= (centroids[:, 2] > 1.0) & (centroids[:, 2] < 2.0)
        assert not in_hole.any(), kind

        _, world, inside = chart_texels(surface, atlas.placements[index], atlas)
        # Texel centres sit on odd multiples of 0.01 m, so the count is exact: 200 x 150 texels
        # on the 4 x 3 m rectangle less the 50 x 50 under the hole.
        assert inside.sum() == 200 * 150 - 50 * 50, kind
        x, z = world[:, 0], world[:, 2]
        texel_in_hole = (x > 1.0) & (x < 2.0) & (z > 1.0) & (z < 2.0)
        assert not np.any(inside & texel_in_hole), kind


def test_triangle_windings_agree_with_normals(surfaces, room):
    atlas = build_atlas(surfaces)
    for part, s in zip(surfaces_to_parts(surfaces, atlas), surfaces, strict=True):
        a, b, c = (part.positions[part.indices[:, k]] for k in range(3))
        cross = np.cross(b - a, c - a)
        length = np.linalg.norm(cross, axis=1)
        solid = length > 1e-12
        facing = cross[solid] @ s.normal
        assert np.all(facing > 0), s.name
        np.testing.assert_allclose(part.normals, np.tile(s.normal, (len(part.positions), 1)))


def test_uvs_stay_inside_their_chart(surfaces):
    atlas = build_atlas(surfaces)
    parts = surfaces_to_parts(surfaces, atlas)
    for part, placement in zip(parts, atlas.placements, strict=True):
        assert part.page == placement.page
        assert np.all(part.uvs >= 0.0) and np.all(part.uvs <= 1.0), part.name
        px = part.uvs * atlas.page_size
        assert np.all(px[:, 0] >= placement.x - 1e-6), part.name
        assert np.all(px[:, 0] <= placement.x + placement.width + 1e-6), part.name
        assert np.all(px[:, 1] >= placement.y - 1e-6), part.name
        assert np.all(px[:, 1] <= placement.y + placement.height + 1e-6), part.name


def test_floor_triangles_sum_to_polygon_area(room, surfaces):
    atlas = build_atlas(surfaces)
    parts = surfaces_to_parts(surfaces, atlas)
    floor = next(p for p in parts if p.kind == "floor")
    a, b, c = (floor.positions[floor.indices[:, k]] for k in range(3))
    area = float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)) / 2)
    assert area == pytest.approx(4.0 * 3.0)


def test_chart_texels_integrate_to_surface_area(surfaces):
    atlas = build_atlas(surfaces, texel_size=0.02)
    floor_index = next(i for i, s in enumerate(surfaces) if s.kind == "floor")
    _, _, inside = chart_texels(surfaces[floor_index], atlas.placements[floor_index], atlas)
    assert inside.sum() * atlas.texel_size**2 == pytest.approx(12.0, rel=0.02)


def test_charts_do_not_overlap_on_a_page(surfaces):
    atlas = build_atlas(surfaces)
    rects = [(p.page, p.x, p.y, p.x + p.width, p.y + p.height) for p in atlas.placements]
    for i, (pa, ax0, ay0, ax1, ay1) in enumerate(rects):
        assert ax0 >= 0 and ay0 >= 0 and ax1 <= atlas.page_size and ay1 <= atlas.page_size
        for pb, bx0, by0, bx1, by1 in rects[i + 1 :]:
            if pa != pb:
                continue
            separate = ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0
            assert separate, (i, pa)


def test_tiny_page_forces_coarser_texels(surfaces):
    default = build_atlas(surfaces)
    tiny = build_atlas(surfaces, page_size=256)
    assert default.texel_size == pytest.approx(0.01)
    assert tiny.texel_size > default.texel_size
    assert tiny.pages >= 1
    for placement in tiny.placements:
        assert placement.x + placement.width <= 256
        assert placement.y + placement.height <= 256


def test_atlas_with_no_room_for_a_chart_raises(surfaces):
    with pytest.raises(ValueError):
        build_atlas(surfaces, page_size=8, padding=2, max_pages=1)


def test_uvs_map_back_to_each_vertex_through_its_chart(surfaces):
    # Inverting the atlas mapping must give back each vertex's world position. A mirrored or
    # swapped UV mapping fails here even though every UV stays inside its chart.
    atlas = build_atlas(surfaces)
    parts = surfaces_to_parts(surfaces, atlas)
    for part, surface, placement in zip(parts, surfaces, atlas.placements, strict=True):
        px = part.uvs * atlas.page_size
        u = placement.u_min + (px[:, 0] - placement.x - atlas.padding) * atlas.texel_size
        v = placement.v_max - (px[:, 1] - placement.y - atlas.padding) * atlas.texel_size
        rebuilt = surface.origin + u[:, None] * surface.u_axis + v[:, None] * surface.v_axis
        np.testing.assert_allclose(rebuilt, part.positions, atol=1e-9, err_msg=surface.name)


def test_triangle_centroids_land_on_texels_holding_their_world_points(surfaces):
    # The texel a triangle's UV centroid addresses must hold the world point of that centroid:
    # this is the texel the texturing pass writes to.
    atlas = build_atlas(surfaces)
    parts = surfaces_to_parts(surfaces, atlas)
    for part, surface, placement in zip(parts, surfaces, atlas.placements, strict=True):
        _, world, _ = chart_texels(surface, placement, atlas)
        centroid_uv = part.uvs[part.indices].mean(axis=1) * atlas.page_size
        cols = np.floor(centroid_uv[:, 0]).astype(np.int64) - placement.x
        rows = np.floor(centroid_uv[:, 1]).astype(np.int64) - placement.y
        inside_chart = (
            (cols >= 0) & (cols < placement.width) & (rows >= 0) & (rows < placement.height)
        )
        assert inside_chart.all(), surface.name
        texel_world = world[rows * placement.width + cols]
        centroid_world = part.positions[part.indices].mean(axis=1)
        gap = np.linalg.norm(texel_world - centroid_world, axis=1)
        # Half a texel per axis puts the texel centre within a half diagonal of the point.
        assert gap.max() <= atlas.texel_size * np.sqrt(0.5) + 1e-9, surface.name


def test_small_pages_split_charts_across_pages_without_overlap(room):
    surfaces = layout_surfaces(room)
    atlas = build_atlas(surfaces, page_size=512)
    assert atlas.pages > 1
    assert {p.page for p in atlas.placements} == set(range(atlas.pages))
    rects = [(p.page, p.x, p.y, p.x + p.width, p.y + p.height) for p in atlas.placements]
    for i, (pa, ax0, ay0, ax1, ay1) in enumerate(rects):
        assert 0 <= ax0 < ax1 <= atlas.page_size and 0 <= ay0 < ay1 <= atlas.page_size
        for pb, bx0, by0, bx1, by1 in rects[i + 1 :]:
            if pa == pb:
                assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0, (i, pa)
    parts = surfaces_to_parts(surfaces, atlas)
    for part, placement in zip(parts, atlas.placements, strict=True):
        assert part.page == placement.page
        assert np.all(part.uvs >= 0.0) and np.all(part.uvs <= 1.0), part.name


def _rectangle_with_obstacles(footprints: list[list[list[float]]]) -> dict:
    layout = rectangle_layout((-3.0, -3.0), (3.0, 3.0), elevation=0.0, height=2.5)
    layout["obstacles"] = [
        {
            "id": f"box-{i}",
            "floorId": "floor-0",
            "footprint": footprint,
            "height": 0.8,
            "provenance": "user",
        }
        for i, footprint in enumerate(footprints)
    ]
    return layout


def _obstacle_layouts() -> dict[str, dict]:
    synthetic = _synthetic()
    clockwise = [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]]  # signed area is negative
    concave = [[1.0, 1.0], [2.0, 1.0], [2.0, 1.5], [1.5, 1.5], [1.5, 2.0], [1.0, 2.0]]
    return {
        "synthetic": synthetic,
        "clockwise": _rectangle_with_obstacles([clockwise]),
        "concave": _rectangle_with_obstacles([concave]),
    }


@pytest.mark.parametrize("name", ["synthetic", "clockwise", "concave"])
def test_obstacle_faces_wind_and_point_outward(name):
    layout = _obstacle_layouts()[name]
    surfaces = layout_surfaces(layout)
    footprints = {o["id"]: o["footprint"] for o in layout["obstacles"]}
    atlas = build_atlas(surfaces)
    parts = surfaces_to_parts(surfaces, atlas)
    checked_sides = Counter()
    for part, s in zip(parts, surfaces, strict=True):
        if s.kind != "obstacle":
            continue
        a, b, c = (part.positions[part.indices[:, k]] for k in range(3))
        cross = np.cross(b - a, c - a)
        solid = np.linalg.norm(cross, axis=1) > 1e-12
        assert np.all(cross[solid] @ s.normal > 0), s.name
        footprint = footprints[s.extras["obstacleId"]]
        if s.name.endswith(":top"):
            np.testing.assert_allclose(s.normal, [0, 1, 0], atol=1e-12)
            continue
        # A side: its normal is horizontal and points from the footprint outwards.
        assert abs(s.normal[1]) < 1e-12, s.name
        centre_uv = s.outer.mean(axis=0)
        centre = s.origin + centre_uv[0] * s.u_axis + centre_uv[1] * s.v_axis
        outward = np.array([s.normal[0], s.normal[2]])
        probe = np.array([centre[0], centre[2]])
        assert not point_in_polygon(probe + 0.01 * outward, footprint), s.name
        assert point_in_polygon(probe - 0.01 * outward, footprint), s.name
        checked_sides[s.extras["obstacleId"]] += 1
    assert checked_sides == {oid: len(fp) for oid, fp in footprints.items()}


def test_cap_normals_face_away_from_the_piece(room):
    # cap0 sits at a piece's start and faces back along the wall; cap1 at its end faces on.
    surfaces = layout_surfaces(room)
    walls = {w["id"]: w for w in room["walls"]}
    caps = [s for s in surfaces if s.name.endswith((":cap0", ":cap1"))]
    # Door: lintel (2) plus the wall pieces either side (1 + 1). Window: closed opening,
    # sill and lintel (2 each) plus the wall pieces either side (1 + 1).
    assert len(caps) == 12
    for s in caps:
        wall = walls[s.extras["wallId"]]
        d = np.array(wall["end"], dtype=np.float64) - np.array(wall["start"], dtype=np.float64)
        d = np.array([d[0], 0.0, d[1]]) / np.linalg.norm(d)
        along = float(s.normal @ d)
        expected = -1.0 if s.name.endswith(":cap0") else 1.0
        assert along == pytest.approx(expected, abs=1e-12), s.name
