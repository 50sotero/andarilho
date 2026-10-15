"""Furniture in the layout: the object frame, semantic validation, collision and bounds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from andarilho_worker.geometry.collide import capsule_is_clear
from andarilho_worker.geometry.glb import read_glb
from andarilho_worker.geometry.layout import (
    LayoutError,
    collision_glb,
    collision_parts,
    furniture_footprint,
    furniture_points,
    furniture_transform,
    layout_bounds,
    part_box,
    rectangle_layout,
    validate_layout,
)
from andarilho_worker.geometry.polygon import signed_area

SOFA_SIZE = [2.0, 0.9, 0.9]  # width, height, depth in metres
CORNERS = [(u, v, w) for u in (0, 1) for v in (0, 1) for w in (0, 1)]


def _item(**changes) -> dict:
    """A sofa at (1, 2) on floor-0 with one solid box part and the given changes applied."""
    item = {
        "id": "sofa",
        "floorId": "floor-0",
        "category": "sofa",
        "label": "sofa",
        "position": [1.0, 2.0],
        "yawRadians": 0.0,
        "size": list(SOFA_SIZE),
        "solid": True,
        "parts": [{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#7a4a2a"}],
    }
    item.update(changes)
    return item


def _room(*furniture: dict) -> dict:
    """A 7 x 7 m room whose spawn sits at (0.5, 0, 0.5), with the given pieces in it."""
    layout = rectangle_layout((-3.0, -3.0), (4.0, 4.0), elevation=0.0, height=2.5)
    layout["furniture"] = list(furniture)
    return layout


def _spec_point(item: dict, elevation: float, u: float, v: float, w: float) -> np.ndarray:
    """A part fraction's world point, written out from the layout contract's formulas."""
    width, height, depth = item["size"]
    px, pz = item["position"]
    yaw = item["yawRadians"]
    lx = (u - 0.5) * width
    lz = (w - 0.5) * depth
    x = px + lx * math.cos(yaw) + lz * math.sin(yaw)
    z = pz - lx * math.sin(yaw) + lz * math.cos(yaw)
    return np.array([x, elevation + v * height, z])


def _rounded(points) -> set[tuple[float, ...]]:
    return {tuple(round(float(c), 9) for c in p) for p in points}


@pytest.mark.parametrize("yaw,front", [(0.0, (0.0, 1.0)), (math.pi / 2, (1.0, 0.0))])
def test_front_of_a_piece_faces_plus_z_at_zero_yaw_and_plus_x_at_a_half_turn(yaw, front):
    item = _item(yawRadians=yaw, position=[0.0, 0.0])
    centre, front_point = furniture_points(
        furniture_transform(item, 0.0), [(0.5, 0.0, 0.5), (0.5, 0.0, 1.0)]
    )
    direction = front_point - centre
    assert direction[1] == pytest.approx(0.0)
    # Local +z is half the depth in front of the centre.
    np.testing.assert_allclose([direction[0], direction[2]], np.array(front) * 0.45, atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, 0.7, -2.1])
def test_part_points_follow_the_layout_contract_formulas(yaw):
    item = _item(yawRadians=yaw, position=[-0.4, 1.3])
    fractions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.25, 0.6, 0.8), (0.9, 0.1, 0.05)]
    got = furniture_points(furniture_transform(item, 0.5), fractions)
    for point, fraction in zip(got, fractions, strict=True):
        np.testing.assert_allclose(point, _spec_point(item, 0.5, *fraction), atol=1e-12)


def test_a_sub_box_part_has_the_eight_corners_of_its_convention_box():
    item = _item(yawRadians=1.1, position=[0.3, -0.2])
    part = {"shape": "box", "min": [0.25, 0.1, 0.0], "max": [0.5, 0.6, 1.0], "color": "#ffffff"}
    corners = part_box(item, part, 0.2)
    assert corners.shape == (8, 3)
    expected = []
    for bits in CORNERS:
        fraction = [part["max"][i] if on_max else part["min"][i] for i, on_max in enumerate(bits)]
        expected.append(_spec_point(item, 0.2, *fraction))
    np.testing.assert_allclose(corners, np.array(expected), atol=1e-12)


def test_a_full_part_covers_the_footprint_and_the_height():
    item = _item(yawRadians=0.4, position=[-1.0, 0.5])
    full = {"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#ffffff"}
    corners = part_box(item, full, 0.5)
    assert corners[:, 1].min() == pytest.approx(0.5)
    assert corners[:, 1].max() == pytest.approx(0.5 + SOFA_SIZE[1])
    assert _rounded(corners[:, [0, 2]]) == _rounded(furniture_footprint(item))


def test_footprint_is_the_width_and_depth_turned_counter_clockwise():
    item = _item(yawRadians=0.7)
    footprint = np.array(furniture_footprint(item))
    assert signed_area(footprint) == pytest.approx(SOFA_SIZE[0] * SOFA_SIZE[2])
    np.testing.assert_allclose(footprint.mean(axis=0), [1.0, 2.0], atol=1e-12)
    edges = sorted(np.linalg.norm(np.roll(footprint, -1, axis=0) - footprint, axis=1))
    assert edges == pytest.approx([0.9, 0.9, 2.0, 2.0])


@pytest.mark.parametrize(
    "item,code",
    [
        (_item(floorId="nowhere"), "UNKNOWN_FLOOR"),
        (_item(size=[6.5, 0.9, 0.9]), "INVALID_FURNITURE"),  # width over 6 m
        (_item(size=[2.0, 0.9, 6.1]), "INVALID_FURNITURE"),  # depth over 6 m
        (_item(size=[2.0, 4.5, 0.9]), "INVALID_FURNITURE"),  # height over 4 m
    ],
)
def test_furniture_findings_name_each_broken_size_or_floor(item, code):
    assert code in {f.code for f in validate_layout(_room(item))}


def test_a_part_with_min_not_below_max_is_rejected():
    flat = {"shape": "box", "min": [0.5, 0, 0], "max": [0.5, 1, 1], "color": "#ffffff"}
    findings = validate_layout(_room(_item(parts=[flat])))
    assert {f.code for f in findings} == {"INVALID_FURNITURE"}
    assert "part 0" in findings[0].message


def test_a_piece_without_parts_fails_the_schema():
    assert {f.code for f in validate_layout(_room(_item(parts=[])))} == {"SCHEMA"}


def test_furniture_ids_share_one_namespace_with_openings_and_obstacles():
    layout = _room(_item(id="box-1"))
    layout["obstacles"] = [
        {"id": "box-1", "floorId": "floor-0", "footprint": [[1, 1], [2, 1], [2, 2]], "height": 1.0}
    ]
    assert "DUPLICATE_ID" in {f.code for f in validate_layout(layout)}


def test_a_valid_piece_passes_validation():
    assert validate_layout(_room(_item(yawRadians=0.3))) == []


def test_solid_furniture_blocks_the_spawn_and_a_passable_piece_does_not():
    # The spawn is at (0.5, 0.5), inside this table's footprint.
    table = _item(id="table", position=[0.5, 0.5], size=[1.0, 0.9, 1.0])
    findings = validate_layout(_room(table))
    assert {f.code for f in findings} == {"SPAWN_BLOCKED"}
    assert validate_layout(_room({**table, "solid": False})) == []


def test_a_piece_on_the_spawn_floor_is_a_collision_box_over_its_footprint():
    item = _item(yawRadians=0.5)
    layout = _room(item)
    parts = [p for p in collision_parts(layout) if p.name.startswith("furniture:")]
    assert [p.name for p in parts] == ["furniture:sofa"]
    part = parts[0]
    assert part.kind == "furniture" and part.extras == {"furnitureId": "sofa"}
    assert _rounded(part.positions[:, [0, 2]]) == _rounded(furniture_footprint(item))
    assert part.positions[:, 1].min() == pytest.approx(0.0)
    assert part.positions[:, 1].max() == pytest.approx(SOFA_SIZE[1])


def test_a_piece_on_an_upper_floor_stands_at_that_floors_elevation():
    layout = _room(_item(floorId="upper"))
    layout["floors"].append(
        {
            "id": "upper",
            "elevation": 3.0,
            "polygons": [[[-3.0, -3.0], [4.0, -3.0], [4.0, 4.0], [-3.0, 4.0]]],
            "holes": [],
        }
    )
    part = next(p for p in collision_parts(layout) if p.name == "furniture:sofa")
    assert part.positions[:, 1].min() == pytest.approx(3.0)
    assert part.positions[:, 1].max() == pytest.approx(3.0 + SOFA_SIZE[1])


def test_a_piece_that_is_not_solid_adds_no_collision():
    with_piece = collision_parts(_room(_item(solid=False)))
    assert all(not p.name.startswith("furniture:") for p in with_piece)
    assert len(with_piece) == len(collision_parts(_room()))


def test_solid_is_true_when_the_key_is_missing():
    item = _item()
    del item["solid"]
    assert any(p.name == "furniture:sofa" for p in collision_parts(_room(item)))


def test_solid_pieces_block_the_capsule_and_passable_ones_do_not():
    solid = _room(_item(position=[-1.5, -1.5], size=[1.0, 0.9, 1.0]))
    nodes = read_glb(collision_glb(solid))[1]
    assert not capsule_is_clear(nodes, [-1.5, 0.0, -1.5])
    assert capsule_is_clear(nodes, [2.0, 0.0, 2.0])
    passable = _room(_item(position=[-1.5, -1.5], size=[1.0, 0.9, 1.0], solid=False))
    nodes = read_glb(collision_glb(passable))[1]
    assert capsule_is_clear(nodes, [-1.5, 0.0, -1.5])


def test_bounds_cover_every_drawn_part_even_when_the_piece_is_not_solid():
    lamp = _item(id="lamp", solid=False, position=[3.8, 0.0], size=[1.0, 2.0, 1.0])
    layout = _room(lamp)  # the room's walls end at x = 4.0; the lamp reaches 4.3
    bounds = layout_bounds(layout)
    assert bounds["max"][0] >= 4.3
    assert bounds["max"][1] >= 2.0


def test_only_solid_pieces_reach_the_collision_mesh():
    layout = _room(_item(solid=True), _item(id="rug", solid=False, position=[-1.0, -2.0]))
    names = [p.name for p in collision_parts(layout) if p.name.startswith("furniture:")]
    assert names == ["furniture:sofa"]


def test_the_collision_glb_of_a_furnished_room_reads_back():
    doc, nodes = read_glb(collision_glb(_room(_item(solid=True))))
    assert any(node.kind == "furniture" for node in nodes)
    furniture_node = doc["nodes"][-1]
    assert furniture_node["name"] == "furniture:sofa"
    assert furniture_node["extras"] == {"kind": "furniture", "furnitureId": "sofa"}


def test_a_bad_piece_stops_the_collision_build():
    with pytest.raises(LayoutError):
        collision_parts(_room(_item(size=[7.0, 0.9, 0.9])))
