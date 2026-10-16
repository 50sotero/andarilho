"""Rooms, walls, doors and furniture on the proposal (geometry.rooms), from synthetic walks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scene_analysis_fixtures import keyframe

from andarilho_worker.geometry.grounding import GroundedDoor, GroundedObject
from andarilho_worker.geometry.layout import rectangle_layout, validate_layout
from andarilho_worker.geometry.rooms import (
    _union_outline,
    attach_doors,
    extend_unsnapped_sides,
    place_furniture,
    rooms_layout,
    segment_rooms,
    snap_rectangle,
)


def _wall(rng, fixed: str, coord: float, start: float, end: float, *, gap=None, n=300):
    """Wall-height points on one wall: ``fixed`` is "x" or "z" and ``coord`` is its value.

    ``gap`` leaves a doorway (a span of ``start``..``end`` with no points) out.
    """
    along = rng.uniform(start, end, n)
    if gap is not None:
        along = along[(along < gap[0]) | (along > gap[1])]
    heights = rng.uniform(0.5, 2.0, len(along))
    if fixed == "x":
        points = np.column_stack([np.full(len(along), coord), heights, along])
    else:
        points = np.column_stack([along, heights, np.full(len(along), coord)])
    return points + rng.normal(0, 0.01, points.shape)


def _walk(xs, z: float = 1.5) -> np.ndarray:
    """Camera centres along a walk at eye height, one per entry of ``xs``."""
    return np.column_stack([np.asarray(xs, float), np.full(len(xs), 1.5), np.full(len(xs), z)])


def _keyframes(cameras: np.ndarray, labels: dict[int, str | None]):
    """Keyframes at the walk's cameras listed in ``labels`` (index -> room, or None)."""
    keyframes, frames = [], []
    for index, room in sorted(labels.items()):
        keyframes.append(keyframe(index + 1, cameras[index], cameras[index] + [1.0, 0.0, 0.0]))
        frames.append({"frame": index + 1, "room": room, "doors": [], "objects": []})
    return keyframes, frames


def _segment(cameras, labels):
    keyframes, frames = _keyframes(cameras, labels)
    return segment_rooms(keyframes, frames, cameras[:, [0, 2]])


def _two_room_scene():
    """Room A (x 0..3.5) and room B (x 3.5..7), 3 m deep, with a doorway in the partition.

    The walk goes from A through the doorway into B and is labelled by its keyframes.
    """
    rng = np.random.default_rng(21)
    points = np.vstack(
        [
            _wall(rng, "x", 0.0, 0.0, 3.0),
            _wall(rng, "x", 7.0, 0.0, 3.0),
            _wall(rng, "z", 0.0, 0.0, 7.0),
            _wall(rng, "z", 3.0, 0.0, 7.0),
            _wall(rng, "x", 3.5, 0.0, 3.0, gap=(1.0, 2.0)),
        ]
    )
    cameras = _walk(np.linspace(0.5, 6.5, 20))
    labels = {0: "r1", 5: "r1", 9: "r1", 12: "r2", 16: "r2", 19: "r2"}
    return points, cameras, _segment(cameras, labels)


def _open_doors(layout):
    return [o for o in layout["openings"] if o["state"] == "open"]


def _wall_by_id(layout, wall_id):
    return next(w for w in layout["walls"] if w["id"] == wall_id)


# --------------------------------------------------------------------------
# Segmenting the walk
# --------------------------------------------------------------------------


def test_the_walk_is_split_into_rooms_where_its_keyframes_change_room():
    cameras = _walk(np.linspace(0.5, 6.5, 20))
    rooms = _segment(cameras, {0: "r1", 5: "r1", 9: "r1", 12: "r2", 16: "r2", 19: "r2"})
    assert [room.id for room in rooms] == ["r1", "r2"]
    # The change falls midway between keyframes 9 and 12, which sit 0.95 m apart.
    assert list(rooms[0].indices) == list(range(11))
    assert list(rooms[1].indices) == list(range(11, 20))
    assert rooms[0].keyframes == (1, 6, 10)
    assert rooms[1].keyframes == (13, 17, 20)
    assert rooms[0].entry is None
    # The walk's first step into room two lies between its last camera in one and its first.
    assert rooms[1].entry == pytest.approx(
        [(cameras[10, 0] + cameras[11, 0]) / 2, (cameras[10, 2] + cameras[11, 2]) / 2]
    )


def test_a_room_with_too_few_keyframes_joins_the_room_before_it():
    cameras = _walk(np.linspace(0.5, 9.5, 30))
    rooms = _segment(cameras, {0: "r1", 4: "r1", 9: "r1", 12: "r2", 20: "r3", 25: "r3"})
    assert [room.id for room in rooms] == ["r1", "r3"]
    assert rooms[0].keyframes == (1, 5, 10, 13)  # r2's single keyframe came with it


def test_a_first_room_too_small_joins_the_room_after_it():
    cameras = _walk(np.linspace(0.5, 9.5, 30))
    rooms = _segment(cameras, {0: "r1", 10: "r2", 14: "r2", 18: "r2"})
    assert [room.id for room in rooms] == ["r2"]
    assert len(rooms[0].indices) == 30


def test_keyframes_without_a_room_are_no_evidence():
    cameras = _walk(np.linspace(0.5, 6.5, 20))
    rooms = _segment(cameras, {0: None, 2: None, 5: "r1", 9: "r1", 12: "r2", 16: "r2"})
    assert [room.id for room in rooms] == ["r1", "r2"]
    assert rooms[0].keyframes == (6, 10)  # frames 1 and 3 had no room, so only 6 and 10 count


def test_no_labelled_keyframes_give_no_rooms():
    cameras = _walk(np.linspace(0.5, 6.5, 20))
    assert _segment(cameras, {0: None, 5: None}) == []


# --------------------------------------------------------------------------
# Extending sides to the wall beyond them
# --------------------------------------------------------------------------


def _hallway_points(rng):
    """A 1.8 m wide hallway, 6 m long: side walls at x = 0 and 1.8, a near wall and an end wall.

    The side walls are sparse, as reconstructed walls usually are; the end wall is dense.
    """
    # Furniture and stray points at chest height spread along the hallway, so that a wall's
    # peak stands out from a background, as it does in a real capture.
    clutter = np.column_stack(
        [rng.uniform(0.2, 1.6, 150), rng.uniform(1.0, 1.1, 150), rng.uniform(0.0, 6.0, 150)]
    )
    return np.vstack(
        [
            _wall(rng, "x", 0.0, 0.0, 6.0, n=80),
            _wall(rng, "x", 1.8, 0.0, 6.0, n=80),
            _wall(rng, "z", 0.0, 0.0, 1.8, n=80),
            _wall(rng, "z", 6.0, 0.0, 1.8),
            clutter,
        ]
    )


def test_a_side_the_walk_never_reached_extends_to_the_wall_beyond_it():
    rng = np.random.default_rng(31)
    points = _hallway_points(rng)
    # The walk goes down the middle and stops at z = 3: the end wall is 3 m past its last camera.
    cameras = np.column_stack([np.full(20, 0.9), np.full(20, 1.5), np.linspace(0.5, 3.0, 20)])
    min_xz, max_xz = [0.9 - 0.75, 0.5 - 0.75], [0.9 + 0.75, 3.0 + 0.75]
    lo, hi, snapped = snap_rectangle(min_xz, max_xz, points=points, cameras=cameras, height=2.5)
    assert set(snapped) == {"west", "east", "south"}  # the walls beside and behind snap
    lo, hi, moved = extend_unsnapped_sides(
        lo, hi, points=points, cameras=cameras, height=2.5, snapped=set(snapped)
    )
    assert list(moved) == ["north"]
    assert hi[1] == pytest.approx(6.0, abs=0.05)
    assert lo == pytest.approx([0.0, 0.0], abs=0.05)
    assert hi[0] == pytest.approx(1.8, abs=0.05)


def test_a_side_that_snapped_is_not_extended_again():
    rng = np.random.default_rng(32)
    points = _hallway_points(rng)
    cameras = np.column_stack([np.full(20, 0.6), np.full(20, 1.5), np.linspace(0.5, 3.0, 20)])
    lo, hi, moved = extend_unsnapped_sides(
        [-0.15, -0.25],
        [1.35, 3.75],
        points=points,
        cameras=cameras,
        height=2.5,
        snapped={"west", "east", "south", "north"},
    )
    assert moved == {}
    assert hi == [1.35, 3.75]


def test_the_reach_limits_the_extension():
    rng = np.random.default_rng(33)
    points = _hallway_points(rng)
    cameras = np.column_stack([np.full(20, 0.6), np.full(20, 1.5), np.linspace(0.5, 3.0, 20)])
    # The end wall is 2.25 m beyond the rectangle: out of reach at 1 m.
    _, hi, moved = extend_unsnapped_sides(
        [-0.15, 0.0], [1.35, 3.75], points=points, cameras=cameras, height=2.5,
        snapped={"west", "east", "south"}, reach=1.0,
    )  # fmt: skip
    assert moved == {} and hi[1] == 3.75


def test_flat_clutter_never_pulls_a_side_outwards():
    cameras = np.column_stack([np.full(20, 0.6), np.full(20, 1.5), np.linspace(0.5, 3.0, 20)])
    clutter = np.column_stack(
        [np.linspace(0, 1.2, 2000), np.full(2000, 1.0), np.linspace(0, 6, 2000)]
    )
    _, hi, moved = extend_unsnapped_sides(
        [-0.15, -0.25], [1.35, 3.75], points=clutter, cameras=cameras, height=2.5,
        snapped={"west", "east", "south"},
    )  # fmt: skip
    assert moved == {} and hi[1] == 3.75


# --------------------------------------------------------------------------
# Rooms as a layout
# --------------------------------------------------------------------------


def test_one_room_is_the_rectangle_layout_of_its_outline():
    cameras = _walk(np.linspace(1.0, 3.0, 12))
    rooms = _segment(cameras, {0: "r1", 4: "r1", 8: "r1"})
    spawn = {"feet": [1.0, 0.0, 1.5], "yawRadians": 0.5, "floorId": "floor-0"}
    layout = rooms_layout(
        rooms, points=np.empty((0, 3)), cameras=cameras, height=2.5, floor_y=0.0, spawn=spawn
    )
    expected = rectangle_layout((0.25, 0.75), (3.75, 2.25), elevation=0.0, height=2.5, spawn=spawn)
    assert layout == expected
    assert validate_layout(layout) == []


def test_two_rooms_share_one_interior_wall_with_an_open_door_where_the_walk_passes():
    points, cameras, rooms = _two_room_scene()
    assert [room.id for room in rooms] == ["r1", "r2"]
    layout = rooms_layout(rooms, points=points, cameras=cameras, height=2.5, floor_y=0.0)
    assert validate_layout(layout) == []
    assert len(layout["floors"][0]["polygons"]) == 1  # the outline of the two rooms' union

    (door,) = _open_doors(layout)
    wall = _wall_by_id(layout, door["wallId"])
    start, end = np.array(wall["start"]), np.array(wall["end"])
    assert start[0] == pytest.approx(end[0])  # the interior wall runs along z
    unit = (end - start) / np.linalg.norm(end - start)
    centre = start + (door["offset"] + door["width"] / 2) * unit
    assert centre[1] == pytest.approx(1.5, abs=0.1)  # the walk crosses at z = 1.5
    assert 3.5 < centre[0] < 4.0  # between the cameras either side of the split
    assert door["width"] == pytest.approx(0.9)
    assert door["top"] == pytest.approx(2.05)
    assert door["provenance"] == "proposal"
    # Exactly one wall runs along the split: the interior one, which carries the doorway.
    across = [w for w in layout["walls"] if w["start"][0] == w["end"][0]]
    on_split = [w for w in across if w["start"][0] == pytest.approx(centre[0], abs=1e-6)]
    assert on_split == [wall]


def test_touching_rooms_outline_one_l_shaped_floor_that_validates():
    polygons = _union_outline([(0.0, 0.0, 4.0, 3.0), (4.0, 0.0, 7.0, 1.5)])
    outers, holes = polygons
    assert len(outers) == 1 and holes == []
    assert len(outers[0]) == 6  # an L: the two rooms with the corner notch cut away
    assert (
        validate_layout(
            {
                "schemaVersion": "1.0",
                "floors": [{"id": "floor-0", "elevation": 0.0, "polygons": outers, "holes": []}],
                "walls": [],
                "openings": [],
                "obstacles": [],
                "landmarks": [],
                "spawn": {"feet": [1.0, 0.0, 1.0], "yawRadians": 0.0, "floorId": "floor-0"},
            }
        )
        == []
    )


def test_a_ring_of_rooms_outlines_its_hole():
    rects = [(0.0, 0.0, 3.0, 1.0), (0.0, 2.0, 3.0, 3.0), (0.0, 1.0, 1.0, 2.0), (2.0, 1.0, 3.0, 2.0)]
    outers, holes = _union_outline(rects)
    assert len(outers) == 1 and len(holes) == 1
    assert len(outers[0]) == 4 and len(holes[0]) == 4


# --------------------------------------------------------------------------
# Doors from the analysis
# --------------------------------------------------------------------------


def _door(x, z, direction, *, width=0.9, top=2.05, state="closed", leads_to=None, frame=1):
    return GroundedDoor(
        frame=frame,
        position=(x, z),
        direction=direction,
        width=width,
        top=top,
        state=state,
        leads_to=leads_to,
    )


def test_exterior_doors_are_closed_and_interior_doors_are_open_unless_unknown():
    points, cameras, rooms = _two_room_scene()
    layout = rooms_layout(rooms, points=points, cameras=cameras, height=2.5, floor_y=0.0)
    split = _open_doors(layout)[0]
    split_wall = _wall_by_id(layout, split["wallId"])
    split_x = split_wall["start"][0]
    doors = [
        _door(1.5, 0.0, (1.0, 0.0)),  # on the south wall: exterior, so a closed panel
        _door(split_x, 0.4, (0.0, 1.0), state="open", leads_to="r2"),  # interior, room known
        _door(split_x, 2.5, (0.0, 1.0), state="open", leads_to="r9"),  # interior, unknown room
    ]
    result = attach_doors(layout, doors, room_ids=["r1", "r2"])
    states = {o["id"]: o["state"] for o in result["openings"]}
    assert sorted(states.values()) == ["closed", "closed", "open", "open"]
    new_ids = [o["id"] for o in result["openings"] if o["provenance"] == "proposal"]
    assert len(set(new_ids)) == len(new_ids)
    assert validate_layout(result) == []


def test_a_grounded_door_over_an_open_door_from_the_walk_is_not_added_again():
    points, cameras, rooms = _two_room_scene()
    layout = rooms_layout(rooms, points=points, cameras=cameras, height=2.5, floor_y=0.0)
    split = _open_doors(layout)[0]
    split_x = _wall_by_id(layout, split["wallId"])["start"][0]
    again = attach_doors(layout, [_door(split_x, 1.5, (0.0, 1.0), state="open")], ["r1", "r2"])
    assert again["openings"] == layout["openings"]


def _single_room():
    return rectangle_layout((0.0, 0.0), (12.0, 3.0), elevation=0.0, height=2.5)


def test_doors_on_one_wall_are_capped_at_eight():
    layout = _single_room()
    doors = [_door(0.5 + 1.1 * k, 0.0, (1.0, 0.0)) for k in range(10)]
    result = attach_doors(layout, doors, room_ids=[])
    assert len(result["openings"]) == 8


def test_two_detections_of_one_door_keep_the_wider_one():
    layout = _single_room()
    doors = [_door(3.0, 0.0, (1.0, 0.0), width=0.9), _door(3.1, 0.0, (1.0, 0.0), width=1.0)]
    result = attach_doors(layout, doors, room_ids=[])
    (door,) = result["openings"]
    assert door["width"] == pytest.approx(1.0)
    assert door["offset"] == pytest.approx(3.1 - 0.5, abs=1e-9)


def test_a_grounded_door_is_clamped_to_a_plausible_size_and_height():
    layout = _single_room()
    wide = _door(3.0, 0.0, (1.0, 0.0), width=2.5, top=3.0)
    narrow = _door(8.0, 0.0, (1.0, 0.0), width=0.3, top=1.0)
    result = attach_doors(layout, [wide, narrow], room_ids=[])
    first, second = sorted(result["openings"], key=lambda o: o["offset"])
    assert first["width"] == pytest.approx(1.6)  # the widest a door may be
    assert first["top"] == pytest.approx(2.3)
    assert second["width"] == pytest.approx(0.6)  # the narrowest
    assert second["top"] == pytest.approx(1.8)


def test_a_door_is_skipped_when_its_wall_is_too_short_or_too_far_or_turned():
    short = rectangle_layout((0.0, 0.0), (0.9, 3.0), elevation=0.0, height=2.5)
    assert attach_doors(short, [_door(0.45, 0.0, (1.0, 0.0))], room_ids=[])["openings"] == []
    layout = _single_room()
    far = _door(3.0, 1.0, (1.0, 0.0))  # a metre from the wall: too far
    turned = _door(3.0, 0.0, (0.0, 1.0))  # across the wall, not along it
    assert attach_doors(layout, [far, turned], room_ids=[])["openings"] == []


# --------------------------------------------------------------------------
# Furniture
# --------------------------------------------------------------------------


def _object(
    object_id: str,
    position,
    size=(2.0, 0.9, 0.9),
    *,
    confidence=0.9,
    solid=True,
    category="sofa",
) -> GroundedObject:
    return GroundedObject(
        id=object_id,
        category=category,
        label=f"a {category}",
        solid=solid,
        parts=[{"shape": "box", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#8a6d3b"}],
        position=position,
        size=size,
        confidence=confidence,
        detections=3,
    )


def _room(x1: float = 4.0, z1: float = 3.0, spawn=None):
    spawn = spawn or {"feet": [0.5, 0.0, 0.5], "yawRadians": 0.0, "floorId": "floor-0"}
    return rectangle_layout((0.0, 0.0), (x1, z1), elevation=0.0, height=2.5, spawn=spawn)


def test_a_sofa_by_a_wall_stands_flush_against_it_and_faces_the_room():
    layout = _room()
    # 0.3 m from the z = 0 wall, so its back should stand 0.07 + 0.45 m from it.
    result, dropped = place_furniture(layout, [_object("o1", (2.0, 0.3))], cameras=[[2, 1.5, 2.5]])
    assert dropped == 0
    (sofa,) = result["furniture"]
    assert sofa["position"] == pytest.approx([2.0, 0.07 + 0.45], abs=1e-6)
    assert sofa["yawRadians"] == pytest.approx(0.0, abs=1e-9)  # front = +z, into the room
    assert sofa["floorId"] == "floor-0" and sofa["provenance"] == "proposal"
    assert sofa["size"] == pytest.approx([2.0, 0.9, 0.9])
    assert sofa["id"] == "furniture-1"
    assert validate_layout(result) == []


def test_an_object_against_the_far_wall_faces_back_into_the_room():
    layout = _room()
    result, _ = place_furniture(layout, [_object("o1", (2.0, 2.7))], cameras=[[2, 1.5, 0.5]])
    (sofa,) = result["furniture"]
    assert sofa["position"][1] == pytest.approx(3.0 - 0.07 - 0.45, abs=1e-6)
    assert abs(sofa["yawRadians"]) == pytest.approx(math.pi, abs=1e-9)


def test_an_object_in_the_open_faces_the_cameras_on_a_quarter_turn():
    layout = _room(x1=8.0, z1=8.0)
    # The cameras are off to the left (-x) of the object, which stands in the middle of the room.
    result, _ = place_furniture(
        layout, [_object("o1", (4.0, 4.0))], cameras=[[0.5, 1.5, 4.0], [0.8, 1.5, 3.6]]
    )
    (sofa,) = result["furniture"]
    assert sofa["yawRadians"] == pytest.approx(-math.pi / 2, abs=1e-9)  # front = -x


def test_an_object_that_overlaps_a_placed_one_by_too_much_is_dropped():
    layout = _room(x1=8.0, z1=8.0)
    objects = [_object("o1", (4.0, 4.0), confidence=0.9), _object("o2", (4.1, 4.0), confidence=0.5)]
    result, dropped = place_furniture(layout, objects, cameras=[[1, 1.5, 1]])
    assert len(result["furniture"]) == 1 and dropped == 1
    assert result["furniture"][0]["label"] == "a sofa"


def test_objects_are_placed_most_confident_first_and_small_overlaps_are_kept():
    layout = _room(x1=8.0, z1=8.0)
    # The two sofas share a quarter of the first one's area: under the 40 % limit.
    objects = [_object("o1", (4.0, 4.0), confidence=0.5), _object("o2", (5.5, 4.0), confidence=0.9)]
    result, dropped = place_furniture(layout, objects, cameras=[[1, 1.5, 1]])
    assert dropped == 0 and len(result["furniture"]) == 2
    assert result["furniture"][0]["position"][0] == pytest.approx(5.5)
    assert result["furniture"][0]["confidence"] == pytest.approx(0.9)


def test_an_object_just_outside_the_floor_is_moved_in_and_one_far_outside_is_dropped():
    layout = _room()
    near, far = _object("o1", (-0.2, 1.5)), _object("o2", (-3.0, 1.5), category="chair")
    result, dropped = place_furniture(layout, [near, far], cameras=[[2, 1.5, 1.5]])
    assert dropped == 1
    (sofa,) = result["furniture"]
    assert sofa["position"][0] > 0.0
    assert validate_layout(result) == []


def test_an_object_taller_than_the_room_is_clamped_and_a_wide_one_shrinks_to_fit():
    layout = _room(x1=2.0, z1=8.0)
    tall_and_wide = _object("o1", (1.0, 4.0), size=(3.0, 3.0, 0.5))
    result, _ = place_furniture(layout, [tall_and_wide], cameras=[[1, 1.5, 1]])
    (item,) = result["furniture"]
    assert item["size"][1] == pytest.approx(2.5 - 0.02)
    assert item["size"][0] == pytest.approx(2.0 - 0.1)  # the room's width less 0.1 m


def test_at_most_sixty_four_objects_are_placed():
    layout = _room(x1=12.0, z1=12.0)
    objects = [
        _object(f"o{i}", (0.5 + 1.0 * (i % 11), 0.5 + 1.0 * (i // 11)), size=(0.5, 0.5, 0.5))
        for i in range(70)
    ]
    result, dropped = place_furniture(layout, objects, cameras=[[1, 1.5, 1]])
    assert len(result["furniture"]) == 64 and dropped == 6


def test_the_spawn_moves_off_a_solid_object_and_keeps_its_yaw():
    spawn = {"feet": [2.0, 0.0, 1.5], "yawRadians": 0.7, "floorId": "floor-0"}
    layout = _room(spawn=spawn)
    sofa = _object("o1", (2.0, 1.5))  # the walk started on the sofa
    result, _ = place_furniture(layout, [sofa], cameras=[[2, 1.5, 0.5]])
    (item,) = result["furniture"]
    feet = np.array([result["spawn"]["feet"][0], result["spawn"]["feet"][2]])
    centre = np.array(item["position"])
    # The sofa's footprint at yaw 0: half-extents 1.0 m across and 0.45 m deep.
    dx = max(abs(feet[0] - centre[0]) - 1.0, 0.0)
    dz = max(abs(feet[1] - centre[1]) - 0.45, 0.0)
    assert math.hypot(dx, dz) >= 0.35
    assert result["spawn"]["yawRadians"] == pytest.approx(0.7)
    assert result["spawn"]["feet"][1] == 0.0
    assert np.allclose(np.round(feet, 1), feet)  # the nearest free point of the 0.1 m grid
    assert validate_layout(result) == []


def test_the_spawn_stays_where_it_is_when_nothing_is_near_it():
    layout = _room(spawn={"feet": [0.5, 0.0, 0.5], "yawRadians": 0.0, "floorId": "floor-0"})
    result, _ = place_furniture(layout, [_object("o1", (3.0, 2.0))], cameras=[[1, 1.5, 1]])
    assert result["spawn"]["feet"] == [0.5, 0.0, 0.5]


def test_grounded_furniture_in_a_two_room_layout_stays_inside_its_room():
    points, cameras, rooms = _two_room_scene()
    layout = rooms_layout(rooms, points=points, cameras=cameras, height=2.5, floor_y=0.0)
    sofa = _object("o1", (1.5, 0.4))  # in room one, near the south wall
    result, dropped = place_furniture(layout, [sofa], cameras=cameras)
    assert dropped == 0 and len(result["furniture"]) == 1
    assert validate_layout(result) == []
